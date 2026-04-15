"""GATv2 (Graph Attention Network v2) multi-head attention layer.

Reference: "How Attentive are Graph Attention Networks?" (Brody et al., ICLR 2022)
https://arxiv.org/abs/2105.14491

Drop-in replacement for the original GRU-based `GNN` class in
`model/gran_mixture_bernoulli.py`. Same `forward()` signature so it can be
swapped via a config flag.

Key difference from vanilla GAT:
    GAT v1:   alpha_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
    GATv2:    alpha_ij = softmax_j(a^T LeakyReLU([W_l h_i || W_r h_j]))
                                    ^^^^^^^^^^^^
    The LeakyReLU is applied BEFORE the dot product with `a`, which makes
    the attention "dynamic" (strictly more expressive than GAT v1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """One GATv2 attention layer.

    Given a graph (N nodes, E directed edges, optional per-edge features),
    compute updated node features by attention-weighted aggregation of neighbors.

    Forward pass interface:
        in  node_feat: (N, in_dim)           per-node input features
        in  edges:     (E, 2) long           edge list [src, tgt]
        in  edge_feat: (E, edge_feat_dim) or None
        out           : (N, out_dim)         per-node output features

    Args:
        in_dim:         input node feature dimension
        out_dim:        output node feature dimension (must be divisible by num_heads)
        edge_feat_dim:  per-edge feature dim (0 = no edge features)
        num_heads:      number of attention heads (H)
        negative_slope: LeakyReLU slope
        dropout:        attention weight dropout

    Shape glossary (used in forward):
        N = num nodes,  E = num edges,  H = num heads,  D = head_dim (= out_dim // H)
    """

    def __init__(self, in_dim, out_dim, edge_feat_dim, num_heads=4,
                 negative_slope=0.2, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        # Three separate projection matrices, one per role:
        #   W_l h_i  -> "left"   message (source side of edge)
        #   W_r h_j  -> "right"  message (target side of edge)
        #   W_v h_j  -> "value"  to be aggregated at the target
        # Each outputs (N, out_dim) = (N, H*D).
        self.W_l = nn.Linear(in_dim, out_dim, bias=False)
        self.W_r = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)

        # Optional edge feature projection (same out_dim).
        # When edges carry per-edge features (e.g. one-hot attention ids in GRAN),
        # they get concatenated into the attention input as a third part.
        if edge_feat_dim > 0:
            self.W_e = nn.Linear(edge_feat_dim, out_dim, bias=False)
            att_input_dim = self.head_dim * 3      # [left || right || edge] per head
        else:
            self.W_e = None
            att_input_dim = self.head_dim * 2      # [left || right] per head

        # Attention vector `a` -- one per head, dotted with the leaky-relu'd input.
        # Shape: (num_heads, att_input_dim). Each head learns its own pattern.
        self.attn = nn.Parameter(torch.zeros(num_heads, att_input_dim))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, edges, edge_feat=None):
        """Attention aggregation in 6 steps.

        Running example used throughout the comments:
            N = 10 nodes, E = 20 edges, in_dim = 64, out_dim = 128,
            num_heads H = 4, head_dim D = 32.
        """
        N = node_feat.shape[0]        # e.g. 10
        H = self.num_heads            # e.g. 4
        D = self.head_dim             # e.g. 32

        # --- Step 1: Project node features into three per-head views -----------
        # Each linear maps (N, in_dim=64) -> (N, out_dim=128)
        # View as (N, H=4, D=32) so every head has its own D-dim subspace.
        h_l = self.W_l(node_feat).view(N, H, D)   # (10, 4, 32)
        h_r = self.W_r(node_feat).view(N, H, D)   # (10, 4, 32)
        h_v = self.W_v(node_feat).view(N, H, D)   # (10, 4, 32)

        # --- Edge case: graph with zero edges ----------------------------------
        # No message passing to do. Just return the value projection so isolated
        # nodes still get some learned representation. Shape (N, H*D) = (10, 128).
        if edges.shape[0] == 0:
            return h_v.view(N, H * D)

        # --- Step 2: Gather per-edge source / target features ------------------
        # edges: (E=20, 2) long tensor of [src, tgt] pairs.
        src, tgt = edges[:, 0], edges[:, 1]       # each (E=20,)

        msg_l = h_l[src]   # (E=20, H=4, D=32) -- source side of each edge
        msg_r = h_r[tgt]   # (E=20, H=4, D=32) -- target side of each edge

        # --- Step 3: Build attention input via concat then LeakyReLU -----------
        # (This is the GATv2 twist: nonlinearity BEFORE dot product with `a`.)
        if self.W_e is not None and edge_feat is not None:
            e_proj = self.W_e(edge_feat).view(-1, H, D)       # (E, H, D)
            attn_input = torch.cat([msg_l, msg_r, e_proj], dim=-1)  # (E, H, 3D)
        else:
            attn_input = torch.cat([msg_l, msg_r], dim=-1)          # (E, H, 2D)

        attn_input = self.leaky_relu(attn_input)   # elementwise LeakyReLU

        # --- Step 4: Attention score per edge per head -------------------------
        # Dot product between attn_input and the learned `a` vector per head.
        # self.attn: (H, att_input_dim) -> unsqueeze to (1, H, att_input_dim)
        # attn_input: (E, H, att_input_dim)
        # sum over last dim -> (E, H) raw attention logits.
        e = (attn_input * self.attn.unsqueeze(0)).sum(dim=-1)   # (E=20, H=4)

        # --- Step 5: Softmax of attention scores, grouped by target node -------
        # We want, for each target node t, alpha_{i->t} to sum to 1 across
        # all incoming edges i. Since nodes have variable in-degree, we do a
        # manual "segmented softmax" via scatter_reduce / scatter_add:
        #
        #   1) subtract per-target max for numerical stability
        #   2) exponentiate
        #   3) divide by per-target sum

        # (a) per-target max across heads: (N, H)
        e_max = torch.zeros(N, H, device=node_feat.device)
        e_max = e_max.scatter_reduce(
            0,
            tgt.unsqueeze(1).expand(-1, H),   # (E, H) index: which target node
            e,
            reduce='amax',
            include_self=True,
        )
        # Broadcast subtract: e is (E,H); e_max[tgt] is (E,H).
        e = e - e_max[tgt]
        e = torch.exp(e)                                        # (E, H)

        # (b) per-target sum
        e_sum = torch.zeros(N, H, device=node_feat.device)
        e_sum = e_sum.scatter_add(0, tgt.unsqueeze(1).expand(-1, H), e)  # (N, H)

        # (c) normalize
        alpha = e / (e_sum[tgt] + 1e-10)                        # (E, H)
        alpha = self.dropout(alpha)

        # --- Step 6: Aggregate value vectors weighted by attention --------------
        # h_v[src]: (E, H, D)   the neighbors' value vectors
        # alpha: (E, H) -> unsqueeze to (E, H, 1) for broadcasting
        # Element-wise multiply to get weighted messages.
        msg = h_v[src] * alpha.unsqueeze(-1)                    # (E, H, D)

        # Scatter-add messages into output buffer by target node.
        out = torch.zeros(N, H, D, device=node_feat.device)
        out = out.scatter_add(
            0,
            tgt.unsqueeze(1).unsqueeze(2).expand(-1, H, D),
            msg,
        )   # (N, H, D)

        # Concatenate head outputs back to (N, H*D = out_dim).
        return out.view(N, H * D)


class GATv2(nn.Module):
    """Multi-layer GATv2 stack.

    Drop-in replacement for the original `GNN` class in
    `model/gran_mixture_bernoulli.py`. Matches the same forward() signature so
    GRANv2 can switch between the two via a `use_gatv2` config flag:

        if config.model.use_gatv2:
            self.decoder = GATv2(...)
        else:
            self.decoder = GNN(...)     # original GRU-based

    Forward interface:
        in  node_feat: (N, node_state_dim)
        in  edge:      (E, 2)  long
        in  edge_feat: (E, edge_feat_dim) or None
        in  graph_idx: (N,) long or None
        out           : (N, node_state_dim)            if has_graph_output=False
                       (num_graphs, graph_output_dim)  if has_graph_output=True
    """

    def __init__(self, node_state_dim, edge_feat_dim, num_heads=4,
                 num_layer=1, has_residual=True, dropout=0.0,
                 has_graph_output=False, output_hidden_dim=128,
                 graph_output_dim=None):
        super().__init__()
        self.num_layer = num_layer
        self.has_residual = has_residual
        self.has_graph_output = has_graph_output

        # Each layer is a self-loop over node_state_dim (in == out).
        # Stacking many layers widens the receptive field.
        self.layers = nn.ModuleList([
            GATv2Layer(
                in_dim=node_state_dim,
                out_dim=node_state_dim,
                edge_feat_dim=edge_feat_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layer)
        ])

        # LayerNorm between layers helps training stability in deep stacks.
        self.norms = nn.ModuleList([
            nn.LayerNorm(node_state_dim) for _ in range(num_layer)
        ])

        # Optional graph-level output head (matches original GNN class).
        # Averages node features weighted by a learned attention to produce
        # one vector per graph. Not used by GRAN's edge prediction path.
        if has_graph_output:
            self.graph_output_head_att = nn.Sequential(
                nn.Linear(node_state_dim, output_hidden_dim),
                nn.ReLU(),
                nn.Linear(output_hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.graph_output_head = nn.Sequential(
                nn.Linear(node_state_dim, graph_output_dim),
            )

    def forward(self, node_feat, edge, edge_feat, graph_idx=None):
        """Run L layers of GATv2 attention with residual + norm + ReLU.

        Running example:
            N = 15 nodes, E = 40 edges, node_state_dim = 64, num_layer = 3.
        """
        state = node_feat                              # (N=15, 64)

        # --- Stacked attention layers -----------------------------------------
        for ii in range(self.num_layer):
            # 1. Attention aggregation (replaces message + GRU update of orig GNN)
            new_state = self.layers[ii](state, edge, edge_feat)   # (N, 64)
            # 2. LayerNorm stabilizes training with deep stacks
            new_state = self.norms[ii](new_state)
            # 3. Residual connection: keeps gradient flow healthy
            if self.has_residual:
                new_state = new_state + state
            # 4. Nonlinearity
            state = F.relu(new_state)

        # --- Optional graph-level pooling -------------------------------------
        # Only used when caller asks for one vector per graph (e.g. graph
        # classification heads). GRAN's edge-prediction path returns per-node
        # features, so this branch is skipped.
        if self.has_graph_output and graph_idx is not None:
            num_graph = graph_idx.max() + 1
            node_att_weight = self.graph_output_head_att(state)    # (N, 1)
            node_output = self.graph_output_head(state)            # (N, graph_output_dim)

            reduce_output = torch.zeros(
                num_graph, node_output.shape[1], device=node_feat.device,
            )
            reduce_output = reduce_output.scatter_add(
                0,
                graph_idx.unsqueeze(1).expand(-1, node_output.shape[1]),
                node_output * node_att_weight,
            )

            const = torch.zeros(num_graph, device=node_feat.device)
            const = const.scatter_add(
                0, graph_idx,
                torch.ones(node_output.shape[0], device=node_feat.device),
            )
            reduce_output = reduce_output / const.view(-1, 1)
            return reduce_output                     # (num_graphs, graph_output_dim)

        return state                                  # (N, node_state_dim)
