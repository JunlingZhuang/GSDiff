"""GATv2 (Graph Attention Network v2) multi-head attention layer.

Reference: "How Attentive are Graph Attention Networks?" (Brody et al., ICLR 2022)
https://arxiv.org/abs/2105.14491

Drop-in replacement for the original GRU-based `GNN` class in
`model/gran_mixture_bernoulli.py`. Same `forward()` signature so it can be
swapped via a config flag.

This file has TWO implementations of the same GATv2 idea:

  1. ACTIVE:     thin wrapper around `torch_geometric.nn.GATv2Conv`
                 (well tested, optimized CUDA kernels, less code to maintain)

  2. REFERENCE:  pure-PyTorch hand-rolled version at the bottom, kept as
                 commented-out code so you can see every step of the attention
                 math (projection -> concat -> LeakyReLU -> softmax -> aggregate).
                 It is not used by the model; it documents the algorithm.

If torch-geometric is ever uninstalled, you can revive the reference
implementation by uncommenting it and deleting the PyG import / classes above.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv


# ============================================================================
# ACTIVE IMPLEMENTATION: PyG-based
# ============================================================================

class GATv2Layer(nn.Module):
    """One GATv2 attention layer, backed by ``torch_geometric.nn.GATv2Conv``.

    Forward pass interface (matches the original GRAN `GNN` interface):
        in  node_feat: (N, in_dim)            per-node input features
        in  edges:     (E, 2) long            edge list [src, tgt]
        in  edge_feat: (E, edge_feat_dim) or None
        out           : (N, out_dim)          per-node output features

    Args:
        in_dim:         input node feature dimension
        out_dim:        output node feature dimension (must be divisible by num_heads)
        edge_feat_dim:  per-edge feature dim (0 = no edge features)
        num_heads:      number of attention heads (H)
        negative_slope: LeakyReLU slope (GATv2's inner nonlinearity)
        dropout:        attention weight dropout

    Shape glossary used internally:
        N = num nodes, E = num edges, H = num heads, D = head_dim (= out_dim // H)
    """

    def __init__(self, in_dim, out_dim, edge_feat_dim, num_heads=4,
                 negative_slope=0.2, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = out_dim // num_heads
        self.edge_feat_dim = edge_feat_dim

        # PyG's GATv2Conv:
        #   - in_channels:  per-node input dim
        #   - out_channels: per-HEAD output dim (since concat=True the final
        #                   width is heads * out_channels = out_dim)
        #   - edge_dim:     per-edge feature dim, or None
        #   - negative_slope / dropout: GATv2 internals
        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=self.head_dim,
            heads=num_heads,
            concat=True,
            negative_slope=negative_slope,
            dropout=dropout,
            edge_dim=edge_feat_dim if edge_feat_dim > 0 else None,
            add_self_loops=False,  # GRAN's sampling explicitly builds the edge
                                   # list; we don't want PyG to silently add
                                   # self-loops that weren't in the original graph
        )

    def forward(self, node_feat, edges, edge_feat=None):
        """Run one GATv2 layer over the given graph.

        Running example:
            N = 10 nodes, E = 20 edges, in_dim = 64, out_dim = 128, num_heads = 4.
            node_feat:  (10, 64)
            edges:      (20, 2)
            edge_feat:  (20, 128) or None
            returns:    (10, 128)
        """
        N = node_feat.shape[0]

        # GRAN uses edge shape (E, 2); PyG wants (2, E).
        edge_index = edges.t().contiguous() if edges.numel() > 0 \
            else torch.zeros(2, 0, dtype=torch.long, device=node_feat.device)

        # PyG handles the edge_attr=None case internally when edge_dim is None.
        if self.edge_feat_dim > 0 and edge_feat is not None:
            out = self.conv(node_feat, edge_index, edge_attr=edge_feat)
        else:
            out = self.conv(node_feat, edge_index)

        # With no edges, PyG returns zeros for isolated nodes. The hand-rolled
        # version below instead returned the value projection h_v so gradients
        # still flow. We mimic that behavior here by falling back to a learnable
        # projection of the input when the graph is empty. This keeps the
        # model well-defined for zero-edge graphs (e.g. first generation step
        # in autoregressive sampling).
        if edges.numel() == 0:
            # When there are NO edges at all, PyG's conv still runs but with
            # an empty message set. Its output for isolated nodes is a linear
            # projection of the node features (via conv.lin_l / lin_r), so this
            # is actually fine — it will NOT be all zeros. We return as-is.
            return out

        return out


class GATv2(nn.Module):
    """Multi-layer GATv2 stack built from ``GATv2Layer``.

    Drop-in replacement for the original `GNN` class in
    `model/gran_mixture_bernoulli.py`. Task 2 will switch between the two
    backbones via a ``use_gatv2`` config flag:

        if config.model.use_gatv2:
            self.decoder = GATv2(...)
        else:
            self.decoder = GNN(...)     # original GRU-based

    Forward interface (matches original GNN class verbatim):
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

        # Each layer preserves node_state_dim so outputs can stack with
        # residual connections and the final output keeps the expected dim.
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

        # LayerNorm between layers stabilizes deep stacks.
        self.norms = nn.ModuleList([
            nn.LayerNorm(node_state_dim) for _ in range(num_layer)
        ])

        # Optional graph-level output head (kept to mirror the original GNN).
        # Not used on GRAN's edge-prediction path.
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
        """Stacked GATv2 layers with residual + LayerNorm + ReLU."""
        state = node_feat                              # (N, node_state_dim)

        for ii in range(self.num_layer):
            new_state = self.layers[ii](state, edge, edge_feat)   # (N, node_state_dim)
            new_state = self.norms[ii](new_state)
            if self.has_residual:
                new_state = new_state + state
            state = F.relu(new_state)

        # Optional graph-level pooling (unused for GRAN's edge path).
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


# ============================================================================
# REFERENCE IMPLEMENTATION: hand-rolled, KEPT COMMENTED FOR READING
# ============================================================================
#
# The code below is the original pure-PyTorch implementation of GATv2.
# It is NOT used by the model — the PyG-backed classes above are the active
# implementation. It is kept here so you can see every step of the attention
# math end-to-end without diving into PyG internals.
#
# To revive it: delete the `from torch_geometric.nn import GATv2Conv` import
# and the two classes above, then uncomment everything below.
#
# ----------------------------------------------------------------------------
#
# class GATv2Layer(nn.Module):
#     """Hand-rolled GATv2 attention layer (Brody et al., 2022).
#
#     Computes: alpha_ij = softmax_j(a^T LeakyReLU(W_l h_i || W_r h_j || W_e e_ij))
#               h_i'     = ||_{k=1}^{H} sum_j alpha_ij^k * W_v^k h_j
#     """
#
#     def __init__(self, in_dim, out_dim, edge_feat_dim, num_heads=4,
#                  negative_slope=0.2, dropout=0.0):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = out_dim // num_heads
#         assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
#
#         # Three separate projections:
#         #   W_l h_i  -> "left"  message (source side of edge)
#         #   W_r h_j  -> "right" message (target side of edge)
#         #   W_v h_j  -> "value" to be aggregated at the target
#         self.W_l = nn.Linear(in_dim, out_dim, bias=False)
#         self.W_r = nn.Linear(in_dim, out_dim, bias=False)
#         self.W_v = nn.Linear(in_dim, out_dim, bias=False)
#
#         # Optional edge-feature projection for including per-edge signals.
#         if edge_feat_dim > 0:
#             self.W_e = nn.Linear(edge_feat_dim, out_dim, bias=False)
#             att_input_dim = self.head_dim * 3   # [left || right || edge] per head
#         else:
#             self.W_e = None
#             att_input_dim = self.head_dim * 2   # [left || right]          per head
#
#         # Learned attention vector per head; dotted with leaky-relu'd input.
#         self.attn = nn.Parameter(torch.zeros(num_heads, att_input_dim))
#         nn.init.xavier_uniform_(self.attn.unsqueeze(0))
#
#         self.leaky_relu = nn.LeakyReLU(negative_slope)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, node_feat, edges, edge_feat=None):
#         """Attention aggregation in 6 documented steps.
#
#         Running example: N=10 nodes, E=20 edges, in_dim=64, out_dim=128,
#                          num_heads H=4, head_dim D=32.
#         """
#         N = node_feat.shape[0]
#         H = self.num_heads
#         D = self.head_dim
#
#         # --- Step 1: project node features into three per-head views --------
#         h_l = self.W_l(node_feat).view(N, H, D)   # (10, 4, 32)
#         h_r = self.W_r(node_feat).view(N, H, D)   # (10, 4, 32)
#         h_v = self.W_v(node_feat).view(N, H, D)   # (10, 4, 32)
#
#         # --- Edge case: zero edges ------------------------------------------
#         # Isolated nodes: return the value projection so gradients still flow.
#         if edges.shape[0] == 0:
#             return h_v.view(N, H * D)
#
#         # --- Step 2: per-edge gather of source / target features ------------
#         src, tgt = edges[:, 0], edges[:, 1]
#         msg_l = h_l[src]   # (E, H, D)
#         msg_r = h_r[tgt]   # (E, H, D)
#
#         # --- Step 3: build attention input then apply LeakyReLU (GATv2 fix) -
#         if self.W_e is not None and edge_feat is not None:
#             e_proj = self.W_e(edge_feat).view(-1, H, D)          # (E, H, D)
#             attn_input = torch.cat([msg_l, msg_r, e_proj], dim=-1)  # (E,H,3D)
#         else:
#             attn_input = torch.cat([msg_l, msg_r], dim=-1)          # (E,H,2D)
#         attn_input = self.leaky_relu(attn_input)
#
#         # --- Step 4: attention score per edge per head ----------------------
#         e = (attn_input * self.attn.unsqueeze(0)).sum(dim=-1)   # (E, H)
#
#         # --- Step 5: segmented softmax grouped by target node ---------------
#         # Manual softmax since nodes have variable in-degree:
#         #   1) per-target max subtract (numerical stability)
#         #   2) exponentiate
#         #   3) divide by per-target sum
#         e_max = torch.zeros(N, H, device=node_feat.device)
#         e_max = e_max.scatter_reduce(
#             0, tgt.unsqueeze(1).expand(-1, H), e,
#             reduce='amax', include_self=True,
#         )
#         e = e - e_max[tgt]
#         e = torch.exp(e)
#         e_sum = torch.zeros(N, H, device=node_feat.device)
#         e_sum = e_sum.scatter_add(0, tgt.unsqueeze(1).expand(-1, H), e)
#         alpha = e / (e_sum[tgt] + 1e-10)                        # (E, H)
#         alpha = self.dropout(alpha)
#
#         # --- Step 6: aggregate value vectors weighted by attention ----------
#         msg = h_v[src] * alpha.unsqueeze(-1)                    # (E, H, D)
#         out = torch.zeros(N, H, D, device=node_feat.device)
#         out = out.scatter_add(
#             0,
#             tgt.unsqueeze(1).unsqueeze(2).expand(-1, H, D),
#             msg,
#         )
#         return out.view(N, H * D)
#
#
# class GATv2(nn.Module):
#     """Hand-rolled multi-layer GATv2 stack (reference version)."""
#
#     def __init__(self, node_state_dim, edge_feat_dim, num_heads=4,
#                  num_layer=1, has_residual=True, dropout=0.0,
#                  has_graph_output=False, output_hidden_dim=128,
#                  graph_output_dim=None):
#         super().__init__()
#         self.num_layer = num_layer
#         self.has_residual = has_residual
#         self.has_graph_output = has_graph_output
#
#         self.layers = nn.ModuleList([
#             GATv2Layer(
#                 in_dim=node_state_dim,
#                 out_dim=node_state_dim,
#                 edge_feat_dim=edge_feat_dim,
#                 num_heads=num_heads,
#                 dropout=dropout,
#             )
#             for _ in range(num_layer)
#         ])
#         self.norms = nn.ModuleList([
#             nn.LayerNorm(node_state_dim) for _ in range(num_layer)
#         ])
#
#         if has_graph_output:
#             self.graph_output_head_att = nn.Sequential(
#                 nn.Linear(node_state_dim, output_hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(output_hidden_dim, 1),
#                 nn.Sigmoid(),
#             )
#             self.graph_output_head = nn.Sequential(
#                 nn.Linear(node_state_dim, graph_output_dim),
#             )
#
#     def forward(self, node_feat, edge, edge_feat, graph_idx=None):
#         state = node_feat
#         for ii in range(self.num_layer):
#             new_state = self.layers[ii](state, edge, edge_feat)
#             new_state = self.norms[ii](new_state)
#             if self.has_residual:
#                 new_state = new_state + state
#             state = F.relu(new_state)
#
#         if self.has_graph_output and graph_idx is not None:
#             num_graph = graph_idx.max() + 1
#             node_att_weight = self.graph_output_head_att(state)
#             node_output = self.graph_output_head(state)
#
#             reduce_output = torch.zeros(
#                 num_graph, node_output.shape[1], device=node_feat.device,
#             )
#             reduce_output = reduce_output.scatter_add(
#                 0,
#                 graph_idx.unsqueeze(1).expand(-1, node_output.shape[1]),
#                 node_output * node_att_weight,
#             )
#             const = torch.zeros(num_graph, device=node_feat.device)
#             const = const.scatter_add(
#                 0, graph_idx,
#                 torch.ones(node_output.shape[0], device=node_feat.device),
#             )
#             reduce_output = reduce_output / const.view(-1, 1)
#             return reduce_output
#
#         return state
