"""GRANv2 — GRAN extended with node-attribute prediction, partial-graph
conditioning, and an optional GATv2 backbone.

This is a superset of the original ``GRANMixtureBernoulli`` class. The original
class is kept intact in ``model/gran_mixture_bernoulli.py`` so existing scripts
continue to work unchanged. GRANv2 extends it along four axes:

  1. **Node attribute head** (``self.output_attr``):
     a small MLP that predicts a per-node categorical label
     (e.g. room type for floorplans) from the GNN's final node state.

  2. **Partial-graph conditioning in sampling**:
     ``_sampling()`` accepts an optional ``(partial_A, partial_attrs, start_idx)``
     triple. If given, the first ``n_partial`` rows / cols of the adjacency
     and the first ``n_partial`` attr slots are filled in before the
     autoregressive loop, and generation starts from ``start_idx`` instead
     of 0. This is the "graph completion" use-case.

  3. **Optional GATv2 backbone**:
     when ``config.model.use_gatv2 == True``, ``self.decoder`` is a
     ``model.gatv2.GATv2`` stack instead of the original GRU-based ``GNN``.
     Both classes share the exact same forward signature, so the rest of
     the code path is untouched.

  4. **Optional class-weighted attribute loss** (plan A):
     ``config.model.attr_class_weight`` accepts ``None`` (uniform),
     an explicit list of length ``num_attr_classes``, or the string
     ``'auto'`` (runner computes inverse-frequency weights from
     ``graphs_train``). Stored as the buffer ``self.attr_class_weights``
     and passed to ``F.cross_entropy`` every forward pass.

Forward interface matches the original class:
    training (is_sampling=False) -> (edge_loss, attr_loss)
    sampling (is_sampling=True)  -> (A_list, attr_list)

Shape glossary used throughout:
    B   = batch size
    N   = self.max_num_nodes (e.g. 20)
    H   = self.hidden_dim     (e.g. 64)
    K   = self.block_size     (e.g. 1 — generate K new nodes per step)
    S   = self.sample_stride  (e.g. 1)
    C   = self.num_canonical_order
    L   = self.num_mix_component
    A   = number of attribute classes (e.g. 7 for room types)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gran_mixture_bernoulli import GNN, mixture_bernoulli_loss
from model.gatv2 import GATv2

__all__ = ['GRANv2']


class GRANv2(nn.Module):
    """GRAN Mixture-of-Bernoulli model + attribute head + GATv2 option."""

    def __init__(self, config):
        """Build the model from an EasyDict config.

        Relevant config fields:
            config.device                   -- torch device string
            config.model.max_num_nodes      -- N (padded graph size)
            config.model.hidden_dim         -- H (GNN hidden dim)
            config.model.embedding_dim      -- input projection width
            config.model.num_GNN_layers     -- depth of the GNN
            config.model.num_mix_component  -- L (mixture-of-Bernoulli size)
            config.model.num_attr_classes   -- A (node-attribute classes)  [v2]
            config.model.use_gatv2          -- bool, pick backbone         [v2]
            config.model.gatv2_num_heads    -- heads when use_gatv2=True   [v2]
            config.model.block_size         -- K, nodes per autoregressive step
            config.model.sample_stride      -- S, stride between steps
        """
        super().__init__()
        self.config = config
        self.device = config.device

        # ---- core GRAN hyperparameters (same as GRANMixtureBernoulli) -----
        self.max_num_nodes = config.model.max_num_nodes         # N,  e.g. 20
        self.hidden_dim = config.model.hidden_dim               # H,  e.g. 64
        self.is_sym = config.model.is_sym
        self.block_size = config.model.block_size               # K,  e.g. 1
        self.sample_stride = config.model.sample_stride         # S,  e.g. 1
        self.num_GNN_prop = config.model.num_GNN_prop
        self.num_GNN_layers = config.model.num_GNN_layers
        self.edge_weight = getattr(config.model, 'edge_weight', 1.0)
        self.dimension_reduce = config.model.dimension_reduce
        self.has_attention = config.model.has_attention
        self.num_canonical_order = config.model.num_canonical_order
        self.num_mix_component = config.model.num_mix_component  # L,  e.g. 5

        # ---- v2-specific hyperparameters ---------------------------------
        # A = number of attribute classes (e.g. 7 room types).
        self.num_attr_classes = config.model.num_attr_classes
        self.use_gatv2 = getattr(config.model, 'use_gatv2', False)
        self.gatv2_num_heads = getattr(config.model, 'gatv2_num_heads', 4)

        # Fixed sub-hyperparameters (identical to original class)
        self.output_dim = 1
        self.has_rand_feat = False
        self.att_edge_dim = 64

        # ---- Plan C-1: attribute-conditioned edge head --------------------
        # When on, the edge head sees the attribute embedding of both
        # endpoints alongside the standard diff = h_u - h_v. This lets the
        # model learn pair-specific structural priors (e.g. high P(edge)
        # for Bedroom-Bathroom, low P(edge) for Living-Living).
        #
        # We reserve embedding row index ``num_attr_classes`` as the
        # "unknown / not-yet-sampled" slot. It is used at sampling time for
        # the K-block self-edges (whose own attr is being predicted in the
        # same step) and as a safe default if any consumer feeds in an
        # out-of-range attr id.
        #
        # NOTE: this block must be declared BEFORE the edge-head MLPs below
        # because ``edge_head_in_dim`` depends on ``self.attr_embedding_dim``.
        self.use_attr_conditioned_edge = getattr(
            config.model, 'use_attr_conditioned_edge', False)
        if self.use_attr_conditioned_edge:
            self.attr_embedding_dim = int(
                getattr(config.model, 'attr_embedding_dim', 32))
            self.attr_embedding = nn.Embedding(
                num_embeddings=self.num_attr_classes + 1,
                embedding_dim=self.attr_embedding_dim,
            )
            nn.init.normal_(self.attr_embedding.weight, mean=0.0, std=0.1)
        else:
            self.attr_embedding_dim = 0
            self.attr_embedding = None

        # ---- edge heads: input is diff_h_uv (width hidden_dim) plus,
        # when use_attr_conditioned_edge=True, the concatenated attr
        # embeddings of both endpoints (each width attr_embedding_dim).
        edge_head_in_dim = self.hidden_dim + 2 * self.attr_embedding_dim

        self.output_theta = nn.Sequential(
            nn.Linear(edge_head_in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim * self.num_mix_component))

        self.output_alpha = nn.Sequential(
            nn.Linear(edge_head_in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component))

        # ---- NEW: node-attribute head -------------------------------------
        # Maps per-node hidden state (*, H) -> per-node class logits (*, A).
        # Example: H=64, A=7 -> accepts (10, 64), returns (10, 7).
        self.output_attr = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_attr_classes))

        # ---- v2 structural fix: per-node degree-rank embedding ------------
        # Optional ordering-invariant structural signal. Indexed by each
        # node's current rank (0 = highest-degree among already-generated
        # nodes, N-1 = lowest). Added to initial node features before GNN
        # propagation in both _inference and _sampling. Gated by
        # ``config.model.use_degree_feature`` (default False) so old
        # checkpoints still load cleanly.
        self.use_degree_feature = getattr(
            config.model, 'use_degree_feature', False)
        if self.use_degree_feature:
            self.degree_embedding = nn.Embedding(
                num_embeddings=self.max_num_nodes,
                embedding_dim=config.model.embedding_dim,
            )
            nn.init.normal_(self.degree_embedding.weight, mean=0.0, std=0.1)

        # ---- v2 structural fix (Path 1): ordering-id embedding -------------
        # When ``num_canonical_order > 1`` the dataset emits each graph under
        # several canonical orderings (DFS + BFS + k-core). Without an explicit
        # "which ordering am I seeing right now" signal, the model learns a
        # mixture distribution over orderings and that mixture matches NO
        # single ordering at sampling time. This embedding table (one row per
        # ordering id 0..C-1) is added to every node's initial feature BEFORE
        # GNN propagation, giving the model a direct ordering signal.
        # Gated by ``config.model.use_ordering_id`` (default False) so old
        # checkpoints still load cleanly; at sampling time we always pin
        # ordering id 0 (the first ordering, e.g. DFS).
        self.use_ordering_id = getattr(
            config.model, 'use_ordering_id', False)
        if self.use_ordering_id:
            self.ordering_embedding = nn.Embedding(
                num_embeddings=max(1, self.num_canonical_order),
                embedding_dim=config.model.embedding_dim,
            )
            nn.init.normal_(self.ordering_embedding.weight, mean=0.0, std=0.1)

        # ---- dimension-reduction input embed (unchanged) ------------------
        # Adjacency-row inputs have width N (max_num_nodes). When
        # dimension_reduce=True we project them down to embedding_dim first
        # so the GNN input width is fixed.
        if self.dimension_reduce:
            self.embedding_dim = config.model.embedding_dim  # e.g. 64
            self.decoder_input = nn.Sequential(
                nn.Linear(self.max_num_nodes, self.embedding_dim))
        else:
            self.embedding_dim = self.max_num_nodes

        # ---- decoder: GNN (default) OR GATv2 (opt-in) ---------------------
        # Both backbones implement the same forward signature:
        #     (node_feat: (N,H), edges: (E,2), edge_feat: (E,2*att_edge_dim))
        #         -> (N, H)
        if self.use_gatv2:
            # GATv2 wraps torch_geometric.nn.GATv2Conv; see model/gatv2.py.
            self.decoder = GATv2(
                node_state_dim=self.hidden_dim,
                edge_feat_dim=2 * self.att_edge_dim,
                num_heads=self.gatv2_num_heads,
                num_layer=self.num_GNN_layers,
                has_residual=True,
                dropout=0.0,
            )
        else:
            self.decoder = GNN(
                msg_dim=self.hidden_dim,
                node_state_dim=self.hidden_dim,
                edge_feat_dim=2 * self.att_edge_dim,
                num_prop=self.num_GNN_prop,
                num_layer=self.num_GNN_layers,
                has_attention=self.has_attention,
            )

        # ---- losses -------------------------------------------------------
        pos_weight = torch.ones([1]) * self.edge_weight
        self.adj_loss_func = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction='none')

        # Per-class weights for the attribute cross-entropy. Three options:
        #   None / unset       -> uniform (behaves like plain CrossEntropyLoss)
        #   list / tuple of A  -> fixed weights, used verbatim
        #   the string "auto"  -> placeholder uniform; runner is expected to
        #                         compute inverse-frequency weights from the
        #                         training set and call
        #                         ``model.set_attr_class_weights(w)``.
        # Stored as a buffer so DataParallel / .to(device) / state_dict work.
        self.attr_class_weight_mode = getattr(
            config.model, 'attr_class_weight', None)
        attr_w = self._build_initial_attr_class_weights(
            self.attr_class_weight_mode, self.num_attr_classes)
        self.register_buffer('attr_class_weights', attr_w)

    # ======================================================================
    # Attribute class weight helpers
    # ======================================================================
    @staticmethod
    def _build_initial_attr_class_weights(mode, num_classes):
        """Return the tensor to register as ``attr_class_weights``.

        Called exactly once in ``__init__``. The resulting tensor is
        *always* registered as a buffer so behavior is uniform across code
        paths (DataParallel replication, checkpointing, etc.) — None vs
        'auto' vs explicit list is collapsed into a length-``num_classes``
        float tensor of per-class weights.

        Mapping:
            None / unset      -> ones (no-op under CrossEntropyLoss)
            list / tuple      -> copied verbatim (must have length num_classes)
            'auto'            -> ones placeholder (runner fills via setter)
        """
        if mode is None:
            return torch.ones(num_classes, dtype=torch.float32)
        if isinstance(mode, (list, tuple)):
            if len(mode) != num_classes:
                raise ValueError(
                    'attr_class_weight list length %d != num_attr_classes %d'
                    % (len(mode), num_classes))
            return torch.tensor(list(mode), dtype=torch.float32)
        if isinstance(mode, str) and mode == 'auto':
            return torch.ones(num_classes, dtype=torch.float32)
        raise ValueError(
            'Invalid attr_class_weight: %r (expected None, list, or "auto")'
            % (mode,))

    def set_attr_class_weights(self, weights):
        """Overwrite the in-place attribute class weight buffer.

        Intended to be called by the runner exactly once, right after
        model construction, when ``config.model.attr_class_weight == 'auto'``.
        Input shape: (num_attr_classes,) float tensor. The copy is in-place
        so DataParallel replicas (which share buffers via broadcast_buffers
        = True, the default) pick up the update.
        """
        if weights.shape != (self.num_attr_classes,):
            raise ValueError(
                'set_attr_class_weights expects shape (%d,), got %s'
                % (self.num_attr_classes, tuple(weights.shape)))
        self.attr_class_weights.data.copy_(
            weights.to(self.attr_class_weights.device,
                       dtype=self.attr_class_weights.dtype))

    # ======================================================================
    # Training inference (unchanged from original class)
    # ======================================================================
    def _inference(self, A_pad=None, edges=None,
                   node_idx_gnn=None, node_idx_feat=None, att_idx=None):
        """Generate adj in row-wise autoregressive fashion (training path).

        Running shapes (example with B=4, C=1, N=20, H=64):
            A_pad:         (4, 1, 20, 20)
            node_feat:     (80, 64)   after decoder_input, before pad
            padded:        (81, 64)   leading zero row for "new nodes"
            node_state:    (M, 64)    M = len(node_idx_feat) after GNN
            log_theta/log_alpha: (num_preds, L)
            node_state returned too, so the caller can also compute attr loss.
        """
        B, C, N_max, _ = A_pad.shape
        H = self.hidden_dim
        # Keep a reference to the (B, C, N, N) form so we can compute
        # structural features (degree rank) before flattening.
        A_pad_bcnn = A_pad
        A_pad = A_pad.view(B * C * N_max, -1)                   # (B*C*N, N)

        if self.dimension_reduce:
            node_feat = self.decoder_input(A_pad)               # (B*C*N, H)
        else:
            node_feat = A_pad                                    # (B*C*N, N)

        # ---- v2 structural fix: add degree-rank embedding -----------------
        # For each (b, c, n) node, count how many already-generated
        # neighbours it has (sum over last dim of the lower-triangular
        # A_pad), rank them descending, and add an embedding of the rank.
        # Shapes: degrees (B, C, N) -> inv_ranks (B, C, N) -> flat (B*C*N,).
        if self.use_degree_feature:
            degrees = A_pad_bcnn.sum(dim=-1)                     # (B, C, N)
            # argsort twice to get rank of each element (descending = 0
            # is highest-degree).
            ranks_sort = degrees.argsort(dim=-1, descending=True)
            inv_ranks = ranks_sort.argsort(dim=-1)               # (B, C, N)
            inv_ranks = inv_ranks.clamp(
                min=0, max=self.max_num_nodes - 1).long()
            deg_emb = self.degree_embedding(inv_ranks.view(-1))  # (B*C*N, H)
            node_feat = node_feat + deg_emb

        # ---- v2 structural fix (Path 1): add ordering-id embedding --------
        # Every "real" node in ``node_feat`` lives in slot
        #   flat = batch * C * N + order * N + pos
        # so we can recover the ordering index directly from the slot. This
        # keeps the dataset untouched (no new field needed in collate_fn).
        # We add the ordering embedding BEFORE the leading zero-padding row
        # is prepended so the row index used below (after the +1 shift in
        # node_idx_feat) is consistent.
        if self.use_ordering_id:
            # Per-slot ordering ids, shape (B*C*N,) long.
            # flat index = b*C*N + c*N + n -> c = (flat // N) % C.
            flat = torch.arange(
                B * C * N_max, device=node_feat.device, dtype=torch.long)
            order_idx = (flat // N_max) % max(1, C)              # (B*C*N,)
            ord_emb = self.ordering_embedding(order_idx)         # (B*C*N, H)
            node_feat = node_feat + ord_emb

        # Prepend a zero row as feature for the newly-generated nodes.
        node_feat = F.pad(node_feat, (0, 0, 1, 0), 'constant', value=0.0)

        # Symmetry-breaking edge feature (one-hot for new-node flag).
        att_idx = att_idx.view(-1, 1)
        att_edge_feat = torch.zeros(edges.shape[0],
                                    2 * self.att_edge_dim).to(node_feat.device)
        att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
        att_edge_feat = att_edge_feat.scatter(
            1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

        # GNN forward: produces updated per-node hidden states.
        node_state = self.decoder(
            node_feat[node_idx_feat], edges, edge_feat=att_edge_feat)
        # node_state shape: (len(node_idx_feat), H)

        # Pairwise diff between candidate edge endpoints -> edge logits.
        diff = node_state[node_idx_gnn[:, 0], :] - node_state[node_idx_gnn[:, 1], :]
        log_theta = self.output_theta(diff).view(-1, self.num_mix_component)
        log_alpha = self.output_alpha(diff).view(-1, self.num_mix_component)
        return log_theta, log_alpha, node_state

    # ======================================================================
    # Autoregressive sampling with optional partial-graph conditioning
    # ======================================================================
    def _sampling(self, B, partial_A=None, partial_attrs=None,
                  fixed_edges_next=None, skip_edge_sampling=False,
                  return_attr_logits=False, start_idx=0):
        """Generate adjacency + node attrs row-by-row.

        Args:
            B:             batch size.
            partial_A:     optional (B, n_partial, n_partial) float tensor of
                           known edges. When given, the top-left block of A
                           is initialized with partial_A and generation only
                           predicts the *remaining* rows.
            partial_attrs: optional (B, n_partial) long tensor of known
                           attribute labels for the first n_partial nodes.
            fixed_edges_next: optional (B, jj_max) float binary tensor. When
                           ``skip_edge_sampling=True``, this replaces the
                           Bernoulli draw for the first autoregressive step
                           after ``start_idx``. Only the first ``jj`` columns
                           are consumed — extra columns are ignored. Shape is
                           conventionally (B, n_partial) for the Mode-B path
                           where the user specifies which of the existing
                           ``n_partial`` nodes the new node connects to.
            skip_edge_sampling: when True, the first iteration writes
                           ``fixed_edges_next`` into A instead of sampling
                           from the mixture-of-Bernoulli. Subsequent
                           iterations (if any) fall back to normal sampling.
            return_attr_logits: when True, also return a per-node attribute
                           logits tensor of shape (B, N_pad, A). Slots that
                           were never populated (either padding or never
                           predicted) contain zeros.
            start_idx:     autoregressive loop starts from max(start_idx,
                           n_partial). Allows skipping ahead (e.g. to resume).

        Returns:
            A:          (B, N_pad, N_pad) float     symmetric adjacency
            node_attrs: (B, N_pad) long             predicted attribute class
            attr_logits (optional): (B, N_pad, A) float, only when
                        ``return_attr_logits=True``.

        Example (B=2, N=20, K=1, A=5):
            A returned shape:          (2, 20, 20)
            node_attrs returned shape: (2, 20), values in [0, 5).
        """
        with torch.no_grad():
            K = self.block_size
            S = self.sample_stride
            H = self.hidden_dim
            N = self.max_num_nodes

            # Round N up so the loop produces a whole number of blocks.
            mod_val = (N - K) % S
            if mod_val > 0:
                N_pad = N - K - mod_val + int(np.ceil((K + mod_val) / S)) * S
            else:
                N_pad = N

            # Blank canvases for the adjacency and the predicted attr labels.
            # A:          (B, N_pad, N_pad)
            # node_attrs: (B, N_pad) long
            A = torch.zeros(B, N_pad, N_pad).to(self.device)
            node_attrs = torch.zeros(B, N_pad, dtype=torch.long).to(self.device)
            # Optional per-node attr logits cache, populated when the caller
            # sets return_attr_logits=True. Shape: (B, N_pad, A).
            attr_logits_all = None
            if return_attr_logits:
                attr_logits_all = torch.zeros(
                    B, N_pad, self.num_attr_classes).to(self.device)

            # ---- partial-graph conditioning ------------------------------
            # Write any known rows into A and any known attrs into node_attrs,
            # then jump the loop start to max(start_idx, n_partial).
            n_partial = 0
            if partial_A is not None:
                n_partial = partial_A.shape[1]
                assert n_partial <= N_pad, "partial_A larger than N_pad"
                A[:, :n_partial, :n_partial] = partial_A.to(self.device)
            if partial_attrs is not None:
                n_pa = partial_attrs.shape[1]
                node_attrs[:, :n_pa] = partial_attrs.to(self.device)
                n_partial = max(n_partial, n_pa)

            loop_start = max(int(start_idx), int(n_partial))
            # Round loop_start down to a multiple of S so the loop boundaries
            # remain consistent with the original algorithm.
            loop_start = (loop_start // S) * S
            # Track whether we've already consumed ``fixed_edges_next``. We
            # only honour it on the FIRST iteration of the loop; later
            # iterations fall back to standard Bernoulli sampling. This keeps
            # Mode B ("predict attr for the next node only") well-defined.
            fixed_edges_consumed = False

            dim_input = self.embedding_dim if self.dimension_reduce else self.max_num_nodes
            # Cache of per-node GNN hidden states, used to speed up repeated
            # decoder_input calls on rows we already finalized.
            # Shape: (B, N_pad, dim_input)
            node_state = torch.zeros(B, N_pad, dim_input).to(self.device)

            for ii in range(loop_start, N_pad, S):
                jj = ii + K
                if jj > N_pad:
                    break

                # Reset rows we haven't drawn yet (keeps things lower-tri).
                A[:, ii:, :] = 0.0
                A = torch.tril(A, diagonal=-1)
                # Put the known partial block back (tril may have wiped it).
                if partial_A is not None and n_partial > 0:
                    A[:, :n_partial, :n_partial] = partial_A.to(self.device)

                # Refresh cached node_state for the rows finalized so far.
                if ii >= K:
                    if self.dimension_reduce:
                        node_state[:, ii - K:ii, :] = self.decoder_input(
                            A[:, ii - K:ii, :N])
                    else:
                        node_state[:, ii - K:ii, :] = A[:, ii - S:ii, :N]
                else:
                    if self.dimension_reduce:
                        node_state[:, :ii, :] = self.decoder_input(A[:, :ii, :N])
                    else:
                        node_state[:, :ii, :] = A[:, ii - S:ii, :N]

                # Pad K zero rows at the tail for the freshly-generated nodes.
                # node_state_in shape: (B, jj, H)
                node_state_in = F.pad(
                    node_state[:, :ii, :], (0, 0, 0, K), 'constant', value=0.0)

                # ---- v2 structural fix (Path 1): add ordering-id embed ----
                # At sampling time there's no "real" ordering — we just pin
                # ordering id 0 (the first canonical ordering used during
                # training, e.g. DFS). The same vector is broadcast to every
                # node in the (B, jj) grid.
                if self.use_ordering_id:
                    ord_idx = torch.zeros(
                        B, jj, device=self.device, dtype=torch.long)
                    ord_emb = self.ordering_embedding(ord_idx)   # (B, jj, H)
                    node_state_in = node_state_in + ord_emb

                # ---- v2 structural fix: add degree-rank embedding ---------
                # At step ``ii`` only the first ``ii`` rows of A are finalised,
                # so we compute degree ranks over that prefix only. The K
                # new-node rows get rank 0 (the default clamp) but we pad
                # their degree contributions with zeros so they never win
                # a "highest rank" tie unless no other node has any edges.
                if self.use_degree_feature and ii > 0:
                    # A[:, :ii, :ii] is the finalised lower-triangular adj
                    # prefix; summing over last dim gives each node's current
                    # degree. Symmetry is only enforced at the very end of
                    # the loop, so sum over dim=-1 + dim=-2 is not needed
                    # here: entries at (r, c) for c > r are all zero in the
                    # ongoing lower-tri canvas.
                    prefix_adj = A[:, :ii, :ii]                   # (B, ii, ii)
                    prefix_deg = (prefix_adj + prefix_adj.transpose(1, 2)).sum(
                        dim=-1)                                   # (B, ii)
                    # Pad the K new-node rows with -1 so argsort(descending)
                    # places them last and inv_ranks for them is ii..ii+K-1,
                    # then clamp to max_num_nodes-1.
                    pad = torch.full((B, K), fill_value=-1.0,
                                     device=self.device, dtype=prefix_deg.dtype)
                    degrees_full = torch.cat([prefix_deg, pad], dim=1)  # (B, jj)
                    ranks_sort = degrees_full.argsort(
                        dim=-1, descending=True)
                    inv_ranks = ranks_sort.argsort(dim=-1)        # (B, jj)
                    inv_ranks = inv_ranks.clamp(
                        min=0, max=self.max_num_nodes - 1).long()
                    deg_emb = self.degree_embedding(inv_ranks)    # (B, jj, H)
                    node_state_in = node_state_in + deg_emb

                # Build the (fully connected, lower-tri) candidate edge list
                # used during one sampling step. Every old node connects to
                # every new node via a candidate edge.
                adj = F.pad(A[:, :ii, :ii], (0, K, 0, K),
                            'constant', value=1.0)            # (B, jj, jj)
                adj = torch.tril(adj, diagonal=-1)
                adj = adj + adj.transpose(1, 2)
                edges = [
                    adj[bb].to_sparse().coalesce().indices() + bb * adj.shape[1]
                    for bb in range(B)
                ]
                edges = torch.cat(edges, dim=1).t()            # (E, 2)

                # One-hot "is new node" marker per node (for symmetry-break).
                att_idx = torch.cat([torch.zeros(ii).long(),
                                     torch.arange(1, K + 1)]).to(self.device)
                att_idx = att_idx.view(1, -1).expand(B, -1).contiguous().view(-1, 1)

                att_edge_feat = torch.zeros(edges.shape[0],
                                            2 * self.att_edge_dim).to(self.device)
                att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
                att_edge_feat = att_edge_feat.scatter(
                    1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

                # GNN forward on the flattened (B*jj, H) node features.
                node_state_out = self.decoder(
                    node_state_in.view(-1, H), edges, edge_feat=att_edge_feat)
                node_state_out = node_state_out.view(B, jj, -1)   # (B, jj, H)

                # Build pairwise diffs between (new node row) and (all cols).
                idx_row, idx_col = np.meshgrid(np.arange(ii, jj), np.arange(jj))
                idx_row = torch.from_numpy(idx_row.reshape(-1)).long().to(self.device)
                idx_col = torch.from_numpy(idx_col.reshape(-1)).long().to(self.device)

                diff = node_state_out[:, idx_row, :] - node_state_out[:, idx_col, :]
                diff = diff.view(-1, node_state.shape[2])
                log_theta = self.output_theta(diff)
                log_alpha = self.output_alpha(diff)

                log_theta = log_theta.view(B, -1, K, self.num_mix_component)
                log_theta = log_theta.transpose(1, 2)           # (B, K, jj, L)

                log_alpha = log_alpha.view(B, -1, self.num_mix_component)
                prob_alpha = F.softmax(log_alpha.mean(dim=1), -1)
                alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1).long()

                if skip_edge_sampling and (not fixed_edges_consumed) \
                        and fixed_edges_next is not None:
                    # ---- Mode B: user specifies which existing nodes the
                    # K new nodes connect to. We write those fixed edges
                    # into A instead of drawing from the mixture-of-Bernoulli.
                    # fixed_edges_next: (B, >=ii) binary — columns beyond
                    # index ``ii`` (the new-node rows themselves) are filled
                    # with zeros so the new nodes don't self-connect.
                    fe = fixed_edges_next.to(self.device).float()
                    assert fe.shape[0] == B, \
                        "fixed_edges_next batch dim must match B"
                    assert fe.shape[1] >= ii, \
                        "fixed_edges_next must have >= ii columns (one per existing node)"
                    # Build a (B, jj) connection row: user-specified for the
                    # first ``ii`` columns, zero for the K new-node columns.
                    fe_row = torch.zeros(B, jj).to(self.device)
                    fe_row[:, :ii] = fe[:, :ii]
                    # Broadcast the same connection pattern to all K new rows.
                    A[:, ii:jj, :jj] = fe_row.unsqueeze(1).expand(-1, K, -1)
                    fixed_edges_consumed = True
                else:
                    # ---- Mode A / unconditional: sample edges from the
                    # mixture of Bernoulli as in the original GRAN.
                    prob = []
                    for bb in range(B):
                        prob += [torch.sigmoid(log_theta[bb, :, :, alpha[bb]])]
                    prob = torch.stack(prob, dim=0)             # (B, K, jj)
                    A[:, ii:jj, :jj] = torch.bernoulli(prob[:, :jj - ii, :])

                # ---- attribute prediction for the new nodes --------------
                # node_state_out[:, ii:jj, :] is (B, K, H); convert to class
                # logits via output_attr, then SAMPLE from the softmax
                # distribution (multinomial). Argmax would collapse diversity
                # — e.g. the most frequent class (Bedroom ~38% in RPLAN) would
                # end up dominating every new node.
                #
                # attr_sampling_temperature (config.model.attr_temperature)
                # scales logits before softmax:
                #   < 1.0  -> sharper  (closer to argmax)
                #   = 1.0  -> default, pure model distribution
                #   > 1.0  -> flatter  (more diverse)
                # Default 1.0 means "trust the model distribution".
                #
                # Limitation: when skip_edge_sampling=True we keep the
                # pre-edge-decision node_state_out (the GNN has not seen the
                # user's fixed edges). A second GNN pass would give a more
                # accurate logit — future work if needed.
                new_node_feat = node_state_out[:, ii:jj, :]     # (B, K, H)
                new_attr_logits = self.output_attr(
                    new_node_feat.reshape(-1, H))               # (B*K, A)
                new_attr_logits = new_attr_logits.view(B, K, -1)

                temp = float(
                    getattr(self.config.model, 'attr_temperature', 1.0) or 1.0)
                scaled_logits = new_attr_logits / temp
                probs = F.softmax(scaled_logits, dim=-1)        # (B, K, A)
                # multinomial works on 2D, so flatten batch/time dims
                flat_probs = probs.view(-1, probs.shape[-1])    # (B*K, A)
                flat_samples = torch.multinomial(flat_probs, 1).squeeze(-1)
                new_attrs = flat_samples.view(B, K)             # (B, K) long

                node_attrs[:, ii:jj] = new_attrs
                if return_attr_logits:
                    attr_logits_all[:, ii:jj, :] = new_attr_logits

            if self.is_sym:
                A = torch.tril(A, diagonal=-1)
                A = A + A.transpose(1, 2)

            if return_attr_logits:
                return A, node_attrs, attr_logits_all
            return A, node_attrs

    # ======================================================================
    # Convenience inference APIs (Task 7)
    # ======================================================================
    @torch.no_grad()
    def expand_graph(self, partial_A, partial_attrs, num_target_nodes):
        """Continue autoregressive generation from a partial graph.

        Mode A of the Task-7 inference APIs. The partial adjacency and
        attribute labels are pinned, and the model samples the remaining
        rows / cols autoregressively until ``num_target_nodes`` is reached.

        Args:
            partial_A:        (B, n_partial, n_partial) float tensor — known
                              adjacency entries. Must be symmetric if the
                              model was built with ``is_sym=True``.
            partial_attrs:    (B, n_partial) long tensor — known attribute
                              labels, values in ``[0, num_attr_classes)``.
            num_target_nodes: int, must be ``>= n_partial`` and
                              ``<= max_num_nodes``. The final graph has
                              exactly this many nodes.

        Returns:
            A:     (B, num_target_nodes, num_target_nodes) float adjacency.
                   The top-left ``n_partial`` block equals ``partial_A``.
            attrs: (B, num_target_nodes) long tensor; the first ``n_partial``
                   entries equal ``partial_attrs``.
        """
        B = partial_A.shape[0]
        n_partial = partial_A.shape[1]
        assert num_target_nodes >= n_partial, \
            "num_target_nodes must be >= n_partial"
        assert num_target_nodes <= self.max_num_nodes, \
            "num_target_nodes must be <= max_num_nodes"

        A, attrs = self._sampling(
            B,
            partial_A=partial_A.to(self.device),
            partial_attrs=partial_attrs.to(self.device),
            start_idx=n_partial,
        )
        return (A[:, :num_target_nodes, :num_target_nodes],
                attrs[:, :num_target_nodes])

    @torch.no_grad()
    def predict_attr_with_edges(self, partial_A, partial_attrs, fixed_edges):
        """Predict the attribute logits of a NEW node whose edges are fixed.

        Mode B of the Task-7 inference APIs. Given a partial graph and a
        user-specified connection pattern for the next node, the model
        skips the Bernoulli edge draw and returns the raw attribute logits
        produced by ``self.output_attr`` for that new node.

        Args:
            partial_A:     (B, n_partial, n_partial) float adjacency.
            partial_attrs: (B, n_partial) long attribute labels.
            fixed_edges:   (B, n_partial) binary float/long tensor — 1 where
                           the new node connects to an existing node, 0
                           otherwise.

        Returns:
            attr_logits: (B, num_attr_classes) float — raw (pre-softmax)
                         logits for the new node's attribute class. Apply
                         softmax to obtain a probability distribution.
        """
        B = partial_A.shape[0]
        n_partial = partial_A.shape[1]
        assert fixed_edges.shape == (B, n_partial), \
            "fixed_edges must have shape (B, n_partial)"
        assert n_partial + self.block_size <= self.max_num_nodes, \
            "partial graph leaves no room for a new node"

        # One autoregressive step with the user-specified edges pinned.
        # The GNN sees the partial graph (without the new edges) and
        # predicts the new node's attr from its pre-edge-decision hidden
        # state — see the limitation note in ``_sampling``.
        _, _, attr_logits_all = self._sampling(
            B,
            partial_A=partial_A.to(self.device),
            partial_attrs=partial_attrs.to(self.device),
            fixed_edges_next=fixed_edges.to(self.device).float(),
            skip_edge_sampling=True,
            return_attr_logits=True,
            start_idx=n_partial,
        )
        # attr_logits_all is (B, N_pad, A); the new-node row is index
        # ``n_partial`` (the block_size=1 case) or the first row of a
        # larger block. We collapse the K-block to a single vector by
        # taking the first new row.
        return attr_logits_all[:, n_partial, :]

    # ======================================================================
    # Forward dispatch
    # ======================================================================
    def forward(self, input_dict):
        """Training path returns (edge_loss, attr_loss).
        Sampling path returns (A_list, attr_list).

        See ``GRANMixtureBernoulli.forward`` for key names.
        """
        is_sampling = input_dict.get('is_sampling', False)
        batch_size = input_dict.get('batch_size', None)
        A_pad = input_dict.get('adj', None)
        node_idx_gnn = input_dict.get('node_idx_gnn', None)
        node_idx_feat = input_dict.get('node_idx_feat', None)
        att_idx = input_dict.get('att_idx', None)
        subgraph_idx = input_dict.get('subgraph_idx', None)
        edges = input_dict.get('edges', None)
        label = input_dict.get('label', None)
        num_nodes_pmf = input_dict.get('num_nodes_pmf', None)
        subgraph_idx_base = input_dict.get('subgraph_idx_base', None)
        # Optional v2 inputs
        node_attr_label = input_dict.get('node_attr_label', None)
        node_attr_idx = input_dict.get('node_attr_idx', None)
        # [v2] node_attrs: (B, C, N) int64 produced by GRANDataV2.collate_fn.
        # When provided, the model derives per-subgraph-node labels from
        # ``node_idx_feat`` (see mapping in the training path below) so the
        # runner doesn't need to compute node_attr_label / node_attr_idx.
        node_attrs = input_dict.get('node_attrs', None)
        partial_A = input_dict.get('partial_A', None)
        partial_attrs = input_dict.get('partial_attrs', None)
        start_idx = input_dict.get('start_idx', 0)

        if not is_sampling:
            # ---- training --------------------------------------------------
            B, _, N, _ = A_pad.shape

            log_theta, log_alpha, node_state = self._inference(
                A_pad=A_pad,
                edges=edges,
                node_idx_gnn=node_idx_gnn,
                node_idx_feat=node_idx_feat,
                att_idx=att_idx)

            # Edge (mixture-of-Bernoulli) loss.
            #
            # sum_order_log_prob=True uses `sum log P(pi_c)` across canonical
            # orderings (the ORIGINAL GRAN paper loss). With multi-ordering
            # (num_canonical_order > 1), the default logsumexp formulation
            # produces technically-valid-but-confusing negative values
            # (due to per-edge normalization inside the logsumexp). The
            # `sum` variant keeps loss non-negative and mirrors the paper.
            edge_loss = mixture_bernoulli_loss(
                label, log_theta, log_alpha,
                self.adj_loss_func, subgraph_idx, subgraph_idx_base,
                self.num_canonical_order,
                sum_order_log_prob=True)

            # ---- attribute loss -------------------------------------------
            # Three input shapes are supported, in order of preference:
            #
            #   (a) Explicit (node_attr_label, node_attr_idx) pair — legacy
            #       path where the caller has already flattened labels to
            #       match node_state.
            #   (b) ``node_attrs`` of shape (B, C, N) — produced by
            #       GRANDataV2.collate_fn. We derive per-subgraph-node labels
            #       by inverting the packing convention used for
            #       ``node_idx_feat`` in GRANData.collate_fn:
            #
            #           node_idx_feat[i] = 0                       -> padding
            #           node_idx_feat[i] = v, v > 0
            #               flat   = v - 1
            #               batch  = flat // (C * N)
            #               order  = (flat // N) % C
            #               node_p = flat % N
            #
            #       which gives a label of node_attrs[batch, order, node_p]
            #       per non-padding row of node_state. We then slice out the
            #       non-padding rows and compute cross-entropy.
            #   (c) Neither — return a safe zero tied to the compute graph.
            if node_attr_label is not None and node_attr_idx is not None:
                # (a) explicit labels for specific node_state indices
                attr_feat = node_state[node_attr_idx]           # (M, H)
                attr_logits = self.output_attr(attr_feat)       # (M, A)
                attr_loss = F.cross_entropy(
                    attr_logits, node_attr_label,
                    weight=self.attr_class_weights)
            elif node_attrs is not None and node_idx_feat is not None:
                # (b) derive labels from (B, C, N) attrs + node_idx_feat
                C = self.num_canonical_order
                N = self.max_num_nodes
                # node_state has shape (len(node_idx_feat), H). Rows where
                # node_idx_feat == 0 are the leading zero-padding / new-node
                # placeholder rows; they have no ground-truth attribute and
                # must be masked out.
                valid_mask = node_idx_feat > 0                  # (M,)
                if valid_mask.any():
                    flat = (node_idx_feat[valid_mask] - 1).long()  # (M',)
                    batch_idx = flat // (C * N)
                    order_idx = (flat // N) % C
                    node_pos = flat % N
                    # Gather labels; shape (M',) long in [0, A).
                    labels = node_attrs[batch_idx, order_idx, node_pos]
                    attr_feat = node_state[valid_mask]          # (M', H)
                    attr_logits = self.output_attr(attr_feat)   # (M', A)
                    attr_loss = F.cross_entropy(
                        attr_logits, labels,
                        weight=self.attr_class_weights)
                else:
                    # All rows are padding — unusual, but keep graph-tied.
                    attr_loss = (node_state.sum() * 0.0)
            else:
                # (c) Safe zero scalar tied to the graph so .backward() works
                # even if the caller doesn't wire up attr labels yet.
                attr_loss = (node_state.sum() * 0.0)

            return edge_loss, attr_loss

        # ---- sampling ------------------------------------------------------
        A, attrs = self._sampling(
            batch_size,
            partial_A=partial_A,
            partial_attrs=partial_attrs,
            start_idx=start_idx,
        )

        # Draw per-graph node counts from the empirical PMF.
        num_nodes_pmf_t = torch.from_numpy(num_nodes_pmf).to(self.device)
        num_nodes = torch.multinomial(
            num_nodes_pmf_t, batch_size, replacement=True) + 1  # (B,)

        A_list = [A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)]
        attr_list = [attrs[ii, :num_nodes[ii]] for ii in range(batch_size)]
        return A_list, attr_list
