"""GRANDataV2: extends GRANData to carry per-node attribute labels.

This dataset loader mirrors the original :class:`dataset.gran_data.GRANData`
but additionally tracks a single integer attribute per node (e.g. a room-type
class label).  The attribute is read from ``G.nodes[n]['attr']`` on the input
networkx graphs and re-ordered in lockstep with the adjacency matrices so
that the resulting tensors align with the per-ordering canonical orderings
(DFS / BFS / k-core / degree / ...).

The precomputed pickle files store either:
  * the legacy ``adj_list`` (list of ``np.ndarray``), or
  * the new ``(adj_list, attr_list)`` tuple

so old precompute directories from :class:`GRANData` are still readable
(attributes then default to all zeros).
"""

import os
import time
import glob
import pickle
from collections import defaultdict

import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

from dataset.gran_data import GRANData


class GRANDataV2(GRANData):
    """GRANData variant that additionally returns integer node attributes.

    Each graph ``G`` passed to the constructor is expected to have a
    per-node integer attribute stored at ``G.nodes[n]['attr']``; missing
    attributes default to ``0``.

    The output of :meth:`__getitem__` extends the parent's dict with a
    ``node_attrs`` entry of shape ``(C, N_max)`` (int64) where ``C`` is
    ``num_canonical_order`` and ``N_max`` is ``max_num_nodes``.  Unused
    slots past the real node count are zero-padded.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, config, graphs, tag='train'):
        """Build (or load) the precomputed attributed graph shards.

        We do NOT call ``super().__init__`` because the parent constructor
        dumps only ``adj_list`` to disk; we need to dump
        ``(adj_list, attr_list)`` so we replicate its bookkeeping here.

        Shapes / types of persisted objects:
          * ``adj_list``  : list[np.ndarray of shape (N, N)] length = C
          * ``attr_list`` : list[np.ndarray of shape (N,) int64] length = C
        """
        self.config = config
        self.data_path = config.dataset.data_path
        self.model_name = config.model.name
        self.max_num_nodes = config.model.max_num_nodes
        self.block_size = config.model.block_size
        self.stride = config.model.sample_stride

        self.graphs = graphs
        self.num_graphs = len(graphs)
        self.npr = np.random.RandomState(config.seed)
        self.node_order = config.dataset.node_order
        self.num_canonical_order = config.model.num_canonical_order
        self.tag = tag
        self.num_fwd_pass = config.dataset.num_fwd_pass
        self.is_sample_subgraph = config.dataset.is_sample_subgraph
        self.num_subgraph_batch = config.dataset.num_subgraph_batch
        self.is_overwrite_precompute = config.dataset.is_overwrite_precompute

        if self.is_sample_subgraph:
            assert self.num_subgraph_batch > 0

        self.save_path = os.path.join(
            self.data_path, '{}_{}_{}_{}_{}_{}_{}_precompute'.format(
                config.model.name, config.dataset.name, tag, self.block_size,
                self.stride, self.num_canonical_order, self.node_order))

        if not os.path.isdir(self.save_path) or self.is_overwrite_precompute:
            self.file_names = []
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)

            self.config.dataset.save_path = self.save_path
            for index in tqdm(range(self.num_graphs)):
                G = self.graphs[index]
                data = self._get_graph_data(G)  # (adj_list, attr_list)
                tmp_path = os.path.join(self.save_path,
                                        '{}_{}.p'.format(tag, index))
                pickle.dump(data, open(tmp_path, 'wb'))
                self.file_names += [tmp_path]
        else:
            self.file_names = glob.glob(os.path.join(self.save_path, '*.p'))

    # ------------------------------------------------------------------
    # Graph preprocessing
    # ------------------------------------------------------------------
    def _get_graph_data(self, G):
        """Compute canonical orderings and reorder both adj and attrs.

        Returns:
            adj_list  : list[np.ndarray (N, N) float] length = C
            attr_list : list[np.ndarray (N,) int64]   length = C
        where each ``attr_list[i]`` is the per-node attribute vector
        reordered to match ``adj_list[i]``'s node ordering.
        """
        # Read per-node attribute (default 0 when missing).
        # attr_arr shape: (N,)
        attr_arr = np.array(
            [int(G.nodes[n].get('attr', 0)) for n in G.nodes()],
            dtype=np.int64,
        )
        # Map from node id -> position in G.nodes() so we can translate
        # reorderings (which are expressed as node-id lists) into index
        # permutations of attr_arr.
        node_to_pos = {n: i for i, n in enumerate(G.nodes())}

        def reorder_attrs(nodelist):
            """Permute ``attr_arr`` to match ``nodelist`` ordering.

            nodelist : iterable of node ids (length = N)
            returns  : np.ndarray (N,) int64
            """
            return np.array(
                [attr_arr[node_to_pos[n]] for n in nodelist],
                dtype=np.int64,
            )

        # ``nx.to_numpy_matrix`` was removed in networkx >= 3.0; use the
        # replacement ``to_numpy_array``.  We keep a local alias so the
        # rest of this method reads like the parent implementation.
        _to_np = getattr(nx, 'to_numpy_array', None) or nx.to_numpy_matrix

        # ---- Original ordering (adj_0) --------------------------------
        original_nodelist = list(G.nodes())
        adj_0 = np.array(_to_np(G))
        attr_0 = attr_arr.copy()  # same ordering as G.nodes()

        node_degree_list = [(n, d) for n, d in G.degree()]

        # ---- Degree descent (adj_1) -----------------------------------
        degree_sequence_desc = sorted(
            node_degree_list, key=lambda tt: tt[1], reverse=True)
        nodelist_1 = [dd[0] for dd in degree_sequence_desc]
        adj_1 = np.array(_to_np(G, nodelist=nodelist_1))
        attr_1 = reorder_attrs(nodelist_1)

        # ---- Degree ascent (adj_2) ------------------------------------
        degree_sequence_asc = sorted(node_degree_list, key=lambda tt: tt[1])
        nodelist_2 = [dd[0] for dd in degree_sequence_asc]
        adj_2 = np.array(_to_np(G, nodelist=nodelist_2))
        attr_2 = reorder_attrs(nodelist_2)

        # ---- BFS & DFS from largest-degree node in each CC ------------
        CGs = [G.subgraph(c) for c in nx.connected_components(G)]
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

        node_list_bfs = []
        node_list_dfs = []
        for ii in range(len(CGs)):
            cc_degree_list = [(n, d) for n, d in CGs[ii].degree()]
            cc_degree_sorted = sorted(
                cc_degree_list, key=lambda tt: tt[1], reverse=True)
            bfs_tree = nx.bfs_tree(CGs[ii], source=cc_degree_sorted[0][0])
            dfs_tree = nx.dfs_tree(CGs[ii], source=cc_degree_sorted[0][0])
            node_list_bfs += list(bfs_tree.nodes())
            node_list_dfs += list(dfs_tree.nodes())

        adj_3 = np.array(_to_np(G, nodelist=node_list_bfs))
        adj_4 = np.array(_to_np(G, nodelist=node_list_dfs))
        attr_3 = reorder_attrs(node_list_bfs)
        attr_4 = reorder_attrs(node_list_dfs)

        # ---- k-core (adj_5) -------------------------------------------
        num_core = nx.core_number(G)
        core_order_list = sorted(list(set(num_core.values())), reverse=True)
        degree_dict = dict(G.degree())
        core_to_node = defaultdict(list)
        for nn, kk in num_core.items():
            core_to_node[kk] += [nn]

        node_list_5 = []
        for kk in core_order_list:
            sort_node_tuple = sorted(
                [(nn, degree_dict[nn]) for nn in core_to_node[kk]],
                key=lambda tt: tt[1],
                reverse=True,
            )
            node_list_5 += [nn for nn, dd in sort_node_tuple]

        adj_5 = np.array(_to_np(G, nodelist=node_list_5))
        attr_5 = reorder_attrs(node_list_5)

        # ---- Select orderings (mirror parent class logic) -------------
        if self.num_canonical_order == 5:
            adj_list = [adj_0, adj_1, adj_3, adj_4, adj_5]
            attr_list = [attr_0, attr_1, attr_3, attr_4, attr_5]
        else:
            if self.node_order == 'degree_decent':
                adj_list = [adj_1]
                attr_list = [attr_1]
            elif self.node_order == 'degree_accent':
                adj_list = [adj_2]
                attr_list = [attr_2]
            elif self.node_order == 'BFS':
                adj_list = [adj_3]
                attr_list = [attr_3]
            elif self.node_order == 'DFS':
                adj_list = [adj_4]
                attr_list = [attr_4]
            elif self.node_order == 'k_core':
                adj_list = [adj_5]
                attr_list = [attr_5]
            elif self.node_order == 'DFS+BFS':
                adj_list = [adj_4, adj_3]
                attr_list = [attr_4, attr_3]
            elif self.node_order == 'DFS+BFS+k_core':
                adj_list = [adj_4, adj_3, adj_5]
                attr_list = [attr_4, attr_3, attr_5]
            elif self.node_order == 'DFS+BFS+k_core+degree_decent':
                adj_list = [adj_4, adj_3, adj_5, adj_1]
                attr_list = [attr_4, attr_3, attr_5, attr_1]
            elif self.node_order == 'all':
                adj_list = [adj_4, adj_3, adj_5, adj_1, adj_0]
                attr_list = [attr_4, attr_3, attr_5, attr_1, attr_0]
            else:
                adj_list = [adj_0]
                attr_list = [attr_0]

        return adj_list, attr_list

    # ------------------------------------------------------------------
    # Per-sample access
    # ------------------------------------------------------------------
    def _load_shard(self, path):
        """Load a pickle shard with backward-compat handling.

        Returns:
            adj_list  : list[np.ndarray (N, N)]
            attr_list : list[np.ndarray (N,) int64]
        """
        payload = pickle.load(open(path, 'rb'))
        if isinstance(payload, tuple) and len(payload) == 2:
            # New format: (adj_list, attr_list)
            adj_list, attr_list = payload
        else:
            # Legacy format: just adj_list. Fill attrs with zeros so the
            # pipeline still runs end-to-end without crashing.
            adj_list = payload
            attr_list = [
                np.zeros(adj.shape[0], dtype=np.int64) for adj in adj_list
            ]
        return adj_list, attr_list

    def __getitem__(self, index):
        """Return a list of ``num_fwd_pass`` data dicts (same as parent).

        In addition to the parent fields, every dict contains:
          * ``node_attrs`` : np.ndarray of shape (C, N_max) int64
            (converted to torch.int64 tensor in :meth:`collate_fn`)
        """
        K = self.block_size
        N = self.max_num_nodes
        S = self.stride
        C = self.num_canonical_order

        # ---- Load shard (supports legacy & new formats) ---------------
        adj_list, attr_list = self._load_shard(self.file_names[index])
        num_nodes = adj_list[0].shape[0]
        num_subgraphs = int(np.floor((num_nodes - K) / S) + 1)

        if self.is_sample_subgraph:
            if self.num_subgraph_batch < num_subgraphs:
                num_subgraphs_pass = int(
                    np.floor(self.num_subgraph_batch / self.num_fwd_pass))
            else:
                num_subgraphs_pass = int(
                    np.floor(num_subgraphs / self.num_fwd_pass))
            end_idx = min(num_subgraphs, self.num_subgraph_batch)
        else:
            num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))
            end_idx = num_subgraphs

        rand_perm_idx = self.npr.permutation(num_subgraphs).tolist()

        # ---- Pre-pad node_attrs once; shape (C, N_max) int64 ---------
        # Each row is ordering i's attrs, 0-padded out to N_max.
        node_attrs_padded = np.zeros((C, N), dtype=np.int64)
        for i in range(C):
            n_i = attr_list[i].shape[0]
            node_attrs_padded[i, :n_i] = attr_list[i]

        start_time = time.time()
        data_batch = []
        for ff in range(self.num_fwd_pass):
            ff_idx_start = num_subgraphs_pass * ff
            if ff == self.num_fwd_pass - 1:
                ff_idx_end = end_idx
            else:
                ff_idx_end = (ff + 1) * num_subgraphs_pass

            rand_idx = rand_perm_idx[ff_idx_start:ff_idx_end]

            edges = []
            node_idx_gnn = []
            node_idx_feat = []
            label = []
            subgraph_size = []
            subgraph_idx = []
            att_idx = []
            subgraph_node_attrs = []   # Plan C-1: aligned with node_state rows
            subgraph_count = 0

            for ii in range(len(adj_list)):
                adj_full = adj_list[ii]

                idx = -1
                for jj in range(0, num_nodes, S):
                    idx += 1
                    if jj + K > num_nodes:
                        break
                    if idx not in rand_idx:
                        continue

                    adj_block = np.pad(
                        adj_full[:jj, :jj], ((0, K), (0, K)),
                        'constant',
                        constant_values=1.0)
                    adj_block = np.tril(adj_block, k=-1)
                    adj_block = adj_block + adj_block.transpose()
                    adj_block = torch.from_numpy(adj_block).to_sparse()
                    edges += [adj_block.coalesce().indices().long()]

                    if jj == 0:
                        att_idx += [np.arange(1, K + 1).astype(np.uint8)]
                    else:
                        att_idx += [
                            np.concatenate([
                                np.zeros(jj).astype(np.uint8),
                                np.arange(1, K + 1).astype(np.uint8),
                            ])
                        ]

                    if jj == 0:
                        node_idx_feat += [np.ones(K) * np.inf]
                    else:
                        node_idx_feat += [
                            np.concatenate([
                                np.arange(jj) + ii * N,
                                np.ones(K) * np.inf,
                            ])
                        ]

                    idx_row_gnn, idx_col_gnn = np.meshgrid(
                        np.arange(jj, jj + K), np.arange(jj + K))
                    idx_row_gnn = idx_row_gnn.reshape(-1, 1)
                    idx_col_gnn = idx_col_gnn.reshape(-1, 1)
                    node_idx_gnn += [
                        np.concatenate([idx_row_gnn, idx_col_gnn],
                                       axis=1).astype(np.int64)
                    ]

                    label += [
                        adj_full[idx_row_gnn, idx_col_gnn].flatten().astype(
                            np.uint8)
                    ]

                    # ---- Plan C-1: per-subgraph-row attr label ----------
                    # node_state for this subgraph has jj+K rows in this
                    # canonical ordering. Existing-node rows take their
                    # real attr; new-node rows take their GROUND-TRUTH attr
                    # (teacher forcing). Positions beyond n_real_nodes
                    # (rare -- only when the graph is shorter than jj+K)
                    # take the "unknown" slot id = num_attr_classes.
                    # Backward-compat: if the config has no attr-class
                    # count, fall back to 0 for the unknown slot (legacy
                    # consumers ignore this field anyway).
                    A_classes = getattr(
                        self.config.model, 'num_attr_classes', 0)
                    n_real = attr_list[ii].shape[0]
                    sg_attrs = np.full(jj + K, A_classes, dtype=np.int64)
                    take_n = min(jj + K, n_real)
                    sg_attrs[:take_n] = attr_list[ii][:take_n]
                    subgraph_node_attrs.append(sg_attrs)

                    subgraph_size += [jj + K]
                    subgraph_idx += [
                        np.ones_like(label[-1]).astype(np.int64) *
                        subgraph_count
                    ]
                    subgraph_count += 1

            cum_size = np.cumsum([0] + subgraph_size).astype(np.int64)
            for ii in range(len(edges)):
                edges[ii] = edges[ii] + cum_size[ii]
                node_idx_gnn[ii] = node_idx_gnn[ii] + cum_size[ii]

            data = {}
            # data['adj'] shape: (C, N, N) after tril
            data['adj'] = np.tril(np.stack(adj_list, axis=0), k=-1)
            data['edges'] = torch.cat(edges, dim=1).t().long() \
                if len(edges) > 0 else torch.zeros((0, 2), dtype=torch.long)
            data['node_idx_gnn'] = np.concatenate(node_idx_gnn) \
                if len(node_idx_gnn) > 0 else np.zeros((0, 2), dtype=np.int64)
            data['node_idx_feat'] = np.concatenate(node_idx_feat) \
                if len(node_idx_feat) > 0 else np.zeros((0,))
            data['label'] = np.concatenate(label) \
                if len(label) > 0 else np.zeros((0,), dtype=np.uint8)
            data['att_idx'] = np.concatenate(att_idx) \
                if len(att_idx) > 0 else np.zeros((0,), dtype=np.uint8)
            data['subgraph_idx'] = np.concatenate(subgraph_idx) \
                if len(subgraph_idx) > 0 else np.zeros((0,), dtype=np.int64)
            data['subgraph_count'] = subgraph_count
            data['num_nodes'] = num_nodes
            data['subgraph_size'] = subgraph_size
            data['num_count'] = sum(subgraph_size)
            # NEW: per-ordering attribute labels, 0-padded to N_max.
            # shape: (C, N_max) int64
            data['node_attrs'] = node_attrs_padded
            # Plan C-1: per-state-row attr labels, aligned with node_idx_feat.
            # shape: (M,) int64 where M = sum_subgraphs(jj+K).
            data['subgraph_node_attrs'] = np.concatenate(subgraph_node_attrs) \
                if len(subgraph_node_attrs) > 0 \
                else np.zeros((0,), dtype=np.int64)
            data_batch += [data]

        end_time = time.time()
        return data_batch

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------
    def collate_fn(self, batch):
        """Batch a list of samples into tensors.

        Delegates the existing fields to :meth:`GRANData.collate_fn` and
        then stacks ``node_attrs`` into a ``(B, C, N_max)`` int64 tensor.
        """
        batch_data = super().collate_fn(batch)

        for ff in range(self.num_fwd_pass):
            # Each bb[ff]['node_attrs'] is (C, N_max) int64; stacking on
            # axis 0 yields (B, C, N_max).
            stacked = np.stack(
                [bb[ff]['node_attrs'] for bb in batch], axis=0)
            batch_data[ff]['node_attrs'] = torch.from_numpy(stacked).long()

            # ---- Plan C-1: concat per-sample subgraph_node_attrs ----
            # Each bb[ff]['subgraph_node_attrs'] is a 1D int64 array of
            # variable length; concat across batch yields a flat (M_total,)
            # tensor aligned with batch_data[ff]['node_idx_feat'].
            sgna_list = [bb[ff]['subgraph_node_attrs'] for bb in batch]
            batch_data[ff]['subgraph_node_attrs'] = torch.from_numpy(
                np.concatenate(sgna_list)).long()

        return batch_data
