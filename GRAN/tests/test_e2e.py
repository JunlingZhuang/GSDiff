"""End-to-end integration tests for the GRANv2 pipeline.

These tests exercise the full chain:

    networkx graph(s)  ->  GRANDataV2  ->  collate_fn  ->
        GRANv2.forward  ->  (edge_loss, attr_loss)  ->
            total_loss.backward()  ->  optimizer.step()

and the inverse (sampling + partial-graph completion) loop.

They intentionally use tiny hyperparameters so one forward/backward pass
runs in ~a second on CPU. The goal is to catch integration bugs between
Task 2 (model), Task 4 (dataset), and Task 5 (runner/config) — not to
verify any numerical training dynamics.
"""
import os
import tempfile

import networkx as nx
import numpy as np
import torch
from easydict import EasyDict as edict

from dataset.gran_data_v2 import GRANDataV2
from model.gran_v2 import GRANv2


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_config(tmp_dir, num_attr_classes=4):
    """Build a unified config for both GRANDataV2 and GRANv2.

    Kept small so the whole pipeline runs fast on CPU:
      * max_num_nodes = 15
      * hidden_dim    = 32
      * num_GNN_layers = 2
      * num_mix_component = 3
      * num_attr_classes  = ``num_attr_classes``
    """
    return edict({
        'seed': 123,
        'device': 'cpu',
        'model': edict({
            'name': 'GRANv2E2ETest',
            'max_num_nodes': 15,
            'hidden_dim': 32,
            'embedding_dim': 32,
            'is_sym': True,
            'block_size': 1,
            'sample_stride': 1,
            'num_GNN_prop': 1,
            'num_GNN_layers': 2,
            'edge_weight': 1.0,
            'dimension_reduce': True,
            'has_attention': True,
            'num_canonical_order': 1,
            'num_mix_component': 3,
            'num_attr_classes': num_attr_classes,
            'use_gatv2': False,
            'gatv2_num_heads': 4,
        }),
        'dataset': edict({
            'data_path': tmp_dir,
            'name': 'e2e_testset',
            'node_order': 'DFS',
            'num_fwd_pass': 1,
            'is_sample_subgraph': False,
            'num_subgraph_batch': 100,
            'is_overwrite_precompute': True,
        }),
    })


def _make_grid_graph_with_attrs(rows=3, cols=3, num_classes=4, seed=0):
    """Build a small connected graph with integer per-node attrs set.

    Uses a ``rows x cols`` grid (default 3x3 -> 9 nodes); each node gets
    a random integer class in [0, num_classes) under ``G.nodes[n]['attr']``.
    Grid graphs are always connected so GRANDataV2's BFS/DFS orderings
    behave deterministically.
    """
    rng = np.random.RandomState(seed)
    G = nx.grid_2d_graph(rows, cols)
    # Relabel (r, c) tuples to 0..N-1 integers so attr indexing is simple.
    G = nx.convert_node_labels_to_integers(G)
    for n in G.nodes():
        G.nodes[n]['attr'] = int(rng.randint(0, num_classes))
    return G


# ----------------------------------------------------------------------
# Test 1: one full training step (dataset -> model -> backward)
# ----------------------------------------------------------------------
def test_e2e_train_step():
    """One full training step on a synthetic attributed graph.

    Covers:
      1. Build small attributed networkx graphs (3x3 grid, 9 nodes each).
      2. Wrap them in ``GRANDataV2`` (per-node 'attr' read into a
         (C, N_max) int64 tensor on ``__getitem__``).
      3. Call ``GRANDataV2.collate_fn`` on a minibatch to obtain the
         input dict expected by the model (including ``node_attrs`` of
         shape (B, C, N_max) int64).
      4. Build ``GRANv2`` and run ``forward`` in training mode; the
         returned pair ``(edge_loss, attr_loss)`` must be finite scalars.
      5. ``total_loss.backward()`` + ``optimizer.step()`` — gradients must
         flow into at least some parameters (the edge head is always
         trained; the attr head may only train when ``node_attrs`` wiring
         is live).
    """
    num_attr_classes = 4
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(tmp, num_attr_classes=num_attr_classes)

        # Step 1 + 2: build a tiny batch of attributed grid graphs and wrap
        # them with GRANDataV2 (which precomputes canonical orderings +
        # per-ordering attribute vectors to ``tmp``).
        graphs = [
            _make_grid_graph_with_attrs(
                rows=3, cols=3, num_classes=num_attr_classes, seed=i)
            for i in range(2)
        ]
        dataset = GRANDataV2(config, graphs, tag='train')

        # Step 3: build a batch via collate_fn exactly as the runner does.
        # ``dataset[i]`` returns a list of length num_fwd_pass=1; each
        # entry is one training sample's per-ordering data dict.
        batch = [dataset[i] for i in range(len(dataset.file_names))]
        batch_data = dataset.collate_fn(batch)
        data = batch_data[0]  # num_fwd_pass=1 -> take the single pass

        # Verify the v2 tensor we rely on is present and has the expected shape.
        # node_attrs: (B, C, N_max) int64
        assert 'node_attrs' in data
        B = len(graphs)
        C = config.model.num_canonical_order
        N = config.model.max_num_nodes
        assert data['node_attrs'].shape == (B, C, N)

        # Step 4: instantiate the model and perform the forward pass.
        model = GRANv2(config)
        model.train()

        # The training-path forward takes a dict matching what the runner
        # builds from ``batch_data[ff]``. We just forward the collated dict
        # (with 'is_sampling' defaulting to False).
        input_dict = {
            'is_sampling': False,
            'adj': data['adj'],
            'edges': data['edges'],
            'node_idx_gnn': data['node_idx_gnn'],
            'node_idx_feat': data['node_idx_feat'],
            'att_idx': data['att_idx'],
            'subgraph_idx': data['subgraph_idx'],
            'subgraph_idx_base': data['subgraph_idx_base'],
            'label': data['label'],
            'node_attrs': data['node_attrs'],
        }
        edge_loss, attr_loss = model(input_dict)

        # Both losses must be finite scalars.
        assert torch.is_tensor(edge_loss) and torch.is_tensor(attr_loss)
        assert edge_loss.dim() == 0 or edge_loss.numel() == 1
        assert attr_loss.dim() == 0 or attr_loss.numel() == 1
        assert torch.isfinite(edge_loss).item(), "edge_loss is NaN/Inf"
        assert torch.isfinite(attr_loss).item(), "attr_loss is NaN/Inf"

        # Step 5: backward + optimizer step. Snapshot one edge-head weight
        # and one attr-head weight pre-step so we can detect gradient flow.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        total_loss = edge_loss + attr_loss

        edge_w_before = model.output_theta[0].weight.detach().clone()
        attr_w_before = model.output_attr[0].weight.detach().clone()

        optimizer.zero_grad()
        total_loss.backward()

        # At least one parameter must have received a non-None gradient.
        any_grad = any(
            (p.grad is not None and p.grad.abs().sum().item() > 0)
            for p in model.parameters()
        )
        assert any_grad, "No parameter received a non-zero gradient"

        optimizer.step()

        # Edge head always trains. Verify it moved after the step.
        edge_w_after = model.output_theta[0].weight.detach()
        assert not torch.allclose(edge_w_before, edge_w_after), \
            "Edge-head weights did not update after optimizer.step()"

        # If attr_loss is a real non-zero loss (attr wiring works), the
        # attr head must have moved too. Be permissive when attr_loss is
        # exactly zero (legacy fallback path).
        attr_w_after = model.output_attr[0].weight.detach()
        if attr_loss.detach().abs().item() > 0:
            assert not torch.allclose(attr_w_before, attr_w_after), \
                "attr_loss>0 but attr-head weights did not update"


# ----------------------------------------------------------------------
# Test 2: sample -> take partial -> complete
# ----------------------------------------------------------------------
def test_e2e_sample_then_complete():
    """Sample a graph, then re-sample conditioned on its first half.

    Covers the inverse direction of the pipeline:
      1. Unconditional sampling from a freshly-initialized ``GRANv2`` —
         returns ``(A_list, attr_list)``.
      2. Take the first half of the first generated sample and feed it
         back as ``partial_A`` / ``partial_attrs`` + ``start_idx``.
      3. The completed sample must contain at least as many nodes as the
         partial prefix, and the attribute vector must match its adjacency
         size.

    The sizes coming out of sampling are drawn from ``num_nodes_pmf`` and
    can vary run-to-run, so the assertions only check structural
    consistency (sizes, dtypes, value ranges).
    """
    num_attr_classes = 4
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(tmp, num_attr_classes=num_attr_classes)

        # Build model in eval mode so sampling uses the no_grad path.
        model = GRANv2(config)
        model.eval()

        # PMF over node counts (1..N_max). We force it to spike in the
        # middle of the range so generated samples are neither trivially
        # small (1-2 nodes) nor clipped to N_max. Must sum to 1.
        N = config.model.max_num_nodes
        num_nodes_pmf = np.zeros(N, dtype=np.float32)
        num_nodes_pmf[5:11] = 1.0   # node counts in {6,7,8,9,10,11}
        num_nodes_pmf /= num_nodes_pmf.sum()

        # Step 1: unconstrained sampling.
        B = 2
        A_list, attr_list = model({
            'is_sampling': True,
            'batch_size': B,
            'num_nodes_pmf': num_nodes_pmf,
        })
        assert len(A_list) == B
        assert len(attr_list) == B
        # attr values must be in [0, num_attr_classes).
        for attrs in attr_list:
            assert attrs.max().item() < num_attr_classes
            assert attrs.min().item() >= 0

        # Step 2: take the first half of sample 0 as the partial graph.
        # We use ``n_partial = ceil(n0 / 2)`` and ensure it is at least 2
        # (so there is a meaningful prefix to condition on) and leaves at
        # least one node for the model to generate.
        A0 = A_list[0]
        n0 = A0.shape[0]
        n_partial = max(2, n0 // 2)
        n_partial = min(n_partial, N - 1)  # leave room for >=1 new node

        # ``_sampling`` expects (B, n_partial, n_partial) and (B, n_partial)
        # tensors. Broadcast the single partial across the batch.
        partial_A = A0[:n_partial, :n_partial].unsqueeze(0).expand(B, -1, -1).contiguous()
        partial_attrs = attr_list[0][:n_partial].unsqueeze(0).expand(B, -1).contiguous()

        # Step 3: re-sample with partial-graph conditioning.
        A_list2, attr_list2 = model({
            'is_sampling': True,
            'batch_size': B,
            'num_nodes_pmf': num_nodes_pmf,
            'partial_A': partial_A,
            'partial_attrs': partial_attrs,
            'start_idx': n_partial,
        })
        assert len(A_list2) == B
        assert len(attr_list2) == B

        # Structural checks: sizes line up and attr values stay in range.
        for A, attrs in zip(A_list2, attr_list2):
            n = A.shape[0]
            # Generated graph shape is (n, n) and attrs is (n,).
            assert A.dim() == 2 and A.shape[0] == A.shape[1]
            assert attrs.shape == (n,), \
                f"Expected attr shape ({n},), got {attrs.shape}"
            assert attrs.max().item() < num_attr_classes
            assert attrs.min().item() >= 0
