# Plan C-1: Attr-Conditioned Edge Head — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GRAN v2's edge head condition on the *attribute* of both endpoints, so it can learn pair-specific structural priors (e.g. P(edge | Living, Living) ≈ 0; P(edge | Bedroom, Bathroom) high).

**Architecture:** Add a learnable `nn.Embedding(num_attr_classes + 1, attr_emb_dim)`. In training (teacher forcing), look up GROUND-TRUTH attr embeddings for both endpoints of each candidate edge and concat into the edge head input alongside the existing `diff = h_u - h_v`. In sampling (autoregressive), reorder the loop: sample attr_k FIRST (from the GNN hidden state), then use the embedded sampled attrs of all endpoints to condition the edge head. Slot `+1` reserved for "unknown / not-yet-sampled" (edge head's K-block self-edges).

Backward-compatible: gated by `config.model.use_attr_conditioned_edge` (default `False`). Flag-off path is byte-identical to today's GRANv2.

**Tech Stack:** PyTorch 2.0.1, Python 3.10, pytest. The codebase already has `tests/test_attr_class_weight.py` as a reference for model-level test patterns.

---

## File Structure

| File | Purpose | Action |
|------|---------|--------|
| `GRAN/model/gran_v2.py` | Add `attr_embedding`, change edge head input dim, modify `_inference` and `_sampling` | Modify |
| `GRAN/dataset/gran_data_v2.py` | Emit a per-`node_state`-row attr label tensor `subgraph_node_attrs` so `_inference` can gather endpoint attrs cheaply | Modify |
| `GRAN/tests/test_attr_cond_edge.py` | Unit tests for all behavior changes | Create |
| `GRAN/config/gran_v2_rplan_planc.yaml` | New config: same as baseline `gran_v2_rplan.yaml` with `use_attr_conditioned_edge: true`, `attr_embedding_dim: 32`, fresh exp_name | Create |

The dataset change is the cleanest way to plumb attr labels to the model at every candidate-edge endpoint without re-deriving them inside `_inference`. It's an additive output key (existing consumers ignore unknown keys), so no breakage.

---

## Task 1: Test scaffolding — config helper + smoke imports

**Files:**
- Create: `GRAN/tests/test_attr_cond_edge.py`

- [ ] **Step 1: Write the file scaffold (config helper + one trivial passing test)**

```python
"""Tests for Plan C-1 attr-conditioned edge head.

When ``config.model.use_attr_conditioned_edge`` is True, GRANv2:
  1. Registers an ``attr_embedding`` table of shape (A+1, attr_emb_dim).
  2. Sets edge-head input dim to ``hidden_dim + 2 * attr_emb_dim`` (was hidden_dim).
  3. In training (``_inference``), gathers GROUND-TRUTH attr embeddings for
     both endpoints of every candidate edge and concats them into the edge
     head input alongside the existing diff = h_u - h_v.
  4. In sampling (``_sampling``), samples attr_k BEFORE deciding edges (k, j),
     and feeds embedded sampled attrs into the edge head.

When the flag is off (default), behavior is byte-identical to the previous
GRANv2 — verified by parameter count, attr_embedding absence, and a
forward-pass equivalence check against an old-style model.
"""

import tempfile

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from dataset.gran_data_v2 import GRANDataV2
from model.gran_v2 import GRANv2


def _cfg(use_attr_cond_edge=False, attr_emb_dim=8,
         num_attr_classes=5, tmp_dir=None):
    cfg = edict({
        'device': 'cpu',
        'seed': 42,
        'model': edict({
            'name': 'GRANv2',
            'max_num_nodes': 8,
            'hidden_dim': 16,
            'embedding_dim': 16,
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
        }),
        'dataset': edict({
            'data_path': tmp_dir or '.',
            'name': 'attrcondtest',
            'node_order': 'DFS',
            'num_fwd_pass': 1,
            'is_sample_subgraph': False,
            'num_subgraph_batch': 100,
            'is_overwrite_precompute': True,
        }),
    })
    if use_attr_cond_edge:
        cfg.model.use_attr_conditioned_edge = True
        cfg.model.attr_embedding_dim = attr_emb_dim
    return cfg


def _make_graph(n_nodes=6, seed=0, num_classes=5):
    rng = np.random.RandomState(seed)
    G = nx.erdos_renyi_graph(n_nodes, 0.5, seed=seed)
    if not nx.is_connected(G):
        comps = [list(c) for c in nx.connected_components(G)]
        for cc in comps[1:]:
            G.add_edge(comps[0][0], cc[0])
    for i in range(n_nodes):
        G.nodes[i]['attr'] = int(rng.randint(0, num_classes))
    return G


def test_smoke_imports_and_helpers():
    """Smoke test: helpers build successfully."""
    cfg = _cfg()
    assert cfg.model.num_attr_classes == 5
    G = _make_graph(n_nodes=4, seed=0)
    assert G.number_of_nodes() == 4
```

- [ ] **Step 2: Run the test**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_attr_cond_edge.py -v`
Expected: PASS (1 passed).

- [ ] **Step 3: Commit**

```bash
git add GRAN/tests/test_attr_cond_edge.py
git commit -m "test(planc): add scaffolding for attr-conditioned edge head tests"
```

---

## Task 2: attr_embedding registered when flag on; absent when flag off

**Files:**
- Modify: `GRAN/model/gran_v2.py:60-220` (the `__init__` method)
- Modify: `GRAN/tests/test_attr_cond_edge.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `GRAN/tests/test_attr_cond_edge.py`:

```python
def test_attr_embedding_absent_when_flag_off():
    """Default GRANv2 has no attr_embedding (backward compat)."""
    cfg = _cfg(use_attr_cond_edge=False)
    model = GRANv2(cfg)
    assert not hasattr(model, 'attr_embedding') or model.attr_embedding is None
    # Also: use_attr_conditioned_edge attribute should default to False.
    assert getattr(model, 'use_attr_conditioned_edge', False) is False


def test_attr_embedding_registered_when_flag_on():
    """Flag on -> attr_embedding shape (A+1, attr_emb_dim)."""
    cfg = _cfg(use_attr_cond_edge=True, attr_emb_dim=8, num_attr_classes=5)
    model = GRANv2(cfg)
    assert hasattr(model, 'attr_embedding')
    # +1 for the "unknown / not-yet-sampled" slot used for K-block self-edges
    # and for endpoints whose attr hasn't been decided yet.
    assert model.attr_embedding.num_embeddings == 5 + 1
    assert model.attr_embedding.embedding_dim == 8
    # Initial weights should be small (std=0.1 init, see __init__).
    assert model.attr_embedding.weight.abs().mean().item() < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_attr_cond_edge.py::test_attr_embedding_registered_when_flag_on -v`
Expected: FAIL with `AttributeError: 'GRANv2' object has no attribute 'attr_embedding'` (or similar).

- [ ] **Step 3: Implement attr_embedding registration in GRANv2.__init__**

In `GRAN/model/gran_v2.py`, find the section after the `use_ordering_id` block (around line 165) and BEFORE the `dimension_reduce` block (around line 171). Add:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_attr_cond_edge.py -v`
Expected: PASS (3 passed — smoke + 2 new).

- [ ] **Step 5: Commit**

```bash
git add GRAN/model/gran_v2.py GRAN/tests/test_attr_cond_edge.py
git commit -m "feat(planc): register attr_embedding when use_attr_conditioned_edge=True"
```

---

## Task 3: Edge head input dim adjusts when flag on

**Files:**
- Modify: `GRAN/model/gran_v2.py:109-121` (output_theta + output_alpha construction)
- Modify: `GRAN/tests/test_attr_cond_edge.py` (add test)

The edge head MLPs (`output_theta`, `output_alpha`) currently take input of width `hidden_dim`. With Plan C-1, input becomes `concat([diff_h, a_u_emb, a_v_emb])` of width `hidden_dim + 2 * attr_embedding_dim`.

- [ ] **Step 1: Write the failing test**

Append to `GRAN/tests/test_attr_cond_edge.py`:

```python
def test_edge_head_input_dim_grows_when_flag_on():
    """First Linear of output_theta and output_alpha grows by 2*attr_emb."""
    cfg_off = _cfg(use_attr_cond_edge=False)
    model_off = GRANv2(cfg_off)
    in_off = model_off.output_theta[0].in_features
    assert in_off == cfg_off.model.hidden_dim

    cfg_on = _cfg(use_attr_cond_edge=True, attr_emb_dim=8)
    model_on = GRANv2(cfg_on)
    in_on = model_on.output_theta[0].in_features
    assert in_on == cfg_on.model.hidden_dim + 2 * 8
    # output_alpha matches output_theta input dim
    assert model_on.output_alpha[0].in_features == in_on
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_attr_cond_edge.py::test_edge_head_input_dim_grows_when_flag_on -v`
Expected: FAIL — `output_theta[0].in_features` is still `hidden_dim`.

- [ ] **Step 3: Adjust edge-head construction**

In `GRAN/model/gran_v2.py`, find the `output_theta` / `output_alpha` declarations (around lines 109-121). The `attr_embedding` block (added in Task 2) lives BELOW these — that's the wrong order, since edge-head construction needs to know `attr_embedding_dim`. **Move the attr_embedding block to BEFORE the edge-head MLPs** (i.e. immediately after the line `self.att_edge_dim = 64` near line 103), then change the edge-head input dim computation to:

```python
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
```

(`attr_embedding_dim` is 0 when the flag is off, so `edge_head_in_dim == hidden_dim` in that path — exactly the old behavior.)

- [ ] **Step 4: Run all tests to verify pass + no regressions**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_attr_cond_edge.py tests/test_gran_v2_model.py tests/test_attr_class_weight.py -v`
Expected: ALL PASS — flag-off path is unchanged, flag-on path has new dim.

- [ ] **Step 5: Commit**

```bash
git add GRAN/model/gran_v2.py GRAN/tests/test_attr_cond_edge.py
git commit -m "feat(planc): widen edge head input by 2*attr_emb when flag on"
```

---

## Task 4: Dataset emits per-state-row attr labels

**Files:**
- Modify: `GRAN/dataset/gran_data_v2.py` (in `__getitem__`, around line 290-410)
- Modify: `GRAN/tests/test_attr_cond_edge.py` (add test)

The model needs the attr label of every endpoint of every candidate edge. Cleanest approach: emit a `subgraph_node_attrs` array of shape `(M,)` aligned with `node_state` rows. M = `sum(jj+K for each subgraph in batch)`. For each subgraph row position `p` (0..jj+K-1), the attr is `attr_list[order_idx][p]` if `p < n_real_nodes` else `num_attr_classes` (the "unknown" slot).

- [ ] **Step 1: Write the failing test**

Append to `GRAN/tests/test_attr_cond_edge.py`:

```python
def test_dataset_emits_subgraph_node_attrs_aligned_with_node_state():
    """``subgraph_node_attrs`` shape matches what node_state will have.

    For each subgraph of size jj+K, the array stores the ground-truth attr
    of every row (existing nodes get their real attr; padding/out-of-range
    slots get num_attr_classes — the 'unknown' slot id).
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(num_attr_classes=5, tmp_dir=tmp)
        graphs = [_make_graph(n_nodes=6, seed=i, num_classes=5)
                  for i in range(3)]
        ds = GRANDataV2(cfg, graphs, tag='train')
        sample = ds[0]
        assert 'subgraph_node_attrs' in sample
        assert sample['subgraph_node_attrs'].dtype.kind in ('i', 'u')
        # The per-subgraph row count equals subgraph_size summed:
        # each subgraph contributes (jj+K) rows.
        assert sample['subgraph_node_attrs'].shape[0] == \
            sample['node_idx_feat'].shape[0]
        # Values must be in [0, num_attr_classes] (last slot reserved).
        assert sample['subgraph_node_attrs'].min() >= 0
        assert sample['subgraph_node_attrs'].max() <= cfg.model.num_attr_classes


def test_collate_stacks_subgraph_node_attrs():
    """collate_fn concatenates per-sample subgraph_node_attrs into one tensor."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(num_attr_classes=5, tmp_dir=tmp)
        graphs = [_make_graph(n_nodes=6, seed=i, num_classes=5)
                  for i in range(3)]
        ds = GRANDataV2(cfg, graphs, tag='train')
        batch = ds.collate_fn([ds[i] for i in range(len(graphs))])
        data = batch[0]
        assert 'subgraph_node_attrs' in data
        assert isinstance(data['subgraph_node_attrs'], torch.Tensor)
        assert data['subgraph_node_attrs'].dtype == torch.long
        # Aligns with node_idx_feat after collate (both flattened across batch).
        assert data['subgraph_node_attrs'].shape == data['node_idx_feat'].shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_attr_cond_edge.py::test_dataset_emits_subgraph_node_attrs_aligned_with_node_state -v`
Expected: FAIL — `'subgraph_node_attrs' not in sample`.

- [ ] **Step 3: Implement in `__getitem__` (per-sample)**

In `GRAN/dataset/gran_data_v2.py`, the per-subgraph loop in `__getitem__` is around lines 319-378. Two edits:

**(a)** Initialize an accumulator next to `node_idx_feat = []` (line 312). Replace lines 310-316 (the block that initializes `edges`, `node_idx_gnn`, `node_idx_feat`, etc.) by adding `subgraph_node_attrs` to the list:

```python
            edges = []
            node_idx_gnn = []
            node_idx_feat = []
            label = []
            subgraph_size = []
            subgraph_idx = []
            att_idx = []
            subgraph_node_attrs = []   # Plan C-1: aligned with node_state rows
            subgraph_count = 0
```

**(b)** Inside the `for jj in range(0, num_nodes, S)` loop (around line 323), AFTER the `node_idx_gnn` and `label` blocks (i.e. just before `subgraph_size += [jj + K]` at line 373), append:

```python
                    # ---- Plan C-1: per-subgraph-row attr label ----------
                    # node_state for this subgraph has jj+K rows in this
                    # canonical ordering. Existing-node rows take their
                    # real attr; new-node rows take their GROUND-TRUTH attr
                    # (teacher forcing). Positions beyond n_real_nodes
                    # (rare — only when the graph is shorter than jj+K)
                    # take the "unknown" slot id = num_attr_classes.
                    A_classes = self.config.model.num_attr_classes
                    n_real = attr_list[ii].shape[0]
                    sg_attrs = np.full(jj + K, A_classes, dtype=np.int64)
                    take_n = min(jj + K, n_real)
                    sg_attrs[:take_n] = attr_list[ii][:take_n]
                    subgraph_node_attrs.append(sg_attrs)
```

Then, in the data-dict assembly block (around lines 385-406), add:

```python
            data['subgraph_node_attrs'] = np.concatenate(subgraph_node_attrs) \
                if len(subgraph_node_attrs) > 0 else np.zeros((0,), dtype=np.int64)
```

- [ ] **Step 4: Implement in `collate_fn`**

In the same file's `collate_fn` (around lines 419-428), find the loop that handles `node_attrs` stacking. Add stacking for `subgraph_node_attrs`. Each sample's array is 1D of variable length; concatenate them across batch and convert to a torch long tensor:

```python
            # ---- Plan C-1: concat per-sample subgraph_node_attrs ----
            sgna_list = [bb[ff]['subgraph_node_attrs'] for bb in batch]
            batch_data[ff]['subgraph_node_attrs'] = torch.from_numpy(
                np.concatenate(sgna_list)).long()
```

- [ ] **Step 5: Run tests to verify they pass + no regressions**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_attr_cond_edge.py tests/test_gran_data_v2.py -v`
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add GRAN/dataset/gran_data_v2.py GRAN/tests/test_attr_cond_edge.py
git commit -m "feat(planc): emit per-state-row attr labels from GRANDataV2"
```

---

## Task 5: `_inference` uses attr embeddings in edge head input

**Files:**
- Modify: `GRAN/model/gran_v2.py:273-350` (the `_inference` method)
- Modify: `GRAN/model/gran_v2.py:734-846` (the `forward` method — pass new key through)
- Modify: `GRAN/tests/test_attr_cond_edge.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `GRAN/tests/test_attr_cond_edge.py`:

```python
def test_inference_forward_pass_with_flag_on():
    """End-to-end training forward returns finite scalar losses."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(use_attr_cond_edge=True, attr_emb_dim=8,
                   num_attr_classes=5, tmp_dir=tmp)
        model = GRANv2(cfg)
        graphs = [_make_graph(n_nodes=6, seed=i, num_classes=5)
                  for i in range(3)]
        ds = GRANDataV2(cfg, graphs, tag='train')
        batch = ds.collate_fn([ds[i] for i in range(len(graphs))])
        data = batch[0]

        input_dict = {
            'is_sampling': False,
            'adj': data['adj'],
            'edges': data['edges'],
            'node_idx_gnn': data['node_idx_gnn'],
            'node_idx_feat': data['node_idx_feat'],
            'att_idx': data['att_idx'],
            'label': data['label'],
            'subgraph_idx': data['subgraph_idx'],
            'subgraph_idx_base': data['subgraph_idx_base'],
            'node_attrs': data['node_attrs'],
            'subgraph_node_attrs': data['subgraph_node_attrs'],
        }
        edge_loss, attr_loss = model(input_dict)
        assert torch.isfinite(edge_loss)
        assert torch.isfinite(attr_loss)
        assert edge_loss.item() > 0
        assert attr_loss.item() > 0


def test_inference_uses_attr_embeddings_in_edge_logits():
    """Edge logits depend on attr_embedding weights when flag on.

    Strategy: build two flag-on models with identical state EXCEPT for
    attr_embedding weights, run forward on the same batch, verify the
    edge_loss differs.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(use_attr_cond_edge=True, attr_emb_dim=8,
                   num_attr_classes=5, tmp_dir=tmp)
        m1 = GRANv2(cfg)
        m2 = GRANv2(cfg)
        # Copy all params from m1 to m2 then perturb only attr_embedding.
        sd1 = m1.state_dict()
        sd2 = m2.state_dict()
        for k, v in sd1.items():
            if k != 'attr_embedding.weight':
                sd2[k].copy_(v)
        # Set m2's attr_embedding to a different value.
        m2.attr_embedding.weight.data = torch.randn_like(
            m2.attr_embedding.weight) * 0.5

        graphs = [_make_graph(n_nodes=6, seed=i, num_classes=5)
                  for i in range(3)]
        ds = GRANDataV2(cfg, graphs, tag='train')
        batch = ds.collate_fn([ds[i] for i in range(len(graphs))])
        data = batch[0]
        input_dict = {
            'is_sampling': False,
            'adj': data['adj'],
            'edges': data['edges'],
            'node_idx_gnn': data['node_idx_gnn'],
            'node_idx_feat': data['node_idx_feat'],
            'att_idx': data['att_idx'],
            'label': data['label'],
            'subgraph_idx': data['subgraph_idx'],
            'subgraph_idx_base': data['subgraph_idx_base'],
            'node_attrs': data['node_attrs'],
            'subgraph_node_attrs': data['subgraph_node_attrs'],
        }
        m1.eval(); m2.eval()
        with torch.no_grad():
            e1, _ = m1(input_dict)
            e2, _ = m2(input_dict)
        assert not torch.isclose(e1, e2), (
            "Edge loss did not change when attr_embedding changed — "
            "attr embeddings are not flowing into the edge head")


def test_inference_flag_off_unchanged():
    """Flag off + identical seed -> identical edge loss as before this change.

    We construct a flag-off model and check that omitting subgraph_node_attrs
    from input_dict still works (backward compat with existing call sites).
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(use_attr_cond_edge=False, num_attr_classes=5, tmp_dir=tmp)
        model = GRANv2(cfg)
        graphs = [_make_graph(n_nodes=6, seed=i, num_classes=5)
                  for i in range(3)]
        ds = GRANDataV2(cfg, graphs, tag='train')
        batch = ds.collate_fn([ds[i] for i in range(len(graphs))])
        data = batch[0]
        input_dict = {
            'is_sampling': False,
            'adj': data['adj'],
            'edges': data['edges'],
            'node_idx_gnn': data['node_idx_gnn'],
            'node_idx_feat': data['node_idx_feat'],
            'att_idx': data['att_idx'],
            'label': data['label'],
            'subgraph_idx': data['subgraph_idx'],
            'subgraph_idx_base': data['subgraph_idx_base'],
            'node_attrs': data['node_attrs'],
            # Deliberately omit subgraph_node_attrs to verify back-compat.
        }
        edge_loss, attr_loss = model(input_dict)
        assert torch.isfinite(edge_loss)
        assert torch.isfinite(attr_loss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_attr_cond_edge.py -v`
Expected: at least `test_inference_forward_pass_with_flag_on` and `test_inference_uses_attr_embeddings_in_edge_logits` fail (likely with a Linear in_features mismatch when `_inference` feeds plain `diff` into the new wider edge-head MLP).

- [ ] **Step 3: Modify `_inference` signature + body**

In `GRAN/model/gran_v2.py`, change `_inference` to accept an optional `subgraph_node_attrs` arg and use it when `use_attr_conditioned_edge=True`:

Replace lines 273-274 (signature) with:

```python
    def _inference(self, A_pad=None, edges=None,
                   node_idx_gnn=None, node_idx_feat=None, att_idx=None,
                   subgraph_node_attrs=None):
```

Replace the diff-and-edge-head block (lines 346-350) with:

```python
        # Pairwise diff between candidate edge endpoints -> edge logits.
        diff = node_state[node_idx_gnn[:, 0], :] - node_state[node_idx_gnn[:, 1], :]

        # ---- Plan C-1: append attr-pair embeddings to edge head input ----
        # When the flag is on, the edge head sees the attr embedding of
        # both endpoints alongside the standard node-state diff. Asymmetric
        # concat (a_u, a_v) is fine: the dataset's multi-ordering averaging
        # already smooths over u/v swaps.
        if self.use_attr_conditioned_edge and subgraph_node_attrs is not None:
            # Clamp out-of-range to the "unknown" slot defensively.
            attrs = subgraph_node_attrs.clamp(min=0, max=self.num_attr_classes)
            a_u = self.attr_embedding(attrs[node_idx_gnn[:, 0]])
            a_v = self.attr_embedding(attrs[node_idx_gnn[:, 1]])
            edge_in = torch.cat([diff, a_u, a_v], dim=-1)
        else:
            edge_in = diff

        log_theta = self.output_theta(edge_in).view(-1, self.num_mix_component)
        log_alpha = self.output_alpha(edge_in).view(-1, self.num_mix_component)
        return log_theta, log_alpha, node_state
```

- [ ] **Step 4: Plumb the new key through `forward`**

In `GRAN/model/gran_v2.py` around line 758 (the input_dict unpacking in `forward`), add:

```python
        subgraph_node_attrs = input_dict.get('subgraph_node_attrs', None)
```

And in the `_inference` call (around lines 767-772), add the new arg:

```python
            log_theta, log_alpha, node_state = self._inference(
                A_pad=A_pad,
                edges=edges,
                node_idx_gnn=node_idx_gnn,
                node_idx_feat=node_idx_feat,
                att_idx=att_idx,
                subgraph_node_attrs=subgraph_node_attrs)
```

- [ ] **Step 5: Update runner to pass the new key**

`gran_runner_v2.py` already conditionally threads `node_attrs` into `data` at lines 292-294 (training path) and lines 901-902 (validation path). Add an analogous conditional block for `subgraph_node_attrs` at BOTH locations.

After the existing block at lines 292-294:

```python
                            # [v2] thread node_attrs through when provided by the loader
                            if 'node_attrs' in batch_data[dd][ff]:
                                data['node_attrs'] = batch_data[dd][ff]['node_attrs'].pin_memory().to(gpu_id, non_blocking=True)
                            # [planc] thread per-state-row attr labels for the
                            # attr-conditioned edge head.
                            if 'subgraph_node_attrs' in batch_data[dd][ff]:
                                data['subgraph_node_attrs'] = batch_data[dd][ff]['subgraph_node_attrs'].pin_memory().to(gpu_id, non_blocking=True)
```

Apply the same edit at lines 901-902 (without `pin_memory()` if the surrounding val-path code style omits it — match local style verbatim).

- [ ] **Step 6: Run all tests**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: ALL PASS, including pre-existing tests (no regression).

- [ ] **Step 7: Commit**

```bash
git add GRAN/model/gran_v2.py GRAN/runner/gran_runner_v2.py GRAN/tests/test_attr_cond_edge.py
git commit -m "feat(planc): _inference concats attr embeddings into edge head input"
```

---

## Task 6: `_sampling` reorders to attr-then-edge

**Files:**
- Modify: `GRAN/model/gran_v2.py:355-639` (the `_sampling` method)
- Modify: `GRAN/tests/test_attr_cond_edge.py` (add test)

The current order in `_sampling` is: GNN forward → sample edges → sample attrs. Plan C-1 requires: GNN forward → sample attrs → sample edges (so the edge head can see the just-sampled attr_k).

Existing endpoints (j < ii) already have attrs in `node_attrs[:, :ii]` from prior iterations. New endpoints (k in [ii, jj)) get the just-sampled `new_attrs`.

- [ ] **Step 1: Write the failing test**

Append to `GRAN/tests/test_attr_cond_edge.py`:

```python
def test_sampling_returns_correct_shapes_with_flag_on():
    """End-to-end sampling produces (B, N) adjacency and (B, N) attrs."""
    cfg = _cfg(use_attr_cond_edge=True, attr_emb_dim=8, num_attr_classes=5)
    model = GRANv2(cfg)
    model.eval()
    B = 2
    A, attrs = model._sampling(B)
    assert A.shape == (B, cfg.model.max_num_nodes, cfg.model.max_num_nodes)
    assert attrs.shape == (B, cfg.model.max_num_nodes)
    # Attrs are in [0, num_attr_classes) (they were SAMPLED from the model
    # head, never set to the "unknown" slot).
    assert attrs.min().item() >= 0
    assert attrs.max().item() < cfg.model.num_attr_classes


def test_sampling_attr_decisions_affect_edge_decisions():
    """When attr embedding is large+distinct per class, swapping an existing
    node's attr (via partial_attrs) measurably changes downstream edge probs.

    Strategy: pin n_partial=2 nodes with all-zero adjacency. Run sampling
    once with partial_attrs=[0, 0] and once with partial_attrs=[1, 4].
    Verify the resulting adjacency distributions differ over many seeds.
    """
    cfg = _cfg(use_attr_cond_edge=True, attr_emb_dim=16, num_attr_classes=5)
    model = GRANv2(cfg)
    # Amplify attr_embedding so the signal is strong even with random init.
    with torch.no_grad():
        model.attr_embedding.weight.mul_(5.0)
    model.eval()
    B = 8
    n_partial = 2
    partial_A = torch.zeros(B, n_partial, n_partial)

    torch.manual_seed(0)
    A_a, _ = model._sampling(
        B,
        partial_A=partial_A,
        partial_attrs=torch.zeros(B, n_partial, dtype=torch.long),
        start_idx=n_partial,
    )
    torch.manual_seed(0)
    A_b, _ = model._sampling(
        B,
        partial_A=partial_A,
        partial_attrs=torch.tensor(
            [[1, 4]] * B, dtype=torch.long),
        start_idx=n_partial,
    )
    # The two adjacencies should differ on at least some entries; if the
    # attr signal weren't reaching the edge head, identical seed +
    # identical h_uv => identical Bernoulli draws => identical A.
    assert not torch.equal(A_a, A_b), (
        "Adjacency identical with different partial_attrs — attr signal "
        "is not reaching the edge head during sampling")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_attr_cond_edge.py::test_sampling_attr_decisions_affect_edge_decisions -v`
Expected: FAIL — current `_sampling` doesn't feed attrs into edge head.

- [ ] **Step 3: Reorder the `_sampling` loop body**

In `GRAN/model/gran_v2.py`, find the per-iteration body inside `for ii in range(loop_start, N_pad, S):` (around lines 451-631).

The current order is:
1. GNN forward → `node_state_out` (~line 547)
2. Compute `log_theta`, `log_alpha` from diff (~lines 552-559)
3. Sample alpha & edges (~lines 561-595)
4. Compute `new_attr_logits` from `node_state_out[:, ii:jj, :]` and sample `new_attrs` (~lines 597-629)

New order (when flag on):
1. GNN forward → `node_state_out`
2. Sample `new_attrs` for the K new nodes from `output_attr(node_state_out[:, ii:jj, :])`
3. Write `new_attrs` into `node_attrs[:, ii:jj]` BEFORE computing edge logits
4. Build edge head input: `concat([diff, a_u_emb, a_v_emb])` where the attr ids come from `node_attrs` (now populated for all rows up to jj)
5. Compute `log_theta`, `log_alpha`, sample alpha & edges

Concretely, replace the body section between "GNN forward" (line 547) and the end of the iteration with the following structure:

```python
                # GNN forward on the flattened (B*jj, H) node features.
                node_state_out = self.decoder(
                    node_state_in.view(-1, H), edges, edge_feat=att_edge_feat)
                node_state_out = node_state_out.view(B, jj, -1)   # (B, jj, H)

                # ---- Plan C-1: sample attrs for the K new nodes BEFORE
                # the edge head runs, so the edge head can condition on
                # the just-sampled attrs.
                # When the flag is OFF this block runs in its OLD location
                # (after edge sampling) — see further down.
                if self.use_attr_conditioned_edge:
                    new_attrs = self._sample_new_node_attrs(
                        node_state_out[:, ii:jj, :], H)
                    node_attrs[:, ii:jj] = new_attrs
                    if return_attr_logits:
                        # Recompute logits with the sampled attrs already
                        # written, for consistency with the new order.
                        new_node_feat = node_state_out[:, ii:jj, :]
                        attr_logits_all[:, ii:jj, :] = self.output_attr(
                            new_node_feat.reshape(-1, H)).view(B, K, -1)

                # Build pairwise diffs between (new node row) and (all cols).
                idx_row, idx_col = np.meshgrid(np.arange(ii, jj), np.arange(jj))
                idx_row = torch.from_numpy(idx_row.reshape(-1)).long().to(self.device)
                idx_col = torch.from_numpy(idx_col.reshape(-1)).long().to(self.device)

                diff = node_state_out[:, idx_row, :] - node_state_out[:, idx_col, :]
                diff = diff.view(-1, node_state.shape[2])

                # ---- Plan C-1: append attr-pair embeddings to edge head ----
                if self.use_attr_conditioned_edge:
                    # Lookup attrs for both endpoints. Both are populated
                    # in node_attrs already: existing rows from prior loop
                    # iterations, new rows from the just-sampled new_attrs
                    # written above.
                    # node_attrs has shape (B, N_pad). idx_row/idx_col are
                    # node positions in the (B, jj) grid. We want a 2D gather
                    # indexed by (b, idx_row[e]) for every batch b and edge e.
                    a_row = node_attrs[:, idx_row]                # (B, E_per_b)
                    a_col = node_attrs[:, idx_col]                # (B, E_per_b)
                    a_row = a_row.clamp(min=0, max=self.num_attr_classes)
                    a_col = a_col.clamp(min=0, max=self.num_attr_classes)
                    a_u_emb = self.attr_embedding(a_row).view(
                        -1, self.attr_embedding_dim)
                    a_v_emb = self.attr_embedding(a_col).view(
                        -1, self.attr_embedding_dim)
                    edge_in = torch.cat([diff, a_u_emb, a_v_emb], dim=-1)
                else:
                    edge_in = diff

                log_theta = self.output_theta(edge_in)
                log_alpha = self.output_alpha(edge_in)

                log_theta = log_theta.view(B, -1, K, self.num_mix_component)
                log_theta = log_theta.transpose(1, 2)           # (B, K, jj, L)

                log_alpha = log_alpha.view(B, -1, self.num_mix_component)
                prob_alpha = F.softmax(log_alpha.mean(dim=1), -1)
                alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1).long()

                if skip_edge_sampling and (not fixed_edges_consumed) \
                        and fixed_edges_next is not None:
                    fe = fixed_edges_next.to(self.device).float()
                    assert fe.shape[0] == B, \
                        "fixed_edges_next batch dim must match B"
                    assert fe.shape[1] >= ii, \
                        "fixed_edges_next must have >= ii columns"
                    fe_row = torch.zeros(B, jj).to(self.device)
                    fe_row[:, :ii] = fe[:, :ii]
                    A[:, ii:jj, :jj] = fe_row.unsqueeze(1).expand(-1, K, -1)
                    fixed_edges_consumed = True
                else:
                    prob = []
                    for bb in range(B):
                        prob += [torch.sigmoid(log_theta[bb, :, :, alpha[bb]])]
                    prob = torch.stack(prob, dim=0)             # (B, K, jj)
                    A[:, ii:jj, :jj] = torch.bernoulli(prob[:, :jj - ii, :])

                # ---- attribute prediction (LEGACY ORDER: flag off) -------
                # Same logic as before, runs only when flag is off.
                if not self.use_attr_conditioned_edge:
                    new_node_feat = node_state_out[:, ii:jj, :]
                    new_attr_logits = self.output_attr(
                        new_node_feat.reshape(-1, H))
                    new_attr_logits = new_attr_logits.view(B, K, -1)
                    temp = float(getattr(
                        self.config.model, 'attr_temperature', 1.0) or 1.0)
                    scaled_logits = new_attr_logits / temp
                    probs = F.softmax(scaled_logits, dim=-1)
                    flat_probs = probs.view(-1, probs.shape[-1])
                    flat_samples = torch.multinomial(
                        flat_probs, 1).squeeze(-1)
                    new_attrs = flat_samples.view(B, K)
                    node_attrs[:, ii:jj] = new_attrs
                    if return_attr_logits:
                        attr_logits_all[:, ii:jj, :] = new_attr_logits
```

Add the helper method below the `_sampling` method (or above it; placement doesn't matter):

```python
    def _sample_new_node_attrs(self, new_node_feat, H):
        """Sample attr for K new nodes from softmax(output_attr(h)).

        Plan C-1 helper. Same temperature-scaling logic as the legacy
        attr-sampling block at the end of ``_sampling``, factored out so
        it can run before the edge head.

        Args:
            new_node_feat: (B, K, H) hidden states for the K new nodes.
            H:             hidden dim, passed in to avoid re-derivation.

        Returns:
            (B, K) long tensor of sampled attr ids in [0, num_attr_classes).
        """
        B, K, _ = new_node_feat.shape
        new_attr_logits = self.output_attr(new_node_feat.reshape(-1, H))
        new_attr_logits = new_attr_logits.view(B, K, -1)
        temp = float(
            getattr(self.config.model, 'attr_temperature', 1.0) or 1.0)
        scaled_logits = new_attr_logits / temp
        probs = F.softmax(scaled_logits, dim=-1)
        flat_probs = probs.view(-1, probs.shape[-1])
        flat_samples = torch.multinomial(flat_probs, 1).squeeze(-1)
        return flat_samples.view(B, K)
```

- [ ] **Step 4: Run all tests**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: ALL PASS, including the new sampling tests.

- [ ] **Step 5: Commit**

```bash
git add GRAN/model/gran_v2.py GRAN/tests/test_attr_cond_edge.py
git commit -m "feat(planc): _sampling samples attr before edges + conditions edge head"
```

---

## Task 7: Gradient flow sanity check

**Files:**
- Modify: `GRAN/tests/test_attr_cond_edge.py` (add test)

- [ ] **Step 1: Write the test**

Append to `GRAN/tests/test_attr_cond_edge.py`:

```python
def test_attr_embedding_gradient_flows_from_edge_loss():
    """Backprop from edge_loss must reach attr_embedding.weight.grad."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(use_attr_cond_edge=True, attr_emb_dim=8,
                   num_attr_classes=5, tmp_dir=tmp)
        model = GRANv2(cfg)
        model.train()
        graphs = [_make_graph(n_nodes=6, seed=i, num_classes=5)
                  for i in range(3)]
        ds = GRANDataV2(cfg, graphs, tag='train')
        batch = ds.collate_fn([ds[i] for i in range(len(graphs))])
        data = batch[0]
        input_dict = {
            'is_sampling': False,
            'adj': data['adj'],
            'edges': data['edges'],
            'node_idx_gnn': data['node_idx_gnn'],
            'node_idx_feat': data['node_idx_feat'],
            'att_idx': data['att_idx'],
            'label': data['label'],
            'subgraph_idx': data['subgraph_idx'],
            'subgraph_idx_base': data['subgraph_idx_base'],
            'node_attrs': data['node_attrs'],
            'subgraph_node_attrs': data['subgraph_node_attrs'],
        }
        edge_loss, attr_loss = model(input_dict)
        # Backprop ONLY edge_loss to isolate the attr-embedding-via-edge-head
        # gradient path (attr head also touches attr_embedding indirectly
        # via output_attr — wait, no, output_attr does NOT use attr_embedding;
        # only the edge head does. So edge_loss is the only path.)
        edge_loss.backward()
        g = model.attr_embedding.weight.grad
        assert g is not None, "attr_embedding has no .grad after edge_loss.backward()"
        assert g.abs().sum().item() > 0, "attr_embedding.grad is all zeros"
```

- [ ] **Step 2: Run the test**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_attr_cond_edge.py::test_attr_embedding_gradient_flows_from_edge_loss -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add GRAN/tests/test_attr_cond_edge.py
git commit -m "test(planc): assert attr_embedding gradient flows from edge loss"
```

---

## Task 8: Plan C config + integration smoke test

**Files:**
- Create: `GRAN/config/gran_v2_rplan_planc.yaml`

- [ ] **Step 1: Copy baseline config and toggle the flag**

Read `GRAN/config/gran_v2_rplan.yaml` and write `GRAN/config/gran_v2_rplan_planc.yaml` as an exact copy with these diffs:

- `model.use_attr_conditioned_edge: true` (new key, after `gatv2_num_heads`)
- `model.attr_embedding_dim: 32` (new key)
- `exp_name: GRANv2_rplan_planc`
- `exp_dir: exp/GRANv2_rplan_planc`
- `test.test_model_dir:` set to empty / placeholder so the runner doesn't try to load a baseline checkpoint by mistake. Comment with a note: "set after first training run completes".
- Keep everything else identical (300 epochs, lr 2e-4, batch 64, 20k graphs, etc.) so this is a clean A/B comparison vs. baseline.

- [ ] **Step 2: Smoke-test config loads + builds model successfully**

Run from project root:

```bash
cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -c "
from utils.arg_helper import get_config
cfg = get_config('config/gran_v2_rplan_planc.yaml', is_test=False)
print('use_attr_conditioned_edge:', cfg.model.get('use_attr_conditioned_edge'))
print('attr_embedding_dim:', cfg.model.get('attr_embedding_dim'))
print('exp_name:', cfg.exp_name)
from model.gran_v2 import GRANv2
model = GRANv2(cfg)
print('model has attr_embedding:', hasattr(model, 'attr_embedding') and model.attr_embedding is not None)
print('attr_embedding.num_embeddings:', model.attr_embedding.num_embeddings)
print('output_theta in_features:', model.output_theta[0].in_features)
n_params = sum(p.numel() for p in model.parameters())
print(f'total params: {n_params:,}')
"
```

Expected output:
```
use_attr_conditioned_edge: True
attr_embedding_dim: 32
exp_name: GRANv2_rplan_planc
model has attr_embedding: True
attr_embedding.num_embeddings: 8
output_theta in_features: 320   # 256 + 2*32
total params: <some number>
```

- [ ] **Step 3: Commit**

```bash
git add GRAN/config/gran_v2_rplan_planc.yaml
git commit -m "config(planc): new config with use_attr_conditioned_edge=True"
```

---

## Task 9: Final regression sweep

**Files:** none modified.

- [ ] **Step 1: Run the full GRAN test suite**

Run: `cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/ -v --tb=short`
Expected: ALL PASS.

- [ ] **Step 2: If any pre-existing test fails, investigate before continuing**

Common failure modes to look for:
- A test that constructed `input_dict` without `subgraph_node_attrs` and now fails because the runner / model insists on the key. **Fix:** ensure all flag-off paths tolerate missing key (already specified in Task 5 design).
- An equivalence test comparing two models (one with flag off) that drifted because of the edge-head dim change. **Fix:** verify `edge_head_in_dim == hidden_dim` exactly when flag is off.

- [ ] **Step 3: Final commit — only if there are accumulated edits not yet committed**

```bash
git status
# if dirty:
git add -A
git commit -m "test(planc): fix downstream regressions from edge head input dim change"
```

---

## After the plan is executed

(NOT part of this plan; record here so the engineer knows the next step.)

1. Run training:
   ```bash
   cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe run_exp.py -c config/gran_v2_rplan_planc.yaml
   ```
2. After training finishes, set `test.test_model_dir` in the planc config to the new `exp/GRANv2_rplan_planc/<run_dir>/` and run testing:
   ```bash
   cd GRAN && D:/Github/GSDiff/.venv/Scripts/python.exe run_exp.py -c config/gran_v2_rplan_planc.yaml -t
   ```
3. Compare attr-aware metrics in `test_results.json` vs. baseline & Plan A. Targets (from plan-c-design.md §3.1.6):
   - `endpoint_attr_pair_kl` < 0.6
   - `0-0` (Living-Living) edges < 5%
   - `1-2` (Bedroom-Bathroom) edges > 10%
   - `living_count_kl` significantly down
   - per-graph 1-Living rate > 50%
4. Write up as Exp 6 in `experiments.md`.
5. Decide on Plan C-2 based on results (per plan-c-design.md §3.2).
