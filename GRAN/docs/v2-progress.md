# GRAN v2 Implementation Progress

Tracking file-level changes for each task in `plan-gran-v2-upgrade.md`.

---

## Task 1: GATv2 Multi-Head Attention Layer ✅

**Status:** DONE (updated to use PyG backend)
**Commits:** `8586323` (hand-rolled), then replaced with PyG-backed version
**Date:** 2026-04-15

### Files created (3 new)

| File | Lines | Purpose |
|------|-------|---------|
| `model/gatv2.py` | ~380 | `GATv2Layer` + `GATv2` classes; thin wrapper over `torch_geometric.nn.GATv2Conv` with hand-rolled reference version kept commented-out for reading |
| `tests/__init__.py` | 0 | Makes `tests/` a Python package |
| `tests/test_gatv2.py` | 56 | 4 unit tests (shape, dynamic attention, multi-layer, empty edges) |

### Files modified (0)

None. GATv2 is added as a parallel module — the original `GNN` class in
`model/gran_mixture_bernoulli.py` is untouched so existing checkpoints remain
loadable. Task 2 will add a `use_gatv2` config flag to switch between backbones.

### Implementation choice: PyG vs hand-rolled

After initial implementation, switched to `torch_geometric.nn.GATv2Conv`:

| | Hand-rolled (original) | PyG (current) |
|---|---|---|
| Correctness | manual scatter_softmax | battle-tested |
| Code size | ~150 lines | ~30 lines |
| Maintenance | ours | PyG team |
| Dependency | none | torch-geometric |

The hand-rolled version is kept at the bottom of `gatv2.py` as commented-out
reference code. You can see every step of the attention math (projection,
concat, LeakyReLU, segmented softmax, weighted aggregate) without diving into
PyG internals.

### What the code does

**`GATv2Layer`** — one attention layer implementing the GATv2 formula from
Brody et al. 2022:

```
alpha_ij = softmax_j(a^T LeakyReLU(W_l h_i || W_r h_j || W_e e_ij))
h_i'     = ||_{k=1}^{H} sum_j alpha_ij^k · W_v^k h_j
```

The GATv2 key fix is applying LeakyReLU BEFORE the dot product with the
attention vector `a` (vanilla GAT had it after), making attention dynamic.

Active implementation wraps `torch_geometric.nn.GATv2Conv`:
- Converts GRAN's `(E, 2)` edge list to PyG's `(2, E)` edge_index
- Disables `add_self_loops` (GRAN builds the edge list explicitly)
- `concat=True` → output dim = heads × head_dim = out_dim

Interface matches the existing `GNN.forward()`:
```python
forward(node_feat, edge, edge_feat, graph_idx=None)
```

**`GATv2`** — multi-layer stack:
- L `GATv2Layer` blocks
- LayerNorm + residual + ReLU between layers
- Optional graph-level pooling head (matches original `GNN`)

### Test results

```
tests/test_gatv2.py::test_gatv2_layer_output_shape       PASSED
tests/test_gatv2.py::test_gatv2_layer_attention_varies   PASSED
tests/test_gatv2.py::test_gatv2_multi_layer_stack        PASSED
tests/test_gatv2.py::test_gatv2_empty_edges              PASSED
4 passed in 1.56s
```

### Known issues / notes

- **Environment**: tests run against `D:\Github\GSDiff\.venv` (the GSDiff
  parent venv that has torch 2.0.1+cu118 + torch-geometric 2.7.0). The
  dedicated GRAN venv `GRAN\.venv` does not have torch installed. Future
  tasks can either install torch into GRAN's venv or keep using the parent.
- **New dependency**: `torch-geometric==2.7.0` (pure Python, no C++
  extensions needed for GATv2Conv). Not added to GRAN's pyproject.toml yet.
- No changes to `model/__init__.py` yet — Task 2 will re-export the new classes.

---

## Task 2: GRANv2 Model with Node Attribute Head ✅

**Status:** DONE
**Commit:** `156fe8a`
**Date:** 2026-04-16

### Files created (2 new)

| File | Lines | Purpose |
|------|-------|---------|
| `model/gran_v2.py` | 437 | `GRANv2` class: attr head + GATv2 option + partial graph `_sampling()` |
| `tests/test_gran_v2_model.py` | 76 | 4 unit tests |

### Files modified (1)

| File | Change |
|------|--------|
| `model/__init__.py` | Added `from model.gatv2 import *` and `from model.gran_v2 import *` |

### What the code does

**`GRANv2`** extends the original `GRANMixtureBernoulli` architecture with:

1. **`output_attr` head** — `MLP(hidden_dim → hidden_dim → num_attr_classes)`
   - In: node state `(N, H)` from GNN
   - Out: class logits `(N, A)` where A = num_attr_classes (e.g. 7)
   - Loss: `nn.CrossEntropyLoss`

2. **Backbone switch** — `use_gatv2` config flag
   - `True`: `self.decoder = GATv2(...)` (Task 1's PyG-backed implementation)
   - `False`: `self.decoder = GNN(...)` (original GRU-based)

3. **`_sampling(B, partial_A, partial_attrs, start_idx)`**
   - Injects known adjacency + attrs into initial state
   - Autoregressive loop starts from `max(start_idx, n_partial)`
   - At each step: GNN propagation → edge sampling (theta/alpha) + attr prediction
   - Returns `(A, node_attrs)` where `node_attrs = argmax(attr_logits)`

4. **`forward()` return values**
   - Training: `(edge_loss, attr_loss)` tuple
   - Sampling: `(A_list, attr_list)` tuple

### Test results

```
tests/test_gran_v2_model.py::test_gran_v2_has_attr_head                  PASSED
tests/test_gran_v2_model.py::test_gran_v2_sampling_returns_adj_and_attrs PASSED
tests/test_gran_v2_model.py::test_gran_v2_with_gatv2_backbone            PASSED
tests/test_gran_v2_model.py::test_gran_v2_with_gru_backbone              PASSED
4 passed in 1.83s (total 8 with Task 1)
```

### Architecture diagram

```
GRANv2.__init__():
    ┌── output_theta  ← same as original (edge mixture logits)
    ├── output_alpha  ← same as original (mixture weights)
    ├── output_attr   ← NEW (node attribute classification)
    ├── decoder_input ← same (adj row → embedding)
    ├── decoder       ← GNN or GATv2 (switchable via config)
    ├── adj_loss_func ← BCE (same)
    └── attr_loss_func← CrossEntropy (NEW)

GRANv2._sampling():
    A = zeros(B, N_pad, N_pad)
    [inject partial_A if given]
    for ii in range(start_idx, N_pad):
        node_state = GNN/GATv2(...)
        edges = bernoulli(theta[alpha])  ← same as original
        attrs[ii] = argmax(output_attr(node_state[ii]))  ← NEW
    return (A, attrs)
```

---

## Task 3: Partial Graph Completion Tests ✅

**Status:** DONE
**Commit:** `2db2fcf`
**Date:** 2026-04-17

### Files created (1 new)

| File | Lines | Purpose |
|------|-------|---------|
| `tests/test_partial_graph.py` | ~60 | 3 tests exercising GRANv2's partial graph modes |

### Files modified (0)

This task is TEST-ONLY — the implementation was already in Task 2's `_sampling()`.

### What the tests verify

1. **`test_partial_graph_completion`** — Give 5-node chain graph as prefix,
   generate to 8-9 nodes. Verify the original chain edges are preserved.
2. **`test_attribute_only_prediction`** — Give complete 8-node graph with
   `start_idx=N`. No new edges generated; only attributes predicted.
3. **`test_unconditional_still_works`** — Without any partial input, behavior
   matches original GRAN unconditional generation.

### Test results

```
tests/test_partial_graph.py::test_partial_graph_completion   PASSED
tests/test_partial_graph.py::test_attribute_only_prediction  PASSED
tests/test_partial_graph.py::test_unconditional_still_works  PASSED
3 passed (total 11 with Tasks 1+2)
```

### Minor fix during testing

`num_nodes_pmf` in input_dict must be a numpy array (the model calls
`torch.from_numpy()` on it). Updated the tests to wrap PMF lists with
`np.array()`. No model code changed.

---

## Task 4: GRANDataV2 Dataset ✅

**Status:** DONE
**Commit:** `651c0a7`
**Date:** 2026-04-17

### Files created (2 new)

| File | Lines | Purpose |
|------|-------|---------|
| `dataset/gran_data_v2.py` | 430 | `GRANDataV2(GRANData)` subclass with node attribute support |
| `tests/test_gran_data_v2.py` | 89 | 3 unit tests |

### Files modified (1)

| File | Change |
|------|--------|
| `dataset/__init__.py` | Added `from dataset.gran_data_v2 import *` |

### What the code does

**`GRANDataV2`** extends the parent `GRANData` to include node attribute
labels alongside the adjacency matrices. Key overrides:

1. **`_get_graph_data(G)`** — returns `(adj_list, attr_list)` instead of just
   `adj_list`. For each canonical ordering (DFS/BFS/k-core/degree/etc) the
   attrs are reordered in lockstep with the adjacency matrix.

2. **`_load_shard(path)`** — backward-compatible loader:
   - New pickle format: `(adj_list, attr_list)` tuple
   - Old pickle format: just `adj_list` → defaults attrs to zeros

3. **`__getitem__(index)`** — adds `'node_attrs'` key to the returned data
   dict. Shape: `(C, N_max)` int64 numpy array (padded with 0 for virtual
   nodes beyond the real graph size).

4. **`collate_fn(batch)`** — calls parent `collate_fn` first, then stacks
   `node_attrs` across the batch into `(B, C, N_max)` int64 tensor.

### Test results

```
tests/test_gran_data_v2.py::test_getitem_returns_node_attrs    PASSED
tests/test_gran_data_v2.py::test_attrs_match_original_graph    PASSED
tests/test_gran_data_v2.py::test_missing_attrs_default_to_zero PASSED
3 passed (total 14 with Tasks 1-3)
```

### Known issues / notes

- **networkx 3.1 compatibility**: the parent `dataset/gran_data.py` still
  calls `nx.to_numpy_matrix` which was removed in networkx 3.0. In v2 we use
  `getattr(nx, 'to_numpy_array', nx.to_numpy_matrix)` to handle both. The
  parent file is left untouched per task boundary — may need attention in
  Task 5 when wiring up training, since `GRANData` itself would break on
  this networkx version.
- **Not a subclass of parent's `__init__`**: parent persists only `adj_list`
  to disk, but v2 needs `(adj_list, attr_list)`. The v2 `__init__` re-does
  the precompute bookkeeping so the stored pickle matches the new schema.

---

## Task 5: GranRunnerV2 and Configs ⏳

**Status:** Not started

---

## Task 6: End-to-End Integration Test ⏳

**Status:** Not started

---

## Task 7: Extended Inference Modes ⏳

**Status:** Not started

---

## Summary of all changes so far

### New files
- `GRAN/model/gatv2.py` (Task 1)
- `GRAN/model/gran_v2.py` (Task 2)
- `GRAN/dataset/gran_data_v2.py` (Task 4)
- `GRAN/tests/__init__.py` (Task 1)
- `GRAN/tests/test_gatv2.py` (Task 1)
- `GRAN/tests/test_gran_v2_model.py` (Task 2)
- `GRAN/tests/test_partial_graph.py` (Task 3)
- `GRAN/tests/test_gran_data_v2.py` (Task 4)

### Modified files
- `GRAN/model/__init__.py` (Task 2 — added imports for gatv2 + gran_v2)
- `GRAN/dataset/__init__.py` (Task 4 — added import for gran_data_v2)

### Known issues to address before Task 5

- **networkx compatibility**: `dataset/gran_data.py` still uses the removed
  `nx.to_numpy_matrix`. Needs to be updated before running the full runner
  (Task 5 will fail otherwise when the parent class materializes adjacency
  matrices).
