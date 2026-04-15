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

## Task 2: GRANv2 Model with Node Attribute Head ⏳

**Status:** Not started

_Pending. Will update when implementation begins._

---

## Task 3: Partial Graph Completion Tests ⏳

**Status:** Not started

---

## Task 4: GRANDataV2 Dataset ⏳

**Status:** Not started

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
- `GRAN/model/gatv2.py`
- `GRAN/tests/__init__.py`
- `GRAN/tests/test_gatv2.py`

### Modified files
(none yet)
