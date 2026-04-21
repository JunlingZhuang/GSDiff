# GRANv2 Structural Fixes: Multi Canonical Orderings + Degree Features

> **Status:** 📝 Planned. Not yet implemented.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the "model doesn't know Living Room should be the hub" problem
by forcing GRANv2 to learn *structural* features rather than *positional*
features.

**Architecture:** Two small, orthogonal improvements:

1. **Multi canonical orderings** — train each graph under 3-5 different
   node orderings (DFS, BFS, k-core, degree-descent, default). Same graph
   seen from many angles → model can't memorize "position 0 = Living".
2. **Structural node features** — inject each node's *current degree*
   (count of already-generated neighbors) as an embedding added to its
   initial feature. Model gets a direct signal for "you are a high-degree
   node" which is the real substrate of "you are Living".

**Prereq:** GRANv2 (Tasks 1-7 of `plan-gran-v2-upgrade.md`) is done. The
production RPLAN pipeline works (`plan-rplan-to-gran.md`).

**Tech Stack:** Python 3.10, PyTorch 2.0.1+cu118, torch-geometric 2.7.0

---

## Why these two changes (and not position embedding)

Position embedding (e.g. `nn.Embedding(max_num_nodes, hidden_dim)` indexed
by generation order) looks tempting but is a **false signal**: if the user
runs sampling with a different implicit ordering, or if the training
corpus gets a new ordering strategy, the learned "pos 0 → Living"
pattern breaks. It's memorizing ordering noise, not structure.

**Structural features** are ordering-invariant. A high-degree node is a
high-degree node regardless of its index. Combined with multi-ordering
training, the model is forced to learn graph structure, not position.

See `research-landscape-and-roadmap.md` §3-4 for the broader rationale.

---

## Task A: Enable Multi Canonical Orderings

Zero code change; config-only. GRAN already supports this via
`num_canonical_order` and `node_order` but our current config uses the
single-ordering default.

**Files:**
- Modify: `config/gran_v2_rplan.yaml`
- Verify: `dataset/gran_data_v2.py` handles `num_canonical_order > 1` correctly

### Steps

- [ ] **Step 1: Confirm v2 dataset handles multi orderings**

Read `dataset/gran_data_v2.py`, specifically `_get_graph_data`. The parent
`GRANData._get_graph_data` returns a list of adjacency matrices (one per
ordering). The v2 override must return aligned `(adj_list, attr_list)`
where `attr_list[i]` is the room types reordered to match `adj_list[i]`'s
canonical ordering.

If this is already correct: nothing to fix. If the v2 override only
handles the first ordering: patch it.

- [ ] **Step 2: Update config/gran_v2_rplan.yaml**

Change dataset section:
```yaml
dataset:
  node_order: 'DFS+BFS+k_core'    # was 'DFS'
model:
  num_canonical_order: 3          # was 1
```

**Why 3 and not 5**: DFS/BFS/k-core cover the "hub-centric" orderings,
which are the ones most consistent with the "Living is central" prior.
Degree-ascent and default add ordering noise that may hurt more than help
on a small-graph task. Start with 3, can try 5 later.

- [ ] **Step 3: Smoke test with debug config**

Run on the grid debug config first (since it's tiny and fast):
```bash
cd D:/Github/GSDiff/GRAN
D:/Github/GSDiff/.venv/Scripts/python.exe run_exp.py -c config/gran_v2_grid_debug.yaml
```
If it doesn't crash on the first epoch, the dataset pipeline handles
multi-ordering correctly.

- [ ] **Step 4: Commit**

```bash
cd D:/Github/GSDiff
git add GRAN/config/gran_v2_rplan.yaml GRAN/config/gran_v2_grid_debug.yaml
git commit -m "config: enable 3 canonical orderings (DFS+BFS+k_core)"
```

### Expected effect

- Training data effectively 3x larger (same graphs, 3 orderings)
- Training time per epoch ~3x longer (still fast on RPLAN: ~15s/epoch → 45s/epoch)
- Model can no longer rely on "first node is Living" → forced to learn
  structural features (provided Task B is also done; otherwise it may
  learn nothing new)

---

## Task B: Add Degree-Rank Structural Feature

Add a learnable embedding indexed by each node's **current degree rank**
(position in the sort of already-generated nodes by degree, descending).
This is ordering-invariant: regardless of generation order, "the
highest-degree node right now" always maps to rank 0.

**Files:**
- Modify: `model/gran_v2.py` — `__init__` + `_inference` + `_sampling`
- Create: `tests/test_degree_feature.py` — 2 unit tests

### Why degree *rank* instead of raw degree

Raw degree values are unbounded and have a skewed distribution. Rank is
bounded (0 to N-1) and has uniform semantics across different graph
sizes ("I'm the top-1 highest-degree node" means the same thing in a
6-node graph and an 8-node graph).

### Steps

- [ ] **Step 1: Write failing tests**

```python
# tests/test_degree_feature.py
import torch
from easydict import EasyDict as edict
from model.gran_v2 import GRANv2


def _cfg(use_degree_feat=True):
    return edict({
        'device': 'cpu',
        'model': edict({
            'max_num_nodes': 10,
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
            'num_attr_classes': 5,
            'use_gatv2': False,
            'use_degree_feature': use_degree_feat,
        })
    })


def test_degree_embedding_module_exists_when_enabled():
    """When config.model.use_degree_feature=True, the model must have a
    degree_embedding submodule."""
    model = GRANv2(_cfg(use_degree_feat=True))
    assert hasattr(model, 'degree_embedding')
    # The embedding table should be indexable up to max_num_nodes-1.
    assert model.degree_embedding.num_embeddings == 10


def test_degree_feature_off_by_default():
    """When the flag is absent or False, no degree_embedding is added
    (keeps backward-compat with existing checkpoints)."""
    cfg = _cfg(use_degree_feat=False)
    # Remove the flag entirely to simulate old config.
    del cfg.model.use_degree_feature
    model = GRANv2(cfg)
    assert not hasattr(model, 'degree_embedding')
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd D:/Github/GSDiff/GRAN
D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_degree_feature.py -v
```
Expected: FAIL on `hasattr(model, 'degree_embedding')`.

- [ ] **Step 3: Add `degree_embedding` module to `GRANv2.__init__`**

Insert after the existing `self.output_attr` definition in `model/gran_v2.py`:

```python
# v2 structural fix: per-node degree-rank embedding.
# Indexed by each node's current rank in the already-generated subgraph
# (0 = highest-degree, N-1 = lowest). Gives the model an ordering-invariant
# structural signal so it can learn e.g. "rank 0 tends to be Living Room".
self.use_degree_feature = getattr(config.model, 'use_degree_feature', False)
if self.use_degree_feature:
    self.degree_embedding = nn.Embedding(
        num_embeddings=self.max_num_nodes,
        embedding_dim=self.embedding_dim,
    )
    nn.init.normal_(self.degree_embedding.weight, mean=0.0, std=0.1)
```

- [ ] **Step 4: Inject degree-rank embedding in `_inference`**

Find the block where `node_feat` is computed from the adjacency rows
(around line 170 of `gran_v2.py` after the parent class pattern). After
`node_feat` is assembled but before the GNN propagation:

```python
if self.use_degree_feature:
    # A_pad: (B, C, N, N), each row i tells "node i's neighbors in the
    # already-generated prefix". Sum over last dim -> current degree.
    degrees = A_pad.sum(dim=-1)                        # (B, C, N) long-ish
    # Rank: descending sort gives rank 0 for highest degree.
    # argsort(ranks) twice to get inverse permutation = rank of each node.
    # Clamp so new (all-zero) rows don't crash the embedding lookup.
    ranks = degrees.argsort(dim=-1, descending=True)   # sort order
    inv_ranks = ranks.argsort(dim=-1)                  # (B, C, N) rank of each node
    inv_ranks = inv_ranks.clamp(max=self.max_num_nodes - 1).long()
    deg_emb = self.degree_embedding(inv_ranks.view(-1))  # (B*C*N, H)
    node_feat = node_feat + deg_emb
```

Do the same injection at the analogous point in `_sampling` — each
autoregressive step, compute the current-state degree rank of each node
and add it to `node_state`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_degree_feature.py -v
```
Expected: both tests PASS.

- [ ] **Step 6: Run all existing tests (no regressions)**

```bash
D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: 24 passed (22 existing + 2 new).

- [ ] **Step 7: Commit**

```bash
cd D:/Github/GSDiff
git add GRAN/model/gran_v2.py GRAN/tests/test_degree_feature.py
git commit -m "feat(v2): optional degree-rank embedding as structural node feature"
```

---

## Task C: Retrain and Compare

Verify the fixes actually work on RPLAN.

**Files:**
- Modify: `config/gran_v2_rplan.yaml` — add `model.use_degree_feature: true`

### Steps

- [ ] **Step 1: Turn on degree feature in config**

```yaml
model:
  use_degree_feature: true
```

- [ ] **Step 2: Retrain on the same 20k subset**

```bash
cd D:/Github/GSDiff/GRAN
D:/Github/GSDiff/.venv/Scripts/python.exe run_exp.py -c config/gran_v2_rplan.yaml
```

Expected time: ~15-30 min (multi-ordering is 3x slower per epoch but train
converges faster on better features).

- [ ] **Step 3: Evaluate MMD on the new checkpoint**

```bash
D:/Github/GSDiff/.venv/Scripts/python.exe run_exp.py \
    -c exp/GRANv2_rplan/<new-run-id>/config.yaml -t
```

- [ ] **Step 4: Compare against baseline run (without these fixes)**

Key metrics:

| Metric | Baseline (run `45216`) | With fixes | Expected |
|--------|----------------------|-----------|----------|
| TEST degree MMD | 0.0036 | ? | unchanged / slight improvement |
| TEST clustering MMD | 0.0164 | ? | unchanged |
| TEST spectral MMD | 0.0079 | ? | unchanged |
| Visual: is Living at hub? | No (often missing) | **Yes (should emerge)** | — |
| Attr class balance | Skewed | **Closer to training distribution** | — |

Load the two `gen_grid.png` files side-by-side and eyeball whether:
- Living Room (red) appears as the high-degree node more often
- Room type distribution is closer to reference
- Fewer "0 Living" or "2 Living" floorplans

- [ ] **Step 5: Commit comparison notes into v2-progress.md**

Add a section:
```markdown
## Structural Fixes (multi canonical ordering + degree rank) — <date>
- Config: `gran_v2_rplan.yaml` with `num_canonical_order: 3`, `use_degree_feature: true`
- Baseline MMD vs TEST: ...
- New MMD vs TEST: ...
- Visual quality improvement: <notes>
```

---

## Rollback

Both changes are **opt-in and backward-compatible**:
- `num_canonical_order` defaults to 1 (single ordering) if unset
- `use_degree_feature` defaults to False if unset

Old checkpoints can be loaded with the old config. Nothing is forcibly
changed.

---

## What this does NOT solve

Even with both fixes, these remain unaddressed:

1. **Generation time** — still O(N³) autoregressive, bad for 30+ node graphs.
2. **Global structure guarantees** — model will *tend* to put Living at the
   hub but won't always. If you need strict "exactly one Living Room per
   floorplan", post-processing or architectural change is still needed.
3. **Arbitrary constraint satisfaction** — this plan doesn't add the
   constraint-aware generation from DStruct2Design etc.

Those are out of scope and belong in a diffusion-based v3.

---

## Summary of changes

| File | Change type | Lines |
|------|-------------|-------|
| `config/gran_v2_rplan.yaml` | modify | ~2 |
| `model/gran_v2.py` | modify | ~40 (init + 2 injection sites) |
| `tests/test_degree_feature.py` | create | ~50 |
| `docs/v2-progress.md` | modify | ~20 (comparison notes) |

**Total work:** ~1-2 hours code, ~30 min retrain, ~10 min evaluation.
