# MSD Hierarchical GSDiff

Status: data construction started

This document tracks the new MSD floor-level, coarse-to-fine GSDiff path. It is
intentionally separate from the original RPLAN GSDiff pipeline and from the
DiGress graph-generation baseline.

## Goal

Train a GSDiff-style graph-to-vector-floorplan model for MSD building-complex
floors without flattening each full floor into one very large corner graph.

The proposed structure is coarse-to-fine:

```text
Full MSD room graph
-> coarse floor graph
-> coarse unit/public/stair/corridor layout
-> local room-level GSDiff inside each coarse region
-> stitching / wall snapping / geometry cleanup
```

This is still a building-level MSD task. It does not reduce MSD to isolated
apartment-level generation.

## Dataset Version

The first dataset is `msd_hier_v6`.

Inputs:

```text
Raw geometry: datasets/msd/raw/mds_V2_5.372k.csv
Wall graph:   digress/data/msd_wall_v6/graphs.p
```

The room-level edge labels are aligned with the existing MSD wall v6 graph:

```text
none / wall / passage / door / entrance
```

Raw CSV geometry is used for polygons, `floor_id`, `unit_id`, `roomtype`, and
group construction. The v6 graph pickle is used as the preferred source for
room connectivity. If a floor cannot be matched to the v6 graph by `floor_id`
and room count, the script falls back to geometry adjacency and marks the
sample as `geometry_fallback`.

## Generated Layout

Command:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_msd_hierarchy.py --vis-samples 50 --clean
```

Output:

```text
datasets/msd_hier_v6/
  floors/{train,val,test}/*.pkl
  coarse/{train,val,test}/*.pkl
  local/{train,val,test}/*.pkl
  vis/sample_*.png
  stats.json
```

The output directory is ignored by git through the repository-level `datasets`
ignore rule.

## Dataset Loader

Added:

```text
datasets/msd_hier.py
```

Smoke command:

```powershell
.\.venv\Scripts\python.exe datasets\msd_hier.py --root datasets\msd_hier_v6 --split train --batch-size 4
```

Smoke result:

```text
coarse samples 4297
coarse node_features (4, 64, 14)
coarse edge_types (4, 64, 64)
coarse target_bboxes (4, 64, 4)
local samples 27550
local node_features (4, 32, 13)
local edge_types (4, 32, 32)
local target_bboxes (4, 32, 4)
```

The loader preserves typed v6 edges and also exposes a binary compatibility
tensor:

```text
edge_types:
  none / wall / passage / door / entrance / touch / adjacent

edge_present:
  0 for none, 1 for any real edge
```

This matters because the original GSDiff edge handling is effectively binary.
The first smoke model can use `edge_present` to stay close to the original
pipeline. A later MSD-specific model should use `edge_types` through an edge
embedding, because wall/door/passage/entrance carry useful building-level
semantics.

## BBox Smoke Model

Added:

```text
gsdiff/msd_hier/bbox_model.py
scripts/trainval_msd_hier_bbox.py
```

This is not the final diffusion model. It is a small edge-aware transformer
that predicts normalized bboxes from the hierarchical graphs. Its purpose is to
verify the new data path and give a fast learnability signal before implementing
corner/polygon diffusion.

The model consumes both:

```text
edge_present  # original GSDiff-style binary compatibility
edge_types    # MSD v6 typed edges, embedded in the smoke model
```

Coarse CUDA smoke:

```powershell
.\.venv\Scripts\python.exe scripts\trainval_msd_hier_bbox.py --task coarse --steps 3 --batch-size 8 --d-model 64 --layers 2 --heads 4 --val-every 2 --val-batches 2 --log-every 1 --out-dir outputs\msd_hier_bbox_coarse_smoke
```

Result:

```text
train_samples=4297
val_samples=537
params=123012
step 1 train_loss=0.1317
step 2 train_loss=0.1348 val_loss=0.0950
step 3 train_loss=0.1188 val_loss=0.0892
```

Local CUDA smoke:

```powershell
.\.venv\Scripts\python.exe scripts\trainval_msd_hier_bbox.py --task local --steps 3 --batch-size 16 --d-model 64 --layers 2 --heads 4 --val-every 2 --val-batches 2 --log-every 1 --out-dir outputs\msd_hier_bbox_local_smoke
```

Result:

```text
train_samples=27550
val_samples=3508
params=122948
step 1 train_loss=0.2018
step 2 train_loss=0.1694 val_loss=0.1700
step 3 train_loss=0.1592 val_loss=0.1612
```

Status:

```text
usable smoke path
```

Evidence:

```text
Both coarse and local loaders run on CUDA.
Forward/backward/checkpoint/validation all complete.
Loss decreases over the 3-step local smoke and validation is finite for both tasks.
```

Weakness:

```text
The target is bbox only, not polygon/corner geometry.
This does not yet prove final floorplan generation quality.
PyTorch currently emits a MultiheadAttention mask dtype performance warning
during smoke runs; it does not block training but should be cleaned up later.
```

## Grouping Rule

Current automatic grouping:

```text
unit group:
  room has valid unit_id and roomtype is not Stairs

public/stair/corridor group:
  roomtype is Stairs, or unit_id is missing
  public candidates are split by geometry connected components
```

Important correction from the first smoke test:

```text
Do not classify every Corridor as public.
```

In MSD, many corridors are inside a unit. Treating every corridor as public
caused each apartment to split into many one-room groups. The current rule keeps
valid-unit corridors inside their unit group.

## Current Full-Dataset Statistics

Run date: 2026-05-18

Status: usable as first data artifact; needs visual QA before training.

Evidence:

```text
floors: 5372
train floors: 4297
val floors: 537
test floors: 538
train local samples: 27550
val local samples: 3508
test local samples: 3507
```

Global stats:

```text
rooms_per_floor:
  min=15, median=25, mean=30.78, p90=49, p95=62, p99=116, max=331

raw_corners_per_floor:
  min=68, median=218, mean=264.50, p90=439, p95=554, p99=985.48, max=3308

groups_per_floor:
  min=1, median=5, mean=6.43, p90=10.9, p95=15, p99=32, max=174

rooms_per_group:
  min=1, median=5, mean=4.78, p90=10, p95=11, p99=13, max=24
```

Interpretation:

```text
Direct full-floor GSDiff remains too large for a single-scale model.
The local fine stage is much smaller: p95 is 11 rooms per group.
The coarse stage is also smaller than raw room/corner generation for most floors.
```

## Known Weaknesses

Status: not ready for final training claims.

Weakness:

```text
Some floors have many public/stair singleton groups.
Some samples have zero coarse edges, usually when the entire floor collapses to
one coarse group.
The current coarse target stores union polygons and bbox, but the first model
should probably start with bbox or simplified polygon only.
The initial full run emitted several Shapely distance RuntimeWarnings. The
script now routes geometry distance calls through `safe_distance`; rerun full
generation if warning-free logs are required.
```

Next:

```text
1. Inspect datasets/msd_hier_v6/vis/sample_*.png.
2. Run longer bbox smoke, e.g. 1k-5k steps, to check learnability.
3. Add bbox visualization for predicted coarse/local layouts.
4. Decide whether the next target is simplified polygon vertices or GSDiff-style corner graph.
5. Add polygon/corner diffusion after bbox models are stable.
```
