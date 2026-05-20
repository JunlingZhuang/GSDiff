# procedural/

Rule-based (procedural) floor plan generator for MSD-style apartments.

Independent from the ML stack (`gsdiff/`, `digress/`, `GRAN/`). No PyTorch dependency. Pure geometry + graph.

## Why this exists

- GSDiff is ML (graph diffusion). This module is the deterministic, rules-only alternative.
- Same input contract as `/api/generate/topology`: `(rooms, adjacency, boundary)`.
- Output: room polygons + walls (deterministic, debuggable, fast).
- Intended to mount under `/api/generate/procedural` in `app/backend/`.

## Layout

```
procedural/
├── README.md
├── __init__.py
├── notebooks/
│   └── 01_observe_msd.ipynb     # eyeball MSD samples → derive rules
├── rules.py                     # hand-written rules + (optional) MSD-derived constants
├── geometry.py                  # shapely / boundary helpers
└── generator.py                 # main entry: generate(graph, boundary) -> polygons
```

## Pipeline (target)

```
graph_in (nodes=types, edges=adjacency) + boundary polygon
  │
  ├─ Step 1: identify hub node (highest-degree, usually Living/Corridor)
  ├─ Step 2: extract circulation backbone (Steiner tree)
  ├─ Step 3: embed graph into boundary (medial-axis aware)
  ├─ Step 4: split boundary along edges → candidate polygons
  ├─ Step 5: bipartite-match polygons ↔ graph nodes (area + adjacency priors)
  └─ Step 6: emit walls (shapely difference / buffer)
                  ↓
            room polygons + walls
```

## Env

Uses repo `.venv` (Python 3.10). Add jupyter + ipykernel:

```bash
uv pip install --python ../.venv/Scripts/python.exe jupyter ipykernel
../.venv/Scripts/python.exe -m ipykernel install --user --name gsdiff --display-name "GSDiff (.venv)"
```

Then open `notebooks/01_observe_msd.ipynb` and pick the `GSDiff (.venv)` kernel.
