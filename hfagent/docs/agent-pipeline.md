# Agent Pipeline

End-to-end request flow from the browser to an editable floor plan.

## Overview

```
Browser (AgentWorkspace.tsx)
  │  POST /api/agent/generate  { text: "generate a clinic..." }
  ▼
FastAPI  hfagent/api.py  :8100
  │  understand(text, client)        # understand.py
  │  → Gemini generate_json(prompt, PROGRAM_SCHEMA)
  │  → sanitize_program()            # drop unknown types, clamp counts
  │  → program = { building_type, rooms, adjacency }
  │
  │  generate_plan(program, client, session)   # pipeline.py
  │  # correction loop, up to 3 rounds
       │
       ├─ Pass 1: generate_colorblock(program, pipeline="two-pass")
       │    generate_colorblock.py
       │    → build_realistic_prompt() → Gemini image → *.realistic.png
       │    → CONVERT_PROMPT           → Gemini image → *.png  (color-block)
       │
       ├─ cv_parse(image)
       │    cv_parse.py
       │    → pixel classify → morphology clean → dilate (half-wall)
       │    → grid_contour → snap_axes
       │    → Plan { rooms: [Room{type, polygon}] }
       │
       ├─ count violations  (surplus / missing room types)
       │    exact?  → early stop
       │    stall ≥ 2 rounds with no improvement? → early stop, keep best
       │
       ├─ feed image + violations back to Gemini → next round image
       │
       └─ best round Plan →
            fix_room_counts(plan, requested)    # deterministic repair
            plan_fixes.py
            → relabel / merge / split  (shapely geometry)
            → guarantees exact counts, zero API calls
            │
            plan_to_wallgraph(plan)
            build_wallgraph.py
            → quantize vertices → merge coincident nodes
            → T-junction insert → deduplicate walls
            → WallGraph { nodes, walls, rooms }
  ▼
API returns JSON
  { id, program, report, plan, wallgraph }
  ▼
agentToKernelGraph(wallgraph)    # agent-plan.ts (frontend)
  → mm→m unit conversion + y-axis flip
  → WallGraph (kernel format)
  ▼
FloorPlanEditor2D + FloorPlanView3D   # immediately editable
```

## Stages

### 1. Understand — `hfagent/understand.py`

Converts free-form natural language into a structured building program.

- Calls `Gemini.generate_json()` with a strict JSON schema (`PROGRAM_SCHEMA`) so the model is forced to output valid structured data — no post-hoc parsing.
- `sanitize_program()` drops any room types not in the palette, clamps counts to 1–12, and validates adjacency pairs. LLM hallucinations are stripped here, not downstream.

Output: `{ building_type, rooms: [{type, count, approx_area_m2?}], adjacency }`

### 2. Generate Color-Block — `hfagent/tools/generate_colorblock.py`

Two-pass image generation — better layout quality than direct color-block generation.

- **Pass 1** (`build_realistic_prompt`): generate a realistic architectural floor plan drawing, labeled with room names.
- **Pass 2** (`CONVERT_PROMPT`): instruct Gemini to re-color by room-type legend, remove all doors / windows / annotations. Produces a clean color-block PNG.

Single-pass (`pipeline="direct"`) is available but two-pass is the default — it gives professional corridor layouts rather than diagram-style blobs.

### 3. Parse — `hfagent/tools/cv_parse.py`

Converts the color-block PNG into a `Plan` (list of typed room polygons).

1. Classify each pixel to the nearest palette color (`ROOM_RGB`).
2. Morphological clean (open + close) to remove door arcs and noise.
3. Dilate by half-wall width to close gaps between adjacent rooms.
4. **Grid-contour** (`_grid_contour`): trace polygon on a coarse grid → constructively orthogonal output.
5. **Global axis-snap** (`_snap_axes`): cluster all x/y coordinates across all rooms, snap clusters to their mean → shared wall edges are exact, not approximate.

### 4. Correction Loop — `hfagent/pipeline.py`

Verifier-driven iteration (§5.3 discipline):

| Round | Action |
|-------|--------|
| Generate image | `generate_colorblock()` |
| Verify | `cv_parse()` → count violations |
| Exact? | Early stop — return immediately |
| Stall ≥ 2? | Early stop — keep best round |
| Otherwise | Feed `image + violations` back to Gemini, next round |

The VLM only draws; deterministic code (`cv_parse`) adjudicates. The LLM never self-judges compliance.

### 5. Deterministic Count Repair — `hfagent/tools/plan_fixes.py`

Backstop when the correction loop exits with residual violations. Operates in Plan (JSON) space — zero API cost, guaranteed termination.

| Operation | When | How |
|-----------|------|-----|
| **Relabel** | Surplus type A, missing type B | Rename the smallest surplus room — no geometry change |
| **Merge** | Two adjacent rooms of same type, count too high | Shapely buffer + union |
| **Split** | Largest room of a type, count too low | Bisect across longer axis with shapely box intersection |

### 6. Wall Graph — `hfagent/tools/build_wallgraph.py`

Converts the flat polygon array into a topology-correct wall graph.

1. **Quantize + merge** coincident vertices → shared `WallNode` entries.
2. **T-junction split**: for each wall edge (a→b), find nodes lying strictly on it and insert them into the loop — both rooms reference the same node.
3. **Wall dedup**: canonical `(min_id, max_id)` key; each wall records which `RoomFace` loops it belongs to.

Output: `WallGraph { nodes, walls, rooms }` — walls are first-class citizens, rooms are node-id loops with no private coordinates.

### 7. Frontend Conversion — `app/components/floorplan/agent-plan.ts`

`agentToKernelGraph(wallgraph)` converts the API response to the frontend kernel format:

- Unit conversion: mm → m (or px → m depending on `units` field)
- Y-axis flip: image coordinates (top-left origin) → world coordinates (bottom-left origin)
- Returns a `WallGraph` identical in structure to what `FloorPlanEditor2D` and `FloorPlanView3D` use natively — the generated plan is immediately editable.

## Key Design Principles

- **VLM only draws** — it never judges its own output. `cv_parse` (deterministic code) is the verifier.
- **Correction loop is pixel-space** — feed the image + violation list back, re-generate.
- **Count repair is Plan-space** — shapely geometry ops (relabel / merge / split), no API cost.
- **4 layers of correctness**: prompt engineering → pixel correction loop → deterministic repair → wall graph topology.

## File Map

| File | Role |
|------|------|
| `hfagent/api.py` | FastAPI endpoint, port 8100 — flat routes, `Depends(get_client)` |
| `hfagent/graph.py` | `run_pipeline()` — understand + generate_plan, two sequential calls |
| `hfagent/pipeline.py` | `generate_plan()` — correction loop, deterministic repair, wall graph |
| `hfagent/understand.py` | NL → structured program |
| `hfagent/run_phase0a.py` | CLI harness for offline evaluation |
| `hfagent/tools/generate_colorblock.py` | Two-pass image generation |
| `hfagent/tools/cv_parse.py` | Color-block PNG → Plan |
| `hfagent/tools/plan_fixes.py` | Deterministic count repair |
| `hfagent/tools/build_wallgraph.py` | Plan → WallGraph topology |
| `hfagent/schema/palette.py` | Room type ↔ RGB mapping |
| `hfagent/schema/plan.py` | Pydantic Plan / Room / Wall |
| `hfagent/schema/wallgraph.py` | WallNode / WallSeg / RoomFace |
| `app/components/floorplan/agent-plan.ts` | API response → kernel WallGraph |
| `app/components/floorplan/AgentWorkspace.tsx` | Agent tab UI |
| `app/lib/api.ts` | `generateAgentPlan()` fetch call |
