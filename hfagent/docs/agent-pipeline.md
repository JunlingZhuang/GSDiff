# Agent Pipeline

End-to-end request flow from a natural-language brief to an editable floor plan
with rooms, walls and doors.

## Overview

```
Browser (AgentWorkspace.tsx)
  │  POST /api/agent/generate  { text: "generate a clinic..." }
  ▼
FastAPI  hfagent/api.py  :8100
  │  understand(text, client)              # nodes/understand.py
  │  → Gemini generate_json(prompt, PROGRAM_SCHEMA)
  │  → sanitize_program()                  # drop unknown types, clamp counts
  │  → program = { building_type, rooms, adjacency }
  │
  │  generate_plan(program, client, session)        # floor_plan_generate.py
  │  # real2color correction loop, up to max_correction_rounds
       │
       ├─ FloorPlanGenerator (tools/floor_plan_generator.py)
       │    Pass 1  generate_real_plan()    text  → realistic architectural plan
       │            (rooms labelled patient_room_1.. ; doors drawn)   → *.real.png
       │    Pass 2  to_colorblock(real_png) image → flat colour blocks,
       │            ALL text/doors/windows removed (clean fill to parse)  → *.png
       │
       ├─ cv_parse(image)                   # tools/image_parser.py
       │    pixel classify → morphology / internal-hole fill → per-room instance map
       │    → nearest-room GAP FILL (rooms meet at wall centreline)
       │    → grid contour (orthogonal) → global axis snap
       │    → Plan { rooms: [Room{type, polygon}] }
       │
       ├─ count violations  (surplus / missing room types)
       │    exact? → early stop (no further rounds)
       │    else  → feed *.real.png + violation list back to Gemini,
       │            re-convert, next round
       │
       └─ keep the least-violating round →
            fix_room_counts(plan, requested)         # tools/plan_fixes.py
            → relabel / merge / split (shapely)  → exact counts, zero API
            │
            extract_room_adjacency(real_png)         # tools/room_adjacency_extractor.py
            → Gemini reads the REALISTIC plan → RoomGraph
            → door edges {room_a, room_b} ONLY — connectivity, no coordinates
            │
            place_doors(fixed_plan, room_graph)      # tools/door_placer.py
            → for each connected room-type pair sharing a wall,
              put a door CENTRED on that shared wall
            → Plan with walls + doors
  ▼
API returns JSON
  { id, program, report, plan, room_graph }
  ▼
FloorPlanEditor2D + FloorPlanView3D   (frontend)
```

`generate_plan` returns `(report, plan_dict, room_graph_dict)`. Per-program
artifacts are written under the run directory (see Outputs).

## Stages

### 1. Understand — `hfagent/nodes/understand.py`

Free-form natural language → structured building program. A *node* (always runs,
makes an LLM call), kept in `nodes/` apart from the pure-function `tools/`.

- `Gemini.generate_json()` with a strict `PROGRAM_SCHEMA` — the model is forced to
  emit valid structured data, no post-hoc parsing.
- `sanitize_program()` drops room types not in the palette, clamps counts 1–12,
  validates adjacency. Hallucinations are stripped here, not downstream.

Output: `{ building_type, rooms: [{type, count, approx_area_m2?}], adjacency }`

### 2. Generate — real2color — `hfagent/tools/floor_plan_generator.py`

Two Gemini image calls per round; the realistic plan is an image-space
chain-of-thought that lifts layout quality far above asking for blocks directly.

- **Pass 1 `generate_real_plan`** — program → a realistic architectural drawing. Rooms
  are labelled with **instance names** (`patient_room_1 … patient_room_N`) so the
  later adjacency read can name door endpoints unambiguously; doors are drawn.
  - **Two input entries, one output.** Without a boundary (`build_real_prompt`) the
    footprint is free. With a boundary image (`build_boundary_prompt`, passed as
    `generate_plan(..., boundary=<png bytes>)` or the `/api/agent/generate-from-boundary`
    endpoint), the outer walls follow that exact outline. This is the **only** point that
    differs between the two entries — everything from pass-2 onward is identical.
- **Pass 2 `to_colorblock`** — realistic plan → flat colour blocks: read each label,
  strip the trailing number, fill with that base type's legend colour, then **remove
  ALL text, doors, windows and fixtures**. A clean, unbroken fill is exactly what the
  parser segments — any retained text punches holes in room masks (see Findings).

There is no `generation_mode` (removed — it only ever had one value). The path is chosen
by `config.json: structure_mode` (see 2b); one mode, `direct_colorblock`, is a single-pass
mode that skips the realistic plan entirely.

### 2b. Structure mode — how the realistic plan becomes a `Plan` (`config.json: structure_mode`)

Two interchangeable ways to turn the realistic plan into a parsed `Plan`. Pick via
`structure_mode`; both feed the identical downstream (repair → walls → doors → adjacency).

- **`colorblock`** (default) — pass-2 `to_colorblock` renders a flat colour-block image,
  then `cv_parse` (Stage 3) reads it. Image-native, but image models won't fully obey a
  segmentation mask (leaked text, door arcs painted as corridor spikes, gradients) which
  pollutes the parse.
- **`json`** (default) — `tools/structure_reader.read_structure` has the VLM *read* the
  realistic plan into one JSON object: each room as the **grid cells it covers** on a fixed
  `GRID_COLS×GRID_ROWS` (32×18, 16:9) grid — `rooms:[{id,type,cells:[[c1,r1,c2,r2],…]}]` +
  `doors:[{room_a,room_b}]`. `tools/rectify.rectify` paints the cells onto one label grid
  (later room wins overlaps) and traces each room's region into an orthogonal polygon.
  **Skips `to_colorblock` and `cv_parse`** (the two brittle stages) and reads doors in the
  same call (no separate `extract_room_adjacency`). Why **cells, not pixel boxes**: VLMs are
  unreliable at precise coordinates (they skew the extents → warped aspect ratio); discrete
  cell choice needs no precise coords, so the building's aspect is fixed by the grid, and a
  bent corridor / L-shaped room keeps its shape (several cell-rects). JSON is schema-
  validated and counts are exact. MVP trade-off: a room split into disconnected cell blocks
  keeps only its largest block.

### 3. Parse — `hfagent/tools/image_parser.py`  (colorblock mode only)

Colour-block PNG → `Plan` (typed room polygons), fully deterministic.

1. Classify every pixel to the nearest palette colour (`ROOM_RGB`, capped distance).
2. Per type: open (despeckle) → close (reconnect across thin text strokes) → drop
   sub-room blobs. Each surviving connected component becomes one **instance id**.
   Label / door-arc holes are deliberately left for step 3 — never filled here — so a
   colour block can never paint over an enclosed room of a different colour.
3. **Centre-line fill** (`_grow_to_centerline`): every non-room pixel **within half a
   wall** of a seed is handed to the nearest seed. Adjacent rooms therefore meet exactly
   on the wall mid-line and label / door holes are reclaimed by their surrounding room,
   while the black wall band itself survives as the real divider — walls are boundaries,
   not gaps. The half-wall cap is the whole point: it replaced an earlier *whole-footprint*
   nearest-seed flood that ate walls, let a room balloon across a missing neighbour, and
   sprouted long medial-axis triangles where three rooms met.
4. **Full-resolution corners** (`_region_corners`): trace each region's outer boundary
   (`RETR_EXTERNAL`, so a toilet bitten out of a corner keeps the room L/U-shaped) and
   reduce it to its real corners with `approxPolyDP`. No coarse grid — small rooms
   (≈45 px toilets) keep their true extent.
5. **Global axis snap then right-angle** (`raster_geometry.snap_axes`, then
   `_rectilinearize`): cluster all x/y across all rooms onto shared lines so chamfered
   corners become truly axis-aligned and shared walls are exactly collinear; only then
   force right angles. Snapping first is what keeps right-angling from picking a
   wrong-side corner and self-intersecting. Clusters are anchored on the group's first
   value (span ≤ tol), never chained on the previous value.

### 4. Correction Loop — `hfagent/floor_plan_generate.py`

Verifier-driven iteration — the VLM only draws, deterministic `cv_parse` adjudicates.

| Step | Action |
|------|--------|
| Generate | `generate_real_plan` (round 1) / edit realistic plan with feedback (round ≥2) → `to_colorblock` |
| Verify | `cv_parse` → count room types vs program |
| Exact? | **Early stop** — no further rounds |
| Otherwise | Feed `*.real.png` + quantified violations back to Gemini, re-convert |
| On cap | Keep the **least-violating** round |

`max_correction_rounds` (`config.json`) is the single loop cap — there is no stall
detection (it was removed as redundant; see Findings).

### 5. Deterministic Count Repair — `hfagent/tools/plan_fixes.py`

Backstop when the loop exits with residual violations. Plan (JSON) space — zero API
cost, guaranteed termination.

| Operation | When | How |
|-----------|------|-----|
| **Relabel** | Surplus type A, missing type B | Rename the smallest surplus room |
| **Merge** | Two adjacent same-type rooms, count too high | Shapely buffer + union |
| **Split** | Largest room of a type, count too low | Bisect across the longer axis |

### 6. Room Adjacency — `hfagent/tools/room_adjacency_extractor.py`

Which rooms connect through a door, read from the **realistic** plan (the colour block
has no doors — pass 2 erased them).

- The LLM returns **connectivity only**: a list of `{room_a, room_b}` edges, using the
  instance labels (`exterior` for an outside door). **No coordinates** — the model is
  unreliable at pixel/normalized positions (it mixes axes and scales), so geometry, not
  the model, decides where a door sits.
- Output: `RoomGraph` (`schema/roomgraph.py`) — `rooms` are nodes, `doors` are edges.

### 7. Walls — `hfagent/tools/plan_to_walls.py`

`plan_to_walls(plan)` canonicalises room-polygon boundaries into the explicit
`walls[]` geometry layer of the authoritative plan (docs/agent/02-data-model.md
§3.2). It snaps near-collinear rectilinear edges onto global axes, unions
overlapping / touching intervals, and emits one `Wall` per maximal occupied segment.
This keeps large parsed plans from exploding into hundreds of duplicate wall
fragments when room polygons are slightly misaligned or meet at T-junctions. Doors
hang on these canonical walls.

Out of scope here: reconstructing a full editable WallGraph topology with shared
node ids. Room polygons stay independent; frontend CAD editing still owns node-level
merge / T-insertion behavior.

### 8. Door Placement — `hfagent/tools/door_placer.py`

`RoomGraph` connectivity + reconstructed geometry → physical doors **hung on real walls**.

- For every pair of rooms whose **types** are connected in the graph and which actually
  share a wall, hang a `Door` **centred** on that shared wall. Matching is by type (not
  instance): instance identity is lost in the colour block, and "every patient_room–
  corridor shared wall gets a door" is the intended behaviour without fragile OCR.
- The shared wall is the longest straight segment of `a.boundary ∩ b.buffer(tol)` —
  robust to the sub-pixel gaps/overlaps left by rasterised parsing, and never the
  diagonal chord across a stepped corner. The door then hangs on the collinear
  `plan_to_walls` wall nearest that segment: `Door.wall_id` + `position` (0..1).
- Exterior edges get a door on the room's longest outside wall.
- `place_doors` also returns, per door, the two room ids it links — used to build the
  `adjacency_graph` (edges carry `via = door id`). `render_plan` renders the openings
  natively from `plan.walls` + `plan.doors`.

The assembled result is a single authoritative `Plan` (docs §3.2): `rooms` + complete
`walls` + doors-on-walls + `adjacency_graph` — written as `plan.json` and returned as
the response's `plan`.

## Key Design Principles

- **LLM-Modulo throughout** — the model proposes *structure* (a program, a layout, a
  connectivity graph); deterministic code *adjudicates and places* (parse, count repair,
  door geometry). The model never judges its own output and never supplies coordinates.
- **Correction is pixel-space; repair is Plan-space.** Counts are a structural constraint
  and are fixed in structure space, not by re-prompting (which avalanches on complex
  plans).
- **Room graph, not wall graph.** The structural output is rooms-as-nodes /
  doors-as-edges. There is no wall-segment topology layer.

## Outputs (per program, under the run directory)

```
gemini_rN.real.png    realistic plan for round N (labels + doors visible here)
gemini_rN.png         colour block for round N (clean fill, parser input)
parsed.json           cv_parse of the best round
fixed.json            after deterministic count repair (rooms only)
recon.png             deterministic re-render of the fixed plan
graph.json            RoomGraph — rooms + door edges (connectivity, no coords)
plan.json             AUTHORITATIVE Plan (docs §3.2): rooms + walls + doors-on-walls + adjacency_graph
recon_with_door.png   recon.png with door openings cut into the walls
report.json           counts, rounds, fixes, room_graph summary (+ walls count)
```

Run directories are timestamped: `out/eval/<YYYYmmdd-HHMMSS>/` (CLI), `out/api/<ts>/`
(API), `out/tests/<test_name>/` (pytest).

## File Map

| File | Role |
|------|------|
| `hfagent/api.py` | FastAPI endpoint :8100 — flat routes, `Depends(get_client)` |
| `hfagent/graph.py` | `run_pipeline()` — understand + generate_plan |
| `hfagent/floor_plan_generate.py` | `generate_plan()` — correction loop, repair, doors |
| `hfagent/nodes/understand.py` | NL → structured program (LLM node) |
| `hfagent/evaluate.py` | CLI sweep over `programs.json` (real Gemini) |
| `hfagent/tools/floor_plan_generator.py` | real2color generation + room-adjacency read |
| `hfagent/tools/image_parser.py` | `cv_parse` colour-block PNG → Plan |
| `hfagent/tools/room_adjacency_extractor.py` | realistic plan → RoomGraph (connectivity) |
| `hfagent/tools/plan_to_walls.py` | room polygons → complete deduped wall list |
| `hfagent/tools/door_placer.py` | RoomGraph + plan → doors hung on real walls + adjacency edges |
| `hfagent/tools/plan_fixes.py` | deterministic count repair |
| `hfagent/tools/render_plan.py` | Plan → PNG (rooms + wall openings) |
| `hfagent/schema/palette.py` | room type ↔ RGB |
| `hfagent/schema/plan.py` | Pydantic Plan / Room / Wall / Door |
| `hfagent/schema/roomgraph.py` | RoomGraph / RoomNode / Door (graph edge) |

> Findings & design-decision history: [phase0a-findings.md](phase0a-findings.md),
> [phase0b-findings.md](phase0b-findings.md),
> [phase1-doors-findings.md](phase1-doors-findings.md).
