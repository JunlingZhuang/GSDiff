# CLAUDE.md — hfagent

Guidance for Claude Code when working inside `hfagent/` (the healthcare floor-plan
generation backend). This overrides general repo defaults for this subtree.

## What hfagent is

A text → floor-plan agent. A natural-language brief becomes a structured room
program, which becomes a labelled architectural plan, which becomes a flat
colour-block image, which is parsed back into a vector plan plus a **room graph**
(rooms connected by doors). FastAPI serves it; the frontend renders/edits it.

## Pipeline (one flow; the path is chosen by `config.json` structure_mode)

```
text
 └─ understand()                nodes/understand.py     -> program {rooms, adjacency}
     └─ generate_plan()         floor_plan_generate.py  -> report, plan, room_graph
         └─ FloorPlanGenerator  tools/floor_plan_generator.py
             generate_real_plan(program)        text  -> realistic plan PNG   (pass 1)
             to_colorblock(real_png)            image -> colour-block PNG      (pass 2)
             extract_room_adjacency(real_png)   image -> RoomGraph (doors)
```

There is ONE flow and **no `generation_mode`** (it was removed — it only ever had one
value). `config.json` holds a `modes` block — one entry per structure mode, each with its
own `description`, `text_model`, `image_model` and `boundary` path — plus a top-level
`structure_mode` selecting the active one. Most modes run the two-pass **real2color** image
chain above (realistic plan → colour-block); one — `direct_colorblock` — skips pass-1 and
draws the colour-block in a single pass (see `docs/direct-colorblock-findings.md`). Do not
reintroduce a `generation_mode` or a `pipeline` parameter name.

**Two input entries, one pipeline.** Pass-1 has two forms — a program alone
(`build_real_prompt`, free footprint) or a program **+ a boundary image**
(`build_boundary_prompt`, outer walls follow the given outline; pass
`generate_plan(..., boundary=<png bytes>)`, CLI `--boundary`, or the
`/api/agent/generate-from-boundary` endpoint). Pass-1 is the ONLY divergence; the
realflow plan and everything downstream of it are identical for both. Keep it that
way — never fork the downstream.

**Structure modes — how the program becomes a `Plan`** (each is a `config.json: modes.*`
block with its own `text_model` / `image_model` / `boundary`; the active one is
`config.json: structure_mode`, overridable with `--structure-mode` or
`generate_plan(..., structure_mode=...)`):
- `colorblock` (current config default) — `to_colorblock` image + `cv_parse`. Brittle: image
  models won't draw a clean segmentation mask (leaked text, door arcs as corridor spikes).
- `json` — `tools/structure_reader.read_structure` (VLM reads the realistic plan
  into `rooms[cells]+doors` JSON: each room as the **grid cells it covers** on a fixed
  32×18 grid) + `tools/rectify.rectify` (paint cells → trace orthogonal `Plan`). Skips
  `to_colorblock` + `cv_parse`, reads doors in the same call. **Cells, not pixel coords** —
  the model only picks discrete cells, so it can't skew the building's aspect (fixed by the
  grid) and an L/T corridor keeps its shape; counts/types are schema-checked.
- `direct_colorblock` — **one pass, no realistic plan**: the image model draws the flat
  colour-block straight from the program (`generate_colorblock_direct` +
  `build_direct_colorblock_prompt`), then `cv_parse`. Skips `generate_real_plan` +
  `to_colorblock`, so the conversion step that injects door-arc blobs / count drift is
  gone — cleaner geometry, one image call per round, ~half the cost. Doors come from
  `program["adjacency"]`, not an image read (see Doors below). Best on `gemini-3-pro-image`
  for plans up to ~33–40 rooms; degrades past ~80 (see `docs/direct-colorblock-findings.md`).
- `linework` — **geometry-only, no OCR, no room typing**: pass-1 draws a clean line plan
  (`build_linework_prompt`), then `tools/linework_tracer.trace_linework` deterministically
  traces exactly the drawn walls (faithful trace + short-stub pass, nothing invented),
  confirms a door ONLY where a quarter-circle swing arc straddles a wall gap, bridges the
  walls at confirmed doors, post-processes (posts absorbed, faces merged, ink-gated corner
  snap, whiskers dropped) and polygonizes the closed rooms into an untyped px-unit `Plan`
  (`type="unknown"`, ids `r1..rN`). Runs a **single round** — untyped rooms give no
  per-type violation signal — and the report compares total rooms traced vs the program
  total. Every run writes the fixed trace artifacts (see Output artifacts).

All structure modes share the SAME downstream (`fix_room_counts` → `plan_to_walls` /
`place_doors` → unified `Plan`). The divergence is ONLY in how the `Plan` is produced —
do not fork the downstream.

### Correction loop (in `generate_plan`)

- Runs up to `max_correction_rounds` (config). Each round: real2color → `cv_parse`
  → count rooms vs program.
- **If a round has zero violations, break immediately** — no further rounds.
- Otherwise feed the violations back as an edit instruction to the realistic plan
  and retry. Keep the **least-violating** round; deterministic `fix_room_counts`
  repairs whatever the VLM never resolved.
- There is no stall detection — `max_correction_rounds` is the single cap. Do not
  add stall counters.
- `linework` is exempt: it always runs exactly ONE round (untyped rooms have no
  per-type count signal to correct on) and skips `fix_room_counts`.

## Doors — LLM reads the graph, geometry places the door

Split of responsibility, on purpose:

1. **The LLM returns connectivity only.** `tools/room_adjacency_extractor.py`
   (`extract_room_adjacency`) asks the model which room pairs share a door and
   returns a `RoomGraph` (`schema/roomgraph.py`): rooms as nodes, doors as edges
   `{room_a, room_b}` — **no coordinates**. Models are unreliable at pixel/normalized
   door positions (they mix axes and scales), so we never trust coordinates from them.
2. **Post-processing places the physical door.** `tools/door_placer.py`
   (`place_doors`) takes the reconstructed plan + the graph and, for every pair of
   rooms whose **types** are connected and which actually share a wall, drops a door
   **centred** on that shared wall. Exterior edges get a centred door on the room's
   longest outside wall. `render_plan` renders the openings natively from
   `plan.walls` + `plan.doors`.

Rules:
- **Room graph, not wall graph.** No `WallGraph` / party-wall topology. Don't add one.
- **Never ask the LLM for door coordinates.** Connectivity only; geometry decides position.
- **`direct_colorblock` sources connectivity from `program["adjacency"]`**, not an image
  read (it has no realistic plan): one `RoomNode` per type, one `Door` per adjacency pair
  (`_room_graph_from_program`), then the SAME `place_doors` hangs doors by type on shared
  walls. Same "connectivity + geometric placement" split — the connectivity is just the
  program the LLM already produced.
- **`linework` sources connectivity from the drawing's own door arcs** — geometric
  detection, no LLM read at all: each arc-confirmed door probes one point on either side
  of its opening and names the two flanking plan room ids (`"exterior"` when a side is
  not a room). Those edges use exact ids (`r1..rN`), so `place_doors` matches them via
  `exact_connected`, not by type.
- Matching is by room **type**, not instance — instance identity is lost in the colour
  block, and "every patient_room–corridor shared wall gets a door" is the intended
  behaviour without fragile disambiguation.
- Rooms with `count > 1` are labelled `type_1 .. type_N` in `build_real_prompt` so the
  LLM can name edge endpoints; `build_convert_prompt` strips the number for the legend
  colour. Keep `room_instance_ids()` in sync with that labelling.

## Layout: nodes vs tools

- `nodes/` — orchestration steps that **always run** and make a model call
  (e.g. `understand`). Reserved for LangGraph nodes (graph wiring lands in Phase 4).
- `tools/` — pure, individually testable functions a node calls
  (`floor_plan_generator`, `room_adjacency_extractor`, `image_parser`,
  `render_plan`, `plan_fixes`).
- `schema/` — pydantic models (`plan`, `roomgraph`) + the shared `palette`.
- `floor_plan_generate.py` is the orchestrator; `graph.py`/`api.py` are thin
  wrappers; `evaluate.py` is the batch CLI harness.

## Output artifacts (per program, under the run dir)

```
gemini_rN.real.png    realistic plan for round N (doors visible here)
gemini_rN.png         colour-block for round N (parser input)
parsed.json           cv_parse of the best round
fixed.json            after deterministic count repair
recon.png             deterministic re-render of the fixed plan
graph.json            RoomGraph: rooms + door edges (connectivity, no coords)
doors.json            fixed plan with walls + doors placed on shared walls
recon_with_door.png   recon.png with door openings cut into the shared walls
report.json           counts, rounds, fixes, room_graph summary
```

`structure_mode=linework` replaces the colour-block round images and the rendered
`recon.png` with the tracer's FIXED artifact set, written every run:

```
walls_overlay.png     faithful traced walls in red on the source drawing
doors_overlay.png     faithful walls grey + every detected swing arc/leaf/hinge
recon.png             bridged walls (pre-postprocess) + door arc markers
recon_post.png        post-processed wall graph + door arc markers
rooms_colorful.png    closed room polygons, distinct deterministic colours,
                      walls black on top, doors as white gaps
```

`generate_plan` returns `(report, plan_dict, room_graph_dict)` — the third element
is the room graph (key `room_graph` in API/`run_pipeline` responses).

## Models

Model IDs are **never hardcoded**. `llm.py` resolves text/image models from
`models.list()` by preference order (overridable via `HFAGENT_TEXT_MODEL` /
`HFAGENT_IMAGE_MODEL`). `generate_image` and `generate_json` both accept a list
mixing `str` and image `bytes` for multimodal turns. When adding a capability,
extend `llm.py`, don't pin a model elsewhere.

**Image bytes must be format-agnostic.** Auto-resolution means the image model
can change under you, and different models return images differently — raw PNG
bytes vs **base64-encoded JPEG text** in `inline_data.data` (e.g.
`gemini-3-pro-image`). `decode_image_bytes()` sniffs magic bytes and base64-decodes
when needed; outgoing image parts sniff their mime type (`_sniff_mime`) instead of
assuming PNG. Never write `inline_data.data` to disk directly, and never hardcode
`mime_type="image/png"` for an image whose source you don't control.

## Testing

- Run: `python -m pytest hfagent/tests/ -q` (from repo root).
- Deterministic — no real API calls. `tests/mock_vlm.py` scripts image responses
  (2 image calls per real2color round) and `generate_json` returns scripted doors.
- Persistent artifacts go to `hfagent/out/tests/<test_name>/` via the `out_dir`
  fixture (`tests/conftest.py`) — not pytest's temp dir — so images are inspectable.
- `tests/test_programs.py` runs every entry in `programs.json` through the pipeline
  with a perfect synthetic plan (`synth.make_plan_for_program`). New room types
  must be added to `schema/palette.py` first.

### Mandatory for any pipeline change

The mock tests prove wiring, **not** model behaviour. Any **major change to the
generation pipeline** (real2color flow, correction loop, prompts, room-adjacency
extraction, parsing, rendering, schemas) MUST be validated by a **full real run**:

```
python -m hfagent.evaluate          # ALL programs in programs.json, real Gemini
```

Run the full suite — never just `--n 1`. Inspect the resulting
`gemini_r*.real.png`, `recon_with_door.png`, `graph.json` and `summary.json`
before considering the change done. Requires a `GEMINI` key in `.env.local`.

## Conventions

- **Names and architecture must be optimal — this is a hard requirement, not a
  preference.** Every variable, function, file and module name must be the clearest
  intention-revealing name for what it does; if a better name exists, rename to it
  (and update every reference) rather than leave a stale or vague one. The code
  architecture must be the best reasonable structure: single responsibility, no dead
  code, no duplicated logic, no leftover scaffolding, nodes vs tools kept clean.
  When a change makes an existing name or structure suboptimal, fixing it is part of
  that change.
- OOP where it carries state: `FloorPlanGenerator` holds `program` + `client` so
  they aren't threaded through every call. `run()` is the full single-shot flow
  (both passes + room-adjacency extraction + file writes).
- Output dirs are timestamped (`out/eval/<YYYYmmdd-HHMMSS>/`, `out/api/<ts>/`) so
  runs never clobber each other. Don't write to a fixed run dir.
- Never commit `.env.local`.
- Do not add `Co-Authored-By: Claude` or any Claude attribution to commit messages.
