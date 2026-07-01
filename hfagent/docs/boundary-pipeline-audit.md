# Boundary pipeline audit (read-only)

Audit of the claim: the **input-boundary** feature only affects **pass-1** (the
realistic / first image generation) and is otherwise **orthogonal to
`structure_mode`**, so it should be a single ON/OFF toggle that composes with
every structure mode — injecting a boundary constraint into pass-1 and passing
the boundary image through.

Scope: code-reading only. No edits to `config.json`, `floor_plan_generate.py`, or
`hfagent/CLAUDE.md`. Citations are `file:line` at read time.

## TL;DR

- **The claim holds.** `boundary` enters the pipeline at exactly one point — the
  `FloorPlanGenerator` constructor (`floor_plan_generate.py:149`) — and is consumed
  only by the two pass-1 producers (`generate_real_plan`, `generate_colorblock_direct`).
  Nothing downstream of pass-1 takes a boundary argument.
- **All 4 structure modes already support boundary** at the generator level: the
  correct boundary-aware prompt builder is selected AND the boundary image is
  prepended to the model turn. No per-mode wiring gap.
- **The real gap is exposure.** `boundary` is reachable via the programmatic arg,
  `evaluate.py --boundary`, a per-program `"boundary"` field, and the
  `/api/agent/generate-from-boundary` endpoint — but it is **not in `config.json`**,
  `load_config()` does not return it, and the two config-driven entry points
  (`api.py /api/agent/generate`, `graph.py run_pipeline`) cannot enable it. There is
  no single config toggle, which is what the orthogonality of the feature warrants.

## 1. Where `boundary` flows

Single ingress, single store, two consumers:

- `generate_plan(..., boundary: bytes | None = None, ...)` — the only place a
  boundary enters the orchestrator. `floor_plan_generate.py:119`.
- Passed straight into the generator at construction (once per run, before the
  correction loop): `FloorPlanGenerator(program, client, boundary=boundary, ...)`.
  `floor_plan_generate.py:146-152`.
- Stored as `self.boundary`. `floor_plan_generator.py:282`.
- Consumed only by the two pass-1 producers:
  - `generate_real_plan()` — `floor_plan_generator.py:290-308`.
  - `generate_colorblock_direct()` — `floor_plan_generator.py:316-326`.

`generate_real_plan` has three branches, all keyed on `self.boundary`/`drawing_mode`:

- linework: `build_linework_prompt(program, boundary=self.boundary is not None)`,
  then `contents = [self.boundary, prompt] if self.boundary is not None else prompt`.
  `floor_plan_generator.py:297-301`.
- standard + boundary: `build_boundary_prompt(program)` and
  `generate_image([self.boundary, prompt])`. `floor_plan_generator.py:302-305`.
- standard + free footprint: `build_real_prompt(program)`, prompt only.
  `floor_plan_generator.py:306-308`.

`generate_colorblock_direct`:
`build_direct_colorblock_prompt(program, boundary=self.boundary is not None)`, then
`contents = [self.boundary, prompt] if self.boundary is not None else prompt`.
`floor_plan_generator.py:323-326`.

Boundary-aware prompt builders:

- `build_boundary_prompt` — `floor_plan_generator.py:114-126` (adds a `BOUNDARY (hard
  requirement)` block; shares `_PLAN_RULES` with the free-footprint prompt).
- `build_linework_prompt(program, boundary=False)` — `floor_plan_generator.py:141-163`
  (boundary intro at `143-148`).
- `build_direct_colorblock_prompt(program, boundary=False)` —
  `floor_plan_generator.py:218-259` (boundary intro at `231-237`).

External consumers that supply the bytes:

- `evaluate.py` — `--boundary` flag (`63-64`), per-program `"boundary"` field with
  precedence `args.boundary or program.get("boundary")` (`102-106`), existence checks
  (`48-49`), passed to `generate_plan` (`113`).
- `api.py` — `/api/agent/generate-from-boundary` reads the upload and passes it
  (`78-95`, bytes at `94`).

## 2. Per-mode pass-1 boundary support

For each structure mode: the pass-1 entrypoint, whether the boundary-aware prompt
builder is used, and whether the boundary **image** is actually handed to the model.

| structure_mode | pass-1 entrypoint | boundary-aware prompt builder used? | boundary image passed to model? | gap |
|---|---|---|---|---|
| `colorblock` | `generate_real_plan()` standard branch — `floor_plan_generate.py:195`, `floor_plan_generator.py:302-305` | yes — `build_boundary_prompt` (`:303`) | yes — `generate_image([self.boundary, prompt])` (`:305`) | none |
| `json` | `generate_real_plan()` standard branch — `floor_plan_generate.py:195`, then `read_structure(real_png, ...)` at `:221` | yes — `build_boundary_prompt` (`:303`) | yes — `[self.boundary, prompt]` (`:305`) | none — boundary lands in `real_png`; `read_structure` reads the already-bounded plan (correct, no boundary arg) |
| `linework` | `generate_real_plan()` linework branch — `floor_plan_generate.py:195`, `floor_plan_generator.py:297-301`; then `parse_linework(real_png, program)` at `:239` | yes — `build_linework_prompt(program, boundary=True)` (`:298`) | yes — `[self.boundary, prompt]` (`:300`) | none |
| `direct_colorblock` | `generate_colorblock_direct()` — `floor_plan_generate.py:171`, `floor_plan_generator.py:316-326` | yes — `build_direct_colorblock_prompt(program, boundary=True)` (`:323`) | yes — `[self.boundary, prompt]` (`:325`) | none |

Answers to the specific sub-questions:

- (a) **linework** — yes. `generate_real_plan` (drawing_mode=linework) calls
  `build_linework_prompt(program, boundary=self.boundary is not None)` and prepends
  the image via `[self.boundary, prompt]`. `floor_plan_generator.py:297-301`.
- (b) **direct_colorblock** — yes on both halves. `generate_colorblock_direct` uses
  `build_direct_colorblock_prompt(program, boundary=...)` and prepends the image
  (`floor_plan_generator.py:323-326`); and `generate_plan`'s direct branch routes the
  boundary through because the generator is built once with `boundary=boundary`
  (`floor_plan_generate.py:146-152`) and the direct branch calls
  `generator.generate_colorblock_direct()` (`floor_plan_generate.py:171`), which reads
  `self.boundary`. `direct_colorblock` has no realistic plan, so the boundary correctly
  injects into the colour-block generation instead.
- (c) **json** — confirmed. It consumes the realistic plan from `generate_real_plan`
  (`floor_plan_generate.py:221`), so the boundary flows through pass-1 into `real_png`
  and `structure_reader.read_structure` then reads the bounded realistic plan. The
  reader takes no boundary argument, which is correct.

**Correction-loop note (consistent with orthogonality).** The boundary is applied
only on round 1 (the round that calls a pass-1 producer). Rounds 2+ edit the round-1
image with violation feedback and do **not** re-pass the boundary:
real2color family edits `real_png` (`floor_plan_generate.py:210`), direct edits the
colour-block (`floor_plan_generate.py:187`). The footprint persists because the
boundary is already baked into the round-1 image and the feedback text instructs the
model to keep the same footprint. This matches "boundary only affects pass-1."

## 3. Boundary touches nothing downstream of pass-1

Every post-pass-1 stage takes the realistic/parsed image, the program, and/or the
plan — never a boundary. Signatures (none has a `boundary` parameter):

- `cv_parse(image, px_per_mm=None, min_room_px=400, wall_px=None)` — `image_parser.py:148`.
- `read_structure(real_png, client, program, prompt_logger=None)` — `structure_reader.py:92`.
- `parse_linework(image, program, ocr_reader=None)` — `linework_parser.py:234`.
- `fix_room_counts(plan, requested)` — `plan_fixes.py:36`.
- `place_doors(plan, room_graph)` — `door_placer.py:105`.
- `render_plan(...)` — `render_plan.py:21`.
- `extract_room_adjacency(real_png, client, program, prompt_logger=None)` —
  `room_adjacency_extractor.py:72`.

A repo-wide grep for `boundary` confirms the only other hits are unrelated geometry
("a region's outer **boundary**", "shared **boundary** segment") in `door_placer.py`,
`image_parser.py`, `infer_openings.py`, and `raster_geometry.py` — i.e. polygon edges,
not the input-boundary image. No downstream module imports or reads the boundary PNG.

The existing test encodes this contract: `test_boundary_entry_runs_pass1_then_shared_downstream`
asserts pass-1 used the boundary prompt and that the downstream artifacts (plan,
room graph, walls) are produced identically. `tests/test_boundary.py:32-43`.

## 4. Gaps

1. **No `config.json` boundary key.** `config.json` (current state read) exposes
   `generation_mode`, `structure_mode`, `max_correction_rounds`, `image_size`,
   `image_aspect`, `image_model` — no boundary. `load_config()` likewise returns only
   those six keys (`floor_plan_generate.py:52-70`), so any boundary key in config would
   be silently dropped.
2. **Config-driven entry points cannot enable a boundary.**
   `api.py /api/agent/generate` (`55-75`) and `graph.py run_pipeline` (`16-27`) call
   `generate_plan` from config but pass no boundary. Only the dedicated upload endpoint
   `/api/agent/generate-from-boundary` and the `evaluate.py` flag/field can turn it on.
   There is no way to say "every generation in this deployment is boundary-constrained
   by this default outline."
3. **`structure_mode` options are only described in prose**, not enumerated. The valid
   set lives in a validator (`floor_plan_generate.py:58`) and inside `_structure_mode_doc`;
   `config.json` does not list the options as data.

No per-mode generator gap was found — all four modes are correctly wired for boundary.

## 5. Recommendations (design only)

### (a) Enumerate `structure_mode` options in config.json

Keep the existing `value` + `_<key>_doc` convention and add a sibling list of valid
options. `load_config()` reads keys explicitly via `cfg.get(...)`, so any extra
`_`-prefixed sibling is ignored and non-breaking.

```jsonc
"structure_mode": "colorblock",
"_structure_mode_options": ["colorblock", "json", "linework", "direct_colorblock"],
"_structure_mode_doc": "how the program becomes a Plan. Selected value is set above; choose one of _structure_mode_options. colorblock = ... | json = ... | linework = ... | direct_colorblock = ... Override with --structure-mode."
```

- `structure_mode` stays the single selected/default value (so nothing downstream
  changes).
- `_structure_mode_options` is the authoritative enumeration the user asked for.
- Optional symmetry: `"_generation_mode_options": ["real2color"]` for consistency.
- Optional hardening (not required): have `load_config()` validate
  `structure_mode in cfg["_structure_mode_options"]` instead of the hardcoded tuple at
  `floor_plan_generate.py:58`, keeping the option list single-sourced in config.

### (b) Expose boundary as a config toggle, orthogonal to structure_mode

Boundary is a pass-1 **input**, not a structure choice, so it belongs as its own
top-level key — not nested under `structure_mode`. A pure boolean can't carry the
image, so mirror the existing `"image_model": ""` "empty = off" convention and make
the toggle a **path string** (empty = OFF / free footprint; non-empty = ON, the
boundary image to constrain pass-1):

```jsonc
"boundary": "",
"_boundary_doc": "Optional building-outline image (PNG) constraining pass-1 only. Empty = OFF (free footprint). Set to a path (relative to hfagent/) to turn ON: the realflow plan's outer walls follow that outline. Orthogonal to structure_mode — composes with all of them; injected into pass-1 and never used downstream. Precedence: CLI --boundary > per-program \"boundary\" > this config default. The /api/agent/generate-from-boundary upload overrides per request."
```

Consumption (where each piece should read it):

- `load_config()` (`floor_plan_generate.py:52-70`): add
  `"boundary": cfg.get("boundary", "")` to the returned dict so all callers see it.
- `evaluate.py` (`102-106`): extend the existing precedence chain to fall back to the
  config default, e.g. `bpath = args.boundary or program.get("boundary") or cfg["boundary"]`,
  then read the bytes as today.
- `api.py /api/agent/generate` (`55-75`) and `graph.py run_pipeline` (`16-27`): if
  `cfg["boundary"]` is non-empty, resolve the path, read the bytes, and pass
  `boundary=...` into `generate_plan`. Keep `/api/agent/generate-from-boundary` for
  per-request uploads (a runtime override, not a deployment default).

Because the generator already routes the boundary correctly for every mode, this is
the **only** change required to make boundary a real config-level toggle — no
generator or downstream edits. If a literal boolean is preferred for the UI, add
`"boundary_enabled": false` gating a separate `"boundary_path"`, but the single
path-string form above is simpler and matches the file's existing `""`-means-off style.

### (c) Modes lacking boundary support / minimal fix

None at the generator level — `colorblock`, `json`, `linework`, and
`direct_colorblock` all inject the boundary into pass-1 with the correct prompt
builder and the image prepended. The only "mode that doesn't support boundary" in
practice is **any generation triggered through the config-driven paths**
(`/api/agent/generate`, `run_pipeline`), because the boundary is not wired from
config. The minimal fix is recommendation (b): add the `boundary` key, return it from
`load_config()`, and have those two entry points + `evaluate.py`'s fallback consume it.
No change to `FloorPlanGenerator` or any downstream tool is needed.
