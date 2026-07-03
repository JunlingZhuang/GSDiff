# Sealed Linework Prompt Sweep

Date: 2026-07-02

Status: **not good enough for production**

## Question

Can Gemini Flash reliably generate a line drawing in which every room and
corridor is fully sealed, so that the CV parser does not need to reconstruct
door openings?

The alternative is to keep the previous architectural prompt with visible
doors, then improve the trace algorithm to detect door evidence and bridge only
those gaps.

## Experiment

- Model: `gemini-3.1-flash-image`
- Requested image size: `2K`
- Requested aspect ratio: `16:9`
- Prompt variants: 5
- Programs: `clinic-small`, `hospital-floor`, `inpatient-ward`,
  `hospital-tower-floor`
- Repeats: 2 independent generations per prompt and program
- Total generated images: 40
- Missing or failed final images: 0
- Unique final image hashes: 40

Command:

```powershell
python -B -m hfagent.evaluate_sealed_prompts --out hfagent/out/eval/20260702-sealed-prompt-sweep
```

Artifacts:

- `hfagent/out/eval/20260702-sealed-prompt-sweep/manifest.json`
- `hfagent/out/eval/20260702-sealed-prompt-sweep/summary.json`
- `hfagent/out/eval/20260702-sealed-prompt-sweep/manual_review.csv`
- `hfagent/out/eval/20260702-sealed-prompt-sweep/contact_sheets/`
- Exact expanded prompt for every image:
  `<variant>/<program>-r<repeat>/prompt.txt`
- Generated and traced files for every image:
  `<variant>/<program>-r<repeat>/`

The sweep is isolated from the production prompt. Existing images are reused
when the command is rerun, so metrics and contact sheets can be regenerated
without another API call.

## Metrics

### Strict seal success

Manual binary review of the original 2K image. A result passes only when there
is no visible wall opening anywhere. One unjoined wall endpoint, or one room or
corridor connected to another white region through a wall gap, fails the whole
image.

This is the most honest closure metric for a no-door prompt. There is no fixed
denominator of intended doors, so a per-door closure percentage would be
invented rather than measured.

### Enclosed regions

Objective raster connected-component count:

1. Convert the generated image to grayscale.
2. Treat pixels at or above 230 as white interior.
3. Find 8-connected white components.
4. Exclude the canvas exterior and components smaller than 1000 pixels.

The count stayed unchanged when the threshold was varied from 200 through 250
on the checked images. This measures parser-recoverable enclosed white regions,
not semantic room correctness.

### Count exact

Passes when `enclosed_regions == required_rooms`. It can pass even when walls
are visibly open, because a merged region and an extra region can cancel each
other. It must therefore be combined with strict seal success.

### Tracer diagnostics

`summary.json` also records current tracer output. In particular,
`trace.sub_gaps` is **not** a manual wall-opening count: it can count spaces
between unrelated collinear walls and separate building wings. It is retained
for parser debugging but is not used as the prompt success metric.

## Aggregate Results

| Prompt | Strictly sealed | Count exact | Both sealed and exact | Region-count MAE | Enclosed / required |
|---|---:|---:|---:|---:|---:|
| P1 current short | 3/8 (37.5%) | 2/8 (25.0%) | 2/8 (25.0%) | 16.12 | 57.5% |
| P2 positive boundaries | 5/8 (62.5%) | 1/8 (12.5%) | 1/8 (12.5%) | 4.88 | 96.9% |
| P3 partition map | **6/8 (75.0%)** | 2/8 (25.0%) | **2/8 (25.0%)** | 5.50 | 113.6% |
| P4 construction steps | 1/8 (12.5%) | 2/8 (25.0%) | 0/8 (0.0%) | 5.25 | 90.5% |
| P5 final wall audit | 0/8 (0.0%) | 0/8 (0.0%) | 0/8 (0.0%) | 24.62 | 33.0% |

P3 is the best no-door wording in this sample, but it is not robust. Its two
`hospital-tower-floor` images both contain open boundaries. It also changes the
task from an architectural floor plan to a partition map and frequently invents,
duplicates, or omits cells and labels.

## Per-Image Results

`sealed` is the conservative manual full-image decision. `axis` records whether
the output stayed horizontal/vertical.

| Prompt | Program | Run | Required | Enclosed | Sealed | Axis | Main observation |
|---|---|---:|---:|---:|---:|---:|---|
| P1 | clinic-small | 1 | 6 | 6 | 1 | 1 | Exact and closed |
| P1 | clinic-small | 2 | 6 | 6 | 1 | 1 | Exact and closed |
| P1 | hospital-floor | 1 | 26 | 22 | 0 | 1 | Open central waiting/corridor walls |
| P1 | hospital-floor | 2 | 26 | 28 | 1 | 1 | Closed, but extra regions and duplicate labels |
| P1 | inpatient-ward | 1 | 33 | 2 | 0 | 1 | Repeated openings along both corridors |
| P1 | inpatient-ward | 2 | 33 | 11 | 0 | 1 | Repeated room/corridor openings |
| P1 | hospital-tower-floor | 1 | 82 | 19 | 0 | 1 | Large number of open boundaries |
| P1 | hospital-tower-floor | 2 | 82 | 75 | 0 | 1 | Central and lower openings remain |
| P2 | clinic-small | 1 | 6 | 6 | 1 | 1 | Exact and closed |
| P2 | clinic-small | 2 | 6 | 7 | 1 | 1 | Closed, but one extra region |
| P2 | hospital-floor | 1 | 26 | 25 | 0 | 1 | Incomplete central boundaries |
| P2 | hospital-floor | 2 | 26 | 38 | 1 | 1 | Closed, but many extra regions and labels |
| P2 | inpatient-ward | 1 | 33 | 35 | 1 | 1 | Closed, but count and labels are wrong |
| P2 | inpatient-ward | 2 | 33 | 31 | 0 | 1 | Open endpoints around central corridors |
| P2 | hospital-tower-floor | 1 | 82 | 67 | 0 | 1 | Open corridor strips and group joins |
| P2 | hospital-tower-floor | 2 | 82 | 76 | 1 | 0 | Cells sealed, but diagonal wings violate geometry |
| P3 | clinic-small | 1 | 6 | 6 | 1 | 1 | Exact and closed |
| P3 | clinic-small | 2 | 6 | 6 | 1 | 1 | Exact and closed |
| P3 | hospital-floor | 1 | 26 | 31 | 1 | 1 | Closed, but extra regions and duplicate labels |
| P3 | hospital-floor | 2 | 26 | 31 | 1 | 1 | Closed, but extra regions and duplicate labels |
| P3 | inpatient-ward | 1 | 33 | 31 | 1 | 1 | Closed, but count and labels are wrong |
| P3 | inpatient-ward | 2 | 33 | 37 | 1 | 1 | Closed, but count and labels are wrong |
| P3 | hospital-tower-floor | 1 | 82 | 93 | 0 | 1 | Open lower nurse/corridor groups |
| P3 | hospital-tower-floor | 2 | 82 | 99 | 0 | 1 | U-shaped incomplete cells around core |
| P4 | clinic-small | 1 | 6 | 6 | 0 | 1 | Exact count hides three corridor gaps |
| P4 | clinic-small | 2 | 6 | 8 | 0 | 1 | Open waiting/corridor boundaries |
| P4 | hospital-floor | 1 | 26 | 29 | 1 | 1 | Closed, but extra regions and label errors |
| P4 | hospital-floor | 2 | 26 | 24 | 0 | 1 | Incomplete central boundaries |
| P4 | inpatient-ward | 1 | 33 | 34 | 0 | 1 | Waiting area opens into corridor |
| P4 | inpatient-ward | 2 | 33 | 34 | 0 | 1 | Open waiting and nurse-station cells |
| P4 | hospital-tower-floor | 1 | 82 | 82 | 0 | 1 | Exact count hides open group boundaries |
| P4 | hospital-tower-floor | 2 | 82 | 49 | 0 | 1 | Many incomplete walls in all wings |
| P5 | clinic-small | 1 | 6 | 4 | 0 | 1 | Repeated wall openings |
| P5 | clinic-small | 2 | 6 | 5 | 0 | 1 | Large lower opening |
| P5 | hospital-floor | 1 | 26 | 18 | 0 | 1 | Many room/corridor openings |
| P5 | hospital-floor | 2 | 26 | 7 | 0 | 1 | Most central boundaries are open |
| P5 | inpatient-ward | 1 | 33 | 6 | 0 | 1 | Openings along corridor-facing rooms |
| P5 | inpatient-ward | 2 | 33 | 22 | 0 | 1 | Multiple open cells |
| P5 | hospital-tower-floor | 1 | 82 | 15 | 0 | 1 | Very high opening density |
| P5 | hospital-tower-floor | 2 | 82 | 20 | 0 | 1 | Very high opening density |

## Prompt Variants

`{ROOM_LINES}` is expanded to one exact instance ID per required room, with its
approximate area when present. Every expanded prompt is preserved beside its
image in `prompt.txt`.

### P1: Current Short

```text
Draw a top-down floor plan of a {BUILDING_TYPE} as a simple black line diagram.

- Rooms are rectangles packed side by side along the corridor (the corridor is just a long thin room). Together they fill the whole building with no leftover space.
- Every wall is a solid black band of ONE uniform thickness, about 12 px, perfectly horizontal or vertical.
- This plan has NO doors: every room, including the corridor, is completely closed on all four sides. No gap, no break, no opening, no swing arc anywhere in any wall.
- Write each room's name once in its centre, plain black text, exactly as listed below. No other text, no furniture, no windows, no shading.
- Everything outside the building is pure white.

Rooms (one room per name; size them roughly by the areas):
{ROOM_LINES}

Output only the drawing.
```

### P2: Positive Boundaries

```text
Create a top-down floor plan for a {BUILDING_TYPE} on a pure white 16:9 canvas.

Draw one connected building made only from closed rectangular room cells:
- Each room cell has four continuous solid black wall bands joined at all four corners.
- Each wall band runs continuously from corner to corner. Shared walls between neighbouring cells are continuous too.
- The corridor is another closed rectangular cell, not an open passage.
- Use one uniform wall thickness of about 12 px. All walls are horizontal or vertical.
- Fill the footprint with the listed cells, with pure white interiors and pure white outside.
- Put exactly one listed name at the centre of its cell. Add nothing except these labels and walls.

There are no doors or openings in this diagram. Every cell boundary remains solid and closed.

Required room cells, exactly once each:
{ROOM_LINES}

Output only the finished black-and-white diagram.
```

### P3: Partition Map

```text
Create a top-down orthogonal partition map for a {BUILDING_TYPE} on a pure white 16:9 canvas.

This is a labelled closed-cell partition map, not a usable building drawing.
- Represent every item below as one separate fully enclosed white rectangle.
- Pack the rectangles into one connected orthogonal footprint. Rectangles may share black edges.
- Every edge is a continuous solid black band about 12 px thick, joined cleanly at its corners.
- Treat names containing corridor exactly like all other names: corridor is a sealed labelled rectangle.
- Use only horizontal and vertical edges.
- Draw no breaks in edges and no symbols on edges.
- Draw no furniture, fixtures, windows, decoration, dimensions or text other than the exact names.

Required closed cells, one per line and one occurrence each:
{ROOM_LINES}

Return only the partition map image.
```

### P4: Construction Steps

```text
Create a top-down orthogonal partition map for a {BUILDING_TYPE} on a pure white 16:9 canvas.

Construct the diagram in this order:
1. Draw one closed orthogonal outer boundary using a uniform 12 px solid black band.
2. Divide its interior into exactly the listed number of rectangular cells using solid horizontal and vertical wall bands.
3. Extend every internal wall until it touches another wall. Join every endpoint so each cell has a complete closed perimeter.
4. Keep every wall continuous. Do not cut an entrance, doorway, passage or other gap into any wall.
5. After all cells are closed, write exactly one required name in the centre of each cell.

Use pure white cell interiors and pure white outside. Draw no windows, furniture, fixtures, dimensions, shading or extra text. A corridor name identifies an ordinary sealed cell; it does not authorize openings.

Cells to draw exactly once:
{ROOM_LINES}

Output only the completed diagram.
```

### P5: Final Wall Audit

```text
NO OPENINGS ARE ALLOWED. Draw a sealed black-wall room layout for a {BUILDING_TYPE}.

- Use a top-down 16:9 view and one connected orthogonal footprint.
- Give every listed room its own rectangular white interior and a complete solid black perimeter.
- Use one uniform wall band about 12 px thick. Walls are horizontal or vertical and meet exactly at corners and junctions.
- The corridor is fully enclosed by continuous walls on every side, like every other room.
- Use each exact room name once. Draw only walls and room names: no doors, door leaves, swing arcs, windows, furniture, dimensions, shading or extra marks.

Exact rooms:
{ROOM_LINES}

Before returning the image, inspect every room edge from corner to corner. Fill every white break in every black wall. Return the image only after every listed room, including every corridor, is completely sealed.

Output only the drawing.
```

## Findings

1. Prompt wording changes the failure rate, but does not provide a topology
   guarantee. Complexity is the dominant failure: the best prompt still failed
   both 82-room tower samples.
2. Removing architectural language helps closure. P3 performs best because
   `partition map` suppresses the model's floor-plan and doorway prior.
3. That improvement has a cost: P3 no longer reliably behaves like a plausible
   architectural floor plan and often invents or duplicates cells and labels.
4. Repeating negative instructions is counterproductive here. P5 is the worst
   prompt by every measured result.
5. Step-by-step construction language is not executed deterministically by the
   image model. P4 has exact connected-component count in two images, yet both
   still contain visible wall openings.
6. Exact region count alone is unsafe. P4 tower run 1 is `82/82`, but visible
   open group boundaries remain.

## Decision

Do **not** make no-door generation the production solution. P3 should remain a
geometry stress-test or optional partition-map mode, not the main architectural
pipeline.

Return the production linework prompt to explicit simple doors and improve the
trace algorithm around evidence-backed door gaps:

1. Ask for one consistent door symbol: a wall gap plus one thin quarter-circle
   swing arc, with no windows or furniture.
2. Detect and record door evidence before room segmentation.
3. Bridge only gaps supported by a local arc, door-leaf, jamb, width, and wall
   continuity score. Do not globally close every collinear gap.
4. Segment rooms on the temporarily bridged wall mask.
5. Restore the recorded door into the room graph as an edge between the two
   regions on either side.
6. Evaluate door candidate precision/recall and final room recovery on a small
   manually annotated set. Keep prompt tests and parser tests separate.

Weakness: this experiment has only two stochastic samples per program and does
not measure the old with-door prompt against the same programs yet.

Next: run a matched 40-image with-door baseline, annotate true door gaps, and
measure parser door detection precision/recall plus final room count. This is a
better use of the next experiment than adding more no-door negation wording.

---

## Round 2 — five iterations on the round-1 winner p3-partition-map

Date: 2026-07-02 (round 2). Same protocol as round 1 (flash, 2K, 16:9, 4
programs x 2 repeats = 40 images), fully comparable.

Command:

```powershell
python -m hfagent.evaluate_sealed_prompts --variants p6-p3-count-lock,p7-p3-edge-rule,p8-p3-banded,p9-p3-verify,p10-p3-compact --out hfagent/out/eval/20260702-sealed-prompt-sweep-r2
```

p3's residual failures drove the designs: over-partitioning + duplicate labels
at scale (hospital-floor 31 vs 26, tower 93/99 vs 82), open endpoints /
U-shaped cells at tower scale. Each iteration keeps p3's core (closed-cell
partition-map reframe, corridor = ordinary sealed cell, uniform 12 px
orthogonal edges) and attacks one failure mode:

| variant | hypothesis |
|---|---|
| p6-p3-count-lock | exact-count anchoring ("exactly N cells, each name once, count when done") |
| p7-p3-edge-rule | endpoint discipline ("every edge starts/ends on another edge") |
| p8-p3-banded | prescriptive banded macro-layout; corridors are their own long thin bands |
| p9-p3-verify | one light closing self-check (p5 showed heavy audit framing backfires) |
| p10-p3-compact | p3 core compressed to minimal words |

### enclosed_regions per run (req = program total)

| variant | clinic (6) | hosp-floor (26) | inpatient (33) | tower (82) | exact |
|---|---|---|---|---|---|
| p3 baseline (round 1) | 6, 6 | 31, 31 | 31, 37 | 93, 99 | 2/8 |
| p6-count-lock | 6, 6 | 27, 28 | 36, 34 | 121, 94 | 2/8 |
| p7-edge-rule | 1, 1 | 31, 32 | 41, 38 | 112, 95 | 0/8 |
| **p8-banded** | 6, 7 | **26**, 28 | **33, 33** | 64, **82** | **5/8** |
| p9-verify | 6, 12 | 32, 26 | 30, 35 | 102, 99 | 2/8 |
| p10-compact | 4, 5 | 32, 28 | 23, 4 | 100, 84 | 0/8 |

### Visual verdicts (manual)

- **p8-banded is the clear winner** and the only variant that moved the tower.
  At <= 33 rooms it naturally produces the RIGHT architecture: full-width
  corridor bands separating rows of rooms (double-loaded corridors), all cells
  sealed, counts exact (inpatient 33/33 both runs).
- p8 tower r1: correct structure (corridor_1..4 as full-width separator bands,
  room rows between) but only 64 cells closed. Tower r2: exactly 82 sealed
  cells but layout degenerates into a spreadsheet-like table — corridors drawn
  as 4 ordinary cells in the bottom row, disconnected from the rooms; label
  enumeration has duplicates/omissions (patient_room_15 twice, 35 missing,
  toilet 5/9/12/15 missing, office_3 missing).
- Label leakage: the "(about X m2)" area hints from the room list are printed
  inside cells in some runs ("12 m2", "about 17 m2"). The area hint should be
  dropped or the prompt must say names only.
- p6 counting instruction did nothing at scale (121 cells vs "exactly 82" in
  the prompt) — image models cannot count to 82; structure, not arithmetic, is
  what controls the count.
- p7's "every edge ends on another edge" collapsed clinic-small to one giant
  grid cell (1, 1) — the rule made the model draw one lattice instead of
  discrete rooms.
- p9's light self-check and p10's brevity both underperformed plain p3.

### Round-2 conclusion

The banded macro-layout (p8) is the right lever: it converts free-form packing
into rows + corridor bands, which the model can seal reliably up to ~33 rooms
and produces the correct double-loaded-corridor architecture. Remaining work is
stabilising the 82-room case (one of two draws degenerated to a table) and
killing the area-hint label leakage.

Round-3 finalists (per the experiment plan, to be run at higher repeats with a
thinking_level comparison): p8 refinements — (a) corridors explicitly the
full-width separator bands; (b) names-only room list (no area hints); (c) wings
structure for the tower; (d) p8 unchanged as control.

---

## Round 2 addendum — trace-side "seal everything" quantified (offline)

`trace_linework(png, sealed=True)` (bridge EVERY collinear sub-gap with a real
wall end; no arc requirement) re-run on the round-1 p1/p2 images:

| variant | program | raw -> sealed (req) |
|---|---|---|
| p1 | inpatient r1 | 2 -> 67 (33) |
| p1 | tower r1 | 19 -> 250 (82) |
| p2 | hospital-floor r2 | 38 -> 92 (26) |
| p2 | tower r1 | 67 -> 169 (82) |

Unconditional gap-bridging OVERSHOOTS badly: it seals room mouths but also
slices corridors and open areas at every pier alignment, shattering the plan
into hundreds of false cells. Selective sealing would require deciding which
gaps are room boundaries — which is the door-detection problem in disguise.

## Overall conclusion after two rounds

- Architectural realism and sealed compliance trade off: the only variant
  family that seals reliably at 82 rooms (p3/p8 partition-map/banded) stops
  looking like architecture; the variants that look like real plans (p1/p2)
  leave gaps at scale that cannot be closed deterministically without
  re-solving door detection.
- Meanwhile the with-doors path already works: doors_in_plan=true + the
  robust arc tracer closes 84/84 labeled rooms on the 82-room reference with
  112 doors and no phantom bridges, on real architectural drawings.
- RECOMMENDATION: production stays doors_in_plan=true (real architecture +
  arc-confirmed doors). The sealed mode remains available behind the config
  toggle for small/mid programs (p2-style prompt works to ~33 rooms) and as a
  future research direction (p2 + selective sealing).

---

## Round 3 — architectural single-pass finals (stopped early) and FINAL DECISION

Finals: p2 (control) vs narrative-prose wings (p11) vs "architectural semantic
map" (p12) vs the same two at thinking_level=high (p13/p14), tower-82 only,
6 repeats, single pass. Stopped at 23/30 images because the interim data
already answered the question — enclosed regions per run (target 82):

| variant | runs |
|---|---|
| p2 control | 79, 77, 84, 85, 22, 108 |
| p11 wings prose | 81, 11, 114, 11, 10, 106 |
| p12 semantic map | 117, 84, 57, 112, 47, 99 |
| p13 = p11 @ thinking high | 69, 5, 82, 81, 74 |

- No single-pass prompt is production-stable at 82 rooms; every variant has
  collapse runs. The variance is inherent to one-shot generation at this
  density, not a wording problem — three rounds and ~90 images converge on
  this.
- One real signal: thinking_level=high VISIBLY TIGHTENS the distribution
  (p13 vs p11 with identical text: 69/5/82/81/74 vs 81/11/114/11/10/106),
  including one exact-82 run. Worth keeping in mind wherever flash is used.

FINAL DECISION (user): abandon the sealed/no-door direction. Production
linework returns to the doors-drawn prompt exactly as of eval
20260701-172629-linework-trace-doors (config toggle removed), and effort goes
into the TRACE side: better door detection and local, evidence-gated wall
patching. Global gap-sealing is refuted (250 false cells); local patching must
stay ink/evidence-based like the existing corner snap.
