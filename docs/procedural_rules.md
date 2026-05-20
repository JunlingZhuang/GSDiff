# Procedural Floor-Plan Rules (MSD)

Rules for the rule-based / procedural generator in `procedural/`. Every constant
in this document is derived from `digress/data/msd_wall_v6/graphs.p` (5372
graphs, edge kinds: wall + door + passage + entrance) by running
`procedural/mine_priors.py`, which writes `procedural/priors.json`.

The generator (`procedural/generator.py`) is the consumer; the predicates
(`procedural/rules.py`) are the checkers; this document is the spec.

## 1. Why this exists

GSDiff / DiGress produce floor plans by learned diffusion. This module is the
deterministic alternative — same input contract (bubble diagram + boundary),
deterministic output, no GPU, no training. Used as:

1. App fallback when ML models are unavailable.
2. Sanity baseline for ML metrics.
3. Cheap data augmentation (deterministic seeds → controlled diversity).

## 2. Data the rules come from

`msd_wall_v6` is the latest wall-inference pass over MSD (see file timestamps
in `digress/data/msd_wall_*`). It distinguishes four edge kinds per room pair:

| kind | count | semantics |
|---|---:|---|
| `wall` | 160,629 | rooms share a wall, no door between them |
| `door` | 126,135 | door between rooms |
| `passage` | 21,813 | open passage (no door) |
| `entrance` | 20,222 | entrance door (from outside) |

The **access graph** = `door + passage + entrance` (168k edges) is what the
user supplies as the bubble diagram. The **adjacency graph** = access graph +
`wall` is what the generator's output must satisfy geometrically (every edge
in the output, regardless of kind, requires the two polygons to physically
touch).

## 3. Per-room priors (data-driven constants)

Median values from 5372 graphs. `n` is the number of room instances pooled
across the dataset.

| Room | n | area p10–p50–p90 (m²) | aspect p50 | rect frac | degree p50 |
|---|---:|---:|---:|---:|---:|
| Storeroom | 7,367 | 1.0 — **2.5** — 9.1 | 1.5 | **0.60** | 3 |
| Bathroom | 28,276 | 1.8 — **3.8** — 5.6 | 1.4 | 0.48 | 3 |
| Stairs | 9,561 | 1.7 — **4.8** — 18.8 | 1.3 | 0.52 | 4 |
| Balcony | 19,783 | 3.1 — **7.7** — 18.9 | 2.4 | 0.27 | 2 |
| Kitchen | 18,000 | 5.4 — **7.9** — 12.4 | 1.4 | 0.24 | 3 |
| Corridor | 25,738 | 2.9 — **8.2** — 16.9 | 1.8 | **0.15** | **6** |
| Dining | 540 | 6.7 — **10.4** — 18.0 | 1.3 | 0.19 | 4 |
| Bedroom | 42,584 | 10.6 — **14.4** — 19.0 | 1.4 | 0.46 | 4 |
| Livingroom | 13,492 | 17.9 — **27.6** — 39.8 | 1.4 | **0.08** | 5 |

Highlights that drive the rules:

- **Livingroom is the largest** — always (`area` ~2× Bedroom, ~7× Bathroom).
- **Corridor is the hub** — median degree 6, max 48 across the dataset.
- **Bathroom is small but bimodal** — p10=1.8 (WC) vs p90=5.6 (full bath).
- **Livingroom & Corridor are non-rectangular** — rect_frac 0.08 / 0.15. Any
  generator that forces rectangles will fail on these two types.
- **Bedroom & Bathroom & Storeroom are mostly rectangles** (rect_frac 0.46 /
  0.48 / 0.60) — generator may default to oriented bounding rectangles here.

## 4. Adjacency priors

Two distinct distributions; treat them differently in the generator.

### 4.1 Access pairs (door / passage / entrance) — user input

Top 10 most common room-pair connections when there is a door or opening:

| pair | count | reading |
|---|---:|---|
| Bedroom — Corridor | 38,384 | bedrooms enter via corridor |
| Bathroom — Corridor | 24,948 | bathrooms enter via corridor |
| Corridor — Livingroom | 14,389 | living enters via corridor |
| Balcony — Livingroom | 11,941 | balcony off living |
| Corridor — Stairs | 11,374 | stairs from corridor |
| Corridor — Kitchen | 11,161 | kitchen off corridor |
| Corridor — Corridor | 10,352 | multi-corridor networks |
| Balcony — Bedroom | 9,642 | balcony off bedroom |
| Kitchen — Livingroom | 9,034 | open-plan kitchen+living |
| Corridor — Storeroom | 5,306 | storage off corridor |

### 4.2 Wall pairs (touching, no door) — generator decides

Top 10 most common room-pair physical adjacencies without a door:

| pair | count | reading |
|---|---:|---|
| Bedroom — Bedroom | 23,030 | bedrooms cluster (sleeping wing) |
| Bathroom — Bedroom | 17,330 | en-suite by wall, not door |
| Bedroom — Livingroom | 10,224 | bedrooms back onto living |
| Bathroom — Kitchen | 9,979 | wet-room plumbing stack |
| Bedroom — Kitchen | 8,609 | bedrooms back onto kitchen |
| Bedroom — Corridor | 6,662 | corridor wall is bedroom's edge |
| Bathroom — Corridor | 6,438 | corridor wall is bathroom's edge |
| Bathroom — Bathroom | 6,387 | back-to-back baths (plumbing) |
| Balcony — Bedroom | 6,334 | balcony wall, no door |
| Bathroom — Livingroom | 6,250 | bathroom backs onto living |

Three patterns that become rules:

- **Wet rooms cluster** (Bathroom — Bathroom, Bathroom — Kitchen): plumbing
  efficiency. Generator should bias bathrooms toward shared walls with other
  wet rooms.
- **Sleeping wing** (Bedroom — Bedroom 23k): bedrooms cluster, separate from
  living/kitchen.
- **Bedroom ↔ Bathroom is wall-first, door-second** (17k wall vs 1.3k door):
  en-suite access in MSD usually goes through the corridor, not via a direct
  bedroom door.

## 5. The 9 rules

Each rule is a predicate in `procedural/rules.py`. The generator both _uses_
the rules (as scoring functions) and _validates against_ the rules (as
acceptance tests).

### R1 — Corridor is the backbone

Identify all Corridor nodes in `graph_in`. Connect them into a Steiner tree
that touches every non-Corridor node by following access edges. This tree is
the generator's first geometric scaffold.

> **Why:** Corridor median degree is 6 (max 48). Empirically `Corridor` is the
> only type whose degree distribution is centered above 3. Confirmed across
> 5372 graphs without counter-example for residential apartments.

### R2 — Leaves hang off the backbone

Every non-Corridor node attaches to exactly one Corridor or Livingroom node in
the backbone tree. Exceptions:

- Balcony attaches to Livingroom **or** Bedroom (frequencies are comparable).
- Kitchen attaches to either Corridor or Livingroom (open-plan layouts).

> **Why:** Bedroom — Corridor access (38k) is 13× more common than Bedroom —
> Livingroom access (2,960). Bathroom never connects to Livingroom directly via
> a door. Bedrooms enter via corridors; this is a Swiss-residential pattern,
> not a universal one.

### R3 — Area within band

Each generated polygon's area must fall in `[p10, p90]` for its room type
(table §3). Generator soft-targets `p50`.

> **Why:** Direct constants from MSD. The p10/p90 band absorbs the natural
> spread (e.g. WC vs full bath under "Bathroom"); going outside the band is
> distributionally rare.

### R4 — Livingroom is the largest in its apartment

Within one apartment (one connected access-graph component), the Livingroom
polygon's area must be ≥ every other room's area.

> **Why:** Holds across the entire dataset — Livingroom p10 (17.9) already
> exceeds every other room's p50 except Bedroom (14.4) and Corridor (8.2).

### R5 — Bedrooms have an exterior wall

Each Bedroom polygon must share at least one edge with the apartment boundary
(window requirement).

> **Why:** Building-code requirement in Switzerland (and most jurisdictions).
> While not measured directly in the priors, every spot-checked MSD sample
> satisfies this.

### R6 — Bathrooms may be interior

Bathroom polygons are permitted (not required) to have no exterior wall.

> **Why:** Bathrooms are ventilated mechanically. p50 area 3.8 m² fits interior
> placement against a plumbing wall.

### R7 — Balconies on the exterior + attached to Living or Bedroom

Balcony polygon must share an edge with the apartment boundary **and** must be
access-connected to a Livingroom (preferred, 11,941 cases) or Bedroom (9,642
cases).

> **Why:** A balcony with no exterior boundary is geometrically nonsensical.
> Frequencies above show the two acceptable parents.

### R8 — Shape complexity by type

- Livingroom and Corridor polygons: arbitrary non-self-intersecting polygons,
  may be L-shaped (rect_frac 0.08 / 0.15).
- Bedroom, Bathroom, Storeroom: emitted as axis-aligned or rotated rectangles
  unless boundary geometry forces a notch (rect_frac 0.46–0.60).
- Kitchen, Balcony, Stairs, Dining: rectangle is acceptable but allow L-shape
  for boundary-fit cases (rect_frac 0.19–0.27).

> **Why:** Empirical shape complexity per type, §3.

### R9 — Wet-room clustering

Bias the optimizer (soft cost, not hard constraint) toward placing Bathrooms
adjacent to other Bathrooms or to Kitchens by shared wall:

- Bathroom — Bathroom: 6,387 wall-pair occurrences
- Bathroom — Kitchen: 9,979 wall-pair occurrences

Cost reduction when these adjacencies are realised; no penalty when absent.

> **Why:** Plumbing-stack efficiency, recorded as a strong distributional
> signal in the wall-pair counts.

## 6. Generator pipeline

`procedural/generator.py::generate(graph_in, boundary) -> {polygons, walls}`:

```
1. parse_inputs(graph_in, boundary)
2. backbone = identify_backbone(graph_in)              # R1, R2
3. embedded = embed_in_boundary(backbone, boundary)    # places hub + arms
4. polygons = grow_rooms(embedded, priors)             # R3, R8
5. for room in polygons:
       enforce_exterior_if_required(room)              # R5, R7
       enforce_area_in_band(room)                      # R3
6. score = soft_score(polygons, priors)                # R4, R9
7. polygons = local_search(polygons, score)            # simulated annealing
8. walls = derive_walls(polygons)
9. return { "polygons": polygons, "walls": walls }
```

Each stage uses `rules.py` predicates as feasibility checks and as scoring
terms. The generator never trains; it just searches over geometric moves
under fixed rules.

## 7. What is *not* in this spec

- **Furniture layout** (chairs, beds, sinks). MSD doesn't have it; generator
  ends at room polygons + walls.
- **Door positions** along walls. Could be derived later from `door` /
  `passage` edges in `graph_in`, but not in the MVP.
- **Multi-floor stacking**. Single floor only.
- **Structural elements** (columns, shafts). The generator places them only
  if they are present in the input `boundary`; it does not invent them.
- **Validation against MSD ground truth**. This is procedural generation, not
  ML. Use the predicates in `rules.py` to validate plausibility instead of
  diffing against `graph_out`.

## 8. Refreshing the priors

Whenever the wall-inference algorithm changes (`msd_wall_v6` → `msd_wall_v7`,
say), update the `GRAPHS_PATH` constant in `procedural/mine_priors.py`, re-run
it, and re-derive any constants in this document from the new
`procedural/priors.json`. The rule list (§5) is intentionally written so the
rules survive minor numerical drift — only the constants need updating.
