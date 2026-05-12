# MSD Wall-Edge Inference

This document describes how `digress/scripts/prepare_msd_graphs.py` produces the
extra `wall` edges that distinguish `msd_wall` from the vanilla `msd` dataset.

> **Status**: V3 is the algorithm currently committed in
> `prepare_msd_graphs.py`. V1 and V2 reconstructed below from the natural
> evolution of the heuristic — please cross-check against your local notes /
> editor history and fill in the gaps marked `TODO`.

---

## 1. Why we need wall inference at all

The MSD `graph_out/` pickles only carry the **access graph**: rooms with
`passage` / `door` / `entrance` edges. Two rooms that share a wall but no
opening (e.g. a bedroom next to a bathroom, with no door between them) are
recorded as **no edge** there.

For floorplan reconstruction or for a model that has to recover spatial
adjacency, this is information loss. The fix is to add a fifth edge class
`wall`, which marks "rooms physically adjacent but not connected".

The raw MSD CSV (`raw/.../v2.csv`) does carry `WALL` polygons. The job of
`--add-wall-edges` is to read those and infer the missing `wall` edges from
geometry.

```
edges in msd:        none / passage / door / entrance              (4 classes)
edges in msd_wall:   none / wall / passage / door / entrance       (5 classes)
```

`wall` is added **only as a fallback**: if a pair of rooms is already linked
by `passage` / `door` / `entrance`, it stays that way and the wall is not
recorded as a separate edge.

---

## 2. Algorithm versions

### V1 — naive co-touch (earliest, no segment logic)

> Reconstructed. **TODO: confirm exact form from notes/local editor history.**

Idea (the most direct reading of "rooms that share a wall"):

```
for each WALL polygon w in the floor:
    rooms_touching_w = [r for r in rooms if r.boundary touches w (within eps)]
    for every pair (a, b) in rooms_touching_w:
        add wall edge (a, b)
```

Failure modes that pushed us off V1:

- **Long shared walls double-count.** A single MSD `WALL` polygon often runs
  across an entire interior partition (it is drawn as one rigid body for the
  building, not per room-pair). Five rooms touching the same wall produce
  C(5,2)=10 wall edges, including pairs at opposite ends of the wall that
  are not actually adjacent.
- **Same-side rooms get linked.** Two rooms on the same side of a long wall
  (e.g. two bedrooms, both south of a corridor wall) are flagged as
  wall-neighbors despite never sharing the wall as a partition between them.

### V2 — add opposite-side check

> Reconstructed. **TODO: confirm if this was actually a separate iteration or
> if V1→V3 happened in one step.**

Idea: walls are partitions, so two rooms can share a wall **only if the wall
is between them**. Implement that as: their centroids must lie on opposite
sides of the wall's principal axis.

```
for each wall w:
    tangent, normal = principal_axes(w)              # PCA of wall vertices
    for each contact c on w:
        c.side = sign((c.room_centroid - c.point) · normal)
    for every (a, b) on this wall:
        if sign(a.side) * sign(b.side) >= 0:  continue   # same side, drop
        add wall edge (a.room, b.room)
```

What V2 fixed:

- Eliminates the most obvious "two rooms on the same side of a corridor wall"
  false positives.

What V2 still got wrong:

- **Diagonal rooms on a long wall.** Room A at the far north end of a long
  central wall, Room B at the far south end. Centroids are on opposite sides,
  so the side check passes — but they are diagonally placed, not actually
  adjacent.
- Result: wall edges get assigned to pairs of rooms that are nowhere near
  each other locally.

### V3 — current: opposite-side **and** segment-locality

Implemented in `prepare_msd_graphs.py` around
`extract_structural_graph_from_floor` (line ~279). Three thresholds, all
applied per-wall:

```python
# 1) Detect a meaningful contact between each room and this wall
contact = room.boundary.intersection(wall.buffer(wall_contact_eps))
if contact.length < wall_min_contact_length:
    drop                                          # contact too short to count

# 2) Tag which side of the wall the room sits on
side = sign((room_centroid - contact_centroid) · wall_normal)

# 3) Pair up only rooms that are local AND on opposite sides
for (a, b) in contacts on this wall:
    if a.side * b.side >= 0:                      continue   # same side
    if distance(a.contact_segment, b.contact_segment) > wall_segment_gap:
        continue                                  # not local on this wall
    add wall edge (a.room, b.room)
```

Then the edge-assignment loop applies a **priority**:

```
passage   if room-room distance < 0.04
door      else if any DOOR polygon touches both rooms (eps 0.05)
entrance  else if any ENTRANCE_DOOR polygon touches both
wall      else if (a, b) ∈ wall_pairs
none      otherwise
```

So `wall` is strictly a fallback — rooms that have a real opening between
them get the access-graph label, not `wall`.

What V3 fixed compared to V2:

- A wall with two rooms 3 metres apart on opposite ends no longer auto-pairs.
  The `wall_segment_gap` filter forces the two contact segments themselves to
  be locally close on the same wall.

What V3 still gets wrong:

- See section **5. Known failure modes** below — the segment-gap default
  (0.45) is generous, and there are floorplan layouts that defeat the
  segment check.

---

## 3. Default thresholds

| Threshold | Default | What it controls | Tightening makes it… | Loosening makes it… |
|---|---|---|---|---|
| `wall_contact_eps` | 0.01 | Room boundary is treated as touching a wall if it lies within this buffer of the wall polygon. Compensates for floating-point gaps in the CSV geometry. | Miss real contacts where the CSV has small gaps. | Tag near-misses as contacts (almost no harm at 0.01). |
| `wall_min_contact_length` | 0.02 | Contact length must be at least this many length units to count. Filters single-point or hair-line contacts. | Drop short but valid contacts (e.g. small bathrooms). | Tag corner kisses as wall contacts. |
| `wall_segment_gap` | 0.45 | Two rooms' contact segments on the same wall must be within this distance of each other to count as a wall-pair. | False negatives: misses real adjacent pairs whose contact segments are slightly separated by a doorway. | False positives: pairs distant rooms that happen to touch the same long wall. |

> Units: MSD `v2.csv` geometry is in metres (per MSD repo conventions). 0.45 m
> is roughly the width of one doorway, which is intentional — the original
> reasoning was "two contact segments separated by one doorway should still
> count as the same shared wall". In practice this is too generous; see V3
> failure modes.

The chosen defaults (`0.01 / 0.02 / 0.45`) are recorded both in
`prepare_msd_graphs.py` and on every output graph as
`graph.graph["wall_contact_eps"]`, `graph.graph["wall_min_contact_length"]`,
`graph.graph["wall_segment_gap"]` so the parameters used at preparation time
travel with the data.

---

## 4. Edge-assignment priority and why `wall` is last

```
passage > door > entrance > wall > none
```

Rationale: the access graph (door/passage/entrance) is **ground truth** in
MSD; we trust it. `wall` is **inferred from geometry** and is therefore lower
confidence. If the access graph already says two rooms are connected by a
door, we never overwrite it with `wall`, even if the geometry says they also
share a wall (which is in fact always true: a door sits in a wall).

`passage` is checked first because it uses the simplest test (room
polygons within 0.04 of each other) and reliably catches "rooms separated
only by an open arch" cases.

---

## 5. Known failure modes (V3)

### 5.1 Diagonal rooms via a single long wall

Two rooms at opposite corners of the apartment, both touching segments of
one long L-shaped or Z-shaped `WALL` polygon. The side check passes
(centroids really are on opposite sides), and if the contact segments are
within 0.45 of each other along the wall's length, V3 still pairs them.

Concretely: an apartment with a vertical interior wall that runs the full
depth — one room top-left, one room bottom-right — both touch the wall on
opposite sides, and the segment gap might or might not catch them
depending on how much corridor sits between.

### 5.2 Rooms separated by a thin corridor

Wall ↔ corridor ↔ wall. The two rooms each have a contact segment on
"their" wall, but those are different MSD `WALL` polygons in most cases, so
V3 does **not** false-positive this. The hazard is when MSD draws the two
walls as a single non-convex `WALL` polygon — then both rooms touch
segments of the same polygon, on opposite sides, and may be locally close
along it.

### 5.3 Corner kiss

Two rooms meeting at a single corner point. `wall_min_contact_length=0.02`
filters most of these out, but if floating-point precision puts the contact
length slightly above 0.02 (or if rooms share a tiny edge by accident in
the CSV), a wall edge is added.

### 5.4 Real shared walls missed

The dual problem: legitimate adjacent rooms are dropped if the MSD `WALL`
polygon between them is not actually drawn (some apartments rely on
implicit partitioning that the CSV does not record), or if the contact
length is just below `wall_min_contact_length`.

---

## 6. How to sanity-check the inferred labels

`prepare_msd_graphs.py --visualize 10` writes per-sample 2x3 panels to
`digress/data/msd_wall/vis/sample_*.png`:

- top-left: room polygons coloured by type
- top-middle: access edges only, drawn over geometry
- top-right: bubble diagram with all edges (incl. `wall`)
- bottom-left/middle/right: bubble variants with area labels / edge labels

The `bubble diagram (edge labels)` panel is the cleanest place to spot
suspicious wall edges — if you see `wall` between two rooms that are clearly
not next to each other in the geometry view, that is a V3 failure.

There is currently **no automated quality metric** for the wall inference.
A useful next step is a script that:

1. Picks N random samples
2. For each, computes the rate of wall edges that pair rooms whose minimum
   inter-polygon distance is greater than some sanity threshold (say
   0.5 m). Anything above that distance is almost certainly a false
   positive.
3. Reports the rate per-sample and globally.

---

## 7. Open improvement directions

### Direction A: drop wall entirely, use vanilla `msd`

Skip wall inference, train on the access-graph-only `msd` dataset (already
implemented). Matches the House Diffusion baseline; trades the noisy `wall`
signal for clean labels. Recommended if downstream tasks do not strictly
require spatial adjacency in the graph.

### Direction B: tighten `wall_segment_gap`

Lower from 0.45 to ~0.10 — meaning "the two contact segments must be
nearly touching on the same wall to count as a shared wall". Cheap to try
(no code change), but risks raising the false-negative rate on layouts
where doorways legitimately split a long shared wall.

### Direction C: re-derive from inter-room geometry

Replace "do they touch the same WALL polygon?" with a direct geometric
test:

```
for each pair (a, b):
    d = distance(room_a, room_b)
    if d > eps:                     continue        # not adjacent
    midline = shortest line between room_a and room_b boundaries
    if midline lies inside any WALL polygon (not inside any room polygon):
        add wall edge (a, b)
```

Pros: directly answers "are these two rooms separated by a wall?"; resilient
to MSD's habit of drawing one `WALL` polygon spanning many partitions.

Cons: needs careful handling of corner-kiss (the midline degenerates at
distance 0), and of cases where the MSD CSV does not draw a wall between
two rooms that are nonetheless adjacent (false negatives).

Estimated effort: half-day to a day. Recommended if `msd_wall` is a
required signal for the downstream pipeline.

### Direction D: side-step inference, label from a learned model

If MSD's room geometry is rich enough, train a small classifier on a
hand-labelled subset to score each candidate room-pair as wall / no-wall,
then propagate. Higher-effort, mainly useful if Directions B/C still fail
on edge cases.

---

## 8. History

> **TODO**: please fill in actual dates and the parameter values used at
> each iteration. The version sequence above (V1 → V2 → V3) is reconstructed
> from the natural evolution of the heuristic; the *exact* sequence of what
> was tried locally is not in git because `prepare_msd_graphs.py` is still
> untracked at the time of writing.

| Version | Approx. date | What changed | Why moved on |
|---|---|---|---|
| V1 | TODO | naive co-touch, no side check, no segment locality | over-paired rooms on the same long wall |
| V2 | TODO | added opposite-side check using wall normal | diagonal rooms still passed |
| V3 | (current) | added `wall_segment_gap` segment-locality filter | works for most cases; failure modes in §5 |

When implementing the next iteration, please:

1. Commit `prepare_msd_graphs.py` first so the prior version is captured in
   git history.
2. Update §2 with a new V$N$ entry describing the change and the failure
   mode it addresses.
3. Keep the parameter table in §3 in sync.

---

## 9. Where this lives in code

| Code | What it does |
|---|---|
| `digress/scripts/prepare_msd_graphs.py::_room_wall_contact` | Step 1: room-wall contact detection with `eps` and `min_contact_length` |
| `digress/scripts/prepare_msd_graphs.py::_wall_axes` | PCA on wall vertices, returns tangent/normal — the basis for the side check |
| `digress/scripts/prepare_msd_graphs.py::_local_wall_pairs` | Steps 2 + 3: side check and segment-gap pairing |
| `digress/scripts/prepare_msd_graphs.py::extract_structural_graph_from_floor` | Orchestration + final priority over passage/door/entrance/wall |
| `digress/scripts/prepare_msd_graphs.py::load_graphs_from_csv` (with `add_wall_edges=True`) | Calls `extract_structural_graph_from_floor` per `floor_id` |
| `digress/scripts/download_msd_dataset.py::--add-wall-edges` | Forwards the flag and threshold args to `prepare_msd_graphs.py` |
