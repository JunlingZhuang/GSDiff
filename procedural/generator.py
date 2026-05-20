"""Procedural floor-plan generator.

Pipeline (BSP-first):
  graph_in (bubble diagram) + boundary polygon
    ─► Stage 1: identify_backbone        # R1 (informational, not used by BSP)
    ─► Stage 2: bsp_partition            # graph-guided recursive split (PRIMARY)
    ─► Stage 3: enforce_hard_rules       # R3, R5, R7 check
    ─► Stage 4: local_search (optional)  # SA refinement, off by default
    ─► Stage 5: derive_walls             # extract walls from final polygons
    ─► return Layout

BSP guarantees 100% packing + 0 overlap by construction. SA is now a
post-processing pass for shape refinement when needed.

See docs/procedural_rules.md for the rules.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

import networkx as nx
from shapely.affinity import rotate as shp_rotate
from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from . import bsp, carve, io_msd, rules

# ---- types --------------------------------------------------------------------


@dataclass
class Room:
    node_id: int
    room_type: str
    polygon: Polygon


@dataclass
class Layout:
    rooms: dict[int, Room] = field(default_factory=dict)
    boundary: Polygon | None = None
    walls: list = field(default_factory=list)


# ---- Stage 1: backbone --------------------------------------------------------


def identify_backbone(graph: nx.Graph) -> nx.Graph:
    """R1: Corridor nodes form the backbone tree.

    Returns a subgraph containing the Steiner tree connecting all Corridor
    nodes (via any access edges). For graphs with no Corridor nodes, returns
    a tree rooted at the highest-degree node.
    """
    hubs = rules.identify_hub_nodes(graph)
    if not hubs:
        hubs = [max(graph.nodes, key=lambda n: graph.degree(n))]

    if len(hubs) == 1:
        return graph.subgraph(hubs).copy()

    access_edges = [(u, v) for u, v, d in graph.edges(data=True)
                    if d.get("connectivity") in rules.ACCESS_KINDS]
    access_g = graph.edge_subgraph(access_edges).copy()
    if not all(h in access_g for h in hubs):
        access_g = graph

    try:
        from networkx.algorithms.approximation import steiner_tree
        return steiner_tree(access_g, hubs, method="mehlhorn")
    except (nx.NetworkXError, nx.NodeNotFound):
        return access_g.subgraph(hubs).copy()


# ---- Stage 2: embed in boundary ----------------------------------------------


def _node_type(graph: nx.Graph, n: int) -> str | None:
    t = graph.nodes[n].get("room_type")
    if isinstance(t, int):
        return rules.ROOM_NAMES[t]
    return t


def dominant_angle(boundary: Polygon) -> float:
    """Detect the building's main axis as the length-weighted mode of edge
    angles (mod 90°).

    Buildings are rarely axis-aligned in world coords; cutting with arbitrary
    angles produces triangular fragments. Aligning BSP cuts to the building's
    own wall directions keeps every room a clean (rotated) rectangle.
    """
    coords = list(boundary.exterior.coords)
    bins = np.zeros(90)
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        length = math.hypot(dx, dy)
        ang = int(round(math.degrees(math.atan2(dy, dx)))) % 90
        bins[ang] += length
    return float(int(np.argmax(bins)))


def embed_in_boundary(
    graph: nx.Graph,
    backbone: nx.Graph,
    boundary: Polygon,
) -> dict[int, tuple[float, float]]:
    """Place nodes by exterior-wall requirement.

    Exterior-required rooms (Bedroom, Living, Balcony, Kitchen) → outer ring,
    placed by angle around the boundary perimeter to keep graph-neighbours
    near each other. Interior-allowed rooms (Corridor, Bathroom, Storeroom,
    Stairs) → inner blob via spring layout seeded at boundary centroid.
    """
    minx, miny, maxx, maxy = boundary.bounds
    cx, cy = boundary.centroid.coords[0]
    radius = 0.45 * min(maxx - minx, maxy - miny)

    exterior_types = rules.EXTERIOR_REQUIRED
    exterior_nodes = [n for n in graph.nodes if _node_type(graph, n) in exterior_types]
    interior_nodes = [n for n in graph.nodes if n not in exterior_nodes]

    pos: dict[int, tuple[float, float]] = {}

    # 1. exterior ring — order by access-graph traversal so neighbours stay close
    if exterior_nodes:
        access_edges = [(u, v) for u, v, d in graph.edges(data=True)
                        if d.get("connectivity") in rules.ACCESS_KINDS]
        access_g = graph.edge_subgraph(access_edges).copy() if access_edges else graph
        # BFS-order from an exterior node; mix nodes that aren't connected
        order = []
        seen = set()
        for start in exterior_nodes:
            if start in seen:
                continue
            try:
                for n in nx.bfs_tree(access_g, start):
                    if n in exterior_nodes and n not in seen:
                        order.append(n)
                        seen.add(n)
            except nx.NetworkXError:
                pass
        for n in exterior_nodes:
            if n not in seen:
                order.append(n)
                seen.add(n)

        for i, n in enumerate(order):
            angle = 2 * math.pi * i / len(order)
            pos[n] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    # 2. interior nodes — spring layout seeded around centroid
    if interior_nodes:
        seed = {n: pos[n] for n in exterior_nodes if n in pos}
        for n in interior_nodes:
            seed[n] = (cx + random.gauss(0, radius * 0.2),
                       cy + random.gauss(0, radius * 0.2))
        try:
            sp = nx.spring_layout(graph, pos=seed, fixed=list(pos) or None,
                                  iterations=30, seed=0, scale=radius * 0.5)
            for n in interior_nodes:
                xy = sp[n]
                pos[n] = (cx + xy[0], cy + xy[1])
        except Exception:
            for n in interior_nodes:
                pos[n] = seed[n]

    return pos


# ---- Stage 3: grow rooms ------------------------------------------------------


def grow_rooms(
    graph: nx.Graph,
    positions: dict[int, tuple[float, float]],
    boundary: Polygon,
) -> dict[int, Room]:
    """Emit a sized rectangle per node at its position.

    Area = priors p50, aspect = priors p50. Rectangle is axis-aligned;
    local_search rotates/scales/translates from here.
    """
    out: dict[int, Room] = {}
    for n in graph.nodes:
        t = _node_type(graph, n)
        if t is None or t not in rules.load_priors()["room_priors"]:
            continue
        target_area = rules.area_target(t)
        aspect = rules.load_priors()["room_priors"][t]["aspect"]["p50"]
        short = (target_area / aspect) ** 0.5
        long = short * aspect
        cx, cy = positions[n]
        rect = box(cx - long / 2, cy - short / 2, cx + long / 2, cy + short / 2)
        out[n] = Room(node_id=n, room_type=t, polygon=rect)
    return out


# ---- Stage 4: hard rules ------------------------------------------------------


def enforce_hard_rules(layout: Layout) -> rules.ValidationReport:
    rooms_map = {n: (r.polygon, r.room_type) for n, r in layout.rooms.items()}
    return rules.validate_layout(rooms_map, layout.boundary)


# ---- Stage 5: local search (simulated annealing) -----------------------------


# weights chosen so each term is O(10) at typical violation scales.
W_OVERLAP = 80.0       # area of pairwise overlap, m²        (hard)
W_OUT_OF_BOUNDS = 40.0 # area outside boundary, m²           (hard)
W_REQ_ADJ = 30.0       # m of distance between rooms that MUST touch  (hard — input graph)
W_GAP = 3.0            # unfilled boundary area, m²          (soft — packing pressure)
W_AREA = 1.0           # |area - target| / target            (soft — R3)
W_EXTERIOR = 5.0       # missing required exterior wall      (soft — R5/R7)


def _cost(layout: Layout, graph: nx.Graph,
          required_edges: list[tuple[int, int]]) -> float:
    """Cost = hard violations (overlap, oob, required-adjacency) + soft (gap, area, exterior).

    `required_edges` is the user-supplied bubble-diagram adjacency — these are
    constraints, not preferences. Every edge in `required_edges` means "rooms u
    and v MUST share at least one boundary segment" in the final layout.
    """
    rooms = layout.rooms
    boundary = layout.boundary

    polys = {n: r.polygon for n, r in rooms.items()}
    items = list(polys.items())

    # overlap (pairwise)
    overlap = 0.0
    for i, (na, pa) in enumerate(items):
        for nb, pb in items[i + 1:]:
            if pa.intersects(pb):
                inter = pa.intersection(pb)
                if not inter.is_empty:
                    overlap += inter.area

    # out-of-bounds
    out_of_bounds = 0.0
    for n, p in polys.items():
        try:
            inside = p.intersection(boundary).area
            out_of_bounds += max(0.0, p.area - inside)
        except Exception:
            out_of_bounds += p.area

    # gap: portion of boundary not covered by any room (drives packing)
    try:
        union = unary_union(list(polys.values()))
        filled = union.intersection(boundary).area
        gap = max(0.0, boundary.area - filled)
    except Exception:
        gap = 0.0

    # area deviation
    area_dev = 0.0
    for n, r in rooms.items():
        target = rules.area_target(r.room_type)
        area_dev += abs(r.polygon.area - target) / max(target, 1e-6)

    # exterior wall requirement
    exterior_miss = 0
    for n, r in rooms.items():
        if r.room_type in rules.EXTERIOR_REQUIRED:
            if not rules.shares_edge_with_boundary(r.polygon, boundary, eps=0.3):
                exterior_miss += 1

    # REQUIRED adjacency: input graph constraint — these rooms MUST touch
    req_adj_dist = 0.0
    for u, v in required_edges:
        if u in polys and v in polys:
            d = polys[u].distance(polys[v])
            req_adj_dist += d

    return (W_OVERLAP * overlap
            + W_OUT_OF_BOUNDS * out_of_bounds
            + W_REQ_ADJ * req_adj_dist
            + W_GAP * gap
            + W_AREA * area_dev
            + W_EXTERIOR * exterior_miss)


def _snap_to_neighbor(poly: Polygon, neighbor: Polygon) -> Polygon:
    """Translate `poly` so its closest face aligns with `neighbor`'s closest face.

    Looks at the 4 axis-aligned bbox edges of each, picks the pair with the
    smallest gap (signed gap can be negative = overlap; we still snap).
    """
    a = poly.bounds  # minx, miny, maxx, maxy
    b = neighbor.bounds
    options = [
        # (|gap|, dx, dy)
        (abs(b[0] - a[2]), b[0] - a[2], 0),   # A right edge ↔ B left edge
        (abs(b[2] - a[0]), b[2] - a[0], 0),   # A left  edge ↔ B right edge
        (abs(b[1] - a[3]), 0, b[1] - a[3]),   # A top   edge ↔ B bot   edge
        (abs(b[3] - a[1]), 0, b[3] - a[1]),   # A bot   edge ↔ B top   edge
    ]
    options.sort(key=lambda t: t[0])
    _, dx, dy = options[0]
    return shp_translate(poly, xoff=dx, yoff=dy)


def _swap_centers(poly_a: Polygon, poly_b: Polygon) -> tuple[Polygon, Polygon]:
    """Swap the centroids of two polygons; each keeps its own size/shape."""
    ca = poly_a.centroid
    cb = poly_b.centroid
    new_a = shp_translate(poly_a, xoff=cb.x - ca.x, yoff=cb.y - ca.y)
    new_b = shp_translate(poly_b, xoff=ca.x - cb.x, yoff=ca.y - cb.y)
    return new_a, new_b


def _random_move(layout: Layout, graph: nx.Graph, rng: random.Random,
                 step_size: float) -> tuple[str, list[tuple[int, Polygon]]]:
    """Pick a random move. Return (move_name, [(node, new_poly), ...])."""
    node_ids = list(layout.rooms)
    n = rng.choice(node_ids)
    poly = layout.rooms[n].polygon

    # bias: as T cools, prefer packing moves (snap) over exploration (translate)
    if step_size > 1.5:
        op = rng.choices(("translate", "resize", "aspect", "snap", "swap"),
                         weights=(3, 2, 2, 1, 1))[0]
    else:
        op = rng.choices(("translate", "resize", "aspect", "snap", "swap"),
                         weights=(2, 2, 2, 3, 1))[0]

    if op == "translate":
        dx = rng.gauss(0, step_size)
        dy = rng.gauss(0, step_size)
        return op, [(n, shp_translate(poly, xoff=dx, yoff=dy))]

    if op == "resize":
        f = 1.0 + rng.gauss(0, 0.08 + 0.04 * step_size)
        f = max(0.7, min(1.4, f))
        return op, [(n, shp_scale(poly, xfact=f, yfact=f, origin="center"))]

    if op == "aspect":
        f = 1.0 + rng.gauss(0, 0.08 + 0.04 * step_size)
        f = max(0.7, min(1.4, f))
        return op, [(n, shp_scale(poly, xfact=f, yfact=1 / f, origin="center"))]

    if op == "snap":
        nbrs = [m for m in graph.neighbors(n) if m in layout.rooms]
        if not nbrs:
            return "translate", [(n, shp_translate(poly, xoff=rng.gauss(0, step_size), yoff=rng.gauss(0, step_size)))]
        m = rng.choice(nbrs)
        return op, [(n, _snap_to_neighbor(poly, layout.rooms[m].polygon))]

    if op == "swap":
        # swap with another room of the same type if possible, else any room
        t = layout.rooms[n].room_type
        same_type = [k for k in node_ids if k != n and layout.rooms[k].room_type == t]
        m = rng.choice(same_type) if same_type else rng.choice([k for k in node_ids if k != n])
        new_a, new_b = _swap_centers(poly, layout.rooms[m].polygon)
        return op, [(n, new_a), (m, new_b)]

    return op, [(n, poly)]


def _extract_required_edges(graph: nx.Graph, edge_kinds: Iterable[str] | None) -> list[tuple[int, int]]:
    """Pull the user-input adjacency constraint from a graph.

    By default uses ACCESS_KINDS (door / passage / entrance) — these are the
    edges a user draws in a bubble diagram. Pass `None` to use every edge
    (e.g. when the input already includes wall edges as constraints).
    """
    if edge_kinds is None:
        return list(graph.edges())
    kinds = set(edge_kinds)
    return [(u, v) for u, v, d in graph.edges(data=True)
            if d.get("connectivity") in kinds]


def local_search(
    layout: Layout,
    graph: nx.Graph,
    max_iters: int = 4000,
    T0: float = 5.0,
    T_final: float = 0.05,
    seed: int = 0,
    verbose: bool = False,
    required_edge_kinds: Iterable[str] | None = ("door", "passage", "entrance"),
) -> Layout:
    """Simulated annealing.

    `required_edge_kinds` defines which input-graph edges are treated as
    hard adjacency constraints (must-touch). Default = access graph
    (door / passage / entrance), matching the bubble diagram a user would draw.
    Pass `None` to require every edge in the graph (use when input already
    includes wall edges).
    """
    rng = random.Random(seed)

    required_edges = _extract_required_edges(graph, required_edge_kinds)
    if verbose:
        print(f"  required adjacency edges (constraint): {len(required_edges)}")

    current_cost = _cost(layout, graph, required_edges)
    best_layout = Layout(
        rooms={n: Room(r.node_id, r.room_type, Polygon(r.polygon)) for n, r in layout.rooms.items()},
        boundary=layout.boundary,
    )
    best_cost = current_cost

    log_T = math.log(T_final / T0)
    accepts = 0
    move_stats = {}
    for it in range(max_iters):
        T = T0 * math.exp(log_T * it / max(max_iters - 1, 1))
        step_size = max(0.5, T)

        op, updates = _random_move(layout, graph, rng, step_size)
        # save old polys + apply updates
        olds = [(n, layout.rooms[n].polygon) for n, _ in updates]
        for n, new_poly in updates:
            layout.rooms[n].polygon = new_poly
        new_cost = _cost(layout, graph, required_edges)

        dE = new_cost - current_cost
        if dE < 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            current_cost = new_cost
            accepts += 1
            move_stats[op] = move_stats.get(op, 0) + 1
            if new_cost < best_cost:
                best_cost = new_cost
                best_layout = Layout(
                    rooms={k: Room(r.node_id, r.room_type, Polygon(r.polygon))
                           for k, r in layout.rooms.items()},
                    boundary=layout.boundary,
                )
        else:
            for n, old_poly in olds:
                layout.rooms[n].polygon = old_poly

        if verbose and (it + 1) % 1000 == 0:
            print(f"  iter {it+1:>5}  T={T:.3f}  cost={current_cost:>8.2f}  best={best_cost:>8.2f}  accept={accepts/(it+1):.2%}")

    if verbose:
        print(f"  done. best cost={best_cost:.2f} from {len(layout.rooms)} rooms")
        print(f"  accepts by move: {move_stats}")
    return best_layout


# ---- Stage 6: walls -----------------------------------------------------------


def derive_walls(layout: Layout) -> list:
    """Walls = polygon boundaries (one per room, duplicated edges acceptable for now).

    Future: merge shared edges, derive door positions from access edges.
    """
    return [room.polygon.boundary for room in layout.rooms.values()]


# ---- public entry point -------------------------------------------------------


def _bsp_grow(graph: nx.Graph, boundary: Polygon,
              fit_mode: str = "bbox_clip") -> dict[int, Room]:
    """Run BSP partition (with weighted KL) + wrap result as Room objects.

    fit_mode:
      "bbox_clip" (default) — BSP the boundary's bounding box so every room is a
          clean axis-aligned rectangle, then clip rooms to the real boundary so
          they don't spill into concave voids. Rooms stay rectangular except
          where the boundary cuts them. Matches "boundary as a bounding box".
      "exact" — BSP the real boundary directly. Rooms tile it 100% but contort
          to fill concave corners (non-rectangular).
    """
    target_areas: dict[int, float] = {}
    types: dict[int, str] = {}
    for n in graph.nodes:
        t = _node_type(graph, n)
        if t is None or t not in rules.load_priors()["room_priors"]:
            continue
        types[n] = t
        target_areas[n] = rules.area_target(t)

    subgraph = graph.subgraph(types).copy()
    bsp.annotate_edge_weights(subgraph)

    region = boundary
    if fit_mode == "bbox_clip":
        region = box(*boundary.bounds)   # clean rectangle → clean rectangular rooms

    polys = bsp.hierarchical_partition(subgraph, region, target_areas)

    out: dict[int, Room] = {}
    for n, poly in polys.items():
        if poly.is_empty:
            continue
        if fit_mode == "bbox_clip":
            clipped = poly.intersection(boundary)
            if clipped.is_empty or clipped.area < 0.05:
                continue
            if isinstance(clipped, MultiPolygon):
                clipped = max(clipped.geoms, key=lambda g: g.area)
            poly = clipped
        out[n] = Room(node_id=n, room_type=types[n], polygon=poly)
    return out


def generate(
    graph: nx.Graph,
    boundary: Polygon,
    axis_angle: float | None = None,
    fit_mode: str = "bbox_clip",
    do_carve: bool = False,
    carve_iters: int = 30,
    sa_refine_iters: int = 0,
    seed: int = 0,
    verbose: bool = False,
    required_edge_kinds: Iterable[str] | None = ("door", "passage", "entrance"),
) -> Layout:
    """Generate a floor plan from a bubble diagram and an apartment boundary.

    Pipeline: BSP (rectangles, 100% packed) -> carve (L-shapes, R3 fixup) ->
    optional SA refine.

    Args:
        graph: networkx Graph. Nodes need `room_type` (int or string). Edges
            need `connectivity` in {door, passage, entrance, wall}.
        boundary: shapely Polygon of the apartment outline.
        axis_angle: rotation in degrees for BSP cut directions. None (default)
            = auto-detect the building's dominant wall angle, so every room is
            a clean rectangle aligned to the building (no triangular fragments).
            Pass a number to force a specific angle, or 0 for world-axis cuts.
        do_carve: run the carve pass after BSP.
        carve_iters: max carve transfers.
        sa_refine_iters: optional SA refinement iterations (default 0).
        seed: RNG seed (used by SA only).
        verbose: print progress.
        required_edge_kinds: edge kinds counted as hard adjacency constraints
            during SA. Ignored if sa_refine_iters == 0.

    Returns:
        Layout (rooms + walls).
    """
    random.seed(seed)

    if axis_angle is None:
        axis_angle = dominant_angle(boundary)

    # Axis rotation: rotate boundary into BSP's axis-aligned frame, then rotate
    # output back. Equivalent to BSP cuts at `axis_angle`.
    if axis_angle != 0.0:
        cx, cy = boundary.centroid.x, boundary.centroid.y
        rotated_boundary = shp_rotate(boundary, -axis_angle, origin=(cx, cy))
        rooms_map = _bsp_grow(graph, rotated_boundary, fit_mode=fit_mode)
        if do_carve:
            tmp = Layout(rooms=rooms_map, boundary=rotated_boundary)
            tmp, n_carves = carve.carve_pass(tmp, graph, max_iters=carve_iters)
            rooms_map = tmp.rooms
            if verbose:
                print(f"  carve: {n_carves} area-transfers accepted")
        # rotate every room polygon back into the original frame
        for room in rooms_map.values():
            room.polygon = shp_rotate(room.polygon, axis_angle, origin=(cx, cy))
        layout = Layout(rooms=rooms_map, boundary=boundary)
    else:
        rooms_map = _bsp_grow(graph, boundary, fit_mode=fit_mode)
        layout = Layout(rooms=rooms_map, boundary=boundary)
        if do_carve:
            layout, n_carves = carve.carve_pass(layout, graph, max_iters=carve_iters)
            if verbose:
                print(f"  carve: {n_carves} area-transfers accepted")

    if sa_refine_iters > 0:
        if verbose:
            print(f"  SA refining for {sa_refine_iters} iters")
        layout = local_search(layout, graph, max_iters=sa_refine_iters, seed=seed,
                              T0=1.0, T_final=0.05,
                              verbose=verbose, required_edge_kinds=required_edge_kinds)

    layout.walls = derive_walls(layout)
    return layout


# ---- smoke test ---------------------------------------------------------------

if __name__ == "__main__":
    import pickle
    import time
    from collections import Counter
    from pathlib import Path

    import matplotlib.pyplot as plt
    from shapely.geometry import MultiPolygon

    p = Path(__file__).resolve().parents[1] / "digress" / "data" / "msd_wall_v6" / "graphs.p"
    with open(p, "rb") as f:
        graphs = pickle.load(f)

    # find a small graph for the demo (MSD min is 15 — multi-apartment buildings)
    small = sorted(graphs, key=lambda g: g.number_of_nodes())
    g = next(x for x in small if 15 <= x.number_of_nodes() <= 18)
    print(f"Picked graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # PROPER boundary: extract from struct_in (the real MSD input), NOT from GT polygons
    boundary = io_msd.boundary_for_processed_graph(g, split="train")
    if boundary is None:
        print(f"WARNING: no struct_in for graph ID {g.graph.get('ID')}, falling back to GT union")
        polys = [Polygon(d["geometry"]) for _, d in g.nodes(data=True)]
        boundary_raw = unary_union([p.buffer(0.02) for p in polys]).buffer(-0.02)
        boundary = (max(boundary_raw.geoms, key=lambda g: g.area)
                    if isinstance(boundary_raw, MultiPolygon) else boundary_raw)
    print(f"Boundary area (from struct_in): {boundary.area:.1f} m²")

    t0 = time.time()
    layout_bsp = generate(g, boundary, do_carve=False, verbose=True)
    bsp_time = time.time() - t0

    t1 = time.time()
    layout = generate(g, boundary, do_carve=True, carve_iters=40, verbose=True)
    carve_time = time.time() - t1

    print(f"\nBSP only         : {bsp_time:.3f}s")
    print(f"BSP + carve      : {carve_time:.3f}s")

    report = enforce_hard_rules(layout)
    print(f"Validation: ok={report.ok}, {len(report.failures)} failures")
    for f in report.failures[:8]:
        print(f"  - {f}")

    # check constraint satisfaction explicitly per edge kind
    print()
    for kinds, label in (
        (("door", "passage", "entrance"), "Access  "),
        (("wall",),                       "Wall    "),
        (("door", "passage", "entrance", "wall"), "All     "),
    ):
        req = _extract_required_edges(g, kinds)
        violated = [(u, v, layout.rooms[u].polygon.distance(layout.rooms[v].polygon))
                    for u, v in req
                    if u in layout.rooms and v in layout.rooms
                    and layout.rooms[u].polygon.distance(layout.rooms[v].polygon) > 0.05]
        pct = (1 - len(violated) / max(len(req), 1)) * 100
        print(f"  {label} edges: {len(req):>3} required, {len(violated):>3} violated  -> {pct:5.1f}% satisfied")

    # ---- visualization: 4 panels in a row ------------------------------------
    # 1. input bubble diagram (graph only, no geometry)
    # 2. ground truth polygons + graph overlay (edges colored by connectivity)
    # 3. BSP only + graph overlay
    # 4. BSP + carve + graph overlay

    EDGE_STYLE = {
        "entrance": dict(color="#d62728", linewidth=2.2, linestyle="-", alpha=0.9),
        "door":     dict(color="#1f77b4", linewidth=1.6, linestyle="-", alpha=0.85),
        "passage":  dict(color="#2ca02c", linewidth=1.6, linestyle="--", alpha=0.85),
        "wall":     dict(color="#888888", linewidth=0.8, linestyle=":", alpha=0.6),
    }

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    cmap = plt.cm.tab20
    type_to_idx = {t: i for i, t in enumerate(rules.ROOM_NAMES)}

    # Per-node centroids for each panel (input panel uses GT centroids for readability)
    gt_centroids = {n: tuple(g.nodes[n]['centroid']) if 'centroid' in g.nodes[n]
                    else Polygon(g.nodes[n]['geometry']).centroid.coords[0]
                    for n in g.nodes}

    def draw_polys(ax, rooms_iter, title, get_type, get_poly):
        bx, by = boundary.exterior.xy
        ax.plot(bx, by, color='black', linewidth=2, alpha=0.4)
        centroids = {}
        for item in rooms_iter:
            poly = get_poly(item)
            if poly is None or poly.is_empty:
                continue
            try:
                x, y = poly.exterior.xy
            except AttributeError:
                continue
            t = get_type(item)
            ax.fill(x, y, color=cmap(type_to_idx[t] % 20), alpha=0.55,
                    edgecolor='black', linewidth=0.5)
            cx, cy = poly.centroid.coords[0]
            n = item[0] if isinstance(item, tuple) else item.node_id
            centroids[n] = (cx, cy)
            ax.text(cx, cy, f"{n}\n{t[:4]}", ha='center', va='center', fontsize=7)
        ax.set_aspect('equal')
        ax.set_title(title)
        return centroids

    def overlay_graph_edges(ax, graph, centroids):
        for u, v, d in graph.edges(data=True):
            if u not in centroids or v not in centroids:
                continue
            kind = d.get("connectivity", "wall")
            style = EDGE_STYLE.get(kind, EDGE_STYLE["wall"])
            cx_u, cy_u = centroids[u]
            cx_v, cy_v = centroids[v]
            ax.plot([cx_u, cx_v], [cy_u, cy_v], **style)
        # scatter nodes on top
        for n, (cx, cy) in centroids.items():
            ax.plot(cx, cy, "o", color="white", markersize=4, markeredgecolor="black", zorder=10)

    # Panel 1: input bubble diagram (graph drawn on GT centroids, no polygons)
    ax = axes[0]
    edge_kind_counts = Counter(d.get("connectivity", "?") for _, _, d in g.edges(data=True))
    overlay_graph_edges(ax, g, gt_centroids)
    for n, (cx, cy) in gt_centroids.items():
        t = rules.ROOM_NAMES[g.nodes[n]['room_type']]
        ax.scatter(cx, cy, c=[cmap(type_to_idx[t] % 20)], s=420, edgecolors='black', zorder=20)
        ax.text(cx, cy, f"{n}\n{t[:4]}", ha='center', va='center', fontsize=6, zorder=21)
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, color='black', linewidth=1, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    title = f"Input bubble diagram\n{dict(edge_kind_counts)}"
    ax.set_title(title)
    # legend
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], label=k, **{**v, 'alpha': 1.0}) for k, v in EDGE_STYLE.items()]
    ax.legend(handles=legend, loc='lower left', fontsize=8, framealpha=0.9)

    # Panel 2: ground truth + graph overlay
    centroids_gt = draw_polys(
        axes[1], g.nodes(data=True),
        f"Ground truth ({g.number_of_nodes()} rooms)",
        lambda nd: rules.ROOM_NAMES[nd[1]['room_type']],
        lambda nd: Polygon(nd[1]['geometry']) if nd[1].get('geometry') else None,
    )
    overlay_graph_edges(axes[1], g, centroids_gt)

    # Panel 3: BSP only
    centroids_bsp = draw_polys(
        axes[2], layout_bsp.rooms.values(), "BSP only (rectangles)",
        lambda r: r.room_type, lambda r: r.polygon,
    )
    overlay_graph_edges(axes[2], g, centroids_bsp)

    # Panel 4: BSP + carve
    centroids_carve = draw_polys(
        axes[3], layout.rooms.values(), "BSP + carve (with L-shapes)",
        lambda r: r.room_type, lambda r: r.polygon,
    )
    overlay_graph_edges(axes[3], g, centroids_carve)

    out_path = Path(__file__).resolve().parent / "smoke_output.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\nSaved 4-panel detail to {out_path}")

    # ---- 10-sample grid: pick diverse sizes and render GT | Generated -------
    print("\nRendering 10-sample grid...")
    target_sizes = [15, 17, 19, 21, 23, 25, 28, 32, 38, 45]
    selected: list = []
    seen_sizes: set[int] = set()
    by_size: dict[int, list] = {}
    for graph in graphs:
        by_size.setdefault(graph.number_of_nodes(), []).append(graph)
    for sz in target_sizes:
        pool = by_size.get(sz) or by_size.get(min(by_size, key=lambda k: abs(k - sz)))
        if pool:
            cand = pool[0]
            selected.append(cand)
            seen_sizes.add(cand.number_of_nodes())

    # Input | GT | Generated (auto building-axis angle, seeds for variety)
    SEEDS = (0, 1, 2)
    n_cols = 2 + len(SEEDS)
    fig2, axes2 = plt.subplots(len(selected), n_cols,
                                 figsize=(5 * n_cols, 4 * len(selected)))
    for i, sample in enumerate(selected):
        # Real boundary from struct_in
        bnd_i = io_msd.boundary_for_processed_graph(sample, split="train")
        if bnd_i is None:
            polys_i = [Polygon(d['geometry']) for _, d in sample.nodes(data=True)
                       if d.get('geometry')]
            bnd_raw = unary_union([p.buffer(0.02) for p in polys_i]).buffer(-0.02)
            bnd_i = (max(bnd_raw.geoms, key=lambda g: g.area)
                     if isinstance(bnd_raw, MultiPolygon) else bnd_raw)

        # Panel 1: Input bubble diagram (nodes + colored edges, no geometry)
        ax = axes2[i, 0]
        bx, by = bnd_i.exterior.xy
        ax.plot(bx, by, color='black', linewidth=1.0, alpha=0.25, linestyle='--')
        gt_cents_i = {n: tuple(sample.nodes[n]['centroid']) if 'centroid' in sample.nodes[n]
                      else Polygon(sample.nodes[n]['geometry']).centroid.coords[0]
                      for n in sample.nodes if sample.nodes[n].get('geometry')}
        # draw edges first
        for u, v, d in sample.edges(data=True):
            if u not in gt_cents_i or v not in gt_cents_i:
                continue
            kind = d.get("connectivity", "wall")
            style = EDGE_STYLE.get(kind, EDGE_STYLE["wall"])
            ax.plot([gt_cents_i[u][0], gt_cents_i[v][0]],
                    [gt_cents_i[u][1], gt_cents_i[v][1]], **style)
        # then nodes on top
        for n, (cx, cy) in gt_cents_i.items():
            t = rules.ROOM_NAMES[sample.nodes[n]['room_type']]
            ax.scatter(cx, cy, c=[cmap(type_to_idx[t] % 20)], s=180,
                       edgecolors='black', linewidth=0.5, zorder=10)
            ax.text(cx, cy, t[:4], ha='center', va='center', fontsize=5, zorder=11)
        ax.set_aspect('equal')
        kinds = Counter(d.get('connectivity', '?') for _, _, d in sample.edges(data=True))
        ax.set_title(f"Sample {i+1}: input ({sample.number_of_nodes()}n / "
                     f"door={kinds.get('door',0)} wall={kinds.get('wall',0)})")

        # Panel 2: Ground truth polygons
        ax = axes2[i, 1]
        ax.plot(bx, by, color='black', linewidth=1.5, alpha=0.4)
        for n, d in sample.nodes(data=True):
            poly = Polygon(d['geometry'])
            if poly.is_empty:
                continue
            x, y = poly.exterior.xy
            t = rules.ROOM_NAMES[d['room_type']]
            ax.fill(x, y, color=cmap(type_to_idx[t] % 20), alpha=0.55,
                    edgecolor='black', linewidth=0.4)
            cx, cy = poly.centroid.coords[0]
            ax.text(cx, cy, t[:4], ha='center', va='center', fontsize=6)
        ax.set_aspect('equal')
        ax.set_title(f"Sample {i+1}: GT")

        # Panels 3..N: Generated at auto building-axis angle, different seeds
        for j, sd in enumerate(SEEDS):
            ax = axes2[i, 2 + j]
            try:
                ly = generate(sample, bnd_i, axis_angle=None, seed=sd,
                              do_carve=True, carve_iters=40, verbose=False)
            except Exception as e:
                ax.set_title(f"Sample {i+1}: seed={sd} (failed)")
                continue
            ax.plot(bx, by, color='black', linewidth=1.5, alpha=0.4)
            centroids_i = {}
            for n, room in ly.rooms.items():
                if room.polygon.is_empty:
                    continue
                try:
                    x, y = room.polygon.exterior.xy
                except AttributeError:
                    continue
                ax.fill(x, y, color=cmap(type_to_idx[room.room_type] % 20), alpha=0.55,
                        edgecolor='black', linewidth=0.4)
                cx, cy = room.polygon.centroid.coords[0]
                centroids_i[n] = (cx, cy)
                ax.text(cx, cy, room.room_type[:4], ha='center', va='center', fontsize=6)
            for u, v, d in sample.edges(data=True):
                if u not in centroids_i or v not in centroids_i:
                    continue
                kind = d.get("connectivity", "wall")
                if kind == "wall":
                    continue
                style = EDGE_STYLE.get(kind, EDGE_STYLE["wall"])
                ax.plot([centroids_i[u][0], centroids_i[v][0]],
                        [centroids_i[u][1], centroids_i[v][1]], **style)
            ax.set_aspect('equal')
            ax.set_title(f"Sample {i+1}: Gen (seed={sd})")

    grid_path = Path(__file__).resolve().parent / "samples_grid.png"
    plt.tight_layout()
    plt.savefig(grid_path, dpi=90)
    plt.close(fig2)
    print(f"Saved 10-sample grid to {grid_path}")
    print(f"  picked sizes: {sorted(seen_sizes)}")
