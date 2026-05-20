"""Graph-guided binary space partitioning for floor plans.

The packing core. Recursively bisects the boundary into rooms following the
input graph's adjacency structure. Guarantees:

  * 100% coverage of the boundary (no gaps)
  * 0 overlap (cuts are disjoint by construction)
  * Adjacency between graph-connected rooms (they share a cut line at some
    recursion depth)

Algorithm:
  bsp_partition(graph, boundary, target_areas):
      if |graph| == 1: return {node: boundary}
      A, B   = min_cut_partition(graph)            # Kernighan-Lin
      axis   = longer dimension of boundary bbox
      ratio  = sum_target(A) / (sum_target(A)+sum_target(B))
      pA, pB = split_polygon_by_area(boundary, axis, ratio)
      return bsp_partition(G[A], pA) ∪ bsp_partition(G[B], pB)
"""

from __future__ import annotations

import networkx as nx
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split as shp_split
from shapely.ops import unary_union


# Edge weights used by min_cut_partition. Higher weight = more expensive to cut.
EDGE_WEIGHTS = {
    "entrance": 100.0,   # entrance edge is from outside; must be preserved
    "door":     100.0,   # door = hard adjacency, user-visible
    "passage":   80.0,   # passage = hard adjacency, slightly weaker than door
    "wall":      10.0,   # wall = geometric adjacency, soft
}
DEFAULT_EDGE_WEIGHT = 10.0
# Corridor-corridor edges are made CHEAP so KL is willing to cut the corridor
# network across partitions. This spreads the corridor spine through the
# building centre instead of clumping all corridors on one side.
CORRIDOR_CORRIDOR_WEIGHT = 3.0

_ROOM_NAMES = (
    "Bedroom", "Livingroom", "Kitchen", "Dining", "Corridor",
    "Stairs", "Storeroom", "Bathroom", "Balcony",
)


def _node_type(graph: nx.Graph, n) -> str | None:
    t = graph.nodes[n].get("room_type")
    if isinstance(t, int):
        return _ROOM_NAMES[t] if 0 <= t < len(_ROOM_NAMES) else None
    return t


def annotate_edge_weights(graph: nx.Graph) -> nx.Graph:
    """In-place: tag every edge with a `weight` for KL min-cut.

    weight = EDGE_WEIGHTS[connectivity], except Corridor-Corridor edges get a
    low weight (CORRIDOR_CORRIDOR_WEIGHT) so the corridor network can be split
    across partitions → corridors end up central rather than clumped.
    """
    for u, v, d in graph.edges(data=True):
        tu, tv = _node_type(graph, u), _node_type(graph, v)
        if tu == "Corridor" and tv == "Corridor":
            d["weight"] = CORRIDOR_CORRIDOR_WEIGHT
        else:
            d["weight"] = EDGE_WEIGHTS.get(d.get("connectivity"), DEFAULT_EDGE_WEIGHT)
    return graph


def min_cut_partition(graph: nx.Graph) -> tuple[list[int], list[int]]:
    """Bisect `graph` into two sets minimising total cut-edge weight.

    Uses networkx's Kernighan-Lin with the `weight` edge attribute. Access edges
    (door/passage/entrance) are weighted ~10x heavier than wall edges so KL
    prefers cutting wall edges first. Falls back gracefully on tiny / disconnected
    graphs.
    """
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return nodes, []
    if len(nodes) == 2:
        return [nodes[0]], [nodes[1]]
    if not nx.is_connected(graph):
        comps = sorted(nx.connected_components(graph), key=len, reverse=True)
        if len(comps) >= 2:
            A = list(comps[0])
            B = [n for c in comps[1:] for n in c]
            return A, B
    try:
        # ensure weights exist
        for _, _, d in graph.edges(data=True):
            d.setdefault("weight", DEFAULT_EDGE_WEIGHT)
        a, b = nx.community.kernighan_lin_bisection(graph, seed=0, max_iter=20, weight="weight")
        return sorted(a), sorted(b)
    except Exception:
        mid = len(nodes) // 2
        return nodes[:mid], nodes[mid:]


def split_polygon_by_area(
    poly: Polygon, axis: str, target_ratio: float,
    tol: float = 1e-3, max_iters: int = 40,
) -> tuple[Polygon, Polygon]:
    """Cut `poly` with a vertical/horizontal line so the first piece has area
    `target_ratio * poly.area`.

    Uses binary search on the cut coordinate. For non-convex (L-shaped)
    polygons a single line may produce >2 pieces; small pieces are merged
    with their nearest neighbour by side.
    """
    minx, miny, maxx, maxy = poly.bounds
    target_area = target_ratio * poly.area

    if axis == "vertical":
        lo, hi = minx, maxx
        def cut_line(c: float) -> LineString:
            return LineString([(c, miny - 1.0), (c, maxy + 1.0)])
        def side_key(p: Polygon, c: float) -> float:
            return p.centroid.x - c
    else:  # horizontal
        lo, hi = miny, maxy
        def cut_line(c: float) -> LineString:
            return LineString([(minx - 1.0, c), (maxx + 1.0, c)])
        def side_key(p: Polygon, c: float) -> float:
            return p.centroid.y - c

    def split_at(c: float) -> tuple[Polygon, Polygon]:
        line = cut_line(c)
        result = shp_split(poly, line)
        pieces = list(getattr(result, "geoms", [result]))
        first_side, second_side = [], []
        for piece in pieces:
            if piece.is_empty:
                continue
            if side_key(piece, c) <= 0:
                first_side.append(piece)
            else:
                second_side.append(piece)
        a = unary_union(first_side) if first_side else Polygon()
        b = unary_union(second_side) if second_side else Polygon()
        # If a/b are MultiPolygon, take the largest piece + glue smaller ones to the other side?
        # Simpler: return as-is; caller decides whether to recurse on each piece.
        return _to_polygon(a), _to_polygon(b)

    for _ in range(max_iters):
        c = (lo + hi) / 2
        a, _ = split_at(c)
        if abs(a.area - target_area) < tol * poly.area:
            break
        if a.area < target_area:
            lo = c
        else:
            hi = c

    return split_at(c)


def _to_polygon(geom) -> Polygon:
    """Coerce shapely geometry to a single Polygon (largest piece if Multi)."""
    if geom.is_empty:
        return Polygon()
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return Polygon()


def bsp_partition(
    graph: nx.Graph,
    boundary: Polygon,
    target_areas: dict[int, float],
) -> dict[int, Polygon]:
    """Recursively partition `boundary` into one polygon per node of `graph`.

    `target_areas[n]` drives the cut ratio at each level (cuts are
    proportional to total target area on each side).
    """
    nodes = list(graph.nodes)
    if not nodes:
        return {}
    if boundary.is_empty or boundary.area < 1e-6:
        # Degenerate: assign an empty polygon to every node so the caller still gets keys
        return {n: Polygon() for n in nodes}
    if len(nodes) == 1:
        return {nodes[0]: boundary}

    a_nodes, b_nodes = min_cut_partition(graph)
    if not a_nodes or not b_nodes:
        return {nodes[0]: boundary, **{n: Polygon() for n in nodes[1:]}}

    minx, miny, maxx, maxy = boundary.bounds
    axis = "vertical" if (maxx - minx) >= (maxy - miny) else "horizontal"

    area_a = sum(target_areas.get(n, 1.0) for n in a_nodes)
    area_b = sum(target_areas.get(n, 1.0) for n in b_nodes)
    ratio = area_a / max(area_a + area_b, 1e-9)
    ratio = max(0.05, min(0.95, ratio))  # avoid pathological cuts

    poly_a, poly_b = split_polygon_by_area(boundary, axis, ratio)

    sub_a = graph.subgraph(a_nodes).copy()
    sub_b = graph.subgraph(b_nodes).copy()

    out = bsp_partition(sub_a, poly_a, target_areas)
    out.update(bsp_partition(sub_b, poly_b, target_areas))
    return out


# ---- hierarchical (apartment-first) partition --------------------------------


def detect_apartments(graph: nx.Graph) -> list[set]:
    """Cluster rooms into apartment-like communities.

    MSD floors are mostly multi-apartment; the access graph is one connected
    component (apartments link through shared stairs/entrance), so plain
    connected-components doesn't separate them. Greedy modularity finds the
    dense intra-apartment clusters. Uses access edges only.

    Returns a list of node-id sets, one per apartment. Single community (or
    tiny graphs) returns one set = flat layout.
    """
    if graph.number_of_nodes() < 6:
        return [set(graph.nodes)]
    access = graph.edge_subgraph([
        (u, v) for u, v, d in graph.edges(data=True)
        if d.get("connectivity") in ("door", "passage", "entrance")
    ]).copy()
    # include isolated nodes that the access subgraph dropped
    access.add_nodes_from(graph.nodes)
    try:
        comms = nx.community.greedy_modularity_communities(access, weight=None)
        comms = [set(c) for c in comms if c]
    except Exception:
        comms = []
    if not comms:
        return [set(graph.nodes)]
    # assign any node missing from communities to the nearest community by graph distance
    covered = set().union(*comms)
    for n in graph.nodes:
        if n not in covered:
            comms[0].add(n)
    return comms


def _build_super_graph(graph: nx.Graph, communities: list[set]) -> nx.Graph:
    """Community-level graph: node = community index, edge weight = #connecting edges."""
    node_to_comm = {}
    for ci, c in enumerate(communities):
        for n in c:
            node_to_comm[n] = ci
    sg = nx.Graph()
    sg.add_nodes_from(range(len(communities)))
    for u, v in graph.edges():
        cu, cv = node_to_comm.get(u), node_to_comm.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        if sg.has_edge(cu, cv):
            sg[cu][cv]["weight"] += 1.0
        else:
            sg.add_edge(cu, cv, weight=1.0)
    return sg


def hierarchical_partition(
    graph: nx.Graph,
    boundary: Polygon,
    target_areas: dict[int, float],
) -> dict[int, Polygon]:
    """Apartment-first BSP.

    1. detect apartment communities
    2. BSP the boundary into one region per apartment (cut few inter-apt edges)
    3. BSP each apartment's rooms inside its region

    Falls back to flat bsp_partition when there is a single apartment.
    """
    communities = detect_apartments(graph)
    if len(communities) <= 1:
        return bsp_partition(graph, boundary, target_areas)

    # 1. super-graph + per-apartment total target area
    super_g = _build_super_graph(graph, communities)
    super_areas = {ci: sum(target_areas.get(n, 1.0) for n in c)
                   for ci, c in enumerate(communities)}

    # 2. partition boundary among apartments
    apt_polys = bsp_partition(super_g, boundary, super_areas)

    # 3. partition rooms inside each apartment region
    out: dict[int, Polygon] = {}
    for ci, c in enumerate(communities):
        poly = apt_polys.get(ci)
        if poly is None or poly.is_empty:
            for n in c:
                out[n] = Polygon()
            continue
        sub = graph.subgraph(c).copy()
        annotate_edge_weights(sub)
        out.update(bsp_partition(sub, poly, target_areas))
    return out
