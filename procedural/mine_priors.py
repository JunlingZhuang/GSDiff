"""Mine procedural priors from MSD wall v6 graphs.

Reads digress/data/msd_wall_v6/graphs.p (5372 graphs with wall + door + passage +
entrance edges) and produces procedural/priors.json — the data-derived constants
that procedural/rules.py uses.

Usage:
    python procedural/mine_priors.py
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

ROOM_NAMES = [
    "Bedroom", "Livingroom", "Kitchen", "Dining", "Corridor",
    "Stairs", "Storeroom", "Bathroom", "Balcony",
]
EDGE_KINDS = ("wall", "door", "passage", "entrance")
ACCESS_KINDS = ("door", "passage", "entrance")

GRAPHS_PATH = Path(__file__).resolve().parents[1] / "digress" / "data" / "msd_wall_v6" / "graphs.p"
OUT_PATH = Path(__file__).resolve().parent / "priors.json"


def pct(arr, qs=(10, 25, 50, 75, 90)):
    a = np.asarray(arr)
    return {f"p{q}": float(round(np.percentile(a, q), 3)) for q in qs}


def main():
    print(f"Loading {GRAPHS_PATH} ...")
    with open(GRAPHS_PATH, "rb") as f:
        graphs = pickle.load(f)
    print(f"  {len(graphs)} graphs")

    # ---- per-room metrics ----
    room_stats = defaultdict(lambda: defaultdict(list))   # type -> metric -> [values]
    degree_by_type_kind = defaultdict(lambda: defaultdict(list))  # type -> kind -> [degrees]
    apt_node_counts = []
    apt_edge_counts = []
    edge_kind_global = Counter()

    # ---- pairwise metrics ----
    pair_count = {k: Counter() for k in EDGE_KINDS}        # (a,b) -> count
    # NOTE: shared_wall_length is intentionally NOT mined here.
    # MSD wall polygons have thickness, so room polygons never share a true edge;
    # buffer-based approximation gave nonsense numbers in earlier iterations.
    # The generator computes shared-wall length on-the-fly from its own layout.

    for g_idx, g in enumerate(graphs):
        apt_node_counts.append(g.number_of_nodes())
        apt_edge_counts.append(g.number_of_edges())

        polys = {}
        types = {}
        for n, d in g.nodes(data=True):
            try:
                poly = Polygon(d["geometry"])
                if not poly.is_valid or poly.area < 0.05:
                    continue
            except Exception:
                continue
            polys[n] = poly
            tname = ROOM_NAMES[d["room_type"]]
            types[n] = tname

            bbox = poly.minimum_rotated_rectangle
            bx, by = bbox.exterior.coords.xy
            edges = [((bx[i + 1] - bx[i]) ** 2 + (by[i + 1] - by[i]) ** 2) ** 0.5 for i in range(4)]
            long_e, short_e = max(edges[:2]), min(edges[:2])

            s = room_stats[tname]
            s["area"].append(poly.area)
            s["aspect"].append(long_e / max(short_e, 1e-6))
            s["vertices"].append(len(d["geometry"]) - 1)
            s["convexity"].append(poly.area / poly.convex_hull.area)

        # per-type degree split by edge kind
        deg_per_node_kind = defaultdict(lambda: Counter())
        for u, v, ed in g.edges(data=True):
            kind = ed.get("connectivity", "?")
            edge_kind_global[kind] += 1
            deg_per_node_kind[u][kind] += 1
            deg_per_node_kind[v][kind] += 1

            if u not in types or v not in types:
                continue
            a, b = sorted([types[u], types[v]])
            if kind in EDGE_KINDS:
                pair_count[kind][(a, b)] += 1

        for n, t in types.items():
            for kind in EDGE_KINDS:
                degree_by_type_kind[t][kind].append(deg_per_node_kind[n][kind])
            degree_by_type_kind[t]["total"].append(sum(deg_per_node_kind[n].values()))

        if (g_idx + 1) % 1000 == 0:
            print(f"  processed {g_idx + 1} / {len(graphs)}")

    # ---- assemble priors ----
    priors = {
        "source": "digress/data/msd_wall_v6/graphs.p",
        "graph_count": len(graphs),
        "edge_kind_totals": dict(edge_kind_global.most_common()),
        "apartment": {
            "nodes_per_graph": pct(apt_node_counts),
            "edges_per_graph": pct(apt_edge_counts),
        },
        "room_priors": {},
        "adjacency_priors": {
            "wall_pairs": {},
            "access_pairs": {},
        },
    }

    for tname in ROOM_NAMES:
        s = room_stats.get(tname)
        if not s or not s["area"]:
            continue
        priors["room_priors"][tname] = {
            "n": len(s["area"]),
            "area_m2": pct(s["area"]),
            "aspect": pct(s["aspect"]),
            "vertices": pct(s["vertices"]),
            "convexity": pct(s["convexity"]),
            "rect_fraction": float(round((np.asarray(s["vertices"]) == 4).mean(), 3)),
            "degree_total": pct(degree_by_type_kind[tname]["total"]),
            "degree_wall": pct(degree_by_type_kind[tname]["wall"]),
            "degree_door": pct(degree_by_type_kind[tname]["door"]),
        }

    # adjacency pairs — wall only and access combined
    for pair, cnt in pair_count["wall"].most_common():
        priors["adjacency_priors"]["wall_pairs"][f"{pair[0]}|{pair[1]}"] = {"count": cnt}

    access_pair_count = Counter()
    for k in ACCESS_KINDS:
        for pair, cnt in pair_count[k].items():
            access_pair_count[pair] += cnt
    for pair, cnt in access_pair_count.most_common():
        priors["adjacency_priors"]["access_pairs"][f"{pair[0]}|{pair[1]}"] = {"count": cnt}

    OUT_PATH.write_text(json.dumps(priors, indent=2))
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")

    # ---- console summary ----
    print("\n=== APARTMENT ===")
    print(f"  nodes/apt p50={priors['apartment']['nodes_per_graph']['p50']:.0f}, "
          f"p90={priors['apartment']['nodes_per_graph']['p90']:.0f}")
    print(f"  edges/apt p50={priors['apartment']['edges_per_graph']['p50']:.0f}, "
          f"p90={priors['apartment']['edges_per_graph']['p90']:.0f}")
    print(f"  edge kinds: {priors['edge_kind_totals']}")

    print("\n=== ROOM PRIORS (area p50 / rect_frac / degree_total p50) ===")
    for t, p in priors["room_priors"].items():
        print(f"  {t:<12} n={p['n']:>6}  area={p['area_m2']['p50']:>6.2f}  "
              f"rect={p['rect_fraction']:.2f}  deg={p['degree_total']['p50']:.1f}")

    print("\n=== TOP 10 WALL PAIRS (rooms touching but no door) ===")
    items = sorted(priors["adjacency_priors"]["wall_pairs"].items(), key=lambda kv: -kv[1]["count"])
    for pair, info in items[:10]:
        print(f"  {pair:<30} count={info['count']:>6}")

    print("\n=== TOP 10 ACCESS PAIRS (door+passage+entrance) ===")
    items = sorted(priors["adjacency_priors"]["access_pairs"].items(), key=lambda kv: -kv[1]["count"])
    for pair, info in items[:10]:
        print(f"  {pair:<30} count={info['count']:>6}")


if __name__ == "__main__":
    main()
