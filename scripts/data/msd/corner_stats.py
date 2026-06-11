"""MSD corner-count statistics for GSDiff scaling analysis.

For each MSD graph (rooms-with-polygons), reconstruct the corner-wall view:
- corners = unique polygon vertices across all room polygons, merged with eps tol
- walls = polygon-boundary segments between consecutive merged vertices

Print quantiles of corner count, wall count, room count.
"""
import glob
import os
import pickle
import sys

import numpy as np


def merged_corners_and_walls(graph, eps=1e-3):
    pts = []
    polys_idx = []
    for _, ndata in graph.nodes(data=True):
        geom = ndata.get("geometry")
        if not geom:
            continue
        coords = list(geom)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        idxs = []
        for x, y in coords:
            idxs.append(len(pts))
            pts.append((x, y))
        polys_idx.append(idxs)

    if not pts:
        return 0, 0

    arr = np.asarray(pts, dtype=np.float64)
    keys = np.round(arr / eps).astype(np.int64)
    keys_view = keys.view([("", keys.dtype)] * 2).ravel()
    _, inv = np.unique(keys_view, return_inverse=True)

    n_corners = int(inv.max() + 1)

    walls = set()
    for idxs in polys_idx:
        n = len(idxs)
        for k in range(n):
            a = int(inv[idxs[k]])
            b = int(inv[idxs[(k + 1) % n]])
            if a == b:
                continue
            walls.add((min(a, b), max(a, b)))
    return n_corners, len(walls)


def quantiles(values, qs=(0, 25, 50, 75, 90, 95, 99, 100)):
    arr = np.asarray(values)
    return {q: float(np.percentile(arr, q)) for q in qs}


def fmt(q):
    return ", ".join(f"p{k}={v:.0f}" for k, v in q.items())


def main():
    base = "datasets/msd/raw/modified-swiss-dwellings-v2"
    splits = ["train", "test"]

    rows = []
    for split in splits:
        files = sorted(glob.glob(os.path.join(base, split, "graph_out", "*.pickle")))
        for fp in files:
            with open(fp, "rb") as f:
                G = pickle.load(f)
            n_corners, n_walls = merged_corners_and_walls(G)
            rows.append({
                "split": split,
                "rooms": G.number_of_nodes(),
                "room_edges": G.number_of_edges(),
                "corners": n_corners,
                "walls": n_walls,
            })

    print(f"total graphs: {len(rows)}")
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)

    for split, rs in by_split.items():
        print(f"\n=== {split} ({len(rs)} graphs) ===")
        for key in ("rooms", "corners", "walls"):
            vals = [r[key] for r in rs]
            print(f"  {key:8s}  mean={np.mean(vals):.1f}  {fmt(quantiles(vals))}")

    all_corners = [r["corners"] for r in rows]
    thresholds = [53, 100, 150, 200, 256, 300, 400, 512]
    print("\n=== coverage if N_max = T (fraction of graphs with corners <= T) ===")
    for T in thresholds:
        frac = float(np.mean([c <= T for c in all_corners]))
        print(f"  N_max={T:4d}  coverage={frac*100:5.1f}%  dropped={len(all_corners)-int(round(frac*len(all_corners)))} graphs")


if __name__ == "__main__":
    main()
