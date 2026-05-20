"""MSD I/O helpers — load the proper inputs for the procedural generator.

The MSD task is defined as:
    input  : struct_in (512x512x3 numpy)  +  graph_in (bubble diagram)
    target : graph_out (room polygons)

`struct_in` channel 0 is a binary structure mask (walls + columns + shafts).
Channels 1 and 2 are per-pixel world (x, y) coordinates in metres — they map
image pixels back to MSD's metric coordinate frame.

This module extracts the apartment boundary polygon directly from struct_in
without touching graph_out. That is the input the procedural generator should
receive in production.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_fill_holes
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage import measure

MSD_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "msd" / "raw" / "modified-swiss-dwellings-v2"
MSD_CSV = Path(__file__).resolve().parents[1] / "datasets" / "msd" / "raw" / "mds_V2_5.372k.csv"


def extract_boundary_from_struct(struct: np.ndarray) -> Polygon | None:
    """Pull the building's outer polygon out of a struct_in array.

    Correct algorithm (was previously over-filling via binary_fill_holes):
      1. wall_mask = ch0 < 127.5            (walls drawn dark on white)
      2. space_mask = ~wall_mask            (rooms + outside)
      3. label space_mask, mark labels touching image edge as "outside"
      4. interior = space - outside         (rooms only, no exterior)
      5. building = interior + walls        (rooms + the walls bounding them)
      6. largest connected component of building → main footprint
      7. contour + pixel→world via chx/chy
    """
    if struct.ndim != 3 or struct.shape[2] < 3:
        return None
    ch0 = struct[:, :, 0].astype(np.float32)
    chx = struct[:, :, 1].astype(np.float32)
    chy = struct[:, :, 2].astype(np.float32)
    h, w = ch0.shape

    wall_mask = ch0 < 127.5
    space_mask = ~wall_mask

    labeled_space, n_space = measure.label(space_mask, return_num=True)
    if n_space == 0:
        return None
    edge_labels = set()
    edge_labels.update(labeled_space[0, :].tolist())
    edge_labels.update(labeled_space[-1, :].tolist())
    edge_labels.update(labeled_space[:, 0].tolist())
    edge_labels.update(labeled_space[:, -1].tolist())
    edge_labels.discard(0)

    interior_mask = (labeled_space > 0) & np.isin(labeled_space, list(edge_labels), invert=True)
    if not interior_mask.any():
        return None

    building_mask = interior_mask | wall_mask
    labeled_bld, n_bld = measure.label(building_mask, return_num=True)
    if n_bld == 0:
        return None
    sizes = [(i, (labeled_bld == i).sum()) for i in range(1, n_bld + 1)]
    largest = max(sizes, key=lambda kv: kv[1])[0]
    main = (labeled_bld == largest)

    # CRITICAL: pad with False ring so a building that touches the image edge
    # still produces a single CLOSED contour (otherwise find_contours returns
    # multiple open arcs at the edge — was the source of the triangle bugs).
    PAD = 2
    main_padded = np.pad(main, PAD, mode="constant", constant_values=False)
    contours = measure.find_contours(main_padded.astype(float), 0.5)
    if not contours:
        return None
    biggest = max(contours, key=len)

    # undo padding offset; clip to valid index range
    rs = np.clip(biggest[:, 0].astype(int) - PAD, 0, h - 1)
    cs = np.clip(biggest[:, 1].astype(int) - PAD, 0, w - 1)
    world_x = chx[rs, cs]
    world_y = chy[rs, cs]

    poly = Polygon(list(zip(world_x, world_y)))
    if not poly.is_valid:
        poly = poly.buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    poly = poly.simplify(0.05, preserve_topology=True)
    return poly if poly.area > 1.0 else None


def _find_split(graph_id: int) -> str | None:
    for split in ("train", "test"):
        if (MSD_ROOT / split / "struct_in" / f"{graph_id}.npy").exists():
            return split
    return None


def load_msd_sample(graph_id: int, split: str | None = None) -> dict:
    """Load a full MSD sample (graph_in/graph_out/struct_in/boundary).

    If `split` is None, auto-detects from train/test.
    Missing files return None for that key rather than raising.
    """
    if split is None:
        split = _find_split(graph_id)
    if split is None:
        return {"graph_in": None, "graph_out": None, "struct_in": None, "boundary": None}
    base = MSD_ROOT / split
    out = {"graph_in": None, "graph_out": None, "struct_in": None, "boundary": None}
    for key, ext in (("graph_in", "pickle"), ("graph_out", "pickle"), ("struct_in", "npy")):
        p = base / key / f"{graph_id}.{ext}"
        if not p.exists():
            continue
        if ext == "pickle":
            with open(p, "rb") as f:
                out[key] = pickle.load(f)
        else:
            out[key] = np.load(p).astype(np.float32)
    if out["struct_in"] is not None:
        out["boundary"] = extract_boundary_from_struct(out["struct_in"])
    return out


@lru_cache(maxsize=1)
def _load_csv():
    import pandas as pd
    return pd.read_csv(MSD_CSV, usecols=["floor_id", "entity_type", "entity_subtype", "geom"])


def extract_boundary_from_csv(floor_id: int) -> Polygon | None:
    """Vector boundary from MSD CSV — exact, no pixel noise.

    WALL polygons are sparse line segments with door/window gaps, so unioning
    them gives 18+ disconnected pieces. ROOM polygons (entity_type='area')
    cover the entire buildable area without gaps — union them instead.

    The result is the exact area BSP should partition (matches what GT room
    polygons sum to).
    """
    df = _load_csv()
    sub = df[df["floor_id"] == floor_id]
    if sub.empty:
        return None
    rooms_df = sub[sub["entity_type"] == "area"]
    polys = []
    for g_str in rooms_df["geom"]:
        try:
            geom = wkt.loads(g_str)
        except Exception:
            continue
        if isinstance(geom, Polygon) and not geom.is_empty:
            polys.append(geom)
        elif isinstance(geom, MultiPolygon):
            polys.extend(g for g in geom.geoms if not g.is_empty)
    if not polys:
        return None
    # Rooms are separated by wall-thickness gaps (~0.1-0.2 m), so a tiny buffer
    # won't connect them. Buffer out 0.2 m (mitre join keeps corners sharp) to
    # bridge the wall gaps, union, then buffer back. The footprint then includes
    # the interior wall area, which is correct (the building outline encloses it).
    BUF = 0.2
    merged = unary_union([p.buffer(BUF, join_style=2) for p in polys]).buffer(-BUF, join_style=2)
    if isinstance(merged, MultiPolygon):
        merged = max(merged.geoms, key=lambda g: g.area)
    if not isinstance(merged, Polygon) or merged.is_empty:
        return None
    outer = Polygon(merged.exterior.coords)
    return outer.simplify(0.05, preserve_topology=True) if outer.is_valid else outer


def boundary_for_processed_graph(graph, split: str | None = None,
                                  prefer_csv: bool = True) -> Polygon | None:
    """Given a graph from msd_wall_v6/graphs.p (carries graph.graph['ID']),
    fetch its struct_in and return the extracted boundary polygon. Auto-detects
    split if not specified.
    """
    gid = graph.graph.get("ID")
    if gid is None:
        return None
    if prefer_csv:
        b = extract_boundary_from_csv(gid)
        if b is not None and not b.is_empty:
            return b
    if split is None:
        split = _find_split(gid)
    if split is None:
        return None
    npy_path = MSD_ROOT / split / "struct_in" / f"{gid}.npy"
    if not npy_path.exists():
        return None
    struct = np.load(npy_path).astype(np.float32)
    return extract_boundary_from_struct(struct)
