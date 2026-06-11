"""Prepare a floor-level hierarchical MSD dataset for GSDiff experiments.

This is an additive MSD pipeline. It does not touch the existing RPLAN/GSDiff
preprocessing outputs. The generated files are intended as QA-friendly
intermediate artifacts before training a hierarchical GSDiff model.

Default wall graph source is the MSD wall v6 graph pickle produced by the
DiGress preprocessing pipeline. Raw CSV geometry is still used for unit/public
grouping and target polygons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import shutil
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


ROOM_TYPES = [
    "Bedroom",
    "Livingroom",
    "Kitchen",
    "Dining",
    "Corridor",
    "Stairs",
    "Storeroom",
    "Bathroom",
    "Balcony",
]

PUBLIC_ROOM_TYPES = {"Stairs"}

ROOM_COLORS = {
    "Bedroom": "#8da0cb",
    "Livingroom": "#47b39c",
    "Kitchen": "#f1c27d",
    "Dining": "#fdae6b",
    "Corridor": "#fdd0a2",
    "Stairs": "#72246c",
    "Storeroom": "#ffd92f",
    "Bathroom": "#bdbdbd",
    "Balcony": "#a6d854",
}

GROUP_COLORS = {
    "unit": "#60a5fa",
    "corridor": "#f97316",
    "stairs": "#7c3aed",
    "public": "#94a3b8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MSD hierarchical floor/coarse/local samples.")
    parser.add_argument(
        "--source",
        default="datasets/msd/raw/mds_V2_5.372k.csv",
        help="MSD raw CSV path.",
    )
    parser.add_argument(
        "--wall-graphs",
        default="digress/data/msd_wall_v6/graphs.p",
        help="Optional MSD wall v6 graphs.p path. Use empty string to disable.",
    )
    parser.add_argument("--out", default="datasets/msd_hier_v6", help="Output dataset directory.")
    parser.add_argument("--limit-floors", type=int, default=None, help="Optional smoke-test floor limit.")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--simplify-tolerance", type=float, default=0.03)
    parser.add_argument("--adjacency-eps", type=float, default=0.05)
    parser.add_argument("--corner-merge-eps", type=float, default=1e-3)
    parser.add_argument(
        "--split-disconnected-units",
        action="store_true",
        help="Split same-unit rooms by geometry components. Disabled by default because MSD unit_id is the primary apartment grouping signal.",
    )
    parser.add_argument("--min-rooms", type=int, default=2)
    parser.add_argument("--max-rooms", type=int, default=None)
    parser.add_argument("--max-corners", type=int, default=None)
    parser.add_argument("--vis-samples", type=int, default=50)
    parser.add_argument("--clean", action="store_true", help="Remove existing output directory first.")
    return parser.parse_args()


def normalize_unit_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        as_float = float(text)
    except ValueError:
        return text
    if not math.isfinite(as_float):
        return None
    as_int = int(as_float)
    if abs(as_float - as_int) < 1e-6:
        return str(as_int)
    return text


def clean_geometry(geom) -> Polygon | MultiPolygon | None:
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    polygons = [part for part in getattr(geom, "geoms", []) if isinstance(part, Polygon)]
    if not polygons:
        return None
    return unary_union(polygons)


def safe_distance(geom_a, geom_b) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            distance = geom_a.distance(geom_b)
        except Exception:
            return float("inf")
    if not math.isfinite(distance):
        return float("inf")
    return float(distance)


def load_selected_floor_ids(source: Path, limit_floors: int | None) -> set[str] | None:
    if limit_floors is None:
        return None
    selected: list[str] = []
    seen: set[str] = set()
    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            floor_id = str(row["floor_id"])
            if floor_id not in seen:
                seen.add(floor_id)
                selected.append(floor_id)
                if len(selected) >= limit_floors:
                    break
    return set(selected)


def load_rooms_from_csv(source: Path, selected_floor_ids: set[str] | None) -> dict[str, list[dict]]:
    floors: dict[str, list[dict]] = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            floor_id = str(row["floor_id"])
            if selected_floor_ids is not None and floor_id not in selected_floor_ids:
                continue
            if row.get("entity_type") != "area":
                continue
            room_type = row.get("roomtype") or row.get("entity_subtype")
            if room_type not in ROOM_TYPES:
                continue
            try:
                geom = clean_geometry(wkt.loads(row["geom"]))
            except Exception:
                continue
            if geom is None:
                continue
            if isinstance(geom, MultiPolygon):
                # Room targets should be single polygons. Keep the largest part
                # and let QA expose any floors where this is a bad assumption.
                geom = max(geom.geoms, key=lambda part: part.area)
            room_index = len(floors[floor_id])
            floors[floor_id].append(
                {
                    "room_index": room_index,
                    "floor_id": floor_id,
                    "area_id": row.get("area_id"),
                    "unit_id": normalize_unit_id(row.get("unit_id")),
                    "room_type": room_type,
                    "polygon": geom,
                }
            )
    return floors


def load_wall_graphs(path: Path | None) -> dict[str, nx.Graph]:
    if path is None or not path.exists():
        return {}
    with path.open("rb") as f:
        graphs = pickle.load(f)
    return {str(graph.graph.get("ID")): graph for graph in graphs if graph.graph.get("ID") is not None}


def components_by_geometry(room_indices: list[int], rooms: list[dict], eps: float) -> list[list[int]]:
    if not room_indices:
        return []
    graph = nx.Graph()
    graph.add_nodes_from(room_indices)
    for pos, i in enumerate(room_indices):
        geom_i = rooms[i]["polygon"]
        for j in room_indices[pos + 1 :]:
            geom_j = rooms[j]["polygon"]
            if safe_distance(geom_i, geom_j) <= eps or geom_i.intersects(geom_j):
                graph.add_edge(i, j)
    return [sorted(component) for component in nx.connected_components(graph)]


def polygons_to_coords(geom) -> list[list[tuple[float, float]]]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        return []
    coords = []
    for polygon in polygons:
        exterior = [(float(x), float(y)) for x, y in polygon.exterior.coords]
        if len(exterior) >= 4:
            coords.append(exterior)
    return coords


def unique_corner_count(polygons: list, eps: float) -> int:
    points = []
    for geom in polygons:
        if isinstance(geom, Polygon):
            coords = list(geom.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            points.extend(coords)
        elif isinstance(geom, MultiPolygon):
            for polygon in geom.geoms:
                coords = list(polygon.exterior.coords)
                if len(coords) > 1 and coords[0] == coords[-1]:
                    coords = coords[:-1]
                points.extend(coords)
    if not points:
        return 0
    arr = np.asarray(points, dtype=np.float64)
    keys = np.round(arr / eps).astype(np.int64)
    return int(len({(int(x), int(y)) for x, y in keys}))


def simplify_polygon(geom, tolerance: float):
    if tolerance <= 0:
        return geom
    simplified = geom.simplify(tolerance, preserve_topology=True)
    return clean_geometry(simplified) or geom


def build_groups(rooms: list[dict], eps: float, simplify_tolerance: float, split_disconnected_units: bool) -> list[dict]:
    unit_candidates: dict[str, list[int]] = defaultdict(list)
    public_candidates: list[int] = []

    for room in rooms:
        idx = room["room_index"]
        unit_id = room["unit_id"]
        if room["room_type"] in PUBLIC_ROOM_TYPES or unit_id is None:
            public_candidates.append(idx)
        else:
            unit_candidates[unit_id].append(idx)

    groups = []
    for unit_id in sorted(unit_candidates, key=lambda value: int(value) if value.isdigit() else value):
        if split_disconnected_units:
            components = components_by_geometry(unit_candidates[unit_id], rooms, eps)
        else:
            components = [sorted(unit_candidates[unit_id])]
        for component_index, room_indices in enumerate(components):
            groups.append(make_group("unit", f"unit_{unit_id}_c{component_index}", room_indices, rooms, simplify_tolerance))

    for component_index, room_indices in enumerate(components_by_geometry(public_candidates, rooms, eps)):
        counts = Counter(rooms[i]["room_type"] for i in room_indices)
        if counts.get("Stairs", 0) == len(room_indices):
            group_type = "stairs"
        elif counts.get("Corridor", 0) >= max(1, len(room_indices) // 2):
            group_type = "corridor"
        else:
            group_type = "public"
        groups.append(make_group(group_type, f"{group_type}_c{component_index}", room_indices, rooms, simplify_tolerance))

    for group_index, group in enumerate(groups):
        group["group_index"] = group_index
    return groups


def make_group(group_type: str, group_key: str, room_indices: list[int], rooms: list[dict], simplify_tolerance: float) -> dict:
    geoms = [rooms[i]["polygon"] for i in room_indices]
    union = simplify_polygon(clean_geometry(unary_union(geoms)), simplify_tolerance)
    minx, miny, maxx, maxy = union.bounds
    return {
        "group_index": -1,
        "group_key": group_key,
        "group_type": group_type,
        "room_indices": sorted(room_indices),
        "room_type_counts": dict(Counter(rooms[i]["room_type"] for i in room_indices)),
        "polygon": union,
        "bbox": [float(minx), float(miny), float(maxx), float(maxy)],
        "area": float(union.area),
    }


def room_edges_from_wall_graph(wall_graph: nx.Graph | None, num_rooms: int) -> list[dict]:
    if wall_graph is None or wall_graph.number_of_nodes() != num_rooms:
        return []
    edges = []
    for u, v, data in wall_graph.edges(data=True):
        edges.append(
            {
                "source": int(u),
                "target": int(v),
                "edge_type": str(data.get("connectivity", "wall")),
            }
        )
    return edges


def room_edges_from_geometry(rooms: list[dict], eps: float) -> list[dict]:
    edges = []
    for i, room_i in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            if safe_distance(room_i["polygon"], rooms[j]["polygon"]) <= eps:
                edges.append({"source": i, "target": j, "edge_type": "adjacent"})
    return edges


def build_coarse_edges(groups: list[dict], room_edges: list[dict], eps: float) -> list[dict]:
    room_to_group = {}
    for group in groups:
        for room_index in group["room_indices"]:
            room_to_group[room_index] = group["group_index"]

    edge_counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for edge in room_edges:
        group_a = room_to_group.get(edge["source"])
        group_b = room_to_group.get(edge["target"])
        if group_a is None or group_b is None or group_a == group_b:
            continue
        key = tuple(sorted((group_a, group_b)))
        edge_counts[key][edge["edge_type"]] += 1

    coarse_edges = []
    for i, group_i in enumerate(groups):
        for j in range(i + 1, len(groups)):
            group_j = groups[j]
            key = (i, j)
            distance = safe_distance(group_i["polygon"], group_j["polygon"])
            if key in edge_counts:
                edge_type = edge_counts[key].most_common(1)[0][0]
                coarse_edges.append(
                    {
                        "source": i,
                        "target": j,
                        "edge_type": edge_type,
                        "room_edge_counts": dict(edge_counts[key]),
                        "distance": distance,
                    }
                )
            elif distance <= eps:
                coarse_edges.append(
                    {
                        "source": i,
                        "target": j,
                        "edge_type": "touch",
                        "room_edge_counts": {},
                        "distance": distance,
                    }
                )
    return coarse_edges


def primitive_room(room: dict, simplify_tolerance: float) -> dict:
    polygon = simplify_polygon(room["polygon"], simplify_tolerance)
    centroid = polygon.centroid
    return {
        "room_index": room["room_index"],
        "area_id": room["area_id"],
        "unit_id": room["unit_id"],
        "room_type": room["room_type"],
        "polygon": polygons_to_coords(polygon),
        "bbox": [float(x) for x in polygon.bounds],
        "centroid": [float(centroid.x), float(centroid.y)],
        "area": float(polygon.area),
    }


def primitive_group(group: dict) -> dict:
    return {
        "group_index": group["group_index"],
        "group_key": group["group_key"],
        "group_type": group["group_type"],
        "room_indices": group["room_indices"],
        "room_type_counts": group["room_type_counts"],
        "polygon": polygons_to_coords(group["polygon"]),
        "bbox": group["bbox"],
        "area": group["area"],
    }


def split_floor_ids(floor_ids: list[str], seed: int, train_ratio: float, val_ratio: float) -> dict[str, set[str]]:
    rng = random.Random(seed)
    ids = list(floor_ids)
    rng.shuffle(ids)
    train_len = int(len(ids) * train_ratio)
    val_len = int(len(ids) * val_ratio)
    return {
        "train": set(ids[:train_len]),
        "val": set(ids[train_len : train_len + val_len]),
        "test": set(ids[train_len + val_len :]),
    }


def split_for_floor(floor_id: str, split_ids: dict[str, set[str]]) -> str:
    for split, ids in split_ids.items():
        if floor_id in ids:
            return split
    raise KeyError(f"floor_id {floor_id} missing from split ids")


def save_pickle(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def draw_floor(sample: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    for room in sample["rooms"]:
        color = ROOM_COLORS.get(room["room_type"], "#d1d5db")
        for polygon in room["polygon"]:
            patch = MplPolygon(polygon, closed=True, facecolor=color, edgecolor="#111827", linewidth=0.4, alpha=0.75)
            axes[0].add_patch(patch)
    axes[0].set_title(f"floor {sample['floor_id']} rooms")

    for group in sample["groups"]:
        color = GROUP_COLORS.get(group["group_type"], "#94a3b8")
        for polygon in group["polygon"]:
            patch = MplPolygon(polygon, closed=True, facecolor=color, edgecolor="#111827", linewidth=0.8, alpha=0.35)
            axes[1].add_patch(patch)
        minx, miny, maxx, maxy = group["bbox"]
        axes[1].text((minx + maxx) / 2, (miny + maxy) / 2, str(group["group_index"]), fontsize=7, ha="center", va="center")
    for edge in sample["coarse_edges"]:
        source = sample["groups"][edge["source"]]["bbox"]
        target = sample["groups"][edge["target"]]["bbox"]
        sx, sy = (source[0] + source[2]) / 2, (source[1] + source[3]) / 2
        tx, ty = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
        axes[1].plot([sx, tx], [sy, ty], color="#475569", linewidth=0.8, alpha=0.8)
    axes[1].set_title("coarse groups")

    for ax in axes:
        ax.autoscale()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def summarize(values: list[int | float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    out = Path(args.out)
    wall_graph_path = Path(args.wall_graphs) if args.wall_graphs else None

    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    selected_floor_ids = load_selected_floor_ids(source, args.limit_floors)
    floors = load_rooms_from_csv(source, selected_floor_ids)
    wall_graphs = load_wall_graphs(wall_graph_path)

    floor_ids = sorted(floors.keys(), key=lambda value: int(float(value)) if str(value).replace(".", "", 1).isdigit() else value)
    split_ids = split_floor_ids(floor_ids, args.split_seed, args.train_ratio, args.val_ratio)

    stats = {
        "dataset": "msd_hier_v6",
        "source": str(source),
        "wall_graphs": str(wall_graph_path) if wall_graph_path else None,
        "wall_version": "v6" if wall_graphs else None,
        "filters": {
            "min_rooms": args.min_rooms,
            "max_rooms": args.max_rooms,
            "max_corners": args.max_corners,
            "simplify_tolerance": args.simplify_tolerance,
            "adjacency_eps": args.adjacency_eps,
            "corner_merge_eps": args.corner_merge_eps,
            "split_disconnected_units": args.split_disconnected_units,
        },
        "splits": {},
        "global": {},
        "skipped": [],
    }
    measurements = defaultdict(list)
    split_counts = defaultdict(Counter)
    visualized = 0

    for floor_id in floor_ids:
        rooms = floors[floor_id]
        floor_corners = unique_corner_count([room["polygon"] for room in rooms], args.corner_merge_eps)
        if len(rooms) < args.min_rooms:
            stats["skipped"].append({"floor_id": floor_id, "reason": "min_rooms", "rooms": len(rooms)})
            continue
        if args.max_rooms is not None and len(rooms) > args.max_rooms:
            stats["skipped"].append({"floor_id": floor_id, "reason": "max_rooms", "rooms": len(rooms)})
            continue
        if args.max_corners is not None and floor_corners > args.max_corners:
            stats["skipped"].append({"floor_id": floor_id, "reason": "max_corners", "corners": floor_corners})
            continue

        wall_graph = wall_graphs.get(str(int(float(floor_id))) if str(floor_id).replace(".", "", 1).isdigit() else floor_id)
        room_edges = room_edges_from_wall_graph(wall_graph, len(rooms))
        if not room_edges:
            room_edges = room_edges_from_geometry(rooms, args.adjacency_eps)

        groups = build_groups(rooms, args.adjacency_eps, args.simplify_tolerance, args.split_disconnected_units)
        coarse_edges = build_coarse_edges(groups, room_edges, args.adjacency_eps)

        split = split_for_floor(floor_id, split_ids)
        primitive_rooms = [primitive_room(room, args.simplify_tolerance) for room in rooms]
        primitive_groups = [primitive_group(group) for group in groups]

        floor_sample = {
            "floor_id": floor_id,
            "split": split,
            "rooms": primitive_rooms,
            "room_edges": room_edges,
            "groups": primitive_groups,
            "coarse_edges": coarse_edges,
            "metadata": {
                "wall_version": "v6" if wall_graph is not None else "geometry_fallback",
                "num_rooms": len(rooms),
                "num_groups": len(groups),
                "num_room_edges": len(room_edges),
                "num_coarse_edges": len(coarse_edges),
                "num_floor_corners_raw": floor_corners,
            },
        }
        save_pickle(floor_sample, out / "floors" / split / f"{floor_id}.pkl")

        coarse_sample = {
            "floor_id": floor_id,
            "split": split,
            "nodes": primitive_groups,
            "edges": coarse_edges,
            "metadata": floor_sample["metadata"],
        }
        save_pickle(coarse_sample, out / "coarse" / split / f"{floor_id}.pkl")

        for group in groups:
            group_room_indices = set(group["room_indices"])
            local_edges = [
                edge for edge in room_edges
                if edge["source"] in group_room_indices and edge["target"] in group_room_indices
            ]
            local_sample = {
                "floor_id": floor_id,
                "split": split,
                "group": primitive_group(group),
                "rooms": [primitive_rooms[i] for i in group["room_indices"]],
                "room_edges": local_edges,
                "metadata": {
                    "wall_version": floor_sample["metadata"]["wall_version"],
                    "num_rooms": len(group["room_indices"]),
                    "num_room_edges": len(local_edges),
                    "group_type": group["group_type"],
                },
            }
            save_pickle(local_sample, out / "local" / split / f"{floor_id}_g{group['group_index']:03d}.pkl")

        split_counts[split]["floors"] += 1
        split_counts[split]["local_samples"] += len(groups)
        split_counts[split]["coarse_samples"] += 1
        measurements["rooms_per_floor"].append(len(rooms))
        measurements["groups_per_floor"].append(len(groups))
        measurements["room_edges_per_floor"].append(len(room_edges))
        measurements["coarse_edges_per_floor"].append(len(coarse_edges))
        measurements["raw_corners_per_floor"].append(floor_corners)
        measurements["rooms_per_group"].extend(len(group["room_indices"]) for group in groups)

        if visualized < args.vis_samples:
            draw_floor(floor_sample, out / "vis" / f"sample_{visualized:03d}_{floor_id}.png")
            visualized += 1

    for split, counts in split_counts.items():
        stats["splits"][split] = dict(counts)
    stats["global"] = {name: summarize(values) for name, values in measurements.items()}
    stats["num_skipped"] = len(stats["skipped"])

    with (out / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote MSD hierarchical dataset: {out}")
    print(json.dumps({"splits": stats["splits"], "global": stats["global"], "num_skipped": stats["num_skipped"]}, indent=2))


if __name__ == "__main__":
    main()
