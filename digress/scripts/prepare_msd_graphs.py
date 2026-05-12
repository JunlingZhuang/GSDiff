import argparse
import json
import pickle
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import Polygon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as MplPolygon
from shapely import wkt


NODE_CLASS_NAMES = [
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

NODE_NAMES = ["Bedroom", "Living", "Kitchen", "Dining", "Corridor", "Stairs", "Storage", "Bathroom", "Balcony"]
NODE_COLORS = ["#8da0cb", "#47b39c", "#f1c27d", "#fdae6b", "#fdd0a2", "#72246c", "#ffd92f", "#bdbdbd", "#a6d854"]
EDGE_COLORS = {
    "wall": "#334155",
    "passage": "#64748b",
    "door": "#c2410c",
    "entrance": "#d97706",
    1: "#334155",
    2: "#64748b",
    3: "#c2410c",
    4: "#d97706",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
MSD_CODE_DIR = REPO_ROOT / "datasets" / "msd"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MSD NetworkX graphs for phase-1 vanilla DiGress training."
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "MSD raw path. Supported: graph_out directory, directory containing graph_out, "
            "pickle with graph list, directory of per-floor pickles, or CSV with floor geometries."
        ),
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "data" / "msd" / "graphs.p"),
        help="Output pickle path. Default: digress/data/msd/graphs.p",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional graph limit for smoke tests.")
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument(
        "--add-wall-edges",
        action="store_true",
        help=(
            "When source is CSV, add wall edges using WALL polygons touching two rooms. "
            "This changes the edge decoder to none/wall/passage/door/entrance."
        ),
    )
    parser.add_argument(
        "--wall-contact-eps",
        type=float,
        default=0.01,
        help="Geometry tolerance for room-wall contact when --add-wall-edges is used with CSV.",
    )
    parser.add_argument(
        "--wall-room-pair-max-distance",
        type=float,
        default=0.15,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wall-min-contact-length",
        type=float,
        default=0.02,
        help="Minimum room-wall boundary contact length required for a wall edge candidate.",
    )
    parser.add_argument(
        "--wall-segment-gap",
        type=float,
        default=0.45,
        help="Maximum distance between two room-wall contact segments on the same wall (v3 default 0.45, v4 default 0.10).",
    )
    parser.add_argument(
        "--wall-version",
        choices=["v1", "v2", "v3", "v4", "v5", "v6"],
        default="v3",
        help=(
            "Wall-edge inference algorithm. "
            "v1=co-touch (no filter), "
            "v2=v1+opposite-side, "
            "v3=v2+segment-gap (current), "
            "v4=v3 with stricter segment-gap default 0.10, "
            "v5=room-centric (Direction C: distance + midline-in-wall), "
            "v6=v5 + minimum mutual-proximity boundary arc length on both rooms (filters diagonal corner false positives)."
        ),
    )
    parser.add_argument(
        "--wall-room-distance-eps",
        type=float,
        default=0.5,
        help="v5 only: max distance (in MSD units, ~m) between two rooms to consider as a wall-pair candidate.",
    )
    parser.add_argument(
        "--wall-midline-samples",
        type=int,
        default=16,
        help="v5 only: number of points sampled along the room-to-room midline for wall coverage check.",
    )
    parser.add_argument(
        "--wall-coverage-threshold",
        type=float,
        default=0.6,
        help="v5 only: min fraction of midline samples that must lie inside any WALL polygon.",
    )
    parser.add_argument(
        "--wall-touch-share-min-length",
        type=float,
        default=0.04,
        help="v5/v6 only: when two rooms touch directly, minimum shared boundary length to count (filters corner kisses).",
    )
    parser.add_argument(
        "--wall-min-shared-arc-length",
        type=float,
        default=0.5,
        help=(
            "v6 only: minimum length of each room's boundary segment that faces the other room. "
            "Filters diagonal-corner false positives where shortest_line passes through a wall intersection. "
            "Default 0.5 (~half a doorway width); typical real shared walls are 1-3."
        ),
    )
    parser.add_argument(
        "--vis-out",
        default=str(PROJECT_ROOT / "data" / "msd" / "sample_graphs.png"),
        help="Output PNG grid for checking extracted dataset graphs. Use empty string to disable.",
    )
    parser.add_argument("--num-vis", type=int, default=32)
    parser.add_argument("--vis-cols", type=int, default=4)
    parser.add_argument(
        "--visualize",
        type=int,
        default=10,
        help="Save RPLAN-style 2x3 per-sample diagnostics for first N graphs.",
    )
    parser.add_argument(
        "--vis-dir",
        default=str(PROJECT_ROOT / "data" / "msd" / "vis"),
        help="Directory for per-sample diagnostic PNGs.",
    )
    parser.add_argument(
        "--stats-out",
        default=str(PROJECT_ROOT / "data" / "msd" / "dataset_stats.json"),
        help="Output JSON dataset statistics. Use empty string to disable.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print preprocessing progress every N floors. Use 0 to disable.",
    )
    return parser.parse_args()


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def clear_processed_cache(out_path):
    processed_dir = out_path.parent / "processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed stale processed cache: {processed_dir}")


def flatten_graph_payload(payload):
    if isinstance(payload, nx.Graph):
        return [payload]
    if isinstance(payload, dict):
        if "graphs" in payload:
            return flatten_graph_payload(payload["graphs"])
        graphs = []
        for value in payload.values():
            graphs.extend(flatten_graph_payload(value))
        return graphs
    if isinstance(payload, (list, tuple)):
        graphs = []
        for item in payload:
            if isinstance(item, nx.Graph):
                graphs.append(item)
        return graphs
    return []


def load_graphs_from_pickle(path):
    return flatten_graph_payload(load_pickle(path))


def load_graphs_from_directory(path):
    graph_dir = path / "graph_out" if (path / "graph_out").is_dir() else path
    pickle_files = sorted([
        *graph_dir.glob("*.pickle"),
        *graph_dir.glob("*.pkl"),
    ])
    if not pickle_files:
        raise FileNotFoundError(f"No .pickle/.pkl graph files found in {graph_dir}")

    graphs = []
    for pickle_file in pickle_files:
        graphs.extend(load_graphs_from_pickle(pickle_file))
    return graphs


def _safe_wkt_loads(value):
    try:
        geom = wkt.loads(value)
    except Exception:
        return None
    if geom.is_empty or not geom.is_valid:
        return None
    bounds = np.asarray(geom.bounds, dtype=float)
    if not np.isfinite(bounds).all():
        return None
    return geom


def _safe_distance(geom_a, geom_b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        distance = geom_a.distance(geom_b)
    if not np.isfinite(distance):
        return None
    return float(distance)


def _touches_both(separator, room_a, room_b, eps):
    distance_a = _safe_distance(separator, room_a)
    if distance_a is None or distance_a >= eps:
        return False
    distance_b = _safe_distance(separator, room_b)
    return distance_b is not None and distance_b < eps


def _wall_axes(wall):
    coords = np.asarray(wall.exterior.coords, dtype=float)
    if coords.shape[0] < 2:
        return None, None
    centered = coords[:, :2] - coords[:, :2].mean(axis=0)
    cov = centered.T @ centered
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None, None
    tangent = eigvecs[:, int(np.argmax(eigvals))]
    norm = np.linalg.norm(tangent)
    if norm <= 1e-12:
        return None, None
    tangent = tangent / norm
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    return tangent, normal


def _contact_point(contact_geom):
    point = contact_geom.centroid
    coords = np.asarray(point.coords[0], dtype=float)
    return coords[:2] if np.isfinite(coords[:2]).all() else None


def _room_wall_contact(room, wall, eps, min_contact_length):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        contact = room.boundary.intersection(wall.buffer(eps))
    if contact.is_empty:
        return None
    length = float(contact.length)
    if not np.isfinite(length) or length < min_contact_length:
        return None
    point = _contact_point(contact)
    if point is None:
        return None
    return {
        "geometry": contact,
        "length": length,
        "point": point,
    }


def _local_wall_pairs(room_contacts, wall_segment_gap, *, apply_side_check=True, apply_gap_check=True, side_eps=1e-6):
    """Build wall-pair candidates from per-wall contacts.

    Toggleable filters control which version of the heuristic is applied:
      - apply_side_check=False, apply_gap_check=False  -> V1 (naive co-touch)
      - apply_side_check=True,  apply_gap_check=False  -> V2 (opposite-side)
      - apply_side_check=True,  apply_gap_check=True   -> V3 / V4
    """
    pairs = []
    for idx, contact_a in enumerate(room_contacts):
        for contact_b in room_contacts[idx + 1:]:
            if apply_side_check and contact_a["side"] * contact_b["side"] >= -side_eps:
                continue
            if apply_gap_check:
                distance = _safe_distance(contact_a["geometry"], contact_b["geometry"])
                if distance is None or distance > wall_segment_gap:
                    continue
            pairs.append((contact_a["room_idx"], contact_b["room_idx"]))
    return pairs


def _collect_wall_pairs_wall_centric(
    rooms,
    walls,
    wall_contact_eps,
    wall_min_contact_length,
    wall_segment_gap,
    *,
    apply_side_check,
    apply_gap_check,
):
    """V1 / V2 / V3 / V4 — pair rooms via shared WALL polygons.

    The three filter knobs encode the version:
      V1: apply_side_check=False, apply_gap_check=False
      V2: apply_side_check=True,  apply_gap_check=False
      V3: apply_side_check=True,  apply_gap_check=True   (gap=0.45)
      V4: apply_side_check=True,  apply_gap_check=True   (gap=0.10)
    """
    wall_pairs = set()
    room_centroids = [np.array([room.centroid.x, room.centroid.y], dtype=float) for room, _ in rooms]
    for wall in walls:
        contacts = []
        if apply_side_check:
            _, wall_normal = _wall_axes(wall)
            if wall_normal is None:
                continue
        else:
            wall_normal = None

        for room_idx, (room, _) in enumerate(rooms):
            contact = _room_wall_contact(room, wall, wall_contact_eps, wall_min_contact_length)
            if contact is None:
                continue
            if apply_side_check:
                side = float(np.dot(room_centroids[room_idx] - contact["point"], wall_normal))
                if not np.isfinite(side) or abs(side) < 1e-6:
                    continue
                contact["side"] = side
            else:
                contact["side"] = 0.0  # placeholder, not used when side check disabled
            contact["room_idx"] = room_idx
            contacts.append(contact)
        wall_pairs.update(_local_wall_pairs(
            contacts,
            wall_segment_gap,
            apply_side_check=apply_side_check,
            apply_gap_check=apply_gap_check,
        ))
    return wall_pairs


def _collect_wall_pairs_room_centric(
    rooms,
    walls,
    *,
    distance_eps,
    midline_samples,
    coverage_threshold,
    touch_share_min_length,
    min_shared_arc_length=0.0,
):
    """V5 / V6 — Direction C: pair rooms whose shortest connector lies inside a WALL polygon.

    Key idea: forget which WALL polygon each room touches. Instead, for every pair of rooms
    that are physically close, check whether the geometric line between them is mostly
    covered by wall material. This decouples the heuristic from how MSD draws WALL polygons.

    `min_shared_arc_length` (V6 only, V5 passes 0) additionally requires that each room
    has a boundary segment of at least this length facing the other room. This filters
    diagonal-corner false positives where two rooms meet only at a corner: the shortest
    line happens to pass through wall material at the corner intersection, but the rooms
    do not actually share a wall.
    """
    wall_pairs = set()
    if not walls:
        return wall_pairs

    from shapely.geometry import MultiPolygon
    from shapely.ops import unary_union

    # Union all walls once so point-in-wall checks reduce to a single contains() call.
    walls_union = unary_union(walls)

    for i, (room_a, _) in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            room_b, _ = rooms[j]
            distance = _safe_distance(room_a, room_b)
            if distance is None or distance > distance_eps:
                continue

            # V6 pre-filter: each room must face the other along a substantial boundary
            # segment, not just a single point. Skipped for V5 (min_shared_arc_length=0).
            if min_shared_arc_length > 0:
                buffer_radius = float(distance) + 0.05
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    try:
                        arc_a = room_a.boundary.intersection(room_b.buffer(buffer_radius))
                        arc_b = room_b.boundary.intersection(room_a.buffer(buffer_radius))
                    except Exception:
                        continue
                arc_a_length = float(arc_a.length) if hasattr(arc_a, "length") else 0.0
                arc_b_length = float(arc_b.length) if hasattr(arc_b, "length") else 0.0
                if not (np.isfinite(arc_a_length) and np.isfinite(arc_b_length)):
                    continue
                if arc_a_length < min_shared_arc_length or arc_b_length < min_shared_arc_length:
                    continue

            if distance <= 1e-9:
                # Rooms touch directly. Their shared boundary IS the candidate "wall".
                # Require minimum shared length to filter corner kisses, then check that
                # a WALL polygon overlaps the shared boundary.
                shared = room_a.boundary.intersection(room_b.boundary)
                if shared.is_empty:
                    continue
                shared_length = float(shared.length) if hasattr(shared, "length") else 0.0
                if shared_length < touch_share_min_length:
                    continue
                if walls_union.buffer(1e-6).intersects(shared):
                    wall_pairs.add((i, j))
                continue

            # Rooms are separated. Sample the shortest connecting line and ask:
            # what fraction of it lies inside any WALL polygon?
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    line = shapely.shortest_line(room_a, room_b)
                except Exception:
                    continue
            if line is None or line.is_empty:
                continue
            line_length = float(line.length)
            if not np.isfinite(line_length) or line_length <= 1e-9:
                continue
            line_coords = np.asarray(line.coords, dtype=float)
            if line_coords.size == 0 or not np.isfinite(line_coords).all():
                continue

            samples_inside = 0
            for k in range(1, midline_samples + 1):
                # Skip endpoints (which sit on room boundaries, not walls).
                t = k / (midline_samples + 1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    try:
                        point = line.interpolate(t, normalized=True)
                        inside = walls_union.contains(point) or walls_union.touches(point)
                    except Exception:
                        inside = False
                if inside:
                    samples_inside += 1
            if samples_inside / midline_samples >= coverage_threshold:
                wall_pairs.add((i, j))

    return wall_pairs


def extract_structural_graph_from_floor(
    df_floor,
    floor_id,
    room_names,
    room_mapping,
    wall_contact_eps,
    wall_min_contact_length,
    wall_segment_gap,
    *,
    wall_version="v3",
    wall_room_distance_eps=0.5,
    wall_midline_samples=16,
    wall_coverage_threshold=0.6,
    wall_touch_share_min_length=0.04,
    wall_min_shared_arc_length=0.5,
):
    mapping = {name: idx for idx, name in enumerate(room_names)}
    rooms = []
    doors = []
    entrance_doors = []
    walls = []

    for row in df_floor.itertuples(index=False):
        geom = getattr(row, "geom")
        subtype = getattr(row, "entity_subtype")
        roomtype = getattr(row, "roomtype")
        if geom is None:
            continue
        if subtype == "DOOR":
            doors.append(geom)
        elif subtype == "ENTRANCE_DOOR":
            entrance_doors.append(geom)
        elif subtype == "WALL":
            walls.append(geom)
        elif roomtype in room_names[:9]:
            rooms.append((geom, roomtype))

    graph = nx.Graph()
    graph.graph["ID"] = floor_id
    graph.graph["wall_inference_version"] = wall_version
    graph.graph["wall_contact_eps"] = wall_contact_eps
    graph.graph["wall_min_contact_length"] = wall_min_contact_length
    graph.graph["wall_segment_gap"] = wall_segment_gap
    if wall_version in {"v5", "v6"}:
        graph.graph["wall_room_distance_eps"] = wall_room_distance_eps
        graph.graph["wall_midline_samples"] = wall_midline_samples
        graph.graph["wall_coverage_threshold"] = wall_coverage_threshold
        graph.graph["wall_touch_share_min_length"] = wall_touch_share_min_length
    if wall_version == "v6":
        graph.graph["wall_min_shared_arc_length"] = wall_min_shared_arc_length
    for idx, (room, roomtype) in enumerate(rooms):
        graph.add_node(
            idx,
            geometry=list(zip(*room.exterior.coords.xy)),
            room_type=mapping[roomtype],
            centroid=np.array([room.centroid.x, room.centroid.y]),
        )

    if wall_version in {"v5", "v6"}:
        wall_pairs = _collect_wall_pairs_room_centric(
            rooms,
            walls,
            distance_eps=wall_room_distance_eps,
            midline_samples=wall_midline_samples,
            coverage_threshold=wall_coverage_threshold,
            touch_share_min_length=wall_touch_share_min_length,
            min_shared_arc_length=wall_min_shared_arc_length if wall_version == "v6" else 0.0,
        )
    else:
        apply_side_check = wall_version in {"v2", "v3", "v4"}
        apply_gap_check = wall_version in {"v3", "v4"}
        wall_pairs = _collect_wall_pairs_wall_centric(
            rooms,
            walls,
            wall_contact_eps,
            wall_min_contact_length,
            wall_segment_gap,
            apply_side_check=apply_side_check,
            apply_gap_check=apply_gap_check,
        )

    wall_edges = 0
    for i, (room_a, _) in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            room_b, _ = rooms[j]

            room_distance = _safe_distance(room_a, room_b)
            if room_distance is None:
                continue

            if room_distance < 0.04:
                graph.add_edge(i, j, connectivity="passage")
                continue

            if any(_touches_both(door, room_a, room_b, 0.05) for door in doors):
                graph.add_edge(i, j, connectivity="door")
                continue

            if any(_touches_both(entrance, room_a, room_b, 0.05) for entrance in entrance_doors):
                graph.add_edge(i, j, connectivity="entrance")
                continue

            if (i, j) in wall_pairs or (j, i) in wall_pairs:
                graph.add_edge(i, j, connectivity="wall")
                wall_edges += 1

    graph.graph["wall_edges_added"] = wall_edges
    return graph


def load_graphs_from_csv(
    path,
    add_wall_edges=False,
    wall_contact_eps=0.01,
    wall_room_pair_max_distance=0.15,
    wall_min_contact_length=0.02,
    wall_segment_gap=0.45,
    wall_version="v3",
    wall_room_distance_eps=0.5,
    wall_midline_samples=16,
    wall_coverage_threshold=0.6,
    wall_touch_share_min_length=0.04,
    wall_min_shared_arc_length=0.5,
    limit=None,
    min_nodes=2,
    max_nodes=None,
    progress_every=100,
):
    if str(MSD_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(MSD_CODE_DIR))

    try:
        import pandas as pd
        from constants import ROOM_MAPPING, ROOM_NAMES
        from graphs import extract_access_graph, get_geometries_from_id
    except ImportError as exc:
        raise ImportError(
            "CSV extraction requires pandas and shapely-compatible MSD code dependencies. "
            "Install them in digress/.venv or use Kaggle graph_out pickle files instead."
        ) from exc

    df = pd.read_csv(path)
    if "roomtype" not in df.columns:
        if "entity_subtype" not in df.columns:
            raise KeyError("CSV needs either 'roomtype' or 'entity_subtype' column")
        df["roomtype"] = df["entity_subtype"].map(ROOM_MAPPING)
    room_names = ROOM_NAMES[:9]
    if add_wall_edges:
        required = {"floor_id", "entity_subtype", "roomtype", "geom"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"CSV wall extraction missing columns: {sorted(missing)}")
        df = df[df["entity_subtype"].isin([
            "WALL",
            "DOOR",
            "ENTRANCE_DOOR",
            *ROOM_MAPPING.keys(),
        ])].copy()

    graphs = []
    floor_iter = df.groupby("floor_id", sort=True) if add_wall_edges else [
        (floor_id, None) for floor_id in sorted(df.floor_id.unique().tolist())
    ]
    total_floors = int(df.floor_id.nunique())
    processed_floors = 0
    skipped_floors = 0
    for floor_id, grouped_floor_df in floor_iter:
        processed_floors += 1
        try:
            if add_wall_edges:
                floor_df = grouped_floor_df.copy()
                room_count = int(floor_df["roomtype"].isin(room_names).sum())
                if room_count < min_nodes:
                    skipped_floors += 1
                    continue
                if max_nodes is not None and room_count > max_nodes:
                    skipped_floors += 1
                    continue
                floor_df["geom"] = floor_df["geom"].apply(_safe_wkt_loads)
                floor_df = floor_df[floor_df["geom"].notna()]
                graph = extract_structural_graph_from_floor(
                    floor_df,
                    floor_id=floor_id,
                    room_names=ROOM_NAMES,
                    room_mapping=ROOM_MAPPING,
                    wall_contact_eps=wall_contact_eps,
                    wall_min_contact_length=wall_min_contact_length,
                    wall_segment_gap=wall_segment_gap,
                    wall_version=wall_version,
                    wall_room_distance_eps=wall_room_distance_eps,
                    wall_midline_samples=wall_midline_samples,
                    wall_coverage_threshold=wall_coverage_threshold,
                    wall_touch_share_min_length=wall_touch_share_min_length,
                    wall_min_shared_arc_length=wall_min_shared_arc_length,
                )
            else:
                geoms, room_types = get_geometries_from_id(df, floor_id, column="roomtype")
                graph = extract_access_graph(geoms, room_types, ROOM_NAMES, floor_id)
        except Exception as exc:
            print(f"Skipping floor_id={floor_id}: {exc}")
            skipped_floors += 1
            continue
        n_nodes = graph.number_of_nodes()
        if n_nodes < min_nodes:
            skipped_floors += 1
            continue
        if max_nodes is not None and n_nodes > max_nodes:
            skipped_floors += 1
            continue
        graphs.append(graph)
        if progress_every > 0 and (
            processed_floors % progress_every == 0
            or (limit is not None and len(graphs) >= limit)
            or processed_floors == total_floors
        ):
            print(
                "Preprocessing MSD floors: "
                f"{processed_floors}/{total_floors} scanned, "
                f"{len(graphs)} kept, {skipped_floors} skipped",
                flush=True,
            )
        if limit is not None and len(graphs) >= limit:
            break
    if progress_every > 0 and processed_floors % progress_every != 0:
        print(
            "Preprocessing MSD floors: "
            f"{processed_floors}/{total_floors} scanned, "
            f"{len(graphs)} kept, {skipped_floors} skipped",
            flush=True,
        )
    return graphs


def normalize_graph(graph):
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    graph.graph.pop("image", None)
    return graph


def filter_graphs(graphs, min_nodes, max_nodes, limit):
    filtered = []
    for graph in graphs:
        graph = normalize_graph(graph)
        n_nodes = graph.number_of_nodes()
        if n_nodes < min_nodes:
            continue
        if max_nodes is not None and n_nodes > max_nodes:
            continue
        filtered.append(graph)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def _node_attr(graph, node):
    data = graph.nodes[node]
    value = data.get("room_type", data.get("attr", data.get("roomtype", 0)))
    if hasattr(value, "item"):
        value = value.item()
    return int(value) if not isinstance(value, str) else NODE_CLASS_NAMES.index(value)


def _edge_attr(data):
    value = data.get("connectivity", data.get("edge_type", "passage"))
    if hasattr(value, "item"):
        value = value.item()
    return value


def _edge_name(value):
    if isinstance(value, str):
        return value
    names = {1: "wall", 2: "passage", 3: "door", 4: "entrance"}
    return names.get(int(value), str(value))


def _geometry(graph, node):
    geometry = graph.nodes[node].get("geometry")
    if geometry is None:
        return None
    return np.asarray(geometry, dtype=float)


def _polygon_area(points):
    if points is None or len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _node_area(graph, node):
    area = graph.nodes[node].get("area")
    if area is not None:
        if hasattr(area, "item"):
            area = area.item()
        return float(area)
    return _polygon_area(_geometry(graph, node))


def _series_stats(values):
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p25": None,
            "p75": None,
            "sum": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "std": round(float(arr.std()), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "sum": round(float(arr.sum()), 6),
    }


def _count_table(counter, ordered_keys):
    total = sum(counter.values())
    table = {}
    for key in ordered_keys:
        count = int(counter.get(key, 0))
        table[str(key)] = {
            "count": count,
            "proportion": round(count / total, 8) if total else 0.0,
        }
    return table


def _edge_names_from_graphs(graphs):
    names = sorted({
        _edge_name(_edge_attr(data))
        for graph in graphs
        for _, _, data in graph.edges(data=True)
    })
    preferred_order = ["wall", "passage", "door", "entrance"]
    return [name for name in preferred_order if name in names] + [
        name for name in names if name not in preferred_order
    ]


def build_dataset_stats(graphs, source, filters):
    node_counts = []
    edge_counts = []
    avg_degrees = []
    edge_densities = []
    total_areas = []
    room_counts = Counter()
    edge_type_counts = Counter()
    dense_edge_counts = Counter()
    node_count_histogram = Counter()
    edge_count_histogram = Counter()
    area_by_room = defaultdict(list)
    all_room_areas = []

    for graph in graphs:
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        node_counts.append(n_nodes)
        edge_counts.append(n_edges)
        node_count_histogram[n_nodes] += 1
        edge_count_histogram[n_edges] += 1
        avg_degrees.append((2.0 * n_edges / n_nodes) if n_nodes else 0.0)

        possible_edges = n_nodes * (n_nodes - 1) // 2
        non_edges = max(0, possible_edges - n_edges)
        dense_edge_counts["none"] += non_edges
        edge_densities.append(n_edges / possible_edges if possible_edges else 0.0)

        graph_area = 0.0
        for node in graph.nodes():
            room_type = _node_attr(graph, node)
            room_name = NODE_CLASS_NAMES[room_type]
            area = _node_area(graph, node)
            room_counts[room_name] += 1
            area_by_room[room_name].append(area)
            all_room_areas.append(area)
            graph_area += area
        total_areas.append(graph_area)

        for _, _, data in graph.edges(data=True):
            edge_name = _edge_name(_edge_attr(data))
            edge_type_counts[edge_name] += 1
            dense_edge_counts[edge_name] += 1

    room_names = NODE_CLASS_NAMES
    edge_names = _edge_names_from_graphs(graphs)
    dense_edge_names = ["none", *edge_names]
    return {
        "dataset": "msd",
        "source": str(source),
        "filters": filters,
        "graph_count": len(graphs),
        "graph_level": {
            "nodes_per_graph": _series_stats(node_counts),
            "edges_per_graph": _series_stats(edge_counts),
            "avg_degree": _series_stats(avg_degrees),
            "edge_density": _series_stats(edge_densities),
            "total_room_area_per_graph": _series_stats(total_areas),
        },
        "room_types": _count_table(room_counts, room_names),
        "edge_types_present_edges_only": _count_table(edge_type_counts, edge_names),
        "edge_types_dense_adjacency_including_none": _count_table(dense_edge_counts, dense_edge_names),
        "room_area_all_nodes": _series_stats(all_room_areas),
        "room_area_by_type": {
            room_name: _series_stats(area_by_room.get(room_name, []))
            for room_name in room_names
        },
        "node_count_histogram": {str(key): int(value) for key, value in sorted(node_count_histogram.items())},
        "edge_count_histogram": {str(key): int(value) for key, value in sorted(edge_count_histogram.items())},
    }


def save_dataset_stats(graphs, path, source, filters):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = build_dataset_stats(graphs, source=source, filters=filters)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved dataset statistics to {path}")
    return stats


def _position(graph):
    positions = {}
    for node, data in graph.nodes(data=True):
        centroid = data.get("centroid")
        if centroid is None:
            geometry = _geometry(graph, node)
            if geometry is None or len(geometry) == 0:
                return nx.spring_layout(graph, seed=0, k=0.9, iterations=80)
            centroid = geometry.mean(axis=0)
        if hasattr(centroid, "detach"):
            centroid = centroid.detach().cpu().numpy()
        positions[node] = np.asarray(centroid, dtype=float)
    return positions


def _set_geometry_limits(ax, graph):
    coords = []
    for node in graph.nodes():
        geometry = _geometry(graph, node)
        if geometry is not None and len(geometry) > 0:
            coords.append(geometry)
    if not coords:
        return False
    all_coords = np.concatenate(coords, axis=0)
    min_xy = all_coords.min(axis=0)
    max_xy = all_coords.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    pad = span.max() * 0.08
    ax.set_xlim(min_xy[0] - pad, max_xy[0] + pad)
    ax.set_ylim(max_xy[1] + pad, min_xy[1] - pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return True


def draw_floor_shapes(ax, graph, title, alpha=0.95):
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    for node in graph.nodes():
        geometry = _geometry(graph, node)
        if geometry is None or len(geometry) < 3:
            continue
        color = NODE_COLORS[_node_attr(graph, node) % len(NODE_COLORS)]
        ax.add_patch(MplPolygon(geometry, closed=True, facecolor=color, edgecolor="black", linewidth=0.5, alpha=alpha))
    if not _set_geometry_limits(ax, graph):
        draw_graph(ax, graph, title)


def draw_access_edges(ax, graph, title):
    draw_floor_shapes(ax, graph, title, alpha=0.22)
    pos = _position(graph)
    for u, v, data in graph.edges(data=True):
        edge_type = _edge_attr(data)
        src = pos[u]
        dst = pos[v]
        color = EDGE_COLORS.get(edge_type, "#64748b")
        style = ":" if edge_type in {"wall", 1} else "--" if edge_type in {"door", 3} else "-." if edge_type in {"entrance", 4} else "-"
        ax.plot([src[0], dst[0]], [src[1], dst[1]], color=color, linewidth=2.0, linestyle=style)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=60,
        node_color="black",
    )


def draw_graph(ax, graph, title):
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    if graph.number_of_nodes() == 0:
        return

    pos = _position(graph)
    node_colors = [NODE_COLORS[_node_attr(graph, node) % len(NODE_COLORS)] for node in graph.nodes()]
    edge_colors = [EDGE_COLORS.get(_edge_attr(data), "#64748b") for _, _, data in graph.edges(data=True)]
    labels = {node: NODE_NAMES[_node_attr(graph, node) % len(NODE_NAMES)] for node in graph.nodes()}
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=420,
        edgecolors="black",
        linewidths=0.6,
    )
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=1.2)
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=5)


def draw_bubble(ax, graph, title, background=False, show_area=False, show_edge_labels=False):
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if background:
        draw_floor_shapes(ax, graph, title, alpha=0.20)
    else:
        _set_geometry_limits(ax, graph)

    pos = _position(graph)
    for u, v, data in graph.edges(data=True):
        edge_type = _edge_attr(data)
        src = pos[u]
        dst = pos[v]
        color = EDGE_COLORS.get(edge_type, "#64748b")
        style = ":" if edge_type in {"wall", 1} else "--" if edge_type in {"passage", 2} else "-"
        ax.plot([src[0], dst[0]], [src[1], dst[1]], color=color, linewidth=2.2, linestyle=style, alpha=0.9)
        if show_edge_labels:
            mid = (src + dst) / 2
            label = str(edge_type)
            if edge_type == 1:
                label = "wall"
            elif edge_type == 2:
                label = "passage"
            elif edge_type == 3:
                label = "door"
            elif edge_type == 4:
                label = "entrance"
            ax.text(
                mid[0],
                mid[1],
                label,
                fontsize=6,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
            )

    for node in graph.nodes():
        point = pos[node]
        attr = _node_attr(graph, node)
        geometry = _geometry(graph, node)
        area = graph.nodes[node].get("area", _polygon_area(geometry))
        radius = 0.025
        if geometry is not None:
            coords = np.concatenate([_geometry(graph, n) for n in graph.nodes() if _geometry(graph, n) is not None])
            radius = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1])) * 0.035
        circle = plt.Circle(
            (point[0], point[1]),
            radius,
            facecolor=NODE_COLORS[attr % len(NODE_COLORS)],
            edgecolor="black",
            linewidth=1.0,
            alpha=0.92,
            zorder=3,
        )
        ax.add_patch(circle)
        label = NODE_NAMES[attr % len(NODE_NAMES)]
        if show_area:
            label = f"{label[:4]}\n{area:.2f}"
        ax.text(point[0], point[1], label, fontsize=6, ha="center", va="center", weight="bold", zorder=4)

    if not _set_geometry_limits(ax, graph):
        ax.relim()
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")


def visualize_sample(graph, out_path, index):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    draw_floor_shapes(axes[0, 0], graph, "MSD room geometry")
    draw_access_edges(axes[0, 1], graph, "Access edges on geometry")
    draw_bubble(axes[0, 2], graph, "Bubble diagram on geometry", background=True)
    draw_bubble(axes[1, 0], graph, "Bubble diagram (with area)", show_area=True)
    draw_bubble(axes[1, 1], graph, "Bubble diagram (edge labels)", show_edge_labels=True)
    draw_graph(axes[1, 2], graph, "Abstract graph (full labels)")

    room_legend = [
        Patch(facecolor=NODE_COLORS[i], label=NODE_NAMES[i])
        for i in range(len(NODE_NAMES))
    ]
    edge_legend = [
        plt.Line2D([0], [0], color=EDGE_COLORS["wall"], linewidth=2, linestyle=":", label="wall"),
        plt.Line2D([0], [0], color=EDGE_COLORS["passage"], linewidth=2, label="passage"),
        plt.Line2D([0], [0], color=EDGE_COLORS["door"], linewidth=2, linestyle="--", label="door"),
        plt.Line2D([0], [0], color=EDGE_COLORS["entrance"], linewidth=2, linestyle="-.", label="entrance"),
    ]
    fig.legend(
        handles=room_legend + edge_legend,
        loc="lower center",
        ncol=7,
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )
    graph_id = graph.graph.get("ID", index)
    edge_counts = Counter(_edge_name(_edge_attr(data)) for _, _, data in graph.edges(data=True))
    edge_summary = ", ".join(
        f"{name}={edge_counts.get(name, 0)}"
        for name in ["wall", "passage", "door", "entrance"]
        if name in edge_counts or name == "wall"
    )
    wall_contact_eps = graph.graph.get("wall_contact_eps")
    wall_min_contact_length = graph.graph.get("wall_min_contact_length")
    wall_segment_gap = graph.graph.get("wall_segment_gap")
    params = []
    if wall_contact_eps is not None:
        params.append(f"wall_contact_eps={wall_contact_eps}")
    if wall_min_contact_length is not None:
        params.append(f"wall_min_contact_length={wall_min_contact_length}")
    if wall_segment_gap is not None:
        params.append(f"wall_segment_gap={wall_segment_gap}")
    subtitle = " | ".join(params)
    if subtitle:
        subtitle = f"\n{subtitle}"
    plt.suptitle(
        (
            f"MSD graph {graph_id} "
            f"({graph.number_of_nodes()} rooms, {graph.number_of_edges()} edges; {edge_summary})"
            f"{subtitle}"
        ),
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_sample_visualizations(graphs, out_dir, count):
    if count <= 0:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, graph in enumerate(graphs[:count]):
        graph_id = graph.graph.get("ID", idx)
        out_path = out_dir / f"sample_{idx:03d}_{graph_id}.png"
        visualize_sample(graph, out_path, idx)
    print(f"Saved {min(count, len(graphs))} per-sample visualizations to {out_dir}")


def save_graph_grid(graphs, path, num_vis, cols):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graphs = graphs[:num_vis]
    if not graphs:
        return
    cols = max(1, cols)
    rows = int(np.ceil(len(graphs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for idx, graph in enumerate(graphs):
        draw_graph(axes[idx], graph, f"MSD {idx}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved sample visualization to {path}")


def main():
    args = parse_args()
    source = Path(args.source)
    if not source.is_absolute():
        source = (PROJECT_ROOT / source).resolve()

    # V4 = V3 algorithm with stricter default segment gap. If the user accepted
    # the V3 default (0.45) but selected V4, auto-tighten to 0.10 to honour the
    # V4 contract. Explicit user-set values are kept as-is.
    if args.wall_version == "v4" and args.wall_segment_gap == 0.45:
        args.wall_segment_gap = 0.10

    out = Path(args.out)
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()

    # If the user did not override --out, route non-default versions to a
    # version-suffixed directory so V3 artifacts are not overwritten by V1/V2/V4/V5
    # runs and side-by-side comparison stays possible.
    default_out = (PROJECT_ROOT / "data" / "msd" / "graphs.p").resolve()
    if args.add_wall_edges and out == default_out:
        if args.wall_version == "v3":
            out = (PROJECT_ROOT / "data" / "msd_wall" / "graphs.p").resolve()
        else:
            out = (PROJECT_ROOT / "data" / f"msd_wall_{args.wall_version}" / "graphs.p").resolve()

    if args.add_wall_edges and source.suffix.lower() != ".csv":
        raise ValueError("--add-wall-edges requires CSV source so WALL polygons are available.")

    if source.is_dir():
        graphs = load_graphs_from_directory(source)
    elif source.suffix.lower() in {".pickle", ".pkl"}:
        graphs = load_graphs_from_pickle(source)
    elif source.suffix.lower() == ".csv":
        graphs = load_graphs_from_csv(
            source,
            add_wall_edges=args.add_wall_edges,
            wall_contact_eps=args.wall_contact_eps,
            wall_room_pair_max_distance=args.wall_room_pair_max_distance,
            wall_min_contact_length=args.wall_min_contact_length,
            wall_segment_gap=args.wall_segment_gap,
            wall_version=args.wall_version,
            wall_room_distance_eps=args.wall_room_distance_eps,
            wall_midline_samples=args.wall_midline_samples,
            wall_coverage_threshold=args.wall_coverage_threshold,
            wall_touch_share_min_length=args.wall_touch_share_min_length,
            wall_min_shared_arc_length=args.wall_min_shared_arc_length,
            limit=args.limit,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            progress_every=args.progress_every,
        )
    else:
        raise ValueError(f"Unsupported MSD source: {source}")

    graphs = filter_graphs(
        graphs,
        args.min_nodes,
        args.max_nodes,
        None if source.suffix.lower() == ".csv" else args.limit,
    )
    if not graphs:
        raise ValueError("No MSD graphs left after filtering")

    save_pickle(graphs, out)
    clear_processed_cache(out)

    # If user accepted the per-arg defaults but the wall-version routed us to a
    # version-suffixed output dir, also redirect vis/stats outputs into that dir
    # so artifacts stay grouped by version.
    default_vis_out = (PROJECT_ROOT / "data" / "msd" / "sample_graphs.png").resolve()
    default_vis_dir = (PROJECT_ROOT / "data" / "msd" / "vis").resolve()
    default_stats_out = (PROJECT_ROOT / "data" / "msd" / "dataset_stats.json").resolve()

    def _route_to_out_dir(user_path, default_path, filename):
        path = Path(user_path)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        else:
            path = path.resolve()
        if path == default_path:
            return out.parent / filename
        return path

    if args.vis_out:
        vis_out = _route_to_out_dir(args.vis_out, default_vis_out, "sample_graphs.png")
        save_graph_grid(graphs, vis_out, args.num_vis, args.vis_cols)
    if args.visualize > 0:
        vis_dir = _route_to_out_dir(args.vis_dir, default_vis_dir, "vis")
        save_sample_visualizations(graphs, vis_dir, args.visualize)
    stats = None
    if args.stats_out:
        stats_out = _route_to_out_dir(args.stats_out, default_stats_out, "dataset_stats.json")
        stats = save_dataset_stats(
            graphs,
            stats_out,
            source=source,
            filters={
                "min_nodes": args.min_nodes,
                "max_nodes": args.max_nodes,
                "limit": args.limit,
                "add_wall_edges": args.add_wall_edges,
                "wall_version": args.wall_version,
                "wall_contact_eps": args.wall_contact_eps,
                "wall_min_contact_length": args.wall_min_contact_length,
                "wall_segment_gap": args.wall_segment_gap,
                "wall_room_distance_eps": args.wall_room_distance_eps,
                "wall_midline_samples": args.wall_midline_samples,
                "wall_coverage_threshold": args.wall_coverage_threshold,
                "wall_touch_share_min_length": args.wall_touch_share_min_length,
                "wall_min_shared_arc_length": args.wall_min_shared_arc_length,
            },
        )
    sizes = [graph.number_of_nodes() for graph in graphs]
    edges = [graph.number_of_edges() for graph in graphs]
    print(f"Saved {len(graphs)} MSD graphs to {out}")
    if args.add_wall_edges:
        print(
            f"Wall inference: version={args.wall_version}, "
            f"segment_gap={args.wall_segment_gap}, "
            f"contact_eps={args.wall_contact_eps}, "
            f"min_contact_length={args.wall_min_contact_length}"
        )
        if args.wall_version in {"v5", "v6"}:
            print(
                f"  {args.wall_version} params: room_distance_eps={args.wall_room_distance_eps}, "
                f"midline_samples={args.wall_midline_samples}, "
                f"coverage_threshold={args.wall_coverage_threshold}"
            )
            if args.wall_version == "v6":
                print(f"  v6 extra: min_shared_arc_length={args.wall_min_shared_arc_length}")
    print(f"Nodes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes) / len(sizes):.2f}")
    print(f"Edges: min={min(edges)}, max={max(edges)}, avg={sum(edges) / len(edges):.2f}")
    if stats is not None:
        area_stats = stats["graph_level"]["total_room_area_per_graph"]
        print(
            "Area per graph: "
            f"min={area_stats['min']}, max={area_stats['max']}, avg={area_stats['mean']}"
        )


if __name__ == "__main__":
    main()
