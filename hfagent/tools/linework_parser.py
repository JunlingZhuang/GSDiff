# -*- coding: utf-8 -*-
"""OCR + vector wall-graph parser for generated architectural line plans."""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from hfagent.schema.plan import Plan, Room
from hfagent.schema.roomgraph import Door, RoomGraph, RoomNode
from hfagent.tools.wall_graph import OpeningCandidate, WallGraphResult, reconstruct_wall_graph


@dataclass
class OCRText:
    text: str
    box: list[list[float]]
    confidence: float
    center: tuple[float, float]
    room_type: str | None = None
    match_score: float = 0.0


@lru_cache(maxsize=1)
def _easyocr_reader():
    try:
        import easyocr
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "structure_mode=linework requires EasyOCR; install hfagent requirements"
        ) from exc
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _decode(image: bytes | str | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, bytes):
        array = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(image, str):
        array = cv2.imread(image, cv2.IMREAD_COLOR)
    elif isinstance(image, Image.Image):
        array = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        array = np.asarray(image, dtype=np.uint8).copy()
    if array is None or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected a decodable RGB floor-plan image")
    return array


def _normalise_text(text: str) -> str:
    value = text.lower().strip().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    return re.sub(r"_+", "_", value).strip("_")


def _match_type(text: str, room_types: list[str]) -> tuple[str | None, float]:
    value = re.sub(r"_?\d+$", "", _normalise_text(text))
    if not value:
        return None, 0.0
    score, room_type = max(
        (SequenceMatcher(None, value, room_type).ratio(), room_type)
        for room_type in room_types
    )
    return (room_type, score) if score >= 0.52 else (None, score)


def _ocr(image: np.ndarray, room_types: list[str], reader=None) -> tuple[list[OCRText], list[OCRText]]:
    backend = reader or _easyocr_reader()
    raw_result = backend.readtext(
        image,
        detail=1,
        paragraph=False,
        mag_ratio=3.0,
        canvas_size=4096,
        text_threshold=0.45,
        low_text=0.25,
        link_threshold=0.45,
        width_ths=0.10,
        height_ths=0.30,
        ycenter_ths=0.30,
        add_margin=0.02,
    )
    all_text: list[OCRText] = []
    matched: list[OCRText] = []
    for box, text, confidence in raw_result:
        points = [[float(x), float(y)] for x, y in box]
        item = OCRText(
            text=str(text),
            box=points,
            confidence=float(confidence),
            center=(sum(point[0] for point in points) / 4, sum(point[1] for point in points) / 4),
        )
        item.room_type, item.match_score = _match_type(item.text, room_types)
        all_text.append(item)
        if item.room_type is not None:
            matched.append(item)
    return all_text, matched


def _wall_protect_mask(image: np.ndarray) -> np.ndarray:
    """Boolean mask of axis-aligned wall pixels (long H/V strokes).

    A label printed inside a tight room (e.g. a ~24 px toilet) overlaps that room's
    short partition walls. Whiting out the whole OCR box therefore punches holes in
    those walls, breaking them into sub-threshold fragments that never become walls.
    Walls run far beyond a glyph, so a directional open keeps the through-wall and
    drops the glyph — we protect exactly those pixels during erasure.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    span = max(20, min(image.shape[:2]) // 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (span, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, span))
    walls = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel) | cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
    return walls > 0


def _erase_text(image: np.ndarray, detections: list[OCRText]) -> np.ndarray:
    """Remove OCR glyphs before line extraction, preserving walls under the boxes."""
    cleaned = image.copy()
    height, width = image.shape[:2]
    protect = _wall_protect_mask(image)
    for item in detections:
        points = np.asarray(item.box, dtype=np.float64)
        x0, y0 = np.floor(points.min(axis=0) - 1).astype(int)
        x1, y1 = np.ceil(points.max(axis=0) + 1).astype(int)
        x0, x1 = max(0, x0), min(width - 1, x1)
        y0, y1 = max(0, y0), min(height - 1, y1)
        # Large OCR boxes are usually false positives on wall geometry.
        if x1 - x0 > width * 0.20 or y1 - y0 > height * 0.08:
            continue
        # white out only the glyph pixels; keep walls that run through the box
        region = cleaned[y0:y1 + 1, x0:x1 + 1]
        region[~protect[y0:y1 + 1, x0:x1 + 1]] = (255, 255, 255)
    return cleaned


def _white_fraction(gray: np.ndarray, polygon: Polygon) -> float:
    min_x, min_y, max_x, max_y = polygon.bounds
    x0, y0 = max(0, int(min_x)), max(0, int(min_y))
    x1, y1 = min(gray.shape[1], int(max_x) + 1), min(gray.shape[0], int(max_y) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    points = np.asarray(
        [(round(x - x0), round(y - y0)) for x, y in polygon.exterior.coords],
        np.int32,
    )
    cv2.fillPoly(mask, [points], 1)
    pixels = gray[y0:y1, x0:x1][mask > 0]
    return float((pixels > 200).mean()) if pixels.size else 0.0


def _polygon_texts(polygon: Polygon, texts: list[OCRText], tolerance: float) -> list[OCRText]:
    region = polygon.buffer(tolerance)
    return [item for item in texts if region.covers(Point(item.center))]


def _best_label(items: list[OCRText]) -> OCRText | None:
    if not items:
        return None
    return max(items, key=lambda item: item.match_score * max(item.confidence, 0.25))


def _opening_rooms(
    opening: OpeningCandidate,
    polygons: list[Polygon],
    polygon_room: dict[int, Room],
    wall_width: int,
) -> tuple[str, str] | None:
    center_x, center_y = opening.center
    offset = max(2.0, wall_width * 1.5)
    if opening.orientation == "horizontal":
        probes = (Point(center_x, center_y - offset), Point(center_x, center_y + offset))
    else:
        probes = (Point(center_x - offset, center_y), Point(center_x + offset, center_y))

    room_ids: list[str] = []
    for probe in probes:
        found = next(
            (polygon_room[index].id for index, polygon in enumerate(polygons) if polygon.buffer(0.5).covers(probe)),
            "",
        )
        room_ids.append(found)
    if not room_ids[0] or not room_ids[1] or room_ids[0] == room_ids[1]:
        return None
    return room_ids[0], room_ids[1]


def _debug_image(
    image: np.ndarray,
    wall_graph: WallGraphResult,
    polygons: list[Polygon],
    rooms: list[Room],
    matched_text: list[OCRText],
    door_openings: list[OpeningCandidate],
) -> bytes:
    output = image.copy()
    for segment in wall_graph.segments:
        if segment.orientation == "horizontal":
            start = (round(segment.start), round(segment.axis))
            end = (round(segment.end), round(segment.axis))
        else:
            start = (round(segment.axis), round(segment.start))
            end = (round(segment.axis), round(segment.end))
        cv2.line(output, start, end, (0, 180, 0), 1)
    for polygon, room in zip(polygons, rooms):
        points = np.asarray(polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [points], True, (255, 0, 0), 2)
        centroid = polygon.representative_point()
        cv2.putText(
            output,
            f"{room.id}:{room.type}",
            (round(centroid.x), round(centroid.y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 0, 180),
            1,
        )
    for item in matched_text:
        points = np.asarray(item.box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [points], True, (255, 120, 0), 1)
    for opening in door_openings:
        cv2.circle(output, tuple(map(round, opening.center)), 5, (0, 0, 255), 2)
    success, encoded = cv2.imencode(".png", output)
    return encoded.tobytes() if success else b""


def parse_linework(image, program: dict, ocr_reader=None) -> tuple[Plan, RoomGraph, dict, bytes]:
    """Return ``(Plan, RoomGraph, diagnostics, debug_png)`` from a real plan."""
    source = _decode(image)
    room_types = [room["type"] for room in program["rooms"]]
    all_text, matched_text = _ocr(source, room_types, reader=ocr_reader)
    cleaned = _erase_text(source, all_text)
    cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    wall_graph = reconstruct_wall_graph(cleaned_gray)

    kept_polygons: list[Polygon] = []
    rooms: list[Room] = []
    labelled = 0
    for polygon in wall_graph.polygons:
        hits = _polygon_texts(polygon, matched_text, tolerance=wall_graph.wall_width * 0.5)
        best = _best_label(hits)
        white_fraction = _white_fraction(cleaned_gray, polygon)
        # Center-line polygonization occasionally encloses a wall/symbol cell. A
        # real room is mostly white, or has a valid OCR room label inside it.
        if best is None and white_fraction < 0.72:
            continue
        room_type = best.room_type if best is not None else "unknown"
        name = _normalise_text(best.text) if best is not None else None
        if best is not None:
            labelled += 1
        coordinates = [(float(x), float(y)) for x, y in list(polygon.exterior.coords)[:-1]]
        if len(coordinates) < 3:
            continue
        room = Room(id=f"r{len(rooms) + 1}", type=room_type, name=name, polygon=coordinates)
        kept_polygons.append(polygon)
        rooms.append(room)

    polygon_room = {index: room for index, room in enumerate(rooms)}
    pairs: set[tuple[str, str]] = set()
    used_openings: list[OpeningCandidate] = []
    for opening in wall_graph.openings:
        if opening.kind != "door":
            continue
        pair = _opening_rooms(opening, kept_polygons, polygon_room, wall_graph.wall_width)
        if pair is None:
            continue
        key = tuple(sorted(pair))
        if key in pairs:
            continue
        pairs.add(key)
        used_openings.append(opening)

    graph = RoomGraph(
        rooms=[RoomNode(id=room.id, type=room.type) for room in rooms],
        doors=[Door(room_a=room_a, room_b=room_b) for room_a, room_b in sorted(pairs)],
    )
    expected_rooms = sum(int(room.get("count", 1)) for room in program["rooms"])
    label_rate = labelled / len(rooms) if rooms else 0.0
    region_coverage = len(rooms) / expected_rooms if expected_rooms else 0.0
    diagnostics = {
        "geometry_backend": "vector_wall_graph",
        "wall_width_px": wall_graph.wall_width,
        "wall_segments": len(wall_graph.segments),
        "polygonized_regions": len(wall_graph.polygons),
        "room_regions": len(rooms),
        "labelled_rooms": labelled,
        "unlabelled_rooms": len(rooms) - labelled,
        "label_rate": round(label_rate, 4),
        "region_coverage": round(region_coverage, 4),
        "opening_candidates": {
            "doors": sum(opening.kind == "door" for opening in wall_graph.openings),
            "windows": sum(opening.kind == "window" for opening in wall_graph.openings),
            "mapped_doors": len(pairs),
        },
        "door_candidates": [
            {
                "center": [round(opening.center[0], 1), round(opening.center[1], 1)],
                "orientation": opening.orientation,
                "width_px": round(opening.end - opening.start, 1),
            }
            for opening in used_openings
        ],
        "ocr_matches": [
            {
                "text": item.text,
                "type": item.room_type,
                "confidence": round(item.confidence, 4),
                "match_score": round(item.match_score, 4),
                "center": [round(item.center[0], 1), round(item.center[1], 1)],
            }
            for item in matched_text
        ],
        "parser_confident": label_rate >= 0.80 and region_coverage >= 0.65,
    }
    plan = Plan(units="px", rooms=rooms)
    debug = _debug_image(source, wall_graph, kept_polygons, rooms, matched_text, used_openings)
    return plan, graph, diagnostics, debug
