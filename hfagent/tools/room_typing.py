# -*- coding: utf-8 -*-
"""Room typing for the linework mode: read the drawn room labels, return types.

The tracer stays geometry-only and emits untyped rooms (r1..rN); this tool runs
AFTER it and assigns each room a program type by reading the label drawn inside
it. It is a CLOSED-VOCABULARY problem, not open OCR: the program already names
every legal room type, so the recognised string only has to pick its nearest
vocabulary entry — "patlent_room_12" and "tollet_3" still land on the right
type through edit-distance matching, which is what makes a small local CRNN
sufficient.

Models (both optional, in ``data/models/``, graceful no-op when absent):
- text detection: shared ``text_mask.detect_label_quads`` (PP-OCRv3 DB);
- text recognition: ``text_recognition_CRNN_EN.onnx`` (opencv_zoo CRNN, 36-char
  lowercase+digit charset — underscores in labels simply vanish, which the
  vocabulary match absorbs).
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon

from hfagent.tools.text_mask import detect_label_quads

RECOGNITION_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "models" / "text_recognition_CRNN_EN.onnx"

_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
_INPUT_SIZE = (100, 32)
_MIN_MATCH_SCORE = 0.6


@dataclass
class RoomTypeGuess:
    """One room's label reading: base type + instance + evidence."""
    type: str
    instance: int | None
    score: float
    text: str

    @property
    def name(self) -> str:
        return f"{self.type}_{self.instance}" if self.instance is not None else self.type


def _recognize_window(net, gray: np.ndarray, vertices: np.ndarray) -> str:
    """CRNN read of one rectified window (official opencv_zoo pre/postprocess)."""
    target = np.array([[0, _INPUT_SIZE[1] - 1], [0, 0],
                       [_INPUT_SIZE[0] - 1, 0],
                       [_INPUT_SIZE[0] - 1, _INPUT_SIZE[1] - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(vertices.astype(np.float32), target)
    crop = cv2.warpPerspective(gray, matrix, _INPUT_SIZE)
    blob = cv2.dnn.blobFromImage(crop, size=_INPUT_SIZE, mean=127.5, scalefactor=1 / 127.5)
    net.setInput(blob)
    out = net.forward()
    raw = ""
    for i in range(out.shape[0]):
        c = int(np.argmax(out[i][0]))
        raw += _CHARSET[c - 1] if c != 0 else "-"
    chars = []
    for i, ch in enumerate(raw):
        if ch != "-" and not (i > 0 and ch == raw[i - 1]):
            chars.append(ch)
    return "".join(chars)


def _split_wide_quad(quad: np.ndarray) -> list[np.ndarray]:
    """Split a wide text quad into overlapping ~window-aspect chunks.

    Two reasons: the CRNN window is a fixed 100x32 (~3:1) and squashes a whole
    "patient_room_12" quad (aspect ~18:1) into mush; and with thin partitions
    the detector sometimes merges ADJACENT rooms' labels into one quad — chunk
    sub-quads carry their own positions, so each chunk lands in its own room
    and the per-room reading-order concatenation reassembles each label. The
    vocabulary match absorbs the duplicated characters at chunk seams.
    """
    vertices = quad.reshape((4, 2)).astype(np.float64)   # [bl, tl, tr, br]
    bl, tl, tr, br = vertices
    width = max(float(np.linalg.norm(tr - tl)), 1.0)
    height = max(float(np.linalg.norm(tl - bl)), 1.0)
    natural = _INPUT_SIZE[0] / _INPUT_SIZE[1]
    aspect = width / height
    if aspect <= natural * 1.6:
        return [quad]
    chunks = int(np.ceil(aspect / natural))
    step = 1.0 / chunks
    overlap = step * 0.18
    out = []
    for k in range(chunks):
        f0 = max(0.0, k * step - overlap)
        f1 = min(1.0, (k + 1) * step + overlap)
        out.append(np.array([bl + (br - bl) * f0, tl + (tr - tl) * f0,
                             tl + (tr - tl) * f1, bl + (br - bl) * f1], dtype=np.int32))
    return out


def _recognize(net, gray: np.ndarray, quad: np.ndarray) -> str:
    """CRNN read of one (window-sized) quad."""
    return _recognize_window(net, gray, quad.reshape((4, 2)).astype(np.float64))


def match_program_type(text: str, vocabulary: list[str]) -> tuple[str | None, int | None, float]:
    """Nearest program room type for a recognised label string.

    Returns ``(type, instance, score)`` — type None below the match floor.
    Trailing digits are the instance number ("patientroom12" -> patient_room, 12);
    area sublabels like "16m2" match no vocabulary entry and fall away.
    """
    clean = "".join(ch for ch in text.lower() if ch.isalnum())
    if not clean:
        return None, None, 0.0
    stem = clean.rstrip("0123456789")
    digits = clean[len(stem):]
    instance = int(digits) if digits else None
    best_type, best_score = None, 0.0
    for room_type in vocabulary:
        key = "".join(ch for ch in room_type.lower() if ch.isalnum())
        score = SequenceMatcher(None, stem, key).ratio()
        if score > best_score:
            best_type, best_score = room_type, score
    if best_score < _MIN_MATCH_SCORE:
        return None, None, best_score
    return best_type, instance, best_score


def type_rooms(png: bytes, plan, program: dict) -> dict[str, RoomTypeGuess]:
    """Read each plan room's drawn label; return {room_id: RoomTypeGuess}.

    Every candidate reading in a room competes: each detected quad inside the
    room polygon is recognised on its own AND as part of the room's left-to-
    right concatenation (labels sometimes split into word quads), each line
    is tried upright and rotated (corridor labels run vertically), and the
    highest vocabulary score wins the room.
    """
    if not RECOGNITION_MODEL_PATH.exists():
        return {}
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {}
    quads = detect_label_quads(image)
    if not quads:
        return {}
    net = cv2.dnn.readNet(str(RECOGNITION_MODEL_PATH))

    vocabulary = [r["type"] for r in program.get("rooms", [])]
    rooms = []
    for room in plan.rooms:
        poly = ShapelyPolygon(room.polygon)
        if not poly.is_valid:
            poly = poly.buffer(0)
        rooms.append((room.id, poly))
    # Room.polygon stores exterior rings only — a corridor that WRAPS other rooms
    # "contains" their labels too. Smallest containing room wins, so scan small→large.
    rooms.sort(key=lambda item: item[1].area if not item[1].is_empty else float("inf"))

    per_room: dict[str, list[tuple[float, np.ndarray, str]]] = {}
    for quad in [piece for q in quads for piece in _split_wide_quad(q)]:
        vertices = quad.reshape((4, 2)).astype(np.float64)   # [bl, tl, tr, br]
        bl, tl, tr, br = vertices
        # a label drawn near a small room's ceiling pokes out of the traced
        # polygon into the wall band — its BOTTOM edge is still the room's;
        # anchor there first, then the centre, then the nearest room within
        # two label heights
        bottom_center = Point(*((bl + br) / 2.0 - (0.0, 1.0)))
        center = Point(float(quad[:, 0].mean()), float(quad[:, 1].mean()))
        owner = next((rid for rid, poly in rooms
                      if not poly.is_empty and poly.contains(bottom_center)), None)
        if owner is None:
            owner = next((rid for rid, poly in rooms
                          if not poly.is_empty and poly.contains(center)), None)
        if owner is None:
            reach = 2.0 * max(8.0, float(quad[:, 1].max() - quad[:, 1].min()))
            near = [(poly.distance(bottom_center), rid)
                    for rid, poly in rooms if not poly.is_empty]
            distance, rid = min(near, default=(None, None))
            if distance is None or distance > reach:
                continue
            owner = rid
        w = float(quad[:, 0].max() - quad[:, 0].min())
        h = float(quad[:, 1].max() - quad[:, 1].min())
        readings = [_recognize(net, image, quad)]
        if h > 1.4 * w:
            # vertical label (rotated corridor text): read at every vertex
            # rotation and keep whatever the vocabulary recognises best
            for roll in (1, 2, 3):
                rotated = np.array(np.roll(quad.reshape(4, 2), roll, axis=0),
                                   dtype=np.int32)
                readings.append(_recognize(net, image, rotated))
        text = max(readings,
                   key=lambda s: (match_program_type(s, vocabulary)[2], len(s)))
        per_room.setdefault(owner, []).append((float(center.y), quad, text))

    areas = sorted(ShapelyPolygon(r.polygon).area for r in plan.rooms) or [0.0]
    median_area = areas[len(areas) // 2]
    room_area = {r.id: ShapelyPolygon(r.polygon).area for r in plan.rooms}
    circulation = {"corridor", "waiting"}

    guesses: dict[str, RoomTypeGuess] = {}
    for rid, entries in per_room.items():
        candidates = [text for _, _, text in entries]
        # reading-order concatenations repair labels split across quads
        entries.sort(key=lambda e: (round(e[0] / 20.0), float(e[1][:, 0].min())))
        candidates.append("".join(text for _, _, text in entries))
        # a merged circulation mega-space swallows unclosed member rooms AND
        # their labels — and the member label often reads BETTER than the
        # rotated corridor text, so in oversized rooms any circulation label
        # above the match floor outranks score itself
        oversized = room_area.get(rid, 0.0) > 2.5 * median_area
        best: RoomTypeGuess | None = None
        best_key: tuple | None = None
        for text in candidates:
            room_type, instance, score = match_program_type(text, vocabulary)
            if room_type is None:
                continue
            key = (1 if (oversized and room_type in circulation) else 0,
                   score,
                   1 if instance is not None else 0)
            if best_key is None or key > best_key:
                best_key = key
                best = RoomTypeGuess(type=room_type, instance=instance, score=score, text=text)
        if best is not None:
            guesses[rid] = best
    return guesses


def program_overview(rooms_colorful_png: bytes, plan, program: dict | None) -> bytes:
    """``rooms_colorful`` recoloured by PROGRAM TYPE + room names + colour legend.

    Built on the tracer's rooms_colorful rendering so walls and door gaps stay
    pixel-identical: each room's own fill colour is repainted with the shared
    ``ROOM_RGB`` palette colour of its assigned type (untyped rooms go grey),
    the room's name is written inside it, and a legend strip on the right maps
    every colour to its type with typed count vs the program's required count.
    """
    from hfagent.schema.palette import ROOM_RGB

    unknown_rgb = (205, 205, 205)
    image = cv2.imdecode(np.frombuffer(rooms_colorful_png, np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]

    for room in plan.rooms:
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.array(room.polygon, np.int32).reshape(-1, 1, 2)], 1)
        inside = image[mask == 1]
        if inside.size == 0:
            continue
        # the room's current fill = dominant non-wall, non-door colour inside it
        colours, counts = np.unique(inside.reshape(-1, 3), axis=0, return_counts=True)
        keep = [i for i, c in enumerate(colours)
                if not (c.max() < 60 or c.min() > 240)]     # skip wall black / door white
        if not keep:
            continue
        fill = colours[keep[int(np.argmax(counts[keep]))]]
        rgb = ROOM_RGB.get(room.type, unknown_rgb)
        target = (rgb[2], rgb[1], rgb[0])                    # palette is RGB, cv2 is BGR
        repaint = (mask == 1) & (np.abs(image.astype(int) - fill).sum(axis=2) <= 30)
        image[repaint] = target

    for room in plan.rooms:
        pts = np.array(room.polygon)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        label = room.name or (room.type if room.type != "unknown" else "?")
        scale = 0.55 if width > 2000 else 0.45
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        origin = (max(2, cx - size[0] // 2), cy + size[1] // 2)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 1, cv2.LINE_AA)

    typed_counts: dict[str, int] = {}
    for room in plan.rooms:
        typed_counts[room.type] = typed_counts.get(room.type, 0) + 1
    required = {r["type"]: r.get("count", 1) for r in (program or {}).get("rooms", [])}
    order = list(required) + [t for t in typed_counts if t not in required]
    entries = [(t, typed_counts.get(t, 0), required.get(t)) for t in dict.fromkeys(order)]

    panel_w = 360
    panel = np.full((height, panel_w, 3), 255, np.uint8)
    cv2.putText(panel, "PROGRAM", (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 2, cv2.LINE_AA)
    y = 92
    for room_type, have, want in entries:
        rgb = ROOM_RGB.get(room_type, unknown_rgb)
        cv2.rectangle(panel, (24, y - 20), (52, y + 8), (rgb[2], rgb[1], rgb[0]), -1)
        cv2.rectangle(panel, (24, y - 20), (52, y + 8), (0, 0, 0), 1)
        text = f"{room_type}  {have}" + (f" / {want}" if want is not None else "")
        cv2.putText(panel, text, (64, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 0, 0), 1, cv2.LINE_AA)
        y += 46
        if y > height - 20:
            break
    out = np.hstack([image, panel])
    ok, buf = cv2.imencode(".png", out)
    return buf.tobytes() if ok else b""


def typing_overlay(png: bytes, plan, guesses: dict[str, RoomTypeGuess]) -> bytes:
    """The source drawing with each room's assigned type written at its centroid."""
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    for room in plan.rooms:
        pts = np.array(room.polygon)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        guess = guesses.get(room.id)
        label = guess.name if guess else "?"
        color = (0, 160, 0) if guess else (0, 0, 230)
        cv2.putText(image, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", image)
    return buf.tobytes() if ok else b""
