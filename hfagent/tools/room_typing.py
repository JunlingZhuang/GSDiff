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
        text = _recognize(net, image, quad)
        w = float(quad[:, 0].max() - quad[:, 0].min())
        h = float(quad[:, 1].max() - quad[:, 1].min())
        if h > 1.4 * w:      # vertical label (rotated corridor text): read rotated too
            rotated = np.array(np.roll(quad, 1, axis=0), dtype=np.int32)
            alt = _recognize(net, image, rotated)
            if len(alt) > len(text):
                text = alt
        per_room.setdefault(owner, []).append((float(center.y), quad, text))

    guesses: dict[str, RoomTypeGuess] = {}
    for rid, entries in per_room.items():
        candidates = [text for _, _, text in entries]
        # reading-order concatenations repair labels split across quads
        entries.sort(key=lambda e: (round(e[0] / 20.0), float(e[1][:, 0].min())))
        candidates.append("".join(text for _, _, text in entries))
        best: RoomTypeGuess | None = None
        for text in candidates:
            room_type, instance, score = match_program_type(text, vocabulary)
            if room_type is None:
                continue
            if best is None or score > best.score or (score == best.score
                                                      and instance is not None and best.instance is None):
                best = RoomTypeGuess(type=room_type, instance=instance, score=score, text=text)
        if best is not None:
            guesses[rid] = best
    return guesses


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
