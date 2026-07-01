# -*- coding: utf-8 -*-
"""Tool: read a realistic floor plan into a STRUCTURED room layout (rectangles + doors).

The alternative to to_colorblock + cv_parse. Image models won't reliably draw a clean
segmentation mask — they keep door arcs, window ticks, text and gradients, which then
pollute the parser. Instead the VLM READS the realistic drawing and returns JSON: each
room as one or more rectangles (normalized 0-1) plus door connectivity, which `rectify`
turns into a clean Plan deterministically. The realistic image is still drawn by the
image model (good layout); only the structure extraction changes.

The model does chain-of-thought (reason about the layout first, then emit the JSON) and
gives RELATIVE rectangles — these are used only for layout, never for absolute proportion.
The building's true aspect ratio is measured from the realistic image pixels in `rectify`
(`building_aspect`), not taken from the model, which skews extents.

One JSON call replaces both to_colorblock (pass 2) and extract_room_adjacency.

Public API:
    read_structure(real_png, client, program) -> (list[RoomShape], RoomGraph)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from hfagent.schema.palette import ROOM_RGB
from hfagent.schema.roomgraph import Door, RoomGraph, RoomNode


@dataclass
class RoomShape:
    id: str                                          # instance label, e.g. "patient_room_3"
    type: str                                        # base room type ∈ palette
    rects: list[tuple[float, float, float, float]]   # [x1,y1,x2,y2] normalized 0-1; one or more


_STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "rects": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                },
                "required": ["id", "type", "rects"],
            },
        },
        "doors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"room_a": {"type": "string"}, "room_b": {"type": "string"}},
                "required": ["room_a", "room_b"],
            },
        },
    },
    "required": ["rooms", "doors"],
}


def _build_structure_prompt(program: dict) -> str:
    types = sorted({r["type"] for r in program["rooms"]})
    return f"""The image above is a top-down architectural floor plan. Read it into structured JSON.

First think briefly in "reasoning": the overall footprint, where the corridor runs, how the rooms are arranged. THEN list every room and every door.

Use a normalized coordinate system over the WHOLE image: x from 0.0 (left edge) to 1.0 (right edge), y from 0.0 (top) to 1.0 (bottom). Keep each room's position and relative size faithful to the drawing.

For each ROOM return:
  id: the exact instance label printed inside the room (e.g. "patient_room_3"; if a type has only one room, just the type, e.g. "corridor")
  type: the base room type, one of: {", ".join(types)}
  rects: a list of axis-aligned rectangles [x1, y1, x2, y2] (normalized) that together cover the room. Most rooms = ONE rectangle. A bent corridor or an L-shaped room = 2-3 rectangles that together cover its whole area. Rooms should tile the building with no gaps or overlaps.

For each DOOR (an opening / quarter-circle swing arc between two rooms) return room_a and room_b using the room ids above ("exterior" only for a real entrance door). Ignore windows.

Return ONE JSON object {{"reasoning": "...", "rooms": [...], "doors": [...]}}. No extra text."""


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def read_structure(
    real_png: bytes,
    client,
    program: dict,
    prompt_logger: Callable[[str, str], None] | None = None,
) -> tuple[list[RoomShape], RoomGraph]:
    prompt = _build_structure_prompt(program)
    if prompt_logger is not None:
        prompt_logger("structure JSON", prompt)
    raw = client.generate_json(
        [real_png, prompt], schema=_STRUCTURE_SCHEMA
    )
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    rooms: list[RoomShape] = []
    for r in data.get("rooms") or []:
        if not isinstance(r, dict) or str(r.get("type", "")) not in ROOM_RGB:
            continue
        rects: list[tuple[float, float, float, float]] = []
        for rc in r.get("rects") or []:
            if not (isinstance(rc, (list, tuple)) and len(rc) == 4):
                continue
            try:
                x1, y1, x2, y2 = (_clamp01(float(v)) for v in rc)
            except (TypeError, ValueError):
                continue
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 - x1 > 1e-4 and y2 - y1 > 1e-4:
                rects.append((x1, y1, x2, y2))
        if not rects:
            continue
        rtype = str(r["type"])
        rooms.append(RoomShape(id=str(r.get("id") or rtype), type=rtype, rects=rects))

    valid_ids = {rs.id for rs in rooms} | {"exterior"}
    doors: list[Door] = []
    seen: set[tuple[str, str]] = set()
    for d in data.get("doors") or []:
        if not isinstance(d, dict):
            continue
        a, b = str(d.get("room_a", "")), str(d.get("room_b", ""))
        if a not in valid_ids or b not in valid_ids or a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        doors.append(Door(room_a=a, room_b=b))

    nodes = [RoomNode(id=rs.id, type=rs.type) for rs in rooms]
    return rooms, RoomGraph(rooms=nodes, doors=doors)
