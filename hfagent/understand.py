# -*- coding: utf-8 -*-
"""Tool ⓪ understand: natural-language brief -> structured program JSON.

First real use of the TEXT model (gemini-3.1-pro family) with a forced
response schema. The LLM only proposes STRUCTURE (room list / counts /
adjacency); everything downstream is the validated Phase-0 pipeline. Output is
sanitised deterministically — unknown room types and malformed adjacency are
dropped, counts clamped — so a hallucinated program cannot reach generation.
"""
from __future__ import annotations

import json

from hfagent.schema.palette import ROOM_TYPES

PROGRAM_SCHEMA = {
    "type": "object",
    "properties": {
        "building_type": {"type": "string"},
        "rooms": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ROOM_TYPES},
                    "count": {"type": "integer", "minimum": 1, "maximum": 12},
                    "approx_area_m2": {"type": "number", "minimum": 2, "maximum": 200},
                },
                "required": ["type", "count"],
            },
        },
        "adjacency": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
        },
    },
    "required": ["building_type", "rooms"],
}


def build_understand_prompt(text: str) -> str:
    return f"""You turn a building brief into a structured room program for a healthcare floor plan generator.

AVAILABLE ROOM TYPES (use ONLY these): {", ".join(ROOM_TYPES)}

RULES:
- Cover everything the brief asks for; fill sensible healthcare defaults for what it omits
  (a clinic needs waiting + toilet; more than 3 rooms needs a corridor; a ward wing needs a nurse_station).
- Every room type that patients or staff must reach should be adjacent to "corridor" (or "waiting" for small clinics).
- Keep counts realistic and within 1..12 per type. Include approx_area_m2 when the brief implies sizes.

BRIEF:
{text}

Return the program JSON only."""


def sanitize_program(program: dict) -> dict:
    rooms = []
    seen: set[str] = set()
    for r in program.get("rooms", []):
        t = r.get("type")
        if t not in ROOM_TYPES or t in seen:
            continue
        seen.add(t)
        room = {"type": t, "count": max(1, min(12, int(r.get("count", 1))))}
        if isinstance(r.get("approx_area_m2"), (int, float)) and r["approx_area_m2"] > 0:
            room["approx_area_m2"] = float(r["approx_area_m2"])
        rooms.append(room)
    if not rooms:
        raise ValueError("program has no usable rooms")
    types = {r["type"] for r in rooms}
    adjacency = [
        [a, b] for a, b in (pair for pair in program.get("adjacency", []) if len(pair) == 2)
        if a in types and b in types and a != b
    ]
    return {
        "building_type": str(program.get("building_type", "healthcare facility")),
        "rooms": rooms,
        "adjacency": adjacency,
    }


def understand(text: str, client) -> dict:
    """client: anything with generate_json(prompt, schema) -> str."""
    raw = client.generate_json(build_understand_prompt(text), schema=PROGRAM_SCHEMA)
    return sanitize_program(json.loads(raw))
