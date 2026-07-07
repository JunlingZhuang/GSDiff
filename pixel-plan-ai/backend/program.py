from __future__ import annotations

import math
from typing import Any

from constants import area_for


def normalize_program(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Program must be a JSON object.")
    raw_rooms = value.get("rooms")
    if not isinstance(raw_rooms, list) or not raw_rooms:
        raise ValueError("Program.rooms must be a non-empty array.")

    rooms: list[dict[str, Any]] = []
    for index, item in enumerate(raw_rooms):
        if not isinstance(item, dict) or not isinstance(item.get("type"), str) or not item["type"].strip():
            raise ValueError(f"rooms[{index}].type is required.")
        count = max(1, min(200, round(float(item.get("count", 1)))))
        room: dict[str, Any] = {"type": item["type"].strip(), "count": count}
        if item.get("approx_area_ft2") is not None:
            area_ft2 = float(item["approx_area_ft2"])
            if not math.isfinite(area_ft2) or area_ft2 <= 0:
                raise ValueError(f"rooms[{index}].approx_area_ft2 must be positive.")
            room["approx_area_ft2"] = area_ft2
            room["approx_area_m2"] = area_ft2 * 0.09290304
        elif item.get("approx_area_m2") is not None:
            area = float(item["approx_area_m2"])
            if not math.isfinite(area) or area <= 0:
                raise ValueError(f"rooms[{index}].approx_area_m2 must be positive.")
            room["approx_area_m2"] = area
        rooms.append(room)

    adjacency: list[list[str]] = []
    raw_adjacency = value.get("adjacency", [])
    if not isinstance(raw_adjacency, list):
        raise ValueError("Program.adjacency must be an array.")
    for index, pair in enumerate(raw_adjacency):
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(item, str) for item in pair):
            raise ValueError(f"adjacency[{index}] must contain two room type strings.")
        adjacency.append([pair[0].strip(), pair[1].strip()])

    if sum(room["count"] for room in rooms) > 240:
        raise ValueError("The prototype supports at most 240 room instances.")
    return {
        "building_type": str(value.get("building_type", "unspecified building")),
        "rooms": rooms,
        "adjacency": adjacency,
    }


def expand_rooms(program: dict[str, Any]) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for room in program["rooms"]:
        for index in range(1, room["count"] + 1):
            instances.append(
                {
                    "id": f"{room['type']}-{index}",
                    "type": room["type"],
                    "target_area_m2": area_for(room),
                }
            )
    return instances


def calculate_scale(program: dict[str, Any], width: int, height: int, corridor_ratio: float = 0.14) -> float:
    programmed = sum(
        area_for(room) * room["count"]
        for room in program["rooms"]
        if room["type"] not in {"corridor", "circulation"}
    )
    gross_area = programmed / max(0.5, 1.0 - corridor_ratio)
    return math.sqrt(gross_area / (width * height))
