"""Deterministic seed -> code translator for trace mode (/api/refine).

A traced floor plan arrives as neutral grid JSON. Instead of teaching the
coding agent our format, we translate the seed into the same kind of program
the agent itself writes: FOOTPRINT_RECTS / ROOM_DATA / DOORS data literals
plus a fixed rasterizing builder. The translation runs as attempt 0 of the
normal repair loop, so a failing seed is repaired as "the agent's own code".
"""

from __future__ import annotations

from typing import Any

SEED_MIN_WIDTH, SEED_MAX_WIDTH = 32, 160
SEED_MIN_HEIGHT, SEED_MAX_HEIGHT = 24, 120
MAX_SEED_ROOMS = 120
MAX_SEED_DOORS = 200


def _flatten_cells(raw: Any, width: int, height: int) -> list[int]:
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        if len(raw) != height or any(not isinstance(row, list) or len(row) != width for row in raw):
            raise ValueError(f"seed.cells rows must form a {height}x{width} matrix.")
        flat = [cell for row in raw for cell in row]
    elif isinstance(raw, list):
        flat = list(raw)
    else:
        raise ValueError("seed.cells must be a flat list or a row-nested matrix of integers.")
    if len(flat) != width * height:
        raise ValueError(f"seed.cells must contain exactly width*height = {width * height} values.")
    cells: list[int] = []
    for value in flat:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("seed.cells values must be integers.")
        if value < -2:
            raise ValueError("seed.cells values must be -2 (outside), -1 (unassigned), or room indexes.")
        cells.append(value)
    return cells


def normalize_seed(value: Any) -> dict[str, Any]:
    """Validated neutral seed: width/height/meters_per_cell/cells/rooms/doors."""
    if not isinstance(value, dict):
        raise ValueError("seed must be an object with width, height, meters_per_cell, cells, and rooms.")
    try:
        width = int(value.get("width", 0))
        height = int(value.get("height", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("seed.width and seed.height must be integers.") from error
    if not SEED_MIN_WIDTH <= width <= SEED_MAX_WIDTH or not SEED_MIN_HEIGHT <= height <= SEED_MAX_HEIGHT:
        raise ValueError(
            f"seed dimensions must be {SEED_MIN_WIDTH}x{SEED_MIN_HEIGHT} to {SEED_MAX_WIDTH}x{SEED_MAX_HEIGHT} cells."
        )
    try:
        meters_per_cell = float(value.get("meters_per_cell", 0.0) or 0.0)
    except (TypeError, ValueError) as error:
        raise ValueError("seed.meters_per_cell must be a number.") from error
    if not 0.0 < meters_per_cell <= 10.0:
        raise ValueError("seed.meters_per_cell must be a positive number no greater than 10.")

    rooms_raw = value.get("rooms")
    if not isinstance(rooms_raw, list) or not rooms_raw or len(rooms_raw) > MAX_SEED_ROOMS:
        raise ValueError(f"seed.rooms must be a list of 1 to {MAX_SEED_ROOMS} rooms.")
    rooms: list[dict[str, str]] = []
    seen_room_ids: set[str] = set()
    for index, room in enumerate(rooms_raw):
        if not isinstance(room, dict):
            raise ValueError("Each seed room must be an object with id and type.")
        room_id = str(room.get("id") or f"room_{index}").strip()
        room_type = str(room.get("type") or "").strip()
        if not room_type:
            raise ValueError(f"seed room '{room_id}' is missing its type.")
        if room_id in seen_room_ids:
            raise ValueError(f"Duplicate seed room id: {room_id}.")
        seen_room_ids.add(room_id)
        rooms.append({"id": room_id, "type": room_type})

    cells = _flatten_cells(value.get("cells"), width, height)
    owned = [0] * len(rooms)
    for cell in cells:
        if cell >= len(rooms):
            raise ValueError(f"seed.cells references room index {cell} but only {len(rooms)} rooms exist.")
        if cell >= 0:
            owned[cell] += 1
    for index, count in enumerate(owned):
        if count == 0:
            raise ValueError(f"seed room '{rooms[index]['id']}' owns no cells.")

    doors_raw = value.get("doors") or []
    if not isinstance(doors_raw, list) or len(doors_raw) > MAX_SEED_DOORS:
        raise ValueError(f"seed.doors must be a list of at most {MAX_SEED_DOORS} doors.")
    doors: list[dict[str, Any]] = []
    seen_door_ids: set[str] = set()
    for index, door in enumerate(doors_raw):
        if not isinstance(door, dict):
            raise ValueError("Each seed door must be an object.")
        door_id = str(door.get("id") or f"door_{index}").strip()
        if door_id in seen_door_ids:
            raise ValueError(f"Duplicate seed door id: {door_id}.")
        seen_door_ids.add(door_id)
        from_room = str(door.get("from_room", "")).strip()
        to_room_raw = door.get("to_room")
        to_room = None if to_room_raw in (None, "") else str(to_room_raw).strip()
        if from_room not in seen_room_ids or (to_room is not None and to_room not in seen_room_ids):
            raise ValueError(f"seed door '{door_id}' references an unknown room.")
        orientation = str(door.get("orientation", "")).strip()
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError(f"seed door '{door_id}' orientation must be horizontal or vertical.")
        try:
            x = int(door.get("x"))
            y = int(door.get("y"))
            width_cells = int(door.get("width_cells", 1))
        except (TypeError, ValueError) as error:
            raise ValueError(f"seed door '{door_id}' coordinates must be integers.") from error
        if not 0 <= x <= width or not 0 <= y <= height:
            raise ValueError(f"seed door '{door_id}' is outside the canvas.")
        if not 1 <= width_cells <= 6:
            raise ValueError(f"seed door '{door_id}' width_cells must be between 1 and 6.")
        doors.append(
            {
                "id": door_id,
                "from_room": from_room,
                "to_room": to_room,
                "x": x,
                "y": y,
                "orientation": orientation,
                "width_cells": width_cells,
            }
        )

    return {
        "width": width,
        "height": height,
        "meters_per_cell": round(meters_per_cell, 4),
        "cells": cells,
        "rooms": rooms,
        "doors": doors,
    }


def _decompose_rectangles(owns: list[bool], width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Greedy maximal rectangles covering all True cells (row-major sweep)."""
    consumed = [False] * (width * height)
    rectangles: list[tuple[int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            offset = y * width + x
            if not owns[offset] or consumed[offset]:
                continue
            rect_width = 1
            while x + rect_width < width and owns[offset + rect_width] and not consumed[offset + rect_width]:
                rect_width += 1
            rect_height = 1
            while y + rect_height < height:
                row_offset = (y + rect_height) * width + x
                if any(
                    not owns[row_offset + column] or consumed[row_offset + column]
                    for column in range(rect_width)
                ):
                    break
                rect_height += 1
            for row in range(y, y + rect_height):
                for column in range(x, x + rect_width):
                    consumed[row * width + column] = True
            rectangles.append((x, y, rect_width, rect_height))
    return rectangles


def seed_to_code(seed: dict[str, Any]) -> str:
    """The seed as a complete agent-style program: data literals + fixed builder."""
    width = seed["width"]
    height = seed["height"]
    cells = seed["cells"]

    footprint_rects = _decompose_rectangles([cell != -2 for cell in cells], width, height)
    room_lines: list[str] = []
    for index, room in enumerate(seed["rooms"]):
        rects = _decompose_rectangles([cell == index for cell in cells], width, height)
        rect_text = ", ".join(f"({x}, {y}, {w}, {h})" for x, y, w, h in rects)
        room_lines.append(f'    ("{room["id"]}", "{room["type"]}", [{rect_text}]),')
    footprint_text = "\n".join(
        f"    ({x}, {y}, {w}, {h})," for x, y, w, h in footprint_rects
    )
    door_lines = [
        f'    ("{door["id"]}", "{door["from_room"]}", '
        + (f'"{door["to_room"]}"' if door["to_room"] is not None else "None")
        + f', {door["x"]}, {door["y"]}, "{door["orientation"]}", {door["width_cells"]}),'
        for door in seed["doors"]
    ]

    return f'''# Deterministic translation of a traced floor plan (trace mode seed).
# The plan IS the data below; the builder only rasterizes it. Repairs should
# adjust these literals, not replace the layout.
WIDTH = {width}
HEIGHT = {height}
METERS_PER_CELL = {seed["meters_per_cell"]}

# (x, y, w, h) cell rectangles covering the building footprint.
FOOTPRINT_RECTS = [
{footprint_text}
]

# (room_id, room_type, [(x, y, w, h), ...]) — order defines grid indexes.
ROOM_DATA = [
{chr(10).join(room_lines)}
]

# (door_id, from_room, to_room_or_None, x, y, orientation, width_cells)
DOORS = [
{chr(10).join(door_lines)}
]

footprint = [[False] * WIDTH for _ in range(HEIGHT)]
for rect_x, rect_y, rect_w, rect_h in FOOTPRINT_RECTS:
    for row in range(rect_y, rect_y + rect_h):
        for col in range(rect_x, rect_x + rect_w):
            footprint[row][col] = True

grid = [[-2] * WIDTH for _ in range(HEIGHT)]
for row in range(HEIGHT):
    for col in range(WIDTH):
        if footprint[row][col]:
            grid[row][col] = -1

for room_index, (room_id, room_type, rects) in enumerate(ROOM_DATA):
    for rect_x, rect_y, rect_w, rect_h in rects:
        for row in range(rect_y, rect_y + rect_h):
            for col in range(rect_x, rect_x + rect_w):
                if grid[row][col] >= 0 and grid[row][col] != room_index:
                    raise ValueError(
                        "Room " + room_id + " overlaps room " + ROOM_DATA[grid[row][col]][0]
                    )
                grid[row][col] = room_index
                footprint[row][col] = True

result = {{
    "width": WIDTH,
    "height": HEIGHT,
    "meters_per_cell": METERS_PER_CELL,
    "footprint": footprint,
    "grid": grid,
    "rooms": [{{"id": room_id, "type": room_type}} for room_id, room_type, _ in ROOM_DATA],
    "doors": [
        {{
            "id": door_id,
            "from_room": from_room,
            "to_room": to_room,
            "x": door_x,
            "y": door_y,
            "orientation": orientation,
            "width_cells": width_cells,
        }}
        for door_id, from_room, to_room, door_x, door_y, orientation, width_cells in DOORS
    ],
}}
'''
