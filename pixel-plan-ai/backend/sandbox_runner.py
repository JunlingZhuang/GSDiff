from __future__ import annotations

import builtins
import contextlib
import io
import json
import sys
from typing import Any

from code_policy import AUTHORIZED_IMPORTS, validate_generated_code
from constants import color_for
from pixel_plan import PixelPlan


def restricted_import(
    name: str,
    globals_value: dict[str, object] | None = None,
    locals_value: dict[str, object] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> object:
    if level or name.split(".", 1)[0] not in AUTHORIZED_IMPORTS:
        raise ImportError(f"Import is not authorized: {name}")
    return builtins.__import__(name, globals_value, locals_value, fromlist, level)


SAFE_BUILTINS = {
    "__build_class__": builtins.__build_class__,
    "__import__": restricted_import,
    "abs": abs,
    "all": all,
    "any": any,
    "BaseException": BaseException,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "IndexError": IndexError,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "KeyError": KeyError,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "OverflowError": OverflowError,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "RuntimeError": RuntimeError,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "StopIteration": StopIteration,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
    "ZeroDivisionError": ZeroDivisionError,
}


def plain_value(value: Any) -> Any:
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, dict):
        return {str(key): plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"Result contains unsupported value type: {type(value).__name__}.")


def normalize_matrix(value: Any, width: int, height: int, label: str) -> list[list[Any]]:
    matrix = plain_value(value)
    if not isinstance(matrix, list) or len(matrix) != height:
        raise ValueError(f"{label} must contain exactly {height} rows.")
    if any(not isinstance(row, list) or len(row) != width for row in matrix):
        raise ValueError(f"Every {label} row must contain exactly {width} cells.")
    return matrix


def validate_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 80:
        raise ValueError(f"{label} is invalid.")
    if any(not (character.isalnum() or character in "_-") for character in value):
        raise ValueError(f"{label} contains invalid characters.")
    return value


def validate_doors(
    raw_doors: Any,
    width: int,
    height: int,
    footprint: list[int],
    cells: list[int],
    room_indexes: dict[str, int],
    room_types: list[str],
) -> list[dict[str, Any]]:
    doors = plain_value(raw_doors or [])
    if not isinstance(doors, list):
        raise ValueError("result['doors'] must be a list.")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def room_at(cell_x: int, cell_y: int) -> int:
        if cell_x < 0 or cell_x >= width or cell_y < 0 or cell_y >= height:
            return -2
        offset = cell_y * width + cell_x
        if not footprint[offset]:
            return -2
        return cells[offset]

    for raw in doors:
        if not isinstance(raw, dict):
            raise ValueError("Each door must be a dictionary.")
        door_id = validate_identifier(raw.get("id"), "Door id")
        if door_id in seen_ids:
            raise ValueError(f"Duplicate door id: {door_id}.")
        seen_ids.add(door_id)
        from_room = validate_identifier(raw.get("from_room"), "Door from_room")
        to_room = raw.get("to_room")
        if to_room is not None:
            to_room = validate_identifier(to_room, "Door to_room")
        if from_room not in room_indexes or (to_room is not None and to_room not in room_indexes):
            raise ValueError(f"Door {door_id} references an unknown room.")
        x = raw.get("x")
        y = raw.get("y")
        width_cells = raw.get("width_cells", 1)
        orientation = raw.get("orientation")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (x, y, width_cells)):
            raise ValueError(f"Door {door_id} coordinates and width must be integers.")
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError(f"Door {door_id} orientation must be horizontal or vertical.")
        if not 1 <= width_cells <= 6:
            raise ValueError(f"Door {door_id} width must be between 1 and 6 cells.")
        if not 0 <= x <= width or not 0 <= y <= height:
            raise ValueError(f"Door {door_id} is outside the canvas.")

        from_index = room_indexes[from_room]
        to_index = room_indexes[to_room] if to_room is not None else -2
        expected = {from_index, to_index}
        boundary_pairs: list[tuple[int, int]] = []
        for offset in range(width_cells):
            if orientation == "horizontal":
                boundary_pairs.append((room_at(x + offset, y - 1), room_at(x + offset, y)))
            else:
                boundary_pairs.append((room_at(x - 1, y + offset), room_at(x, y + offset)))
        if any({first, second} != expected for first, second in boundary_pairs):
            destination = to_room if to_room is not None else "outside"
            raise ValueError(
                f"Door {door_id} is not fully on the shared boundary between {from_room} and {destination}."
            )
        circulation_types = {"corridor", "circulation"}
        from_type = room_types[from_index]
        to_type = room_types[to_index] if to_index >= 0 else None
        if to_index < 0:
            swing_index = from_index
        elif from_type in circulation_types and to_type not in circulation_types:
            swing_index = to_index
        elif to_type in circulation_types and from_type not in circulation_types:
            swing_index = from_index
        else:
            swing_index = from_index
        first_index, second_index = boundary_pairs[0]
        if swing_index == first_index:
            swing_side = "north" if orientation == "horizontal" else "west"
        elif swing_index == second_index:
            swing_side = "south" if orientation == "horizontal" else "east"
        else:
            raise ValueError(f"Door {door_id} swing room is not on its shared boundary.")
        normalized.append(
            {
                "id": door_id,
                "from_room": from_room,
                "to_room": to_room,
                "x": x,
                "y": y,
                "orientation": orientation,
                "width_cells": width_cells,
                "swing_side": swing_side,
            }
        )
    return normalized


def normalize_result(raw_result: Any) -> dict[str, Any]:
    result = plain_value(raw_result)
    if not isinstance(result, dict):
        raise ValueError("Top-level variable result must be a dictionary.")
    width = result.get("width")
    height = result.get("height")
    meters_per_cell = result.get("meters_per_cell")
    if not isinstance(width, int) or isinstance(width, bool) or not 8 <= width <= 200:
        raise ValueError("result['width'] must be an integer from 8 to 200.")
    if not isinstance(height, int) or isinstance(height, bool) or not 8 <= height <= 160:
        raise ValueError("result['height'] must be an integer from 8 to 160.")
    if not isinstance(meters_per_cell, (int, float)) or isinstance(meters_per_cell, bool) or not 0 < meters_per_cell <= 10:
        raise ValueError("result['meters_per_cell'] must be a positive number no greater than 10.")

    raw_rooms = result.get("rooms")
    if not isinstance(raw_rooms, list) or not raw_rooms:
        raise ValueError("result['rooms'] must be a non-empty list.")
    room_indexes: dict[str, int] = {}
    room_types: list[str] = []
    for index, room in enumerate(raw_rooms):
        if not isinstance(room, dict):
            raise ValueError("Each room must be a dictionary.")
        room_id = validate_identifier(room.get("id"), "Room id")
        room_type = validate_identifier(room.get("type"), "Room type")
        if room_id in room_indexes:
            raise ValueError(f"Duplicate room id: {room_id}.")
        room_indexes[room_id] = index
        room_types.append(room_type)

    grid = normalize_matrix(result.get("grid"), width, height, "result['grid']")
    raw_footprint = result.get("footprint")
    footprint_matrix = (
        normalize_matrix(raw_footprint, width, height, "result['footprint']")
        if raw_footprint is not None
        else [[cell != -2 and cell != "__outside__" for cell in row] for row in grid]
    )
    footprint: list[int] = []
    cells: list[int] = []
    pixel_counts = [0] * len(raw_rooms)
    minimum_x = [width] * len(raw_rooms)
    minimum_y = [height] * len(raw_rooms)
    maximum_x = [-1] * len(raw_rooms)
    maximum_y = [-1] * len(raw_rooms)

    for y, row in enumerate(grid):
        for x, raw_cell in enumerate(row):
            inside = bool(footprint_matrix[y][x])
            footprint.append(int(inside))
            if not inside:
                cells.append(-1)
                continue
            if raw_cell is None or raw_cell == -1:
                cells.append(-1)
                continue
            if isinstance(raw_cell, str):
                if raw_cell not in room_indexes:
                    raise ValueError(f"Grid references unknown room id: {raw_cell}.")
                room_index = room_indexes[raw_cell]
            elif isinstance(raw_cell, int) and not isinstance(raw_cell, bool):
                room_index = raw_cell
                if not 0 <= room_index < len(raw_rooms):
                    raise ValueError(f"Grid contains invalid room index: {room_index}.")
            else:
                raise ValueError("Grid cells must be room indexes, room ids, -1, or None.")
            cells.append(room_index)
            pixel_counts[room_index] += 1
            minimum_x[room_index] = min(minimum_x[room_index], x)
            minimum_y[room_index] = min(minimum_y[room_index], y)
            maximum_x[room_index] = max(maximum_x[room_index], x)
            maximum_y[room_index] = max(maximum_y[room_index], y)

    rooms: list[dict[str, Any]] = []
    for index, raw_room in enumerate(raw_rooms):
        if pixel_counts[index] == 0:
            raise ValueError(f"Room {raw_room['id']} has no pixels.")
        rooms.append(
            {
                "id": raw_room["id"],
                "type": room_types[index],
                "color": color_for(room_types[index]),
                "pixel_count": pixel_counts[index],
                "bounds": {
                    "x": minimum_x[index],
                    "y": minimum_y[index],
                    "width": maximum_x[index] - minimum_x[index] + 1,
                    "height": maximum_y[index] - minimum_y[index] + 1,
                },
            }
        )
    doors = validate_doors(result.get("doors"), width, height, footprint, cells, room_indexes, room_types)
    return {
        "width": width,
        "height": height,
        "meters_per_cell": float(meters_per_cell),
        "footprint": footprint,
        "cells": cells,
        "rooms": rooms,
        "doors": doors,
    }


def main() -> int:
    code = sys.stdin.read()
    try:
        tree = validate_generated_code(code)
        scope: dict[str, object] = {
            "__builtins__": SAFE_BUILTINS,
            "__name__": "generated_layout",
            "PixelPlan": PixelPlan,
        }
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            exec(compile(tree, "generated_layout.py", "exec"), scope, scope)
        raw_result = scope.get("result")
        if raw_result is not None:
            plan = normalize_result(raw_result)
        else:
            legacy_plan = scope.get("plan")
            if not isinstance(legacy_plan, PixelPlan):
                raise ValueError("Generated code did not produce top-level variable result.")
            plan = legacy_plan.finish()
        print(json.dumps({"ok": True, "result": plan}, separators=(",", ":")))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
