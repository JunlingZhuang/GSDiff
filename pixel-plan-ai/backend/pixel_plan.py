from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from constants import color_for


@dataclass(frozen=True)
class Door:
    id: str
    from_room: str
    to_room: str | None
    x: int
    y: int
    orientation: str
    width_cells: int


class PixelPlan:
    def __init__(self, width: int, height: int, meters_per_cell: float = 0.5) -> None:
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("Canvas dimensions must be integers.")
        if not 8 <= width <= 200 or not 8 <= height <= 160:
            raise ValueError("Canvas dimensions are outside the allowed range.")
        if not isinstance(meters_per_cell, (int, float)) or not 0 < meters_per_cell <= 10:
            raise ValueError("meters_per_cell must be between 0 and 10.")
        self.width = width
        self.height = height
        self.meters_per_cell = float(meters_per_cell)
        self.cells = [-1] * (width * height)
        self.footprint = [True] * (width * height)
        self.rooms: list[dict[str, Any]] = []
        self.room_indexes: dict[str, int] = {}
        self.doors: list[Door] = []

    @staticmethod
    def _validate_id(value: str, label: str) -> None:
        if not isinstance(value, str) or not value or len(value) > 80:
            raise ValueError(f"{label} is invalid.")
        if any(not (char.isalnum() or char in "_-") for char in value):
            raise ValueError(f"{label} contains invalid characters.")

    def _paint(self, room_id: str, room_type: str, points: list[tuple[int, int]]) -> str:
        self._validate_id(room_id, "Room id")
        self._validate_id(room_type, "Room type")
        if room_id in self.room_indexes:
            raise ValueError(f"Duplicate room id: {room_id}")
        if not points:
            raise ValueError(f"Room {room_id} has no pixels.")
        room_index = len(self.rooms)
        min_x, min_y = self.width, self.height
        max_x = max_y = -1
        for x, y in points:
            if not isinstance(x, int) or not isinstance(y, int):
                raise ValueError("Pixel coordinates must be integers.")
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                raise ValueError(f"Room {room_id} extends outside the canvas at ({x}, {y}).")
            offset = y * self.width + x
            if not self.footprint[offset]:
                raise ValueError(f"Room {room_id} extends outside the building footprint at ({x}, {y}).")
            if self.cells[offset] != -1:
                other = self.rooms[self.cells[offset]]["id"]
                raise ValueError(f"Room {room_id} overlaps {other} at ({x}, {y}).")
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)
        for x, y in points:
            self.cells[y * self.width + x] = room_index
        self.room_indexes[room_id] = room_index
        self.rooms.append(
            {
                "id": room_id,
                "type": room_type,
                "color": color_for(room_type),
                "pixel_count": len(points),
                "bounds": {"x": min_x, "y": min_y, "width": max_x - min_x + 1, "height": max_y - min_y + 1},
            }
        )
        return room_id

    def fill_rect(self, room_id: str, room_type: str, x: int, y: int, width: int, height: int) -> str:
        if not all(isinstance(value, int) for value in (x, y, width, height)):
            raise ValueError("Rectangle values must be integers.")
        if width <= 0 or height <= 0:
            raise ValueError("Rectangle dimensions must be positive.")
        points = [(px, py) for py in range(y, y + height) for px in range(x, x + width)]
        return self._paint(room_id, room_type, points)

    def _polygon_points(self, vertices: list[list[int]] | list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not isinstance(vertices, list) or not 3 <= len(vertices) <= 64:
            raise ValueError("A polygon needs 3 to 64 vertices.")
        normalized: list[tuple[int, int]] = []
        for vertex in vertices:
            if not isinstance(vertex, (list, tuple)) or len(vertex) != 2 or not all(isinstance(value, int) for value in vertex):
                raise ValueError("Polygon vertices must be integer [x, y] pairs.")
            normalized.append((vertex[0], vertex[1]))
        min_x = max(0, min(x for x, _ in normalized))
        max_x = min(self.width - 1, max(x for x, _ in normalized))
        min_y = max(0, min(y for _, y in normalized))
        max_y = min(self.height - 1, max(y for _, y in normalized))
        points: list[tuple[int, int]] = []
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                px, py = x + 0.5, y + 0.5
                inside = False
                previous = len(normalized) - 1
                for current in range(len(normalized)):
                    xi, yi = normalized[current]
                    xj, yj = normalized[previous]
                    crosses = (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-9) + xi
                    if crosses:
                        inside = not inside
                    previous = current
                if inside:
                    points.append((x, y))
        return points

    def set_footprint(self, vertices: list[list[int]] | list[tuple[int, int]]) -> None:
        if self.rooms:
            raise ValueError("The footprint must be defined before rooms are painted.")
        points = self._polygon_points(vertices)
        if not points:
            raise ValueError("The building footprint has no pixels.")
        self.footprint = [False] * (self.width * self.height)
        for x, y in points:
            self.footprint[y * self.width + x] = True

    def fill_polygon(self, room_id: str, room_type: str, vertices: list[list[int]] | list[tuple[int, int]]) -> str:
        return self._paint(room_id, room_type, self._polygon_points(vertices))

    def fill_corridor_path(
        self,
        corridor_id: str,
        vertices: list[list[int]] | list[tuple[int, int]],
        thickness: int,
    ) -> str:
        if not isinstance(vertices, list) or not 2 <= len(vertices) <= 32:
            raise ValueError("A corridor path needs 2 to 32 vertices.")
        if not isinstance(thickness, int) or not 2 <= thickness <= min(self.width, self.height) // 3:
            raise ValueError("Corridor thickness is outside the allowed range.")
        normalized: list[tuple[int, int]] = []
        for vertex in vertices:
            if not isinstance(vertex, (list, tuple)) or len(vertex) != 2 or not all(
                isinstance(value, int) for value in vertex
            ):
                raise ValueError("Corridor vertices must be integer [x, y] pairs.")
            normalized.append((vertex[0], vertex[1]))

        points: set[tuple[int, int]] = set()
        offset_start = -(thickness // 2)
        for (start_x, start_y), (end_x, end_y) in zip(normalized, normalized[1:]):
            if start_x != end_x and start_y != end_y:
                raise ValueError("Corridor path segments must be horizontal or vertical.")
            if start_x == end_x and start_y == end_y:
                raise ValueError("Corridor path contains a zero-length segment.")
            if start_y == end_y:
                for x in range(min(start_x, end_x), max(start_x, end_x) + 1):
                    for delta in range(offset_start, offset_start + thickness):
                        points.add((x, start_y + delta))
            else:
                for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
                    for delta in range(offset_start, offset_start + thickness):
                        points.add((start_x + delta, y))
        return self._paint(corridor_id, "corridor", sorted(points, key=lambda point: (point[1], point[0])))

    def fill_cross_corridors(
        self,
        corridor_ids: list[str] | tuple[str, str, str, str],
        center_x: int,
        center_y: int,
        thickness: int,
    ) -> None:
        if not isinstance(corridor_ids, (list, tuple)) or len(corridor_ids) != 4:
            raise ValueError("Cross corridors need four ids ordered north, south, west, east.")
        if not all(isinstance(room_id, str) for room_id in corridor_ids):
            raise ValueError("Cross corridor ids must be strings.")
        if not all(isinstance(value, int) for value in (center_x, center_y, thickness)):
            raise ValueError("Cross corridor values must be integers.")
        if not 2 <= thickness <= min(self.width, self.height) // 3:
            raise ValueError("Cross corridor thickness is outside the allowed range.")

        left = center_x - thickness // 2
        right = left + thickness
        top = center_y - thickness // 2
        bottom = top + thickness
        if left <= 0 or right >= self.width or top <= 0 or bottom >= self.height:
            raise ValueError("Cross corridor center leaves no usable room bands.")

        regions = (
            (corridor_ids[0], lambda x, y: left <= x < right and y < top),
            (corridor_ids[1], lambda x, y: left <= x < right and y >= bottom),
            (corridor_ids[2], lambda x, y: top <= y < bottom and x < center_x),
            (corridor_ids[3], lambda x, y: top <= y < bottom and x >= center_x),
        )
        for room_id, contains in regions:
            points = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if self.footprint[y * self.width + x] and contains(x, y)
            ]
            self._paint(room_id, "corridor", points)

    def fill_room_band(
        self,
        rooms: list[list[object]] | list[tuple[object, ...]],
        x: int,
        y: int,
        width: int,
        height: int,
        axis: str,
        corridor_id: str,
        door_side: str,
    ) -> None:
        if not isinstance(rooms, list) or not rooms:
            raise ValueError("A room band needs at least one [id, type, weight] item.")
        if axis not in {"x", "y"}:
            raise ValueError("Room band axis must be x or y.")
        if door_side not in {"top", "right", "bottom", "left"}:
            raise ValueError("Room band door_side is invalid.")
        if corridor_id not in self.room_indexes:
            raise ValueError(f"Room band references unknown corridor: {corridor_id}")
        if not all(isinstance(value, int) for value in (x, y, width, height)) or width <= 0 or height <= 0:
            raise ValueError("Room band bounds must be positive integers.")
        parsed: list[tuple[str, str, float]] = []
        for item in rooms:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError("Room band items must be [id, type, weight].")
            room_id, room_type, weight = item
            if not isinstance(room_id, str) or not isinstance(room_type, str):
                raise ValueError("Room band ids and types must be strings.")
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError("Room band weights must be positive numbers.")
            parsed.append((room_id, room_type, float(weight)))

        total_length = width if axis == "x" else height
        if len(parsed) > total_length:
            raise ValueError("Room band has more rooms than available pixels along its axis.")
        weight_sum = sum(item[2] for item in parsed)
        raw_lengths = [item[2] / weight_sum * total_length for item in parsed]
        lengths = [max(1, int(value)) for value in raw_lengths]
        while sum(lengths) < total_length:
            target = max(range(len(lengths)), key=lambda index: raw_lengths[index] - lengths[index])
            lengths[target] += 1
        while sum(lengths) > total_length:
            candidates = [index for index, value in enumerate(lengths) if value > 1]
            if not candidates:
                raise ValueError("Room band cannot allocate a positive length to every room.")
            target = max(candidates, key=lambda index: lengths[index] - raw_lengths[index])
            lengths[target] -= 1

        cursor = x if axis == "x" else y
        for (room_id, room_type, _), length in zip(parsed, lengths, strict=True):
            room_x = cursor if axis == "x" else x
            room_y = y if axis == "x" else cursor
            room_width = length if axis == "x" else width
            room_height = height if axis == "x" else length
            self.fill_rect(room_id, room_type, room_x, room_y, room_width, room_height)
            self._add_auto_shared_door(
                f"door-{room_id}",
                room_id,
                corridor_id,
                door_side,
                2,
            )
            cursor += length

    def _add_auto_shared_door(
        self,
        door_id: str,
        from_room: str,
        to_room: str,
        preferred_side: str,
        preferred_width: int,
    ) -> str:
        room = self.rooms[self.room_indexes[from_room]]
        bounds = room["bounds"]
        side_order = [preferred_side] + [
            side for side in ("top", "right", "bottom", "left") if side != preferred_side
        ]
        for side in side_order:
            horizontal = side in {"top", "bottom"}
            available = bounds["width"] if horizontal else bounds["height"]
            for width_cells in range(min(preferred_width, available), 0, -1):
                for offset in range(available - width_cells + 1):
                    if side == "top":
                        x, y, orientation = bounds["x"] + offset, bounds["y"], "horizontal"
                    elif side == "bottom":
                        x = bounds["x"] + offset
                        y = bounds["y"] + bounds["height"]
                        orientation = "horizontal"
                    elif side == "left":
                        x, y, orientation = bounds["x"], bounds["y"] + offset, "vertical"
                    else:
                        x = bounds["x"] + bounds["width"]
                        y = bounds["y"] + offset
                        orientation = "vertical"
                    try:
                        return self.add_door(
                            door_id,
                            from_room,
                            to_room,
                            x,
                            y,
                            orientation,
                            width_cells,
                        )
                    except ValueError as error:
                        if "is not fully on the shared boundary" not in str(error):
                            raise
        raise ValueError(f"Room {from_room} has no shared boundary with corridor {to_room}.")

    def add_door(
        self,
        door_id: str,
        from_room: str,
        to_room: str | None,
        x: int,
        y: int,
        orientation: str,
        width_cells: int = 2,
    ) -> str:
        self._validate_id(door_id, "Door id")
        if any(door.id == door_id for door in self.doors):
            raise ValueError(f"Duplicate door id: {door_id}")
        if from_room not in self.room_indexes:
            raise ValueError(f"Unknown door room: {from_room}")
        if to_room is not None and to_room not in self.room_indexes:
            raise ValueError(f"Unknown door room: {to_room}")
        if not isinstance(x, int) or not isinstance(y, int) or not isinstance(width_cells, int):
            raise ValueError("Door coordinates and width must be integers.")
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError("Door orientation must be horizontal or vertical.")
        if not 1 <= width_cells <= 6:
            raise ValueError("Door width must be 1 to 6 cells.")
        if not 0 <= x <= self.width or not 0 <= y <= self.height:
            raise ValueError("Door is outside the canvas.")
        from_index = self.room_indexes[from_room]
        to_index = self.room_indexes[to_room] if to_room is not None else -2

        def room_at(cell_x: int, cell_y: int) -> int:
            if cell_x < 0 or cell_x >= self.width or cell_y < 0 or cell_y >= self.height:
                return -2
            offset = cell_y * self.width + cell_x
            if not self.footprint[offset]:
                return -2
            return self.cells[offset]

        boundary_pairs: list[tuple[int, int]] = []
        for offset in range(width_cells):
            if orientation == "horizontal":
                boundary_pairs.append((room_at(x + offset, y - 1), room_at(x + offset, y)))
            else:
                boundary_pairs.append((room_at(x - 1, y + offset), room_at(x, y + offset)))
        expected = {from_index, to_index}
        if any({first, second} != expected for first, second in boundary_pairs):
            destination = to_room if to_room is not None else "outside"
            raise ValueError(
                f"Door {door_id} is not fully on the shared boundary between {from_room} and {destination}."
            )
        self.doors.append(Door(door_id, from_room, to_room, x, y, orientation, width_cells))
        return door_id

    def add_main_entrance(
        self,
        door_id: str,
        corridor_id: str,
        side: str,
        width_cells: int = 2,
    ) -> str:
        if corridor_id not in self.room_indexes:
            raise ValueError(f"Unknown entrance corridor: {corridor_id}")
        if side not in {"north", "east", "south", "west"}:
            raise ValueError("Entrance side must be north, east, south, or west.")
        if not isinstance(width_cells, int) or not 1 <= width_cells <= 6:
            raise ValueError("Entrance width must be 1 to 6 cells.")
        corridor_index = self.room_indexes[corridor_id]

        candidates: list[tuple[int, int, str]] = []
        for y in range(self.height):
            for x in range(self.width):
                if self.cells[y * self.width + x] != corridor_index:
                    continue
                if side == "north" and (y == 0 or not self.footprint[(y - 1) * self.width + x]):
                    candidates.append((x, y, "horizontal"))
                elif side == "south" and (
                    y == self.height - 1 or not self.footprint[(y + 1) * self.width + x]
                ):
                    candidates.append((x, y + 1, "horizontal"))
                elif side == "west" and (x == 0 or not self.footprint[y * self.width + x - 1]):
                    candidates.append((x, y, "vertical"))
                elif side == "east" and (
                    x == self.width - 1 or not self.footprint[y * self.width + x + 1]
                ):
                    candidates.append((x + 1, y, "vertical"))

        if not candidates:
            raise ValueError(f"Corridor {corridor_id} has no {side} exterior boundary.")
        fixed_index = 1 if side in {"north", "south"} else 0
        edge_value = min(item[fixed_index] for item in candidates) if side in {"north", "west"} else max(
            item[fixed_index] for item in candidates
        )
        edge = [item for item in candidates if item[fixed_index] == edge_value]
        edge.sort(key=lambda item: item[0] if item[2] == "horizontal" else item[1])
        for start in range(len(edge) - width_cells + 1):
            run = edge[start : start + width_cells]
            positions = [item[0] if item[2] == "horizontal" else item[1] for item in run]
            if positions == list(range(positions[0], positions[0] + width_cells)):
                x, y, orientation = run[0]
                return self.add_door(door_id, corridor_id, None, x, y, orientation, width_cells)
        raise ValueError(f"Corridor {corridor_id} has no {width_cells}-cell {side} entrance span.")

    def finish(self) -> dict[str, Any]:
        if not self.rooms:
            raise ValueError("The plan contains no rooms.")
        return {
            "width": self.width,
            "height": self.height,
            "meters_per_cell": self.meters_per_cell,
            "footprint": [1 if value else 0 for value in self.footprint],
            "cells": self.cells,
            "rooms": self.rooms,
            "doors": [door.__dict__ for door in self.doors],
        }
