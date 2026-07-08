# -*- coding: utf-8 -*-
"""Rasterize a traced px-unit Plan into the neutral seed-grid JSON that
pixel-plan-ai's /api/refine accepts (trace mode / mode 3).

The raster follows pixel-plan-ai's model: rooms TILE the footprint and walls
are implicit cell boundaries, so after painting each room polygon the wall
strips between rooms are absorbed by a bounded label expansion. Doors are not
taken from the trace's px geometry (only logical room_a/room_b edges survive
``trace_linework``); instead each edge is placed centred on the LONGEST shared
boundary run of its two rooms in the raster, which guarantees the door lands
on a real boundary and survives the pixel-plan sandbox door validator.
"""
from __future__ import annotations

import numpy as np
import cv2

from hfagent.schema.plan import Plan
from hfagent.schema.roomgraph import RoomGraph

GRID_MIN_W, GRID_MAX_W = 32, 160
GRID_MIN_H, GRID_MAX_H = 24, 120
# physical door widths; converted to cells at the chosen scale
INTERIOR_DOOR_METERS = 0.9
EXTERIOR_DOOR_METERS = 1.8
FALLBACK_METERS_PER_CELL = 0.3


def _safe_identifier(value: str, fallback: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in (value or "").strip())
    return cleaned[:80] or fallback


def _expand_rooms_into_walls(grid: np.ndarray, footprint: np.ndarray, rounds: int) -> None:
    """Grow room labels into unassigned footprint cells (the traced wall strips)
    so rooms tile the footprint the way pixel-plan-ai plans do. Bounded rounds
    keep genuinely unassigned pockets (courtyards, unlabeled areas) as -1."""
    for _ in range(max(0, rounds)):
        unassigned = (grid == -1) & footprint
        if not unassigned.any():
            return
        best = np.full(grid.shape, np.iinfo(np.int32).max, np.int32)
        for shifted in (
            np.roll(grid, 1, axis=0),
            np.roll(grid, -1, axis=0),
            np.roll(grid, 1, axis=1),
            np.roll(grid, -1, axis=1),
        ):
            neighbor = shifted.copy()
            neighbor[neighbor < 0] = np.iinfo(np.int32).max
            best = np.minimum(best, neighbor)
        claimable = unassigned & (best != np.iinfo(np.int32).max)
        grid[claimable] = best[claimable]


def _boundary_runs(grid: np.ndarray, index_a: int, index_b: int) -> list[tuple[str, int, int, int]]:
    """Contiguous shared-boundary runs between two labels (or a label and -2
    outside). Returns (orientation, x, y, length) per run using pixel-plan-ai's
    door convention: a vertical door at (x, y) separates (x-1, y) / (x, y);
    a horizontal door at (x, y) separates (x, y-1) / (x, y)."""
    height, width = grid.shape
    pair = {index_a, index_b}
    runs: list[tuple[str, int, int, int]] = []

    padded = np.full((height + 2, width + 2), -2, np.int32)
    padded[1:-1, 1:-1] = grid

    for x in range(width + 1):
        run_start, run_len = -1, 0
        for y in range(height + 1):
            left = padded[y + 1, x]
            right = padded[y + 1, x + 1]
            on_boundary = y < height and left != right and {int(left), int(right)} == pair
            if on_boundary:
                if run_start < 0:
                    run_start = y
                run_len += 1
            elif run_start >= 0:
                runs.append(("vertical", x, run_start, run_len))
                run_start, run_len = -1, 0
    for y in range(height + 1):
        run_start, run_len = -1, 0
        for x in range(width + 1):
            above = padded[y, x + 1]
            below = padded[y + 1, x + 1]
            on_boundary = x < width and above != below and {int(above), int(below)} == pair
            if on_boundary:
                if run_start < 0:
                    run_start = x
                run_len += 1
            elif run_start >= 0:
                runs.append(("horizontal", run_start, y, run_len))
                run_start, run_len = -1, 0
    return runs


def rasterize_plan(
    plan: Plan,
    room_graph: RoomGraph,
    *,
    meters_per_pixel: float | None = None,
    wall_width_px: float | None = None,
    target_width_cells: int = 96,
) -> dict:
    """The traced plan as pixel-plan-ai seed JSON (plus a ``meta`` report)."""
    if not plan.rooms:
        raise ValueError("rasterize_plan: the traced plan has no rooms")

    min_x, min_y, max_x, max_y = plan.bounds()
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    target_width_cells = max(GRID_MIN_W, min(GRID_MAX_W, target_width_cells))
    cell_px = span_x / (target_width_cells - 2)
    if round(span_y / cell_px) + 2 > GRID_MAX_H:
        cell_px = span_y / (GRID_MAX_H - 2)
    width = max(GRID_MIN_W, min(GRID_MAX_W, int(round(span_x / cell_px)) + 2))
    height = max(GRID_MIN_H, min(GRID_MAX_H, int(round(span_y / cell_px)) + 2))

    grid = np.full((height, width), -2, np.int32)
    kept_rooms: list[dict] = []
    dropped: list[str] = []
    seen_ids: set[str] = set()
    for room in plan.rooms:
        points = np.array(
            [[(x - min_x) / cell_px + 1.0, (y - min_y) / cell_px + 1.0] for x, y in room.polygon],
            np.float64,
        )
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        mask &= (grid == -2).astype(np.uint8)  # first room wins contested cells
        if not mask.any():
            dropped.append(room.id)
            continue
        index = len(kept_rooms)
        grid[mask.astype(bool)] = index
        room_id = _safe_identifier(room.name or room.id, f"room_{index}")
        if room_id in seen_ids:
            room_id = _safe_identifier(f"{room_id}_{index}", f"room_{index}")
        seen_ids.add(room_id)
        kept_rooms.append({"id": room_id, "type": _safe_identifier(room.type, "unknown"), "trace_id": room.id})

    if not kept_rooms:
        raise ValueError("rasterize_plan: no room rasterized to at least one cell")

    wall_cells = max(1, int(round((wall_width_px or cell_px) / cell_px)))
    room_mask = (grid >= 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * wall_cells + 1, 2 * wall_cells + 1))
    footprint = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, kernel).astype(bool)
    grid[footprint & (grid == -2)] = -1
    _expand_rooms_into_walls(grid, footprint, rounds=wall_cells + 2)
    # cells the expansion could not reach stay -1; shrink footprint back to
    # assigned cells so exterior doors see -2 right at the room edge
    grid[(grid == -1)] = -2
    footprint = grid >= 0

    meters_per_cell = round(
        cell_px * meters_per_pixel if meters_per_pixel else FALLBACK_METERS_PER_CELL, 4
    )

    trace_to_index = {room["trace_id"]: index for index, room in enumerate(kept_rooms)}
    doors: list[dict] = []
    unplaced: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    for edge in room_graph.doors:
        pair_key = tuple(sorted((edge.room_a, edge.room_b)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        exterior = "exterior" in pair_key
        interior_ids = [name for name in pair_key if name != "exterior"]
        if not interior_ids or any(name not in trace_to_index for name in interior_ids):
            unplaced.append("-".join(pair_key))
            continue
        index_a = trace_to_index[interior_ids[0]]
        index_b = -2 if exterior else trace_to_index[interior_ids[1]]
        runs = _boundary_runs(grid, index_a, index_b)
        if not runs:
            unplaced.append("-".join(pair_key))
            continue
        orientation, run_x, run_y, run_len = max(runs, key=lambda run: run[3])
        door_meters = EXTERIOR_DOOR_METERS if exterior else INTERIOR_DOOR_METERS
        width_cells = max(1, min(6, min(run_len, int(round(door_meters / meters_per_cell)))))
        offset = (run_len - width_cells) // 2
        door = {
            "id": f"door_{len(doors)}",
            "from_room": kept_rooms[index_a]["id"],
            "to_room": None if exterior else kept_rooms[index_b]["id"],
            "orientation": orientation,
            "width_cells": width_cells,
            "x": run_x + (offset if orientation == "horizontal" else 0),
            "y": run_y + (offset if orientation == "vertical" else 0),
        }
        doors.append(door)

    return {
        "width": width,
        "height": height,
        "meters_per_cell": meters_per_cell,
        "cells": [int(v) for v in grid.flatten()],
        "rooms": [{"id": room["id"], "type": room["type"]} for room in kept_rooms],
        "doors": doors,
        "meta": {
            "cell_px": round(cell_px, 3),
            "scale_calibrated": meters_per_pixel is not None,
            "rooms_dropped": dropped,
            "doors_unplaced": unplaced,
        },
    }
