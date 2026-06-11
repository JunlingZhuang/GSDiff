# -*- coding: utf-8 -*-
"""Colour legend shared by ALL three tools.

The same palette drives: the prompt sent to Gemini (generate_colorblock), the
deterministic renderer (render_plan) and the parser (cv_parse). Keeping it in
one place is what makes the round-trip test meaningful.

Colours are chosen to be far apart in RGB so nearest-colour classification is
robust to VLM colour drift.
"""
from __future__ import annotations

# room type -> RGB
ROOM_RGB: dict[str, tuple[int, int, int]] = {
    "patient_room": (66, 133, 244),   # blue
    "corridor": (251, 188, 4),        # yellow
    "exam_room": (52, 168, 83),       # green
    "waiting": (255, 109, 1),         # orange
    "toilet": (171, 71, 188),         # purple
    "nurse_station": (233, 30, 99),   # pink
    "storage": (121, 85, 72),         # brown
    "office": (0, 188, 212),          # cyan
}

WALL_RGB: tuple[int, int, int] = (0, 0, 0)        # walls / outlines
BACKGROUND_RGB: tuple[int, int, int] = (255, 255, 255)  # outside

ROOM_TYPES = list(ROOM_RGB.keys())


def rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def nearest_room_type(rgb: tuple[float, float, float]) -> tuple[str, float]:
    """Nearest legend colour for a pixel; returns (room_type, distance)."""
    best, bd = "", float("inf")
    for t, c in ROOM_RGB.items():
        d = sum((a - b) ** 2 for a, b in zip(rgb, c)) ** 0.5
        if d < bd:
            best, bd = t, d
    return best, bd
