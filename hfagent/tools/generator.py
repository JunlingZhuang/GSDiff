# -*- coding: utf-8 -*-
"""Tool ①: structured program -> colour-block PNG via two Gemini calls.

Public API:
    text_to_real_plan(program, client) -> bytes
    real_plan_to_colorblock(real_png, program, client) -> bytes
    generate_colorblock(program, client, out_path, generation_mode) -> Path

generation_mode="real2color" pipeline:
    text_to_real_plan      — text prompt -> realistic architectural floor plan (PNG bytes)
    real_plan_to_colorblock — realistic plan image -> flat colour-block diagram (PNG bytes)
    generate_colorblock    — orchestrates both calls, writes *.real.png + out_path to disk
"""
from __future__ import annotations

from pathlib import Path

from hfagent.schema.palette import BACKGROUND_RGB, ROOM_RGB, WALL_RGB, rgb_hex


# ── prompt builders (pure, unit-testable) ────────────────────────────────────

def build_real_prompt(program: dict) -> str:
    rooms = "\n".join(
        f"- {r.get('count', 1)} x {r['type'].replace('_', ' ')}"
        + (f", each about {r['approx_area_m2']} m2" if r.get("approx_area_m2") else "")
        for r in program["rooms"]
    )
    adjacency = "\n".join(
        f"- every {a} opens onto a {b}" for a, b in program.get("adjacency", [])
    )
    return f"""Design a realistic, professionally laid-out 2D architectural floor plan (top-down) for a {program['building_type']}.

Draw it as a clean architectural drawing: orthogonal walls, sensible room proportions, realistic circulation. No furniture, no dimension lines. LABEL every room with its exact type name from the program below (small plain text inside the room) — the labels are required for a later processing step.

ROOM PROGRAM (exact counts are a hard requirement):
{rooms}

CIRCULATION:
{adjacency if adjacency else '- (none)'}

Before finalising, count the rooms of each type in your drawing and verify they match the program EXACTLY.

Output only the floor plan drawing."""


_CONVERT_PROMPT_TEMPLATE = """Convert the floor plan drawing above into a flat colour-block diagram for computer-vision parsing.

HARD RULES:
- Preserve every room's position, size and count EXACTLY as drawn above. Do not add, remove, merge or move rooms.
- Use the text label inside each room to determine its type, then fill it with that type's legend colour below.
- REMOVE all door swings, door arcs, window symbols, fixtures and text labels. Close every wall opening — walls become solid unbroken black lines.
- Each room is ONE single continuous flat block of its legend colour. A room must never be split by lines or symbols.
- The corridor is ONE single continuous block of its colour, even where doors used to be.
- No gradients, no textures, no text, no furniture.
- Walls: pure black {wall}. Background outside the building: pure white {bg}.

ROOM LEGEND (exact fill colours):
{legend}

Output only the converted diagram image."""


def build_convert_prompt(program: dict) -> str:
    legend = "\n".join(
        f"- {r['type'].replace('_', ' ')}: {rgb_hex(ROOM_RGB[r['type']])}"
        for r in program["rooms"]
    )
    return _CONVERT_PROMPT_TEMPLATE.format(
        wall=rgb_hex(WALL_RGB), bg=rgb_hex(BACKGROUND_RGB), legend=legend
    )


# ── generation steps ─────────────────────────────────────────────────────────

def text_to_real_plan(program: dict, client) -> bytes:
    """Pass 1 — text prompt → realistic architectural floor plan (PNG bytes)."""
    return client.generate_image(build_real_prompt(program))


def real_plan_to_colorblock(real_png: bytes, program: dict, client) -> bytes:
    """Pass 2 — realistic plan image → flat colour-block diagram (PNG bytes)."""
    return client.generate_image([real_png, build_convert_prompt(program)])


# ── file-writing orchestrator ─────────────────────────────────────────────────

def generate_colorblock(
    program: dict,
    client,
    out_path: str | Path,
    generation_mode: str = "real2color",
) -> Path:
    """Run the full real2color pipeline and write outputs to disk.

    Writes:
        out_path          — colour-block PNG (parser input)
        out_path.real.png — realistic intermediate (correction-loop input)
    """
    if generation_mode != "real2color":
        raise ValueError(f"Unknown generation_mode '{generation_mode}' — use 'real2color'")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    real_png = text_to_real_plan(program, client)
    out.with_suffix(".real.png").write_bytes(real_png)

    colorblock = real_plan_to_colorblock(real_png, program, client)
    out.write_bytes(colorblock)

    return out
