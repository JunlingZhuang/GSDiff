# -*- coding: utf-8 -*-
"""Tool ① generate_colorblock: structured program -> Gemini -> colour-block PNG.

I/O contract:
    build_real_prompt(program) -> str          (pure, unit-testable)
    build_convert_prompt(program) -> str       (pure, unit-testable)
    generate_colorblock(program, client, out_path, generation_mode) -> Path

generation_mode="real2color": two Gemini calls per round —
    1. generate a realistic architectural plan (text-guided)
    2. convert it to a flat colour-block diagram (image-to-image)
"""
from __future__ import annotations

from pathlib import Path

from hfagent.schema.palette import BACKGROUND_RGB, ROOM_RGB, WALL_RGB, rgb_hex


def build_real_prompt(program: dict) -> str:
    """Prompt for pass 1: realistic architectural floor plan with labelled rooms."""
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

Before finalising, count the rooms of each type in your drawing and verify they match the program EXACTLY — do not add extra rooms of any type, do not drop any.

Output only the floor plan drawing."""


CONVERT_PROMPT_TEMPLATE = """Convert the floor plan drawing above into a flat colour-block diagram for computer-vision parsing.

HARD RULES:
- Preserve every room's position, size and count EXACTLY as drawn above. Do not add, remove, merge or move rooms.
- Use the text label inside each room to determine its type, then colour it with that type's legend colour below.
- REMOVE all door swings, door arcs, door openings, window symbols, fixtures and text labels. Close every wall opening — walls become solid unbroken black lines.
- Each room becomes ONE single continuous flat block of its legend colour below. A room must never be split into multiple blocks by lines or symbols.
- The corridor is ONE single continuous block of its colour, even where doors used to be.
- No gradients, no textures, no text, no furniture.
- Walls: pure black {wall} lines of uniform thickness. Background outside the building: pure white {bg}.

ROOM LEGEND (exact fill colours):
{legend}

Output only the converted diagram image."""


def build_convert_prompt(program: dict) -> str:
    """Prompt for pass 2: convert realistic plan to flat colour-block diagram."""
    legend = "\n".join(
        f"- {r['type'].replace('_', ' ')}: {rgb_hex(ROOM_RGB[r['type']])}"
        for r in program["rooms"]
    )
    return CONVERT_PROMPT_TEMPLATE.format(
        wall=rgb_hex(WALL_RGB), bg=rgb_hex(BACKGROUND_RGB), legend=legend
    )


def generate_colorblock(
    program: dict,
    client,
    out_path: str | Path,
    generation_mode: str = "real2color",
) -> Path:
    """Generate a colour-block floor plan PNG from a structured program.

    generation_mode="real2color":
        Pass 1 — realistic architectural plan (saved as *.real.png)
        Pass 2 — convert to flat colour-block diagram (saved as out_path)
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if generation_mode == "real2color":
        real_png = client.generate_image(build_real_prompt(program))
        out.with_suffix(".real.png").write_bytes(real_png)
        colorblock = client.generate_image([real_png, build_convert_prompt(program)])
    else:
        raise ValueError(f"Unknown generation_mode '{generation_mode}' — use 'real2color'")

    out.write_bytes(colorblock)
    return out
