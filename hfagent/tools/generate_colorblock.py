# -*- coding: utf-8 -*-
"""Tool ① generate_colorblock: structured program -> Gemini -> colour-block PNG.

I/O contract:
    build_prompt(program) -> str                       (pure, unit-testable)
    generate_colorblock(program, client, out_path)     (client mockable)

`program` is the structured requirement the upstream LLM step will eventually
produce; Phase 0 hand-writes them:
    {
      "building_type": "community clinic",
      "rooms": [{"type": "exam_room", "count": 3}, {"type": "waiting", "count": 1}],
      "adjacency": [["exam_room", "corridor"], ["waiting", "corridor"]],
    }
"""
from __future__ import annotations

from pathlib import Path

from hfagent.schema.palette import BACKGROUND_RGB, ROOM_RGB, WALL_RGB, rgb_hex


def build_prompt(program: dict) -> str:
    """Build the image-generation prompt. Example — for this program:

        {"building_type": "tiny clinic",
         "rooms": [{"type": "exam_room", "count": 2}, {"type": "corridor", "count": 1}],
         "adjacency": [["exam_room", "corridor"]]}

    the returned prompt is:

        Generate a single top-down 2D architectural floor plan diagram of a tiny clinic.

        STRICT STYLE RULES — this image will be parsed by a computer vision program:
        - Flat solid colour blocks only. No gradients, no shadows, no textures, no 3D,
          no furniture, no text, no labels, no dimension lines, no people, no icons.
        - Background outside the building: pure white #FFFFFF.
        - Walls: pure black #000000 lines of uniform thickness separating every room.
        - Every room is one flat rectangle (or L-shape) filled with EXACTLY its legend
          colour below. Do not blend or restyle the colours.
        - Rooms must tile the building footprint completely with no gaps and no
          overlaps. Orthogonal (axis-aligned) layout only.

        ROOM LEGEND (exact fill colours):
        - exam_room: fill colour exactly #34A853 (2 room(s))
        - corridor: fill colour exactly #FBBC04 (1 room(s))

        ADJACENCY REQUIREMENTS:
        - exam_room must share a wall with corridor

        COUNT CHECK — the room counts are a hard requirement, not a suggestion:
        - exactly 2 block(s) of #34A853 (exam_room)
        - exactly 1 block(s) of #FBBC04 (corridor)
        Before finalising the image, count the colour blocks of each colour and verify
        they match EXACTLY. Do not add extra rooms to fill space — make the remaining
        rooms larger instead.

        Include one corridor connecting the rooms if a corridor is listed. Output only
        the diagram image.
    """
    def legend_line(r: dict) -> str:
        line = f"- {r['type']}: fill colour exactly {rgb_hex(ROOM_RGB[r['type']])} ({r.get('count', 1)} room(s))"
        if r.get("approx_area_m2"):  # optional size hint for more complex programs
            line += f", each roughly {r['approx_area_m2']} m2 relative to the other rooms"
        return line

    legend = "\n".join(legend_line(r) for r in program["rooms"])
    adjacency = "\n".join(f"- {a} must share a wall with {b}" for a, b in program.get("adjacency", []))
    counts = "\n".join(
        f"- exactly {r.get('count', 1)} block(s) of {rgb_hex(ROOM_RGB[r['type']])} ({r['type']})"
        for r in program["rooms"]
    )
    return f"""Generate a single top-down 2D architectural floor plan diagram of a {program['building_type']}.

STRICT STYLE RULES — this image will be parsed by a computer vision program:
- Flat solid colour blocks only. No gradients, no shadows, no textures, no 3D, no furniture, no text, no labels, no dimension lines, no people, no icons.
- Background outside the building: pure white {rgb_hex(BACKGROUND_RGB)}.
- Walls: pure black {rgb_hex(WALL_RGB)} lines of uniform thickness separating every room.
- Every room is one flat rectangle (or L-shape) filled with EXACTLY its legend colour below. Do not blend or restyle the colours.
- Rooms must tile the building footprint completely with no gaps and no overlaps. Orthogonal (axis-aligned) layout only.

ROOM LEGEND (exact fill colours):
{legend}

ADJACENCY REQUIREMENTS:
{adjacency if adjacency else '- (none)'}

COUNT CHECK — the room counts are a hard requirement, not a suggestion:
{counts}
Before finalising the image, count the colour blocks of each colour and verify they match EXACTLY. Do not add extra rooms to fill space — make the remaining rooms larger instead.

Include one corridor connecting the rooms if a corridor is listed. Output only the diagram image."""


def build_realistic_prompt(program: dict) -> str:
    """Pass 1 of the two-pass pipeline: ask for a REALISTIC architectural plan
    first, so the layout comes from the model's real-floorplan prior instead of
    its diagram prior."""
    rooms = "\n".join(
        f"- {r.get('count', 1)} x {r['type'].replace('_', ' ')}"
        + (f", each about {r['approx_area_m2']} m2" if r.get("approx_area_m2") else "")
        for r in program["rooms"]
    )
    adjacency = "\n".join(f"- every {a} opens onto a {b}" for a, b in program.get("adjacency", []))
    return f"""Design a realistic, professionally laid-out 2D architectural floor plan (top-down) for a {program['building_type']}.

Draw it as a clean architectural drawing: orthogonal walls, sensible room proportions, realistic circulation. No furniture, no dimension lines. LABEL every room with its exact type name from the program below (small plain text inside the room) — the labels are required for a later processing step.

ROOM PROGRAM (exact counts are a hard requirement):
{rooms}

CIRCULATION:
{adjacency if adjacency else '- (none)'}

Before finalising, count the rooms of each type in your drawing and verify they match the program EXACTLY — do not add extra rooms of any type, do not drop any.

Output only the floor plan drawing."""


CONVERT_PROMPT_HEADER = """Convert the floor plan drawing above into a flat colour-block diagram for computer-vision parsing.

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
    """Pass 2 of the two-pass pipeline: faithful image-to-image conversion."""
    legend = "\n".join(
        f"- {r['type'].replace('_', ' ')}: {rgb_hex(ROOM_RGB[r['type']])}" for r in program["rooms"]
    )
    return CONVERT_PROMPT_HEADER.format(
        wall=rgb_hex(WALL_RGB), bg=rgb_hex(BACKGROUND_RGB), legend=legend
    )


def generate_colorblock(
    program: dict, client, out_path: str | Path, pipeline: str = "direct"
) -> Path:
    """client: hfagent.llm.GeminiClient or any object with .generate_image(contents)->bytes.

    pipeline:
      "direct"   program -> colour-block image (one call)
      "two-pass" program -> realistic plan image -> convert to colour-block
                 (the realistic intermediate is saved next to out_path as
                 *.realistic.png for inspection)
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if pipeline == "two-pass":
        realistic = client.generate_image(build_realistic_prompt(program))
        out.with_suffix(".realistic.png").write_bytes(realistic)
        png = client.generate_image([realistic, build_convert_prompt(program)])
    else:
        png = client.generate_image(build_prompt(program))
    out.write_bytes(png)
    return out
