# -*- coding: utf-8 -*-
"""Tool ①: structured program -> colour-block PNG via the real2color pipeline.

Public API:
    FloorPlanGenerator(program, client)
        .generate_real_plan() -> bytes              pass 1: text -> realistic plan
        .to_colorblock(real_png) -> bytes            pass 2: realistic plan -> colour-block
        .extract_room_adjacency(real_png) -> RoomGraph   doors read from the realistic plan
        .run(out_path) -> Path                       all three + write files to disk

Prompt builders are module-level so tests can verify their content independently:
    build_real_prompt(program) -> str
    build_convert_prompt(program) -> str
"""
from __future__ import annotations

from pathlib import Path

from hfagent.schema.palette import BACKGROUND_RGB, ROOM_RGB, WALL_RGB, rgb_hex
from hfagent.schema.roomgraph import RoomGraph
from hfagent.tools.room_adjacency_extractor import extract_room_adjacency


# ── prompt builders (pure, unit-testable) ────────────────────────────────────

def _program_blocks(program: dict) -> tuple[str, str]:
    """Render the program's room list and circulation into prompt text blocks."""
    lines = []
    for r in program["rooms"]:
        count = r.get("count", 1)
        display = r["type"].replace("_", " ")
        area = f", each about {r['approx_area_m2']} m2" if r.get("approx_area_m2") else ""
        if count > 1:
            # Numbered labels enable unambiguous door extraction later.
            label_note = f" — label each one {r['type']}_1, {r['type']}_2 … {r['type']}_{count}"
        else:
            label_note = f" — label it {r['type']}"
        lines.append(f"- {count} x {display}{area}{label_note}")
    adjacency = "\n".join(f"- every {a} opens onto a {b}" for a, b in program.get("adjacency", []))
    return "\n".join(lines), (adjacency or "- (none)")


# Shared drawing rules — identical whether the footprint is free (build_real_prompt)
# or given (build_boundary_prompt), so both produce the same kind of realflow plan.
_PLAN_RULES = """VIEW
- Strictly TOP-DOWN orthographic plan view (looking straight down, bird's-eye).
- NO perspective, NO 3D, NO isometric/axonometric, NO camera tilt or vanishing points.
- Do NOT rotate or skew the building. Its outer walls run purely horizontal and vertical, axis-aligned to the image edges.

WALLS
- Draw every wall as a SOLID BLACK FILLED band (poché) — continuous black fill between and around rooms, NOT a thin single outline and NOT a hairline.
- ONE fixed, uniform thickness for EVERY wall — exterior walls, interior partitions and party walls are ALL the same thickness. Do NOT make exterior or load-bearing walls thicker than the others.
- Keep walls THIN and uniform: a thin filled black band about 12 px wide on the ~1400 px-wide output (≈1% of the image width) — never a hairline, never thicker than ~20 px.
- Only right angles: no diagonal, curved or angled walls. Every room is fully enclosed.

LAYOUT
- Every room is rectangular (or a clean L-shape) with orthogonal corners.
- Rooms completely TILE the building footprint — no gaps, no overlaps, no leftover background slivers inside the outer walls.
- Organise rooms along a clear corridor/circulation spine; size rooms roughly by the areas given (the corridor is a long thin band).
- One single connected building footprint, not scattered blocks.

DOORS (required — they are read back from this drawing)
- Draw a door at every connection as a SIMPLE gap in the wall with ONE plain quarter-circle swing arc — nothing more. No door frame, no panelled leaf, no threshold, no hinge marks, no door tag or number.
- Every enclosed room must have at least one door to a corridor / circulation space so it is reachable.
- Put doors only where two spaces should connect (see CIRCULATION).

LABELS
- One label per room, centred, small plain black text, the EXACT instance name from the program below.
- No other text anywhere.

EXCLUDE — keep it a clean SCHEMATIC plan, NOT a construction / working drawing. Do NOT draw any of:
- furniture, fixtures, equipment, sanitary ware, beds, sinks, desks;
- windows of any kind;
- door construction detail (frames, leaves, thresholds, swing-radius dimensions);
- dimension lines or strings, grid / column lines, section / elevation / detail markers;
- hatching, fill patterns, textures, gradients or shadows (walls stay flat solid black);
- schedules, legends, keynotes, callouts, scale bars, north arrows, title blocks.
- Background outside the building is pure white and empty."""


def _compose_prompt(intro: str, program: dict) -> str:
    rooms, adjacency = _program_blocks(program)
    return f"""{intro}

{_PLAN_RULES}

ROOM PROGRAM (exact counts and labels are hard requirements):
{rooms}

CIRCULATION:
{adjacency}

Before finalising, count the rooms of each type and verify they match the program EXACTLY.

Output only the floor plan drawing."""


def build_real_prompt(program: dict) -> str:
    """Pass-1 prompt, FREE footprint: text program -> realflow plan."""
    intro = (
        f"Draw a single 2D architectural floor plan for a {program['building_type']}. "
        "Follow every rule below — each is a hard requirement, not a preference."
    )
    return _compose_prompt(intro, program)


def build_boundary_prompt(program: dict) -> str:
    """Pass-1 prompt, GIVEN footprint: the image above is the building outline;
    fill it with the program -> realflow plan. Same downstream as build_real_prompt."""
    intro = (
        f"The image above is the EXACT building outline (footprint) for a {program['building_type']}. "
        "Draw a single 2D architectural floor plan that fills this outline. "
        "Follow every rule below — each is a hard requirement.\n\n"
        "BOUNDARY (hard requirement)\n"
        "- Your building's OUTER walls MUST follow the given outline EXACTLY — same shape and "
        "proportions, every notch and step. Do not change, simplify, rotate or rescale it.\n"
        "- Lay ALL rooms INSIDE that outline; together they fill the whole footprint with no empty leftover area."
    )
    return _compose_prompt(intro, program)


_CONVERT_PROMPT_TEMPLATE = """Convert the floor plan drawing above into a flat colour-block diagram for computer-vision parsing.

HARD RULES:
- Preserve every room's position, size and count EXACTLY as drawn above. Do not add, remove, merge or move rooms.
- Use the text label inside each room to identify its base type (strip any trailing _number), then fill it with that type's legend colour below.
  Example: "patient_room_3" → fill with the patient_room colour.
- REMOVE all text labels, door swings, door arcs, window symbols and fixtures. Close every wall opening — walls become solid unbroken black lines.
- Each room is ONE single continuous flat block of its legend colour. A room must never be split by any line, symbol or text — this clean fill is what the parser segments into blocks.
- The corridor is ONE single continuous block of its colour, even where doors used to be.
- No gradients, no textures, no text, no furniture.
- Walls: pure black {wall}, THIN and uniform — keep the same thin wall thickness as the drawing above (about 12 px / ~1% of image width). Never thicken walls into wide black bands.
- Background outside the building: pure white {bg}.

ROOM LEGEND (exact fill colours — apply by base room type, ignore instance numbers):
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


# ── generator class ───────────────────────────────────────────────────────────

class FloorPlanGenerator:
    """Generates a colour-block floor plan PNG from a structured room program.

    Holds program + client so they don't need to be threaded through every call.
    The correction loop in pipeline.py calls to_colorblock() directly with an
    already-corrected real_png, bypassing generate_real_plan().
    """

    def __init__(self, program: dict, client, boundary: bytes | None = None):
        self.program = program
        self.client = client
        self.boundary = boundary  # given building outline (PNG bytes), or None for a free footprint

    def generate_real_plan(self) -> bytes:
        """Pass 1 — program -> realflow plan (PNG bytes).

        Two input entries, one output: with a boundary image the outer walls follow
        that exact footprint; without one the footprint is free. Everything after
        this (to_colorblock, parsing, doors, walls) is identical for both.
        """
        if self.boundary is not None:
            return self.client.generate_image([self.boundary, build_boundary_prompt(self.program)])
        return self.client.generate_image(build_real_prompt(self.program))

    def to_colorblock(self, real_png: bytes) -> bytes:
        """Pass 2 — realistic plan image -> flat colour-block diagram (PNG bytes)."""
        return self.client.generate_image([real_png, build_convert_prompt(self.program)])

    def extract_room_adjacency(self, real_png: bytes) -> RoomGraph:
        """Read the room adjacency graph (rooms + doors) from the realistic plan image.

        Doors live only in the realistic drawing — the colour-block conversion
        erases them — so this must run on real_png, not the reconstructed plan.
        """
        return extract_room_adjacency(real_png, self.client, self.program)

    def run(self, out_path: str | Path) -> Path:
        """Run the full single-shot flow and write outputs to disk.

        Writes:
            out_path            — colour-block PNG (parser input)
            out_path.real.png   — realistic intermediate (correction-loop input)
            out_path.graph.json — room graph (rooms + doors)
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        real_png = self.generate_real_plan()
        out.with_suffix(".real.png").write_bytes(real_png)

        colorblock = self.to_colorblock(real_png)
        out.write_bytes(colorblock)

        graph = self.extract_room_adjacency(real_png)
        out.with_suffix(".graph.json").write_text(graph.model_dump_json(indent=2), encoding="utf-8")

        return out
