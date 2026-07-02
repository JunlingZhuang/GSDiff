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
from typing import Callable

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


# Doors section of the shared rules — swapped for _NO_DOORS_RULES when a mode is
# configured with doors_in_plan=false (sealed rooms; no door symbols to mis-parse).
_DOORS_RULES = """DOORS (required — they are read back from this drawing)
- Draw a door at every connection as a SIMPLE gap in the wall with ONE plain quarter-circle swing arc — nothing more. No door frame, no panelled leaf, no threshold, no hinge marks, no door tag or number.
- Every enclosed room must have at least one door to a corridor / circulation space so it is reachable.
- Put doors only where two spaces should connect (see CIRCULATION)."""

_NO_DOORS_RULES = """DOORS — NONE. This plan has NO doors at all:
- Every room is COMPLETELY sealed by unbroken walls: no gaps, no openings, no door leaves, no quarter-circle swing arcs anywhere in the drawing.
- Walls run continuous straight through where a door would normally be — do not leave any passage between rooms or to the corridor.
- The CIRCULATION list below is adjacency guidance for the layout only, NOT openings."""

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

{doors_rules}

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


def _compose_prompt(
    intro: str,
    program: dict,
    extra_rules: str = "",
    plan_rules: str = _PLAN_RULES,
    doors: bool = True,
) -> str:
    rooms, adjacency = _program_blocks(program)
    plan_rules = plan_rules.replace("{doors_rules}", _DOORS_RULES if doors else _NO_DOORS_RULES)
    if not doors:
        # adjacency stays layout guidance, but "opens onto" implies an opening
        adjacency = adjacency.replace("opens onto", "sits next to")
    return f"""{intro}

{plan_rules}
{extra_rules}

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


_LINEWORK_RULES = """

CV LINEWORK PROFILE (hard requirements)
- This drawing will be parsed by OCR and deterministic computer vision. Use only pure white room interiors and crisp black marks.
- Room labels must use the EXACT underscore-and-number spelling from the program, for example office_1 and patient_room_2. Draw every label once, on one horizontal line, in a large plain sans-serif font, centred well away from walls and doors. Never omit, duplicate, rotate, wrap or abbreviate a label.
- Area values are layout guidance only. Do not print square metres, dimensions or any text below the room label.
- Do NOT draw windows, glazing lines or openings that resemble windows. Exterior walls are continuous except at a real entrance door.
- Every door uses the same simple symbol: one clear wall gap, one thin straight door leaf and one thin quarter-circle swing arc. Use single-leaf hinged doors only. Do not draw double doors, sliding doors, pocket doors or decorative door frames.
- A wall gap is allowed only for a door. Apart from walls, the standard door symbol and room labels, draw no other black lines.
"""

_LINEWORK_RULES_NO_DOORS = """

CV LINEWORK PROFILE (hard requirements)
- This drawing will be parsed by deterministic computer vision. Use only pure white room interiors and crisp black marks.
- Room labels must use the EXACT underscore-and-number spelling from the program, for example office_1 and patient_room_2. Draw every label once, on one horizontal line, in a large plain sans-serif font, centred well away from walls. Never omit, duplicate, rotate, wrap or abbreviate a label.
- Area values are layout guidance only. Do not print square metres, dimensions or any text below the room label.
- Do NOT draw windows, glazing lines or openings that resemble windows. ALL walls — exterior and interior — are fully continuous unbroken bands.
- Do NOT draw any door: no wall gaps, no door leaves, no swing arcs. There is no opening anywhere in any wall.
- Apart from walls and room labels, draw no other black lines.
"""


def build_linework_prompt(program: dict, boundary: bool = False, doors: bool = True) -> str:
    """Parser-oriented real-plan prompt used only by structure_mode=linework.

    ``doors=False`` (config: modes.linework.doors_in_plan) draws every room fully
    sealed — no door symbols at all. Door detection/bridging then has nothing to do
    and room closure is pure wall tracing.
    """
    if boundary:
        intro = (
            f"The image above is the EXACT building outline (footprint) for a {program['building_type']}. "
            "Draw a single clean 2D architectural line plan that fills this outline. "
            "The outer walls follow every step and notch of the supplied outline exactly."
        )
    else:
        intro = (
            f"Draw a single clean 2D architectural line plan for a {program['building_type']}. "
            "Follow every rule below as a hard requirement."
        )
    linework_plan_rules = _PLAN_RULES.replace(
        "centred, small plain black text",
        "centred, large plain black text",
    )
    return _compose_prompt(
        intro,
        program,
        extra_rules=_LINEWORK_RULES if doors else _LINEWORK_RULES_NO_DOORS,
        plan_rules=linework_plan_rules,
        doors=doors,
    )


_CONVERT_PROMPT_TEMPLATE = """Repaint the floor plan above as a flat colour diagram.

- Fill each room with one solid colour from the legend below, chosen by the room's text label (ignore any number, e.g. "patient_room_3" uses the patient_room colour).
- Also fill the small area inside each door's swing arc with the colour of the room that door belongs to, so every room is one continuous block of colour with no notch or gap where a door was.
- Keep the walls as solid black {wall} lines. Keep everything outside the building white {bg}.
- The result has only the flat legend colours, black walls and white outside — no text, no furniture, no door arcs, no window marks, no shading.

Colour legend (by room type):
{legend}

Output only the colour image."""


def build_convert_prompt(program: dict) -> str:
    legend = "\n".join(
        f"- {r['type'].replace('_', ' ')}: {rgb_hex(ROOM_RGB[r['type']])}"
        for r in program["rooms"]
    )
    return _CONVERT_PROMPT_TEMPLATE.format(
        wall=rgb_hex(WALL_RGB), bg=rgb_hex(BACKGROUND_RGB), legend=legend
    )


# ── single-pass direct colour-block (experimental) ───────────────────────────
# Skips the realistic plan entirely: the model draws the flat colour-block diagram
# straight from the program. Removes the realistic->colour-block conversion step,
# which is where door-arc blobs and count drift creep in.

_DIRECT_COLORBLOCK_RULES = """VIEW
- Strictly TOP-DOWN orthographic plan view, looking straight down (bird's-eye).
- NO perspective, NO 3D, NO tilt. The building is axis-aligned: every wall runs purely horizontal or vertical, parallel to the image edges.

ROOMS AS FLAT COLOUR BLOCKS
- Draw each room as ONE solid block of flat colour — a filled rectangle (or a clean L-shape) in the single legend colour for that room's type.
- Use ONLY the exact flat legend colours. Inside a room: no shading, gradients, textures, patterns or coloured outlines — just the one flat fill.
- Rooms of the SAME type share the SAME colour; that is expected. Two same-type rooms that touch MUST still have a black wall between them so they read as two separate blocks, never one merged block.

WALLS
- Between every two rooms, and around the whole building, draw a SOLID BLACK band of ONE uniform thickness (about 12 px on the ~1400 px-wide image). Right angles only; every room fully enclosed.

LAYOUT
- Rooms completely TILE the footprint — no gaps, no white slivers inside the outer walls, no overlaps.
- Organise rooms along a clear corridor/circulation spine (the corridor is itself a coloured block); size each room roughly by its given area.
- One single connected building footprint.

NOTHING ELSE — this is a clean colour-block diagram, not a realistic drawing:
- NO text, NO room labels, NO numbers anywhere.
- NO doors, NO door gaps, NO door swing arcs.
- NO furniture, fixtures, windows, dimension lines, grid lines, legends, north arrows or title blocks.
- Everything outside the building is pure white."""


def build_direct_colorblock_prompt(program: dict, boundary: bool = False) -> str:
    """Single-pass prompt: program -> flat colour-block diagram (no realistic plan)."""
    legend_lines = []
    for r in program["rooms"]:
        count = r.get("count", 1)
        display = r["type"].replace("_", " ")
        area = f", each about {r['approx_area_m2']} m2" if r.get("approx_area_m2") else ""
        legend_lines.append(f"- {count} x {display} — fill colour {rgb_hex(ROOM_RGB[r['type']])}{area}")
    rooms_block = "\n".join(legend_lines)
    adjacency = "\n".join(
        f"- a {a} sits next to a {b}" for a, b in program.get("adjacency", [])
    ) or "- (none)"

    if boundary:
        intro = (
            f"The image above is the EXACT building outline for a {program['building_type']}. "
            "Draw, directly inside that outline, a FLAT COLOUR-BLOCK floor plan: a simplified "
            "schematic where each room is one solid block of colour separated by black walls. "
            "The outer walls follow the given outline exactly. Follow every rule below as a hard requirement."
        )
    else:
        intro = (
            f"Draw a 2D architectural floor plan for a {program['building_type']} directly as a "
            "FLAT COLOUR-BLOCK diagram: a simplified schematic where each room is one solid block "
            "of colour separated by black walls. Follow every rule below as a hard requirement."
        )

    return f"""{intro}

{_DIRECT_COLORBLOCK_RULES}

ROOM PROGRAM — draw exactly these rooms, each filled with the colour shown (same type = same colour, but each instance is its own block separated by walls):
{rooms_block}

Walls are {rgb_hex(WALL_RGB)}; everything outside the building is {rgb_hex(BACKGROUND_RGB)}.

CIRCULATION (which rooms sit next to each other along the corridor):
{adjacency}

Before finishing, count the colour blocks of each type and verify they match the program EXACTLY.

Output only the colour-block diagram."""


# ── generator class ───────────────────────────────────────────────────────────

class FloorPlanGenerator:
    """Generates a colour-block floor plan PNG from a structured room program.

    Holds program + client so they don't need to be threaded through every call.
    The correction loop in pipeline.py calls to_colorblock() directly with an
    already-corrected real_png, bypassing generate_real_plan().
    """

    def __init__(
        self,
        program: dict,
        client,
        boundary: bytes | None = None,
        prompt_logger: Callable[[str, str], None] | None = None,
        drawing_mode: str = "standard",
        doors_in_plan: bool = True,
    ):
        self.program = program
        self.client = client
        self.boundary = boundary  # given building outline (PNG bytes), or None for a free footprint
        self.prompt_logger = prompt_logger
        self.drawing_mode = drawing_mode
        self.doors_in_plan = doors_in_plan  # False: sealed rooms, no door symbols (linework)

    def _log_prompt(self, stage: str, prompt: str) -> None:
        if self.prompt_logger is not None:
            self.prompt_logger(stage, prompt)

    def generate_real_plan(self) -> bytes:
        """Pass 1 — program -> realflow plan (PNG bytes).

        Two input entries, one output: with a boundary image the outer walls follow
        that exact footprint; without one the footprint is free. Everything after
        this (to_colorblock, parsing, doors, walls) is identical for both.
        """
        if self.drawing_mode == "linework":
            prompt = build_linework_prompt(
                self.program, boundary=self.boundary is not None, doors=self.doors_in_plan
            )
            self._log_prompt("CV linework real plan", prompt)
            contents = [self.boundary, prompt] if self.boundary is not None else prompt
            return self.client.generate_image(contents)
        if self.boundary is not None:
            prompt = build_boundary_prompt(self.program)
            self._log_prompt("real plan with boundary", prompt)
            return self.client.generate_image([self.boundary, prompt])
        prompt = build_real_prompt(self.program)
        self._log_prompt("real plan", prompt)
        return self.client.generate_image(prompt)

    def to_colorblock(self, real_png: bytes) -> bytes:
        """Pass 2 — realistic plan image -> flat colour-block diagram (PNG bytes)."""
        prompt = build_convert_prompt(self.program)
        self._log_prompt("colour-block conversion", prompt)
        return self.client.generate_image([real_png, prompt])

    def generate_colorblock_direct(self) -> bytes:
        """Single pass — program text -> flat colour-block diagram (PNG bytes).

        Experimental alternative to generate_real_plan + to_colorblock: the model
        draws the colour-block directly, skipping the realistic intermediate (and the
        conversion step that introduces door-arc blobs / count drift).
        """
        prompt = build_direct_colorblock_prompt(self.program, boundary=self.boundary is not None)
        self._log_prompt("direct colour-block", prompt)
        contents = [self.boundary, prompt] if self.boundary is not None else prompt
        return self.client.generate_image(contents)

    def extract_room_adjacency(self, real_png: bytes) -> RoomGraph:
        """Read the room adjacency graph (rooms + doors) from the realistic plan image.

        Doors live only in the realistic drawing — the colour-block conversion
        erases them — so this must run on real_png, not the reconstructed plan.
        """
        return extract_room_adjacency(
            real_png,
            self.client,
            self.program,
            prompt_logger=self.prompt_logger,
        )

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
