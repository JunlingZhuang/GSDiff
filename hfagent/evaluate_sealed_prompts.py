# -*- coding: utf-8 -*-
"""Controlled Gemini Flash sweep for sealed-room linework prompts.

This experiment intentionally does not use or mutate the production linework
prompt. It generates the same programs with five no-door prompt variants, saves
every exact prompt and image, and records geometry-only tracer diagnostics for
later manual review.

From the repository root::

    python -m hfagent.evaluate_sealed_prompts

The default matrix is 5 prompts x 4 programs x 2 repeats = 40 images.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from hfagent.evaluate import DEFAULT_PROGRAMS, load_programs
from hfagent.floor_plan_generate import load_config
from hfagent.llm import GeminiClient
from hfagent.tools.floor_plan_generator import area_sqft
from hfagent.tools.linework_tracer import trace_linework
from hfagent.tools.room_adjacency_extractor import room_instance_ids


DEFAULT_PROGRAM_NAMES = (
    "clinic-small",
    "hospital-floor",
    "inpatient-ward",
    "hospital-tower-floor",
)
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"


def _room_lines(program: dict) -> str:
    areas = {room["type"]: area_sqft(room) for room in program["rooms"]}
    lines = []
    for node in room_instance_ids(program):
        area = areas.get(node.type)
        lines.append(f"- {node.id}" + (f" (about {area} sq ft)" if area else ""))
    return "\n".join(lines)


def _shared_intro(program: dict, *, call_it_floor_plan: bool) -> str:
    subject = "top-down floor plan" if call_it_floor_plan else "top-down orthogonal partition map"
    return f"Create a {subject} for a {program['building_type']} on a pure white 16:9 canvas."


def _prompt_current_short(program: dict) -> str:
    """P1: exact wording of the current short sealed production candidate."""
    return f"""Draw a top-down floor plan of a {program['building_type']} as a simple black line diagram.

- Rooms are rectangles packed side by side along the corridor (the corridor is just a long thin room). Together they fill the whole building with no leftover space.
- Every wall is a solid black band of ONE uniform thickness, about 12 px, perfectly horizontal or vertical.
- This plan has NO doors: every room, including the corridor, is completely closed on all four sides. No gap, no break, no opening, no swing arc anywhere in any wall.
- Write each room's name once in its centre, plain black text, exactly as listed below. No other text, no furniture, no windows, no shading.
- Everything outside the building is pure white.

Rooms (one room per name; size them roughly by the areas):
{_room_lines(program)}

Output only the drawing."""


def _prompt_positive_boundaries(program: dict) -> str:
    """P2: describe complete boundaries positively instead of relying on negation."""
    return f"""{_shared_intro(program, call_it_floor_plan=True)}

Draw one connected building made only from closed rectangular room cells:
- Each room cell has four continuous solid black wall bands joined at all four corners.
- Each wall band runs continuously from corner to corner. Shared walls between neighbouring cells are continuous too.
- The corridor is another closed rectangular cell, not an open passage.
- Use one uniform wall thickness of about 12 px. All walls are horizontal or vertical.
- Fill the footprint with the listed cells, with pure white interiors and pure white outside.
- Put exactly one listed name at the centre of its cell. Add nothing except these labels and walls.

There are no doors or openings in this diagram. Every cell boundary remains solid and closed.

Required room cells, exactly once each:
{_room_lines(program)}

Output only the finished black-and-white diagram."""


def _prompt_partition_map(program: dict) -> str:
    """P3: remove architectural floor-plan language that may trigger a door prior."""
    return f"""{_shared_intro(program, call_it_floor_plan=False)}

This is a labelled closed-cell partition map, not a usable building drawing.
- Represent every item below as one separate fully enclosed white rectangle.
- Pack the rectangles into one connected orthogonal footprint. Rectangles may share black edges.
- Every edge is a continuous solid black band about 12 px thick, joined cleanly at its corners.
- Treat names containing corridor exactly like all other names: corridor is a sealed labelled rectangle.
- Use only horizontal and vertical edges.
- Draw no breaks in edges and no symbols on edges.
- Draw no furniture, fixtures, windows, decoration, dimensions or text other than the exact names.

Required closed cells, one per line and one occurrence each:
{_room_lines(program)}

Return only the partition map image."""


def _prompt_construction_steps(program: dict) -> str:
    """P4: give an explicit drawing order that completes walls before labels."""
    return f"""{_shared_intro(program, call_it_floor_plan=False)}

Construct the diagram in this order:
1. Draw one closed orthogonal outer boundary using a uniform 12 px solid black band.
2. Divide its interior into exactly the listed number of rectangular cells using solid horizontal and vertical wall bands.
3. Extend every internal wall until it touches another wall. Join every endpoint so each cell has a complete closed perimeter.
4. Keep every wall continuous. Do not cut an entrance, doorway, passage or other gap into any wall.
5. After all cells are closed, write exactly one required name in the centre of each cell.

Use pure white cell interiors and pure white outside. Draw no windows, furniture, fixtures, dimensions, shading or extra text. A corridor name identifies an ordinary sealed cell; it does not authorize openings.

Cells to draw exactly once:
{_room_lines(program)}

Output only the completed diagram."""


def _prompt_final_wall_audit(program: dict) -> str:
    """P5: repeat a concrete wall audit immediately before output."""
    return f"""NO OPENINGS ARE ALLOWED. Draw a sealed black-wall room layout for a {program['building_type']}.

- Use a top-down 16:9 view and one connected orthogonal footprint.
- Give every listed room its own rectangular white interior and a complete solid black perimeter.
- Use one uniform wall band about 12 px thick. Walls are horizontal or vertical and meet exactly at corners and junctions.
- The corridor is fully enclosed by continuous walls on every side, like every other room.
- Use each exact room name once. Draw only walls and room names: no doors, door leaves, swing arcs, windows, furniture, dimensions, shading or extra marks.

Exact rooms:
{_room_lines(program)}

Before returning the image, inspect every room edge from corner to corner. Fill every white break in every black wall. Return the image only after every listed room, including every corridor, is completely sealed.

Output only the drawing."""


# ---- round 2: five iterations on the round-1 winner p3-partition-map ---------------
# p3's residual failures (manual_review.csv): over-partitioning + duplicate labels at
# scale (hospital-floor 31 vs 26, tower 93/99 vs 82), a few open endpoints / U-shaped
# incomplete cells at tower scale. Each iteration keeps p3's winning core (closed-cell
# partition-map reframe, corridor = ordinary sealed cell, uniform 12 px orthogonal
# edges) and attacks ONE failure mode.


def _prompt_p3_count_lock(program: dict) -> str:
    """P6: p3 + exact cell-count anchoring against over-partitioning / duplicate labels."""
    total = len(room_instance_ids(program))
    return f"""{_shared_intro(program, call_it_floor_plan=False)}

This is a labelled closed-cell partition map, not a usable building drawing. It contains EXACTLY {total} cells — one cell per name below, no more, no fewer.
- Represent every name below as one separate fully enclosed white rectangle. A name is used exactly once; no name is ever repeated, and no unnamed cell exists.
- Pack the rectangles into one connected orthogonal footprint. Rectangles may share black edges.
- Every edge is a continuous solid black band about 12 px thick, joined cleanly at its corners.
- Treat names containing corridor exactly like all other names: corridor is a sealed labelled rectangle.
- Use only horizontal and vertical edges. Draw no breaks in edges and no symbols on edges.
- Draw no furniture, fixtures, windows, decoration, dimensions or text other than the exact names.

Required closed cells, one per line and one occurrence each ({total} total):
{_room_lines(program)}

Count the labelled cells before finishing: exactly {total}. Return only the partition map image."""


def _prompt_p3_edge_rule(program: dict) -> str:
    """P7: p3 + endpoint discipline (an edge never ends in open space) against open endpoints."""
    return f"""{_shared_intro(program, call_it_floor_plan=False)}

This is a labelled closed-cell partition map, not a usable building drawing.
- Represent every name below as one separate fully enclosed white rectangle.
- Pack the rectangles into one connected orthogonal footprint. Rectangles may share black edges.
- Every edge is a continuous solid black band about 12 px thick, joined cleanly at its corners.
- EVERY edge begins on another edge and ends on another edge — at a corner or a T-junction. An edge never stops in open white space, and no cell's outline has a missing side.
- Treat names containing corridor exactly like all other names: corridor is a sealed labelled rectangle.
- Use only horizontal and vertical edges. Draw no breaks in edges and no symbols on edges.
- Draw no furniture, fixtures, windows, decoration, dimensions or text other than the exact names.

Required closed cells, one per line and one occurrence each:
{_room_lines(program)}

Return only the partition map image."""


def _prompt_p3_banded(program: dict) -> str:
    """P8: p3 + prescriptive banded macro-layout to tame 82-cell complexity."""
    return f"""{_shared_intro(program, call_it_floor_plan=False)}

This is a labelled closed-cell partition map, not a usable building drawing. Arrange it as horizontal bands: each band is a row of rectangles side by side, and the rows are stacked; each corridor name is one long thin band of its own. This banded arrangement keeps every rectangle simple and fully closed.
- Represent every name below as one separate fully enclosed white rectangle.
- Rectangles may share black edges; rows share the horizontal edges between them.
- Every edge is a continuous solid black band about 12 px thick, joined cleanly at its corners.
- Treat names containing corridor exactly like all other names: corridor is a sealed labelled rectangle.
- Use only horizontal and vertical edges. Draw no breaks in edges and no symbols on edges.
- Draw no furniture, fixtures, windows, decoration, dimensions or text other than the exact names.

Required closed cells, one per line and one occurrence each:
{_room_lines(program)}

Return only the partition map image."""


def _prompt_p3_verify(program: dict) -> str:
    """P9: p3 + one light closing self-check (p5 showed heavy audit framing backfires)."""
    return f"""{_shared_intro(program, call_it_floor_plan=False)}

This is a labelled closed-cell partition map, not a usable building drawing.
- Represent every item below as one separate fully enclosed white rectangle.
- Pack the rectangles into one connected orthogonal footprint. Rectangles may share black edges.
- Every edge is a continuous solid black band about 12 px thick, joined cleanly at its corners.
- Treat names containing corridor exactly like all other names: corridor is a sealed labelled rectangle.
- Use only horizontal and vertical edges. Draw no breaks in edges and no symbols on edges.
- Draw no furniture, fixtures, windows, decoration, dimensions or text other than the exact names.

Required closed cells, one per line and one occurrence each:
{_room_lines(program)}

Finally, confirm each rectangle's outline is one closed ring of black. Return only the partition map image."""


def _prompt_p3_compact(program: dict) -> str:
    """P10: p3's core compressed to the fewest words (does brevity hold at scale?)."""
    total = len(room_instance_ids(program))
    return f"""{_shared_intro(program, call_it_floor_plan=False)}

A labelled closed-cell partition map: {total} fully enclosed white rectangles packed into one connected orthogonal footprint, every edge a continuous solid black band about 12 px thick, horizontal or vertical only, cleanly joined. Each name below gets exactly one sealed rectangle labelled at its centre — corridor names included. Nothing else appears in the image.

{_room_lines(program)}

Return only the partition map image."""


# ---- round 3: single-pass finals — architectural framing, narrative prose ----------
# Round-2 verdict: partition-map family (p3/p8) seals but stops looking like
# architecture; architectural p2 keeps realism but leaks at 82. Finals stay
# ARCHITECTURAL and single-pass, written as narrative paragraphs (docs: a
# descriptive paragraph beats a bullet list), room list names-only (the
# "(about X m2)" hints leaked into labels in rounds 1-2). Two levers:
# double-loaded-corridor wing structure in prose, and the user's
# "architectural semantic map" framing; each also at thinking_level=high
# (docs: 3.1-flash supports minimal (default) | high).


def _room_names_inline(program: dict) -> str:
    return ", ".join(node.id for node in room_instance_ids(program))


def _prompt_wings_prose(program: dict) -> str:
    """P11: narrative prose + double-loaded corridor wings, sealed, names-only."""
    return f"""Create a top-down floor plan of a {program['building_type']}, drawn as a clean black-and-white line diagram on a pure white 16:9 canvas.

The floor is organised the way a real building of this kind is: the corridors form the spine of the layout, and along both sides of every corridor runs a row of rooms — double-loaded corridor wings that join into one connected orthogonal footprint. In this diagram every space, including each corridor, is drawn as its own fully enclosed rectangle: all four of its walls are continuous solid black bands about 12 px thick, perfectly horizontal or vertical, meeting cleanly at the corners, and neighbouring spaces share the wall between them. Every space carries exactly one name at its centre in plain black text, and the diagram contains nothing else — only sealed rooms, shared walls and names, with pure white everywhere outside the building.

The spaces are: {_room_names_inline(program)}.

Output only the drawing."""


def _prompt_semantic_map(program: dict) -> str:
    """P12: narrative prose, 'architectural semantic map' framing, sealed, names-only."""
    return f"""Create a top-down architectural semantic map of a {program['building_type']} on a pure white 16:9 canvas.

A semantic map shows the floor's spatial structure the way an architect's zoning diagram does: every functional space appears as one sealed rectangle bounded by continuous solid black lines about 12 px thick, all perfectly horizontal or vertical, joined cleanly at the corners. The spaces are arranged exactly as the real floor plan would arrange them — corridors run as long spines with rooms lining both sides, and the whole floor forms one connected orthogonal footprint — but because this is a semantic map rather than a working drawing, every boundary is unbroken and each space is a completely closed cell. The only text is one name at the centre of each space, in plain black; there is nothing else in the image, and everything outside the building is pure white.

The spaces are: {_room_names_inline(program)}.

Output only the map."""


PROMPT_VARIANTS: dict[str, Callable[[dict], str]] = {
    "p1-current-short": _prompt_current_short,
    "p2-positive-boundaries": _prompt_positive_boundaries,
    "p3-partition-map": _prompt_partition_map,
    "p4-construction-steps": _prompt_construction_steps,
    "p5-final-wall-audit": _prompt_final_wall_audit,
    "p6-p3-count-lock": _prompt_p3_count_lock,
    "p7-p3-edge-rule": _prompt_p3_edge_rule,
    "p8-p3-banded": _prompt_p3_banded,
    "p9-p3-verify": _prompt_p3_verify,
    "p10-p3-compact": _prompt_p3_compact,
    "p11-wings-prose": _prompt_wings_prose,
    "p12-semantic-map": _prompt_semantic_map,
    "p13-wings-high": _prompt_wings_prose,
    "p14-semantic-high": _prompt_semantic_map,
}

# thinking_level per variant (gemini-3.1-flash-image: minimal (default) | high).
VARIANT_THINKING: dict[str, str] = {
    "p13-wings-high": "high",
    "p14-semantic-high": "high",
}


MANUAL_FIELDS = (
    "variant",
    "program",
    "repeat",
    "required_rooms",
    "enclosed_regions",
    "count_exact",
    "strict_seal_success",
    "orthogonal_axis_aligned",
    "label_or_count_notes",
    "notes",
)


def count_enclosed_regions(png: bytes, *, threshold: int = 230, min_area: int = 1000) -> int:
    """Count parser-recoverable white regions enclosed away from the canvas edge."""
    gray = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("could not decode generated image")

    white = (gray >= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    height, width = gray.shape
    enclosed = 0
    for component in range(1, count):
        x, y, w, h, area = stats[component]
        touches_canvas = x == 0 or y == 0 or x + w == width or y + h == height
        if not touches_canvas and area >= min_area:
            enclosed += 1
    return enclosed


def _write_manual_template(path: Path, rows: list[dict]) -> None:
    existing = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("variant"), row.get("program"), row.get("repeat"))
                existing[key] = row
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_FIELDS)
        writer.writeheader()
        for row in rows:
            key = (row.get("variant"), row.get("program"), str(row.get("repeat")))
            old = existing.get(key, {})
            writer.writerow({
                field: row.get(field, "") if field in {
                    "variant", "program", "repeat", "required_rooms",
                    "enclosed_regions", "count_exact",
                } else old.get(field, "")
                for field in MANUAL_FIELDS
            })


def _write_contact_sheets(
    out_dir: Path,
    variants: list[str],
    names: list[str],
    repeats: int,
) -> None:
    """Write one 4-column x repeat-row visual index per prompt variant."""
    from PIL import Image, ImageDraw

    cell_w, cell_h, label_h = 688, 384, 32
    sheets_dir = out_dir / "contact_sheets"
    sheets_dir.mkdir(exist_ok=True)
    for variant in variants:
        sheet = Image.new("RGB", (cell_w * len(names), (cell_h + label_h) * repeats), "white")
        draw = ImageDraw.Draw(sheet)
        for repeat in range(1, repeats + 1):
            for column, name in enumerate(names):
                image_path = out_dir / variant / f"{name}-r{repeat}" / "image.png"
                if not image_path.exists():
                    continue
                with Image.open(image_path) as source:
                    thumb = source.convert("RGB")
                    thumb.thumbnail((cell_w, cell_h))
                    x = column * cell_w + (cell_w - thumb.width) // 2
                    y0 = (repeat - 1) * (cell_h + label_h)
                    y = y0 + label_h + (cell_h - thumb.height) // 2
                    sheet.paste(thumb, (x, y))
                draw.text((column * cell_w + 8, y0 + 8), f"{name} r{repeat}", fill="black")
        sheet.save(sheets_dir / f"{variant}.png")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", default=DEFAULT_PROGRAMS)
    parser.add_argument("--program", "-p", default=",".join(DEFAULT_PROGRAM_NAMES))
    parser.add_argument("--variants", default=",".join(PROMPT_VARIANTS))
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    programs = load_programs(Path(args.programs))
    names = _parse_csv(args.program)
    variants = _parse_csv(args.variants)
    missing_programs = [name for name in names if name not in programs]
    missing_variants = [name for name in variants if name not in PROMPT_VARIANTS]
    if missing_programs:
        raise SystemExit(f"unknown programs: {missing_programs}")
    if missing_variants:
        raise SystemExit(f"unknown variants: {missing_variants}")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    cfg = load_config()
    client = GeminiClient(
        # This sweep never calls generate_text/generate_json. Supplying a sentinel
        # avoids an unnecessary models.list() request before the first image.
        text_model="unused-by-sealed-prompt-sweep",
        image_model=args.image_model,
        image_size=cfg["image_size"],
        image_aspect=cfg["image_aspect"],
    )
    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "out" / "eval" /
        f"{time.strftime('%Y%m%d-%H%M%S')}-sealed-prompt-sweep"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "image_model": client.image_model,
        "image_size": client.image_size,
        "image_aspect": client.image_aspect,
        "programs": names,
        "variants": variants,
        "repeat": args.repeat,
        "total_images": len(names) * len(variants) * args.repeat,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"output: {out_dir}")

    summary: list[dict] = []
    manual_rows: list[dict] = []
    for variant in variants:
        builder = PROMPT_VARIANTS[variant]
        for name in names:
            program = programs[name]
            required_rooms = sum(room.get("count", 1) for room in program["rooms"])
            prompt = builder(program)
            for repeat in range(1, args.repeat + 1):
                label = f"{variant}/{name}-r{repeat}"
                work_dir = out_dir / variant / f"{name}-r{repeat}"
                work_dir.mkdir(parents=True, exist_ok=True)
                (work_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                image_path = work_dir / "image.png"
                print(f"[{len(summary) + 1}/{manifest['total_images']}] {label}", flush=True)

                row = {
                    "variant": variant,
                    "program": name,
                    "repeat": repeat,
                    "required_rooms": required_rooms,
                }
                try:
                    if image_path.exists():
                        png = image_path.read_bytes()
                        print("  reuse existing image.png", flush=True)
                    else:
                        png = client.generate_image(
                            prompt, thinking_level=VARIANT_THINKING.get(variant)
                        )
                        image_path.write_bytes(png)
                    enclosed_regions = count_enclosed_regions(png)
                    row["enclosed_regions"] = enclosed_regions
                    row["count_exact"] = enclosed_regions == required_rooms
                    trace = trace_linework(png)
                    for filename, payload in trace.artifacts.items():
                        (work_dir / filename).write_bytes(payload)
                    row["trace"] = trace.diagnostics
                    row["image"] = str(image_path.relative_to(out_dir)).replace("\\", "/")
                    (work_dir / "trace.json").write_text(
                        json.dumps(trace.diagnostics, indent=2), encoding="utf-8"
                    )
                    print(
                        f"  enclosed={enclosed_regions}/{required_rooms} "
                        f"trace_rooms={trace.diagnostics['rooms_closed']} "
                        f"gaps={trace.diagnostics['sub_gaps']} "
                        f"doors={trace.diagnostics['doors_detected']}",
                        flush=True,
                    )
                except Exception as exc:
                    row["error"] = str(exc)
                    print(f"  ERROR: {exc}", flush=True)

                summary.append(row)
                manual_rows.append(row)
                (out_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                _write_manual_template(out_dir / "manual_review.csv", manual_rows)

    print(f"done: {len(summary)} image rows -> {out_dir}")
    _write_contact_sheets(out_dir, variants, names, args.repeat)
    print(f"contact sheets: {out_dir / 'contact_sheets'}")
    print(f"manual review template: {out_dir / 'manual_review.csv'}")


if __name__ == "__main__":
    main()
