from __future__ import annotations

import json
from typing import Any

# Room-scale sibling of gemini.build_prompt, kept deliberately short: the domain is one ICU
# room, not a whole floor. The validator (room_validator.validate_room) is the sole judge, so
# this prompt teaches the coordinate contract and restates the rules the validator enforces.

RESULT_CONTRACT = """result = {
    "room": {
        "width_ft": float,
        "depth_ft": float,
        "door": {"wall": "N|S|E|W", "offset_ft": float, "width_ft": float}
    },
    "assets": [
        {
            "id": unique_string,
            "type": catalog_type,
            "x_ft": float, "y_ft": float,
            "w_ft": float, "d_ft": float,
            "rotation_deg": 0 | 90 | 180 | 270,
            "wall": "N|S|E|W" or None
        }
    ]
}"""

COORDINATE_RULES = """Coordinate system (feet, 0.25 ft grid):
- x runs 0 at the West wall to width_ft at the East wall. y runs 0 at the South wall to depth_ft at the North wall.
- (x_ft, y_ft) is the SOUTH-WEST (minimum) corner of an asset; its footprint covers [x_ft, x_ft + w_ft] by [y_ft, y_ft + d_ft].
- Every asset must lie fully inside the room. Snap all coordinates and sizes to 0.25 ft.
- w_ft and d_ft are the footprint AS PLACED. rotation_deg 90 or 270 swaps the catalog width and depth; 0 and 180 keep them.
- A wall-anchored asset must be flush to its declared wall (within 0.25 ft): N means y_ft + d_ft = depth_ft, S means y_ft = 0, E means x_ft + w_ft = width_ft, W means x_ft = 0. Set "wall" to that wall; use None for floor, ceiling, and mobile assets.
- door.offset_ft is the door CENTER measured along its wall from the origin corner (from x=0 for N/S walls, from y=0 for E/W walls)."""

EXECUTION_ENVIRONMENT = """Execution environment:
- Authorized imports: numpy, shapely, scipy, networkx, PIL, math, random, statistics, itertools, and collections.
- Functions, classes, loops, comprehensions, and normal control flow are allowed. The executor blocks files, network, processes, and dynamic execution.
- Keep the program deterministic (seed any random generator). Do not print the result and do not save files.
- Assign the final layout to a top-level variable named result using the exact contract below."""

BEHAVIORAL_RULES = """Behavioral rules:
- Write one complete standalone Python program every attempt; return the entire program, never a patch.
- Prefer deriving placements with code (compute clearances, wall offsets, and the bed axis) over hard-coded magic coordinates.
- A deterministic validator is the sole judge. Never claim the layout passes; state what you changed and which named checks it targets.
- On a repair, read the validator delta and failed checks, then make the MINIMAL edit that fixes the named failures while leaving passing placements untouched.

Layout guidance (mirrors the validator, all values in feet):
- Put the bed head (short side) against one wall that is at least the minimum headwall width, leaving a head-to-wall gap inside the target band. That wall is the headwall.
- Keep the bed foot clearance and both long-side clearances (one transfer side, one other side) free of floor and wall assets; mobile assets may sit in a SIDE clearance but never in the foot clearance.
- Place the two ceiling booms flanking the bed head on OPPOSITE sides of the bed long axis, within mount reach and coverage of the head.
- Put the patient monitor and IV pole near the bed head on one side (the equipment side); put the visitor chair on the OPPOSITE side.
- Keep casework off the headwall and clear of the door swing. Put the handwash sink within reach of the door.
- Give the door at least the minimum clear width and keep a clear path of the required width from the door to the bed foot."""


def catalog_table(catalog: dict[str, Any]) -> str:
    lines = []
    for entry in catalog.get("assets", []):
        width, depth = entry["footprint_ft"]
        lines.append(
            f"- {entry['type']} ({entry['label']}): footprint {width} x {depth} ft, "
            f"anchor {entry['anchor']}, need {entry.get('count_required', 1)}. {entry.get('notes', '')}".rstrip()
        )
    return "\n".join(lines)


def build_room_prompt(room_request: dict[str, Any], rules: dict[str, Any], catalog: dict[str, Any]) -> str:
    width = room_request.get("width_ft")
    depth = room_request.get("depth_ft")
    user_prompt = str(room_request.get("prompt", "") or "").strip()
    repair_context = room_request.get("repair_context")
    previous_code = room_request.get("previous_code")

    user_line = f"\nUser request: {user_prompt}" if user_prompt else ""
    repair_block = repair_context or "This is the first attempt."
    if previous_code:
        repair_block = (
            f"{repair_block}\n\nPrevious complete program:\n{previous_code}\n"
            "Diagnose the exact failure and return a complete revised program."
        )

    return f"""You are an ICU room layout coding agent.
Design a single-patient ICU room, then write a complete executable Python program that assigns the layout to a top-level variable named result.
Return JSON matching the required response schema (code, strategy, assumptions).

{EXECUTION_ENVIRONMENT}

Result contract:
{RESULT_CONTRACT}

{COORDINATE_RULES}

Asset catalog (place exactly the required count of each type):
{catalog_table(catalog)}

ICU schematic planning rules (the validator enforces these exact values):
{json.dumps(rules, indent=2)}

{BEHAVIORAL_RULES}

Room to lay out: width_ft = {width}, depth_ft = {depth}.{user_line}

Repair context:
{repair_block}
"""
