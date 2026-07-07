from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Any

from constants import area_for
from healthcare_rules import rules_for_prompt, uses_healthcare_rules

DEFAULT_MODEL = "gemini-3.5-flash"

SYSTEM_INSTRUCTION = """You are a U.S. healthcare floor-plan coding agent.
You produce complete executable Python, then use exact executor and validator feedback to revise it.
Treat room function, zoning, circulation access, physical clear dimensions, area, compactness, and aspect ratio as first-class design constraints.
Never trade away clinical room usability merely to satisfy room counts. Distinguish patient care, clinical support, staff support, public support, and building support spaces.
An inpatient en-suite toilet should default to an inboard corner near the room entrance and corridor, preserving the exterior wall for the patient bed and window. The patient-space pixels may form an L around that toilet, while a rectangular bed/clear zone remains and the patient space keeps direct corridor access. Nested and outboard variants are allowed only when the design request or program calls for them. Exam rooms, offices, storage rooms, nurse stations, and waiting rooms are independent spaces and must never be carved into or wrapped by a patient room.
Return one final implementation only. The code must not contain abandoned layout alternatives, repeated redefinitions of room lists, self-correction commentary, or draft coordinates. Put reasoning in the strategy field, not inside the code.
The result is a schematic planning study, not construction documentation or a claim of code compliance."""

FEET_PER_METER = 3.280839895
SQUARE_FEET_PER_SQUARE_METER = 10.763910417


def normalize_usage_metadata(model: str, raw: Any) -> dict[str, Any]:
    metadata = raw if isinstance(raw, dict) else {}
    prompt_tokens = int(metadata.get("promptTokenCount", 0) or 0)
    candidate_tokens = int(metadata.get("candidatesTokenCount", 0) or 0)
    thinking_tokens = int(metadata.get("thoughtsTokenCount", 0) or 0)
    total_tokens = int(metadata.get("totalTokenCount", 0) or 0)
    if not total_tokens:
        total_tokens = prompt_tokens + candidate_tokens + thinking_tokens
    price_per_million: tuple[float, float] | None = None
    if model.startswith("gemini-3.1-pro"):
        price_per_million = (2.0, 12.0)
    elif model.startswith("gemini-3.5-flash"):
        price_per_million = (2.7, 16.2)
    elif model.startswith("gemini-3.1-flash-lite"):
        price_per_million = (0.25, 1.5)
    estimated_cost = None
    if price_per_million is not None:
        input_price, output_price = price_per_million
        estimated_cost = round(
            (prompt_tokens * input_price + (candidate_tokens + thinking_tokens) * output_price) / 1_000_000,
            6,
        )
    return {
        "prompt_tokens": prompt_tokens,
        "candidate_tokens": candidate_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
    }


def pixel_planning_targets(program: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    meters_per_cell = float(options["meters_per_cell"])
    cell_area_m2 = meters_per_cell**2
    cell_area_ft2 = cell_area_m2 * SQUARE_FEET_PER_SQUARE_METER
    cell_dimension_ft = meters_per_cell * FEET_PER_METER
    rules = rules_for_prompt(program) if uses_healthcare_rules(program) else {}
    room_rules = rules.get("room_types", {})
    targets: dict[str, Any] = {}
    for room in program["rooms"]:
        room_type = room["type"]
        if room_type in {"corridor", "circulation"}:
            continue
        target_area_m2 = area_for(room)
        target_cells = target_area_m2 / cell_area_m2
        rule = room_rules.get(room_type, {})
        tolerance = float(rule.get("area_tolerance_fraction", 0.35))
        minimum_area_ft2 = float(rule.get("minimum_area_ft2", 0.0))
        lower_cells = math.ceil(max(target_cells * (1.0 - tolerance), minimum_area_ft2 / cell_area_ft2))
        upper_cells = max(lower_cells, math.floor(target_cells * (1.0 + tolerance)))
        minimum_side = max(1, math.ceil(float(rule.get("minimum_clear_dimension_ft", 0.0)) / cell_dimension_ft))
        maximum_aspect = float(rule.get("hard_max_aspect_ratio", 4.0))
        best: tuple[float, int, int] | None = None
        for rectangle_width in range(minimum_side, min(int(options["width"]), 60) + 1):
            estimated_height = max(minimum_side, round(target_cells / rectangle_width))
            for rectangle_height in range(max(minimum_side, estimated_height - 1), estimated_height + 2):
                cell_count = rectangle_width * rectangle_height
                aspect = max(rectangle_width, rectangle_height) / min(rectangle_width, rectangle_height)
                if not lower_cells <= cell_count <= upper_cells or aspect > maximum_aspect:
                    continue
                cost = abs(cell_count - target_cells) + abs(rectangle_width - rectangle_height) * 0.02
                if best is None or cost < best[0]:
                    best = (cost, rectangle_width, rectangle_height)
        targets[room_type] = {
            "count": room["count"],
            "target_area_ft2": round(target_area_m2 * SQUARE_FEET_PER_SQUARE_METER),
            "target_cells": round(target_cells, 1),
            "acceptable_cell_count": [lower_cells, upper_cells],
            "minimum_side_cells": minimum_side,
            "recommended_rectangle_cells": [best[1], best[2]] if best else None,
        }
    return {
        "meters_per_cell": meters_per_cell,
        "feet_per_cell": round(cell_dimension_ft, 3),
        "square_feet_per_cell": round(cell_area_ft2, 3),
        "room_types": targets,
    }


def layout_family_guidance(program: dict[str, Any]) -> str:
    building_type = str(program.get("building_type", "")).lower()
    if "inpatient" in building_type and "tower" not in building_type:
        return """Use a non-tower inpatient ward topology:
- Treat each patient_room as one bed. Place patient-room and toilet modules in ordered perimeter bands. Default each en-suite toilet to a corridor-side/inboard corner near the patient-room entrance, fill the remainder of the module with patient_room pixels, and connect the toilet only to that patient room. Preserve a compact rectangular bed/clear zone toward the exterior wall and window side.
- Use exactly the requested corridor count as a connected spine, loop, or two connected ward segments. Every patient room must retain a direct shared boundary with a corridor even when a toilet is attached.
- Place nurse stations centrally along circulation. Place exam rooms, offices, storage, and waiting as independent compact rectangles with direct corridor boundaries.
- Generate repeated patient modules with loops from the precomputed module dimensions. Do not write one literal coordinate dictionary per patient room.
- This is a ward, not a hospital tower. Do not invent an H-shaped tower or central-core tower unless the building type explicitly says tower."""
    return "Use the program, healthcare profile, and precomputed pixel targets to select an appropriate compact layout family."


def output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "A complete Python layout algorithm that assigns the final semantic plan to top-level variable result.",
            },
            "strategy": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["code", "strategy", "assumptions"],
        "additionalProperties": False,
    }


def build_prompt(
    program: dict[str, Any],
    design_request: str,
    options: dict[str, Any],
    repair_context: str | None,
) -> str:
    healthcare_rules = rules_for_prompt(program) if uses_healthcare_rules(program) else {}
    pixel_targets = pixel_planning_targets(program, options)
    family_guidance = layout_family_guidance(program)
    return f"""You are a specialized floor-plan coding agent.
Read the architectural program, design a spatial strategy, and write a complete executable Python layout algorithm. Work like a coding agent: inspect the previous code and exact executor or validator feedback, revise the implementation, and return the entire corrected program on every attempt.
Return JSON matching the required response schema.

The code field must contain a standalone Python program. There is no custom floor-plan API. Write the geometry, allocation, rasterization, search, and optimization logic yourself.

Execution environment:
- Authorized imports: numpy, shapely, scipy, networkx, PIL, math, random, statistics, itertools, and collections.
- Functions, classes, loops, recursion, comprehensions, exceptions, and normal Python control flow are allowed.
- The executor blocks files, network, processes, dynamic execution, and private runtime attributes.
- Keep the program deterministic. Seed random generators if they are used.
- Do not use PixelPlan. Do not print the result and do not save files.

Assign the final data to a top-level variable named result with this contract:

result = {{
    "width": integer,
    "height": integer,
    "meters_per_cell": positive_number,
    "footprint": 2D boolean array shaped [height, width],
    "grid": 2D integer array shaped [height, width],
    "rooms": [{{"id": unique_string, "type": program_type}}, ...],
    "doors": [
        {{
            "id": unique_string,
            "from_room": room_id,
            "to_room": room_id_or_None,
            "x": integer,
            "y": integer,
            "orientation": "horizontal" or "vertical",
            "width_cells": integer_1_to_6
        }},
        ...
    ]
}}

Grid contract:
- Each nonnegative grid value is an index into result["rooms"].
- Use -1 for an unassigned cell inside the footprint and -2 outside the footprint.
- The explicit footprint must be False outside the building and True inside it.
- Every listed room must own at least one grid cell.
- Build reusable Python helpers for painting regions, detecting overlaps, finding shared boundaries, and placing doors.
- Treat nonnegative grid cells as owned: a painting helper must reject any attempt to overwrite a cell owned by a different room.
- Do not append a room record until its geometry has been allocated successfully.
- Before assigning result, run an internal ownership audit: every room index from 0 through len(rooms) - 1 must appear in grid, every nonnegative grid index must be valid, and requested counts by type must match exactly.
- A horizontal door at (x, y) separates cells above and below that boundary. A vertical door separates cells left and right. Every door pixel must lie on the exact shared boundary of its two rooms. An exterior door uses to_room=None and must separate its from_room from footprint exterior.
- Never guess or hard-code door coordinates before the grid is complete. Generate doors only after every room has its final pixels.
- Implement one shared-boundary scanner and use it for every internal door. For a vertical door, scan x from 1 to width - 1 and compare grid[y, x - 1] with grid[y, x]. For a horizontal door, scan y from 1 to height - 1 and compare grid[y - 1, x] with grid[y, x]. Match the unordered pair of room indexes, choose a valid candidate, and use width_cells=1 unless a whole contiguous run was verified.
- Implement one exterior-boundary scanner for the entrance. It must find a corridor cell adjacent to footprint=False or the canvas exterior and emit the matching boundary coordinate and orientation.
- If two intended spaces have no shared boundary, change the layout. Never fabricate a door coordinate.

Layout requirements:
- Use exactly {options['width']} by {options['height']} pixels.
- Every requested room instance needs one unique id.
- Include the requested number of corridor instances.
- The corridor count must match the program exactly. Do not add an extra core, lobby, or connector with type corridor; merge such geometry into one of the requested corridor indexes.
- No overlap and no geometry outside the canvas.
- Prefer compact spaces and continuous corridors.
- Every non-corridor room needs a door, normally connected to a corridor.
- Door coordinates must lie on the shared room boundary.
- Include one exterior main entrance with to_room_id_or_none set to None.
- Satisfy every requested type adjacency where practical.
- Fill most of the canvas.
- Prefer algorithms and loops over hundreds of repeated literal assignments. Keep the code under 50 KB even for large hospitals.
- For large programs, use data lists plus loops or helper functions to allocate every requested room instance exactly once.
- Derive integer room-module widths and depths from meters_per_cell and the physical rules below before placing modules. Audit actual pixel area, minimum dimension, bounding-box aspect ratio, and compactness for every room before returning result.
- For a hospital tower, create an explicit winged footprint whose row or column spans vary. A smaller inset rectangle is still rectangular and will be rejected. L, T, H, U, cross, and other multi-wing circulation topologies are valid. Multiple corridor regions may touch along boundaries but must not overlap.
- In a tower, place patient-room bands at the footprint perimeter on both sides of the four corridor arms. Never represent a tower as full-width horizontal strips.
- Make normal occupiable rooms compact and approximately rectangular. Do not create L-shaped patient rooms that wrap around unrelated support spaces.
- Only a toilet may be planned as an attached room inside a patient-room module. Exam rooms, offices, storage rooms, nurse stations, and waiting rooms must be independent rooms with their own corridor access.
- Treat the following U.S. healthcare planning profile as mandatory acceptance criteria. Convert physical feet to cells using meters_per_cell. The validator checks every room instance, not only one example of each type.

U.S. healthcare planning profile:
{json.dumps(healthcare_rules, indent=2)}

Precomputed pixel planning targets for this exact canvas and scale:
{json.dumps(pixel_targets, indent=2)}

Use the recommended rectangle dimensions or another rectangle within each acceptable cell-count range. Do not give all room types one shared module size.

Building-type layout guidance:
{family_guidance}

Suggested scale: {options['meters_per_cell']} meters per cell.

Program:
{json.dumps(program, indent=2)}

Design request:
{design_request or 'Create a compact and legible plan with practical circulation.'}

Repair context:
{repair_context or 'This is the first attempt.'}
"""


def generate_gemini_code(
    api_key: str,
    model: str | None,
    program: dict[str, Any],
    design_request: str,
    options: dict[str, Any],
    repair_context: str | None = None,
    thinking_level: str | None = None,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")
    model_name = model or DEFAULT_MODEL
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    requested_room_count = sum(int(room.get("count", 1)) for room in program.get("rooms", []))
    complex_request = "tower" in str(program.get("building_type", "")).lower() or requested_room_count >= 40
    output_token_setting = (
        os.environ.get("GEMINI_COMPLEX_MAX_OUTPUT_TOKENS", "40000")
        if complex_request
        else os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "20000")
    )
    max_output_tokens = max(8000, min(60000, int(output_token_setting)))
    generation_config: dict[str, Any] = {
        "temperature": 0.2,
        "maxOutputTokens": max_output_tokens,
        "responseMimeType": "application/json",
        "responseJsonSchema": output_schema(),
    }
    if thinking_level:
        normalized_level = thinking_level.strip().lower()
        if normalized_level not in {"minimal", "low", "medium", "high"}:
            raise ValueError("Gemini thinking level must be minimal, low, medium, or high.")
        generation_config["thinkingConfig"] = {"thinkingLevel": normalized_level}

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": build_prompt(program, design_request, options, repair_context)}],
            }
        ],
        "generationConfig": generation_config,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    timeout_seconds = max(30.0, float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "300")))
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(body).get("error", {}).get("message", body)
        except json.JSONDecodeError:
            message = body
        raise ValueError(f"Gemini request failed: {message}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"Gemini network request failed: {error.reason}") from error

    candidates = response_payload.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        raise ValueError("Gemini returned no candidate text.")
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        structured = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Gemini returned invalid structured JSON: {error.msg}.") from error
    if not isinstance(structured.get("code"), str):
        raise ValueError("Gemini output is missing the code string.")
    if not isinstance(structured.get("strategy"), str):
        raise ValueError("Gemini output is missing the strategy string.")
    if not isinstance(structured.get("assumptions"), list):
        raise ValueError("Gemini output is missing the assumptions array.")
    structured["model"] = model_name
    structured["usage"] = normalize_usage_metadata(model_name, response_payload.get("usageMetadata"))
    return structured
