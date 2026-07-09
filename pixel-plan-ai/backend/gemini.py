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


def attempt_tools_enabled() -> bool:
    """True when the in-attempt self-check tool loop is switched on (docs #8).

    Unset or 0/false/no/off (case-insensitive) keeps the byte-identical single-call path.
    """
    return os.environ.get("GEMINI_ATTEMPT_TOOLS", "").strip().lower() not in {"", "0", "false", "no", "off"}


# In-attempt self-check tool (docs/claude-code-lessons.md #8): behind GEMINI_ATTEMPT_TOOLS the model
# may call this to run a draft in the real sandbox + validator before submitting its structured answer.
ATTEMPT_TOOL_DECLARATION = {
    "name": "execute_and_validate",
    "description": (
        "Run a complete candidate program in the real sandbox and architectural validator. "
        "Returns score, rejected flag, and compact validator feedback. "
        "Use it to self-check before submitting your final answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "A complete standalone Python program that assigns the final plan to top-level variable result.",
            },
        },
        "required": ["code"],
    },
}

# Appended after the repair-context layer only when the tool loop is active.
ATTEMPT_TOOLS_GUIDANCE = (
    "You may call execute_and_validate at most 2 times to test a draft before answering. "
    "After the final tool result, return the required JSON answer."
)

# Verdict-ownership + anti-thrashing rules added 2026-07-08 (docs/claude-code-lessons.md #3); pair with the validator delta feedback.
# Durable coding contract (execution env, result/grid contracts) moved here 2026-07-08 for role separation + prefix caching (docs/claude-code-lessons.md #12).
# Micro-examples added 2026-07-08; the door example is locked by a sandbox test (docs/claude-code-lessons.md #11).
SYSTEM_INSTRUCTION = """You are a U.S. healthcare floor-plan coding agent.
You produce complete executable Python, then use exact executor and validator feedback to revise it.
Treat room function, zoning, circulation access, physical clear dimensions, area, compactness, and aspect ratio as first-class design constraints.
Never trade away clinical room usability merely to satisfy room counts. Distinguish patient care, clinical support, staff support, public support, and building support spaces.
An inpatient en-suite toilet should default to an inboard corner near the room entrance and corridor, preserving the exterior wall for the patient bed and window. The patient-space pixels may form an L around that toilet, while a rectangular bed/clear zone remains and the patient space keeps direct corridor access. Nested and outboard variants are allowed only when the design request or program calls for them. Exam rooms, offices, storage rooms, nurse stations, and waiting rooms are independent spaces and must never be carved into or wrapped by a patient room.
Return one final implementation only. The code must not contain abandoned layout alternatives, repeated redefinitions of room lists, self-correction commentary, or draft coordinates. Put reasoning in the strategy field, not inside the code.
The validator is the sole judge of correctness. Never state or imply that the plan passes; report what you changed and which named failures it targets.
Before changing approach, diagnose WHY the previous attempt failed from the validator delta when one is present. Prefer the minimal edit that fixes the named failures; never discard parts that already pass.
Example: if the delta reports fixed: doors:Rooms with doors and still_failing: area:U.S. room-area range, edit only the failing rooms' geometry; leave the door code untouched.
The result is a schematic planning study, not construction documentation or a claim of code compliance.

The code field must contain a standalone Python program. There is no custom floor-plan API. Write the geometry, allocation, rasterization, search, and optimization logic yourself.

Execution environment:
- Authorized imports: numpy, shapely, scipy, networkx, PIL, math, random, statistics, itertools, and collections.
- Functions, classes, loops, recursion, comprehensions, exceptions, and normal Python control flow are allowed.
- The executor blocks files, network, processes, dynamic execution, and private runtime attributes.
- Keep the program deterministic. Seed random generators if they are used.
- Do not use PixelPlan. Do not print the result and do not save files.

Assign the final data to a top-level variable named result with this contract:

result = {
    "width": integer,
    "height": integer,
    "meters_per_cell": positive_number,
    "footprint": 2D boolean array shaped [height, width],
    "grid": 2D integer array shaped [height, width],
    "rooms": [{"id": unique_string, "type": program_type}, ...],
    "doors": [
        {
            "id": unique_string,
            "from_room": room_id,
            "to_room": room_id_or_None,
            "x": integer,
            "y": integer,
            "orientation": "horizontal" or "vertical",
            "width_cells": integer_1_to_6
        },
        ...
    ]
}

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
Example (door derivation): if grid[4][10] == 2 (exam_room_1) and grid[5][10] == 0 (corridor_1), a horizontal door on that boundary is {"id": "d1", "from_room": "exam_room_1", "to_room": "corridor_1", "x": 10, "y": 5, "orientation": "horizontal", "width_cells": 1} — y names the row BELOW the boundary, and the unordered pair {grid[y-1][x], grid[y][x]} must equal the two rooms for every covered x."""

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


def sum_attempt_usage(model: str, raw_metadatas: list[Any]) -> dict[str, Any]:
    """Normalize every per-call usageMetadata in one attempt and sum the token fields.

    estimated_cost_usd is summed only across the calls that priced (None otherwise).
    """
    total = {
        "prompt_tokens": 0,
        "candidate_tokens": 0,
        "thinking_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": None,
    }
    for raw in raw_metadatas:
        row = normalize_usage_metadata(model, raw)
        total["prompt_tokens"] += row["prompt_tokens"]
        total["candidate_tokens"] += row["candidate_tokens"]
        total["thinking_tokens"] += row["thinking_tokens"]
        total["total_tokens"] += row["total_tokens"]
        if row["estimated_cost_usd"] is not None:
            total["estimated_cost_usd"] = round((total["estimated_cost_usd"] or 0.0) + row["estimated_cost_usd"], 6)
    return total


def call_gemini(
    api_key: str,
    model_name: str,
    contents: list[dict[str, Any]],
    generation_config: dict[str, Any],
    system_instruction_text: str,
    tools: list[dict[str, Any]] | None = None,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """POST one (possibly multi-turn) generateContent request and return the parsed JSON."""
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_instruction_text}]},
        "contents": contents,
        "generationConfig": generation_config,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_config is not None:
        payload["toolConfig"] = tool_config
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    timeout_seconds = max(30.0, float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "300")))
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(body).get("error", {}).get("message", body)
        except json.JSONDecodeError:
            message = body
        raise ValueError(f"Gemini request failed: {message}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"Gemini network request failed: {error.reason}") from error


def finish_reason(response_payload: dict[str, Any]) -> str | None:
    """The first candidate's finishReason (e.g. "STOP", "MAX_TOKENS"), or None when absent."""
    candidates = response_payload.get("candidates", [])
    if not candidates:
        return None
    return candidates[0].get("finishReason")


def call_gemini_with_truncation_retry(
    api_key: str,
    model_name: str,
    contents: list[dict[str, Any]],
    generation_config: dict[str, Any],
    system_instruction_text: str,
    tools: list[dict[str, Any]] | None = None,
    tool_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """call_gemini, but recover from an output-token cut-off at the API layer.

    If the first response stops at finishReason MAX_TOKENS, retry the SAME request ONCE with
    maxOutputTokens doubled (capped at 60000). Returns (response, [usageMetadata, ...]) so the
    caller sums every HTTP call's usage. If the retry is ALSO truncated, raise a clear ValueError.

    Incident 2026-07-09 (24-room program, GEMINI_MAX_OUTPUT_TOKENS=20000): the model's JSON hit the
    output cap mid-string. Three consecutive repair attempts died with "'{' was never closed" and
    that fake Python syntax error was fed back into the model's repair context — telling it to hunt
    a bug that lived in the token budget, not the code. Truncation is detected here and must never
    masquerade as a code error.
    """
    usage_metadatas: list[Any] = []
    response = call_gemini(
        api_key, model_name, contents, generation_config, system_instruction_text,
        tools=tools, tool_config=tool_config,
    )
    usage_metadatas.append(response.get("usageMetadata"))
    if finish_reason(response) != "MAX_TOKENS":
        return response, usage_metadatas
    doubled_cap = min(60000, int(generation_config.get("maxOutputTokens", 0)) * 2)
    doubled_config = dict(generation_config)
    doubled_config["maxOutputTokens"] = doubled_cap
    retry_response = call_gemini(
        api_key, model_name, contents, doubled_config, system_instruction_text,
        tools=tools, tool_config=tool_config,
    )
    usage_metadatas.append(retry_response.get("usageMetadata"))
    if finish_reason(retry_response) == "MAX_TOKENS":
        raise ValueError(
            f"Gemini output was truncated at the {doubled_cap} token limit twice; "
            "the program is too large for the output budget. "
            "Raise GEMINI_MAX_OUTPUT_TOKENS or simplify the program."
        )
    return retry_response, usage_metadatas


def first_candidate_parts(response_payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = response_payload.get("candidates", [])
    if not candidates:
        return []
    return candidates[0].get("content", {}).get("parts", []) or []


def find_function_call(parts: list[dict[str, Any]]) -> dict[str, Any] | None:
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("functionCall"), dict):
            return part["functionCall"]
    return None


def parse_structured_answer(response_payload: dict[str, Any]) -> dict[str, Any]:
    """Extract, unfence, json-parse and shape-validate a structured code answer."""
    parts = first_candidate_parts(response_payload)
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
    return structured


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


# Label-conflict precedence added 2026-07-08: a live candidate drawing duplicated exam_room_3 (docs/claude-code-lessons.md #13).
REFERENCE_IMAGE_GUIDANCE = """A reference floor-plan DRAWING of this exact program is attached.
Your program must TRANSCRIBE that drawing onto the cell grid, not invent a new design:
- Orientation: grid row y=0 is the TOP edge of the drawing and y grows DOWNWARD; column x=0 is the LEFT edge. Do not mirror or rotate the layout.
- Reproduce its building massing, wing arrangement, corridor topology and room placement; the grid is a rasterization of THIS drawing at the requested canvas size.
- The text label inside each drawn room names its program instance; the quarter-circle swing arcs are the doors; place your doors on the same shared boundaries.
- Keep every room's position and relative proportion close to the drawing. Deviate only where the drawing physically cannot satisfy a validation rule (for example an undersized room), and keep such deviations local.
- Encode the layout you SEE as data (room rectangles read off the drawing) plus painting loops; do not substitute a generic packing algorithm.
- If the drawing's room labels conflict with the program JSON (duplicate or missing labels), the program wins: keep the program's exact room counts and map surplus or missing labels onto the closest sensible geometry."""


# Escape hatch added 2026-07-08: the first trace-mode E2E hit a seed missing the waiting room entirely (docs/claude-code-lessons.md #9).
SEED_REPAIR_GUIDANCE = """The current program encodes an EXISTING traced floor plan as data literals
(FOOTPRINT_RECTS / ROOM_DATA / DOORS) followed by a fixed rasterizing builder.
Repair it; do not redesign it:
- Fix validator failures by minimally adjusting the data literals: nudge rectangle bounds, resize or split one room's rectangles, move or add door entries on real shared boundaries.
- Preserve the traced massing, corridor topology, and relative room placement. Do not reorder rooms, do not swap the builder for a generic packing or banding algorithm, and do not regenerate the layout from scratch.
- Keep the ROOM_DATA/DOORS + builder structure in every revision so later repairs stay local. You may extend the builder with ownership audits or shared-boundary door scanners, but geometry always stays in the data literals.
- If a named failure cannot be fixed by a local edit (for example a required room is missing from the trace entirely), add the minimal new geometry required, placed consistently with the traced topology; this is the only case where new rooms may be introduced.
- Example of a minimal repair: to widen exam_room_2 by one cell, change its rectangle (40, 8, 10, 12) to (40, 8, 11, 12) inside ROOM_DATA and touch nothing else."""


def build_prompt(
    program: dict[str, Any],
    design_request: str,
    options: dict[str, Any],
    repair_context: str | None,
    with_reference_image: bool = False,
    with_seed_repair: bool = False,
) -> str:
    healthcare_rules = rules_for_prompt(program) if uses_healthcare_rules(program) else {}
    pixel_targets = pixel_planning_targets(program, options)
    family_guidance = layout_family_guidance(program)
    if with_seed_repair:
        family_guidance = SEED_REPAIR_GUIDANCE
        if with_reference_image:
            family_guidance += (
                "\n\nThe original drawing this plan was traced from is attached for visual grounding "
                "(grid row y=0 is the TOP edge; do not mirror). Use it to resolve ambiguity, "
                "but geometry edits still happen in the data literals."
            )
    elif with_reference_image:
        family_guidance = REFERENCE_IMAGE_GUIDANCE
    # Section order = variation frequency (static -> mode -> program -> attempt) so implicit prefix caching survives repair attempts (docs/claude-code-lessons.md #4).
    # Numbers below mirror validator.py thresholds exactly — update together (docs/claude-code-lessons.md #10).
    return f"""You are a specialized floor-plan coding agent.
Read the architectural program, design a spatial strategy, and write a complete executable Python layout algorithm. Work like a coding agent: inspect the previous code and exact executor or validator feedback, revise the implementation, and return the entire corrected program on every attempt.
Return JSON matching the required response schema.

Layout requirements:
- Every requested room instance needs one unique id.
- Include the requested number of corridor instances.
- The corridor count must match the program exactly. Do not add an extra core, lobby, or connector with type corridor; merge such geometry into one of the requested corridor indexes.
- No overlap and no geometry outside the canvas.
- Keep every room compact: the proportion check rejects any room whose bounding-box aspect ratio (longer clear side / shorter clear side) exceeds its type's hard_max_aspect_ratio (default 4.0), whose shorter clear dimension is below its type's minimum_clear_dimension_ft, or whose fill ratio (owned cells / bounding-box cells) is under its type's minimum_compactness. Corridors carry no proportion limit, but every non-corridor room must share a boundary with a corridor so the access check passes.
- Every non-corridor room needs a door, normally connected to a corridor.
- Door coordinates must lie on the shared room boundary.
- Include one exterior main entrance with to_room_id_or_none set to None.
- Satisfy every requested type adjacency where practical.
- Assign rooms to at least 85% of the footprint cells; the validator rejects coverage below 0.85.
- Prefer algorithms and loops over hundreds of repeated literal assignments. Keep the code under 50 KB even for large hospitals.
- For large programs, use data lists plus loops or helper functions to allocate every requested room instance exactly once.
- Derive integer room-module widths and depths from meters_per_cell and the physical rules below before placing modules. Audit actual pixel area, minimum dimension, bounding-box aspect ratio, and compactness for every room before returning result.
- For a hospital tower, create an explicit winged footprint whose row or column spans vary. A smaller inset rectangle is still rectangular and will be rejected. L, T, H, U, cross, and other multi-wing circulation topologies are valid. Multiple corridor regions may touch along boundaries but must not overlap.
- In a tower, place patient-room bands at the footprint perimeter on both sides of the four corridor arms. Never represent a tower as full-width horizontal strips.
- Make normal occupiable rooms compact and approximately rectangular. Do not create L-shaped patient rooms that wrap around unrelated support spaces.
- Only a toilet may be planned as an attached room inside a patient-room module. Exam rooms, offices, storage rooms, nurse stations, and waiting rooms must be independent rooms with their own corridor access.
- Treat the following U.S. healthcare planning profile as mandatory acceptance criteria. Convert physical feet to cells using meters_per_cell. The validator checks every room instance, not only one example of each type.

Use the recommended rectangle dimensions or another rectangle within each acceptable cell-count range. Do not give all room types one shared module size.

Building-type layout guidance:
{family_guidance}

U.S. healthcare planning profile:
{json.dumps(healthcare_rules, indent=2)}

Precomputed pixel planning targets for this exact canvas and scale:
{json.dumps(pixel_targets, indent=2)}

Canvas: use exactly {options['width']} by {options['height']} pixels.

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
    reference_image: dict[str, str] | None = None,
    seed_repair: bool = False,
    attempt_executor: Any = None,
    max_tool_calls: int = 2,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")
    model_name = model or DEFAULT_MODEL
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

    # Only build the tool loop when a real executor is wired AND the flag is on;
    # otherwise the request is byte-identical to the plain single structured call.
    use_tools = attempt_executor is not None and attempt_tools_enabled()
    prompt_text = build_prompt(
        program, design_request, options, repair_context,
        with_reference_image=reference_image is not None,
        with_seed_repair=seed_repair,
    )
    if use_tools:
        prompt_text = prompt_text + "\n\n" + ATTEMPT_TOOLS_GUIDANCE
    parts: list[dict[str, Any]] = [{"text": prompt_text}]
    if reference_image is not None:
        # the drawing rides along on EVERY attempt, so repairs stay anchored to
        # the reference design instead of drifting toward a generic layout
        parts.append({
            "inlineData": {
                "mimeType": reference_image["mime"],
                "data": reference_image["data"],
            },
        })

    if not use_tools:
        response_payload, usage_metadatas = call_gemini_with_truncation_retry(
            api_key, model_name, [{"role": "user", "parts": parts}], generation_config, SYSTEM_INSTRUCTION
        )
        structured = parse_structured_answer(response_payload)
        structured["model"] = model_name
        # sum_attempt_usage over one or (after a truncation retry) two calls; identical to
        # normalize_usage_metadata for a single call.
        structured["usage"] = sum_attempt_usage(model_name, usage_metadatas)
        return structured

    # In-attempt self-check loop (docs #8): the model may call execute_and_validate up to
    # max_tool_calls times against the real sandbox+validator, then we take one tools-free
    # schema-constrained call as the final structured answer. Probe confirmed tools and
    # responseJsonSchema can coexist, so the generation_config is unchanged across phases.
    contents: list[dict[str, Any]] = [{"role": "user", "parts": parts}]
    tools = [{"functionDeclarations": [ATTEMPT_TOOL_DECLARATION]}]
    tool_config = {"functionCallingConfig": {"mode": "AUTO"}}
    usage_metadatas: list[Any] = []
    calls_used = 0
    while calls_used < max_tool_calls:
        # Tool-phase calls also get the doubled-budget retry and raise on double-truncation.
        tool_response, tool_usage = call_gemini_with_truncation_retry(
            api_key, model_name, contents, generation_config, SYSTEM_INSTRUCTION,
            tools=tools, tool_config=tool_config,
        )
        usage_metadatas.extend(tool_usage)
        model_parts = first_candidate_parts(tool_response)
        function_call = find_function_call(model_parts)
        if function_call is None:
            break  # the model chose to answer directly; the final phase re-asks under the schema
        contents.append({"role": "model", "parts": model_parts})
        code_argument = str((function_call.get("args") or {}).get("code", ""))
        feedback = attempt_executor(code_argument)
        contents.append({
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": function_call.get("name") or ATTEMPT_TOOL_DECLARATION["name"],
                    "response": feedback,
                },
            }],
        })
        calls_used += 1
    else:
        # Budget consumed without an early break: the model used every tool call.
        # Tell it to finalize instead of silently dropping its next intent (docs #8 guard).
        if calls_used:
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": ATTEMPT_TOOL_DECLARATION["name"],
                        "response": {"note": "tool budget exhausted; submit your final JSON now"},
                    },
                }],
            })

    # Final phase gets the same MAX_TOKENS retry as the plain single-call path (docs #8).
    final_response, final_usage = call_gemini_with_truncation_retry(
        api_key, model_name, contents, generation_config, SYSTEM_INSTRUCTION
    )
    usage_metadatas.extend(final_usage)
    structured = parse_structured_answer(final_response)
    structured["model"] = model_name
    structured["usage"] = sum_attempt_usage(model_name, usage_metadatas)
    return structured
