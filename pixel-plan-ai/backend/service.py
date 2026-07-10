from __future__ import annotations

import math
import os
import time
from typing import Any

from code_policy import validate_program_contract
from gemini import attempt_tools_enabled, generate_gemini_code
from job_progress import append_job_event, publish_job_progress
from plan_image import generate_plan_images
from program import calculate_scale, normalize_program
from runtime import execute_pixel_code
from seed_code import normalize_seed, seed_to_code
from validator import validate_plan


MAX_REFERENCE_IMAGE_BASE64_LENGTH = 8_000_000


def normalize_reference_image(value: Any) -> dict[str, str] | None:
    """Validated ``{mime, data}`` reference drawing, or None when absent."""
    if value in (None, "", {}):
        return None
    if not isinstance(value, dict):
        raise ValueError("reference_image must be an object with mime and data.")
    mime = str(value.get("mime", "image/png"))
    data = str(value.get("data", ""))
    if not mime.startswith("image/"):
        raise ValueError("reference_image.mime must be an image mime type.")
    if not data:
        raise ValueError("reference_image.data must be base64 image bytes.")
    if len(data) > MAX_REFERENCE_IMAGE_BASE64_LENGTH:
        raise ValueError("reference_image exceeds the 8 MB base64 limit.")
    return {"mime": mime, "data": data}


def room_type_from_issue(issue: str) -> str:
    room_id = issue.split(" ", 1)[0]
    head, separator, tail = room_id.rpartition("_")
    return head if separator and tail.isdigit() else room_id


def issue_room_id(issue: str) -> str:
    token = issue.split(" ", 1)[0]
    head, separator, tail = token.rpartition("_")
    return token if separator and tail.isdigit() else "plan"


def compact_validation_feedback(validation: dict[str, Any]) -> str:
    failed_checks = [check for check in validation.get("checks", []) if not check.get("pass")]
    sections = [f"Validation score: {validation.get('score', 0)}."]
    if failed_checks:
        check_lines = [f"{check.get('category', 'general')}: {check.get('label', 'failed')}" for check in failed_checks]
        sections.append("Failed checks:\n- " + "\n- ".join(check_lines[:20]))

    failed_areas = [row for row in validation.get("areas", []) if not row.get("pass")]
    area_groups: dict[str, list[dict[str, Any]]] = {}
    for row in failed_areas:
        area_groups.setdefault(str(row.get("type", "unknown")), []).append(row)
    if area_groups:
        area_lines: list[str] = []
        for room_type, rows in sorted(area_groups.items()):
            actual_values = [float(row.get("actual_ft2", 0)) for row in rows]
            actual_cell_values = [int(row.get("actual_cells", 0)) for row in rows]
            target = round(float(rows[0].get("target_ft2", 0)))
            target_cells = float(rows[0].get("target_cells", 0))
            tolerance = round(float(rows[0].get("tolerance_percent", 0)))
            actual_range = (
                f"{round(min(actual_values))} sf"
                if min(actual_values) == max(actual_values)
                else f"{round(min(actual_values))}-{round(max(actual_values))} sf"
            )
            area_lines.append(
                f"{room_type}: {len(rows)} failing rooms, actual {actual_range} / "
                f"{min(actual_cell_values)}-{max(actual_cell_values)} cells, target {target} sf / "
                f"about {target_cells:.1f} cells, tolerance {tolerance}%."
            )
        sections.append("Area failures by type:\n- " + "\n- ".join(area_lines))

    issues = [str(issue) for issue in validation.get("issues", [])]
    representative_groups: list[tuple[str, tuple[str, ...], str]] = [
        ("Door/access examples", (" has no door.", " requires direct corridor access", " requires access from"), "doors"),
        ("Proportion examples", (" proportion fails ",), "proportion"),
        ("Zoning examples", (" is embedded in ",), "zoning"),
        ("Other geometry examples", ("Tower ", "Too few patient rooms", "circulation does not"), "geometry"),
    ]
    consumed: set[str] = set()
    for heading, markers, slug in representative_groups:
        by_type: dict[str, str] = {}
        for issue in issues:
            if issue in consumed or not any(marker in issue for marker in markers):
                continue
            room_type = room_type_from_issue(issue)
            by_type.setdefault(room_type, issue)
            consumed.add(issue)
        if by_type:
            # Stable rule/room tags let the model pattern-match the same failure
            # across repair attempts (docs/claude-code-lessons.md #2).
            tagged = [
                f'<validator_error rule="{slug}" room="{issue_room_id(issue)}">{issue}</validator_error>'
                for issue in list(by_type.values())[:8]
            ]
            sections.append(f"{heading}:\n- " + "\n- ".join(tagged))

    return "\n".join(sections)


def failed_check_keys(validation: dict[str, Any]) -> dict[str, str]:
    keys: dict[str, str] = {}
    for check in validation.get("checks", []):
        if check.get("pass"):
            continue
        category = str(check.get("category", "general"))
        label = str(check.get("label", "failed"))
        # Drop tokens carrying a count or percentage so the key survives value churn.
        stable_tokens = [token for token in label.split() if not any(char.isdigit() or char == "%" for char in token)]
        stable_label = " ".join(stable_tokens).rstrip(":")
        keys[f"{category}:{stable_label}"] = label
    return keys


def validation_delta(previous: dict[str, Any], current: dict[str, Any]) -> str:
    previous_keys = failed_check_keys(previous)
    current_keys = failed_check_keys(current)
    fixed = sorted(key for key in previous_keys if key not in current_keys)
    still_failing = sorted(key for key in current_keys if key in previous_keys)
    new_failures = sorted(key for key in current_keys if key not in previous_keys)
    fixed_text = ", ".join(fixed) or "none"
    still_text = ", ".join(f"{key} ({current_keys[key]})" for key in still_failing) or "none"
    new_text = ", ".join(f"{key} ({current_keys[key]})" for key in new_failures) or "none"
    return (
        "[validator delta vs previous candidate — computed by the harness from executing both candidates; treat as ground truth]\n"
        f"fixed: {fixed_text}\n"
        f"still_failing: {still_text}\n"
        f"new_failures: {new_text}"
    )


def normalize_options(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    width = max(32, min(160, round(float(source.get("width", 96)))))
    height = max(24, min(120, round(float(source.get("height", 64)))))
    return {"width": width, "height": height}


def execute_and_validate(
    code: str,
    program: dict[str, Any],
    enforce_ai_contract: bool = False,
) -> dict[str, Any]:
    if enforce_ai_contract:
        validate_program_contract(code, program)
    sanitized, plan = execute_pixel_code(code)
    return {"code": sanitized, "plan": plan, "validation": validate_plan(plan, program)}


def _normalized_room_centroids(
    container: dict[str, Any],
) -> tuple[list[tuple[float, float] | None], list[int]]:
    """Per-room centroids normalized into the occupied bounding box, plus cell counts.

    The occupied bbox is taken over every room cell (values ``>= 0``) on this side alone,
    then each room's centroid (mean of its own cell coords) is mapped to ``u=(cx-bx)/bw``,
    ``v=(cy-by)/bh`` (a 0-width/height span is guarded to 1). Removing each side's own
    translation and scale is what makes the metric scale-invariant. A room with no cells
    has centroid ``None``.
    """
    rooms = container.get("rooms", [])
    width = int(container.get("width", 0)) or 1
    sum_x = [0.0] * len(rooms)
    sum_y = [0.0] * len(rooms)
    counts = [0] * len(rooms)
    min_x = min_y = max_x = max_y = None
    for position, room_index in enumerate(container.get("cells", [])):
        if not isinstance(room_index, int) or not 0 <= room_index < len(rooms):
            continue
        x, y = position % width, position // width
        sum_x[room_index] += x
        sum_y[room_index] += y
        counts[room_index] += 1
        min_x = x if min_x is None or x < min_x else min_x
        max_x = x if max_x is None or x > max_x else max_x
        min_y = y if min_y is None or y < min_y else min_y
        max_y = y if max_y is None or y > max_y else max_y
    if min_x is None:  # no room cells on this side
        return [None] * len(rooms), counts
    bw = (max_x - min_x) or 1
    bh = (max_y - min_y) or 1
    centroids: list[tuple[float, float] | None] = []
    for index in range(len(rooms)):
        if counts[index] == 0:
            centroids.append(None)
        else:
            cx, cy = sum_x[index] / counts[index], sum_y[index] / counts[index]
            centroids.append(((cx - min_x) / bw, (cy - min_y) / bh))
    return centroids, counts


def layout_fidelity(seed: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Scale-invariant relative-layout similarity of a candidate plan to the traced seed.

    v2 (user decision 2026-07-10): a legitimate in-place rescale — shrinking rooms to hit
    ft^2 targets — must not read as a layout change. Each side is normalized into its own
    occupied bounding box, so uniform scaling and translation cost nothing; only rearranging
    rooms does. The score blends two components, both in ``[0, 1]``:

    * position (0.6) — per matched room, displacement ``d = hypot(du, dv)`` between the seed
      and candidate normalized centroids scores ``max(0, 1 - d/0.5)`` (zero credit at
      half-building displacement), size-weighted by seed cell count, unmatched rooms carrying
      their weight at zero credit.
    * arrangement (0.4) — over all matched room pairs, the fraction of relative orderings
      preserved: for each axis a pair counts only when the seed separation exceeds 0.08, and
      is preserved when the sign of the seed separation matches the candidate's (1.0 when no
      pair is decisive on either axis).

    Rooms are matched by EXACT id first (ids survive most ROOM_DATA repairs); each remaining
    seed room then greedily takes the nearest unused same-type candidate by normalized centroid
    distance (overlap matching breaks under rescale). Unmatched seed rooms earn no position
    credit and are listed in ``unmatched``. ``movers`` reports up to five matched rooms with
    the lowest position score as ``(seed_id, displacement percent of the building diagonal)``.
    Both sides carry a flat ``cells`` list indexed row-major (``y*width + x``) whose nonnegative
    values index the matching ``rooms`` list.
    """
    seed_rooms = seed.get("rooms", [])
    plan_rooms = plan.get("rooms", [])
    seed_centroids, seed_counts = _normalized_room_centroids(seed)
    plan_centroids, _plan_counts = _normalized_room_centroids(plan)

    plan_index_by_id: dict[str, int] = {}
    for index, room in enumerate(plan_rooms):
        plan_index_by_id.setdefault(str(room.get("id")), index)

    # (seed_id, (su, sv), (pu, pv), position_score, weight, displacement_percent)
    matches: list[tuple[str, tuple[float, float], tuple[float, float], float, int, float]] = []
    consumed_plan: set[int] = set()
    remaining_seed: list[int] = []
    diagonal = math.hypot(1.0, 1.0)  # building diagonal in normalized space

    def record_match(seed_index: int, plan_index: int) -> None:
        seed_uv = seed_centroids[seed_index]
        plan_uv = plan_centroids[plan_index]
        assert seed_uv is not None and plan_uv is not None
        distance = math.hypot(seed_uv[0] - plan_uv[0], seed_uv[1] - plan_uv[1])
        position_score = max(0.0, 1.0 - distance / 0.5)
        matches.append((
            str(seed_rooms[seed_index].get("id")),
            seed_uv,
            plan_uv,
            position_score,
            seed_counts[seed_index],
            round(distance / diagonal * 100, 1),
        ))

    # Pass 1: exact id match — the id survives most ROOM_DATA repairs.
    for seed_index, room in enumerate(seed_rooms):
        if seed_centroids[seed_index] is None:
            continue  # a seed room with no cells cannot be scored or missed
        plan_index = plan_index_by_id.get(str(room.get("id")))
        if plan_index is not None and plan_index not in consumed_plan and plan_centroids[plan_index] is not None:
            consumed_plan.add(plan_index)
            record_match(seed_index, plan_index)
        else:
            remaining_seed.append(seed_index)

    # Pass 2: greedy same-type match, nearest normalized centroid first.
    candidate_pairs: list[tuple[float, int, int]] = []  # (distance, seed_index, plan_index)
    for seed_index in remaining_seed:
        seed_type = str(seed_rooms[seed_index].get("type"))
        su, sv = seed_centroids[seed_index]
        for plan_index, room in enumerate(plan_rooms):
            plan_uv = plan_centroids[plan_index]
            if plan_index in consumed_plan or plan_uv is None or str(room.get("type")) != seed_type:
                continue
            candidate_pairs.append((math.hypot(su - plan_uv[0], sv - plan_uv[1]), seed_index, plan_index))
    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_seed: set[int] = set()
    for _distance, seed_index, plan_index in candidate_pairs:
        if seed_index in matched_seed or plan_index in consumed_plan:
            continue
        consumed_plan.add(plan_index)
        matched_seed.add(seed_index)
        record_match(seed_index, plan_index)

    unmatched = [
        str(seed_rooms[seed_index].get("id"))
        for seed_index in remaining_seed
        if seed_index not in matched_seed
    ]

    # Position component: seed-cell-weighted mean of the per-room position scores, with the
    # weight of every unmatched seed room dragging the mean down at zero credit.
    matched_weight = sum(weight for _, _, _, _, weight, _ in matches)
    unmatched_weight = sum(
        seed_counts[seed_index] for seed_index in remaining_seed if seed_index not in matched_seed
    )
    total_weight = matched_weight + unmatched_weight
    if total_weight:
        position = sum(score * weight for _, _, _, score, weight, _ in matches) / total_weight
    else:
        position = 1.0

    # Arrangement component: fraction of decisive relative orderings preserved per axis.
    considered = 0
    preserved = 0
    for a in range(len(matches)):
        (_, (sua, sva), (pua, pva), _, _, _) = matches[a]
        for b in range(a + 1, len(matches)):
            (_, (sub, svb), (pub, pvb), _, _, _) = matches[b]
            if abs(sua - sub) > 0.08:
                considered += 1
                preserved += (sua - sub > 0) == (pua - pub > 0)
            if abs(sva - svb) > 0.08:
                considered += 1
                preserved += (sva - svb > 0) == (pva - pvb > 0)
    arrangement = preserved / considered if considered else 1.0

    score = round(100 * (0.6 * position + 0.4 * arrangement), 1)
    movers = [
        (seed_id, displacement)
        for seed_id, _, _, position_score, _, displacement in sorted(
            matches, key=lambda item: (item[3], item[0])
        )[:5]
    ]
    return {"score": score, "movers": movers, "unmatched": unmatched}


def format_fidelity_line(fidelity: dict[str, Any]) -> str | None:
    """Compose the one-line seed-fidelity summary, or None when fidelity is undefined."""
    score = fidelity.get("score")
    if score is None:
        return None
    line = f"layout fidelity vs traced seed: {score}%"
    movers = [(room_id, percent) for room_id, percent in fidelity.get("movers", []) if percent > 0]
    if movers:
        line += " — biggest moves: " + ", ".join(
            f"{room_id} (moved {percent:g}%)" for room_id, percent in movers
        )
    unmatched = fidelity.get("unmatched", [])
    if unmatched:
        line += "; missing: " + ", ".join(unmatched)
    return f"[{line}]"


# Blocker categories that must pass before a sub-override candidate is accepted. Area and
# proportion are gated separately on a compliance ratio, so they are deliberately not listed.
HARD_CATEGORIES = {
    "access",
    "adjacency",
    "count",
    "doors",
    "massing",
    "circulation",
    "perimeter",
    "zoning",
}


def has_hard_failures(validation: dict[str, Any]) -> bool:
    """True when any hard-category acceptance check failed."""
    return any(
        check["category"] in HARD_CATEGORIES and not check["pass"]
        for check in validation.get("checks", [])
    )


def candidate_rejection_reason(result: dict[str, Any]) -> str:
    validation = result["validation"]
    area_rows = validation.get("areas", [])
    area_compliance_ratio = (
        sum(bool(row.get("pass")) for row in area_rows) / len(area_rows)
        if area_rows
        else 0.0
    )
    minimum_area_compliance = max(
        0.0,
        min(1.0, float(os.environ.get("AREA_MIN_ACCEPTANCE_RATIO", "0.80"))),
    )
    occupiable_room_count = sum(
        room["type"] not in {"corridor", "circulation"}
        for room in result.get("plan", {}).get("rooms", [])
    )
    proportion_compliance_ratio = (
        float(validation.get("summary", {}).get("proportion_compliant_rooms", 0)) / occupiable_room_count
        if occupiable_room_count
        else 1.0
    )
    minimum_proportion_compliance = max(
        0.0,
        min(1.0, float(os.environ.get("PROPORTION_MIN_ACCEPTANCE_RATIO", "0.90"))),
    )
    area_blocks = area_compliance_ratio < minimum_area_compliance and any(
        check["category"] == "area" and not check["pass"] for check in validation["checks"]
    )
    proportion_blocks = proportion_compliance_ratio < minimum_proportion_compliance and any(
        check["category"] == "proportion" and not check["pass"] for check in validation["checks"]
    )
    blocked = has_hard_failures(validation) or area_blocks or proportion_blocks
    # User decision 2026-07-09 — high-score drafts accept with failures still listed;
    # transparency comes from the validation panel, not from blocking. A draft at or above
    # ACCEPT_SCORE_OVERRIDE is a usable draft even with hard blockers (0 disables the override).
    override = float(os.environ.get("ACCEPT_SCORE_OVERRIDE", "90"))
    if override > 0 and validation["score"] >= override:
        return ""
    if validation["score"] >= 72 and not blocked:
        return ""

    return compact_validation_feedback(validation)


def make_iteration(
    attempt: int,
    phase: str,
    source: str,
    model: str | None,
    status: str,
    started: float,
    message: str,
    result: dict[str, Any] | None,
    code: str | None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "phase": phase,
        "source": source,
        "model": model,
        "status": status,
        "duration_ms": round((time.perf_counter() - started) * 1000),
        "score": result["validation"]["score"] if result else None,
        "message": message,
        "issues": result["validation"]["issues"][:8] if result else [],
        "code": code,
        "usage": usage,
    }


def execute_code_action(payload: dict[str, Any]) -> dict[str, Any]:
    program = normalize_program(payload.get("program"))
    code = str(payload.get("code", ""))
    if not code.strip():
        raise ValueError("No Python code was provided.")
    action = str(payload.get("action", "run"))
    if action not in {"run", "inspect"}:
        raise ValueError("Execution action must be run or inspect.")
    started = time.perf_counter()
    result = execute_and_validate(code, program, enforce_ai_contract=True)
    rejection = candidate_rejection_reason(result)
    status = "accepted" if not rejection else "rejected"
    message = (
        "Code executed and passed the acceptance threshold."
        if not rejection
        else rejection
    )
    iteration = make_iteration(1, action, "user-code", None, status, started, message, result, result["code"])
    return {
        **result,
        "accepted": not rejection,
        "source": "manual-run" if action == "run" else "deterministic-inspection",
        "provider": None,
        "model": None,
        "strategy": "User-edited Python executed in the isolated floor-plan runtime.",
        "assumptions": [],
        "program": program,
        "prompt": str(payload.get("prompt", "")),
        "iterations": [iteration],
        "inspection": {
            "accepted": not rejection,
            "summary": message,
            "issues": result["validation"]["issues"],
        },
    }


def generate_plan(
    payload: dict[str, Any],
    *,
    seed_repair: bool = False,
    preseeded_iterations: list[dict[str, Any]] | None = None,
    initial_error: str | None = None,
    initial_validation: dict[str, Any] | None = None,
    fidelity_seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    program = normalize_program(payload.get("program"))
    action = str(payload.get("action", "generate"))
    if action not in {"generate", "fix", "revise"}:
        raise ValueError("Agent action must be generate, fix, or revise.")
    design_request = str(payload.get("prompt", ""))
    current_code = str(payload.get("current_code", ""))
    if action in {"fix", "revise"} and not current_code.strip():
        raise ValueError(f"The {action} action requires the current complete Python program.")

    options = normalize_options(payload.get("options"))
    # A seed carries its own physical scale; everything else derives it from the program.
    scale_override = (payload.get("options") or {}).get("meters_per_cell") if isinstance(payload.get("options"), dict) else None
    options["meters_per_cell"] = (
        round(float(scale_override), 4)
        if scale_override
        else round(calculate_scale(program, options["width"], options["height"]), 4)
    )
    reference_image = normalize_reference_image(payload.get("reference_image"))
    mode = str(payload.get("mode", "auto"))
    api_key = os.environ.get("GEMINI_API_KEY", "")
    quality_model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    fast_model = os.environ.get("GEMINI_FAST_MODEL", "gemini-3.1-flash-lite")
    fast_thinking = os.environ.get("GEMINI_FAST_THINKING_LEVEL", "minimal")
    quality_thinking = os.environ.get("GEMINI_THINKING_LEVEL", "low")
    repair_thinking = os.environ.get("GEMINI_REPAIR_THINKING_LEVEL", quality_thinking)
    complex_model = os.environ.get("GEMINI_COMPLEX_MODEL", quality_model)
    complex_thinking = os.environ.get("GEMINI_COMPLEX_THINKING_LEVEL", "medium")
    max_attempts = max(1, min(5, int(os.environ.get("GEMINI_MAX_ATTEMPTS", "5"))))
    max_total_tokens = max(0, int(os.environ.get("GEMINI_MAX_TOTAL_TOKENS", "0")))
    max_budget_usd = max(0.0, float(os.environ.get("GEMINI_MAX_BUDGET_USD", "0")))
    # docs #8: behind the flag the model self-checks drafts against the real sandbox+validator
    # mid-attempt; unset leaves generation byte-identical (attempt_executor stays None).
    attempt_tools_on = attempt_tools_enabled()
    requested_room_count = sum(room["count"] for room in program["rooms"])
    complex_program = "tower" in program["building_type"].lower() or requested_room_count >= 40
    iterations: list[dict[str, Any]] = list(preseeded_iterations or [])

    def publish_checkpoint(
        phase: str,
        *,
        result: dict[str, Any] | None = None,
        code: str | None = None,
        source: str | None = None,
        model: str | None = None,
        message: str | None = None,
    ) -> None:
        publish_job_progress(
            iterations,
            phase=phase,
            result=result,
            code=code,
            source=source,
            model=model,
            message=message,
        )

    if mode != "auto":
        raise ValueError("Generation mode must be auto.")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured. Add the key before running the coding agent.")

    previous_code = current_code
    latest_error = ""
    first_error = ""
    previous_validation: dict[str, Any] | None = initial_validation
    best_result: dict[str, Any] | None = None
    best_candidate: dict[str, Any] | None = None
    spent_tokens = 0
    spent_usd = 0.0
    stop_reason = "attempts_exhausted"
    # Refine-mode layout fidelity: the last score measured against the traced seed (None off-refine).
    last_fidelity_score: float | None = None

    if action == "fix":
        if initial_error is not None:
            # The caller (seed refine) already executed the code; skip the probe.
            latest_error = initial_error
        else:
            try:
                current_result = execute_and_validate(current_code, program, enforce_ai_contract=True)
                latest_error = candidate_rejection_reason(current_result)
                if not latest_error:
                    latest_error = "The current code runs, but the user requested an additional quality pass."
            except Exception as error:
                latest_error = str(error)
    elif action == "revise":
        latest_error = f"User revision request: {design_request or 'Improve the current plan.'}"

    for attempt_number in range(1, max_attempts + 1):
        if (max_total_tokens and spent_tokens >= max_total_tokens) or (
            max_budget_usd and spent_usd >= max_budget_usd
        ):
            latest_error = (
                f"Stopped before attempt {attempt_number}: spent {spent_tokens} tokens "
                f"(~${spent_usd:.3f}) against limits {max_total_tokens or 'off'} tokens / "
                f"${max_budget_usd or 'off'}."
            )
            stop_reason = "budget_exhausted"
            break
        started = time.perf_counter()
        candidate: dict[str, Any] | None = None
        result: dict[str, Any] | None = None
        if action == "generate" and attempt_number == 1:
            model = complex_model if complex_program else fast_model
            thinking_level = complex_thinking if complex_program else fast_thinking
            phase = "initial"
            agent_context = None
        else:
            model = complex_model if complex_program else quality_model
            thinking_level = (
                complex_thinking
                if complex_program
                else (repair_thinking if attempt_number > 1 or action == "fix" else quality_thinking)
            )
            phase = action if attempt_number == 1 and action != "generate" else "repair"
            if action == "revise" and attempt_number == 1:
                agent_context = (
                    "Revise the existing complete floor-plan program according to the user request. "
                    "Preserve working behavior that is not affected. Return the entire revised program, not a patch.\n\n"
                    f"User request:\n{design_request or 'Improve the current plan.'}\n\n"
                    f"Existing complete program:\n{previous_code}"
                )
            else:
                agent_context = (
                    f"Code-agent iteration {attempt_number} of {max_attempts}.\n"
                    f"The previous complete program failed or was rejected with:\n{latest_error}\n\n"
                    f"Previous complete program:\n{previous_code}\n\n"
                    "Diagnose the exact failure and return a complete revised program. Do not return a patch."
                )
        running_iteration = {
            "attempt": attempt_number,
            "phase": phase,
            "source": "gemini",
            "model": model,
            "status": "running",
            "duration_ms": 0,
            "started_at_ms": round(time.time() * 1000),
            "score": None,
            "message": f"Waiting for {model} to return a complete Python program.",
            "issues": [],
            "code": None,
        }
        append_job_event({"e": "attempt_start", "attempt": attempt_number, "phase": phase, "model": model})
        publish_job_progress(
            [*iterations, running_iteration],
            phase=phase,
            source="gemini",
            model=model,
            message=running_iteration["message"],
        )
        attempt_executor = None
        if attempt_tools_on:
            def attempt_executor(code: str, _attempt: int = attempt_number) -> dict[str, Any]:
                """Run one candidate through the real sandbox+validator for the model's self-check (docs #8)."""
                try:
                    tool_result = execute_and_validate(code, program, enforce_ai_contract=True)
                    reason = candidate_rejection_reason(tool_result)
                    feedback = {
                        "score": tool_result["validation"]["score"],
                        "rejected": bool(reason),
                        "feedback": (reason or "passes acceptance")[:4000],
                    }
                except Exception as tool_error:
                    feedback = {"score": 0, "rejected": True, "feedback": str(tool_error)[:4000]}
                append_job_event({
                    "e": "tool_check",
                    "attempt": _attempt,
                    "score": feedback["score"],
                    "rejected": feedback["rejected"],
                })
                return feedback
        try:
            candidate = generate_gemini_code(
                api_key,
                model,
                program,
                design_request,
                options,
                agent_context,
                thinking_level=thinking_level,
                reference_image=reference_image,
                seed_repair=seed_repair,
                attempt_executor=attempt_executor,
                max_tool_calls=2,
            )
            append_job_event({
                "e": "model_returned",
                "attempt": attempt_number,
                "model": candidate["model"],
                "tokens": candidate.get("usage", {}).get("total_tokens"),
            })
            usage_row = candidate.get("usage") or {}
            spent_tokens += int(usage_row.get("total_tokens") or 0)
            spent_usd += float(usage_row.get("estimated_cost_usd") or 0.0)
            previous_code = candidate["code"]
            result = execute_and_validate(previous_code, program, enforce_ai_contract=True)
            latest_error = candidate_rejection_reason(result)
            append_job_event({
                "e": "validator_verdict",
                "attempt": attempt_number,
                "score": result["validation"]["score"],
                "rejected": bool(latest_error),
            })
            prior_validation = previous_validation
            previous_validation = result["validation"]
            if latest_error and prior_validation is not None:
                latest_error = validation_delta(prior_validation, result["validation"]) + "\n" + latest_error
            # Refine only: measure similarity to the traced seed so preservation is enforced,
            # not merely suggested. A fidelity drop is a repair signal, and the floor
            # (REFINE_FIDELITY_MIN) rejects an otherwise-passing but drifted candidate.
            # User decision 2026-07-10: relative-layout preservation is a first-class
            # acceptance criterion in refine mode, so the floor defaults ON at 60 (0 disables).
            if fidelity_seed is not None:
                fidelity = layout_fidelity(fidelity_seed, result["plan"])
                last_fidelity_score = fidelity.get("score")
                fidelity_line = format_fidelity_line(fidelity)
                fidelity_floor = max(0.0, float(os.environ.get("REFINE_FIDELITY_MIN", "60")))
                if (
                    fidelity_floor > 0
                    and last_fidelity_score is not None
                    and last_fidelity_score < fidelity_floor
                    and not latest_error
                ):
                    drift_note = (
                        "the layout drifted too far from the traced seed; "
                        "move rooms back toward their seed positions"
                    )
                    latest_error = f"{fidelity_line}\n{drift_note}" if fidelity_line else drift_note
                elif latest_error and fidelity_line and fidelity_line not in latest_error:
                    latest_error = f"{latest_error}\n{fidelity_line}"
            if best_result is None or result["validation"]["score"] > best_result["validation"]["score"]:
                best_result = result
                best_candidate = candidate
            if not latest_error:
                iterations.append(
                    make_iteration(
                        attempt_number,
                        phase,
                        "gemini",
                        candidate["model"],
                        "accepted",
                        started,
                        f"Agent {phase} program passed execution and architectural validation.",
                        result,
                        previous_code,
                        candidate.get("usage"),
                    )
                )
                append_job_event({"e": "attempt_end", "attempt": attempt_number, "status": "accepted"})
                publish_checkpoint(
                    phase,
                    result=result,
                    code=previous_code,
                    source="gemini",
                    model=candidate["model"],
                    message=iterations[-1]["message"],
                )
                if action == "generate":
                    source = "gemini" if attempt_number == 1 else "gemini-repaired"
                else:
                    source = "gemini-fixed" if action == "fix" else "gemini-revised"
                append_job_event({"e": "run_end", "stop_reason": "accepted", "accepted": True})
                return {
                    **result,
                    "accepted": True,
                    "source": source,
                    "provider": "Google Gemini",
                    "model": candidate["model"],
                    "strategy": candidate["strategy"],
                    "assumptions": candidate["assumptions"],
                    "usage": candidate.get("usage"),
                    "usage_total": {"total_tokens": spent_tokens, "estimated_cost_usd": round(spent_usd, 6)},
                    "stop_reason": "accepted",
                    "seed_fidelity": last_fidelity_score,
                    "program": program,
                    "prompt": design_request,
                    "repair_note": first_error if attempt_number > 1 else None,
                    "iterations": iterations,
                    # Accepted despite hard-category failures => the score override let it through.
                    **({"accepted_via": "score_override"} if has_hard_failures(result["validation"]) else {}),
                }
            if best_result is not result and best_result is not None and best_candidate is not None:
                regressed_error = latest_error
                previous_code = best_candidate["code"]
                # Next delta must compare against the code the model is actually shown.
                previous_validation = best_result["validation"]
                latest_error = (
                    "The latest executable candidate regressed. Continue from the best executable program below. "
                    f"Latest candidate feedback: {regressed_error} "
                    f"Best candidate feedback: {candidate_rejection_reason(best_result)}"
                )
            status = "rejected"
        except Exception as error:
            attempt_error = str(error)
            if best_result is not None and best_candidate is not None:
                previous_code = best_candidate["code"]
                # Next delta must compare against the code the model is actually shown.
                previous_validation = best_result["validation"]
                latest_error = (
                    "The latest attempted edit failed before validation. Continue from the best executable program below. "
                    f"Failed edit error: {attempt_error} "
                    f"Best candidate feedback: {candidate_rejection_reason(best_result)}"
                )
            else:
                latest_error = attempt_error
            status = "failed"

        if attempt_number == 1:
            first_error = latest_error
        iterations.append(
            make_iteration(
                attempt_number,
                phase,
                "gemini",
                candidate["model"] if candidate else model,
                status,
                started,
                latest_error,
                result,
                candidate["code"] if candidate else None,
                candidate.get("usage") if candidate else None,
            )
        )
        append_job_event({"e": "attempt_end", "attempt": attempt_number, "status": status})
        publish_checkpoint(
            phase,
            result=result,
            code=candidate["code"] if candidate else None,
            source="gemini",
            model=candidate["model"] if candidate else model,
            message=iterations[-1]["message"],
        )

    if stop_reason == "budget_exhausted":
        exhaustion = latest_error
    else:
        exhaustion = f"The floor-plan coding agent exhausted {max_attempts} attempts. Last failure: {latest_error}"
    if best_result is None or best_candidate is None:
        raise ValueError(exhaustion)
    result_source = "gemini-unaccepted" if action == "generate" else f"gemini-{action}-unaccepted"
    append_job_event({"e": "run_end", "stop_reason": stop_reason, "accepted": False})
    return {
        **best_result,
        "accepted": False,
        "source": result_source,
        "provider": "Google Gemini",
        "model": best_candidate["model"],
        "strategy": best_candidate["strategy"],
        "assumptions": best_candidate["assumptions"],
        "usage": best_candidate.get("usage"),
        "usage_total": {"total_tokens": spent_tokens, "estimated_cost_usd": round(spent_usd, 6)},
        "stop_reason": stop_reason,
        "seed_fidelity": last_fidelity_score,
        "program": program,
        "prompt": design_request,
        "ai_error": exhaustion,
        "iterations": iterations,
    }


def generate_images_action(payload: dict[str, Any]) -> dict[str, Any]:
    """Candidate plan drawings for image mode; the user picks one in the UI."""
    program = normalize_program(payload.get("program"))
    count = max(1, min(4, int(payload.get("count", 3) or 3)))
    api_key = os.environ.get("GEMINI_API_KEY", "")
    publish_job_progress(
        [],
        phase="images",
        source="gemini-image",
        model=os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image"),
        message=f"Drawing {count} candidate floor plans.",
    )
    images = generate_plan_images(api_key, program, count)
    return {"images": images, "program": program, "count": len(images)}


def refine_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Trace mode: deterministic seed translation as attempt 0, then the normal repair loop."""
    program = normalize_program(payload.get("program"))
    seed = normalize_seed(payload.get("seed"))
    seed_program_code = seed_to_code(seed)
    started = time.perf_counter()
    append_job_event({"e": "attempt_start", "attempt": 0, "phase": "seed"})
    publish_job_progress(
        [],
        phase="seed",
        source="seed-translator",
        message="Executing the deterministic translation of the traced plan.",
    )
    result: dict[str, Any] | None = None
    rejection = ""
    try:
        result = execute_and_validate(seed_program_code, program, enforce_ai_contract=True)
        rejection = candidate_rejection_reason(result)
    except Exception as error:
        rejection = str(error)
    message = (
        "Seed plan executed and passed validation; no AI repair was needed."
        if result is not None and not rejection
        else rejection
    )
    seed_iteration = make_iteration(
        0,
        "seed",
        "seed-translator",
        None,
        "accepted" if result is not None and not rejection else "rejected",
        started,
        message,
        result,
        seed_program_code,
    )
    publish_job_progress(
        [seed_iteration],
        phase="seed",
        result=result,
        code=seed_program_code,
        source="seed-translator",
        message=message,
    )
    append_job_event({
        "e": "validator_verdict",
        "attempt": 0,
        "score": result["validation"]["score"] if result is not None else None,
        "rejected": bool(rejection),
    })
    append_job_event({"e": "attempt_end", "attempt": 0, "status": seed_iteration["status"]})
    if result is not None and not rejection:
        append_job_event({"e": "run_end", "stop_reason": "accepted", "accepted": True})
        return {
            **result,
            "accepted": True,
            "source": "seed-translated",
            "provider": None,
            "model": None,
            "strategy": "Deterministic translation of the traced seed plan; the validator passed without AI repair.",
            "assumptions": [],
            "stop_reason": "accepted",
            "usage_total": {"total_tokens": 0, "estimated_cost_usd": 0.0},
            # The seed against itself is 100.0; computed anyway so every refine result carries the field.
            "seed_fidelity": layout_fidelity(seed, result["plan"]).get("score"),
            "program": program,
            "prompt": str(payload.get("prompt", "")),
            "iterations": [seed_iteration],
            # Accepted despite hard-category failures => the score override let it through.
            **({"accepted_via": "score_override"} if has_hard_failures(result["validation"]) else {}),
        }

    repair_payload = {
        **payload,
        "action": "fix",
        "current_code": seed_program_code,
        "options": {
            "width": seed["width"],
            "height": seed["height"],
            "meters_per_cell": seed["meters_per_cell"],
        },
    }
    outcome = generate_plan(
        repair_payload,
        seed_repair=True,
        preseeded_iterations=[seed_iteration],
        initial_error=rejection,
        initial_validation=result["validation"] if result is not None else None,
        fidelity_seed=seed,
    )
    source_map = {"gemini-fixed": "seed-repaired", "gemini-fix-unaccepted": "seed-repair-unaccepted"}
    outcome["source"] = source_map.get(outcome["source"], outcome["source"])
    return outcome


def dispatch_job(payload: dict[str, Any]) -> dict[str, Any]:
    job_kind = str(payload.get("job_kind", "agent"))
    if job_kind == "execute":
        return execute_code_action(payload)
    if job_kind == "agent":
        return generate_plan(payload)
    if job_kind == "images":
        return generate_images_action(payload)
    if job_kind == "refine":
        return refine_plan(payload)
    if job_kind == "room":
        from room_service import generate_room_plan  # lazy import keeps the room flow isolated

        return generate_room_plan(payload)
    raise ValueError("Unknown job kind.")
