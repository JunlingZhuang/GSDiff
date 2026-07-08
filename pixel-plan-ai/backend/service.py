from __future__ import annotations

import os
import time
from typing import Any

from code_policy import validate_program_contract
from gemini import generate_gemini_code
from job_progress import publish_job_progress
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
    hard_categories = {
        "access",
        "adjacency",
        "count",
        "doors",
        "massing",
        "circulation",
        "perimeter",
        "zoning",
    }
    failed_blockers = [
        check["label"]
        for check in validation["checks"]
        if (
            check["category"] in hard_categories
            or (check["category"] == "area" and area_compliance_ratio < minimum_area_compliance)
            or (
                check["category"] == "proportion"
                and proportion_compliance_ratio < minimum_proportion_compliance
            )
        )
        and not check["pass"]
    ]
    if validation["score"] >= 72 and not failed_blockers:
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
        publish_job_progress(
            [*iterations, running_iteration],
            phase=phase,
            source="gemini",
            model=model,
            message=running_iteration["message"],
        )
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
            )
            usage_row = candidate.get("usage") or {}
            spent_tokens += int(usage_row.get("total_tokens") or 0)
            spent_usd += float(usage_row.get("estimated_cost_usd") or 0.0)
            previous_code = candidate["code"]
            result = execute_and_validate(previous_code, program, enforce_ai_contract=True)
            latest_error = candidate_rejection_reason(result)
            prior_validation = previous_validation
            previous_validation = result["validation"]
            if latest_error and prior_validation is not None:
                latest_error = validation_delta(prior_validation, result["validation"]) + "\n" + latest_error
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
                    "program": program,
                    "prompt": design_request,
                    "repair_note": first_error if attempt_number > 1 else None,
                    "iterations": iterations,
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
    if result is not None and not rejection:
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
            "program": program,
            "prompt": str(payload.get("prompt", "")),
            "iterations": [seed_iteration],
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
    raise ValueError("Unknown job kind.")
