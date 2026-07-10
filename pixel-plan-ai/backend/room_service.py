from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from gemini import generate_room_code
from job_progress import append_job_event, publish_job_progress
from room_prompt import build_room_prompt
from room_runtime import execute_room_code
from room_validator import validate_room
# Reused from the floor service unchanged (isolation rule allows importing these).
from service import failed_check_keys, make_iteration, validation_delta

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Hard blocker categories for the ICU room flow: a failure in any of these blocks acceptance
# below the score override, mirroring the floor service's HARD_CATEGORIES gate.
ROOM_HARD_CATEGORIES = {"count", "overlap", "clearance", "access", "anchor"}


@lru_cache(maxsize=1)
def load_room_rules() -> dict[str, Any]:
    value = json.loads((DATA_DIR / "icu_room_rules.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "room" not in value:
        raise ValueError("The ICU room rules file is invalid.")
    return value


@lru_cache(maxsize=1)
def load_room_catalog() -> dict[str, Any]:
    value = json.loads((DATA_DIR / "icu_assets.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("assets"), list):
        raise ValueError("The ICU asset catalog is invalid.")
    return value


def snap_ft(value: float) -> float:
    return round(round(value / 0.25) * 0.25, 4)


def normalize_room_request(payload: dict[str, Any]) -> dict[str, Any]:
    room = payload.get("room")
    if not isinstance(room, dict):
        raise ValueError("payload['room'] must be an object with width_ft and depth_ft.")
    try:
        width = snap_ft(float(room["width_ft"]))
        depth = snap_ft(float(room["depth_ft"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Room width_ft and depth_ft are required numbers.") from error
    if not 8.0 <= width <= 40.0 or not 8.0 <= depth <= 40.0:
        raise ValueError("Room width_ft and depth_ft must each be between 8 and 40 ft.")
    if width * depth < 150.0:
        raise ValueError("Room area must be at least 150 sf.")
    return {"width_ft": width, "depth_ft": depth, "prompt": str(payload.get("prompt", "") or "")}


def compact_room_feedback(validation: dict[str, Any]) -> str:
    failed = [check for check in validation.get("checks", []) if not check.get("pass")]
    sections = [f"Validation score: {validation.get('score', 0)}."]
    if failed:
        lines = [f"{check.get('category', 'general')}: {check.get('label', 'failed')}" for check in failed]
        sections.append("Failed checks:\n- " + "\n- ".join(lines[:20]))
    issues = [str(issue) for issue in validation.get("issues", [])]
    if issues:
        sections.append("Issues:\n- " + "\n- ".join(issues[:12]))
    return "\n".join(sections)


def room_rejection_reason(validation: dict[str, Any]) -> str:
    """Empty string when the candidate is acceptable, else the compact repair feedback."""
    override = float(os.environ.get("ACCEPT_SCORE_OVERRIDE", "90"))
    hard_fail = any(
        check["category"] in ROOM_HARD_CATEGORIES and not check["pass"]
        for check in validation.get("checks", [])
    )
    if override > 0 and validation["score"] >= override:
        return ""
    if validation["score"] >= 72 and not hard_fail:
        return ""
    return compact_room_feedback(validation)


def generate_room_plan(payload: dict[str, Any]) -> dict[str, Any]:
    room_request = normalize_room_request(payload)
    rules = load_room_rules()
    catalog = load_room_catalog()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured. Add the key before running the room agent.")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    thinking_level = os.environ.get("GEMINI_ROOM_THINKING_LEVEL", os.environ.get("GEMINI_THINKING_LEVEL", "low"))
    max_attempts = max(1, min(5, int(os.environ.get("GEMINI_ROOM_MAX_ATTEMPTS", os.environ.get("GEMINI_MAX_ATTEMPTS", "5")))))

    iterations: list[dict[str, Any]] = []
    previous_code = ""
    previous_validation: dict[str, Any] | None = None
    latest_error = ""
    best: dict[str, Any] | None = None  # {"code","room","validation","candidate"}
    spent_tokens = 0
    spent_usd = 0.0
    stop_reason = "attempts_exhausted"

    for attempt_number in range(1, max_attempts + 1):
        started = time.perf_counter()
        phase = "initial" if attempt_number == 1 else "repair"
        prompt_request = {
            **room_request,
            "repair_context": None if attempt_number == 1 else latest_error,
            "previous_code": None if attempt_number == 1 else previous_code,
        }
        prompt_text = build_room_prompt(prompt_request, rules, catalog)
        append_job_event({"e": "attempt_start", "attempt": attempt_number, "phase": phase, "model": model})
        publish_job_progress(
            [*iterations, {
                "attempt": attempt_number,
                "phase": phase,
                "source": "gemini-room",
                "model": model,
                "status": "running",
                "duration_ms": 0,
                "started_at_ms": round(time.time() * 1000),
                "score": None,
                "message": f"Waiting for {model} to return a complete room program.",
                "issues": [],
                "code": None,
            }],
            phase=phase,
            source="gemini-room",
            model=model,
            message=f"Waiting for {model} to return a complete room program.",
        )
        candidate: dict[str, Any] | None = None
        result: dict[str, Any] | None = None
        try:
            candidate = generate_room_code(api_key, model, prompt_text, thinking_level=thinking_level)
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
            sanitized_code, room = execute_room_code(previous_code)
            validation = validate_room(room, rules, catalog)
            result = {"code": sanitized_code, "room": room, "validation": validation}
            latest_error = room_rejection_reason(validation)
            append_job_event({
                "e": "validator_verdict",
                "attempt": attempt_number,
                "score": validation["score"],
                "rejected": bool(latest_error),
            })
            if latest_error and previous_validation is not None:
                latest_error = validation_delta(previous_validation, validation) + "\n" + latest_error
            previous_validation = validation
            if best is None or validation["score"] > best["validation"]["score"]:
                best = {**result, "candidate": candidate}
            status = "accepted" if not latest_error else "rejected"
        except Exception as error:  # execution or model failure: keep looping with the message
            latest_error = str(error)
            status = "failed"

        iterations.append(
            make_iteration(
                attempt_number,
                phase,
                "gemini-room",
                candidate["model"] if candidate else model,
                status,
                started,
                (f"Room {phase} program passed the schematic validator." if status == "accepted" else latest_error),
                result,
                candidate["code"] if candidate else None,
                candidate.get("usage") if candidate else None,
            )
        )
        append_job_event({"e": "attempt_end", "attempt": attempt_number, "status": status})
        publish_job_progress(
            iterations,
            phase=phase,
            result=result,
            code=candidate["code"] if candidate else None,
            source="gemini-room",
            model=candidate["model"] if candidate else model,
            message=iterations[-1]["message"],
        )

        if result is not None and not latest_error:
            stop_reason = "accepted"
            source = "gemini-room" if attempt_number == 1 else "gemini-room-repaired"
            append_job_event({"e": "run_end", "stop_reason": "accepted", "accepted": True})
            return _envelope(
                accepted=True,
                source=source,
                result=result,
                candidate=candidate,
                room_request=room_request,
                iterations=iterations,
                stop_reason="accepted",
                spent_tokens=spent_tokens,
                spent_usd=spent_usd,
            )

    append_job_event({"e": "run_end", "stop_reason": stop_reason, "accepted": False})
    if best is None:
        raise ValueError(f"The ICU room coding agent exhausted {max_attempts} attempts. Last failure: {latest_error}")
    return _envelope(
        accepted=False,
        source="gemini-room-unaccepted",
        result=best,
        candidate=best["candidate"],
        room_request=room_request,
        iterations=iterations,
        stop_reason=stop_reason,
        spent_tokens=spent_tokens,
        spent_usd=spent_usd,
        ai_error=f"The ICU room coding agent exhausted {max_attempts} attempts. Last failure: {latest_error}",
    )


def _envelope(
    *,
    accepted: bool,
    source: str,
    result: dict[str, Any],
    candidate: dict[str, Any] | None,
    room_request: dict[str, Any],
    iterations: list[dict[str, Any]],
    stop_reason: str,
    spent_tokens: int,
    spent_usd: float,
    ai_error: str | None = None,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "accepted": accepted,
        "source": source,
        "provider": "Google Gemini",
        "model": candidate["model"] if candidate else None,
        "strategy": candidate.get("strategy") if candidate else "",
        "assumptions": candidate.get("assumptions") if candidate else [],
        "code": result["code"],
        "room": result["room"],
        "validation": result["validation"],
        "iterations": iterations,
        "program": {"width_ft": room_request["width_ft"], "depth_ft": room_request["depth_ft"]},
        "prompt": room_request.get("prompt", ""),
        "stop_reason": stop_reason,
        "usage": candidate.get("usage") if candidate else None,
        "usage_total": {"total_tokens": spent_tokens, "estimated_cost_usd": round(spent_usd, 6)},
        "seed_fidelity": None,
    }
    if ai_error is not None:
        envelope["ai_error"] = ai_error
    return envelope
