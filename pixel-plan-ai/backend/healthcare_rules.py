from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "healthcare_rules_us.json"
HEALTHCARE_TERMS = {
    "clinic",
    "health",
    "hospital",
    "inpatient",
    "outpatient",
    "ward",
}


@lru_cache(maxsize=1)
def load_healthcare_rules() -> dict[str, Any]:
    value = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("room_types"), dict):
        raise ValueError("The U.S. healthcare rules file is invalid.")
    return value


def uses_healthcare_rules(program: dict[str, Any]) -> bool:
    building_type = str(program.get("building_type", "")).lower()
    return any(term in building_type for term in HEALTHCARE_TERMS)


def room_rule(room_type: str) -> dict[str, Any] | None:
    value = load_healthcare_rules()["room_types"].get(room_type)
    return value if isinstance(value, dict) else None


def rules_for_prompt(program: dict[str, Any]) -> dict[str, Any]:
    rules = load_healthcare_rules()
    requested_types = {str(room.get("type", "")) for room in program.get("rooms", [])}
    return {
        "profile": rules["profile"],
        "jurisdiction": rules["jurisdiction"],
        "status": rules["status"],
        "allowed_module_containment": rules["allowed_module_containment"],
        "door_swing_defaults": rules.get("door_swing_defaults", {}),
        "room_types": {
            room_type: rule
            for room_type, rule in rules["room_types"].items()
            if room_type in requested_types
        },
    }
