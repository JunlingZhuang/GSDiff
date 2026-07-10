from __future__ import annotations

import builtins
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

from code_policy import AUTHORIZED_IMPORTS, validate_generated_code

# The scaffolding below (restricted_import, SAFE_BUILTINS, plain_value, validate_identifier)
# is intentionally DUPLICATED from sandbox_runner.py. Room generation is a separate, isolated
# pipeline; sharing a base with the floor runner would couple the two flows the isolation rule
# keeps apart. Keep the two runners independent even though the sandboxing is identical.

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "icu_assets.json"
SNAP_FT = 0.25
FOOTPRINT_TOLERANCE_FT = 0.01
BOUNDS_TOLERANCE_FT = 0.01
VALID_WALLS = {"N", "S", "E", "W"}
VALID_ROTATIONS = {0, 90, 180, 270}


def restricted_import(
    name: str,
    globals_value: dict[str, object] | None = None,
    locals_value: dict[str, object] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> object:
    if level or name.split(".", 1)[0] not in AUTHORIZED_IMPORTS:
        raise ImportError(f"Import is not authorized: {name}")
    return builtins.__import__(name, globals_value, locals_value, fromlist, level)


SAFE_BUILTINS = {
    "__build_class__": builtins.__build_class__,
    "__import__": restricted_import,
    "abs": abs,
    "all": all,
    "any": any,
    "BaseException": BaseException,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "IndexError": IndexError,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "KeyError": KeyError,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "OverflowError": OverflowError,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "RuntimeError": RuntimeError,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "StopIteration": StopIteration,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
    "ZeroDivisionError": ZeroDivisionError,
}


def plain_value(value: Any) -> Any:
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, dict):
        return {str(key): plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"Result contains unsupported value type: {type(value).__name__}.")


def validate_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 80:
        raise ValueError(f"{label} is invalid.")
    if any(not (character.isalnum() or character in "_-") for character in value):
        raise ValueError(f"{label} contains invalid characters.")
    return value


def load_catalog() -> dict[str, dict[str, Any]]:
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assets = raw.get("assets") if isinstance(raw, dict) else None
    if not isinstance(assets, list) or not assets:
        raise ValueError("The ICU asset catalog is invalid.")
    catalog: dict[str, dict[str, Any]] = {}
    for entry in assets:
        catalog[str(entry["type"])] = entry
    return catalog


def as_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    return float(value)


def snap(value: float) -> float:
    """Round to the 0.25 ft planning grid, stripping binary-float dust."""
    return round(round(value / SNAP_FT) * SNAP_FT, 4)


def normalize_room(raw_room: Any) -> dict[str, Any]:
    if not isinstance(raw_room, dict):
        raise ValueError("result['room'] must be a dictionary.")
    width = snap(as_number(raw_room.get("width_ft"), "room width_ft"))
    depth = snap(as_number(raw_room.get("depth_ft"), "room depth_ft"))
    if not 0 < width <= 100 or not 0 < depth <= 100:
        raise ValueError("Room width_ft and depth_ft must be positive and at most 100 ft.")
    raw_door = raw_room.get("door")
    if not isinstance(raw_door, dict):
        raise ValueError("result['room']['door'] must be a dictionary.")
    wall = raw_door.get("wall")
    if wall not in VALID_WALLS:
        raise ValueError("Door wall must be one of N, S, E, or W.")
    offset = snap(as_number(raw_door.get("offset_ft"), "door offset_ft"))
    door_width = snap(as_number(raw_door.get("width_ft"), "door width_ft"))
    if door_width <= 0:
        raise ValueError("Door width_ft must be positive.")
    wall_length = width if wall in {"N", "S"} else depth
    if offset - door_width / 2 < -BOUNDS_TOLERANCE_FT or offset + door_width / 2 > wall_length + BOUNDS_TOLERANCE_FT:
        raise ValueError("Door opening extends past the end of its wall.")
    return {
        "width_ft": width,
        "depth_ft": depth,
        "door": {"wall": wall, "offset_ft": offset, "width_ft": door_width},
    }


def normalize_assets(raw_assets: Any, room: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("result['assets'] must be a non-empty list.")
    width = room["width_ft"]
    depth = room["depth_ft"]
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_assets:
        if not isinstance(raw, dict):
            raise ValueError("Each asset must be a dictionary.")
        asset_id = validate_identifier(raw.get("id"), "Asset id")
        if asset_id in seen_ids:
            raise ValueError(f"Duplicate asset id: {asset_id}.")
        seen_ids.add(asset_id)
        asset_type = raw.get("type")
        if asset_type not in catalog:
            raise ValueError(f"Asset {asset_id} has unknown type: {asset_type}.")
        rotation_raw = raw.get("rotation_deg", 0)
        if isinstance(rotation_raw, bool) or not isinstance(rotation_raw, (int, float)) or int(rotation_raw) not in VALID_ROTATIONS:
            raise ValueError(f"Asset {asset_id} rotation_deg must be 0, 90, 180, or 270.")
        rotation = int(rotation_raw)
        x = snap(as_number(raw.get("x_ft"), f"Asset {asset_id} x_ft"))
        y = snap(as_number(raw.get("y_ft"), f"Asset {asset_id} y_ft"))
        w = snap(as_number(raw.get("w_ft"), f"Asset {asset_id} w_ft"))
        d = snap(as_number(raw.get("d_ft"), f"Asset {asset_id} d_ft"))
        if w <= 0 or d <= 0:
            raise ValueError(f"Asset {asset_id} footprint must be positive.")
        if (
            x < -BOUNDS_TOLERANCE_FT
            or y < -BOUNDS_TOLERANCE_FT
            or x + w > width + BOUNDS_TOLERANCE_FT
            or y + d > depth + BOUNDS_TOLERANCE_FT
        ):
            raise ValueError(f"Asset {asset_id} lies outside the room bounds.")
        catalog_w, catalog_d = (float(value) for value in catalog[asset_type]["footprint_ft"])
        expected_w, expected_d = (catalog_d, catalog_w) if rotation in {90, 270} else (catalog_w, catalog_d)
        if abs(w - expected_w) > FOOTPRINT_TOLERANCE_FT or abs(d - expected_d) > FOOTPRINT_TOLERANCE_FT:
            raise ValueError(
                f"Asset {asset_id} footprint {w} x {d} does not match the catalog footprint for "
                f"{asset_type} at rotation {rotation} (expected {expected_w} x {expected_d})."
            )
        wall = raw.get("wall")
        if wall is not None and wall not in VALID_WALLS:
            raise ValueError(f"Asset {asset_id} wall must be one of N, S, E, W, or null.")
        normalized.append(
            {
                "id": asset_id,
                "type": asset_type,
                "x_ft": x,
                "y_ft": y,
                "w_ft": w,
                "d_ft": d,
                "rotation_deg": rotation,
                "wall": wall,
                "anchor": str(catalog[asset_type]["anchor"]),
            }
        )
    return normalized


def normalize_result(raw_result: Any) -> dict[str, Any]:
    result = plain_value(raw_result)
    if not isinstance(result, dict):
        raise ValueError("Top-level variable result must be a dictionary.")
    catalog = load_catalog()
    room = normalize_room(result.get("room"))
    assets = normalize_assets(result.get("assets"), room, catalog)
    return {"room": room, "assets": assets}


def main() -> int:
    code = sys.stdin.read()
    try:
        tree = validate_generated_code(code)
        scope: dict[str, object] = {
            "__builtins__": SAFE_BUILTINS,
            "__name__": "generated_room",
        }
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            exec(compile(tree, "generated_room.py", "exec"), scope, scope)
        raw_result = scope.get("result")
        if raw_result is None:
            raise ValueError("Generated code did not produce top-level variable result.")
        plan = normalize_result(raw_result)
        print(json.dumps({"ok": True, "result": plan}, separators=(",", ":")))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
