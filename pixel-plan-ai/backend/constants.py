from __future__ import annotations

PALETTE: dict[str, str] = {
    "waiting": "#E8A23A",
    "exam_room": "#52A6A2",
    "patient_room": "#6E8ED6",
    "nurse_station": "#D96B7B",
    "office": "#8A75C9",
    "toilet": "#5F7890",
    "storage": "#A87854",
    "corridor": "#E8DEC8",
    "circulation": "#E8DEC8",
    "unassigned": "#F2EFE8",
}

DEFAULT_AREAS: dict[str, float] = {
    "waiting": 30.0,
    "exam_room": 16.0,
    "patient_room": 18.0,
    "nurse_station": 20.0,
    "office": 12.0,
    "toilet": 6.0,
    "storage": 10.0,
    "corridor": 35.0,
    "circulation": 35.0,
}


def color_for(room_type: str) -> str:
    return PALETTE.get(room_type, "#9B8D7A")


def area_for(room: dict[str, object]) -> float:
    explicit = room.get("approx_area_m2")
    if isinstance(explicit, (int, float)) and explicit > 0:
        return float(explicit)
    return DEFAULT_AREAS.get(str(room.get("type", "")), 12.0)
