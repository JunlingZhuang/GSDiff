# -*- coding: utf-8 -*-
"""Tool: extract the room adjacency graph (which rooms connect by a door).

The LLM reads ONLY the connectivity from the realistic drawing — which pairs of
rooms have a door between them. It returns no coordinates: door positions are not
reliable from the model, so door_placer puts each door centred on the wall the two
connected rooms physically share in the reconstructed plan.

Public API:
    extract_room_adjacency(real_png, client, program) -> RoomGraph

Nodes come from the program (one per room instance, numbered to match the labels
we asked the generator to draw). Door edges come from a JSON vision call.
"""
from __future__ import annotations

import json

from hfagent.schema.roomgraph import Door, RoomGraph, RoomNode


def room_instance_ids(program: dict) -> list[RoomNode]:
    """Expand the program into one node per room instance, matching image labels.

    count == 1 -> "corridor"; count > 1 -> "patient_room_1" .. "patient_room_N".
    Must stay in sync with build_real_prompt's labelling.
    """
    nodes: list[RoomNode] = []
    for r in program["rooms"]:
        rtype = r["type"]
        count = r.get("count", 1)
        if count == 1:
            nodes.append(RoomNode(id=rtype, type=rtype))
        else:
            nodes.extend(RoomNode(id=f"{rtype}_{i}", type=rtype) for i in range(1, count + 1))
    return nodes


_DOOR_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "room_a": {"type": "string"},
            "room_b": {"type": "string"},
        },
        "required": ["room_a", "room_b"],
    },
}


def _build_door_prompt(nodes: list[RoomNode]) -> str:
    labels = ", ".join(n.id for n in nodes)
    return f"""Look at the architectural floor plan above. Identify which rooms are \
connected by a DOOR — an opening you can walk through, drawn as a gap in a wall with a \
door leaf and/or a quarter-circle swing arc.

Report DOORS ONLY, not windows (a window is a gap in an EXTERIOR wall with thin parallel
glazing lines, NO swing arc and NO leaf — ignore every window).

Return ONLY the connectivity — no coordinates. For each door:
  room_a: label of the first room it connects (use "exterior" only for a real entrance door)
  room_b: label of the second room it connects

Use ONLY these room labels (read the text printed inside each room):
{labels}

Return a JSON array, one {{"room_a", "room_b"}} object per door. No extra text."""


def extract_room_adjacency(real_png: bytes, client, program: dict) -> RoomGraph:
    nodes = room_instance_ids(program)
    valid_ids = {n.id for n in nodes} | {"exterior"}

    raw = client.generate_json([real_png, _build_door_prompt(nodes)], schema=_DOOR_JSON_SCHEMA)
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        items = []
    if not isinstance(items, list):
        items = []

    doors: list[Door] = []
    seen: set[tuple[str, str]] = set()
    for d in items:
        if not isinstance(d, dict):
            continue
        try:
            a, b = str(d["room_a"]), str(d["room_b"])
        except KeyError:
            continue
        # keep only edges between labels we actually drew, dedup, no self-loops
        if a not in valid_ids or b not in valid_ids or a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        doors.append(Door(room_a=a, room_b=b))

    return RoomGraph(rooms=nodes, doors=doors)
