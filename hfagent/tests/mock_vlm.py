# -*- coding: utf-8 -*-
"""Scripted mock VLM for orchestration tests.

Each round of the real2color pipeline makes exactly 2 generate_image calls:
  call 1 — realistic plan prompt
  call 2 — convert to colour-block
The direct_colorblock structure_mode makes exactly 1 generate_image call per round
(program -> colour-block, no realistic plan); construct MockVLM with
image_calls_per_round=1 for that mode.

MockVLM returns the nth script plan for every call in round n, advancing to the
next plan every `image_calls_per_round` calls. This isolates orchestration logic
from real VLM noise.

generate_json is also stubbed: returns `door_script` as a JSON array on each call,
defaulting to an empty list (no doors). Tests that exercise door extraction pass
a list of door dicts: [{"room_a": "...", "room_b": "...", "x": 0.5, "y": 0.5}].
"""
from __future__ import annotations

import io
import json
from collections import Counter

from hfagent.schema.plan import Plan
from hfagent.tools.render_plan import render_plan


class MockVLM:
    image_model = "mock-vlm"
    text_model = "mock-llm"

    def __init__(
        self,
        script: list[Plan],
        door_script: list[dict] | None = None,
        image_calls_per_round: int = 2,
    ):
        self.script = script
        self.door_script: list[dict] = door_script or []
        self.image_calls_per_round = image_calls_per_round
        self.calls = 0
        self.feedbacks: list[str] = []
        self.last_plan: Plan = script[0]

    def generate_image(self, contents) -> bytes:
        if isinstance(contents, list):
            feedback_str = next((c for c in contents if isinstance(c, str)), None)
            if feedback_str:
                self.feedbacks.append(feedback_str)

        round_idx = self.calls // self.image_calls_per_round
        plan = self.script[min(round_idx, len(self.script) - 1)]
        self.last_plan = plan
        self.calls += 1

        buf = io.BytesIO()
        render_plan(plan, px_per_mm=0.05).save(buf, "PNG")
        return buf.getvalue()

    def generate_json(self, contents, schema=None) -> str:
        # structure read (json mode) asks for an object with "rooms"; door read asks
        # for a plain array. Distinguish by the schema so both paths get valid data.
        if isinstance(schema, dict) and "rooms" in (schema.get("properties") or {}):
            return json.dumps(self._structure_for(self.last_plan))
        return json.dumps(self.door_script)

    def _structure_for(self, plan: Plan) -> dict:
        """Derive rooms (normalized rectangles) + doors from the last rendered plan, mirroring
        what a VLM would read back. Each room's bbox is normalized to the plan bounds."""
        xs = [x for r in plan.rooms for x, _ in r.polygon]
        ys = [y for r in plan.rooms for _, y in r.polygon]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        sx, sy = (maxx - minx) or 1.0, (maxy - miny) or 1.0
        totals = Counter(r.type for r in plan.rooms)
        seen: Counter = Counter()
        rooms = []
        for r in plan.rooms:
            seen[r.type] += 1
            rid = r.type if totals[r.type] == 1 else f"{r.type}_{seen[r.type]}"
            rx = [x for x, _ in r.polygon]
            ry = [y for _, y in r.polygon]
            rect = [(min(rx) - minx) / sx, (min(ry) - miny) / sy,
                    (max(rx) - minx) / sx, (max(ry) - miny) / sy]
            rooms.append({"id": rid, "type": r.type, "rects": [rect]})
        return {"rooms": rooms, "doors": self.door_script}
