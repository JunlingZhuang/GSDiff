# -*- coding: utf-8 -*-
"""Scripted mock VLM for Phase 0-B orchestration validation.

Plays a predetermined sequence of Plans: round 1 returns plans[0] rendered as a
colour-block PNG, each correction round returns the next plan in the script.
This isolates the ORCHESTRATION (correction loop, best-round, early-stop) from
real VLM noise — exactly the mock-isolation called for by docs/agent §7.3.
"""
from __future__ import annotations

import io

from hfagent.schema.plan import Plan
from hfagent.tools.render_plan import render_plan


class MockVLM:
    image_model = "mock-vlm"
    text_model = "mock-llm"

    def __init__(self, script: list[Plan]):
        self.script = script
        self.calls = 0
        self.feedbacks: list[str] = []

    def generate_image(self, contents) -> bytes:
        if isinstance(contents, list):  # correction round: [image bytes, feedback]
            self.feedbacks.append(next(c for c in contents if isinstance(c, str)))
        plan = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        img = render_plan(plan, px_per_mm=0.05)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
