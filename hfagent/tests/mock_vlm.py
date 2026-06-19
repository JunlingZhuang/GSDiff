# -*- coding: utf-8 -*-
"""Scripted mock VLM for orchestration tests.

Each round of the real2color pipeline makes exactly 2 generate_image calls:
  call 1 — realistic plan prompt
  call 2 — convert to colour-block

MockVLM returns the nth script plan for both calls in round n, advancing to
the next plan every 2 calls. This isolates orchestration logic from real VLM noise.
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
        if isinstance(contents, list):
            feedback_str = next((c for c in contents if isinstance(c, str)), None)
            if feedback_str:
                self.feedbacks.append(feedback_str)

        round_idx = self.calls // 2  # 2 calls per round in real2color mode
        plan = self.script[min(round_idx, len(self.script) - 1)]
        self.calls += 1

        buf = io.BytesIO()
        render_plan(plan, px_per_mm=0.05).save(buf, "PNG")
        return buf.getvalue()
