# -*- coding: utf-8 -*-
"""Sequential pipeline: understand -> generate_plan.

Thin wrapper used by the API and tests. Keeps the two-step flow explicit
without LangGraph overhead (which adds no value for two sequential nodes
with no branching — LangGraph conditional edges are planned for Phase 4).
"""
from __future__ import annotations

from pathlib import Path

from hfagent.floor_plan_generate import generate_plan, load_config
from hfagent.nodes.understand import understand


def run_pipeline(text: str, client, out_dir: str | Path) -> dict:
    """understand -> generate_plan, returns the full result dict."""
    out_dir = Path(out_dir)
    cfg = load_config()
    program = understand(text, client)
    boundary = None
    if cfg["boundary"]:
        bpath = Path(cfg["boundary"])
        if not bpath.exists():
            bpath = Path(__file__).parent / cfg["boundary"]
        boundary = bpath.read_bytes() if bpath.exists() else None
    report, plan, room_graph = generate_plan(
        program, client, out_dir,
        max_rounds=cfg["max_correction_rounds"],
        boundary=boundary,
        structure_mode=cfg["structure_mode"],
    )
    return {"program": program, "report": report, "plan": plan, "room_graph": room_graph}
