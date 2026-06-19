# -*- coding: utf-8 -*-
"""Core generation pipeline: program -> generate -> parse -> verify -> fix -> wallgraph.

This is the heart of the agent: one function that takes a structured program and
a client, runs the VLM correction loop, repairs counts deterministically, and returns
(report, plan_dict, wallgraph_dict).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from hfagent.tools.wallgraph import plan_to_wallgraph
from hfagent.tools.image_parser import cv_parse
from hfagent.tools.floor_plan_generator import FloorPlanGenerator
from hfagent.tools.plan_fixes import fix_room_counts
from hfagent.tools.render_plan import render_plan

DEFAULT_CONFIG = Path(__file__).parent / "config.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    mode = cfg.get("generation_mode", "real2color")
    if mode not in ("real2color",):
        raise SystemExit(f"config.json: unknown generation_mode '{mode}' (use real2color)")
    return {
        "generation_mode": mode,
        "max_correction_rounds": int(cfg.get("max_correction_rounds", 3)),
    }


@dataclass
class _BestRound:
    violation_count: int
    round_num: int
    plan: object
    png_path: Path


def _count_violations(required_rooms: dict, actual_counts: Counter) -> list[str]:
    """Compare parsed room counts against the program; return human-readable mismatches."""
    violations = []
    for room_type, required in required_rooms.items():
        actual = actual_counts.get(room_type, 0)
        if actual != required:
            violations.append(f"{room_type}: drew {actual} block(s), required exactly {required}")
    for room_type, actual in actual_counts.items():
        if room_type not in required_rooms:
            violations.append(f"{room_type}: drew {actual} block(s), but this room type was NOT requested")
    return violations


def generate_plan(
    program: dict,
    client,
    out_dir: Path,
    name: str = "plan",
    max_rounds: int = 3,
    generation_mode: str = "real2color",
) -> tuple[dict, dict, dict]:
    """Generate, parse, and repair a floor plan from a structured program.

    Runs a VLM correction loop (up to max_rounds), then deterministically repairs
    any remaining count violations via relabel / merge / split.

    Returns (report, plan_dict, wallgraph_dict).
    Writes intermediate files to out_dir/name/.
    """
    work_dir = out_dir / name
    work_dir.mkdir(parents=True, exist_ok=True)
    required_rooms = {r["type"]: r.get("count", 1) for r in program["rooms"]}

    generator = FloorPlanGenerator(program, client)
    rounds: list[dict] = []
    best_round: _BestRound | None = None
    real_png: bytes | None = None

    for round_num in range(1, max_rounds + 1):
        png_path = work_dir / f"gemini_r{round_num}.png"

        # ── generate colour-block image ──────────────────────────────────────
        if round_num == 1:
            generator.run(png_path)
            real_png = png_path.with_suffix(".real.png").read_bytes()
        else:
            # apply violation feedback to the realistic plan, then re-convert to colour blocks
            violation_feedback = (
                "The architectural floor plan above violates its room program:\n- "
                + "\n- ".join(rounds[-1]["violations"])
                + "\nEdit the plan to fix ONLY these violations (add missing rooms, merge or "
                "remove extra ones). Keep the same drawing style, footprint and circulation."
            )
            real_png = client.generate_image([real_png, violation_feedback])
            png_path.with_suffix(".real.png").write_bytes(real_png)
            png_path.write_bytes(generator.to_colorblock(real_png))

        # ── parse + verify ───────────────────────────────────────────────────
        parsed_plan = cv_parse(str(png_path))
        actual_counts = Counter(r.type for r in parsed_plan.rooms)
        violations = _count_violations(required_rooms, actual_counts)
        rounds.append({"round": round_num, "actual_rooms": dict(actual_counts), "violations": violations})

        if best_round is None or len(violations) < best_round.violation_count:
            best_round = _BestRound(len(violations), round_num, parsed_plan, png_path)

        if not violations:
            break  # all room counts match — skip remaining rounds

    assert best_round is not None
    (work_dir / "parsed.json").write_text(best_round.plan.model_dump_json(indent=2), encoding="utf-8")

    fixed_plan, count_fix = fix_room_counts(best_round.plan, required_rooms)
    (work_dir / "fixed.json").write_text(fixed_plan.model_dump_json(indent=2), encoding="utf-8")
    render_plan(fixed_plan, px_per_mm=1.0).save(work_dir / "recon.png")

    wallgraph = plan_to_wallgraph(fixed_plan)
    (work_dir / "wallgraph.json").write_text(wallgraph.model_dump_json(indent=2), encoding="utf-8")

    vlm_converged = not rounds[best_round.round_num - 1]["violations"]
    report = {
        "program": name,
        "image_model": client.image_model,
        "generation_mode": generation_mode,
        "required_rooms": required_rooms,
        "rounds": rounds,
        "best_round": best_round.round_num,
        "best_image": best_round.png_path.name,
        "actual_rooms": rounds[best_round.round_num - 1]["actual_rooms"],
        "room_count_exact": vlm_converged,
        "converged_in": best_round.round_num if vlm_converged else None,
        "count_fix": count_fix,
        "final_count_exact": vlm_converged or count_fix["fixed"],
        "wallgraph": {
            "nodes": len(wallgraph.nodes),
            "walls": len(wallgraph.walls),
            "party_walls": len(wallgraph.adjacency()),
        },
    }
    (work_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report, fixed_plan.model_dump(), wallgraph.model_dump()
