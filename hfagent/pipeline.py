# -*- coding: utf-8 -*-
"""Core generation pipeline: program -> generate -> parse -> verify -> fix -> wallgraph.

This is the heart of the agent: one function that takes a structured program and
a client, runs the correction loop, repairs counts deterministically, and returns
(report, plan_dict, wallgraph_dict).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from hfagent.tools.build_wallgraph import plan_to_wallgraph
from hfagent.tools.cv_parse import cv_parse
from hfagent.tools.generate_colorblock import build_convert_prompt, generate_colorblock
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
    n_violations: int
    index: int
    plan: object
    png_path: Path


def _count_mismatches(requested: dict, parsed_counts: Counter) -> list[str]:
    out = []
    for room_type, wanted in requested.items():
        got = parsed_counts.get(room_type, 0)
        if got != wanted:
            out.append(f"{room_type}: drew {got} block(s), required exactly {wanted}")
    for room_type, got in parsed_counts.items():
        if room_type not in requested:
            out.append(f"{room_type}: drew {got} block(s), but this room type was NOT requested")
    return out


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
    requested = {r["type"]: r.get("count", 1) for r in program["rooms"]}

    rounds: list[dict] = []
    best: _BestRound | None = None
    real_png: bytes | None = None   # realistic intermediate kept for correction feedback
    colorblock_png: bytes | None = None
    stall = 0
    stopped_early = False

    for round_num in range(1, max_rounds + 1):
        png_path = work_dir / f"gemini_r{round_num}.png"

        # ── generate colour-block image ──────────────────────────────────────
        if round_num == 1:
            generate_colorblock(program, client, png_path, generation_mode=generation_mode)
            real_png = png_path.with_suffix(".real.png").read_bytes()
        else:
            # send feedback against the realistic plan (where layout decisions live),
            # then re-convert to colour blocks for parsing
            violation_feedback = (
                "The architectural floor plan above violates its room program:\n- "
                + "\n- ".join(rounds[-1]["mismatches"])
                + "\nEdit the plan to fix ONLY these violations (add missing rooms, merge or "
                "remove extra ones). Keep the same drawing style, footprint and circulation."
            )
            real_png = client.generate_image([real_png, violation_feedback])
            png_path.with_suffix(".real.png").write_bytes(real_png)
            png_path.write_bytes(client.generate_image([real_png, build_convert_prompt(program)]))

        # ── parse + verify ───────────────────────────────────────────────────
        colorblock_png = png_path.read_bytes()
        plan = cv_parse(str(png_path))
        parsed_counts = Counter(r.type for r in plan.rooms)
        violations = _count_mismatches(requested, parsed_counts)
        rounds.append({"round": round_num, "parsed_rooms": dict(parsed_counts), "mismatches": violations})

        if best is None or len(violations) < best.n_violations:
            best = _BestRound(len(violations), round_num, plan, png_path)

        if not violations:
            break  # exact match — no point running more rounds

        # stop early if violations haven't strictly improved for 2 consecutive rounds
        if len(rounds) >= 2 and len(violations) >= len(rounds[-2]["mismatches"]):
            stall += 1
            if stall >= 2:
                stopped_early = True
                break
        else:
            stall = 0

    assert best is not None
    (work_dir / "parsed.json").write_text(best.plan.model_dump_json(indent=2), encoding="utf-8")

    fixed_plan, fix = fix_room_counts(best.plan, requested)
    (work_dir / "fixed.json").write_text(fixed_plan.model_dump_json(indent=2), encoding="utf-8")
    render_plan(fixed_plan, px_per_mm=1.0).save(work_dir / "recon.png")

    wallgraph = plan_to_wallgraph(fixed_plan)
    (work_dir / "wallgraph.json").write_text(wallgraph.model_dump_json(indent=2), encoding="utf-8")

    vlm_exact = not rounds[best.index - 1]["mismatches"]
    report = {
        "program": name,
        "image_model": client.image_model,
        "generation_mode": generation_mode,
        "requested_rooms": requested,
        "rounds": rounds,
        "best_round": best.index,
        "best_image": best.png_path.name,
        "parsed_rooms": rounds[best.index - 1]["parsed_rooms"],
        "room_count_exact": vlm_exact,
        "converged_in": best.index if vlm_exact else None,
        "count_fix": fix,
        "final_count_exact": vlm_exact or fix["fixed"],
        "stopped_early": stopped_early,
        "wallgraph": {
            "nodes": len(wallgraph.nodes),
            "walls": len(wallgraph.walls),
            "party_walls": len(wallgraph.adjacency()),
        },
    }
    (work_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report, fixed_plan.model_dump(), wallgraph.model_dump()
