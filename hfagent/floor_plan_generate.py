# -*- coding: utf-8 -*-
"""Core generation pipeline: program -> generate -> parse -> verify -> fix -> room graph.

This is the heart of the agent: one function that takes a structured program and
a client, runs the VLM correction loop, repairs counts deterministically, reads the
room adjacency (doors) from the best realistic plan, and returns
(report, plan_dict, room_graph_dict).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from hfagent.tools.image_parser import cv_parse
from hfagent.tools.floor_plan_generator import FloorPlanGenerator
from hfagent.tools.door_placer import place_doors
from hfagent.tools.plan_fixes import fix_room_counts
from hfagent.tools.render_plan import render_plan
from hfagent.schema.plan import AdjacencyGraph, AdjEdge

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
    real_png: bytes


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
    boundary: bytes | None = None,
) -> tuple[dict, dict, dict]:
    """Generate, parse, and repair a floor plan from a structured program.

    Runs a VLM correction loop (up to max_rounds), deterministically repairs any
    remaining count violations, then reads the room adjacency (doors) from the best
    realistic plan image.

    `boundary` (PNG bytes) is the optional second input entry: when given, the
    realflow plan's outer walls follow that footprint. Everything downstream of the
    realflow plan is identical with or without it.

    Returns (report, plan_dict, room_graph_dict).
    Writes intermediate files to out_dir/name/.
    """
    work_dir = out_dir / name
    work_dir.mkdir(parents=True, exist_ok=True)
    required_rooms = {r["type"]: r.get("count", 1) for r in program["rooms"]}

    generator = FloorPlanGenerator(program, client, boundary=boundary)
    rounds: list[dict] = []
    best_round: _BestRound | None = None
    real_png: bytes | None = None

    for round_num in range(1, max_rounds + 1):
        png_path = work_dir / f"gemini_r{round_num}.png"

        # ── real2color: realistic plan -> colour-block ───────────────────────
        if round_num == 1:
            real_png = generator.generate_real_plan()
        else:
            # apply violation feedback to the realistic plan, then re-convert
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
            best_round = _BestRound(len(violations), round_num, parsed_plan, png_path, real_png)

        if not violations:
            break  # all room counts match — skip remaining rounds

    assert best_round is not None
    (work_dir / "parsed.json").write_text(best_round.plan.model_dump_json(indent=2), encoding="utf-8")

    fixed_plan, count_fix = fix_room_counts(best_round.plan, required_rooms)
    (work_dir / "fixed.json").write_text(fixed_plan.model_dump_json(indent=2), encoding="utf-8")
    render_plan(fixed_plan, px_per_mm=1.0).save(work_dir / "recon.png")

    # ── room adjacency (doors) read from the best realistic plan ─────────────
    # LLM returns connectivity only; door_placer hangs each door on the real wall the
    # two connected rooms physically share, and reports which rooms each door links.
    room_graph = generator.extract_room_adjacency(best_round.real_png)
    (work_dir / "graph.json").write_text(room_graph.model_dump_json(indent=2), encoding="utf-8")

    # assemble the authoritative plan (docs/agent/02-data-model.md §3.2):
    # rooms + complete walls + doors-on-walls + adjacency_graph (edges link via door id)
    plan, door_pairs = place_doors(fixed_plan, room_graph)
    plan.building_type = program.get("building_type", plan.building_type)
    plan.adjacency_graph = AdjacencyGraph(
        nodes=[r.id for r in plan.rooms],
        edges=[AdjEdge(from_=a, to=b, type="door", via=did) for did, a, b in door_pairs],
    )
    (work_dir / "plan.json").write_text(plan.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    render_plan(plan, px_per_mm=1.0).save(work_dir / "recon_with_door.png")

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
        "room_graph": {
            "rooms": len(room_graph.rooms),
            "doors": len(room_graph.doors),       # logical edges from the LLM
            "placed_doors": len(plan.doors),       # doors hung on real walls
            "walls": len(plan.walls),              # complete geometry-layer wall list
        },
    }
    (work_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report, plan.model_dump(by_alias=True), room_graph.model_dump()
