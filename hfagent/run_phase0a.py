# -*- coding: utf-8 -*-
"""Phase 0-A real-VLM evaluation: program -> Gemini colour-block -> cv_parse -> render back.

Usage (from repo root, hfagent venv active):
    python -m hfagent.run_phase0a                       # all programs in programs.json
    python -m hfagent.run_phase0a --n 1                 # quick smoke run (first program)
    python -m hfagent.run_phase0a --only ward-wing      # named subset (comma-separated)
    python -m hfagent.run_phase0a --programs my.json    # alternative program file

Test inputs live in hfagent/programs.json — edit that file to change/extend the
programs. Room types must exist in schema/palette.py (add a colour there first
when introducing a new type); each room entry supports an optional
"approx_area_m2" size hint.

Outputs per program under hfagent/out/<run>/<program>/:
    gemini.png   raw VLM image
    parsed.json  cv_parse result (units px)
    recon.png    deterministic re-render of the parsed plan
    report.json  room counts per type vs requested + parse stats
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from hfagent.llm import GeminiClient
from hfagent.tools.cv_parse import cv_parse
from hfagent.tools.generate_colorblock import generate_colorblock
from hfagent.tools.build_wallgraph import plan_to_wallgraph
from hfagent.tools.plan_fixes import fix_room_counts
from hfagent.tools.render_plan import render_plan

DEFAULT_PROGRAMS = Path(__file__).parent / "programs.json"
DEFAULT_CONFIG = Path(__file__).parent / "config.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    pipeline = cfg.get("pipeline", "direct")
    if pipeline not in ("direct", "two-pass"):
        raise SystemExit(f"config.json: unknown pipeline '{pipeline}' (use direct | two-pass)")
    return {
        "pipeline": pipeline,
        "max_correction_rounds": int(cfg.get("max_correction_rounds", 3)),
    }


def load_programs(path: Path) -> dict:
    """Load test programs and validate room types against the shared palette,
    so a typo or a not-yet-supported room type fails loudly before any API call."""
    from hfagent.schema.palette import ROOM_RGB

    programs = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, prog in programs.items():
        unknown = sorted({r["type"] for r in prog["rooms"] if r["type"] not in ROOM_RGB})
        if unknown:
            raise SystemExit(
                f"program '{name}' uses unknown room types {unknown} — "
                "add their colours to hfagent/schema/palette.py first"
            )
    return programs


def _mismatches(requested: dict, parsed_counts: Counter) -> list[str]:
    out = []
    for t, want in requested.items():
        got = parsed_counts.get(t, 0)
        if got != want:
            out.append(f"{t}: drew {got} block(s), required exactly {want}")
    for t, got in parsed_counts.items():
        if t not in requested:
            out.append(f"{t}: drew {got} block(s), but this room type was NOT requested")
    return out


def evaluate_one(
    name: str,
    program: dict,
    client: GeminiClient,
    out_dir: Path,
    max_rounds: int = 3,
    pipeline: str = "direct",
) -> dict:
    """Generate -> parse -> verify counts; on mismatch, feed the previous image
    plus quantified violations back for an edit (LLM-Modulo correction loop:
    deterministic verifier judges, hard round cap, keep the best round)."""
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    requested = {r["type"]: r.get("count", 1) for r in program["rooms"]}

    rounds, best = [], None  # best = (n_mismatch, round_idx, plan, png_path)
    png = realistic = None
    stall = 0  # consecutive rounds without strict improvement (§5.3: don't keep burning)
    stopped_early = False
    for i in range(1, max_rounds + 1):
        png_path = d / f"gemini_r{i}.png"
        if i == 1:
            generate_colorblock(program, client, png_path, pipeline=pipeline)
            if pipeline == "two-pass":
                realistic = png_path.with_suffix(".realistic.png").read_bytes()
        elif pipeline == "two-pass":
            # fix the DESIGN, not the diagram: count errors originate in pass 1,
            # so edit the realistic plan, then re-convert it to colour blocks
            feedback = (
                "The architectural floor plan above violates its room program:\n- "
                + "\n- ".join(rounds[-1]["mismatches"])
                + "\nEdit the plan to fix ONLY these violations (add missing rooms, merge or "
                "remove extra ones). Keep the same drawing style, footprint and circulation."
            )
            realistic = client.generate_image([realistic, feedback])
            png_path.with_suffix(".realistic.png").write_bytes(realistic)
            from hfagent.tools.generate_colorblock import build_convert_prompt

            png_path.write_bytes(client.generate_image([realistic, build_convert_prompt(program)]))
        else:
            feedback = (
                "The floor plan image above violates the room-count requirements:\n- "
                + "\n- ".join(rounds[-1]["mismatches"])
                + "\nEdit the image to fix ONLY these count violations (merge or remove extra "
                "blocks, add missing ones). Keep every style rule: flat exact legend colours, "
                "black walls, white background, no text, no gradients."
            )
            png_path.write_bytes(client.generate_image([png, feedback]))
        png = png_path.read_bytes()
        plan = cv_parse(str(png_path))
        parsed_counts = Counter(r.type for r in plan.rooms)
        mism = _mismatches(requested, parsed_counts)
        rounds.append({"round": i, "parsed_rooms": dict(parsed_counts), "mismatches": mism})
        if best is None or len(mism) < best[0]:
            best = (len(mism), i, plan, png_path)
        if not mism:
            break
        # violations not strictly decreasing for 2 consecutive rounds -> stop
        # burning tokens, keep the best round (plan §5.3 discipline)
        if len(rounds) >= 2 and len(mism) >= len(rounds[-2]["mismatches"]):
            stall += 1
            if stall >= 2:
                stopped_early = True
                break
        else:
            stall = 0

    _, best_i, best_plan, best_png = best
    (d / "parsed.json").write_text(best_plan.model_dump_json(indent=2), encoding="utf-8")

    # final, deterministic count repair in Plan space (relabel / merge / split)
    fixed_plan, fix = fix_room_counts(best_plan, requested)
    (d / "fixed.json").write_text(fixed_plan.model_dump_json(indent=2), encoding="utf-8")
    render_plan(fixed_plan, px_per_mm=1.0).save(d / "recon.png")
    # authoritative structure: wall graph (walls first-class, rooms as node loops)
    g = plan_to_wallgraph(fixed_plan)
    (d / "wallgraph.json").write_text(g.model_dump_json(indent=2), encoding="utf-8")

    vlm_exact = not rounds[best_i - 1]["mismatches"]
    report = {
        "program": name,
        "image_model": client.image_model,
        "pipeline": pipeline,
        "requested_rooms": requested,
        "rounds": rounds,
        "best_round": best_i,
        "best_image": best_png.name,
        "parsed_rooms": rounds[best_i - 1]["parsed_rooms"],
        "room_count_exact": vlm_exact,
        "converged_in": best_i if vlm_exact else None,
        "count_fix": fix,
        "final_count_exact": vlm_exact or fix["fixed"],
        "stopped_early": stopped_early,
        "wallgraph": {"nodes": len(g.nodes), "walls": len(g.walls), "party_walls": len(g.adjacency())},
    }
    (d / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="run only the first N programs")
    ap.add_argument("--only", default=None, help="comma-separated program names")
    ap.add_argument("--programs", default=DEFAULT_PROGRAMS, help="path to programs JSON")
    ap.add_argument("--pipeline", default=None, help="override config.json pipeline (direct | two-pass)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    pipeline = args.pipeline or cfg["pipeline"]
    programs = load_programs(Path(args.programs))
    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        missing = [n for n in names if n not in programs]
        if missing:
            raise SystemExit(f"unknown programs {missing}; available: {list(programs)}")
        programs = {n: programs[n] for n in names}

    client = GeminiClient()
    print(f"text model:  {client.text_model}")
    print(f"image model: {client.image_model}")
    print(f"pipeline:    {pipeline} (max {cfg['max_correction_rounds']} correction rounds)")

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "out" / "phase0a"
    reports = []
    for name, program in list(programs.items())[: args.n]:
        print(f"-- {name} ...")
        try:
            r = evaluate_one(
                name, program, client, out_dir,
                max_rounds=cfg["max_correction_rounds"],
                pipeline=program.get("pipeline", pipeline),  # per-program override allowed
            )
        except Exception as e:  # one bad program must not kill the sweep
            r = {"program": name, "error": str(e)}
        print(json.dumps(r, indent=2))
        reports.append(r)

    (out_dir / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    ok = sum(1 for r in reports if r.get("room_count_exact"))
    final_ok = sum(1 for r in reports if r.get("final_count_exact"))
    rounds_used = [r.get("converged_in") for r in reports if r.get("converged_in")]
    print(f"\nVLM-exact: {ok}/{len(reports)} (rounds: {rounds_used}) | after deterministic fix: {final_ok}/{len(reports)}  -> {out_dir}")


if __name__ == "__main__":
    main()
