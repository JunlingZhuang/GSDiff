# -*- coding: utf-8 -*-
"""CLI harness for evaluating the generation pipeline against test programs.

Usage (from repo root, hfagent venv active):
    python -m hfagent.run_phase0a                       # all programs in programs.json
    python -m hfagent.run_phase0a --n 1                 # quick smoke run (first program)
    python -m hfagent.run_phase0a --only ward-wing      # named subset (comma-separated)
    python -m hfagent.run_phase0a --programs my.json    # alternative program file

Test inputs live in hfagent/programs.json. Room types must exist in
schema/palette.py; each room entry supports an optional "approx_area_m2" hint.

Outputs per program under hfagent/out/<run>/<program>/:
    gemini_r*.png   VLM images per correction round
    parsed.json     cv_parse result (units px)
    fixed.json      after deterministic count repair
    recon.png       re-render of fixed plan
    wallgraph.json  wall-graph structure
    report.json     counts, rounds, fix ops
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hfagent.llm import GeminiClient
from hfagent.pipeline import generate_plan, load_config

DEFAULT_PROGRAMS = Path(__file__).parent / "programs.json"


def load_programs(path: Path) -> dict:
    """Load and validate test programs against the shared palette."""
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
            report, _, _ = generate_plan(
                program, client, out_dir, name=name,
                max_rounds=cfg["max_correction_rounds"],
                pipeline=program.get("pipeline", pipeline),
            )
        except Exception as e:
            report = {"program": name, "error": str(e)}
        print(json.dumps(report, indent=2))
        reports.append(report)

    (out_dir / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    ok = sum(1 for r in reports if r.get("room_count_exact"))
    final_ok = sum(1 for r in reports if r.get("final_count_exact"))
    rounds_used = [r.get("converged_in") for r in reports if r.get("converged_in")]
    print(
        f"\nVLM-exact: {ok}/{len(reports)} (rounds: {rounds_used}) | "
        f"after deterministic fix: {final_ok}/{len(reports)}  -> {out_dir}"
    )


if __name__ == "__main__":
    main()
