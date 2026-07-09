# -*- coding: utf-8 -*-
"""CLI harness for evaluating the generation pipeline against test programs.

Usage (from repo root, hfagent venv active):
    python -m hfagent.evaluate                       # all programs in programs.json
    python -m hfagent.evaluate --n 1                 # quick smoke run (first program)
    python -m hfagent.evaluate --only ward-wing      # named subset (comma-separated)
    python -m hfagent.evaluate --programs my.json    # alternative program file
    python -m hfagent.evaluate --repeat 4            # 4 independent runs per program

Test inputs live in hfagent/programs.json. Room types must exist in
schema/palette.py; each room entry supports an optional "approx_area_ft2" hint.

Outputs per run under hfagent/out/<run>/<program>/ (with --repeat N the work dirs
are <program>-r1 .. <program>-rN, each an independent full pipeline run):
    gemini_r*.png         colour-block VLM image per correction round
    gemini_r*.real.png    realistic plan per round (doors visible here)
    parsed.json           cv_parse result (units px)
    fixed.json            after deterministic count repair
    recon.png             re-render of fixed plan
    graph.json            room adjacency graph (rooms + doors)
    recon_with_door.png   recon.png with door markers overlaid
    report.json           counts, rounds, fix ops

structure_mode=linework replaces the colour-block artifacts with the trace set:
walls_overlay.png, doors_overlay.png, recon.png (bridged walls + door arcs),
recon_post.png, rooms_colorful.png.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hfagent.llm import GeminiClient
from hfagent.floor_plan_generate import generate_plan, load_config

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
        if prog.get("boundary") and not (Path(__file__).parent / prog["boundary"]).exists():
            raise SystemExit(f"program '{name}': boundary image not found: {prog['boundary']}")
    return programs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="run only the first N programs")
    ap.add_argument("--only", "--program", "-p", dest="only", default=None,
                    help="run only these program(s) by name (comma-separated)")
    ap.add_argument("--programs", default=DEFAULT_PROGRAMS, help="path to programs JSON")
    ap.add_argument("--structure-mode", default=None, dest="structure_mode",
                    help="override config.json structure_mode (colorblock | json | linework | direct_colorblock)")
    ap.add_argument("--boundary", default=None,
                    help="building-outline image to fit (overrides per-program / per-mode boundary)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each program N times as independent full pipeline runs; "
                         "work dirs become <program>-r1 .. <program>-rN (default 1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    structure_mode = args.structure_mode or cfg["structure_mode"]
    if structure_mode not in cfg["modes"]:
        raise SystemExit(f"unknown structure_mode '{structure_mode}'; available: {sorted(cfg['modes'])}")
    # the chosen mode (config default or --structure-mode) supplies its own models + boundary
    mode_cfg = cfg["modes"][structure_mode]
    mode_boundary = mode_cfg.get("boundary", "")
    programs = load_programs(Path(args.programs))

    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        missing = [n for n in names if n not in programs]
        if missing:
            raise SystemExit(f"unknown programs {missing}; available: {list(programs)}")
        programs = {n: programs[n] for n in names}

    client = GeminiClient(
        text_model=(mode_cfg.get("text_model") or None),
        image_model=(mode_cfg.get("image_model") or None),
        image_size=cfg["image_size"],
        image_aspect=cfg["image_aspect"],
    )
    print(f"text model:       {client.text_model}")
    print(f"image model:      {client.image_model}")
    print(f"image size:       {client.image_size} @ {client.image_aspect}")
    print(f"structure mode:   {structure_mode} (max {cfg['max_correction_rounds']} correction rounds)")
    print(f"mode boundary:    {mode_boundary or '(none)'}")

    # timestamped run dir so repeated evaluations never clobber each other
    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "out" / "eval" / time.strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)  # so summary.json survives a failing program
    def resolve(p: str) -> Path:
        cand = Path(p)
        return cand if cand.exists() else Path(__file__).parent / p

    reports = []
    for name, program in list(programs.items())[: args.n]:
        # second entry, priority: --boundary flag > per-program field > active mode's config path
        boundary = None
        bpath = args.boundary or program.get("boundary") or mode_boundary
        if bpath:
            boundary = resolve(bpath).read_bytes()
        for run in range(1, args.repeat + 1):
            run_name = name if args.repeat == 1 else f"{name}-r{run}"
            print(f"-- {run_name} ...{' [boundary]' if boundary else ''}")
            try:
                report, _, _ = generate_plan(
                    program, client, out_dir, name=run_name,
                    max_rounds=cfg["max_correction_rounds"],
                    boundary=boundary,
                    structure_mode=structure_mode,
                )
            except Exception as e:
                report = {"program": run_name, "error": str(e)}
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
