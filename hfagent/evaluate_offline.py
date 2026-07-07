# -*- coding: utf-8 -*-
"""Offline linework eval: trace + room typing over EXISTING drawings, no Gemini.

    python -m hfagent.evaluate_offline PATH [PATH ...] [--program NAME] [--out DIR]

Each PATH is a plan PNG or a directory searched recursively for ``*.png``.
The program vocabulary for room typing resolves per image: ``--program`` wins,
otherwise the nearest ancestor directory named after a ``programs.json`` entry
(both the curated dataset layout ``data/real-floorplan-dataset/<program>/x.png``
and the eval layout ``out/eval/<ts>/<program>/gemini_r1.real.png`` match).

Per image it writes the tracer's fixed artifact set plus ``typing_overlay.png``
and ``typing.json`` under ``out/eval/<timestamp>-offline/<relative-name>/`` and
prints one summary row: rooms, doors, typed coverage, per-type counts.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from hfagent.tools.linework_tracer import trace_linework
from hfagent.tools.room_typing import type_rooms, typing_overlay

_HFAGENT_DIR = Path(__file__).resolve().parent
DEFAULT_PROGRAMS = _HFAGENT_DIR / "programs.json"


def _collect_images(paths: list[str]) -> list[Path]:
    images: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            images.extend(sorted(p for p in path.rglob("*.png")
                                 if "_review" not in p.parts))
        elif path.suffix.lower() == ".png":
            images.append(path)
    return images


def _program_for(image: Path, programs: dict, forced: str | None) -> tuple[str, dict] | None:
    if forced is not None:
        return forced, programs[forced]
    for part in reversed(image.parent.parts):
        if part in programs:
            return part, programs[part]
    for name in programs:                      # dataset files carry __<program>__ in the name
        if f"__{name}__" in image.stem or image.stem.endswith(name):
            return name, programs[name]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="plan PNGs or directories of them")
    ap.add_argument("--program", default=None,
                    help="force this programs.json entry for every image")
    ap.add_argument("--programs", default=str(DEFAULT_PROGRAMS))
    ap.add_argument("--out", default=None, help="output dir (default out/eval/<ts>-offline)")
    args = ap.parse_args()

    programs = json.loads(Path(args.programs).read_text(encoding="utf-8"))
    if args.program is not None and args.program not in programs:
        raise SystemExit(f"unknown --program {args.program!r}; choose from {sorted(programs)}")
    images = _collect_images(args.paths)
    if not images:
        raise SystemExit("no PNG images found under the given paths")

    out_root = Path(args.out) if args.out else \
        _HFAGENT_DIR / "out" / "eval" / f"{time.strftime('%Y%m%d-%H%M%S')}-offline"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for image in images:
        resolved = _program_for(image, programs, args.program)
        png = image.read_bytes()
        trace = trace_linework(png)
        plan = trace.plan
        guesses: dict = {}
        program_name = None
        if resolved is not None:
            program_name, program = resolved
            guesses = type_rooms(png, plan, program)
            for room in plan.rooms:
                guess = guesses.get(room.id)
                if guess is not None:
                    room.type, room.name = guess.type, guess.name

        work_dir = out_root / (image.parent.name + "__" + image.stem
                               if image.parent.name else image.stem)
        work_dir.mkdir(parents=True, exist_ok=True)
        for filename, payload in trace.artifacts.items():
            (work_dir / filename).write_bytes(payload)
        (work_dir / "typing_overlay.png").write_bytes(typing_overlay(png, plan, guesses))
        (work_dir / "typing.json").write_text(json.dumps(
            {rid: dict(type=g.type, instance=g.instance,
                       score=round(g.score, 3), text=g.text)
             for rid, g in sorted(guesses.items())}, indent=1), encoding="utf-8")

        diag = trace.diagnostics
        row = dict(
            image=str(image),
            program=program_name,
            rooms=diag["rooms_closed"],
            doors=diag["doors_detected"],
            typed=len(guesses),
            type_counts=dict(Counter(g.type for g in guesses.values())),
        )
        rows.append(row)
        coverage = f"{100 * row['typed'] / max(1, row['rooms']):3.0f}%"
        print(f"{image.parent.name + '/' + image.name:60s} rooms={row['rooms']:<4d} "
              f"doors={row['doors']:<4d} typed={row['typed']:<4d} ({coverage}) "
              f"program={program_name or '-'}", flush=True)

    (out_root / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    total_rooms = sum(r["rooms"] for r in rows)
    total_typed = sum(r["typed"] for r in rows)
    print(f"\n{len(rows)} image(s): rooms={total_rooms} doors={sum(r['doors'] for r in rows)} "
          f"typed={total_typed} ({100 * total_typed / max(1, total_rooms):.1f}%)")
    print(f"results -> {out_root}")


if __name__ == "__main__":
    main()
