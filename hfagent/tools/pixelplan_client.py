# -*- coding: utf-8 -*-
"""Thin HTTP client for pixel-plan-ai: refine a traced drawing via /api/refine.

    python -m hfagent.tools.pixelplan_client DRAWING.png --program clinic-small
        [--programs hfagent/programs.json] [--url http://127.0.0.1:8765]
        [--out DIR] [--no-reference]

The ONLY coupling between hfagent and pixel-plan-ai is this HTTP call:
trace_linework + type_rooms + rasterize_plan run here, then the neutral seed
JSON (plus the original drawing as the anchoring reference image) is POSTed.
pixel-plan-ai translates the seed to code, validates it, and repairs it with
its own coding agent only if validation fails.
"""
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
import uuid
from pathlib import Path

_HFAGENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROGRAMS = _HFAGENT_DIR / "programs.json"
DEFAULT_URL = "http://127.0.0.1:8765"


def refine(
    seed: dict,
    program: dict,
    *,
    prompt: str = "",
    base_url: str = DEFAULT_URL,
    reference_png: bytes | None = None,
    timeout_seconds: float = 360.0,
) -> dict:
    """POST the seed to pixel-plan-ai /api/refine and return its result JSON."""
    payload: dict = {
        "request_id": str(uuid.uuid4()),
        "program": program,
        "seed": {key: value for key, value in seed.items() if key != "meta"},
        "prompt": prompt,
    }
    if reference_png is not None:
        payload["reference_image"] = {
            "mime": "image/png",
            "data": base64.b64encode(reference_png).decode("ascii"),
        }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/refine",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(body).get("error", body)
        except json.JSONDecodeError:
            message = body
        raise RuntimeError(f"/api/refine failed ({error.code}): {message}") from error


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", help="traced-style floor plan drawing PNG")
    ap.add_argument("--program", required=True, help="programs.json entry for typing + refine")
    ap.add_argument("--programs", default=str(DEFAULT_PROGRAMS))
    ap.add_argument("--url", default=DEFAULT_URL, help="pixel-plan-ai server base URL")
    ap.add_argument("--out", default=None, help="output dir (default out/refine/<ts>)")
    ap.add_argument("--no-reference", action="store_true",
                    help="do not attach the drawing as the repair reference image")
    args = ap.parse_args()

    from hfagent.tools.linework_tracer import trace_linework
    from hfagent.tools.plan_raster import rasterize_plan
    from hfagent.tools.room_typing import type_rooms

    programs = json.loads(Path(args.programs).read_text(encoding="utf-8"))
    if args.program not in programs:
        raise SystemExit(f"unknown --program {args.program!r}; choose from {sorted(programs)}")
    program = programs[args.program]

    png = Path(args.image).read_bytes()
    print("tracing linework...", flush=True)
    trace = trace_linework(png)
    plan = trace.plan
    print(f"  rooms={trace.diagnostics['rooms_closed']} doors={trace.diagnostics['doors_detected']}")

    print("typing rooms...", flush=True)
    typing = type_rooms(png, plan, program)
    for room in plan.rooms:
        guess = typing.get(room.id)
        if guess is not None:
            room.type, room.name = guess.type, guess.name
    meters_per_pixel = getattr(typing, "meters_per_pixel", None)
    print(f"  typed={len(typing)} meters_per_pixel={meters_per_pixel}")

    seed = rasterize_plan(
        plan,
        trace.room_graph,
        meters_per_pixel=meters_per_pixel,
        wall_width_px=trace.diagnostics.get("wall_width"),
    )
    print(f"rasterized seed: {seed['width']}x{seed['height']} cells, "
          f"{len(seed['rooms'])} rooms, {len(seed['doors'])} doors, "
          f"{seed['meters_per_cell']} m/cell (calibrated={seed['meta']['scale_calibrated']})")
    if seed["meta"]["doors_unplaced"]:
        print(f"  doors unplaced: {seed['meta']['doors_unplaced']}")

    out_dir = Path(args.out) if args.out else \
        _HFAGENT_DIR / "out" / "refine" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seed.json").write_text(json.dumps(seed, indent=1), encoding="utf-8")

    print(f"refining via {args.url}/api/refine ...", flush=True)
    result = refine(
        seed,
        program,
        base_url=args.url,
        reference_png=None if args.no_reference else png,
    )
    (out_dir / "refine_result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    attempts = result.get("iterations", [])
    print(f"refine: source={result.get('source')} accepted={result.get('accepted')} "
          f"score={result.get('validation', {}).get('score')} attempts={len(attempts)}")
    for iteration in attempts:
        print(f"  attempt {iteration['attempt']:>2} {iteration['phase']:<7} "
              f"{iteration['status']:<9} score={iteration['score']}")
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
