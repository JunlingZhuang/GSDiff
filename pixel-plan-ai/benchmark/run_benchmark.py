#!/usr/bin/env python
"""A/B benchmark for the in-attempt self-check tool loop (docs/claude-code-lessons.md #8).

Runs a set of programs through service.generate_plan in-process (no HTTP server) for one
or both arms of GEMINI_ATTEMPT_TOOLS, records per-run outcome/cost/wall metrics, prints a
compact table and writes the raw rows to JSON. Gated adoption: keep the loop only if the
"on" arm beats "off" on acceptance / attempts / cost across the fixed set.

Run from the pixel-plan-ai directory, for example:

    python benchmark/run_benchmark.py \
        --programs clinic-small clinic-mixed outpatient-dept \
        --arms off on \
        --out benchmark/results-<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
SAMPLES_PATH = ROOT / "data" / "samples.json"


def load_dotenv() -> None:
    """Same shape as backend/server.py: read .env and os.environ.setdefault every key."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def set_arm(arm: str) -> None:
    """Flip GEMINI_ATTEMPT_TOOLS for the requested arm."""
    if arm == "on":
        os.environ["GEMINI_ATTEMPT_TOOLS"] = "1"
    else:
        os.environ.pop("GEMINI_ATTEMPT_TOOLS", None)


def run_one(generate_plan: Any, program: dict[str, Any], arm: str) -> dict[str, Any]:
    set_arm(arm)
    started = time.perf_counter()
    result = generate_plan({
        "program": program,
        "options": {"width": 96, "height": 64},
        "mode": "auto",
    })
    wall_seconds = round(time.perf_counter() - started, 2)
    return {
        "accepted": bool(result.get("accepted")),
        "score": result.get("validation", {}).get("score"),
        "attempts": len(result.get("iterations", [])),
        "stop_reason": result.get("stop_reason"),
        "usage_total": result.get("usage_total"),
        "wall_seconds": wall_seconds,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    header = f"{'program':<18} {'arm':<4} {'ok':<4} {'score':<6} {'att':<4} {'stop':<20} {'tokens':<9} {'wall_s':<7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if row.get("error"):
            print(f"{row['program']:<18} {row['arm']:<4} ERROR: {row['error'][:80]}")
            continue
        tokens = ""
        if isinstance(row.get("usage_total"), dict):
            tokens = str(row["usage_total"].get("total_tokens", ""))
        print(
            f"{row['program']:<18} {row['arm']:<4} "
            f"{('yes' if row['accepted'] else 'no'):<4} {str(row['score']):<6} "
            f"{str(row['attempts']):<4} {str(row['stop_reason']):<20} {tokens:<9} {str(row['wall_seconds']):<7}"
        )


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--programs",
        nargs="+",
        default=["clinic-small", "clinic-mixed", "outpatient-dept"],
        help="samples.json keys to benchmark",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["off", "on"],
        default=["off", "on"],
        help="GEMINI_ATTEMPT_TOOLS arms to sweep",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "benchmark" / f"results-{timestamp}.json"),
        help="output JSON path",
    )
    args = parser.parse_args()

    load_dotenv()
    # Events must go nowhere for an in-process sweep: never touch a live progress file.
    os.environ.pop("PIXEL_PLAN_PROGRESS_FILE", None)
    sys.path.append(str(BACKEND))
    from service import generate_plan  # imported after sys.path + .env are ready

    samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for program_key in args.programs:
        for arm in args.arms:
            row: dict[str, Any] = {"program": program_key, "arm": arm}
            program = samples.get(program_key)
            if program is None:
                row["error"] = f"unknown program key '{program_key}'"
                rows.append(row)
                print_table([row])
                continue
            try:
                row.update(run_one(generate_plan, program, arm))
            except Exception as error:  # one failure must not kill the sweep
                row["error"] = str(error)
            rows.append(row)
            print_table([row])

    print("\n=== summary ===")
    print_table(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"generated_at": timestamp, "arms": args.arms, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
