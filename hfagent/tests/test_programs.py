# -*- coding: utf-8 -*-
"""Pipeline smoke tests driven by programs.json.

Each program gets its own MockVLM (a perfect synthetic plan), runs the full
generate_plan pipeline in one round, and writes all artifacts to
hfagent/out/tests/test_programs/<prog_name>/ for visual inspection.

These tests validate that:
  - every room type in programs.json is handled without errors
  - a perfect MockVLM response produces room_count_exact == True
  - recon.png and recon_with_door.png are written
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hfagent.floor_plan_generate import generate_plan
from hfagent.tests.mock_vlm import MockVLM
from hfagent.tests.synth import make_plan_for_program

_PROGRAMS_PATH = Path(__file__).resolve().parent.parent / "programs.json"
PROGRAMS: dict = json.loads(_PROGRAMS_PATH.read_text(encoding="utf-8"))
_OUT_ROOT = Path(__file__).resolve().parent.parent / "out" / "tests" / "test_programs"


@pytest.fixture
def prog_out_dir(request):
    """Output dir keyed by program name, not test node name (avoids bracket issues)."""
    prog_name: str = request.param
    path = _OUT_ROOT / prog_name
    path.mkdir(parents=True, exist_ok=True)
    return path, prog_name


@pytest.mark.parametrize("prog_out_dir", list(PROGRAMS.keys()), indirect=True)
def test_program_pipeline(prog_out_dir):
    out_dir, prog_name = prog_out_dir
    program = PROGRAMS[prog_name]

    plan = make_plan_for_program(program)
    client = MockVLM([plan])

    r, _, _ = generate_plan(program, client, out_dir, name=prog_name, max_rounds=1)

    assert r["room_count_exact"], (
        f"{prog_name}: expected exact room counts from perfect mock, "
        f"got violations: {r['rounds'][0]['violations']}"
    )
    assert (out_dir / prog_name / "recon.png").exists()
    assert (out_dir / prog_name / "recon_with_door.png").exists()
