# -*- coding: utf-8 -*-
"""generate_colorblock contract: prompt construction (pure) + file output (mock client)."""
from pathlib import Path

from hfagent.schema.palette import ROOM_RGB, rgb_hex
from hfagent.tools.floor_plan_generator import (
    FloorPlanGenerator,
    build_convert_prompt,
    build_linework_prompt,
    build_real_prompt,
)

PROGRAM = {
    "building_type": "community clinic",
    "rooms": [{"type": "exam_room", "count": 3}, {"type": "waiting", "count": 1}, {"type": "corridor", "count": 1}],
    "adjacency": [["exam_room", "corridor"], ["waiting", "corridor"]],
}


def test_real_prompt_contains_room_types():
    p = build_real_prompt(PROGRAM)
    assert "exam room" in p
    assert "waiting" in p
    assert "community clinic" in p


def test_convert_prompt_contains_hex_legend():
    p = build_convert_prompt(PROGRAM)
    assert rgb_hex(ROOM_RGB["exam_room"]) in p
    assert rgb_hex(ROOM_RGB["waiting"]) in p
    # plain-English repaint instruction (no jargon like "segmentation mask")
    assert "flat colour diagram" in p
    assert "one solid colour from the legend" in p
    # door swing arc filled with the room's colour so blocks stay continuous
    assert "door's swing arc" in p
    assert "no notch or gap where a door was" in p
    # walls stay black, outside stays white, nothing else survives
    assert "solid black" in p
    assert "no text" in p


def test_linework_prompt_is_ocr_and_cv_friendly():
    prompt = build_linework_prompt(PROGRAM)
    assert "CV LINEWORK PROFILE" in prompt
    assert "EXACT underscore-and-number spelling" in prompt
    assert "Do NOT draw windows" in prompt
    assert "A wall gap is allowed only for a door" in prompt
    assert "single-leaf hinged doors only" in prompt


class FakeClient:
    def __init__(self):
        self.calls = []

    def generate_image(self, contents) -> bytes:
        self.calls.append(contents)
        return b"\x89PNG fake " + str(len(self.calls)).encode()

    def generate_json(self, contents, schema=None) -> str:
        return "[]"  # no doors — room adjacency extraction is exercised in test_orchestration


def test_real2color_makes_two_calls_and_writes_outputs(tmp_path: Path):
    client = FakeClient()
    generator = FloorPlanGenerator(PROGRAM, client)
    out = generator.run(tmp_path / "plan.png")
    # 2 image calls — call 1: realistic-plan prompt (string); call 2: [realistic bytes, convert prompt]
    assert len(client.calls) == 2
    assert isinstance(client.calls[0], str) and "architectural floor plan" in client.calls[0].lower()
    assert isinstance(client.calls[1], list)
    assert client.calls[1][0] == b"\x89PNG fake 1"   # realistic bytes forwarded
    assert "flat colour diagram" in client.calls[1][1]
    # colour-block output is pass-2; realistic intermediate + room graph saved alongside
    assert out.read_bytes() == b"\x89PNG fake 2"
    assert (tmp_path / "plan.real.png").read_bytes() == b"\x89PNG fake 1"
    assert (tmp_path / "plan.graph.json").exists()
