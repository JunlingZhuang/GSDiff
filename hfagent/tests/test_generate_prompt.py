# -*- coding: utf-8 -*-
"""generate_colorblock contract: prompt construction (pure) + file output (mock client)."""
from pathlib import Path

from hfagent.schema.palette import ROOM_RGB, rgb_hex
from hfagent.tools.generator import build_real_prompt, build_convert_prompt, generate_colorblock

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
    assert "Preserve every room" in p


class FakeClient:
    def __init__(self):
        self.calls = []

    def generate_image(self, contents) -> bytes:
        self.calls.append(contents)
        return b"\x89PNG fake " + str(len(self.calls)).encode()


def test_real2color_makes_two_calls_and_writes_both_files(tmp_path: Path):
    client = FakeClient()
    out = generate_colorblock(PROGRAM, client, tmp_path / "plan.png", generation_mode="real2color")
    # call 1: realistic prompt (string); call 2: [realistic image bytes, convert prompt]
    assert len(client.calls) == 2
    assert isinstance(client.calls[0], str) and "realistic" in client.calls[0].lower()
    assert isinstance(client.calls[1], list)
    assert client.calls[1][0] == b"\x89PNG fake 1"   # realistic bytes forwarded
    assert "Preserve every room" in client.calls[1][1]
    # colour-block output is pass-2; realistic intermediate saved alongside
    assert out.read_bytes() == b"\x89PNG fake 2"
    assert (tmp_path / "plan.real.png").read_bytes() == b"\x89PNG fake 1"
