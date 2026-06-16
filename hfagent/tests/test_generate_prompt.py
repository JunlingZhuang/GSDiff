# -*- coding: utf-8 -*-
"""generate_colorblock contract: prompt construction (pure) + file output (mock client)."""
from pathlib import Path

from hfagent.schema.palette import ROOM_RGB, rgb_hex
from hfagent.tools.generate_colorblock import build_prompt, generate_colorblock

PROGRAM = {
    "building_type": "community clinic",
    "rooms": [{"type": "exam_room", "count": 3}, {"type": "waiting", "count": 1}, {"type": "corridor", "count": 1}],
    "adjacency": [["exam_room", "corridor"], ["waiting", "corridor"]],
}


def test_prompt_contains_exact_hex_legend():
    p = build_prompt(PROGRAM)
    assert rgb_hex(ROOM_RGB["exam_room"]) in p
    assert rgb_hex(ROOM_RGB["waiting"]) in p
    assert "3 room(s)" in p
    assert "exam_room must share a wall with corridor" in p
    assert "no text" in p.lower()


class FakeClient:
    def __init__(self):
        self.calls = []

    def generate_image(self, contents) -> bytes:
        self.calls.append(contents)
        return b"\x89PNG fake " + str(len(self.calls)).encode()


def test_generate_writes_file(tmp_path: Path):
    client = FakeClient()
    out = generate_colorblock(PROGRAM, client, tmp_path / "x" / "plan.png")
    assert out.read_bytes() == b"\x89PNG fake 1"
    assert "community clinic" in client.calls[0]


def test_two_pass_pipeline(tmp_path: Path):
    client = FakeClient()
    out = generate_colorblock(PROGRAM, client, tmp_path / "plan.png", pipeline="two-pass")
    # call 1: realistic prompt (text); call 2: [realistic image bytes, convert prompt]
    assert len(client.calls) == 2
    assert "realistic" in client.calls[0]
    assert isinstance(client.calls[1], list)
    assert client.calls[1][0] == b"\x89PNG fake 1"
    assert "Preserve every room" in client.calls[1][1]
    # final colour-block is pass-2 output; realistic intermediate saved alongside
    assert out.read_bytes() == b"\x89PNG fake 2"
    assert (tmp_path / "plan.realistic.png").read_bytes() == b"\x89PNG fake 1"
