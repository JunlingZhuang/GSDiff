# -*- coding: utf-8 -*-
"""THE Phase 0-A gate: synthetic Plan -> render_plan -> cv_parse -> compare.

Passing this proves the parser is trustworthy in isolation (no VLM noise), so
any quality gap measured later on real Gemini images is attributable to the
VLM image alone.
"""
import pytest
from PIL import Image, ImageDraw

from hfagent.schema.palette import ROOM_RGB, WALL_RGB
from hfagent.metrics import compare_plans
from hfagent.tests.synth import ALL
from hfagent.tools.image_parser import cv_parse
from hfagent.tools.render_plan import render_plan

PX_PER_MM = 0.05


@pytest.mark.parametrize("name", list(ALL.keys()))
def test_roundtrip(name):
    truth = ALL[name]()
    img = render_plan(truth, px_per_mm=PX_PER_MM)
    parsed = cv_parse(img, px_per_mm=PX_PER_MM)
    # parser works in image px space; truth is in plan mm space — re-anchor by
    # translating parsed rooms so both bounding boxes start at the same origin
    tb = truth.bounds()
    pb = parsed.bounds()
    dx, dy = tb[0] - pb[0], tb[1] - pb[1]
    for r in parsed.rooms:
        r.polygon = [(x + dx, y + dy) for x, y in r.polygon]

    m = compare_plans(truth, parsed)
    assert m["room_count_match"], m
    assert m["type_accuracy"] == 1.0, m
    assert m["mean_iou"] >= 0.85, m
    assert m["min_iou"] >= 0.75, m


def test_parser_tolerates_internal_black_holes():
    img = Image.new("RGB", (220, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((10, 10, 95, 110), fill=ROOM_RGB["exam_room"])
    draw.rectangle((105, 10, 210, 110), fill=ROOM_RGB["corridor"])
    draw.rectangle((96, 10, 104, 110), fill=WALL_RGB)
    draw.rectangle((35, 42, 68, 58), fill=WALL_RGB)
    draw.rectangle((140, 45, 180, 62), fill=WALL_RGB)

    parsed = cv_parse(img, min_room_px=500, grid_px=4, gap_fill_px=8)
    counts = {t: sum(1 for r in parsed.rooms if r.type == t) for t in ("exam_room", "corridor")}

    assert counts == {"exam_room": 1, "corridor": 1}
    assert all(r.area() > 7000 for r in parsed.rooms)
