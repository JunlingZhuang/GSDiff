# -*- coding: utf-8 -*-
"""THE Phase 0-A gate: synthetic Plan -> render_plan -> cv_parse -> compare.

Passing this proves the parser is trustworthy in isolation (no VLM noise), so
any quality gap measured later on real Gemini images is attributable to the
VLM image alone.
"""
import pytest

from hfagent.metrics import compare_plans
from hfagent.tests.synth import ALL
from hfagent.tools.cv_parse import cv_parse
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
