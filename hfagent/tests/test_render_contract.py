# -*- coding: utf-8 -*-
"""render_plan I/O contract: exact palette colours, walls, white background."""
import numpy as np

from hfagent.schema.palette import BACKGROUND_RGB, ROOM_RGB, WALL_RGB
from hfagent.tests.synth import simple_clinic
from hfagent.tools.render_plan import render_plan


def test_render_uses_exact_legend_colors():
    img = np.asarray(render_plan(simple_clinic()))
    present = {tuple(c) for c in img.reshape(-1, 3)[::13]}
    for t in ("waiting", "exam_room", "corridor", "toilet"):
        assert ROOM_RGB[t] in present, f"missing legend colour for {t}"
    assert BACKGROUND_RGB in present
    assert WALL_RGB in present


def test_render_size_follows_scale():
    plan = simple_clinic()  # 12000 x 12400 mm
    img = render_plan(plan, px_per_mm=0.05, margin_px=40)
    assert img.size == (12000 * 0.05 + 80, 12400 * 0.05 + 80)


def test_render_deterministic():
    a = np.asarray(render_plan(simple_clinic()))
    b = np.asarray(render_plan(simple_clinic()))
    assert np.array_equal(a, b)
