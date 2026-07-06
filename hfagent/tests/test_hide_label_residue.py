# -*- coding: utf-8 -*-
"""Render-side label-residue hiding: safety properties (pure geometry)."""
import numpy as np

from hfagent.tools.linework_tracer import hide_label_residue, polygonize_rooms
from hfagent.tools.text_mask import detect_label_quads
from hfagent.tools.wall_graph import WallSegment


def _square_room():
    """A 100x100 room ringed by four wall segments (ww=6)."""
    return [
        WallSegment("horizontal", 0.0, 0.0, 100.0, 6.0),
        WallSegment("horizontal", 100.0, 0.0, 100.0, 6.0),
        WallSegment("vertical", 0.0, 0.0, 100.0, 6.0),
        WallSegment("vertical", 100.0, 0.0, 100.0, 6.0),
    ]


def _quad(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)


def test_interior_fragment_cluster_is_hidden_but_walls_stay():
    walls = _square_room()
    fragments = [
        WallSegment("horizontal", 50.0, 20.0, 38.0, 4.0),
        WallSegment("horizontal", 50.0, 44.0, 60.0, 4.0),
    ]
    polygons = polygonize_rooms(walls, (200, 200))
    shown = hide_label_residue(walls + fragments, polygons, 6)
    assert len(shown) == 4
    assert all(s in shown for s in walls)


def test_lone_counter_bar_survives():
    walls = _square_room()
    counter = WallSegment("horizontal", 50.0, 20.0, 80.0, 4.0)   # 60 px = 10 ww
    polygons = polygonize_rooms(walls, (200, 200))
    shown = hide_label_residue(walls + [counter], polygons, 6)
    assert counter in shown


def test_quad_conviction_hides_wall_welded_stub_but_not_ring_edges():
    walls = _square_room()
    # label stub welded to the left wall: touches the boundary, so the interior
    # test spares it — the OCR-quad conviction must catch it
    stub = WallSegment("horizontal", 30.0, 0.0, 26.0, 4.0)
    polygons = polygonize_rooms(walls, (200, 200))
    quad = _quad(-5, 22, 40, 38)                 # covers the stub AND grazes the wall
    shown = hide_label_residue(walls + [stub], polygons, 6, text_quads=[quad])
    assert stub not in shown
    assert all(s in shown for s in walls)        # ring edges never hidden


def test_ring_edge_inside_text_quad_is_kept():
    walls = _square_room()
    polygons = polygonize_rooms(walls, (200, 200))
    quad = _quad(-5, -8, 105, 8)                 # a quad riding the whole top wall
    shown = hide_label_residue(list(walls), polygons, 6, text_quads=[quad])
    assert len(shown) == 4


def test_missing_model_returns_no_quads(monkeypatch, tmp_path):
    from hfagent.tools import text_mask
    monkeypatch.setattr(text_mask, "MODEL_PATH", tmp_path / "absent.onnx")
    assert detect_label_quads(np.full((64, 64), 255, np.uint8)) == []
