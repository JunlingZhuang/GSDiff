"""Deterministic tests for the real-plan -> OCR/CV linework parser."""
import cv2
import numpy as np

from hfagent.tools.linework_parser import parse_linework


PROGRAM = {
    "building_type": "test clinic",
    "rooms": [
        {"type": "waiting", "count": 1},
        {"type": "exam_room", "count": 1},
        {"type": "corridor", "count": 1},
    ],
    "adjacency": [["waiting", "corridor"], ["exam_room", "corridor"]],
}


class FakeOCR:
    def readtext(self, image, **kwargs):
        def item(text, x, y):
            return ([[x - 30, y - 8], [x + 30, y - 8], [x + 30, y + 8], [x - 30, y + 8]], text, 0.99)

        return [
            item("waiting", 105, 80),
            item("exam_room", 295, 80),
            item("corridor", 200, 195),
        ]


def _line_plan() -> np.ndarray:
    image = np.full((240, 400, 3), 255, np.uint8)
    black = (0, 0, 0)
    cv2.rectangle(image, (15, 15), (385, 225), black, 9)
    cv2.line(image, (200, 15), (200, 150), black, 9)
    cv2.line(image, (15, 150), (85, 150), black, 9)
    cv2.line(image, (115, 150), (285, 150), black, 9)
    cv2.line(image, (315, 150), (385, 150), black, 9)
    # Thin standard swing arcs are door evidence but are not wall bands.
    cv2.ellipse(image, (85, 150), (30, 30), 0, 270, 360, black, 1)
    cv2.line(image, (85, 150), (85, 120), black, 1)
    cv2.ellipse(image, (285, 150), (30, 30), 0, 270, 360, black, 1)
    cv2.line(image, (285, 150), (285, 120), black, 1)
    return image


def test_linework_parser_extracts_rooms_types_and_door_graph():
    plan, graph, diagnostics, debug_png = parse_linework(
        _line_plan(), PROGRAM, ocr_reader=FakeOCR()
    )

    assert sorted(room.type for room in plan.rooms) == ["corridor", "exam_room", "waiting"]
    assert diagnostics["unlabelled_rooms"] == 0
    assert diagnostics["parser_confident"]
    assert len(graph.rooms) == 3
    assert len(graph.doors) == 2
    assert debug_png.startswith(b"\x89PNG")
