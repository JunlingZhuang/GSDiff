# -*- coding: utf-8 -*-
"""Room typing: closed-vocabulary matching (pure) + label reading (model-gated)."""
import cv2
import numpy as np
import pytest

from hfagent.schema.plan import Plan, Room
from hfagent.tools.room_typing import (RECOGNITION_MODEL_PATH, match_program_type,
                                       type_rooms)
from hfagent.tools.text_mask import MODEL_PATH as DETECTION_MODEL_PATH

VOCAB = ["patient_room", "toilet", "exam_room", "corridor", "nurse_station"]


def test_exact_label_with_instance():
    room_type, instance, score = match_program_type("patientroom12", VOCAB)
    assert (room_type, instance) == ("patient_room", 12)
    assert score == 1.0


def test_ocr_noise_still_lands_on_the_right_type():
    room_type, instance, _ = match_program_type("patlentroom3", VOCAB)
    assert (room_type, instance) == ("patient_room", 3)
    room_type, _, _ = match_program_type("tollet", VOCAB)
    assert room_type == "toilet"


def test_area_sublabel_matches_nothing():
    room_type, _, _ = match_program_type("16m2", VOCAB)
    assert room_type is None


def test_empty_and_garbage_are_rejected():
    assert match_program_type("", VOCAB)[0] is None
    assert match_program_type("###", VOCAB)[0] is None
    assert match_program_type("zzzz", VOCAB)[0] is None


@pytest.mark.skipif(not (RECOGNITION_MODEL_PATH.exists() and DETECTION_MODEL_PATH.exists()),
                    reason="local OCR models not installed")
def test_reads_a_drawn_label_into_a_room_type():
    image = np.full((400, 800), 255, np.uint8)
    cv2.rectangle(image, (40, 40), (760, 360), 0, 8)          # a room outline
    cv2.putText(image, "exam_room_2", (240, 210), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, 0, 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", image)
    plan = Plan(units="px", rooms=[Room(id="r1", type="unknown",
                                        polygon=[(44, 44), (756, 44), (756, 356), (44, 356)])])
    program = {"rooms": [{"type": t, "count": 1} for t in VOCAB]}
    guesses = type_rooms(buf.tobytes(), plan, program)
    assert "r1" in guesses
    assert guesses["r1"].type == "exam_room"
    assert guesses["r1"].instance == 2
