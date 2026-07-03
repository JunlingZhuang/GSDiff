# -*- coding: utf-8 -*-
"""Prompt and metric contracts for the sealed-room Flash sweep."""
import cv2
import numpy as np

from hfagent.evaluate_sealed_prompts import PROMPT_VARIANTS, count_enclosed_regions


PROGRAM = {
    "building_type": "test clinic",
    "rooms": [
        {"type": "exam_room", "count": 2, "approx_area_m2": 16},
        {"type": "corridor", "count": 1},
    ],
    "adjacency": [["exam_room", "corridor"]],
}


def test_sweep_variants_are_distinct_no_door_prompts():
    # p13/p14 intentionally reuse the p11/p12 prompt builders (they differ only by
    # thinking_level), so distinct TEXTS = registry size minus those two aliases.
    prompts = {name: builder(PROGRAM) for name, builder in PROMPT_VARIANTS.items()}
    aliased = {"p13-wings-high", "p14-semantic-high"}
    distinct = [p for name, p in prompts.items() if name not in aliased]
    assert len(set(distinct)) == len(distinct)

    for prompt in prompts.values():
        lower = prompt.lower()
        assert "exam_room_1" in prompt
        assert "exam_room_2" in prompt
        assert "corridor" in prompt
        assert any(word in lower for word in ("closed", "sealed", "continuous"))


def test_partition_map_variant_avoids_floor_plan_prior():
    prompt = PROMPT_VARIANTS["p3-partition-map"](PROGRAM).lower()
    assert "partition map" in prompt
    assert "floor plan" not in prompt
    assert "corridor is a sealed labelled rectangle" in prompt


def test_count_enclosed_regions_ignores_canvas_and_text_specks():
    image = np.full((300, 500), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (200, 250), 0, 12)
    cv2.rectangle(image, (260, 50), (450, 250), 0, 12)
    cv2.circle(image, (100, 100), 4, 0, -1)
    ok, encoded = cv2.imencode(".png", image)

    assert ok
    assert count_enclosed_regions(encoded.tobytes(), min_area=500) == 2
