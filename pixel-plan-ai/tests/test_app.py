from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from code_policy import validate_generated_code, validate_program_contract
from gemini import build_prompt, normalize_usage_metadata
from healthcare_rules import load_healthcare_rules
from job_progress import publish_job_progress
from program import calculate_scale, normalize_program
from runtime import execute_pixel_code
from service import generate_plan
from validator import has_non_rectangular_footprint, validate_plan
import service


class CodePolicyTests(unittest.TestCase):
    def test_us_healthcare_rules_define_room_proportion_ranges(self) -> None:
        rules = load_healthcare_rules()
        self.assertEqual(rules["profile"], "us-healthcare-schematic-v1")
        self.assertEqual(rules["room_types"]["patient_room"]["minimum_clear_dimension_ft"], 10)
        self.assertEqual(
            rules["room_types"]["patient_room"]["en_suite_module"]["preferred_placement"],
            "inboard_corner",
        )
        self.assertTrue(rules["room_types"]["exam_room"]["must_be_independent"])
        self.assertIn(["patient_room", "toilet"], rules["allowed_module_containment"])
        self.assertEqual(rules["door_swing_defaults"]["corridor_to_room"], "swing_into_room")

    def test_agent_prompt_requires_doors_derived_from_final_grid(self) -> None:
        samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))
        program = normalize_program(samples["clinic-small"])
        prompt = build_prompt(
            program,
            "",
            {"width": 64, "height": 40, "meters_per_cell": 0.25},
            None,
        )
        self.assertIn("Never guess or hard-code door coordinates", prompt)
        self.assertIn("shared-boundary scanner", prompt)
        self.assertIn("corridor count must match", prompt)
        self.assertIn("Precomputed pixel planning targets", prompt)
        self.assertIn("recommended_rectangle_cells", prompt)

    def test_gemini_usage_metadata_includes_thinking_cost(self) -> None:
        usage = normalize_usage_metadata(
            "gemini-3.1-pro-preview",
            {
                "promptTokenCount": 20_000,
                "candidatesTokenCount": 10_000,
                "thoughtsTokenCount": 30_000,
                "totalTokenCount": 60_000,
            },
        )
        self.assertEqual(usage["total_tokens"], 60_000)
        self.assertEqual(usage["thinking_tokens"], 30_000)
        self.assertAlmostEqual(usage["estimated_cost_usd"], 0.52)

    def test_inpatient_sample_matches_its_24_bed_us_label(self) -> None:
        samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))
        program = samples["inpatient-ward"]
        counts = {room["type"]: room["count"] for room in program["rooms"]}
        self.assertEqual(counts["patient_room"], 24)
        self.assertEqual(counts["toilet"], 24)
        self.assertIn(["toilet", "patient_room"], program["adjacency"])
        self.assertIn(["storage", "corridor"], program["adjacency"])

    def test_program_accepts_square_feet_and_normalizes_to_square_meters(self) -> None:
        program = normalize_program(
            {
                "building_type": "U.S. outpatient clinic",
                "rooms": [{"type": "exam_room", "count": 1, "approx_area_ft2": 120}],
            }
        )
        self.assertAlmostEqual(program["rooms"][0]["approx_area_m2"], 11.1483648)

    def test_footprint_shape_distinguishes_inset_rectangle_from_l_shape(self) -> None:
        width, height = 8, 6
        rectangle = [int(1 <= x < 7 and 1 <= y < 5) for y in range(height) for x in range(width)]
        l_shape = [int((1 <= x < 3 and 1 <= y < 5) or (1 <= x < 7 and 3 <= y < 5)) for y in range(height) for x in range(width)]
        self.assertFalse(has_non_rectangular_footprint(rectangle, width, height))
        self.assertTrue(has_non_rectangular_footprint(l_shape, width, height))

    def test_accepts_only_pixel_plan_calls(self) -> None:
        code = "\n".join(
            [
                "plan = PixelPlan(32, 24, 0.5)",
                'plan.fill_rect("room-1", "office", 0, 0, 32, 20)',
                'plan.fill_rect("corridor-1", "corridor", 0, 20, 32, 4)',
                'plan.add_door("door-1", "room-1", "corridor-1", 14, 20, "horizontal", 2)',
                'plan.add_door("main-entrance", "corridor-1", None, 0, 21, "vertical", 2)',
                "plan.finish()",
            ]
        )
        validate_generated_code(code)

    def test_rejects_unauthorized_imports(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot import module"):
            validate_generated_code("import os\nplan = PixelPlan(32, 24, 0.5)\nplan.finish()")

    def test_executes_numpy_result_contract(self) -> None:
        code = """import numpy as np
width = 8
height = 8
footprint = np.ones((height, width), dtype=bool)
grid = np.full((height, width), -1, dtype=int)
rooms = [{"id": "office-1", "type": "office"}, {"id": "corridor-1", "type": "corridor"}]
grid[:6, :] = 0
grid[6:, :] = 1
doors = [
    {"id": "door-1", "from_room": "office-1", "to_room": "corridor-1", "x": 3, "y": 6, "orientation": "horizontal", "width_cells": 1},
    {"id": "entrance", "from_room": "corridor-1", "to_room": None, "x": 0, "y": 6, "orientation": "vertical", "width_cells": 2},
]
result = {"width": width, "height": height, "meters_per_cell": 0.5, "footprint": footprint, "grid": grid, "rooms": rooms, "doors": doors}
"""
        validate_generated_code(code)
        _, plan = execute_pixel_code(code)
        self.assertEqual([room["id"] for room in plan["rooms"]], ["office-1", "corridor-1"])
        self.assertEqual(len(plan["doors"]), 2)
        self.assertEqual(plan["doors"][0]["swing_side"], "north")
        self.assertEqual(plan["doors"][1]["swing_side"], "east")

    def test_executes_generated_functions_loops_variables_and_math(self) -> None:
        code = "\n".join(
            [
                "def add_rooms(target, count):",
                "    for index in range(count):",
                '        target.fill_rect(f"room-{index}", "office", index * 4, 0, 4, 8)',
                "plan = PixelPlan(20, 12, 0.5)",
                "add_rooms(plan, 3)",
                'plan.fill_rect("corridor-1", "corridor", 0, 8, 20, 4)',
                "for index in range(3):",
                '    plan.add_door(f"door-{index}", f"room-{index}", "corridor-1", index * 4 + 1, 8, "horizontal", 2)',
                'plan.add_main_entrance("main-entrance", "corridor-1", "west", 2)',
                "plan.finish()",
            ]
        )
        _, plan = execute_pixel_code(code)
        self.assertEqual(len(plan["rooms"]), 4)
        self.assertEqual(len(plan["doors"]), 4)

    def test_rejects_unknown_methods(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot access attribute"):
            validate_generated_code("plan = PixelPlan(32, 24, 0.5)\nplan.run_shell(\"x\")\nplan.finish()")

    def test_rejects_door_away_from_shared_boundary(self) -> None:
        code = "\n".join(
            [
                "plan = PixelPlan(32, 24, 0.5)",
                'plan.fill_rect("room-1", "office", 0, 0, 32, 20)',
                'plan.fill_rect("corridor-1", "corridor", 0, 20, 32, 4)',
                'plan.add_door("door-1", "room-1", "corridor-1", 14, 10, "horizontal", 2)',
                "plan.finish()",
            ]
        )
        with self.assertRaisesRegex(ValueError, "shared boundary"):
            execute_pixel_code(code)

    def test_cross_corridors_and_automatic_entrance_execute(self) -> None:
        code = "\n".join(
            [
                "plan = PixelPlan(40, 30, 0.5)",
                'plan.fill_cross_corridors(["north", "south", "west", "east"], 20, 15, 4)',
                'plan.add_main_entrance("main-entrance", "west", "west", 2)',
                "plan.finish()",
            ]
        )
        _, plan = execute_pixel_code(code)
        self.assertEqual(len(plan["rooms"]), 4)
        self.assertEqual(len(plan["doors"]), 1)
        self.assertIsNone(plan["doors"][0]["to_room"])

    def test_l_shaped_corridor_path_executes(self) -> None:
        code = "\n".join(
            [
                "plan = PixelPlan(40, 30, 0.5)",
                'plan.fill_corridor_path("corridor-l", [[0, 10], [24, 10], [24, 24]], 4)',
                'plan.add_main_entrance("main-entrance", "corridor-l", "west", 2)',
                "plan.finish()",
            ]
        )
        _, plan = execute_pixel_code(code)
        room = plan["rooms"][0]
        self.assertGreater(room["bounds"]["width"], room["bounds"]["height"])
        self.assertEqual(plan["doors"][0]["from_room"], "corridor-l")

    def test_room_band_recovers_from_wrong_preferred_door_side(self) -> None:
        code = "\n".join(
            [
                "plan = PixelPlan(30, 24, 0.5)",
                'plan.fill_rect("corridor-1", "corridor", 0, 10, 30, 4)',
                'plan.fill_room_band([["room-1", "office", 1]], 0, 14, 30, 10, "x", "corridor-1", "bottom")',
                "plan.finish()",
            ]
        )
        _, plan = execute_pixel_code(code)
        self.assertEqual(len(plan["doors"]), 1)
        self.assertEqual(plan["doors"][0]["orientation"], "horizontal")
        self.assertEqual(plan["doors"][0]["y"], 14)

    def test_validator_blocks_skinny_rooms_and_improper_patient_room_nesting(self) -> None:
        width = height = 12
        cells = [-1] * (width * height)

        def paint(index: int, x0: int, y0: int, x1: int, y1: int) -> None:
            for y in range(y0, y1):
                for x in range(x0, x1):
                    cells[y * width + x] = index

        paint(0, 0, 0, 6, 6)
        paint(0, 3, 6, 6, 8)
        paint(1, 0, 6, 3, 8)
        paint(2, 0, 8, 12, 12)
        plan = {
            "width": width,
            "height": height,
            "meters_per_cell": 0.52,
            "footprint": [1] * (width * height),
            "cells": cells,
            "rooms": [
                {"id": "patient_room-1", "type": "patient_room", "color": "#000", "pixel_count": 42, "bounds": {"x": 0, "y": 0, "width": 6, "height": 8}},
                {"id": "office-1", "type": "office", "color": "#000", "pixel_count": 6, "bounds": {"x": 0, "y": 6, "width": 3, "height": 2}},
                {"id": "corridor-1", "type": "corridor", "color": "#000", "pixel_count": 48, "bounds": {"x": 0, "y": 8, "width": 12, "height": 4}},
            ],
            "doors": [
                {"id": "patient-door", "from_room": "patient_room-1", "to_room": "corridor-1", "x": 4, "y": 8, "orientation": "horizontal", "width_cells": 1},
                {"id": "office-door", "from_room": "office-1", "to_room": "corridor-1", "x": 1, "y": 8, "orientation": "horizontal", "width_cells": 1},
                {"id": "entrance", "from_room": "corridor-1", "to_room": None, "x": 0, "y": 12, "orientation": "horizontal", "width_cells": 1},
            ],
        }
        program = normalize_program(
            {
                "building_type": "U.S. hospital inpatient unit",
                "rooms": [
                    {"type": "patient_room", "count": 1, "approx_area_m2": 11.36},
                    {"type": "office", "count": 1, "approx_area_m2": 1.62},
                    {"type": "corridor", "count": 1},
                ],
                "adjacency": [["patient_room", "corridor"], ["office", "corridor"]],
            }
        )
        validation = validate_plan(plan, program)
        category_results = {check["category"]: check["pass"] for check in validation["checks"]}
        self.assertFalse(category_results["area"])
        self.assertFalse(category_results["proportion"])
        self.assertFalse(category_results["zoning"])
        self.assertGreater(validation["summary"]["zoning_violations"], 0)

    def test_large_tower_contract_allows_custom_geometry_code(self) -> None:
        program = {
            "building_type": "hospital tower",
            "rooms": [{"type": "patient_room", "count": 40}],
        }
        code = "\n".join(
            [
                "plan = PixelPlan(40, 30, 0.5)",
                'plan.fill_cross_corridors(["north", "south", "west", "east"], 20, 15, 4)',
                'plan.add_door("main-entrance", "west", None, 0, 13, "vertical", 2)',
                "plan.finish()",
            ]
        )
        validate_program_contract(code, program)


class AgentGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))

    def test_manual_inspection_executes_persistent_code(self) -> None:
        code = "result = {'width': 1, 'height': 1}"
        execution = {
            "code": code,
            "plan": {"rooms": []},
            "validation": {"score": 95, "checks": [], "issues": [], "areas": []},
        }
        with patch.object(service, "execute_and_validate", return_value=execution):
            inspected = service.execute_code_action(
                {
                    "action": "inspect",
                    "program": self.samples["clinic-small"],
                    "code": code,
                }
            )
        self.assertEqual(inspected["source"], "deterministic-inspection")
        self.assertEqual(inspected["iterations"][0]["phase"], "inspect")
        self.assertTrue(inspected["inspection"]["accepted"])

    def test_completed_attempt_is_published_as_an_atomic_progress_checkpoint(self) -> None:
        iteration = {
            "attempt": 1,
            "phase": "initial",
            "source": "gemini",
            "status": "rejected",
            "score": 55,
            "code": "result = {}",
        }
        result = {
            "plan": {"width": 1, "height": 1, "rooms": []},
            "validation": {"score": 55, "checks": [], "issues": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            with patch.dict(os.environ, {"PIXEL_PLAN_PROGRESS_FILE": str(path)}):
                publish_job_progress(
                    [iteration],
                    phase="initial",
                    result=result,
                    code="result = {}",
                    source="gemini",
                    model="test-model",
                    message="First attempt completed.",
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["iterations"], [iteration])
        self.assertEqual(payload["preview"]["code"], "result = {}")
        self.assertEqual(payload["preview"]["validation"]["score"], 55)

    def test_revise_action_sends_existing_complete_code_to_gemini(self) -> None:
        model_output = {
            "code": "revised-code",
            "strategy": "revision",
            "assumptions": [],
            "model": "quality-model",
        }
        execution_output = {
            "code": "revised-code",
            "plan": {},
            "validation": {"score": 95, "checks": [], "issues": []},
        }
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "quality-model"}),
            patch.object(service, "generate_gemini_code", return_value=model_output) as generate_mock,
            patch.object(service, "execute_and_validate", return_value=execution_output),
        ):
            result = service.generate_plan(
                {
                    "action": "revise",
                    "program": self.samples["clinic-small"],
                    "prompt": "Move the main entrance south.",
                    "current_code": "existing-complete-code",
                    "options": {"width": 64, "height": 40},
                }
            )
        revision_context = generate_mock.call_args.args[5]
        self.assertIn("existing-complete-code", revision_context)
        self.assertIn("Move the main entrance south", revision_context)
        self.assertEqual(result["source"], "gemini-revised")
        self.assertEqual(result["iterations"][0]["phase"], "revise")

    def test_repair_iterations_record_rejected_then_accepted(self) -> None:
        model_outputs = [
            {"code": "first", "strategy": "first", "assumptions": [], "model": "test-model"},
            {"code": "second", "strategy": "second", "assumptions": [], "model": "test-model"},
        ]
        execution_outputs = [
            {
                "code": "first",
                "plan": {},
                "validation": {"score": 40, "checks": [], "issues": ["Missing required adjacency."]},
            },
            {
                "code": "second",
                "plan": {},
                "validation": {"score": 91, "checks": [], "issues": []},
            },
        ]
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"}),
            patch.object(service, "generate_gemini_code", side_effect=model_outputs),
            patch.object(service, "execute_and_validate", side_effect=execution_outputs),
        ):
            result = service.generate_plan(
                {
                    "program": self.samples["clinic-small"],
                    "mode": "auto",
                    "options": {"width": 64, "height": 40},
                }
            )
        self.assertEqual(result["source"], "gemini-repaired")
        self.assertEqual([item["phase"] for item in result["iterations"]], ["initial", "repair"])
        self.assertEqual([item["status"] for item in result["iterations"]], ["rejected", "accepted"])
        self.assertEqual(result["iterations"][0]["score"], 40)
        self.assertEqual(result["iterations"][1]["score"], 91)

    def test_repair_reverts_to_best_executable_code_after_failed_edit(self) -> None:
        model_outputs = [
            {"code": "best-code", "strategy": "best", "assumptions": [], "model": "test-model"},
            {"code": "broken-edit", "strategy": "broken", "assumptions": [], "model": "test-model"},
            {"code": "fixed-code", "strategy": "fixed", "assumptions": [], "model": "test-model"},
        ]
        rejected = {
            "code": "best-code",
            "plan": {},
            "validation": {
                "score": 80,
                "checks": [{"category": "area", "label": "U.S. room-area range: 7/8", "pass": False}],
                "issues": ["One room is outside its area range."],
            },
        }
        accepted = {
            "code": "fixed-code",
            "plan": {},
            "validation": {"score": 95, "checks": [], "issues": []},
        }
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"}),
            patch.object(service, "generate_gemini_code", side_effect=model_outputs) as generate_mock,
            patch.object(
                service,
                "execute_and_validate",
                side_effect=[rejected, ValueError("invalid syntax"), accepted],
            ),
        ):
            result = service.generate_plan(
                {
                    "program": self.samples["clinic-small"],
                    "mode": "auto",
                    "options": {"width": 64, "height": 40},
                }
            )
        third_context = generate_mock.call_args_list[2].args[5]
        self.assertIn("best-code", third_context)
        self.assertNotIn("broken-edit", third_context)
        self.assertIn("best executable", third_context)
        self.assertEqual(result["source"], "gemini-repaired")

    def test_tower_architecture_failure_blocks_high_scoring_candidate(self) -> None:
        result = {
            "validation": {
                "score": 95,
                "checks": [
                    {"category": "massing", "label": "Explicit non-rectangular tower footprint", "pass": False},
                    {"category": "doors", "label": "All occupiable rooms have doors", "pass": True},
                ],
                "issues": [],
            }
        }
        reason = service.candidate_rejection_reason(result)
        self.assertIn("Explicit non-rectangular tower footprint", reason)

    def test_small_number_of_area_outliers_is_advisory(self) -> None:
        result = {
            "validation": {
                "score": 90,
                "checks": [
                    {"category": "area", "label": "U.S. room-area range: 8/10", "pass": False},
                ],
                "issues": ["Two rooms are outside their target area range."],
                "areas": [
                    {"pass": index < 8}
                    for index in range(10)
                ],
            }
        }
        self.assertEqual(service.candidate_rejection_reason(result), "")

    def test_widespread_area_failure_still_blocks_acceptance(self) -> None:
        result = {
            "validation": {
                "score": 90,
                "checks": [
                    {"category": "area", "label": "U.S. room-area range: 7/10", "pass": False},
                ],
                "issues": ["Three rooms are outside their target area range."],
                "areas": [
                    {"pass": index < 7}
                    for index in range(10)
                ],
            }
        }
        self.assertIn("U.S. room-area range", service.candidate_rejection_reason(result))

    def test_ninety_percent_proportion_compliance_is_advisory(self) -> None:
        result = {
            "plan": {"rooms": [{"type": "exam_room"} for _ in range(10)]},
            "validation": {
                "score": 90,
                "checks": [
                    {"category": "proportion", "label": "Reasonable room proportion: 9/10", "pass": False},
                ],
                "summary": {"proportion_compliant_rooms": 9},
                "issues": ["One room proportion is outside its preferred range."],
                "areas": [],
            },
        }
        with patch.dict(os.environ, {"PROPORTION_MIN_ACCEPTANCE_RATIO": "0.90"}):
            self.assertEqual(service.candidate_rejection_reason(result), "")

    def test_proportion_compliance_below_ninety_percent_blocks_acceptance(self) -> None:
        result = {
            "plan": {"rooms": [{"type": "exam_room"} for _ in range(10)]},
            "validation": {
                "score": 90,
                "checks": [
                    {"category": "proportion", "label": "Reasonable room proportion: 8/10", "pass": False},
                ],
                "summary": {"proportion_compliant_rooms": 8},
                "issues": ["Two room proportions are outside their preferred ranges."],
                "areas": [],
            },
        }
        with patch.dict(os.environ, {"PROPORTION_MIN_ACCEPTANCE_RATIO": "0.90"}):
            self.assertIn("Reasonable room proportion", service.candidate_rejection_reason(result))

    def test_large_program_starts_with_complex_model_and_medium_thinking(self) -> None:
        model_output = {
            "code": "first",
            "strategy": "tower",
            "assumptions": [],
            "model": "complex-model",
        }
        execution_output = {
            "code": "first",
            "plan": {},
            "validation": {"score": 95, "checks": [], "issues": []},
        }
        with (
            patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "test-key",
                    "GEMINI_FAST_MODEL": "fast-model",
                    "GEMINI_MODEL": "quality-model",
                    "GEMINI_COMPLEX_MODEL": "complex-model",
                    "GEMINI_THINKING_LEVEL": "low",
                    "GEMINI_COMPLEX_THINKING_LEVEL": "medium",
                },
            ),
            patch.object(service, "generate_gemini_code", return_value=model_output) as generate_mock,
            patch.object(service, "execute_and_validate", return_value=execution_output),
        ):
            result = service.generate_plan(
                {
                    "program": self.samples["hospital-tower-floor"],
                    "mode": "auto",
                    "options": {"width": 96, "height": 64},
                }
            )
        self.assertEqual(result["source"], "gemini")
        self.assertEqual(generate_mock.call_args.args[1], "complex-model")
        self.assertEqual(generate_mock.call_args.kwargs["thinking_level"], "medium")

    def test_rejection_feedback_groups_repeated_area_failures(self) -> None:
        validation = {
            "score": 55,
            "checks": [
                {"category": "area", "label": "U.S. room-area range: 1/6", "pass": False},
                {"category": "doors", "label": "Rooms with doors: 0/6", "pass": False},
            ],
            "areas": [
                {
                    "id": f"patient_room_{index}",
                    "type": "patient_room",
                    "actual_ft2": 35,
                    "target_ft2": 172,
                    "actual_cells": 16,
                    "target_cells": 78.5,
                    "tolerance_percent": 25,
                    "pass": False,
                }
                for index in range(1, 6)
            ],
            "issues": [
                f"patient_room_{index} area is 35 sf; target 172 sf with ±25% tolerance."
                for index in range(1, 6)
            ],
        }
        feedback = service.compact_validation_feedback(validation)
        self.assertIn("patient_room: 5 failing rooms", feedback)
        self.assertIn("actual 35 sf / 16-16 cells, target 172 sf / about 78.5 cells", feedback)
        self.assertNotIn("patient_room_5 area", feedback)

    def test_code_agent_reports_failure_after_five_unexecutable_attempts_without_fallback(self) -> None:
        model_outputs = [
            {"code": f"attempt-{index}", "strategy": "repair", "assumptions": [], "model": "test-model"}
            for index in range(1, 6)
        ]

        def execute_side_effect(code: str, program: dict[str, object], enforce_ai_contract: bool = False) -> dict[str, object]:
            raise ValueError(f"executor rejected {code}")

        with (
            patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model", "GEMINI_MAX_ATTEMPTS": "5"},
            ),
            patch.object(service, "generate_gemini_code", side_effect=model_outputs) as generate_mock,
            patch.object(service, "execute_and_validate", side_effect=execute_side_effect),
            patch.object(service, "publish_job_progress") as progress_mock,
        ):
            with self.assertRaisesRegex(ValueError, "exhausted 5 attempts"):
                service.generate_plan(
                    {
                        "program": self.samples["clinic-small"],
                        "mode": "auto",
                        "options": {"width": 64, "height": 40},
                    }
                )
        self.assertEqual(generate_mock.call_count, 5)
        self.assertEqual(progress_mock.call_count, 10)
        running_checkpoints = [
            call.args[0][-1]
            for call in progress_mock.call_args_list
            if call.args and call.args[0] and call.args[0][-1]["status"] == "running"
        ]
        self.assertEqual(len(running_checkpoints), 5)
        self.assertEqual(running_checkpoints[0]["attempt"], 1)
        self.assertIn("Waiting for", running_checkpoints[0]["message"])
        self.assertIn("complete Python program", running_checkpoints[0]["message"])

    def test_generate_keeps_best_executable_ai_candidate_after_exhaustion(self) -> None:
        model_outputs = [
            {"code": f"attempt-{index}", "strategy": "tower", "assumptions": [], "model": "quality-model"}
            for index in range(1, 6)
        ]
        best_execution = {
            "code": "attempt-1",
            "plan": {},
            "validation": {
                "score": 55,
                "checks": [{"category": "area", "label": "U.S. room-area range: 70/78", "pass": False}],
                "issues": ["Several rooms are outside their area ranges."],
                "areas": [],
            },
        }

        def execute_side_effect(code: str, program: dict[str, object], enforce_ai_contract: bool = False) -> dict[str, object]:
            if code == "attempt-1":
                return best_execution
            raise ValueError(f"executor rejected {code}")

        with (
            patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "quality-model", "GEMINI_MAX_ATTEMPTS": "5"},
            ),
            patch.object(service, "generate_gemini_code", side_effect=model_outputs),
            patch.object(service, "execute_and_validate", side_effect=execute_side_effect),
        ):
            result = service.generate_plan(
                {
                    "program": self.samples["hospital-tower-floor"],
                    "mode": "auto",
                    "options": {"width": 96, "height": 64},
                }
            )
        self.assertEqual(result["source"], "gemini-unaccepted")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["code"], "attempt-1")
        self.assertEqual(result["validation"]["score"], 55)
        self.assertEqual(len(result["iterations"]), 5)


class ImageModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))

    def test_reference_image_normalization_accepts_and_rejects(self) -> None:
        self.assertIsNone(service.normalize_reference_image(None))
        self.assertIsNone(service.normalize_reference_image(""))
        normalized = service.normalize_reference_image({"mime": "image/png", "data": "aGVsbG8="})
        self.assertEqual(normalized, {"mime": "image/png", "data": "aGVsbG8="})
        with self.assertRaises(ValueError):
            service.normalize_reference_image("not-an-object")
        with self.assertRaises(ValueError):
            service.normalize_reference_image({"mime": "text/plain", "data": "aGVsbG8="})
        with self.assertRaises(ValueError):
            service.normalize_reference_image({"mime": "image/png", "data": ""})

    def test_prompt_swaps_family_guidance_for_transcription_when_image_attached(self) -> None:
        program = normalize_program(self.samples["clinic-small"])
        options = {"width": 64, "height": 40, "meters_per_cell": 0.25}
        plain = build_prompt(program, "", options, None)
        with_image = build_prompt(program, "", options, None, with_reference_image=True)
        self.assertNotIn("TRANSCRIBE", plain)
        self.assertIn("TRANSCRIBE that drawing onto the cell grid", with_image)
        self.assertIn("Do not mirror or rotate the layout", with_image)

    def test_generate_passes_reference_image_to_every_gemini_attempt(self) -> None:
        model_output = {"code": "ok", "strategy": "s", "assumptions": [], "model": "quality-model"}
        execution_output = {
            "code": "ok",
            "plan": {},
            "validation": {"score": 95, "checks": [], "issues": []},
        }
        reference = {"mime": "image/png", "data": "aGVsbG8="}
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "quality-model"}),
            patch.object(service, "generate_gemini_code", return_value=model_output) as generate_mock,
            patch.object(service, "execute_and_validate", return_value=execution_output),
        ):
            service.generate_plan(
                {
                    "program": self.samples["clinic-small"],
                    "mode": "auto",
                    "options": {"width": 64, "height": 40},
                    "reference_image": reference,
                }
            )
        self.assertEqual(generate_mock.call_args.kwargs["reference_image"], reference)

    def test_generate_without_reference_image_stays_backward_compatible(self) -> None:
        model_output = {"code": "ok", "strategy": "s", "assumptions": [], "model": "quality-model"}
        execution_output = {
            "code": "ok",
            "plan": {},
            "validation": {"score": 95, "checks": [], "issues": []},
        }
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "quality-model"}),
            patch.object(service, "generate_gemini_code", return_value=model_output) as generate_mock,
            patch.object(service, "execute_and_validate", return_value=execution_output),
        ):
            result = service.generate_plan(
                {
                    "program": self.samples["clinic-small"],
                    "mode": "auto",
                    "options": {"width": 64, "height": 40},
                }
            )
        self.assertIsNone(generate_mock.call_args.kwargs["reference_image"])
        self.assertTrue(result["accepted"])

    def test_images_action_returns_candidates_and_clamps_count(self) -> None:
        drawings = [{"mime": "image/png", "data": "aW1n"}] * 4
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(service, "generate_plan_images", return_value=drawings) as images_mock,
        ):
            result = service.generate_images_action(
                {"program": self.samples["clinic-small"], "count": 99}
            )
        self.assertEqual(images_mock.call_args.args[2], 4)
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["images"], drawings)

    def test_dispatch_routes_images_job_kind(self) -> None:
        with patch.object(service, "generate_images_action", return_value={"images": []}) as action_mock:
            service.dispatch_job({"job_kind": "images", "program": {}})
        action_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
