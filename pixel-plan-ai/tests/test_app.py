from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import gemini
from code_policy import validate_generated_code, validate_program_contract
from gemini import SYSTEM_INSTRUCTION, build_prompt, normalize_usage_metadata
from healthcare_rules import load_healthcare_rules
from job_progress import publish_job_progress
from program import calculate_scale, normalize_program
from runtime import execute_pixel_code
from seed_code import normalize_seed, seed_to_code
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
        # The durable coding contract now lives in SYSTEM_INSTRUCTION, not the per-request prompt.
        self.assertIn("Never guess or hard-code door coordinates", SYSTEM_INSTRUCTION)
        self.assertIn("shared-boundary scanner", SYSTEM_INSTRUCTION)
        self.assertIn("Grid contract:", SYSTEM_INSTRUCTION)
        self.assertIn("Authorized imports:", SYSTEM_INSTRUCTION)
        self.assertIn("standalone Python program", SYSTEM_INSTRUCTION)
        self.assertNotIn("Grid contract:", prompt)
        self.assertNotIn("Execution environment:", prompt)
        self.assertNotIn("Authorized imports:", prompt)
        self.assertNotIn("standalone Python program", prompt)
        self.assertNotIn("Never guess or hard-code door coordinates", prompt)
        self.assertNotIn("shared-boundary scanner", prompt)
        # Task-level data still belongs in the per-request prompt.
        self.assertIn("corridor count must match", prompt)
        self.assertIn("Precomputed pixel planning targets", prompt)
        self.assertIn("recommended_rectangle_cells", prompt)

    def test_prompt_layers_order_static_before_program_before_attempt(self) -> None:
        samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))
        program = normalize_program(samples["inpatient-ward"])
        prompt = build_prompt(
            program,
            "",
            {"width": 96, "height": 64, "meters_per_cell": 0.25},
            "Attempt 2 of 5. Fix the named door failures.",
        )
        # "healthcare planning profile" also occurs in a layer-1 bullet, so the section
        # header (with its trailing colon) is the unambiguous layer-3 anchor.
        layout = prompt.index("Layout requirements")
        guidance = prompt.index("Building-type layout guidance")
        profile = prompt.index("U.S. healthcare planning profile:")
        targets = prompt.index("Precomputed pixel planning targets")
        program_section = prompt.index("Program:")
        repair = prompt.index("Repair context:")
        self.assertLess(layout, guidance)
        self.assertLess(guidance, profile)
        self.assertLess(profile, targets)
        self.assertLess(targets, program_section)
        self.assertLess(program_section, repair)

    def test_prompt_static_prefix_is_shared_across_programs(self) -> None:
        samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))
        options = {"width": 96, "height": 64, "meters_per_cell": 0.25}
        a = build_prompt(normalize_program(samples["clinic-small"]), "", options, None)
        b = build_prompt(normalize_program(samples["inpatient-ward"]), "", options, None)
        common = os.path.commonprefix([a, b])
        # Everything before the mode-level layer (layer 2) must be byte-identical.
        self.assertGreaterEqual(len(common), a.index("Building-type layout guidance"))

    def test_system_instruction_states_verdict_ownership_and_anti_thrashing(self) -> None:
        self.assertIn("sole judge of correctness", SYSTEM_INSTRUCTION)
        self.assertIn("minimal edit that fixes the named failures", SYSTEM_INSTRUCTION)

    def test_layout_requirements_state_validator_thresholds_not_adjectives(self) -> None:
        samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))
        program = normalize_program(samples["clinic-small"])
        prompt = build_prompt(
            program,
            "",
            {"width": 64, "height": 40, "meters_per_cell": 0.25},
            None,
        )
        # Coverage threshold from validator.py: coverage >= 0.85.
        self.assertIn("85%", prompt)
        self.assertNotIn("Fill most of the canvas", prompt)
        self.assertNotIn("Prefer compact spaces and continuous corridors", prompt)
        # Proportion check: aspect ratio <= hard_max_aspect_ratio (default 4.0).
        self.assertIn("hard_max_aspect_ratio (default 4.0)", prompt)
        self.assertIn("minimum_compactness", prompt)

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

    def test_samples_carry_square_feet_and_normalize_to_square_meters(self) -> None:
        samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))
        waiting = samples["health-center-large"]["rooms"][0]
        self.assertEqual(waiting["type"], "waiting")
        # Samples now speak square feet; square metres are gone from the user-facing data.
        self.assertEqual(waiting["approx_area_ft2"], 650)
        self.assertNotIn("approx_area_m2", waiting)
        # normalize_program keeps the internal contract metric.
        program = normalize_program(samples["health-center-large"])
        self.assertAlmostEqual(program["rooms"][0]["approx_area_m2"], 650 * 0.09290304)

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

    def test_door_micro_example_survives_the_real_sandbox(self) -> None:
        # Locks the SYSTEM_INSTRUCTION door-derivation micro-example (docs #11):
        # a full 12x8 grid where grid[4][10] == 2 (exam_room_1) sits over
        # grid[5][10] == 0 (corridor_1) must yield the example door dict verbatim
        # once the real sandbox door validator runs. Keep example and test in sync.
        code = "\n".join(
            [
                "width = 12",
                "height = 8",
                "footprint = [[True for _ in range(width)] for _ in range(height)]",
                "grid = [[0 for _ in range(width)] for _ in range(height)]",
                "for y in range(height):",
                "    for x in range(width):",
                "        if y < 5:",
                "            grid[y][x] = 1 if x < 6 else 2",
                "        else:",
                "            grid[y][x] = 0",
                "rooms = [",
                '    {"id": "corridor_1", "type": "corridor"},',
                '    {"id": "waiting_1", "type": "waiting"},',
                '    {"id": "exam_room_1", "type": "exam_room"},',
                "]",
                "doors = [",
                '    {"id": "d1", "from_room": "exam_room_1", "to_room": "corridor_1", "x": 10, "y": 5, "orientation": "horizontal", "width_cells": 1},',
                "]",
                'result = {"width": width, "height": height, "meters_per_cell": 0.5, "footprint": footprint, "grid": grid, "rooms": rooms, "doors": doors}',
            ]
        )
        _, plan = execute_pixel_code(code)
        # The grid realizes the example's exact claims.
        self.assertEqual(plan["cells"][4 * 12 + 10], 2)  # exam_room_1
        self.assertEqual(plan["cells"][5 * 12 + 10], 0)  # corridor_1
        self.assertEqual(len(plan["doors"]), 1)
        door = plan["doors"][0]
        self.assertEqual(door["id"], "d1")
        self.assertEqual(door["from_room"], "exam_room_1")
        self.assertEqual(door["to_room"], "corridor_1")
        self.assertEqual(door["orientation"], "horizontal")
        self.assertEqual(door["swing_side"], "north")

    def test_prompts_carry_the_locked_micro_examples(self) -> None:
        self.assertIn("Example (door derivation)", SYSTEM_INSTRUCTION)
        self.assertIn("unordered pair", SYSTEM_INSTRUCTION)
        samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))
        program = normalize_program(samples["clinic-small"])
        seed_prompt = build_prompt(
            program,
            "",
            {"width": 40, "height": 24, "meters_per_cell": 0.5},
            None,
            with_seed_repair=True,
        )
        self.assertIn("(40, 8, 11, 12)", seed_prompt)

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
        self.assertEqual(result["stop_reason"], "accepted")
        self.assertIn("usage_total", result)
        self.assertEqual(result["usage_total"]["total_tokens"], 0)

    def test_generate_plan_appends_typed_event_timeline(self) -> None:
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
        with tempfile.TemporaryDirectory() as directory:
            progress_path = Path(directory) / "progress.json"
            with (
                patch.dict(
                    os.environ,
                    {
                        "GEMINI_API_KEY": "test-key",
                        "GEMINI_MODEL": "test-model",
                        "PIXEL_PLAN_PROGRESS_FILE": str(progress_path),
                    },
                ),
                patch.object(service, "generate_gemini_code", side_effect=model_outputs),
                patch.object(service, "execute_and_validate", side_effect=execution_outputs),
            ):
                service.generate_plan(
                    {
                        "program": self.samples["clinic-small"],
                        "mode": "auto",
                        "options": {"width": 64, "height": 40},
                    }
                )
            events_path = Path(str(progress_path) + ".events.jsonl")
            self.assertTrue(events_path.is_file())
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self.assertEqual(
            [event["e"] for event in events],
            [
                "attempt_start",
                "model_returned",
                "validator_verdict",
                "attempt_end",
                "attempt_start",
                "model_returned",
                "validator_verdict",
                "attempt_end",
                "run_end",
            ],
        )
        self.assertEqual(events[-1]["stop_reason"], "accepted")

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

    def test_tower_architecture_failure_blocks_sub_override_candidate(self) -> None:
        # 88 sits above the 72 floor but below the 90 score override, so a hard massing
        # failure still blocks acceptance here.
        result = {
            "validation": {
                "score": 88,
                "checks": [
                    {"category": "massing", "label": "Explicit non-rectangular tower footprint", "pass": False},
                    {"category": "doors", "label": "All occupiable rooms have doors", "pass": True},
                ],
                "issues": [],
            }
        }
        reason = service.candidate_rejection_reason(result)
        self.assertIn("Explicit non-rectangular tower footprint", reason)

    def test_score_override_accepts_hard_failure_at_or_above_ninety(self) -> None:
        result = {
            "validation": {
                "score": 93,
                "checks": [
                    {"category": "count", "label": "waiting: 0/1", "pass": False},
                ],
                "issues": ["waiting count is off."],
                "areas": [],
            }
        }
        # Env default override (90): a 93 accepts even with a hard count blocker.
        self.assertEqual(service.candidate_rejection_reason(result), "")
        # Disabling the override restores blocking on the same input.
        with patch.dict(os.environ, {"ACCEPT_SCORE_OVERRIDE": "0"}):
            reason = service.candidate_rejection_reason(result)
        self.assertTrue(reason)
        self.assertIn("waiting: 0/1", reason)

    def test_accepted_result_flags_score_override_only_with_hard_failures(self) -> None:
        model_output = {"code": "ok", "strategy": "s", "assumptions": [], "model": "quality-model"}
        override_execution = {
            "code": "ok",
            "plan": {"rooms": []},
            "validation": {
                "score": 93,
                "checks": [{"category": "count", "label": "waiting: 0/1", "pass": False}],
                "issues": [],
                "areas": [],
            },
        }
        clean_execution = {
            "code": "ok",
            "plan": {"rooms": []},
            "validation": {"score": 95, "checks": [], "issues": [], "areas": []},
        }
        payload = {
            "program": self.samples["clinic-small"],
            "mode": "auto",
            "options": {"width": 64, "height": 40},
        }
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "quality-model"}),
            patch.object(service, "generate_gemini_code", return_value=model_output),
            patch.object(service, "execute_and_validate", return_value=override_execution),
        ):
            override_result = service.generate_plan(payload)
        self.assertTrue(override_result["accepted"])
        self.assertEqual(override_result["accepted_via"], "score_override")

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "quality-model"}),
            patch.object(service, "generate_gemini_code", return_value=model_output),
            patch.object(service, "execute_and_validate", return_value=clean_execution),
        ):
            clean_result = service.generate_plan(payload)
        self.assertTrue(clean_result["accepted"])
        self.assertNotIn("accepted_via", clean_result)

    def test_small_number_of_area_outliers_is_advisory(self) -> None:
        # 88 keeps this below the 90 score override, so the area-ratio path decides acceptance.
        result = {
            "validation": {
                "score": 88,
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
        # 88 keeps this below the 90 score override, so the area-ratio path decides acceptance.
        result = {
            "validation": {
                "score": 88,
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
        # 88 keeps this below the 90 score override, so the proportion-ratio path decides acceptance.
        result = {
            "plan": {"rooms": [{"type": "exam_room"} for _ in range(10)]},
            "validation": {
                "score": 88,
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
        # 88 keeps this below the 90 score override, so the proportion-ratio path decides acceptance.
        result = {
            "plan": {"rooms": [{"type": "exam_room"} for _ in range(10)]},
            "validation": {
                "score": 88,
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

    def test_validator_issue_feedback_carries_stable_rule_and_room_tags(self) -> None:
        validation = {
            "score": 45,
            "checks": [],
            "areas": [],
            "issues": [
                "waiting_0 has no door.",
                "exam_room_2 proportion fails 5.0 > 4.0.",
                "No exterior entrance door was generated.",
            ],
        }
        feedback = service.compact_validation_feedback(validation)
        self.assertIn('<validator_error rule="doors" room="waiting_0">', feedback)
        self.assertIn('<validator_error rule="proportion" room="exam_room_2">', feedback)
        # The plan-level entrance issue matches no group marker, so it is not rendered as a tag.
        self.assertNotIn("No exterior entrance door was generated.", feedback)

    def test_validation_delta_reports_fixed_still_failing_and_new(self) -> None:
        previous = {
            "checks": [
                {"category": "count", "label": "waiting: 0/1", "pass": False},
                {"category": "doors", "label": "Main entrance", "pass": False},
            ]
        }
        current = {
            "checks": [
                {"category": "count", "label": "waiting: 0/1", "pass": False},
            ]
        }
        delta = service.validation_delta(previous, current)
        self.assertIn("fixed: doors:Main entrance", delta)
        self.assertIn("still_failing: count:waiting (waiting: 0/1)", delta)
        self.assertIn("new_failures: none", delta)
        self.assertIn("treat as ground truth", delta)

    def test_repair_context_prepends_validator_delta_after_first_candidate(self) -> None:
        model_outputs = [
            {"code": f"code-{index}", "strategy": "repair", "assumptions": [], "model": "quality-model"}
            for index in range(1, 4)
        ]
        rejected_a_b = {
            "code": "code-1",
            "plan": {"rooms": []},
            "validation": {
                "score": 40,
                "checks": [
                    {"category": "doors", "label": "Main entrance door", "pass": False},
                    {"category": "count", "label": "waiting: 0/1", "pass": False},
                ],
                "issues": ["Main entrance door missing.", "waiting count is off."],
                "areas": [],
            },
        }
        rejected_b = {
            "code": "code-2",
            "plan": {"rooms": []},
            "validation": {
                "score": 50,
                "checks": [
                    {"category": "doors", "label": "Main entrance door", "pass": True},
                    {"category": "count", "label": "waiting: 0/1", "pass": False},
                ],
                "issues": ["waiting count is off."],
                "areas": [],
            },
        }
        accepted = {
            "code": "code-3",
            "plan": {"rooms": []},
            "validation": {"score": 95, "checks": [], "issues": [], "areas": []},
        }
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "quality-model"}),
            patch.object(service, "generate_gemini_code", side_effect=model_outputs) as generate_mock,
            patch.object(service, "execute_and_validate", side_effect=[rejected_a_b, rejected_b, accepted]),
        ):
            service.generate_plan(
                {
                    "action": "generate",
                    "program": self.samples["clinic-small"],
                    "mode": "auto",
                    "options": {"width": 64, "height": 40},
                }
            )
        second_context = generate_mock.call_args_list[1].args[5]
        third_context = generate_mock.call_args_list[2].args[5]
        # No prior executed candidate before the first repair, so no delta yet.
        self.assertNotIn("[validator delta", second_context)
        self.assertIn("[validator delta vs previous candidate", third_context)
        self.assertIn("fixed: doors:Main entrance door", third_context)

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
        self.assertEqual(result["stop_reason"], "attempts_exhausted")

    def test_budget_guard_stops_before_next_attempt_with_typed_reason(self) -> None:
        model_output = {
            "code": "attempt-1",
            "strategy": "s",
            "assumptions": [],
            "model": "quality-model",
            "usage": {"total_tokens": 1000, "estimated_cost_usd": 0.02},
        }
        rejected = {
            "code": "attempt-1",
            "plan": {"rooms": []},
            "validation": {
                "score": 40,
                "checks": [{"category": "doors", "label": "Main entrance door", "pass": False}],
                "areas": [],
                "issues": [],
            },
        }
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MAX_BUDGET_USD": "0.01"}),
            patch.object(service, "generate_gemini_code", return_value=model_output) as generate_mock,
            patch.object(service, "execute_and_validate", return_value=rejected),
        ):
            result = service.generate_plan(
                {
                    "program": self.samples["clinic-small"],
                    "mode": "auto",
                    "options": {"width": 64, "height": 40},
                }
            )
        # First attempt spends $0.02, exceeding the $0.01 ceiling, so attempt 2 never runs.
        self.assertEqual(generate_mock.call_count, 1)
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        self.assertFalse(result["accepted"])
        self.assertEqual(len(result["iterations"]), 1)
        self.assertEqual(result["usage_total"]["total_tokens"], 1000)


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
        self.assertIn("the program wins", with_image)

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


def make_test_seed() -> dict[str, object]:
    """40x24 seed: corridor band with a waiting room above and a toilet below."""
    width, height = 40, 24
    cells = [-2] * (width * height)

    def paint(x0: int, y0: int, x1: int, y1: int, value: int) -> None:
        for y in range(y0, y1):
            for x in range(x0, x1):
                cells[y * width + x] = value

    paint(2, 2, 38, 22, -1)      # footprint interior
    paint(2, 2, 38, 10, 0)       # waiting_0 (top band)
    paint(2, 10, 38, 14, 1)      # corridor_0 (middle band)
    paint(2, 14, 20, 22, 2)      # toilet_0 (bottom left)
    return {
        "width": width,
        "height": height,
        "meters_per_cell": 0.5,
        "cells": cells,
        "rooms": [
            {"id": "waiting_0", "type": "waiting"},
            {"id": "corridor_0", "type": "corridor"},
            {"id": "toilet_0", "type": "toilet"},
        ],
        "doors": [
            {
                "id": "door_waiting",
                "from_room": "waiting_0",
                "to_room": "corridor_0",
                "x": 18,
                "y": 10,
                "orientation": "horizontal",
                "width_cells": 2,
            },
            {
                "id": "door_toilet",
                "from_room": "toilet_0",
                "to_room": "corridor_0",
                "x": 10,
                "y": 14,
                "orientation": "horizontal",
                "width_cells": 1,
            },
        ],
    }


def make_refine_plan(
    room_rects: list[tuple[int, int, int, int, int]],
    width: int = 40,
    height: int = 24,
) -> dict[str, object]:
    """Executed-plan stand-in for the make_test_seed rooms: paint (index, x0, y0, x1, y1)
    rectangles onto a flat row-major cells grid so layout_fidelity can score it."""
    cells = [-1] * (width * height)
    for index, x0, y0, x1, y1 in room_rects:
        for y in range(y0, y1):
            for x in range(x0, x1):
                cells[y * width + x] = index
    return {
        "width": width,
        "height": height,
        "cells": cells,
        "rooms": [
            {"id": "waiting_0", "type": "waiting"},
            {"id": "corridor_0", "type": "corridor"},
            {"id": "toilet_0", "type": "toilet"},
        ],
    }


class SeedRefineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))

    def test_seed_translation_executes_and_reproduces_the_grid_exactly(self) -> None:
        seed = normalize_seed(make_test_seed())
        code = seed_to_code(seed)
        sanitized, plan = execute_pixel_code(code)
        self.assertIn("ROOM_DATA", sanitized)
        self.assertEqual(plan["width"], seed["width"])
        self.assertEqual(plan["height"], seed["height"])
        self.assertEqual(plan["meters_per_cell"], seed["meters_per_cell"])
        self.assertEqual([room["id"] for room in plan["rooms"]], ["waiting_0", "corridor_0", "toilet_0"])
        self.assertEqual(len(plan["doors"]), 2)
        expected_cells = [-1 if cell == -2 else cell for cell in seed["cells"]]
        self.assertEqual(plan["cells"], expected_cells)

    def test_normalize_seed_rejects_malformed_input(self) -> None:
        good = make_test_seed()
        with self.assertRaises(ValueError):
            normalize_seed({**good, "width": 4})
        with self.assertRaises(ValueError):
            normalize_seed({**good, "cells": good["cells"][:-1]})
        with self.assertRaises(ValueError):
            normalize_seed({**good, "rooms": good["rooms"][:1]})  # cells reference index 2
        with self.assertRaises(ValueError):
            normalize_seed({**good, "doors": [{**good["doors"][0], "orientation": "diagonal"}]})
        with self.assertRaises(ValueError):
            normalize_seed({**good, "meters_per_cell": 0})

    def test_refine_short_circuits_without_ai_when_seed_passes(self) -> None:
        passing = {
            "code": "seed-code",
            "plan": {"rooms": []},
            "validation": {"score": 95, "checks": [], "issues": [], "areas": []},
        }
        with (
            patch.object(service, "execute_and_validate", return_value=passing),
            patch.object(service, "generate_gemini_code") as generate_mock,
        ):
            outcome = service.refine_plan(
                {"program": self.samples["clinic-small"], "seed": make_test_seed()}
            )
        generate_mock.assert_not_called()
        self.assertTrue(outcome["accepted"])
        self.assertEqual(outcome["source"], "seed-translated")
        self.assertEqual(outcome["iterations"][0]["attempt"], 0)
        self.assertEqual(outcome["iterations"][0]["phase"], "seed")
        self.assertEqual(outcome["stop_reason"], "accepted")
        self.assertEqual(outcome["usage_total"]["total_tokens"], 0)

    def test_refine_repairs_failing_seed_with_seed_guidance(self) -> None:
        failing = {
            "code": "seed-code",
            "plan": {"rooms": []},
            "validation": {
                "score": 40,
                "checks": [{"category": "doors", "label": "waiting_0 has no door", "pass": False}],
                "issues": ["waiting_0 has no door."],
                "areas": [],
            },
        }
        passing = {
            "code": "repaired-code",
            "plan": {"rooms": []},
            "validation": {"score": 92, "checks": [], "issues": [], "areas": []},
        }
        model_output = {
            "code": "repaired-code",
            "strategy": "moved a door literal",
            "assumptions": [],
            "model": "quality-model",
        }
        with (
            # This test exercises repair guidance, not fidelity; keep the gate out of the way.
            patch.dict(os.environ, {
                "GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "quality-model", "REFINE_FIDELITY_MIN": "0",
            }),
            patch.object(service, "execute_and_validate", side_effect=[failing, passing]),
            patch.object(service, "generate_gemini_code", return_value=model_output) as generate_mock,
        ):
            outcome = service.refine_plan(
                {"program": self.samples["clinic-small"], "seed": make_test_seed()}
            )
        self.assertTrue(generate_mock.call_args.kwargs["seed_repair"])
        repair_context = generate_mock.call_args.args[5]
        self.assertIn("ROOM_DATA", repair_context)
        self.assertTrue(outcome["accepted"])
        self.assertEqual(outcome["source"], "seed-repaired")
        self.assertEqual(outcome["iterations"][0]["phase"], "seed")
        self.assertEqual(outcome["iterations"][0]["status"], "rejected")
        self.assertEqual(outcome["iterations"][1]["phase"], "fix")
        # The repair loop must inherit the seed's physical scale, not re-derive it.
        options = generate_mock.call_args.args[4]
        self.assertEqual(options["meters_per_cell"], 0.5)

    def test_refine_delta_uses_the_seed_validation_as_its_base(self) -> None:
        failing_seed = {
            "code": "seed-code",
            "plan": {"rooms": []},
            "validation": {
                "score": 40,
                "checks": [
                    {"category": "doors", "label": "Main entrance door", "pass": False},
                    {"category": "count", "label": "waiting: 0/1", "pass": False},
                ],
                "issues": ["waiting_0 has no door."],
                "areas": [],
            },
        }
        rejected_repair = {
            "code": "repair-1",
            "plan": {"rooms": []},
            "validation": {
                "score": 55,
                "checks": [
                    {"category": "doors", "label": "Main entrance door", "pass": True},
                    {"category": "count", "label": "waiting: 0/1", "pass": False},
                ],
                "issues": ["waiting count is off."],
                "areas": [],
            },
        }
        passing_repair = {
            "code": "repair-2",
            "plan": {"rooms": []},
            "validation": {"score": 92, "checks": [], "issues": [], "areas": []},
        }
        model_outputs = [
            {"code": "repair-1", "strategy": "first repair", "assumptions": [], "model": "quality-model"},
            {"code": "repair-2", "strategy": "second repair", "assumptions": [], "model": "quality-model"},
        ]
        with (
            # This test exercises the validator delta, not fidelity; keep the gate out of the way.
            patch.dict(os.environ, {
                "GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "quality-model", "REFINE_FIDELITY_MIN": "0",
            }),
            patch.object(
                service,
                "execute_and_validate",
                side_effect=[failing_seed, rejected_repair, passing_repair],
            ),
            patch.object(service, "generate_gemini_code", side_effect=model_outputs) as generate_mock,
        ):
            outcome = service.refine_plan(
                {"program": self.samples["clinic-small"], "seed": make_test_seed()}
            )
        # The delta on the second repair is computed against the seed translation (attempt 0).
        second_repair_context = generate_mock.call_args_list[1].args[5]
        self.assertIn("[validator delta vs previous candidate", second_repair_context)
        self.assertIn("fixed: doors:Main entrance door", second_repair_context)
        self.assertTrue(outcome["accepted"])

    def test_prompt_uses_seed_guidance_and_keeps_reference_note(self) -> None:
        program = normalize_program(self.samples["clinic-small"])
        options = {"width": 40, "height": 24, "meters_per_cell": 0.5}
        seed_prompt = build_prompt(program, "", options, None, with_seed_repair=True)
        self.assertIn("Repair it; do not redesign it", seed_prompt)
        self.assertIn("cannot be fixed by a local edit", seed_prompt)
        self.assertIn("minimal new geometry", seed_prompt)
        self.assertNotIn("TRANSCRIBE that drawing", seed_prompt)
        # Preservation is ordered hard rules now, not advice (user decision 2026-07-10).
        self.assertIn("never relocate a seed room", seed_prompt)
        self.assertIn("centerlines must not move", seed_prompt)
        both_prompt = build_prompt(
            program, "", options, None, with_reference_image=True, with_seed_repair=True
        )
        self.assertIn("Repair it; do not redesign it", both_prompt)
        self.assertIn("visual grounding", both_prompt)

    def test_dispatch_routes_refine_job_kind(self) -> None:
        with patch.object(service, "refine_plan", return_value={"accepted": True}) as refine_mock:
            service.dispatch_job({"job_kind": "refine", "program": {}, "seed": {}})
        refine_mock.assert_called_once()

    @staticmethod
    def _painted_plan(
        width: int,
        height: int,
        rooms: list[dict[str, str]],
        rects: list[tuple[int, int, int, int, int]],
    ) -> dict[str, object]:
        """Paint (index, x0, y0, x1, y1) rectangles onto a row-major cells grid."""
        cells = [-1] * (width * height)
        for index, x0, y0, x1, y1 in rects:
            for y in range(y0, y1):
                for x in range(x0, x1):
                    cells[y * width + x] = index
        return {"width": width, "height": height, "rooms": rooms, "cells": cells}

    def test_layout_fidelity_identical_layout_scores_100(self) -> None:
        rooms = [
            {"id": "waiting_0", "type": "waiting"},
            {"id": "exam_0", "type": "exam"},
            {"id": "exam_1", "type": "exam"},
        ]
        plan = self._painted_plan(
            32, 24, rooms, [(0, 0, 0, 24, 8), (1, 0, 8, 12, 24), (2, 12, 8, 24, 24)]
        )
        fidelity = service.layout_fidelity(plan, plan)
        self.assertEqual(fidelity["score"], 100.0)
        self.assertEqual(fidelity["unmatched"], [])

    def test_layout_fidelity_is_scale_invariant(self) -> None:
        # THE key property: the same relative arrangement, uniformly shrunk to half size and
        # translated elsewhere on the canvas, still reads as the same layout (a legitimate
        # in-place rescale to hit ft2 targets must not collapse the score).
        rooms = [
            {"id": "waiting_0", "type": "waiting"},
            {"id": "exam_0", "type": "exam"},
            {"id": "exam_1", "type": "exam"},
        ]
        # Seed fills a 24x24 bbox at the origin: a full-width band over two stacked rooms.
        seed = self._painted_plan(
            32, 24, rooms, [(0, 0, 0, 24, 8), (1, 0, 8, 12, 24), (2, 12, 8, 24, 24)]
        )
        # Candidate: the identical arrangement at half scale in a 12x12 bbox in the lower-right.
        candidate = self._painted_plan(
            32, 24, rooms, [(0, 16, 8, 28, 12), (1, 16, 12, 22, 20), (2, 22, 12, 28, 20)]
        )
        fidelity = service.layout_fidelity(seed, candidate)
        self.assertGreaterEqual(fidelity["score"], 90.0)
        self.assertEqual(fidelity["unmatched"], [])

    def test_layout_fidelity_penalizes_swapped_rooms(self) -> None:
        rooms = [
            {"id": "room_a", "type": "office"},
            {"id": "room_b", "type": "office"},
            {"id": "room_c", "type": "office"},
        ]
        # room_a left, room_c centre, room_b right.
        seed = self._painted_plan(
            12, 6, rooms, [(0, 0, 0, 3, 6), (2, 4, 0, 8, 6), (1, 9, 0, 12, 6)]
        )
        # room_a and room_b swap ends; room_c holds the centre.
        swapped = self._painted_plan(
            12, 6, rooms, [(0, 9, 0, 12, 6), (2, 4, 0, 8, 6), (1, 0, 0, 3, 6)]
        )
        fidelity = service.layout_fidelity(seed, swapped)
        self.assertLess(fidelity["score"], 60.0)
        movers = dict(fidelity["movers"])
        self.assertIn("room_a", movers)
        self.assertIn("room_b", movers)
        # The swapped rooms carry a nonzero displacement percent; the anchor stays at zero.
        self.assertGreater(movers["room_a"], 0)
        self.assertGreater(movers["room_b"], 0)
        self.assertEqual(movers["room_c"], 0.0)

    def test_refine_repair_feedback_carries_layout_fidelity(self) -> None:
        seed_rejected = {
            "code": "seed",
            "plan": {"rooms": []},
            "validation": {
                "score": 40,
                "checks": [{"category": "doors", "label": "waiting_0 has no door", "pass": False}],
                "issues": ["waiting_0 has no door."],
                "areas": [],
            },
        }
        # Validator-failing repair whose waiting room drifted down two rows from the seed.
        repair_drifted = {
            "code": "repair-1",
            "plan": make_refine_plan([(0, 2, 4, 38, 10), (1, 2, 10, 38, 14), (2, 2, 14, 20, 22)]),
            "validation": {
                "score": 40,
                "checks": [{"category": "doors", "label": "waiting_0 has no door", "pass": False}],
                "issues": ["waiting_0 has no door."],
                "areas": [],
            },
        }
        repair_faithful = {
            "code": "repair-2",
            "plan": make_refine_plan([(0, 2, 2, 38, 10), (1, 2, 10, 38, 14), (2, 2, 14, 20, 22)]),
            "validation": {"score": 92, "checks": [], "issues": [], "areas": []},
        }
        model_output = {"code": "repaired", "strategy": "s", "assumptions": [], "model": "quality-model"}
        with (
            patch.dict(os.environ, {
                "GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "quality-model", "REFINE_FIDELITY_MIN": "0",
            }),
            patch.object(service, "execute_and_validate",
                         side_effect=[seed_rejected, repair_drifted, repair_faithful]),
            patch.object(service, "generate_gemini_code",
                         side_effect=[model_output, model_output]) as generate_mock,
        ):
            outcome = service.refine_plan(
                {"program": self.samples["clinic-small"], "seed": make_test_seed()}
            )
        # The second repair attempt is told how far the first drifted from the traced seed.
        second_repair_context = generate_mock.call_args_list[1].args[5]
        self.assertIn("layout fidelity vs traced seed", second_repair_context)
        self.assertTrue(outcome["accepted"])
        self.assertEqual(outcome["seed_fidelity"], 100.0)

    def test_fidelity_gate_defaults_on_and_rejects_drifted_candidate(self) -> None:
        seed_rejected = {
            "code": "seed",
            "plan": {"rooms": []},
            "validation": {
                "score": 40,
                "checks": [{"category": "doors", "label": "waiting_0 has no door", "pass": False}],
                "issues": ["waiting_0 has no door."],
                "areas": [],
            },
        }
        # Passes the validator (score 92) but waiting and toilet swapped ends -> 15% fidelity,
        # far below the default REFINE_FIDELITY_MIN floor of 60.
        drifted_but_valid = {
            "code": "repair",
            "plan": make_refine_plan([(0, 2, 14, 20, 22), (1, 2, 10, 38, 14), (2, 2, 2, 38, 10)]),
            "validation": {"score": 92, "checks": [], "issues": [], "areas": []},
        }
        model_output = {"code": "repaired", "strategy": "s", "assumptions": [], "model": "quality-model"}
        base_env = {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "quality-model", "GEMINI_MAX_ATTEMPTS": "2"}
        # Default env (no REFINE_FIDELITY_MIN => floor 60): the drifted-but-valid candidate is
        # rejected and the loop keeps repairing, told it drifted too far from the traced seed.
        with (
            patch.dict(os.environ, base_env),
            patch.object(service, "execute_and_validate",
                         side_effect=[seed_rejected, drifted_but_valid, drifted_but_valid]),
            patch.object(service, "generate_gemini_code", return_value=model_output) as gated_mock,
        ):
            os.environ.pop("REFINE_FIDELITY_MIN", None)  # exercise the built-in default floor
            gated = service.refine_plan(
                {"program": self.samples["clinic-small"], "seed": make_test_seed()}
            )
        self.assertFalse(gated["accepted"])
        drift_context = gated_mock.call_args_list[1].args[5]
        self.assertIn("drifted too far", drift_context)
        self.assertIn("layout fidelity vs traced seed", drift_context)
        self.assertEqual(gated["seed_fidelity"], 15.0)
        # Gate explicitly disabled (0): the same candidate is accepted and carries its fidelity.
        with (
            patch.dict(os.environ, {**base_env, "REFINE_FIDELITY_MIN": "0"}),
            patch.object(service, "execute_and_validate",
                         side_effect=[seed_rejected, drifted_but_valid]),
            patch.object(service, "generate_gemini_code", return_value=model_output),
        ):
            ungated = service.refine_plan(
                {"program": self.samples["clinic-small"], "seed": make_test_seed()}
            )
        self.assertTrue(ungated["accepted"])
        self.assertEqual(ungated["seed_fidelity"], 15.0)


def _fake_urlopen_response(payload: dict[str, object]) -> object:
    """Minimal context-manager stand-in for urllib.request.urlopen's return value."""

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    return _Response()


def _function_call_payload(code: str, total_tokens: int) -> dict[str, object]:
    """Response shaped like the live probe: a model turn carrying one functionCall part."""
    return {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{
                    "functionCall": {"name": "execute_and_validate", "args": {"code": code}, "id": "call-1"},
                }],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": total_tokens, "candidatesTokenCount": 0, "totalTokenCount": total_tokens},
    }


def _structured_json_payload(code: str, total_tokens: int) -> dict[str, object]:
    """Response shaped like the live probe: a model turn carrying the final JSON text part."""
    text = json.dumps({"code": code, "strategy": "s", "assumptions": []})
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": total_tokens, "candidatesTokenCount": 0, "totalTokenCount": total_tokens},
    }


def _max_tokens_payload(total_tokens: int) -> dict[str, object]:
    """A candidate cut off at the output-token cap: the JSON is truncated mid-string and the
    finishReason is MAX_TOKENS (the 2026-07-09 24-room incident that read as a syntax error)."""
    return {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": '{"code": "def build():\\n    plan = {'}]},
            "finishReason": "MAX_TOKENS",
        }],
        "usageMetadata": {"promptTokenCount": total_tokens, "candidatesTokenCount": 0, "totalTokenCount": total_tokens},
    }


class AttemptToolLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))
        cls.program = normalize_program(samples["clinic-small"])
        cls.options = {"width": 64, "height": 40, "meters_per_cell": 0.25}

    def test_flag_off_builds_a_single_tools_free_request(self) -> None:
        response = _fake_urlopen_response(_structured_json_payload("off-code", 42))
        with (
            patch.dict(os.environ, {"GEMINI_ATTEMPT_TOOLS": "0"}),
            patch("urllib.request.urlopen", return_value=response) as urlopen_mock,
        ):
            result = gemini.generate_gemini_code(
                "test-key", "test-model", self.program, "", self.options
            )
        self.assertEqual(urlopen_mock.call_count, 1)
        body = json.loads(urlopen_mock.call_args.args[0].data.decode("utf-8"))
        self.assertNotIn("tools", body)
        self.assertNotIn("toolConfig", body)
        self.assertEqual(result["code"], "off-code")
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["usage"]["total_tokens"], 42)

    def test_tool_loop_self_checks_two_drafts_then_submits_final_json(self) -> None:
        recorded: list[str] = []

        def fake_executor(code: str) -> dict[str, object]:
            recorded.append(code)
            return {"score": 40, "rejected": True, "feedback": "f"}

        responses = [
            _fake_urlopen_response(_function_call_payload("draft-1", 100)),
            _fake_urlopen_response(_function_call_payload("draft-2", 200)),
            _fake_urlopen_response(_structured_json_payload("final-code", 300)),
        ]
        with (
            patch.dict(os.environ, {"GEMINI_ATTEMPT_TOOLS": "1"}),
            patch("urllib.request.urlopen", side_effect=responses) as urlopen_mock,
        ):
            result = gemini.generate_gemini_code(
                "test-key", "test-model", self.program, "", self.options,
                attempt_executor=fake_executor, max_tool_calls=2,
            )
        # The executor ran on each of the two drafts the model asked to test.
        self.assertEqual(recorded, ["draft-1", "draft-2"])
        self.assertEqual(urlopen_mock.call_count, 3)
        self.assertEqual(result["code"], "final-code")

        first_body = json.loads(urlopen_mock.call_args_list[0].args[0].data.decode("utf-8"))
        self.assertIn("tools", first_body)
        self.assertEqual(first_body["toolConfig"]["functionCallingConfig"]["mode"], "AUTO")

        third_body = json.loads(urlopen_mock.call_args_list[2].args[0].data.decode("utf-8"))
        self.assertNotIn("tools", third_body)  # final phase is tools-free
        self.assertIn("responseJsonSchema", third_body["generationConfig"])
        function_responses = [
            part["functionResponse"]["response"]
            for turn in third_body["contents"]
            for part in turn.get("parts", [])
            if "functionResponse" in part
        ]
        # Both real validator feedbacks are carried into the final request.
        self.assertEqual(
            function_responses.count({"score": 40, "rejected": True, "feedback": "f"}), 2
        )

    def test_tool_loop_sums_usage_across_all_calls_in_the_attempt(self) -> None:
        responses = [
            _fake_urlopen_response(_function_call_payload("draft-1", 100)),
            _fake_urlopen_response(_function_call_payload("draft-2", 200)),
            _fake_urlopen_response(_structured_json_payload("final-code", 300)),
        ]
        with (
            patch.dict(os.environ, {"GEMINI_ATTEMPT_TOOLS": "1"}),
            patch("urllib.request.urlopen", side_effect=responses),
        ):
            result = gemini.generate_gemini_code(
                "test-key", "test-model", self.program, "", self.options,
                attempt_executor=lambda code: {"score": 0, "rejected": True, "feedback": "x"},
                max_tool_calls=2,
            )
        self.assertEqual(result["usage"]["total_tokens"], 600)

    def test_max_tokens_retries_once_with_doubled_budget(self) -> None:
        # First call is cut off at the output cap; the retry with a doubled budget completes.
        responses = [
            _fake_urlopen_response(_max_tokens_payload(150)),
            _fake_urlopen_response(_structured_json_payload("recovered-code", 250)),
        ]
        with (
            patch.dict(os.environ, {"GEMINI_ATTEMPT_TOOLS": "0", "GEMINI_MAX_OUTPUT_TOKENS": "20000"}),
            patch("urllib.request.urlopen", side_effect=responses) as urlopen_mock,
        ):
            result = gemini.generate_gemini_code(
                "test-key", "test-model", self.program, "", self.options
            )
        self.assertEqual(urlopen_mock.call_count, 2)
        self.assertEqual(result["code"], "recovered-code")
        first_body = json.loads(urlopen_mock.call_args_list[0].args[0].data.decode("utf-8"))
        second_body = json.loads(urlopen_mock.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(
            second_body["generationConfig"]["maxOutputTokens"],
            first_body["generationConfig"]["maxOutputTokens"] * 2,
        )
        # Usage sums both HTTP calls, not just the one that produced the answer.
        self.assertEqual(result["usage"]["total_tokens"], 400)

    def test_double_truncation_raises_a_clear_error_not_a_syntax_error(self) -> None:
        responses = [
            _fake_urlopen_response(_max_tokens_payload(150)),
            _fake_urlopen_response(_max_tokens_payload(150)),
        ]
        with (
            patch.dict(os.environ, {"GEMINI_ATTEMPT_TOOLS": "0", "GEMINI_MAX_OUTPUT_TOKENS": "20000"}),
            patch("urllib.request.urlopen", side_effect=responses) as urlopen_mock,
        ):
            with self.assertRaises(ValueError) as caught:
                gemini.generate_gemini_code(
                    "test-key", "test-model", self.program, "", self.options
                )
        self.assertIn("truncated", str(caught.exception))
        self.assertEqual(urlopen_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
