# -*- coding: utf-8 -*-
"""Core generation pipeline: program -> generate -> parse -> verify -> fix -> room graph.

This is the heart of the agent: one function that takes a structured program and
a client, runs the VLM correction loop, repairs counts deterministically, reads the
room adjacency (doors) from the best realistic plan (or, in direct_colorblock, from
the program's own adjacency), and returns (report, plan_dict, room_graph_dict).
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from hfagent.tools.image_parser import cv_parse
from hfagent.tools.linework_tracer import trace_linework
from hfagent.tools.room_typing import type_rooms, typing_overlay
from hfagent.tools.floor_plan_generator import FloorPlanGenerator
from hfagent.tools.door_placer import place_doors
from hfagent.tools.plan_fixes import fix_room_counts
from hfagent.tools.rectify import building_aspect, rectify
from hfagent.tools.render_plan import render_plan
from hfagent.tools.structure_reader import read_structure
from hfagent.schema.plan import AdjacencyGraph, AdjEdge
from hfagent.schema.roomgraph import Door, RoomGraph, RoomNode

DEFAULT_CONFIG = Path(__file__).parent / "config.json"


def _log(name: str, message: str) -> None:
    """Emit progress immediately; model calls can otherwise look stalled."""
    print(f"[hfagent:{name}] {message}", flush=True)


def _log_prompt(name: str, stage: str, prompt: str) -> None:
    message = (
        f"\n[hfagent:{name}] PROMPT BEGIN [{stage}]\n"
        f"{prompt}\n"
        f"[hfagent:{name}] PROMPT END [{stage}]\n"
    )
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_message = message.encode(encoding, errors="replace").decode(encoding)
    print(safe_message, flush=True)


def _elapsed(started_at: float) -> str:
    return f"{time.perf_counter() - started_at:.1f}s"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    modes = cfg.get("modes", {})
    structure_mode = cfg.get("structure_mode", "colorblock")
    if structure_mode not in modes:
        raise SystemExit(
            f"config.json: structure_mode '{structure_mode}' is not one of the configured "
            f"modes {sorted(modes)}"
        )
    active = modes[structure_mode]
    return {
        "structure_mode": structure_mode,
        "max_correction_rounds": int(cfg.get("max_correction_rounds", 3)),
        "image_size": cfg.get("image_size", "1K"),
        "image_aspect": cfg.get("image_aspect", "16:9"),
        # per-mode settings of the ACTIVE mode (a --structure-mode override re-resolves via "modes")
        "text_model": active.get("text_model", ""),
        "image_model": active.get("image_model", ""),
        "boundary": active.get("boundary", ""),
        "modes": modes,
    }


@dataclass
class _BestRound:
    violation_count: int
    round_num: int
    plan: object
    png_path: Path
    real_png: bytes
    room_graph: object = None  # json: doors read alongside rooms; colorblock/direct: built later


def _count_violations(required_rooms: dict, actual_counts: Counter) -> list[str]:
    """Compare parsed room counts against the program; return human-readable mismatches."""
    violations = []
    for room_type, required in required_rooms.items():
        actual = actual_counts.get(room_type, 0)
        if actual != required:
            violations.append(f"{room_type}: drew {actual} block(s), required exactly {required}")
    for room_type, actual in actual_counts.items():
        if room_type not in required_rooms:
            violations.append(f"{room_type}: drew {actual} block(s), but this room type was NOT requested")
    return violations


def _room_graph_from_program(program: dict) -> RoomGraph:
    """Build the door graph straight from the program's declared adjacency.

    direct_colorblock has no realistic plan to read doors from, so connectivity comes
    from the program the LLM already produced: one node per distinct room type, one
    door edge per adjacency pair. place_doors matches by TYPE, so type-level nodes are
    enough — it hangs a door on every shared wall between two connected types. This is
    the documented "LLM connectivity + geometric placement" split, with the connectivity
    coming from the program rather than a second image read.
    """
    types = sorted({r["type"] for r in program["rooms"]})
    rooms = [RoomNode(id=t, type=t) for t in types]
    doors = [Door(room_a=a, room_b=b) for a, b in program.get("adjacency", [])]
    return RoomGraph(rooms=rooms, doors=doors)


def generate_plan(
    program: dict,
    client,
    out_dir: Path,
    name: str = "plan",
    max_rounds: int = 3,
    boundary: bytes | None = None,
    structure_mode: str = "colorblock",
) -> tuple[dict, dict, dict]:
    """Generate, parse, and repair a floor plan from a structured program.

    Runs a VLM correction loop (up to max_rounds), deterministically repairs any
    remaining count violations, then reads the room adjacency (doors) from the best
    realistic plan image.

    `boundary` (PNG bytes) is the optional second input entry: when given, the
    realflow plan's outer walls follow that footprint. Everything downstream of the
    realflow plan is identical with or without it.

    Returns (report, plan_dict, room_graph_dict).
    Writes intermediate files to out_dir/name/.
    """
    work_dir = out_dir / name
    work_dir.mkdir(parents=True, exist_ok=True)
    required_rooms = {r["type"]: r.get("count", 1) for r in program["rooms"]}
    if structure_mode == "linework":
        # geometry-only mode: rooms are untyped, so the per-type count violations
        # that drive the correction loop do not exist — exactly one round.
        max_rounds = 1
    pipeline_started = time.perf_counter()
    _log(
        name,
        f"start structure={structure_mode}, rounds={max_rounds}, "
        f"boundary={'yes' if boundary else 'no'}, output={work_dir}",
    )
    _log(name, f"required rooms: {required_rooms}")

    generator = FloorPlanGenerator(
        program,
        client,
        boundary=boundary,
        prompt_logger=lambda stage, prompt: _log_prompt(name, stage, prompt),
        drawing_mode="linework" if structure_mode == "linework" else "standard",
    )
    rounds: list[dict] = []
    best_round: _BestRound | None = None
    real_png: bytes | None = None

    for round_num in range(1, max_rounds + 1):
        png_path = work_dir / f"gemini_r{round_num}.png"
        round_started = time.perf_counter()
        _log(name, f"round {round_num}/{max_rounds}: begin")

        round_graph = None
        trace_diag = None
        round_image_path = png_path

        if structure_mode == "direct_colorblock":
            # ── one-pass: program text -> flat colour-block PNG ──────────────
            # No realistic plan: the model draws the colour mask directly, skipping
            # generate_real_plan + to_colorblock. cv_parse reads it like any mask.
            if round_num == 1:
                _log(name, f"round {round_num}: generating colour-block directly from the program")
                colorblock_png = generator.generate_colorblock_direct()
            else:
                # feed the previous colour-block back with the violations to redraw
                _log(
                    name,
                    f"round {round_num}: correcting colour-block for "
                    f"{len(rounds[-1]['violations'])} count violation(s)",
                )
                violation_feedback = (
                    "The colour-block floor plan above violates its room program:\n- "
                    + "\n- ".join(rounds[-1]["violations"])
                    + "\nRedraw it as a flat colour-block diagram, fixing ONLY these violations "
                    "(add the missing colour blocks, or merge/remove the extra ones). Keep the same "
                    "colours, black walls, footprint and layout; still no text, doors, arcs or furniture."
                )
                _log_prompt(name, f"colour-block correction round {round_num}", violation_feedback)
                colorblock_png = client.generate_image([colorblock_png, violation_feedback])
            png_path.write_bytes(colorblock_png)
            _log(name, f"round {round_num}: saved colour-block -> {png_path.name}; parsing with CV parser")
            parsed_plan = cv_parse(str(png_path))
        else:
            # ── real2color family: realistic plan first ──────────────────────
            if round_num == 1:
                _log(name, f"round {round_num}: generating realistic floor plan")
                real_png = generator.generate_real_plan()
            else:
                # apply violation feedback to the realistic plan, then re-convert
                _log(
                    name,
                    f"round {round_num}: correcting realistic plan for "
                    f"{len(rounds[-1]['violations'])} count violation(s)",
                )
                violation_feedback = (
                    "The architectural floor plan above violates its room program:\n- "
                    + "\n- ".join(rounds[-1]["violations"])
                    + "\nEdit the plan to fix ONLY these violations (add missing rooms, merge or "
                    "remove extra ones). Keep the same drawing style, footprint and circulation."
                )
                _log_prompt(name, f"real plan correction round {round_num}", violation_feedback)
                real_png = client.generate_image([real_png, violation_feedback])
            real_path = png_path.with_suffix(".real.png")
            real_path.write_bytes(real_png)
            _log(name, f"round {round_num}: saved realistic plan -> {real_path.name}")

            # ── realistic plan -> parsed Plan (interchangeable structure paths) ──
            if structure_mode == "json":
                # VLM reads the realistic plan into rooms (rectangles) + doors; rectify makes a
                # clean orthogonal Plan deterministically. The building's aspect ratio is measured
                # from the realistic image, not the model. No to_colorblock, no cv_parse.
                _log(name, f"round {round_num}: reading room geometry and doors as JSON")
                shapes, round_graph = read_structure(
                    real_png,
                    client,
                    program,
                    prompt_logger=lambda stage, prompt: _log_prompt(name, stage, prompt),
                )
                _log(name, f"round {round_num}: rectifying {len(shapes)} room shape(s)")
                parsed_plan = rectify(shapes, building_aspect(real_png))
                structure_json = {
                    "rooms": [{"id": rs.id, "type": rs.type, "rects": [list(r) for r in rs.rects]}
                              for rs in shapes],
                    "doors": [{"room_a": d.room_a, "room_b": d.room_b} for d in round_graph.doors],
                }
                (work_dir / f"structure_r{round_num}.json").write_text(
                    json.dumps(structure_json, indent=2), encoding="utf-8"
                )
            elif structure_mode == "linework":
                # Faithful-trace backend (tools/linework_tracer): walls are traced
                # exactly as drawn, a gap becomes a door only when a swing arc
                # confirms it, and the closed wall graph is polygonized into rooms
                # (ids r1..rN, px units). The trace itself is geometry-only; the
                # separate room-typing step then reads the label drawn inside each
                # room (local OCR + program-vocabulary match, tools/room_typing)
                # so the plan comes back with program types. The trace's RoomGraph
                # names the two plan room ids flanking each confirmed door
                # ("exterior" when a side is open ground), so the shared downstream
                # applies unchanged: place_doors hangs each edge on the wall that
                # exact room pair shares (exact_connected) and plan_to_walls derives
                # the wall list from the traced polygons.
                _log(name, f"round {round_num}: tracing walls, door arcs and room polygons")
                trace = trace_linework(real_png)
                parsed_plan, round_graph = trace.plan, trace.room_graph
                trace_diag = trace.diagnostics
                round_image_path = real_path
                for filename, payload in trace.artifacts.items():
                    (work_dir / filename).write_bytes(payload)
                _log(
                    name,
                    f"round {round_num}: traced {trace_diag['rooms_closed']} room(s), "
                    f"{trace_diag['doors_detected']} door(s); wrote {', '.join(trace.artifacts)}",
                )
                _log(name, f"round {round_num}: reading room labels for program types")
                guesses = type_rooms(real_png, parsed_plan, program)
                for room in parsed_plan.rooms:
                    guess = guesses.get(room.id)
                    if guess is not None:
                        room.type, room.name = guess.type, guess.name
                typed_by_id = {room.id: room.type for room in parsed_plan.rooms}
                for node in round_graph.rooms:
                    node.type = typed_by_id.get(node.id, node.type)
                (work_dir / "typing_overlay.png").write_bytes(
                    typing_overlay(real_png, parsed_plan, guesses))
                (work_dir / "typing.json").write_text(json.dumps(
                    {rid: dict(type=g.type, instance=g.instance,
                               score=round(g.score, 3), text=g.text)
                     for rid, g in sorted(guesses.items())}, indent=1), encoding="utf-8")
                trace_diag["room_typing"] = dict(rooms=len(parsed_plan.rooms),
                                                 typed=len(guesses))
                _log(name, f"round {round_num}: typed {len(guesses)}/"
                           f"{len(parsed_plan.rooms)} room(s) from drawn labels")
            else:
                _log(name, f"round {round_num}: converting realistic plan to colour-block mask")
                png_path.write_bytes(generator.to_colorblock(real_png))
                _log(name, f"round {round_num}: parsing colour-block mask with CV parser")
                parsed_plan = cv_parse(str(png_path))

        # ── verify ───────────────────────────────────────────────────────────
        # linework rooms are typed from drawn labels AFTER the trace, but label
        # coverage is best-effort (tiny/rotated text) — the single-round check
        # stays total rooms traced vs the program's total.
        actual_counts = Counter(r.type for r in parsed_plan.rooms)
        if structure_mode == "linework":
            required_total = sum(required_rooms.values())
            traced_total = len(parsed_plan.rooms)
            violations = [] if traced_total == required_total else [
                f"traced {traced_total} room(s); the program totals {required_total}"
            ]
        else:
            violations = _count_violations(required_rooms, actual_counts)
        round_report = {
            "round": round_num,
            "actual_rooms": dict(actual_counts),
            "violations": violations,
        }
        if trace_diag is not None:
            round_report["rooms_traced"] = len(parsed_plan.rooms)
            round_report["rooms_required_total"] = sum(required_rooms.values())
            round_report["trace"] = trace_diag
        rounds.append(round_report)
        _log(
            name,
            f"round {round_num}: parsed {len(parsed_plan.rooms)} rooms, "
            f"violations={len(violations)}, elapsed={_elapsed(round_started)}",
        )
        _log(name, f"round {round_num}: room counts {dict(actual_counts)}")

        if best_round is None or len(violations) < best_round.violation_count:
            best_round = _BestRound(len(violations), round_num, parsed_plan, round_image_path, real_png, round_graph)
            _log(name, f"round {round_num}: selected as current best")

        if not violations:
            _log(name, f"round {round_num}: room counts match; stopping correction loop")
            break  # all room counts match — skip remaining rounds

    assert best_round is not None
    _log(name, f"using round {best_round.round_num} with {best_round.violation_count} violation(s)")
    (work_dir / "parsed.json").write_text(best_round.plan.model_dump_json(indent=2), encoding="utf-8")

    _log(name, "running deterministic room-count repair")
    if structure_mode == "linework":
        # the traced geometry IS the result: label-read types are best-effort, so
        # splitting/relabelling rooms to force per-type counts would fake geometry.
        fixed_plan = best_round.plan.model_copy(deep=True)
        count_fix = {
            "ops": [],
            "fixed": len(fixed_plan.rooms) == sum(required_rooms.values()),
            "fixed_rooms": dict(Counter(room.type for room in fixed_plan.rooms)),
            "skipped": "linework geometry is authoritative; per-type count repair does not apply",
        }
    else:
        fixed_plan, count_fix = fix_room_counts(best_round.plan, required_rooms)
    (work_dir / "fixed.json").write_text(fixed_plan.model_dump_json(indent=2), encoding="utf-8")
    if structure_mode != "linework":  # the trace already wrote recon.png / recon_post.png
        render_plan(fixed_plan, px_per_mm=1.0).save(work_dir / "recon.png")
    _log(name, f"count repair fixed={count_fix['fixed']}")

    # ── room adjacency (doors) ───────────────────────────────────────────────
    # json read the doors alongside the rooms; linework detected them geometrically
    # (arc-confirmed, edges naming exact plan room ids); direct_colorblock has no
    # realistic plan so it takes connectivity straight from the program; colorblock reads
    # it from the best realistic plan now. Either way door_placer hangs each door on the
    # real wall the two connected rooms physically share.
    if structure_mode in ("json", "linework"):
        _log(name, f"using door graph from the best {structure_mode} structure round")
        room_graph = best_round.room_graph
    elif structure_mode == "direct_colorblock":
        _log(name, "building door graph from the program's declared adjacency")
        room_graph = _room_graph_from_program(program)
    else:
        _log(name, "extracting room-door adjacency from the best realistic plan")
        room_graph = generator.extract_room_adjacency(best_round.real_png)
    (work_dir / "graph.json").write_text(room_graph.model_dump_json(indent=2), encoding="utf-8")
    _log(name, f"door graph: rooms={len(room_graph.rooms)}, logical doors={len(room_graph.doors)}")

    # assemble the authoritative plan (docs/agent/02-data-model.md §3.2):
    # rooms + complete walls + doors-on-walls + adjacency_graph (edges link via door id)
    plan, door_pairs = place_doors(fixed_plan, room_graph)
    plan.building_type = program.get("building_type", plan.building_type)
    plan.adjacency_graph = AdjacencyGraph(
        nodes=[r.id for r in plan.rooms],
        edges=[AdjEdge(from_=a, to=b, type="door", via=did) for did, a, b in door_pairs],
    )
    (work_dir / "plan.json").write_text(plan.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    render_plan(plan, px_per_mm=1.0).save(work_dir / "recon_with_door.png")
    _log(name, f"placed doors={len(plan.doors)}, walls={len(plan.walls)}; rendered recon_with_door.png")

    vlm_converged = not rounds[best_round.round_num - 1]["violations"]
    report = {
        "program": name,
        "image_model": client.image_model,
        "structure_mode": structure_mode,
        "required_rooms": required_rooms,
        "rounds": rounds,
        "best_round": best_round.round_num,
        "best_image": best_round.png_path.name,
        "actual_rooms": rounds[best_round.round_num - 1]["actual_rooms"],
        "room_count_exact": vlm_converged,
        "converged_in": best_round.round_num if vlm_converged else None,
        "count_fix": count_fix,
        "final_count_exact": vlm_converged or count_fix["fixed"],
        "room_graph": {
            "rooms": len(room_graph.rooms),
            "doors": len(room_graph.doors),       # logical edges (LLM read or trace)
            "placed_doors": len(plan.doors),       # doors hung on real walls
            "walls": len(plan.walls),              # complete geometry-layer wall list
        },
    }
    (work_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(name, f"complete in {_elapsed(pipeline_started)} -> {work_dir}")

    return report, plan.model_dump(by_alias=True), room_graph.model_dump()
