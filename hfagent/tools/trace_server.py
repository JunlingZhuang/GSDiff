# -*- coding: utf-8 -*-
"""hfagent trace HTTP service: turn a floor-plan drawing into a neutral seed grid.

Run with the GSDiff root venv:

    cd d:/Github/GSDiff && .venv/Scripts/python -m hfagent.tools.trace_server

Endpoints (bind 127.0.0.1, port env ``HFAGENT_TRACE_PORT`` default 8801):

    POST /trace   {"image": {"mime": "...", "data": "<base64>"}, "program": "<name>"}
        -> 200 {"seed": <rasterize_plan dict incl. meta>,
                "diagnostics": {"rooms", "doors", "typed", "meters_per_pixel"},
                "artifacts": {"linework": {"mime": "image/png", "data": "<base64>"}}}
    POST /draw    {"program": "<name>" | <program object>, "count": 1..4 (default 3)}
        -> 200 {"images": [{"mime", "data", "rooms_expected", "rooms_found"}],
                "count": N, "source": "hfagent-linework"}
        Draws each candidate with hfagent's OWN authoritative linework pipeline
        (the user-optimized prompt rules, EXACTLY one image-model call per
        candidate — no correction or redraw rounds: asking an image model to fix
        a drawing makes it lazily patch in filler rooms; discrepancy repair
        belongs to the downstream code agent), then VERIFIES deterministically by
        tracing the drawing back and counting enclosed rooms. The count is
        reported, never "fixed" here.
    GET  /health  -> 200 {"ok": true, "draw": true, "programs": [names]}

This service imports NO pixel-plan-ai code. The only coupling is that the seed
JSON matches what pixel-plan-ai's /api/refine accepts; the studio frontend
reaches this service through a Next.js rewrite so trace mode degrades gracefully
when it is offline. The chain (trace_linework -> type_rooms -> rasterize_plan)
mirrors ``hfagent.tools.pixelplan_client.main``. /draw makes hfagent the
authoritative drawer of floor-plan candidates; pixel-plan-ai's /api/images is the
fallback the studio uses only when this service is offline or the draw fails.
"""
from __future__ import annotations

import base64
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from hfagent.floor_plan_generate import load_config
from hfagent.llm import GeminiClient, _sniff_mime
from hfagent.tools.floor_plan_generator import FloorPlanGenerator
from hfagent.tools.linework_tracer import trace_linework
from hfagent.tools.plan_raster import rasterize_plan
from hfagent.tools.room_typing import type_rooms

_HFAGENT_DIR = Path(__file__).resolve().parent.parent
PROGRAMS_PATH = _HFAGENT_DIR / "programs.json"
PROGRAMS: dict = json.loads(PROGRAMS_PATH.read_text(encoding="utf-8"))
BODY_LIMIT = 10_000_000


def _pick_linework_artifact(artifacts: dict[str, bytes]) -> bytes | None:
    """The single most plan-like trace artifact: ``rooms_colorful.png`` if it
    exists, else the first artifact whose name mentions linework/walls, else the
    first artifact of any kind. Inspects keys defensively (they may change)."""
    if "rooms_colorful.png" in artifacts:
        return artifacts["rooms_colorful.png"]
    for name, payload in artifacts.items():
        lowered = name.lower()
        if "linework" in lowered or "walls" in lowered:
            return payload
    for payload in artifacts.values():
        return payload
    return None


def trace_drawing(png: bytes, program_name: str) -> dict:
    """Run trace_linework -> type_rooms -> rasterize_plan for one drawing."""
    if program_name not in PROGRAMS:
        raise ValueError(f"unknown program {program_name!r}; choose from {sorted(PROGRAMS)}")
    program = PROGRAMS[program_name]

    trace = trace_linework(png)
    plan = trace.plan
    typing = type_rooms(png, plan, program)
    typed = 0
    for room in plan.rooms:
        guess = typing.get(room.id)
        if guess is not None:
            room.type, room.name = guess.type, guess.name
            typed += 1
    meters_per_pixel = getattr(typing, "meters_per_pixel", None)

    seed = rasterize_plan(
        plan,
        trace.room_graph,
        meters_per_pixel=meters_per_pixel,
        wall_width_px=trace.diagnostics.get("wall_width"),
    )

    result: dict = {
        "seed": seed,
        "diagnostics": {
            "rooms": len(seed["rooms"]),
            "doors": len(seed["doors"]),
            "typed": typed,
            "meters_per_pixel": meters_per_pixel,
        },
        "artifacts": {},
    }
    linework = _pick_linework_artifact(trace.artifacts)
    if linework:
        result["artifacts"]["linework"] = {
            "mime": "image/png",
            "data": base64.b64encode(linework).decode("ascii"),
        }
    return result


# ── authoritative candidate drawing (hfagent linework pipeline) ───────────────
# /draw draws floor-plan candidates with hfagent's OWN, user-optimized drawing
# path — build_linework_prompt via FloorPlanGenerator, one image-model call each —
# and verifies every candidate deterministically by tracing it back and counting
# closed rooms. This is why the coupling stays HTTP-only: no prompt text is
# re-implemented here, hfagent's request code is reused as-is.

_DRAW_STRUCTURE_MODE = "linework"
_client_lock = threading.Lock()
_draw_client_singleton: GeminiClient | None = None


def _draw_client() -> GeminiClient:
    """The Gemini client pinned to the linework mode's models, built once and cached.

    Constructed exactly as ``hfagent.evaluate`` builds it — config.json's
    ``modes.linework`` supplies the (auto-resolving) model ids, and GeminiClient does
    its own env loading (.env.local GEMINI / GEMINI_API_KEY / GOOGLE_API_KEY). Model
    discovery runs a single ``models.list()`` on first use; the lock guards the
    threaded /draw fan-out from building it more than once.
    """
    global _draw_client_singleton
    with _client_lock:
        if _draw_client_singleton is None:
            cfg = load_config()
            mode_cfg = cfg["modes"][_DRAW_STRUCTURE_MODE]
            _draw_client_singleton = GeminiClient(
                text_model=(mode_cfg.get("text_model") or None),
                image_model=(mode_cfg.get("image_model") or None),
                image_size=cfg["image_size"],
                image_aspect=cfg["image_aspect"],
            )
        return _draw_client_singleton


def _resolve_program(program: object) -> dict:
    """Accept a program NAME in programs.json or a full program OBJECT."""
    if isinstance(program, str):
        if program not in PROGRAMS:
            raise ValueError(f"unknown program {program!r}; choose from {sorted(PROGRAMS)}")
        return PROGRAMS[program]
    if isinstance(program, dict):
        if not isinstance(program.get("rooms"), list) or not program["rooms"]:
            raise ValueError("program object must include a non-empty 'rooms' list")
        if not isinstance(program.get("building_type"), str) or not program["building_type"]:
            raise ValueError("program object must include a 'building_type' string")
        return program
    raise ValueError("program must be a program name (string) or a program object")


def _expected_rooms(program: dict) -> int:
    """The room count the drawing should enclose = the program's TOTAL instance
    count, corridors INCLUDED.

    The linework tracer polygonizes a corridor as a closed room like any other, and
    hfagent's own linework verify (floor_plan_generate.generate_plan) compares
    ``len(traced rooms)`` against ``sum(required_rooms.values())`` with corridor
    instances in that sum. We match that semantics exactly — corridors count.
    """
    return sum(int(r.get("count", 1)) for r in program["rooms"])


def _draw_one(program: dict) -> bytes:
    """One authoritative linework plan PNG/JPEG via hfagent's own drawing path.

    ``drawing_mode="linework"`` makes generate_real_plan build the linework prompt
    (build_linework_prompt) and issue exactly one image-model call — the same single
    round the linework structure mode uses by design.
    """
    generator = FloorPlanGenerator(program, _draw_client(), drawing_mode="linework")
    return generator.generate_real_plan()


def _rooms_found(png: bytes) -> int:
    """Deterministic verification: trace the drawing, count enclosed rooms."""
    return int(trace_linework(png).diagnostics["rooms_closed"])


def _draw_verified_candidate(program: dict, expected: int) -> dict:
    """Draw one candidate with exactly ONE image-model call and report its traced
    room count. No redraw/correction round (user decision 2026-07-09: image-model
    correction lazily patches in filler rooms; count mismatches are surfaced to
    the picker and repaired by the downstream code agent)."""
    png = _draw_one(program)
    found = _rooms_found(png)
    # The image model's bytes may be PNG or JPEG (auto-resolved model); sniff the
    # real mime instead of assuming PNG (hfagent convention: never hardcode it).
    return {
        "mime": _sniff_mime(png) or "image/png",
        "data": base64.b64encode(png).decode("ascii"),
        "rooms_expected": expected,
        "rooms_found": found,
    }


def draw_candidates(program_ref: object, count: object = 3) -> dict:
    """Draw ``count`` (1..4, clamped) verified linework candidates concurrently."""
    program = _resolve_program(program_ref)
    expected = _expected_rooms(program)
    n = max(1, min(4, int(count)))
    with ThreadPoolExecutor(max_workers=n) as pool:
        images = list(pool.map(lambda _i: _draw_verified_candidate(program, expected), range(n)))
    return {"images": images, "count": len(images), "source": "hfagent-linework"}


class TraceHandler(BaseHTTPRequestHandler):
    server_version = "HfagentTrace/0.1"

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            # "draw": true lets clients feature-detect the authoritative drawer.
            self._send_json(HTTPStatus.OK, {"ok": True, "draw": True, "programs": sorted(PROGRAMS)})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > BODY_LIMIT:
            raise ValueError("Request body exceeds 10 MB.")
        body = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        return body

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/trace":
            self._handle_trace()
        elif path == "/draw":
            self._handle_draw()
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def _handle_trace(self) -> None:
        try:
            body = self._read_json_body()
            image = body.get("image")
            if not isinstance(image, dict) or not image.get("data"):
                raise ValueError("Request must include image.data (base64 PNG).")
            program_name = body.get("program")
            if not isinstance(program_name, str) or not program_name:
                raise ValueError("Request must include a program name.")
            png = base64.b64decode(image["data"])
            self._send_json(HTTPStatus.OK, trace_drawing(png, program_name))
        except Exception as error:  # noqa: BLE001 - report every failure as 400
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _handle_draw(self) -> None:
        try:
            body = self._read_json_body()
            program = body.get("program")
            if program is None:
                raise ValueError("Request must include a program (name or object).")
            self._send_json(HTTPStatus.OK, draw_candidates(program, body.get("count", 3)))
        except Exception as error:  # noqa: BLE001 - report every failure as 400
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def main() -> None:
    port = int(os.environ.get("HFAGENT_TRACE_PORT", "8801"))
    server = ThreadingHTTPServer(("127.0.0.1", port), TraceHandler)
    print(f"hfagent trace service at http://127.0.0.1:{port}")
    print(f"programs: {', '.join(sorted(PROGRAMS))}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
