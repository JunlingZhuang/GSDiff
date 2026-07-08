# -*- coding: utf-8 -*-
"""hfagent trace HTTP service: turn a floor-plan drawing into a neutral seed grid.

Run with the GSDiff root venv:

    cd d:/Github/GSDiff && .venv/Scripts/python -m hfagent.tools.trace_server

Endpoints (bind 127.0.0.1, port env ``HFAGENT_TRACE_PORT`` default 8801):

    POST /trace   {"image": {"mime": "...", "data": "<base64>"}, "program": "<name>"}
        -> 200 {"seed": <rasterize_plan dict incl. meta>,
                "diagnostics": {"rooms", "doors", "typed", "meters_per_pixel"},
                "artifacts": {"linework": {"mime": "image/png", "data": "<base64>"}}}
    GET  /health  -> 200 {"ok": true, "programs": [names]}

This service imports NO pixel-plan-ai code. The only coupling is that the seed
JSON matches what pixel-plan-ai's /api/refine accepts; the studio frontend
reaches this service through a Next.js rewrite so trace mode degrades gracefully
when it is offline. The chain (trace_linework -> type_rooms -> rasterize_plan)
mirrors ``hfagent.tools.pixelplan_client.main``.
"""
from __future__ import annotations

import base64
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

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
            self._send_json(HTTPStatus.OK, {"ok": True, "programs": sorted(PROGRAMS)})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/trace":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > BODY_LIMIT:
                raise ValueError("Request body exceeds 10 MB.")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object.")
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
