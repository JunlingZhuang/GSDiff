# -*- coding: utf-8 -*-
"""FastAPI entry point for hfagent.

Run:
    hfagent/.venv/Scripts/python.exe -m uvicorn hfagent.api:app --port 8100

Sessions persist under hfagent/out/api/<timestamp>/ (file storage for Phase 1;
a database arrives in Phase 5).
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hfagent.llm import GeminiClient
from hfagent.floor_plan_generate import generate_plan, load_config
from hfagent.nodes.understand import sanitize_program, understand

OUT_ROOT = Path(__file__).parent / "out" / "api"
_counter = itertools.count(1)
_cfg = load_config()

app = FastAPI(title="hfagent", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000)


def get_client() -> GeminiClient:
    return GeminiClient()


@app.get("/api/agent/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/agent/generate")
def generate(req: GenerateRequest, client: GeminiClient = Depends(get_client)) -> dict:
    session = OUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    try:
        program = understand(req.text, client)
        report, plan, room_graph = generate_plan(
            program, client, session,
            max_rounds=_cfg["max_correction_rounds"],
            generation_mode=_cfg["generation_mode"],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "id": next(_counter),
        "program": program,
        "report": report,
        "plan": plan,
        "room_graph": room_graph,
    }


@app.post("/api/agent/generate-from-boundary")
def generate_from_boundary(
    program: str = Form(...),          # JSON-encoded structured room program
    boundary: UploadFile = File(...),  # building outline image
    client: GeminiClient = Depends(get_client),
) -> dict:
    """Second entry: structured program + building outline -> plan. Only pass-1
    differs from /generate; everything downstream of the realflow plan is shared."""
    session = OUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    try:
        prog = sanitize_program(json.loads(program))
        report, plan, room_graph = generate_plan(
            prog, client, session,
            max_rounds=_cfg["max_correction_rounds"],
            generation_mode=_cfg["generation_mode"],
            boundary=boundary.file.read(),
        )
    except ValueError as e:  # bad JSON, no usable rooms, etc.
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "id": next(_counter),
        "program": prog,
        "report": report,
        "plan": plan,
        "room_graph": room_graph,
    }
