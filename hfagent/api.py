# -*- coding: utf-8 -*-
"""FastAPI entry point for hfagent.

Run:
    hfagent/.venv/Scripts/python.exe -m uvicorn hfagent.api:app --port 8100

Sessions persist under hfagent/out/api/<timestamp>/ (file storage for Phase 1;
a database arrives in Phase 5).
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hfagent.llm import GeminiClient
from hfagent.pipeline import generate_plan, load_config
from hfagent.understand import understand

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
        report, plan, wallgraph = generate_plan(
            program, client, session,
            max_rounds=_cfg["max_correction_rounds"],
            pipeline=_cfg["pipeline"],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "id": next(_counter),
        "program": program,
        "report": report,
        "plan": plan,
        "wallgraph": wallgraph,
    }
