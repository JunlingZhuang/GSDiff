# -*- coding: utf-8 -*-
"""Phase 1 end-to-end: understand -> pipeline -> API.
All LLM/VLM calls mocked — proves wiring, not models."""
import json

from fastapi.testclient import TestClient

from hfagent.api import app, get_client
from hfagent.graph import run_pipeline
from hfagent.tests.mock_vlm import MockVLM
from hfagent.tests.synth import simple_clinic
from hfagent.nodes.understand import build_understand_prompt, sanitize_program, understand

PROGRAM = {
    "building_type": "community clinic",
    "rooms": [
        {"type": "waiting", "count": 1},
        {"type": "exam_room", "count": 2},
        {"type": "corridor", "count": 1},
        {"type": "toilet", "count": 1},
    ],
    "adjacency": [["waiting", "corridor"], ["exam_room", "corridor"]],
}


# ── ⓪ understand ─────────────────────────────────────────────────────────────

class FakeTextLLM:
    def __init__(self, payload: dict):
        self.payload = payload

    def generate_json(self, prompt: str, schema=None) -> str:
        self.prompt, self.schema = prompt, schema
        return json.dumps(self.payload)


def test_understand_contract():
    llm = FakeTextLLM(PROGRAM)
    program = understand("生成一个小诊所", llm)
    assert program["rooms"][1] == {"type": "exam_room", "count": 2}
    assert llm.schema is not None
    assert "corridor" in llm.prompt and "生成一个小诊所" in llm.prompt


def test_sanitize_drops_hallucinated_types_and_clamps():
    dirty = {
        "building_type": "clinic",
        "rooms": [
            {"type": "operating_theatre", "count": 2},  # not in palette -> dropped
            {"type": "exam_room", "count": 99},          # clamped to 12
        ],
        "adjacency": [["exam_room", "operating_theatre"], ["exam_room", "exam_room"]],
    }
    clean = sanitize_program(dirty)
    assert [r["type"] for r in clean["rooms"]] == ["exam_room"]
    assert clean["rooms"][0]["count"] == 12
    assert clean["adjacency"] == []


def test_prompt_only_offers_palette_types():
    assert "operating_theatre" not in build_understand_prompt("x")


# ── ① + ② pipeline (mock text + mock image) ──────────────────────────────────

class FakeFullClient(MockVLM):
    """MockVLM (scripted images) + canned JSON text model."""

    def __init__(self, program: dict, script):
        super().__init__(script)
        self._program = program

    def generate_json(self, prompt: str, schema=None) -> str:
        return json.dumps(self._program)


def test_pipeline_end_to_end_with_mocks(tmp_path):
    client = FakeFullClient(PROGRAM, [simple_clinic()])
    result = run_pipeline("a small clinic", client, tmp_path)
    assert result["program"]["building_type"] == "community clinic"
    assert result["report"]["final_count_exact"]
    assert len(result["plan"]["rooms"]) == 5
    assert result["plan"]["walls"]                 # geometry layer populated
    assert "adjacency_graph" in result["plan"]      # topology layer present
    assert result["room_graph"]["rooms"]  # one node per room instance
    assert (tmp_path / "plan" / "graph.json").exists()
    assert (tmp_path / "plan" / "plan.json").exists()  # authoritative output


# ── ③ API ─────────────────────────────────────────────────────────────────────

class FakeAPIClient(FakeFullClient):
    """Minimal client for the API route — single-image script, no real Gemini."""
    pass


def test_api_generate_roundtrip(tmp_path):
    fake = FakeAPIClient(PROGRAM, [simple_clinic()])
    app.dependency_overrides[get_client] = lambda: fake
    try:
        c = TestClient(app)
        r = c.post("/api/agent/generate", json={"text": "一个小诊所"})
        assert r.status_code == 200
        body = r.json()
        assert body["program"]["building_type"] == "community clinic"
        assert c.get("/api/agent/health").json() == {"ok": True}
    finally:
        app.dependency_overrides.clear()


def test_api_rejects_unusable_brief():
    class BoomClient:
        image_model = text_model = "boom"
        def generate_json(self, *a, **kw):
            raise ValueError("program has no usable rooms")
        def generate_image(self, *a, **kw):
            raise ValueError("program has no usable rooms")

    app.dependency_overrides[get_client] = lambda: BoomClient()
    try:
        c = TestClient(app)
        assert c.post("/api/agent/generate", json={"text": "你好"}).status_code == 422
    finally:
        app.dependency_overrides.clear()
