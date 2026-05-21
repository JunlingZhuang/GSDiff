"""Endpoint test on a minimal app (avoids loading ML models in lifespan)."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.procedural import router as procedural_router

app = FastAPI()
app.include_router(procedural_router)
client = TestClient(app)


def test_procedural_generates_plan_from_default_boundary():
    body = {
        "graph": {
            "nodes": [
                {"id": 0, "room_type": "Livingroom"},
                {"id": 1, "room_type": "Kitchen"},
                {"id": 2, "room_type": "Bedroom"},
                {"id": 3, "room_type": "Corridor"},
            ],
            "edges": [
                {"source": 0, "target": 3, "connectivity": "door"},
                {"source": 1, "target": 3, "connectivity": "door"},
                {"source": 2, "target": 3, "connectivity": "door"},
            ],
        },
        # simple rectangular boundary in metres
        "boundary": [[0, 0], [10, 0], [10, 8], [0, 8]],
        "seed": 0,
    }
    resp = client.post("/api/generate/procedural", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert {"walls", "openings", "rooms", "grid", "unit"} <= set(data)
    assert len(data["rooms"]) == 4
    assert len(data["walls"]) > 0
