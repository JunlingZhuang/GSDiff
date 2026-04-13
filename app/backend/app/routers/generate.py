"""Generation API endpoints."""

import base64
import io

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.inference import generate_unconstrained, generate_topology, generate_boundary

router = APIRouter(prefix="/api/generate", tags=["generate"])


class TopologyRequest(BaseModel):
    rooms: list[int]
    adjacency: list[list[int]]


class BoundaryRequest(BaseModel):
    boundary_image: str  # base64 encoded 256x256 PNG (black/white boundary outline)


class GenerateResponse(BaseModel):
    image: str
    rooms: int | None = None


def _pil_to_data_uri(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


@router.post("/unconstrained", response_model=GenerateResponse)
def unconstrained():
    try:
        img, room_count = generate_unconstrained()
        return GenerateResponse(image=_pil_to_data_uri(img), rooms=room_count)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/topology", response_model=GenerateResponse)
def topology(req: TopologyRequest):
    if len(req.rooms) < 4 or len(req.rooms) > 8:
        raise HTTPException(400, "Room count must be 4-8")
    if len(req.adjacency) != len(req.rooms):
        raise HTTPException(400, "Adjacency matrix size must match room count")
    try:
        img = generate_topology(req.rooms, req.adjacency)
        return GenerateResponse(image=_pil_to_data_uri(img))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/boundary", response_model=GenerateResponse)
def boundary(req: BoundaryRequest):
    try:
        img = generate_boundary(req.boundary_image)
        return GenerateResponse(image=_pil_to_data_uri(img))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
