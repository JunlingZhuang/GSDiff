"""Generation API endpoints."""

import base64
import io
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.graph_generation import graph_generator
from app.services.inference import generate_unconstrained, generate_topology, generate_boundary

logger = logging.getLogger("app.generate")

router = APIRouter(prefix="/api/generate", tags=["generate"])


class TopologyRequest(BaseModel):
    rooms: list[int]
    adjacency: list[list[int]]


class GraphRequest(BaseModel):
    dataset: str = "rplan"
    model: str | None = None
    num_samples: int = 1
    num_nodes: int | None = None
    seed: int | None = None


class BoundaryRequest(BaseModel):
    boundary_image: str  # base64 encoded 256x256 PNG (black/white boundary outline)


class GraphNode(BaseModel):
    id: int
    attr: int
    room_type: str


class GraphEdge(BaseModel):
    source: int
    target: int
    edge_type: int
    edge_label: str


class GeneratedGraph(BaseModel):
    num_nodes: int
    num_edges: int
    rooms: list[int]
    adjacency: list[list[int]]
    edge_types: list[list[int]]
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class GenerateResponse(BaseModel):
    image: str
    rooms: int | None = None


class GraphResponse(BaseModel):
    dataset: str
    checkpoint: str
    inference_seconds: float
    graphs: list[GeneratedGraph]
    room_types: list[str]
    edge_types: list[str]


class NextNodeRequest(BaseModel):
    dataset: str = "msd_wall"
    model: str = "msd_wall_next_node"
    graph: GeneratedGraph
    num_candidates: int = 1
    seed: int | None = None


class GraphCompletionRequest(BaseModel):
    dataset: str = "msd_wall"
    model: str = "msd_wall_full_completion_v2"
    graph: GeneratedGraph
    target_num_nodes: int = 30
    num_candidates: int = 8
    seed: int | None = None


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


@router.post("/graph", response_model=GraphResponse)
def graph(req: GraphRequest):
    if req.num_samples < 1 or req.num_samples > 8:
        raise HTTPException(400, "num_samples must be 1-8")
    if req.num_nodes is not None:
        max_nodes = graph_generator.max_num_nodes(req.dataset)
        if req.num_nodes < 1 or req.num_nodes > max_nodes:
            raise HTTPException(400, f"num_nodes must be 1-{max_nodes} for {req.dataset}")
    try:
        return graph_generator.generate(
            dataset=req.dataset,
            model=req.model,
            num_samples=req.num_samples,
            num_nodes=req.num_nodes,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("graph sample failed (dataset=%s model=%s)", req.dataset, req.model)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/graph/next-node", response_model=GraphResponse)
def graph_next_node(req: NextNodeRequest):
    if req.num_candidates < 1 or req.num_candidates > 8:
        raise HTTPException(400, "num_candidates must be 1-8")
    try:
        return graph_generator.complete_next_node(
            dataset=req.dataset,
            model=req.model,
            graph=req.graph.model_dump(),
            num_candidates=req.num_candidates,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("next-node completion failed (dataset=%s model=%s)", req.dataset, req.model)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/graph/completion", response_model=GraphResponse)
def graph_completion(req: GraphCompletionRequest):
    if req.target_num_nodes < 2 or req.target_num_nodes > 64:
        raise HTTPException(400, "target_num_nodes must be 2-64")
    if req.num_candidates < 1 or req.num_candidates > 8:
        raise HTTPException(400, "num_candidates must be 1-8")
    try:
        return graph_generator.complete_graph(
            dataset=req.dataset,
            model=req.model,
            graph=req.graph.model_dump(),
            target_num_nodes=req.target_num_nodes,
            num_candidates=req.num_candidates,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("graph completion failed (dataset=%s model=%s)", req.dataset, req.model)
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
