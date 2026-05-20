# Procedural CAD Editor — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate an editable vector floor plan from a bubble diagram + boundary and render it read-only in the app with double-line walls, axis grid, room fills, and openings.

**Architecture:** The backend's procedural generator emits room polygons; a new `plan_export` module derives walls (deduped shared/exterior polygon edges) and openings (from door/passage/entrance graph edges) into a `Plan` dict. A new FastAPI endpoint returns it. The frontend defines a matching `Plan` TypeScript model and renders it in an SVG `FloorPlanEditor2D` inside a new `design` mode. Phase 1 is read-only; interactions and 3D are later phases.

**Tech Stack:** Python (shapely, networkx, pytest, FastAPI/TestClient/httpx) for the backend; Next.js 16 / React 19 / TypeScript / Tailwind / SVG for the frontend. All Python runs on `D:/Github/GSDiff/.venv/Scripts/python.exe`.

**Scope note:** Phase 2 (drag walls / resize rooms / openings edit / room-type change / grid snap / undo-redo) and Phase 3 (react-three-fiber 3D view) are separate plans. This plan stops at a rendered, non-interactive plan.

**Spec:** `docs/superpowers/specs/2026-05-20-procedural-cad-editor-design.md`

---

## File Structure (Phase 1)

Backend:
- Create `procedural/plan_export.py` — `derive_walls`, `derive_openings`, `build_plan` (polygons + graph → Plan dict)
- Create `procedural/tests/__init__.py` (empty package marker)
- Create `procedural/tests/test_plan_export.py` — unit tests for the above
- Create `app/backend/app/routers/procedural.py` — `POST /api/generate/procedural`
- Modify `app/backend/app/main.py` — include the procedural router
- Create `app/backend/tests/__init__.py` (empty)
- Create `app/backend/tests/test_procedural_endpoint.py` — endpoint test on a minimal app

Frontend:
- Create `app/lib/plan.ts` — `Plan` and member types + helpers
- Modify `app/lib/api.ts` — `generateProcedural()`
- Create `app/components/floorplan/walls.ts` — centreline → double-line polygon geometry
- Create `app/components/floorplan/FloorPlanEditor2D.tsx` — read-only SVG renderer
- Modify `app/lib/constants.ts` — add `design` to `GenerationMode` + mode metadata
- Modify `app/components/ModeSelector.tsx` — surface the `design` mode (only if it hardcodes modes)
- Modify `app/components/MainViewer.tsx` — render the editor in `design` mode
- Modify `app/src/app/page.tsx` — `design` generate handler + state

---

## Task 1: Derive walls from room polygons

**Files:**
- Create: `procedural/plan_export.py`
- Create: `procedural/tests/__init__.py`
- Test: `procedural/tests/test_plan_export.py`

- [ ] **Step 1: Create the empty test package marker**

Create `procedural/tests/__init__.py` with no content (empty file).

- [ ] **Step 2: Write the failing test**

Create `procedural/tests/test_plan_export.py`:

```python
from procedural.plan_export import derive_walls


def _square(x0, y0, x1, y1):
    """Closed CCW polygon as list of (x, y), first point repeated at end."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def test_derive_walls_two_adjacent_squares_dedup_shared_edge():
    # Two unit squares sharing the edge x=1, y in [0,1].
    rooms = {
        "rA": _square(0, 0, 1, 1),
        "rB": _square(1, 0, 2, 1),
    }
    walls = derive_walls(rooms, thickness=0.2)
    # 4 + 4 edges, the shared (1,0)-(1,1) edge deduped to one → 7 unique walls.
    assert len(walls) == 7
    # every wall has the requested thickness and non-zero length
    for w in walls:
        assert w["thickness"] == 0.2
        ax, ay = w["a"]
        bx, by = w["b"]
        assert (ax, ay) != (bx, by)
    # the shared segment appears exactly once
    def key(w):
        return tuple(sorted([tuple(w["a"]), tuple(w["b"])]))
    keys = [key(w) for w in walls]
    shared = tuple(sorted([(1.0, 0.0), (1.0, 1.0)]))
    assert keys.count(shared) == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest procedural/tests/test_plan_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'procedural.plan_export'` (or ImportError for `derive_walls`).

- [ ] **Step 4: Write minimal implementation**

Create `procedural/plan_export.py`:

```python
"""Derive a renderable/editable Plan (walls + openings) from procedural output.

The procedural generator emits room polygons. The CAD editor needs walls
(double-line) and openings (doors/passages). This module bridges the two:
shared polygon edges become interior walls, lone edges become exterior walls,
and graph access edges (door/passage/entrance) become openings on the shared
wall between the two rooms.
"""

from __future__ import annotations

import math

ROUND = 3  # coordinate rounding (mm) for coincident-edge matching


def _seg_key(a, b):
    """Order-independent rounded key for an edge segment."""
    pa = (round(a[0], ROUND), round(a[1], ROUND))
    pb = (round(b[0], ROUND), round(b[1], ROUND))
    return tuple(sorted([pa, pb]))


def derive_walls(rooms: dict[str, list[tuple[float, float]]],
                 thickness: float = 0.2) -> list[dict]:
    """Collect polygon edges across all rooms; dedup coincident edges.

    `rooms` maps room_id -> closed polygon (list of (x, y), first point repeated
    at the end). Returns a list of wall dicts: {id, a:[x,y], b:[x,y], thickness}.
    """
    seen: dict[tuple, dict] = {}
    for poly in rooms.values():
        for i in range(len(poly) - 1):
            a, b = poly[i], poly[i + 1]
            if (round(a[0], ROUND), round(a[1], ROUND)) == (round(b[0], ROUND), round(b[1], ROUND)):
                continue  # skip zero-length edge
            k = _seg_key(a, b)
            if k not in seen:
                seen[k] = {"a": [float(a[0]), float(a[1])],
                           "b": [float(b[0]), float(b[1])]}
    walls = []
    for i, (_, ab) in enumerate(seen.items()):
        walls.append({"id": f"w{i}", "a": ab["a"], "b": ab["b"], "thickness": thickness})
    return walls
```

- [ ] **Step 5: Run test to verify it passes**

Run: `D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest procedural/tests/test_plan_export.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add procedural/plan_export.py procedural/tests/__init__.py procedural/tests/test_plan_export.py
git commit -m "Add derive_walls: dedup room-polygon edges into walls"
```

---

## Task 2: Derive openings from graph access edges

**Files:**
- Modify: `procedural/plan_export.py`
- Test: `procedural/tests/test_plan_export.py`

- [ ] **Step 1: Write the failing test (append to test file)**

Append to `procedural/tests/test_plan_export.py`:

```python
from procedural.plan_export import derive_openings


def test_derive_openings_door_edge_makes_one_opening():
    rooms = {"rA": _square(0, 0, 1, 1), "rB": _square(1, 0, 2, 1)}
    walls = derive_walls(rooms, thickness=0.2)
    # graph edge between rA and rB with a door → one opening on the shared wall
    edges = [{"source": "rA", "target": "rB", "connectivity": "door"}]
    openings = derive_openings(rooms, walls, edges)
    assert len(openings) == 1
    o = openings[0]
    assert o["kind"] == "door"
    # the opening's wall is the shared (1,0)-(1,1) segment
    wall = next(w for w in walls if w["id"] == o["wallId"])
    assert _seg_key(wall["a"], wall["b"]) == tuple(sorted([(1.0, 0.0), (1.0, 1.0)]))
    assert 0.0 <= o["t"] <= 1.0


def test_derive_openings_wall_edge_makes_none():
    rooms = {"rA": _square(0, 0, 1, 1), "rB": _square(1, 0, 2, 1)}
    walls = derive_walls(rooms, thickness=0.2)
    edges = [{"source": "rA", "target": "rB", "connectivity": "wall"}]
    openings = derive_openings(rooms, walls, edges)
    assert openings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest procedural/tests/test_plan_export.py -v`
Expected: FAIL with ImportError for `derive_openings`.

- [ ] **Step 3: Write minimal implementation (append to plan_export.py)**

Append to `procedural/plan_export.py`:

```python
ACCESS_KINDS = {"door", "passage", "entrance"}
OPENING_WIDTH = {"door": 0.9, "passage": 1.2, "entrance": 1.0, "window": 1.0}


def _room_edge_keys(poly) -> set:
    keys = set()
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        keys.add(_seg_key(a, b))
    return keys


def derive_openings(rooms: dict[str, list[tuple[float, float]]],
                    walls: list[dict],
                    edges: list[dict]) -> list[dict]:
    """One opening per access edge (door/passage/entrance) on the wall shared by
    the two rooms. wall-type edges produce nothing. If the two rooms share no
    coincident wall (adjacency not realised geometrically), the edge is skipped.
    """
    wall_by_key = {_seg_key(w["a"], w["b"]): w for w in walls}
    openings: list[dict] = []
    idx = 0
    for e in edges:
        kind = e.get("connectivity")
        if kind not in ACCESS_KINDS:
            continue
        ra, rb = e.get("source"), e.get("target")
        if ra not in rooms or rb not in rooms:
            continue
        shared = _room_edge_keys(rooms[ra]) & _room_edge_keys(rooms[rb])
        wall = None
        for k in shared:
            if k in wall_by_key:
                wall = wall_by_key[k]
                break
        if wall is None:
            continue  # adjacency not realised — designer fixes by hand later
        openings.append({
            "id": f"o{idx}",
            "wallId": wall["id"],
            "t": 0.5,
            "width": OPENING_WIDTH.get(kind, 0.9),
            "kind": "door" if kind in ("door", "entrance") else "passage",
        })
        idx += 1
    return openings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest procedural/tests/test_plan_export.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add procedural/plan_export.py procedural/tests/test_plan_export.py
git commit -m "Add derive_openings: access edges -> wall openings"
```

---

## Task 3: Build the full Plan dict from a graph + boundary

**Files:**
- Modify: `procedural/plan_export.py`
- Test: `procedural/tests/test_plan_export.py`

- [ ] **Step 1: Write the failing test (append)**

Append to `procedural/tests/test_plan_export.py`:

```python
import json
import pickle
from pathlib import Path

from procedural.plan_export import build_plan
from procedural import io_msd


def _load_multiapt_graph():
    p = Path(__file__).resolve().parents[2] / "digress" / "data" / "msd_wall_v6" / "graphs.p"
    with open(p, "rb") as f:
        graphs = pickle.load(f)
    # smallest graph for a fast test
    return min(graphs, key=lambda g: g.number_of_nodes())


def test_build_plan_shape_and_json_serializable():
    g = _load_multiapt_graph()
    boundary = io_msd.boundary_for_processed_graph(g)
    assert boundary is not None
    plan = build_plan(g, boundary, seed=0)

    for key in ("walls", "openings", "rooms", "grid", "unit"):
        assert key in plan
    assert plan["unit"] == "m"
    assert len(plan["rooms"]) > 0
    assert len(plan["walls"]) > 0
    # every room references walls and has a polygon + type
    r = plan["rooms"][0]
    for key in ("id", "type", "poly", "wallIds"):
        assert key in r
    # grid carries the building axis angle
    assert "angleDeg" in plan["grid"]
    # whole thing must be JSON-serializable (it crosses the HTTP boundary)
    json.dumps(plan)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest procedural/tests/test_plan_export.py::test_build_plan_shape_and_json_serializable -v`
Expected: FAIL with ImportError for `build_plan`.

- [ ] **Step 3: Write minimal implementation (append to plan_export.py)**

Append to `procedural/plan_export.py`:

```python
import networkx as nx
from shapely.geometry import Polygon

from procedural import generator, rules


def _poly_to_pts(poly: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in poly.exterior.coords]


def build_plan(graph: nx.Graph, boundary: Polygon, seed: int = 0) -> dict:
    """Run the procedural generator, then derive a Plan dict (walls/openings/
    rooms/grid). JSON-serializable; this is the HTTP response body.
    """
    layout = generator.generate(graph, boundary, seed=seed)
    angle = generator.dominant_angle(boundary)

    # rooms: stable string ids r{node}
    rooms_poly: dict[str, list[tuple[float, float]]] = {}
    room_meta: dict[str, dict] = {}
    for n, room in layout.rooms.items():
        if room.polygon.is_empty:
            continue
        rid = f"r{n}"
        pts = [(float(x), float(y)) for x, y in room.polygon.exterior.coords]
        rooms_poly[rid] = pts
        room_meta[rid] = {"type": room.room_type, "node": n}

    walls = derive_walls(rooms_poly, thickness=0.2)

    # graph edges keyed by the same r{node} ids
    edges = []
    for u, v, d in graph.edges(data=True):
        edges.append({"source": f"r{u}", "target": f"r{v}",
                      "connectivity": d.get("connectivity", "wall")})
    openings = derive_openings(rooms_poly, walls, edges)

    # attach wallIds to each room (walls whose segment lies on the room polygon)
    wall_by_key = {_seg_key(w["a"], w["b"]): w for w in walls}
    rooms_out = []
    for rid, pts in rooms_poly.items():
        wall_ids = []
        for k in _room_edge_keys(pts):
            w = wall_by_key.get(k)
            if w:
                wall_ids.append(w["id"])
        rooms_out.append({
            "id": rid,
            "type": room_meta[rid]["type"],
            "poly": [[float(x), float(y)] for x, y in pts],
            "wallIds": wall_ids,
        })

    return {
        "walls": walls,
        "openings": openings,
        "rooms": rooms_out,
        "grid": {"originX": 0.0, "originY": 0.0,
                 "spacingX": 1.0, "spacingY": 1.0, "angleDeg": float(angle)},
        "unit": "m",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest procedural/tests/test_plan_export.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add procedural/plan_export.py procedural/tests/test_plan_export.py
git commit -m "Add build_plan: graph + boundary -> JSON Plan dict"
```

---

## Task 4: Procedural FastAPI endpoint

**Files:**
- Create: `app/backend/app/routers/procedural.py`
- Modify: `app/backend/app/main.py:35-36`
- Create: `app/backend/tests/__init__.py`
- Test: `app/backend/tests/test_procedural_endpoint.py`

- [ ] **Step 1: Create the empty test package marker**

Create `app/backend/tests/__init__.py` with no content.

- [ ] **Step 2: Write the failing endpoint test**

Create `app/backend/tests/test_procedural_endpoint.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run (from `app/backend`):
`cd app/backend && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_procedural_endpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.routers.procedural'`.

- [ ] **Step 4: Write the router**

Create `app/backend/app/routers/procedural.py`:

```python
"""Procedural (rule-based) floor-plan generation endpoint."""

import sys

import networkx as nx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from shapely.geometry import Polygon

from app.config import PROJECT_ROOT

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from procedural.plan_export import build_plan  # noqa: E402

router = APIRouter(prefix="/api/generate", tags=["procedural"])


class GraphNode(BaseModel):
    id: int
    room_type: str


class GraphEdge(BaseModel):
    source: int
    target: int
    connectivity: str = "wall"


class GraphIn(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class ProceduralRequest(BaseModel):
    graph: GraphIn
    boundary: list[list[float]]
    seed: int = 0


def _to_nx(graph: GraphIn) -> nx.Graph:
    g = nx.Graph()
    for node in graph.nodes:
        g.add_node(node.id, room_type=node.room_type)
    for e in graph.edges:
        g.add_edge(e.source, e.target, connectivity=e.connectivity)
    return g


@router.post("/procedural")
def procedural(req: ProceduralRequest):
    if len(req.boundary) < 3:
        raise HTTPException(400, "boundary needs at least 3 points")
    try:
        g = _to_nx(req.graph)
        boundary = Polygon(req.boundary)
        if not boundary.is_valid:
            boundary = boundary.buffer(0)
        return build_plan(g, boundary, seed=req.seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```

- [ ] **Step 5: Run test to verify it passes**

Run (from `app/backend`):
`cd app/backend && D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/test_procedural_endpoint.py -v`
Expected: PASS.

> Note: `build_plan` calls `generator.generate`, which expects `room_type` as a string or int index. The endpoint passes strings (e.g. "Livingroom"); `generator._node_type` already handles string types. No model loading happens — the procedural path is pure geometry.

- [ ] **Step 6: Register the router in main.py**

In `app/backend/app/main.py`, after the existing router imports (around line 12-14), add:

```python
from app.routers.procedural import router as procedural_router
```

and after `app.include_router(models_router)` (around line 36) add:

```python
app.include_router(procedural_router)
```

- [ ] **Step 7: Commit**

```bash
git add app/backend/app/routers/procedural.py app/backend/app/main.py app/backend/tests/__init__.py app/backend/tests/test_procedural_endpoint.py
git commit -m "Add /api/generate/procedural endpoint"
```

---

## Task 5: Frontend Plan data model

**Files:**
- Create: `app/lib/plan.ts`

- [ ] **Step 1: Write the type module**

Create `app/lib/plan.ts`:

```typescript
// Plan data model — mirrors procedural/plan_export.build_plan output.
// Shared by the 2D editor (Phase 1-2) and the future 3D view (Phase 3).

export type Pt = [number, number]; // [x, y] in metres

export interface Wall {
  id: string;
  a: Pt;
  b: Pt;
  thickness: number;
}

export type OpeningKind = 'door' | 'window' | 'passage';

export interface Opening {
  id: string;
  wallId: string;
  t: number; // 0..1 along the wall
  width: number;
  kind: OpeningKind;
}

export type RoomType =
  | 'Bedroom' | 'Livingroom' | 'Kitchen' | 'Dining' | 'Corridor'
  | 'Stairs' | 'Storeroom' | 'Bathroom' | 'Balcony';

export interface Room {
  id: string;
  type: RoomType;
  poly: Pt[];
  wallIds: string[];
}

export interface AxisGrid {
  originX: number;
  originY: number;
  spacingX: number;
  spacingY: number;
  angleDeg: number;
}

export interface Plan {
  walls: Wall[];
  openings: Opening[];
  rooms: Room[];
  grid: AxisGrid;
  unit: 'm';
}

// Bounds of every room/wall point — used to fit the SVG viewBox.
export function planBounds(plan: Plan): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const eat = (x: number, y: number) => {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  };
  for (const r of plan.rooms) for (const [x, y] of r.poly) eat(x, y);
  for (const w of plan.walls) { eat(w.a[0], w.a[1]); eat(w.b[0], w.b[1]); }
  if (!isFinite(minX)) return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  return { minX, minY, maxX, maxY };
}
```

- [ ] **Step 2: Verify it type-checks**

Run (from `app`): `cd app && npx tsc --noEmit`
Expected: no errors referencing `plan.ts`.

- [ ] **Step 3: Commit**

```bash
git add app/lib/plan.ts
git commit -m "Add Plan TypeScript data model"
```

---

## Task 6: API client for procedural generation

**Files:**
- Modify: `app/lib/api.ts`

- [ ] **Step 1: Read the existing api.ts to match its fetch/base-URL pattern**

Run: open `app/lib/api.ts` and note the base URL constant and how an existing POST helper (e.g. `generateTopology`) is written.

- [ ] **Step 2: Add the generateProcedural function**

Append to `app/lib/api.ts` (use the same base URL constant the file already defines — shown here as `API_BASE`; match the actual name in the file):

```typescript
import type { Plan } from '@/lib/plan';

export interface ProceduralGraphInput {
  nodes: { id: number; room_type: string }[];
  edges: { source: number; target: number; connectivity: string }[];
}

export async function generateProcedural(
  graph: ProceduralGraphInput,
  boundary: [number, number][],
  seed = 0,
): Promise<Plan> {
  const res = await fetch(`${API_BASE}/api/generate/procedural`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ graph, boundary, seed }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Procedural generation failed (${res.status})`);
  }
  return (await res.json()) as Plan;
}
```

> If `api.ts` does not already define an exported `API_BASE`, reuse whatever
> constant the existing functions use for the backend origin (do not introduce a
> second base-URL constant).

- [ ] **Step 3: Verify it type-checks**

Run (from `app`): `cd app && npx tsc --noEmit`
Expected: no new errors.

- [ ] **Step 4: Commit**

```bash
git add app/lib/api.ts
git commit -m "Add generateProcedural API client"
```

---

## Task 7: Double-line wall geometry helper

**Files:**
- Create: `app/components/floorplan/walls.ts`

- [ ] **Step 1: Write the helper**

Create `app/components/floorplan/walls.ts`:

```typescript
import type { Wall, Pt } from '@/lib/plan';

// A wall centreline + thickness → the 4 corners of its rectangle, so the SVG
// renders the wall as a filled double-line body rather than a hairline.
export function wallQuad(wall: Wall): Pt[] {
  const [ax, ay] = wall.a;
  const [bx, by] = wall.b;
  const dx = bx - ax;
  const dy = by - ay;
  const len = Math.hypot(dx, dy) || 1;
  // unit normal
  const nx = -dy / len;
  const ny = dx / len;
  const h = wall.thickness / 2;
  return [
    [ax + nx * h, ay + ny * h],
    [bx + nx * h, by + ny * h],
    [bx - nx * h, by - ny * h],
    [ax - nx * h, ay - ny * h],
  ];
}

// SVG path "d" for a closed polygon of points.
export function polyPath(pts: Pt[]): string {
  if (pts.length === 0) return '';
  const [first, ...rest] = pts;
  return `M ${first[0]} ${first[1]} ` + rest.map((p) => `L ${p[0]} ${p[1]}`).join(' ') + ' Z';
}

// Midpoint of a wall offset by t in [0,1] from a→b; used to place openings.
export function pointAlongWall(wall: Wall, t: number): Pt {
  return [wall.a[0] + (wall.b[0] - wall.a[0]) * t, wall.a[1] + (wall.b[1] - wall.a[1]) * t];
}
```

- [ ] **Step 2: Verify it type-checks**

Run (from `app`): `cd app && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add app/components/floorplan/walls.ts
git commit -m "Add wall double-line geometry helpers"
```

---

## Task 8: Read-only 2D SVG renderer

**Files:**
- Create: `app/components/floorplan/FloorPlanEditor2D.tsx`

- [ ] **Step 1: Write the component**

Create `app/components/floorplan/FloorPlanEditor2D.tsx`:

```tsx
'use client';

import { useMemo } from 'react';
import type { Plan, RoomType } from '@/lib/plan';
import { planBounds } from '@/lib/plan';
import { wallQuad, polyPath, pointAlongWall } from './walls';

// Reuse a room-type → colour mapping consistent with the rest of the app.
const ROOM_COLORS: Record<RoomType, string> = {
  Livingroom: '#aec7e8', Bedroom: '#1f77b4', Kitchen: '#ff7f0e',
  Dining: '#ffbb78', Corridor: '#2ca02c', Stairs: '#98df8a',
  Storeroom: '#d62728', Bathroom: '#ff9896', Balcony: '#9467bd',
};

interface Props {
  plan: Plan;
}

export function FloorPlanEditor2D({ plan }: Props) {
  const b = useMemo(() => planBounds(plan), [plan]);
  const pad = 1; // metres
  const vb = `${b.minX - pad} ${b.minY - pad} ${b.maxX - b.minX + 2 * pad} ${b.maxY - b.minY + 2 * pad}`;

  // grid lines spanning the bounds (axis-aligned for Phase 1; rotation in P2)
  const gridLines: { x1: number; y1: number; x2: number; y2: number }[] = [];
  for (let x = Math.floor(b.minX); x <= Math.ceil(b.maxX); x += plan.grid.spacingX) {
    gridLines.push({ x1: x, y1: b.minY - pad, x2: x, y2: b.maxY + pad });
  }
  for (let y = Math.floor(b.minY); y <= Math.ceil(b.maxY); y += plan.grid.spacingY) {
    gridLines.push({ x1: b.minX - pad, y1: y, x2: b.maxX + pad, y2: y });
  }

  return (
    <svg
      viewBox={vb}
      className="h-full w-full bg-background"
      style={{ transform: 'scaleY(-1)' }} // metres: +y is up
    >
      {/* axis grid */}
      <g stroke="#e5e7eb" strokeWidth={0.02}>
        {gridLines.map((l, i) => (
          <line key={`g${i}`} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} strokeDasharray="0.1 0.1" />
        ))}
      </g>

      {/* room fills */}
      <g>
        {plan.rooms.map((r) => (
          <path
            key={r.id}
            d={polyPath(r.poly)}
            fill={ROOM_COLORS[r.type] ?? '#cccccc'}
            fillOpacity={0.5}
            stroke="none"
          />
        ))}
      </g>

      {/* double-line walls (filled rectangles) */}
      <g fill="#1f2937">
        {plan.walls.map((w) => (
          <path key={w.id} d={polyPath(wallQuad(w))} />
        ))}
      </g>

      {/* openings: erase the wall span with a background-coloured rectangle */}
      <g fill="hsl(0 0% 100%)">
        {plan.openings.map((o) => {
          const wall = plan.walls.find((w) => w.id === o.wallId);
          if (!wall) return null;
          const t0 = Math.max(0, o.t - o.width / 2 / (Math.hypot(wall.b[0] - wall.a[0], wall.b[1] - wall.a[1]) || 1));
          const t1 = Math.min(1, o.t + o.width / 2 / (Math.hypot(wall.b[0] - wall.a[0], wall.b[1] - wall.a[1]) || 1));
          const p0 = pointAlongWall(wall, t0);
          const p1 = pointAlongWall(wall, t1);
          const seg = { ...wall, a: p0, b: p1 };
          return <path key={o.id} d={polyPath(wallQuad(seg))} />;
        })}
      </g>

      {/* room labels (flip text back upright) */}
      <g>
        {plan.rooms.map((r) => {
          const cx = r.poly.reduce((s, p) => s + p[0], 0) / r.poly.length;
          const cy = r.poly.reduce((s, p) => s + p[1], 0) / r.poly.length;
          return (
            <text
              key={`t${r.id}`}
              x={cx}
              y={cy}
              fontSize={0.4}
              textAnchor="middle"
              fill="#111827"
              transform={`scale(1,-1) translate(0, ${-2 * cy})`}
            >
              {r.type}
            </text>
          );
        })}
      </g>
    </svg>
  );
}
```

- [ ] **Step 2: Verify it type-checks**

Run (from `app`): `cd app && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add app/components/floorplan/FloorPlanEditor2D.tsx
git commit -m "Add read-only 2D floor-plan SVG renderer"
```

---

## Task 9: Add the `design` mode

**Files:**
- Modify: `app/lib/constants.ts`
- Modify: `app/components/ModeSelector.tsx`

- [ ] **Step 1: Inspect how modes are declared**

Run: open `app/lib/constants.ts`; find the `GenerationMode` union and any `MODES`/mode-metadata array. Open `app/components/ModeSelector.tsx` to see whether it maps over a constant or hardcodes buttons.

- [ ] **Step 2: Add `design` to the mode union and metadata**

In `app/lib/constants.ts`, add `'design'` to the `GenerationMode` union type. If there is a modes-metadata array (label/description/icon), add an entry:

```typescript
{ id: 'design', name: 'Design', description: 'Procedural editable floor plan' },
```

(Match the exact shape of the existing entries — copy their fields.)

- [ ] **Step 3: Surface it in ModeSelector if needed**

If `ModeSelector.tsx` maps over the constant array, no change is needed. If it hardcodes a list of modes, add `design` alongside the others following the same JSX pattern.

- [ ] **Step 4: Verify it type-checks**

Run (from `app`): `cd app && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add app/lib/constants.ts app/components/ModeSelector.tsx
git commit -m "Add design mode to mode selector"
```

---

## Task 10: Wire generation + rendering in the app

**Files:**
- Modify: `app/components/MainViewer.tsx`
- Modify: `app/src/app/page.tsx`

- [ ] **Step 1: Render the editor in design mode (MainViewer)**

In `app/components/MainViewer.tsx`, add an import:

```tsx
import { FloorPlanEditor2D } from '@/components/floorplan/FloorPlanEditor2D';
import type { Plan } from '@/lib/plan';
```

Add a `planDraft?: Plan | null` prop to the `Props` interface. Add a branch (before the final unconstrained fallback) that renders the editor in design mode:

```tsx
  if (mode === 'design') {
    return (
      <div className="flex flex-1 items-center justify-center overflow-hidden bg-background p-6">
        {error ? (
          <div className="rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">{error}</div>
        ) : loading ? (
          <div className="flex flex-col items-center gap-3">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-muted-foreground/20 border-t-muted-foreground/60" />
            <p className="text-sm text-muted-foreground">Generating plan…</p>
          </div>
        ) : planDraft ? (
          <div className="h-full w-full"><FloorPlanEditor2D plan={planDraft} /></div>
        ) : (
          <p className="text-sm text-muted-foreground">Edit the bubble graph, then click Generate.</p>
        )}
      </div>
    );
  }
```

- [ ] **Step 2: Add design generation state + handler (page.tsx)**

In `app/src/app/page.tsx`:

Add imports:

```tsx
import { generateProcedural } from '@/lib/api';
import type { Plan } from '@/lib/plan';
```

Add state near the other `useState` hooks:

```tsx
const [planDraft, setPlanDraft] = useState<Plan | null>(null);
```

Add a handler (uses the existing `graphDraft`/selected graph and a default rectangular boundary for Phase 1; the polygon-draw control is a later task):

```tsx
const handleGenerateProcedural = useCallback(async () => {
  const graphItem = selectedItem?.kind === 'graph' ? selectedItem : null;
  const sourceGraph = graphDraft ?? (graphItem ? graphItem.graph : null);
  if (!sourceGraph) {
    setError('Draw or sample a bubble graph first');
    return;
  }
  setLoading(true);
  setError(null);
  try {
    const nodes = sourceGraph.nodes.map((n) => ({ id: n.id, room_type: n.room_type }));
    const edges = sourceGraph.edges.map((e) => ({
      source: e.source, target: e.target, connectivity: e.edge_label ?? 'wall',
    }));
    // Phase 1 default boundary: a 12x9 m rectangle. Polygon-draw control is a later task.
    const boundary: [number, number][] = [[0, 0], [12, 0], [12, 9], [0, 9]];
    const plan = await generateProcedural({ nodes, edges }, boundary, 0);
    setPlanDraft(plan);
  } catch (e) {
    setError(e instanceof Error ? e.message : 'Procedural generation failed');
  } finally {
    setLoading(false);
  }
}, [graphDraft, selectedItem]);
```

> `room_type` values from the bubble editor are lowercase (e.g. "living"). The
> backend's `generator._node_type` matches the capitalised `ROOM_NAMES`. Map
> them: if the editor emits lowercase, add a small lookup
> (`living→Livingroom`, `bedroom→Bedroom`, …) in this handler before sending.
> Inspect `sourceGraph.nodes[i].room_type` values at runtime and add the map if
> they don't already match the backend's RoomType strings.

- [ ] **Step 3: Pass planDraft to MainViewer and wire the Generate button**

In `page.tsx`, pass the new prop to `<MainViewer ... planDraft={planDraft} />`. In the `GenerateButton` onClick chain, add:

```tsx
else if (activeMode === 'design') handleGenerateProcedural();
```

and include `design` in the button's enabled condition (treat like `graph` — always enabled).

- [ ] **Step 4: Verify build**

Run (from `app`): `cd app && npx tsc --noEmit && npm run build`
Expected: build succeeds.

- [ ] **Step 5: Manual smoke test**

Start backend: `cd app/backend && D:/Github/GSDiff/.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000`
Start frontend: `cd app && npm run dev`
In the browser: switch to **Design** mode, ensure a bubble graph exists (sample or default), click **Generate**. Expect a rendered floor plan: room fills, dark double-line walls, dashed grid, white gaps where doors/passages are, room-type labels.

- [ ] **Step 6: Commit**

```bash
git add app/components/MainViewer.tsx app/src/app/page.tsx
git commit -m "Wire design mode: generate procedural plan and render it"
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** Phase 1 covers spec §4 (data model → `plan.ts`), §5 (endpoint + derivation → Tasks 1-4), §6 (design mode wiring → Tasks 9-10), §7 render layers (grid/fills/double-walls/openings/labels → Task 8). Spec §7 interactions, §8 3D, §9 export are explicitly deferred to Phase 2/3.
- **Boundary input:** Phase 1 uses a fixed default rectangle (Task 10 Step 2), matching spec §6's option (b). The polygon-draw control is a Phase 2 task.
- **Type consistency:** `Plan`/`Wall`/`Opening`/`Room` fields match between `plan_export.build_plan` (Task 3), `plan.ts` (Task 5), `walls.ts` (Task 7), and `FloorPlanEditor2D` (Task 8): `a`/`b`/`thickness`, `wallId`/`t`/`width`/`kind`, `poly`/`type`/`wallIds`, `grid.angleDeg`.
- **Room-type casing:** flagged in Task 10 Step 2 — map editor lowercase → backend RoomType strings.
