# Procedural Floor-Plan CAD Editor — Design

Date: 2026-05-20
Status: approved (design), pending implementation plan

## 1. Goal

Add an editable, professional-grade 2D floor-plan editor to the GSDiff app,
fed by the procedural (rule-based) generator. A designer draws a bubble diagram
+ boundary, generates a plan, then edits it (drag walls, move doors, change room
types) when the generated result isn't satisfactory. The editor is built on a
shared geometry data model so a one-click 2D→3D view and future BIM export
(DXF/IFC) come naturally.

Non-goals for v1: furniture, materials, multi-floor stacking, manual wall
drawing from scratch, dimension annotations. These are future extensions.

## 2. Context

Existing app (`app/`): Next.js 16, React 19, Tailwind 4, shadcn, base-ui,
lucide-react, **D3** (d3-zoom/drag/selection/force).

Reusable components:
- `BubbleGraphCanvas` (D3) — bubble-diagram editor: nodes (room types), edges
  (connectivity), drag, edge-type editing.
- `BoundaryCanvas` — boundary drawing.
- `MainViewer`, `ModeSelector`, `HistoryBar`, left control panel.
- Current generation output is a PNG `<img>` (ML backend) — not editable. The
  procedural editor replaces this with editable vector geometry.

Backend (`app/backend/`): FastAPI. Existing routers: `generate`, `models`,
`retrieve`. The procedural generator lives in `procedural/` (committed:
`generator.generate(graph, boundary) -> Layout` of room polygons).

## 3. Architecture

```
                ┌─────────────────────────┐
                │  Plan data model (TS)    │   core asset, renderer-agnostic
                │  Wall / Opening / Room / │   procedural backend emits this
                │  AxisGrid / Plan         │
                └───────────┬─────────────┘
                    ┌────────┴────────┐
            ┌───────▼──────┐   ┌──────▼────────┐
            │ 2D editor     │   │ 3D view        │
            │ SVG + d3-zoom │   │ react-three-   │   one-click toggle:
            │ double walls  │   │ fiber (extrude)│   same Plan, two renderers
            │ axis grid     │   │                │
            └──────────────┘   └───────────────┘
                          ↓ future
              DXF / IFC / Speckle (web-ifc, @thatopen)
```

Why custom (not react-planner/arcada): react-planner requires React 16 + Redux4
+ ImmutableJS3 + Three 0.94 (2018 stack), incompatible with React 19/Next 16.
arcada is Pixi-only 2D with a closed model that can't ingest our procedural
output. Building a clean shared model + SVG-2D + R3F-3D is React-19-native,
ingests the procedural output directly, and extends to BIM. The "shared model,
two renderers" pattern is exactly how react-planner achieves 2D→3D — we
replicate the proven idea on a modern stack.

## 4. Data model (`app/lib/plan.ts`)

```typescript
type Pt = { x: number; y: number };          // metres, world frame

interface Wall {
  id: string;
  a: Pt; b: Pt;            // centreline endpoints
  thickness: number;       // metres (default 0.2) → rendered as double line
}

interface Opening {
  id: string;
  wallId: string;
  t: number;               // 0..1 position of opening centre along the wall
  width: number;           // metres
  kind: 'door' | 'window' | 'passage';
}

type RoomType =
  | 'Bedroom' | 'Livingroom' | 'Kitchen' | 'Dining' | 'Corridor'
  | 'Stairs' | 'Storeroom' | 'Bathroom' | 'Balcony';

interface Room {
  id: string;
  type: RoomType;
  poly: Pt[];              // closed polygon (fill + label + 3D floor)
  wallIds: string[];       // boundary walls (for drag-linking)
}

interface AxisGrid {
  originX: number; originY: number;
  spacingX: number; spacingY: number;   // metres (default 1.0)
  angleDeg: number;                      // building dominant axis
}

interface Plan {
  walls: Wall[];
  openings: Opening[];
  rooms: Room[];
  grid: AxisGrid;
  unit: 'm';
}
```

DXF/IFC/Speckle export are pure functions `Plan -> string/bytes` added later.

## 5. Backend: `/api/generate/procedural`

New router `app/backend/app/routers/procedural.py`.

Request:
```jsonc
{
  "graph": {
    "nodes": [{ "id": 0, "room_type": "Livingroom" }, ...],
    "edges": [{ "source": 0, "target": 2, "connectivity": "door" }, ...]
  },
  "boundary": [[x0,y0], [x1,y1], ...],   // polygon, metres
  "seed": 0                              // optional, change for a variant
}
```

Response = `Plan`:
```jsonc
{
  "rooms":    [{ "id": "r0", "type": "Livingroom", "poly": [[x,y],...], "wallIds": ["w3","w7",...] }],
  "walls":    [{ "id": "w0", "a": [x,y], "b": [x,y], "thickness": 0.2 }],
  "openings": [{ "id": "o0", "wallId": "w3", "t": 0.5, "width": 0.9, "kind": "door" }],
  "grid":     { "originX": 0, "originY": 0, "spacingX": 1, "spacingY": 1, "angleDeg": 37 },
  "unit": "m"
}
```

Derivation in a new `procedural/plan_export.py`:
1. `generator.generate(graph, boundary, seed)` → room polygons.
2. **Walls**: collect every room polygon edge; merge shared/collinear-coincident
   edges between two rooms into one wall; boundary edges become exterior walls.
   Each wall gets `thickness=0.2`.
3. **Openings**: for each input graph edge with connectivity in
   {door, passage, entrance}, find the wall shared by the two rooms, add an
   Opening at its midpoint (`t≈0.5`, width by kind: door 0.9, passage 1.2,
   window 1.0). `wall`-type edges → no opening (solid wall).
4. **Grid**: `angleDeg` = `generator.dominant_angle(boundary)`; spacing 1.0 m.

The frontend never re-derives walls; the backend owns this so 2D/3D/export all
share one source of truth.

## 6. Frontend integration

New mode `design` in `ModeSelector` / `GenerationMode`. Layout: the existing
left control panel + a split workspace.

```
┌─────────┬───────────────────────┬──────────────────────┐
│ left    │  Input (reuse)        │  FloorPlanEditor (new)│
│ panel   │  BubbleGraphCanvas    │  2D SVG / 3D R3F      │
│ (reuse) │  + boundary           │  toggle               │
└─────────┴───────────────────────┴──────────────────────┘
```

- `page.tsx`: add `design` mode handler `handleGenerateProcedural` → calls
  `generateProcedural(graph, boundary, seed)` in `app/lib/api.ts`, stores the
  returned `Plan` in component state (and history as a new `plan` item kind).
- `MainViewer`: in `design` mode, render `<DesignWorkspace>` (the split view).
- History: add `PlanHistoryItem { kind: 'plan'; plan: Plan }` to `lib/history.ts`.

**Boundary input format**: the existing `BoundaryCanvas` emits a 256×256 PNG
(for the ML boundary model). The procedural endpoint needs a **polygon**
(`[[x,y],...]` in metres). For v1, the design mode takes the boundary as a
polygon via one of: (a) a lightweight polygon-draw tool (click vertices), or
(b) a default/sample boundary loaded from an MSD floor for quick testing.
Extending `BoundaryCanvas` to also emit vertices is deferred; the plan picks
the simplest path (likely a small polygon-draw control reusing d3-drag).

## 7. 2D editor (`app/components/floorplan/FloorPlanEditor2D.tsx`)

SVG canvas, `d3-zoom` for pan/zoom (consistent with `BubbleGraphCanvas`).

Render order (SVG layers, back→front):
1. **Axis grid** — dashed lines at `grid.spacing`, rotated by `grid.angleDeg`.
2. **Room fills** — `poly` filled by room-type colour (reuse `RoomLegend` palette).
3. **Walls (double line)** — each wall centreline offset ±thickness/2 → two
   parallel strokes; joints closed by filling the wall as a thin polygon.
4. **Openings** — gap in the wall at the opening span; doors drawn with a swing
   arc, windows with a thin double line, passages as a plain gap.
5. **Room labels** — type text at polygon centroid.
6. **Selection / handles overlay**.

Interactions (v1):
- **Drag wall**: select a wall, drag perpendicular; the two adjacent rooms'
  polygons update so the shared edge follows (shared-wall linkage). Snap to grid.
- **Resize room**: select a room, bounding-box handles; dragging a handle moves
  the corresponding walls.
- **Openings**: click a wall to add a door at click position; drag an opening
  along its wall; select + Delete to remove.
- **Change room type**: select a room → dropdown (9 types); fill + label update.
- **Grid snap**: all drag operations snap endpoints to the rotated axis grid;
  toggle on/off.

State: a single `Plan` in React state; edits produce a new `Plan` (immutable
update). Undo/redo via a small history stack (future-friendly, simple array).

## 8. 3D view (`app/components/floorplan/FloorPlanView3D.tsx`)

`react-three-fiber` + `@react-three/drei` (OrbitControls). New deps: `three`,
`@react-three/fiber`, `@react-three/drei`.

- Toggle button switches the right pane between 2D editor and 3D view; both read
  the same `Plan`.
- **Walls**: each Wall → a box extruded along its centreline to wall height
  (default 2.8 m), width = thickness. Openings cut rectangular holes (CSG via
  three-bvh-csg, or simpler: split the wall box around the opening span for v1).
- **Floors**: each Room `poly` → a flat extruded slab at z=0.
- Camera: OrbitControls, sensible default framing on the plan bounds.

v1 keeps 3D read-only (no editing in 3D). Editing stays in 2D.

## 9. Phasing

| Area    | v1 (this work)                                              | Future |
|---------|-------------------------------------------------------------|--------|
| Backend | `/api/generate/procedural` → Plan (walls/openings derived)  | per-apartment metadata, multi-floor |
| 2D edit | drag wall / resize room / add-move-delete opening / room type / grid snap / undo-redo | manual wall draw, dimensions, area schedule |
| 3D      | extrude preview + orbit                                     | furniture, materials, levels, edit-in-3D |
| Export  | Plan JSON + SVG download                                    | DXF, IFC (web-ifc/@thatopen), Speckle |

## 10. Component / file map

```
app/lib/plan.ts                                  # Plan data model + helpers
app/lib/api.ts                                   # + generateProcedural()
app/lib/history.ts                               # + PlanHistoryItem
app/components/floorplan/
  DesignWorkspace.tsx                            # split view (graph | editor)
  FloorPlanEditor2D.tsx                          # SVG editor + interactions
  FloorPlanView3D.tsx                            # R3F 3D view
  walls.ts                                        # double-line geometry, joints
  interactions.ts                                 # drag/resize/snap helpers
app/backend/app/routers/procedural.py            # FastAPI endpoint
procedural/plan_export.py                         # polygons → walls/openings/Plan
```

## 11. Risks / open points

- **Shared-wall drag linkage**: keeping adjacent room polygons consistent when a
  wall moves is the trickiest interaction. Mitigation: walls are the source of
  truth; room polygons are recomputed from their `wallIds` after each edit
  rather than edited directly.
- **Wall joints**: clean double-line corners need miter/cleanup. v1 fills each
  wall as a thin rectangle and overlaps at joints (visually acceptable);
  proper miter is a refinement.
- **Opening CSG in 3D**: full boolean cut adds a dependency. v1 may split the
  wall box around the opening instead of CSG.
- **Adjacency satisfaction ~60%** (procedural limitation): the editor is exactly
  the mitigation — the designer fixes unsatisfied adjacencies by hand.

## 12. Testing

- Backend `plan_export`: unit tests that walls dedup correctly, every door/
  passage edge yields exactly one opening on the correct wall, wall-edges yield
  none, total wall length is finite, no zero-length walls.
- Frontend: component smoke (render a sample Plan, assert wall/opening counts);
  interaction tests for drag-snap and room-type change (where practical).
