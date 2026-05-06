# MSD app redesign — design spec

**Date**: 2026-05-06
**Scope**: GSDiff demo app (`app/`) UI overhaul + MSD model swap V3 → V5

## 1. Goals

1. Make the MSD `msd_wall` model usable in the app (currently RPLAN-only effectively, since the MSD checkpoint has only just been wired to the backend).
2. Swap the MSD checkpoint from V3 (`outputs/2026-05-01/15-25-20-msd_wall/checkpoints/msd_wall/best.ckpt`) to V5 (`outputs/2026-05-05/13-58-49-msd_wall_v5/checkpoints/msd_wall_v5/best.ckpt`) once V5 passes its 500-sample confirmation run.
3. Replace `@xyflow/react` with a pure SVG + `d3-force` bubble-diagram component shared between Graph (read-only) and Topology (editable) modes.
4. Restructure the layout so the bubble diagram and the floorplan get a large display area and feel like one connected workflow rather than two disjoint panels.

## 2. Non-goals

- Boundary mode rewrite. Keep its current canvas mechanics; only relocate it into the new right-panel workspace.
- Backend `inference.py` (GSDiff side) changes. The redesign affects routing/state, not GSDiff itself.
- New features beyond MSD enablement (e.g. additional datasets, multi-graph batch generation, account/persistence). Out of scope.

## 3. Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│ Header  (project name + active mode label)                  │
├──────────────┬──────────────────────────────────────────────┤
│ LEFT (320px) │            RIGHT WORKSPACE (flex-1)          │
│  Pure        │                                              │
│  controls    │  ┌────────────────────────────────────────┐  │
│              │  │   MAIN VIEWER                          │  │
│  - Dataset   │  │   (graph SVG / floorplan PNG /         │  │
│  - Mode      │  │    boundary canvas)                    │  │
│  - Settings  │  └────────────────────────────────────────┘  │
│    (mode-    │  ┌────────────────────────────────────────┐  │
│    specific) │  │ HISTORY  [⬡g1][⬡g2][⬜f1*][⬡g3][⬜f2]  │  │
│  - Model     │  │                              [Generate]│  │
│    status    │  └────────────────────────────────────────┘  │
│  -─────────  │                                              │
│  [Generate]  │                                              │
└──────────────┴──────────────────────────────────────────────┘
```

Two clear regions:

- **Left (controls, ~320px)** — slimmed from current 680px. Stateless config + a sticky primary action button.
- **Right (workspace, flex-1)** — big main viewer + bottom history strip. All inputs and outputs live here.

## 4. Layout details

### 4.1 Left panel composition (top → bottom)

| Section | What it shows |
|---|---|
| `<Header>` (rendered above panels, full width) | Project name on left, current mode description on right |
| Dataset selector | RPLAN / MSD pills, current selection highlighted |
| Mode selector | Unconstrained / Graph / Topology / Boundary, vertical list |
| Mode-specific settings | Whatever the active mode needs (see §5) |
| `<ModelStatusPanel>` | Compact list of registered models + state badges |
| Sticky primary action button (`<GenerateButton>`) | Bottom, mode-aware label and handler |

### 4.2 Right workspace composition

| Slot | Component |
|---|---|
| Top (flex-1) | `<MainViewer>` — switches based on (mode, currently-selected history item). Shows graph for Graph/Topology modes, floorplan PNG for Unconstrained/post-generation, boundary canvas for Boundary input. |
| Bottom (~120px) | `<HistoryBar>` — horizontal scroll of generated artifacts (graphs and floorplans), current selection highlighted. Right-aligned action button area for context-relevant secondary action (e.g. "Send to Floorplan" when a graph is selected). |

### 4.3 Empty states

When no history yet, the main viewer shows a per-mode empty state:

- Unconstrained: "Click Generate to sample a random floorplan."
- Graph: "Click Sample Graph to generate a bubble diagram."
- Topology: starts with 4 default rooms in a circle (immediate editable canvas, no empty state).
- Boundary: starts with empty 256×256 canvas + pen tool ready.

## 5. Per-mode behavior

### 5.1 Unconstrained
- **Left settings**: none.
- **Primary action**: "Generate Floorplan".
- **Flow**: button → spinner in main viewer → floorplan PNG appears, added to history as `⬜`, auto-selected.
- **Main viewer interactions**: download button overlay.

### 5.2 Graph
- **Left settings**: optional `num_nodes` slider (1–N for selected dataset), optional seed input.
- **Primary action**: "Sample Graph".
- **Flow**: button → spinner → `<BubbleGraphCanvas mode="view">` appears in main viewer with the new graph; added to history as `⬡`.
- **Secondary action** (in main viewer top-right overlay or history bar right side): "Send to Floorplan" — runs GSDiff topology endpoint with the graph's rooms+adjacency, result added as `⬜` with `sourceGraphId` reference.
- **Main viewer interactions**: drag node to reposition (overrides force layout for that node), zoom, pan. Read-only otherwise.

### 5.3 Topology
- **Main viewer**: `<BubbleGraphCanvas mode="edit">` is the workspace itself. Initial state: 4 default rooms in a circle.
- **Left settings**:
  - Default room type for new nodes (color swatch select)
  - Default edge type for new edges
  - "Reset" button (clears editor back to 4-room initial state)
- **Primary action**: "Generate Floorplan" (uses current edited graph).
- **Editing interactions**:
  - Click empty space → add new node at cursor (default room type)
  - Drag node → reposition (overrides force for that node)
  - Click node → select (visual: outline + delete X)
  - Double-click node → small popover to change room type
  - Drag from node perimeter → drop on another node = create edge (default edge type)
  - Click edge → cycles to the next edge type in this dataset's `edge_decoder` array order, skipping `none` (so for MSD: `wall → passage → door → entrance → wall`); right-click → delete
  - Ctrl+Z / Ctrl+Y for undo/redo. Stack is in-memory only and reset on page reload or "Reset" button click. Stack max depth: 50 operations.
- **Persistence**: editor state saved to `sessionStorage` on every change so mode-switching round-trip preserves work.

### 5.4 Boundary
- **Main viewer**: 256×256 drawing canvas centered with toolbar overlay (pen / eraser / clear).
- **Left settings**: pen size slider, clear button.
- **Primary action**: "Generate Floorplan".
- **Flow**: user draws → click Generate → spinner → floorplan added to history.

## 6. Data flow + state management

### 6.1 Page-level state (in `src/app/page.tsx`)

```typescript
type HistoryItem =
  | { id: string; kind: 'graph'; dataset: DatasetId; graph: GeneratedGraph; createdAt: number }
  | { id: string; kind: 'floorplan'; dataset: DatasetId; image: string; sourceGraphId?: string;
      sourceMode: GenerationMode; createdAt: number };

interface PageState {
  selectedDataset: DatasetId;
  activeMode: GenerationMode;
  history: HistoryItem[];           // global, mixed graph + floorplan
  selectedHistoryId: string | null; // what the main viewer shows
  modeDrafts: {
    topology?: TopologyDraft;        // { nodes, edges } - persisted via sessionStorage
    boundary?: BoundaryDraft;        // { canvasDataUrl } - persisted via sessionStorage
  };
  loading: boolean;
  error: string | null;
}
```

### 6.2 What `<MainViewer>` shows (selection logic)

```
selectedHistoryItem = history.find(i => i.id === selectedHistoryId)

if mode === 'topology':       show <BubbleGraphCanvas edit> with modeDrafts.topology
elif mode === 'boundary':     show <BoundaryCanvas edit> with modeDrafts.boundary
elif selectedHistoryItem:     show that item (graph or floorplan)
else:                          show per-mode empty state
```

Note: in Topology and Boundary modes, the main viewer is the **input editor** (not the selected history item) by default. Clicking any history item that does NOT belong to the current mode auto-switches mode per §6.4 (the only kind of item generated *in* Topology or Boundary mode is a floorplan; clicking it auto-switches to a viewer mode that just renders that floorplan, leaving the editor draft untouched).

The viewer-mode used when auto-switching to view a floorplan is `Unconstrained` (its main viewer is built to display floorplans and has no editor state). This avoids inventing a new "viewer" mode just for cross-mode navigation.

### 6.3 Mode switching

- Every mode switch preserves all state (drafts, history, selection).
- Switching to Topology/Boundary restores the draft from `modeDrafts` if present.
- History is global and survives mode switches.

### 6.4 Cross-mode history navigation

Clicking a graph in history while in (e.g.) Boundary mode:
- Switches `activeMode` to `Graph` (so the bubble graph viewer is active)
- Sets `selectedHistoryId` to the clicked graph
- Shows a small toast or breadcrumb: "Switched to Graph mode to view this item"

This avoids invalid view states (e.g. trying to render a graph while in Boundary mode).

## 7. Component contracts

### 7.1 `<BubbleGraphCanvas>` (new)

The single shared bubble-diagram component. Used by Graph mode (read-only) and Topology mode (editable).

```typescript
interface BubbleGraphCanvasProps {
  graph: GeneratedGraph;
  dataset: DatasetId;
  mode: 'view' | 'edit';
  onChange?: (next: GeneratedGraph) => void;  // required when mode='edit'

  // Edit-mode helpers
  defaultNodeAttr?: number;
  defaultEdgeType?: number;
  onSelectionChange?: (nodeId: number | null) => void;
}
```

Internal:
- `d3-force` simulation (force-x, force-y, force-collide, force-link)
- React-controlled `<svg>` with `<circle>` nodes and `<line>` edges
- d3-zoom for pan/zoom (applied to a top-level `<g>`)
- Manual drag handlers (HTML pointer events on `<circle>`, dispatch back to React state which drives positions)
- Edit-mode interactions wrap React state changes; emit `onChange` with the new graph
- Colors and labels read from `DATASET_SPECS[dataset]`

### 7.2 `<MainViewer>` (new)

Switches its child component based on (mode, selectedHistoryItem, modeDrafts). Just a router — no logic of its own beyond prop assembly.

### 7.3 `<HistoryBar>` (new)

```typescript
interface HistoryBarProps {
  items: HistoryItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onClear?: () => void;
  rightSlot?: React.ReactNode;  // for the contextual action button area
}
```

Each thumbnail is ~80×80, with an icon overlay (`⬡` / `⬜`) and timestamp. Selected item has a thick border. `floorplan` thumbnails use the image; `graph` thumbnails use a tiny rendered SVG (8-12 nodes max, simple force layout).

### 7.4 `<GenerateButton>` (new)

Mode-aware sticky button at the bottom of the left panel. Just a thin wrapper that picks the right label and action handler based on `activeMode` + current state.

| Mode | Label | Handler |
|---|---|---|
| Unconstrained | Generate Floorplan | calls `/generate/unconstrained` |
| Graph | Sample Graph | calls `/generate/graph` |
| Topology | Generate Floorplan | calls `/generate/topology` with current draft |
| Boundary | Generate Floorplan | calls `/generate/boundary` with current canvas |

### 7.5 Components to remove

- `RoomNode.tsx` (react-flow node component, no longer needed)
- `graph-flow.ts` (react-flow graph utility)
- `@xyflow/react` package + `@xyflow/react/dist/style.css` import in `globals.css`
- `TopologyEditor.tsx` is rewritten using `BubbleGraphCanvas`; the file may stay but its react-flow internals are gone.

### 7.6 Components to rewrite

- `TopologyEditor.tsx` — becomes a thin shell wrapping `<BubbleGraphCanvas mode="edit">`.
- `GraphPanel.tsx` — folded into the left-panel + main-viewer split. The current "graph + floorplan generation in one panel" abstraction stops making sense.
- `ResultPanel.tsx` — becomes `<MainViewer>` (or similar). Floorplan rendering moved into the new component.
- `HistoryPanel.tsx` — becomes `<HistoryBar>` with the new mixed-kind item model.
- `BubbleGraphView.tsx` — replaced by `<BubbleGraphCanvas mode="view">`. File deleted.

## 8. Backend changes

### 8.1 V3 → V5 ckpt swap

After V5 500-sample confirmation passes (currently running):

```yaml
# digress/configs/experiment/msd_wall.yaml
test:
    checkpoint: 'outputs/2026-05-05/13-58-49-msd_wall_v5/checkpoints/msd_wall_v5/best.ckpt'
```

`graph_generation.py` already routes `dataset="msd_wall"` to this experiment, so no Python change needed.

If V5 underperforms V3 on the 500-sample run, defer the swap and stay on V3. The Python side stays unchanged either way.

### 8.2 No other backend changes

The `/generate/graph` endpoint already returns `room_types` and `edge_types` per dataset. The earlier refactor of `graph_generation.py` already handles MSD. Nothing else changes here.

## 9. Tech stack additions

- `d3-force` (~25 KB) for graph layout simulation
- `d3-zoom` (~10 KB) for pan/zoom
- `d3-drag` (~8 KB) for node drag handlers
- (no React wrapper, raw d3 modules used inside React component via `useEffect` lifecycle)

Removed: `@xyflow/react` (~150 KB).

Net bundle change: ~−110 KB.

## 10. Visual style

Keep the existing shadcn / Tailwind aesthetic. Only structural layout changes; no new design system.

Specifics:
- Bubble nodes: ~36px radius circles, room-type fill from `DATASET_SPECS`, white text label inside, room-id below
- Edges: 2px stroke, color by edge type, dashed for door/passage/entrance per `EdgeTypeMeta.dashed`
- Selected node: 3px outline in `oklch(0.4 0.18 25)` (warm orange)
- Selected edge: thicker stroke + label visible
- Background: `oklch(0.97 0 0)` (current BubbleGraphView bg)

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Topology editor undo/redo bugs after the rewrite | Keep edit interactions minimal in v1: add/delete node, add/delete edge, cycle edge type, change room type. Defer richer interactions (group selection, copy/paste) until reported. |
| `d3-force` simulation thrashing on each React render | Run simulation in `useEffect` with refs; only push positions back to React state on `tick` and on user drag. Avoid putting force-simulation state in React state. |
| History bar getting cluttered after many generations | Add a cap (last 30 items) + "Clear all" button. Optionally let user pin items. |
| Boundary mode regression after relocating the canvas | Move the existing `BoundaryCanvas` component into the right workspace without touching its drawing internals. |
| V5 connectivity drop (87.5% vs V3 96.9% on 32 samples) is real, not noise | Wait for 500-sample confirmation before swapping. If real, leave V3 in production and run a follow-up V7 with `lambda_train=[2.5, 0]` to balance edge weight vs connectivity. |
| Cross-mode auto-switching when clicking a foreign-mode history item is confusing | Show a small toast on auto-switch; remember user's prior mode in case they want to switch back. |

## 12. Out-of-scope, captured for later

- Persisting history across browser refreshes (currently sessionStorage only for mode drafts; full localStorage history can wait)
- User accounts / cloud-saved sessions
- Batch generation (sample N graphs at once)
- Diffing two graphs side-by-side
- Comparing V3 vs V5 inside the app

## 13. Implementation order (rough)

1. Backend: confirm V5 500-sample results; swap msd_wall.yaml ckpt path if V5 wins.
2. Frontend types: add new `HistoryItem` discriminated union, `PageState` shape.
3. Frontend layout shell: split the new left/right structure in `page.tsx`. Keep existing components in place.
4. Build `<BubbleGraphCanvas>` (new). Test in isolation with a fixed graph fixture before wiring.
5. Wire Graph mode end-to-end with `<BubbleGraphCanvas mode="view">` in the new MainViewer.
6. Build `<HistoryBar>` with the new mixed-kind model, swap in.
7. Rewrite `TopologyEditor` to use `<BubbleGraphCanvas mode="edit">` (delete react-flow imports).
8. Move `<BoundaryCanvas>` into MainViewer, update its container.
9. Remove `@xyflow/react` package + `RoomNode.tsx` + `graph-flow.ts` + old `BubbleGraphView.tsx`.
10. Cross-mode polish: history navigation toasts, mode-draft persistence to sessionStorage.

A separate implementation plan (writing-plans) will break each item into testable PRs.
