# MSD App Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the GSDiff demo app so the MSD `msd_wall` model is fully usable, replace `@xyflow/react` with a pure SVG + `d3-force` bubble-graph component shared across Graph and Topology modes, and reorganize the layout into a slim left controls panel + right workspace with a mixed-kind history bar.

**Architecture:** Slim 320px left panel (controls only) + flex-1 right workspace (main viewer + bottom history bar). All inputs and outputs live on the right. A single `<BubbleGraphCanvas>` component handles both Graph mode (read-only) and Topology mode (editable) via a `mode` prop.

**Tech Stack:** Next.js 16 + React 19 + Tailwind 4 + shadcn (frontend), FastAPI + DiGress (backend). Adds: `d3-force`, `d3-zoom`, `d3-drag`. Removes: `@xyflow/react`.

**Reference spec:** `docs/superpowers/specs/2026-05-06-msd-app-redesign-design.md` (commit 27b7326).

**Important Next.js note:** `app/AGENTS.md` warns that Next.js APIs in this project may differ from training data. Before writing or modifying code in `app/src/app/` (App Router), `app/next.config.ts`, or any code that uses Next-specific features (server components, metadata, fetch caching, route handlers), read the relevant guide in `app/node_modules/next/dist/docs/` first.

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `app/lib/history.ts` | `HistoryItem` discriminated union + helpers (`createGraphItem`, `createFloorplanItem`) |
| `app/lib/bubble-graph/types.ts` | Internal types: `BubbleNodeState`, `BubbleEdgeState`, `BubbleSelection` |
| `app/lib/bubble-graph/use-force-simulation.ts` | React hook wrapping d3-force lifecycle |
| `app/components/bubble-graph/BubbleGraphCanvas.tsx` | Shared bubble graph component (`mode='view' \| 'edit'`) |
| `app/components/bubble-graph/BubbleNode.tsx` | Single node SVG element with drag/click handlers |
| `app/components/bubble-graph/BubbleEdge.tsx` | Single edge SVG element |
| `app/components/HistoryBar.tsx` | Bottom history strip |
| `app/components/HistoryThumb.tsx` | Single thumbnail in history (graph or floorplan) |
| `app/components/MainViewer.tsx` | Right-side workspace router (graph / floorplan / boundary canvas) |
| `app/components/GenerateButton.tsx` | Sticky mode-aware action button |
| `app/src/app/test/bubble-canvas/page.tsx` | Dev fixture route for visual smoke-testing the canvas |

### Modified files

| Path | Change |
|---|---|
| `digress/configs/experiment/msd_wall.yaml` | Conditionally swap `test.checkpoint` V3 → V5 (Task 1) |
| `app/lib/types.ts` | Add `HistoryItem` union, `PageState` shape |
| `app/src/app/page.tsx` | Major restructure: 320px left + flex-1 right workspace |
| `app/components/TopologyEditor.tsx` | Full rewrite: thin shell over `<BubbleGraphCanvas mode="edit">` |
| `app/components/BoundaryCanvas.tsx` | Adapt for placement inside `<MainViewer>` (size + handlers) |
| `app/src/app/globals.css` | Remove `@import "@xyflow/react/dist/style.css"` |
| `app/package.json` | Remove `@xyflow/react`; add `d3-force`, `d3-zoom`, `d3-drag`, `@types/d3-force`, `@types/d3-zoom`, `@types/d3-drag` |

### Deleted files

| Path | Reason |
|---|---|
| `app/components/RoomNode.tsx` | react-flow node, no longer used |
| `app/lib/graph-flow.ts` | react-flow utility, no longer used |
| `app/components/BubbleGraphView.tsx` | Replaced by `<BubbleGraphCanvas mode="view">` |
| `app/components/GraphPanel.tsx` | Folded into left panel + `<MainViewer>` |
| `app/components/ResultPanel.tsx` | Replaced by `<MainViewer>` |
| `app/components/HistoryPanel.tsx` | Replaced by `<HistoryBar>` |

---

## Task Dependency Graph

```
Task 1 (V5 ckpt swap, conditional) ─┐
Task 2 (HistoryItem types) ─────────┼─→ Task 5 (page.tsx restructure)
Task 3 (deps: install d3, remove rf)┘
Task 4 (BubbleGraphCanvas view) ─→ Task 6 (Wire Graph mode) ─→ Task 7 (HistoryBar) ─→ Task 8 (BubbleGraphCanvas edit) ─→ Task 9 (TopologyEditor rewrite)
                                                                                                                                  ↓
                                                                                                                            Task 10 (Boundary into MainViewer)
                                                                                                                                  ↓
                                                                                                                          Task 11 (cleanup + delete dead code)
                                                                                                                                  ↓
                                                                                                                          Task 12 (sessionStorage persistence + cross-mode toast)
```

Each task is one PR. Tasks 1-3 can be done in parallel. Task 4 starts as soon as Task 3 lands. Tasks 5-12 are linear.

---

### Task 1: Backend — Swap MSD checkpoint V3 → V5 (CONDITIONAL)

**Gate:** Run only after the user's 500-sample V5 confirmation run completes. Read `digress/outputs/2026-05-05/13-58-49-msd_wall_v5/test_samples/metrics.json` and compare to V3's metrics. **Skip this task entirely if V5's connectivity is more than 5pp below V3's** — the user will retrain a V7 with `lambda_train=[2.5, 0]` first.

**Files:**
- Modify: `digress/configs/experiment/msd_wall.yaml:35`

- [ ] **Step 1: Verify V5 won the 500-sample shootout**

```bash
cd /d/Github/GSDiff
python -c "
import json
v3 = json.load(open('digress/outputs/2026-05-01/15-25-20-msd_wall/test_samples/metrics.json'))
v5 = json.load(open('digress/outputs/2026-05-05/13-58-49-msd_wall_v5/test_samples/metrics.json'))
print('V3 connected:', v3['generated_summary']['connected_frac'])
print('V5 connected:', v5['generated_summary']['connected_frac'])
print('V3 wall delta abs:', abs(v3['generated_distribution']['edge_type_dist']['wall'] - v3['dataset_distribution']['edge_type_dist']['wall']))
print('V5 wall delta abs:', abs(v5['generated_distribution']['edge_type_dist']['wall'] - v5['dataset_distribution']['edge_type_dist']['wall']))
print()
v5_wins = (v5['generated_summary']['connected_frac'] >= v3['generated_summary']['connected_frac'] - 0.05) and (abs(v5['generated_distribution']['edge_type_dist']['wall'] - v5['dataset_distribution']['edge_type_dist']['wall']) < abs(v3['generated_distribution']['edge_type_dist']['wall'] - v3['dataset_distribution']['edge_type_dist']['wall']))
print('PROCEED with swap?' , v5_wins)
"
```
Expected: `PROCEED with swap? True` to continue. If `False`, STOP — leave V3, do not edit msd_wall.yaml.

- [ ] **Step 2: Edit `digress/configs/experiment/msd_wall.yaml` to point at V5**

Change line 35 from:
```yaml
    checkpoint: 'outputs/2026-05-01/15-25-20-msd_wall/checkpoints/msd_wall/best.ckpt'
```
to:
```yaml
    checkpoint: 'outputs/2026-05-05/13-58-49-msd_wall_v5/checkpoints/msd_wall_v5/best.ckpt'
```

- [ ] **Step 3: Verify the new checkpoint exists and the experiment loads**

```bash
ls digress/outputs/2026-05-05/13-58-49-msd_wall_v5/checkpoints/msd_wall_v5/best.ckpt
cd app/backend && /d/Github/GSDiff/.venv/Scripts/python.exe -c "
from app.services.graph_generation import graph_generator
loaded = graph_generator._load('msd_wall')
print('Checkpoint:', loaded.checkpoint)
print('Room types:', loaded.room_types)
print('Edge types:', loaded.edge_types)
"
```
Expected: shows the V5 ckpt path, lists 9 room types, lists 5 edge types. Loading should take 5-15s.

- [ ] **Step 4: Commit**

```bash
git add digress/configs/experiment/msd_wall.yaml
git commit -m "config: swap msd_wall ckpt V3 -> V5 after 500-sample confirmation"
```

---

### Task 2: Frontend — Add `HistoryItem` types and helpers

**Files:**
- Create: `app/lib/history.ts`
- Modify: `app/lib/types.ts`

- [ ] **Step 1: Create `app/lib/history.ts` with the discriminated union and constructors**

```typescript
import type { DatasetId, GenerationMode } from './constants';
import type { GeneratedGraph } from './types';

export interface GraphHistoryItem {
  id: string;
  kind: 'graph';
  dataset: DatasetId;
  graph: GeneratedGraph;
  createdAt: number;
}

export interface FloorplanHistoryItem {
  id: string;
  kind: 'floorplan';
  dataset: DatasetId;
  image: string;            // data URI
  sourceGraphId?: string;
  sourceMode: GenerationMode;
  createdAt: number;
}

export type HistoryItem = GraphHistoryItem | FloorplanHistoryItem;

const HISTORY_CAP = 30;

let nextId = 0;
function makeId(prefix: string): string {
  nextId += 1;
  return `${prefix}-${Date.now()}-${nextId}`;
}

export function createGraphItem(dataset: DatasetId, graph: GeneratedGraph): GraphHistoryItem {
  return {
    id: makeId('g'),
    kind: 'graph',
    dataset,
    graph,
    createdAt: Date.now(),
  };
}

export function createFloorplanItem(
  dataset: DatasetId,
  image: string,
  sourceMode: GenerationMode,
  sourceGraphId?: string,
): FloorplanHistoryItem {
  return {
    id: makeId('f'),
    kind: 'floorplan',
    dataset,
    image,
    sourceGraphId,
    sourceMode,
    createdAt: Date.now(),
  };
}

/** Prepend a new item; cap to the most recent HISTORY_CAP items. */
export function appendHistory(prev: HistoryItem[], item: HistoryItem): HistoryItem[] {
  return [item, ...prev].slice(0, HISTORY_CAP);
}

export function findHistoryItem(items: HistoryItem[], id: string | null): HistoryItem | undefined {
  if (!id) return undefined;
  return items.find((i) => i.id === id);
}
```

- [ ] **Step 2: Add page-state types to `app/lib/types.ts`**

Append at the bottom of `app/lib/types.ts`:

```typescript
import type { DatasetId, GenerationMode } from './constants';
import type { HistoryItem } from './history';

export interface TopologyDraft {
  graph: GeneratedGraph;
}

export interface BoundaryDraft {
  canvasDataUrl: string;
}

export interface PageState {
  selectedDataset: DatasetId;
  activeMode: GenerationMode;
  history: HistoryItem[];
  selectedHistoryId: string | null;
  modeDrafts: {
    topology?: TopologyDraft;
    boundary?: BoundaryDraft;
  };
  loading: boolean;
  error: string | null;
}
```

- [ ] **Step 3: Type-check**

```bash
cd /d/Github/GSDiff/app && npx tsc --noEmit
```
Expected: no output (success).

- [ ] **Step 4: Commit**

```bash
git add app/lib/history.ts app/lib/types.ts
git commit -m "types: add HistoryItem discriminated union and PageState shape"
```

---

### Task 3: Frontend — Install d3 modules, prepare to remove react-flow

**Files:**
- Modify: `app/package.json`

- [ ] **Step 1: Install d3 modules**

```bash
cd /d/Github/GSDiff/app
npm install d3-force d3-zoom d3-drag d3-selection
npm install --save-dev @types/d3-force @types/d3-zoom @types/d3-drag @types/d3-selection
```

- [ ] **Step 2: Verify installations**

```bash
npm ls d3-force d3-zoom d3-drag d3-selection 2>&1 | grep -E "d3-"
```
Expected: each package shows a version (e.g. `d3-force@3.x.x`).

- [ ] **Step 3: Type-check (no consumers yet, just verifies no breakage)**

```bash
npx tsc --noEmit
```
Expected: no output.

- [ ] **Step 4: Commit**

DO NOT remove `@xyflow/react` yet — components still depend on it. Removal is Task 11.

```bash
git add app/package.json app/package-lock.json
git commit -m "deps: add d3-force/zoom/drag/selection for upcoming bubble graph rewrite"
```

---

### Task 4: Frontend — `<BubbleGraphCanvas>` view-mode (no editing yet)

The component is the heart of the redesign. Build it isolated with a fixture page first, then wire it in Task 6.

**Files:**
- Create: `app/lib/bubble-graph/types.ts`
- Create: `app/lib/bubble-graph/use-force-simulation.ts`
- Create: `app/components/bubble-graph/BubbleGraphCanvas.tsx`
- Create: `app/components/bubble-graph/BubbleNode.tsx`
- Create: `app/components/bubble-graph/BubbleEdge.tsx`
- Create: `app/src/app/test/bubble-canvas/page.tsx` (fixture page)

- [ ] **Step 1: Create internal types**

`app/lib/bubble-graph/types.ts`:
```typescript
export interface BubbleNodeState {
  id: number;
  attr: number;
  x: number;
  y: number;
  fx: number | null;  // fixed x (set when user drags)
  fy: number | null;
}

export interface BubbleEdgeState {
  id: string;          // `${source}-${target}`
  source: number;
  target: number;
  edgeType: number;
}

export interface BubbleSelection {
  nodeId: number | null;
  edgeId: string | null;
}
```

- [ ] **Step 2: Create the d3-force hook**

`app/lib/bubble-graph/use-force-simulation.ts`:
```typescript
import { useEffect, useRef } from 'react';
import { forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide, type Simulation } from 'd3-force';
import type { BubbleNodeState, BubbleEdgeState } from './types';

interface UseForceSimulationParams {
  nodes: BubbleNodeState[];
  edges: BubbleEdgeState[];
  width: number;
  height: number;
  onTick: (nodes: BubbleNodeState[]) => void;
}

/**
 * Lifecycle wrapper around d3-force. The simulation owns the position state;
 * React just renders whatever positions onTick reports.
 *
 * - Simulation is rebuilt when node/edge IDs change (length or set differs).
 * - Simulation does NOT rebuild on simple position updates (drag), to avoid thrash.
 */
export function useForceSimulation({ nodes, edges, width, height, onTick }: UseForceSimulationParams): {
  reheat: () => void;
  pinNode: (nodeId: number, x: number, y: number) => void;
  releaseNode: (nodeId: number) => void;
} {
  const simRef = useRef<Simulation<BubbleNodeState, BubbleEdgeState> | null>(null);
  const nodesRef = useRef<BubbleNodeState[]>(nodes);

  useEffect(() => {
    nodesRef.current = nodes;
    const sim = forceSimulation<BubbleNodeState>(nodes)
      .force('link', forceLink<BubbleNodeState, BubbleEdgeState>(edges).id((d) => d.id).distance(120).strength(0.5))
      .force('charge', forceManyBody<BubbleNodeState>().strength(-400))
      .force('center', forceCenter(width / 2, height / 2))
      .force('collide', forceCollide<BubbleNodeState>(42))
      .on('tick', () => onTick([...nodesRef.current]));
    simRef.current = sim;
    return () => {
      sim.stop();
      simRef.current = null;
    };
    // Rebuild only when node/edge identity changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes.length, edges.length, width, height]);

  const reheat = () => simRef.current?.alpha(0.5).restart();
  // Mutate the simulation's node directly so the next tick respects the new position.
  // The drag handler must call this on the SAME node object the simulation holds —
  // pass a node from the array given to useForceSimulation, NOT a separately-constructed copy.
  const pinNode = (nodeId: number, x: number, y: number) => {
    const node = nodesRef.current.find((n) => n.id === nodeId);
    if (!node) return;
    node.fx = x;
    node.fy = y;
    node.x = x;
    node.y = y;
    simRef.current?.alpha(0.3).restart();
  };
  const releaseNode = (nodeId: number) => {
    const node = nodesRef.current.find((n) => n.id === nodeId);
    if (node) {
      node.fx = null;
      node.fy = null;
    }
  };
  return { reheat, pinNode, releaseNode };
}
```

- [ ] **Step 3: Create `BubbleNode.tsx`**

`app/components/bubble-graph/BubbleNode.tsx`:
```typescript
'use client';

import type { BubbleNodeState } from '@/lib/bubble-graph/types';
import type { RoomTypeMeta } from '@/lib/constants';

interface Props {
  node: BubbleNodeState;
  meta: RoomTypeMeta | undefined;
  selected: boolean;
  onPointerDown?: (e: React.PointerEvent) => void;
  onClick?: (e: React.MouseEvent) => void;
}

export function BubbleNode({ node, meta, selected, onPointerDown, onClick }: Props) {
  const fill = meta?.color ?? '#e5e7eb';
  const textColor = meta?.textColor ?? '#111827';
  const label = meta?.name ?? `class_${node.attr}`;
  return (
    <g
      transform={`translate(${node.x},${node.y})`}
      style={{ cursor: onPointerDown ? 'grab' : 'default' }}
      onPointerDown={onPointerDown}
      onClick={onClick}
    >
      <circle
        r="36"
        fill={fill}
        stroke={selected ? 'oklch(0.55 0.21 35)' : 'oklch(0.2 0 0 / 0.28)'}
        strokeWidth={selected ? 3 : 1.5}
      />
      <text textAnchor="middle" dominantBaseline="middle" fontSize="11" fontWeight="700" fill={textColor} y={-2}>
        {label}
      </text>
      <text textAnchor="middle" dominantBaseline="middle" fontSize="10" fill={textColor} opacity="0.68" y={14}>
        #{node.id}
      </text>
    </g>
  );
}
```

- [ ] **Step 4: Create `BubbleEdge.tsx`**

`app/components/bubble-graph/BubbleEdge.tsx`:
```typescript
'use client';

import type { BubbleEdgeState, BubbleNodeState } from '@/lib/bubble-graph/types';
import type { EdgeTypeMeta } from '@/lib/constants';

interface Props {
  edge: BubbleEdgeState;
  source: BubbleNodeState;
  target: BubbleNodeState;
  meta: EdgeTypeMeta | undefined;
  selected: boolean;
  onClick?: (e: React.MouseEvent) => void;
  onContextMenu?: (e: React.MouseEvent) => void;
}

export function BubbleEdge({ edge, source, target, meta, selected, onClick, onContextMenu }: Props) {
  const stroke = meta?.color ?? '#475569';
  const dashed = meta?.dashed ?? false;
  const showLabel = meta && meta.id > 1; // wall is dominant; only label door/passage/entrance
  return (
    <g onClick={onClick} onContextMenu={onContextMenu} style={{ cursor: onClick ? 'pointer' : 'default' }}>
      <line
        x1={source.x}
        y1={source.y}
        x2={target.x}
        y2={target.y}
        stroke={stroke}
        strokeWidth={selected ? 4 : dashed ? 2.5 : 2}
        strokeDasharray={dashed ? '7 5' : undefined}
        strokeLinecap="round"
      />
      {showLabel && (
        <text
          x={(source.x + target.x) / 2}
          y={(source.y + target.y) / 2 - 6}
          textAnchor="middle"
          fontSize="10"
          fill={stroke}
          paintOrder="stroke"
          stroke="oklch(0.97 0 0)"
          strokeWidth="4"
        >
          {meta?.name}
        </text>
      )}
    </g>
  );
}
```

- [ ] **Step 5: Create `BubbleGraphCanvas.tsx` (view mode only — edit handlers are stubbed)**

`app/components/bubble-graph/BubbleGraphCanvas.tsx`:
```typescript
'use client';

import { useCallback, useMemo, useState } from 'react';
import { DATASET_SPECS, type DatasetId } from '@/lib/constants';
import type { GeneratedGraph } from '@/lib/types';
import type { BubbleNodeState, BubbleEdgeState, BubbleSelection } from '@/lib/bubble-graph/types';
import { useForceSimulation } from '@/lib/bubble-graph/use-force-simulation';
import { BubbleNode } from './BubbleNode';
import { BubbleEdge } from './BubbleEdge';

interface Props {
  graph: GeneratedGraph;
  dataset: DatasetId;
  mode: 'view' | 'edit';
  onChange?: (next: GeneratedGraph) => void;
  defaultNodeAttr?: number;
  defaultEdgeType?: number;
  onSelectionChange?: (sel: BubbleSelection) => void;
  width?: number;
  height?: number;
}

const DEFAULT_WIDTH = 900;
const DEFAULT_HEIGHT = 600;

export function BubbleGraphCanvas({
  graph,
  dataset,
  mode,
  onChange,
  onSelectionChange,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
}: Props) {
  const spec = DATASET_SPECS[dataset];

  const initialNodes: BubbleNodeState[] = useMemo(
    () => graph.nodes.map((n, i) => ({
      id: n.id,
      attr: n.attr,
      x: width / 2 + Math.cos((2 * Math.PI * i) / graph.nodes.length) * 100,
      y: height / 2 + Math.sin((2 * Math.PI * i) / graph.nodes.length) * 100,
      fx: null,
      fy: null,
    })),
    // Re-init only when the graph identity changes, not on each render
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [graph.nodes.length, graph.nodes.map((n) => n.id).join(',')],
  );

  const initialEdges: BubbleEdgeState[] = useMemo(
    () => graph.edges.map((e) => ({
      id: `${e.source}-${e.target}`,
      source: e.source,
      target: e.target,
      edgeType: e.edge_type,
    })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [graph.edges.map((e) => `${e.source}-${e.target}-${e.edge_type}`).join(',')],
  );

  const [nodes, setNodes] = useState<BubbleNodeState[]>(initialNodes);
  const [edges] = useState<BubbleEdgeState[]>(initialEdges);
  const [selection, setSelection] = useState<BubbleSelection>({ nodeId: null, edgeId: null });

  const handleTick = useCallback((next: BubbleNodeState[]) => {
    setNodes([...next]);
  }, []);

  const { pinNode } = useForceSimulation({ nodes: initialNodes, edges: initialEdges, width, height, onTick: handleTick });

  const updateSelection = (next: BubbleSelection) => {
    setSelection(next);
    onSelectionChange?.(next);
  };

  const interactive = mode === 'edit';

  // Drag handler — uses pinNode so the simulation owns position state.
  const dragNode = (nodeId: number) => (e: React.PointerEvent) => {
    e.preventDefault();
    const svg = (e.currentTarget as SVGElement).ownerSVGElement;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const onMove = (ev: PointerEvent) => {
      const x = ((ev.clientX - rect.left) / rect.width) * width;
      const y = ((ev.clientY - rect.top) / rect.height) * height;
      pinNode(nodeId, x, y);
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  const nodeIndex = new Map(nodes.map((n) => [n.id, n]));

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full" role="img" aria-label="Bubble graph">
      <rect width={width} height={height} fill="oklch(0.97 0 0)" rx="18" onClick={() => updateSelection({ nodeId: null, edgeId: null })} />
      <g>
        {edges.map((edge) => {
          const source = nodeIndex.get(edge.source);
          const target = nodeIndex.get(edge.target);
          if (!source || !target) return null;
          const meta = spec.edgeTypes.find((m) => m.id === edge.edgeType);
          return (
            <BubbleEdge
              key={edge.id}
              edge={edge}
              source={source}
              target={target}
              meta={meta}
              selected={selection.edgeId === edge.id}
              onClick={interactive ? () => updateSelection({ nodeId: null, edgeId: edge.id }) : undefined}
            />
          );
        })}
      </g>
      <g>
        {nodes.map((node) => {
          const meta = spec.roomTypes.find((m) => m.id === node.attr);
          return (
            <BubbleNode
              key={node.id}
              node={node}
              meta={meta}
              selected={selection.nodeId === node.id}
              onPointerDown={dragNode(node.id)}
              onClick={interactive ? (e) => { e.stopPropagation(); updateSelection({ nodeId: node.id, edgeId: null }); } : undefined}
            />
          );
        })}
      </g>
      {/* Edit-mode interactions added in Task 8 */}
    </svg>
  );
}
```

- [ ] **Step 6: Create the fixture page**

`app/src/app/test/bubble-canvas/page.tsx`:
```typescript
'use client';

import { BubbleGraphCanvas } from '@/components/bubble-graph/BubbleGraphCanvas';
import type { GeneratedGraph } from '@/lib/types';

const FIXTURE_RPLAN: GeneratedGraph = {
  num_nodes: 6,
  num_edges: 7,
  rooms: [0, 1, 1, 2, 3, 4],
  adjacency: [],
  edge_types: [],
  nodes: [
    { id: 0, attr: 0, room_type: 'living' },
    { id: 1, attr: 1, room_type: 'bedroom' },
    { id: 2, attr: 1, room_type: 'bedroom' },
    { id: 3, attr: 2, room_type: 'bathroom' },
    { id: 4, attr: 3, room_type: 'kitchen' },
    { id: 5, attr: 4, room_type: 'balcony' },
  ],
  edges: [
    { source: 0, target: 1, edge_type: 2, edge_label: 'door' },
    { source: 0, target: 2, edge_type: 2, edge_label: 'door' },
    { source: 0, target: 3, edge_type: 1, edge_label: 'wall' },
    { source: 0, target: 4, edge_type: 2, edge_label: 'door' },
    { source: 0, target: 5, edge_type: 2, edge_label: 'door' },
    { source: 1, target: 3, edge_type: 1, edge_label: 'wall' },
    { source: 4, target: 5, edge_type: 1, edge_label: 'wall' },
  ],
};

export default function BubbleCanvasFixture() {
  return (
    <div className="min-h-screen bg-background p-8">
      <h1 className="mb-4 text-xl font-semibold">BubbleGraphCanvas — view mode (RPLAN fixture)</h1>
      <div className="h-[600px] w-full max-w-[900px] rounded-xl border bg-card p-4">
        <BubbleGraphCanvas graph={FIXTURE_RPLAN} dataset="rplan" mode="view" />
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Type-check + manual smoke test**

```bash
cd /d/Github/GSDiff/app && npx tsc --noEmit
```
Expected: no output.

Then start dev server:
```bash
npm run dev
```

Open `http://localhost:3000/test/bubble-canvas` in browser. Expected:
- 6 colored circles arranged in a force-directed layout (living room near center)
- 7 edges between them (some dashed for `door`)
- Drag a node — it sticks where you dropped it; other nodes settle around it
- Page does not crash on hot reload

- [ ] **Step 8: Commit**

```bash
git add app/lib/bubble-graph/ app/components/bubble-graph/ app/src/app/test/
git commit -m "feat(canvas): add BubbleGraphCanvas view-mode + d3-force hook + fixture page"
```

---

### Task 5: Frontend — Restructure `page.tsx` layout (left 320px + right workspace)

This is the biggest visual change. Keep all existing components plugged in via the new layout — no behavior changes yet, just restructured shell. Graph panel still uses old `BubbleGraphView` until Task 6.

**Files:**
- Modify: `app/src/app/page.tsx`
- Create: `app/components/MainViewer.tsx`
- Create: `app/components/GenerateButton.tsx`

**Pre-step:** Read `app/node_modules/next/dist/docs/01-app/01-getting-started/03-layouts-and-pages.mdx` (if relevant to your changes). The `app/src/app/page.tsx` is already a Client Component (`'use client'`); just structural changes to its return JSX.

- [ ] **Step 1: Create skeleton `MainViewer.tsx`**

`app/components/MainViewer.tsx`:
```typescript
'use client';

import { ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

/**
 * Right-workspace main viewer slot. Just a styled container for now;
 * Task 6 wires it to (mode, selectedHistoryItem, modeDrafts) routing.
 */
export function MainViewer({ children }: Props) {
  return (
    <div className="flex flex-1 items-center justify-center overflow-hidden bg-background p-6">
      <div className="h-full w-full max-w-[1200px]">
        {children}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create `GenerateButton.tsx`**

`app/components/GenerateButton.tsx`:
```typescript
'use client';

import { Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { GenerationMode } from '@/lib/constants';

interface Props {
  mode: GenerationMode;
  loading: boolean;
  onClick: () => void;
  disabled?: boolean;
}

const LABEL: Record<GenerationMode, string> = {
  unconstrained: 'Generate Floorplan',
  graph: 'Sample Graph',
  topology: 'Generate Floorplan',
  boundary: 'Generate Floorplan',
};

export function GenerateButton({ mode, loading, onClick, disabled }: Props) {
  return (
    <Button onClick={onClick} disabled={disabled || loading} className="h-11 w-full text-sm font-medium">
      {loading ? (
        <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-background/30 border-t-background" />
      ) : (
        <Sparkles className="mr-2 h-4 w-4" />
      )}
      {LABEL[mode]}
    </Button>
  );
}
```

- [ ] **Step 3: Restructure `page.tsx` — slim left + right workspace**

Replace the body of `app/src/app/page.tsx` (keep imports + state hooks; only rewrite the `return (...)` JSX). The new return:

```typescript
  return (
    <div className="flex h-full flex-col">
      <Header />

      <div className="flex flex-1 overflow-hidden">
        {/* Left controls panel (320px) */}
        <aside className="flex w-[320px] shrink-0 flex-col border-r border-border/60 bg-card">
          <div className="flex-1 space-y-4 overflow-auto p-4">
            <DatasetSelector value={selectedDataset} onChange={setSelectedDataset} />
            <ModeSelector activeMode={activeMode} onModeChange={setActiveMode} />
            <ModelStatusPanel />

            {/* Mode-specific controls — slim sidebars without canvas/editor for now */}
            {activeMode === 'graph' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Sample bubble graphs from {selectedDatasetMeta.name}.
              </div>
            )}
            {activeMode === 'topology' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Topology editor will move into the right workspace (Task 9).
              </div>
            )}
            {activeMode === 'boundary' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Boundary canvas will move into the right workspace (Task 10).
              </div>
            )}
          </div>

          <div className="border-t border-border/60 p-4">
            <GenerateButton
              mode={activeMode}
              loading={loading}
              onClick={() => {
                if (activeMode === 'unconstrained') handleUnconstrained();
                // Other modes get wired in subsequent tasks
              }}
              disabled={activeMode !== 'unconstrained'}
            />
          </div>
        </aside>

        {/* Right workspace */}
        <div className="flex flex-1 flex-col overflow-hidden">
          <MainViewer>
            <ResultPanel image={image} loading={loading} error={error} dataset={selectedDataset} />
          </MainViewer>

          {/* History bar (placeholder using existing HistoryPanel until Task 7) */}
          {history.length > 0 && (
            <HistoryPanel
              history={history}
              onSelect={handleSelectHistory}
              onClear={handleClearHistory}
              currentImage={image}
            />
          )}
        </div>
      </div>
    </div>
  );
```

Add the new imports near the top:
```typescript
import { MainViewer } from '@/components/MainViewer';
import { GenerateButton } from '@/components/GenerateButton';
```

Remove the old imports for `UnconstrainedPanel`, `GraphPanel`, `TopologyEditor`, `BoundaryCanvas` — they're temporarily disabled in this layout (modes other than `unconstrained` won't generate; subsequent tasks restore them via the new flow).

- [ ] **Step 4: Type-check + lint**

```bash
cd /d/Github/GSDiff/app && npx tsc --noEmit && npm run lint
```
Expected: no type errors, lint passes (or only warnings).

- [ ] **Step 5: Manual smoke test**

```bash
npm run dev
```

Open `http://localhost:3000`. Expected:
- Left panel is ~320px wide with Dataset/Mode/ModelStatus + a placeholder for mode-specific area + sticky "Generate" button at the bottom
- Right panel is much wider, shows current ResultPanel content (image / loading state / empty state)
- History bar appears at the bottom only after first generation
- "Generate Floorplan" button works in Unconstrained mode (calls existing handler); other modes show "wiring in subsequent tasks" placeholders

- [ ] **Step 6: Commit**

```bash
git add app/src/app/page.tsx app/components/MainViewer.tsx app/components/GenerateButton.tsx
git commit -m "ui: restructure page layout to 320px left + flex-1 right workspace"
```

---

### Task 6: Frontend — Wire Graph mode end-to-end with new BubbleGraphCanvas + history

Convert from the old `GraphPanel`/`BubbleGraphView` flow to the new model: Graph mode generates a graph, adds it to history, displays it in MainViewer. "Send to Floorplan" button posts to topology endpoint and adds the floorplan to history.

**Files:**
- Modify: `app/src/app/page.tsx`
- Modify: `app/components/MainViewer.tsx`

- [ ] **Step 1: Replace page state with the new `PageState` shape**

In `app/src/app/page.tsx`:

Replace:
```typescript
const [image, setImage] = useState<string | null>(null);
const [loading, setLoading] = useState(false);
const [error, setError] = useState<string | null>(null);
const [history, setHistory] = useState<HistoryItem[]>([]);
```
with:
```typescript
import {
  appendHistory,
  createGraphItem,
  createFloorplanItem,
  findHistoryItem,
  type HistoryItem,
  type GraphHistoryItem,
} from '@/lib/history';

const [history, setHistory] = useState<HistoryItem[]>([]);
const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
const [loading, setLoading] = useState(false);
const [error, setError] = useState<string | null>(null);
```

Remove the old `HistoryItem` import from `@/lib/types` if it conflicts (the new one comes from `@/lib/history`).

- [ ] **Step 2: Update handlers to write into the new history**

Replace `handleUnconstrained`:
```typescript
const handleUnconstrained = useCallback(async () => {
  setLoading(true);
  setError(null);
  try {
    const res = await generateUnconstrained();
    const item = createFloorplanItem(selectedDataset, res.image, 'unconstrained');
    setHistory((prev) => appendHistory(prev, item));
    setSelectedHistoryId(item.id);
  } catch (e) {
    setError(e instanceof Error ? e.message : 'Generation failed');
  } finally {
    setLoading(false);
  }
}, [selectedDataset]);
```

Add `handleSampleGraph`:
```typescript
const handleSampleGraph = useCallback(async () => {
  setLoading(true);
  setError(null);
  try {
    const res = await generateGraph(selectedDataset);
    const graph = res.graphs[0];
    if (!graph) throw new Error('Server returned no graph');
    const item = createGraphItem(selectedDataset, graph);
    setHistory((prev) => appendHistory(prev, item));
    setSelectedHistoryId(item.id);
  } catch (e) {
    setError(e instanceof Error ? e.message : 'Graph generation failed');
  } finally {
    setLoading(false);
  }
}, [selectedDataset]);
```

Add `handleSendGraphToFloorplan`:
```typescript
const handleSendGraphToFloorplan = useCallback(async (graphItem: GraphHistoryItem) => {
  setLoading(true);
  setError(null);
  try {
    const res = await generateTopology(graphItem.graph.rooms, graphItem.graph.adjacency);
    const item = createFloorplanItem(selectedDataset, res.image, 'graph', graphItem.id);
    setHistory((prev) => appendHistory(prev, item));
    setSelectedHistoryId(item.id);
  } catch (e) {
    setError(e instanceof Error ? e.message : 'Generation failed');
  } finally {
    setLoading(false);
  }
}, [selectedDataset]);
```

- [ ] **Step 3: Make `MainViewer` render based on the selected history item + mode**

Update `MainViewer.tsx`:
```typescript
'use client';

import { ImageIcon, Network } from 'lucide-react';
import { BubbleGraphCanvas } from '@/components/bubble-graph/BubbleGraphCanvas';
import type { GenerationMode } from '@/lib/constants';
import type { HistoryItem } from '@/lib/history';

interface Props {
  mode: GenerationMode;
  selectedItem: HistoryItem | undefined;
  loading: boolean;
  error: string | null;
}

export function MainViewer({ mode, selectedItem, loading, error }: Props) {
  return (
    <div className="flex flex-1 items-center justify-center overflow-hidden bg-background p-6">
      <div className="flex h-full w-full max-w-[1200px] items-center justify-center">
        {error && (
          <div className="rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}
        {!error && loading && (
          <div className="flex flex-col items-center gap-3">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-muted-foreground/20 border-t-muted-foreground/60" />
            <p className="text-sm text-muted-foreground">Generating...</p>
          </div>
        )}
        {!error && !loading && !selectedItem && <EmptyState mode={mode} />}
        {!error && !loading && selectedItem && (
          selectedItem.kind === 'graph' ? (
            <BubbleGraphCanvas graph={selectedItem.graph} dataset={selectedItem.dataset} mode="view" />
          ) : (
            <img src={selectedItem.image} alt="Generated floorplan" className="max-h-full max-w-full rounded-xl border border-border/60 shadow-sm" />
          )
        )}
      </div>
    </div>
  );
}

function EmptyState({ mode }: { mode: GenerationMode }) {
  const Icon = mode === 'graph' ? Network : ImageIcon;
  const text = mode === 'graph'
    ? 'Click Sample Graph to generate a bubble diagram.'
    : 'Click Generate to sample a floorplan.';
  return (
    <div className="flex flex-col items-center gap-3 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-muted">
        <Icon className="h-7 w-7 text-muted-foreground/40" />
      </div>
      <p className="text-sm text-muted-foreground">{text}</p>
    </div>
  );
}
```

- [ ] **Step 4: Update `page.tsx` to use the new MainViewer + wire `handleSampleGraph`**

In `page.tsx`:
- Import `findHistoryItem` from `@/lib/history`
- Compute `selectedItem = findHistoryItem(history, selectedHistoryId)`
- Pass `mode`, `selectedItem`, `loading`, `error` to `<MainViewer>`
- Wire `<GenerateButton>` so Graph mode calls `handleSampleGraph`, Unconstrained calls `handleUnconstrained`. Other modes still disabled.
- Add a "Send to Floorplan" button just below the MainViewer (or inline in the right column) that's enabled only when `selectedItem?.kind === 'graph'`. Clicking it calls `handleSendGraphToFloorplan(selectedItem)`.

- [ ] **Step 5: Remove the old `<HistoryPanel>` placeholder**

It still works visually but takes the legacy `HistoryItem` shape (image + mode). Replace with a no-op until Task 7:
```typescript
{history.length > 0 && (
  <div className="border-t border-border/60 bg-card px-4 py-2 text-xs text-muted-foreground">
    History: {history.length} item(s) — bar wired up in Task 7
  </div>
)}
```

- [ ] **Step 6: Type-check + smoke**

```bash
cd /d/Github/GSDiff/app && npx tsc --noEmit
```
Expected: no errors. (`HistoryItem` from `@/lib/history` is now used; the old import from `@/lib/types` should be gone.)

```bash
npm run dev
```

Open `http://localhost:3000`. Test sequence:
1. Pick RPLAN dataset, switch to Graph mode → click "Sample Graph" → wait → bubble graph appears in main viewer
2. Click "Send to Floorplan" → wait → floorplan appears in main viewer
3. Pick MSD dataset → repeat (first MSD sample loads the model, ~5-15s extra)

- [ ] **Step 7: Commit**

```bash
git add app/src/app/page.tsx app/components/MainViewer.tsx
git commit -m "feat(graph): wire Graph mode through new MainViewer + history shape"
```

---

### Task 7: Frontend — Build `<HistoryBar>` and replace the placeholder

**Files:**
- Create: `app/components/HistoryBar.tsx`
- Create: `app/components/HistoryThumb.tsx`
- Modify: `app/src/app/page.tsx`

- [ ] **Step 1: Build `HistoryThumb.tsx`**

`app/components/HistoryThumb.tsx`:
```typescript
'use client';

import { cn } from '@/lib/utils';
import type { HistoryItem } from '@/lib/history';
import { DATASET_SPECS } from '@/lib/constants';

interface Props {
  item: HistoryItem;
  selected: boolean;
  onClick: () => void;
}

export function HistoryThumb({ item, selected, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'relative flex h-20 w-20 shrink-0 items-center justify-center rounded-lg border-2 bg-muted transition-colors',
        selected ? 'border-foreground' : 'border-border/60 hover:border-foreground/40',
      )}
      title={`${item.kind} · ${new Date(item.createdAt).toLocaleTimeString()}`}
    >
      {item.kind === 'floorplan' ? (
        <img src={item.image} alt="" className="h-full w-full rounded object-cover" />
      ) : (
        <MiniGraphPreview item={item} />
      )}
      <span className="absolute bottom-0.5 left-1 text-[9px] font-bold text-foreground/70">
        {item.kind === 'graph' ? '⬡' : '⬜'}
      </span>
    </button>
  );
}

function MiniGraphPreview({ item }: { item: Extract<HistoryItem, { kind: 'graph' }> }) {
  // 80x80 SVG with nodes laid out in a small circle
  const spec = DATASET_SPECS[item.dataset];
  const cx = 40;
  const cy = 40;
  const r = 26;
  const n = item.graph.nodes.length;
  return (
    <svg viewBox="0 0 80 80" className="h-full w-full">
      {item.graph.nodes.map((node, i) => {
        const angle = (2 * Math.PI * i) / Math.max(1, n);
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        const meta = spec.roomTypes.find((m) => m.id === node.attr);
        return <circle key={node.id} cx={x} cy={y} r="4" fill={meta?.color ?? '#9ca3af'} />;
      })}
    </svg>
  );
}
```

- [ ] **Step 2: Build `HistoryBar.tsx`**

`app/components/HistoryBar.tsx`:
```typescript
'use client';

import { ReactNode } from 'react';
import { Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { HistoryThumb } from './HistoryThumb';
import type { HistoryItem } from '@/lib/history';

interface Props {
  items: HistoryItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onClear?: () => void;
  rightSlot?: ReactNode;
}

export function HistoryBar({ items, selectedId, onSelect, onClear, rightSlot }: Props) {
  return (
    <div className="flex items-center gap-3 border-t border-border/60 bg-card px-4 py-3">
      <div className="flex flex-1 gap-2 overflow-x-auto">
        {items.length === 0 && (
          <p className="text-xs text-muted-foreground">No items yet — generate something to populate the history.</p>
        )}
        {items.map((item) => (
          <HistoryThumb
            key={item.id}
            item={item}
            selected={selectedId === item.id}
            onClick={() => onSelect(item.id)}
          />
        ))}
      </div>
      <div className="flex items-center gap-2">
        {rightSlot}
        {items.length > 0 && onClear && (
          <Button variant="ghost" size="icon" onClick={onClear} title="Clear all">
            <Trash2 className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Wire it into `page.tsx`**

Replace the placeholder div from Task 6 step 5 with:
```typescript
<HistoryBar
  items={history}
  selectedId={selectedHistoryId}
  onSelect={setSelectedHistoryId}
  onClear={() => { setHistory([]); setSelectedHistoryId(null); }}
  rightSlot={selectedItem?.kind === 'graph' ? (
    <Button onClick={() => handleSendGraphToFloorplan(selectedItem)} disabled={loading} size="sm">
      → Send to Floorplan
    </Button>
  ) : null}
/>
```

Add the import:
```typescript
import { HistoryBar } from '@/components/HistoryBar';
import { Button } from '@/components/ui/button';
```

- [ ] **Step 4: Type-check + smoke**

```bash
cd /d/Github/GSDiff/app && npx tsc --noEmit && npm run dev
```

Open `http://localhost:3000`. Test:
1. Generate two graphs → both thumbnails appear in history bar at bottom
2. Click first thumbnail → main viewer switches to it
3. Click "Send to Floorplan" → new floorplan thumbnail appears with `⬜` icon
4. Click between graph and floorplan thumbnails → main viewer swaps content
5. Click trash icon → history clears

- [ ] **Step 5: Commit**

```bash
git add app/components/HistoryBar.tsx app/components/HistoryThumb.tsx app/src/app/page.tsx
git commit -m "feat(history): add HistoryBar with mixed graph/floorplan thumbnails"
```

---

### Task 8: Frontend — Add edit-mode to `<BubbleGraphCanvas>`

Adds: click-to-add node, click-to-select edge, click-to-cycle edge type, double-click-to-change room type, delete via DEL key, undo/redo via Ctrl+Z/Y, drag-from-perimeter to add edges.

**Files:**
- Modify: `app/components/bubble-graph/BubbleGraphCanvas.tsx`
- Modify: `app/src/app/test/bubble-canvas/page.tsx` (add an edit-mode section)

- [ ] **Step 1: Refactor `BubbleGraphCanvas.tsx` to manage editable state**

Wrap the existing nodes/edges state in a small reducer with undo/redo:

Inside `BubbleGraphCanvas.tsx`, replace the `useState<BubbleEdgeState[]>` line with:
```typescript
import { useReducer, useRef, useEffect } from 'react';

type EditAction =
  | { type: 'add-node'; attr: number; x: number; y: number }
  | { type: 'delete-node'; id: number }
  | { type: 'change-node-attr'; id: number; attr: number }
  | { type: 'add-edge'; source: number; target: number; edgeType: number }
  | { type: 'delete-edge'; id: string }
  | { type: 'cycle-edge-type'; id: string; nextType: number };

interface EditState {
  nodes: BubbleNodeState[];
  edges: BubbleEdgeState[];
  past: { nodes: BubbleNodeState[]; edges: BubbleEdgeState[] }[];
  future: { nodes: BubbleNodeState[]; edges: BubbleEdgeState[] }[];
}

const MAX_UNDO = 50;
function pushPast(state: EditState): EditState {
  const past = [...state.past, { nodes: state.nodes, edges: state.edges }].slice(-MAX_UNDO);
  return { ...state, past, future: [] };
}

function reducer(state: EditState, action: EditAction | { type: 'undo' } | { type: 'redo' } | { type: 'tick'; nodes: BubbleNodeState[] }): EditState {
  if (action.type === 'tick') {
    // Position-only update from d3-force; not undoable
    return { ...state, nodes: action.nodes };
  }
  if (action.type === 'undo') {
    if (state.past.length === 0) return state;
    const prev = state.past[state.past.length - 1];
    return {
      ...state,
      nodes: prev.nodes,
      edges: prev.edges,
      past: state.past.slice(0, -1),
      future: [{ nodes: state.nodes, edges: state.edges }, ...state.future],
    };
  }
  if (action.type === 'redo') {
    if (state.future.length === 0) return state;
    const next = state.future[0];
    return {
      ...state,
      nodes: next.nodes,
      edges: next.edges,
      past: [...state.past, { nodes: state.nodes, edges: state.edges }].slice(-MAX_UNDO),
      future: state.future.slice(1),
    };
  }
  const checkpointed = pushPast(state);
  switch (action.type) {
    case 'add-node': {
      const newId = (state.nodes.reduce((m, n) => Math.max(m, n.id), -1)) + 1;
      const node: BubbleNodeState = { id: newId, attr: action.attr, x: action.x, y: action.y, fx: action.x, fy: action.y };
      return { ...checkpointed, nodes: [...state.nodes, node] };
    }
    case 'delete-node': {
      return {
        ...checkpointed,
        nodes: state.nodes.filter((n) => n.id !== action.id),
        edges: state.edges.filter((e) => e.source !== action.id && e.target !== action.id),
      };
    }
    case 'change-node-attr': {
      return { ...checkpointed, nodes: state.nodes.map((n) => (n.id === action.id ? { ...n, attr: action.attr } : n)) };
    }
    case 'add-edge': {
      const id = `${Math.min(action.source, action.target)}-${Math.max(action.source, action.target)}`;
      if (state.edges.some((e) => e.id === id)) return state;
      return { ...checkpointed, edges: [...state.edges, { id, source: action.source, target: action.target, edgeType: action.edgeType }] };
    }
    case 'delete-edge': {
      return { ...checkpointed, edges: state.edges.filter((e) => e.id !== action.id) };
    }
    case 'cycle-edge-type': {
      return { ...checkpointed, edges: state.edges.map((e) => (e.id === action.id ? { ...e, edgeType: action.nextType } : e)) };
    }
  }
}
```

- [ ] **Step 2: Use the reducer in `BubbleGraphCanvas` for edit mode**

In edit mode, replace `setNodes` etc. with `dispatch`. Keep view mode using the original simple `useState` path (don't pay the reducer cost when editing isn't needed).

```typescript
const [editState, dispatch] = useReducer(reducer, { nodes: initialNodes, edges: initialEdges, past: [], future: [] });
const editing = mode === 'edit';
const nodes = editing ? editState.nodes : viewNodes;  // viewNodes is the existing useState
const edges = editing ? editState.edges : initialEdges;
```

When in edit mode, propagate `editState.nodes`/`editState.edges` back via `onChange`:
```typescript
useEffect(() => {
  if (!editing || !onChange) return;
  // Convert back to GeneratedGraph shape
  const next: GeneratedGraph = {
    num_nodes: editState.nodes.length,
    num_edges: editState.edges.length,
    rooms: editState.nodes.map((n) => n.attr),
    adjacency: [],   // recompute if needed; or leave to consumer
    edge_types: [],
    nodes: editState.nodes.map((n) => ({ id: n.id, attr: n.attr, room_type: spec.roomTypes.find((m) => m.id === n.attr)?.name ?? `class_${n.attr}` })),
    edges: editState.edges.map((e) => ({ source: e.source, target: e.target, edge_type: e.edgeType, edge_label: spec.edgeTypes.find((m) => m.id === e.edgeType)?.name ?? `class_${e.edgeType}` })),
  };
  onChange(next);
}, [editing, editState.nodes, editState.edges, onChange, spec]);
```

- [ ] **Step 3: Add edit-mode interactions**

Inside the SVG, when `editing`:
- Background `<rect>` `onClick` adds a node at the click coordinates with `defaultNodeAttr`. Use `e.nativeEvent.offsetX/Y` and convert to viewBox coords.
- Node `onDoubleClick` triggers `handleChangeNodeAttr` — for v1, just bump `attr` to `(attr + 1) % spec.roomTypes.length`. (Color cycle. Real popover later.)
- Edge `onClick` cycles to the next non-`none` edge type (`cycle-edge-type` action). Compute `nextType` from current via:
  ```typescript
  const nonNoneTypes = spec.edgeTypes.filter((m) => m.id !== 0).map((m) => m.id);
  const idx = nonNoneTypes.indexOf(currentEdgeType);
  const nextType = nonNoneTypes[(idx + 1) % nonNoneTypes.length];
  ```
- Edge `onContextMenu` (right-click) → `e.preventDefault(); dispatch({type:'delete-edge',id})`
- Selected node + DEL key → `dispatch({type:'delete-node',id})`
  - Implement via `useEffect` listening to `keydown` on `window`, gated by `editing && selection.nodeId !== null`
- Ctrl+Z / Ctrl+Y → undo/redo (also via window keydown effect)

- [ ] **Step 4: Add edge-creation drag interaction**

When user pointer-downs on a node's perimeter (radius > 30 from center), start an "edge creation" mode:
- Show a dashed rubber-band line from the source node to the cursor
- On pointer-up over another node → `dispatch({type:'add-edge', source, target, edgeType: defaultEdgeType ?? 1})`
- On pointer-up elsewhere → cancel

Implementation sketch (inside BubbleGraphCanvas):
```typescript
const [edgeDraft, setEdgeDraft] = useState<{ sourceId: number; cursor: { x: number; y: number } } | null>(null);

const handleNodePointerDown = (nodeId: number) => (e: React.PointerEvent) => {
  if (!editing) return dragNode(nodeId)(e);
  const target = nodes.find((n) => n.id === nodeId);
  if (!target) return;
  const rect = (e.currentTarget as SVGElement).ownerSVGElement!.getBoundingClientRect();
  const px = ((e.clientX - rect.left) / rect.width) * width;
  const py = ((e.clientY - rect.top) / rect.height) * height;
  const dist = Math.hypot(px - target.x, py - target.y);
  if (dist > 30) {
    // Start edge-drag
    setEdgeDraft({ sourceId: nodeId, cursor: { x: px, y: py } });
    const onMove = (ev: PointerEvent) => {
      const cx = ((ev.clientX - rect.left) / rect.width) * width;
      const cy = ((ev.clientY - rect.top) / rect.height) * height;
      setEdgeDraft({ sourceId: nodeId, cursor: { x: cx, y: cy } });
    };
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      // Hit-test against nodes
      const cx = ((ev.clientX - rect.left) / rect.width) * width;
      const cy = ((ev.clientY - rect.top) / rect.height) * height;
      const hit = nodes.find((n) => n.id !== nodeId && Math.hypot(n.x - cx, n.y - cy) < 36);
      if (hit) dispatch({ type: 'add-edge', source: nodeId, target: hit.id, edgeType: defaultEdgeType ?? 1 });
      setEdgeDraft(null);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  } else {
    dragNode(nodeId)(e);
  }
};
```

Render the rubber-band:
```typescript
{edgeDraft && (() => {
  const src = nodes.find((n) => n.id === edgeDraft.sourceId);
  if (!src) return null;
  return <line x1={src.x} y1={src.y} x2={edgeDraft.cursor.x} y2={edgeDraft.cursor.y} stroke="#94a3b8" strokeWidth="2" strokeDasharray="4 4" />;
})()}
```

- [ ] **Step 5: Add edit-mode demo to fixture**

In `app/src/app/test/bubble-canvas/page.tsx`, add a second section below the view-mode example:

```typescript
function EditModeDemo() {
  const [graph, setGraph] = useState<GeneratedGraph>(FIXTURE_RPLAN);
  return (
    <div className="mt-8">
      <h2 className="mb-4 text-lg font-semibold">Edit mode (RPLAN fixture)</h2>
      <p className="mb-2 text-sm text-muted-foreground">
        Click empty space to add. Click node to select, DEL to delete. Drag node perimeter to another node to add edge.
        Click edge to cycle type, right-click to delete. Ctrl+Z/Y for undo/redo.
      </p>
      <div className="h-[600px] w-full max-w-[900px] rounded-xl border bg-card p-4">
        <BubbleGraphCanvas graph={graph} dataset="rplan" mode="edit" onChange={setGraph} defaultNodeAttr={1} defaultEdgeType={1} />
      </div>
      <pre className="mt-2 max-h-32 overflow-auto rounded bg-muted p-2 text-xs">{JSON.stringify({ n: graph.num_nodes, e: graph.num_edges }, null, 2)}</pre>
    </div>
  );
}

// And render <EditModeDemo /> below the existing view-mode block
```

- [ ] **Step 6: Type-check + manual edit smoke test**

```bash
cd /d/Github/GSDiff/app && npx tsc --noEmit
npm run dev
```

Open `http://localhost:3000/test/bubble-canvas`. Test the editing interactions listed in step 5's instructions. The status JSON below should update on every change.

- [ ] **Step 7: Commit**

```bash
git add app/components/bubble-graph/ app/src/app/test/bubble-canvas/page.tsx
git commit -m "feat(canvas): add edit mode with undo/redo, edge-drag, and full interactions"
```

---

### Task 9: Frontend — Rewrite `TopologyEditor` to use `<BubbleGraphCanvas mode="edit">`

**Files:**
- Modify: `app/components/TopologyEditor.tsx` (full rewrite)
- Modify: `app/src/app/page.tsx` (wire topology mode)

- [ ] **Step 1: Define a default initial graph for topology mode**

At the top of `app/components/TopologyEditor.tsx` (after imports):
```typescript
const DEFAULT_TOPOLOGY_GRAPH: GeneratedGraph = {
  num_nodes: 4,
  num_edges: 3,
  rooms: [0, 1, 2, 3],
  adjacency: [],
  edge_types: [],
  nodes: [
    { id: 0, attr: 0, room_type: 'living' },
    { id: 1, attr: 1, room_type: 'bedroom' },
    { id: 2, attr: 2, room_type: 'bathroom' },
    { id: 3, attr: 3, room_type: 'kitchen' },
  ],
  edges: [
    { source: 0, target: 1, edge_type: 2, edge_label: 'door' },
    { source: 0, target: 2, edge_type: 1, edge_label: 'wall' },
    { source: 0, target: 3, edge_type: 2, edge_label: 'door' },
  ],
};
```

(For MSD this default is fine — RPLAN uses attrs 0-5, MSD uses 0-8, but `attr=0` is "living" in RPLAN and "Bedroom" in MSD. Keep this generic; the actual attrs depend on selected dataset, but the default looks reasonable in both.)

- [ ] **Step 2: Rewrite the file**

Replace the entire `app/components/TopologyEditor.tsx`:
```typescript
'use client';

import { useEffect, useState } from 'react';
import { BubbleGraphCanvas } from '@/components/bubble-graph/BubbleGraphCanvas';
import type { DatasetId } from '@/lib/constants';
import type { GeneratedGraph } from '@/lib/types';

interface Props {
  dataset: DatasetId;
  initialGraph?: GeneratedGraph;
  defaultNodeAttr?: number;
  defaultEdgeType?: number;
  onGraphChange: (graph: GeneratedGraph) => void;
}

const DEFAULT_TOPOLOGY_GRAPH: GeneratedGraph = {
  num_nodes: 4,
  num_edges: 3,
  rooms: [0, 1, 2, 3],
  adjacency: [],
  edge_types: [],
  nodes: [
    { id: 0, attr: 0, room_type: 'living' },
    { id: 1, attr: 1, room_type: 'bedroom' },
    { id: 2, attr: 2, room_type: 'bathroom' },
    { id: 3, attr: 3, room_type: 'kitchen' },
  ],
  edges: [
    { source: 0, target: 1, edge_type: 2, edge_label: 'door' },
    { source: 0, target: 2, edge_type: 1, edge_label: 'wall' },
    { source: 0, target: 3, edge_type: 2, edge_label: 'door' },
  ],
};

export function TopologyEditor({ dataset, initialGraph, defaultNodeAttr, defaultEdgeType, onGraphChange }: Props) {
  const [graph, setGraph] = useState<GeneratedGraph>(initialGraph ?? DEFAULT_TOPOLOGY_GRAPH);

  useEffect(() => {
    onGraphChange(graph);
  }, [graph, onGraphChange]);

  return (
    <BubbleGraphCanvas
      graph={graph}
      dataset={dataset}
      mode="edit"
      onChange={setGraph}
      defaultNodeAttr={defaultNodeAttr}
      defaultEdgeType={defaultEdgeType}
    />
  );
}

export { DEFAULT_TOPOLOGY_GRAPH };
```

- [ ] **Step 3: Wire into `page.tsx` MainViewer**

In `MainViewer.tsx`, add a topology branch:
```typescript
import { TopologyEditor } from '@/components/TopologyEditor';

// In Props, add:
//   topologyDraft?: GeneratedGraph;
//   onTopologyChange?: (g: GeneratedGraph) => void;

// In the render switch, BEFORE the selectedItem check:
if (mode === 'topology') {
  return (
    <div className="flex flex-1 items-center justify-center overflow-hidden bg-background p-6">
      <div className="h-full w-full max-w-[1200px]">
        <TopologyEditor dataset={...} initialGraph={topologyDraft} onGraphChange={onTopologyChange ?? (() => {})} />
      </div>
    </div>
  );
}
```

In `page.tsx`:
- Add `const [topologyDraft, setTopologyDraft] = useState<GeneratedGraph | undefined>(undefined);`
- Pass `topologyDraft` and `setTopologyDraft` into `MainViewer`
- Wire the `<GenerateButton>` for topology mode:
  ```typescript
  const handleGenerateFloorplanFromTopology = useCallback(async () => {
    if (!topologyDraft) return;
    setLoading(true);
    setError(null);
    try {
      const res = await generateTopology(topologyDraft.rooms, topologyDraft.adjacency.length ? topologyDraft.adjacency : computeAdjacency(topologyDraft));
      const item = createFloorplanItem(selectedDataset, res.image, 'topology');
      setHistory((prev) => appendHistory(prev, item));
      setSelectedHistoryId(item.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  }, [topologyDraft, selectedDataset]);
  ```

  Note: `topologyDraft.adjacency` may be empty since the editor doesn't compute it. Add a small helper:
  ```typescript
  function computeAdjacency(g: GeneratedGraph): number[][] {
    const n = g.nodes.length;
    const adj: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
    for (const e of g.edges) { adj[e.source][e.target] = 1; adj[e.target][e.source] = 1; }
    return adj;
  }
  ```

- [ ] **Step 4: Type-check + smoke**

```bash
cd /d/Github/GSDiff/app && npx tsc --noEmit
npm run dev
```

Test:
1. Switch to Topology mode → main viewer shows the 4-node default
2. Edit (add nodes, change room types via double-click, add edges)
3. Click "Generate Floorplan" → floorplan generated, added to history
4. Switch to another mode and back → topology editor restores state (Task 12 will add sessionStorage; for now in-memory state survives within a page session)

- [ ] **Step 5: Commit**

```bash
git add app/components/TopologyEditor.tsx app/components/MainViewer.tsx app/src/app/page.tsx
git commit -m "feat(topology): rewrite TopologyEditor to use BubbleGraphCanvas edit mode"
```

---

### Task 10: Frontend — Move `<BoundaryCanvas>` into `<MainViewer>`

**Files:**
- Modify: `app/components/BoundaryCanvas.tsx`
- Modify: `app/components/MainViewer.tsx`
- Modify: `app/src/app/page.tsx`

- [ ] **Step 1: Read current `BoundaryCanvas`**

Read `app/components/BoundaryCanvas.tsx` to understand its props and internal canvas mechanics. Identify:
- Where the canvas DOM element lives
- Pen/eraser/clear handler signatures
- How it returns the boundary image to the parent

- [ ] **Step 2: Refactor `BoundaryCanvas` to expose its draft**

Add (or surface) a controlled mode where the parent owns `boundaryDraft.canvasDataUrl`:
- New prop: `value?: string` (data URL of current canvas)
- New prop: `onChange?: (dataUrl: string) => void` (called on every stroke)
- Keep existing `onGenerate(dataUrl)` for the explicit generate-button flow if it exists

If the current component uses internal canvas state only, expose the canvas data via a `useEffect` that calls `onChange(canvas.toDataURL('image/png'))` after each draw stroke.

- [ ] **Step 3: Add boundary branch to `MainViewer`**

```typescript
if (mode === 'boundary') {
  return (
    <div className="flex flex-1 items-center justify-center overflow-hidden bg-background p-6">
      <BoundaryCanvas value={boundaryDraft} onChange={onBoundaryChange} />
    </div>
  );
}
```

- [ ] **Step 4: Wire in `page.tsx`**

```typescript
const [boundaryDraft, setBoundaryDraft] = useState<string>('');

const handleGenerateFloorplanFromBoundary = useCallback(async () => {
  if (!boundaryDraft) return;
  setLoading(true);
  setError(null);
  try {
    const res = await generateBoundary(boundaryDraft);
    const item = createFloorplanItem(selectedDataset, res.image, 'boundary');
    setHistory((prev) => appendHistory(prev, item));
    setSelectedHistoryId(item.id);
  } catch (e) {
    setError(e instanceof Error ? e.message : 'Generation failed');
  } finally {
    setLoading(false);
  }
}, [boundaryDraft, selectedDataset]);
```

Wire the `<GenerateButton>` for boundary mode to call this. Pass `boundaryDraft`/`setBoundaryDraft` into `MainViewer`.

- [ ] **Step 5: Type-check + smoke**

```bash
cd /d/Github/GSDiff/app && npx tsc --noEmit && npm run dev
```

Test:
1. Switch to Boundary mode → canvas appears in main viewer
2. Draw a boundary
3. Click "Generate Floorplan" → result added to history

- [ ] **Step 6: Commit**

```bash
git add app/components/BoundaryCanvas.tsx app/components/MainViewer.tsx app/src/app/page.tsx
git commit -m "feat(boundary): move BoundaryCanvas into MainViewer with controlled draft"
```

---

### Task 11: Frontend — Cleanup: remove dead code and `@xyflow/react`

**Files:**
- Delete: `app/components/RoomNode.tsx`
- Delete: `app/lib/graph-flow.ts`
- Delete: `app/components/BubbleGraphView.tsx`
- Delete: `app/components/GraphPanel.tsx`
- Delete: `app/components/ResultPanel.tsx`
- Delete: `app/components/HistoryPanel.tsx`
- Modify: `app/src/app/globals.css`
- Modify: `app/package.json`
- Modify: `app/components/RoomLegend.tsx` (drop the `ROOM_TYPES` shim path, ensure it requires `dataset` prop)
- Modify: `app/lib/constants.ts` (drop the `ROOM_TYPES` and `RoomType` backwards-compat shims)

- [ ] **Step 1: Verify no remaining usages**

```bash
cd /d/Github/GSDiff/app
grep -rn "from '@/components/RoomNode'" src components
grep -rn "from '@/lib/graph-flow'" src components
grep -rn "from '@/components/BubbleGraphView'" src components
grep -rn "from '@/components/GraphPanel'" src components
grep -rn "from '@/components/ResultPanel'" src components
grep -rn "from '@/components/HistoryPanel'" src components
grep -rn "from '@xyflow/react'" src components
grep -rn "ROOM_TYPES\b" src components | grep -v 'ROOM_TYPES_'
```
Expected: all greps return nothing (or only the file being deleted itself).

If any usages remain, fix them before continuing.

- [ ] **Step 2: Delete dead files**

```bash
rm app/components/RoomNode.tsx
rm app/lib/graph-flow.ts
rm app/components/BubbleGraphView.tsx
rm app/components/GraphPanel.tsx
rm app/components/ResultPanel.tsx
rm app/components/HistoryPanel.tsx
```

- [ ] **Step 3: Remove xyflow CSS import from `globals.css`**

In `app/src/app/globals.css`, delete the line:
```css
@import "@xyflow/react/dist/style.css";
```

- [ ] **Step 4: Uninstall `@xyflow/react`**

```bash
cd /d/Github/GSDiff/app
npm uninstall @xyflow/react
```

- [ ] **Step 5: Drop `ROOM_TYPES` shim**

In `app/lib/constants.ts`, remove these lines at the bottom:
```typescript
// Backwards-compat shim: existing callers using ROOM_TYPES default to RPLAN's set.
export const ROOM_TYPES = ROOM_TYPES_RPLAN;
export type RoomType = (typeof ROOM_TYPES_RPLAN)[number];
```

- [ ] **Step 6: Type-check + lint + smoke**

```bash
cd /d/Github/GSDiff/app && npx tsc --noEmit && npm run lint && npm run build
```
Expected: all pass.

```bash
npm run dev
```

Click through all 4 modes for both RPLAN and MSD; verify nothing broke.

- [ ] **Step 7: Commit**

```bash
git add -A app/ # everything cleaned
git commit -m "chore: remove @xyflow/react and replaced legacy components"
```

---

### Task 12: Frontend — `sessionStorage` persistence + cross-mode toast

Polish layer: persist topology and boundary drafts across page reloads, and surface a toast when clicking a history item from a different mode.

**Files:**
- Modify: `app/src/app/page.tsx`
- (No new components — keep the toast minimal: an inline status message at the top of the right workspace, auto-dismisses after 3s)

- [ ] **Step 1: Add a persistence hook**

Inside `app/src/app/page.tsx` (or extract to `app/lib/use-session-state.ts` if you prefer):
```typescript
function useSessionState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === 'undefined') return initial;
    try {
      const raw = window.sessionStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      window.sessionStorage.setItem(key, JSON.stringify(value));
    } catch {}
  }, [key, value]);
  return [value, setValue];
}
```

- [ ] **Step 2: Use it for topology and boundary drafts**

Replace:
```typescript
const [topologyDraft, setTopologyDraft] = useState<GeneratedGraph | undefined>(undefined);
const [boundaryDraft, setBoundaryDraft] = useState<string>('');
```
with:
```typescript
const [topologyDraft, setTopologyDraft] = useSessionState<GeneratedGraph | undefined>('topologyDraft', undefined);
const [boundaryDraft, setBoundaryDraft] = useSessionState<string>('boundaryDraft', '');
```

- [ ] **Step 3: Add cross-mode toast on history click**

In `page.tsx`, wrap the `setSelectedHistoryId` handler that `<HistoryBar>` calls:
```typescript
const [toast, setToast] = useState<string | null>(null);

const handleSelectHistory = useCallback((id: string) => {
  const item = findHistoryItem(history, id);
  if (!item) return;
  // Auto-switch to a viewer mode that can render this item
  let nextMode: GenerationMode = activeMode;
  if (item.kind === 'graph' && activeMode !== 'graph') nextMode = 'graph';
  if (item.kind === 'floorplan' && (activeMode === 'graph')) nextMode = 'unconstrained';
  // Topology and Boundary modes keep their editor visible UNLESS the clicked item is a graph
  if ((activeMode === 'topology' || activeMode === 'boundary') && item.kind === 'graph') nextMode = 'graph';
  if (nextMode !== activeMode) {
    setActiveMode(nextMode);
    setToast(`Switched to ${nextMode} mode to view this item`);
    setTimeout(() => setToast(null), 3000);
  }
  setSelectedHistoryId(id);
}, [activeMode, history]);
```

Render the toast at the top of the right workspace:
```typescript
{toast && (
  <div className="border-b border-border/60 bg-foreground/90 px-4 py-2 text-xs text-background">
    {toast}
  </div>
)}
```

- [ ] **Step 4: Type-check + smoke**

```bash
cd /d/Github/GSDiff/app && npx tsc --noEmit && npm run dev
```

Test:
1. Topology mode → edit some nodes → reload page → drafts restored
2. Boundary mode → draw → reload → drawing restored
3. Generate a graph in Graph mode, switch to Topology mode, click the graph thumbnail in history → switches back to Graph mode with toast

- [ ] **Step 5: Commit**

```bash
git add app/src/app/page.tsx
git commit -m "feat: persist topology/boundary drafts via sessionStorage + cross-mode toast"
```

---

## Final verification (after all 12 tasks)

```bash
cd /d/Github/GSDiff/app
npx tsc --noEmit && npm run lint && npm run build
```
All three pass.

Smoke test in browser (`npm run dev`, http://localhost:3000):
- Both RPLAN and MSD selectable; both work for Graph mode
- All 4 modes generate successfully
- History bar shows mixed graph/floorplan items
- "Send to Floorplan" works on graph items
- Topology editing (add/delete node, add/delete edge, undo/redo) works
- Page reload preserves topology and boundary drafts
- `@xyflow/react` is no longer installed: `npm ls @xyflow/react` returns "(empty)"

---

## Self-Review Notes

**Spec coverage check:**
- §3 architecture overview → Tasks 5, 6, 7 ✓
- §4.1 left panel → Task 5 ✓
- §4.2 right workspace → Tasks 5, 6, 7 ✓
- §4.3 empty states → Task 6 (MainViewer.EmptyState) ✓
- §5.1 Unconstrained → Task 6 ✓
- §5.2 Graph → Tasks 4, 6 ✓
- §5.3 Topology → Tasks 8, 9 ✓
- §5.4 Boundary → Task 10 ✓
- §6.1 page state → Task 2 + 6 ✓
- §6.2 selection logic → Tasks 6, 9, 10 ✓
- §6.3 mode switching → Task 12 ✓
- §6.4 cross-mode nav → Task 12 ✓
- §7.1 BubbleGraphCanvas contract → Tasks 4, 8 ✓
- §7.2 MainViewer → Task 5 ✓
- §7.3 HistoryBar → Task 7 ✓
- §7.4 GenerateButton → Task 5 ✓
- §7.5 components to remove → Task 11 ✓
- §7.6 components to rewrite → Tasks 9, 10, 11 ✓
- §8.1 V5 ckpt swap → Task 1 (conditional) ✓
- §8.2 no other backend → confirmed ✓
- §9 tech stack → Task 3 (install) + Task 11 (uninstall) ✓
- §10 visual style → embedded in Task 4 (BubbleNode/Edge styling) ✓
- §11 risks → addressed across tasks (lambda issue → Task 1 gate; force thrash → Task 4 hook design; history cap → Task 2 HISTORY_CAP=30; boundary regression → Task 10 step 1 read-first; cross-mode confusion → Task 12 toast) ✓
- §12 out-of-scope → respected (no localStorage history, no batch generation, etc.) ✓

No spec gaps found. All requirements covered.
