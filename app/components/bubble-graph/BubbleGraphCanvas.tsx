'use client';

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from 'react';
import { DATASET_SPECS, type DatasetId } from '@/lib/constants';
import type { GeneratedGraph } from '@/lib/types';
import type { BubbleNodeState, BubbleEdgeState, BubbleSelection } from '@/lib/bubble-graph/types';
import { useForceSimulation } from '@/lib/bubble-graph/use-force-simulation';
import { BubbleNode } from './BubbleNode';
import { BubbleEdge } from './BubbleEdge';
import { EditorToolbar } from './EditorToolbar';
import { NodeActionBar } from './NodeActionBar';
import { EdgeActionBar } from './EdgeActionBar';

interface Props {
  graph: GeneratedGraph;
  dataset: DatasetId;
  mode: 'view' | 'edit';
  onChange?: (next: GeneratedGraph) => void;
  /** Initial default room-type attr for newly created nodes (editor-internal after mount) */
  defaultNodeAttrInitial?: number;
  /** Initial default edge type for newly created edges (editor-internal after mount) */
  defaultEdgeTypeInitial?: number;
  onSelectionChange?: (sel: BubbleSelection) => void;
  width?: number;
  height?: number;
  /** Node ids to render with an outer glow ring (e.g. retrieval matches).
   *  Pass an array or a Set; falsy means no highlighting. */
  highlightedNodeIds?: number[] | Set<number> | null;
}

const DEFAULT_WIDTH = 900;
const DEFAULT_HEIGHT = 600;

// ---------------------------------------------------------------------------
// Edit-mode reducer
// ---------------------------------------------------------------------------

type EditAction =
  | { type: 'add-node'; attr: number; x: number; y: number }
  | { type: 'delete-node'; id: number }
  | { type: 'change-node-attr'; id: number; attr: number }
  | { type: 'add-edge'; source: number; target: number; edgeType: number }
  | { type: 'delete-edge'; id: string }
  | { type: 'set-edge-type'; id: string; edgeType: number }
  /** Compound: add a new node then connect source → new-node in ONE undo step */
  | { type: 'add-node-and-edge'; attr: number; x: number; y: number; sourceId: number; edgeType: number }
  /** Restore nodes+edges to a snapshot (used by Reset). Undoable. */
  | { type: 'reset-to'; nodes: BubbleNodeState[]; edges: BubbleEdgeState[] }
  /** Replace editor state because the incoming graph prop changed. Not undoable. */
  | { type: 'replace-from-prop'; nodes: BubbleNodeState[]; edges: BubbleEdgeState[] }
  | { type: 'undo' }
  | { type: 'redo' }
  | { type: 'tick'; nodes: BubbleNodeState[] };

interface EditState {
  nodes: BubbleNodeState[];
  edges: BubbleEdgeState[];
  past: { nodes: BubbleNodeState[]; edges: BubbleEdgeState[] }[];
  future: { nodes: BubbleNodeState[]; edges: BubbleEdgeState[] }[];
}

const MAX_UNDO = 50;

function endpointId(endpoint: number | BubbleNodeState): number {
  return typeof endpoint === 'object' ? endpoint.id : endpoint;
}

function normalizeEdges(edges: BubbleEdgeState[]): BubbleEdgeState[] {
  return edges.map((edge) => ({
    ...edge,
    source: endpointId(edge.source as number | BubbleNodeState),
    target: endpointId(edge.target as number | BubbleNodeState),
  }));
}

function buildInitialNodes(
  graph: GeneratedGraph,
  width: number,
  height: number,
): BubbleNodeState[] {
  const n = graph.nodes.length;
  if (n === 0) return [];

  const centerX = width / 2;
  const centerY = height / 2;
  if (n === 1) {
    const only = graph.nodes[0];
    return [{ id: only.id, attr: only.attr, x: centerX, y: centerY, fx: null, fy: null }];
  }

  const degree = new Map<number, number>();
  for (const node of graph.nodes) degree.set(node.id, 0);
  for (const edge of graph.edges) {
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1);
  }

  // Generated MSD graphs can have 25-30 rooms. A fixed small circle places
  // them almost on top of one another, so seed the force layout with a
  // topology-aware multi-ring layout: hubs near the center, leaves outward.
  const ordered = [...graph.nodes].sort((a, b) => {
    const degreeDelta = (degree.get(b.id) ?? 0) - (degree.get(a.id) ?? 0);
    return degreeDelta !== 0 ? degreeDelta : a.id - b.id;
  });
  const positions = new Map<number, { x: number; y: number }>();

  const useCenterHub = n >= 8 && (degree.get(ordered[0].id) ?? 0) >= 3;
  let cursor = 0;
  if (useCenterHub) {
    positions.set(ordered[0].id, { x: centerX, y: centerY });
    cursor = 1;
  }

  const desiredSpacing = 92;
  let ringIndex = 1;
  while (cursor < ordered.length) {
    const rawRadius = 92 + (ringIndex - 1) * 112;
    const rx = Math.min(width * 0.42, rawRadius * 1.25);
    const ry = Math.min(height * 0.38, rawRadius);
    const approximateCircumference = 2 * Math.PI * Math.sqrt((rx * rx + ry * ry) / 2);
    const capacity = Math.max(6, Math.floor(approximateCircumference / desiredSpacing));
    const remaining = ordered.length - cursor;
    const count = Math.min(capacity, remaining);
    const angleOffset = ringIndex % 2 === 0 ? Math.PI / count : 0;

    for (let i = 0; i < count; i += 1) {
      const angle = angleOffset + (2 * Math.PI * i) / count;
      const jitter = 1 + ((i % 3) - 1) * 0.035;
      const node = ordered[cursor + i];
      positions.set(node.id, {
        x: centerX + Math.cos(angle) * rx * jitter,
        y: centerY + Math.sin(angle) * ry * jitter,
      });
    }
    cursor += count;
    ringIndex += 1;
  }

  return graph.nodes.map((node) => {
    const position = positions.get(node.id) ?? { x: centerX, y: centerY };
    return {
      id: node.id,
      attr: node.attr,
      x: position.x,
      y: position.y,
      fx: null,
      fy: null,
    };
  });
}

function isWallEdge(edge: BubbleEdgeState, edgeTypes: readonly { id: number; name: string }[]): boolean {
  return edgeTypes.find((type) => type.id === edge.edgeType)?.name.toLowerCase() === 'wall';
}

function buildLayoutEdges(
  nodes: BubbleNodeState[],
  edges: BubbleEdgeState[],
  edgeTypes: readonly { id: number; name: string }[],
): BubbleEdgeState[] {
  if (edges.length <= Math.max(12, nodes.length * 1.4)) return edges;

  const nodeIds = new Set(nodes.map((node) => node.id));
  const parent = new Map<number, number>();
  for (const id of nodeIds) parent.set(id, id);

  const find = (id: number): number => {
    const current = parent.get(id) ?? id;
    if (current === id) return id;
    const root = find(current);
    parent.set(id, root);
    return root;
  };
  const union = (a: number, b: number): boolean => {
    if (!nodeIds.has(a) || !nodeIds.has(b)) return false;
    const rootA = find(a);
    const rootB = find(b);
    if (rootA === rootB) return false;
    parent.set(rootB, rootA);
    return true;
  };

  const accessEdges: BubbleEdgeState[] = [];
  const wallEdges: BubbleEdgeState[] = [];
  for (const edge of edges) {
    if (isWallEdge(edge, edgeTypes)) wallEdges.push(edge);
    else accessEdges.push(edge);
  }

  const selected: BubbleEdgeState[] = [];
  for (const edge of accessEdges) {
    selected.push({ ...edge, layoutRole: 'access' });
    union(endpointId(edge.source as number | BubbleNodeState), endpointId(edge.target as number | BubbleNodeState));
  }

  // Wall contacts are useful for connectivity but too dense to use as physics
  // springs. Add only a spanning backbone, then let collision/charge create
  // readable spacing.
  for (const edge of wallEdges) {
    const source = endpointId(edge.source as number | BubbleNodeState);
    const target = endpointId(edge.target as number | BubbleNodeState);
    if (union(source, target)) selected.push({ ...edge, layoutRole: 'wall' });
  }

  return selected.length > 0 ? selected : edges;
}

function buildNeighborhood(
  selectedNodeId: number | null,
  edges: BubbleEdgeState[],
  maxDepth = 3,
): {
  nodeDepth: Map<number, 0 | 1 | 2 | 3>;
  edgeDepth: Map<string, 1 | 2 | 3>;
} {
  const nodeDepth = new Map<number, 0 | 1 | 2 | 3>();
  const edgeDepth = new Map<string, 1 | 2 | 3>();
  if (selectedNodeId === null) return { nodeDepth, edgeDepth };

  const adjacency = new Map<number, { neighbor: number; edgeId: string }[]>();
  for (const edge of edges) {
    const source = endpointId(edge.source as number | BubbleNodeState);
    const target = endpointId(edge.target as number | BubbleNodeState);
    if (!adjacency.has(source)) adjacency.set(source, []);
    if (!adjacency.has(target)) adjacency.set(target, []);
    adjacency.get(source)?.push({ neighbor: target, edgeId: edge.id });
    adjacency.get(target)?.push({ neighbor: source, edgeId: edge.id });
  }

  nodeDepth.set(selectedNodeId, 0);
  const queue: number[] = [selectedNodeId];
  while (queue.length > 0) {
    const current = queue.shift();
    if (current === undefined) continue;
    const depth = nodeDepth.get(current);
    if (depth === undefined || depth >= maxDepth) continue;

    for (const { neighbor, edgeId } of adjacency.get(current) ?? []) {
      const nextDepth = (depth + 1) as 1 | 2 | 3;
      const knownDepth = nodeDepth.get(neighbor);
      if (knownDepth === undefined || nextDepth < knownDepth) {
        nodeDepth.set(neighbor, nextDepth);
        queue.push(neighbor);
      }
      const knownEdgeDepth = edgeDepth.get(edgeId);
      if (knownEdgeDepth === undefined || nextDepth < knownEdgeDepth) {
        edgeDepth.set(edgeId, nextDepth);
      }
    }
  }

  return { nodeDepth, edgeDepth };
}

function pushPast(state: EditState): EditState {
  const past = [...state.past, { nodes: state.nodes, edges: normalizeEdges(state.edges) }].slice(-MAX_UNDO);
  return { ...state, past, future: [] };
}

function reducer(state: EditState, action: EditAction): EditState {
  if (action.type === 'tick') {
    // Merge positions by id rather than replacing the array wholesale.
    // The simulation may be one tick behind a freshly-added node (between
    // dispatch add-node and sim rebuild); in that window, action.nodes
    // is shorter than state.nodes. Nodes the sim doesn't know about keep
    // their current position.
    const simPositions = new Map(action.nodes.map((n) => [n.id, n]));
    return {
      ...state,
      nodes: state.nodes.map((n) => {
        const fromSim = simPositions.get(n.id);
        return fromSim
          ? { ...n, x: fromSim.x, y: fromSim.y, fx: fromSim.fx, fy: fromSim.fy }
          : n;
      }),
    };
  }
  if (action.type === 'undo') {
    if (state.past.length === 0) return state;
    const prev = state.past[state.past.length - 1];
    return {
      ...state,
      nodes: prev.nodes,
      edges: normalizeEdges(prev.edges),
      past: state.past.slice(0, -1),
      future: [{ nodes: state.nodes, edges: normalizeEdges(state.edges) }, ...state.future],
    };
  }
  if (action.type === 'redo') {
    if (state.future.length === 0) return state;
    const next = state.future[0];
    return {
      ...state,
      nodes: next.nodes,
      edges: normalizeEdges(next.edges),
      past: [...state.past, { nodes: state.nodes, edges: normalizeEdges(state.edges) }].slice(-MAX_UNDO),
      future: state.future.slice(1),
    };
  }
  if (action.type === 'replace-from-prop') {
    return {
      nodes: action.nodes.map((n) => ({ ...n })),
      edges: normalizeEdges(action.edges),
      past: [],
      future: [],
    };
  }
  const checkpointed = pushPast(state);
  switch (action.type) {
    case 'add-node': {
      const newId = state.nodes.reduce((m, n) => Math.max(m, n.id), -1) + 1;
      const node: BubbleNodeState = {
        id: newId,
        attr: action.attr,
        x: action.x,
        y: action.y,
        // Leave fx/fy null so the simulation can position this node
        // naturally based on its edges and the global force layout.
        fx: null,
        fy: null,
      };
      return { ...checkpointed, nodes: [...state.nodes, node] };
    }
    case 'delete-node': {
      return {
        ...checkpointed,
        nodes: state.nodes.filter((n) => n.id !== action.id),
        edges: state.edges.filter((e) => {
          const srcId =
            endpointId(e.source as number | BubbleNodeState);
          const tgtId =
            endpointId(e.target as number | BubbleNodeState);
          return srcId !== action.id && tgtId !== action.id;
        }),
      };
    }
    case 'change-node-attr': {
      return {
        ...checkpointed,
        nodes: state.nodes.map((n) => (n.id === action.id ? { ...n, attr: action.attr } : n)),
      };
    }
    case 'add-edge': {
      const id = `${Math.min(action.source, action.target)}-${Math.max(action.source, action.target)}`;
      if (state.edges.some((e) => e.id === id)) return state;
      return {
        ...checkpointed,
        edges: [
          ...state.edges,
          { id, source: action.source, target: action.target, edgeType: action.edgeType },
        ],
      };
    }
    case 'delete-edge': {
      return {
        ...checkpointed,
        edges: state.edges.filter((e) => e.id !== action.id),
      };
    }
    case 'set-edge-type': {
      return {
        ...checkpointed,
        edges: state.edges.map((e) =>
          e.id === action.id ? { ...e, edgeType: action.edgeType } : e,
        ),
      };
    }
    case 'reset-to': {
      return {
        ...checkpointed,
        nodes: action.nodes.map((n) => ({ ...n })),
        edges: normalizeEdges(action.edges),
      };
    }
    case 'add-node-and-edge': {
      // Single undo step: create node + edge.
      // Leave fx/fy null so the simulation can place the new node
      // naturally given its connection to the source node.
      const newId = state.nodes.reduce((m, n) => Math.max(m, n.id), -1) + 1;
      const node: BubbleNodeState = {
        id: newId,
        attr: action.attr,
        x: action.x,
        y: action.y,
        fx: null,
        fy: null,
      };
      const edgeId = `${Math.min(action.sourceId, newId)}-${Math.max(action.sourceId, newId)}`;
      const edge: BubbleEdgeState = {
        id: edgeId,
        source: action.sourceId,
        target: newId,
        edgeType: action.edgeType,
      };
      return {
        ...checkpointed,
        nodes: [...state.nodes, node],
        edges: [...state.edges, edge],
      };
    }
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function BubbleGraphCanvas({
  graph,
  dataset,
  mode,
  onChange,
  defaultNodeAttrInitial = 0,
  defaultEdgeTypeInitial = 1,
  onSelectionChange,
  width: widthProp,
  height: heightProp,
  highlightedNodeIds,
}: Props) {
  const highlightSet = useMemo(() => {
    if (!highlightedNodeIds) return null;
    return highlightedNodeIds instanceof Set
      ? highlightedNodeIds
      : new Set(highlightedNodeIds);
  }, [highlightedNodeIds]);

  const spec = DATASET_SPECS[dataset];

  // Editor-internal toolbar state
  const [defaultNodeAttr, setDefaultNodeAttr] = useState(defaultNodeAttrInitial);
  const [defaultEdgeType, setDefaultEdgeType] = useState(defaultEdgeTypeInitial);

  // Measure container
  const containerRef = useRef<HTMLDivElement>(null);
  const [measured, setMeasured] = useState({ width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT });
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => {
      const rect = el.getBoundingClientRect();
      const w = Math.max(400, Math.round(rect.width));
      const h = Math.max(300, Math.round(rect.height));
      setMeasured((prev) => (prev.width === w && prev.height === h ? prev : { width: w, height: h }));
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const width = widthProp ?? measured.width;
  const height = heightProp ?? measured.height;

  const graphNodeSignature = useMemo(
    () => graph.nodes.map((node) => `${node.id}:${node.attr}`).join('|'),
    [graph.nodes],
  );
  const graphEdgeSignature = useMemo(
    () =>
      graph.edges
        .map((edge) => {
          const source = Math.min(edge.source, edge.target);
          const target = Math.max(edge.source, edge.target);
          return `${source}-${target}:${edge.edge_type}`;
        })
        .sort()
        .join('|'),
    [graph.edges],
  );
  const graphSignature = `${dataset}::${graphNodeSignature}::${graphEdgeSignature}`;

  const initialNodes: BubbleNodeState[] = useMemo(
    () => buildInitialNodes(graph, width, height),
    // Re-init only when graph identity changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [graphNodeSignature, graphEdgeSignature],
  );

  const initialEdges: BubbleEdgeState[] = useMemo(
    () =>
      graph.edges.map((e) => ({
        id: `${e.source}-${e.target}`,
        source: e.source,
        target: e.target,
        edgeType: e.edge_type,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [graphEdgeSignature],
  );

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  const [viewNodes, setViewNodes] = useState<BubbleNodeState[]>(initialNodes);
  const [forceSyncNonce, setForceSyncNonce] = useState(0);

  const [editState, dispatch] = useReducer(reducer, {
    nodes: initialNodes,
    edges: initialEdges,
    past: [],
    future: [],
  });

  const editing = mode === 'edit';
  const nodes = editing ? editState.nodes : viewNodes;
  const edges = editing ? editState.edges : initialEdges;

  const [selection, setSelection] = useState<BubbleSelection>({ nodeId: null, edgeId: null });

  // Hover tracking for "+" handle
  const [hoveredNodeId, setHoveredNodeId] = useState<number | null>(null);

  // Pan + zoom state (in viewBox coordinate space).
  //   - `k` is the scale factor (1 = no zoom)
  //   - `(x, y)` is the translate applied BEFORE scale, so a point P in
  //     "world" coordinates is rendered at viewBox position (P * k + (x, y)).
  const [zoom, setZoom] = useState<{ k: number; x: number; y: number }>({ k: 1, x: 0, y: 0 });

  // Edge-creation rubber-band state (edit mode only)
  const [edgeDraft, setEdgeDraft] = useState<{
    sourceId: number;
    cursor: { x: number; y: number };
  } | null>(null);

  // Keep stable ref to nodes for pointer callbacks
  const nodesRef = useRef<BubbleNodeState[]>(nodes);
  nodesRef.current = nodes;

  // Snapshot of the initial graph (from `graph` prop). Used by Reset.
  // Updates whenever the incoming graph identity changes (new sample / new mount).
  const initialSnapshotRef = useRef<{ nodes: BubbleNodeState[]; edges: BubbleEdgeState[] }>({
    nodes: initialNodes,
    edges: initialEdges,
  });
  const graphSignatureRef = useRef(graphSignature);
  useEffect(() => {
    initialSnapshotRef.current = { nodes: initialNodes, edges: initialEdges };
    if (graphSignatureRef.current === graphSignature) return;
    graphSignatureRef.current = graphSignature;
    dispatch({
      type: 'replace-from-prop',
      nodes: initialNodes.map((n) => ({ ...n })),
      edges: initialEdges.map((e) => ({ ...e })),
    });
    setViewNodes(initialNodes.map((n) => ({ ...n })));
    setSelection({ nodeId: null, edgeId: null });
    setForceSyncNonce((value) => value + 1);
  }, [graphSignature, initialNodes, initialEdges]);

  // ---------------------------------------------------------------------------
  // Force simulation
  // ---------------------------------------------------------------------------

  const handleTick = useCallback(
    (next: BubbleNodeState[]) => {
      if (editing) {
        dispatch({ type: 'tick', nodes: next });
      } else {
        setViewNodes([...next]);
      }
    },
    [editing],
  );

  // Feed the live editor state into the force sim so newly-added nodes/edges
  // participate in the layout. The hook rebuilds the simulation whenever
  // nodes.length or edges.length changes — that's our cue for add/delete.
  //
  // Without this, the sim is stuck on the original `initialNodes`; on every
  // tick it would overwrite editState.nodes with the simulation's 4-node
  // array, making newly-added nodes silently disappear.
  const simInputNodes = editing ? editState.nodes : initialNodes;
  const simInputEdges = editing ? editState.edges : initialEdges;
  const layoutEdges = useMemo(
    () => buildLayoutEdges(simInputNodes, simInputEdges, spec.edgeTypes),
    [simInputNodes, simInputEdges, spec.edgeTypes],
  );

  const { pinNode, releaseNode, reheat } = useForceSimulation({
    nodes: simInputNodes,
    edges: layoutEdges,
    width,
    height,
    onTick: handleTick,
    syncKey: forceSyncNonce,
  });

  // Dispatch wrapper that also bumps the simulation so the graph
  // visibly rebalances after structural edits (add/delete/connect).
  const dispatchAndReheat = useCallback(
    (action: EditAction) => {
      dispatch(action);
      // Slight delay so the reducer's state update is in effect before reheat.
      requestAnimationFrame(() => reheat());
    },
    [reheat],
  );

  // ---------------------------------------------------------------------------
  // onChange emission
  // ---------------------------------------------------------------------------
  const graphContentSignature = useMemo(() => {
    const nodePart = editState.nodes
      .map((node) => `${node.id}:${node.attr}`)
      .join('|');
    const edgePart = editState.edges
      .map((edge) => {
        const source = endpointId(edge.source as number | BubbleNodeState);
        const target = endpointId(edge.target as number | BubbleNodeState);
        return `${Math.min(source, target)}-${Math.max(source, target)}:${edge.edgeType}`;
      })
      .sort()
      .join('|');
    return `${nodePart}::${edgePart}`;
  }, [editState.nodes, editState.edges]);

  useEffect(() => {
    if (!editing || !onChange) return;
    const next: GeneratedGraph = {
      num_nodes: editState.nodes.length,
      num_edges: editState.edges.length,
      rooms: editState.nodes.map((n) => n.attr),
      adjacency: [],
      edge_types: [],
      nodes: editState.nodes.map((n) => ({
        id: n.id,
        attr: n.attr,
        room_type: spec.roomTypes.find((m) => m.id === n.attr)?.name ?? `class_${n.attr}`,
      })),
      edges: editState.edges.map((e) => {
        const srcId =
          typeof e.source === 'object' ? (e.source as unknown as BubbleNodeState).id : e.source;
        const tgtId =
          typeof e.target === 'object' ? (e.target as unknown as BubbleNodeState).id : e.target;
        return {
          source: srcId,
          target: tgtId,
          edge_type: e.edgeType,
          edge_label: spec.edgeTypes.find((m) => m.id === e.edgeType)?.name ?? `class_${e.edgeType}`,
        };
      }),
    };
    onChange(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editing, graphContentSignature]);

  // ---------------------------------------------------------------------------
  // Keyboard handlers
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!editing) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selection.nodeId !== null) {
          dispatch({ type: 'delete-node', id: selection.nodeId });
          setSelection({ nodeId: null, edgeId: null });
          onSelectionChange?.({ nodeId: null, edgeId: null });
        } else if (selection.edgeId !== null) {
          dispatch({ type: 'delete-edge', id: selection.edgeId });
          setSelection({ nodeId: null, edgeId: null });
          onSelectionChange?.({ nodeId: null, edgeId: null });
        }
      } else if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
        e.preventDefault();
        dispatch({ type: 'undo' });
      } else if (
        (e.key === 'y' && (e.ctrlKey || e.metaKey)) ||
        (e.key === 'z' && (e.ctrlKey || e.metaKey) && e.shiftKey)
      ) {
        e.preventDefault();
        dispatch({ type: 'redo' });
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [editing, selection, onSelectionChange]);

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  const updateSelection = useCallback(
    (next: BubbleSelection) => {
      setSelection(next);
      onSelectionChange?.(next);
    },
    [onSelectionChange],
  );

  // Convert "world" coords (the logical position of nodes, ignoring zoom) →
  // container-relative pixels. Factors in the current pan+zoom transform so
  // floating HTML action bars track the on-screen position of the node/edge.
  const viewBoxToContainer = useCallback(
    (vx: number, vy: number): { px: number; py: number } => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return { px: vx, py: vy };
      const transformedX = vx * zoom.k + zoom.x;
      const transformedY = vy * zoom.k + zoom.y;
      const scaleX = rect.width / width;
      const scaleY = rect.height / height;
      return { px: transformedX * scaleX, py: transformedY * scaleY };
    },
    [width, height, zoom],
  );

  // Convert client coordinates to "world" viewBox coordinates (undo zoom).
  const clientToViewBox = useCallback(
    (clientX: number, clientY: number, rect: DOMRect): { x: number; y: number } => {
      const rawX = ((clientX - rect.left) / rect.width) * width;
      const rawY = ((clientY - rect.top) / rect.height) * height;
      return {
        x: (rawX - zoom.x) / zoom.k,
        y: (rawY - zoom.y) / zoom.k,
      };
    },
    [width, height, zoom],
  );

  // Wheel zoom — non-passive native listener so we can preventDefault and
  // stop the browser from scrolling the page while the cursor is over the canvas.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = container.getBoundingClientRect();
      // Cursor position in raw viewBox coords (pre-transform)
      const rawX = ((e.clientX - rect.left) / rect.width) * width;
      const rawY = ((e.clientY - rect.top) / rect.height) * height;
      setZoom((prev) => {
        // Same cursor point in current world coords
        const worldX = (rawX - prev.x) / prev.k;
        const worldY = (rawY - prev.y) / prev.k;
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const newK = Math.max(0.3, Math.min(4, prev.k * factor));
        // Keep the same world point under the cursor
        return {
          k: newK,
          x: rawX - worldX * newK,
          y: rawY - worldY * newK,
        };
      });
    };
    container.addEventListener('wheel', onWheel, { passive: false });
    return () => container.removeEventListener('wheel', onWheel);
  }, [width, height]);

  // Drag node to move it. Pin during drag, release on drop so the node
  // rejoins the force layout — gives a "free force graph" feel.
  // Important: only reheat if the user *actually* dragged. A bare click
  // must not perturb the layout (otherwise selecting a node visibly
  // resettles the whole graph).
  const dragNode = (nodeId: number) => (e: React.PointerEvent) => {
    e.preventDefault();
    const svg = (e.currentTarget as SVGElement).ownerSVGElement;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    let didDrag = false;
    const onMove = (ev: PointerEvent) => {
      didDrag = true;
      const { x, y } = clientToViewBox(ev.clientX, ev.clientY, rect);
      pinNode(nodeId, x, y);
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      if (didDrag) {
        releaseNode(nodeId);
        reheat();
      }
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  // Start edge-drag from the "+" handle
  const startEdgeDrag = (sourceId: number) => (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const svgEl = (e.currentTarget as SVGElement).closest('svg');
    if (!svgEl) return;
    const rect = svgEl.getBoundingClientRect();
    const startPos = clientToViewBox(e.clientX, e.clientY, rect);
    setEdgeDraft({ sourceId, cursor: startPos });

    const onMove = (ev: PointerEvent) => {
      const pos = clientToViewBox(ev.clientX, ev.clientY, rect);
      setEdgeDraft({ sourceId, cursor: pos });
    };
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      const pos = clientToViewBox(ev.clientX, ev.clientY, rect);
      const hit = nodesRef.current.find(
        (n) => n.id !== sourceId && Math.hypot(n.x - pos.x, n.y - pos.y) < 36,
      );
      if (hit) {
        dispatchAndReheat({
          type: 'add-edge',
          source: sourceId,
          target: hit.id,
          edgeType: defaultEdgeType,
        });
      } else {
        // Drop on empty space → add node + edge as one undo step
        dispatchAndReheat({
          type: 'add-node-and-edge',
          attr: defaultNodeAttr,
          x: pos.x,
          y: pos.y,
          sourceId,
          edgeType: defaultEdgeType,
        });
      }
      setEdgeDraft(null);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  const handleAddNodeAtCenter = () => {
    dispatchAndReheat({
      type: 'add-node',
      attr: defaultNodeAttr,
      x: width / 2 + (Math.random() - 0.5) * 80,
      y: height / 2 + (Math.random() - 0.5) * 80,
    });
  };

  // ---------------------------------------------------------------------------
  // Derived
  // ---------------------------------------------------------------------------

  const nodeIndex = new Map(nodes.map((n) => [n.id, n]));

  // Selected node / edge objects for action bars
  const selectedNode =
    selection.nodeId !== null ? nodeIndex.get(selection.nodeId) ?? null : null;
  const selectedEdge =
    selection.edgeId !== null ? edges.find((e) => e.id === selection.edgeId) ?? null : null;
  const neighborhood = useMemo(
    () => buildNeighborhood(selection.nodeId, edges, 3),
    [selection.nodeId, edges],
  );
  const hasNodeNeighborhood = selection.nodeId !== null && neighborhood.nodeDepth.size > 0;
  const layoutEdgeIds = useMemo(
    () => new Set(layoutEdges.map((edge) => edge.id)),
    [layoutEdges],
  );

  // Edge midpoint for EdgeActionBar
  const edgeMidpoint = (() => {
    if (!selectedEdge) return null;
    const srcId =
      typeof selectedEdge.source === 'object'
        ? (selectedEdge.source as unknown as BubbleNodeState).id
        : selectedEdge.source;
    const tgtId =
      typeof selectedEdge.target === 'object'
        ? (selectedEdge.target as unknown as BubbleNodeState).id
        : selectedEdge.target;
    const src = nodeIndex.get(srcId);
    const tgt = nodeIndex.get(tgtId);
    if (!src || !tgt) return null;
    return { x: (src.x + tgt.x) / 2, y: (src.y + tgt.y) / 2 };
  })();

  const renderEdge = (edge: BubbleEdgeState, layer: 'dimmed' | 'active') => {
    const sourceId =
      typeof edge.source === 'object'
        ? (edge.source as unknown as BubbleNodeState).id
        : edge.source;
    const targetId =
      typeof edge.target === 'object'
        ? (edge.target as unknown as BubbleNodeState).id
        : edge.target;
    const source = nodeIndex.get(sourceId);
    const target = nodeIndex.get(targetId);
    if (!source || !target) return null;

    const edgeHopDepth = neighborhood.edgeDepth.get(edge.id) ?? null;
    const inNeighborhood = edgeHopDepth !== null;
    const outsideSelectedNeighborhood =
      hasNodeNeighborhood && !inNeighborhood && selection.edgeId !== edge.id;
    if (layer === 'dimmed' && !outsideSelectedNeighborhood) return null;
    if (layer === 'active' && outsideSelectedNeighborhood) return null;

    const meta = spec.edgeTypes.find((m) => m.id === edge.edgeType);
    const denseGraph = edges.length > Math.max(18, nodes.length * 1.5);
    const wallEdge = isWallEdge(edge, spec.edgeTypes);
    const participatesInLayout = layoutEdgeIds.has(edge.id);
    const muted =
      layer === 'active' &&
      denseGraph &&
      wallEdge &&
      !participatesInLayout &&
      !inNeighborhood &&
      selection.edgeId !== edge.id;
    const dimmed = layer === 'dimmed';
    const showLabel = !dimmed && (selection.edgeId === edge.id || inNeighborhood || !muted);

    return (
      <BubbleEdge
        key={edge.id}
        edge={edge}
        source={source}
        target={target}
        meta={meta}
        selected={selection.edgeId === edge.id}
        muted={muted}
        showLabel={showLabel}
        neighborhoodDepth={edgeHopDepth}
        dimmed={dimmed}
        onClick={
          editing
            ? (e) => {
                e.stopPropagation();
                updateSelection({ nodeId: null, edgeId: edge.id });
              }
            : undefined
        }
        onContextMenu={
          editing
            ? (e) => {
                e.preventDefault();
                dispatch({ type: 'delete-edge', id: edge.id });
                if (selection.edgeId === edge.id) {
                  updateSelection({ nodeId: null, edgeId: null });
                }
              }
            : undefined
        }
      />
    );
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full overflow-hidden rounded-2xl bg-[oklch(0.97_0_0)]"
      onClick={() => {
        // Clicking the container background deselects
        // (SVG children stop propagation)
      }}
    >
      {/* Top toolbar (edit mode only) */}
      {editing && (
        <EditorToolbar
          roomTypes={spec.roomTypes}
          edgeTypes={spec.edgeTypes}
          defaultNodeAttr={defaultNodeAttr}
          defaultEdgeType={defaultEdgeType}
          onDefaultNodeAttrChange={setDefaultNodeAttr}
          onDefaultEdgeTypeChange={setDefaultEdgeType}
          canUndo={editState.past.length > 0}
          canRedo={editState.future.length > 0}
          onUndo={() => dispatch({ type: 'undo' })}
          onRedo={() => dispatch({ type: 'redo' })}
          onAddNode={handleAddNodeAtCenter}
          onReset={() => {
            dispatch({
              type: 'reset-to',
              nodes: initialSnapshotRef.current.nodes.map((n) => ({ ...n })),
              edges: initialSnapshotRef.current.edges.map((e) => ({ ...e })),
            });
            updateSelection({ nodeId: null, edgeId: null });
            setForceSyncNonce((value) => value + 1);
          }}
        />
      )}

      {/* Selected node action bar */}
      {editing && selectedNode && (() => {
        const { px, py } = viewBoxToContainer(selectedNode.x, selectedNode.y);
        return (
          <NodeActionBar
            nodeId={selectedNode.id}
            currentAttr={selectedNode.attr}
            roomTypes={spec.roomTypes}
            px={px}
            py={py}
            onChangeAttr={(attr) => {
              dispatch({ type: 'change-node-attr', id: selectedNode.id, attr });
            }}
            onDelete={() => {
              dispatch({ type: 'delete-node', id: selectedNode.id });
              updateSelection({ nodeId: null, edgeId: null });
            }}
          />
        );
      })()}

      {/* Selected edge action bar */}
      {editing && selectedEdge && edgeMidpoint && (() => {
        const { px, py } = viewBoxToContainer(edgeMidpoint.x, edgeMidpoint.y);
        return (
          <EdgeActionBar
            edgeId={selectedEdge.id}
            currentType={selectedEdge.edgeType}
            edgeTypes={spec.edgeTypes}
            px={px}
            py={py}
            onChangeType={(et) => {
              dispatch({ type: 'set-edge-type', id: selectedEdge.id, edgeType: et });
            }}
            onDelete={() => {
              dispatch({ type: 'delete-edge', id: selectedEdge.id });
              updateSelection({ nodeId: null, edgeId: null });
            }}
          />
        );
      })()}

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="block h-full w-full"
        role="img"
        aria-label="Bubble graph"
        onClick={(e) => {
          // Empty-space click only deselects — never adds nodes.
          // Adding nodes is intentional only: toolbar "+ Add Room" button,
          // or dragging the "+" handle from an existing node into empty space.
          if (e.target !== e.currentTarget) return;
          updateSelection({ nodeId: null, edgeId: null });
        }}
      >
        {/* Pan+zoom transform — content positions are in "world" coords,
            this group applies the user's current zoom/pan. */}
        <g transform={`translate(${zoom.x},${zoom.y}) scale(${zoom.k})`}>
        {/* Edges. Non-neighborhood edges are composited as one dim layer so
            overlapping transparent strokes do not keep getting darker. */}
        <g
          opacity={hasNodeNeighborhood ? 0.08 : 1}
          style={{ transition: 'opacity 220ms ease-out' }}
        >
          {edges.map((edge) => renderEdge(edge, 'dimmed'))}
        </g>
        <g>
          {edges.map((edge) => renderEdge(edge, 'active'))}
        </g>

        {/* Rubber-band edge draft line */}
        {edgeDraft &&
          (() => {
            const src = nodesRef.current.find((n) => n.id === edgeDraft.sourceId);
            if (!src) return null;
            return (
              <line
                x1={src.x}
                y1={src.y}
                x2={edgeDraft.cursor.x}
                y2={edgeDraft.cursor.y}
                stroke="#94a3b8"
                strokeWidth="2"
                strokeDasharray="4 4"
                style={{ pointerEvents: 'none' }}
              />
            );
          })()}

        {/* Nodes */}
        <g>
          {nodes.map((node) => {
            const meta = spec.roomTypes.find((m) => m.id === node.attr);
            const isHovered = editing && hoveredNodeId === node.id;
            const nodeHopDepth = neighborhood.nodeDepth.get(node.id) ?? null;
            const dimmed = hasNodeNeighborhood && nodeHopDepth === null;
            return (
              <g key={node.id}>
                <BubbleNode
                  node={node}
                  meta={meta}
                  selected={selection.nodeId === node.id}
                  highlighted={highlightSet ? highlightSet.has(node.id) : false}
                  neighborhoodDepth={nodeHopDepth}
                  dimmed={dimmed}
                  onPointerDown={dragNode(node.id)}
                  onPointerEnter={editing ? () => setHoveredNodeId(node.id) : undefined}
                  onPointerLeave={editing ? () => setHoveredNodeId((prev) => prev === node.id ? null : prev) : undefined}
                  onClick={
                    editing
                      ? (e) => {
                          e.stopPropagation();
                          updateSelection({ nodeId: node.id, edgeId: null });
                        }
                      : undefined
                  }
                />
                {/* "+" handle: shown when hovered in edit mode */}
                {isHovered && (
                  <circle
                    cx={node.x + 48}
                    cy={node.y}
                    r={10}
                    fill="oklch(0.55 0.21 35)"
                    stroke="white"
                    strokeWidth={2}
                    style={{ cursor: 'crosshair' }}
                    onPointerDown={startEdgeDrag(node.id)}
                    onPointerEnter={() => setHoveredNodeId(node.id)}
                  />
                )}
                {isHovered && (
                  <text
                    x={node.x + 48}
                    y={node.y}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize="14"
                    fontWeight="bold"
                    fill="white"
                    style={{ pointerEvents: 'none', userSelect: 'none' }}
                  >
                    +
                  </text>
                )}
                {/* Delete badge — selected node only, bottom-right of circle
                    (positioned away from the room-type action bar that floats above) */}
                {editing && selection.nodeId === node.id && (
                  <g
                    style={{ cursor: 'pointer' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      dispatch({ type: 'delete-node', id: node.id });
                      updateSelection({ nodeId: null, edgeId: null });
                    }}
                  >
                    <circle
                      cx={node.x + 26}
                      cy={node.y + 26}
                      r={12}
                      fill="oklch(0.58 0.22 25)"
                      stroke="white"
                      strokeWidth={2}
                    />
                    <text
                      x={node.x + 26}
                      y={node.y + 26}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontSize="15"
                      fontWeight="bold"
                      fill="white"
                      style={{ pointerEvents: 'none', userSelect: 'none' }}
                    >
                      ×
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </g>
        {/* Selected edge delete badge — sits at midpoint, offset down */}
        {editing && selectedEdge && edgeMidpoint && (
          <g
            style={{ cursor: 'pointer' }}
            onClick={(e) => {
              e.stopPropagation();
              dispatch({ type: 'delete-edge', id: selectedEdge.id });
              updateSelection({ nodeId: null, edgeId: null });
            }}
          >
            <circle
              cx={edgeMidpoint.x}
              cy={edgeMidpoint.y + 18}
              r={10}
              fill="oklch(0.58 0.22 25)"
              stroke="white"
              strokeWidth={2}
            />
            <text
              x={edgeMidpoint.x}
              y={edgeMidpoint.y + 18}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize="13"
              fontWeight="bold"
              fill="white"
              style={{ pointerEvents: 'none', userSelect: 'none' }}
            >
              ×
            </text>
          </g>
        )}
        </g>{/* end pan+zoom group */}
      </svg>
    </div>
  );
}
