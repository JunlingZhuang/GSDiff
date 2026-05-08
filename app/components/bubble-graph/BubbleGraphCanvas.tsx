'use client';

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';
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

// ---------------------------------------------------------------------------
// Edit-mode reducer
// ---------------------------------------------------------------------------

type EditAction =
  | { type: 'add-node'; attr: number; x: number; y: number }
  | { type: 'delete-node'; id: number }
  | { type: 'change-node-attr'; id: number; attr: number }
  | { type: 'add-edge'; source: number; target: number; edgeType: number }
  | { type: 'delete-edge'; id: string }
  | { type: 'cycle-edge-type'; id: string; nextType: number }
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

function pushPast(state: EditState): EditState {
  const past = [...state.past, { nodes: state.nodes, edges: state.edges }].slice(-MAX_UNDO);
  return { ...state, past, future: [] };
}

function reducer(state: EditState, action: EditAction): EditState {
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
      const newId = state.nodes.reduce((m, n) => Math.max(m, n.id), -1) + 1;
      const node: BubbleNodeState = {
        id: newId,
        attr: action.attr,
        x: action.x,
        y: action.y,
        fx: action.x,
        fy: action.y,
      };
      return { ...checkpointed, nodes: [...state.nodes, node] };
    }
    case 'delete-node': {
      return {
        ...checkpointed,
        nodes: state.nodes.filter((n) => n.id !== action.id),
        edges: state.edges.filter((e) => {
          const srcId =
            typeof e.source === 'object' ? (e.source as unknown as BubbleNodeState).id : e.source;
          const tgtId =
            typeof e.target === 'object' ? (e.target as unknown as BubbleNodeState).id : e.target;
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
    case 'cycle-edge-type': {
      return {
        ...checkpointed,
        edges: state.edges.map((e) =>
          e.id === action.id ? { ...e, edgeType: action.nextType } : e,
        ),
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
  defaultNodeAttr = 0,
  defaultEdgeType = 1,
  onSelectionChange,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
}: Props) {
  const spec = DATASET_SPECS[dataset];

  const initialNodes: BubbleNodeState[] = useMemo(
    () =>
      graph.nodes.map((n, i) => ({
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
    () =>
      graph.edges.map((e) => ({
        id: `${e.source}-${e.target}`,
        source: e.source,
        target: e.target,
        edgeType: e.edge_type,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [graph.edges.map((e) => `${e.source}-${e.target}-${e.edge_type}`).join(',')],
  );

  // ---------------------------------------------------------------------------
  // View-mode state (original simple path)
  // ---------------------------------------------------------------------------
  const [viewNodes, setViewNodes] = useState<BubbleNodeState[]>(initialNodes);

  // ---------------------------------------------------------------------------
  // Edit-mode state (reducer path)
  // ---------------------------------------------------------------------------
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

  // Edge-creation rubber-band state (edit mode only)
  const [edgeDraft, setEdgeDraft] = useState<{
    sourceId: number;
    cursor: { x: number; y: number };
  } | null>(null);

  // Keep a stable ref to the current nodes list for use in pointer event callbacks
  const nodesRef = useRef<BubbleNodeState[]>(nodes);
  nodesRef.current = nodes;

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

  const { pinNode } = useForceSimulation({
    nodes: initialNodes,
    edges: initialEdges,
    width,
    height,
    onTick: handleTick,
  });

  // ---------------------------------------------------------------------------
  // onChange emission for edit mode
  // ---------------------------------------------------------------------------
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
  }, [editing, editState.nodes, editState.edges]);

  // ---------------------------------------------------------------------------
  // Keyboard handlers (edit mode only)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!editing) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't fire if focus is on an input/textarea
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

  const updateSelection = (next: BubbleSelection) => {
    setSelection(next);
    onSelectionChange?.(next);
  };

  // Convert client coordinates to SVG viewBox coordinates.
  // Accepts the SVG element's bounding rect.
  const clientToViewBox = (
    clientX: number,
    clientY: number,
    rect: DOMRect,
  ): { x: number; y: number } => ({
    x: ((clientX - rect.left) / rect.width) * width,
    y: ((clientY - rect.top) / rect.height) * height,
  });

  // Drag node — moves pinNode in the force simulation.
  const dragNode = (nodeId: number) => (e: React.PointerEvent) => {
    e.preventDefault();
    const svg = (e.currentTarget as SVGElement).ownerSVGElement;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const onMove = (ev: PointerEvent) => {
      const { x, y } = clientToViewBox(ev.clientX, ev.clientY, rect);
      pinNode(nodeId, x, y);
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  // Pointer-down handler for edit mode: distinguishes drag vs. edge-creation.
  const handleNodePointerDown = (nodeId: number) => (e: React.PointerEvent) => {
    if (!editing) return dragNode(nodeId)(e);

    const target = nodesRef.current.find((n) => n.id === nodeId);
    if (!target) return;

    const svgEl = (e.currentTarget as SVGElement).ownerSVGElement;
    if (!svgEl) return;
    const rect = svgEl.getBoundingClientRect();
    const { x: px, y: py } = clientToViewBox(e.clientX, e.clientY, rect);
    const dist = Math.hypot(px - target.x, py - target.y);

    if (dist > 30) {
      // Start edge-drag from perimeter
      e.preventDefault();
      setEdgeDraft({ sourceId: nodeId, cursor: { x: px, y: py } });

      const onMove = (ev: PointerEvent) => {
        const pos = clientToViewBox(ev.clientX, ev.clientY, rect);
        setEdgeDraft({ sourceId: nodeId, cursor: pos });
      };
      const onUp = (ev: PointerEvent) => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
        const pos = clientToViewBox(ev.clientX, ev.clientY, rect);
        const hit = nodesRef.current.find(
          (n) => n.id !== nodeId && Math.hypot(n.x - pos.x, n.y - pos.y) < 36,
        );
        if (hit) {
          dispatch({ type: 'add-edge', source: nodeId, target: hit.id, edgeType: defaultEdgeType });
        }
        setEdgeDraft(null);
      };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    } else {
      dragNode(nodeId)(e);
    }
  };

  // ---------------------------------------------------------------------------
  // Derived helpers
  // ---------------------------------------------------------------------------

  const nodeIndex = new Map(nodes.map((n) => [n.id, n]));

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full" role="img" aria-label="Bubble graph">
      {/* Background — click-to-add-node in edit mode */}
      <rect
        width={width}
        height={height}
        fill="oklch(0.97 0 0)"
        rx="18"
        onClick={(e) => {
          if (editing) {
            const svg = e.currentTarget.ownerSVGElement ?? (e.currentTarget as unknown as SVGSVGElement);
            const rect = svg.getBoundingClientRect();
            const { x, y } = clientToViewBox(e.clientX, e.clientY, rect);
            dispatch({ type: 'add-node', attr: defaultNodeAttr, x, y });
          } else {
            updateSelection({ nodeId: null, edgeId: null });
          }
        }}
      />
      {/* Edges */}
      <g>
        {edges.map((edge) => {
          // d3-force replaces source/target numeric ids with node object references
          // after the simulation starts. Resolve to numeric id safely.
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
          const meta = spec.edgeTypes.find((m) => m.id === edge.edgeType);
          return (
            <BubbleEdge
              key={edge.id}
              edge={edge}
              source={source}
              target={target}
              meta={meta}
              selected={selection.edgeId === edge.id}
              onClick={
                editing
                  ? (e) => {
                      e.stopPropagation();
                      // Cycle to next non-none edge type
                      const nonNoneTypes = spec.edgeTypes
                        .filter((m) => m.id !== 0)
                        .map((m) => m.id);
                      const idx = nonNoneTypes.indexOf(edge.edgeType);
                      const nextType = nonNoneTypes[(idx + 1) % nonNoneTypes.length];
                      dispatch({ type: 'cycle-edge-type', id: edge.id, nextType });
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
        })}
      </g>
      {/* Rubber-band edge draft line */}
      {edgeDraft && (() => {
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
          return (
            <BubbleNode
              key={node.id}
              node={node}
              meta={meta}
              selected={selection.nodeId === node.id}
              onPointerDown={handleNodePointerDown(node.id)}
              onClick={
                editing
                  ? (e) => {
                      e.stopPropagation();
                      updateSelection({ nodeId: node.id, edgeId: null });
                    }
                  : undefined
              }
              onDoubleClick={
                editing
                  ? (e) => {
                      e.stopPropagation();
                      // Cycle attr to next room type
                      const nextAttr = (node.attr + 1) % spec.roomTypes.length;
                      dispatch({ type: 'change-node-attr', id: node.id, attr: nextAttr });
                    }
                  : undefined
              }
            />
          );
        })}
      </g>
    </svg>
  );
}
