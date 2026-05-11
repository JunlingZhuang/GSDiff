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
    case 'set-edge-type': {
      return {
        ...checkpointed,
        edges: state.edges.map((e) =>
          e.id === action.id ? { ...e, edgeType: action.edgeType } : e,
        ),
      };
    }
    case 'add-node-and-edge': {
      // Single undo step: create node + edge
      const newId = state.nodes.reduce((m, n) => Math.max(m, n.id), -1) + 1;
      const node: BubbleNodeState = {
        id: newId,
        attr: action.attr,
        x: action.x,
        y: action.y,
        fx: action.x,
        fy: action.y,
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
}: Props) {
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
    // Re-init only when graph identity changes
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
  // State
  // ---------------------------------------------------------------------------
  const [viewNodes, setViewNodes] = useState<BubbleNodeState[]>(initialNodes);

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

  // Edge-creation rubber-band state (edit mode only)
  const [edgeDraft, setEdgeDraft] = useState<{
    sourceId: number;
    cursor: { x: number; y: number };
  } | null>(null);

  // Keep stable ref to nodes for pointer callbacks
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
  // onChange emission
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

  // Convert SVG viewBox coords → container-relative pixels
  const viewBoxToContainer = useCallback(
    (vx: number, vy: number): { px: number; py: number } => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return { px: vx, py: vy };
      const scaleX = rect.width / width;
      const scaleY = rect.height / height;
      return { px: vx * scaleX, py: vy * scaleY };
    },
    [width, height],
  );

  // Convert client coordinates to SVG viewBox coordinates
  const clientToViewBox = (
    clientX: number,
    clientY: number,
    rect: DOMRect,
  ): { x: number; y: number } => ({
    x: ((clientX - rect.left) / rect.width) * width,
    y: ((clientY - rect.top) / rect.height) * height,
  });

  // Drag node to move it
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
        dispatch({
          type: 'add-edge',
          source: sourceId,
          target: hit.id,
          edgeType: defaultEdgeType,
        });
      } else {
        // Drop on empty space → add node + edge as one undo step
        dispatch({
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
    dispatch({
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
          if (e.target !== e.currentTarget) return;
          if (editing) {
            const rect = e.currentTarget.getBoundingClientRect();
            const { x, y } = clientToViewBox(e.clientX, e.clientY, rect);
            dispatch({ type: 'add-node', attr: defaultNodeAttr, x, y });
          } else {
            updateSelection({ nodeId: null, edgeId: null });
          }
        }}
      >
        {/* Edges */}
        <g>
          {edges.map((edge) => {
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
            return (
              <g key={node.id}>
                <BubbleNode
                  node={node}
                  meta={meta}
                  selected={selection.nodeId === node.id}
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
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
