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

  const [nodes, setNodes] = useState<BubbleNodeState[]>(initialNodes);
  const [edges] = useState<BubbleEdgeState[]>(initialEdges);
  const [selection, setSelection] = useState<BubbleSelection>({ nodeId: null, edgeId: null });

  const handleTick = useCallback((next: BubbleNodeState[]) => {
    setNodes([...next]);
  }, []);

  const { pinNode } = useForceSimulation({
    nodes: initialNodes,
    edges: initialEdges,
    width,
    height,
    onTick: handleTick,
  });

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
      <rect
        width={width}
        height={height}
        fill="oklch(0.97 0 0)"
        rx="18"
        onClick={() => updateSelection({ nodeId: null, edgeId: null })}
      />
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
                interactive
                  ? () => updateSelection({ nodeId: null, edgeId: edge.id })
                  : undefined
              }
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
              onClick={
                interactive
                  ? (e) => {
                      e.stopPropagation();
                      updateSelection({ nodeId: node.id, edgeId: null });
                    }
                  : undefined
              }
            />
          );
        })}
      </g>
      {/* Edit-mode interactions added in Task 8 */}
    </svg>
  );
}
