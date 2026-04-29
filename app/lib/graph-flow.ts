import type { Edge, Node } from '@xyflow/react';
import { ROOM_TYPES } from '@/lib/constants';
import type { GeneratedGraph } from '@/lib/types';
import type { RoomNodeData } from '@/components/RoomNode';

export function graphToFlow(graph: GeneratedGraph): { nodes: Node[]; edges: Edge[] } {
  const center = { x: 220, y: 160 };
  const radius = 150;
  const livingNode = graph.nodes.find((node) => node.attr === 0);
  const ringNodes = livingNode
    ? graph.nodes.filter((node) => node.id !== livingNode.id)
    : graph.nodes;

  const nodes: Node[] = graph.nodes.map((graphNode) => {
    const roomType = ROOM_TYPES[graphNode.attr];
    let position = center;
    if (!livingNode || graphNode.id !== livingNode.id) {
      const ringIndex = ringNodes.findIndex((node) => node.id === graphNode.id);
      const angle = -Math.PI / 2 + (2 * Math.PI * ringIndex) / Math.max(1, ringNodes.length);
      position = {
        x: center.x + Math.cos(angle) * radius,
        y: center.y + Math.sin(angle) * radius,
      };
    }

    return {
      id: `room-${graphNode.id}`,
      type: 'room',
      position,
      data: {
        roomTypeId: graphNode.attr,
        label: roomType?.name ?? graphNode.room_type,
      } satisfies RoomNodeData,
    };
  });

  const edges: Edge[] = graph.edges.map((edge) => ({
    id: `edge-${edge.source}-${edge.target}`,
    source: `room-${edge.source}`,
    target: `room-${edge.target}`,
    sourceHandle: 'room-handle',
    targetHandle: 'room-target',
    label: edge.edge_type === 2 ? 'door' : undefined,
    type: 'straight',
    style: {
      stroke: edge.edge_type === 2 ? '#c2410c' : 'oklch(0.439 0 0)',
      strokeWidth: edge.edge_type === 2 ? 2 : 1.75,
      strokeDasharray: edge.edge_type === 2 ? '4 3' : undefined,
    },
  }));

  return { nodes, edges };
}

export function flowToTopology(nodes: Node[], edges: Edge[]): { rooms: number[]; adjacency: number[][] } {
  const nodeIds = nodes.map((node) => node.id);
  const nodeIndexMap = new Map(nodeIds.map((id, index) => [id, index]));
  const rooms = nodes.map((node) => (node.data as unknown as RoomNodeData).roomTypeId);
  const adjacency: number[][] = Array.from({ length: rooms.length }, () => Array(rooms.length).fill(0));

  for (const edge of edges) {
    const sourceIndex = nodeIndexMap.get(edge.source);
    const targetIndex = nodeIndexMap.get(edge.target);
    if (sourceIndex !== undefined && targetIndex !== undefined) {
      adjacency[sourceIndex][targetIndex] = 1;
      adjacency[targetIndex][sourceIndex] = 1;
    }
  }

  return { rooms, adjacency };
}
