'use client';

import { ROOM_TYPES } from '@/lib/constants';
import type { GeneratedGraph } from '@/lib/types';

interface Props {
  graph: GeneratedGraph;
}

interface Point {
  x: number;
  y: number;
}

function layoutGraph(graph: GeneratedGraph): Map<number, Point> {
  const width = 640;
  const height = 390;
  const center = { x: width / 2, y: height / 2 };
  const radius = 135;
  const livingNode = graph.nodes.find((node) => node.attr === 0);
  const ringNodes = livingNode
    ? graph.nodes.filter((node) => node.id !== livingNode.id)
    : graph.nodes;
  const positions = new Map<number, Point>();

  if (livingNode) {
    positions.set(livingNode.id, center);
  }

  for (const node of ringNodes) {
    const index = ringNodes.findIndex((candidate) => candidate.id === node.id);
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / Math.max(1, ringNodes.length);
    positions.set(node.id, {
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius,
    });
  }

  return positions;
}

export function BubbleGraphView({ graph }: Props) {
  const positions = layoutGraph(graph);

  return (
    <svg
      viewBox="0 0 640 390"
      role="img"
      aria-label="Generated bubble graph"
      className="h-full w-full"
    >
      <rect width="640" height="390" rx="18" fill="oklch(0.97 0 0)" />
      <g>
        {graph.edges.map((edge) => {
          const source = positions.get(edge.source);
          const target = positions.get(edge.target);
          if (!source || !target) return null;
          const isDoor = edge.edge_type === 2;
          return (
            <g key={`${edge.source}-${edge.target}`}>
              <line
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                stroke={isDoor ? '#c2410c' : '#475569'}
                strokeWidth={isDoor ? 2.5 : 2}
                strokeDasharray={isDoor ? '7 5' : undefined}
                strokeLinecap="round"
              />
              {isDoor && (
                <text
                  x={(source.x + target.x) / 2}
                  y={(source.y + target.y) / 2 - 6}
                  textAnchor="middle"
                  fontSize="10"
                  fill="#9a3412"
                  paintOrder="stroke"
                  stroke="oklch(0.97 0 0)"
                  strokeWidth="4"
                >
                  door
                </text>
              )}
            </g>
          );
        })}
      </g>
      <g>
        {graph.nodes.map((node) => {
          const point = positions.get(node.id);
          if (!point) return null;
          const room = ROOM_TYPES[node.attr];
          return (
            <g key={node.id}>
              <circle
                cx={point.x}
                cy={point.y}
                r="36"
                fill={room?.color ?? '#e5e7eb'}
                stroke="oklch(0.2 0 0 / 0.28)"
                strokeWidth="1.5"
              />
              <text
                x={point.x}
                y={point.y - 2}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="11"
                fontWeight="700"
                fill={room?.textColor ?? '#111827'}
              >
                {room?.name ?? node.room_type}
              </text>
              <text
                x={point.x}
                y={point.y + 14}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="10"
                fill={room?.textColor ?? '#111827'}
                opacity="0.68"
              >
                #{node.id}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}
