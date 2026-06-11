'use client';

import type { BubbleEdgeState, BubbleNodeState } from '@/lib/bubble-graph/types';
import type { EdgeTypeMeta } from '@/lib/constants';

interface Props {
  edge: BubbleEdgeState;
  source: BubbleNodeState;
  target: BubbleNodeState;
  meta: EdgeTypeMeta | undefined;
  selected: boolean;
  muted?: boolean;
  showLabel?: boolean;
  neighborhoodDepth?: 1 | 2 | 3 | null;
  dimmed?: boolean;
  onClick?: (e: React.MouseEvent) => void;
  onContextMenu?: (e: React.MouseEvent) => void;
}

export function BubbleEdge({
  edge,
  source,
  target,
  meta,
  selected,
  muted = false,
  showLabel = true,
  neighborhoodDepth = null,
  dimmed = false,
  onClick,
  onContextMenu,
}: Props) {
  const stroke = meta?.color ?? '#475569';
  const dashArray = meta?.dashArray;
  // Label every meaningful edge (none has id=0 and isn't rendered anyway).
  const shouldShowLabel = showLabel && !dimmed && meta && meta.id > 0;
  const hopOpacity = neighborhoodDepth === null ? null : [0, 0.9, 0.52, 0.26][neighborhoodDepth];
  const baseOpacity = dimmed ? 1 : muted ? 0.1 : 0.56;
  const effectiveStroke = dimmed ? 'oklch(0.72 0 0)' : stroke;
  return (
    <g
      onClick={onClick}
      onContextMenu={onContextMenu}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
      data-edge-id={edge.id}
    >
      <line
        x1={source.x}
        y1={source.y}
        x2={target.x}
        y2={target.y}
        stroke={effectiveStroke}
        strokeOpacity={selected ? 1 : hopOpacity ?? baseOpacity}
        strokeWidth={selected ? 4 : muted ? 1.2 : dashArray ? 2.3 : 1.8}
        strokeDasharray={dashArray}
        strokeLinecap="round"
        style={{ transition: 'stroke-opacity 220ms ease-out, stroke-width 220ms ease-out' }}
      />
      {shouldShowLabel && (
        <text
          x={(source.x + target.x) / 2}
          y={(source.y + target.y) / 2 - 6}
          textAnchor="middle"
          fontSize="10"
          fontWeight="600"
          fill={effectiveStroke}
          paintOrder="stroke"
          stroke="oklch(0.97 0 0)"
          strokeWidth="4"
          opacity={selected ? 1 : hopOpacity ?? 0.86}
          style={{ transition: 'opacity 220ms ease-out' }}
        >
          {meta?.name}
        </text>
      )}
    </g>
  );
}
