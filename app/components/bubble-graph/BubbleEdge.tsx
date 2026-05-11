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
  const dashArray = meta?.dashArray;
  // Label every meaningful edge (none has id=0 and isn't rendered anyway).
  const showLabel = meta && meta.id > 0;
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
        stroke={stroke}
        strokeWidth={selected ? 4 : dashArray ? 2.5 : 2}
        strokeDasharray={dashArray}
        strokeLinecap="round"
      />
      {showLabel && (
        <text
          x={(source.x + target.x) / 2}
          y={(source.y + target.y) / 2 - 6}
          textAnchor="middle"
          fontSize="10"
          fontWeight="600"
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
