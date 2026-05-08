'use client';

import type { BubbleNodeState } from '@/lib/bubble-graph/types';
import type { RoomTypeMeta } from '@/lib/constants';

interface Props {
  node: BubbleNodeState;
  meta: RoomTypeMeta | undefined;
  selected: boolean;
  onPointerDown?: (e: React.PointerEvent) => void;
  onClick?: (e: React.MouseEvent) => void;
  onDoubleClick?: (e: React.MouseEvent) => void;
}

export function BubbleNode({ node, meta, selected, onPointerDown, onClick, onDoubleClick }: Props) {
  const fill = meta?.color ?? '#e5e7eb';
  const textColor = meta?.textColor ?? '#111827';
  const label = meta?.name ?? `class_${node.attr}`;
  return (
    <g
      transform={`translate(${node.x},${node.y})`}
      style={{ cursor: onPointerDown ? 'grab' : 'default' }}
      onPointerDown={onPointerDown}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
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
