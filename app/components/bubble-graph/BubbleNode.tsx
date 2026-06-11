'use client';

import type { BubbleNodeState } from '@/lib/bubble-graph/types';
import type { RoomTypeMeta } from '@/lib/constants';

interface Props {
  node: BubbleNodeState;
  meta: RoomTypeMeta | undefined;
  selected: boolean;
  /** When true, render an outer glow ring (used in retrieval results to flag
   *  candidate nodes that align with a query node). */
  highlighted?: boolean;
  neighborhoodDepth?: 0 | 1 | 2 | 3 | null;
  dimmed?: boolean;
  onPointerDown?: (e: React.PointerEvent) => void;
  onPointerEnter?: (e: React.PointerEvent) => void;
  onPointerLeave?: (e: React.PointerEvent) => void;
  onClick?: (e: React.MouseEvent) => void;
  onDoubleClick?: (e: React.MouseEvent) => void;
}

export function BubbleNode({
  node,
  meta,
  selected,
  highlighted = false,
  neighborhoodDepth = null,
  dimmed = false,
  onPointerDown,
  onPointerEnter,
  onPointerLeave,
  onClick,
  onDoubleClick,
}: Props) {
  const fill = meta?.color ?? '#e5e7eb';
  const textColor = meta?.textColor ?? '#111827';
  const label = meta?.name ?? `class_${node.attr}`;
  const neighborhoodStrokeOpacity =
    neighborhoodDepth === null ? 0 : [1, 0.75, 0.48, 0.28][neighborhoodDepth];
  const nodeOpacity =
    dimmed ? 0.16 : neighborhoodDepth === null ? 1 : [1, 0.94, 0.82, 0.7][neighborhoodDepth];
  return (
    <g
      transform={`translate(${node.x},${node.y})`}
      style={{
        cursor: onPointerDown ? 'grab' : 'default',
        opacity: nodeOpacity,
        transition: 'opacity 220ms ease-out',
      }}
      onPointerDown={onPointerDown}
      onPointerEnter={onPointerEnter}
      onPointerLeave={onPointerLeave}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      {highlighted && (
        <>
          <circle r="46" fill="none" stroke="oklch(0.78 0.18 75)" strokeWidth={3} opacity={0.45} />
          <circle r="41" fill="none" stroke="oklch(0.68 0.22 50)" strokeWidth={2.5} />
        </>
      )}
      <circle
        r="36"
        fill={fill}
        stroke={
          selected
            ? 'oklch(0.55 0.21 35)'
            : neighborhoodDepth !== null
              ? 'oklch(0.2 0 0)'
              : 'oklch(0.2 0 0 / 0.28)'
        }
        strokeOpacity={selected ? 1 : neighborhoodDepth !== null ? neighborhoodStrokeOpacity : 1}
        strokeWidth={selected ? 3 : neighborhoodDepth !== null ? 2 : 1.5}
        style={{
          transition: 'stroke 220ms ease-out, stroke-opacity 220ms ease-out, stroke-width 220ms ease-out',
        }}
      />
      <text
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize="11"
        fontWeight="700"
        fill={dimmed ? 'oklch(0.45 0 0)' : textColor}
        y={-2}
      >
        {label}
      </text>
      <text
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize="10"
        fill={dimmed ? 'oklch(0.45 0 0)' : textColor}
        opacity={dimmed ? 0.35 : 0.68}
        y={14}
      >
        #{node.id}
      </text>
    </g>
  );
}
