'use client';

import { useMemo } from 'react';
import type { Plan, RoomType } from '@/lib/plan';
import { planBounds } from '@/lib/plan';
import { wallQuad, polyPath, pointAlongWall } from './walls';

// Reuse a room-type → colour mapping consistent with the rest of the app.
const ROOM_COLORS: Record<RoomType, string> = {
  Livingroom: '#aec7e8', Bedroom: '#1f77b4', Kitchen: '#ff7f0e',
  Dining: '#ffbb78', Corridor: '#2ca02c', Stairs: '#98df8a',
  Storeroom: '#d62728', Bathroom: '#ff9896', Balcony: '#9467bd',
};

interface Props {
  plan: Plan;
}

export function FloorPlanEditor2D({ plan }: Props) {
  const b = useMemo(() => planBounds(plan), [plan]);
  const pad = 1; // metres
  const vb = `${b.minX - pad} ${b.minY - pad} ${b.maxX - b.minX + 2 * pad} ${b.maxY - b.minY + 2 * pad}`;

  // grid lines spanning the bounds (axis-aligned for Phase 1; rotation in P2)
  const gridLines: { x1: number; y1: number; x2: number; y2: number }[] = [];
  for (let x = Math.floor(b.minX); x <= Math.ceil(b.maxX); x += plan.grid.spacingX) {
    gridLines.push({ x1: x, y1: b.minY - pad, x2: x, y2: b.maxY + pad });
  }
  for (let y = Math.floor(b.minY); y <= Math.ceil(b.maxY); y += plan.grid.spacingY) {
    gridLines.push({ x1: b.minX - pad, y1: y, x2: b.maxX + pad, y2: y });
  }

  return (
    <svg
      viewBox={vb}
      className="h-full w-full bg-background"
      style={{ transform: 'scaleY(-1)' }} // metres: +y is up
    >
      {/* axis grid */}
      <g stroke="#e5e7eb" strokeWidth={0.02}>
        {gridLines.map((l, i) => (
          <line key={`g${i}`} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} strokeDasharray="0.1 0.1" />
        ))}
      </g>

      {/* room fills */}
      <g>
        {plan.rooms.map((r) => (
          <path
            key={r.id}
            d={polyPath(r.poly)}
            fill={ROOM_COLORS[r.type] ?? '#cccccc'}
            fillOpacity={0.5}
            stroke="none"
          />
        ))}
      </g>

      {/* double-line walls (filled rectangles) */}
      <g fill="#1f2937">
        {plan.walls.map((w) => (
          <path key={w.id} d={polyPath(wallQuad(w))} />
        ))}
      </g>

      {/* openings: erase the wall span with a background-coloured rectangle */}
      <g fill="hsl(0 0% 100%)">
        {plan.openings.map((o) => {
          const wall = plan.walls.find((w) => w.id === o.wallId);
          if (!wall) return null;
          const t0 = Math.max(0, o.t - o.width / 2 / (Math.hypot(wall.b[0] - wall.a[0], wall.b[1] - wall.a[1]) || 1));
          const t1 = Math.min(1, o.t + o.width / 2 / (Math.hypot(wall.b[0] - wall.a[0], wall.b[1] - wall.a[1]) || 1));
          const p0 = pointAlongWall(wall, t0);
          const p1 = pointAlongWall(wall, t1);
          const seg = { ...wall, a: p0, b: p1 };
          return <path key={o.id} d={polyPath(wallQuad(seg))} />;
        })}
      </g>

      {/* room labels (flip text back upright) */}
      <g>
        {plan.rooms.map((r) => {
          const cx = r.poly.reduce((s, p) => s + p[0], 0) / r.poly.length;
          const cy = r.poly.reduce((s, p) => s + p[1], 0) / r.poly.length;
          return (
            <text
              key={`t${r.id}`}
              x={cx}
              y={cy}
              fontSize={0.4}
              textAnchor="middle"
              fill="#111827"
              transform={`scale(1,-1) translate(0, ${-2 * cy})`}
            >
              {r.type}
            </text>
          );
        })}
      </g>
    </svg>
  );
}
