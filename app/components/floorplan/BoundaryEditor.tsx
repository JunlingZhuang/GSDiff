'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import type { Pt } from '@/lib/plan';

interface Props {
  value: Pt[];
  onChange: (pts: Pt[]) => void;
}

// Lightweight polygon editor for the building boundary (metres). Drag vertices,
// click a green midpoint to insert, double-click a vertex to delete, or snap to
// a W×H rectangle. Rendered y-down for simplicity; the generator handles axis.
export function BoundaryEditor({ value, onChange }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragIdx = useRef<number | null>(null);
  const [rect, setRect] = useState({ w: 12, h: 9 });

  const view = useMemo(() => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of value) {
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
    if (!isFinite(minX)) { minX = 0; minY = 0; maxX = 12; maxY = 9; }
    const m = Math.max(1, (maxX - minX + maxY - minY) * 0.08);
    return { x: minX - m, y: minY - m, w: maxX - minX + 2 * m, h: maxY - minY + 2 * m };
  }, [value]);

  const toWorld = useCallback((clientX: number, clientY: number): Pt => {
    const svg = svgRef.current!;
    const p = svg.createSVGPoint();
    p.x = clientX;
    p.y = clientY;
    const wp = p.matrixTransform(svg.getScreenCTM()!.inverse());
    return [wp.x, wp.y];
  }, []);

  const onMove = useCallback(
    (e: React.PointerEvent) => {
      if (dragIdx.current == null) return;
      const w = toWorld(e.clientX, e.clientY);
      const next = value.map((p, i) => (i === dragIdx.current ? w : p));
      onChange(next);
    },
    [onChange, toWorld, value],
  );

  const insertAfter = useCallback(
    (i: number) => {
      const a = value[i];
      const b = value[(i + 1) % value.length];
      const mid: Pt = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      const next = [...value.slice(0, i + 1), mid, ...value.slice(i + 1)];
      onChange(next);
    },
    [onChange, value],
  );

  const removeVertex = useCallback(
    (i: number) => {
      if (value.length <= 3) return;
      onChange(value.filter((_, j) => j !== i));
    },
    [onChange, value],
  );

  const stroke = view.w / 320;
  const dPath = value.length
    ? `M ${value.map((p) => `${p[0]} ${p[1]}`).join(' L ')} Z`
    : '';

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs">
        <span className="text-muted-foreground">Rectangle</span>
        <input
          type="number"
          className="w-14 rounded border border-border bg-background px-1 py-0.5"
          value={rect.w}
          min={1}
          onChange={(e) => setRect((r) => ({ ...r, w: Number(e.target.value) || 1 }))}
        />
        <span className="text-muted-foreground">×</span>
        <input
          type="number"
          className="w-14 rounded border border-border bg-background px-1 py-0.5"
          value={rect.h}
          min={1}
          onChange={(e) => setRect((r) => ({ ...r, h: Number(e.target.value) || 1 }))}
        />
        <span className="text-muted-foreground">m</span>
        <button
          className="rounded border border-border px-2 py-0.5 hover:bg-muted"
          onClick={() => onChange([[0, 0], [rect.w, 0], [rect.w, rect.h], [0, rect.h]])}
        >
          Apply
        </button>
      </div>
      <svg
        ref={svgRef}
        viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
        className="h-44 w-full rounded-md border border-border bg-muted/30"
        onPointerMove={onMove}
        onPointerUp={() => (dragIdx.current = null)}
        onPointerLeave={() => (dragIdx.current = null)}
      >
        {dPath && <path d={dPath} fill="#94a3b8" fillOpacity={0.18} stroke="#64748b" strokeWidth={stroke * 1.5} />}
        {/* insert midpoints */}
        {value.map((p, i) => {
          const b = value[(i + 1) % value.length];
          const mx = (p[0] + b[0]) / 2;
          const my = (p[1] + b[1]) / 2;
          return (
            <circle
              key={`m${i}`}
              cx={mx}
              cy={my}
              r={stroke * 3}
              fill="#22c55e"
              style={{ cursor: 'copy' }}
              onPointerDown={(e) => {
                e.stopPropagation();
                insertAfter(i);
              }}
            />
          );
        })}
        {/* vertices */}
        {value.map((p, i) => (
          <circle
            key={`v${i}`}
            cx={p[0]}
            cy={p[1]}
            r={stroke * 4}
            fill="#0ea5e9"
            stroke="#fff"
            strokeWidth={stroke}
            style={{ cursor: 'grab' }}
            onPointerDown={(e) => {
              e.stopPropagation();
              dragIdx.current = i;
              svgRef.current?.setPointerCapture(e.pointerId);
            }}
            onDoubleClick={(e) => {
              e.stopPropagation();
              removeVertex(i);
            }}
          />
        ))}
      </svg>
      <p className="text-[11px] text-muted-foreground">
        Drag dots to reshape · green = add point · double-click a point to delete
      </p>
    </div>
  );
}
