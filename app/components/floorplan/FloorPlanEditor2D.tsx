'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { select } from 'd3-selection';
import { zoom, zoomIdentity, type ZoomBehavior } from 'd3-zoom';
import { roomColor, ROOM_TYPE_OPTIONS, type Pt, type RoomType, type OpeningKind, type AxisGrid } from '@/lib/plan';
import { polyPath } from './walls';
import {
  type WallGraph,
  wallEnds,
  roomPoly,
  wallSolid,
  multiPolyPath,
  graphBounds,
  dragWallSeg,
  setRoomType,
  addOpening,
  moveOpening,
  deleteOpening,
  setOpeningKind,
  setOpeningWidth,
  splitWall,
  wallLength,
  roomArea,
  paramOnWall,
  perpDrag,
} from './kernel';

const DEFAULT_GRID: AxisGrid = { originX: 0, originY: 0, spacingX: 1, spacingY: 1, angleDeg: 0 };

export type EditorTool = 'select' | 'door' | 'passage' | 'boundary';

interface Props {
  graph: WallGraph | null;
  boundary: Pt[];
  onChange?: (g: WallGraph) => void;
  onBoundaryChange?: (pts: Pt[]) => void;
  gridSnap?: boolean;
  tool?: EditorTool;
  fitSignature?: number;
  /** show length labels on every wall (selected wall is always dimensioned) */
  showDims?: boolean;
}

type Selection = { type: 'room' | 'wall' | 'opening'; id: string } | null;
type DragState =
  | { kind: 'wall'; id: string; start: Pt; orig: WallGraph }
  | { kind: 'opening'; id: string; wallId: string; orig: WallGraph }
  | { kind: 'bvertex'; idx: number; orig: Pt[] }
  | null;

export function FloorPlanEditor2D({
  graph,
  boundary,
  onChange,
  onBoundaryChange,
  gridSnap = false,
  tool = 'select',
  fitSignature = 0,
  showDims = false,
}: Props) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const dragRef = useRef<DragState>(null);
  const toolRef = useRef<EditorTool>(tool);
  const editable = Boolean(onChange);
  const grid = graph?.grid ?? DEFAULT_GRID;

  const [t, setT] = useState({ k: 1, x: 0, y: 0 });
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [sel, setSel] = useState<Selection>(null);

  useEffect(() => {
    toolRef.current = tool;
  }, [tool]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const z = zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.05, 120])
      .filter((event: Event) => {
        if (event.type === 'wheel') return true;
        const target = event.target as Element | null;
        if (target && target.closest('[data-interactive]')) return false;
        if (toolRef.current === 'boundary') return false;
        const me = event as MouseEvent;
        return !me.ctrlKey && me.button === 0;
      })
      .on('zoom', (event) => setT({ k: event.transform.k, x: event.transform.x, y: event.transform.y }));
    zoomRef.current = z;
    select(svg).call(z);
    return () => {
      select(svg).on('.zoom', null);
    };
  }, []);

  const solid = useMemo(() => (graph ? wallSolid(graph) : []), [graph]);

  const bounds = useMemo(() => {
    const b = graph ? graphBounds(graph) : { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
    for (const [x, y] of boundary) {
      if (x < b.minX) b.minX = x;
      if (y < b.minY) b.minY = y;
      if (x > b.maxX) b.maxX = x;
      if (y > b.maxY) b.maxY = y;
    }
    if (!isFinite(b.minX)) return { minX: 0, minY: 0, maxX: 12, maxY: 9 };
    return b;
  }, [graph, boundary]);

  const fitToContent = useCallback(() => {
    const svg = svgRef.current;
    const z = zoomRef.current;
    if (!svg || !z || size.w === 0) return;
    const pad = 1.5;
    const k = Math.min(
      size.w / Math.max(1e-3, bounds.maxX - bounds.minX + 2 * pad),
      size.h / Math.max(1e-3, bounds.maxY - bounds.minY + 2 * pad),
    ) || 1;
    const cx = (bounds.minX + bounds.maxX) / 2;
    const cy = (bounds.minY + bounds.maxY) / 2;
    select(svg).call(z.transform, zoomIdentity.translate(size.w / 2 - k * cx, size.h / 2 - k * -cy).scale(k));
  }, [bounds, size]);

  const fittedFor = useRef<number | null>(null);
  useEffect(() => {
    if (size.w === 0) return;
    if (fittedFor.current !== fitSignature) {
      fittedFor.current = fitSignature;
      fitToContent();
    }
  }, [fitSignature, size, fitToContent]);

  const toWorld = useCallback(
    (cx: number, cy: number): Pt => {
      const rect = svgRef.current!.getBoundingClientRect();
      return [(cx - rect.left - t.x) / t.k, -((cy - rect.top - t.y) / t.k)];
    },
    [t],
  );

  // ── interactions ──
  const beginWall = useCallback(
    (e: React.PointerEvent, wallId: string) => {
      e.stopPropagation();
      setSel({ type: 'wall', id: wallId });
      if (!editable || !graph) return;
      dragRef.current = { kind: 'wall', id: wallId, start: toWorld(e.clientX, e.clientY), orig: graph };
      try {
        svgRef.current?.setPointerCapture(e.pointerId);
      } catch {
        /* synthetic / already-captured pointer */
      }
    },
    [editable, graph, toWorld],
  );

  const clickWallForOpening = useCallback(
    (e: React.PointerEvent, wallId: string) => {
      e.stopPropagation();
      if (!editable || !graph) return;
      const tt = paramOnWall(graph, wallId, toWorld(e.clientX, e.clientY));
      onChange?.(addOpening(graph, wallId, tt, tool === 'passage' ? 'passage' : 'door'));
    },
    [editable, graph, onChange, tool, toWorld],
  );

  const beginOpening = useCallback(
    (e: React.PointerEvent, openingId: string, wallId: string) => {
      e.stopPropagation();
      setSel({ type: 'opening', id: openingId });
      if (!editable || !graph) return;
      dragRef.current = { kind: 'opening', id: openingId, wallId, orig: graph };
      try {
        svgRef.current?.setPointerCapture(e.pointerId);
      } catch {
        /* synthetic / already-captured pointer */
      }
    },
    [editable, graph],
  );

  const beginBoundaryVertex = useCallback(
    (e: React.PointerEvent, idx: number) => {
      e.stopPropagation();
      if (!onBoundaryChange) return;
      dragRef.current = { kind: 'bvertex', idx, orig: boundary };
      try {
        svgRef.current?.setPointerCapture(e.pointerId);
      } catch {
        /* synthetic / already-captured pointer */
      }
    },
    [boundary, onBoundaryChange],
  );

  const insertBoundaryVertex = useCallback(
    (e: React.PointerEvent, i: number) => {
      e.stopPropagation();
      if (!onBoundaryChange) return;
      const a = boundary[i];
      const b = boundary[(i + 1) % boundary.length];
      onBoundaryChange([...boundary.slice(0, i + 1), [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], ...boundary.slice(i + 1)]);
    },
    [boundary, onBoundaryChange],
  );

  const onSvgPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (tool === 'boundary' && onBoundaryChange) {
        onBoundaryChange([...boundary, toWorld(e.clientX, e.clientY)]);
        return;
      }
      setSel(null);
    },
    [boundary, onBoundaryChange, tool, toWorld],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      const d = dragRef.current;
      if (!d) return;
      const p = toWorld(e.clientX, e.clientY);
      if (d.kind === 'wall') {
        onChange?.(dragWallSeg(d.orig, d.id, perpDrag(d.orig, d.id, d.start, p), gridSnap));
      } else if (d.kind === 'opening') {
        onChange?.(moveOpening(d.orig, d.id, paramOnWall(d.orig, d.wallId, p)));
      } else if (d.kind === 'bvertex') {
        onBoundaryChange?.(d.orig.map((pt, i) => (i === d.idx ? p : pt)));
      }
    },
    [gridSnap, onChange, onBoundaryChange, toWorld],
  );

  const endDrag = useCallback((e: React.PointerEvent) => {
    if (dragRef.current) {
      dragRef.current = null;
      try {
        svgRef.current?.releasePointerCapture(e.pointerId);
      } catch {
        /* released */
      }
    }
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && sel?.type === 'opening' && onChange && graph) {
        onChange(deleteOpening(graph, sel.id));
        setSel(null);
      }
      if (e.key === 'Escape') setSel(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [sel, graph, onChange]);

  const gridLines = useMemo(() => {
    const lines: { x1: number; y1: number; x2: number; y2: number }[] = [];
    const sx = grid.spacingX || 1;
    const sy = grid.spacingY || 1;
    const pad = 4;
    const minX = Math.floor(bounds.minX - pad);
    const maxX = Math.ceil(bounds.maxX + pad);
    const minY = Math.floor(bounds.minY - pad);
    const maxY = Math.ceil(bounds.maxY + pad);
    for (let x = minX; x <= maxX; x += sx) lines.push({ x1: x, y1: minY, x2: x, y2: maxY });
    for (let y = minY; y <= maxY; y += sy) lines.push({ x1: minX, y1: y, x2: maxX, y2: y });
    return lines;
  }, [grid, bounds]);

  const angle = grid.angleDeg || 0;
  const gcx = (bounds.minX + bounds.maxX) / 2;
  const gcy = (bounds.minY + bounds.maxY) / 2;
  const sw = 1 / t.k;
  const selWall = sel?.type === 'wall' && graph ? graph.walls.find((w) => w.id === sel.id) : null;

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden bg-background">
      <svg
        ref={svgRef}
        width={size.w}
        height={size.h}
        className="touch-none select-none"
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
        onPointerDown={onSvgPointerDown}
      >
        <g transform={`translate(${t.x},${t.y}) scale(${t.k})`}>
          <g transform="scale(1,-1)">
            {/* grid */}
            <g transform={`rotate(${angle} ${gcx} ${gcy})`} stroke="#e8edf3" strokeWidth={sw} strokeDasharray={`${4 * sw} ${4 * sw}`}>
              {gridLines.map((l, i) => (
                <line key={`g${i}`} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} />
              ))}
            </g>

            {/* boundary */}
            {boundary.length > 1 && (
              <path
                d={polyPath(boundary)}
                fill={graph ? 'none' : '#f1f5f9'}
                fillOpacity={graph ? 0 : 0.6}
                stroke="#64748b"
                strokeWidth={2.5 * sw}
                strokeDasharray={`${6 * sw} ${3 * sw}`}
              />
            )}

            {/* rooms */}
            {graph?.rooms.map((r) => {
              const selected = sel?.type === 'room' && sel.id === r.id;
              return (
                <path
                  key={r.id}
                  data-interactive
                  d={polyPath(roomPoly(graph, r))}
                  fill={roomColor(r.type)}
                  fillOpacity={selected ? 0.68 : 0.38}
                  stroke={selected ? '#0f172a' : 'none'}
                  strokeWidth={selected ? 1.5 * sw : 0}
                  style={{ cursor: tool === 'boundary' ? 'crosshair' : 'pointer' }}
                  onPointerDown={(e) => {
                    if (tool === 'boundary') return;
                    e.stopPropagation();
                    setSel({ type: 'room', id: r.id });
                  }}
                />
              );
            })}

            {/* wall solid: mitred double-line (white body + dark outline) */}
            {solid.length > 0 && (
              <path d={multiPolyPath(solid)} fill="#ffffff" stroke="#1f2937" strokeWidth={1.4 * sw} fillRule="evenodd" style={{ pointerEvents: 'none' }} />
            )}

            {/* door / window symbols */}
            {graph?.openings.map((o) => {
              const w = graph.walls.find((x) => x.id === o.wallId);
              if (!w) return null;
              const { a, b } = wallEnds(graph, w);
              const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
              const half = o.width / 2 / len;
              const g0: Pt = [a[0] + (b[0] - a[0]) * Math.max(0, o.t - half), a[1] + (b[1] - a[1]) * Math.max(0, o.t - half)];
              const g1: Pt = [a[0] + (b[0] - a[0]) * Math.min(1, o.t + half), a[1] + (b[1] - a[1]) * Math.min(1, o.t + half)];
              const L = Math.hypot(g1[0] - g0[0], g1[1] - g0[1]);
              const nx = -(b[1] - a[1]) / len;
              const ny = (b[0] - a[0]) / len;
              const tip: Pt = [g0[0] + nx * L, g0[1] + ny * L];
              return (
                <g key={`sym${o.id}`} style={{ pointerEvents: 'none' }}>
                  {o.kind === 'door' && (
                    <>
                      <line x1={g0[0]} y1={g0[1]} x2={tip[0]} y2={tip[1]} stroke="#c2410c" strokeWidth={1.2 * sw} />
                      <path
                        d={`M ${tip[0].toFixed(3)} ${tip[1].toFixed(3)} A ${L.toFixed(3)} ${L.toFixed(3)} 0 0 1 ${g1[0].toFixed(3)} ${g1[1].toFixed(3)}`}
                        fill="none"
                        stroke="#c2410c"
                        strokeWidth={sw}
                        strokeDasharray={`${2 * sw} ${2 * sw}`}
                      />
                    </>
                  )}
                  {o.kind === 'window' && <line x1={g0[0]} y1={g0[1]} x2={g1[0]} y2={g1[1]} stroke="#0ea5e9" strokeWidth={1.4 * sw} />}
                </g>
              );
            })}

            {/* wall hit-lines (select / drag / add-opening) */}
            {tool !== 'boundary' &&
              graph?.walls.map((w) => {
                const { a, b } = wallEnds(graph, w);
                return (
                  <line
                    key={`hit${w.id}`}
                    data-interactive
                    x1={a[0]}
                    y1={a[1]}
                    x2={b[0]}
                    y2={b[1]}
                    stroke="#000"
                    strokeOpacity={0}
                    strokeWidth={Math.max(w.thickness, 0.45)}
                    style={{ pointerEvents: 'stroke', cursor: tool === 'select' ? 'move' : 'crosshair' }}
                    onPointerDown={(e) => (tool === 'select' ? beginWall(e, w.id) : clickWallForOpening(e, w.id))}
                  />
                );
              })}

            {/* selected wall move gizmo: double-headed arrow along the wall
                normal. Junction nodes are intentionally NOT draggable — walls
                move as rigid collinear chains, which keeps the plan orthogonal. */}
            {selWall && tool === 'select' && graph && (() => {
              const { a, b } = wallEnds(graph, selWall);
              const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
              const ux = (b[0] - a[0]) / len;
              const uy = (b[1] - a[1]) / len;
              const nx = -uy;
              const ny = ux;
              const mx = (a[0] + b[0]) / 2;
              const my = (a[1] + b[1]) / 2;
              const L = 26 * sw;
              const head = 8 * sw;
              const cursor = Math.abs(nx) > Math.abs(ny) ? 'ew-resize' : 'ns-resize';
              const tri = (dir: 1 | -1) => {
                const tx = mx + nx * dir * L;
                const ty = my + ny * dir * L;
                const bx = tx - nx * dir * head;
                const by = ty - ny * dir * head;
                return `${tx},${ty} ${bx + ux * head * 0.55},${by + uy * head * 0.55} ${bx - ux * head * 0.55},${by - uy * head * 0.55}`;
              };
              return (
                <g
                  data-interactive
                  style={{ pointerEvents: 'all', cursor }}
                  onPointerDown={(e) => beginWall(e, selWall.id)}
                >
                  {/* fat invisible hit area */}
                  <line x1={mx - nx * L} y1={my - ny * L} x2={mx + nx * L} y2={my + ny * L} stroke="#000" strokeOpacity={0} strokeWidth={16 * sw} />
                  <line x1={mx - nx * (L - head)} y1={my - ny * (L - head)} x2={mx + nx * (L - head)} y2={my + ny * (L - head)} stroke="#2563eb" strokeWidth={2 * sw} />
                  <polygon points={tri(1)} fill="#2563eb" />
                  <polygon points={tri(-1)} fill="#2563eb" />
                  <circle cx={mx} cy={my} r={4 * sw} fill="#ffffff" stroke="#2563eb" strokeWidth={2 * sw} />
                </g>
              );
            })()}

            {/* opening drag handles */}
            {tool !== 'boundary' &&
              graph?.openings.map((o) => {
                const w = graph.walls.find((x) => x.id === o.wallId);
                if (!w) return null;
                const { a, b } = wallEnds(graph, w);
                const c: Pt = [a[0] + (b[0] - a[0]) * o.t, a[1] + (b[1] - a[1]) * o.t];
                const selected = sel?.type === 'opening' && sel.id === o.id;
                return (
                  <circle
                    key={o.id}
                    data-interactive
                    cx={c[0]}
                    cy={c[1]}
                    r={Math.max(0.1, 5 * sw)}
                    fill={selected ? '#2563eb' : '#0ea5e9'}
                    stroke="#fff"
                    strokeWidth={sw}
                    style={{ pointerEvents: 'all', cursor: 'grab' }}
                    onPointerDown={(e) => beginOpening(e, o.id, o.wallId)}
                  />
                );
              })}

            {/* boundary editing handles */}
            {tool === 'boundary' &&
              onBoundaryChange &&
              boundary.map((p, i) => {
                const b = boundary[(i + 1) % boundary.length];
                return (
                  <g key={`bv${i}`}>
                    <circle data-interactive cx={(p[0] + b[0]) / 2} cy={(p[1] + b[1]) / 2} r={3 * sw} fill="#22c55e" style={{ pointerEvents: 'all', cursor: 'copy' }} onPointerDown={(e) => insertBoundaryVertex(e, i)} />
                    <circle
                      data-interactive
                      cx={p[0]}
                      cy={p[1]}
                      r={5 * sw}
                      fill="#0ea5e9"
                      stroke="#fff"
                      strokeWidth={sw}
                      style={{ pointerEvents: 'all', cursor: 'grab' }}
                      onPointerDown={(e) => beginBoundaryVertex(e, i)}
                      onDoubleClick={(e) => {
                        e.stopPropagation();
                        if (boundary.length > 3) onBoundaryChange(boundary.filter((_, j) => j !== i));
                      }}
                    />
                  </g>
                );
              })}

            {/* labels: room type + area */}
            {graph?.rooms.map((r) => {
              const poly = roomPoly(graph, r);
              const cx = poly.reduce((s, p) => s + p[0], 0) / poly.length;
              const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length;
              const area = roomArea(graph, r);
              return (
                <g key={`t${r.id}`} style={{ pointerEvents: 'none', userSelect: 'none' }} transform={`scale(1,-1) translate(0, ${-2 * cy})`}>
                  <text x={cx} y={cy - 0.12} fontSize={0.4} textAnchor="middle" fill="#0f172a">
                    {r.type}
                  </text>
                  <text x={cx} y={cy + 0.38} fontSize={0.3} textAnchor="middle" fill="#475569">
                    {area.toFixed(1)} m²
                  </text>
                </g>
              );
            })}

            {/* wall dimension labels: every wall when showDims, else the selection */}
            {graph?.walls.map((w) => {
              const selected = sel?.type === 'wall' && sel.id === w.id;
              if (!showDims && !selected) return null;
              const len = wallLength(graph, w);
              if (!selected && len < 0.6) return null; // skip clutter on stubs
              const { a, b } = wallEnds(graph, w);
              const mx = (a[0] + b[0]) / 2;
              const my = (a[1] + b[1]) / 2;
              const nx = -(b[1] - a[1]) / (len || 1);
              const ny = (b[0] - a[0]) / (len || 1);
              const lx = mx + nx * 0.35;
              const ly = my + ny * 0.35;
              return (
                <g key={`dim${w.id}`} style={{ pointerEvents: 'none', userSelect: 'none' }} transform={`scale(1,-1) translate(0, ${-2 * ly})`}>
                  <text x={lx} y={ly} fontSize={0.32} textAnchor="middle" fill={selected ? '#2563eb' : '#64748b'} fontWeight={selected ? 600 : 400}>
                    {len.toFixed(2)} m
                  </text>
                </g>
              );
            })}
          </g>
        </g>
      </svg>

      {/* properties panel */}
      {editable && tool !== 'boundary' && sel && graph && (
        <div className="absolute right-3 top-3 w-52 rounded-xl border border-border/70 bg-card/95 p-3 text-xs shadow-lg backdrop-blur [animation:fpFadeSlide_140ms_ease-out]">
          {sel.type === 'room' && (() => {
            const room = graph.rooms.find((r) => r.id === sel.id);
            if (!room) return null;
            return (
              <div className="space-y-2">
                <p className="font-semibold text-foreground">Room</p>
                <label className="block text-muted-foreground">
                  Type
                  <select className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1 text-foreground" value={room.type} onChange={(e) => onChange?.(setRoomType(graph, room.id, e.target.value as RoomType))}>
                    {ROOM_TYPE_OPTIONS.map((rt) => (
                      <option key={rt} value={rt}>{rt}</option>
                    ))}
                  </select>
                </label>
                <p className="text-muted-foreground">Area: {roomArea(graph, room).toFixed(1)} m²</p>
              </div>
            );
          })()}
          {sel.type === 'opening' && (() => {
            const op = graph.openings.find((o) => o.id === sel.id);
            if (!op) return null;
            return (
              <div className="space-y-2">
                <p className="font-semibold text-foreground">Opening</p>
                <label className="block text-muted-foreground">
                  Kind
                  <select className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1 text-foreground" value={op.kind} onChange={(e) => onChange?.(setOpeningKind(graph, op.id, e.target.value as OpeningKind))}>
                    <option value="door">door</option>
                    <option value="passage">passage</option>
                    <option value="window">window</option>
                  </select>
                </label>
                <label className="block text-muted-foreground">
                  Width: {op.width.toFixed(2)} m
                  <input
                    type="range"
                    min={0.3}
                    max={3}
                    step={0.05}
                    value={op.width}
                    className="mt-1 w-full accent-blue-600"
                    onChange={(e) => onChange?.(setOpeningWidth(graph, op.id, Number(e.target.value)))}
                  />
                </label>
                <button className="w-full rounded-md border border-destructive/30 bg-destructive/10 px-2 py-1 text-destructive transition-colors hover:bg-destructive/20" onClick={() => { onChange?.(deleteOpening(graph, op.id)); setSel(null); }}>
                  Delete opening
                </button>
              </div>
            );
          })()}
          {sel.type === 'wall' && selWall && (
            <div className="space-y-2">
              <p className="font-semibold text-foreground">Wall</p>
              <p className="text-muted-foreground">Length: {wallLength(graph, selWall).toFixed(2)} m</p>
              <button
                className="w-full rounded-md border border-border bg-background px-2 py-1 text-foreground transition-colors hover:bg-accent"
                onClick={() => onChange?.(splitWall(graph, selWall.id, 0.5))}
              >
                Split wall at midpoint
              </button>
              <p className="text-muted-foreground">Drag the wall to move it, or drag a blue node to move the corner.</p>
            </div>
          )}
        </div>
      )}

      {tool === 'boundary' && (
        <div className="absolute left-3 top-3 rounded-lg border border-border/70 bg-card/95 px-3 py-2 text-[11px] text-muted-foreground shadow-md backdrop-blur">
          <span className="font-medium text-foreground">Boundary tool</span> · click empty space to add a point · drag dots · green = insert · double-click to delete
        </div>
      )}
    </div>
  );
}
