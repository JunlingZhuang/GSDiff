// Floor-plan CAD kernel (compute layer).
//
// Single source of truth is a *wall graph*: junction nodes, wall segments
// between nodes, rooms as ordered node-id loops, and openings parameterised
// along a wall. Geometry (room polygons, mitred wall solids) is *derived* from
// the graph, so moving a node updates every wall and room that references it —
// walls never fragment and rooms never tear. Clean corners come from a boolean
// union of the per-segment rectangles (minus opening cut-outs).

import polygonClipping from 'polygon-clipping';
import type { Plan, Pt, RoomType, OpeningKind, AxisGrid } from '@/lib/plan';

export interface WallNode { id: string; x: number; y: number }
export interface WallSeg { id: string; n0: string; n1: string; thickness: number }
export interface RoomFace { id: string; type: RoomType; loop: string[] }
export interface KOpening { id: string; wallId: string; t: number; width: number; kind: OpeningKind }

export interface WallGraph {
  nodes: Record<string, WallNode>;
  walls: WallSeg[];
  rooms: RoomFace[];
  openings: KOpening[];
  grid: AxisGrid;
}

type Ring = [number, number][];
type Poly = Ring[];
type MultiPoly = Poly[];

const Q = 1000; // node-merge precision (1 mm)
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const dist = (a: Pt, b: Pt) => Math.hypot(b[0] - a[0], b[1] - a[1]);
const lerp = (a: Pt, b: Pt, t: number): Pt => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

function widthFor(kind: OpeningKind): number {
  return kind === 'passage' ? 1.2 : kind === 'window' ? 1.0 : 0.9;
}

// ── build the graph from a generated Plan (merge coincident vertices) ───────
export function planToGraph(plan: Plan): WallGraph {
  const nodes: Record<string, WallNode> = {};
  const keyToId = new Map<string, string>();
  let nid = 0;
  const getNode = (x: number, y: number): string => {
    const kx = Math.round(x * Q);
    const ky = Math.round(y * Q);
    const key = `${kx}:${ky}`;
    let id = keyToId.get(key);
    if (!id) {
      id = `n${nid++}`;
      keyToId.set(key, id);
      nodes[id] = { id, x: kx / Q, y: ky / Q };
    }
    return id;
  };

  const rooms: RoomFace[] = plan.rooms.map((r) => {
    const pts = r.poly;
    const ring =
      pts.length > 1 && Math.round(pts[0][0] * Q) === Math.round(pts[pts.length - 1][0] * Q) &&
      Math.round(pts[0][1] * Q) === Math.round(pts[pts.length - 1][1] * Q)
        ? pts.slice(0, -1)
        : pts;
    const loop = ring.map(([x, y]) => getNode(x, y));
    const clean = loop.filter((id, i) => id !== loop[(i - 1 + loop.length) % loop.length]);
    return { id: r.id, type: (r.type as RoomType) ?? 'Bedroom', loop: clean };
  });

  const walls: WallSeg[] = plan.walls
    .map((w) => ({ id: w.id, n0: getNode(w.a[0], w.a[1]), n1: getNode(w.b[0], w.b[1]), thickness: w.thickness }))
    .filter((w) => w.n0 !== w.n1);

  const openings: KOpening[] = plan.openings.map((o) => ({ ...o }));
  return { nodes, walls, rooms, openings, grid: plan.grid };
}

// ── derived geometry ────────────────────────────────────────────────────────
export function nodeXY(g: WallGraph, id: string): Pt {
  const n = g.nodes[id];
  return [n.x, n.y];
}
export function roomPoly(g: WallGraph, room: RoomFace): Pt[] {
  return room.loop.map((id) => nodeXY(g, id));
}
export function wallEnds(g: WallGraph, w: WallSeg): { a: Pt; b: Pt } {
  return { a: nodeXY(g, w.n0), b: nodeXY(g, w.n1) };
}

function segRect(a: Pt, b: Pt, thickness: number): Ring | null {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return null;
  const nx = (-dy / len) * (thickness / 2);
  const ny = (dx / len) * (thickness / 2);
  return [
    [a[0] + nx, a[1] + ny],
    [b[0] + nx, b[1] + ny],
    [b[0] - nx, b[1] - ny],
    [a[0] - nx, a[1] - ny],
    [a[0] + nx, a[1] + ny],
  ];
}

// Mitred wall solid: union of segment rectangles minus opening cut-outs.
// Returns a MultiPolygon (array of polygons; each polygon = [outerRing, ...holes]).
export function wallSolid(g: WallGraph): MultiPoly {
  const rects: Poly[] = [];
  for (const w of g.walls) {
    const { a, b } = wallEnds(g, w);
    const r = segRect(a, b, w.thickness);
    if (r) rects.push([r]);
  }
  if (!rects.length) return [];
  let solid: MultiPoly;
  try {
    solid = polygonClipping.union(rects[0] as Poly, ...(rects.slice(1) as Poly[])) as MultiPoly;
  } catch {
    return rects as unknown as MultiPoly;
  }

  const holes: Poly[] = [];
  for (const o of g.openings) {
    const w = g.walls.find((x) => x.id === o.wallId);
    if (!w) continue;
    const { a, b } = wallEnds(g, w);
    const len = dist(a, b) || 1;
    const half = o.width / 2 / len;
    const p0 = lerp(a, b, clamp01(o.t - half));
    const p1 = lerp(a, b, clamp01(o.t + half));
    const r = segRect(p0, p1, w.thickness * 1.4); // overshoot to fully clear both faces
    if (r) holes.push([r]);
  }
  if (holes.length) {
    try {
      solid = polygonClipping.difference(solid as MultiPoly, holes[0] as Poly, ...(holes.slice(1) as Poly[])) as MultiPoly;
    } catch {
      /* keep solid */
    }
  }
  return solid;
}

export function graphBounds(g: WallGraph): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const id in g.nodes) {
    const n = g.nodes[id];
    if (n.x < minX) minX = n.x;
    if (n.y < minY) minY = n.y;
    if (n.x > maxX) maxX = n.x;
    if (n.y > maxY) maxY = n.y;
  }
  if (!isFinite(minX)) return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  return { minX, minY, maxX, maxY };
}

// ── edit operations (immutable) ─────────────────────────────────────────────
export function snapXY(x: number, y: number, grid: AxisGrid): [number, number] {
  const sx = grid.spacingX || 1;
  const sy = grid.spacingY || 1;
  return [
    grid.originX + Math.round((x - grid.originX) / sx) * sx,
    grid.originY + Math.round((y - grid.originY) / sy) * sy,
  ];
}

export function moveNode(g: WallGraph, id: string, x: number, y: number): WallGraph {
  if (!g.nodes[id]) return g;
  return { ...g, nodes: { ...g.nodes, [id]: { ...g.nodes[id], x, y } } };
}

function wallDir(g: WallGraph, w: WallSeg): [number, number] {
  const a = nodeXY(g, w.n0);
  const b = nodeXY(g, w.n1);
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  return [dx / len, dy / len];
}
function parallel(d1: [number, number], d2: [number, number], eps = 2e-2): boolean {
  return Math.abs(d1[0] * d2[1] - d1[1] * d2[0]) < eps;
}

// All nodes belonging to the maximal *collinear chain* through a wall: walls
// that are parallel to it and connected by a shared node lie on the same
// infinite line, so they must move together when the wall is dragged.
function collinearChainNodes(g: WallGraph, w: WallSeg): Set<string> {
  const dir = wallDir(g, w);
  const chain = new Set<string>([w.id]);
  const nodes = new Set<string>([w.n0, w.n1]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const o of g.walls) {
      if (chain.has(o.id)) continue;
      if (!(nodes.has(o.n0) || nodes.has(o.n1))) continue;
      if (!parallel(wallDir(g, o), dir)) continue;
      chain.add(o.id);
      nodes.add(o.n0);
      nodes.add(o.n1);
      changed = true;
    }
  }
  return nodes;
}

// Drag a wall perpendicular to itself by `d` metres. The whole collinear chain
// translates rigidly, so the wall stays straight and every perpendicular
// neighbour slides along its own axis (stays orthogonal) instead of rotating.
export function dragWallSeg(g: WallGraph, wallId: string, d: number, snap: boolean): WallGraph {
  const w = g.walls.find((x) => x.id === wallId);
  if (!w) return g;
  const { a, b } = wallEnds(g, w);
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  const chainNodes = collinearChainNodes(g, w);
  const nodes = { ...g.nodes };
  for (const id of chainNodes) {
    const p = g.nodes[id];
    let x = p.x + nx * d;
    let y = p.y + ny * d;
    if (snap) {
      [x, y] = snapXY(x, y, g.grid);
    }
    nodes[id] = { ...p, x, y };
  }
  return { ...g, nodes };
}

export function setRoomType(g: WallGraph, roomId: string, type: RoomType): WallGraph {
  return { ...g, rooms: g.rooms.map((r) => (r.id === roomId ? { ...r, type } : r)) };
}

export function addOpening(g: WallGraph, wallId: string, t: number, kind: OpeningKind): WallGraph {
  const id = `o${Date.now().toString(36)}${Math.floor(Math.random() * 1e3)}`;
  return { ...g, openings: [...g.openings, { id, wallId, t: clamp01(t), width: widthFor(kind), kind }] };
}
export function moveOpening(g: WallGraph, id: string, t: number): WallGraph {
  return { ...g, openings: g.openings.map((o) => (o.id === id ? { ...o, t: clamp01(t) } : o)) };
}
export function deleteOpening(g: WallGraph, id: string): WallGraph {
  return { ...g, openings: g.openings.filter((o) => o.id !== id) };
}
export function setOpeningKind(g: WallGraph, id: string, kind: OpeningKind): WallGraph {
  return { ...g, openings: g.openings.map((o) => (o.id === id ? { ...o, kind, width: widthFor(kind) } : o)) };
}

// project a world point onto a wall, returning its parameter t in [0,1]
export function paramOnWall(g: WallGraph, wallId: string, p: Pt): number {
  const w = g.walls.find((x) => x.id === wallId);
  if (!w) return 0.5;
  const { a, b } = wallEnds(g, w);
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const L2 = dx * dx + dy * dy || 1;
  return clamp01(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2);
}

// signed perpendicular component of (to-from) relative to a wall
export function perpDrag(g: WallGraph, wallId: string, from: Pt, to: Pt): number {
  const w = g.walls.find((x) => x.id === wallId);
  if (!w) return 0;
  const { a, b } = wallEnds(g, w);
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  return (to[0] - from[0]) * (-dy / len) + (to[1] - from[1]) * (dx / len);
}

// SVG path "d" for a MultiPolygon (outer rings + holes), even-odd fill.
export function multiPolyPath(mp: MultiPoly): string {
  const f = (n: number) => +n.toFixed(4);
  let d = '';
  for (const poly of mp) {
    for (const ring of poly) {
      if (!ring.length) continue;
      d += `M ${f(ring[0][0])} ${f(ring[0][1])} `;
      for (let i = 1; i < ring.length; i++) d += `L ${f(ring[i][0])} ${f(ring[i][1])} `;
      d += 'Z ';
    }
  }
  return d.trim();
}
