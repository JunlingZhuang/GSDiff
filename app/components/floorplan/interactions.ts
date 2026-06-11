// Pure edit operations on a Plan. Each returns a new Plan (immutable), so the
// 2D editor can keep an undo stack and React re-renders cleanly.

import type { Plan, Pt, Wall, RoomType, OpeningKind } from '@/lib/plan';

const R = 1000; // coincident-point match precision (1 mm)

function keyPt(p: Pt): string {
  return `${Math.round(p[0] * R)}:${Math.round(p[1] * R)}`;
}
function eqPt(a: Pt, b: Pt): boolean {
  return keyPt(a) === keyPt(b);
}

export function snap1(v: number, step: number): number {
  return Math.round(v / step) * step;
}

export function snapPt(p: Pt, grid: Plan['grid']): Pt {
  return [
    grid.originX + snap1(p[0] - grid.originX, grid.spacingX || 1),
    grid.originY + snap1(p[1] - grid.originY, grid.spacingY || 1),
  ];
}

// Move a wall perpendicular to itself by signed distance `d` (metres). Every
// other wall endpoint and room-polygon vertex coincident with the moved
// endpoints follows, so junctions stay connected and adjacent rooms reshape.
export function dragWall(plan: Plan, wallId: string, d: number, gridSnap: boolean): Plan {
  const wall = plan.walls.find((w) => w.id === wallId);
  if (!wall) return plan;
  const [ax, ay] = wall.a;
  const [bx, by] = wall.b;
  const dx = bx - ax;
  const dy = by - ay;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  const oldA: Pt = [ax, ay];
  const oldB: Pt = [bx, by];
  let na: Pt = [ax + nx * d, ay + ny * d];
  let nb: Pt = [bx + nx * d, by + ny * d];
  if (gridSnap) {
    na = snapPt(na, plan.grid);
    nb = snapPt(nb, plan.grid);
  }
  const mapPt = (p: Pt): Pt => (eqPt(p, oldA) ? na : eqPt(p, oldB) ? nb : p);
  return {
    ...plan,
    walls: plan.walls.map((w) => ({ ...w, a: mapPt(w.a), b: mapPt(w.b) })),
    rooms: plan.rooms.map((r) => ({ ...r, poly: r.poly.map(mapPt) })),
  };
}

export function setRoomType(plan: Plan, roomId: string, type: RoomType): Plan {
  return { ...plan, rooms: plan.rooms.map((r) => (r.id === roomId ? { ...r, type } : r)) };
}

export function moveOpening(plan: Plan, openingId: string, t: number): Plan {
  const tt = Math.max(0, Math.min(1, t));
  return { ...plan, openings: plan.openings.map((o) => (o.id === openingId ? { ...o, t: tt } : o)) };
}

export function deleteOpening(plan: Plan, openingId: string): Plan {
  return { ...plan, openings: plan.openings.filter((o) => o.id !== openingId) };
}

export function setOpeningKind(plan: Plan, openingId: string, kind: OpeningKind): Plan {
  const width = OPENING_WIDTH[kind];
  return {
    ...plan,
    openings: plan.openings.map((o) => (o.id === openingId ? { ...o, kind, width } : o)),
  };
}

export const OPENING_WIDTH: Record<OpeningKind, number> = {
  door: 0.9,
  passage: 1.2,
  window: 1.0,
};

export function addOpening(plan: Plan, wallId: string, t: number, kind: OpeningKind = 'door'): Plan {
  const id = `o${Date.now().toString(36)}${Math.floor(Math.random() * 1e3)}`;
  return {
    ...plan,
    openings: [
      ...plan.openings,
      { id, wallId, t: Math.max(0, Math.min(1, t)), width: OPENING_WIDTH[kind], kind },
    ],
  };
}

// Nearest wall (and its parameter t) to a world point — for click-to-add-door
// and for picking a wall to drag.
export function nearestWall(
  plan: Plan,
  p: Pt,
): { wall: Wall; t: number; dist: number } | null {
  let best: { wall: Wall; t: number; dist: number } | null = null;
  for (const w of plan.walls) {
    const [ax, ay] = w.a;
    const [bx, by] = w.b;
    const dx = bx - ax;
    const dy = by - ay;
    const L2 = dx * dx + dy * dy || 1;
    let t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2;
    t = Math.max(0, Math.min(1, t));
    const cx = ax + dx * t;
    const cy = ay + dy * t;
    const dist = Math.hypot(p[0] - cx, p[1] - cy);
    if (!best || dist < best.dist) best = { wall: w, t, dist };
  }
  return best;
}

// Signed perpendicular distance a world point lies off a wall's centreline,
// used to convert a drag vector into a perpendicular offset.
export function perpOffset(wall: Wall, from: Pt, to: Pt): number {
  const dx = wall.b[0] - wall.a[0];
  const dy = wall.b[1] - wall.a[1];
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  return (to[0] - from[0]) * nx + (to[1] - from[1]) * ny;
}
