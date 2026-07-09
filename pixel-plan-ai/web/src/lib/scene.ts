import type { Door, Plan, RoomBounds } from "./types";

// Retained scene graph for the 2D CAD viewport. `buildScene(plan)` is computed
// ONCE per plan (memoised by the viewport) and is the single geometry source of
// truth shared by the renderer (walls / room accents / labels / doors) and the
// interaction layer (hit testing). Every coordinate here is in CELL space (the
// grid the plan is defined on); the renderer multiplies by the effective px/cell
// at draw time, so the same scene serves every zoom level without recomputation.

const OUTSIDE = -2;
const UNASSIGNED = -1;
const SQ_METERS_TO_SQ_FEET = 10.7639;

export interface Segment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface SceneRoom {
  index: number;
  id: string;
  type: string;
  color: string;
  cellCount: number;
  areaFt2: number;
  bounds: RoomBounds;
  // Largest inscribed axis-aligned rectangle of the room (histogram sweep),
  // falling back to the bounding box when the room has no interior rectangle.
  largestRect: { x: number; y: number; w: number; h: number };
  // Unit edges of the room's outline (one per cell face that borders a
  // non-room cell), used for the low-alpha inner wall accent.
  boundary: Segment[];
  // Centre of `largestRect` — where the two-line label lockup anchors.
  labelAnchor: { x: number; y: number };
}

export interface SceneWall {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  kind: "exterior" | "interior";
  // Region ids on each side of the wall (a room index >= 0, UNASSIGNED, or
  // OUTSIDE). Kept for interaction context; rendering only reads `kind`.
  rooms: [number, number];
}

export interface SceneDoor {
  id: string;
  door: Door;
  // Hinge point (arc centre) in cell coords.
  hinge: { x: number; y: number };
  // Opening width in cells.
  span: number;
  horizontal: boolean;
  closedAngle: number;
  openAngle: number;
  isEntrance: boolean;
  // The opening span itself, as a cell-coord segment (used for hit testing).
  opening: Segment;
}

export type Hit =
  | { kind: "door"; door: SceneDoor }
  | { kind: "wall"; wall: SceneWall }
  | { kind: "room"; roomIndex: number };

export interface Scene {
  width: number;
  height: number;
  metersPerCell: number;
  rooms: SceneRoom[];
  walls: SceneWall[];
  doors: SceneDoor[];
  // Hit test a point in CELL float coords. `tolerancePx` is the pick radius in
  // screen pixels; `cellPx` is the effective px/cell so it can be converted to
  // cell space. Priority: door > wall > room.
  hitTest(pt: { x: number; y: number }, tolerancePx: number, cellPx: number): Hit | null;
}

// Largest inscribed axis-aligned rectangle of a room's cells, in cell coords.
// Runs a histogram maximal-rectangle sweep restricted to the room's bounds.
function largestInnerRect(
  cells: number[],
  planWidth: number,
  roomIndex: number,
  bounds: RoomBounds,
): { x: number; y: number; w: number; h: number } | null {
  const bw = bounds.width;
  const bh = bounds.height;
  if (bw <= 0 || bh <= 0) return null;
  const heights = new Array<number>(bw).fill(0);
  let best = { area: 0, x: 0, y: 0, w: 0, h: 0 };
  for (let r = 0; r < bh; r += 1) {
    for (let c = 0; c < bw; c += 1) {
      const occupied = cells[(bounds.y + r) * planWidth + (bounds.x + c)] === roomIndex;
      heights[c] = occupied ? heights[c] + 1 : 0;
    }
    const stack: number[] = [];
    for (let c = 0; c <= bw; c += 1) {
      const cur = c === bw ? 0 : heights[c];
      while (stack.length && heights[stack[stack.length - 1]] > cur) {
        const height = heights[stack.pop() as number];
        const left = stack.length ? stack[stack.length - 1] + 1 : 0;
        const width = c - left;
        const area = height * width;
        if (area > best.area) {
          best = { area, x: bounds.x + left, y: bounds.y + (r - height + 1), w: width, h: height };
        }
      }
      stack.push(c);
    }
  }
  return best.area > 0 ? best : null;
}

// Perpendicular distance from a point to a finite segment (all in cell coords).
function distToSegment(px: number, py: number, s: Segment): number {
  const dx = s.x2 - s.x1;
  const dy = s.y2 - s.y1;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(px - s.x1, py - s.y1);
  let t = ((px - s.x1) * dx + (py - s.y1) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (s.x1 + t * dx), py - (s.y1 + t * dy));
}

export function segmentLengthCells(s: Segment): number {
  return Math.hypot(s.x2 - s.x1, s.y2 - s.y1);
}

// One raw unit edge produced by the region-transition sweep, before merging.
interface RawEdge {
  orient: "v" | "h";
  fixed: number; // x for vertical edges, y for horizontal edges
  start: number; // y for vertical edges, x for horizontal edges (unit length)
  kind: "exterior" | "interior";
  a: number;
  b: number;
}

function buildWalls(plan: Plan): SceneWall[] {
  const W = plan.width;
  const H = plan.height;
  const hasFootprint = Array.isArray(plan.footprint) && plan.footprint.length === W * H;

  // Region id used for wall classification: a specific room, an unassigned but
  // in-footprint cell, or OUTSIDE (past the building envelope / off-grid).
  const regionAt = (x: number, y: number): number => {
    if (x < 0 || y < 0 || x >= W || y >= H) return OUTSIDE;
    const off = y * W + x;
    const v = plan.cells[off];
    if (v !== undefined && v >= 0) return v;
    const inside = hasFootprint ? !!plan.footprint[off] : false;
    return inside ? UNASSIGNED : OUTSIDE;
  };

  const raw: RawEdge[] = [];
  // Vertical edges: boundary between (x-1,y) and (x,y).
  for (let y = 0; y < H; y += 1) {
    for (let x = 0; x <= W; x += 1) {
      const a = regionAt(x - 1, y);
      const b = regionAt(x, y);
      if (a === b) continue;
      raw.push({ orient: "v", fixed: x, start: y, kind: a === OUTSIDE || b === OUTSIDE ? "exterior" : "interior", a, b });
    }
  }
  // Horizontal edges: boundary between (x,y-1) and (x,y).
  for (let y = 0; y <= H; y += 1) {
    for (let x = 0; x < W; x += 1) {
      const a = regionAt(x, y - 1);
      const b = regionAt(x, y);
      if (a === b) continue;
      raw.push({ orient: "h", fixed: y, start: x, kind: a === OUTSIDE || b === OUTSIDE ? "exterior" : "interior", a, b });
    }
  }

  // Merge contiguous, collinear unit edges that share kind AND region pair into
  // maximal segments. Merging only same-kind runs keeps the render identical
  // (butt-capped collinear segments render as one line), and keeping the region
  // pair intact makes each wall's `rooms` meaningful for interaction.
  const groups = new Map<string, RawEdge[]>();
  for (const e of raw) {
    const key = `${e.orient}|${e.fixed}|${e.kind}|${e.a}|${e.b}`;
    const list = groups.get(key);
    if (list) list.push(e);
    else groups.set(key, [e]);
  }

  const walls: SceneWall[] = [];
  for (const list of groups.values()) {
    list.sort((p, q) => p.start - q.start);
    let runStart = list[0].start;
    let runEnd = list[0].start + 1;
    const sample = list[0];
    const flush = (): void => {
      if (sample.orient === "v") {
        walls.push({ x1: sample.fixed, y1: runStart, x2: sample.fixed, y2: runEnd, kind: sample.kind, rooms: [sample.a, sample.b] });
      } else {
        walls.push({ x1: runStart, y1: sample.fixed, x2: runEnd, y2: sample.fixed, kind: sample.kind, rooms: [sample.a, sample.b] });
      }
    };
    for (let i = 1; i < list.length; i += 1) {
      const e = list[i];
      if (e.start === runEnd) {
        runEnd = e.start + 1;
      } else {
        flush();
        runStart = e.start;
        runEnd = e.start + 1;
      }
    }
    flush();
  }
  return walls;
}

function buildRooms(plan: Plan): SceneRoom[] {
  const W = plan.width;
  const H = plan.height;
  const mpc = plan.meters_per_cell;

  const cellRoom = (x: number, y: number): number => {
    if (x < 0 || y < 0 || x >= W || y >= H) return UNASSIGNED;
    const v = plan.cells[y * W + x];
    return v === undefined || v < 0 ? UNASSIGNED : v;
  };

  return plan.rooms.map((room, index) => {
    const bounds = room.bounds;
    const boundary: Segment[] = [];
    for (let y = bounds.y; y < bounds.y + bounds.height; y += 1) {
      for (let x = bounds.x; x < bounds.x + bounds.width; x += 1) {
        if (cellRoom(x, y) !== index) continue;
        if (cellRoom(x - 1, y) !== index) boundary.push({ x1: x, y1: y, x2: x, y2: y + 1 });
        if (cellRoom(x + 1, y) !== index) boundary.push({ x1: x + 1, y1: y, x2: x + 1, y2: y + 1 });
        if (cellRoom(x, y - 1) !== index) boundary.push({ x1: x, y1: y, x2: x + 1, y2: y });
        if (cellRoom(x, y + 1) !== index) boundary.push({ x1: x, y1: y + 1, x2: x + 1, y2: y + 1 });
      }
    }
    const rect = largestInnerRect(plan.cells, W, index, bounds) ?? {
      x: bounds.x,
      y: bounds.y,
      w: bounds.width,
      h: bounds.height,
    };
    return {
      index,
      id: room.id,
      type: room.type,
      color: room.color,
      cellCount: room.pixel_count,
      areaFt2: Math.round(room.pixel_count * mpc * mpc * SQ_METERS_TO_SQ_FEET),
      bounds,
      largestRect: rect,
      boundary,
      labelAnchor: { x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 },
    };
  });
}

function buildDoors(plan: Plan): SceneDoor[] {
  return plan.doors.map((door, index) => {
    const horizontal = door.orientation === "horizontal";
    const span = door.width_cells;
    let closedAngle: number;
    let openAngle: number;
    if (horizontal) {
      closedAngle = 0;
      openAngle = door.swing_side === "south" ? Math.PI / 2 : -Math.PI / 2;
    } else {
      closedAngle = Math.PI / 2;
      openAngle = door.swing_side === "west" ? Math.PI : 0;
    }
    const opening: Segment = horizontal
      ? { x1: door.x, y1: door.y, x2: door.x + span, y2: door.y }
      : { x1: door.x, y1: door.y, x2: door.x, y2: door.y + span };
    return {
      id: door.id ?? `door_${index}`,
      door,
      hinge: { x: door.x, y: door.y },
      span,
      horizontal,
      closedAngle,
      openAngle,
      isEntrance: door.to_room === null,
      opening,
    };
  });
}

export function buildScene(plan: Plan): Scene {
  const W = plan.width;
  const H = plan.height;
  const rooms = buildRooms(plan);
  const walls = buildWalls(plan);
  const doors = buildDoors(plan);

  // Plans are <= ~200 x 160 cells, so walls/doors number in the low hundreds —
  // a flat scan per query is plenty fast, no spatial index needed.
  const hitTest: Scene["hitTest"] = (pt, tolerancePx, cellPx) => {
    const tol = cellPx > 0 ? tolerancePx / cellPx : 0;

    let bestDoor: SceneDoor | null = null;
    let bestDoorDist = tol;
    for (const d of doors) {
      const dist = distToSegment(pt.x, pt.y, d.opening);
      if (dist <= bestDoorDist) {
        bestDoorDist = dist;
        bestDoor = d;
      }
    }
    if (bestDoor) return { kind: "door", door: bestDoor };

    let bestWall: SceneWall | null = null;
    let bestWallDist = tol;
    for (const w of walls) {
      const dist = distToSegment(pt.x, pt.y, w);
      if (dist <= bestWallDist) {
        bestWallDist = dist;
        bestWall = w;
      }
    }
    if (bestWall) return { kind: "wall", wall: bestWall };

    const cx = Math.floor(pt.x);
    const cy = Math.floor(pt.y);
    if (cx >= 0 && cy >= 0 && cx < W && cy < H) {
      const v = plan.cells[cy * W + cx];
      if (v !== undefined && v >= 0) return { kind: "room", roomIndex: v };
    }
    return null;
  };

  return { width: W, height: H, metersPerCell: plan.meters_per_cell, rooms, walls, doors, hitTest };
}
