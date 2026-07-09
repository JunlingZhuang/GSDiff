import type { Plan } from "./types";

// Derives a schematic 3D model (Finch-style massing) purely from the plan
// grid: floor slabs per room, walls along cell boundaries, door openings
// left as gaps. All dimensions in meters; the model is centered on origin.

export interface Box3D {
  center: [number, number, number];
  size: [number, number, number];
  color: string;
}

// A door leaf is a box hinged at one jamb and swung open into a room, so it
// carries a Y rotation on top of the box fields.
export interface DoorLeaf3D extends Box3D {
  rotationY: number;
}

export interface Door3D {
  leaf: DoorLeaf3D;
  // The two slim jamb posts framing the opening ends.
  posts: Box3D[];
}

export interface PlanModel {
  floors: Box3D[];
  walls: Box3D[];
  doors: Door3D[];
  base: Box3D | null;
  extentMeters: number;
}

// Dollhouse-style cutaway: waist-height light walls keep the room colors
// visible from an orbiting camera (full-height walls read as a black box).
const WALL_HEIGHT = 1.5;
const WALL_THICKNESS = 0.14;
const FLOOR_THICKNESS = 0.1;
const BASE_THICKNESS = 0.18;
const WALL_COLOR = "#f1efe9";

// Door components (Finch-scale, no textures). The leaf is a thin panel hinged at
// a jamb and swung 30° into the swing room; jamb posts frame the opening ends.
const DOOR_LEAF_HEIGHT = WALL_HEIGHT * 0.8;
const DOOR_LEAF_THICKNESS = WALL_THICKNESS / 3;
const DOOR_OPEN_RAD = (30 * Math.PI) / 180;
const DOOR_LEAF_COLOR = "#F7F5F1"; // white-ish interior leaf
const DOOR_ENTRANCE_COLOR = "#8FBAF0"; // accent-tinted entry leaf (to_room null)
const DOOR_FRAME_COLOR = "#d8d3c9"; // jamb posts, slightly darker than walls

export function buildPlanModel(plan: Plan): PlanModel {
  const m = plan.meters_per_cell || 0.3;
  const offsetX = (plan.width * m) / 2;
  const offsetZ = (plan.height * m) / 2;
  const cell = (x: number, y: number): number => {
    if (x < 0 || y < 0 || x >= plan.width || y >= plan.height) return -2;
    const value = plan.cells[y * plan.width + x];
    return value === undefined ? -2 : value;
  };

  const floors: Box3D[] = [];
  for (const [roomIndex, room] of plan.rooms.entries()) {
    for (const rect of decomposeRoom(plan, roomIndex)) {
      floors.push({
        center: [
          (rect.x + rect.w / 2) * m - offsetX,
          FLOOR_THICKNESS / 2,
          (rect.y + rect.h / 2) * m - offsetZ,
        ],
        size: [rect.w * m, FLOOR_THICKNESS, rect.h * m],
        color: room.color,
      });
    }
  }

  // Boundary units covered by a door become wall gaps.
  const doorUnits = new Set<string>();
  for (const door of plan.doors) {
    for (let i = 0; i < door.width_cells; i += 1) {
      if (door.orientation === "horizontal") doorUnits.add(`h:${door.x + i},${door.y}`);
      else doorUnits.add(`v:${door.x},${door.y + i}`);
    }
  }

  const walls: Box3D[] = [];
  // Vertical boundaries (constant x, spanning rows): between (x-1,y) and (x,y).
  for (let x = 0; x <= plan.width; x += 1) {
    let runStart = -1;
    for (let y = 0; y <= plan.height; y += 1) {
      const left = cell(x - 1, y);
      const right = cell(x, y);
      const isWall = y < plan.height && left !== right && (left >= 0 || right >= 0) && !doorUnits.has(`v:${x},${y}`);
      if (isWall && runStart < 0) runStart = y;
      if (!isWall && runStart >= 0) {
        walls.push({
          center: [x * m - offsetX, WALL_HEIGHT / 2, ((runStart + y) / 2) * m - offsetZ],
          size: [WALL_THICKNESS, WALL_HEIGHT, (y - runStart) * m + WALL_THICKNESS],
          color: WALL_COLOR,
        });
        runStart = -1;
      }
    }
  }
  // Horizontal boundaries (constant y, spanning columns): between (x,y-1) and (x,y).
  for (let y = 0; y <= plan.height; y += 1) {
    let runStart = -1;
    for (let x = 0; x <= plan.width; x += 1) {
      const above = cell(x, y - 1);
      const below = cell(x, y);
      const isWall = x < plan.width && above !== below && (above >= 0 || below >= 0) && !doorUnits.has(`h:${x},${y}`);
      if (isWall && runStart < 0) runStart = x;
      if (!isWall && runStart >= 0) {
        walls.push({
          center: [((runStart + x) / 2) * m - offsetX, WALL_HEIGHT / 2, y * m - offsetZ],
          size: [(x - runStart) * m + WALL_THICKNESS, WALL_HEIGHT, WALL_THICKNESS],
          color: WALL_COLOR,
        });
        runStart = -1;
      }
    }
  }

  // Door components sit in the wall gaps carved above: a swung leaf plus two jamb
  // posts. All in world meters, centered on origin like the rest of the model.
  const doors: Door3D[] = [];
  for (const door of plan.doors) {
    const horizontal = door.orientation === "horizontal";
    const cells = Math.max(1, door.width_cells);
    const len = cells * m;
    // Jamb A is the hinge; jamb B is the far opening end.
    const aX = door.x;
    const aY = door.y;
    const bX = horizontal ? door.x + cells : door.x;
    const bY = horizontal ? door.y : door.y + cells;
    const worldX = (cx: number): number => cx * m - offsetX;
    const worldZ = (cy: number): number => cy * m - offsetZ;
    const hingeX = worldX(aX);
    const hingeZ = worldZ(aY);

    // Closed leaf runs hinge → far jamb; the swing normal points into the room.
    // Rotating the (orthonormal) closed direction 30° toward the normal opens it.
    const dir: [number, number] = horizontal ? [1, 0] : [0, 1];
    const normal: [number, number] = horizontal
      ? door.swing_side === "north"
        ? [0, -1]
        : [0, 1]
      : door.swing_side === "west"
        ? [-1, 0]
        : [1, 0];
    const c = Math.cos(DOOR_OPEN_RAD);
    const s = Math.sin(DOOR_OPEN_RAD);
    const openX = dir[0] * c + normal[0] * s;
    const openZ = dir[1] * c + normal[1] * s;

    const entrance = door.to_room === null;
    doors.push({
      leaf: {
        // Hinge + half-length along the open direction places the leaf centre; a
        // Y rotation aligns the box's long (local +X) axis with that direction.
        center: [hingeX + openX * (len / 2), DOOR_LEAF_HEIGHT / 2, hingeZ + openZ * (len / 2)],
        size: [len, DOOR_LEAF_HEIGHT, DOOR_LEAF_THICKNESS],
        rotationY: Math.atan2(-openZ, openX),
        color: entrance ? DOOR_ENTRANCE_COLOR : DOOR_LEAF_COLOR,
      },
      posts: [
        {
          center: [hingeX, WALL_HEIGHT / 2, hingeZ],
          size: [WALL_THICKNESS, WALL_HEIGHT, WALL_THICKNESS],
          color: DOOR_FRAME_COLOR,
        },
        {
          center: [worldX(bX), WALL_HEIGHT / 2, worldZ(bY)],
          size: [WALL_THICKNESS, WALL_HEIGHT, WALL_THICKNESS],
          color: DOOR_FRAME_COLOR,
        },
      ],
    });
  }

  const footprintRect = footprintBounds(plan);
  const base = footprintRect
    ? {
        center: [
          (footprintRect.x + footprintRect.w / 2) * m - offsetX,
          -BASE_THICKNESS / 2,
          (footprintRect.y + footprintRect.h / 2) * m - offsetZ,
        ] as [number, number, number],
        size: [
          (footprintRect.w + 2) * m,
          BASE_THICKNESS,
          (footprintRect.h + 2) * m,
        ] as [number, number, number],
        color: "#dedbd3",
      }
    : null;

  return {
    floors,
    walls,
    doors,
    base,
    extentMeters: Math.max(plan.width, plan.height) * m,
  };
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

function decomposeRoom(plan: Plan, roomIndex: number): Rect[] {
  const owns = (x: number, y: number): boolean =>
    x >= 0 && y >= 0 && x < plan.width && y < plan.height && plan.cells[y * plan.width + x] === roomIndex;
  const consumed = new Set<number>();
  const rects: Rect[] = [];
  const bounds = plan.rooms[roomIndex].bounds;
  for (let y = bounds.y; y < bounds.y + bounds.height; y += 1) {
    for (let x = bounds.x; x < bounds.x + bounds.width; x += 1) {
      const key = y * plan.width + x;
      if (!owns(x, y) || consumed.has(key)) continue;
      let w = 1;
      while (owns(x + w, y) && !consumed.has(key + w)) w += 1;
      let h = 1;
      expand: while (y + h < bounds.y + bounds.height) {
        for (let i = 0; i < w; i += 1) {
          const probe = (y + h) * plan.width + x + i;
          if (!owns(x + i, y + h) || consumed.has(probe)) break expand;
        }
        h += 1;
      }
      for (let dy = 0; dy < h; dy += 1) {
        for (let dx = 0; dx < w; dx += 1) consumed.add((y + dy) * plan.width + x + dx);
      }
      rects.push({ x, y, w, h });
    }
  }
  return rects;
}

function footprintBounds(plan: Plan): Rect | null {
  let minX = plan.width;
  let minY = plan.height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < plan.height; y += 1) {
    for (let x = 0; x < plan.width; x += 1) {
      if (!plan.footprint?.[y * plan.width + x]) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0) return null;
  return { x: minX - 1, y: minY - 1, w: maxX - minX + 3, h: maxY - minY + 3 };
}
