import type { Plan } from "./types";

// Derives a schematic 3D model (Finch-style massing) purely from the plan
// grid: floor slabs per room, walls along cell boundaries, door openings
// left as gaps. All dimensions in meters; the model is centered on origin.

export interface Box3D {
  center: [number, number, number];
  size: [number, number, number];
  color: string;
}

export interface PlanModel {
  floors: Box3D[];
  walls: Box3D[];
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
