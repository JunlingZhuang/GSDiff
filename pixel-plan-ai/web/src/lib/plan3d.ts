import type { Plan } from "./types";

// Derives a schematic 3D model (Finch-style massing) purely from the plan
// grid: floor slabs per room, walls along cell boundaries, door openings
// left as gaps. All dimensions in meters; the model is centered on origin.

export interface Box3D {
  center: [number, number, number];
  size: [number, number, number];
  color: string;
}

// A door leaf rendered as a CHILD of a hinge group: `offset` is the leaf box
// centre in the group's LOCAL frame (local +X = unit vector from jamb A to
// jamb B before the swing), so rotating the group about world Y swings the
// whole panel around the hinge jamb like a real door.
export interface DoorLeaf3D {
  offset: [number, number, number];
  size: [number, number, number];
  color: string;
}

export interface Door3D {
  // World position of the hinge group — jamb A, on the floor plane (y = 0).
  hinge: [number, number, number];
  // Total group Y rotation: wall-alignment base + the signed 80° open swing.
  rotationY: number;
  leaf: DoorLeaf3D;
  // The two slim jamb posts capping the opening ends (world-positioned).
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

// Door components (Finch-scale, no textures). The leaf is a floor-standing thin
// panel hinged at jamb A and swung 80° into the swing room — wide open, so it
// reads unmistakably as a door from the orbit camera. Jamb posts cap the ends.
const DOOR_LEAF_HEIGHT = WALL_HEIGHT * 0.85;
const DOOR_LEAF_THICKNESS = WALL_THICKNESS * 0.35;
const DOOR_OPEN_RAD = (80 * Math.PI) / 180;
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

  // Door components sit in the wall gaps carved above: a hinged leaf swung 80°
  // open plus two jamb posts. Same endpoint convention as the gap carving: the
  // opening runs A=(x, y) → B=(x+width_cells, y) for a horizontal door (wall
  // along +X at z=y) and A=(x, y) → B=(x, y+width_cells) for a vertical one
  // (wall along +Z at x=x). All world coords in meters, centered on origin.
  const doors: Door3D[] = [];
  for (const door of plan.doors) {
    const horizontal = door.orientation === "horizontal";
    const cells = Math.max(1, door.width_cells);
    const len = cells * m;
    const worldX = (cx: number): number => cx * m - offsetX;
    const worldZ = (cy: number): number => cy * m - offsetZ;
    // Jamb A carries the hinge; jamb B is the far opening end.
    const bX = horizontal ? door.x + cells : door.x;
    const bY = horizontal ? door.y : door.y + cells;
    const hingeX = worldX(door.x);
    const hingeZ = worldZ(door.y);

    // Wall direction (unit, A→B) and the swing normal pointing into the room.
    const [wx, wz]: [number, number] = horizontal ? [1, 0] : [0, 1];
    const [nx, nz]: [number, number] = horizontal
      ? door.swing_side === "north"
        ? [0, -1]
        : [0, 1]
      : door.swing_side === "west"
        ? [-1, 0]
        : [1, 0];

    // Hinge group rotation. Base angle aligns the group's local +X with the
    // wall direction (three.js Y-rotation by θ maps local +X to world
    // (cosθ, 0, −sinθ), so wallDir +Z needs θ = −π/2). The swing then adds a
    // SIGNED 80° about world Y: rotating wallDir by +δ lands at
    // (wx·cosδ + wz·sinδ, −wx·sinδ + wz·cosδ), whose dot with the swing normal
    // has the sign of cross(wallDir, swingNormal).y = wz·nx − wx·nz — so that
    // cross sign IS the swing sign (verified per side by the dev check below).
    const base = horizontal ? 0 : -Math.PI / 2;
    const sign = Math.sign(wz * nx - wx * nz) || 1;
    const rotationY = base + sign * DOOR_OPEN_RAD;

    // Clear opening between the two jamb posts (each T wide, centred on A / B).
    const leafWidth = Math.max(len - WALL_THICKNESS, WALL_THICKNESS);

    // Dev-time sanity lock: the leaf's far corner (local x = T/2 + leafWidth on
    // the floor plane) must land inside the swing room. Local (L, ·, 0) maps to
    // a world offset of (L·cos rotationY, ·, −L·sin rotationY) from the hinge.
    if (process.env.NODE_ENV !== "production") {
      const farL = WALL_THICKNESS / 2 + leafWidth;
      const farX = hingeX + farL * Math.cos(rotationY);
      const farZ = hingeZ + -farL * Math.sin(rotationY);
      const swingOk = horizontal
        ? door.swing_side === "south"
          ? farZ > hingeZ // south room lies at z > wall z
          : farZ < hingeZ // north room lies at z < wall z
        : door.swing_side === "east"
          ? farX > hingeX // east room lies at x > wall x
          : farX < hingeX; // west room lies at x < wall x
      if (!swingOk) {
        console.warn(
          `plan3d: door ${door.id} leaf swings away from its ${door.swing_side} room`,
        );
      }
    }

    const entrance = door.to_room === null;
    doors.push({
      hinge: [hingeX, 0, hingeZ],
      rotationY,
      leaf: {
        // In the group's local frame: past the hinge post (T/2), reaching the
        // near face of the far post, standing on the floor (bottom at y = 0).
        offset: [leafWidth / 2 + WALL_THICKNESS / 2, DOOR_LEAF_HEIGHT / 2, 0],
        size: [leafWidth, DOOR_LEAF_HEIGHT, DOOR_LEAF_THICKNESS],
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
