import type { Plan } from "./types";
import { buildScene } from "./scene";

// Derives a schematic 3D model (Finch-style massing) from the plan grid:
// floor slabs per room, walls reused from the shared 2D scene graph's UNIQUE
// merged boundary runs, door-height openings with lintel headers above them,
// and hinged leaves with jamb frames. All dimensions in meters; the model is
// centered on origin.

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

  // Walls from UNIQUE boundary runs. The shared 2D scene graph (scene.ts)
  // already derives exactly one merged wall run per region-transition edge, so
  // reusing its walls array guarantees the 3D massing can never emit two
  // co-located boxes for the same boundary — previously a twin wall could span
  // a door opening that had only been carved out of its double. Scene runs
  // split where the flanking region pair or kind changes; 3D renders interior
  // and exterior walls identically (same thickness/height, as before), so
  // collapse touching collinear runs into maximal intervals per boundary line
  // first — otherwise the corner-closure elongation below would make touching
  // pieces overlap.
  const lineKey = (horizontal: boolean, fixed: number): string => (horizontal ? `h:${fixed}` : `v:${fixed}`);
  const lineRuns = new Map<string, [number, number][]>();
  for (const w of buildScene(plan).walls) {
    const horizontal = w.y1 === w.y2;
    const key = lineKey(horizontal, horizontal ? w.y1 : w.x1);
    const span: [number, number] = horizontal
      ? [Math.min(w.x1, w.x2), Math.max(w.x1, w.x2)]
      : [Math.min(w.y1, w.y2), Math.max(w.y1, w.y2)];
    const runs = lineRuns.get(key);
    if (runs) runs.push(span);
    else lineRuns.set(key, [span]);
  }
  for (const runs of lineRuns.values()) {
    runs.sort((p, q) => p[0] - q[0]);
    let write = 0;
    for (let i = 1; i < runs.length; i += 1) {
      if (runs[i][0] <= runs[write][1]) runs[write][1] = Math.max(runs[write][1], runs[i][1]);
      else runs[(write += 1)] = runs[i];
    }
    runs.length = write + 1;
  }

  // Door openings per boundary line, in cell units along the line. A
  // horizontal door sits in the wall running along +X at z = y (opening
  // A=(x,y) → B=(x+width,y)); a vertical door in the wall along +Z at x = x.
  const doorSpans = new Map<string, [number, number][]>();
  for (const door of plan.doors) {
    const horizontal = door.orientation === "horizontal";
    const key = lineKey(horizontal, horizontal ? door.y : door.x);
    const start = horizontal ? door.x : door.y;
    const span: [number, number] = [start, start + Math.max(1, door.width_cells)];
    const spans = doorSpans.get(key);
    if (spans) spans.push(span);
    else doorSpans.set(key, [span]);
  }
  for (const spans of doorSpans.values()) spans.sort((p, q) => p[0] - q[0]);

  // Emit wall boxes by subtracting each line's door spans from its runs:
  // full-height pieces between cuts, and a lintel header over every opening
  // (door height → wall top, full wall thickness, wall color) so the wall
  // reads continuous above the door. Ends created by a door cut stop exactly
  // at the jamb (the frame post caps them); original run ends keep the T/2
  // corner-closure elongation so L/T junctions stay filled.
  const walls: Box3D[] = [];
  const DOOR_HEIGHT = DOOR_LEAF_HEIGHT; // the opening clears exactly the leaf
  const pushWallBox = (
    horizontal: boolean,
    fixed: number,
    from: number,
    to: number,
    doorCutStart: boolean,
    doorCutEnd: boolean,
  ): void => {
    const min = from * m - (doorCutStart ? 0 : WALL_THICKNESS / 2);
    const max = to * m + (doorCutEnd ? 0 : WALL_THICKNESS / 2);
    if (max - min <= 0) return;
    const mid = (min + max) / 2;
    const length = max - min;
    walls.push(
      horizontal
        ? {
            center: [mid - offsetX, WALL_HEIGHT / 2, fixed * m - offsetZ],
            size: [length, WALL_HEIGHT, WALL_THICKNESS],
            color: WALL_COLOR,
          }
        : {
            center: [fixed * m - offsetX, WALL_HEIGHT / 2, mid - offsetZ],
            size: [WALL_THICKNESS, WALL_HEIGHT, length],
            color: WALL_COLOR,
          },
    );
  };
  const pushLintel = (horizontal: boolean, fixed: number, from: number, to: number): void => {
    // Inset to the jamb posts' inner faces so the header never overlaps (and
    // z-fights) the full-height posts that frame the opening.
    const min = from * m + WALL_THICKNESS / 2;
    const max = to * m - WALL_THICKNESS / 2;
    if (max - min <= 0) return;
    const mid = (min + max) / 2;
    const yMid = (DOOR_HEIGHT + WALL_HEIGHT) / 2;
    const yLen = WALL_HEIGHT - DOOR_HEIGHT;
    walls.push(
      horizontal
        ? {
            center: [mid - offsetX, yMid, fixed * m - offsetZ],
            size: [max - min, yLen, WALL_THICKNESS],
            color: WALL_COLOR,
          }
        : {
            center: [fixed * m - offsetX, yMid, mid - offsetZ],
            size: [WALL_THICKNESS, yLen, max - min],
            color: WALL_COLOR,
          },
    );
  };
  for (const [key, runs] of lineRuns) {
    const horizontal = key.startsWith("h:");
    const fixed = Number(key.slice(2));
    const spans = doorSpans.get(key) ?? [];
    for (const [runStart, runEnd] of runs) {
      let cursor = runStart;
      let cursorFromDoor = false;
      for (const [doorStart, doorEnd] of spans) {
        if (doorEnd <= cursor || doorStart >= runEnd) continue;
        const cutStart = Math.max(doorStart, cursor);
        const cutEnd = Math.min(doorEnd, runEnd);
        if (cutStart > cursor) pushWallBox(horizontal, fixed, cursor, cutStart, cursorFromDoor, true);
        pushLintel(horizontal, fixed, cutStart, cutEnd);
        cursor = cutEnd;
        cursorFromDoor = true;
      }
      if (cursor < runEnd) pushWallBox(horizontal, fixed, cursor, runEnd, cursorFromDoor, false);
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

      // Opening-clear lock: no wall box may put mass inside the door opening
      // BELOW the lintel. Checked volume = the clear opening between the jamb
      // posts' inner faces (posts legitimately cap the raw gap ends, and
      // perpendicular walls' corner elongation at a jamb hides inside a post)
      // × the wall band thickness × height up to the door head. Lintels pass
      // because their underside sits exactly at DOOR_LEAF_HEIGHT.
      const eps = 1e-3;
      const alongMin = (horizontal ? door.x : door.y) * m + WALL_THICKNESS / 2 + eps;
      const alongMax = (horizontal ? door.x + cells : door.y + cells) * m - WALL_THICKNESS / 2 - eps;
      const openXMin = horizontal ? alongMin - offsetX : hingeX - WALL_THICKNESS / 2 + eps;
      const openXMax = horizontal ? alongMax - offsetX : hingeX + WALL_THICKNESS / 2 - eps;
      const openZMin = horizontal ? hingeZ - WALL_THICKNESS / 2 + eps : alongMin - offsetZ;
      const openZMax = horizontal ? hingeZ + WALL_THICKNESS / 2 - eps : alongMax - offsetZ;
      const openYMax = DOOR_LEAF_HEIGHT - eps;
      for (const box of walls) {
        const [cx, cy, cz] = box.center;
        const [sx, sy, sz] = box.size;
        const blocks =
          cx + sx / 2 > openXMin &&
          cx - sx / 2 < openXMax &&
          cz + sz / 2 > openZMin &&
          cz - sz / 2 < openZMax &&
          cy - sy / 2 < openYMax &&
          cy + sy / 2 > eps;
        if (blocks) {
          console.warn(`plan3d: wall box crosses the door opening of ${door.id}`);
          break;
        }
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
