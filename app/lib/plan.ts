// Plan data model — mirrors procedural/plan_export.build_plan output.
// Shared by the 2D editor (Phase 1-2) and the future 3D view (Phase 3).

export type Pt = [number, number]; // [x, y] in metres

export interface Wall {
  id: string;
  a: Pt;
  b: Pt;
  thickness: number; // metres
}

export type OpeningKind = 'door' | 'window' | 'passage';

export interface Opening {
  id: string;
  wallId: string;
  t: number; // 0..1 along the wall
  width: number;
  kind: OpeningKind;
}

// Canonical room types. The backend room_type is a raw string, so a value
// outside this union is possible; renderers should fall back to a default.
export type RoomType =
  | 'Bedroom' | 'Livingroom' | 'Kitchen' | 'Dining' | 'Corridor'
  | 'Stairs' | 'Storeroom' | 'Bathroom' | 'Balcony';

export interface Room {
  id: string;
  type: RoomType;
  poly: Pt[];
  wallIds: string[];
}

export interface AxisGrid {
  originX: number;
  originY: number;
  spacingX: number;
  spacingY: number;
  angleDeg: number; // building dominant-axis angle, degrees
}

export interface Plan {
  walls: Wall[];
  openings: Opening[];
  rooms: Room[];
  grid: AxisGrid;
  unit: 'm';
}

// Bounds of every room/wall point — used to fit the SVG viewBox.
export function planBounds(plan: Plan): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const eat = (x: number, y: number) => {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  };
  for (const r of plan.rooms) for (const [x, y] of r.poly) eat(x, y);
  for (const w of plan.walls) { eat(w.a[0], w.a[1]); eat(w.b[0], w.b[1]); }
  if (!isFinite(minX)) return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  return { minX, minY, maxX, maxY };
}
