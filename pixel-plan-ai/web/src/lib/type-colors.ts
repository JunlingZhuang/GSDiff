import type { Door, Plan, PlanRoom, SeedPlan } from "./types";

// Room-type fill colours for the trace raw preview. Mirrors the hex palette in
// pixel-plan-ai/backend/constants.py (color_for) so a traced seed reads with the
// same colours the validated plan will use once /api/refine renders it.
export const DEFAULT_TYPE_COLORS: Record<string, string> = {
  waiting: "#E8A23A",
  exam_room: "#52A6A2",
  patient_room: "#6E8ED6",
  nurse_station: "#D96B7B",
  office: "#8A75C9",
  toilet: "#5F7890",
  storage: "#A87854",
  corridor: "#E8DEC8",
  circulation: "#E8DEC8",
  unassigned: "#F2EFE8",
};

const FALLBACK_COLOR = "#9B8D7A";

export function colorForType(type: string): string {
  return DEFAULT_TYPE_COLORS[type] ?? FALLBACK_COLOR;
}

// Trace seeds may carry the cell grid flat (rasterize_plan) or nested; the
// renderer always wants one flat row-major array.
function flattenCells(cells: number[] | number[][]): number[] {
  return Array.isArray(cells[0]) ? (cells as number[][]).flat() : (cells as number[]);
}

// Per-room pixel_count + bounding box from the cell grid (rooms are painted by
// their index in the seed's rooms list, matching rasterize_plan).
function roomExtents(
  cells: number[],
  width: number,
  height: number,
  roomCount: number,
): { pixel_count: number; bounds: { x: number; y: number; width: number; height: number } }[] {
  const extents = Array.from({ length: roomCount }, () => ({
    pixel_count: 0,
    minX: Infinity,
    minY: Infinity,
    maxX: -Infinity,
    maxY: -Infinity,
  }));
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = cells[y * width + x];
      if (index === undefined || index < 0 || index >= roomCount) continue;
      const extent = extents[index];
      extent.pixel_count += 1;
      if (x < extent.minX) extent.minX = x;
      if (y < extent.minY) extent.minY = y;
      if (x > extent.maxX) extent.maxX = x;
      if (y > extent.maxY) extent.maxY = y;
    }
  }
  return extents.map((extent) =>
    extent.pixel_count > 0
      ? {
          pixel_count: extent.pixel_count,
          bounds: {
            x: extent.minX,
            y: extent.minY,
            width: extent.maxX - extent.minX + 1,
            height: extent.maxY - extent.minY + 1,
          },
        }
      : { pixel_count: 0, bounds: { x: 0, y: 0, width: 0, height: 0 } },
  );
}

// Convert a traced seed grid into a Plan the CAD renderer can draw directly, so
// the raw trace shows on the canvas through the same pathway as a live plan.
// Cells map 1:1: footprint covers every cell >= -1 (rooms + traced wall strips),
// -2 (outside) becomes -1 in the passthrough cells, rooms get palette colours,
// and doors pass through with a placeholder swing side (geometry only, unvalidated).
export function seedToPreviewPlan(seed: SeedPlan): Plan {
  const width = seed.width;
  const height = seed.height;
  const flat = flattenCells(seed.cells);
  const cells = flat.map((value) => (value === -2 ? -1 : value));
  const footprint = flat.map((value) => (value >= -1 ? 1 : 0));

  const extents = roomExtents(flat, width, height, seed.rooms.length);
  const rooms: PlanRoom[] = seed.rooms.map((room, index) => ({
    id: room.id ?? `room_${index}`,
    type: room.type,
    color: colorForType(room.type),
    pixel_count: extents[index]?.pixel_count ?? 0,
    bounds: extents[index]?.bounds ?? { x: 0, y: 0, width: 0, height: 0 },
  }));

  const doors: Door[] = (seed.doors ?? []).map((door, index) => ({
    id: door.id ?? `door_${index}`,
    from_room: door.from_room,
    to_room: door.to_room,
    x: door.x,
    y: door.y,
    orientation: door.orientation,
    width_cells: door.width_cells ?? 1,
    swing_side: "north",
  }));

  return {
    width,
    height,
    meters_per_cell: seed.meters_per_cell,
    footprint,
    cells,
    rooms,
    doors,
  };
}
