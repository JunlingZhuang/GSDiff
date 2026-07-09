import type { Plan } from "./types";
import type { Scene, SceneDoor } from "./scene";

// CAD-grade 2D plan renderer. The plan reads as an architectural linework
// drawing inked on cool drafting paper: a merged footprint silhouette,
// low-alpha room fills with a full-strength inner accent, crisp dark wall
// strokes traced from the cell grid, proper door swing arcs, and a layered
// two-line label lockup. Cells are always square — the canvas takes the plan's
// aspect ratio, never the container's. The caller passes the *effective* px/cell
// (fitBaseCellPx × zoom) and a devicePixelRatio, so the raster is drawn at the
// display's true resolution and re-drawn on every zoom step instead of being
// resampled by a CSS scale transform. Panning stays a CSS translate (which does
// not blur), so this only redraws on plan / zoom / hover / dpr / size changes.
const BACKDROP = "#f7f8fa";
const GRID_MINOR = "rgba(46,90,180,0.045)";
const GRID_MAJOR = "rgba(46,90,180,0.09)";

// Linework palette (dark ink on the paper canvas).
const INK = "26,29,33"; // --foreground channels
const FOOTPRINT_FILL = `rgba(${INK},0.03)`;
const HATCH = `rgba(${INK},0.10)`;
const WALL_INTERIOR = `rgba(${INK},0.50)`;
const WALL_EXTERIOR = `rgba(${INK},0.90)`;
const DOOR_BLUE = "#2E7CEE";
const DOOR_GREEN = "#0E9F6E";
const LABEL_NAME = `rgba(${INK},0.88)`;
const LABEL_AREA = "rgba(102,112,133,0.9)";

const UNASSIGNED = -1;

interface RenderOptions {
  // Effective px/cell at draw time (fitBaseCellPx × zoom); may be fractional.
  cellPx: number;
  // Capped devicePixelRatio for the backing store (viewport clamps to ≤ 2.5).
  dpr: number;
  hoveredRoomIndex?: number;
  preview?: boolean;
}

export function cellScale(width: number): number {
  return Math.max(6, Math.min(14, Math.floor(1280 / width)));
}

export function planCellScale(plan: Plan): number {
  return cellScale(plan.width);
}

// Canvas cannot resolve CSS variables inside ctx.font, so read the resolved
// families off the DOM once and reuse them for every label.
let cachedSans: string | null = null;
let cachedMono: string | null = null;
function fontFamilies(): { sans: string; mono: string } {
  if (cachedSans === null) {
    try {
      const style = getComputedStyle(document.body);
      cachedSans = style.fontFamily || "Inter, sans-serif";
      const mono = style.getPropertyValue("--font-geist-mono").trim();
      cachedMono = mono || "ui-monospace, SFMono-Regular, monospace";
    } catch {
      cachedSans = "Inter, sans-serif";
      cachedMono = "ui-monospace, SFMono-Regular, monospace";
    }
  }
  return { sans: cachedSans, mono: cachedMono ?? "ui-monospace, monospace" };
}

function withAlpha(color: string, alpha: number): string {
  const c = color.trim();
  if (c.startsWith("#")) {
    let hex = c.slice(1);
    if (hex.length === 3) hex = hex.split("").map((ch) => ch + ch).join("");
    if (hex.length >= 6) {
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return `rgba(${r},${g},${b},${alpha})`;
    }
  }
  const match = c.match(/rgba?\(([^)]+)\)/);
  if (match) {
    const [r, g, b] = match[1].split(",").map((s) => s.trim());
    return `rgba(${r},${g},${b},${alpha})`;
  }
  return `rgba(${INK},${alpha})`;
}

function isCirculation(type: string): boolean {
  const t = type.toLowerCase();
  return t.includes("corridor") || t.includes("circulation");
}

function titleCase(id: string): string {
  return id
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function prepareCanvas(
  canvas: HTMLCanvasElement,
  cssWidth: number,
  cssHeight: number,
  dpr: number,
): CanvasRenderingContext2D {
  // CSS size is the plan's display size at the current effective scale; the
  // backing store is dpr× larger so every stroke, arc, and glyph rasterizes at
  // the display's true resolution instead of being resampled by a CSS scale.
  canvas.width = Math.max(1, Math.round(cssWidth * dpr));
  canvas.height = Math.max(1, Math.round(cssHeight * dpr));
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas rendering is unavailable.");
  // One transform maps CSS-pixel drawing coordinates to device pixels; every
  // path, arc, and label below is expressed in CSS pixels and stays crisp.
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return context;
}

// Blueprint layer: paper backdrop + minor grid every cell + major grid every 8,
// spanning the full canvas so the plan floats on a continuous drafting surface.
function drawBlueprint(context: CanvasRenderingContext2D, cols: number, rows: number, scale: number): void {
  const width = cols * scale;
  const height = rows * scale;
  context.fillStyle = BACKDROP;
  context.fillRect(0, 0, width, height);

  context.lineWidth = 1;
  context.strokeStyle = GRID_MINOR;
  context.beginPath();
  for (let x = 0; x <= cols; x += 1) {
    const px = Math.round(x * scale) + 0.5;
    context.moveTo(px, 0);
    context.lineTo(px, height);
  }
  for (let y = 0; y <= rows; y += 1) {
    const py = Math.round(y * scale) + 0.5;
    context.moveTo(0, py);
    context.lineTo(width, py);
  }
  context.stroke();

  context.strokeStyle = GRID_MAJOR;
  context.beginPath();
  for (let x = 0; x <= cols; x += 8) {
    const px = Math.round(x * scale) + 0.5;
    context.moveTo(px, 0);
    context.lineTo(px, height);
  }
  for (let y = 0; y <= rows; y += 8) {
    const py = Math.round(y * scale) + 0.5;
    context.moveTo(0, py);
    context.lineTo(width, py);
  }
  context.stroke();
}

// Empty viewport: just the blueprint grid, sized to a default plan footprint.
export function renderEmptyGrid(
  canvas: HTMLCanvasElement,
  cols: number,
  rows: number,
  { cellPx, dpr }: { cellPx: number; dpr: number },
): void {
  const context = prepareCanvas(canvas, cols * cellPx, rows * cellPx, dpr);
  drawBlueprint(context, cols, rows, cellPx);
}

export function renderPlanToCanvas(
  canvas: HTMLCanvasElement,
  plan: Plan,
  scene: Scene,
  options: RenderOptions,
): void {
  const { cellPx, dpr, hoveredRoomIndex = -1, preview = false } = options;
  // `cellPx` is the effective px/cell (fitBaseCellPx × zoom); every coordinate
  // below derives from it at draw time, so zooming re-rasterizes rather than
  // CSS-scaling. The label hide threshold now reads against this effective size.
  const scale = cellPx;
  const context = prepareCanvas(canvas, plan.width * cellPx, plan.height * cellPx, dpr);
  drawBlueprint(context, plan.width, plan.height, scale);
  drawPlanBody(context, plan, scene, scale, { hoveredRoomIndex, preview });
}

// Everything above the blueprint grid: footprint silhouette + hatch, room fills
// with inner accents, walls, doors, and labels. All geometry is read from the
// retained scene; only per-cell fills (footprint + room bodies) are rebuilt at
// draw time because they depend on the effective scale. Kept as a standalone
// pass so it can be composited onto its own layer without redrawing the grid.
function drawPlanBody(
  context: CanvasRenderingContext2D,
  plan: Plan,
  scene: Scene,
  scale: number,
  { hoveredRoomIndex = -1, preview = false }: { hoveredRoomIndex?: number; preview?: boolean },
): void {
  const W = plan.width;
  const H = plan.height;
  const hasFootprint = Array.isArray(plan.footprint) && plan.footprint.length === W * H;

  // Room index of a cell (>= 0) or UNASSIGNED, out of bounds → UNASSIGNED.
  const cellRoom = (x: number, y: number): number => {
    if (x < 0 || y < 0 || x >= W || y >= H) return UNASSIGNED;
    const v = plan.cells[y * W + x];
    return v === undefined || v < 0 ? UNASSIGNED : v;
  };

  context.save();
  if (preview) context.globalAlpha = 0.6;

  // 3. Footprint body — one merged silhouette fill (composited once so shared
  //    cell edges never accumulate alpha), plus a hatch on unassigned interior.
  if (hasFootprint) {
    const footPath = new Path2D();
    const hatchRegion = new Path2D();
    let hasHatch = false;
    for (let y = 0; y < H; y += 1) {
      for (let x = 0; x < W; x += 1) {
        if (!plan.footprint[y * W + x]) continue;
        footPath.rect(x * scale, y * scale, scale, scale);
        if (cellRoom(x, y) < 0) {
          hatchRegion.rect(x * scale, y * scale, scale, scale);
          hasHatch = true;
        }
      }
    }
    context.fillStyle = FOOTPRINT_FILL;
    context.fill(footPath);

    if (hasHatch) {
      context.save();
      context.clip(hatchRegion);
      context.strokeStyle = HATCH;
      context.lineWidth = 1;
      context.beginPath();
      const pxW = W * scale;
      const pxH = H * scale;
      for (let o = -pxH; o < pxW; o += 6) {
        context.moveTo(o, 0);
        context.lineTo(o + pxH, pxH);
      }
      context.stroke();
      context.restore();
    }
  }

  // 4. Room fills at low alpha + a full-strength inner accent hugging the walls.
  //    Fill bodies are rebuilt from the cell grid; the accent outline comes from
  //    the scene's per-room boundary edge scan.
  scene.rooms.forEach((room) => {
    const bounds = room.bounds;
    const roomIndex = room.index;
    const hovered = roomIndex === hoveredRoomIndex;
    const fillPath = new Path2D();
    for (let y = bounds.y; y < bounds.y + bounds.height; y += 1) {
      for (let x = bounds.x; x < bounds.x + bounds.width; x += 1) {
        if (cellRoom(x, y) !== roomIndex) continue;
        fillPath.rect(x * scale, y * scale, scale, scale);
      }
    }
    const boundaryPath = new Path2D();
    for (const seg of room.boundary) {
      boundaryPath.moveTo(seg.x1 * scale, seg.y1 * scale);
      boundaryPath.lineTo(seg.x2 * scale, seg.y2 * scale);
    }
    const fillAlpha = hovered ? 0.32 : isCirculation(room.type) ? 0.12 : 0.22;
    context.fillStyle = withAlpha(room.color, fillAlpha);
    context.fill(fillPath);

    // Inner accent: clip to the room, then stroke the boundary at double width
    // so the visible band lands ~3px inside the wall line.
    context.save();
    context.clip(fillPath);
    context.strokeStyle = withAlpha(room.color, hovered ? 0.85 : 0.5);
    context.lineWidth = 6;
    context.lineJoin = "round";
    context.stroke(boundaryPath);
    context.restore();
  });

  // 5. Walls — merged region-transition segments carried by the scene.
  const align = (v: number): number => Math.round(v) + 0.5;
  const interiorPath = new Path2D();
  const exteriorPath = new Path2D();
  for (const wall of scene.walls) {
    const path = wall.kind === "exterior" ? exteriorPath : interiorPath;
    path.moveTo(align(wall.x1 * scale), align(wall.y1 * scale));
    path.lineTo(align(wall.x2 * scale), align(wall.y2 * scale));
  }
  context.lineCap = "butt";
  context.lineJoin = "miter";
  context.strokeStyle = WALL_INTERIOR;
  context.lineWidth = 1.25;
  context.stroke(interiorPath);
  context.strokeStyle = WALL_EXTERIOR;
  context.lineWidth = 2.5;
  context.stroke(exteriorPath);

  // 6. Doors — carve an opening, then draw the swing arc + leaf.
  scene.doors.forEach((door) => drawDoor(context, door, scale));

  // 7. Labels — two-line lockup in each room's largest inscribed rectangle.
  const { sans, mono } = fontFamilies();
  context.textAlign = "center";
  context.textBaseline = "middle";
  scene.rooms.forEach((room) => {
    const rect = room.largestRect;
    if (rect.w * scale < 64 || rect.h * scale < 28) return;
    // Snap the label anchor to whole CSS pixels so the two-line lockup lands on
    // a device-pixel-consistent baseline (crisp at dpr ≥ 1).
    const cx = Math.round(room.labelAnchor.x * scale);
    const cy = Math.round(room.labelAnchor.y * scale);

    context.font = `500 11px ${sans}`;
    context.fillStyle = LABEL_NAME;
    context.fillText(titleCase(room.id), cx, cy - 7);

    context.font = `500 9.5px ${mono}`;
    context.fillStyle = LABEL_AREA;
    context.fillText(`${room.areaFt2} ft²`, cx, cy + 6);
  });

  context.restore();
}

function drawDoor(context: CanvasRenderingContext2D, sceneDoor: SceneDoor, scale: number): void {
  const span = sceneDoor.span * scale;
  const x0 = sceneDoor.hinge.x * scale;
  const y0 = sceneDoor.hinge.y * scale;
  const horizontal = sceneDoor.horizontal;
  const base = context.globalAlpha;

  context.save();

  // (a) Erase the wall stroke across the opening span, in the backdrop color.
  context.lineCap = "butt";
  context.strokeStyle = BACKDROP;
  context.lineWidth = Math.max(3.5, scale * 0.34);
  context.beginPath();
  context.moveTo(x0, y0);
  if (horizontal) context.lineTo(x0 + span, y0);
  else context.lineTo(x0, y0 + span);
  context.stroke();

  // Closed leaf points along the wall; open leaf swings perpendicular.
  const closedAngle = sceneDoor.closedAngle;
  const openAngle = sceneDoor.openAngle;

  const entrance = sceneDoor.isEntrance;
  context.strokeStyle = entrance ? DOOR_GREEN : DOOR_BLUE;
  context.globalAlpha = base * 0.9;
  context.lineWidth = 1.25;
  context.lineCap = "round";
  context.lineJoin = "round";

  // (c) Door leaf in the open position + (b) quarter-circle swing arc.
  context.beginPath();
  context.moveTo(x0, y0);
  context.lineTo(x0 + Math.cos(openAngle) * span, y0 + Math.sin(openAngle) * span);
  context.moveTo(x0 + Math.cos(closedAngle) * span, y0 + Math.sin(closedAngle) * span);
  context.arc(x0, y0, span, closedAngle, openAngle, openAngle < closedAngle);
  context.stroke();

  // (d) Main entrance: a small outward chevron away from the swing side.
  if (entrance) {
    const midX = horizontal ? x0 + span / 2 : x0;
    const midY = horizontal ? y0 : y0 + span / 2;
    const outAngle = openAngle + Math.PI;
    const ox = Math.cos(outAngle);
    const oy = Math.sin(outAngle);
    const perpX = -oy;
    const perpY = ox;
    const tipX = midX + ox * 10;
    const tipY = midY + oy * 10;
    const baseX = midX + ox * 4;
    const baseY = midY + oy * 4;
    context.beginPath();
    context.moveTo(baseX + perpX * 5, baseY + perpY * 5);
    context.lineTo(tipX, tipY);
    context.lineTo(baseX - perpX * 5, baseY - perpY * 5);
    context.stroke();
  }

  context.restore();
}
