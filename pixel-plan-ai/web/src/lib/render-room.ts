import { ICU_CATALOG_BY_TYPE } from "./icu-catalog";
import type { RoomAsset, RoomLayout } from "./types";

// CAD-grade 2D renderer for a single ICU room. It shares the floor renderer's
// pipeline ethos (a blueprint paper backdrop, dark ink linework, double-line
// walls, and a DPR-true backing store re-drawn on every zoom step) but is a
// separate module — it never imports or mutates render.ts, so the floor flow is
// untouched. The room coordinate frame is feet with the origin at the SOUTH-WEST
// corner (x east, y north); canvas y is flipped so north renders at the top.

const BACKDROP = "#f7f8fa";
const GRID_MINOR = "rgba(46,90,180,0.05)";
const GRID_MAJOR = "rgba(46,90,180,0.10)";
const INK = "26,29,33";
const WALL_INK = `rgba(${INK},0.9)`;
const DOOR_BLUE = "#2E7CEE";
const PRIMARY = "46,124,238";
const LABEL_INK = `rgba(${INK},0.72)`;
const SANS = "Inter, ui-sans-serif, system-ui, sans-serif";

// Feet of bed clearance to visualize (mirrors icu_room_rules.json bed_clearances_ft).
const FOOT_CLEAR = 5;
const TRANSFER_CLEAR = 5;
const OTHER_CLEAR = 4;
const HEAD_CLEAR = 1;

export type SymbolCache = Map<string, HTMLImageElement>;

export interface RoomRenderOptions {
  // Effective px per foot (fitBasePxPerFt × zoom); may be fractional.
  pxPerFt: number;
  // Capped devicePixelRatio for the backing store.
  dpr: number;
  showClearances: boolean;
  symbols: SymbolCache;
  selectedAssetId?: string | null;
}

function prepareCanvas(
  canvas: HTMLCanvasElement,
  cssWidth: number,
  cssHeight: number,
  dpr: number,
): CanvasRenderingContext2D {
  canvas.width = Math.max(1, Math.round(cssWidth * dpr));
  canvas.height = Math.max(1, Math.round(cssHeight * dpr));
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas rendering is unavailable.");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return context;
}

// Blueprint layer: paper backdrop + a 1 ft minor grid and a 4 ft major grid.
function drawBlueprint(context: CanvasRenderingContext2D, width: number, depth: number, s: number): void {
  const w = width * s;
  const h = depth * s;
  context.fillStyle = BACKDROP;
  context.fillRect(0, 0, w, h);

  const lines = (step: number, color: string) => {
    context.strokeStyle = color;
    context.lineWidth = 1;
    context.beginPath();
    for (let x = 0; x <= width + 1e-6; x += step) {
      const px = Math.round(x * s) + 0.5;
      context.moveTo(px, 0);
      context.lineTo(px, h);
    }
    for (let y = 0; y <= depth + 1e-6; y += step) {
      const py = Math.round(y * s) + 0.5;
      context.moveTo(0, py);
      context.lineTo(w, py);
    }
    context.stroke();
  };
  lines(1, GRID_MINOR);
  lines(4, GRID_MAJOR);
}

// Blueprint-only render for the pre-generation state (default room footprint).
export function renderEmptyRoom(
  canvas: HTMLCanvasElement,
  widthFt: number,
  depthFt: number,
  { pxPerFt, dpr }: { pxPerFt: number; dpr: number },
): void {
  const context = prepareCanvas(canvas, widthFt * pxPerFt, depthFt * pxPerFt, dpr);
  drawBlueprint(context, widthFt, depthFt, pxPerFt);
}

export function renderRoom(canvas: HTMLCanvasElement, layout: RoomLayout, options: RoomRenderOptions): void {
  const { pxPerFt: s, dpr } = options;
  const width = layout.room.width_ft;
  const depth = layout.room.depth_ft;
  const context = prepareCanvas(canvas, width * s, depth * s, dpr);
  const w = width * s;
  const h = depth * s;

  // Canvas y for a room y (north is up): py = (depth - y) * s.
  const cy = (yFt: number) => (depth - yFt) * s;

  drawBlueprint(context, width, depth, s);

  // Room interior wash so the floor reads distinct from the paper margin.
  context.fillStyle = `rgba(${INK},0.025)`;
  context.fillRect(0, 0, w, h);

  // Double-line perimeter wall band (hollow poché) via an even-odd ring fill.
  const band = Math.max(3, Math.min(10, s * 0.5));
  const ring = new Path2D();
  ring.rect(0, 0, w, h);
  ring.rect(band, band, w - 2 * band, h - 2 * band);
  context.fillStyle = WALL_INK;
  context.fill(ring, "evenodd");

  drawDoor(context, layout, s, band, cy, w, h);

  if (options.showClearances) {
    const bed = layout.assets.find((asset) => asset.type === "icu_bed");
    if (bed) drawClearances(context, bed, width, depth, s, cy);
  }

  for (const asset of layout.assets) {
    drawAsset(context, asset, s, cy, options);
  }

  if (options.selectedAssetId) {
    const asset = layout.assets.find((item) => item.id === options.selectedAssetId);
    if (asset) {
      context.save();
      context.strokeStyle = `rgba(${PRIMARY},0.9)`;
      context.lineWidth = 2;
      context.setLineDash([]);
      context.strokeRect(asset.x_ft * s, cy(asset.y_ft + asset.d_ft), asset.w_ft * s, asset.d_ft * s);
      context.restore();
    }
  }
}

function drawDoor(
  context: CanvasRenderingContext2D,
  layout: RoomLayout,
  s: number,
  band: number,
  cy: (yFt: number) => number,
  w: number,
  h: number,
): void {
  const door = layout.room.door;
  const doorWpx = door.width_ft * s;
  const half = door.width_ft / 2;

  let hinge: [number, number];
  let closed: [number, number];
  let open: [number, number];
  let slot: [number, number, number, number]; // x, y, w, h

  if (door.wall === "S") {
    const ax = (door.offset_ft - half) * s;
    const bx = (door.offset_ft + half) * s;
    hinge = [ax, h];
    closed = [bx, h];
    open = [ax, h - doorWpx];
    slot = [ax, h - band, bx - ax, band];
  } else if (door.wall === "N") {
    const ax = (door.offset_ft - half) * s;
    const bx = (door.offset_ft + half) * s;
    hinge = [ax, 0];
    closed = [bx, 0];
    open = [ax, doorWpx];
    slot = [ax, 0, bx - ax, band];
  } else if (door.wall === "W") {
    const yLow = cy(door.offset_ft - half);
    const yHigh = cy(door.offset_ft + half);
    hinge = [0, yLow];
    closed = [0, yHigh];
    open = [doorWpx, yLow];
    slot = [0, yHigh, band, yLow - yHigh];
  } else {
    const yLow = cy(door.offset_ft - half);
    const yHigh = cy(door.offset_ft + half);
    hinge = [w, yLow];
    closed = [w, yHigh];
    open = [w - doorWpx, yLow];
    slot = [w - band, yHigh, band, yLow - yHigh];
  }

  context.save();
  // Open the wall across the slot.
  context.fillStyle = BACKDROP;
  context.fillRect(slot[0], slot[1], slot[2], slot[3]);

  // Jamb caps: short ink ticks closing the double-wall ends.
  context.strokeStyle = WALL_INK;
  context.lineWidth = 1.2;
  context.beginPath();
  if (door.wall === "S" || door.wall === "N") {
    const yTop = slot[1];
    const yBot = slot[1] + slot[3];
    context.moveTo(hinge[0], yTop);
    context.lineTo(hinge[0], yBot);
    context.moveTo(closed[0], yTop);
    context.lineTo(closed[0], yBot);
  } else {
    const xL = slot[0];
    const xR = slot[0] + slot[2];
    context.moveTo(xL, hinge[1]);
    context.lineTo(xR, hinge[1]);
    context.moveTo(xL, closed[1]);
    context.lineTo(xR, closed[1]);
  }
  context.stroke();

  // Swing leaf + quarter-circle arc.
  const closedAngle = Math.atan2(closed[1] - hinge[1], closed[0] - hinge[0]);
  const openAngle = Math.atan2(open[1] - hinge[1], open[0] - hinge[0]);
  let delta = openAngle - closedAngle;
  while (delta > Math.PI) delta -= 2 * Math.PI;
  while (delta < -Math.PI) delta += 2 * Math.PI;
  context.strokeStyle = DOOR_BLUE;
  context.globalAlpha = 0.9;
  context.lineWidth = 1.4;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(hinge[0], hinge[1]);
  context.lineTo(open[0], open[1]);
  context.moveTo(closed[0], closed[1]);
  context.arc(hinge[0], hinge[1], doorWpx, closedAngle, openAngle, delta < 0);
  context.stroke();
  context.restore();
}

interface ClearRect {
  rect: [number, number, number, number];
}

function bedClearanceRects(bed: RoomAsset, width: number, depth: number): ClearRect[] {
  const bx = bed.x_ft;
  const by = bed.y_ft;
  const bw = bed.w_ft;
  const bd = bed.d_ft;
  const rects: ClearRect[] = [];
  if (bd >= bw) {
    // Long axis vertical → head/foot on N/S, transfer/other on E/W.
    const southGap = by;
    const northGap = depth - (by + bd);
    const headNorth = northGap <= southGap;
    if (headNorth) {
      rects.push({ rect: [bx, by - FOOT_CLEAR, bw, FOOT_CLEAR] });
      rects.push({ rect: [bx, by + bd, bw, HEAD_CLEAR] });
    } else {
      rects.push({ rect: [bx, by + bd, bw, FOOT_CLEAR] });
      rects.push({ rect: [bx, by - HEAD_CLEAR, bw, HEAD_CLEAR] });
    }
    const eastBig = width - (bx + bw) >= bx;
    const east = eastBig ? TRANSFER_CLEAR : OTHER_CLEAR;
    const west = eastBig ? OTHER_CLEAR : TRANSFER_CLEAR;
    rects.push({ rect: [bx + bw, by, east, bd] });
    rects.push({ rect: [bx - west, by, west, bd] });
  } else {
    // Long axis horizontal → head/foot on E/W, transfer/other on N/S.
    const westGap = bx;
    const eastGap = width - (bx + bw);
    const headEast = eastGap <= westGap;
    if (headEast) {
      rects.push({ rect: [bx - FOOT_CLEAR, by, FOOT_CLEAR, bd] });
      rects.push({ rect: [bx + bw, by, HEAD_CLEAR, bd] });
    } else {
      rects.push({ rect: [bx + bw, by, FOOT_CLEAR, bd] });
      rects.push({ rect: [bx - HEAD_CLEAR, by, HEAD_CLEAR, bd] });
    }
    const northBig = depth - (by + bd) >= by;
    const north = northBig ? TRANSFER_CLEAR : OTHER_CLEAR;
    const south = northBig ? OTHER_CLEAR : TRANSFER_CLEAR;
    rects.push({ rect: [bx, by + bd, bw, north] });
    rects.push({ rect: [bx, by - south, bw, south] });
  }
  return rects;
}

function drawClearances(
  context: CanvasRenderingContext2D,
  bed: RoomAsset,
  width: number,
  depth: number,
  s: number,
  cy: (yFt: number) => number,
): void {
  context.save();
  context.strokeStyle = `rgba(${PRIMARY},0.35)`;
  context.fillStyle = `rgba(${PRIMARY},0.06)`;
  context.lineWidth = 1;
  context.setLineDash([5, 4]);
  for (const { rect } of bedClearanceRects(bed, width, depth)) {
    const [rx, ry, rw, rh] = rect;
    const x = rx * s;
    const y = cy(ry + rh);
    context.fillRect(x, y, rw * s, rh * s);
    context.strokeRect(x, y, rw * s, rh * s);
  }
  context.restore();
}

function drawAsset(
  context: CanvasRenderingContext2D,
  asset: RoomAsset,
  s: number,
  cy: (yFt: number) => number,
  options: RoomRenderOptions,
): void {
  const catalog = ICU_CATALOG_BY_TYPE[asset.type];
  // Unrotated footprint drives the symbol size; the rotate() produces the swap.
  const cw = catalog ? catalog.footprint_ft[0] : asset.w_ft;
  const cd = catalog ? catalog.footprint_ft[1] : asset.d_ft;
  const centerX = (asset.x_ft + asset.w_ft / 2) * s;
  const centerY = cy(asset.y_ft + asset.d_ft / 2);
  const iw = cw * s;
  const ih = cd * s;
  const alpha = asset.anchor === "ceiling" ? 0.55 : asset.anchor === "mobile" ? 0.85 : 1;
  const image = options.symbols.get(asset.type);

  context.save();
  context.globalAlpha = alpha;
  context.translate(centerX, centerY);
  // rotation_deg is counter-clockwise in the room's y-up frame (matching the 3D
  // viewport's documented convention). Canvas y points DOWN, so a positive room
  // CCW turn is a NEGATIVE canvas rotation — hence the sign flip. Without it the
  // 2D symbol spun the opposite way from the GLB, so a validator-passing facing
  // (front-south at rot 0; wall map N:0 S:180 E:270 W:90) looked correct in 3D
  // but backwards in 2D. See scripts/generate-3d-assets.mjs for the convention.
  context.rotate(-(asset.rotation_deg * Math.PI) / 180);
  if (image && image.complete && image.naturalWidth > 0) {
    context.drawImage(image, -iw / 2, -ih / 2, iw, ih);
    if (asset.anchor === "ceiling") {
      context.setLineDash([4, 3]);
      context.strokeStyle = `rgba(${INK},0.5)`;
      context.lineWidth = 1;
      context.strokeRect(-iw / 2, -ih / 2, iw, ih);
    }
  } else {
    // Fallback: a labeled rect so a missing/failed SVG never leaves a blank.
    context.setLineDash(asset.anchor === "ceiling" ? [4, 3] : []);
    context.fillStyle = `rgba(${INK},0.05)`;
    context.strokeStyle = `rgba(${INK},0.6)`;
    context.lineWidth = 1.2;
    context.fillRect(-iw / 2, -ih / 2, iw, ih);
    context.strokeRect(-iw / 2, -ih / 2, iw, ih);
    if (Math.min(iw, ih) > 18) {
      context.setLineDash([]);
      context.globalAlpha = alpha;
      context.fillStyle = LABEL_INK;
      context.font = `500 9px ${SANS}`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(catalog ? catalog.label : asset.type, 0, 0);
    }
  }
  context.restore();

  // Screen-space label beneath the footprint when the zoom gives it room.
  if (s >= 14 && image && image.complete && image.naturalWidth > 0) {
    const label = catalog ? catalog.label : asset.type.replaceAll("_", " ");
    context.save();
    context.globalAlpha = Math.min(1, alpha + 0.15);
    context.fillStyle = LABEL_INK;
    context.font = `500 10px ${SANS}`;
    context.textAlign = "center";
    context.textBaseline = "top";
    context.fillText(label, centerX, cy(asset.y_ft) + 3);
    context.restore();
  }
}
