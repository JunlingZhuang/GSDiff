import type { Door, Plan } from "./types";

// Ported from the legacy frontend/app.ts renderer. The plan sheet keeps the
// light drafting palette; it now sits on a dark blueprint backdrop with a faint
// cell grid so the full-bleed viewport reads as a drafting surface. Cells are
// always square — the canvas takes the plan's aspect ratio, never the container's.
const PAPER = "#f7f6f2";
const WALL = "rgba(31,36,42,.78)";
const BACKDROP = "#101318";
const GRID_MINOR = "rgba(120,150,200,0.06)";
const GRID_MAJOR = "rgba(120,150,200,0.11)";

export function cellScale(width: number): number {
  return Math.max(6, Math.min(14, Math.floor(1280 / width)));
}

export function planCellScale(plan: Plan): number {
  return cellScale(plan.width);
}

function prepareCanvas(
  canvas: HTMLCanvasElement,
  cols: number,
  rows: number,
  scale: number,
): CanvasRenderingContext2D {
  const deviceScale = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = cols * scale * deviceScale;
  canvas.height = rows * scale * deviceScale;
  canvas.style.width = `${cols * scale}px`;
  canvas.style.height = `${rows * scale}px`;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas rendering is unavailable.");
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.scale(deviceScale, deviceScale);
  return context;
}

// Blueprint layer: dark backdrop + minor grid every cell + major grid every 8.
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
export function renderEmptyGrid(canvas: HTMLCanvasElement, cols = 96, rows = 64): void {
  const scale = cellScale(cols);
  const context = prepareCanvas(canvas, cols, rows, scale);
  drawBlueprint(context, cols, rows, scale);
}

export function renderPlanToCanvas(canvas: HTMLCanvasElement, plan: Plan): void {
  const scale = planCellScale(plan);
  const context = prepareCanvas(canvas, plan.width, plan.height, scale);
  drawBlueprint(context, plan.width, plan.height, scale);

  for (let y = 0; y < plan.height; y += 1) {
    for (let x = 0; x < plan.width; x += 1) {
      const offset = y * plan.width + x;
      if (plan.footprint?.[offset]) {
        context.fillStyle = PAPER;
        context.fillRect(x * scale, y * scale, scale + 0.4, scale + 0.4);
      }
    }
  }

  for (let y = 0; y < plan.height; y += 1) {
    for (let x = 0; x < plan.width; x += 1) {
      const roomIndex = plan.cells[y * plan.width + x];
      const room = roomIndex === undefined || roomIndex < 0 ? undefined : plan.rooms[roomIndex];
      if (!room) continue;
      context.fillStyle = room.color;
      context.fillRect(x * scale, y * scale, scale + 0.4, scale + 0.4);
    }
  }

  context.strokeStyle = WALL;
  context.lineWidth = Math.max(1, scale * 0.09);
  context.beginPath();
  for (let y = 0; y < plan.height; y += 1) {
    for (let x = 0; x < plan.width; x += 1) {
      const index = plan.cells[y * plan.width + x];
      if (index === undefined || index < 0) continue;
      if (x === 0 || plan.cells[y * plan.width + x - 1] !== index) {
        context.moveTo(x * scale, y * scale);
        context.lineTo(x * scale, (y + 1) * scale);
      }
      if (y === 0 || plan.cells[(y - 1) * plan.width + x] !== index) {
        context.moveTo(x * scale, y * scale);
        context.lineTo((x + 1) * scale, y * scale);
      }
      if (x === plan.width - 1) {
        context.moveTo((x + 1) * scale, y * scale);
        context.lineTo((x + 1) * scale, (y + 1) * scale);
      }
      if (y === plan.height - 1) {
        context.moveTo(x * scale, (y + 1) * scale);
        context.lineTo((x + 1) * scale, (y + 1) * scale);
      }
    }
  }
  context.stroke();

  if (scale >= 7 && plan.rooms.length <= 90) {
    context.textAlign = "center";
    context.textBaseline = "middle";
    for (const [roomIndex, room] of plan.rooms.entries()) {
      const bounds = room.bounds;
      if (bounds.width * scale < 32 || bounds.height * scale < 20) continue;
      const targetX = bounds.x + bounds.width / 2;
      const targetY = bounds.y + bounds.height / 2;
      let labelX = targetX;
      let labelY = targetY;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let y = bounds.y; y < bounds.y + bounds.height; y += 1) {
        for (let x = bounds.x; x < bounds.x + bounds.width; x += 1) {
          if (plan.cells[y * plan.width + x] !== roomIndex) continue;
          const distance = (x + 0.5 - targetX) ** 2 + (y + 0.5 - targetY) ** 2;
          if (distance < bestDistance) {
            bestDistance = distance;
            labelX = x + 0.5;
            labelY = y + 0.5;
          }
        }
      }
      const label = room.id.replaceAll("_", " ");
      context.font = `600 ${Math.max(7, Math.min(10, scale * 0.72))}px var(--font-geist-mono), ui-monospace, monospace`;
      context.fillStyle = "rgba(24,29,35,.8)";
      context.fillText(label.length > 17 ? `${label.slice(0, 15)}...` : label, labelX * scale, labelY * scale);
    }
  }
  plan.doors.forEach((door) => drawDoor(context, door, scale));
}

function drawDoor(context: CanvasRenderingContext2D, door: Door, scale: number): void {
  const width = door.width_cells * scale;
  const x = door.x * scale;
  const y = door.y * scale;
  context.save();
  context.lineCap = "square";
  context.strokeStyle = PAPER;
  context.lineWidth = Math.max(3, scale * 0.35);
  context.beginPath();
  if (door.orientation === "horizontal") {
    context.moveTo(x, y);
    context.lineTo(x + width, y);
  } else {
    context.moveTo(x, y);
    context.lineTo(x, y + width);
  }
  context.stroke();
  context.strokeStyle = door.to_room === null ? "#e2593a" : "#1f242a";
  context.lineWidth = Math.max(1, scale * 0.12);
  context.beginPath();
  let openAngle = 0;
  let closedAngle = 0;
  if (door.orientation === "horizontal") {
    openAngle = door.swing_side === "south" ? Math.PI / 2 : -Math.PI / 2;
    closedAngle = 0;
    context.moveTo(x, y);
    context.lineTo(x + Math.cos(openAngle) * width, y + Math.sin(openAngle) * width);
    context.moveTo(x + width, y);
    context.arc(x, y, width, closedAngle, openAngle, openAngle < closedAngle);
  } else {
    openAngle = door.swing_side === "west" ? Math.PI : 0;
    closedAngle = Math.PI / 2;
    context.moveTo(x, y);
    context.lineTo(x + Math.cos(openAngle) * width, y + Math.sin(openAngle) * width);
    context.moveTo(x, y + width);
    context.arc(x, y, width, closedAngle, openAngle, openAngle < closedAngle);
  }
  context.stroke();
  context.restore();
}
