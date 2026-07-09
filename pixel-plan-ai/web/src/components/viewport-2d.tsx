"use client";

import * as React from "react";

import { cellScale, clearLayer, planCellScale, renderEmptyGrid, renderOverlay, renderPlanLayer, roomDisplayName } from "@/lib/render";
import { buildScene, segmentLengthCells } from "@/lib/scene";
import type { Hit } from "@/lib/scene";
import type { CandidateImage, Plan, PlanRoom } from "@/lib/types";
import { candidateDataUrl, prettyType } from "@/lib/types";
import { cn } from "@/lib/utils";

const DEFAULT_COLS = 96;
const DEFAULT_ROWS = 64;

// The floating panels overlap the viewport; fit-to-view reserves their gutters
// so the plan lands centered in the visible drawing area, not behind a panel.
const PANEL_LEFT = 324;
const PANEL_RIGHT = 344;
const PANEL_RIGHT_CLOSED = 12;
const FIT_PADDING = 48;

// Whole-plan rasters are cheap (plans are ~96×64), but at extreme zoom the
// backing store could exceed the browser's max canvas area. Past this cap we
// trade backing-store resolution (never a CSS scale) so we never crash.
const MAX_BACKING_PX = 8192;
// devicePixelRatio is capped here so a 3×/4× display never quadruples fill cost.
const MAX_DPR = 2.5;
// Pick radius (screen px) for hovering walls / doors — converted to cell space.
const HIT_TOLERANCE_PX = 6;

function resolveDpr(): number {
  if (typeof window === "undefined") return 1;
  return Math.min(MAX_DPR, window.devicePixelRatio || 1);
}

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

interface Viewport2DProps {
  plan: Plan | null;
  reference: CandidateImage | null;
  showReference: boolean;
  referenceOpacity: number;
  preview?: boolean;
  rightPanelOpen?: boolean;
  selectedRoomId?: string | null;
  onViewChange?: (zoom: number) => void;
  onHoverCell?: (cell: { x: number; y: number } | null) => void;
  onHoverEntity?: (label: string | null) => void;
  onSelectRoom?: (roomId: string | null) => void;
}

export interface Viewport2DHandle {
  zoomBy: (factor: number) => void;
  fit: () => void;
  exportPng: () => string | null;
}

// Which entity is under the cursor. Kept separate from the raw cursor position
// so the overlay only repaints when the entity changes, not on every move.
interface HoverState {
  hit: Hit;
  // Present only when the hovered entity is a room (drives the tooltip).
  room: PlanRoom | null;
}

function hitKey(hit: Hit): string {
  if (hit.kind === "room") return `r${hit.roomIndex}`;
  if (hit.kind === "wall") return `w${hit.wall.x1},${hit.wall.y1},${hit.wall.x2},${hit.wall.y2}`;
  return `d${hit.door.id}`;
}

// zoom is the factor relative to the fitted baseline (1 = fitted). base is the
// fitted px/cell; the effective px/cell drawn at any moment is base × zoom.
// x / y are the plan's top-left offset in CSS pixels (applied as a CSS translate
// — translation never blurs, so panning does not force a re-raster).
interface View {
  zoom: number;
  x: number;
  y: number;
  base: number;
}

export const Viewport2D = React.forwardRef<Viewport2DHandle, Viewport2DProps>(function Viewport2D(
  {
    plan,
    reference,
    showReference,
    referenceOpacity,
    preview = false,
    rightPanelOpen = false,
    selectedRoomId = null,
    onViewChange,
    onHoverCell,
    onHoverEntity,
    onSelectRoom,
  },
  handleRef,
) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  // Three stacked layers sharing one display box: the blueprint grid, the plan
  // linework, and the interaction overlay. Splitting them lets a hover / select
  // repaint just the (cheap) overlay without re-rasterizing the plan or grid.
  const gridCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const planCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const [view, setView] = React.useState<View>({ zoom: 1, x: 0, y: 0, base: cellScale(DEFAULT_COLS) });
  const viewRef = React.useRef(view);
  viewRef.current = view;
  const planSizeRef = React.useRef<string>("");
  const rightPanelOpenRef = React.useRef(rightPanelOpen);
  rightPanelOpenRef.current = rightPanelOpen;
  // Active pan gesture. `panning` distinguishes a pan (middle-drag / space-drag)
  // from a potential click-select on the left button. vx/vy accumulate the last
  // pointer velocity (px/frame) so release can spin up inertia.
  const dragRef = React.useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
    panning: boolean;
    moved: boolean;
    lastX: number;
    lastY: number;
    lastT: number;
    vx: number;
    vy: number;
  } | null>(null);
  const inertiaRef = React.useRef<number | null>(null);
  const spaceHeldRef = React.useRef(false);
  const [spaceHeld, setSpaceHeld] = React.useState(false);
  const [panningActive, setPanningActive] = React.useState(false);
  // Last entity label pushed to the read-out, so we only re-notify on change.
  const entityLabelRef = React.useRef<string | null>(null);
  // Which entity is hovered (drives the overlay) vs. where the cursor is (drives
  // only the floating tooltip). Splitting them keeps the overlay from repainting
  // on every pointer move within one entity.
  const [hover, setHover] = React.useState<HoverState | null>(null);
  const hoverKeyRef = React.useRef<string>("");
  const [pointer, setPointer] = React.useState<{ x: number; y: number } | null>(null);
  const hoveredRoomIndex = hover && hover.hit.kind === "room" ? hover.hit.roomIndex : -1;
  const [dpr, setDpr] = React.useState(resolveDpr);

  const dims = React.useMemo(() => {
    const cols = plan?.width ?? DEFAULT_COLS;
    const rows = plan?.height ?? DEFAULT_ROWS;
    // Natural px/cell — used only to bound the fitted scale's readability floor.
    const scale = plan ? planCellScale(plan) : cellScale(cols);
    return { cols, rows, scale };
  }, [plan]);

  // Retained scene graph — the single geometry source of truth for both the
  // renderer and hit testing. Computed once per plan, reused across zoom/pan.
  const scene = React.useMemo(() => (plan ? buildScene(plan) : null), [plan]);

  const selectedRoomIndex = React.useMemo(() => {
    if (!plan || !selectedRoomId) return -1;
    return plan.rooms.findIndex((room) => room.id === selectedRoomId);
  }, [plan, selectedRoomId]);

  // Effective px/cell and the plan's display box at the current zoom. Derived in
  // render so the wrapper and the canvas backing store size stay in lock-step.
  const cellPx = view.base * view.zoom;
  const contentSize = { width: dims.cols * cellPx, height: dims.rows * cellPx };

  // Effective px/cell + capped dpr for the current view. Backing-store guard:
  // shrink dpr (never the CSS size) past the browser's max canvas area.
  const backingFor = React.useCallback(() => {
    const v = viewRef.current;
    const effectiveCellPx = v.base * v.zoom;
    const contentW = dims.cols * effectiveCellPx;
    const contentH = dims.rows * effectiveCellPx;
    const capScale = Math.min(1, MAX_BACKING_PX / (contentW * dpr), MAX_BACKING_PX / (contentH * dpr));
    return { effectiveCellPx, contentW, contentH, effectiveDpr: dpr * capScale };
  }, [dims.cols, dims.rows, dpr]);

  // Grid + plan layers: re-rasterized on plan / zoom / scale / preview / dpr.
  // Panning does NOT trigger this — the layers are CSS-translated. Hover does
  // NOT trigger this either — that only repaints the overlay.
  const drawScene = React.useCallback(() => {
    const { effectiveCellPx, contentW, contentH, effectiveDpr } = backingFor();
    if (!(effectiveCellPx > 0)) return;
    const grid = gridCanvasRef.current;
    if (grid) renderEmptyGrid(grid, dims.cols, dims.rows, { cellPx: effectiveCellPx, dpr: effectiveDpr });
    const planCanvas = planCanvasRef.current;
    if (!planCanvas) return;
    if (plan && scene) {
      renderPlanLayer(planCanvas, plan, scene, { cellPx: effectiveCellPx, dpr: effectiveDpr, preview });
    } else {
      clearLayer(planCanvas, contentW, contentH, effectiveDpr);
    }
  }, [plan, scene, dims.cols, dims.rows, preview, dpr, backingFor]);

  // Overlay layer: hover highlight + selection. Cheap, repaints on pointer /
  // selection changes (and re-rasterizes with the scene on zoom/dpr).
  const drawOverlay = React.useCallback(() => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;
    const { effectiveCellPx, contentW, contentH, effectiveDpr } = backingFor();
    if (!(effectiveCellPx > 0)) return;
    if (plan && scene) {
      renderOverlay(overlay, plan, scene, {
        cellPx: effectiveCellPx,
        dpr: effectiveDpr,
        hover: hover?.hit ?? null,
        selectedRoomIndex,
        metersPerCell: plan.meters_per_cell,
      });
    } else {
      clearLayer(overlay, contentW, contentH, effectiveDpr);
    }
  }, [plan, scene, hover, selectedRoomIndex, backingFor]);

  const fit = React.useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const bounds = container.getBoundingClientRect();
    const rightInset = rightPanelOpenRef.current ? PANEL_RIGHT : PANEL_RIGHT_CLOSED;
    const availLeft = PANEL_LEFT + FIT_PADDING;
    const availTop = FIT_PADDING;
    const availWidth = Math.max(80, bounds.width - rightInset - FIT_PADDING - availLeft);
    const availHeight = Math.max(80, bounds.height - FIT_PADDING - availTop);
    // px/cell that lands the plan inside the visible rect, with the same
    // readability floor/ceiling the old CSS-scale fit used (0.2×–8× natural).
    const fitCellPx = Math.max(
      0.2 * dims.scale,
      Math.min(8 * dims.scale, availWidth / dims.cols, availHeight / dims.rows),
    );
    const contentW = dims.cols * fitCellPx;
    const contentH = dims.rows * fitCellPx;
    setView({
      zoom: 1,
      base: fitCellPx,
      x: availLeft + (availWidth - contentW) / 2,
      y: availTop + (availHeight - contentH) / 2,
    });
    onViewChange?.(1);
  }, [dims.cols, dims.rows, dims.scale, onViewChange]);

  const stopInertia = React.useCallback(() => {
    if (inertiaRef.current !== null) {
      cancelAnimationFrame(inertiaRef.current);
      inertiaRef.current = null;
    }
  }, []);

  // Fling the pan on release: carry the last pointer velocity, decaying 0.92 per
  // frame until it drops below 0.5px/frame. Disabled under reduced motion.
  const startInertia = React.useCallback(
    (vx: number, vy: number) => {
      stopInertia();
      if (prefersReducedMotion()) return;
      let velX = vx;
      let velY = vy;
      const step = (): void => {
        velX *= 0.92;
        velY *= 0.92;
        if (Math.hypot(velX, velY) < 0.5) {
          inertiaRef.current = null;
          return;
        }
        setView((current) => ({ ...current, x: current.x + velX, y: current.y + velY }));
        inertiaRef.current = requestAnimationFrame(step);
      };
      inertiaRef.current = requestAnimationFrame(step);
    },
    [stopInertia],
  );

  const zoomAt = React.useCallback(
    (factor: number, clientX?: number, clientY?: number) => {
      const container = containerRef.current;
      if (!container) return;
      stopInertia();
      const bounds = container.getBoundingClientRect();
      const pivotX = clientX === undefined ? bounds.width / 2 : clientX - bounds.left;
      const pivotY = clientY === undefined ? bounds.height / 2 : clientY - bounds.top;
      setView((current) => {
        const zoom = Math.max(0.05, Math.min(12, current.zoom * factor));
        const scaleChange = zoom / current.zoom;
        const next = {
          ...current,
          zoom,
          x: pivotX - (pivotX - current.x) * scaleChange,
          y: pivotY - (pivotY - current.y) * scaleChange,
        };
        onViewChange?.(zoom);
        return next;
      });
    },
    [onViewChange, stopInertia],
  );

  React.useImperativeHandle(
    handleRef,
    () => ({
      zoomBy: (factor: number) => zoomAt(factor),
      fit,
      // Composite the grid + plan layers (never the interaction overlay) into an
      // offscreen canvas so the exported PNG matches the printed drawing exactly.
      exportPng: () => {
        const grid = gridCanvasRef.current;
        const planCanvas = planCanvasRef.current;
        if (!grid) return null;
        const out = document.createElement("canvas");
        out.width = grid.width;
        out.height = grid.height;
        const ctx = out.getContext("2d");
        if (!ctx) return null;
        ctx.drawImage(grid, 0, 0);
        if (planCanvas && plan) ctx.drawImage(planCanvas, 0, 0);
        return out.toDataURL("image/png");
      },
    }),
    [zoomAt, fit, plan],
  );

  // Re-fit when the plan's footprint changes (also fits the default empty grid
  // once on mount). Zoom/pan are otherwise preserved across re-renders.
  React.useEffect(() => {
    const signature = plan ? `${plan.width}x${plan.height}` : `empty-${DEFAULT_COLS}x${DEFAULT_ROWS}`;
    if (planSizeRef.current !== signature) {
      planSizeRef.current = signature;
      fit();
    }
  }, [plan, fit]);

  // Re-raster the grid + plan layers on plan / zoom / scale / preview / dpr
  // change. Panning (view.x / view.y) intentionally does not appear here — it is
  // a CSS translate — and neither does hover, which only repaints the overlay.
  React.useEffect(() => {
    drawScene();
  }, [drawScene, view.base, view.zoom]);

  // Repaint the overlay on hover / selection change, and re-rasterize it when
  // the scene box changes (zoom / dpr).
  React.useEffect(() => {
    drawOverlay();
  }, [drawOverlay, view.base, view.zoom]);

  // Native listener: React wheel events are passive and cannot preventDefault.
  React.useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      // 1.1× per notch (deltaY ≈ ±100), smooth for fractional trackpad deltas,
      // anchored on the cursor so the point under it stays put.
      zoomAt(Math.pow(1.1, -event.deltaY / 100), event.clientX, event.clientY);
    };
    container.addEventListener("wheel", onWheel, { passive: false });
    return () => container.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  // Container resize → redraw (rAF-debounced so a resize drag redraws once/frame).
  React.useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return;
    let raf = 0;
    const observer = new ResizeObserver(() => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        drawScene();
        drawOverlay();
      });
    });
    observer.observe(container);
    return () => {
      observer.disconnect();
      if (raf) cancelAnimationFrame(raf);
    };
  }, [drawScene, drawOverlay]);

  // devicePixelRatio can change (moving the window across displays / OS zoom).
  // The media query is pinned to the current dppx, so re-subscribe when it flips.
  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const query = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    const onChange = () => setDpr(resolveDpr());
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, [dpr]);

  // Window keyboard shortcuts (ignored while typing in a field). Space toggles
  // pan-drag; F fits; +/− zoom about centre; 0 resets to 100%; Esc clears.
  React.useEffect(() => {
    const isTyping = (): boolean => {
      const el = document.activeElement as HTMLElement | null;
      if (!el) return false;
      return el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable;
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTyping()) return;
      if (event.code === "Space") {
        spaceHeldRef.current = true;
        setSpaceHeld(true);
        event.preventDefault();
        return;
      }
      switch (event.key) {
        case "f":
        case "F":
          event.preventDefault();
          fit();
          break;
        case "+":
        case "=":
          event.preventDefault();
          zoomAt(1.1);
          break;
        case "-":
        case "_":
          event.preventDefault();
          zoomAt(1 / 1.1);
          break;
        case "0":
          event.preventDefault();
          zoomAt(1 / viewRef.current.zoom);
          break;
        case "Escape":
          onSelectRoom?.(null);
          break;
        default:
          break;
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        spaceHeldRef.current = false;
        setSpaceHeld(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [fit, zoomAt, onSelectRoom]);

  // Stop any in-flight inertia on unmount.
  React.useEffect(() => stopInertia, [stopInertia]);

  // Push an entity label to the read-out only when it changes, to avoid
  // re-rendering the parent HUD on every pointer move within one entity.
  const emitEntity = React.useCallback(
    (label: string | null) => {
      if (entityLabelRef.current === label) return;
      entityLabelRef.current = label;
      onHoverEntity?.(label);
    },
    [onHoverEntity],
  );

  // A short human label for a hit, for the read-out pill.
  const entityLabel = React.useCallback(
    (hit: Hit): string | null => {
      if (!plan || !scene) return null;
      if (hit.kind === "room") {
        const room = scene.rooms[hit.roomIndex];
        return room ? `${roomDisplayName(room.id)} · ${room.areaFt2} ft²` : null;
      }
      const ftPerCell = plan.meters_per_cell * 3.28084;
      if (hit.kind === "wall") {
        return `wall ${Math.round(segmentLengthCells(hit.wall) * ftPerCell)} ft`;
      }
      return `door ${(hit.door.span * plan.meters_per_cell).toFixed(1)} m`;
    },
    [plan, scene],
  );

  const updateHover = React.useCallback(
    (clientX: number, clientY: number) => {
      const container = containerRef.current;
      if (!container) return;
      const bounds = container.getBoundingClientRect();
      const v = viewRef.current;
      const effectiveCellPx = v.base * v.zoom;
      if (!(effectiveCellPx > 0)) return;
      const localX = clientX - bounds.left;
      const localY = clientY - bounds.top;
      const cellX = (localX - v.x) / effectiveCellPx;
      const cellY = (localY - v.y) / effectiveCellPx;
      const cx = Math.floor(cellX);
      const cy = Math.floor(cellY);
      const inGrid = cx >= 0 && cy >= 0 && cx < dims.cols && cy < dims.rows;
      onHoverCell?.(inGrid ? { x: cx, y: cy } : null);
      setPointer({ x: localX, y: localY });

      const hit = plan && scene ? scene.hitTest({ x: cellX, y: cellY }, HIT_TOLERANCE_PX, effectiveCellPx) : null;
      if (!hit) {
        if (hoverKeyRef.current !== "") {
          hoverKeyRef.current = "";
          setHover(null);
        }
        emitEntity(null);
        return;
      }
      // Only touch the overlay-driving state when the entity actually changes.
      const key = hitKey(hit);
      if (key !== hoverKeyRef.current) {
        hoverKeyRef.current = key;
        const room = hit.kind === "room" && plan ? plan.rooms[hit.roomIndex] ?? null : null;
        setHover({ hit, room });
      }
      emitEntity(entityLabel(hit));
    },
    [dims.cols, dims.rows, plan, scene, onHoverCell, emitEntity, entityLabel],
  );

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 && event.button !== 1) return;
    stopInertia();
    // Middle-drag always pans; left-drag pans only while Space is held, else it
    // is a candidate click-select.
    const panning = event.button === 1 || (event.button === 0 && spaceHeldRef.current);
    const current = viewRef.current;
    const now = performance.now();
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: current.x,
      originY: current.y,
      panning,
      moved: false,
      lastX: event.clientX,
      lastY: event.clientY,
      lastT: now,
      vx: 0,
      vy: 0,
    };
    if (panning) setPanningActive(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (drag && drag.pointerId === event.pointerId) {
      const dx = event.clientX - drag.startX;
      const dy = event.clientY - drag.startY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) drag.moved = true;
      if (drag.panning) {
        const now = performance.now();
        const dt = Math.max(1, now - drag.lastT);
        // px per ~frame (16ms), for the release fling.
        drag.vx = ((event.clientX - drag.lastX) / dt) * 16;
        drag.vy = ((event.clientY - drag.lastY) / dt) * 16;
        drag.lastX = event.clientX;
        drag.lastY = event.clientY;
        drag.lastT = now;
        setView((current) => ({ ...current, x: drag.originX + dx, y: drag.originY + dy }));
        return;
      }
    }
    updateHover(event.clientX, event.clientY);
  };

  const selectAt = React.useCallback(
    (clientX: number, clientY: number) => {
      if (!plan || !onSelectRoom) return;
      const container = containerRef.current;
      if (!container) return;
      const bounds = container.getBoundingClientRect();
      const v = viewRef.current;
      const effectiveCellPx = v.base * v.zoom;
      if (!(effectiveCellPx > 0)) return;
      // Selection is room-only: resolve the cell directly (ignoring wall/door
      // pick priority) so clicking anywhere inside a room selects it.
      const cx = Math.floor((clientX - bounds.left - v.x) / effectiveCellPx);
      const cy = Math.floor((clientY - bounds.top - v.y) / effectiveCellPx);
      let roomId: string | null = null;
      if (cx >= 0 && cy >= 0 && cx < plan.width && cy < plan.height) {
        const idx = plan.cells[cy * plan.width + cx];
        if (idx !== undefined && idx >= 0) roomId = plan.rooms[idx]?.id ?? null;
      }
      onSelectRoom(roomId);
    },
    [plan, onSelectRoom],
  );

  const onPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    if (drag.panning) {
      setPanningActive(false);
      // Ignore stale velocity: if the pointer paused before release, don't fling.
      const fresh = performance.now() - drag.lastT < 60;
      startInertia(fresh ? drag.vx : 0, fresh ? drag.vy : 0);
    } else if (!drag.moved && event.button === 0) {
      selectAt(event.clientX, event.clientY);
    }
  };

  const onPointerLeave = () => {
    hoverKeyRef.current = "";
    setHover(null);
    setPointer(null);
    onHoverCell?.(null);
    emitEntity(null);
  };

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative h-full w-full touch-none overflow-hidden bg-canvas",
        panningActive
          ? "cursor-grabbing"
          : spaceHeld
            ? "cursor-grab"
            : hoveredRoomIndex >= 0
              ? "cursor-pointer"
              : "cursor-default",
      )}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onPointerLeave={onPointerLeave}
    >
      <div
        className="absolute left-0 top-0"
        style={{
          width: contentSize.width,
          height: contentSize.height,
          // Translate only — the plan is rasterized at its final size, so there
          // is no CSS scale to resample and blur it.
          transform: `translate(${view.x}px, ${view.y}px)`,
        }}
      >
        {/* Three stacked layers; pointer events are handled on the container. */}
        <canvas
          ref={gridCanvasRef}
          className={cn("absolute left-0 top-0 block", plan && "shadow-[0_4px_24px_rgba(16,24,40,0.10)]")}
        />
        <canvas ref={planCanvasRef} className="pointer-events-none absolute left-0 top-0 block" />
        {plan && reference && showReference ? (
          /* Stretched to the grid box on purpose: this is the same mapping
             the transcription uses, so rooms should land on themselves. */
          <img
            src={candidateDataUrl(reference)}
            alt="Reference drawing overlay"
            className="pointer-events-none absolute left-0 top-0 h-full w-full"
            style={{ opacity: referenceOpacity / 100 }}
          />
        ) : null}
        <canvas ref={overlayCanvasRef} className="pointer-events-none absolute left-0 top-0 block" />
      </div>

      {hover?.room && pointer ? (
        <div
          className="pointer-events-none absolute z-20 flex items-center gap-2 rounded-md border border-border bg-card/90 px-2 py-1 text-[11px] text-foreground shadow-lg backdrop-blur-md"
          style={{ left: pointer.x + 14, top: pointer.y + 14 }}
        >
          <span
            className="size-2.5 shrink-0 rounded-[3px] border border-black/30"
            style={{ background: hover.room.color }}
            aria-hidden
          />
          <span className="font-medium capitalize">{prettyType(hover.room.type)}</span>
          {plan ? (
            <span className="font-mono tabular-nums text-muted-foreground">
              {Math.round(hover.room.pixel_count * plan.meters_per_cell * plan.meters_per_cell * 10.7639)} ft²
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
});
