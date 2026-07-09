"use client";

import * as React from "react";

import { cellScale, clearLayer, planCellScale, renderEmptyGrid, renderOverlay, renderPlanLayer } from "@/lib/render";
import { buildScene } from "@/lib/scene";
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

function resolveDpr(): number {
  if (typeof window === "undefined") return 1;
  return Math.min(MAX_DPR, window.devicePixelRatio || 1);
}

interface Viewport2DProps {
  plan: Plan | null;
  reference: CandidateImage | null;
  showReference: boolean;
  referenceOpacity: number;
  preview?: boolean;
  rightPanelOpen?: boolean;
  onViewChange?: (zoom: number) => void;
  onHoverCell?: (cell: { x: number; y: number } | null) => void;
}

export interface Viewport2DHandle {
  zoomBy: (factor: number) => void;
  fit: () => void;
  exportPng: () => string | null;
}

interface HoverState {
  cx: number;
  cy: number;
  px: number;
  py: number;
  roomIndex: number;
  room: PlanRoom | null;
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
  { plan, reference, showReference, referenceOpacity, preview = false, rightPanelOpen = false, onViewChange, onHoverCell },
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
  const dragRef = React.useRef<{ pointerId: number; startX: number; startY: number; originX: number; originY: number } | null>(null);
  const [hover, setHover] = React.useState<HoverState | null>(null);
  const hoveredRoomIndex = hover?.roomIndex ?? -1;
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
      const hover: Hit | null = hoveredRoomIndex >= 0 ? { kind: "room", roomIndex: hoveredRoomIndex } : null;
      renderOverlay(overlay, plan, scene, {
        cellPx: effectiveCellPx,
        dpr: effectiveDpr,
        hover,
        selectedRoomIndex: -1,
        metersPerCell: plan.meters_per_cell,
      });
    } else {
      clearLayer(overlay, contentW, contentH, effectiveDpr);
    }
  }, [plan, scene, hoveredRoomIndex, backingFor]);

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

  const zoomAt = React.useCallback(
    (factor: number, clientX?: number, clientY?: number) => {
      const container = containerRef.current;
      if (!container) return;
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
    [onViewChange],
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
      zoomAt(Math.exp(-event.deltaY * 0.0016), event.clientX, event.clientY);
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
      const cx = Math.floor((localX - v.x) / effectiveCellPx);
      const cy = Math.floor((localY - v.y) / effectiveCellPx);
      if (cx < 0 || cy < 0 || cx >= dims.cols || cy >= dims.rows) {
        setHover(null);
        onHoverCell?.(null);
        return;
      }
      const rawIndex = plan ? plan.cells[cy * dims.cols + cx] : -1;
      const roomIndex = rawIndex !== undefined && rawIndex >= 0 ? rawIndex : -1;
      const room = plan && roomIndex >= 0 ? plan.rooms[roomIndex] ?? null : null;
      setHover({ cx, cy, px: localX, py: localY, roomIndex: room ? roomIndex : -1, room });
      onHoverCell?.({ x: cx, y: cy });
    },
    [dims.cols, dims.rows, plan, onHoverCell],
  );

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 && event.button !== 1) return;
    const current = viewRef.current;
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: current.x,
      originY: current.y,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (drag && drag.pointerId === event.pointerId) {
      setView((current) => ({
        ...current,
        x: drag.originX + event.clientX - drag.startX,
        y: drag.originY + event.clientY - drag.startY,
      }));
    }
    updateHover(event.clientX, event.clientY);
  };

  const onPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  };

  const onPointerLeave = () => {
    setHover(null);
    onHoverCell?.(null);
  };

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative h-full w-full touch-none overflow-hidden bg-canvas",
        hover?.room ? "cursor-pointer" : "cursor-grab active:cursor-grabbing",
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

      {hover?.room ? (
        <div
          className="pointer-events-none absolute z-20 flex items-center gap-2 rounded-md border border-border bg-card/90 px-2 py-1 text-[11px] text-foreground shadow-lg backdrop-blur-md"
          style={{ left: hover.px + 14, top: hover.py + 14 }}
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
