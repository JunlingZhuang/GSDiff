"use client";

import * as React from "react";

import { cellScale, planCellScale, renderEmptyGrid, renderPlanToCanvas } from "@/lib/render";
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

export const Viewport2D = React.forwardRef<Viewport2DHandle, Viewport2DProps>(function Viewport2D(
  { plan, reference, showReference, referenceOpacity, preview = false, rightPanelOpen = false, onViewChange, onHoverCell },
  handleRef,
) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [view, setView] = React.useState({ zoom: 1, x: 0, y: 0 });
  const viewRef = React.useRef(view);
  viewRef.current = view;
  const planSizeRef = React.useRef<string>("");
  // The fitted scale is the 100% baseline; the zoom cluster multiplies on top.
  const baseZoomRef = React.useRef(1);
  const rightPanelOpenRef = React.useRef(rightPanelOpen);
  rightPanelOpenRef.current = rightPanelOpen;
  const dragRef = React.useRef<{ pointerId: number; startX: number; startY: number; originX: number; originY: number } | null>(null);
  const [hover, setHover] = React.useState<HoverState | null>(null);
  const hoveredRoomIndex = hover?.roomIndex ?? -1;

  const dims = React.useMemo(() => {
    const cols = plan?.width ?? DEFAULT_COLS;
    const rows = plan?.height ?? DEFAULT_ROWS;
    const scale = plan ? planCellScale(plan) : cellScale(cols);
    return { cols, rows, scale };
  }, [plan]);

  const contentSize = { width: dims.cols * dims.scale, height: dims.rows * dims.scale };

  const fit = React.useCallback(() => {
    const container = containerRef.current;
    if (!container || !contentSize.width) return;
    const bounds = container.getBoundingClientRect();
    const rightInset = rightPanelOpenRef.current ? PANEL_RIGHT : PANEL_RIGHT_CLOSED;
    const availLeft = PANEL_LEFT + FIT_PADDING;
    const availTop = FIT_PADDING;
    const availWidth = Math.max(80, bounds.width - rightInset - FIT_PADDING - availLeft);
    const availHeight = Math.max(80, bounds.height - FIT_PADDING - availTop);
    // Fit the plan into the visible rect; keep a readability floor on the base.
    const base = Math.max(
      0.2,
      Math.min(8, availWidth / contentSize.width, availHeight / contentSize.height),
    );
    baseZoomRef.current = base;
    const next = {
      zoom: base,
      x: availLeft + (availWidth - contentSize.width * base) / 2,
      y: availTop + (availHeight - contentSize.height * base) / 2,
    };
    setView(next);
    onViewChange?.(1);
  }, [contentSize.width, contentSize.height, onViewChange]);

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
          zoom,
          x: pivotX - (pivotX - current.x) * scaleChange,
          y: pivotY - (pivotY - current.y) * scaleChange,
        };
        onViewChange?.(zoom / (baseZoomRef.current || 1));
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
      exportPng: () => canvasRef.current?.toDataURL("image/png") ?? null,
    }),
    [zoomAt, fit],
  );

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (plan) renderPlanToCanvas(canvas, plan, { hoveredRoomIndex, preview });
    else renderEmptyGrid(canvas, DEFAULT_COLS, DEFAULT_ROWS);
    const signature = plan ? `${plan.width}x${plan.height}` : `empty-${DEFAULT_COLS}x${DEFAULT_ROWS}`;
    if (planSizeRef.current !== signature) {
      planSizeRef.current = signature;
      fit();
    }
  }, [plan, fit, hoveredRoomIndex, preview]);

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

  const updateHover = React.useCallback(
    (clientX: number, clientY: number) => {
      const container = containerRef.current;
      if (!container) return;
      const bounds = container.getBoundingClientRect();
      const v = viewRef.current;
      const localX = clientX - bounds.left;
      const localY = clientY - bounds.top;
      const cx = Math.floor((localX - v.x) / (dims.scale * v.zoom));
      const cy = Math.floor((localY - v.y) / (dims.scale * v.zoom));
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
    [dims.scale, dims.cols, dims.rows, plan, onHoverCell],
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
        "relative h-full w-full touch-none overflow-hidden bg-[#101318]",
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
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
          transformOrigin: "0 0",
        }}
      >
        <canvas ref={canvasRef} className={plan ? "block shadow-[0_2px_28px_rgba(0,0,0,0.45)]" : "block"} />
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
