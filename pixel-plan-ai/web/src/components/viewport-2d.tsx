"use client";

import * as React from "react";

import { planCellScale, renderPlanToCanvas } from "@/lib/render";
import type { CandidateImage, Plan } from "@/lib/types";
import { candidateDataUrl } from "@/lib/types";

interface Viewport2DProps {
  plan: Plan | null;
  reference: CandidateImage | null;
  showReference: boolean;
  referenceOpacity: number;
  onViewChange?: (zoom: number) => void;
}

export interface Viewport2DHandle {
  zoomBy: (factor: number) => void;
  fit: () => void;
  exportPng: () => string | null;
}

export const Viewport2D = React.forwardRef<Viewport2DHandle, Viewport2DProps>(function Viewport2D(
  { plan, reference, showReference, referenceOpacity, onViewChange },
  handleRef,
) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [view, setView] = React.useState({ zoom: 1, x: 0, y: 0 });
  const viewRef = React.useRef(view);
  viewRef.current = view;
  const planSizeRef = React.useRef<string>("");
  const dragRef = React.useRef<{ pointerId: number; startX: number; startY: number; originX: number; originY: number } | null>(null);

  const contentSize = React.useMemo(() => {
    if (!plan) return { width: 0, height: 0 };
    const scale = planCellScale(plan);
    return { width: plan.width * scale, height: plan.height * scale };
  }, [plan]);

  const fit = React.useCallback(() => {
    const container = containerRef.current;
    if (!container || !contentSize.width) return;
    const bounds = container.getBoundingClientRect();
    const zoom = Math.min(bounds.width / contentSize.width, bounds.height / contentSize.height) * 0.9;
    const next = {
      zoom,
      x: (bounds.width - contentSize.width * zoom) / 2,
      y: (bounds.height - contentSize.height * zoom) / 2,
    };
    setView(next);
    onViewChange?.(next.zoom);
  }, [contentSize, onViewChange]);

  const zoomAt = React.useCallback(
    (factor: number, clientX?: number, clientY?: number) => {
      const container = containerRef.current;
      if (!container) return;
      const bounds = container.getBoundingClientRect();
      const pivotX = clientX === undefined ? bounds.width / 2 : clientX - bounds.left;
      const pivotY = clientY === undefined ? bounds.height / 2 : clientY - bounds.top;
      setView((current) => {
        const zoom = Math.max(0.15, Math.min(6, current.zoom * factor));
        const scaleChange = zoom / current.zoom;
        const next = {
          zoom,
          x: pivotX - (pivotX - current.x) * scaleChange,
          y: pivotY - (pivotY - current.y) * scaleChange,
        };
        onViewChange?.(next.zoom);
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
    if (!plan || !canvasRef.current) return;
    renderPlanToCanvas(canvasRef.current, plan);
    const signature = `${plan.width}x${plan.height}`;
    if (planSizeRef.current !== signature) {
      planSizeRef.current = signature;
      fit();
    }
  }, [plan, fit]);

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
    if (!drag || drag.pointerId !== event.pointerId) return;
    setView((current) => ({
      ...current,
      x: drag.originX + event.clientX - drag.startX,
      y: drag.originY + event.clientY - drag.startY,
    }));
  };

  const onPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  };

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full cursor-grab touch-none overflow-hidden active:cursor-grabbing"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      {plan ? (
        <div
          className="absolute left-0 top-0"
          style={{
            width: contentSize.width,
            height: contentSize.height,
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
            transformOrigin: "0 0",
          }}
        >
          <canvas ref={canvasRef} className="block shadow-[0_2px_18px_rgba(31,36,42,0.14)]" />
          {reference && showReference ? (
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
      ) : null}
    </div>
  );
});
