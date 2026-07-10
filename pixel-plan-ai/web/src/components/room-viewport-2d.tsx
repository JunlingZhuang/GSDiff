"use client";

import * as React from "react";

import { ICU_CATALOG } from "@/lib/icu-catalog";
import { renderEmptyRoom, renderRoom, type SymbolCache } from "@/lib/render-room";
import type { RoomLayout } from "@/lib/types";
import { cn } from "@/lib/utils";

const DEFAULT_WIDTH_FT = 16.5;
const DEFAULT_DEPTH_FT = 16.75;

// Fit reserves the floating-panel gutters (left 300 + 12, right 320 + 12) so the
// room lands centered in the visible drawing area, matching the floor viewport.
const PANEL_LEFT = 324;
const PANEL_RIGHT = 344;
const PANEL_RIGHT_CLOSED = 12;
const FIT_PADDING = 48;

const MAX_BACKING_PX = 8192;
const MAX_DPR = 2.5;

function resolveDpr(): number {
  if (typeof window === "undefined") return 1;
  return Math.min(MAX_DPR, window.devicePixelRatio || 1);
}

interface RoomViewport2DProps {
  layout: RoomLayout | null;
  showClearances: boolean;
  preview?: boolean;
  rightPanelOpen?: boolean;
  selectedAssetId?: string | null;
  onSelectAsset?: (id: string | null) => void;
  onViewChange?: (zoom: number) => void;
}

export interface RoomViewport2DHandle {
  zoomBy: (factor: number) => void;
  fit: () => void;
  exportPng: () => string | null;
}

interface View {
  zoom: number;
  x: number;
  y: number;
  base: number;
}

export const RoomViewport2D = React.forwardRef<RoomViewport2DHandle, RoomViewport2DProps>(
  function RoomViewport2D(
    { layout, showClearances, preview = false, rightPanelOpen = false, selectedAssetId = null, onSelectAsset, onViewChange },
    handleRef,
  ) {
    const containerRef = React.useRef<HTMLDivElement>(null);
    const canvasRef = React.useRef<HTMLCanvasElement>(null);
    const [view, setView] = React.useState<View>({ zoom: 1, x: 0, y: 0, base: 24 });
    const viewRef = React.useRef(view);
    viewRef.current = view;
    const [dpr, setDpr] = React.useState(resolveDpr);
    const sizeSignatureRef = React.useRef<string>("");

    const widthFt = layout?.room.width_ft ?? DEFAULT_WIDTH_FT;
    const depthFt = layout?.room.depth_ft ?? DEFAULT_DEPTH_FT;

    // Symbol image cache — loaded once; a load bumps `symbolsReady` to redraw.
    const symbolsRef = React.useRef<SymbolCache>(new Map());
    const [symbolsReady, setSymbolsReady] = React.useState(0);
    React.useEffect(() => {
      let active = true;
      for (const entry of ICU_CATALOG) {
        if (symbolsRef.current.has(entry.type)) continue;
        const image = new Image();
        image.onload = () => {
          if (active) setSymbolsReady((value) => value + 1);
        };
        image.src = entry.symbol;
        symbolsRef.current.set(entry.type, image);
      }
      return () => {
        active = false;
      };
    }, []);

    const backingFor = React.useCallback(() => {
      const v = viewRef.current;
      const pxPerFt = v.base * v.zoom;
      const contentW = widthFt * pxPerFt;
      const contentH = depthFt * pxPerFt;
      const capScale = Math.min(1, MAX_BACKING_PX / (contentW * dpr), MAX_BACKING_PX / (contentH * dpr));
      return { pxPerFt, contentW, contentH, effectiveDpr: dpr * capScale };
    }, [widthFt, depthFt, dpr]);

    const draw = React.useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const { pxPerFt, effectiveDpr } = backingFor();
      if (!(pxPerFt > 0)) return;
      if (layout) {
        renderRoom(canvas, layout, {
          pxPerFt,
          dpr: effectiveDpr,
          showClearances,
          symbols: symbolsRef.current,
          selectedAssetId,
        });
      } else {
        renderEmptyRoom(canvas, widthFt, depthFt, { pxPerFt, dpr: effectiveDpr });
      }
    }, [layout, showClearances, selectedAssetId, widthFt, depthFt, backingFor]);

    const fit = React.useCallback(() => {
      const container = containerRef.current;
      if (!container) return;
      const bounds = container.getBoundingClientRect();
      const rightInset = rightPanelOpen ? PANEL_RIGHT : PANEL_RIGHT_CLOSED;
      const availLeft = PANEL_LEFT + FIT_PADDING;
      const availTop = FIT_PADDING;
      const availWidth = Math.max(80, bounds.width - rightInset - FIT_PADDING - availLeft);
      const availHeight = Math.max(80, bounds.height - FIT_PADDING - availTop);
      const base = Math.max(6, Math.min(64, availWidth / widthFt, availHeight / depthFt));
      const contentW = widthFt * base;
      const contentH = depthFt * base;
      setView({
        zoom: 1,
        base,
        x: availLeft + (availWidth - contentW) / 2,
        y: availTop + (availHeight - contentH) / 2,
      });
      onViewChange?.(1);
    }, [widthFt, depthFt, rightPanelOpen, onViewChange]);

    const zoomAt = React.useCallback(
      (factor: number, clientX?: number, clientY?: number) => {
        const container = containerRef.current;
        if (!container) return;
        const bounds = container.getBoundingClientRect();
        const pivotX = clientX === undefined ? bounds.width / 2 : clientX - bounds.left;
        const pivotY = clientY === undefined ? bounds.height / 2 : clientY - bounds.top;
        setView((current) => {
          const zoom = Math.max(0.1, Math.min(12, current.zoom * factor));
          const scaleChange = zoom / current.zoom;
          onViewChange?.(zoom);
          return {
            ...current,
            zoom,
            x: pivotX - (pivotX - current.x) * scaleChange,
            y: pivotY - (pivotY - current.y) * scaleChange,
          };
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

    // Re-fit when the room footprint changes (also fits the default grid once).
    React.useEffect(() => {
      const signature = `${widthFt}x${depthFt}`;
      if (sizeSignatureRef.current !== signature) {
        sizeSignatureRef.current = signature;
        fit();
      }
    }, [widthFt, depthFt, fit]);

    // Re-raster on zoom / dpr / content / toggle / symbol-load. Panning is a CSS
    // translate (view.x / view.y are intentionally absent here).
    React.useEffect(() => {
      draw();
    }, [draw, view.base, view.zoom, dpr, symbolsReady]);

    // Native wheel listener (React wheel is passive and cannot preventDefault).
    React.useEffect(() => {
      const container = containerRef.current;
      if (!container) return;
      const onWheel = (event: WheelEvent) => {
        event.preventDefault();
        zoomAt(Math.pow(1.1, -event.deltaY / 100), event.clientX, event.clientY);
      };
      container.addEventListener("wheel", onWheel, { passive: false });
      return () => container.removeEventListener("wheel", onWheel);
    }, [zoomAt]);

    // Container resize → redraw (rAF-debounced).
    React.useEffect(() => {
      const container = containerRef.current;
      if (!container || typeof ResizeObserver === "undefined") return;
      let raf = 0;
      const observer = new ResizeObserver(() => {
        if (raf) return;
        raf = requestAnimationFrame(() => {
          raf = 0;
          draw();
        });
      });
      observer.observe(container);
      return () => {
        observer.disconnect();
        if (raf) cancelAnimationFrame(raf);
      };
    }, [draw]);

    React.useEffect(() => {
      if (typeof window === "undefined") return;
      const query = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
      const onChange = () => setDpr(resolveDpr());
      query.addEventListener("change", onChange);
      return () => query.removeEventListener("change", onChange);
    }, [dpr]);

    // Pan (drag) + click-select. A drag beyond the threshold pans; a clean click
    // selects the topmost asset under the cursor.
    const dragRef = React.useRef<{
      pointerId: number;
      startX: number;
      startY: number;
      originX: number;
      originY: number;
      moved: boolean;
    } | null>(null);
    const [panning, setPanning] = React.useState(false);

    const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0 && event.button !== 1) return;
      const current = viewRef.current;
      dragRef.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        originX: current.x,
        originY: current.y,
        moved: false,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
    };

    const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
      const drag = dragRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;
      const dx = event.clientX - drag.startX;
      const dy = event.clientY - drag.startY;
      if (!drag.moved && (Math.abs(dx) > 3 || Math.abs(dy) > 3)) {
        drag.moved = true;
        setPanning(true);
      }
      if (drag.moved) setView((c) => ({ ...c, x: drag.originX + dx, y: drag.originY + dy }));
    };

    const selectAt = React.useCallback(
      (clientX: number, clientY: number) => {
        if (!layout || !onSelectAsset) return;
        const container = containerRef.current;
        if (!container) return;
        const bounds = container.getBoundingClientRect();
        const v = viewRef.current;
        const pxPerFt = v.base * v.zoom;
        if (!(pxPerFt > 0)) return;
        const roomX = (clientX - bounds.left - v.x) / pxPerFt;
        const roomY = depthFt - (clientY - bounds.top - v.y) / pxPerFt;
        // Topmost first (assets are painted in array order).
        for (let i = layout.assets.length - 1; i >= 0; i -= 1) {
          const a = layout.assets[i];
          if (roomX >= a.x_ft && roomX <= a.x_ft + a.w_ft && roomY >= a.y_ft && roomY <= a.y_ft + a.d_ft) {
            onSelectAsset(a.id === selectedAssetId ? null : a.id);
            return;
          }
        }
        onSelectAsset(null);
      },
      [layout, onSelectAsset, depthFt, selectedAssetId],
    );

    const onPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
      const drag = dragRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;
      dragRef.current = null;
      if (drag.moved) {
        setPanning(false);
      } else if (event.button === 0) {
        selectAt(event.clientX, event.clientY);
      }
    };

    const contentSize = { width: widthFt * view.base * view.zoom, height: depthFt * view.base * view.zoom };

    return (
      <div
        ref={containerRef}
        className={cn("relative h-full w-full touch-none overflow-hidden bg-canvas", panning ? "cursor-grabbing" : "cursor-default")}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <div
          className="absolute left-0 top-0"
          style={{
            width: contentSize.width,
            height: contentSize.height,
            transform: `translate(${view.x}px, ${view.y}px)`,
          }}
        >
          <canvas
            ref={canvasRef}
            className={cn("absolute left-0 top-0 block", layout && "shadow-[0_4px_24px_rgba(16,24,40,0.10)]")}
          />
        </div>
      </div>
    );
  },
);
