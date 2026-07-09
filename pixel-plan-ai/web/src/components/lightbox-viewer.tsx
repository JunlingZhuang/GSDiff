"use client";

import * as React from "react";
import { Check, ChevronLeft, ChevronRight, Maximize, Minus, Plus, XIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogTitle } from "@/components/ui/dialog";
import type { CandidateImage } from "@/lib/types";
import { candidateDataUrl } from "@/lib/types";
import { cn } from "@/lib/utils";

const MIN_SCALE = 0.5;
const MAX_SCALE = 6;
const WHEEL_STEP = 1.15;
const DOUBLE_TAP_SCALE = 2.5;

interface View {
  scale: number;
  x: number;
  y: number;
}

const FIT: View = { scale: 1, x: 0, y: 0 };

export interface LightboxViewerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  image: CandidateImage | null;
  // Candidate-strip context (omit all for the single-image artifact viewer):
  index?: number;
  count?: number;
  selectedIndex?: number;
  onIndex?: (index: number) => void;
  onSelect?: (index: number) => void;
}

// Large, zoomable image inspector shared by the candidate strip and the trace
// linework artifact. Scale 1 is the contain-fitted baseline; wheel zooms at the
// cursor, drag pans, double-click toggles 1x <-> 2.5x, arrows flip candidates.
export function LightboxViewer({
  open,
  onOpenChange,
  title,
  image,
  index,
  count,
  selectedIndex,
  onIndex,
  onSelect,
}: LightboxViewerProps) {
  const stageRef = React.useRef<HTMLDivElement>(null);
  const [view, setView] = React.useState<View>(FIT);
  const [panning, setPanning] = React.useState(false);
  const dragRef = React.useRef<{ pointerId: number; startX: number; startY: number; originX: number; originY: number } | null>(
    null,
  );

  const navigable = typeof index === "number" && typeof count === "number" && count > 1 && !!onIndex;
  const isSelected = typeof index === "number" && index === selectedIndex;
  const canSelect = typeof index === "number" && !!onSelect && index !== selectedIndex;

  const reset = React.useCallback(() => setView(FIT), []);

  // Snap back to the fitted baseline whenever the viewer opens or the image
  // changes — done during render (React's "adjust state on prop change") with a
  // ref guard so a stale zoom never paints for the new image.
  const resetKeyRef = React.useRef<string | null>(null);
  const resetKey = open ? `${index ?? "single"}:${image?.mime ?? ""}:${image?.data.length ?? 0}` : null;
  if (resetKeyRef.current !== resetKey) {
    resetKeyRef.current = resetKey;
    if (view.scale !== 1 || view.x !== 0 || view.y !== 0) setView(FIT);
  }

  // Zoom about a screen point (cursor), keeping that point under the cursor.
  // Coordinates are taken relative to the stage centre (the transform origin).
  const zoomAbout = React.useCallback((factor: number, clientX?: number, clientY?: number) => {
    const stage = stageRef.current;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const pivotX = (clientX ?? rect.left + rect.width / 2) - rect.left - rect.width / 2;
    const pivotY = (clientY ?? rect.top + rect.height / 2) - rect.top - rect.height / 2;
    setView((current) => {
      const scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, current.scale * factor));
      const k = scale / current.scale;
      return {
        scale,
        x: pivotX - k * (pivotX - current.x),
        y: pivotY - k * (pivotY - current.y),
      };
    });
  }, []);

  // Native wheel listener: React wheel events are passive and cannot preventDefault.
  React.useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !open) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      zoomAbout(Math.pow(WHEEL_STEP, -event.deltaY / 100), event.clientX, event.clientY);
    };
    stage.addEventListener("wheel", onWheel, { passive: false });
    return () => stage.removeEventListener("wheel", onWheel);
  }, [open, zoomAbout]);

  // Arrow keys flip between candidates (candidate context only). Esc is the
  // Dialog's own default and is left untouched.
  React.useEffect(() => {
    if (!open || !navigable || typeof index !== "number" || typeof count !== "number" || !onIndex) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        onIndex((index - 1 + count) % count);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        onIndex((index + 1) % count);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, navigable, index, count, onIndex]);

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: view.x,
      originY: view.y,
    };
    setPanning(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const nextX = drag.originX + (event.clientX - drag.startX);
    const nextY = drag.originY + (event.clientY - drag.startY);
    setView((current) => ({ ...current, x: nextX, y: nextY }));
  };

  const onPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    setPanning(false);
  };

  const onDoubleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (view.scale > 1.01) {
      reset();
    } else {
      zoomAbout(DOUBLE_TAP_SCALE / view.scale, event.clientX, event.clientY);
    }
  };

  const step = (delta: -1 | 1) => {
    if (!navigable || typeof index !== "number" || typeof count !== "number" || !onIndex) return;
    onIndex((index + delta + count) % count);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        showCloseButton={false}
        className="flex h-[88vh] w-[92vw] max-w-none flex-col gap-0 overflow-hidden p-0"
      >
        {/* header */}
        <div className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2">
          <DialogTitle className="text-[13px] font-medium">{title}</DialogTitle>
          {isSelected ? (
            <span className="inline-flex items-center gap-1 rounded-md border border-primary/40 bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
              <Check className="size-2.5" /> selected
            </span>
          ) : null}
          <div className="ml-auto flex items-center gap-2">
            {canSelect ? (
              <Button
                size="sm"
                className="h-7 gap-1.5 text-[11px] font-medium"
                onClick={() => {
                  if (typeof index === "number") onSelect?.(index);
                }}
              >
                <Check className="size-3" /> Use this drawing
              </Button>
            ) : null}
            <DialogClose render={<Button variant="ghost" size="icon-sm" className="size-7" />}>
              <XIcon />
              <span className="sr-only">Close</span>
            </DialogClose>
          </div>
        </div>

        {/* stage */}
        <div
          ref={stageRef}
          className={cn(
            "relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-[#F2F4F7]",
            panning ? "cursor-grabbing" : "cursor-grab",
          )}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onDoubleClick={onDoubleClick}
        >
          {image ? (
            <img
              src={candidateDataUrl(image)}
              alt={title}
              draggable={false}
              className="pointer-events-none h-full w-full select-none object-contain"
              style={{
                transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
                transformOrigin: "center center",
                willChange: "transform",
              }}
            />
          ) : null}

          {/* candidate flip chevrons */}
          {navigable ? (
            <div className="pointer-events-none absolute inset-0" onPointerDown={(event) => event.stopPropagation()}>
              <Button
                size="icon"
                variant="ghost"
                className="pointer-events-auto absolute left-3 top-1/2 size-9 -translate-y-1/2 rounded-full border border-border bg-card/90 shadow-lg backdrop-blur-md"
                onClick={() => step(-1)}
                aria-label="Previous candidate"
              >
                <ChevronLeft className="size-4" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                className="pointer-events-auto absolute right-3 top-1/2 size-9 -translate-y-1/2 rounded-full border border-border bg-card/90 shadow-lg backdrop-blur-md"
                onClick={() => step(1)}
                aria-label="Next candidate"
              >
                <ChevronRight className="size-4" />
              </Button>
            </div>
          ) : null}

          {/* zoom cluster — matches the viewport chrome */}
          <div
            className="absolute bottom-3 right-3 flex items-center gap-0.5 rounded-lg border border-border bg-card/90 p-0.5 shadow-lg backdrop-blur-md"
            onPointerDown={(event) => event.stopPropagation()}
            onDoubleClick={(event) => event.stopPropagation()}
          >
            <Button size="icon" variant="ghost" className="size-7" onClick={() => zoomAbout(1 / WHEEL_STEP)} aria-label="Zoom out">
              <Minus className="size-3.5" />
            </Button>
            <span className="w-12 text-center font-mono text-[11px] tabular-nums text-muted-foreground">
              {Math.round(view.scale * 100)}%
            </span>
            <Button size="icon" variant="ghost" className="size-7" onClick={() => zoomAbout(WHEEL_STEP)} aria-label="Zoom in">
              <Plus className="size-3.5" />
            </Button>
            <Button size="icon" variant="ghost" className="size-7" onClick={reset} aria-label="Fit">
              <Maximize className="size-3.5" />
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
