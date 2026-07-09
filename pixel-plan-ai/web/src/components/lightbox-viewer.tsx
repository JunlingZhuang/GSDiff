"use client";

import * as React from "react";
import { Check, ChevronLeft, ChevronRight, Maximize, Minus, Plus, XIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogTitle } from "@/components/ui/dialog";
import type { CandidateImage } from "@/lib/types";
import { candidateDataUrl } from "@/lib/types";
import { cn } from "@/lib/utils";

const MIN_SCALE = 0.5;
const MAX_SCALE = 8;
const WHEEL_STEP = 1.15;
const DOUBLE_TAP_SCALE = 2.5;
// Per-frame easing toward the target transform — high enough to feel instant,
// low enough that a wheel flick glides instead of stepping.
const LERP = 0.35;

interface View {
  scale: number;
  x: number;
  y: number;
}

const FIT: View = { scale: 1, x: 0, y: 0 };

export interface LightboxImage {
  image: CandidateImage;
  title: string;
}

export interface LightboxViewerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  // Single / candidate-strip API (controlled by the parent):
  title?: string;
  image?: CandidateImage | null;
  index?: number;
  count?: number;
  selectedIndex?: number;
  onIndex?: (index: number) => void;
  onSelect?: (index: number) => void;
  // Image-set API (self-contained flippable list, e.g. drawing + trace linework).
  // When provided it takes precedence over `image`/`title`; flipping is internal.
  images?: LightboxImage[];
  startIndex?: number;
}

// Large, zoomable image inspector shared by the candidate strip and the trace
// artifact gallery. Scale 1 is the contain-fitted baseline (sharp — the 2K source
// is downscaled, never upscaled). Transform lives in refs and is applied to the
// <img> via requestAnimationFrame; wheel eases toward a target scale at the
// cursor, pointer-capture drag pans, double-click toggles 1x <-> 2.5x, arrows and
// chevrons flip between images.
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
  images,
  startIndex,
}: LightboxViewerProps) {
  const stageRef = React.useRef<HTMLDivElement>(null);
  const imgRef = React.useRef<HTMLImageElement>(null);

  // Transform state: `view` is what's painted this frame, `target` is where we are
  // easing toward. Both are refs so zoom/pan never trigger a React render.
  const viewRef = React.useRef<View>({ ...FIT });
  const targetRef = React.useRef<View>({ ...FIT });
  const rafRef = React.useRef<number | null>(null);
  const naturalRef = React.useRef<{ w: number; h: number } | null>(null);

  const [panning, setPanning] = React.useState(false);
  // Lightweight sync of the eased zoom to the % readout (integer percent only).
  const [displayZoom, setDisplayZoom] = React.useState(100);
  const lastDisplayRef = React.useRef(100);
  // Explicit pixel size of the contain-fitted baseline (scale 1). Kept in state so
  // the <img> is sized from its natural resolution — no CSS stretch to blur it.
  const [baseSize, setBaseSize] = React.useState<{ w: number; h: number } | null>(null);

  const dragRef = React.useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  } | null>(null);

  // Image-set mode manages its own index; single/candidate mode is parent-driven.
  const usingSet = Array.isArray(images) && images.length > 0;
  const [setIndex, setSetIndex] = React.useState(0);

  const activeImage = usingSet ? images![setIndex]?.image ?? null : image ?? null;
  const activeTitle = usingSet ? images![setIndex]?.title ?? "" : title ?? "";
  const activeIndex = usingSet ? setIndex : index;
  const activeCount = usingSet ? images!.length : count;
  const goTo = usingSet ? setSetIndex : onIndex;

  const navigable =
    typeof activeIndex === "number" && typeof activeCount === "number" && activeCount > 1 && !!goTo;
  const isSelected = !usingSet && typeof index === "number" && index === selectedIndex;
  const canSelect = !usingSet && typeof index === "number" && !!onSelect && index !== selectedIndex;

  const src = activeImage ? candidateDataUrl(activeImage) : null;

  const applyTransform = React.useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    const v = viewRef.current;
    img.style.transform = `translate3d(${v.x}px, ${v.y}px, 0) scale(${v.scale})`;
  }, []);

  const syncDisplay = React.useCallback(() => {
    const rounded = Math.round(viewRef.current.scale * 100);
    if (rounded !== lastDisplayRef.current) {
      lastDisplayRef.current = rounded;
      setDisplayZoom(rounded);
    }
  }, []);

  // Ease `view` toward `target` one frame at a time; stop once settled.
  const kick = React.useCallback(() => {
    if (rafRef.current != null) return;
    const frame = () => {
      const v = viewRef.current;
      const t = targetRef.current;
      const ds = t.scale - v.scale;
      const dx = t.x - v.x;
      const dy = t.y - v.y;
      if (Math.abs(ds) < 0.001 && Math.abs(dx) < 0.05 && Math.abs(dy) < 0.05) {
        viewRef.current = { ...t };
        applyTransform();
        syncDisplay();
        rafRef.current = null;
        return;
      }
      viewRef.current = { scale: v.scale + ds * LERP, x: v.x + dx * LERP, y: v.y + dy * LERP };
      applyTransform();
      syncDisplay();
      rafRef.current = requestAnimationFrame(frame);
    };
    rafRef.current = requestAnimationFrame(frame);
  }, [applyTransform, syncDisplay]);

  const snapToFit = React.useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    viewRef.current = { ...FIT };
    targetRef.current = { ...FIT };
    lastDisplayRef.current = 100;
    setDisplayZoom(100);
    applyTransform();
  }, [applyTransform]);

  // Zoom toward a target scale about a screen point, keeping that point fixed.
  // Pivot math is relative to `target` so successive wheel ticks accumulate.
  const zoomTo = React.useCallback(
    (nextScale: number, clientX?: number, clientY?: number) => {
      const stage = stageRef.current;
      if (!stage) return;
      const rect = stage.getBoundingClientRect();
      const pivotX = (clientX ?? rect.left + rect.width / 2) - rect.left - rect.width / 2;
      const pivotY = (clientY ?? rect.top + rect.height / 2) - rect.top - rect.height / 2;
      const t = targetRef.current;
      const scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, nextScale));
      const k = scale / t.scale;
      targetRef.current = {
        scale,
        x: pivotX - k * (pivotX - t.x),
        y: pivotY - k * (pivotY - t.y),
      };
      kick();
    },
    [kick],
  );

  const easeToFit = React.useCallback(() => {
    targetRef.current = { ...FIT };
    kick();
  }, [kick]);

  // Reset the transform whenever the viewer opens or the shown image changes.
  // Done during render (React's "adjust state on prop change") with a ref guard so
  // a stale zoom never paints for the new image; the actual DOM apply happens on
  // the next image load / effect, not here.
  const srcGuardRef = React.useRef<string | null>(null);
  const guardKey = open ? src : null;
  if (srcGuardRef.current !== guardKey) {
    srcGuardRef.current = guardKey;
    viewRef.current = { ...FIT };
    targetRef.current = { ...FIT };
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (lastDisplayRef.current !== 100) {
      lastDisplayRef.current = 100;
      setDisplayZoom(100);
    }
    // Hide until the new image is measured so we never flash it at a stale size.
    if (open && baseSize !== null) setBaseSize(null);
  }

  // Seed the internal index from startIndex each time the set viewer opens.
  React.useEffect(() => {
    if (open && usingSet) {
      const list = images!;
      setSetIndex(Math.max(0, Math.min(list.length - 1, startIndex ?? 0)));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Measure the contain-fitted baseline from the image's natural resolution and
  // the current stage size. Never upscale past natural (small art stays crisp);
  // 2K art downscales to fit (sharp). Recomputed on load and on stage resize.
  const measure = React.useCallback(() => {
    const stage = stageRef.current;
    const nat = naturalRef.current;
    if (!stage || !nat || nat.w === 0 || nat.h === 0) return;
    const availW = stage.clientWidth;
    const availH = stage.clientHeight;
    if (availW === 0 || availH === 0) return;
    const fit = Math.min(availW / nat.w, availH / nat.h, 1);
    setBaseSize({ w: Math.round(nat.w * fit), h: Math.round(nat.h * fit) });
  }, []);

  const onImgLoad = React.useCallback(
    (event: React.SyntheticEvent<HTMLImageElement>) => {
      const img = event.currentTarget;
      naturalRef.current = { w: img.naturalWidth, h: img.naturalHeight };
      snapToFit();
      measure();
    },
    [measure, snapToFit],
  );

  // Recompute the baseline (and snap to fit) when the stage is resized.
  React.useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !open) return;
    const observer = new ResizeObserver(() => {
      measure();
      snapToFit();
    });
    observer.observe(stage);
    return () => observer.disconnect();
  }, [open, measure, snapToFit]);

  // Native wheel listener: React wheel events are passive and cannot preventDefault.
  React.useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !open) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const factor = Math.pow(WHEEL_STEP, -event.deltaY / 100);
      zoomTo(targetRef.current.scale * factor, event.clientX, event.clientY);
    };
    stage.addEventListener("wheel", onWheel, { passive: false });
    return () => stage.removeEventListener("wheel", onWheel);
  }, [open, zoomTo]);

  // Cancel any in-flight ease when the viewer closes/unmounts.
  React.useEffect(() => {
    if (open) return;
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, [open]);

  const flip = React.useCallback(
    (delta: -1 | 1) => {
      if (!navigable || typeof activeIndex !== "number" || typeof activeCount !== "number" || !goTo) return;
      goTo((activeIndex + delta + activeCount) % activeCount);
    },
    [navigable, activeIndex, activeCount, goTo],
  );

  // Arrow keys flip between images. Esc is the Dialog's own default, left untouched.
  React.useEffect(() => {
    if (!open || !navigable) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        flip(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        flip(1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, navigable, flip]);

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    // Freeze any running ease at the painted position so the drag tracks 1:1.
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    targetRef.current = { ...viewRef.current };
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: viewRef.current.x,
      originY: viewRef.current.y,
    };
    setPanning(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const nextX = drag.originX + (event.clientX - drag.startX);
    const nextY = drag.originY + (event.clientY - drag.startY);
    // No easing during drag — apply directly for a hand-locked feel.
    viewRef.current = { ...viewRef.current, x: nextX, y: nextY };
    targetRef.current = { ...targetRef.current, x: nextX, y: nextY };
    applyTransform();
  };

  const onPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    setPanning(false);
  };

  const onDoubleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (targetRef.current.scale > 1.01) {
      easeToFit();
    } else {
      zoomTo(DOUBLE_TAP_SCALE, event.clientX, event.clientY);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent fullscreen showCloseButton={false} className="gap-0">
        {/* header */}
        <div className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2">
          <DialogTitle className="text-[13px] font-medium">{activeTitle}</DialogTitle>
          {isSelected ? (
            <span className="inline-flex items-center gap-1 rounded-md border border-primary/40 bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
              <Check className="size-2.5" /> selected
            </span>
          ) : null}
          {navigable && typeof activeIndex === "number" && typeof activeCount === "number" ? (
            <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
              {activeIndex + 1} / {activeCount}
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

        {/* stage — spans (viewport - 32px) minus the header height on both axes */}
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
          {src ? (
            <img
              ref={imgRef}
              src={src}
              alt={activeTitle}
              draggable={false}
              onLoad={onImgLoad}
              className="pointer-events-none max-w-none max-h-none select-none"
              style={{
                width: baseSize ? `${baseSize.w}px` : undefined,
                height: baseSize ? `${baseSize.h}px` : undefined,
                visibility: baseSize ? "visible" : "hidden",
                transformOrigin: "center center",
                imageRendering: "auto",
                willChange: "transform",
              }}
            />
          ) : null}

          {/* image-set / candidate flip chevrons */}
          {navigable ? (
            <div className="pointer-events-none absolute inset-0" onPointerDown={(event) => event.stopPropagation()}>
              <Button
                size="icon"
                variant="ghost"
                className="pointer-events-auto absolute left-3 top-1/2 size-9 -translate-y-1/2 rounded-full border border-border bg-card/90 shadow-lg backdrop-blur-md"
                onClick={() => flip(-1)}
                aria-label="Previous image"
              >
                <ChevronLeft className="size-4" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                className="pointer-events-auto absolute right-3 top-1/2 size-9 -translate-y-1/2 rounded-full border border-border bg-card/90 shadow-lg backdrop-blur-md"
                onClick={() => flip(1)}
                aria-label="Next image"
              >
                <ChevronRight className="size-4" />
              </Button>
            </div>
          ) : null}

          {/* zoom cluster — synced to the eased scale */}
          <div
            className="absolute bottom-3 right-3 flex items-center gap-0.5 rounded-lg border border-border bg-card/90 p-0.5 shadow-lg backdrop-blur-md"
            onPointerDown={(event) => event.stopPropagation()}
            onDoubleClick={(event) => event.stopPropagation()}
          >
            <Button
              size="icon"
              variant="ghost"
              className="size-7"
              onClick={() => zoomTo(targetRef.current.scale / WHEEL_STEP)}
              aria-label="Zoom out"
            >
              <Minus className="size-3.5" />
            </Button>
            <span className="w-12 text-center font-mono text-[11px] tabular-nums text-muted-foreground">
              {displayZoom}%
            </span>
            <Button
              size="icon"
              variant="ghost"
              className="size-7"
              onClick={() => zoomTo(targetRef.current.scale * WHEEL_STEP)}
              aria-label="Zoom in"
            >
              <Plus className="size-3.5" />
            </Button>
            <Button size="icon" variant="ghost" className="size-7" onClick={easeToFit} aria-label="Fit">
              <Maximize className="size-3.5" />
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
