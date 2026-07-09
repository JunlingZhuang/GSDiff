"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { Box, Eye, Grid2x2, Maximize, Minus, Plus, Square, StopCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Viewport2D, type Viewport2DHandle } from "@/components/viewport-2d";
import type { Studio } from "@/hooks/use-studio";
import { cn } from "@/lib/utils";

const Viewport3D = dynamic(() => import("@/components/viewport-3d").then((module_) => module_.Viewport3D), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-xs text-muted-foreground">Loading 3D engine…</div>
  ),
});

// Bottom/side HUD chrome is inset to clear the floating panels
// (left panel 300 + 12 inset, right panel 320 + 12 inset, plus a 12px gap).
const CLEAR_LEFT = "left-[324px]";
const CLEAR_RIGHT = "right-[344px]";

export function Viewport({ studio }: { studio: Studio }) {
  const viewportRef = React.useRef<Viewport2DHandle>(null);
  const [zoomPercent, setZoomPercent] = React.useState(100);
  const [hoverCell, setHoverCell] = React.useState<{ x: number; y: number } | null>(null);
  const [hoverEntity, setHoverEntity] = React.useState<string | null>(null);
  const plan = studio.activePlan;
  const busy = !!studio.busyAction;
  const is2d = studio.viewport === "2d";

  const downloadPng = React.useCallback(() => {
    const url = viewportRef.current?.exportPng();
    if (!url) return;
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "pixel-plan.png";
    anchor.click();
  }, []);

  React.useEffect(() => {
    (window as unknown as { __pixelPlanExportPng?: () => void }).__pixelPlanExportPng = downloadPng;
  }, [downloadPng]);

  const handleViewChange = React.useCallback((zoom: number) => setZoomPercent(Math.round(zoom * 100)), []);
  const handleHoverCell = React.useCallback((cell: { x: number; y: number } | null) => setHoverCell(cell), []);
  const handleHoverEntity = React.useCallback((label: string | null) => setHoverEntity(label), []);

  const ftPerCell = plan ? (plan.meters_per_cell * 3.28084).toFixed(2) : null;

  return (
    <div className="relative isolate h-full min-h-0 w-full overflow-hidden bg-canvas">
      {!is2d && plan ? (
        /* absolute cage: the R3F canvas must never drive layout height */
        <div className="absolute inset-0 overflow-hidden">
          <Viewport3D plan={plan} />
        </div>
      ) : (
        <Viewport2D
          ref={viewportRef}
          plan={plan}
          reference={studio.reference?.image ?? null}
          showReference={studio.showReference}
          referenceOpacity={studio.referenceOpacity}
          preview={busy}
          rightPanelOpen={!!plan}
          selectedRoomId={studio.selectedRoomId}
          onViewChange={handleViewChange}
          onHoverCell={handleHoverCell}
          onHoverEntity={handleHoverEntity}
          onSelectRoom={studio.setSelectedRoomId}
        />
      )}

      {/* empty state: the faint grid is already visible behind this line */}
      {!plan ? (
        <div className="pointer-events-none absolute inset-0 z-[5] flex items-center justify-center">
          <p className="text-[13px] text-muted-foreground">Pick a program and generate</p>
        </div>
      ) : null}

      {/* generation scanline sweep (or static pulsing border under reduced motion) */}
      {busy ? <div className="scan-line" aria-hidden /> : null}

      {/* 2D / 3D switcher */}
      <div className="absolute left-1/2 top-3 z-30 flex -translate-x-1/2 items-center gap-0.5 rounded-lg border border-border bg-card/90 p-0.5 shadow-lg backdrop-blur-md">
        <Button
          size="sm"
          variant={is2d ? "default" : "ghost"}
          className="h-7 gap-1.5 px-3 text-xs font-medium"
          onClick={() => studio.setViewport("2d")}
        >
          <Grid2x2 className="size-3.5" /> 2D
        </Button>
        <Button
          size="sm"
          variant={studio.viewport === "3d" ? "default" : "ghost"}
          className="h-7 gap-1.5 px-3 text-xs font-medium"
          onClick={() => studio.setViewport("3d")}
          disabled={!plan}
        >
          <Box className="size-3.5" /> 3D
        </Button>
      </div>

      {/* reference compare (2D only, after a transcription) */}
      {is2d && plan && studio.reference ? (
        <div className={cn("absolute top-3 z-30 flex items-center gap-2 rounded-lg border border-border bg-card/90 px-2 py-1.5 shadow-lg backdrop-blur-md", CLEAR_LEFT)}>
          <Button
            size="sm"
            variant={studio.showReference ? "default" : "ghost"}
            className="h-6 gap-1.5 px-2 text-[11px] font-medium"
            onClick={() => studio.setShowReference(!studio.showReference)}
          >
            {studio.showReference ? <Eye className="size-3" /> : <Square className="size-3" />}
            Reference
          </Button>
          {studio.showReference ? (
            <div className="flex w-28 items-center gap-1.5">
              <Slider
                value={[studio.referenceOpacity]}
                min={10}
                max={95}
                step={5}
                onValueChange={(value) => studio.setReferenceOpacity(Array.isArray(value) ? (value[0] ?? 55) : value)}
              />
              <span className="w-8 font-mono text-[10px] tabular-nums text-muted-foreground">{studio.referenceOpacity}%</span>
            </div>
          ) : null}
        </div>
      ) : null}

      {/* raw trace preview banner (2D only, before the seed is validated) */}
      {is2d && plan && studio.showingTracePreview ? (
        <div className={cn("absolute top-3 z-30 flex items-center gap-1.5 rounded-lg border border-warning/50 bg-card/90 px-2 py-1.5 shadow-lg backdrop-blur-md", CLEAR_LEFT)}>
          <span className="size-1.5 shrink-0 rounded-full bg-warning" aria-hidden />
          <span className="text-[11px] font-medium text-foreground">raw trace</span>
          <span className="text-[11px] text-muted-foreground">· not yet validated</span>
        </div>
      ) : null}

      {/* readout (2D only, with a plan) */}
      {is2d && plan ? (
        <div className={cn("absolute bottom-3 z-30 rounded-lg border border-border bg-card/90 px-2.5 py-1.5 font-mono text-[11px] tabular-nums text-muted-foreground shadow-lg backdrop-blur-md", CLEAR_LEFT)}>
          {plan.width}×{plan.height} · {ftPerCell} ft/cell
          {hoverEntity ? <span className="text-foreground"> · {hoverEntity}</span> : hoverCell ? ` · ${hoverCell.x},${hoverCell.y}` : ""}
        </div>
      ) : null}

      {/* zoom cluster (2D only) */}
      {is2d ? (
        <div className={cn("absolute bottom-3 z-30 flex items-center gap-0.5 rounded-lg border border-border bg-card/90 p-0.5 shadow-lg backdrop-blur-md", CLEAR_RIGHT)}>
          <Button size="icon" variant="ghost" className="size-7" onClick={() => viewportRef.current?.zoomBy(1 / 1.25)}>
            <Minus className="size-3.5" />
          </Button>
          <span className="w-12 text-center font-mono text-[11px] tabular-nums text-muted-foreground">{zoomPercent}%</span>
          <Button size="icon" variant="ghost" className="size-7" onClick={() => viewportRef.current?.zoomBy(1.25)}>
            <Plus className="size-3.5" />
          </Button>
          <Button size="icon" variant="ghost" className="size-7" onClick={() => viewportRef.current?.fit()}>
            <Maximize className="size-3.5" />
          </Button>
        </div>
      ) : null}

      {/* generation chip */}
      {busy ? (
        <div className="absolute bottom-6 left-1/2 z-40 flex -translate-x-1/2 items-center gap-3 rounded-xl border border-border bg-card/90 px-3 py-2 shadow-[0_12px_32px_rgba(16,24,40,0.10)] backdrop-blur-md">
          <span className="size-2 shrink-0 animate-pulse rounded-full bg-primary" aria-hidden />
          <p className="max-w-[360px] truncate text-[12px] text-foreground">{studio.phaseText || "Working…"}</p>
          <span className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">{studio.elapsed}s</span>
          <Button
            size="sm"
            variant="destructive"
            className="h-7 shrink-0 gap-1.5 px-2.5 text-[11px] font-medium"
            onClick={() => void studio.stopGeneration()}
          >
            <StopCircle className="size-3.5" /> Stop
          </Button>
        </div>
      ) : null}
    </div>
  );
}
