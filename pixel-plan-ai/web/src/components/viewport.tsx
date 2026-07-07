"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { Box, Eye, Grid2x2, Loader2, Maximize, Minus, Plus, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Viewport2D, type Viewport2DHandle } from "@/components/viewport-2d";
import type { Studio } from "@/hooks/use-studio";

const Viewport3D = dynamic(() => import("@/components/viewport-3d").then((module_) => module_.Viewport3D), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-xs text-muted-foreground">Loading 3D engine…</div>
  ),
});

export function Viewport({ studio }: { studio: Studio }) {
  const viewportRef = React.useRef<Viewport2DHandle>(null);
  const [zoomPercent, setZoomPercent] = React.useState(100);
  const plan = studio.activePlan;

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

  return (
    <div className="relative h-full min-h-0 w-full bg-[#eceae4] [background-image:radial-gradient(rgba(31,36,42,0.09)_1px,transparent_1px)] [background-size:22px_22px]">
      {plan ? (
        studio.viewport === "2d" ? (
          <Viewport2D
            ref={viewportRef}
            plan={plan}
            reference={studio.reference?.image ?? null}
            showReference={studio.showReference}
            referenceOpacity={studio.referenceOpacity}
            onViewChange={(zoom) => setZoomPercent(Math.round(zoom * 100))}
          />
        ) : (
          /* absolute cage: the R3F canvas must never drive layout height */
          <div className="absolute inset-0 overflow-hidden">
            <Viewport3D plan={plan} />
          </div>
        )
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
          <div className="grid grid-cols-2 gap-1 opacity-40">
            {["#7fb5a8", "#e3b04b", "#8d9fb8", "#d8d3c8"].map((color) => (
              <span key={color} className="h-5 w-5 rounded-[3px]" style={{ background: color }} />
            ))}
          </div>
          <p className="text-sm font-semibold tracking-wide text-foreground/70">No plan yet</p>
          <p className="max-w-56 text-xs leading-relaxed text-muted-foreground">
            Configure the program on the left and generate the first layout.
          </p>
        </div>
      )}

      {/* 2D / 3D switcher */}
      <div className="absolute left-1/2 top-3 flex -translate-x-1/2 items-center gap-0.5 rounded-lg border bg-background/95 p-0.5 shadow-sm backdrop-blur">
        <Button
          size="sm"
          variant={studio.viewport === "2d" ? "default" : "ghost"}
          className="h-7 gap-1.5 px-3 text-xs font-semibold"
          onClick={() => studio.setViewport("2d")}
        >
          <Grid2x2 className="size-3.5" /> 2D
        </Button>
        <Button
          size="sm"
          variant={studio.viewport === "3d" ? "default" : "ghost"}
          className="h-7 gap-1.5 px-3 text-xs font-semibold"
          onClick={() => studio.setViewport("3d")}
          disabled={!plan}
        >
          <Box className="size-3.5" /> 3D
        </Button>
      </div>

      {/* zoom controls (2D only) */}
      {studio.viewport === "2d" && plan ? (
        <div className="absolute bottom-3 right-3 flex items-center gap-0.5 rounded-lg border bg-background/95 p-0.5 shadow-sm backdrop-blur">
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

      {/* reference compare (2D only, after a transcription) */}
      {studio.viewport === "2d" && plan && studio.reference ? (
        <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-lg border bg-background/95 px-2 py-1.5 shadow-sm backdrop-blur">
          <Button
            size="sm"
            variant={studio.showReference ? "default" : "ghost"}
            className="h-6 gap-1.5 px-2 text-[11px] font-semibold"
            onClick={() => studio.setShowReference(!studio.showReference)}
          >
            {studio.showReference ? <Eye className="size-3" /> : <Square className="size-3" />}
            REFERENCE
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
              <span className="w-7 font-mono text-[10px] tabular-nums text-muted-foreground">{studio.referenceOpacity}%</span>
            </div>
          ) : null}
        </div>
      ) : null}

      {/* generation progress card */}
      {studio.busyAction ? (
        <div className="absolute bottom-6 left-1/2 w-[min(430px,90%)] -translate-x-1/2 rounded-xl border bg-background/97 p-4 shadow-lg backdrop-blur">
          <div className="flex items-center gap-3">
            <Loader2 className="size-4 shrink-0 animate-spin text-primary" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-xs font-medium">{studio.phaseText}</p>
              <p className="mt-0.5 text-[11px] text-muted-foreground">
                {studio.transcribing && studio.busyAction === "generate" ? "Transcription run" : "Agent loop"} ·{" "}
                {studio.elapsed}s elapsed
              </p>
            </div>
            <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={() => void studio.stopGeneration()}>
              STOP
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
