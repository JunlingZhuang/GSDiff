"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { Box, Grid2x2, Maximize, Minus, Plus, Ruler } from "lucide-react";

import { RoomViewport2D, type RoomViewport2DHandle } from "@/components/room-viewport-2d";
import { RunLogDrawer } from "@/components/run-log-drawer";
import { Button } from "@/components/ui/button";
import type { RoomStudio } from "@/hooks/use-room-studio";
import type { Studio } from "@/hooks/use-studio";
import { cn } from "@/lib/utils";

const RoomViewport3D = dynamic(
  () => import("@/components/room-viewport-3d").then((module_) => module_.RoomViewport3D),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">Loading 3D engine…</div>
    ),
  },
);

// HUD chrome is inset to clear the floating panels (left 300 + 12, right 320 + 12).
const CLEAR_LEFT = "left-[324px]";
const CLEAR_RIGHT = "right-[344px]";

// The room viewport: mounts the 2D symbol renderer or the 3D asset viewport, with
// the same HUD language as the floor viewport (2D/3D switch, zoom cluster, run-log
// drawer). It stays a separate component from the floor Viewport so that flow is
// untouched.
export function RoomViewport({ room }: { room: RoomStudio }) {
  const viewportRef = React.useRef<RoomViewport2DHandle>(null);
  const [zoomPercent, setZoomPercent] = React.useState(100);
  const [showClearances, setShowClearances] = React.useState(true);
  const layout = room.activeRoom;
  const busy = !!room.busyAction;
  const is2d = room.viewport === "2d";

  const handleViewChange = React.useCallback((zoom: number) => setZoomPercent(Math.round(zoom * 100)), []);

  const exportPng = React.useCallback(() => {
    const url = viewportRef.current?.exportPng();
    if (!url) return;
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "icu-room.png";
    anchor.click();
  }, []);

  React.useEffect(() => {
    (window as unknown as { __roomPlanExportPng?: () => void }).__roomPlanExportPng = exportPng;
  }, [exportPng]);

  const hasBed = !!layout?.assets.some((asset) => asset.type === "icu_bed");

  return (
    <div className="relative isolate h-full min-h-0 w-full overflow-hidden bg-canvas">
      {!is2d && layout ? (
        <div className="absolute inset-0 overflow-hidden">
          <RoomViewport3D layout={layout} />
        </div>
      ) : (
        <RoomViewport2D
          ref={viewportRef}
          layout={layout}
          showClearances={showClearances}
          preview={busy}
          rightPanelOpen={!!room.result}
          selectedAssetId={room.selectedAssetId}
          onSelectAsset={room.setSelectedAssetId}
          onViewChange={handleViewChange}
        />
      )}

      {!layout ? (
        <div className="pointer-events-none absolute inset-0 z-[5] flex items-center justify-center">
          <p className="text-[13px] text-muted-foreground">Set room dimensions and generate</p>
        </div>
      ) : null}

      {busy ? <div className="scan-line" aria-hidden /> : null}

      {/* 2D / 3D switcher */}
      <div className="absolute left-1/2 top-3 z-30 flex -translate-x-1/2 items-center gap-0.5 rounded-lg border border-border bg-card/90 p-0.5 shadow-lg backdrop-blur-md">
        <Button
          size="sm"
          variant={is2d ? "default" : "ghost"}
          className="h-7 gap-1.5 px-3 text-xs font-medium"
          onClick={() => room.setViewport("2d")}
        >
          <Grid2x2 className="size-3.5" /> 2D
        </Button>
        <Button
          size="sm"
          variant={room.viewport === "3d" ? "default" : "ghost"}
          className="h-7 gap-1.5 px-3 text-xs font-medium"
          onClick={() => room.setViewport("3d")}
          disabled={!layout}
        >
          <Box className="size-3.5" /> 3D
        </Button>
      </div>

      {/* clearances chip (2D only, with a bed) */}
      {is2d && hasBed ? (
        <div className={cn("absolute top-3 z-30", CLEAR_LEFT)}>
          <Button
            size="sm"
            variant={showClearances ? "default" : "outline"}
            className="h-7 gap-1.5 px-2.5 text-[11px] font-medium"
            aria-pressed={showClearances}
            onClick={() => setShowClearances((value) => !value)}
          >
            <Ruler className="size-3.5" /> Clearances
          </Button>
        </div>
      ) : null}

      {/* dimensions readout (with a layout) */}
      {layout ? (
        <div
          className={cn(
            "absolute bottom-3 z-30 rounded-lg border border-border bg-card/90 px-2.5 py-1.5 font-mono text-[11px] tabular-nums text-muted-foreground shadow-lg backdrop-blur-md",
            CLEAR_LEFT,
          )}
        >
          {layout.room.width_ft} × {layout.room.depth_ft} ft · {Math.round(layout.room.width_ft * layout.room.depth_ft)} ft²
        </div>
      ) : null}

      {/* zoom cluster (2D only) */}
      {is2d ? (
        <div
          className={cn(
            "absolute bottom-3 z-30 flex items-center gap-0.5 rounded-lg border border-border bg-card/90 p-0.5 shadow-lg backdrop-blur-md",
            CLEAR_RIGHT,
          )}
        >
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

      {/* generation ticker + live per-attempt run-log drawer — the shared drawer
          reads only the run-log subset (busyAction, live iterations/events,
          runLogOpen, phaseText, elapsed, stopGeneration), all provided by the
          room hook, so the Studio cast is safe and needs no shared-file change. */}
      <RunLogDrawer studio={room as unknown as Studio} />
    </div>
  );
}
