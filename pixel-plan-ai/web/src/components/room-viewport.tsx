"use client";

import * as React from "react";
import { Box, Grid2x2 } from "lucide-react";

import { RunLogDrawer } from "@/components/run-log-drawer";
import { Button } from "@/components/ui/button";
import type { RoomStudio } from "@/hooks/use-room-studio";
import type { Studio } from "@/hooks/use-studio";
import { cn } from "@/lib/utils";

// HUD chrome is inset to clear the floating panels (left 300 + 12, right 320 + 12).
const CLEAR_LEFT = "left-[324px]";

// The room viewport shell: the 2D/3D switch, the empty/summary state, and the
// shared generation run-log drawer. The 2D symbol renderer and 3D asset
// viewport mount here in the next step; keeping them out of this file for now
// means the flow switch + panels ship independently of the renderers.
export function RoomViewport({ room }: { room: RoomStudio }) {
  const layout = room.activeRoom;
  const busy = !!room.busyAction;
  const is2d = room.viewport === "2d";

  return (
    <div className="relative isolate h-full min-h-0 w-full overflow-hidden bg-canvas">
      <div className="pointer-events-none absolute inset-0 z-[5] flex items-center justify-center">
        <p className="text-[13px] text-muted-foreground">
          {layout
            ? `${layout.assets.length} assets placed · ${room.viewport.toUpperCase()} renderer loading next`
            : "Set room dimensions and generate"}
        </p>
      </div>

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

      {/* dimensions readout */}
      {layout ? (
        <div
          className={cn(
            "absolute bottom-3 z-30 rounded-lg border border-border bg-card/90 px-2.5 py-1.5 font-mono text-[11px] tabular-nums text-muted-foreground shadow-lg backdrop-blur-md",
            CLEAR_LEFT,
          )}
        >
          {layout.room.width_ft} × {layout.room.depth_ft} ft
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
