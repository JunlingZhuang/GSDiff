"use client";

import * as React from "react";

import { InspectorPanel } from "@/components/inspector-panel";
import { ProgramPanel } from "@/components/program-panel";
import { RoomInspector } from "@/components/room-inspector";
import { RoomPanel } from "@/components/room-panel";
import { RoomTopBar } from "@/components/room-top-bar";
import { RoomViewport } from "@/components/room-viewport";
import { TopBar, type StudioFlow } from "@/components/top-bar";
import { Viewport } from "@/components/viewport";
import { useRoomStudio } from "@/hooks/use-room-studio";
import { useStudio } from "@/hooks/use-studio";

// Shared floating-panel shell (glass card, fixed full height) used by both flows.
const PANEL_LEFT =
  "absolute left-3 top-3 z-20 flex h-[calc(100vh-70px)] w-[300px] flex-col overflow-hidden rounded-xl border border-border bg-card/90 shadow-[0_12px_32px_rgba(16,24,40,0.08)] backdrop-blur-md";
const PANEL_RIGHT =
  "absolute right-3 top-3 z-20 flex h-[calc(100vh-70px)] w-[320px] flex-col overflow-hidden rounded-xl border border-border bg-card/90 shadow-[0_12px_32px_rgba(16,24,40,0.08)] backdrop-blur-md animate-in fade-in slide-in-from-right-4 duration-300";

export default function StudioPage() {
  // Top-level flow switch. Floor is the original pipeline (rendered EXACTLY as
  // before); Room is the parallel ICU single-room flow. Both hooks mount every
  // render so hook order is stable regardless of the active flow.
  const [flow, setFlow] = React.useState<StudioFlow>("floor");
  const studio = useStudio();
  const room = useRoomStudio();

  if (flow === "room") {
    return (
      <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground">
        <RoomTopBar room={room} flow={flow} onFlowChange={setFlow} />
        <main className="relative min-h-0 flex-1">
          <RoomViewport room={room} />
          <div className={PANEL_LEFT}>
            <RoomPanel room={room} />
          </div>
          {room.result ? (
            <div className={PANEL_RIGHT}>
              <RoomInspector room={room} />
            </div>
          ) : null}
        </main>
      </div>
    );
  }

  // Floor flow — unchanged from the original tree, plus the top-bar flow switch.
  // The inspector floats in only once there is an actual plan to inspect —
  // a final result or a live/preview plan. It must NOT mount as an empty shell
  // the moment generation starts (i.e. on busyAction, code, or iteration logs).
  const showInspector = !!(studio.result || studio.activePlan);

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground">
      <TopBar studio={studio} flow={flow} onFlowChange={setFlow} />
      <main className="relative min-h-0 flex-1">
        {/* full-bleed blueprint viewport */}
        <Viewport studio={studio} />

        {/* floating left panel — fixed full-height tool card: header + scrolling
            body + footer never collapse or grow past the viewport */}
        <div className={PANEL_LEFT}>
          <ProgramPanel studio={studio} />
        </div>

        {/* floating right panel — same fixed full-height treatment; hidden until
            there is something to inspect */}
        {showInspector ? (
          <div className={PANEL_RIGHT}>
            <InspectorPanel studio={studio} />
          </div>
        ) : null}
      </main>
    </div>
  );
}
