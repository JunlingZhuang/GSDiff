"use client";

import { InspectorPanel } from "@/components/inspector-panel";
import { ProgramPanel } from "@/components/program-panel";
import { TopBar } from "@/components/top-bar";
import { Viewport } from "@/components/viewport";
import { useStudio } from "@/hooks/use-studio";

export default function StudioPage() {
  const studio = useStudio();

  // The inspector floats in only once there is an actual plan to inspect —
  // a final result or a live/preview plan. It must NOT mount as an empty shell
  // the moment generation starts (i.e. on busyAction, code, or iteration logs).
  const showInspector = !!(studio.result || studio.activePlan);

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground">
      <TopBar studio={studio} />
      <main className="relative min-h-0 flex-1">
        {/* full-bleed blueprint viewport */}
        <Viewport studio={studio} />

        {/* floating left panel — a tool card that hugs its content, not full-height */}
        <div className="absolute left-3 top-3 z-20 flex h-auto max-h-[calc(100vh-70px)] w-[300px] flex-col overflow-hidden rounded-xl border border-border bg-card/92 shadow-2xl backdrop-blur-md">
          <ProgramPanel studio={studio} />
        </div>

        {/* floating right panel — hidden until there is something to inspect */}
        {showInspector ? (
          <div className="absolute right-3 top-3 bottom-3 z-20 flex w-[320px] flex-col overflow-hidden rounded-xl border border-border bg-card/92 shadow-2xl backdrop-blur-md animate-in fade-in slide-in-from-right-4 duration-300">
            <InspectorPanel studio={studio} />
          </div>
        ) : null}
      </main>
    </div>
  );
}
