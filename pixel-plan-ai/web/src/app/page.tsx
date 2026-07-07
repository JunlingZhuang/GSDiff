"use client";

import { InspectorPanel } from "@/components/inspector-panel";
import { ProgramPanel } from "@/components/program-panel";
import { TopBar } from "@/components/top-bar";
import { Viewport } from "@/components/viewport";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { useStudio } from "@/hooks/use-studio";

export default function StudioPage() {
  const studio = useStudio();

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground">
      <TopBar studio={studio} />
      <ResizablePanelGroup orientation="horizontal" className="min-h-0 flex-1">
        <ResizablePanel defaultSize="22%" minSize="280px" maxSize="34%" className="h-full">
          <ProgramPanel studio={studio} />
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel defaultSize="56%" minSize="30%" className="h-full">
          <Viewport studio={studio} />
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel defaultSize="22%" minSize="280px" maxSize="34%" className="h-full">
          <InspectorPanel studio={studio} />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
