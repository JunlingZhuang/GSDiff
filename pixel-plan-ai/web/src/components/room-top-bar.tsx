"use client";

import * as React from "react";
import { ClipboardCopy, Download, FileJson, ImageDown } from "lucide-react";

import { FlowSwitch, type StudioFlow } from "@/components/top-bar";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type { RoomStudio } from "@/hooks/use-room-studio";
import { cn } from "@/lib/utils";

function scoreTone(score: number): string {
  if (score >= 90) return "border-success/40 bg-success/10 text-success";
  if (score >= 72) return "border-warning/40 bg-warning/10 text-warning";
  return "border-destructive/40 bg-destructive/10 text-destructive";
}

// Room-flow sibling of TopBar: same chrome, but titled by the room preset and
// wired to the room hook. It shares the FlowSwitch so the two flows toggle from
// the same spot. Kept separate so the floor top bar stays untouched.
export function RoomTopBar({
  room,
  flow,
  onFlowChange,
}: {
  room: RoomStudio;
  flow: StudioFlow;
  onFlowChange: (flow: StudioFlow) => void;
}) {
  const busy = !!room.busyAction;

  const downloadJson = React.useCallback(() => {
    if (!room.result) return;
    const blob = new Blob([JSON.stringify(room.result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "icu-room.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }, [room.result]);

  const exportPng = React.useCallback(() => {
    (window as unknown as { __roomPlanExportPng?: () => void }).__roomPlanExportPng?.();
  }, []);

  const score = room.result?.validation.score ?? null;

  return (
    <header className="flex h-[46px] shrink-0 items-center gap-3 border-b border-border bg-background px-3">
      <FlowSwitch flow={flow} onFlowChange={onFlowChange} />
      <div className="flex items-center gap-2">
        <span className="grid grid-cols-2 gap-px" aria-hidden>
          {["var(--primary)", "var(--success)", "var(--warning)", "var(--muted-foreground)"].map((color) => (
            <i key={color} className="size-1.5 rounded-[1px]" style={{ background: color }} />
          ))}
        </span>
        <span className="text-[13px] font-semibold text-foreground">Pixel Plan</span>
        <span className="rounded-md border border-border px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
          Room
        </span>
      </div>

      <div className="flex min-w-0 flex-1 items-center justify-center gap-2">
        <span
          className={cn(
            "size-1.5 shrink-0 rounded-full",
            busy ? "animate-pulse bg-primary" : "bg-muted-foreground/50",
          )}
          aria-hidden
        />
        <span className="truncate text-[13px] text-muted-foreground">Single-patient ICU room</span>
      </div>

      <div className="flex items-center gap-2">
        <span
          className={cn(
            "rounded-md border px-2 py-0.5 font-mono text-[11px] whitespace-nowrap",
            room.health?.gemini_configured
              ? "border-border text-muted-foreground"
              : "border-destructive/40 text-destructive",
          )}
        >
          {room.health ? (room.health.gemini_configured ? room.health.model : "no api key") : "checking…"}
        </span>

        <DropdownMenu>
          <DropdownMenuTrigger
            className={cn(buttonVariants({ variant: "outline", size: "sm" }), "h-7 gap-1.5 px-2.5 text-[12px]")}
          >
            <Download className="size-3.5" />
            Export
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem disabled={!room.result} onClick={exportPng}>
              <ImageDown />
              PNG image
            </DropdownMenuItem>
            <DropdownMenuItem disabled={!room.result} onClick={downloadJson}>
              <FileJson />
              JSON result
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              disabled={!room.code.trim()}
              onClick={() => void navigator.clipboard.writeText(room.code)}
            >
              <ClipboardCopy />
              Copy code
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {room.result && score !== null ? (
          <span className={cn("rounded-md border px-2 py-0.5 font-mono text-[11px] tabular-nums", scoreTone(score))}>
            {score}
          </span>
        ) : null}
      </div>
    </header>
  );
}
