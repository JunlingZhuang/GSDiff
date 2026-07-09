"use client";

import * as React from "react";
import { ChevronDown, ChevronUp, Loader2, StopCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Studio } from "@/hooks/use-studio";
import type { GenerationIteration } from "@/lib/types";
import { cn } from "@/lib/utils";

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

const STATUS_TONE: Record<GenerationIteration["status"], string> = {
  running: "border-primary/40 bg-primary/10 text-primary",
  accepted: "border-success/40 bg-success/10 text-success",
  rejected: "border-warning/40 bg-warning/10 text-warning",
  failed: "border-destructive/40 bg-destructive/10 text-destructive",
};

const STATUS_LABEL: Record<GenerationIteration["status"], string> = {
  running: "generating",
  accepted: "accepted",
  rejected: "rejected",
  failed: "failed",
};

function StatusChip({ status }: { status: GenerationIteration["status"] }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-medium",
        STATUS_TONE[status],
      )}
    >
      {status === "running" ? <Loader2 className="size-2.5 animate-spin" aria-hidden /> : null}
      {STATUS_LABEL[status]}
    </span>
  );
}

// One attempt: header chips, clamped validator message, and (once complete) the
// generated code behind a closed-by-default disclosure. The in-flight attempt
// shows the animated status and no code section.
function RunLogCard({ iteration }: { iteration: GenerationIteration }) {
  const running = iteration.status === "running";
  const [expanded, setExpanded] = React.useState(false);
  const [showExpand, setShowExpand] = React.useState(false);
  const [codeOpen, setCodeOpen] = React.useState(false);
  const messageRef = React.useRef<HTMLParagraphElement>(null);

  // Reveal the expand toggle only when the message actually overflows six lines;
  // once revealed it stays so the collapse control survives expansion.
  React.useEffect(() => {
    const element = messageRef.current;
    if (!element || expanded) return;
    if (element.scrollHeight > element.clientHeight + 1) setShowExpand(true);
  }, [iteration.message, expanded]);

  const seconds = !running && iteration.duration_ms > 0 ? (iteration.duration_ms / 1000).toFixed(1) : null;
  const hasCode = !running && !!iteration.code?.trim();

  return (
    <article className={cn("rounded-lg border bg-card/60 p-2.5", running ? "border-primary/40" : "border-border")}>
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-mono text-[11px] tabular-nums text-foreground">
          #{String(iteration.attempt).padStart(2, "0")}
        </span>
        <span className="text-[11px] capitalize text-muted-foreground">{iteration.phase}</span>
        <span className="rounded-md border border-border bg-secondary/40 px-1 py-0.5 font-mono text-[10px] text-muted-foreground">
          {iteration.model ?? iteration.source}
        </span>
        <StatusChip status={iteration.status} />
        <span className="ml-auto flex items-center gap-2 font-mono text-[10px] tabular-nums text-muted-foreground">
          {iteration.score !== null ? <span className="text-foreground">score {iteration.score}</span> : null}
          {seconds ? <span>{seconds}s</span> : null}
        </span>
      </div>

      {iteration.message ? (
        <>
          <p
            ref={messageRef}
            className={cn(
              "mt-1.5 whitespace-pre-wrap font-mono text-[11px] leading-snug text-muted-foreground",
              !expanded && "line-clamp-6",
            )}
          >
            {iteration.message}
          </p>
          {showExpand ? (
            <button
              type="button"
              className="mt-0.5 text-[10px] font-medium text-primary hover:underline"
              onClick={() => setExpanded((value) => !value)}
            >
              {expanded ? "collapse" : "expand"}
            </button>
          ) : null}
        </>
      ) : null}

      {hasCode ? (
        <div className="mt-1.5">
          <button
            type="button"
            className="flex items-center gap-1 text-[10px] font-medium text-muted-foreground hover:text-foreground"
            onClick={() => setCodeOpen((value) => !value)}
          >
            <ChevronDown className={cn("size-3 transition-transform", codeOpen && "rotate-180")} />
            Code
          </button>
          {codeOpen ? (
            <ScrollArea className="mt-1 max-h-56 rounded-md border border-border bg-secondary/30">
              <pre className="p-2 font-mono text-[10.5px] leading-relaxed text-foreground">{iteration.code}</pre>
            </ScrollArea>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

// Bottom-center generation ticker + a collapsible drawer of per-attempt cards.
// Renders while a run is active OR a completed run's log is still on hand, so the
// user can review the last run before starting another.
export function RunLogDrawer({ studio }: { studio: Studio }) {
  const busy = !!studio.busyAction;
  const items = studio.liveIterations;
  const open = studio.runLogOpen;
  const hasLog = busy || items.length > 0;
  const scrollWrapRef = React.useRef<HTMLDivElement>(null);
  const prevCountRef = React.useRef(0);

  // Auto-scroll to the newest card only when a fresh attempt appears (a length
  // change), not on every poll. Reduced motion jumps without a scroll animation.
  React.useEffect(() => {
    const count = items.length;
    if (open && count > prevCountRef.current) {
      const viewport = scrollWrapRef.current?.querySelector<HTMLElement>('[data-slot="scroll-area-viewport"]');
      if (viewport) {
        viewport.scrollTo({ top: viewport.scrollHeight, behavior: prefersReducedMotion() ? "auto" : "smooth" });
      }
    }
    prevCountRef.current = count;
  }, [items.length, open]);

  if (!hasLog) return null;

  return (
    <div className="absolute bottom-6 left-1/2 z-40 flex -translate-x-1/2 flex-col items-center gap-2">
      {open ? (
        <div
          ref={scrollWrapRef}
          className="flex max-h-[46vh] w-[min(720px,90vw)] flex-col overflow-hidden rounded-xl border border-border bg-card/90 shadow-[0_12px_32px_rgba(16,24,40,0.14)] backdrop-blur-md"
        >
          <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border px-3 py-2">
            <span className="label-xs">Run log</span>
            <span className="font-mono text-[10px] tabular-nums text-muted-foreground">
              {items.length} {items.length === 1 ? "attempt" : "attempts"}
            </span>
          </div>
          <ScrollArea className="min-h-0 flex-1">
            <div className="flex flex-col gap-1.5 p-2.5">
              {items.length ? (
                items.map((iteration, index) => <RunLogCard key={index} iteration={iteration} />)
              ) : (
                <p className="px-1 py-6 text-center text-[11px] text-muted-foreground">
                  Waiting for the first attempt…
                </p>
              )}
            </div>
          </ScrollArea>
        </div>
      ) : null}

      {/* ticker chip — the live phase + Stop while busy, always the Log toggle */}
      <div className="flex items-center gap-3 rounded-xl border border-border bg-card/90 px-3 py-2 shadow-[0_12px_32px_rgba(16,24,40,0.10)] backdrop-blur-md">
        {busy ? (
          <>
            <span className="size-2 shrink-0 animate-pulse rounded-full bg-primary" aria-hidden />
            <p className="max-w-[360px] truncate text-[12px] text-foreground">{studio.phaseText || "Working…"}</p>
            <span className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">{studio.elapsed}s</span>
          </>
        ) : (
          <>
            <span className="size-2 shrink-0 rounded-full bg-muted-foreground/50" aria-hidden />
            <p className="text-[12px] text-muted-foreground">Run log</p>
          </>
        )}
        <Button
          size="sm"
          variant="ghost"
          className="h-7 shrink-0 gap-1.5 px-2.5 text-[11px] font-medium"
          aria-pressed={open}
          onClick={() => studio.setRunLogOpen(!open)}
        >
          <ChevronUp className={cn("size-3.5 transition-transform", open && "rotate-180")} />
          Log
        </Button>
        {busy ? (
          <Button
            size="sm"
            variant="destructive"
            className="h-7 shrink-0 gap-1.5 px-2.5 text-[11px] font-medium"
            onClick={() => void studio.stopGeneration()}
          >
            <StopCircle className="size-3.5" /> Stop
          </Button>
        ) : null}
      </div>
    </div>
  );
}
