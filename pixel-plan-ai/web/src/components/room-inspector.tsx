"use client";

import * as React from "react";
import { Check, ChevronDown, CircleX } from "lucide-react";

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { RoomStudio } from "@/hooks/use-room-studio";
import { prettyType } from "@/lib/types";
import { cn } from "@/lib/utils";

function scoreDot(score: number): string {
  if (score >= 90) return "bg-success ring-success/25";
  if (score >= 72) return "bg-warning ring-warning/25";
  return "bg-destructive ring-destructive/25";
}

function SectionHeader({ title, meta }: { title: string; meta?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-border pb-1.5">
      <span className="label-xs">{title}</span>
      {meta ? <span className="font-mono text-[11px] tabular-nums text-muted-foreground">{meta}</span> : null}
    </div>
  );
}

// Shared column template so the placements header and rows stay aligned.
const ASSET_GRID = "grid grid-cols-[minmax(0,1fr)_2.5rem_2.5rem_2.5rem] items-center gap-1.5";

function ft(value: number): string {
  // Trim trailing zeros on the 0.25 grid (8.25, 16, 0.75).
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)));
}

export function RoomInspector({ room }: { room: RoomStudio }) {
  const layout = room.activeRoom;
  const validation = room.activeValidation;
  const needsRevision = room.result?.accepted === false;

  const failedChecks = validation?.checks.filter((check) => !check.pass) ?? [];
  const passedChecks = validation?.checks.filter((check) => check.pass) ?? [];

  const selected = room.selectedAssetId;

  return (
    <ScrollArea className="h-full">
      <div className="flex flex-col gap-4 p-3">
        {validation ? (
          <div className="flex flex-col gap-2 rounded-lg border border-border bg-secondary/30 p-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className={cn("size-2.5 rounded-full ring-4", scoreDot(validation.score))} aria-hidden />
                <span className="font-mono text-2xl leading-none tabular-nums">{validation.score}</span>
              </div>
              <span
                className={cn(
                  "rounded-md border px-2 py-0.5 text-[11px]",
                  needsRevision
                    ? "border-destructive/40 bg-destructive/10 text-destructive"
                    : "border-success/40 bg-success/10 text-success",
                )}
              >
                {needsRevision ? "needs revision" : "accepted"}
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5 font-mono text-[11px] tabular-nums text-muted-foreground">
              <span className="rounded-md border border-border bg-secondary/40 px-1.5 py-0.5">
                {validation.summary.checks_passed}/{validation.summary.checks_total} checks
              </span>
              <span className="rounded-md border border-border bg-secondary/40 px-1.5 py-0.5">
                {validation.summary.assets_placed} assets
              </span>
              {layout ? (
                <span className="rounded-md border border-border bg-secondary/40 px-1.5 py-0.5">
                  {ft(layout.room.width_ft)}×{ft(layout.room.depth_ft)} ft
                </span>
              ) : null}
            </div>
          </div>
        ) : null}

        {validation ? (
          <section>
            <SectionHeader title="Validation" meta={`${passedChecks.length}/${validation.checks.length}`} />
            <div className="mt-2 flex flex-col gap-1">
              {failedChecks.map((check, index) => (
                <div key={index} className="flex items-start gap-2 text-[12px] leading-snug">
                  <CircleX className="mt-px size-3.5 shrink-0 text-destructive" />
                  <span>{check.label}</span>
                </div>
              ))}
              {passedChecks.length ? (
                <Collapsible defaultOpen={failedChecks.length === 0}>
                  <CollapsibleTrigger className="group flex w-full items-center justify-between gap-2 py-1 text-left outline-none">
                    <span className="flex items-center gap-1.5 text-[12px] text-muted-foreground">
                      <Check className="size-3.5 text-success" />
                      <span className="font-mono">{passedChecks.length}</span> passing
                    </span>
                    <ChevronDown className="size-3.5 text-muted-foreground transition-transform group-data-[panel-open]:rotate-180" />
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="flex flex-col gap-1 pt-1">
                      {passedChecks.map((check, index) => (
                        <div
                          key={index}
                          className="flex items-start gap-2 text-[12px] leading-snug text-muted-foreground"
                        >
                          <Check className="mt-px size-3 shrink-0 text-success" />
                          <span>{check.label}</span>
                        </div>
                      ))}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              ) : null}
            </div>
          </section>
        ) : null}

        {layout && layout.assets.length ? (
          <section>
            <SectionHeader title="Placements" meta={String(layout.assets.length)} />
            <div className={cn(ASSET_GRID, "mt-2 border-b border-border pb-1")}>
              <span className="label-xs">Type</span>
              <span className="label-xs text-right">X</span>
              <span className="label-xs text-right">Y</span>
              <span className="label-xs text-right">Rot</span>
            </div>
            <div className="flex flex-col">
              {layout.assets.map((asset) => (
                <button
                  key={asset.id}
                  type="button"
                  onClick={() =>
                    room.setSelectedAssetId(selected === asset.id ? null : asset.id)
                  }
                  className={cn(
                    ASSET_GRID,
                    "px-1 py-1 text-left transition-colors hover:bg-secondary/40",
                    selected === asset.id && "bg-primary/5",
                  )}
                >
                  <span className="flex min-w-0 flex-col">
                    <span className="truncate text-[12px] capitalize text-foreground">
                      {prettyType(asset.type)}
                    </span>
                    <span className="truncate font-mono text-[10px] tabular-nums text-muted-foreground">
                      {ft(asset.w_ft)}×{ft(asset.d_ft)} ft{asset.wall ? ` · ${asset.wall}` : ""}
                    </span>
                  </span>
                  <span className="text-right font-mono text-[11px] tabular-nums text-muted-foreground">
                    {ft(asset.x_ft)}
                  </span>
                  <span className="text-right font-mono text-[11px] tabular-nums text-muted-foreground">
                    {ft(asset.y_ft)}
                  </span>
                  <span className="text-right font-mono text-[11px] tabular-nums text-muted-foreground">
                    {asset.rotation_deg}°
                  </span>
                </button>
              ))}
            </div>
          </section>
        ) : null}

        {room.result ? (
          <section>
            <SectionHeader title="Run" />
            <div className="mt-2 grid grid-cols-2 gap-1.5">
              {runRows(room).map(([label, value]) => (
                <div key={label} className="rounded-md border border-border bg-secondary/30 px-2 py-1.5">
                  <p className="label-xs">{label}</p>
                  <p className="truncate font-mono text-[11px] tabular-nums">{value}</p>
                </div>
              ))}
            </div>
          </section>
        ) : null}
      </div>
    </ScrollArea>
  );
}

function runRows(room: RoomStudio): [string, string][] {
  const result = room.result;
  if (!result) return [];
  const rows: [string, string][] = [
    ["Source", result.source],
    ["Model", result.model ?? "None"],
    ["Score", `${result.validation.score}/100`],
    ["Attempts", String(result.iterations.length)],
  ];
  if (result.stop_reason) rows.push(["Stop reason", result.stop_reason.replaceAll("_", " ")]);
  if (result.usage_total) {
    const cost = result.usage_total.estimated_cost_usd;
    rows.push([
      "Run cost",
      `${result.usage_total.total_tokens.toLocaleString()} tok · ${cost == null ? "$—" : `$${cost.toFixed(3)}`}`,
    ]);
  }
  return rows;
}
