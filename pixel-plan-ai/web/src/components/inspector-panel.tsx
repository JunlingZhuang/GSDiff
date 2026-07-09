"use client";

import * as React from "react";
import { Check, ChevronDown, CircleCheck, CircleX, Code2, Gauge, History, Loader2, Table2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import type { Studio } from "@/hooks/use-studio";
import { roomDisplayName } from "@/lib/render";
import type { GenerationIteration } from "@/lib/types";
import { prettyType } from "@/lib/types";
import { cn } from "@/lib/utils";

function scoreDot(score: number): string {
  if (score >= 90) return "bg-success ring-success/25";
  if (score >= 72) return "bg-warning ring-warning/25";
  return "bg-destructive ring-destructive/25";
}

function scoreChipTone(score: number): string {
  if (score >= 90) return "border-success/40 bg-success/10 text-success";
  if (score >= 72) return "border-warning/40 bg-warning/10 text-warning";
  return "border-destructive/40 bg-destructive/10 text-destructive";
}

function Disclosure({
  label,
  meta,
  defaultOpen = false,
  children,
}: {
  label: React.ReactNode;
  meta?: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  return (
    <Collapsible defaultOpen={defaultOpen}>
      <CollapsibleTrigger className="group flex w-full items-center justify-between gap-2 py-1 text-left outline-none">
        <span className="flex items-center gap-1.5">{label}</span>
        <span className="flex items-center gap-1.5">
          {meta}
          <ChevronDown className="size-3.5 text-muted-foreground transition-transform group-data-[panel-open]:rotate-180" />
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent>{children}</CollapsibleContent>
    </Collapsible>
  );
}

function SectionHeader({ title, meta }: { title: string; meta?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-border pb-1.5">
      <span className="label-xs">{title}</span>
      {meta ? <span className="font-mono text-[11px] tabular-nums text-muted-foreground">{meta}</span> : null}
    </div>
  );
}

export function InspectorPanel({ studio }: { studio: Studio }) {
  const plan = studio.activePlan;
  const validation = studio.activeValidation;

  const legend = React.useMemo(() => {
    if (!plan) return [];
    const groups = new Map<string, { count: number; color: string }>();
    for (const room of plan.rooms) {
      const item = groups.get(room.type) ?? { count: 0, color: room.color };
      item.count += 1;
      groups.set(room.type, item);
    }
    return [...groups.entries()];
  }, [plan]);

  const failedChecks = validation?.checks.filter((check) => !check.pass) ?? [];
  const passedChecks = validation?.checks.filter((check) => check.pass) ?? [];
  const needsRevision = studio.result?.accepted === false;

  const selectedRoom = React.useMemo(() => {
    if (!plan || !studio.selectedRoomId) return null;
    const room = plan.rooms.find((item) => item.id === studio.selectedRoomId);
    if (!room) return null;
    return {
      room,
      areaFt2: Math.round(room.pixel_count * plan.meters_per_cell * plan.meters_per_cell * 10.7639),
    };
  }, [plan, studio.selectedRoomId]);

  return (
    <Tabs defaultValue="inspect" className="flex h-full min-h-0 flex-col gap-0">
      <div className="border-b border-border p-2">
        <TabsList className="grid h-8 w-full grid-cols-4">
          <TabsTrigger value="inspect" className="gap-1 text-[11px] font-medium">
            <Gauge className="size-3.5" /> Inspect
          </TabsTrigger>
          <TabsTrigger value="code" className="gap-1 text-[11px] font-medium">
            <Code2 className="size-3.5" /> Code
          </TabsTrigger>
          <TabsTrigger value="data" className="gap-1 text-[11px] font-medium">
            <Table2 className="size-3.5" /> Data
          </TabsTrigger>
          <TabsTrigger value="log" className="gap-1 text-[11px] font-medium">
            <History className="size-3.5" /> Log
          </TabsTrigger>
        </TabsList>
      </div>

      <TabsContent value="inspect" className="mt-0 min-h-0 flex-1">
        <ScrollArea className="h-full">
          <div className="flex flex-col gap-4 p-3">
            {selectedRoom ? (
              <div className="flex items-center gap-2 rounded-lg border border-primary/40 bg-primary/5 px-2.5 py-2">
                <span
                  className="size-3 shrink-0 rounded-[3px] border border-black/20"
                  style={{ background: selectedRoom.room.color }}
                  aria-hidden
                />
                <div className="flex min-w-0 flex-1 flex-col">
                  <span className="truncate text-[12px] font-medium text-foreground">
                    {roomDisplayName(selectedRoom.room.id)}
                  </span>
                  <span className="truncate text-[11px] capitalize text-muted-foreground">
                    {prettyType(selectedRoom.room.type)}
                  </span>
                </div>
                <span className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">
                  {selectedRoom.areaFt2} ft²
                </span>
              </div>
            ) : null}

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
                    {validation.summary.generated_rooms}/{validation.summary.expected_rooms} rooms
                  </span>
                  <span className="rounded-md border border-border bg-secondary/40 px-1.5 py-0.5">
                    {validation.summary.satisfied_adjacencies}/{validation.summary.required_adjacencies} adj
                  </span>
                  <span className="rounded-md border border-border bg-secondary/40 px-1.5 py-0.5">
                    {validation.summary.coverage_percent}% cover
                  </span>
                </div>
              </div>
            ) : null}

            {validation ? (
              <section>
                <SectionHeader
                  title="Validation"
                  meta={`${passedChecks.length}/${validation.checks.length}`}
                />
                <div className="mt-2 flex flex-col gap-1">
                  {failedChecks.map((check, index) => (
                    <div key={index} className="flex items-start gap-2 text-[12px] leading-snug">
                      <CircleX className="mt-px size-3.5 shrink-0 text-destructive" />
                      <span>{check.label}</span>
                    </div>
                  ))}
                  {passedChecks.length ? (
                    <Disclosure
                      label={
                        <span className="flex items-center gap-1.5 text-[12px] text-muted-foreground">
                          <Check className="size-3.5 text-success" />
                          <span className="font-mono">{passedChecks.length}</span> passing
                        </span>
                      }
                    >
                      <div className="flex flex-col gap-1 pt-1">
                        {passedChecks.map((check, index) => (
                          <div key={index} className="flex items-start gap-2 text-[12px] leading-snug text-muted-foreground">
                            <Check className="mt-px size-3 shrink-0 text-success" />
                            <span>{check.label}</span>
                          </div>
                        ))}
                      </div>
                    </Disclosure>
                  ) : null}
                </div>
              </section>
            ) : null}

            {legend.length ? (
              <section>
                <SectionHeader title="Legend" meta={String(plan?.rooms.length ?? 0)} />
                <div className="mt-2 grid grid-cols-2 gap-1.5">
                  {legend.map(([type, item]) => (
                    <div key={type} className="flex items-center gap-1.5 text-[12px]">
                      <span
                        className="size-3 shrink-0 rounded-[3px] border border-black/20"
                        style={{ background: item.color }}
                      />
                      <span className="flex-1 truncate capitalize">{prettyType(type)}</span>
                      <span className="font-mono tabular-nums text-muted-foreground">{item.count}</span>
                    </div>
                  ))}
                </div>
              </section>
            ) : null}

            {plan && plan.doors.length ? (
              <section>
                <Disclosure
                  label={<span className="label-xs">Doors</span>}
                  meta={<span className="font-mono text-[11px] tabular-nums text-muted-foreground">{plan.doors.length}</span>}
                >
                  <div className="flex flex-col gap-1 pt-1.5">
                    {plan.doors.map((door) => (
                      <div key={door.id} className="flex items-center justify-between gap-2 text-[11px]">
                        <span className="font-mono tabular-nums">
                          {door.x},{door.y}
                        </span>
                        <span className="min-w-0 flex-1 truncate text-muted-foreground">
                          {door.from_room} → {door.to_room ?? "outside"}
                        </span>
                        <span className="font-mono text-muted-foreground">{door.swing_side}</span>
                      </div>
                    ))}
                  </div>
                </Disclosure>
              </section>
            ) : null}
          </div>
        </ScrollArea>
      </TabsContent>

      <TabsContent value="code" className="mt-0 flex min-h-0 flex-1 flex-col">
        <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
          <span className="truncate font-mono text-[10px] text-muted-foreground">{studio.codeSource}</span>
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="outline"
              className="h-6 px-2 text-[11px] font-medium"
              disabled={!!studio.busyAction || !studio.code.trim()}
              onClick={() => void studio.runAgentAction("run")}
            >
              Run
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-6 px-2 text-[11px] font-medium"
              disabled={!!studio.busyAction || !studio.code.trim()}
              onClick={() => void studio.runAgentAction("inspect")}
            >
              Inspect
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-6 px-2 text-[11px] font-medium"
              disabled={!!studio.busyAction || !studio.code.trim()}
              onClick={() => void studio.runAgentAction("fix")}
            >
              Fix
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-6 px-2 text-[11px] font-medium"
              disabled={!studio.code.trim()}
              onClick={() => void navigator.clipboard.writeText(studio.code)}
            >
              Copy
            </Button>
          </div>
        </div>
        <Textarea
          value={studio.code}
          onChange={(event) => studio.setCode(event.target.value)}
          spellCheck={false}
          placeholder="# Generate a plan to inspect and edit the code."
          className="min-h-0 flex-1 resize-none rounded-none border-0 font-mono text-[10.5px] leading-relaxed focus-visible:ring-0"
        />
        <div className="flex flex-col gap-1.5 border-t border-border p-2.5">
          <Textarea
            value={studio.revision}
            onChange={(event) => studio.setRevision(event.target.value)}
            placeholder="Revision instruction, e.g. move the main entrance to the south."
            className="h-14 resize-none text-[11px]"
          />
          <Button
            size="sm"
            className="h-7 text-[11px] font-medium"
            disabled={!!studio.busyAction || !studio.code.trim() || !studio.revision.trim()}
            onClick={() => void studio.runAgentAction("revise")}
          >
            Revise with AI
          </Button>
          {studio.result?.inspection ? (
            <p className={cn("text-[11px]", studio.result.inspection.accepted ? "text-success" : "text-destructive")}>
              {studio.result.inspection.summary}
            </p>
          ) : null}
        </div>
      </TabsContent>

      <TabsContent value="data" className="mt-0 min-h-0 flex-1">
        <ScrollArea className="h-full">
          <div className="flex flex-col gap-4 p-3">
            {studio.result ? (
              <section>
                <SectionHeader title="Metadata" />
                <div className="mt-2 grid grid-cols-2 gap-1.5">
                  {metadataRows(studio).map(([label, value]) => (
                    <div key={label} className="rounded-md border border-border bg-secondary/30 px-2 py-1.5">
                      <p className="label-xs">{label}</p>
                      <p className="truncate font-mono text-[11px] tabular-nums">{value}</p>
                    </div>
                  ))}
                </div>
              </section>
            ) : null}
            {studio.result?.validation.areas.length ? (
              <section>
                <SectionHeader title="Area report" />
                <div className="mt-2 overflow-x-auto">
                  <table className="w-full text-[10.5px]">
                    <thead>
                      <tr className="border-b border-border text-left">
                        <th className="label-xs py-1 pr-2">Space</th>
                        <th className="label-xs py-1 pr-2">Target</th>
                        <th className="label-xs py-1 pr-2">Actual</th>
                        <th className="label-xs py-1">Err</th>
                      </tr>
                    </thead>
                    <tbody>
                      {studio.result.validation.areas.map((row) => (
                        <tr key={row.id} className="border-b border-border/50">
                          <td className="py-1 pr-2 font-mono">{row.id}</td>
                          <td className="py-1 pr-2 font-mono tabular-nums">
                            {row.target_ft2 ?? Math.round(row.target_m2 * 10.7639)} sf
                          </td>
                          <td className="py-1 pr-2 font-mono tabular-nums">
                            {row.actual_ft2 ?? Math.round(row.actual_m2 * 10.7639)} sf
                          </td>
                          <td className={cn("py-1 font-mono tabular-nums", row.pass ? "text-success" : "text-destructive")}>
                            {row.error_percent}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            ) : null}
          </div>
        </ScrollArea>
      </TabsContent>

      <TabsContent value="log" className="mt-0 min-h-0 flex-1">
        <ScrollArea className="h-full">
          <div className="flex flex-col gap-2 p-3">
            {studio.iterations.map((iteration, index) => (
              <IterationCard key={index} iteration={iteration} />
            ))}
          </div>
        </ScrollArea>
      </TabsContent>
    </Tabs>
  );
}

function metadataRows(studio: Studio): [string, string][] {
  const result = studio.result;
  if (!result) return [];
  const rows: [string, string][] = [
    ["Source", result.source],
    ["Model", result.model ?? "None"],
    ["Score", `${result.validation.score}/100`],
    ["Grid", `${result.plan.width} × ${result.plan.height}`],
    ["Scale", `${result.plan.meters_per_cell.toFixed(3)} m/cell`],
    ["Rooms", String(result.plan.rooms.length)],
    ["Doors", String(result.plan.doors.length)],
    ["Coverage", `${result.validation.summary.coverage_percent}%`],
    ["Rules", result.validation.summary.rule_profile ?? "generic-schematic"],
    ["Reference", studio.reference ? `candidate ${studio.reference.index + 1}` : "none"],
  ];
  if (result.stop_reason) {
    rows.push(["Stop reason", result.stop_reason.replaceAll("_", " ")]);
  }
  if (result.usage_total) {
    const cost = result.usage_total.estimated_cost_usd;
    rows.push([
      "Run cost",
      `${result.usage_total.total_tokens.toLocaleString()} tok · ${cost == null ? "$—" : `$${cost.toFixed(3)}`}`,
    ]);
  }
  return rows;
}

function IterationCard({ iteration }: { iteration: GenerationIteration }) {
  const StatusIcon =
    iteration.status === "accepted" ? CircleCheck : iteration.status === "running" ? Loader2 : CircleX;
  const statusColor =
    iteration.status === "accepted"
      ? "text-success"
      : iteration.status === "running"
        ? "text-primary"
        : "text-destructive";
  const usage = iteration.usage;
  return (
    <article className="rounded-lg border border-border p-2.5">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5">
          <StatusIcon className={cn("size-3.5", statusColor, iteration.status === "running" && "animate-spin")} />
          <span className="font-mono text-[11px] tabular-nums">attempt {String(iteration.attempt).padStart(2, "0")}</span>
        </span>
        {iteration.score !== null ? (
          <span
            className={cn(
              "rounded-md border px-1.5 py-0.5 font-mono text-[10px] tabular-nums",
              scoreChipTone(iteration.score),
            )}
          >
            {iteration.score}
          </span>
        ) : null}
      </div>
      <p className="mt-1 text-[11px] leading-snug text-muted-foreground">
        {iteration.phase} · {iteration.model ?? iteration.source} · {iteration.message}
      </p>
      {iteration.issues.length ? (
        <ul className="mt-1 list-inside list-disc text-[10.5px] text-destructive/90">
          {iteration.issues.slice(0, 4).map((issue, index) => (
            <li key={index}>{issue}</li>
          ))}
        </ul>
      ) : null}
      <div className="mt-1.5 flex gap-3 font-mono text-[10px] tabular-nums text-muted-foreground">
        <span>{(iteration.duration_ms / 1000).toFixed(1)}s</span>
        <span>{usage ? `${usage.total_tokens.toLocaleString()} tok` : "— tok"}</span>
        <span>{usage?.estimated_cost_usd == null ? "$—" : `$${usage.estimated_cost_usd.toFixed(3)}`}</span>
      </div>
    </article>
  );
}
