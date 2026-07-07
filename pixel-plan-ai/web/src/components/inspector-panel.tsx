"use client";

import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import type { Studio } from "@/hooks/use-studio";
import type { GenerationIteration } from "@/lib/types";
import { prettyType } from "@/lib/types";

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

  return (
    <Tabs defaultValue="inspect" className="flex h-full min-h-0 flex-col gap-0">
      <div className="border-b px-2 py-2">
        <TabsList className="grid h-8 w-full grid-cols-4">
          {["inspect", "code", "data", "log"].map((tab) => (
            <TabsTrigger key={tab} value={tab} className="text-[10px] font-bold uppercase tracking-wide">
              {tab}
            </TabsTrigger>
          ))}
        </TabsList>
      </div>

      <TabsContent value="inspect" className="mt-0 min-h-0 flex-1">
        <ScrollArea className="h-full">
          <div className="flex flex-col gap-4 p-3">
            <div className="rounded-lg border bg-muted/30 p-3">
              <div className="flex items-end justify-between">
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-muted-foreground">Plan score</p>
                  <p className="mt-1 text-3xl font-bold tabular-nums leading-none">
                    {validation ? validation.score : "—"}
                  </p>
                </div>
                {studio.result?.accepted === false ? (
                  <Badge variant="destructive" className="text-[10px]">NEEDS REVISION</Badge>
                ) : validation ? (
                  <Badge className="bg-emerald-600 text-[10px] hover:bg-emerald-600">ACCEPTED</Badge>
                ) : null}
              </div>
              {validation ? (
                <p className="mt-2 text-[11px] text-muted-foreground">
                  {validation.summary.generated_rooms}/{validation.summary.expected_rooms} rooms ·{" "}
                  {validation.summary.satisfied_adjacencies}/{validation.summary.required_adjacencies} adjacencies ·{" "}
                  {validation.summary.coverage_percent}% coverage
                </p>
              ) : null}
            </div>

            <section>
              <SectionTitle
                title="Validation"
                meta={validation ? `${validation.checks.filter((check) => check.pass).length}/${validation.checks.length} PASS` : "—"}
              />
              <div className="mt-2 flex flex-col gap-1">
                {validation?.checks.slice(0, 30).map((check, index) => (
                  <div key={index} className="flex items-start gap-2 text-[11px] leading-snug">
                    <span className={`mt-px font-bold ${check.pass ? "text-emerald-600" : "text-destructive"}`}>
                      {check.pass ? "✓" : "!"}
                    </span>
                    <span className={check.pass ? "text-foreground/80" : "text-destructive"}>{check.label}</span>
                  </div>
                )) ?? <EmptyCopy>Checks appear after generation.</EmptyCopy>}
              </div>
            </section>

            <section>
              <SectionTitle title="Legend" meta={plan ? `${plan.rooms.length} SPACES` : "0"} />
              <div className="mt-2 flex flex-col gap-1">
                {legend.length ? (
                  legend.map(([type, item]) => (
                    <div key={type} className="flex items-center gap-2 text-[11px]">
                      <span className="size-3 shrink-0 rounded-[3px] border border-black/10" style={{ background: item.color }} />
                      <span className="flex-1 capitalize">{prettyType(type)}</span>
                      <span className="font-mono text-muted-foreground">×{item.count}</span>
                    </div>
                  ))
                ) : (
                  <EmptyCopy>No program data yet.</EmptyCopy>
                )}
              </div>
            </section>

            <section>
              <SectionTitle title="Doors" meta={plan ? String(plan.doors.length) : "0"} />
              <div className="mt-2 flex flex-col gap-1.5">
                {plan?.doors.slice(0, 28).map((door) => (
                  <div key={door.id} className="text-[11px] leading-snug">
                    <b className="font-mono">{door.id}</b>
                    <p className="text-muted-foreground">
                      {door.from_room} → {door.to_room ?? "OUTSIDE"} · swings {door.swing_side} @ {door.x},{door.y}
                    </p>
                  </div>
                )) ?? <EmptyCopy>No doors yet.</EmptyCopy>}
              </div>
            </section>
          </div>
        </ScrollArea>
      </TabsContent>

      <TabsContent value="code" className="mt-0 flex min-h-0 flex-1 flex-col">
        <div className="flex items-center justify-between gap-2 border-b px-3 py-2">
          <span className="truncate font-mono text-[10px] text-muted-foreground">{studio.codeSource}</span>
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="outline"
              className="h-6 px-2 text-[10px] font-bold"
              disabled={!!studio.busyAction || !studio.code.trim()}
              onClick={() => void studio.runAgentAction("run")}
            >
              RUN
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-6 px-2 text-[10px] font-bold"
              disabled={!!studio.busyAction || !studio.code.trim()}
              onClick={() => void studio.runAgentAction("inspect")}
            >
              INSPECT
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-6 px-2 text-[10px] font-bold"
              disabled={!!studio.busyAction || !studio.code.trim()}
              onClick={() => void studio.runAgentAction("fix")}
            >
              FIX AI
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-6 px-2 text-[10px] font-bold"
              disabled={!studio.code.trim()}
              onClick={() => void navigator.clipboard.writeText(studio.code)}
            >
              COPY
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
        <div className="flex flex-col gap-1.5 border-t p-2.5">
          <Textarea
            value={studio.revision}
            onChange={(event) => studio.setRevision(event.target.value)}
            placeholder="Revision instruction, e.g. move the main entrance to the south."
            className="h-14 resize-none text-[11px]"
          />
          <Button
            size="sm"
            className="h-7 text-[10px] font-bold tracking-wide"
            disabled={!!studio.busyAction || !studio.code.trim() || !studio.revision.trim()}
            onClick={() => void studio.runAgentAction("revise")}
          >
            REVISE WITH AI
          </Button>
          {studio.result?.inspection ? (
            <p className={`text-[11px] ${studio.result.inspection.accepted ? "text-emerald-700" : "text-destructive"}`}>
              {studio.result.inspection.summary}
            </p>
          ) : null}
        </div>
      </TabsContent>

      <TabsContent value="data" className="mt-0 min-h-0 flex-1">
        <ScrollArea className="h-full">
          <div className="flex flex-col gap-4 p-3">
            <section>
              <SectionTitle title="Metadata" meta="" />
              <div className="mt-2 grid grid-cols-2 gap-1.5">
                {studio.result ? (
                  metadataRows(studio).map(([label, value]) => (
                    <div key={label} className="rounded-md border bg-muted/30 px-2 py-1.5">
                      <p className="text-[9px] font-bold uppercase tracking-wide text-muted-foreground">{label}</p>
                      <p className="truncate font-mono text-[11px]">{value}</p>
                    </div>
                  ))
                ) : (
                  <EmptyCopy>No data.</EmptyCopy>
                )}
              </div>
            </section>
            <section>
              <SectionTitle title="Area report" meta="" />
              <div className="mt-2 overflow-x-auto">
                <table className="w-full text-[10.5px]">
                  <thead>
                    <tr className="border-b text-left text-[9px] uppercase tracking-wide text-muted-foreground">
                      <th className="py-1 pr-2">Space</th>
                      <th className="py-1 pr-2">Target</th>
                      <th className="py-1 pr-2">Actual</th>
                      <th className="py-1">Err</th>
                    </tr>
                  </thead>
                  <tbody>
                    {studio.result?.validation.areas.map((row) => (
                      <tr key={row.id} className="border-b border-border/50">
                        <td className="py-1 pr-2 font-mono">{row.id}</td>
                        <td className="py-1 pr-2 tabular-nums">{row.target_ft2 ?? Math.round(row.target_m2 * 10.7639)} sf</td>
                        <td className="py-1 pr-2 tabular-nums">{row.actual_ft2 ?? Math.round(row.actual_m2 * 10.7639)} sf</td>
                        <td className={`py-1 tabular-nums ${row.pass ? "text-emerald-700" : "text-destructive"}`}>
                          {row.error_percent}%
                        </td>
                      </tr>
                    )) ?? null}
                  </tbody>
                </table>
                {!studio.result ? <EmptyCopy>No area data.</EmptyCopy> : null}
              </div>
            </section>
          </div>
        </ScrollArea>
      </TabsContent>

      <TabsContent value="log" className="mt-0 min-h-0 flex-1">
        <ScrollArea className="h-full">
          <div className="flex flex-col gap-2 p-3">
            {studio.iterations.length ? (
              studio.iterations.map((iteration, index) => <IterationCard key={index} iteration={iteration} />)
            ) : (
              <EmptyCopy>Generate a plan to inspect its attempts.</EmptyCopy>
            )}
          </div>
        </ScrollArea>
      </TabsContent>
    </Tabs>
  );
}

function SectionTitle({ title, meta }: { title: string; meta: string }) {
  return (
    <div className="flex items-center justify-between border-b pb-1.5">
      <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-muted-foreground">{title}</span>
      <span className="font-mono text-[10px] text-muted-foreground">{meta}</span>
    </div>
  );
}

function EmptyCopy({ children }: { children: React.ReactNode }) {
  return <p className="text-[11px] text-muted-foreground">{children}</p>;
}

function metadataRows(studio: Studio): [string, string][] {
  const result = studio.result;
  if (!result) return [];
  return [
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
}

function IterationCard({ iteration }: { iteration: GenerationIteration }) {
  const statusColor =
    iteration.status === "accepted"
      ? "text-emerald-700"
      : iteration.status === "running"
        ? "text-primary"
        : "text-destructive";
  const usage = iteration.usage;
  return (
    <article className="rounded-lg border p-2.5">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[10px] font-bold">ATTEMPT {String(iteration.attempt).padStart(2, "0")}</span>
        <span className={`text-[10px] font-bold uppercase ${statusColor}`}>{iteration.status}</span>
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
      <div className="mt-1.5 flex gap-3 font-mono text-[10px] text-muted-foreground">
        <span>score {iteration.score ?? "—"}</span>
        <span>{(iteration.duration_ms / 1000).toFixed(1)}s</span>
        <span>{usage ? `${usage.total_tokens.toLocaleString()} tok` : "— tok"}</span>
        <span>{usage?.estimated_cost_usd == null ? "$—" : `$${usage.estimated_cost_usd.toFixed(3)}`}</span>
      </div>
    </article>
  );
}
