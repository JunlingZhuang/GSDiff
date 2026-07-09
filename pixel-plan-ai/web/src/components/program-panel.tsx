"use client";

import * as React from "react";
import {
  ArrowLeftRight,
  Braces,
  Check,
  ChevronDown,
  CircleAlert,
  CircleCheck,
  Image as ImageIcon,
  ImagePlus,
  Layers,
  Loader2,
  Plus,
  Route,
  Sparkles,
  X,
} from "lucide-react";

import { LightboxViewer } from "@/components/lightbox-viewer";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import type { Studio } from "@/hooks/use-studio";
import type { CandidateImage, Program, StudioMode } from "@/lib/types";
import { candidateDataUrl, prettyType } from "@/lib/types";
import { cn } from "@/lib/utils";

// Shared grid template for the rooms table (header + rows stay column-aligned):
// Type takes the remaining width, Count / ft² are fixed mono columns, and the
// trailing 1.25rem cell holds the hover-only remove button.
const ROOM_GRID = "grid grid-cols-[minmax(0,1fr)_3rem_4rem_1.25rem] items-center gap-1.5";

function Chip({
  children,
  tone = "default",
}: {
  children: React.ReactNode;
  tone?: "default" | "accent";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border border-border bg-secondary/40 px-1.5 py-0.5 text-[11px]",
        tone === "accent" ? "text-foreground" : "text-muted-foreground",
      )}
    >
      {children}
    </span>
  );
}

function Num({ children }: { children: React.ReactNode }) {
  return <span className="font-mono text-foreground">{children}</span>;
}

function Disclosure({
  open,
  onOpenChange,
  label,
  meta,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  label: React.ReactNode;
  meta?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Collapsible open={open} onOpenChange={onOpenChange}>
      <CollapsibleTrigger className="group flex w-full items-center justify-between gap-2 py-1 text-left outline-none">
        <span className="flex items-center gap-2">{label}</span>
        <span className="flex items-center gap-1.5">
          {meta}
          <ChevronDown className="size-3.5 text-muted-foreground transition-transform group-data-[panel-open]:rotate-180" />
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent>{children}</CollapsibleContent>
    </Collapsible>
  );
}

// Draw + candidate grid — shared by image mode and trace step 1 (selection state
// lives in the studio hook, so both surfaces pick the same drawing).
function CandidateStrip({ studio }: { studio: Studio }) {
  return (
    <div className="flex flex-col gap-2">
      <Button
        size="sm"
        variant="outline"
        className="h-8 w-full gap-1.5 text-[12px]"
        disabled={studio.candidatesBusy || !!studio.busyAction || !studio.program}
        onClick={() => void studio.drawCandidates()}
      >
        <ImagePlus className="size-3.5" />
        {studio.candidatesBusy ? "Drawing…" : "Draw 3 candidates"}
      </Button>
      {studio.candidates.length || studio.candidatesBusy ? (
        <div className="flex flex-wrap gap-1.5">
          {studio.candidatesBusy ? (
            <Chip>drawing…</Chip>
          ) : (
            <Chip>
              <Num>{studio.candidates.length}</Num> drawings
            </Chip>
          )}
          {studio.selectedCandidate >= 0 ? (
            <Chip tone="accent">
              #<Num>{studio.selectedCandidate + 1}</Num> selected
            </Chip>
          ) : null}
        </div>
      ) : null}
      <div className="grid grid-cols-3 gap-1.5">
        {studio.candidatesBusy
          ? [0, 1, 2].map((index) => <Skeleton key={index} className="aspect-video w-full rounded-md" />)
          : studio.candidates.map((candidate, index) => (
              <button
                key={index}
                type="button"
                className={cn(
                  "relative overflow-hidden rounded-md border-2 bg-white transition-shadow",
                  index === studio.selectedCandidate
                    ? "border-primary shadow-[0_0_0_1px_var(--primary)]"
                    : "border-border hover:border-foreground/40",
                )}
                onClick={() =>
                  index === studio.selectedCandidate
                    ? studio.setLightbox(index)
                    : studio.selectCandidate(index)
                }
              >
                {/* contain, never crop: the user must see the whole drawing */}
                <img
                  src={candidateDataUrl(candidate)}
                  alt={`Candidate plan ${index + 1}`}
                  className="aspect-video w-full object-contain"
                />
                {index === studio.selectedCandidate ? (
                  <span className="absolute right-0.5 top-0.5 rounded-full bg-primary p-0.5 text-primary-foreground">
                    <Check className="size-2.5" />
                  </span>
                ) : null}
              </button>
            ))}
      </div>
    </div>
  );
}

// One row of the trace pipeline stepper: a numbered chip-row header (a check when
// complete) plus expandable content; a completed step collapses to a summary row.
function Step({
  index,
  title,
  state,
  open,
  onToggle,
  summary,
  children,
}: {
  index: number;
  title: string;
  state: "done" | "active" | "pending";
  open: boolean;
  onToggle: () => void;
  summary?: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border",
        state === "pending" ? "border-border/60 bg-secondary/20" : "border-border bg-secondary/30",
      )}
    >
      <button
        type="button"
        onClick={onToggle}
        disabled={state === "pending"}
        className="flex w-full items-center gap-2 px-2 py-1.5 text-left outline-none disabled:cursor-not-allowed disabled:opacity-70"
      >
        <span
          className={cn(
            "flex size-4 shrink-0 items-center justify-center rounded-full font-mono text-[10px]",
            state === "done"
              ? "bg-primary text-primary-foreground"
              : state === "active"
                ? "border border-primary text-primary"
                : "border border-border text-muted-foreground",
          )}
        >
          {state === "done" ? <Check className="size-2.5" /> : index}
        </span>
        <span className="label-xs flex-1">{title}</span>
        {!open ? summary : null}
      </button>
      {open && children ? <div className="border-t border-border/60 px-2 py-2">{children}</div> : null}
    </div>
  );
}

// The building-type field lives in the fixed panel header (it names what you're
// editing), so it is split out of the scrolling ProgramEditor body. It writes
// through the same programText single-source-of-truth as every other edit.
function BuildingInput({ studio }: { studio: Studio }) {
  const program = studio.program;
  return (
    <Input
      value={program?.building_type ?? ""}
      disabled={!program}
      onChange={(event) => {
        if (!program) return;
        const next = JSON.parse(JSON.stringify(program)) as Program;
        next.building_type = event.target.value;
        studio.setProgramText(JSON.stringify(next, null, 2));
      }}
      placeholder="e.g. outpatient clinic"
      className="h-8 text-[13px]"
    />
  );
}

// The human-facing program editor: a rooms table (type / count / ft²) plus an
// adjacency chip editor. `programText` (the JSON string on the studio hook) stays
// the single source of truth — every edit here clones it, applies one immutable
// mutation, and writes it back. When the JSON has been hand-edited into an
// invalid state the editor disables itself and points at the Advanced disclosure.
function ProgramEditor({ studio }: { studio: Studio }) {
  const program = studio.program;
  const [adjA, setAdjA] = React.useState("");
  const [adjB, setAdjB] = React.useState("");

  const editProgram = (mutate: (draft: Program) => Program): void => {
    if (!program) return;
    const next = mutate(JSON.parse(JSON.stringify(program)) as Program);
    studio.setProgramText(JSON.stringify(next, null, 2));
  };

  const rooms = program?.rooms ?? [];
  const adjacency = program?.adjacency ?? [];

  // Distinct room types drive the adjacency add-row Selects; cheap to derive on
  // each render (a program has a handful of rooms), so no memo is warranted.
  const roomTypes: string[] = [];
  for (const room of rooms) {
    const type = (room.type ?? "").trim();
    if (type && !roomTypes.includes(type)) roomTypes.push(type);
  }

  const programmedSpaces = rooms.reduce((total, room) => total + (Number(room.count) || 0), 0);
  const programmedFt2 = rooms.reduce(
    (total, room) => total + (Number(room.count) || 0) * (Number(room.approx_area_ft2) || 0),
    0,
  );

  const pairExists = (a: string, b: string): boolean =>
    adjacency.some(([x, y]) => (x === a && y === b) || (x === b && y === a));
  const canAddAdjacency = !!adjA && !!adjB && adjA !== adjB && !pairExists(adjA, adjB);

  const updateRoomType = (index: number, value: string) =>
    editProgram((draft) => ({
      ...draft,
      rooms: draft.rooms.map((room, i) => (i === index ? { ...room, type: value } : room)),
    }));

  const updateRoomCount = (index: number, raw: string) => {
    const digits = raw.replace(/[^\d]/g, "");
    editProgram((draft) => ({
      ...draft,
      rooms: draft.rooms.map((room, i) =>
        i === index ? { ...room, count: digits === "" ? 0 : Number(digits) } : room,
      ),
    }));
  };

  const updateRoomArea = (index: number, raw: string) => {
    const digits = raw.replace(/[^\d]/g, "");
    editProgram((draft) => ({
      ...draft,
      rooms: draft.rooms.map((room, i) => {
        if (i !== index) return room;
        if (digits === "") {
          // Blank ft² drops the field entirely — a preset row with no target area
          // must round-trip through the JSON without gaining a 0.
          const rest = { ...room };
          delete rest.approx_area_ft2;
          return rest;
        }
        return { ...room, approx_area_ft2: Number(digits) };
      }),
    }));
  };

  const removeRoom = (index: number) =>
    editProgram((draft) => ({ ...draft, rooms: draft.rooms.filter((_, i) => i !== index) }));

  const addRoom = () =>
    editProgram((draft) => ({ ...draft, rooms: [...(draft.rooms ?? []), { type: "room", count: 1 }] }));

  const removeAdjacency = (index: number) =>
    editProgram((draft) => ({ ...draft, adjacency: (draft.adjacency ?? []).filter((_, i) => i !== index) }));

  const addAdjacency = () => {
    if (!canAddAdjacency) return;
    editProgram((draft) => ({
      ...draft,
      adjacency: [...(draft.adjacency ?? []), [adjA, adjB] as [string, string]],
    }));
    setAdjA("");
    setAdjB("");
  };

  if (!program) {
    return (
      <p className="flex items-center gap-1.5 rounded-lg border border-dashed border-border bg-secondary/20 px-2.5 py-2 text-[11px] text-muted-foreground">
        <CircleAlert className="size-3 shrink-0 text-destructive" />
        JSON has errors — fix it under Advanced.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Rooms table — inline-editable, renders full-height inside the panel's
          single scroll context (no nested scrollbar of its own) */}
      <div className="flex flex-col gap-1.5">
        <div className="overflow-hidden rounded-lg border border-border">
          <div className={cn(ROOM_GRID, "border-b border-border bg-secondary/30 px-2 py-1.5")}>
            <span className="label-xs">Type</span>
            <span className="label-xs text-right">Count</span>
            <span className="label-xs text-right">ft²</span>
            <span />
          </div>
          <div className="flex flex-col">
            {rooms.length === 0 ? (
              <p className="px-2 py-3 text-center text-[11px] text-muted-foreground">No rooms yet.</p>
            ) : (
              rooms.map((room, index) => (
                <div key={index} className={cn(ROOM_GRID, "group px-2 py-1 hover:bg-secondary/30")}>
                  <Input
                    value={room.type ?? ""}
                    onChange={(event) => updateRoomType(index, event.target.value)}
                    className="h-7 px-2 text-[13px]"
                  />
                  <Input
                    inputMode="numeric"
                    value={room.count == null ? "" : String(room.count)}
                    onChange={(event) => updateRoomCount(index, event.target.value)}
                    className="h-7 px-1.5 text-right font-mono text-[12px] tabular-nums"
                  />
                  <Input
                    inputMode="numeric"
                    value={room.approx_area_ft2 == null ? "" : String(room.approx_area_ft2)}
                    onChange={(event) => updateRoomArea(index, event.target.value)}
                    placeholder="—"
                    className="h-7 px-1.5 text-right font-mono text-[12px] tabular-nums"
                  />
                  <button
                    type="button"
                    onClick={() => removeRoom(index)}
                    aria-label="Remove room"
                    className="flex size-5 items-center justify-center rounded-sm text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
                  >
                    <X className="size-3" />
                  </button>
                </div>
              ))
            )}
          </div>
          <div className="border-t border-border px-2 py-1">
            <button
              type="button"
              onClick={addRoom}
              className="flex items-center gap-1 rounded-sm py-0.5 text-[12px] text-muted-foreground hover:text-foreground"
            >
              <Plus className="size-3" /> Add room
            </button>
          </div>
        </div>
        <p className="font-mono text-[11px] tabular-nums text-muted-foreground">
          <span className="text-foreground">{programmedSpaces}</span> spaces ·{" "}
          <span className="text-foreground">{programmedFt2.toLocaleString("en-US")}</span> ft² programmed
        </p>
      </div>

      {/* Adjacency — chips with a remove ×, plus a two-Select add row */}
      <div className="flex flex-col gap-1.5">
        <span className="label-xs">Adjacency</span>
        <div className="flex flex-wrap gap-1.5">
          {adjacency.length === 0 ? (
            <span className="text-[11px] text-muted-foreground">No adjacencies yet.</span>
          ) : (
            adjacency.map((pair, index) => (
              <span
                key={`${pair[0]}-${pair[1]}-${index}`}
                className="inline-flex items-center gap-1 rounded-md border border-border bg-secondary/40 py-0.5 pr-1 pl-1.5 text-[11px] text-muted-foreground"
              >
                <span>{prettyType(pair[0] ?? "")}</span>
                <ArrowLeftRight className="size-2.5 shrink-0 opacity-70" />
                <span>{prettyType(pair[1] ?? "")}</span>
                <button
                  type="button"
                  onClick={() => removeAdjacency(index)}
                  aria-label="Remove adjacency"
                  className="ml-0.5 flex size-3.5 items-center justify-center rounded-sm hover:text-destructive"
                >
                  <X className="size-2.5" />
                </button>
              </span>
            ))
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <Select value={adjA} onValueChange={(value) => setAdjA(value as string)}>
            <SelectTrigger size="sm" className="h-7 flex-1 text-[12px]">
              <SelectValue placeholder="Room" />
            </SelectTrigger>
            <SelectContent>
              {roomTypes.map((type) => (
                <SelectItem key={type} value={type} className="text-[12px]">
                  {prettyType(type)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <ArrowLeftRight className="size-3 shrink-0 text-muted-foreground" />
          <Select value={adjB} onValueChange={(value) => setAdjB(value as string)}>
            <SelectTrigger size="sm" className="h-7 flex-1 text-[12px]">
              <SelectValue placeholder="Room" />
            </SelectTrigger>
            <SelectContent>
              {roomTypes.map((type) => (
                <SelectItem key={type} value={type} className="text-[12px]">
                  {prettyType(type)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            size="icon-sm"
            variant="outline"
            className="size-7 shrink-0"
            disabled={!canAddAdjacency}
            onClick={addAdjacency}
            aria-label="Add adjacency"
          >
            <Plus className="size-3.5" />
          </Button>
        </div>
      </div>
    </div>
  );
}

export function ProgramPanel({ studio }: { studio: Studio }) {
  const imageMode = studio.mode === "image";
  const traceMode = studio.mode === "trace";
  const [canvasOpen, setCanvasOpen] = React.useState(false);
  const [advancedOpen, setAdvancedOpen] = React.useState(false);
  const [openStep, setOpenStep] = React.useState(1);
  // Which image the trace gallery seeds the lightbox on (0 = drawing, 1 = trace).
  const [traceStart, setTraceStart] = React.useState(0);

  const trace = studio.traceResult;
  const drawingPicked = studio.selectedCandidate >= 0;
  const traced = !!trace;

  // The trace step presents the picked drawing and the traced linework as a
  // flippable image set. Only include images that actually exist.
  const traceDrawing = drawingPicked ? studio.candidates[studio.selectedCandidate] ?? null : null;
  const traceLinework = trace?.artifacts.linework ?? null;
  const traceImages = React.useMemo(() => {
    const list: { image: CandidateImage; title: string; caption: string }[] = [];
    if (traceDrawing) {
      list.push({
        image: traceDrawing,
        title: `Drawing — candidate ${studio.selectedCandidate + 1}`,
        caption: "drawing",
      });
    }
    if (traceLinework) list.push({ image: traceLinework, title: "Traced linework", caption: "trace" });
    return list;
  }, [traceDrawing, traceLinework, studio.selectedCandidate]);

  // Advance the stepper as each stage completes; headers still let the user jump back.
  React.useEffect(() => {
    if (drawingPicked) setOpenStep((step) => (step === 1 ? 2 : step));
  }, [drawingPicked]);
  React.useEffect(() => {
    if (traced) setOpenStep((step) => (step === 2 ? 3 : step));
  }, [traced]);

  const ftPerCell = studio.activePlan
    ? (studio.activePlan.meters_per_cell * 3.28084).toFixed(2)
    : null;

  const refining = studio.busyAction === "generate";
  const step1State = drawingPicked ? "done" : "active";
  const step2State = traced ? "done" : drawingPicked ? "active" : "pending";
  const step3State = studio.result && !studio.showingTracePreview ? "done" : studio.effectiveSeed ? "active" : "pending";
  const toggleStep = (index: number) => setOpenStep((step) => (step === index ? 0 : index));

  // The Advanced disclosure (JSON escape hatch + trace seed paste) is shared by
  // every mode and force-opens whenever an input is in an invalid, hand-editable
  // state so the fix is always one glance away.
  const advancedForced =
    !!studio.programError || (traceMode && (studio.tracerOnline === false || !!studio.seedError));
  const advancedOpenEffective = advancedOpen || advancedForced;

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* HEADER — fixed: preset picker + the building name that identifies what
          you're editing. Never scrolls. */}
      <div className="flex shrink-0 flex-col gap-2 border-b border-border px-3 py-2.5">
        <div className="flex flex-col gap-1.5">
          <span className="label-xs">Program</span>
          <Select value={studio.sampleKey} onValueChange={(value) => value && studio.loadSample(value)}>
            <SelectTrigger className="h-8 w-full text-[13px]">
              <SelectValue placeholder="Pick a sample" />
            </SelectTrigger>
            <SelectContent>
              {Object.keys(studio.samples).map((key) => (
                <SelectItem key={key} value={key} className="text-[13px]">
                  {key}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="label-xs">Building</span>
          <BuildingInput studio={studio} />
        </div>
      </div>

      {/* BODY — the single scroll context for the whole editor. */}
      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-4 p-3">
          {/* Human-facing program editor — rooms table + adjacency chips. The raw
              JSON escape hatch lives in Advanced at the bottom of the panel. */}
          <section>
            <ProgramEditor studio={studio} />
          </section>

          {/* mode */}
          <section className="flex flex-col gap-2">
            <span className="label-xs">Mode</span>
            <Tabs value={studio.mode} onValueChange={(value) => studio.setMode(value as StudioMode)}>
              <TabsList className="grid h-8 w-full grid-cols-3">
                <TabsTrigger value="program" className="gap-1.5 text-[11px] font-medium">
                  <Braces className="size-3.5" /> Program
                </TabsTrigger>
                <TabsTrigger value="image" className="gap-1.5 text-[11px] font-medium">
                  <ImageIcon className="size-3.5" /> Image
                </TabsTrigger>
                <TabsTrigger value="trace" className="gap-1.5 text-[11px] font-medium">
                  <Route className="size-3.5" /> Trace
                </TabsTrigger>
              </TabsList>
            </Tabs>

            {imageMode ? (
              <div className="rounded-lg border border-border bg-secondary/30 p-2">
                <CandidateStrip studio={studio} />
              </div>
            ) : null}

            {traceMode ? (
              <div className="flex flex-col gap-2">
                {/* Step 1 — pick a Gemini drawing */}
                <Step
                  index={1}
                  title="Drawing"
                  state={step1State}
                  open={openStep === 1}
                  onToggle={() => toggleStep(1)}
                  summary={
                    drawingPicked && studio.candidates[studio.selectedCandidate] ? (
                      <span className="flex items-center gap-1.5">
                        <img
                          src={candidateDataUrl(studio.candidates[studio.selectedCandidate])}
                          alt="Picked drawing"
                          className="size-5 rounded border border-border bg-white object-contain"
                        />
                        <span className="text-[11px] text-muted-foreground">candidate {studio.selectedCandidate + 1}</span>
                      </span>
                    ) : null
                  }
                >
                  <CandidateStrip studio={studio} />
                </Step>

                {/* Step 2 — trace the drawing into a seed grid */}
                <Step
                  index={2}
                  title="Trace"
                  state={step2State}
                  open={openStep === 2}
                  onToggle={() => toggleStep(2)}
                  summary={
                    traced ? (
                      <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
                        {trace.diagnostics.rooms}r · {trace.diagnostics.doors}d
                      </span>
                    ) : null
                  }
                >
                  {traced ? (
                    <div className="flex flex-col gap-2">
                      <div className="flex flex-wrap gap-1.5">
                        <Chip>
                          <Num>{trace.diagnostics.rooms}</Num> rooms
                        </Chip>
                        <Chip>
                          <Num>{trace.diagnostics.doors}</Num> doors
                        </Chip>
                        <Chip>
                          <Num>{trace.seed.meters_per_cell.toFixed(2)}</Num> m/cell
                        </Chip>
                        <Chip tone="accent">
                          typed <Num>{trace.diagnostics.typed}</Num>/<Num>{trace.diagnostics.rooms}</Num>
                        </Chip>
                      </div>
                      {/* drawing + linework as a flippable image set — click either
                          thumbnail to open the lightbox seeded on that image */}
                      {traceImages.length ? (
                        <div className="grid grid-cols-2 gap-1.5">
                          {traceImages.map((item, i) => (
                            <button
                              key={item.caption}
                              type="button"
                              className="group relative overflow-hidden rounded-md border-2 border-border bg-white transition-shadow hover:border-foreground/40"
                              onClick={() => {
                                setTraceStart(i);
                                studio.setTraceLightboxOpen(true);
                              }}
                            >
                              <img
                                src={candidateDataUrl(item.image)}
                                alt={item.title}
                                className="aspect-video w-full object-contain"
                              />
                              <span className="absolute bottom-0.5 left-0.5 inline-flex items-center gap-1 rounded bg-background/85 px-1 py-0.5 text-[9px] font-medium text-muted-foreground backdrop-blur-sm">
                                <Layers className="size-2.5" /> {item.caption}
                              </span>
                            </button>
                          ))}
                        </div>
                      ) : null}
                      <div className="flex items-center">
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 gap-1.5 text-[11px]"
                          disabled={studio.traceBusy || !!studio.busyAction || !drawingPicked}
                          onClick={() => void studio.runTrace()}
                        >
                          {studio.traceBusy ? <Loader2 className="size-3 animate-spin" /> : <Route className="size-3" />}
                          Re-trace
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <Button
                      size="sm"
                      className="h-8 w-full gap-1.5 text-[12px]"
                      disabled={studio.traceBusy || !!studio.busyAction || !drawingPicked || studio.tracerOnline === false}
                      onClick={() => void studio.runTrace()}
                    >
                      {studio.traceBusy ? <Loader2 className="size-3.5 animate-spin" /> : <Route className="size-3.5" />}
                      {studio.traceBusy ? "Tracing…" : "Trace drawing"}
                    </Button>
                  )}
                </Step>

                {/* Step 3 — confirm and refine the traced seed */}
                <Step
                  index={3}
                  title="Refine"
                  state={step3State}
                  open={openStep === 3}
                  onToggle={() => toggleStep(3)}
                  summary={
                    studio.result && !studio.showingTracePreview ? (
                      <span className="text-[11px] text-muted-foreground">refined</span>
                    ) : null
                  }
                >
                  <div className="flex flex-col gap-1.5">
                    {studio.showingTracePreview ? (
                      <p className="text-[11px] text-muted-foreground">
                        Raw trace is on the canvas. Confirm to translate + validate it as code.
                      </p>
                    ) : null}
                    <Button
                      className="h-9 w-full gap-2 text-[13px] font-medium"
                      disabled={!!studio.busyAction || !studio.program || !studio.effectiveSeed}
                      onClick={() => void studio.runAgentAction("generate")}
                    >
                      {refining ? <Loader2 className="size-3.5 animate-spin" /> : <Sparkles className="size-3.5" />}
                      {refining ? "Refining…" : "Refine traced plan"}
                    </Button>
                  </div>
                </Step>

                {studio.tracerOnline === false ? (
                  <p className="px-1 text-[11px] text-muted-foreground">
                    Tracer offline — paste a seed under Advanced below.
                  </p>
                ) : null}
              </div>
            ) : null}
          </section>

          {/* brief */}
          <section className="flex flex-col gap-2">
            <span className="label-xs">Brief</span>
            {/* Auto-grows to its content (field-sizing) so a fitting brief never
                shows a scrollbar; caps out and scrolls with the themed bar. */}
            <Textarea
              value={studio.prompt}
              onChange={(event) => studio.setPrompt(event.target.value)}
              placeholder="Entrance on the west, waiting near entry, patient rooms along the perimeter."
              className="min-h-20 max-h-40 resize-y text-[12px] leading-relaxed"
            />
          </section>

          {/* canvas */}
          <section>
            <Disclosure
              open={canvasOpen}
              onOpenChange={setCanvasOpen}
              label={<span className="label-xs">Canvas</span>}
              meta={
                <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
                  {studio.gridWidth} × {studio.gridHeight}
                  {ftPerCell ? ` · ${ftPerCell} ft/cell` : ""}
                </span>
              }
            >
              <div className="flex flex-col gap-2 pt-2">
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex items-center gap-1.5">
                    <Label className="text-[11px] text-muted-foreground">W</Label>
                    <Input
                      type="number"
                      min={32}
                      max={160}
                      value={studio.gridWidth}
                      onChange={(event) => studio.setGridWidth(Number(event.target.value) || 96)}
                      className="h-8 font-mono text-xs"
                    />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Label className="text-[11px] text-muted-foreground">H</Label>
                    <Input
                      type="number"
                      min={24}
                      max={120}
                      value={studio.gridHeight}
                      onChange={(event) => studio.setGridHeight(Number(event.target.value) || 64)}
                      className="h-8 font-mono text-xs"
                    />
                  </div>
                </div>
                <p className="text-[11px] text-muted-foreground">
                  Cells stay square; the grid ratio follows a selected drawing.
                </p>
              </div>
            </Disclosure>
          </section>

          {/* Advanced — shared bottom escape hatch in every mode. The raw program
              JSON lives here (no longer the primary input); trace mode also keeps
              its direct seed paste/upload path here. Force-opens on any error. */}
          <section>
            <Disclosure
              open={advancedOpenEffective}
              onOpenChange={setAdvancedOpen}
              label={<span className="label-xs">Advanced</span>}
              meta={
                studio.programError ? (
                  <span className="flex items-center gap-1 text-[10px] text-destructive">
                    <CircleAlert className="size-3" /> JSON invalid
                  </span>
                ) : traceMode && studio.seedError ? (
                  <span className="flex items-center gap-1 text-[10px] text-destructive">
                    <CircleAlert className="size-3" /> seed invalid
                  </span>
                ) : null
              }
            >
              <div className="flex flex-col gap-3 pt-2">
                {/* Program JSON — the escape hatch for hand-editing the program */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <span className="label-xs">Program JSON</span>
                    <button
                      type="button"
                      className="text-[11px] text-muted-foreground hover:text-foreground"
                      onClick={() => {
                        if (studio.program) studio.setProgramText(JSON.stringify(studio.program, null, 2));
                      }}
                    >
                      Format
                    </button>
                  </div>
                  <Textarea
                    value={studio.programText}
                    onChange={(event) => studio.setProgramText(event.target.value)}
                    spellCheck={false}
                    className="h-44 resize-y font-mono text-[11px] leading-relaxed"
                  />
                  {studio.programError ? (
                    <p className="flex items-center gap-1 text-[11px] text-destructive">
                      <CircleAlert className="size-3 shrink-0" />
                      <span className="font-mono">{studio.programError}</span>
                    </p>
                  ) : (
                    <p className="flex items-center gap-1 text-[11px] text-success">
                      <CircleCheck className="size-3 shrink-0" /> valid
                    </p>
                  )}
                </div>

                {/* Trace mode keeps its direct seed paste/upload path here */}
                {traceMode ? (
                  <div className="flex flex-col gap-2 rounded-lg border border-border bg-secondary/30 p-2">
                    <div className="flex items-center justify-between">
                      <span className="label-xs">Seed JSON</span>
                      <label className="cursor-pointer text-[11px] font-medium text-primary hover:underline">
                        Upload JSON
                        <input
                          type="file"
                          accept=".json,application/json"
                          className="hidden"
                          onChange={(event) => {
                            const file = event.target.files?.[0];
                            if (!file) return;
                            void file.text().then((text) => studio.setSeedText(text));
                            event.target.value = "";
                          }}
                        />
                      </label>
                    </div>
                    <Textarea
                      value={studio.seedText}
                      onChange={(event) => studio.setSeedText(event.target.value)}
                      spellCheck={false}
                      placeholder='Paste traced seed JSON: {"width":..,"height":..,"cells":[..],"rooms":[..],"doors":[..]}'
                      className="h-28 resize-y font-mono text-[10px] leading-relaxed"
                    />
                    {studio.seedError ? (
                      <p className="flex items-center gap-1 text-[11px] text-destructive">
                        <CircleAlert className="size-3 shrink-0" />
                        <span className="font-mono">{studio.seedError}</span>
                      </p>
                    ) : studio.seed ? (
                      <div className="flex flex-wrap gap-1.5">
                        <Chip>
                          <Num>{studio.seed.width}×{studio.seed.height}</Num> cells
                        </Chip>
                        <Chip>
                          <Num>{studio.seed.rooms.length}</Num> rooms
                        </Chip>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </Disclosure>
          </section>
        </div>
      </ScrollArea>

      {/* FOOTER — fixed: primary action + status line, always visible. */}
      <div className="shrink-0 border-t border-border p-3">
        {/* Trace mode's primary action is the stepper's step 3 (Refine). */}
        {traceMode ? null : (
          <Button
            className="h-9 w-full gap-2 text-[13px] font-medium"
            disabled={
              !!studio.busyAction ||
              !studio.program ||
              (imageMode && studio.selectedCandidate < 0)
            }
            onClick={() => void studio.runAgentAction("generate")}
          >
            <Sparkles className="size-3.5" />
            {studio.busyAction === "generate"
              ? studio.transcribing
                ? "Transcribing…"
                : "Generating…"
              : imageMode
                ? "Transcribe drawing"
                : studio.result
                  ? "Re-generate plan"
                  : "Generate plan"}
          </Button>
        )}
        <p
          className={cn(
            "truncate font-mono text-[11px] tabular-nums",
            traceMode ? "" : "mt-2",
            studio.status.error ? "text-destructive" : "text-muted-foreground",
          )}
        >
          {studio.status.text}
        </p>
      </div>

      {/* candidate lightbox — large, zoomable, with candidate flipping */}
      <LightboxViewer
        open={studio.lightbox !== null}
        onOpenChange={(open) => !open && studio.setLightbox(null)}
        title={`Candidate ${studio.lightbox === null ? "" : studio.lightbox + 1}`}
        image={studio.lightbox === null ? null : studio.candidates[studio.lightbox] ?? null}
        index={studio.lightbox ?? undefined}
        count={studio.candidates.length}
        selectedIndex={studio.selectedCandidate}
        onIndex={(index) => studio.setLightbox(index)}
        onSelect={(index) => {
          studio.selectCandidate(index);
          studio.setLightbox(null);
        }}
      />

      {/* trace artifacts — same zoomable viewer, drawing + linework as a set */}
      <LightboxViewer
        open={studio.traceLightboxOpen}
        onOpenChange={(open) => studio.setTraceLightboxOpen(open)}
        images={traceImages}
        startIndex={traceStart}
      />
    </div>
  );
}
