"use client";

import * as React from "react";
import {
  Braces,
  Check,
  ChevronDown,
  CircleAlert,
  CircleCheck,
  Image as ImageIcon,
  ImagePlus,
  Layers,
  Loader2,
  Route,
  Sparkles,
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
import type { CandidateImage, StudioMode } from "@/lib/types";
import { candidateDataUrl } from "@/lib/types";
import { cn } from "@/lib/utils";

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

export function ProgramPanel({ studio }: { studio: Studio }) {
  const imageMode = studio.mode === "image";
  const traceMode = studio.mode === "trace";
  const [jsonOpen, setJsonOpen] = React.useState(false);
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

  const corridors = React.useMemo(
    () =>
      studio.program?.rooms?.reduce(
        (total, room) => total + (room.type.toLowerCase().includes("corridor") ? Number(room.count || 1) : 0),
        0,
      ) ?? 0,
    [studio.program],
  );

  const ftPerCell = studio.activePlan
    ? (studio.activePlan.meters_per_cell * 3.28084).toFixed(2)
    : null;

  const refining = studio.busyAction === "generate";
  const step1State = drawingPicked ? "done" : "active";
  const step2State = traced ? "done" : drawingPicked ? "active" : "pending";
  const step3State = studio.result && !studio.showingTracePreview ? "done" : studio.effectiveSeed ? "active" : "pending";
  const toggleStep = (index: number) => setOpenStep((step) => (step === index ? 0 : index));

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-col gap-1.5 border-b border-border px-3 py-2.5">
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

      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-4 p-3">
          {/* summary chips + JSON disclosure */}
          <section className="flex flex-col gap-2">
            {studio.program ? (
              <div className="flex flex-wrap gap-1.5">
                <Chip>
                  <Num>{studio.requestedSpaces}</Num> spaces
                </Chip>
                {corridors > 0 ? (
                  <Chip>
                    <Num>{corridors}</Num> corridor
                  </Chip>
                ) : null}
                {studio.program.building_type ? (
                  <Chip tone="accent">
                    <span className="capitalize">{studio.program.building_type.replaceAll("_", " ")}</span>
                  </Chip>
                ) : null}
              </div>
            ) : null}

            <Disclosure
              open={jsonOpen || !!studio.programError}
              onOpenChange={setJsonOpen}
              label={<span className="label-xs">JSON</span>}
              meta={
                studio.programError ? (
                  <span className="flex items-center gap-1 text-[10px] text-destructive">
                    <CircleAlert className="size-3" /> invalid
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-[10px] text-success">
                    <CircleCheck className="size-3" /> valid
                  </span>
                )
              }
            >
              <div className="flex flex-col gap-1.5 pt-2">
                <div className="flex justify-end">
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
                  <p className="text-[11px] text-destructive">
                    <span className="font-mono">{studio.programError}</span>
                  </p>
                ) : null}
              </div>
            </Disclosure>
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
                  <p className="px-1 text-[11px] text-muted-foreground">Tracer offline — paste a seed below.</p>
                ) : null}

                {/* Fallback: the direct seed paste/upload path stays available always */}
                <Disclosure
                  open={advancedOpen || studio.tracerOnline === false || !!studio.seedError}
                  onOpenChange={setAdvancedOpen}
                  label={<span className="label-xs">Advanced</span>}
                  meta={
                    studio.seedError ? (
                      <span className="flex items-center gap-1 text-[10px] text-destructive">
                        <CircleAlert className="size-3" /> invalid
                      </span>
                    ) : null
                  }
                >
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
                </Disclosure>
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
        </div>
      </ScrollArea>

      <div className="border-t border-border p-3">
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
