"use client";

import * as React from "react";
import { Check, ImagePlus, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
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
import { candidateDataUrl } from "@/lib/types";

function SectionLabel({ index, children }: { index: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.14em] text-muted-foreground">
      <span className="text-primary/70">{index}</span>
      {children}
    </div>
  );
}

export function ProgramPanel({ studio }: { studio: Studio }) {
  const imageMode = studio.mode === "image";

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b px-4 py-3">
        <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Input / Program</p>
        <h1 className="mt-1 text-sm font-semibold leading-tight">Program to pixel plan</h1>
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-5 p-4">
          <section className="flex flex-col gap-2">
            <SectionLabel index="01">Sample program</SectionLabel>
            <Select value={studio.sampleKey} onValueChange={(value) => value && studio.loadSample(value)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder="Pick a sample" />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(studio.samples).map((key) => (
                  <SelectItem key={key} value={key} className="text-xs">
                    {key}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </section>

          <section className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <SectionLabel index="02">Program JSON</SectionLabel>
              <button
                type="button"
                className="text-[10px] font-bold tracking-wide text-muted-foreground hover:text-foreground"
                onClick={() => {
                  if (studio.program) studio.setProgramText(JSON.stringify(studio.program, null, 2));
                }}
              >
                FORMAT
              </button>
            </div>
            <Textarea
              value={studio.programText}
              onChange={(event) => studio.setProgramText(event.target.value)}
              spellCheck={false}
              className="h-44 resize-y font-mono text-[11px] leading-relaxed"
            />
            <p className={`text-[11px] ${studio.programError ? "text-destructive" : "text-emerald-700"}`}>
              {studio.programError
                ? `Invalid JSON · ${studio.programError}`
                : `Valid JSON · ${studio.requestedSpaces} requested spaces`}
            </p>
          </section>

          <section className="flex flex-col gap-2">
            <SectionLabel index="03">Generation mode</SectionLabel>
            <Tabs value={studio.mode} onValueChange={(value) => studio.setMode(value as "program" | "image")}>
              <TabsList className="grid h-8 w-full grid-cols-3">
                <TabsTrigger value="program" className="text-[10px] font-bold tracking-wide">
                  PROGRAM
                </TabsTrigger>
                <TabsTrigger value="image" className="text-[10px] font-bold tracking-wide">
                  FROM IMAGE
                </TabsTrigger>
                <TabsTrigger value="trace" disabled className="text-[10px] font-bold tracking-wide" title="Coming soon: refine a traced plan">
                  FROM TRACE
                </TabsTrigger>
              </TabsList>
            </Tabs>

            {imageMode ? (
              <div className="flex flex-col gap-2 rounded-lg border bg-muted/40 p-2.5">
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 gap-1.5 text-[11px] font-bold tracking-wide"
                  disabled={studio.candidatesBusy || !!studio.busyAction || !studio.program}
                  onClick={() => void studio.drawCandidates()}
                >
                  <ImagePlus className="size-3.5" />
                  {studio.candidatesBusy ? "DRAWING…" : "DRAW 3 CANDIDATE PLANS"}
                </Button>
                <p className="text-[11px] leading-snug text-muted-foreground">
                  {studio.candidates.length
                    ? studio.selectedCandidate >= 0
                      ? `Candidate ${studio.selectedCandidate + 1} selected · Generate transcribes it.`
                      : "Click a drawing to select it; click again to zoom."
                    : "Gemini draws three schematic plans; pick the one to transcribe."}
                </p>
                <div className="grid grid-cols-3 gap-1.5">
                  {studio.candidatesBusy
                    ? [0, 1, 2].map((index) => <Skeleton key={index} className="aspect-video w-full rounded-md" />)
                    : studio.candidates.map((candidate, index) => (
                        <button
                          key={index}
                          type="button"
                          className={`relative overflow-hidden rounded-md border-2 bg-white transition-shadow ${
                            index === studio.selectedCandidate
                              ? "border-primary shadow-[0_0_0_1px_var(--primary)]"
                              : "border-border hover:border-foreground/40"
                          }`}
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
            ) : null}
          </section>

          <section className="flex flex-col gap-2">
            <SectionLabel index="04">Design brief</SectionLabel>
            <Textarea
              value={studio.prompt}
              onChange={(event) => studio.setPrompt(event.target.value)}
              placeholder="Example: entrance on the west, waiting near entry, patient rooms along the perimeter."
              className="h-20 resize-y text-[11px] leading-relaxed"
            />
          </section>

          <section className="flex flex-col gap-2">
            <SectionLabel index="05">Canvas</SectionLabel>
            <div className="grid grid-cols-2 gap-2">
              <div className="flex items-center gap-1.5">
                <Label className="text-[10px] font-bold text-muted-foreground">W</Label>
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
                <Label className="text-[10px] font-bold text-muted-foreground">H</Label>
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
            <p className="text-[11px] text-muted-foreground">Cells stay square; the grid ratio follows a selected drawing.</p>
          </section>
        </div>
      </ScrollArea>

      <div className="border-t p-3">
        <Button
          className="h-10 w-full gap-2 text-xs font-bold tracking-wide"
          disabled={!!studio.busyAction || !studio.program || (imageMode && studio.selectedCandidate < 0)}
          onClick={() => void studio.runAgentAction("generate")}
        >
          <Sparkles className="size-3.5" />
          {studio.busyAction === "generate"
            ? studio.transcribing
              ? "TRANSCRIBING…"
              : "GENERATING…"
            : imageMode
              ? "TRANSCRIBE DRAWING"
              : studio.result
                ? "RE-GENERATE PLAN"
                : "GENERATE PLAN"}
        </Button>
        <p className={`mt-2 truncate text-[11px] ${studio.status.error ? "text-destructive" : "text-muted-foreground"}`}>
          {studio.status.text}
        </p>
      </div>

      {/* candidate lightbox */}
      <Dialog open={studio.lightbox !== null} onOpenChange={(open) => !open && studio.setLightbox(null)}>
        <DialogContent className="max-w-4xl p-3">
          <DialogTitle className="text-xs font-bold tracking-wide">
            CANDIDATE {studio.lightbox === null ? "" : studio.lightbox + 1} / DRAWING PREVIEW
          </DialogTitle>
          {studio.lightbox !== null && studio.candidates[studio.lightbox] ? (
            <>
              <img
                src={candidateDataUrl(studio.candidates[studio.lightbox])}
                alt={`Candidate plan ${studio.lightbox + 1} full preview`}
                className="max-h-[72vh] w-full rounded-md border bg-white object-contain"
              />
              <div className="flex justify-between gap-2">
                <div className="flex gap-1">
                  {studio.candidates.map((_, index) => (
                    <Button
                      key={index}
                      size="sm"
                      variant={index === studio.lightbox ? "default" : "outline"}
                      className="h-7 w-8 text-[11px]"
                      onClick={() => studio.setLightbox(index)}
                    >
                      {index + 1}
                    </Button>
                  ))}
                </div>
                <Button
                  size="sm"
                  className="h-7 gap-1.5 text-[11px] font-bold"
                  onClick={() => {
                    if (studio.lightbox !== null) studio.selectCandidate(studio.lightbox);
                    studio.setLightbox(null);
                  }}
                >
                  <Check className="size-3" /> USE THIS DRAWING
                </Button>
              </div>
            </>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
