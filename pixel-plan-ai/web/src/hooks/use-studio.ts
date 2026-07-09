"use client";

import * as React from "react";
import { toast } from "sonner";

import {
  cancelGeneration,
  fetchHealth,
  fetchProgress,
  fetchSamples,
  fetchTracerHealth,
  postAgentAction,
  postDrawCandidates,
  postTrace,
} from "@/lib/api";
import type {
  AgentAction,
  CandidateImage,
  GenerationIteration,
  GenerationResult,
  HealthResponse,
  Plan,
  Program,
  Samples,
  SeedPlan,
  StudioMode,
  TraceResponse,
  Validation,
} from "@/lib/types";
import { candidateDataUrl } from "@/lib/types";
import { seedToPreviewPlan } from "@/lib/type-colors";

export type ViewportMode = "2d" | "3d";

export interface StudioStatus {
  text: string;
  error: boolean;
}

const DEFAULT_PROMPTS: Record<string, string> = {
  "inpatient-ward":
    "Place patient rooms along the perimeter, nurse stations centrally, and provide direct doors from every occupied room to circulation.",
  default:
    "Create a compact, legible plan. Put the main entrance on the west side and connect every occupied room to circulation with a door.",
};

export function useStudio() {
  const [samples, setSamples] = React.useState<Samples>({});
  const [sampleKey, setSampleKey] = React.useState<string>("");
  const [programText, setProgramText] = React.useState<string>("");
  const [prompt, setPrompt] = React.useState<string>(DEFAULT_PROMPTS.default);
  const [gridWidth, setGridWidth] = React.useState(96);
  const [gridHeight, setGridHeight] = React.useState(64);
  const [health, setHealth] = React.useState<HealthResponse | null>(null);

  const [mode, setMode] = React.useState<StudioMode>("program");
  const [viewport, setViewport] = React.useState<ViewportMode>("2d");
  const [seedText, setSeedText] = React.useState("");
  const [candidates, setCandidates] = React.useState<CandidateImage[]>([]);
  const [selectedCandidate, setSelectedCandidate] = React.useState(-1);
  const [candidatesBusy, setCandidatesBusy] = React.useState(false);
  const [lightbox, setLightbox] = React.useState<number | null>(null);

  // Trace pipeline: hfagent tracer service (separate localhost process reached
  // through the /trace-api rewrite) turns a picked drawing into a seed grid.
  const [tracerOnline, setTracerOnline] = React.useState<boolean | null>(null);
  const [traceBusy, setTraceBusy] = React.useState(false);
  const [traceResult, setTraceResult] = React.useState<TraceResponse | null>(null);
  const [traceLightboxOpen, setTraceLightboxOpen] = React.useState(false);

  const [result, setResult] = React.useState<GenerationResult | null>(null);
  const [livePlan, setLivePlan] = React.useState<Plan | null>(null);
  const [liveValidation, setLiveValidation] = React.useState<Validation | null>(null);
  const [iterations, setIterations] = React.useState<GenerationIteration[]>([]);
  const [code, setCode] = React.useState("");
  const [codeSource, setCodeSource] = React.useState("—");
  const [revision, setRevision] = React.useState("");

  const [busyAction, setBusyAction] = React.useState<AgentAction | null>(null);
  const [phaseText, setPhaseText] = React.useState("");
  const [elapsed, setElapsed] = React.useState(0);
  const [status, setStatus] = React.useState<StudioStatus>({ text: "Ready", error: false });

  const [reference, setReference] = React.useState<{ image: CandidateImage; index: number } | null>(null);
  const [showReference, setShowReference] = React.useState(false);
  const [referenceOpacity, setReferenceOpacity] = React.useState(55);

  // 2D viewport selection (room id), surfaced to the inspector. Cleared whenever
  // the active plan is replaced so a stale id never highlights a different room.
  const [selectedRoomId, setSelectedRoomId] = React.useState<string | null>(null);

  const abortRef = React.useRef<AbortController | null>(null);
  const requestIdRef = React.useRef<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [samplesPayload, healthPayload] = await Promise.all([fetchSamples(), fetchHealth()]);
        if (cancelled) return;
        setSamples(samplesPayload);
        setHealth(healthPayload);
        const firstKey = Object.keys(samplesPayload)[0];
        if (firstKey) {
          setSampleKey(firstKey);
          setProgramText(JSON.stringify(samplesPayload[firstKey], null, 2));
          setPrompt(DEFAULT_PROMPTS[firstKey] ?? DEFAULT_PROMPTS.default);
        }
      } catch (error) {
        if (!cancelled) setStatus({ text: error instanceof Error ? error.message : String(error), error: true });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  React.useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const tracer = await fetchTracerHealth(controller.signal);
        setTracerOnline(!!tracer.ok);
      } catch {
        setTracerOnline(false); // offline — trace mode falls back to paste/upload
      }
    })();
    return () => controller.abort();
  }, []);

  React.useEffect(() => {
    if (!busyAction) return;
    const startedAt = performance.now();
    setElapsed(0);
    const timer = window.setInterval(() => {
      setElapsed(Math.floor((performance.now() - startedAt) / 1000));
    }, 250);
    return () => window.clearInterval(timer);
  }, [busyAction]);

  const program = React.useMemo<Program | null>(() => {
    try {
      return JSON.parse(programText) as Program;
    } catch {
      return null;
    }
  }, [programText]);

  const programError = React.useMemo<string | null>(() => {
    try {
      JSON.parse(programText);
      return null;
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  }, [programText]);

  const requestedSpaces = React.useMemo(
    () => program?.rooms?.reduce((total, room) => total + Number(room.count || 1), 0) ?? 0,
    [program],
  );

  // The traced seed drives the raw preview and takes precedence over any pasted
  // seed once a trace has run; the preview Plan renders through the same
  // activePlan pathway as a live/result plan (see below).
  const tracedSeed = traceResult?.seed ?? null;
  const tracePreviewPlan = React.useMemo<Plan | null>(
    () => (traceResult ? seedToPreviewPlan(traceResult.seed) : null),
    [traceResult],
  );

  const activePlan = livePlan ?? result?.plan ?? tracePreviewPlan ?? null;
  const activeValidation = liveValidation ?? result?.validation ?? null;
  // Raw trace preview is on screen only while nothing validated has replaced it.
  const showingTracePreview = !livePlan && !result && !!tracePreviewPlan;

  // Drop any selection when the plan on the canvas is swapped out. Reset during
  // render (React's "adjust state on prop change" pattern) rather than in an
  // effect, so a stale highlight never paints for a frame.
  const prevPlanRef = React.useRef<Plan | null>(null);
  if (prevPlanRef.current !== activePlan) {
    prevPlanRef.current = activePlan;
    if (selectedRoomId !== null) setSelectedRoomId(null);
  }

  const seed = React.useMemo<SeedPlan | null>(() => {
    if (!seedText.trim()) return null;
    try {
      const parsed = JSON.parse(seedText) as SeedPlan;
      if (!parsed || typeof parsed !== "object") return null;
      if (!parsed.width || !parsed.height || !parsed.cells || !parsed.rooms?.length) return null;
      return parsed;
    } catch {
      return null;
    }
  }, [seedText]);

  const seedError = React.useMemo<string | null>(() => {
    if (!seedText.trim()) return null;
    try {
      const parsed = JSON.parse(seedText) as SeedPlan;
      if (!parsed?.width || !parsed?.height) return "seed needs width and height";
      if (!parsed?.cells) return "seed needs a cells grid";
      if (!parsed?.rooms?.length) return "seed needs a rooms list";
      return null;
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  }, [seedText]);

  // Refine consumes the traced seed when present, else the pasted/uploaded one.
  const effectiveSeed = tracedSeed ?? seed;

  const loadSample = React.useCallback(
    (key: string) => {
      const sample = samples[key];
      if (!sample) return;
      setSampleKey(key);
      setProgramText(JSON.stringify(sample, null, 2));
      setPrompt(DEFAULT_PROMPTS[key] ?? DEFAULT_PROMPTS.default);
    },
    [samples],
  );

  const transcribing = mode === "image" && selectedCandidate >= 0;

  const pollProgress = React.useCallback(
    async (requestId: string, signal: AbortSignal, isFinished: () => boolean) => {
      while (!signal.aborted && !isFinished()) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, 700));
        if (signal.aborted || isFinished()) return;
        try {
          const progress = await fetchProgress(requestId, signal);
          if (progress.iterations.length) setIterations(progress.iterations);
          const latest = progress.iterations[progress.iterations.length - 1];
          if (latest) {
            const score =
              latest.status === "running"
                ? "waiting for Gemini"
                : latest.score === null
                  ? "no executable plan"
                  : `score ${latest.score}`;
            setPhaseText(`Attempt ${latest.attempt} ${latest.phase} · ${latest.status} · ${score}`);
          } else if (progress.message) {
            setPhaseText(progress.message);
          }
          const preview = progress.preview;
          if (preview?.code) {
            setCode(preview.code);
            setCodeSource(`LIVE · ${[preview.source ?? "gemini", preview.model].filter(Boolean).join(" · ")}`);
          }
          if (preview?.plan && preview.validation) {
            setLivePlan(preview.plan);
            setLiveValidation(preview.validation);
          }
        } catch {
          // Polling is best-effort; the POST response carries the final result.
        }
      }
    },
    [],
  );

  const runAgentAction = React.useCallback(
    async (action: AgentAction) => {
      if (busyAction || !program) return;
      if (action !== "generate" && !code.trim()) {
        setStatus({ text: "Generate a plan before using code-agent actions.", error: true });
        return;
      }
      if (action === "revise" && !revision.trim()) {
        setStatus({ text: "Enter a revision instruction first.", error: true });
        return;
      }
      if (action === "generate" && mode === "image" && selectedCandidate < 0) {
        setStatus({ text: "Draw candidates and pick one drawing first.", error: true });
        return;
      }
      if (action === "generate" && mode === "trace" && !effectiveSeed) {
        setStatus({ text: "Trace a drawing or paste a valid seed JSON first.", error: true });
        return;
      }
      const requestId = crypto.randomUUID();
      const controller = new AbortController();
      abortRef.current = controller;
      requestIdRef.current = requestId;
      const isTranscription = action === "generate" && mode === "image" && selectedCandidate >= 0;
      const isRefinement = action === "generate" && mode === "trace" && !!effectiveSeed;
      // Anchor the refine on the picked drawing too (same reference infra as
      // image mode): the traced seed sets geometry, the drawing sets intent.
      const referenceImage =
        isTranscription || (isRefinement && selectedCandidate >= 0)
          ? candidates[selectedCandidate]
          : null;
      if (referenceImage) setReference({ image: referenceImage, index: selectedCandidate });
      setBusyAction(action);
      setShowReference(false);
      setPhaseText(
        isRefinement
          ? "Executing the seed translation, then repairing only if validation fails"
          : isTranscription
            ? "Gemini is transcribing the selected drawing into floor-plan Python"
            : action === "generate"
              ? "Gemini is writing a complete Python program"
              : action === "fix"
                ? "Gemini is diagnosing and fixing the current code"
                : action === "revise"
                  ? "Gemini is revising the complete Python program"
                  : "Executing the current Python program",
      );
      setStatus({
        text: isRefinement
          ? "Refining the traced plan..."
          : isTranscription
            ? "Transcribing the drawing, then validating..."
            : "Working...",
        error: false,
      });
      let finished = false;
      const progressTask =
        action === "run" || action === "inspect"
          ? Promise.resolve()
          : pollProgress(requestId, controller.signal, () => finished);
      const startedAt = performance.now();
      try {
        const payload = await postAgentAction({
          requestId,
          action,
          program,
          prompt: action === "revise" ? revision : prompt,
          code: action === "run" || action === "inspect" ? code : undefined,
          currentCode: action === "fix" || action === "revise" ? code : undefined,
          width: gridWidth,
          height: gridHeight,
          referenceImage,
          seed: isRefinement ? effectiveSeed : null,
          signal: controller.signal,
        });
        setResult(payload);
        setLivePlan(null);
        setLiveValidation(null);
        setIterations(payload.iterations);
        setCode(payload.code);
        setCodeSource(payload.model ? `${payload.source} · ${payload.model}` : payload.source);
        const seconds = ((performance.now() - startedAt) / 1000).toFixed(1);
        const acceptance = payload.accepted === false ? " · needs revision" : "";
        setStatus({ text: `Done in ${seconds}s · ${payload.source}${acceptance}`, error: false });
        if (payload.accepted === false) {
          toast.warning("Plan generated but not accepted", { description: payload.ai_error ?? "Best attempt kept." });
        }
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          setStatus({ text: "Generation stopped · adjust inputs or re-generate", error: false });
        } else {
          const message = error instanceof Error ? error.message : String(error);
          setStatus({ text: message, error: true });
          toast.error("Generation failed", { description: message });
        }
      } finally {
        finished = true;
        await progressTask;
        if (requestIdRef.current === requestId) {
          abortRef.current = null;
          requestIdRef.current = null;
          setBusyAction(null);
        }
      }
    },
    [busyAction, program, code, revision, mode, selectedCandidate, candidates, effectiveSeed, prompt, gridWidth, gridHeight, pollProgress],
  );

  const stopGeneration = React.useCallback(async () => {
    const requestId = requestIdRef.current;
    if (!requestId) return;
    setStatus({ text: "Stopping generation...", error: false });
    try {
      await cancelGeneration(requestId);
    } finally {
      abortRef.current?.abort();
    }
  }, []);

  const drawCandidates = React.useCallback(async () => {
    if (candidatesBusy || busyAction || !program) return;
    setCandidatesBusy(true);
    setCandidates([]);
    setSelectedCandidate(-1);
    setTraceResult(null); // new drawings invalidate any prior trace
    try {
      const images = await postDrawCandidates(program, 3);
      setCandidates(images);
      toast.success(`${images.length} drawings ready`, { description: "Click one to select it." });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus({ text: message, error: true });
      toast.error("Drawing failed", { description: message });
    } finally {
      setCandidatesBusy(false);
    }
  }, [candidatesBusy, busyAction, program]);

  const runTrace = React.useCallback(async () => {
    if (traceBusy || busyAction) return;
    const candidate = candidates[selectedCandidate];
    if (!candidate) {
      setStatus({ text: "Pick a drawing candidate before tracing.", error: true });
      return;
    }
    if (!sampleKey) {
      setStatus({ text: "Select a program before tracing.", error: true });
      return;
    }
    setTraceBusy(true);
    setStatus({ text: "Tracing the drawing into a seed grid...", error: false });
    try {
      const response = await postTrace(candidate, sampleKey);
      setTraceResult(response);
      // A fresh trace starts a fresh pipeline: drop any prior refine result so
      // the raw preview (not a stale validated plan) shows on the canvas, and
      // clear a prior reference overlay so it can't collide with the preview chip.
      setResult(null);
      setLivePlan(null);
      setLiveValidation(null);
      setReference(null);
      setShowReference(false);
      const d = response.diagnostics;
      setStatus({
        text: `Traced · ${d.rooms} rooms · ${d.doors} doors · typed ${d.typed}/${d.rooms}`,
        error: false,
      });
      toast.success("Drawing traced", {
        description: "Raw seed preview is on the canvas — confirm to refine.",
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus({ text: message, error: true });
      toast.error("Trace failed", { description: message });
    } finally {
      setTraceBusy(false);
    }
  }, [traceBusy, busyAction, candidates, selectedCandidate, sampleKey]);

  const selectCandidate = React.useCallback(
    (index: number) => {
      setSelectedCandidate(index);
      setTraceResult(null); // a different drawing needs a fresh trace
      const candidate = candidates[index];
      if (!candidate) return;
      // Match the grid to the drawing's aspect ratio so transcription does
      // not stretch the layout (drawings are 16:9, default grid is 3:2).
      const image = new Image();
      image.onload = () => {
        if (!image.naturalWidth || !image.naturalHeight) return;
        const aspect = image.naturalWidth / image.naturalHeight;
        setGridHeight((current) => {
          const matched = Math.max(24, Math.min(120, Math.round(gridWidth / aspect)));
          if (matched !== current) {
            toast.info(`Canvas matched to the drawing`, {
              description: `${gridWidth} × ${matched} cells (drawing is ${image.naturalWidth}:${image.naturalHeight}).`,
            });
          }
          return matched;
        });
      };
      image.src = candidateDataUrl(candidate);
    },
    [candidates, gridWidth],
  );

  return {
    samples,
    sampleKey,
    loadSample,
    programText,
    setProgramText,
    program,
    programError,
    requestedSpaces,
    prompt,
    setPrompt,
    gridWidth,
    setGridWidth,
    gridHeight,
    setGridHeight,
    health,
    mode,
    setMode,
    viewport,
    setViewport,
    seedText,
    setSeedText,
    seed,
    seedError,
    candidates,
    selectedCandidate,
    selectCandidate,
    candidatesBusy,
    drawCandidates,
    lightbox,
    setLightbox,
    tracerOnline,
    traceBusy,
    traceResult,
    runTrace,
    effectiveSeed,
    showingTracePreview,
    traceLightboxOpen,
    setTraceLightboxOpen,
    result,
    activePlan,
    activeValidation,
    iterations,
    code,
    setCode,
    codeSource,
    revision,
    setRevision,
    busyAction,
    phaseText,
    elapsed,
    status,
    transcribing,
    runAgentAction,
    stopGeneration,
    reference,
    showReference,
    setShowReference,
    referenceOpacity,
    setReferenceOpacity,
    selectedRoomId,
    setSelectedRoomId,
  };
}

export type Studio = ReturnType<typeof useStudio>;
