"use client";

import * as React from "react";
import { toast } from "sonner";

import { cancelGeneration, fetchHealth, fetchProgress, postRoom } from "@/lib/api";
import type {
  GenerationEvent,
  GenerationIteration,
  HealthResponse,
  RoomResult,
  RoomValidation,
} from "@/lib/types";

export type RoomViewportMode = "2d" | "3d";

export interface RoomStatus {
  text: string;
  error: boolean;
}

// v1 ships a single preset; the select is present so the flow reads as a
// library that will grow, but the dimensions are the only editable input.
export const ROOM_PRESETS = [{ key: "icu-room-v1", label: "ICU Room v1" }] as const;

const DEFAULT_WIDTH_FT = 16.5;
const DEFAULT_DEPTH_FT = 16.75;
const DEFAULT_ROOM_PROMPT =
  "Lay out a calm single-patient ICU room: bed head on a headwall with clear transfer and foot space, equipment on one side, and the visitor chair on the other.";

export function useRoomStudio() {
  const [preset, setPreset] = React.useState<string>(ROOM_PRESETS[0].key);
  const [widthFt, setWidthFt] = React.useState<number>(DEFAULT_WIDTH_FT);
  const [depthFt, setDepthFt] = React.useState<number>(DEFAULT_DEPTH_FT);
  const [prompt, setPrompt] = React.useState<string>(DEFAULT_ROOM_PROMPT);
  const [viewport, setViewport] = React.useState<RoomViewportMode>("2d");
  const [health, setHealth] = React.useState<HealthResponse | null>(null);

  const [result, setResult] = React.useState<RoomResult | null>(null);
  const [liveValidation, setLiveValidation] = React.useState<RoomValidation | null>(null);
  const [iterations, setIterations] = React.useState<GenerationIteration[]>([]);
  const [liveIterations, setLiveIterations] = React.useState<GenerationIteration[]>([]);
  const [liveEvents, setLiveEvents] = React.useState<GenerationEvent[]>([]);
  const [runLogOpen, setRunLogOpen] = React.useState(false);
  const [code, setCode] = React.useState("");
  const [codeSource, setCodeSource] = React.useState("—");

  // "generate" while a run is in flight, else null — the same truthiness the
  // shared RunLogDrawer reads (RunLogHost subset).
  const [busyAction, setBusyAction] = React.useState<"generate" | null>(null);
  const [phaseText, setPhaseText] = React.useState("");
  const [elapsed, setElapsed] = React.useState(0);
  const [status, setStatus] = React.useState<RoomStatus>({ text: "Ready", error: false });

  const [selectedAssetId, setSelectedAssetId] = React.useState<string | null>(null);

  const abortRef = React.useRef<AbortController | null>(null);
  const requestIdRef = React.useRef<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const payload = await fetchHealth();
        if (!cancelled) setHealth(payload);
      } catch {
        // Health is advisory in the room flow; the badge simply shows "checking…".
      }
    })();
    return () => {
      cancelled = true;
    };
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

  const areaFt2 = React.useMemo(() => widthFt * depthFt, [widthFt, depthFt]);

  const activeRoom = result?.room ?? null;
  const activeValidation = liveValidation ?? result?.validation ?? null;

  // Drop any asset selection when the layout on the canvas is replaced.
  const prevRoomRef = React.useRef<typeof activeRoom>(null);
  if (prevRoomRef.current !== activeRoom) {
    prevRoomRef.current = activeRoom;
    if (selectedAssetId !== null) setSelectedAssetId(null);
  }

  // Duplicated from useStudio.pollProgress by design (isolation rule): the room
  // flow must not share mutable surface with the floor hook. It reads the same
  // GET /api/generate/<id> snapshot; room previews carry validation but no plan
  // geometry, so only the validator verdict + iterations update live.
  const pollProgress = React.useCallback(
    async (requestId: string, signal: AbortSignal, isFinished: () => boolean) => {
      while (!signal.aborted && !isFinished()) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, 700));
        if (signal.aborted || isFinished()) return;
        try {
          const progress = await fetchProgress(requestId, signal);
          if (progress.iterations.length) {
            setIterations(progress.iterations);
            setLiveIterations(progress.iterations);
          }
          if (progress.events) setLiveEvents(progress.events);
          const latest = progress.iterations[progress.iterations.length - 1];
          if (latest) {
            const score =
              latest.status === "running"
                ? "waiting for the model"
                : latest.score === null
                  ? "no executable layout"
                  : `score ${latest.score}`;
            setPhaseText(`Attempt ${latest.attempt} ${latest.phase} · ${latest.status} · ${score}`);
          } else if (progress.message) {
            setPhaseText(progress.message);
          }
          const preview = progress.preview;
          if (preview?.code) {
            setCode(preview.code);
            setCodeSource(`LIVE · ${[preview.source ?? "gemini-room", preview.model].filter(Boolean).join(" · ")}`);
          }
          if (preview?.validation) {
            setLiveValidation(preview.validation as unknown as RoomValidation);
          }
        } catch {
          // Polling is best-effort; the POST response carries the final result.
        }
      }
    },
    [],
  );

  const generateRoom = React.useCallback(async () => {
    if (busyAction) return;
    const requestId = crypto.randomUUID();
    const controller = new AbortController();
    abortRef.current = controller;
    requestIdRef.current = requestId;
    setBusyAction("generate");
    setLiveIterations([]);
    setLiveEvents([]);
    setRunLogOpen(false);
    setPhaseText("Gemini is laying out the ICU room, then repairing only if the validator rejects it");
    setStatus({ text: "Laying out the room…", error: false });
    let finished = false;
    let finalIterations: GenerationIteration[] | null = null;
    const progressTask = pollProgress(requestId, controller.signal, () => finished);
    const startedAt = performance.now();
    try {
      const payload = await postRoom({
        requestId,
        widthFt,
        depthFt,
        prompt,
        signal: controller.signal,
      });
      setResult(payload);
      setLiveValidation(null);
      setIterations(payload.iterations);
      finalIterations = payload.iterations;
      setCode(payload.code);
      setCodeSource(payload.model ? `${payload.source} · ${payload.model}` : payload.source);
      const seconds = ((performance.now() - startedAt) / 1000).toFixed(1);
      const acceptance = payload.accepted === false ? " · needs revision" : "";
      setStatus({ text: `Done in ${seconds}s · ${payload.source}${acceptance}`, error: false });
      if (payload.accepted === false) {
        toast.warning("Room generated but not accepted", {
          description: payload.ai_error ?? "Best attempt kept.",
        });
      }
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        setStatus({ text: "Generation stopped · adjust inputs or re-generate", error: false });
      } else {
        const message = error instanceof Error ? error.message : String(error);
        setStatus({ text: message, error: true });
        toast.error("Room generation failed", { description: message });
      }
    } finally {
      finished = true;
      await progressTask;
      if (requestIdRef.current === requestId) {
        if (finalIterations) setLiveIterations(finalIterations);
        abortRef.current = null;
        requestIdRef.current = null;
        setBusyAction(null);
      }
    }
  }, [busyAction, widthFt, depthFt, prompt, pollProgress]);

  const stopGeneration = React.useCallback(async () => {
    const requestId = requestIdRef.current;
    if (!requestId) return;
    setStatus({ text: "Stopping generation…", error: false });
    try {
      await cancelGeneration(requestId);
    } finally {
      abortRef.current?.abort();
    }
  }, []);

  return {
    preset,
    setPreset,
    widthFt,
    setWidthFt,
    depthFt,
    setDepthFt,
    areaFt2,
    prompt,
    setPrompt,
    viewport,
    setViewport,
    health,
    result,
    activeRoom,
    activeValidation,
    iterations,
    liveIterations,
    liveEvents,
    runLogOpen,
    setRunLogOpen,
    code,
    codeSource,
    busyAction,
    phaseText,
    elapsed,
    status,
    selectedAssetId,
    setSelectedAssetId,
    generateRoom,
    stopGeneration,
  };
}

export type RoomStudio = ReturnType<typeof useRoomStudio>;
