"use client";

import * as React from "react";
import { toast } from "sonner";

import { cancelGeneration, fetchHealth, fetchProgress, postRoom } from "@/lib/api";
import { ICU_CATALOG, ICU_CATALOG_BY_TYPE } from "@/lib/icu-catalog";
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

// Presets are IMMUTABLE templates. v1 ships one; a user duplicates a preset into a
// custom program (below) to edit its per-asset counts.
export const ROOM_PRESETS = [{ key: "icu-room-v1", label: "ICU Room v1" }] as const;

// A user-owned, editable clone of a preset. Only custom programs allow changing
// asset counts; they persist client-side under this localStorage key.
export const ROOM_PROGRAMS_STORAGE_KEY = "ppai.room.programs.v1";

export interface CustomProgram {
  id: string;
  name: string;
  basePreset: string; // e.g. "icu-room-v1"
  counts: Record<string, number>;
}

// The default per-type counts for a preset: the catalog's count_required. Cloning a
// preset seeds a custom program with these; the steppers then edit within min/max.
export function presetCounts(): Record<string, number> {
  return Object.fromEntries(ICU_CATALOG.map((entry) => [entry.type, entry.count_required]));
}

function clampCount(type: string, value: number): number {
  const entry = ICU_CATALOG_BY_TYPE[type];
  if (!entry) return Math.max(0, Math.round(value));
  return Math.max(entry.min_count, Math.min(entry.max_count, Math.round(value)));
}

const DEFAULT_WIDTH_FT = 16.5;
const DEFAULT_DEPTH_FT = 16.75;
const DEFAULT_ROOM_PROMPT =
  "Lay out a calm single-patient ICU room: bed head on a headwall with clear transfer and foot space, equipment on one side, and the visitor chair on the other.";

export function useRoomStudio() {
  // activeProgramId is either a preset key ("icu-room-v1") or a custom program id.
  const [activeProgramId, setActiveProgramId] = React.useState<string>(ROOM_PRESETS[0].key);
  const [programs, setPrograms] = React.useState<CustomProgram[]>([]);
  const [programsLoaded, setProgramsLoaded] = React.useState(false);
  // Set by createProgram so the panel can auto-focus the inline rename input once.
  const [renameFocusId, setRenameFocusId] = React.useState<string | null>(null);

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

  // Load persisted custom programs once on mount (client-only).
  React.useEffect(() => {
    try {
      const raw = window.localStorage.getItem(ROOM_PROGRAMS_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as CustomProgram[];
        if (Array.isArray(parsed)) {
          setPrograms(
            parsed.filter(
              (item): item is CustomProgram =>
                !!item && typeof item.id === "string" && typeof item.name === "string" && !!item.counts,
            ),
          );
        }
      }
    } catch {
      // A corrupt or unavailable store is non-fatal: the user simply starts from the preset.
    }
    setProgramsLoaded(true);
  }, []);

  // Persist after the initial load so we never clobber the store with the empty default.
  React.useEffect(() => {
    if (!programsLoaded) return;
    try {
      window.localStorage.setItem(ROOM_PROGRAMS_STORAGE_KEY, JSON.stringify(programs));
    } catch {
      // Ignore quota / privacy-mode failures; programs stay in memory for the session.
    }
  }, [programs, programsLoaded]);

  const activeProgram = React.useMemo(
    () => programs.find((program) => program.id === activeProgramId) ?? null,
    [programs, activeProgramId],
  );
  const isCustomActive = activeProgram !== null;

  // The counts driving the payload + panel: a custom program's edited counts, or the
  // immutable preset defaults. Presets send no counts to the backend (defaults apply).
  const activeCounts = React.useMemo<Record<string, number>>(
    () => activeProgram?.counts ?? presetCounts(),
    [activeProgram],
  );
  const totalAssetCount = React.useMemo(
    () => ICU_CATALOG.reduce((sum, entry) => sum + (activeCounts[entry.type] ?? 0), 0),
    [activeCounts],
  );

  // Duplicate the ACTIVE program (preset or custom) into a new editable custom entry,
  // select it, and flag it for inline rename focus.
  const createProgram = React.useCallback(() => {
    const sourceName = activeProgram
      ? activeProgram.name
      : (ROOM_PRESETS.find((preset) => preset.key === activeProgramId)?.label ?? "ICU Room");
    const basePreset = activeProgram?.basePreset ?? activeProgramId;
    const id =
      typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `prog-${Date.now()}`;
    const clone: CustomProgram = {
      id,
      name: `${sourceName} copy`,
      basePreset,
      counts: { ...activeCounts },
    };
    setPrograms((current) => [...current, clone]);
    setActiveProgramId(id);
    setRenameFocusId(id);
    return id;
  }, [activeProgram, activeProgramId, activeCounts]);

  const renameProgram = React.useCallback((id: string, name: string) => {
    setPrograms((current) => current.map((program) => (program.id === id ? { ...program, name } : program)));
  }, []);

  const deleteProgram = React.useCallback(
    (id: string) => {
      setPrograms((current) => current.filter((program) => program.id !== id));
      setActiveProgramId((current) => (current === id ? ROOM_PRESETS[0].key : current));
    },
    [],
  );

  // Edit one asset count on the ACTIVE custom program, clamped to the catalog bounds.
  // A no-op on a preset (presets are immutable).
  const setAssetCount = React.useCallback(
    (type: string, value: number) => {
      const id = activeProgramId;
      setPrograms((current) =>
        current.map((program) =>
          program.id === id
            ? { ...program, counts: { ...program.counts, [type]: clampCount(type, value) } }
            : program,
        ),
      );
    },
    [activeProgramId],
  );

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
        // Presets send nothing (backend defaults apply); custom programs send their counts.
        assets: isCustomActive ? activeCounts : null,
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
  }, [busyAction, widthFt, depthFt, prompt, isCustomActive, activeCounts, pollProgress]);

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
    activeProgramId,
    setActiveProgramId,
    programs,
    activeProgram,
    isCustomActive,
    activeCounts,
    totalAssetCount,
    createProgram,
    renameProgram,
    deleteProgram,
    setAssetCount,
    renameFocusId,
    setRenameFocusId,
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
