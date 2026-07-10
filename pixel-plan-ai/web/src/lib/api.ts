import type {
  AgentAction,
  CandidateImage,
  GenerationProgress,
  GenerationResult,
  HealthResponse,
  Program,
  RoomResult,
  Samples,
  SeedPlan,
  TraceResponse,
  TracerDrawResponse,
  TracerHealth,
} from "./types";

async function readJson<T>(response: Response): Promise<T> {
  // Read the body once as text so a non-JSON transport error (e.g. a proxy
  // returning plain-text "Internal Server Error" when a long generation
  // exceeds the proxy timeout) can be reported verbatim instead of leaking a
  // JSON parser's "Unexpected token 'I'..." message.
  const body = await response.text();
  let payload: (T & { error?: string }) | undefined;
  try {
    payload = JSON.parse(body) as T & { error?: string };
  } catch {
    const reason = body.trim().slice(0, 120) || response.statusText || "no response body";
    throw new Error(`Backend request failed (${response.status}): ${reason}`);
  }
  if (!response.ok || payload.error) {
    throw new Error(payload.error ?? `Request failed with ${response.status}.`);
  }
  return payload;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return readJson(await fetch("/api/health", { cache: "no-store" }));
}

export async function fetchSamples(): Promise<Samples> {
  return readJson(await fetch("/api/samples", { cache: "no-store" }));
}

export interface AgentRequest {
  requestId: string;
  action: AgentAction;
  program: Program;
  prompt: string;
  code?: string;
  currentCode?: string;
  width: number;
  height: number;
  referenceImage?: CandidateImage | null;
  seed?: SeedPlan | null;
  signal: AbortSignal;
}

export async function postAgentAction(request: AgentRequest): Promise<GenerationResult> {
  const endpoint = request.seed
    ? "/api/refine"
    : request.action === "run" || request.action === "inspect"
      ? "/api/execute"
      : "/api/generate";
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: request.signal,
    body: JSON.stringify({
      request_id: request.requestId,
      action: request.action,
      program: request.program,
      prompt: request.prompt,
      code: request.code,
      current_code: request.currentCode,
      mode: "auto",
      options: { width: request.width, height: request.height },
      reference_image: request.referenceImage ?? undefined,
      seed: request.seed ?? undefined,
    }),
  });
  return readJson(await Promise.resolve(response));
}

export interface RoomRequest {
  requestId: string;
  widthFt: number;
  depthFt: number;
  prompt: string;
  // Requested per-type asset counts (custom programs only); omitted for a preset,
  // which lets the backend apply its catalog defaults.
  assets?: Record<string, number> | null;
  signal: AbortSignal;
}

// POST /api/room runs on the shared job runner, so progress + events are polled
// through the same GET /api/generate/<request_id> as the floor flow.
export async function postRoom(request: RoomRequest): Promise<RoomResult> {
  const response = await fetch("/api/room", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: request.signal,
    body: JSON.stringify({
      request_id: request.requestId,
      room: { width_ft: request.widthFt, depth_ft: request.depthFt },
      prompt: request.prompt,
      assets: request.assets ?? undefined,
    }),
  });
  return readJson(await Promise.resolve(response));
}

export async function fetchProgress(requestId: string, signal: AbortSignal): Promise<GenerationProgress> {
  const response = await fetch(`/api/generate/${encodeURIComponent(requestId)}`, {
    method: "GET",
    signal,
    cache: "no-store",
  });
  return readJson(await Promise.resolve(response));
}

export async function cancelGeneration(requestId: string): Promise<void> {
  await fetch(`/api/generate/${encodeURIComponent(requestId)}`, { method: "DELETE" });
}

// The hfagent tracer is a separate localhost service reached through the
// /trace-api rewrite; both calls stay best-effort so trace mode can degrade to
// the paste/upload seed path whenever the service is offline.
export async function fetchTracerHealth(signal?: AbortSignal): Promise<TracerHealth> {
  return readJson(await fetch("/trace-api/health", { cache: "no-store", signal }));
}

export async function postTrace(image: CandidateImage, program: string): Promise<TraceResponse> {
  const response = await fetch("/trace-api/trace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image, program }),
  });
  return readJson(await Promise.resolve(response));
}

// Draw candidates with hfagent's authoritative linework pipeline (each candidate
// deterministically room-count verified). The full program object goes over the
// wire so a hand-edited program still drives the drawing; the caller falls back to
// the pixel-plan drawer (postDrawCandidates) whenever this rejects.
export async function postDrawViaTracer(program: Program, count: number): Promise<TracerDrawResponse> {
  const response = await fetch("/trace-api/draw", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ program, count }),
  });
  const payload = await readJson<TracerDrawResponse>(await Promise.resolve(response));
  if (!payload.images?.length) throw new Error("The hfagent tracer returned no drawings.");
  return payload;
}

export async function postDrawCandidates(program: Program, count: number): Promise<CandidateImage[]> {
  const response = await fetch("/api/images", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id: crypto.randomUUID(), program, count }),
  });
  const payload = await readJson<{ images?: CandidateImage[] }>(await Promise.resolve(response));
  if (!payload.images?.length) throw new Error("The image model returned no drawings.");
  return payload.images;
}
