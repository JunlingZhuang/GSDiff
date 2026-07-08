import type {
  AgentAction,
  CandidateImage,
  GenerationProgress,
  GenerationResult,
  HealthResponse,
  Program,
  Samples,
  SeedPlan,
} from "./types";

async function readJson<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as T & { error?: string };
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
