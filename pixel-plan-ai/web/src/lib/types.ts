export interface ProgramRoom {
  type: string;
  count: number;
  approx_area_m2?: number;
  approx_area_ft2?: number;
}

export interface Program {
  building_type: string;
  rooms: ProgramRoom[];
  adjacency: [string, string][];
}

export interface RoomBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PlanRoom {
  id: string;
  type: string;
  color: string;
  pixel_count: number;
  bounds: RoomBounds;
}

export interface Door {
  id: string;
  from_room: string;
  to_room: string | null;
  x: number;
  y: number;
  orientation: "horizontal" | "vertical";
  width_cells: number;
  swing_side: "north" | "south" | "east" | "west";
}

export interface TokenUsage {
  prompt_tokens: number;
  candidate_tokens: number;
  thinking_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number | null;
}

export interface Plan {
  width: number;
  height: number;
  meters_per_cell: number;
  footprint: number[];
  cells: number[];
  rooms: PlanRoom[];
  doors: Door[];
}

export interface Check {
  category: string;
  label: string;
  pass: boolean;
}

export interface AreaRow {
  id: string;
  type: string;
  target_m2: number;
  actual_m2: number;
  error_percent: number;
  pass: boolean;
  target_ft2?: number;
  actual_ft2?: number;
  tolerance_percent?: number;
}

export interface Validation {
  score: number;
  checks: Check[];
  issues: string[];
  areas: AreaRow[];
  summary: {
    expected_rooms: number;
    generated_rooms: number;
    doors: number;
    rooms_with_doors: number;
    required_adjacencies: number;
    satisfied_adjacencies: number;
    coverage_percent: number;
    rule_profile?: string;
  };
}

export interface GenerationIteration {
  attempt: number;
  phase: "initial" | "repair" | "run" | "inspect" | "fix" | "revise";
  source: string;
  model: string | null;
  status: "running" | "accepted" | "rejected" | "failed";
  duration_ms: number;
  started_at_ms?: number;
  score: number | null;
  message: string;
  issues: string[];
  code?: string | null;
  usage?: TokenUsage | null;
}

export interface GenerationProgress {
  status: "starting" | "running" | "complete" | "missing";
  phase: string;
  message: string;
  active: boolean;
  iterations: GenerationIteration[];
  preview: {
    code: string | null;
    source: string | null;
    model: string | null;
    plan?: Plan;
    validation?: Validation;
  } | null;
}

export interface GenerationResult {
  accepted?: boolean;
  source: string;
  provider: string | null;
  model: string | null;
  strategy: string;
  assumptions: string[];
  code: string;
  plan: Plan;
  validation: Validation;
  program: Program;
  prompt: string;
  generated_at: string;
  iterations: GenerationIteration[];
  ai_error?: string;
  stop_reason?: string;
  usage_total?: { total_tokens: number; estimated_cost_usd: number | null };
  inspection?: {
    accepted: boolean;
    summary: string;
    issues: string[];
  };
}

export interface HealthResponse {
  ok: boolean;
  gemini_configured: boolean;
  model: string;
  fast_model?: string;
  quality_model?: string;
  complex_model?: string;
}

export interface CandidateImage {
  mime: string;
  data: string;
}

export type Samples = Record<string, Program>;

export type AgentAction = "generate" | "run" | "inspect" | "fix" | "revise";

export type StudioMode = "program" | "image" | "trace";

export interface SeedPlan {
  width: number;
  height: number;
  meters_per_cell: number;
  cells: number[] | number[][];
  rooms: { id?: string; type: string }[];
  doors?: {
    id?: string;
    from_room: string;
    to_room: string | null;
    x: number;
    y: number;
    orientation: "horizontal" | "vertical";
    width_cells?: number;
  }[];
}

export interface TraceDiagnostics {
  rooms: number;
  doors: number;
  typed: number;
  meters_per_pixel: number | null;
}

export interface TraceResponse {
  seed: SeedPlan & { meta?: Record<string, unknown> };
  diagnostics: TraceDiagnostics;
  artifacts: { linework?: CandidateImage };
}

export interface TracerHealth {
  ok: boolean;
  programs: string[];
}

export function candidateDataUrl(candidate: CandidateImage): string {
  return `data:${candidate.mime};base64,${candidate.data}`;
}

export function prettyType(type: string): string {
  return type.replaceAll("_", " ");
}
