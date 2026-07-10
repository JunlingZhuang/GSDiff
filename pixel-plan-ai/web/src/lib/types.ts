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

// Typed JSONL timeline emitted alongside the progress snapshot (backend
// job_progress.append_job_event). The discriminant is `e`; `t` is a unix
// timestamp in seconds. Extra fields vary by event type, so they stay optional.
export interface GenerationEvent {
  e:
    | "attempt_start"
    | "model_returned"
    | "validator_verdict"
    | "exec_result"
    | "tool_check"
    | "attempt_end"
    | "run_end"
    | string;
  t: number;
  attempt?: number;
  phase?: string;
  model?: string | null;
  score?: number | null;
  rejected?: boolean;
  status?: string;
  stop_reason?: string;
  accepted?: boolean;
  tokens?: number | null;
}

export interface GenerationProgress {
  status: "starting" | "running" | "complete" | "missing";
  phase: string;
  message: string;
  active: boolean;
  iterations: GenerationIteration[];
  events?: GenerationEvent[];
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
  // Present when the tracer service exposes POST /draw (authoritative drawer).
  draw?: boolean;
}

// Deterministic room-count verification returned per candidate by the hfagent
// tracer's /draw: how many rooms the trace found vs the program's total, and
// whether the candidate was redrawn once (the drawer's hard retry cap).
export interface DrawDiagnostics {
  rooms_expected: number;
  rooms_found: number;
  retried: boolean;
}

export interface TracerDrawImage extends CandidateImage, DrawDiagnostics {}

export interface TracerDrawResponse {
  images: TracerDrawImage[];
  count: number;
  source: string;
}

// --- ICU room flow (parallel to the floor flow above) ----------------------
// The room backend (POST /api/room) speaks feet: x = 0 at the West wall, y = 0
// at the South wall, and an asset's (x_ft, y_ft) is its SOUTH-WEST corner.
export type RoomWall = "N" | "S" | "E" | "W";

export interface RoomDoor {
  wall: RoomWall;
  offset_ft: number;
  width_ft: number;
}

export interface RoomDims {
  width_ft: number;
  depth_ft: number;
  door: RoomDoor;
}

export interface RoomAsset {
  id: string;
  type: string;
  x_ft: number;
  y_ft: number;
  w_ft: number;
  d_ft: number;
  rotation_deg: number;
  wall: RoomWall | null;
  anchor: string;
}

export interface RoomLayout {
  room: RoomDims;
  assets: RoomAsset[];
}

// The room validator mirrors the floor check shape ({category,label,pass}) but
// carries a room-scale summary and no per-space area report.
export interface RoomValidation {
  score: number;
  checks: Check[];
  issues: string[];
  summary: {
    assets_placed: number;
    checks_passed: number;
    checks_total: number;
    rule_profile?: string;
  };
}

// The floor-style envelope returned by POST /api/room.
export interface RoomResult {
  accepted?: boolean;
  source: string;
  provider?: string | null;
  model: string | null;
  strategy?: string;
  assumptions?: string[];
  code: string;
  room: RoomLayout;
  validation: RoomValidation;
  program?: { width_ft: number; depth_ft: number };
  prompt?: string;
  iterations: GenerationIteration[];
  stop_reason?: string;
  usage_total?: { total_tokens: number; estimated_cost_usd: number | null };
  ai_error?: string;
  generated_at?: string;
}

export function candidateDataUrl(candidate: CandidateImage): string {
  return `data:${candidate.mime};base64,${candidate.data}`;
}

export function prettyType(type: string): string {
  return type.replaceAll("_", " ");
}
