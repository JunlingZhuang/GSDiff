import type { DatasetId, GraphModelId, RetrievalMode } from './constants';
import type { Plan } from './plan';
import type {
  GeneratedGraph,
  GraphGenerationResponse,
  ModelStatusResponse,
  RetrieveResponse,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function readError(res: Response, fallback: string): Promise<Error> {
  try {
    const body = await res.json();
    return new Error(body.detail || fallback);
  } catch {
    return new Error(fallback);
  }
}

export async function generateUnconstrained(): Promise<{ image: string; rooms?: number }> {
  const res = await fetch(`${API_BASE}/api/generate/unconstrained`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(120000),
  });
  if (!res.ok) throw await readError(res, 'Generation failed');
  return res.json();
}

export async function generateGraph(
  dataset: DatasetId,
  model?: GraphModelId,
): Promise<GraphGenerationResponse> {
  const res = await fetch(`${API_BASE}/api/generate/graph`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, model, num_samples: 1 }),
    signal: AbortSignal.timeout(180000),
  });
  if (!res.ok) throw await readError(res, 'Graph generation failed');
  return res.json();
}

export async function completeNextNode(
  dataset: DatasetId,
  model: GraphModelId,
  graph: GeneratedGraph,
): Promise<GraphGenerationResponse> {
  const res = await fetch(`${API_BASE}/api/generate/graph/next-node`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, model, graph, num_candidates: 1 }),
    signal: AbortSignal.timeout(180000),
  });
  if (!res.ok) throw await readError(res, 'Next-node completion failed');
  return res.json();
}

export async function completeGraph(
  dataset: DatasetId,
  model: GraphModelId,
  graph: GeneratedGraph,
  targetNumNodes = 30,
): Promise<GraphGenerationResponse> {
  const res = await fetch(`${API_BASE}/api/generate/graph/completion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset,
      model,
      graph,
      target_num_nodes: targetNumNodes,
      num_candidates: 8,
    }),
    signal: AbortSignal.timeout(240000),
  });
  if (!res.ok) throw await readError(res, 'Graph completion failed');
  return res.json();
}

export async function getModelStatus(): Promise<ModelStatusResponse> {
  const res = await fetch(`${API_BASE}/api/models/status`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(10000),
  });
  if (!res.ok) throw await readError(res, 'Failed to load model status');
  return res.json();
}

export async function generateTopology(
  rooms: number[],
  adjacency: number[][],
): Promise<{ image: string }> {
  const res = await fetch(`${API_BASE}/api/generate/topology`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rooms, adjacency }),
    signal: AbortSignal.timeout(120000),
  });
  if (!res.ok) throw await readError(res, 'Generation failed');
  return res.json();
}

export async function generateBoundary(boundaryImage: string): Promise<{ image: string }> {
  const res = await fetch(`${API_BASE}/api/generate/boundary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ boundary_image: boundaryImage }),
    signal: AbortSignal.timeout(120000),
  });
  if (!res.ok) throw await readError(res, 'Generation failed');
  return res.json();
}

export async function retrieveGraph(
  query: GeneratedGraph,
  mode: RetrievalMode,
  k: number,
): Promise<RetrieveResponse> {
  // The server only needs node room types + edge endpoints + edge types, not
  // the full GeneratedGraph (no need to send adjacency matrix).
  const body = {
    rooms: query.rooms,
    edges: query.edges.map((e) => ({ source: e.source, target: e.target, edge_type: e.edge_type })),
    mode,
    k,
  };
  const res = await fetch(`${API_BASE}/api/retrieve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) throw await readError(res, 'Retrieval failed');
  return res.json();
}

export interface ProceduralGraphInput {
  nodes: { id: number; room_type: string }[];
  edges: { source: number; target: number; connectivity: string }[];
}

export async function generateProcedural(
  graph: ProceduralGraphInput,
  boundary: [number, number][],
  seed = 0,
  axisAngle: number | null = null,
): Promise<Plan> {
  const res = await fetch(`${API_BASE}/api/generate/procedural`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ graph, boundary, seed, axis_angle: axisAngle }),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) throw await readError(res, 'Procedural generation failed');
  return res.json() as Promise<Plan>;
}
