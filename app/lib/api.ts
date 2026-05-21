import type { DatasetId } from './constants';
import type { Plan } from './plan';
import type { GraphGenerationResponse, ModelStatusResponse } from './types';

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

export async function generateGraph(dataset: DatasetId): Promise<GraphGenerationResponse> {
  const res = await fetch(`${API_BASE}/api/generate/graph`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, num_samples: 1 }),
    signal: AbortSignal.timeout(180000),
  });
  if (!res.ok) throw await readError(res, 'Graph generation failed');
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

export interface ProceduralGraphInput {
  nodes: { id: number; room_type: string }[];
  edges: { source: number; target: number; connectivity: string }[];
}

export async function generateProcedural(
  graph: ProceduralGraphInput,
  boundary: [number, number][],
  seed = 0,
): Promise<Plan> {
  const res = await fetch(`${API_BASE}/api/generate/procedural`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ graph, boundary, seed }),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) throw await readError(res, 'Procedural generation failed');
  return res.json() as Promise<Plan>;
}
