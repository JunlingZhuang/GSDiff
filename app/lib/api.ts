const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function generateUnconstrained(): Promise<{ image: string; rooms?: number }> {
  const res = await fetch(`${API_BASE}/api/generate/unconstrained`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(120000),
  });
  if (!res.ok) throw new Error('Generation failed');
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
  if (!res.ok) throw new Error('Generation failed');
  return res.json();
}

export async function generateBoundary(boundaryImage: string): Promise<{ image: string }> {
  const res = await fetch(`${API_BASE}/api/generate/boundary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ boundary_image: boundaryImage }),
    signal: AbortSignal.timeout(120000),
  });
  if (!res.ok) throw new Error('Generation failed');
  return res.json();
}
