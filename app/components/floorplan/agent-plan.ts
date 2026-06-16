// Bridge between the healthcare agent backend (hfagent, port 8100) and the
// CAD kernel: the backend wall graph is structurally isomorphic to the
// kernel's WallGraph by design, so conversion is unit scaling plus bookkeeping.

import type { RoomType } from '@/lib/plan';
import type { WallGraph } from './kernel';

export interface AgentWallGraphJSON {
  units: string; // 'px' (parsed from imagery) or 'mm'
  nodes: Record<string, { id: string; x: number; y: number }>;
  walls: { id: string; n0: string; n1: string; thickness: number; height?: number; rooms: string[] }[];
  rooms: { id: string; type: string; loop: string[] }[];
}

export interface AgentProgram {
  building_type: string;
  rooms: { type: string; count: number; approx_area_m2?: number }[];
  adjacency: [string, string][];
}

export interface AgentGenerateResult {
  id: number;
  program: AgentProgram;
  report: {
    rounds: { round: number; mismatches: string[] }[];
    room_count_exact: boolean;
    final_count_exact: boolean;
    pipeline: string;
    image_model: string;
  };
  wallgraph: AgentWallGraphJSON;
}

// px plans are parsed from ~1400px renders of ~25m buildings: 0.02 m/px keeps
// real-world-ish dimensions so areas/lengths read sensibly in the editor.
const M_PER_PX = 0.02;
const WALL_THICKNESS_M = 0.2;

export function agentToKernelGraph(wg: AgentWallGraphJSON): WallGraph {
  const s = wg.units === 'mm' ? 0.001 : M_PER_PX;
  const nodes: WallGraph['nodes'] = {};
  for (const id in wg.nodes) {
    const n = wg.nodes[id];
    // image y grows downward; the editor world is y-up — flip to match the
    // generated imagery orientation
    nodes[id] = { id, x: n.x * s, y: -n.y * s };
  }
  return {
    nodes,
    walls: wg.walls.map((w) => ({ id: w.id, n0: w.n0, n1: w.n1, thickness: WALL_THICKNESS_M })),
    rooms: wg.rooms.map((r) => ({ id: r.id, type: r.type as RoomType, loop: r.loop })),
    openings: [],
    grid: { originX: 0, originY: 0, spacingX: 1, spacingY: 1, angleDeg: 0 },
  };
}
