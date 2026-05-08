'use client';

import { BubbleGraphCanvas } from '@/components/bubble-graph/BubbleGraphCanvas';
import type { GeneratedGraph } from '@/lib/types';

const FIXTURE_RPLAN: GeneratedGraph = {
  num_nodes: 6,
  num_edges: 7,
  rooms: [0, 1, 1, 2, 3, 4],
  adjacency: [],
  edge_types: [],
  nodes: [
    { id: 0, attr: 0, room_type: 'living' },
    { id: 1, attr: 1, room_type: 'bedroom' },
    { id: 2, attr: 1, room_type: 'bedroom' },
    { id: 3, attr: 2, room_type: 'bathroom' },
    { id: 4, attr: 3, room_type: 'kitchen' },
    { id: 5, attr: 4, room_type: 'balcony' },
  ],
  edges: [
    { source: 0, target: 1, edge_type: 2, edge_label: 'door' },
    { source: 0, target: 2, edge_type: 2, edge_label: 'door' },
    { source: 0, target: 3, edge_type: 1, edge_label: 'wall' },
    { source: 0, target: 4, edge_type: 2, edge_label: 'door' },
    { source: 0, target: 5, edge_type: 2, edge_label: 'door' },
    { source: 1, target: 3, edge_type: 1, edge_label: 'wall' },
    { source: 4, target: 5, edge_type: 1, edge_label: 'wall' },
  ],
};

export default function BubbleCanvasFixture() {
  return (
    <div className="min-h-screen bg-background p-8">
      <h1 className="mb-4 text-xl font-semibold">BubbleGraphCanvas — view mode (RPLAN fixture)</h1>
      <div className="h-[600px] w-full max-w-[900px] rounded-xl border bg-card p-4">
        <BubbleGraphCanvas graph={FIXTURE_RPLAN} dataset="rplan" mode="view" />
      </div>
    </div>
  );
}
