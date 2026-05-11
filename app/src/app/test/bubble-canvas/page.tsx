'use client';

import { useState } from 'react';
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

function EditModeDemo() {
  const [graph, setGraph] = useState<GeneratedGraph>(FIXTURE_RPLAN);
  return (
    <div className="mt-8">
      <h2 className="mb-4 text-lg font-semibold">Edit mode (RPLAN fixture)</h2>
      <p className="mb-2 text-sm text-muted-foreground">
        Click empty space to add a node. Click a node to select, DEL to delete. Drag node perimeter
        to another node to add an edge. Click an edge to cycle its type, right-click to delete.
        Double-click a node to cycle its room type. Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) for
        undo/redo.
      </p>
      <div className="h-[600px] w-full max-w-[900px] rounded-xl border bg-card p-4">
        <BubbleGraphCanvas
          graph={graph}
          dataset="rplan"
          mode="edit"
          onChange={setGraph}
          defaultNodeAttrInitial={1}
          defaultEdgeTypeInitial={1}
        />
      </div>
      <pre className="mt-2 max-h-32 overflow-auto rounded bg-muted p-2 text-xs">
        {JSON.stringify({ n: graph.num_nodes, e: graph.num_edges }, null, 2)}
      </pre>
    </div>
  );
}

export default function BubbleCanvasFixture() {
  return (
    <div className="min-h-screen bg-background p-8">
      <h1 className="mb-4 text-xl font-semibold">BubbleGraphCanvas — view mode (RPLAN fixture)</h1>
      <div className="h-[600px] w-full max-w-[900px] rounded-xl border bg-card p-4">
        <BubbleGraphCanvas graph={FIXTURE_RPLAN} dataset="rplan" mode="view" />
      </div>
      <EditModeDemo />
    </div>
  );
}
