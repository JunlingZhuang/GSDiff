'use client';

import { useEffect, useState } from 'react';
import { BubbleGraphCanvas } from '@/components/bubble-graph/BubbleGraphCanvas';
import type { DatasetId } from '@/lib/constants';
import type { GeneratedGraph } from '@/lib/types';

interface Props {
  dataset: DatasetId;
  initialGraph?: GeneratedGraph;
  defaultNodeAttrInitial?: number;
  defaultEdgeTypeInitial?: number;
  onGraphChange: (graph: GeneratedGraph) => void;
}

export const DEFAULT_TOPOLOGY_GRAPH: GeneratedGraph = {
  num_nodes: 4,
  num_edges: 3,
  rooms: [0, 1, 2, 3],
  adjacency: [],
  edge_types: [],
  nodes: [
    { id: 0, attr: 0, room_type: 'living' },
    { id: 1, attr: 1, room_type: 'bedroom' },
    { id: 2, attr: 2, room_type: 'bathroom' },
    { id: 3, attr: 3, room_type: 'kitchen' },
  ],
  edges: [
    { source: 0, target: 1, edge_type: 2, edge_label: 'door' },
    { source: 0, target: 2, edge_type: 1, edge_label: 'wall' },
    { source: 0, target: 3, edge_type: 2, edge_label: 'door' },
  ],
};

export function TopologyEditor({
  dataset,
  initialGraph,
  defaultNodeAttrInitial,
  defaultEdgeTypeInitial,
  onGraphChange,
}: Props) {
  const [graph, setGraph] = useState<GeneratedGraph>(initialGraph ?? DEFAULT_TOPOLOGY_GRAPH);

  useEffect(() => {
    onGraphChange(graph);
  }, [graph, onGraphChange]);

  return (
    <BubbleGraphCanvas
      graph={graph}
      dataset={dataset}
      mode="edit"
      onChange={setGraph}
      defaultNodeAttrInitial={defaultNodeAttrInitial}
      defaultEdgeTypeInitial={defaultEdgeTypeInitial}
    />
  );
}
