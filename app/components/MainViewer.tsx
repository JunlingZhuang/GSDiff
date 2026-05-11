'use client';

import { ImageIcon } from 'lucide-react';
import { BubbleGraphCanvas } from '@/components/bubble-graph/BubbleGraphCanvas';
import { BoundaryCanvas } from '@/components/BoundaryCanvas';
import type { DatasetId, GenerationMode } from '@/lib/constants';
import type { HistoryItem } from '@/lib/history';
import type { GeneratedGraph } from '@/lib/types';

interface Props {
  mode: GenerationMode;
  selectedItem: HistoryItem | undefined;
  loading: boolean;
  error: string | null;
  dataset: DatasetId;
  boundaryDraft?: string;
  onBoundaryChange?: (dataUrl: string) => void;
  onGenerateBoundary?: () => void;
  onGraphDraftChange?: (g: GeneratedGraph) => void;
}

// Initial canvas state when there is no selected graph yet. Four rooms in a
// circle gives the user something to immediately edit rather than an empty
// canvas. The same default is used whether the user is starting from scratch
// or has just switched into Graph mode.
const DEFAULT_GRAPH: GeneratedGraph = {
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

export function MainViewer({
  mode,
  selectedItem,
  loading,
  error,
  dataset,
  boundaryDraft,
  onBoundaryChange,
  onGenerateBoundary,
  onGraphDraftChange,
}: Props) {
  // Boundary mode: always render the drawing canvas.
  if (mode === 'boundary') {
    return (
      <div className="flex flex-1 items-center justify-center overflow-hidden bg-background p-6">
        <BoundaryCanvas
          value={boundaryDraft}
          onChange={onBoundaryChange}
          onGenerate={() => onGenerateBoundary?.()}
          loading={loading}
        />
      </div>
    );
  }

  // Graph mode: always render the editable bubble canvas. Source is either
  // the currently-selected sampled graph from history, or the small default
  // graph so the user has something to start editing from.
  if (mode === 'graph') {
    const sourceGraph =
      selectedItem?.kind === 'graph' ? selectedItem.graph : DEFAULT_GRAPH;
    return (
      <div className="flex flex-1 items-center justify-center overflow-hidden bg-background p-6">
        <div className="h-full w-full">
          {error ? (
            <div className="flex h-full items-center justify-center">
              <div className="rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
                {error}
              </div>
            </div>
          ) : loading ? (
            <div className="flex h-full items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-muted-foreground/20 border-t-muted-foreground/60" />
                <p className="text-sm text-muted-foreground">Sampling graph...</p>
              </div>
            </div>
          ) : (
            <BubbleGraphCanvas
              graph={sourceGraph}
              dataset={selectedItem?.kind === 'graph' ? selectedItem.dataset : dataset}
              mode="edit"
              onChange={onGraphDraftChange}
            />
          )}
        </div>
      </div>
    );
  }

  // Unconstrained mode (and the default floorplan viewer): show the selected
  // floorplan, an empty-state hint, a spinner, or an error.
  return (
    <div className="flex flex-1 items-center justify-center overflow-hidden bg-background p-6">
      <div className="flex h-full w-full max-w-full items-center justify-center">
        {error && (
          <div className="rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}
        {!error && loading && (
          <div className="flex flex-col items-center gap-3">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-muted-foreground/20 border-t-muted-foreground/60" />
            <p className="text-sm text-muted-foreground">Generating...</p>
          </div>
        )}
        {!error && !loading && !selectedItem && <FloorplanEmptyState />}
        {!error && !loading && selectedItem?.kind === 'floorplan' && (
          <img
            src={selectedItem.image}
            alt="Generated floorplan"
            className="max-h-full max-w-full rounded-xl border border-border/60 shadow-sm"
          />
        )}
      </div>
    </div>
  );
}

function FloorplanEmptyState() {
  return (
    <div className="flex flex-col items-center gap-3 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-muted">
        <ImageIcon className="h-7 w-7 text-muted-foreground/40" />
      </div>
      <p className="text-sm text-muted-foreground">
        Click Generate to sample a floorplan.
      </p>
    </div>
  );
}
