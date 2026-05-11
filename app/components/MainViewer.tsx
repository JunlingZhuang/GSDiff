'use client';

import { ImageIcon, Network } from 'lucide-react';
import { BubbleGraphCanvas } from '@/components/bubble-graph/BubbleGraphCanvas';
import { BoundaryCanvas } from '@/components/BoundaryCanvas';
import { TopologyEditor } from '@/components/TopologyEditor';
import type { DatasetId, GenerationMode } from '@/lib/constants';
import type { HistoryItem } from '@/lib/history';
import type { GeneratedGraph } from '@/lib/types';

interface Props {
  mode: GenerationMode;
  selectedItem: HistoryItem | undefined;
  loading: boolean;
  error: string | null;
  dataset: DatasetId;
  topologyDraft?: GeneratedGraph;
  onTopologyChange?: (g: GeneratedGraph) => void;
  boundaryDraft?: string;
  onBoundaryChange?: (dataUrl: string) => void;
  onGenerateBoundary?: () => void;
}

export function MainViewer({
  mode,
  selectedItem,
  loading,
  error,
  dataset,
  topologyDraft,
  onTopologyChange,
  boundaryDraft,
  onBoundaryChange,
  onGenerateBoundary,
}: Props) {
  // Boundary mode: always render the canvas regardless of selectedItem
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

  // Topology mode: always render the editor regardless of selectedItem
  if (mode === 'topology') {
    return (
      <div className="flex flex-1 items-center justify-center overflow-hidden bg-background p-6">
        <div className="h-full w-full max-w-full">
          <TopologyEditor
            dataset={dataset}
            initialGraph={topologyDraft}
            onGraphChange={onTopologyChange ?? (() => {})}
          />
        </div>
      </div>
    );
  }

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
        {!error && !loading && !selectedItem && <EmptyState mode={mode} />}
        {!error && !loading && selectedItem && (
          selectedItem.kind === 'graph' ? (
            <BubbleGraphCanvas
              graph={selectedItem.graph}
              dataset={selectedItem.dataset}
              mode="view"
            />
          ) : (
            <img
              src={selectedItem.image}
              alt="Generated floorplan"
              className="max-h-full max-w-full rounded-xl border border-border/60 shadow-sm"
            />
          )
        )}
      </div>
    </div>
  );
}

function EmptyState({ mode }: { mode: GenerationMode }) {
  const Icon = mode === 'graph' ? Network : ImageIcon;
  const text =
    mode === 'graph'
      ? 'Click Sample Graph to generate a bubble diagram.'
      : 'Click Generate to sample a floorplan.';
  return (
    <div className="flex flex-col items-center gap-3 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-muted">
        <Icon className="h-7 w-7 text-muted-foreground/40" />
      </div>
      <p className="text-sm text-muted-foreground">{text}</p>
    </div>
  );
}
