'use client';

import { useCallback, useState } from 'react';
import { Network, Send, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { BubbleGraphView } from '@/components/BubbleGraphView';
import type { GeneratedGraph } from '@/lib/types';

interface Props {
  datasetName: string;
  onSampleGraph: () => Promise<GeneratedGraph | null>;
  onGenerateFloorplan: (rooms: number[], adjacency: number[][]) => void;
  loading: boolean;
}

export function GraphPanel({ datasetName, onSampleGraph, onGenerateFloorplan, loading }: Props) {
  const [graph, setGraph] = useState<GeneratedGraph | null>(null);
  const [sampling, setSampling] = useState(false);

  const handleSample = useCallback(async () => {
    setSampling(true);
    try {
      const nextGraph = await onSampleGraph();
      if (!nextGraph) return;
      setGraph(nextGraph);
    } finally {
      setSampling(false);
    }
  }, [onSampleGraph]);

  const handleGenerateFloorplan = useCallback(() => {
    if (!graph) return;
    onGenerateFloorplan(graph.rooms, graph.adjacency);
  }, [graph, onGenerateFloorplan]);

  return (
    <div className="flex flex-1 flex-col">
      <div className="border-b border-border/60 px-4 py-3">
        <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          DiGress Graph Generator
        </p>
        <p className="mt-1 text-sm text-muted-foreground">
          Sample a {datasetName} room graph first, then pass its topology to GSDiff.
        </p>
      </div>

      <div className="relative min-h-[340px] flex-1 bg-muted/20">
        {!graph && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-8 text-center pointer-events-none">
            <Network className="mb-3 h-9 w-9 text-muted-foreground/35" />
            <p className="text-sm font-medium text-foreground">No graph yet</p>
            <p className="mt-1 max-w-xs text-xs leading-relaxed text-muted-foreground">
              Generate a bubble graph with room types and adjacency before creating a floorplan.
            </p>
          </div>
        )}
        {graph && <BubbleGraphView graph={graph} />}
      </div>

      <div className="space-y-3 border-t border-border/60 p-4">
        {graph && (
          <div className="rounded-xl border border-border/70 bg-background px-3 py-2 text-xs text-muted-foreground">
            <span className="font-medium text-foreground">{graph.num_nodes}</span> rooms,
            {' '}
            <span className="font-medium text-foreground">{graph.num_edges}</span> edges
          </div>
        )}
        <div className="grid grid-cols-2 gap-2">
          <Button
            onClick={handleSample}
            disabled={loading || sampling}
            className="h-10 text-sm font-medium"
            variant="secondary"
          >
            {sampling ? (
              <span className="mr-2 h-3.5 w-3.5 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-muted-foreground" />
            ) : (
              <Sparkles className="mr-2 h-3.5 w-3.5" />
            )}
            Generate Graph
          </Button>
          <Button
            onClick={handleGenerateFloorplan}
            disabled={!graph || loading || sampling}
            className="h-10 text-sm font-medium"
          >
            <Send className="mr-2 h-3.5 w-3.5" />
            Send to GSDiff
          </Button>
        </div>
      </div>
    </div>
  );
}
