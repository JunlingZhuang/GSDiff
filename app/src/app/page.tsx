'use client';

import { useState, useCallback } from 'react';
import { Header } from '@/components/Header';
import { ModeSelector } from '@/components/ModeSelector';
import { DatasetSelector } from '@/components/DatasetSelector';
import { ModelStatusPanel } from '@/components/ModelStatusPanel';
import { MainViewer } from '@/components/MainViewer';
import { GenerateButton } from '@/components/GenerateButton';
import {
  generateUnconstrained,
  generateGraph,
  generateTopology,
} from '@/lib/api';
import { DATASETS, type DatasetId, type GenerationMode } from '@/lib/constants';
import { Button } from '@/components/ui/button';
import { HistoryBar } from '@/components/HistoryBar';
import {
  prependHistory,
  createGraphItem,
  createFloorplanItem,
  findHistoryItem,
  type HistoryItem,
  type GraphHistoryItem,
} from '@/lib/history';

export default function Home() {
  const [activeMode, setActiveMode] = useState<GenerationMode>('unconstrained');
  const [selectedDataset, setSelectedDataset] = useState<DatasetId>('rplan');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedDatasetMeta = DATASETS.find((dataset) => dataset.id === selectedDataset) ?? DATASETS[0];
  const selectedItem = findHistoryItem(history, selectedHistoryId);

  const handleUnconstrained = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await generateUnconstrained();
      const item = createFloorplanItem(selectedDataset, res.image, 'unconstrained');
      setHistory((prev) => prependHistory(prev, item));
      setSelectedHistoryId(item.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  }, [selectedDataset]);

  const handleSampleGraph = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await generateGraph(selectedDataset);
      const graph = res.graphs[0];
      if (!graph) throw new Error('Server returned no graph');
      const item = createGraphItem(selectedDataset, graph);
      setHistory((prev) => prependHistory(prev, item));
      setSelectedHistoryId(item.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Graph generation failed');
    } finally {
      setLoading(false);
    }
  }, [selectedDataset]);

  const handleSendGraphToFloorplan = useCallback(async (graphItem: GraphHistoryItem) => {
    setLoading(true);
    setError(null);
    try {
      const res = await generateTopology(graphItem.graph.rooms, graphItem.graph.adjacency);
      const item = createFloorplanItem(selectedDataset, res.image, 'graph', graphItem.id);
      setHistory((prev) => prependHistory(prev, item));
      setSelectedHistoryId(item.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  }, [selectedDataset]);

  return (
    <div className="flex h-full flex-col">
      <Header />

      <div className="flex flex-1 overflow-hidden">
        {/* Left controls panel (320px) */}
        <aside className="flex w-[320px] shrink-0 flex-col border-r border-border/60 bg-card">
          <div className="flex-1 space-y-4 overflow-auto p-4">
            <DatasetSelector value={selectedDataset} onChange={setSelectedDataset} />
            <ModeSelector activeMode={activeMode} onModeChange={setActiveMode} />
            <ModelStatusPanel />

            {/* Mode-specific controls */}
            {activeMode === 'graph' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Sample bubble graphs from {selectedDatasetMeta.name}.
              </div>
            )}
            {activeMode === 'topology' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Topology editor will move into the right workspace (Task 9).
              </div>
            )}
            {activeMode === 'boundary' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Boundary canvas will move into the right workspace (Task 10).
              </div>
            )}
          </div>

          <div className="border-t border-border/60 p-4">
            <GenerateButton
              mode={activeMode}
              loading={loading}
              onClick={() => {
                if (activeMode === 'unconstrained') handleUnconstrained();
                else if (activeMode === 'graph') handleSampleGraph();
              }}
              disabled={activeMode !== 'unconstrained' && activeMode !== 'graph'}
            />
          </div>
        </aside>

        {/* Right workspace */}
        <div className="flex flex-1 flex-col overflow-hidden">
          <MainViewer
            mode={activeMode}
            selectedItem={selectedItem}
            loading={loading}
            error={error}
          />

          <HistoryBar
            items={history}
            selectedId={selectedHistoryId}
            onSelect={setSelectedHistoryId}
            onClear={() => { setHistory([]); setSelectedHistoryId(null); }}
            rightSlot={selectedItem?.kind === 'graph' ? (
              <Button onClick={() => handleSendGraphToFloorplan(selectedItem)} disabled={loading} size="sm">
                Send to Floorplan
              </Button>
            ) : null}
          />
        </div>
      </div>
    </div>
  );
}
