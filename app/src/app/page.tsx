'use client';

import { useState, useCallback } from 'react';
import { Header } from '@/components/Header';
import { ModeSelector } from '@/components/ModeSelector';
import { DatasetSelector } from '@/components/DatasetSelector';
import { ModelStatusPanel } from '@/components/ModelStatusPanel';
import { ResultPanel } from '@/components/ResultPanel';
import { HistoryPanel } from '@/components/HistoryPanel';
import { MainViewer } from '@/components/MainViewer';
import { GenerateButton } from '@/components/GenerateButton';
import {
  generateUnconstrained,
  generateGraph,
  generateTopology,
  generateBoundary,
} from '@/lib/api';
import { DATASETS, type DatasetId, type GenerationMode } from '@/lib/constants';
import type { GeneratedGraph, HistoryItem } from '@/lib/types';

export default function Home() {
  const [activeMode, setActiveMode] = useState<GenerationMode>('unconstrained');
  const [selectedDataset, setSelectedDataset] = useState<DatasetId>('rplan');
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const selectedDatasetMeta = DATASETS.find((dataset) => dataset.id === selectedDataset) ?? DATASETS[0];

  const addToHistory = useCallback((img: string, mode: GenerationMode) => {
    setHistory(prev => [{
      id: Date.now(),
      image: img,
      mode,
      timestamp: new Date(),
    }, ...prev]);
  }, []);

  const handleUnconstrained = useCallback(async () => {
    setLoading(true);
    setError(null);
    setImage(null);
    try {
      const res = await generateUnconstrained();
      setImage(res.image);
      addToHistory(res.image, 'unconstrained');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  }, [addToHistory]);

  const handleTopology = useCallback(
    async (rooms: number[], adjacency: number[][]) => {
      setLoading(true);
      setError(null);
      setImage(null);
      try {
        const res = await generateTopology(rooms, adjacency);
        setImage(res.image);
        addToHistory(res.image, 'topology');
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Generation failed');
      } finally {
        setLoading(false);
      }
    },
    [addToHistory]
  );

  const handleSampleGraph = useCallback(async (): Promise<GeneratedGraph | null> => {
    setError(null);
    try {
      const res = await generateGraph(selectedDataset);
      return res.graphs[0] ?? null;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Graph generation failed');
      return null;
    }
  }, [selectedDataset]);

  const handleBoundary = useCallback(async (boundaryImage: string) => {
    setLoading(true);
    setError(null);
    setImage(null);
    try {
      const res = await generateBoundary(boundaryImage);
      setImage(res.image);
      addToHistory(res.image, 'boundary');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  }, [addToHistory]);

  const handleSelectHistory = useCallback((item: HistoryItem) => {
    setImage(item.image);
    setError(null);
  }, []);

  const handleClearHistory = useCallback(() => {
    setHistory([]);
  }, []);

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

            {/* Mode-specific controls — slim sidebars without canvas/editor for now */}
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
                // Other modes get wired in subsequent tasks
              }}
              disabled={activeMode !== 'unconstrained'}
            />
          </div>
        </aside>

        {/* Right workspace */}
        <div className="flex flex-1 flex-col overflow-hidden">
          <MainViewer>
            <ResultPanel image={image} loading={loading} error={error} dataset={selectedDataset} />
          </MainViewer>

          {/* History bar (placeholder using existing HistoryPanel until Task 7) */}
          {history.length > 0 && (
            <HistoryPanel
              history={history}
              onSelect={handleSelectHistory}
              onClear={handleClearHistory}
              currentImage={image}
            />
          )}
        </div>
      </div>
    </div>
  );
}
