'use client';

import { useState, useCallback } from 'react';
import { Header } from '@/components/Header';
import { ModeSelector } from '@/components/ModeSelector';
import { DatasetSelector } from '@/components/DatasetSelector';
import { ModelStatusPanel } from '@/components/ModelStatusPanel';
import { UnconstrainedPanel } from '@/components/UnconstrainedPanel';
import { GraphPanel } from '@/components/GraphPanel';
import { TopologyEditor } from '@/components/TopologyEditor';
import { BoundaryCanvas } from '@/components/BoundaryCanvas';
import { ResultPanel } from '@/components/ResultPanel';
import { HistoryPanel } from '@/components/HistoryPanel';
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

      <div className="flex flex-1 overflow-hidden flex-col lg:flex-row">
        {/* Left panel */}
        <div className="flex w-full flex-col border-b border-border/60 bg-card lg:w-[680px] lg:shrink-0 lg:border-b-0 lg:border-r">
          {/* Mode selector */}
          <div className="border-b border-border/60 p-4">
            <div className="space-y-4">
              <DatasetSelector
                value={selectedDataset}
                onChange={setSelectedDataset}
              />
              <ModeSelector
                activeMode={activeMode}
                onModeChange={setActiveMode}
              />
              <ModelStatusPanel />
            </div>
          </div>

          {/* Mode-specific controls */}
          <div className="flex flex-1 flex-col overflow-auto">
            <div
              className={`flex-1 flex-col ${activeMode === 'unconstrained' ? 'flex' : 'hidden'}`}
            >
              <UnconstrainedPanel
                onGenerate={handleUnconstrained}
                loading={loading}
              />
            </div>
            <div
              className={`flex-1 flex-col ${activeMode === 'graph' ? 'flex' : 'hidden'}`}
            >
              <GraphPanel
                datasetName={selectedDatasetMeta.name}
                onSampleGraph={handleSampleGraph}
                onGenerateFloorplan={handleTopology}
                loading={loading}
              />
            </div>
            <div
              className={`flex-1 flex-col ${activeMode === 'topology' ? 'flex' : 'hidden'}`}
            >
              <TopologyEditor
                onGenerate={handleTopology}
                onSampleGraph={handleSampleGraph}
                datasetName={selectedDatasetMeta.name}
                loading={loading}
              />
            </div>
            <div
              className={`flex-1 flex-col ${activeMode === 'boundary' ? 'flex' : 'hidden'}`}
            >
              <BoundaryCanvas
                onGenerate={handleBoundary}
                loading={loading}
              />
            </div>
          </div>
        </div>

        {/* Right panel */}
        <div className="flex flex-1 flex-col overflow-hidden bg-background">
          <div className="flex-1 overflow-auto">
            <ResultPanel image={image} loading={loading} error={error} />
          </div>

          {/* History bar at bottom */}
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
