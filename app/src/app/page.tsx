'use client';

import { useState, useCallback } from 'react';
import { Header } from '@/components/Header';
import { ModeSelector } from '@/components/ModeSelector';
import { UnconstrainedPanel } from '@/components/UnconstrainedPanel';
import { TopologyEditor } from '@/components/TopologyEditor';
import { BoundaryCanvas } from '@/components/BoundaryCanvas';
import { ResultPanel } from '@/components/ResultPanel';
import { HistoryPanel } from '@/components/HistoryPanel';
import {
  generateUnconstrained,
  generateTopology,
  generateBoundary,
} from '@/lib/api';
import type { GenerationMode } from '@/lib/constants';
import type { HistoryItem } from '@/lib/types';

export default function Home() {
  const [activeMode, setActiveMode] = useState<GenerationMode>('unconstrained');
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

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
        <div className="flex w-full flex-col border-b border-border/60 bg-card lg:w-[520px] lg:shrink-0 lg:border-b-0 lg:border-r">
          {/* Mode selector */}
          <div className="border-b border-border/60 p-4">
            <ModeSelector
              activeMode={activeMode}
              onModeChange={setActiveMode}
            />
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
              className={`flex-1 flex-col ${activeMode === 'topology' ? 'flex' : 'hidden'}`}
            >
              <TopologyEditor
                onGenerate={handleTopology}
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
