'use client';

import { useState, useEffect, useCallback } from 'react';
import { Header } from '@/components/Header';
import { ModeSelector } from '@/components/ModeSelector';
import { DatasetSelector } from '@/components/DatasetSelector';
import { MainViewer } from '@/components/MainViewer';
import { GenerateButton } from '@/components/GenerateButton';
import {
  generateUnconstrained,
  generateGraph,
  generateTopology,
  generateBoundary,
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
import type { GeneratedGraph } from '@/lib/types';

// ---------------------------------------------------------------------------
// useSessionState — persists state in sessionStorage so drafts survive
// page reloads within the same browser tab.  Guarded against SSR (Next.js
// server render) and private-browsing / quota failures.
// ---------------------------------------------------------------------------
function useSessionState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === 'undefined') return initial;
    try {
      const raw = window.sessionStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      window.sessionStorage.setItem(key, JSON.stringify(value));
    } catch {
      // quota exceeded or private browsing — silently ignore
    }
  }, [key, value]);
  return [value, setValue];
}

// ---------------------------------------------------------------------------
// Helper: build an adjacency matrix from a GeneratedGraph's edges.
// The editor does not compute adjacency eagerly, so we derive it here before
// passing to the topology generation API.
// ---------------------------------------------------------------------------
function computeAdjacency(g: GeneratedGraph): number[][] {
  const n = g.nodes.length;
  const adj: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (const e of g.edges) {
    adj[e.source][e.target] = 1;
    adj[e.target][e.source] = 1;
  }
  return adj;
}

export default function Home() {
  const [activeMode, setActiveMode] = useState<GenerationMode>('unconstrained');
  const [selectedDataset, setSelectedDataset] = useState<DatasetId>('rplan');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [topologyDraft, setTopologyDraft] = useSessionState<GeneratedGraph | undefined>('topologyDraft', undefined);
  const [boundaryDraft, setBoundaryDraft] = useSessionState<string>('boundaryDraft', '');
  const [toast, setToast] = useState<string | null>(null);
  // Transient edits to the currently-displayed graph (Graph mode).
  // Resets to null whenever the selected history item changes.
  const [graphDraft, setGraphDraft] = useState<GeneratedGraph | null>(null);
  useEffect(() => {
    setGraphDraft(null);
  }, [selectedHistoryId]);

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
      // Use the live-edited draft if available, otherwise fall back to the original graph.
      const sourceGraph = graphDraft ?? graphItem.graph;
      const rooms = sourceGraph.rooms;
      const adjacency =
        sourceGraph.adjacency.length > 0
          ? sourceGraph.adjacency
          : computeAdjacency(sourceGraph);
      const res = await generateTopology(rooms, adjacency);
      const item = createFloorplanItem(selectedDataset, res.image, 'graph', graphItem.id);
      setHistory((prev) => prependHistory(prev, item));
      setSelectedHistoryId(item.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  }, [selectedDataset, graphDraft]);

  const handleGenerateFloorplanFromTopology = useCallback(async () => {
    if (!topologyDraft) return;
    setLoading(true);
    setError(null);
    try {
      const adjacency =
        topologyDraft.adjacency.length > 0
          ? topologyDraft.adjacency
          : computeAdjacency(topologyDraft);
      const res = await generateTopology(topologyDraft.rooms, adjacency);
      const item = createFloorplanItem(selectedDataset, res.image, 'topology');
      setHistory((prev) => prependHistory(prev, item));
      setSelectedHistoryId(item.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  }, [topologyDraft, selectedDataset]);

  const handleSelectHistory = useCallback((id: string) => {
    const item = findHistoryItem(history, id);
    if (!item) return;
    // Auto-switch to a viewer mode that can render this item.
    // Rules per spec §6.4 and §6.2:
    //   • graph item → must be in Graph mode to see it
    //   • floorplan item while in Graph mode → switch to Unconstrained (no editor there)
    //   • floorplan item in Topology/Boundary mode → switch to Unconstrained
    //     (the editor occupies the main viewer in those modes, so the floorplan can't display)
    let nextMode: GenerationMode = activeMode;
    if (item.kind === 'graph' && activeMode !== 'graph') {
      nextMode = 'graph';
    } else if (item.kind === 'floorplan' && activeMode === 'graph') {
      nextMode = 'unconstrained';
    } else if (item.kind === 'floorplan' && (activeMode === 'topology' || activeMode === 'boundary')) {
      nextMode = 'unconstrained';
    }
    if (nextMode !== activeMode) {
      setActiveMode(nextMode);
      setToast(`Switched to ${nextMode} mode to view this item`);
      setTimeout(() => setToast(null), 3000);
    }
    setSelectedHistoryId(id);
  }, [activeMode, history]);

  const handleGenerateFloorplanFromBoundary = useCallback(async () => {
    if (!boundaryDraft) return;
    setLoading(true);
    setError(null);
    try {
      // boundaryDraft is a full data URL; strip the prefix to get raw base64
      const base64 = boundaryDraft.split(',')[1] || '';
      const res = await generateBoundary(base64);
      const item = createFloorplanItem(selectedDataset, res.image, 'boundary');
      setHistory((prev) => prependHistory(prev, item));
      setSelectedHistoryId(item.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  }, [boundaryDraft, selectedDataset]);

  return (
    <div className="flex h-full flex-col">
      <Header />

      <div className="flex flex-1 overflow-hidden">
        {/* Left controls panel (320px) */}
        <aside className="flex w-[320px] shrink-0 flex-col border-r border-border/60 bg-card">
          <div className="flex-1 space-y-4 overflow-auto p-4">
            <DatasetSelector value={selectedDataset} onChange={setSelectedDataset} />
            <ModeSelector activeMode={activeMode} onModeChange={setActiveMode} />

            {/* Mode-specific controls */}
            {activeMode === 'graph' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Sample bubble graphs from {selectedDatasetMeta.name}.
              </div>
            )}
            {activeMode === 'topology' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Edit the bubble graph in the workspace. Click empty space to add a room, drag from a node&apos;s edge to connect, right-click an edge to delete. Ctrl+Z/Y to undo/redo.
              </div>
            )}
            {activeMode === 'boundary' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Draw a boundary in the workspace, then click Generate to synthesise a floorplan.
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
                else if (activeMode === 'topology') handleGenerateFloorplanFromTopology();
                else if (activeMode === 'boundary') handleGenerateFloorplanFromBoundary();
              }}
              disabled={
                activeMode !== 'unconstrained' &&
                activeMode !== 'graph' &&
                !(activeMode === 'topology' && topologyDraft !== undefined) &&
                !(activeMode === 'boundary' && boundaryDraft !== '')
              }
            />
          </div>
        </aside>

        {/* Right workspace */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Cross-mode toast — auto-dismisses after 3 s */}
          {toast && (
            <div className="border-b border-border/60 bg-foreground/90 px-4 py-2 text-xs text-background">
              {toast}
            </div>
          )}
          <MainViewer
            mode={activeMode}
            selectedItem={selectedItem}
            loading={loading}
            error={error}
            dataset={selectedDataset}
            topologyDraft={topologyDraft}
            onTopologyChange={setTopologyDraft}
            boundaryDraft={boundaryDraft}
            onBoundaryChange={setBoundaryDraft}
            onGenerateBoundary={handleGenerateFloorplanFromBoundary}
            onGraphDraftChange={setGraphDraft}
          />

          <HistoryBar
            items={history}
            selectedId={selectedHistoryId}
            onSelect={handleSelectHistory}
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
