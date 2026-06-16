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
  completeNextNode,
  completeGraph,
  generateTopology,
  generateBoundary,
  generateProcedural,
} from '@/lib/api';
import {
  DATASETS,
  GRAPH_MODELS,
  type DatasetId,
  type GenerationMode,
  type GraphModelId,
  type GraphModelTask,
} from '@/lib/constants';
import { GraphModelSelector } from '@/components/GraphModelSelector';
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
import type { Plan } from '@/lib/plan';

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

function graphTaskForMode(mode: GenerationMode): GraphModelTask | null {
  if (mode === 'graph') return 'sample';
  if (mode === 'next_node') return 'next_node';
  if (mode === 'graph_completion') return 'completion';
  return null;
}

// The procedural backend expects the canonical room names from
// procedural/rules.py (ROOM_NAMES). Bubble-graph nodes carry looser labels:
// lowercase demo types ("living"), or MSD display names ("Living Room",
// "Storage"). Normalise both to the backend's RoomType strings.
const ROOM_TYPE_TO_BACKEND: Record<string, string> = {
  living: 'Livingroom', livingroom: 'Livingroom', 'living room': 'Livingroom',
  bedroom: 'Bedroom',
  kitchen: 'Kitchen',
  dining: 'Dining', diningroom: 'Dining', 'dining room': 'Dining',
  corridor: 'Corridor', hallway: 'Corridor',
  stairs: 'Stairs', stair: 'Stairs',
  storage: 'Storeroom', storeroom: 'Storeroom', store: 'Storeroom',
  bathroom: 'Bathroom', bath: 'Bathroom',
  balcony: 'Balcony',
};

function toBackendRoomType(roomType: string): string {
  return ROOM_TYPE_TO_BACKEND[roomType.trim().toLowerCase()] ?? roomType;
}

export default function Home() {
  const [activeMode, setActiveMode] = useState<GenerationMode>('unconstrained');
  const [selectedDataset, setSelectedDataset] = useState<DatasetId>('rplan');
  const [selectedGraphModel, setSelectedGraphModel] = useState<GraphModelId>('rplan');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [boundaryDraft, setBoundaryDraft] = useSessionState<string>('boundaryDraft', '');
  const [toast, setToast] = useState<string | null>(null);
  // Transient edits to the currently-displayed graph (Graph mode).
  // Resets to null whenever the selected history item changes.
  const [graphDraft, setGraphDraft] = useState<GeneratedGraph | null>(null);
  useEffect(() => {
    setGraphDraft(null);
  }, [selectedHistoryId]);
  // Procedural floor plan produced in Design mode; planFitKey bumps on each
  // fresh generation so the editor refits the view (edits keep the view).
  const [planDraft, setPlanDraft] = useState<Plan | null>(null);
  const [planFitKey, setPlanFitKey] = useState(0);

  const selectedDatasetMeta = DATASETS.find((dataset) => dataset.id === selectedDataset) ?? DATASETS[0];
  const selectedItem = findHistoryItem(history, selectedHistoryId);

  useEffect(() => {
    const task = graphTaskForMode(activeMode);
    if (!task) return;

    const compatible = GRAPH_MODELS.find(
      (model) => model.dataset === selectedDataset && model.task === task && model.enabled,
    );
    if (compatible) {
      if (selectedGraphModel !== compatible.id) setSelectedGraphModel(compatible.id);
      return;
    }

    const fallback = GRAPH_MODELS.find((model) => model.task === task && model.enabled);
    if (fallback) {
      setSelectedDataset(fallback.dataset);
      setSelectedGraphModel(fallback.id);
    }
  }, [activeMode, selectedDataset, selectedGraphModel]);

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
      const res = await generateGraph(selectedDataset, selectedGraphModel);
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
  }, [selectedDataset, selectedGraphModel]);

  const handleNextNode = useCallback(async () => {
    const graphItem =
      selectedItem?.kind === 'graph' ? (selectedItem as GraphHistoryItem) : null;
    const sourceGraph = graphDraft ?? graphItem?.graph;
    if (!sourceGraph) {
      setError('Draw or sample a graph before running next-node completion');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const res = await completeNextNode(selectedDataset, selectedGraphModel, sourceGraph);
      const graph = res.graphs[0];
      if (!graph) throw new Error('Server returned no next-node candidate');
      const item = createGraphItem(selectedDataset, graph);
      setHistory((prev) => prependHistory(prev, item));
      setSelectedHistoryId(item.id);
      setGraphDraft(graph);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Next-node completion failed');
    } finally {
      setLoading(false);
    }
  }, [selectedDataset, selectedGraphModel, graphDraft, selectedItem]);

  const handleGraphCompletion = useCallback(async () => {
    const graphItem =
      selectedItem?.kind === 'graph' ? (selectedItem as GraphHistoryItem) : null;
    const sourceGraph = graphDraft ?? graphItem?.graph;
    if (!sourceGraph) {
      setError('Draw or sample a partial graph before running graph completion');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const res = await completeGraph(selectedDataset, selectedGraphModel, sourceGraph);
      const graph = res.graphs[0];
      if (!graph) throw new Error('Server returned no completed graph');
      const item = createGraphItem(selectedDataset, graph);
      setHistory((prev) => prependHistory(prev, item));
      setSelectedHistoryId(item.id);
      setGraphDraft(graph);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Graph completion failed');
    } finally {
      setLoading(false);
    }
  }, [selectedDataset, selectedGraphModel, graphDraft, selectedItem]);

  const handleSendCurrentGraphToFloorplan = useCallback(async () => {
    // Source: live-edited draft > selected sampled-graph item.
    // If neither exists yet (user clicked before editing or sampling), bail.
    const graphItem =
      selectedItem?.kind === 'graph' ? (selectedItem as GraphHistoryItem) : null;
    const sourceGraph = graphDraft ?? graphItem?.graph;
    if (!sourceGraph) return;
    setLoading(true);
    setError(null);
    try {
      const rooms = sourceGraph.rooms;
      const adjacency =
        sourceGraph.adjacency.length > 0
          ? sourceGraph.adjacency
          : computeAdjacency(sourceGraph);
      const res = await generateTopology(rooms, adjacency);
      const item = createFloorplanItem(selectedDataset, res.image, 'graph', graphItem?.id);
      setHistory((prev) => prependHistory(prev, item));
      setSelectedHistoryId(item.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  }, [selectedDataset, graphDraft, selectedItem]);

  const handleGenerateProcedural = useCallback(
    async (boundary: [number, number][], axisAngle: number | null) => {
      // Source graph: live-edited draft > selected sampled-graph item.
      const graphItem =
        selectedItem?.kind === 'graph' ? (selectedItem as GraphHistoryItem) : null;
      const sourceGraph = graphDraft ?? graphItem?.graph;
      if (!sourceGraph) {
        setError('Sample or edit a bubble graph first');
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const nodes = sourceGraph.nodes.map((n) => ({
          id: n.id,
          room_type: toBackendRoomType(n.room_type),
        }));
        const edges = sourceGraph.edges.map((e) => ({
          source: e.source,
          target: e.target,
          connectivity: e.edge_label ?? 'wall',
        }));
        const plan = await generateProcedural({ nodes, edges }, boundary, 0, axisAngle);
        setPlanDraft(plan);
        setPlanFitKey((k) => k + 1);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Procedural generation failed');
      } finally {
        setLoading(false);
      }
    },
    [graphDraft, selectedItem],
  );

  const handleSelectHistory = useCallback((id: string) => {
    const item = findHistoryItem(history, id);
    if (!item) return;
    // Auto-switch to a viewer mode that can render this item:
    //   • graph item → Graph mode (editable canvas)
    //   • floorplan item while in Graph or Boundary mode → Unconstrained (the only floorplan viewer)
    let nextMode: GenerationMode = activeMode;
    if (
      item.kind === 'graph' &&
      activeMode !== 'graph' &&
      activeMode !== 'next_node' &&
      activeMode !== 'graph_completion'
    ) {
      nextMode = 'graph';
    } else if (item.kind === 'floorplan' && activeMode !== 'unconstrained') {
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
            {graphTaskForMode(activeMode) && (
              <GraphModelSelector
                dataset={selectedDataset}
                task={graphTaskForMode(activeMode)!}
                value={selectedGraphModel}
                onChange={setSelectedGraphModel}
              />
            )}

            {/* Mode-specific controls */}
            {activeMode === 'graph' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Edit the bubble graph in the workspace, or click <span className="font-medium text-foreground">Sample Graph</span> to populate it from {selectedDatasetMeta.name}. Use the canvas toolbar to add rooms / connect / reset.
              </div>
            )}
            {activeMode === 'next_node' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Edit or select a partial MSD graph, then click <span className="font-medium text-foreground">Suggest Next Node</span>. The model adds one room and predicts its target-to-known edge types.
              </div>
            )}
            {activeMode === 'graph_completion' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Edit or select a partial MSD graph, then click <span className="font-medium text-foreground">Complete Graph</span>. The app uses full-completion v2 with top-k reranking and connectivity repair.
              </div>
            )}
            {activeMode === 'boundary' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Draw a boundary in the workspace, then click Generate to synthesise a floorplan.
              </div>
            )}
            {activeMode === 'retrieve' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Retrieval searches the MSD corpus only. Draw a partial bubble graph in the workspace; use the algorithm selector and Retrieve button there to find top-k matches.
              </div>
            )}
            {activeMode === 'design' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Draw or select a bubble graph (left), pick a boundary and axis, then <span className="font-medium text-foreground">Generate Plan</span> in the workspace toolbar. The plan is editable: drag walls, change room types, add or move doors.
              </div>
            )}
            {activeMode === 'agent' && (
              <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
                Describe the building in natural language in the workspace; the agent plans the program, generates imagery, parses it into a wall graph and repairs counts deterministically. The result is fully editable in 2D with a synced 3D view. Requires the hfagent backend on port 8100.
              </div>
            )}
          </div>

          {/* Design/agent modes own their Generate button inside the workspace. */}
          {activeMode !== 'retrieve' && activeMode !== 'design' && activeMode !== 'agent' && (
            <div className="border-t border-border/60 p-4">
              <GenerateButton
                mode={activeMode}
                loading={loading}
                onClick={() => {
                  if (activeMode === 'unconstrained') handleUnconstrained();
                  else if (activeMode === 'graph') handleSampleGraph();
                  else if (activeMode === 'next_node') handleNextNode();
                  else if (activeMode === 'graph_completion') handleGraphCompletion();
                  else if (activeMode === 'boundary') handleGenerateFloorplanFromBoundary();
                }}
                disabled={
                  activeMode !== 'unconstrained' &&
                  activeMode !== 'graph' &&
                  activeMode !== 'next_node' &&
                  activeMode !== 'graph_completion' &&
                  !(activeMode === 'boundary' && boundaryDraft !== '')
                }
              />
            </div>
          )}
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
            boundaryDraft={boundaryDraft}
            onBoundaryChange={setBoundaryDraft}
            onGenerateBoundary={handleGenerateFloorplanFromBoundary}
            onGraphDraftChange={setGraphDraft}
            planDraft={planDraft}
            planFitKey={planFitKey}
            onGenerateProcedural={handleGenerateProcedural}
            onPlanChange={setPlanDraft}
          />

          <HistoryBar
            items={history}
            selectedId={selectedHistoryId}
            onSelect={handleSelectHistory}
            onClear={() => { setHistory([]); setSelectedHistoryId(null); }}
            rightSlot={activeMode === 'graph' || activeMode === 'next_node' || activeMode === 'graph_completion' ? (
              <Button onClick={handleSendCurrentGraphToFloorplan} disabled={loading} size="sm">
                Send to Floorplan
              </Button>
            ) : null}
          />
        </div>
      </div>
    </div>
  );
}
