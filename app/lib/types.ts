import type { DatasetId, GenerationMode } from './constants';
import type { HistoryItem as NewHistoryItem } from './history';

export interface HistoryItem {
  id: number;
  image: string;
  mode: GenerationMode;
  timestamp: Date;
}

export interface GeneratedGraphNode {
  id: number;
  attr: number;
  room_type: string;
}

export interface GeneratedGraphEdge {
  source: number;
  target: number;
  edge_type: number;
  edge_label: string;
}

export interface GeneratedGraph {
  num_nodes: number;
  num_edges: number;
  rooms: number[];
  adjacency: number[][];
  edge_types: number[][];
  nodes: GeneratedGraphNode[];
  edges: GeneratedGraphEdge[];
}

export interface GraphGenerationResponse {
  dataset: string;
  checkpoint: string;
  inference_seconds: number;
  graphs: GeneratedGraph[];
  room_types: string[];
  edge_types: string[];
}

export type ModelLoadState = 'not_loaded' | 'loading' | 'loaded' | 'error';

export interface ModelStatusItem {
  key: string;
  group: string;
  label: string;
  state: ModelLoadState;
  device: string | null;
  load_seconds: number | null;
  error: string | null;
  metadata: Record<string, string>;
}

export interface ModelStatusResponse {
  device: string;
  gpu: {
    name: string;
    allocated_mb: number;
    reserved_mb: number;
  } | null;
  models: ModelStatusItem[];
}

export interface TopologyDraft {
  graph: GeneratedGraph;
}

export interface BoundaryDraft {
  canvasDataUrl: string;
}

export interface PageState {
  selectedDataset: DatasetId;
  activeMode: GenerationMode;
  history: NewHistoryItem[];
  selectedHistoryId: string | null;
  modeDrafts: {
    topology?: TopologyDraft;
    boundary?: BoundaryDraft;
  };
  loading: boolean;
  error: string | null;
}
