import type { DatasetId, GenerationMode } from './constants';
import type { GeneratedGraph } from './types';

export interface GraphHistoryItem {
  id: string;
  kind: 'graph';
  dataset: DatasetId;
  graph: GeneratedGraph;
  createdAt: number;
}

export interface FloorplanHistoryItem {
  id: string;
  kind: 'floorplan';
  dataset: DatasetId;
  image: string;            // data URI
  sourceGraphId?: string;
  sourceMode: GenerationMode;
  createdAt: number;
}

export type HistoryItem = GraphHistoryItem | FloorplanHistoryItem;

const HISTORY_CAP = 30;

let nextId = 0;
function makeId(prefix: string): string {
  nextId += 1;
  return `${prefix}-${Date.now()}-${nextId}`;
}

export function createGraphItem(dataset: DatasetId, graph: GeneratedGraph): GraphHistoryItem {
  return {
    id: makeId('g'),
    kind: 'graph',
    dataset,
    graph,
    createdAt: Date.now(),
  };
}

export function createFloorplanItem(
  dataset: DatasetId,
  image: string,
  sourceMode: GenerationMode,
  sourceGraphId?: string,
): FloorplanHistoryItem {
  return {
    id: makeId('f'),
    kind: 'floorplan',
    dataset,
    image,
    sourceGraphId,
    sourceMode,
    createdAt: Date.now(),
  };
}

/** Prepend a new item; cap to the most recent HISTORY_CAP items. */
export function appendHistory(prev: HistoryItem[], item: HistoryItem): HistoryItem[] {
  return [item, ...prev].slice(0, HISTORY_CAP);
}

export function findHistoryItem(items: HistoryItem[], id: string | null): HistoryItem | undefined {
  if (!id) return undefined;
  return items.find((i) => i.id === id);
}
