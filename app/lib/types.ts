import type { GenerationMode } from './constants';

export interface HistoryItem {
  id: number;
  image: string;
  mode: GenerationMode;
  timestamp: Date;
}
