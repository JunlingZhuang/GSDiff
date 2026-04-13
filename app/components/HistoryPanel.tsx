'use client';

import { Trash2, Clock } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { GENERATION_MODES } from '@/lib/constants';
import type { HistoryItem } from '@/lib/types';

interface Props {
  history: HistoryItem[];
  onSelect: (item: HistoryItem) => void;
  onClear: () => void;
  currentImage: string | null;
}

export function HistoryPanel({ history, onSelect, onClear, currentImage }: Props) {
  return (
    <div className="border-t border-border/60 bg-card">
      <div className="flex items-center justify-between px-4 py-2">
        <div className="flex items-center gap-1.5">
          <Clock className="h-3 w-3 text-muted-foreground" />
          <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            History ({history.length})
          </span>
        </div>
        <Button variant="ghost" size="sm" onClick={onClear} className="text-xs h-6 px-2">
          <Trash2 className="mr-1 h-3 w-3" />
          Clear
        </Button>
      </div>
      <div className="flex gap-2 overflow-x-auto px-4 pb-3">
        {history.map((item) => {
          const mode = GENERATION_MODES.find(m => m.id === item.mode);
          const isActive = currentImage === item.image;
          return (
            <button
              key={item.id}
              onClick={() => onSelect(item)}
              className={`group relative shrink-0 overflow-hidden rounded-lg border-2 transition-all duration-150 hover:shadow-md ${
                isActive
                  ? 'border-foreground shadow-md'
                  : 'border-border/60 hover:border-foreground/30'
              }`}
            >
              <img
                src={item.image}
                alt="Generated floorplan"
                className="h-16 w-16 object-cover"
              />
              <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-1 py-0.5">
                <span className="text-[9px] font-medium text-white">
                  {mode?.name ?? item.mode}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
