'use client';

import { ReactNode } from 'react';
import { Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { HistoryThumb } from './HistoryThumb';
import type { HistoryItem } from '@/lib/history';

interface Props {
  items: HistoryItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onClear?: () => void;
  rightSlot?: ReactNode;
}

export function HistoryBar({ items, selectedId, onSelect, onClear, rightSlot }: Props) {
  return (
    <div className="flex items-center gap-3 border-t border-border/60 bg-card px-4 py-3">
      <div className="flex flex-1 gap-2 overflow-x-auto">
        {items.length === 0 && (
          <p className="text-xs text-muted-foreground">No items yet — generate something to populate the history.</p>
        )}
        {items.map((item) => (
          <HistoryThumb
            key={item.id}
            item={item}
            selected={selectedId === item.id}
            onClick={() => onSelect(item.id)}
          />
        ))}
      </div>
      <div className="flex items-center gap-2">
        {rightSlot}
        {items.length > 0 && onClear && (
          <Button variant="ghost" size="icon" onClick={onClear} title="Clear all">
            <Trash2 className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
}
