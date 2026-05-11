'use client';

import { X } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { RoomTypeMeta } from '@/lib/constants';

interface Props {
  nodeId: number;
  currentAttr: number;
  roomTypes: readonly RoomTypeMeta[];
  /** Position in container pixels (top-left of container = 0,0) */
  px: number;
  py: number;
  onChangeAttr: (attr: number) => void;
  onDelete: () => void;
}

export function NodeActionBar({ currentAttr, roomTypes, px, py, onChangeAttr, onDelete }: Props) {
  // Labeled chips so users can see the room-type name (especially useful
  // when there are 9 types, as in MSD). Approximate chip width ~80px each.
  const barWidth = roomTypes.length * 82 + 28;
  const left = Math.max(4, px - barWidth / 2);
  const top = Math.max(4, py - 56);

  return (
    <div
      className="absolute z-30 flex items-center gap-1 rounded-lg border border-border bg-card px-2 py-1.5 shadow-md"
      style={{ left, top, pointerEvents: 'auto' }}
      onPointerDown={(e) => e.stopPropagation()}
    >
      {roomTypes.map((rt) => {
        const selected = rt.id === currentAttr;
        return (
          <button
            key={rt.id}
            type="button"
            title={rt.name}
            onClick={() => onChangeAttr(rt.id)}
            className={cn(
              'flex h-7 items-center gap-1.5 rounded-md border px-1.5 transition-colors',
              selected
                ? 'border-foreground bg-muted'
                : 'border-transparent hover:bg-muted/60',
            )}
          >
            <span
              className="inline-block h-3 w-3 shrink-0 rounded-sm border border-black/10"
              style={{ background: rt.color }}
            />
            <span
              className={cn(
                'text-[10px] font-medium',
                selected ? 'text-foreground' : 'text-muted-foreground',
              )}
            >
              {rt.name}
            </span>
          </button>
        );
      })}
      <div className="mx-0.5 h-5 w-px bg-border" />
      <button
        type="button"
        title="Delete node"
        onClick={onDelete}
        className="flex h-6 w-6 items-center justify-center rounded hover:bg-destructive/20 text-destructive"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
