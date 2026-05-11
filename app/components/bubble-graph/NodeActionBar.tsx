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
  // Bar width: swatches (20px each + 4px gap) + X button + padding
  const barWidth = roomTypes.length * 24 + 32 + 16;
  // Position above the node, centered horizontally
  const left = Math.max(4, px - barWidth / 2);
  const top = Math.max(4, py - 56); // 56px above the node centre

  return (
    <div
      className="absolute z-30 flex items-center gap-1 rounded-lg border border-border bg-card px-2 py-1 shadow-md"
      style={{ left, top, pointerEvents: 'auto' }}
      onPointerDown={(e) => e.stopPropagation()}
    >
      {roomTypes.map((rt) => (
        <button
          key={rt.id}
          type="button"
          title={rt.name}
          onClick={() => onChangeAttr(rt.id)}
          className={cn(
            'h-5 w-5 rounded border-2 transition-transform hover:scale-110',
            rt.id === currentAttr ? 'border-foreground' : 'border-transparent',
          )}
          style={{ background: rt.color }}
        />
      ))}
      <div className="mx-0.5 h-4 w-px bg-border" />
      <button
        type="button"
        title="Delete node"
        onClick={onDelete}
        className="flex h-5 w-5 items-center justify-center rounded hover:bg-destructive/20 text-destructive"
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  );
}
