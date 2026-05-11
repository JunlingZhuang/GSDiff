'use client';

import { X } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { EdgeTypeMeta } from '@/lib/constants';

interface Props {
  edgeId: string;
  currentType: number;
  edgeTypes: readonly EdgeTypeMeta[];
  /** Position in container pixels — the midpoint of the edge */
  px: number;
  py: number;
  onChangeType: (et: number) => void;
  onDelete: () => void;
}

export function EdgeActionBar({ currentType, edgeTypes, px, py, onChangeType, onDelete }: Props) {
  const nonNone = edgeTypes.filter((et) => et.id !== 0);
  const barWidth = nonNone.length * 24 + 32 + 16;
  const left = Math.max(4, px - barWidth / 2);
  const top = Math.max(4, py - 40);

  return (
    <div
      className="absolute z-30 flex items-center gap-1 rounded-lg border border-border bg-card px-2 py-1 shadow-md"
      style={{ left, top, pointerEvents: 'auto' }}
      onPointerDown={(e) => e.stopPropagation()}
    >
      {nonNone.map((et) => (
        <button
          key={et.id}
          type="button"
          title={et.name}
          onClick={() => onChangeType(et.id)}
          className={cn(
            'h-5 w-5 rounded border-2 transition-transform hover:scale-110',
            et.id === currentType ? 'border-foreground' : 'border-transparent',
          )}
          style={{ background: et.color }}
        />
      ))}
      <div className="mx-0.5 h-4 w-px bg-border" />
      <button
        type="button"
        title="Delete edge"
        onClick={onDelete}
        className="flex h-5 w-5 items-center justify-center rounded hover:bg-destructive/20 text-destructive"
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  );
}
