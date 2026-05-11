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
  // Rough width estimate so we can center the bar above the midpoint.
  // Each chip is ~64px (swatch line + label); separator + X button add ~28.
  const barWidth = nonNone.length * 66 + 28;
  const left = Math.max(4, px - barWidth / 2);
  const top = Math.max(4, py - 48);

  return (
    <div
      className="absolute z-30 flex items-center gap-1 rounded-lg border border-border bg-card px-2 py-1.5 shadow-md"
      style={{ left, top, pointerEvents: 'auto' }}
      onPointerDown={(e) => e.stopPropagation()}
    >
      {nonNone.map((et) => {
        const selected = et.id === currentType;
        return (
          <button
            key={et.id}
            type="button"
            title={et.name}
            onClick={() => onChangeType(et.id)}
            className={cn(
              'flex h-7 items-center gap-1.5 rounded-md border px-1.5 transition-colors',
              selected
                ? 'border-foreground bg-muted'
                : 'border-transparent hover:bg-muted/60',
            )}
          >
            {/* Mini line showing the edge's actual stroke pattern */}
            <svg width="16" height="6" className="shrink-0">
              <line
                x1="0"
                y1="3"
                x2="16"
                y2="3"
                stroke={et.color}
                strokeWidth="2"
                strokeDasharray={et.dashArray}
                strokeLinecap="round"
              />
            </svg>
            <span
              className={cn(
                'text-[10px] font-medium',
                selected ? 'text-foreground' : 'text-muted-foreground',
              )}
            >
              {et.name}
            </span>
          </button>
        );
      })}
      <div className="mx-0.5 h-5 w-px bg-border" />
      <button
        type="button"
        title="Delete edge"
        onClick={onDelete}
        className="flex h-6 w-6 items-center justify-center rounded hover:bg-destructive/20 text-destructive"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
