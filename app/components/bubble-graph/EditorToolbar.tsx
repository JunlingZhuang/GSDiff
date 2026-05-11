'use client';

import { Undo2, Redo2, ChevronDown } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type { RoomTypeMeta, EdgeTypeMeta } from '@/lib/constants';

interface Props {
  roomTypes: readonly RoomTypeMeta[];
  edgeTypes: readonly EdgeTypeMeta[];
  defaultNodeAttr: number;
  defaultEdgeType: number;
  onDefaultNodeAttrChange: (attr: number) => void;
  onDefaultEdgeTypeChange: (et: number) => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onAddNode: () => void;
}

type SwatchItem = { id: number; name: string; color: string; dashArray?: string | undefined };

function SwatchPreview({ item, size }: { item: SwatchItem; size: 'sm' | 'md' }) {
  // Edge items carry a dashArray field — render a line preview.
  // Room items don't — render a colored square.
  const isEdge = 'dashArray' in item;
  if (isEdge) {
    const w = size === 'sm' ? 16 : 22;
    return (
      <svg width={w} height={6} className="shrink-0">
        <line
          x1="0"
          y1="3"
          x2={w}
          y2="3"
          stroke={item.color}
          strokeWidth="2"
          strokeDasharray={item.dashArray}
          strokeLinecap="round"
        />
      </svg>
    );
  }
  const px = size === 'sm' ? 'h-3 w-3' : 'h-4 w-4';
  return (
    <span
      className={cn('inline-block shrink-0 rounded-sm border border-black/10', px)}
      style={{ background: item.color }}
    />
  );
}

function SwatchDropdown({
  items,
  value,
  onChange,
  fallbackLabel,
  skipIds,
}: {
  items: readonly SwatchItem[];
  value: number;
  onChange: (id: number) => void;
  /** Used if no item matches `value`. */
  fallbackLabel: string;
  skipIds?: number[];
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const filtered = items.filter((i) => !skipIds?.includes(i.id));
  const current = items.find((i) => i.id === value);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-border bg-background px-2.5 text-xs font-medium hover:bg-muted focus-visible:outline-none"
      >
        {current ? <SwatchPreview item={current} size="sm" /> : null}
        <span className="max-w-[88px] truncate">{current?.name ?? fallbackLabel}</span>
        <ChevronDown className="h-3 w-3 opacity-60" />
      </button>
      {open && (
        <div
          className="absolute left-0 top-full z-50 mt-1 flex flex-col gap-0.5 rounded-lg border border-border bg-card p-1 shadow-md"
          style={{ minWidth: 140 }}
        >
          {filtered.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                onChange(item.id);
                setOpen(false);
              }}
              className={cn(
                'flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-colors',
                item.id === value ? 'bg-muted font-medium' : 'hover:bg-muted/60',
              )}
            >
              <SwatchPreview item={item} size="md" />
              <span className="flex-1 truncate">{item.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function EditorToolbar({
  roomTypes,
  edgeTypes,
  defaultNodeAttr,
  defaultEdgeType,
  onDefaultNodeAttrChange,
  onDefaultEdgeTypeChange,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onAddNode,
}: Props) {
  return (
    <div className="absolute left-3 top-3 z-20 flex items-center gap-1.5 rounded-xl border border-border bg-card px-2 py-1.5 shadow-sm">
      {/* Add Room button + room-type picker */}
      <button
        type="button"
        onClick={onAddNode}
        className="inline-flex h-8 items-center gap-1 rounded-lg border border-border bg-background px-2.5 text-xs font-semibold hover:bg-muted"
      >
        <span
          className="inline-block h-3 w-3 rounded-sm border border-black/10"
          style={{ background: roomTypes.find((r) => r.id === defaultNodeAttr)?.color ?? '#e5e7eb' }}
        />
        + Add Room
      </button>
      <SwatchDropdown
        items={roomTypes}
        value={defaultNodeAttr}
        onChange={onDefaultNodeAttrChange}
        fallbackLabel="room"
      />

      <div className="mx-1 h-5 w-px bg-border" />

      {/* Default edge type */}
      <SwatchDropdown
        items={edgeTypes}
        value={defaultEdgeType}
        onChange={onDefaultEdgeTypeChange}
        fallbackLabel="edge"
        skipIds={[0]}
      />

      <div className="mx-1 h-5 w-px bg-border" />

      {/* Undo / Redo */}
      <Button
        variant="ghost"
        size="icon-sm"
        onClick={onUndo}
        disabled={!canUndo}
        title="Undo (Ctrl+Z)"
      >
        <Undo2 />
      </Button>
      <Button
        variant="ghost"
        size="icon-sm"
        onClick={onRedo}
        disabled={!canRedo}
        title="Redo (Ctrl+Y)"
      >
        <Redo2 />
      </Button>
    </div>
  );
}
