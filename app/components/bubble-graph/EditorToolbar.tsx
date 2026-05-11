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

function SwatchDropdown({
  items,
  value,
  onChange,
  label,
  skipIds,
}: {
  items: readonly { id: number; name: string; color: string }[];
  value: number;
  onChange: (id: number) => void;
  label: string;
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
        <span
          className="inline-block h-3 w-3 rounded-sm border border-black/10"
          style={{ background: current?.color ?? '#e5e7eb' }}
        />
        <span className="max-w-[80px] truncate">{label}</span>
        <ChevronDown className="h-3 w-3 opacity-60" />
      </button>
      {open && (
        <div className="absolute left-0 top-full z-50 mt-1 flex flex-wrap gap-1.5 rounded-lg border border-border bg-card p-2 shadow-md"
          style={{ minWidth: 140, maxWidth: 220 }}>
          {filtered.map((item) => (
            <button
              key={item.id}
              type="button"
              title={item.name}
              onClick={() => { onChange(item.id); setOpen(false); }}
              className={cn(
                'h-6 w-6 rounded border-2 transition-transform hover:scale-110',
                item.id === value ? 'border-foreground' : 'border-transparent',
              )}
              style={{ background: item.color }}
            />
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
        label="type"
      />

      <div className="mx-1 h-5 w-px bg-border" />

      {/* Default edge type */}
      <SwatchDropdown
        items={edgeTypes}
        value={defaultEdgeType}
        onChange={onDefaultEdgeTypeChange}
        label="edge"
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
