'use client';

import { cn } from '@/lib/utils';
import type { HistoryItem } from '@/lib/history';
import { DATASET_SPECS } from '@/lib/constants';

interface Props {
  item: HistoryItem;
  selected: boolean;
  onClick: () => void;
}

export function HistoryThumb({ item, selected, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'relative flex h-20 w-20 shrink-0 items-center justify-center rounded-lg border-2 bg-muted transition-colors',
        selected ? 'border-foreground' : 'border-border/60 hover:border-foreground/40',
      )}
      title={`${item.kind} · ${new Date(item.createdAt).toLocaleTimeString()}`}
    >
      {item.kind === 'floorplan' ? (
        <img src={item.image} alt="" className="h-full w-full rounded object-cover" />
      ) : (
        <MiniGraphPreview item={item} />
      )}
      <span className="absolute bottom-0.5 left-1 text-[9px] font-bold text-foreground/70">
        {item.kind === 'graph' ? '⬡' : '⬜'}
      </span>
    </button>
  );
}

function MiniGraphPreview({ item }: { item: Extract<HistoryItem, { kind: 'graph' }> }) {
  // 80x80 SVG with nodes laid out in a small circle
  const spec = DATASET_SPECS[item.dataset];
  const cx = 40;
  const cy = 40;
  const r = 26;
  const n = item.graph.nodes.length;
  return (
    <svg viewBox="0 0 80 80" className="h-full w-full">
      {item.graph.nodes.map((node, i) => {
        const angle = (2 * Math.PI * i) / Math.max(1, n);
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        const meta = spec.roomTypes.find((m) => m.id === node.attr);
        return <circle key={node.id} cx={x} cy={y} r="4" fill={meta?.color ?? '#9ca3af'} />;
      })}
    </svg>
  );
}
