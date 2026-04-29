'use client';

import { Database } from 'lucide-react';
import { DATASETS, type DatasetId } from '@/lib/constants';
import { cn } from '@/lib/utils';

interface Props {
  value: DatasetId;
  onChange: (dataset: DatasetId) => void;
}

export function DatasetSelector({ value, onChange }: Props) {
  return (
    <div className="space-y-1.5">
      <p className="px-1 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        Dataset
      </p>
      <div className="grid grid-cols-2 gap-2">
        {DATASETS.map((dataset) => {
          const isActive = value === dataset.id;
          return (
            <button
              key={dataset.id}
              type="button"
              disabled={!dataset.enabled}
              onClick={() => onChange(dataset.id)}
              className={cn(
                'rounded-xl border px-3 py-2 text-left transition-all duration-200',
                isActive
                  ? 'border-foreground bg-foreground text-background shadow-sm'
                  : 'border-border/70 bg-background hover:border-foreground/30',
                !dataset.enabled && 'cursor-not-allowed opacity-45 hover:border-border/70'
              )}
            >
              <div className="flex items-center gap-2">
                <Database
                  className={cn(
                    'h-3.5 w-3.5',
                    isActive ? 'text-background/80' : 'text-muted-foreground'
                  )}
                />
                <span className="text-sm font-medium">{dataset.name}</span>
              </div>
              <p
                className={cn(
                  'mt-1 text-xs leading-tight',
                  isActive ? 'text-background/60' : 'text-muted-foreground'
                )}
              >
                {dataset.description}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
