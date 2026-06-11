'use client';

import { Cpu } from 'lucide-react';
import {
  GRAPH_MODELS,
  type DatasetId,
  type GraphModelId,
  type GraphModelTask,
} from '@/lib/constants';
import { cn } from '@/lib/utils';

interface Props {
  dataset: DatasetId;
  task: GraphModelTask;
  value: GraphModelId;
  onChange: (model: GraphModelId) => void;
}

export function GraphModelSelector({ dataset, task, value, onChange }: Props) {
  const models = GRAPH_MODELS.filter(
    (model) => model.dataset === dataset && model.task === task,
  );

  return (
    <div className="space-y-1.5">
      <p className="px-1 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        Graph Model
      </p>
      {models.length === 0 ? (
        <div className="rounded-xl border border-border/70 bg-background p-3 text-xs text-muted-foreground">
          No {task.replace('_', ' ')} checkpoint is configured for this dataset yet.
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {models.map((model) => {
            const isActive = value === model.id;
            return (
              <button
                key={model.id}
                type="button"
                disabled={!model.enabled}
                onClick={() => onChange(model.id)}
                className={cn(
                  'rounded-xl border px-3 py-2 text-left transition-all duration-200',
                  isActive
                    ? 'border-foreground bg-foreground text-background shadow-sm'
                    : 'border-border/70 bg-background hover:border-foreground/30',
                  !model.enabled && 'cursor-not-allowed opacity-45 hover:border-border/70',
                )}
              >
                <div className="flex items-center gap-2">
                  <Cpu
                    className={cn(
                      'h-3.5 w-3.5',
                      isActive ? 'text-background/80' : 'text-muted-foreground',
                    )}
                  />
                  <span className="text-sm font-medium">{model.name}</span>
                </div>
                <p
                  className={cn(
                    'mt-1 text-xs leading-tight',
                    isActive ? 'text-background/60' : 'text-muted-foreground',
                  )}
                >
                  {model.description}
                </p>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
