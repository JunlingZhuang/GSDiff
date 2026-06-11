'use client';

import { Sparkles, Share2, PenTool, Network, Search, PlusCircle, PencilRuler } from 'lucide-react';
import { cn } from '@/lib/utils';
import { GENERATION_MODES, type GenerationMode } from '@/lib/constants';

const iconMap = {
  Sparkles,
  Network,
  Share2,
  PenTool,
  Search,
  PlusCircle,
  PencilRuler,
} as const;

interface Props {
  activeMode: GenerationMode;
  onModeChange: (mode: GenerationMode) => void;
}

export function ModeSelector({ activeMode, onModeChange }: Props) {
  return (
    <div className="space-y-1.5">
      <p className="px-1 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        Generation Mode
      </p>
      <div className="space-y-1">
        {GENERATION_MODES.map((mode) => {
          const Icon = iconMap[mode.icon];
          const isActive = activeMode === mode.id;
          return (
            <button
              key={mode.id}
              onClick={() => onModeChange(mode.id)}
              className={cn(
                'group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-all duration-200',
                isActive
                  ? 'bg-foreground text-background shadow-sm'
                  : 'hover:bg-muted'
              )}
            >
              <div
                className={cn(
                  'flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-colors duration-200',
                  isActive
                    ? 'bg-background/15'
                    : 'bg-muted group-hover:bg-background'
                )}
              >
                <Icon
                  className={cn(
                    'h-4 w-4 transition-colors duration-200',
                    isActive ? 'text-background' : 'text-muted-foreground'
                  )}
                />
              </div>
              <div className="min-w-0">
                <p
                  className={cn(
                    'text-sm font-medium leading-none',
                    isActive ? 'text-background' : 'text-foreground'
                  )}
                >
                  {mode.name}
                </p>
                <p
                  className={cn(
                    'mt-1 text-xs leading-none',
                    isActive ? 'text-background/60' : 'text-muted-foreground'
                  )}
                >
                  {mode.description}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
