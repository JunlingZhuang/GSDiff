'use client';

import { useEffect, useState } from 'react';
import { Cpu, Loader2 } from 'lucide-react';
import { getModelStatus } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { ModelLoadState, ModelStatusResponse } from '@/lib/types';

const stateLabel: Record<ModelLoadState, string> = {
  not_loaded: 'not loaded',
  loading: 'loading',
  loaded: 'loaded',
  error: 'error',
};

const stateClass: Record<ModelLoadState, string> = {
  not_loaded: 'bg-muted text-muted-foreground',
  loading: 'bg-amber-100 text-amber-800',
  loaded: 'bg-emerald-100 text-emerald-800',
  error: 'bg-destructive/10 text-destructive',
};

/**
 * Compact model-status indicator that sits in the header.
 * Replaces the left-panel ModelStatusPanel; full detail appears on hover/focus.
 */
export function ModelStatusBadge() {
  const [status, setStatus] = useState<ModelStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const next = await getModelStatus();
        if (!cancelled) {
          setStatus(next);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load model status');
      }
    }
    load();
    const intervalId = window.setInterval(load, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  const models = status?.models ?? [];
  const loaded = models.filter((m) => m.state === 'loaded').length;
  const loading = models.filter((m) => m.state === 'loading').length;
  const errored = models.filter((m) => m.state === 'error').length;

  const dotClass = errored > 0
    ? 'bg-destructive'
    : loading > 0
      ? 'bg-amber-500'
      : loaded > 0
        ? 'bg-emerald-500'
        : 'bg-muted-foreground/40';

  return (
    <div
      className="relative"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <button
        type="button"
        className="flex h-8 items-center gap-1.5 rounded-lg px-2 transition-colors hover:bg-muted focus:bg-muted focus:outline-none"
        aria-label="Model status"
      >
        <Cpu className="h-3.5 w-3.5 text-muted-foreground" />
        <span className={cn('h-2 w-2 rounded-full', dotClass)} />
        <span className="text-[11px] tabular-nums text-muted-foreground">
          {loaded}/{models.length}
        </span>
        {loading > 0 && <Loader2 className="h-3 w-3 animate-spin text-amber-600" />}
      </button>

      {open && (
        <div className="absolute right-0 top-full z-50 mt-1 w-80 rounded-xl border border-border/70 bg-popover p-3 text-popover-foreground shadow-lg">
          <div className="mb-2 flex items-center justify-between border-b border-border/40 pb-2">
            <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Models
            </p>
            <span className="text-[11px] text-muted-foreground">
              {status?.gpu ? `GPU · ${status.gpu.reserved_mb} MB reserved` : status?.device ?? '—'}
            </span>
          </div>

          {error ? (
            <p className="text-xs text-destructive">{error}</p>
          ) : models.length === 0 ? (
            <p className="text-xs text-muted-foreground">No models registered yet.</p>
          ) : (
            <div className="max-h-64 space-y-1.5 overflow-auto pr-1">
              {models.map((model) => (
                <div key={model.key} className="flex items-center justify-between gap-2 text-xs">
                  <span className="min-w-0 truncate">
                    <span className="text-muted-foreground">{model.group}</span>
                    <span className="mx-1 text-muted-foreground/50">/</span>
                    <span>{model.label}</span>
                  </span>
                  <span
                    className={cn(
                      'shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium',
                      stateClass[model.state],
                    )}
                    title={model.error ?? undefined}
                  >
                    {stateLabel[model.state]}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
