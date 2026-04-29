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

export function ModelStatusPanel() {
  const [status, setStatus] = useState<ModelStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadStatus() {
      try {
        const nextStatus = await getModelStatus();
        if (!cancelled) {
          setStatus(nextStatus);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Failed to load model status');
        }
      }
    }

    loadStatus();
    const intervalId = window.setInterval(loadStatus, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  const models = status?.models ?? [];
  const loaded = models.filter((model) => model.state === 'loaded').length;
  const loading = models.filter((model) => model.state === 'loading').length;

  return (
    <div className="rounded-xl border border-border/70 bg-background p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu className="h-3.5 w-3.5 text-muted-foreground" />
          <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Models
          </p>
        </div>
        {loading > 0 && <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-700" />}
      </div>

      {error ? (
        <p className="text-xs text-destructive">{error}</p>
      ) : (
        <>
          <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {loaded}/{models.length} loaded
            </span>
            <span>{status?.gpu ? `${status.gpu.reserved_mb} MB reserved` : status?.device}</span>
          </div>
          <div className="max-h-32 space-y-1 overflow-auto pr-1">
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
                    stateClass[model.state]
                  )}
                  title={model.error ?? undefined}
                >
                  {stateLabel[model.state]}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
