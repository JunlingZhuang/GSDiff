'use client';

import { Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface Props {
  onGenerate: () => void;
  loading: boolean;
}

export function UnconstrainedPanel({ onGenerate, loading }: Props) {
  return (
    <div className="flex flex-1 flex-col">
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-10 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-muted mb-4">
          <Sparkles className="h-6 w-6 text-muted-foreground" />
        </div>
        <h3 className="text-base font-medium">Unconstrained Generation</h3>
        <p className="mt-2 max-w-xs text-sm text-muted-foreground leading-relaxed">
          The model decides everything -- room count, layout, types, and
          connectivity. Each generation produces a unique floorplan.
        </p>
      </div>

      <div className="border-t border-border/60 p-4">
        <Button
          onClick={onGenerate}
          disabled={loading}
          className="w-full h-10 text-sm font-medium"
          size="lg"
        >
          {loading ? (
            <>
              <div className="mr-2 h-3.5 w-3.5 animate-spin rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground" />
              Generating...
            </>
          ) : (
            <>
              <Sparkles className="mr-2 h-3.5 w-3.5" />
              Generate Floorplan
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
