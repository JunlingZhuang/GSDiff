'use client';

import { Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { GenerationMode } from '@/lib/constants';

interface Props {
  mode: GenerationMode;
  loading: boolean;
  onClick: () => void;
  disabled?: boolean;
}

const LABEL: Record<GenerationMode, string> = {
  unconstrained: 'Generate Floorplan',
  graph: 'Sample Graph',
  topology: 'Generate Floorplan',
  boundary: 'Generate Floorplan',
};

export function GenerateButton({ mode, loading, onClick, disabled }: Props) {
  return (
    <Button onClick={onClick} disabled={disabled || loading} className="h-11 w-full text-sm font-medium">
      {loading ? (
        <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-background/30 border-t-background" />
      ) : (
        <Sparkles className="mr-2 h-4 w-4" />
      )}
      {LABEL[mode]}
    </Button>
  );
}
