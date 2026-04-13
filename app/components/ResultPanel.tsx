'use client';

import { Download, ImageIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { RoomLegend } from '@/components/RoomLegend';

interface Props {
  image: string | null;
  loading: boolean;
  error: string | null;
}

export function ResultPanel({ image, loading, error }: Props) {
  const handleDownload = () => {
    if (!image) return;
    const a = document.createElement('a');
    a.href = image;
    a.download = `gsdiff-floorplan-${Date.now()}.png`;
    a.click();
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/60 px-6 py-3">
        <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          Result
        </p>
        {image && !loading && (
          <Button variant="ghost" size="sm" onClick={handleDownload} className="text-xs">
            <Download className="mr-1.5 h-3 w-3" />
            Download
          </Button>
        )}
      </div>

      {/* Main content area */}
      <div className="relative flex flex-1 flex-col items-center justify-center p-6">
        {error && (
          <div className="mb-4 w-full max-w-md rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center gap-4">
            <div className="relative h-64 w-64 overflow-hidden rounded-xl bg-muted">
              <div className="skeleton-pulse absolute inset-0 bg-gradient-to-br from-muted via-muted-foreground/5 to-muted" />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="flex flex-col items-center gap-3">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-muted-foreground/20 border-t-muted-foreground/60" />
                  <p className="text-sm font-medium text-muted-foreground">
                    Generating...
                  </p>
                  <p className="text-xs text-muted-foreground/60">
                    This takes about 30 seconds
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {!loading && !image && !error && (
          <div className="flex flex-col items-center gap-3 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-muted">
              <ImageIcon className="h-7 w-7 text-muted-foreground/40" />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">
                No floorplan yet
              </p>
              <p className="mt-1 text-xs text-muted-foreground/60">
                Configure a mode and click generate
              </p>
            </div>
          </div>
        )}

        {!loading && image && (
          <div className="flex flex-col items-center gap-4">
            <div className="overflow-hidden rounded-xl border border-border/60 shadow-sm">
              <img
                src={image}
                alt="Generated floorplan"
                className="max-h-[500px] max-w-full object-contain"
              />
            </div>
          </div>
        )}

        {/* Legend — floating over canvas */}
        <div className="absolute bottom-4 right-4 rounded-xl border border-border/60 bg-card/90 backdrop-blur-sm shadow-lg px-4 py-3">
          <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground mb-2">
            Room Legend
          </p>
          <RoomLegend />
        </div>
      </div>
    </div>
  );
}
