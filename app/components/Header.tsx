'use client';

import { ExternalLink, LayoutDashboard } from 'lucide-react';

export function Header() {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border/60 bg-card px-6">
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-foreground">
          <LayoutDashboard className="h-4 w-4 text-background" />
        </div>
        <div>
          <h1 className="text-sm font-semibold tracking-tight leading-none">
            GSDiff
          </h1>
          <p className="text-[11px] text-muted-foreground leading-tight mt-0.5">
            Structural Graph Diffusion
          </p>
        </div>
      </div>

      <a
        href="https://github.com/JunlingZhuang/GSDiff"
        target="_blank"
        rel="noopener noreferrer"
        className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      >
        <ExternalLink className="h-4 w-4" />
      </a>
    </header>
  );
}
