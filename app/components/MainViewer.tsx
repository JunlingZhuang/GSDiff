'use client';

import { ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

/**
 * Right-workspace main viewer slot. Just a styled container for now;
 * Task 6 wires it to (mode, selectedHistoryItem, modeDrafts) routing.
 */
export function MainViewer({ children }: Props) {
  return (
    <div className="flex flex-1 items-center justify-center overflow-hidden bg-background p-6">
      <div className="h-full w-full max-w-[1200px]">
        {children}
      </div>
    </div>
  );
}
