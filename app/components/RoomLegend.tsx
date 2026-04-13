'use client';

import { ROOM_TYPES } from '@/lib/constants';

export function RoomLegend() {
  return (
    <div className="grid grid-cols-3 gap-x-6 gap-y-2.5">
      {ROOM_TYPES.map((rt) => (
        <div key={rt.id} className="flex items-center gap-2">
          <span
            className="inline-block h-4 w-4 rounded ring-1 ring-black/10 shrink-0"
            style={{ backgroundColor: rt.color }}
          />
          <span className="text-sm text-foreground">{rt.name}</span>
        </div>
      ))}
    </div>
  );
}
