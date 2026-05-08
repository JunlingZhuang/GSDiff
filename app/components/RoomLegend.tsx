'use client';

import { DATASET_SPECS, type DatasetId } from '@/lib/constants';

interface Props {
  dataset: DatasetId;
}

export function RoomLegend({ dataset }: Props) {
  const roomTypes = DATASET_SPECS[dataset].roomTypes;
  // 9 room types (MSD) wraps better in 3 columns; 6 (RPLAN) also fits cleanly.
  return (
    <div className="grid grid-cols-3 gap-x-6 gap-y-2.5">
      {roomTypes.map((rt) => (
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
