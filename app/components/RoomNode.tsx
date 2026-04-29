'use client';

import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { ROOM_TYPES } from '@/lib/constants';

export interface RoomNodeData {
  roomTypeId: number;
  label: string;
  [key: string]: unknown;
}

function RoomNodeComponent({ data, selected }: NodeProps) {
  const nodeData = data as unknown as RoomNodeData;
  const roomType = ROOM_TYPES[nodeData.roomTypeId];
  if (!roomType) return null;

  const handleStyle =
    '!h-4 !w-4 !border-2 !border-background !bg-foreground/55 !opacity-0 transition-opacity group-hover:!opacity-80';

  return (
    <div className="group relative">
      {/* A full-bubble invisible handle keeps topology editing unconstrained. */}
      <Handle
        type="source"
        position={Position.Right}
        id="room-handle"
        className={handleStyle}
        isConnectable
      />
      <Handle
        type="target"
        position={Position.Left}
        id="room-target"
        className={handleStyle}
        isConnectable
      />

      <div
        className="flex h-16 w-16 items-center justify-center rounded-full border-2 transition-all duration-150 cursor-grab active:cursor-grabbing"
        style={{
          backgroundColor: roomType.color,
          borderColor: selected ? 'oklch(0.205 0 0)' : 'oklch(0.2 0 0 / 0.18)',
          boxShadow: selected
            ? '0 0 0 3px oklch(0.205 0 0 / 0.16)'
            : 'none',
        }}
      >
        <span
          className="text-[10px] font-semibold leading-tight text-center px-1 select-none"
          style={{ color: roomType.textColor }}
        >
          {nodeData.label}
          </span>
      </div>
    </div>
  );
}

export const RoomNode = memo(RoomNodeComponent);
