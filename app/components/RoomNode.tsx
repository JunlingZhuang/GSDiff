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

  const handleStyle = "!w-3 !h-3 !bg-muted-foreground/60 !border-2 !border-background hover:!bg-foreground transition-colors";

  return (
    <>
      {/* Each position has both source and target handles for bidirectional connections */}
      <Handle type="source" position={Position.Top} id="top-src" className={handleStyle} isConnectable />
      <Handle type="target" position={Position.Top} id="top-tgt" className={handleStyle} isConnectable />
      <Handle type="source" position={Position.Bottom} id="bottom-src" className={handleStyle} isConnectable />
      <Handle type="target" position={Position.Bottom} id="bottom-tgt" className={handleStyle} isConnectable />
      <Handle type="source" position={Position.Left} id="left-src" className={handleStyle} isConnectable />
      <Handle type="target" position={Position.Left} id="left-tgt" className={handleStyle} isConnectable />
      <Handle type="source" position={Position.Right} id="right-src" className={handleStyle} isConnectable />
      <Handle type="target" position={Position.Right} id="right-tgt" className={handleStyle} isConnectable />

      <div
        className="flex h-16 w-16 items-center justify-center rounded-full border-2 shadow-sm transition-all duration-150 cursor-grab active:cursor-grabbing"
        style={{
          backgroundColor: roomType.color,
          borderColor: selected ? 'oklch(0.205 0 0)' : 'transparent',
          boxShadow: selected
            ? '0 0 0 3px oklch(0.205 0 0 / 0.15)'
            : '0 1px 3px rgba(0,0,0,0.08)',
        }}
      >
        <span
          className="text-[10px] font-semibold leading-tight text-center px-1 select-none"
          style={{ color: roomType.textColor }}
        >
          {nodeData.label}
        </span>
      </div>
    </>
  );
}

export const RoomNode = memo(RoomNodeComponent);
