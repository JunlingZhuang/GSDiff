'use client';

import { useCallback, useMemo } from 'react';
import {
  ReactFlow,
  useNodesState,
  useEdgesState,
  addEdge,
  Controls,
  Background,
  BackgroundVariant,
  type Connection,
  type Node,
  type Edge,
} from '@xyflow/react';
import { Share2, Trash2, Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { RoomNode, type RoomNodeData } from '@/components/RoomNode';
import { ROOM_TYPES, MIN_ROOMS, MAX_ROOMS } from '@/lib/constants';

interface Props {
  onGenerate: (rooms: number[], adjacency: number[][]) => void;
  loading: boolean;
}

const nodeTypes = { room: RoomNode };

export function TopologyEditor({ onGenerate, loading }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const onConnect = useCallback(
    (connection: Connection) => {
      // Prevent duplicate edges
      const exists = edges.some(
        (e) =>
          (e.source === connection.source && e.target === connection.target) ||
          (e.source === connection.target && e.target === connection.source)
      );
      if (exists) return;
      setEdges((eds) =>
        addEdge(
          {
            ...connection,
            style: { stroke: 'oklch(0.439 0 0)', strokeWidth: 2 },
          },
          eds
        )
      );
    },
    [edges, setEdges]
  );

  const addRoom = useCallback(
    (typeId: number) => {
      if (nodes.length >= MAX_ROOMS) return;
      const roomType = ROOM_TYPES[typeId];
      const id = `room-${Date.now()}`;
      // Place nodes in a grid pattern
      const col = nodes.length % 3;
      const row = Math.floor(nodes.length / 3);
      const newNode: Node = {
        id,
        type: 'room',
        position: { x: 80 + col * 120, y: 60 + row * 120 },
        data: {
          roomTypeId: typeId,
          label: roomType.name,
        } satisfies RoomNodeData,
      };
      setNodes((nds) => [...nds, newNode]);
    },
    [nodes.length, setNodes]
  );

  const deleteSelected = useCallback(() => {
    setNodes((nds) => nds.filter((n) => !n.selected));
    setEdges((eds) => {
      const selectedNodeIds = new Set(
        nodes.filter((n) => n.selected).map((n) => n.id)
      );
      return eds.filter(
        (e) =>
          !e.selected &&
          !selectedNodeIds.has(e.source) &&
          !selectedNodeIds.has(e.target)
      );
    });
  }, [nodes, setNodes, setEdges]);

  const clearAll = useCallback(() => {
    setNodes([]);
    setEdges([]);
  }, [setNodes, setEdges]);

  const handleGenerate = useCallback(() => {
    // Build room type array and adjacency matrix from React Flow state
    const nodeIds = nodes.map((n) => n.id);
    const nodeIndexMap = new Map(nodeIds.map((id, i) => [id, i]));
    const rooms = nodes.map(
      (n) => (n.data as unknown as RoomNodeData).roomTypeId
    );

    const n = rooms.length;
    const adjacency: number[][] = Array.from({ length: n }, () =>
      Array(n).fill(0)
    );

    for (const edge of edges) {
      const i = nodeIndexMap.get(edge.source);
      const j = nodeIndexMap.get(edge.target);
      if (i !== undefined && j !== undefined) {
        adjacency[i][j] = 1;
        adjacency[j][i] = 1;
      }
    }

    onGenerate(rooms, adjacency);
  }, [nodes, edges, onGenerate]);

  const hasSelection = useMemo(
    () => nodes.some((n) => n.selected) || edges.some((e) => e.selected),
    [nodes, edges]
  );

  const canGenerate = nodes.length >= MIN_ROOMS && !loading;

  return (
    <div className="flex flex-1 flex-col">
      {/* Room palette */}
      <div className="border-b border-border/60 px-4 py-3">
        <div className="flex items-center justify-between mb-2">
          <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Add Rooms ({nodes.length}/{MAX_ROOMS})
          </p>
          {nodes.length > 0 && (
            <button
              onClick={clearAll}
              className="text-[11px] text-muted-foreground hover:text-destructive transition-colors"
            >
              Clear all
            </button>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {ROOM_TYPES.map((rt) => (
            <button
              key={rt.id}
              disabled={nodes.length >= MAX_ROOMS}
              onClick={() => addRoom(rt.id)}
              className="flex items-center gap-1.5 rounded-lg border border-border/80 bg-background px-2 py-1.5 text-xs font-medium transition-all duration-150 hover:border-foreground/20 hover:shadow-sm disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-full ring-1 ring-black/5"
                style={{ backgroundColor: rt.color }}
              />
              {rt.name}
              <Plus className="h-3 w-3 text-muted-foreground" />
            </button>
          ))}
        </div>
      </div>

      {/* React Flow canvas */}
      <div className="relative flex-1 min-h-[280px]">
        {nodes.length === 0 && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center pointer-events-none">
            <Share2 className="h-8 w-8 text-muted-foreground/30 mb-2" />
            <p className="text-sm text-muted-foreground/60">
              Add rooms to start building a topology
            </p>
            <p className="text-xs text-muted-foreground/40 mt-1">
              Drag between nodes to create adjacency edges
            </p>
          </div>
        )}
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
          deleteKeyCode={['Backspace', 'Delete']}
          className="bg-muted/30"
          proOptions={{ hideAttribution: true }}
        >
          <Controls
            showInteractive={false}
            className="!shadow-sm !border !border-border/60 !rounded-lg"
          />
          <Background
            variant={BackgroundVariant.Dots}
            gap={20}
            size={1}
            color="oklch(0.708 0 0 / 0.3)"
          />
        </ReactFlow>

        {/* Floating toolbar */}
        {hasSelection && (
          <div className="absolute bottom-14 left-1/2 z-20 -translate-x-1/2">
            <Button
              variant="destructive"
              size="sm"
              onClick={deleteSelected}
              className="shadow-lg"
            >
              <Trash2 className="mr-1.5 h-3.5 w-3.5" />
              Delete selected
            </Button>
          </div>
        )}
      </div>

      {/* Generate button */}
      <div className="border-t border-border/60 p-4">
        {nodes.length > 0 && nodes.length < MIN_ROOMS && (
          <p className="mb-2 text-center text-xs text-muted-foreground">
            Add {MIN_ROOMS - nodes.length} more room{MIN_ROOMS - nodes.length > 1 ? 's' : ''} to generate
          </p>
        )}
        <Button
          onClick={handleGenerate}
          disabled={!canGenerate}
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
              <Share2 className="mr-2 h-3.5 w-3.5" />
              Generate from Topology
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
