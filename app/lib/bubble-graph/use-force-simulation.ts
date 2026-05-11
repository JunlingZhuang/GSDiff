import { useEffect, useRef } from 'react';
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  type Simulation,
} from 'd3-force';
import type { BubbleNodeState, BubbleEdgeState } from './types';

interface UseForceSimulationParams {
  nodes: BubbleNodeState[];
  edges: BubbleEdgeState[];
  width: number;
  height: number;
  onTick: (nodes: BubbleNodeState[]) => void;
}

export function useForceSimulation({
  nodes,
  edges,
  width,
  height,
  onTick,
}: UseForceSimulationParams): {
  reheat: () => void;
  pinNode: (nodeId: number, x: number, y: number) => void;
  releaseNode: (nodeId: number) => void;
} {
  const simRef = useRef<Simulation<BubbleNodeState, BubbleEdgeState> | null>(null);
  const nodesRef = useRef<BubbleNodeState[]>(nodes);

  useEffect(() => {
    nodesRef.current = nodes;
    const sim = forceSimulation<BubbleNodeState>(nodes)
      .force(
        'link',
        forceLink<BubbleNodeState, BubbleEdgeState>(edges)
          .id((d) => d.id)
          .distance(120)
          .strength(0.5),
      )
      .force('charge', forceManyBody<BubbleNodeState>().strength(-400))
      .force('center', forceCenter(width / 2, height / 2))
      .force('collide', forceCollide<BubbleNodeState>(42))
      // Use d3-force defaults (alphaDecay 0.0228, alphaMin 0.001) so the
      // simulation feels "alive" — graph rebalances smoothly after edits.
      .on('tick', () => onTick([...nodesRef.current]));
    simRef.current = sim;
    return () => {
      sim.stop();
      simRef.current = null;
    };
    // Rebuild only when node/edge identity changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes.length, edges.length, width, height]);

  const reheat = () => simRef.current?.alpha(0.5).restart();

  // Mutate the simulation's node directly so the next tick respects the new position.
  // The drag handler must call this on the SAME node object the simulation holds —
  // pass a node from the array given to useForceSimulation, NOT a separately-constructed copy.
  const pinNode = (nodeId: number, x: number, y: number) => {
    const node = nodesRef.current.find((n) => n.id === nodeId);
    if (!node) return;
    node.fx = x;
    node.fy = y;
    node.x = x;
    node.y = y;
    // Tiny alpha bump: enough to trigger one tick so React sees the new
    // position, but not so much that all other nodes resettle dramatically.
    simRef.current?.alpha(0.05).restart();
  };

  const releaseNode = (nodeId: number) => {
    const node = nodesRef.current.find((n) => n.id === nodeId);
    if (node) {
      node.fx = null;
      node.fy = null;
    }
  };

  return { reheat, pinNode, releaseNode };
}
