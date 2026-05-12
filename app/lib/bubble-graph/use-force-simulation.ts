import { useEffect, useRef } from 'react';
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  type ForceLink,
  type ForceCenter,
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

/**
 * Force-directed graph simulation, owned by this hook.
 *
 * Architectural note: the simulation is created ONCE per component lifetime
 * and updated *in place* when nodes/edges/dimensions change. We do NOT
 * tear down and re-create the sim on every add/delete, because rebuilding
 * resets alpha to 1, which makes the whole graph visibly resettle — the
 * user perceives it as "the graph keeps spreading out every time I add a
 * node." Instead, `sim.nodes()` and the link force's `.links()` are
 * mutated in place and we apply a small alpha bump (0.3) so just the
 * affected element nudges into place.
 */
// Per-node charge strength: connected nodes repel each other strongly to
// produce a clean spread; isolated (degree-zero) nodes get almost no charge
// so they don't push the rest of the graph apart when they're sitting in
// the middle waiting to be wired up.
const CHARGE_CONNECTED = -450;
const CHARGE_ISOLATED = -30;

function computeDegrees(
  nodes: BubbleNodeState[],
  edges: BubbleEdgeState[],
): Map<number, number> {
  const map = new Map<number, number>();
  for (const n of nodes) map.set(n.id, 0);
  for (const e of edges) {
    const s = typeof e.source === 'object' ? (e.source as unknown as BubbleNodeState).id : e.source;
    const t = typeof e.target === 'object' ? (e.target as unknown as BubbleNodeState).id : e.target;
    map.set(s, (map.get(s) ?? 0) + 1);
    map.set(t, (map.get(t) ?? 0) + 1);
  }
  return map;
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
  const onTickRef = useRef(onTick);
  onTickRef.current = onTick;

  // Degrees by node id — read by the charge strength function. Re-computed
  // every time the graph structure changes; the closure reads from this
  // ref so the strength function stays current.
  const degreesRef = useRef<Map<number, number>>(computeDegrees(nodes, edges));
  const chargeStrengthFn = (d: BubbleNodeState): number => {
    const deg = degreesRef.current.get(d.id) ?? 0;
    return deg > 0 ? CHARGE_CONNECTED : CHARGE_ISOLATED;
  };

  // -------------------------------------------------------------------------
  // Lifecycle — create / destroy simulation. Runs only when canvas size
  // changes; node/edge updates are handled in separate effects below.
  // -------------------------------------------------------------------------
  useEffect(() => {
    nodesRef.current = nodes;
    const sim = forceSimulation<BubbleNodeState>(nodes)
      .force(
        'link',
        forceLink<BubbleNodeState, BubbleEdgeState>(edges)
          .id((d) => d.id)
          // distance + strength tuned together: stiff springs at moderate
          // distance keep edges visually uniform. Tweaking these is the
          // primary lever for "tighter" vs "looser" layout.
          .distance(105)
          .strength(0.9),
      )
      .force(
        'charge',
        forceManyBody<BubbleNodeState>()
          .strength(chargeStrengthFn)
          .distanceMax(500),
      )
      .force(
        'center',
        forceCenter(width / 2, height / 2).strength(0.05),
      )
      .force('collide', forceCollide<BubbleNodeState>(44))
      .on('tick', () => onTickRef.current([...nodesRef.current]));
    simRef.current = sim;
    return () => {
      sim.stop();
      simRef.current = null;
    };
    // Intentionally NOT depending on nodes/edges identity — those are
    // handled by the dedicated effects below. Re-creating the simulation
    // for every node addition is exactly the "graph keeps spreading" bug.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height]);

  // -------------------------------------------------------------------------
  // Sync simulation nodes/edges when their count changes (add/delete).
  // Uses in-place updates so existing node positions are preserved.
  // -------------------------------------------------------------------------
  useEffect(() => {
    const sim = simRef.current;
    if (!sim) return;
    nodesRef.current = nodes;
    sim.nodes(nodes);
    const linkForce = sim.force('link') as ForceLink<BubbleNodeState, BubbleEdgeState> | undefined;
    if (linkForce) linkForce.links(edges);
    // Refresh per-node degrees so the charge force re-evaluates isolated vs
    // connected. d3-force caches strength values from the function, so we
    // must re-set .strength() to make it pick up the new degrees.
    degreesRef.current = computeDegrees(nodes, edges);
    const charge = sim.force('charge') as {
      strength: (fn: (d: BubbleNodeState) => number) => void;
    } | undefined;
    if (charge) charge.strength(chargeStrengthFn);
    // Mild reheat so the new/removed element settles into place without
    // throwing the rest of the graph around.
    sim.alpha(0.3).restart();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes.length, edges.length]);

  // -------------------------------------------------------------------------
  // Keep center force in sync with canvas size (resize, etc.) without
  // rebuilding the whole simulation.
  // -------------------------------------------------------------------------
  useEffect(() => {
    const sim = simRef.current;
    if (!sim) return;
    const center = sim.force('center') as ForceCenter<BubbleNodeState> | undefined;
    if (center) {
      center.x(width / 2).y(height / 2);
    }
  }, [width, height]);

  const reheat = () => simRef.current?.alpha(0.3).restart();

  const pinNode = (nodeId: number, x: number, y: number) => {
    const node = nodesRef.current.find((n) => n.id === nodeId);
    if (!node) return;
    node.fx = x;
    node.fy = y;
    node.x = x;
    node.y = y;
    // Tiny alpha so the render reflects the new position without
    // disturbing other nodes.
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
