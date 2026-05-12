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
import { FORCE_LAYOUT_CONFIG } from './config/force-layout';

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
function endpointId(endpoint: number | BubbleNodeState): number {
  return typeof endpoint === 'object' ? endpoint.id : endpoint;
}

function cloneLinksForSimulation(edges: BubbleEdgeState[]): BubbleEdgeState[] {
  return edges.map((edge) => ({
    ...edge,
    // d3.forceLink mutates source/target from ids into node references.
    // Keep that mutation inside these throwaway link objects; never let it
    // leak into React reducer state, or old links can keep pointing at stale
    // node objects after ticks rebuild node arrays.
    source: endpointId(edge.source as number | BubbleNodeState),
    target: endpointId(edge.target as number | BubbleNodeState),
  }));
}

function computeDegrees(
  nodes: BubbleNodeState[],
  edges: BubbleEdgeState[],
): Map<number, number> {
  const map = new Map<number, number>();
  for (const n of nodes) map.set(n.id, 0);
  for (const e of edges) {
    const s = endpointId(e.source as number | BubbleNodeState);
    const t = endpointId(e.target as number | BubbleNodeState);
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
    return deg > 0
      ? FORCE_LAYOUT_CONFIG.connectedChargeStrength
      : FORCE_LAYOUT_CONFIG.isolatedChargeStrength;
  };
  const linkStrengthFn = (link: BubbleEdgeState): number => {
    const sourceId = endpointId(link.source as number | BubbleNodeState);
    const targetId = endpointId(link.target as number | BubbleNodeState);
    const sourceDegree = degreesRef.current.get(sourceId) ?? 1;
    const targetDegree = degreesRef.current.get(targetId) ?? 1;
    const maxDegree = Math.max(sourceDegree, targetDegree);
    // Hub links need to be stiffer; otherwise a star graph's leaves repel each
    // other and stretch the older links much longer than newly-added links.
    return Math.min(
      FORCE_LAYOUT_CONFIG.linkMaxStrength,
      FORCE_LAYOUT_CONFIG.linkBaseStrength
        + maxDegree * FORCE_LAYOUT_CONFIG.linkStrengthPerDegree,
    );
  };

  // -------------------------------------------------------------------------
  // Lifecycle — create / destroy simulation. Runs only when canvas size
  // changes; node/edge updates are handled in separate effects below.
  // -------------------------------------------------------------------------
  useEffect(() => {
    nodesRef.current = nodes;
    const simulationLinks = cloneLinksForSimulation(edges);
    const sim = forceSimulation<BubbleNodeState>(nodes)
      .force(
        'link',
        forceLink<BubbleNodeState, BubbleEdgeState>(simulationLinks)
          .id((d) => d.id)
          // distance + strength tuned together: stiff springs at moderate
          // distance keep edges visually uniform. Tweaking these is the
          // primary lever for "tighter" vs "looser" layout.
          .distance(FORCE_LAYOUT_CONFIG.linkDistance)
          .strength(linkStrengthFn)
          // D3's documented way to make link distance constraints more rigid.
          // This is important for hub-and-spoke floorplan graphs where a
          // newly-added leaf should not make older hub links stretch out.
          .iterations(FORCE_LAYOUT_CONFIG.linkIterations),
      )
      .force(
        'charge',
        forceManyBody<BubbleNodeState>()
          .strength(chargeStrengthFn)
          .distanceMax(FORCE_LAYOUT_CONFIG.chargeDistanceMax),
      )
      .force(
        'center',
        forceCenter(width / 2, height / 2).strength(FORCE_LAYOUT_CONFIG.centerStrength),
      )
      .force('collide', forceCollide<BubbleNodeState>(FORCE_LAYOUT_CONFIG.collideRadius))
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
    const simulationLinks = cloneLinksForSimulation(edges);
    if (linkForce) linkForce.links(simulationLinks);
    // Refresh per-node degrees so the charge force re-evaluates isolated vs
    // connected. d3-force caches strength values from the function, so we
    // must re-set .strength() to make it pick up the new degrees.
    degreesRef.current = computeDegrees(nodes, edges);
    if (linkForce) {
      linkForce
        .distance(FORCE_LAYOUT_CONFIG.linkDistance)
        .strength(linkStrengthFn)
        .iterations(FORCE_LAYOUT_CONFIG.linkIterations);
    }
    const charge = sim.force('charge') as {
      strength: (fn: (d: BubbleNodeState) => number) => void;
    } | undefined;
    if (charge) charge.strength(chargeStrengthFn);
    // Mild reheat so the new/removed element settles into place without
    // throwing the rest of the graph around.
    sim.alpha(FORCE_LAYOUT_CONFIG.structuralReheatAlpha).restart();
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

  const reheat = () => simRef.current?.alpha(FORCE_LAYOUT_CONFIG.structuralReheatAlpha).restart();

  const pinNode = (nodeId: number, x: number, y: number) => {
    const node = nodesRef.current.find((n) => n.id === nodeId);
    if (!node) return;
    node.fx = x;
    node.fy = y;
    node.x = x;
    node.y = y;
    // Tiny alpha so the render reflects the new position without
    // disturbing other nodes.
    simRef.current?.alpha(FORCE_LAYOUT_CONFIG.dragReheatAlpha).restart();
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
