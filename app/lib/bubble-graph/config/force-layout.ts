/**
 * Centralized D3 force-layout tuning for the editable bubble graph.
 *
 * The editor uses D3 as a live layout helper, not as the source of truth.
 * Keep these parameters together so graph-editing behavior can be tuned
 * without hunting through React components.
 */
export const FORCE_LAYOUT_CONFIG = {
  /**
   * Target length for every rendered graph edge, in SVG viewBox units.
   * D3 treats this as a spring target, not a hard geometric constraint.
   */
  linkDistance: 105,

  /**
   * Number of link-force relaxation passes per simulation tick.
   * Higher values make edge lengths more rigid and help hub-and-spoke graphs
   * keep all spokes at similar lengths.
   */
  linkIterations: 8,

  /**
   * Base link spring strength before degree-based adjustment.
   * Lower values allow looser layouts; higher values keep edges tighter.
   */
  linkBaseStrength: 0.72,

  /**
   * Extra link spring strength added for each degree of the higher-degree
   * endpoint. This prevents hub links from becoming weaker than leaf links.
   */
  linkStrengthPerDegree: 0.07,

  /**
   * Upper bound for computed link spring strength.
   * D3 expects link strength in the [0, 1] range.
   */
  linkMaxStrength: 1,

  /**
   * Repulsive force for nodes with at least one edge.
   * Keep this weaker than link stiffness so newly-added leaves do not push old
   * leaves far away from the hub.
   */
  connectedChargeStrength: -220,

  /**
   * Repulsive force for isolated nodes.
   * Isolated rooms should not disturb an already-edited graph much.
   */
  isolatedChargeStrength: -30,

  /**
   * Maximum range for charge interactions.
   * Smaller values reduce long-distance spreading in larger graphs.
   */
  chargeDistanceMax: 260,

  /**
   * Radius used by the collision force.
   * This should stay close to the rendered node radius plus label padding.
   */
  collideRadius: 44,

  /**
   * Strength for gently keeping the whole graph near the canvas center.
   * Too high will fight user edits; too low lets graphs drift off-screen.
   */
  centerStrength: 0.05,

  /**
   * Alpha used after structural edits such as adding/removing nodes or edges.
   * Higher values rebalance faster but visibly disturb more of the graph.
   */
  structuralReheatAlpha: 0.3,

  /**
   * Alpha used while dragging a node.
   * Kept low so drag updates are visible without resettling the whole graph.
   */
  dragReheatAlpha: 0.05,
} as const;
