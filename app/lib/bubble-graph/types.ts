export interface BubbleNodeState {
  id: number;
  attr: number;
  x: number;
  y: number;
  fx: number | null;  // fixed x (set when user drags)
  fy: number | null;
}

export interface BubbleEdgeState {
  id: string;          // `${source}-${target}`
  source: number;
  target: number;
  edgeType: number;
  /** Internal layout hint. Wall edges are displayed but should not act like strong access constraints. */
  layoutRole?: 'wall' | 'access';
}

export interface BubbleSelection {
  nodeId: number | null;
  edgeId: string | null;
}
