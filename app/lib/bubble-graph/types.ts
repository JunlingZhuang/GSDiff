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
}

export interface BubbleSelection {
  nodeId: number | null;
  edgeId: string | null;
}
