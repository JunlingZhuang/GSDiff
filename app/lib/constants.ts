export const ROOM_TYPES = [
  { id: 0, name: 'Living Room', color: '#F4F1DE', textColor: '#4a4637' },
  { id: 1, name: 'Bedroom', color: '#EAB69F', textColor: '#5c3a2b' },
  { id: 2, name: 'Bathroom', color: '#6B705C', textColor: '#ffffff' },
  { id: 3, name: 'Kitchen', color: '#E07A5F', textColor: '#ffffff' },
  { id: 4, name: 'Balcony', color: '#5F797B', textColor: '#ffffff' },
  { id: 5, name: 'Storage', color: '#F2CC8F', textColor: '#5c4a2b' },
] as const;

export type RoomType = (typeof ROOM_TYPES)[number];

export const MIN_ROOMS = 4;
export const MAX_ROOMS = 8;

export const GENERATION_MODES = [
  {
    id: 'unconstrained' as const,
    name: 'Unconstrained',
    description: 'Generate a random floorplan from scratch',
    icon: 'Sparkles',
  },
  {
    id: 'topology' as const,
    name: 'Topology',
    description: 'Define room adjacency as a bubble diagram',
    icon: 'Share2',
  },
  {
    id: 'boundary' as const,
    name: 'Boundary',
    description: 'Draw a building outline to fill with rooms',
    icon: 'PenTool',
  },
] as const;

export type GenerationMode = (typeof GENERATION_MODES)[number]['id'];
