// RPLAN room types — index matches digress/configs/dataset/rplan.yaml node_decoder.
export const ROOM_TYPES_RPLAN = [
  { id: 0, name: 'Living Room', color: '#F4F1DE', textColor: '#4a4637' },
  { id: 1, name: 'Bedroom', color: '#EAB69F', textColor: '#5c3a2b' },
  { id: 2, name: 'Bathroom', color: '#6B705C', textColor: '#ffffff' },
  { id: 3, name: 'Kitchen', color: '#E07A5F', textColor: '#ffffff' },
  { id: 4, name: 'Balcony', color: '#5F797B', textColor: '#ffffff' },
  { id: 5, name: 'Storage', color: '#F2CC8F', textColor: '#5c4a2b' },
] as const;

// MSD room types — index matches digress/configs/dataset/msd_wall.yaml node_decoder.
// Colors mirror digress/src/analysis/msd_utils.py MSD_NODE_COLORS for consistency.
export const ROOM_TYPES_MSD = [
  { id: 0, name: 'Bedroom', color: '#8da0cb', textColor: '#1f2937' },
  { id: 1, name: 'Living Room', color: '#47b39c', textColor: '#ffffff' },
  { id: 2, name: 'Kitchen', color: '#f1c27d', textColor: '#5c3a2b' },
  { id: 3, name: 'Dining', color: '#fdae6b', textColor: '#5c3a2b' },
  { id: 4, name: 'Corridor', color: '#fdd0a2', textColor: '#5c3a2b' },
  { id: 5, name: 'Stairs', color: '#72246c', textColor: '#ffffff' },
  { id: 6, name: 'Storage', color: '#ffd92f', textColor: '#5c4a2b' },
  { id: 7, name: 'Bathroom', color: '#bdbdbd', textColor: '#1f2937' },
  { id: 8, name: 'Balcony', color: '#a6d854', textColor: '#1f2937' },
] as const;

// RPLAN edge types — node_decoder index 0 reserved for "no edge".
// dashArray follows SVG stroke-dasharray syntax. undefined = solid.
export const EDGE_TYPES_RPLAN = [
  { id: 0, name: 'none', color: '#cbd5e1', dashArray: undefined },
  { id: 1, name: 'wall', color: '#1e293b', dashArray: undefined },
  { id: 2, name: 'door', color: '#c2410c', dashArray: '7 5' },
] as const;

// MSD edge types — index matches msd_wall.yaml edge_decoder.
// Each type has a visually distinct stroke pattern so all four are
// recognizable at a glance without reading labels:
//   wall      solid dark slate
//   passage   medium-dashed mid slate
//   door      long-dashed warm red
//   entrance  dotted amber
export const EDGE_TYPES_MSD = [
  { id: 0, name: 'none', color: '#cbd5e1', dashArray: undefined },
  { id: 1, name: 'wall', color: '#1e293b', dashArray: undefined },
  { id: 2, name: 'passage', color: '#64748b', dashArray: '4 3' },
  { id: 3, name: 'door', color: '#c2410c', dashArray: '8 5' },
  { id: 4, name: 'entrance', color: '#b45309', dashArray: '2 3' },
] as const;

export type RoomTypeMeta = { id: number; name: string; color: string; textColor: string };
export type EdgeTypeMeta = { id: number; name: string; color: string; dashArray: string | undefined };

export const DATASETS = [
  {
    id: 'rplan',
    name: 'RPLAN',
    description: 'Residential floorplan graphs (6 room types, 2 edge types)',
    enabled: true,
  },
  {
    id: 'msd_wall',
    name: 'MSD',
    description: 'Modified Swiss Dwellings (9 room types, 4 edge types)',
    enabled: true,
  },
] as const;

export type DatasetId = (typeof DATASETS)[number]['id'];

// Per-dataset spec used by frontend rendering. The order of `roomTypes` and
// `edgeTypes` MUST match the digress dataset config decoders.
export const DATASET_SPECS: Record<DatasetId, { roomTypes: readonly RoomTypeMeta[]; edgeTypes: readonly EdgeTypeMeta[] }> = {
  rplan: { roomTypes: ROOM_TYPES_RPLAN, edgeTypes: EDGE_TYPES_RPLAN },
  msd_wall: { roomTypes: ROOM_TYPES_MSD, edgeTypes: EDGE_TYPES_MSD },
};

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
    id: 'graph' as const,
    name: 'Graph',
    description: 'Sample a room graph before floorplan generation',
    icon: 'Network',
  },
  {
    id: 'topology' as const,
    name: 'Graph Editor',
    description: 'Manually edit a bubble diagram, then generate a floorplan',
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
