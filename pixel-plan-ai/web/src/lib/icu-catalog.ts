// Frontend mirror of data/icu_assets.json for the ICU room flow. The Python
// backend is the validation authority; this typed copy drives the panel
// checklist and the 2D/3D renderers (symbol + model asset paths under
// /public/assets). Keep the entries in sync with data/icu_assets.json.

export type IcuAnchor = "floor" | "wall" | "ceiling" | "mobile";

export interface IcuCatalogEntry {
  type: string;
  label: string;
  count_required: number;
  /** [width_ft, depth_ft] catalog footprint, before rotation. */
  footprint_ft: [number, number];
  anchor: IcuAnchor;
  /** Plan-view SVG symbol, viewBox sized footprint_ft * 24 (1 ft = 24 units). */
  symbol: string;
  /** Binary GLB model, meters at real-world scale (1 ft = 0.3048 m). */
  model: string;
}

export const ICU_CATALOG: IcuCatalogEntry[] = [
  {
    type: "icu_bed",
    label: "ICU Bed",
    count_required: 1,
    footprint_ft: [3.5, 7.5],
    anchor: "floor",
    symbol: "/assets/2d/icu_bed.svg",
    model: "/assets/3d/icu_bed.glb",
  },
  {
    type: "ceiling_boom",
    label: "Ceiling Boom",
    count_required: 2,
    footprint_ft: [1.5, 1.5],
    anchor: "ceiling",
    symbol: "/assets/2d/ceiling_boom.svg",
    model: "/assets/3d/ceiling_boom.glb",
  },
  {
    type: "patient_monitor",
    label: "Patient Monitor",
    count_required: 1,
    footprint_ft: [1.5, 0.75],
    anchor: "wall",
    symbol: "/assets/2d/patient_monitor.svg",
    model: "/assets/3d/patient_monitor.glb",
  },
  {
    type: "iv_pole",
    label: "IV Pole",
    count_required: 1,
    footprint_ft: [1.25, 1.25],
    anchor: "mobile",
    symbol: "/assets/2d/iv_pole.svg",
    model: "/assets/3d/iv_pole.glb",
  },
  {
    type: "visitor_chair",
    label: "Visitor Chair",
    count_required: 1,
    footprint_ft: [2.0, 2.0],
    anchor: "floor",
    symbol: "/assets/2d/visitor_chair.svg",
    model: "/assets/3d/visitor_chair.glb",
  },
  {
    type: "casework",
    label: "Casework / Supply Cabinet",
    count_required: 1,
    footprint_ft: [6.0, 2.0],
    anchor: "wall",
    symbol: "/assets/2d/casework.svg",
    model: "/assets/3d/casework.glb",
  },
  {
    type: "handwash_sink",
    label: "Handwash Sink",
    count_required: 1,
    footprint_ft: [2.0, 1.75],
    anchor: "wall",
    symbol: "/assets/2d/handwash_sink.svg",
    model: "/assets/3d/handwash_sink.glb",
  },
  {
    type: "overbed_table",
    label: "Overbed Table",
    count_required: 1,
    footprint_ft: [2.5, 1.25],
    anchor: "mobile",
    symbol: "/assets/2d/overbed_table.svg",
    model: "/assets/3d/overbed_table.glb",
  },
];

export const ICU_CATALOG_BY_TYPE: Record<string, IcuCatalogEntry> = Object.fromEntries(
  ICU_CATALOG.map((entry) => [entry.type, entry]),
);
