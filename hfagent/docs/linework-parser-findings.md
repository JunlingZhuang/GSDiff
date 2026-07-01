# Linework Parser Baseline

## Status

Status: promising geometry; room semantics are not yet good enough.

The mode is selected with:

```powershell
python -m hfagent.evaluate --program hospital-tower-floor --structure-mode linework
```

It preserves the existing `colorblock` and `json` modes. The final contracts
remain `Plan` and `RoomGraph`; no AI-generated colour mask is used. A deterministic
mask is rendered after parsing for compatibility.

## Pipeline

```text
real line plan
-> EasyOCR labels
-> erase OCR glyphs from a geometry-only copy
-> multi-scale horizontal/vertical wall extraction
-> wall-axis clustering and junction snapping
-> classify each local gap as door, window or open passage
-> Shapely polygonize of the planar wall graph
-> OCR point-in-polygon room typing
-> door gaps mapped to polygons on both sides
-> Plan + RoomGraph
```

There is no global morphology closing and no free-space connected-component
room segmentation. Directional morphology is used only to extract wall strokes.

Artifacts are written beside normal evaluation outputs:

- `linework_r*.debug.png`: walls, OCR boxes, polygons and door candidates
- `linework_r*.json`: parser confidence and detections
- `gemini_r*.png`: deterministic mask rendered from parsed polygons

## Real Evaluation

The first implementation used global gap closing and failed:

- `hfagent/out/eval/20260629-151504/hospital-tower-floor`
- `hfagent/out/eval/20260629-151843/hospital-tower-floor`

It split corridors, leaked through exterior openings and produced fragmented
reconstructions. That implementation has been removed.

The replacement vector-wall parser was tested offline on the same source drawing:

- Debug: `hfagent/out/eval/20260629-151843/hospital-tower-floor/linework_vector_v3.debug.png`
- Recon: `hfagent/out/eval/20260629-151843/hospital-tower-floor/linework_vector_v3.recon.png`
- 124 vector wall segments produced 82 polygonized regions.
- 81 regions survived geometry validation, versus 82 requested by the program.
- 69 interior door pairs were mapped from local opening candidates.
- The reconstruction now has continuous shared boundaries and preserves the four
  wings and central circulation geometry.
- 21 regions remain `unknown` because the generated source drawing omitted,
  duplicated or corrupted labels. Local crop OCR recovered only one additional
  label, so these regions are deliberately not guessed from the program.

## Current Decision

Keep `structure_mode=colorblock` as the default until a fresh real evaluation is
reviewed. The vector wall graph is now a viable geometry backend. The remaining
limitation is semantic labelling on roughly 80-room drawings, not polygon closure.
Linework mode never runs deterministic count splitting/relabeling because that
would corrupt trustworthy geometry to hide OCR or generation errors.
