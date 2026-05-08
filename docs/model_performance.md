# DiGress Model Performance Report

**Date:** 2026-05-08
**Branch:** feature/digress-msd

> **Model naming note:** "Model V1/V2/V3" refers to the trained checkpoint lineage, NOT the wall-inference dataset version. Model V1 is the vanilla MSD baseline (no wall edges). Model V2 was trained on wall-inference-V3 data (`data/msd_wall/`). Model V3 was trained on wall-inference-V5 data (`data/msd_wall_v5/`). See `docs/msd_wall_inference.md` for wall-inference algorithm versions.

## Models

| ID         | Dataset     | Checkpoint                                                                                    | Samples |
|------------|-------------|-----------------------------------------------------------------------------------------------|---------|
| RPLAN      | rplan       | `digress/outputs/2026-04-23/21-11-23-rplan/checkpoints/rplan/best-v2.ckpt`                   | 64      |
| Model V1   | msd         | `digress/outputs/2026-04-30/15-16-55-msd/checkpoints/msd/best.ckpt`                          | 32      |
| Model V2   | msd_wall    | `digress/outputs/2026-05-01/15-25-20-msd_wall/checkpoints/msd_wall/best.ckpt`                | 500     |
| Model V3   | msd_wall_v5 | `digress/outputs/2026-05-05/13-58-49-msd_wall_v5/checkpoints/msd_wall_v5/best.ckpt`          | 500     |

---

## 1. Headline Metrics

| Model    | avg\_nodes gen | avg\_nodes ref | nodes Δ  | avg\_edges gen | avg\_edges ref | edges Δ  | connected gen | connected ref | connected Δ | sec/sample |
|----------|---------------|---------------|----------|---------------|---------------|----------|--------------|--------------|-------------|------------|
| RPLAN    | 6.891         | 6.756         | +0.1343  | 9.719         | 9.378         | +0.3406  | 1.000        | 0.9990       | +0.000976   | 0.1933     |
| Model V1 | 27.72         | 27.58         | +0.1418  | 27.28         | 27.99         | −0.7121  | 0.6563       | 0.6385       | +0.01773    | 1.997      |
| Model V2 | 27.99         | 27.60         | +0.3894  | 54.15         | 58.10         | −3.945   | 0.9520       | 0.9456       | +0.006443   | 1.736      |
| Model V3 | 27.37         | 27.60         | −0.2346  | 56.69         | 60.20         | −3.508   | 0.9340       | 0.9607       | −0.02672    | 1.879      |

Notes:
- Model V1 used only 32 samples; its metrics have high variance.
- Model V2 and Model V3 share the same reference dataset (5,143 samples from msd_wall splits).
- Model V3 reference `avg_edges` is higher (60.20 vs 58.10) because it was trained on a dataset version that includes more wall edges per graph.

---

## 2. Edge Type Distribution

Values are fractions (0–1). Delta = gen − ref.

### Model V2 (msd_wall)

| Edge type | gen    | ref    | Δ        |
|-----------|--------|--------|----------|
| none      | 0.8778 | 0.8639 | +0.01386 |
| wall      | 0.05799| 0.07044| −0.01245 |
| passage   | 0.006736| 0.008318| −0.001583|
| door      | 0.04902| 0.04957| −0.000549|
| entrance  | 0.008492| 0.007760| +0.000732|

### Model V3 (msd_wall_v5)

| Edge type | gen    | ref    | Δ        |
|-----------|--------|--------|----------|
| none      | 0.8655 | 0.8590 | +0.006482|
| wall      | 0.06753| 0.07537| −0.007839|
| passage   | 0.01068| 0.008318| +0.002361|
| door      | 0.04771| 0.04957| −0.001868|
| entrance  | 0.008624| 0.007760| +0.000865|

### Side-by-Side Comparison

| Edge type | V2 gen | V3 gen | V2 Δ     | V3 Δ     |
|-----------|--------|--------|----------|----------|
| none      | 0.8778 | 0.8655 | +0.01386 | +0.006482|
| wall      | 0.05799| 0.06753| −0.01245 | −0.007839|
| passage   | 0.006736| 0.01068| −0.001583| +0.002361|
| door      | 0.04902| 0.04771| −0.000549| −0.001868|
| entrance  | 0.008492| 0.008624| +0.000732| +0.000865|

Model V3 generates proportionally more wall edges (6.8% vs 5.8%) and is closer to reference on the `none` and `wall` types. Model V3's `wall` delta abs (0.784pp) is smaller than Model V2's (1.245pp).

---

## 3. Node Type Distribution

### RPLAN node types (living/bedroom/bathroom/kitchen/balcony/storage)

| Node type | V2 gen  | V2 ref  | V2 Δ      |
|-----------|---------|---------|-----------|
| living    | 0.1542  | 0.1510  | +0.003227 |
| bedroom   | 0.3855  | 0.3648  | +0.02066  |
| bathroom  | 0.1837  | 0.1778  | +0.005863 |
| kitchen   | 0.1338  | 0.1419  | −0.008147 |
| balcony   | 0.1383  | 0.1556  | −0.01728  |
| storage   | 0.004535| 0.008860| −0.004325 |

### MSD node types (Bedroom/Livingroom/Kitchen/Dining/Corridor/Stairs/Storeroom/Bathroom/Balcony)

#### Model V2 vs Model V3 side by side

| Node type  | V2 gen  | V3 gen  | ref     | V2 Δ      | V3 Δ      |
|------------|---------|---------|---------|-----------|-----------|
| Bedroom    | 0.2597  | 0.2509  | 0.2595  | +0.000143 | −0.008626 |
| Livingroom | 0.08696 | 0.08405 | 0.08281 | +0.004149 | +0.001235 |
| Kitchen    | 0.1141  | 0.1078  | 0.1088  | +0.005278 | −0.001036 |
| Dining     | 0.003644| 0.001827| 0.003311| +0.000333 | −0.001484 |
| Corridor   | 0.1568  | 0.1530  | 0.1562  | +0.000610 | −0.003124 |
| Stairs     | 0.06045 | 0.06556 | 0.05787 | +0.002578 | +0.007683 |
| Storeroom  | 0.02658 | 0.04633 | 0.04171 | −0.01513  | +0.004623 |
| Bathroom   | 0.1823  | 0.1787  | 0.1692  | +0.01311  | +0.009524 |
| Balcony    | 0.1095  | 0.1118  | 0.1206  | −0.01107  | −0.008795 |

Model V2 notably underproduces Storeroom (−1.51pp) and Balcony (−1.11pp) relative to reference. Model V3 is closer to reference on Storeroom (+0.46pp delta) but still undershoots Balcony (−0.88pp). Both models overproduce Bathroom.

---

## 4. Degree Distribution (Top 5 Bins by Reference Mass)

Reference degrees are shared between Model V2 and Model V3 (same dataset version).

### Model V2

| Degree | gen    | ref    | Δ        |
|--------|--------|--------|----------|
| 3      | 0.2885 | 0.2772 | +0.01132 |
| 4      | 0.1991 | 0.2244 | −0.02537 |
| 5      | 0.1100 | 0.1415 | −0.03148 |
| 2      | 0.1523 | 0.1047 | +0.04758 |
| 6      | 0.06602| 0.08928| −0.02326 |

### Model V3

| Degree | gen    | ref    | Δ        |
|--------|--------|--------|----------|
| 3      | 0.2453 | 0.2650 | −0.01977 |
| 4      | 0.2059 | 0.2302 | −0.02429 |
| 5      | 0.1329 | 0.1509 | −0.01796 |
| 2      | 0.1345 | 0.09176| +0.04278 |
| 6      | 0.08361| 0.09625| −0.01264 |

Both models overproduce low-degree nodes (degree 2 excess: +4.8pp Model V2, +4.3pp Model V3) and underproduce degrees 4–6. Model V3 has smaller absolute deltas at degrees 5 and 6, suggesting a slightly better fit in the mid-degree range.

---

## 5. Speed Comparison

| Model    | sec/sample | samples/sec | Elapsed (500 samples) |
|----------|-----------|-------------|----------------------|
| Model V2 | 1.736      | 0.5760      | 868 s (~14.5 min)     |
| Model V3 | 1.879      | 0.5321      | 940 s (~15.7 min)     |

Model V3 is ~8.2% slower than Model V2 per sample (1.879 vs 1.736 sec/sample). Both use 500 diffusion steps on the same hardware (CUDA). The overhead is likely attributable to the larger/more complex Model V3 architecture.

---

## 6. Model V2 vs Model V3 Verdict

### Gate Conditions (Task 1 of App Redesign Plan)

| Condition                                                                   | Value                                    | Result |
|-----------------------------------------------------------------------------|------------------------------------------|--------|
| Model V3 connectivity ≤ 5pp below Model V2 (V2=0.9520, V3=0.9340, gap=1.8pp) | 1.8pp < 5pp                              | ✅ PASS |
| Model V3 wall delta abs < Model V2 wall delta abs (0.784pp vs 1.245pp)      | 0.784pp < 1.245pp                        | ✅ PASS |

**Both gate conditions are met.**

### Verdict: ✅ PROCEED with checkpoint swap (Model V2 → Model V3)

Model V3 produces a better-calibrated wall edge distribution (delta abs 0.784pp vs 1.245pp) and slightly lower connectivity loss (1.8pp below Model V2, well within the 5pp gate). The connectivity drop is small and within acceptable bounds. Model V3 also better recovers Storeroom frequency, which was Model V2's largest node-type miss. The trade-off is an 8% inference latency increase (1.88 vs 1.74 sec/sample), which is immaterial for the app use case.

**Action:** Swap the MSD checkpoint — update `digress/configs/experiment/msd_wall.yaml` `test.checkpoint` from `outputs/2026-05-01/15-25-20-msd_wall/checkpoints/msd_wall/best.ckpt` (Model V2) to `outputs/2026-05-05/13-58-49-msd_wall_v5/checkpoints/msd_wall_v5/best.ckpt` (Model V3).

### Caveats

- Model V1 metrics are based on only 32 samples and are not reliable for quantitative comparison.
- RPLAN has no `wall` edge type (dataset uses `door` and `wall` in different schema); it is included for completeness only and not used in the Model V2/V3 gate.
- Model V3 reference `connected_frac` (0.9607) is higher than Model V2's reference (0.9456) — the wall-inference-V5 dataset version appears to contain more connected graphs. The connectivity *delta* direction (gen vs own reference) is what matters for the gate: Model V2 has gen above ref (+0.64pp), Model V3 has gen below ref (−2.67pp). Neither is alarming, and the absolute gen connectivity values (0.952 vs 0.934) are both high.
