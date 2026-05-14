# Absorbing D3PM Implementation Notes

This document explains the first absorbing-state D3PM refactor for MSD bubble
graph generation. It is written as a practical reference for readers who are
new to diffusion models.

## Goal

The existing DiGress pipeline already supports unconditional graph generation:

```text
random noisy graph -> denoised full graph
```

The absorbing refactor changes the training objective to:

```text
partially masked graph -> recover the masked node/edge classes
```

This is the foundation for three tasks with one model:

- Unconditional generation: all valid node/edge slots start as `[MASK]`.
- Partial graph completion: known slots stay fixed, unknown slots are `[MASK]`.
- Node attribute prediction: the target node type is `[MASK]`, edges stay known.

The first implementation focuses on the core absorbing objective and
unconditional all-mask sampling. Partial completion and node-type prediction can
be added on top of the same model with anchor masks.

## Compatibility

The old DiGress training path is preserved.

Use the original model:

```powershell
cd D:\Github\GSDiff\digress
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall.yaml
```

Use the absorbing model:

```powershell
cd D:\Github\GSDiff\digress
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_absorbing.yaml
```

The switch is config-driven:

```yaml
model:
  transition: absorbing
```

Any config with `model.transition: marginal` continues to use the original
vanilla DiGress class.

## New Files

`digress/src/diffusion/absorbing_transition.py`

Defines `AbsorbingGraphTransition`. This builds transition matrices that send
every node/edge class toward one special `[MASK]` class. In plain language:
once a clean class becomes `[MASK]` during noising, it remains unknown.

`digress/src/diffusion/absorbing_utils.py`

Contains small helper functions for the absorbing path:

- Detect whether a config requested `transition=absorbing`.
- Expand model input/output dimensions with one internal `[MASK]` class.
- Pad clean one-hot tensors with a zero mask column.
- Strip accidental final `[MASK]` predictions before visualization/API output.

`digress/src/diffusion_model_absorbing.py`

Defines `AbsorbingDenoisingDiffusion`, a separate LightningModule subclass. It
keeps the DiGress GraphTransformer backbone but changes:

- Forward noising: replace random node/edge positions with `[MASK]`.
- Training loss: compute cross entropy only at masked positions.
- Validation metric: report masked validation CE as `val_epoch_NLL` for
  checkpoint compatibility.
- Sampling: start from all `[MASK]` and fill positions progressively.

`digress/configs/experiment/msd_wall_absorbing.yaml`

Training/test config for the first MSD-wall absorbing model. It intentionally
sets `model.extra_features: null` because the old cycle/spectral features treat
all non-zero edge classes as observed edges. With a `[MASK]` edge class, that
would incorrectly treat unknown edges as real graph structure.

## Changed Files

`digress/src/main.py`

Adds config-driven model selection:

```text
transition=marginal  -> DiscreteDenoisingDiffusion
transition=absorbing -> AbsorbingDenoisingDiffusion
```

It also expands dataset dimensions for absorbing runs after normal DiGress
dimension inference.

`digress/src/diffusion_model_discrete.py`

Adds an `absorbing` transition branch in the base initializer. This branch is
only used by absorbing configs and does not affect old marginal configs.

`digress/scripts/test_graph_generation.py`

Adds absorbing checkpoint loading support. The same config-driven test script
can now load either vanilla or absorbing DiGress checkpoints.

`app/backend/app/services/graph_generation.py`

Adds absorbing checkpoint loading support for the backend graph generation
service. Existing `msd_wall` and `rplan` API behavior is unchanged.

## Training Behavior

For each batch:

1. The preprocessed graph provides clean room types and edge types.
2. The model appends one extra internal class, `[MASK]`.
3. A random diffusion step `t` is sampled.
4. Nodes and edges are independently replaced by `[MASK]` according to the
   absorbing schedule.
5. The network predicts original classes.
6. Loss is computed only where the input was `[MASK]`.

The logged training values mean:

- `train_masked_loss`: weighted absorbing loss.
- `masked_node_CE`: cross entropy for masked node types only.
- `masked_edge_CE`: cross entropy for masked edge types only.
- `avg_masked_nodes`: average number of masked node positions per batch.
- `avg_masked_edges`: average number of masked edge positions per batch.

## Test Command

After training, test generation with:

```powershell
cd D:\Github\GSDiff\digress
.\.venv\Scripts\python.exe scripts\test_graph_generation.py msd_wall_absorbing
```

If multiple checkpoints exist, the script searches for the latest checkpoint
under `outputs/**/checkpoints/msd_wall_absorbing/`. You can still override:

```powershell
.\.venv\Scripts\python.exe scripts\test_graph_generation.py msd_wall_absorbing test.checkpoint=outputs/.../best.ckpt
```

## Current Scope

Implemented now:

- Config-driven absorbing model path.
- Internal `[MASK]` vocabulary expansion.
- Absorbing transition.
- Masked-position training/validation loss.
- All-mask unconditional sampling.
- Test script/backend loading compatibility.

Not implemented yet:

- Anchor-aware partial graph completion.
- Single-node attribute prediction endpoint.
- Mask-aware cycle/spectral/centrality features.
- True MaskGIT confidence schedule tuning.

Those are the next layers after the absorbing training objective is validated.

## Experiment Log: Absorbing MVP V1

Run directory:

```text
digress/outputs/2026-05-13/16-55-13-msd_wall_absorbing/
```

Best checkpoint:

```text
digress/outputs/2026-05-13/16-55-13-msd_wall_absorbing/checkpoints/msd_wall_absorbing/best.ckpt
```

Training command:

```powershell
cd D:\Github\GSDiff\digress
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_absorbing.yaml
```

Test command:

```powershell
.\.venv\Scripts\python.exe scripts\test_graph_generation.py msd_wall_absorbing test.checkpoint=outputs/2026-05-13/16-55-13-msd_wall_absorbing/checkpoints/msd_wall_absorbing/best.ckpt
```

### Training Results

The model trained for 200 epochs. The best validation checkpoint was saved at
epoch 170.

| Epoch | Split | Total masked loss | Node CE | Edge CE | Notes |
|---:|---|---:|---:|---:|---|
| 1 | train | 3.126 | 1.966 | 0.580 | Initial learning point |
| 10 | val | 2.633 | 1.627 | 0.503 | First validation |
| 90 | val | 2.446 | 1.486 | 0.480 | Earlier best region |
| 170 | val | 2.394 | 1.469 | 0.463 | Best checkpoint |
| 200 | val | 2.471 | 1.515 | 0.478 | Final epoch, worse than best |

Interpretation:

- The absorbing objective learns: both node CE and edge CE are below random
  guessing baselines.
- Node type prediction is still the harder part.
- Validation improves until around epoch 170, then becomes worse. The final
  checkpoint should not be used over `best.ckpt`.

Reference random CE levels:

```text
node random baseline: ln(9 room types)  = 2.197
edge random baseline: ln(5 edge types)  = 1.609
```

### Generation Test Results

Test output:

```text
digress/outputs/2026-05-13/16-55-13-msd_wall_absorbing/test_samples/
```

Generated 32 samples on CUDA.

| Metric | Generated | Reference MSD-wall | Delta |
|---|---:|---:|---:|
| Average nodes | 26.63 | 27.60 | -0.98 |
| Average edges | 67.09 | 58.10 | +9.00 |
| Connected fraction | 0.969 | 0.946 | +0.023 |
| Seconds per sample | 0.050 | n/a | n/a |

Node distribution deltas:

| Node type | Generated | Reference | Delta |
|---|---:|---:|---:|
| Bedroom | 0.2805 | 0.2595 | +0.0210 |
| Livingroom | 0.0833 | 0.0828 | +0.0005 |
| Kitchen | 0.0915 | 0.1088 | -0.0173 |
| Dining | 0.0012 | 0.0033 | -0.0021 |
| Corridor | 0.1338 | 0.1562 | -0.0224 |
| Stairs | 0.0528 | 0.0579 | -0.0051 |
| Storeroom | 0.0246 | 0.0417 | -0.0171 |
| Bathroom | 0.1854 | 0.1692 | +0.0163 |
| Balcony | 0.1467 | 0.1206 | +0.0261 |

Edge distribution deltas:

| Edge type | Generated | Reference | Delta |
|---|---:|---:|---:|
| none | 0.8313 | 0.8639 | -0.0326 |
| wall | 0.0885 | 0.0704 | +0.0181 |
| passage | 0.0068 | 0.0083 | -0.0015 |
| door | 0.0663 | 0.0496 | +0.0167 |
| entrance | 0.0070 | 0.0078 | -0.0008 |

### V1 Verdict

Absorbing MVP V1 is a successful sanity check, but it is not the final model.

What worked:

- The code path trains end to end.
- The model learns a non-random masked reconstruction objective.
- The sampler produces connected graphs with reasonable node counts.
- The same test script can load either vanilla or absorbing checkpoints.

What failed or needs improvement:

- Generated graphs are too dense: about 9 extra edges per graph.
- `none` edge is underproduced, while `wall` and `door` are overproduced.
- Some node classes are biased: `Balcony`, `Bedroom`, and `Bathroom` are high;
  `Corridor`, `Kitchen`, and `Storeroom` are low.
- V1 disables DiGress structural features, so the model loses useful graph
  topology signal.

## V2 Architecture Adjustment

V2 keeps the absorbing objective and changes the parts that caused dense graphs.

Implemented V2 changes:

- Add mask-aware structural features.
- Restore cycle/spectral features using only observed real edges.
- Exclude `[MASK]` edges from the adjacency matrix.
- Reduce or recalibrate edge pressure so the model predicts more `none` edges.
- Add a sampling-only `none` edge logit bias.

New config:

```text
digress/configs/experiment/msd_wall_absorbing_v2.yaml
```

Training command:

```powershell
cd D:\Github\GSDiff\digress
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_absorbing_v2.yaml
```

Mask-aware adjacency rule:

```python
observed_real_edge = edge_type != none and edge_type != mask
```

This matters because the original DiGress feature code treats every non-zero
edge class as a real edge. In absorbing D3PM, `[MASK]` is also non-zero, but it
means "unknown", not "connected". If `[MASK]` is counted as an edge, cycle and
spectral features become polluted by fake structure.

V2 success target:

```text
Average generated edges should move from 67.09 toward the reference 58.10.
The edge none/wall/door distribution should become closer to reference.
Node distribution should not regress substantially.
```

V2 changed files:

```text
digress/src/diffusion/extra_features.py
digress/src/diffusion_model_absorbing.py
digress/configs/experiment/msd_wall_absorbing_v2.yaml
```

Code behavior:

- `extra_features.py` now uses `observed_adjacency_from_noisy_data`.
- Vanilla DiGress behavior is unchanged because normal configs do not pass a
  `mask_idx_E`.
- Absorbing runs pass `mask_idx_E` in `noisy_data`, so structural features can
  distinguish real edges from unknown `[MASK]` edges.
- `edge_none_logit_bias` only affects sampling. It does not change the training
  loss or the checkpoint weights.

Initial smoke checks:

```text
py_compile: passed
CPU fast_dev_run: passed
untrained sample_batch smoke: passed, no final [MASK] tokens
```
