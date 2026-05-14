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
