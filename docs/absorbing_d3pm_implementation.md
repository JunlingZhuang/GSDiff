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

The first implementation focused on the core absorbing objective and
unconditional all-mask sampling. The current implementation also includes a
partial graph completion test path using anchor masks.

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

`digress/scripts/test_graph_completion.py`

Config-driven partial graph completion evaluator. It loads an absorbing
checkpoint, hides a fraction of nodes and edge slots from real test graphs,
keeps the visible subgraph fixed with anchor masks, and asks the model to fill
the unknown node/edge classes.

`digress/configs/experiment/msd_wall_absorbing_completion.yaml`

Task-tuned absorbing training config for partial graph completion. It uses a
mixed curriculum of random absorbing masks and completion-shaped masks.

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

`digress/configs/experiment/msd_wall_absorbing_v2.yaml`

Adds the `completion` config block used by `test_graph_completion.py`.

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
- `full_completion_batches`: fraction of logged batches that used full graph
  completion masks in the current epoch.
- `next_node_batches`: fraction of logged batches that used next-node masks in
  the current epoch.

### Completion-Style Masking Curriculum

The original V2 absorbing model was trained with random independent masks. That
is enough for unconditional generation, but it has train/test mismatch for the
product task: users provide a partial graph and expect the model to complete
the unknown part.

The completion-tuned config adds a task-shaped masking strategy:

```yaml
model:
  absorbing_masking:
    strategy: mixed
    eval_strategy: full_completion
    random_probability: 0.1
    full_completion_probability: 0.4
    next_node_probability: 0.5
    known_ratio_start: 0.8
    known_ratio_end: 0.5
    eval_known_ratio: 0.5
    curriculum_epochs: 120
    known_ratio_jitter: 0.1
```

Meaning:

- `strategy: mixed`: training batches are sampled from several task-shaped mask
  patterns.
- `random_probability: 0.1`: 10% of batches keep the old random absorbing
  objective as regularization.
- `full_completion_probability: 0.4`: 40% of batches hide an unknown subgraph
  and train full graph completion.
- `next_node_probability: 0.5`: 50% of batches hide one target node and all of
  its target-to-known edge classes. This directly trains "add one room and
  decide its connections".
- `known_ratio_start -> known_ratio_end`: early training gives the model easier
  full-completion partial graphs with more known nodes, then gradually hides
  more nodes.
- `eval_strategy: full_completion`: validation loss measures full completion by
  default. Override to `next_node` when selecting a checkpoint for interactive
  one-node expansion.

Completion mask shape:

```text
Known nodes: keep true room type.
Known-known edges: keep true edge class, including none.
Unknown nodes: [MASK].
Known-unknown and unknown-unknown edges: [MASK].
```

Next-node mask shape:

```text
Known graph: keep all existing node types and existing-existing edge types.
Target next node: node type = [MASK].
Target-to-known edges: edge type = [MASK].
Loss: target node type CE + target-to-known edge type CE.
edge_type=none means "do not connect"; non-none classes mean the predicted
connection type.
```

New training config:

```text
digress/configs/experiment/msd_wall_absorbing_completion.yaml
```

Training command:

```powershell
cd D:\Github\GSDiff\digress
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_absorbing_completion.yaml
```

After training, test both modes:

```powershell
.\.venv\Scripts\python.exe scripts\test_graph_generation.py msd_wall_absorbing_completion test.checkpoint=outputs/.../checkpoints/msd_wall_absorbing_completion/best.ckpt
.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_absorbing_completion completion.checkpoint=outputs/.../checkpoints/msd_wall_absorbing_completion/best.ckpt
.\.venv\Scripts\python.exe scripts\test_next_node_completion.py msd_wall_absorbing_completion next_node.checkpoint=outputs/.../checkpoints/msd_wall_absorbing_completion/best.ckpt
```

Next-node evaluator outputs:

```text
next_node_samples.json   # input graph, model next-node completion, reference
next_node_metrics.json   # target-node and target-edge metrics
next_node_grid.png       # one row per case: input / completed / reference
```

The key next-node metrics are:

- `target_node_accuracy`: whether the masked next room type is correct.
- `target_edge_accuracy_all`: edge type accuracy for target-to-known slots,
  including many `none` slots.
- `target_edge_accuracy_present`: edge type accuracy only on true non-none
  target connections.
- `connection_precision/recall/f1`: whether the model chooses the same existing
  nodes to connect to, ignoring edge type.
- `predicted_connected_to_known_frac`: how often the predicted target node has
  at least one non-none edge back to the known graph.

Initial result for the epoch-200 completion checkpoint:

```text
num_samples=32
target_node_accuracy=0.250
target_edge_accuracy_all=0.817
target_edge_accuracy_present=0.277
connection_f1=0.364
predicted_connected_to_known_frac=0.906
reference_connected_to_known_frac=1.000
avg_predicted_target_degree=3.66
avg_reference_target_degree=4.06
completed connected_frac=0.875
reference connected_frac=0.938
```

Interpretation:

```text
Next-node behavior is substantially more usable than full-completion behavior:
the target node usually reconnects to the known graph and average degree is
close to the reference. However, room type accuracy and exact true-connection
recovery are still weak, so this is not a final interaction-quality model.
```

### Multi-Checkpoint Model Plan

The current product direction uses multiple task-specific checkpoints instead
of forcing one checkpoint to be optimal for every task. This avoids needing a
task-type embedding right now and makes app behavior easier to tune.

Required checkpoints:

```text
msd_wall_absorbing_v2
  Purpose: unconditional graph generation baseline.
  Status: trained.

msd_wall_absorbing_completion
  Purpose: unified research baseline for full completion + next-node.
  Status: trained.

msd_wall_absorbing_next_node
  Purpose: app interaction, "add one room" and predict its target-to-known
  edge types.
  Status: config ready; train next.

msd_wall_absorbing_full_completion
  Purpose: one-click completion from a partial input graph to a full graph.
  Status: config ready; train in parallel only if a second GPU is available,
  otherwise run after next-node.
```

Next-node-specialized training:

```powershell
cd D:\Github\GSDiff\digress
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_absorbing_next_node.yaml
```

Next-node-specialized testing:

```powershell
.\.venv\Scripts\python.exe scripts\test_next_node_completion.py msd_wall_absorbing_next_node next_node.checkpoint=outputs/.../checkpoints/msd_wall_absorbing_next_node/best.ckpt
```

Full-completion-specialized training:

```powershell
cd D:\Github\GSDiff\digress
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_absorbing_full_completion.yaml
```

Full-completion-specialized testing:

```powershell
.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_absorbing_full_completion completion.checkpoint=outputs/.../checkpoints/msd_wall_absorbing_full_completion/best.ckpt
```

Initial specialized config choices:

```text
next_node:
  random=0.05, full_completion=0.10, next_node=0.85
  eval_strategy=next_node
  lambda_train=[1.5, 0]

full_completion:
  random=0.05, full_completion=0.90, next_node=0.05
  eval_strategy=full_completion
  lambda_train=[2.0, 0]
  known_ratio_start=0.7, known_ratio_end=0.5, curriculum_epochs=80
```

Why no task conditioning here:

```text
Task conditioning is most useful when one unified checkpoint must handle many
tasks equally well. With task-specific checkpoints, the task is already defined
by the checkpoint and the training distribution, so task conditioning is not
the highest-priority change.
```

## Partial Graph Completion

Completion is an inference-only mode on the same absorbing checkpoint. It does
not require a separate model architecture.

Mechanism:

- Known node types are passed in as normal class IDs.
- Known edge slots are passed in as normal edge class IDs, including class `0`
  for known non-edges.
- Unknown valid node/edge slots are replaced with the internal `[MASK]` class.
- Anchor masks mark known slots. During MaskGIT-style sampling, anchor slots
  are visible to the network but are never selected for re-sampling.

Current evaluator:

```powershell
cd D:\Github\GSDiff\digress
.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_absorbing_v2 completion.checkpoint=outputs/2026-05-14/14-47-40-msd_wall_absorbing_v2/checkpoints/msd_wall_absorbing_v2/best.ckpt
```

Useful overrides:

```powershell
completion.num_samples=64
completion.known_ratio=0.5
completion.grid_samples=12
completion.out_dir=outputs/msd_wall_completion_eval
model.edge_none_logit_bias=-0.05
```

Outputs:

```text
completion_samples.json   # partial input, model completion, reference graph
completion_metrics.json   # accuracy, graph statistics, distribution deltas
completion_grid.png       # one row per case: partial / completed / reference
```

Important limitation:

```text
The current completion script fixes the target node count from the reference
test graph. It evaluates type/edge completion quality, not automatic graph-size
growth from an arbitrary user sketch.
```

Primary metrics:

- `node_unknown_accuracy`: node type accuracy only on hidden nodes.
- `edge_masked_accuracy_all`: edge type accuracy on all hidden edge slots,
  including the many `none` slots.
- `edge_masked_accuracy_present`: edge type accuracy only where the reference
  hidden edge is a real edge. This is stricter and more useful for checking
  whether the model recovers wall/door/passage/entrance edges.

Initial smoke result on the V2 unconditional checkpoint:

```text
num_samples=8
known_ratio=0.5
node_unknown_accuracy=0.1513
edge_masked_accuracy_all=0.8038
edge_masked_accuracy_present=0.0754
completed avg_edges=62.00
reference avg_edges=60.88
```

Interpretation:

```text
The unconditional V2 checkpoint can preserve graph-level density during
completion, but it is weak at recovering the exact hidden room types and exact
real edge classes. For serious partial-completion quality, the next training
run should use a completion-style masking curriculum instead of only random
independent absorbing masks.
```

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
- Anchor-aware partial graph completion sampling.
- Completion-style masking curriculum for training.
- Test script/backend loading compatibility.

Not implemented yet:

- Single-node attribute prediction endpoint.
- Graphormer-style centrality encoding.
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

## Experiment Log: Absorbing V2

Run directory:

```text
digress/outputs/2026-05-14/14-47-40-msd_wall_absorbing_v2/
```

Best checkpoint used for test:

```text
digress/outputs/2026-05-14/14-47-40-msd_wall_absorbing_v2/checkpoints/msd_wall_absorbing_v2/best.ckpt
```

Test output:

```text
digress/outputs/2026-05-14/14-47-40-msd_wall_absorbing_v2/test_samples/
```

### V2 Training Results

V2 was stopped around epoch 146 after validation stopped improving. The best
validation checkpoint was epoch 100.

| Epoch | Split | Total masked loss | Node CE | Edge CE | Notes |
|---:|---|---:|---:|---:|---|
| 10 | val | 2.099 | 1.613 | 0.486 | First validation |
| 50 | val | 1.997 | 1.530 | 0.466 | Clear improvement |
| 80 | val | 1.963 | 1.493 | 0.470 | Still improving |
| 100 | val | 1.932 | 1.473 | 0.459 | Best checkpoint |
| 140 | val | 2.007 | 1.545 | 0.462 | Worse than best |

Compared with V1, V2 reaches a similar or better node/edge CE much earlier,
while using mask-aware structural features. This confirms that restoring
structural features did not break absorbing training.

### V2 Generation Test Results

Generated 32 samples on CUDA.

| Metric | V1 Generated | V2 Generated | Reference MSD-wall | V2 Delta |
|---|---:|---:|---:|---:|
| Average nodes | 26.63 | 26.34 | 27.60 | -1.26 |
| Average edges | 67.09 | 42.88 | 58.10 | -15.22 |
| Connected fraction | 0.969 | 0.688 | 0.946 | -0.258 |
| Seconds per sample | 0.050 | 0.061 | n/a | n/a |

Node distribution deltas:

| Node type | Generated | Reference | Delta |
|---|---:|---:|---:|
| Bedroom | 0.2835 | 0.2595 | +0.0240 |
| Livingroom | 0.0676 | 0.0828 | -0.0152 |
| Kitchen | 0.1246 | 0.1088 | +0.0157 |
| Dining | 0.0083 | 0.0033 | +0.0050 |
| Corridor | 0.1732 | 0.1562 | +0.0170 |
| Stairs | 0.0558 | 0.0579 | -0.0021 |
| Storeroom | 0.0214 | 0.0417 | -0.0204 |
| Bathroom | 0.1815 | 0.1692 | +0.0123 |
| Balcony | 0.0842 | 0.1206 | -0.0364 |

Edge distribution deltas:

| Edge type | Generated | Reference | Delta |
|---|---:|---:|---:|
| none | 0.8856 | 0.8639 | +0.0217 |
| wall | 0.0518 | 0.0704 | -0.0187 |
| passage | 0.0076 | 0.0083 | -0.0007 |
| door | 0.0458 | 0.0496 | -0.0038 |
| entrance | 0.0093 | 0.0078 | +0.0015 |

### V2 Verdict

V2 fixed one problem but over-corrected another.

What improved:

- Edge type proportions are much closer to reference than V1.
- `door` and `passage` are now well calibrated.
- `none` is no longer underproduced.
- Training loss is better than V1 and uses mask-aware structural features.

What got worse:

- Generated graphs are now too sparse: 42.88 edges vs 58.10 reference.
- Connectivity drops badly: 0.688 vs 0.946 reference.
- The sampling-only `edge_none_logit_bias=0.4` is too strong.
- Lowering `lambda_train` from 2 to 1 may also reduce edge reconstruction
  pressure too much.

Conclusion:

```text
V1 was too dense.
V2 is too sparse.
The next version should keep mask-aware structural features but reduce the
none-edge correction.
```

Recommended V3 direction:

- Keep `extra_features: all`.
- Keep mask-aware adjacency.
- Set `edge_none_logit_bias` to `0.0` or at most `0.15`.
- Consider restoring `lambda_train` to `[2, 0]`, or try `[1.5, 0]`.
- Re-test generation density before training a full long run.

### V2 Sampling Calibration Follow-Up

After the initial V2 test, we found that command-line overrides such as
`model.edge_none_logit_bias=0.0` were not being applied after loading a
checkpoint. Lightning restores the training-time `cfg` stored inside the
checkpoint, so the test script and backend loader now re-apply sampling-only
fields after `load_from_checkpoint`.

Changed files:

```text
digress/scripts/test_graph_generation.py
app/backend/app/services/graph_generation.py
```

The corrected sweep used the epoch-200 V2 `best.ckpt`.

| edge_none_logit_bias | Samples | Avg nodes | Avg edges | Connected frac | Interpretation |
|---:|---:|---:|---:|---:|---|
| 0.0 | 128 | 25.71 | 51.91 | 0.883 | Too sparse |
| -0.2 | 128 | 29.41 | 66.87 | 0.977 | Too dense |
| -0.1 | 128 | 27.91 | 60.62 | 0.930 | Close |
| -0.075 | 256 | 29.89 | 64.25 | 0.941 | Too dense |
| -0.05 | 256 | 28.46 | 59.70 | 0.926 | Best current calibration |

Reference:

```text
avg_nodes=27.60
avg_edges=58.10
connected_frac=0.946
test split connected_frac=0.920
```

Selected default:

```yaml
model:
  lambda_train: [1, 0]
  edge_none_logit_bias: -0.05
```

Why keep `lambda_train: [1, 0]` for now:

- V2 with mask-aware structural features reached better validation loss than V1.
- With calibrated sampling (`edge_none_logit_bias=-0.05`), average edge count
  is close to reference: 59.70 vs 58.10.
- Edge type distribution is also close:

| Edge type | Generated | Reference | Delta |
|---|---:|---:|---:|
| none | 0.8702 | 0.8639 | +0.0063 |
| wall | 0.0636 | 0.0704 | -0.0068 |
| passage | 0.0062 | 0.0083 | -0.0021 |
| door | 0.0518 | 0.0496 | +0.0022 |
| entrance | 0.0081 | 0.0078 | +0.0004 |

Current recommendation:

```text
Do not retrain immediately just to change lambda_train.
Use V2 best checkpoint with edge_none_logit_bias=-0.05 as the current best
absorbing unconditional baseline.
```

If future partial-completion tests show missing edges or disconnected
completion, then train V3 with `lambda_train: [1.5, 0]` while keeping
mask-aware structural features.

Calibration rule for future models/datasets:

```text
Train-time hyperparameters such as lambda_train must be selected per dataset
and usually require retraining.

Sampling-only calibration such as edge_none_logit_bias must be selected per
checkpoint/dataset pair, because every checkpoint can have a different edge
density bias.
```
