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

This is the foundation for graph-only masked prediction. Earlier experiments
tried to make one checkpoint handle all downstream graph tasks equally well,
but the current direction is more specific:

- Keep `msd_wall_absorbing_v2` as the unconditional graph generation baseline.
- Use task-shaped absorbing checkpoints for app-facing graph completion.
- Prefer Graph Policy V2 for the app: one policy step predicts one next room
  and its target-to-current edges; full completion is obtained by repeatedly
  applying that policy.

The first implementation focused on the core absorbing objective and
unconditional all-mask sampling. The current implementation also includes a
partial graph completion test path using anchor masks and an autoregressive
Graph Policy V2 path.

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

`digress/configs/experiment/msd_wall_graph_policy_v2.yaml`

Current recommended app-facing graph policy config. It trains one step:

```text
current partial graph + target summary condition + one masked next node
-> next room type + next-node-to-current edge types
```

Full graph completion should use this policy autoregressively instead of a
single one-shot full-completion denoising pass.

`digress/scripts/test_autoregressive_completion.py`

Config-driven evaluator that repeatedly applies a next-node / graph-policy
checkpoint until the target graph reaches the reference node count. This is the
diagnostic path for app-style full completion.

## Changed Files

`digress/src/main.py`

Adds config-driven model selection:

```text
transition=marginal  -> DiscreteDenoisingDiffusion
transition=absorbing -> AbsorbingDenoisingDiffusion
```

It also expands dataset dimensions for absorbing runs after normal DiGress
dimension inference.

`digress/src/diffusion/absorbing_utils.py`

Adds optional graph-level condition dimensions to absorbing configs. Existing
absorbing configs keep the old dimensions unless `model.graph_condition.enabled`
is true.

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
- `graph_policy_batches`: fraction of logged batches that used Graph Policy V2
  masks in the current epoch.

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
  Status: trained; useful diagnostic, but superseded by Graph Policy V2 because
  it lacks target-size / remaining-room context.

msd_wall_absorbing_full_completion
  Purpose: one-click completion from a partial input graph to a full graph.
  Status: trained; current checkpoint is not good enough because it
  over-connects completed graphs.

msd_wall_graph_policy_v2
  Purpose: current recommended app-facing policy. One step predicts one next
  room and its target-to-current edges; full completion loops this policy.
  Status: code ready; 1-epoch smoke passed; needs full 200-epoch training.
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

Full-completion-specialized result, 2026-05-19:

```text
Status: trained, but not good enough for app use.

Run:
  outputs/2026-05-18/15-59-34-msd_wall_absorbing_full_completion

Checkpoint:
  outputs/2026-05-18/15-59-34-msd_wall_absorbing_full_completion/checkpoints/msd_wall_absorbing_full_completion/best.ckpt

Training:
  completed 200 epochs
  best validation at epoch 180
  train_loss: 3.9423 at epoch 1 -> 3.4734 at epoch 200
  train node CE: 1.9729 -> 1.7736
  train edge CE: 0.9847 -> 0.8499
  val_nll: 3.6716 at epoch 10 -> best 3.5730 at epoch 180 -> 3.5761 at epoch 200

Evaluation command:
  .\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_absorbing_full_completion completion.checkpoint=outputs/2026-05-18/15-59-34-msd_wall_absorbing_full_completion/checkpoints/msd_wall_absorbing_full_completion/best.ckpt completion.out_dir=outputs/2026-05-18/15-59-34-msd_wall_absorbing_full_completion/completion_eval_best completion.num_samples=128 completion.grid_samples=12

Evaluation output:
  outputs/2026-05-18/15-59-34-msd_wall_absorbing_full_completion/completion_eval_best/completion_metrics.json
  outputs/2026-05-18/15-59-34-msd_wall_absorbing_full_completion/completion_eval_best/completion_grid.png

Metrics:
  node_unknown_accuracy: 0.1797
  edge_masked_accuracy_all: 0.6911
  edge_masked_accuracy_present: 0.1444
  completed avg_edges: 105.16
  reference avg_edges: 62.44
  completed connected_frac: 1.000
  reference connected_frac: 0.930
  inference: 38.61 s / 128 samples, 0.302 s per sample, 16 MaskGIT steps

Weakness:
  The model is heavily over-connected. Connected fraction is high only because
  it predicts too many edges. This checkpoint should not be treated as the
  app-facing full-completion model.

Next:
  Calibrate full-completion inference with a positive edge_none_logit_bias
  sweep, analogous to next-node. If inference calibration cannot reduce
  avg_edges near the reference without collapsing edge-present accuracy, train
  a new full-completion checkpoint with stronger none/present balance or a
  degree/density regularizer.
```

Autoregressive full-completion smoke test, 2026-05-19:

```text
Status: implemented and tested, but not yet good enough.

Idea:
  Use the next-node checkpoint as a policy. Start from the known partial graph,
  add one masked target node, sample node type and target-to-known edges, append
  it to the graph, then repeat until the reference node count is reached.

Script:
  scripts/test_autoregressive_completion.py

Command:
  .\.venv\Scripts\python.exe scripts\test_autoregressive_completion.py msd_wall_absorbing_next_node autoregressive.checkpoint=outputs/2026-05-18/13-38-27-msd_wall_absorbing_next_node/checkpoints/msd_wall_absorbing_next_node/best.ckpt autoregressive.out_dir=outputs/2026-05-18/13-38-27-msd_wall_absorbing_next_node/autoregressive_completion_smoke_32 autoregressive.num_samples=32 autoregressive.grid_samples=8 autoregressive.known_ratio=0.5 model.edge_none_logit_bias=0.4

Output:
  outputs/2026-05-18/13-38-27-msd_wall_absorbing_next_node/autoregressive_completion_smoke_32/autoregressive_completion_metrics.json
  outputs/2026-05-18/13-38-27-msd_wall_absorbing_next_node/autoregressive_completion_smoke_32/autoregressive_completion_grid.png

Metrics:
  num_samples: 32
  node_unknown_accuracy: 0.2179
  edge_masked_accuracy_all: 0.7523
  edge_masked_accuracy_present: 0.1061
  edge_presence_f1: 0.1859
  typed_edge_f1: 0.0906
  completed avg_edges: 77.12
  reference avg_edges: 61.50
  completed connected_frac: 0.938
  reference connected_frac: 0.938

Weakness:
  This is not materially better than calibrated one-shot full completion. It
  controls connectedness and edge density better than the uncalibrated full
  checkpoint, but exact GT edge recovery is still weak.

Important limitation:
  The current evaluator uses a random hidden reference-node order so direct
  GT-aligned metrics are defined. In the app, the order would be user-driven
  or sampled/reranked. This means the smoke test is an early diagnostic, not a
  final paper-quality autoregressive evaluation.

Next:
  Do not rely on naive iterative inference alone. Train an autoregressive-style
  checkpoint explicitly, or add top-k candidate sampling with reranking by
  degree prior, edge-density prior, and optional retrieved-neighbor priors.
```

### Graph Policy V2: Current Direction

Decision, 2026-05-19:

```text
Use Graph Policy V2 as the next training target.

This is not a one-shot full-completion model. It is an autoregressive graph
policy: every step predicts exactly one next room and its connections to the
current graph. Full completion is produced by running the same policy in a
loop until the requested target node count is reached.
```

Why this is the chosen direction:

- One-shot full completion is too ambiguous and the trained checkpoint
  over-connected badly.
- The old next-node checkpoint was more app-relevant, but it did not know the
  final target graph size or remaining room inventory.
- Graph Policy V2 adds that missing global context while keeping the task
  simple enough: predict one node and its target-to-current edges.
- The same checkpoint supports both app flows: one-step "add room" and
  multi-step full graph completion.

Model input:

```text
current partial graph
+ graph-level target summary
+ one masked target node
```

Graph-level target summary:

```text
[current_known_count, target_total_count, remaining_count] / max_nodes
+ optional remaining room-type inventory histogram
```

Model output:

```text
target room type
target-to-current edge presence/type
```

Important limitation:

```text
This is still graph-only. It does not predict room geometry, room placement, or
floorplan polygons. Geometry-conditioned graph policy should be a separate
experiment because it changes the data schema and model inputs.
```

Changed / added files for Graph Policy V2:

```text
digress/configs/experiment/msd_wall_graph_policy_v2.yaml
digress/src/diffusion/absorbing_utils.py
digress/src/diffusion_model_absorbing.py
digress/src/main.py
digress/scripts/test_next_node_completion.py
digress/scripts/test_autoregressive_completion.py
digress/scripts/test_graph_generation.py
```

Backend note:

```text
app/backend/app/services/graph_generation.py also uses the updated absorbing
dimension helper so graph-condition-aware checkpoints can be loaded later, but
the app endpoint still needs separate wiring for Graph Policy V2 inference.
```

Architecture changes:

- Optional graph condition is appended to DiGress global `y` features only when
  `model.graph_condition.enabled: true`.
- `graph_policy` masking samples a current known subgraph, selects one hidden
  target node, and treats all other future nodes as inactive padding for that
  training step.
- The edge loss can use `split_presence_type`, which separates edge existence,
  edge class, and edge-density calibration instead of using one CE over all
  edge classes.
- Existing absorbing configs keep the old behavior unless they opt into graph
  condition or split edge loss.

Graph Policy V2 config:

```yaml
general:
  name: msd_wall_graph_policy_v2
train:
  n_epochs: 200
  batch_size: 8
model:
  transition: absorbing
  lambda_train: [2.0, 0]
  edge_loss_mode: split_presence_type
  edge_density_loss_weight: 0.5
  graph_condition:
    enabled: true
    room_type_inventory: true
  absorbing_masking:
    strategy: mixed
    eval_strategy: graph_policy
    random_probability: 0.02
    next_node_probability: 0.08
    graph_policy_probability: 0.90
```

Training command:

```powershell
cd D:\Github\GSDiff\digress
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_graph_policy_v2.yaml
```

Next-node evaluation command after training:

```powershell
.\.venv\Scripts\python.exe scripts\test_next_node_completion.py msd_wall_graph_policy_v2 next_node.checkpoint=outputs/.../checkpoints/msd_wall_graph_policy_v2/best.ckpt next_node.num_samples=128 next_node.grid_samples=12
```

Autoregressive full-completion evaluation command after training:

```powershell
.\.venv\Scripts\python.exe scripts\test_autoregressive_completion.py msd_wall_graph_policy_v2 autoregressive.checkpoint=outputs/.../checkpoints/msd_wall_graph_policy_v2/best.ckpt autoregressive.num_samples=128 autoregressive.grid_samples=12 autoregressive.known_ratio=0.5
```

Smoke result, 2026-05-19:

```text
Status: code ready for full training.

Command:
  .\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_graph_policy_v2.yaml train.n_epochs=1 train.batch_size=2 general.name=debug_graph_policy_v2 general.samples_to_generate=0 general.final_model_samples_to_generate=0 general.sample_every_val=1000

Result:
  py_compile passed.
  1-epoch training smoke completed.
  test/absorbing_masked_loss: 4.7809

Weakness:
  This is only a functionality smoke test, not evidence of model quality.
  The checkpoint still needs a real 200-epoch run and task-specific evaluation.

Next:
  Train `msd_wall_graph_policy_v2` for 200 epochs, then evaluate both
  next-node and autoregressive full completion with the commands above.
```

Graph Policy V2 first long-run observation, 2026-05-19:

```text
Status: training started, but the first config is not healthy enough to keep as
the default recommendation.

Run:
  outputs/2026-05-19/13-18-38-msd_wall_graph_policy_v2

Command:
  .\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_graph_policy_v2.yaml

Observed validation:
  epoch 10: val_masked_loss=4.5677, node=1.9316, edge=1.3181
  epoch 20: val_masked_loss=4.4042, node=1.8723, edge=1.2660
  epoch 30: val_masked_loss=4.2371, node=1.8033, edge=1.2169
  epoch 40: val_masked_loss=4.2102, node=1.7737, edge=1.2183  best so far
  epoch 50: val_masked_loss=4.2191, node=1.7628, edge=1.2282
  epoch 60: val_masked_loss=4.2512, node=1.7395, edge=1.2559

Interpretation:
  Node validation keeps improving, but edge validation starts worsening after
  epoch 30-40. The total validation loss therefore plateaus and then regresses.
  This is not a runtime crash; it is a training/config issue.

Likely cause:
  The curriculum moves from known_ratio=0.85 toward 0.30 over 140 epochs. For
  app-style next-room completion, that becomes too sparse too early. The model
  is learning room type distribution but losing edge stability. The current
  history also only records the combined edge loss, so it is hard to tell
  whether presence, type, or density is the main failure.

Code update:
  `digress/src/diffusion_model_absorbing.py` now records edge loss components:
  `train_e_presence_ce`, `train_e_type_ce`, `train_e_density`,
  `val_e_presence_ce`, `val_e_type_ce`, `val_e_density`.

New config:
  `digress/configs/experiment/msd_wall_graph_policy_v2_stable.yaml`

Changes:
  lr: 1e-4 -> 5e-5
  known_ratio_end: 0.30 -> 0.50
  curriculum_epochs: 140 -> 80
  known_ratio_jitter: 0.20 -> 0.10
  edge_density_loss_weight: 0.5 -> 0.2
  check_val_every_n_epochs: 10 -> 5

Next:
  Prefer starting a new run with `msd_wall_graph_policy_v2_stable.yaml` if the
  first run continues to regress. Keep the best checkpoint from the first run
  as a diagnostic only, not as the app-facing checkpoint.
```

Graph Policy V2 stable run observation, 2026-05-19:

```text
Status: better than the first Graph Policy V2 run, but still plateauing.

Run:
  outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable

Checkpoint:
  outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable/checkpoints/msd_wall_graph_policy_v2_stable/best.ckpt

Observed validation:
  epoch 25:  val_masked_loss=4.3087, node=1.7963, edge=1.2562
  epoch 35:  val_masked_loss=4.1407, node=1.7520, edge=1.1944
  epoch 65:  val_masked_loss=3.9958, node=1.6784, edge=1.1587
  epoch 95:  val_masked_loss=3.9914, node=1.6830, edge=1.1542
  epoch 125: val_masked_loss=3.9770, node=1.6756, edge=1.1507  best so far
  epoch 155: val_masked_loss=3.9898, node=1.7050, edge=1.1424

Interpretation:
  The stable config improved validation from the first run: best val moved
  from about 4.2102 to 3.9770. The train loss staying around 3.7-3.8 is not by
  itself a failure because this objective is node_CE + 2 * edge_loss. However,
  after epoch 125 the validation score has not made a new best for about 30
  epochs.

Weakness:
  Edge validation is still around 1.14-1.15 and node validation is around
  1.68-1.70. This indicates the graph policy is learning but remains
  uncertain, especially for exact next-room type and edge targets.

Next:
  If no new best appears by epoch 170-180, stop the run and evaluate the
  epoch-125 `best.ckpt` with next-node and autoregressive completion tests.
  Do not judge this checkpoint by training loss alone.
```

Graph Policy V2 stable evaluation, 2026-05-19:

```text
Status: next-node is promising; autoregressive full completion is not good
enough yet.

Checkpoint:
  outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable/checkpoints/msd_wall_graph_policy_v2_stable/best.ckpt

Next-node command:
  .\.venv\Scripts\python.exe scripts\test_next_node_completion.py msd_wall_graph_policy_v2_stable next_node.checkpoint=outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable/checkpoints/msd_wall_graph_policy_v2_stable/best.ckpt next_node.num_samples=128 next_node.grid_samples=12

Next-node outputs:
  outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable/next_node_eval_best_128/next_node_metrics.json
  outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable/next_node_eval_best_128/next_node_grid.png

Next-node edge_none_logit_bias sweep:
  bias +0.4: node_acc=0.7031, edge_present_acc=0.1719, conn_f1=0.2691, degree=2.49 vs 3.95
  bias  0.0: node_acc=0.7031, edge_present_acc=0.1996, conn_f1=0.2903, degree=3.37 vs 3.95
  bias -0.2: node_acc=0.7031, edge_present_acc=0.2628, conn_f1=0.3482, degree=3.95 vs 3.95
  bias -0.4: node_acc=0.6953, edge_present_acc=0.2708, conn_f1=0.3445, degree=4.44 vs 3.95

Next-node recommendation:
  Use `model.edge_none_logit_bias=-0.2` for this checkpoint. It gives the best
  connection F1 and matches the reference target degree almost exactly.

Autoregressive commands:
  .\.venv\Scripts\python.exe scripts\test_autoregressive_completion.py msd_wall_graph_policy_v2_stable autoregressive.checkpoint=outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable/checkpoints/msd_wall_graph_policy_v2_stable/best.ckpt autoregressive.num_samples=64 autoregressive.grid_samples=12 autoregressive.known_ratio=0.5
  .\.venv\Scripts\python.exe scripts\test_autoregressive_completion.py msd_wall_graph_policy_v2_stable autoregressive.checkpoint=outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable/checkpoints/msd_wall_graph_policy_v2_stable/best.ckpt autoregressive.num_samples=64 autoregressive.grid_samples=12 autoregressive.known_ratio=0.5 model.edge_none_logit_bias=-0.2
  .\.venv\Scripts\python.exe scripts\test_autoregressive_completion.py msd_wall_graph_policy_v2_stable autoregressive.checkpoint=outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable/checkpoints/msd_wall_graph_policy_v2_stable/best.ckpt autoregressive.num_samples=64 autoregressive.grid_samples=12 autoregressive.known_ratio=0.5 model.edge_none_logit_bias=-0.4

Autoregressive outputs:
  outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable/autoregressive_eval_best_64/autoregressive_completion_metrics.json
  outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable/autoregressive_eval_bias_m0p2_64/autoregressive_completion_metrics.json
  outputs/2026-05-19/13-57-07-msd_wall_graph_policy_v2_stable/autoregressive_eval_bias_m0p4_64/autoregressive_completion_metrics.json

Autoregressive results:
  bias +0.4: node_unknown=0.2477, edge_present=0.0298, presence_f1=0.0827, typed_f1=0.0435, edges=34.12 vs 64.27, connected=0.062 vs 0.938
  bias -0.2: node_unknown=0.2886, edge_present=0.0491, presence_f1=0.1227, typed_f1=0.0588, edges=48.50 vs 64.27, connected=0.375 vs 0.938
  bias -0.4: node_unknown=0.2733, edge_present=0.0596, presence_f1=0.1389, typed_f1=0.0662, edges=54.69 vs 64.27, connected=0.391 vs 0.938

Distribution notes:
  For autoregressive full completion, edge-type JS improves from 0.00487
  at bias +0.4 to 0.00114 at bias -0.4, and degree JS improves from 0.1590
  to 0.0334. This confirms calibration helps distribution match, but does not
  fix connectedness.

Weakness:
  Autoregressive full completion accumulates local edge errors. Even when edge
  count becomes closer to reference, many generated graphs remain disconnected.
  This checkpoint is therefore usable as a next-node prototype but not as a
  high-quality one-click full-completion model.

Next:
  For app next-node, use this checkpoint with `edge_none_logit_bias=-0.2`.
  For full completion, add inference-time connectivity repair / reranking or
  train with explicit connectedness/bridge-edge pressure. Do not present the
  current autoregressive full-completion output as final quality.
```

### Full Completion V2

Decision, 2026-05-19:

```text
Do not use naive autoregressive next-node as the main one-click full-completion
model. The single next-node step is usable, but repeated steps accumulate edge
errors and leave many generated graphs disconnected.

Train a separate one-shot full-completion checkpoint instead.
```

New config:

```text
digress/configs/experiment/msd_wall_full_completion_v2.yaml
```

Training command:

```powershell
cd D:\Github\GSDiff\digress
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_full_completion_v2.yaml
```

Evaluation command after training:

```powershell
.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_full_completion_v2 completion.checkpoint=outputs/.../checkpoints/msd_wall_full_completion_v2/best.ckpt completion.num_samples=128 completion.grid_samples=12
```

What changed:

- Full completion is one-shot: partial graph in, all missing nodes/edges out.
- Graph condition is enabled: current count, target count, remaining count, and
  remaining room-type inventory are appended to global `y`.
- Edge loss uses `split_presence_type`.
- Added `edge_degree_loss_weight` to penalize wrong per-node masked-edge degree.
- Full-completion eval now passes graph condition into `complete_batch`.
- Completion eval can optionally report deterministic connectivity repair
  metrics separately from raw model metrics.

New / changed files:

```text
digress/configs/experiment/msd_wall_full_completion_v2.yaml
digress/src/diffusion/absorbing_losses.py
digress/src/diffusion_model_absorbing.py
digress/scripts/graph_completion_eval.py
digress/scripts/test_graph_completion.py
```

Component split:

```text
test_graph_completion.py is now the CLI driver only.
graph_completion_eval.py contains reusable completion metrics, graph drawing,
graph-condition helpers, and connectivity repair.
absorbing_losses.py contains absorbing masked loss logic.
```

Smoke check:

```text
py_compile passed.
Config/model construction passed:
  input_dims: {'X': 16, 'E': 6, 'y': 24}
  output_dims: {'X': 10, 'E': 6, 'y': 0}
  graph_condition_dim: 12
```

Expected evaluation standard:

```text
Report raw metrics first:
  node_unknown_accuracy
  edge_presence_f1
  typed_edge_f1
  avg_edges vs reference
  connected_frac vs reference
  JS/KL/TV distribution similarity

Report repaired metrics separately:
  repaired metrics are app-facing diagnostics, not proof that the raw model is
  good enough.
```

Full Completion V2 evaluation, 2026-05-20:

```text
Status: raw model is still weak semantically, but top-k graph-only reranking
makes one-click completion structurally usable for the app prototype.

Run:
  outputs/2026-05-19/18-58-03-msd_wall_full_completion_v2

Checkpoint:
  outputs/2026-05-19/18-58-03-msd_wall_full_completion_v2/checkpoints/msd_wall_full_completion_v2/best.ckpt

Training:
  completed 200 epochs
  best validation at epoch 145
  best val_masked_loss=4.4979
  final epoch 200 val_masked_loss=4.5340

Raw eval at default edge_none_logit_bias=-0.2:
  output: outputs/2026-05-19/18-58-03-msd_wall_full_completion_v2/completion_eval_best_128
  node_unknown_accuracy=0.2524
  edge_present_accuracy=0.0976
  edge_presence_f1=0.1749
  typed_edge_f1=0.0829
  avg_edges=78.97 vs 62.44 reference
  connected_frac=0.906 vs 0.930 reference
  interpretation: connectedness is close, but the graph is over-connected.

Bias sweep:
  bias 0.0: avg_edges=74.53 vs 64.27, connected=0.797
  bias 0.2: avg_edges=65.47 vs 64.27, connected=0.734
  bias 0.4: avg_edges=54.59 vs 64.27, connected=0.531

Interpretation:
  `edge_none_logit_bias=0.2` calibrates edge count well, but raw connectedness
  drops. This shows the problem is not just edge quantity; the model is not
  reliably selecting bridge edges.

Top-k graph-only rerank:
  command:
    .\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_full_completion_v2 completion.checkpoint=outputs/2026-05-19/18-58-03-msd_wall_full_completion_v2/checkpoints/msd_wall_full_completion_v2/best.ckpt completion.out_dir=outputs/2026-05-19/18-58-03-msd_wall_full_completion_v2/completion_eval_bias_0p2_rerank8_64 completion.num_samples=64 completion.grid_samples=8 completion.repair_connectivity=true completion.rerank_candidates=8 completion.rerank.expected_degree=4.2 completion.rerank.connected_weight=4.0 completion.rerank.edge_count_weight=1.0 completion.rerank.isolated_weight=0.25 model.edge_none_logit_bias=0.2

  output:
    outputs/2026-05-19/18-58-03-msd_wall_full_completion_v2/completion_eval_bias_0p2_rerank8_64/completion_metrics.json
    outputs/2026-05-19/18-58-03-msd_wall_full_completion_v2/completion_eval_bias_0p2_rerank8_64/completion_grid.png

  raw first candidate:
    node_unknown_accuracy=0.2528
    edge_present_accuracy=0.0790
    edge_presence_f1=0.1583
    typed_edge_f1=0.0780
    avg_edges=65.48 vs 64.27 reference
    connected_frac=0.766 vs 0.938 reference

  reranked candidate:
    node_unknown_accuracy=0.2334
    edge_present_accuracy=0.0885
    edge_presence_f1=0.1689
    typed_edge_f1=0.0873
    avg_edges=65.53 vs 64.27 reference
    connected_frac=1.000 vs 0.938 reference
    degree_js=0.0071
    edge_type_js=0.00023

Decision:
  For app one-click full completion, use this checkpoint with
  `edge_none_logit_bias=0.2` and `rerank_candidates=8` as the current prototype.
  For paper/model-quality claims, report raw metrics separately and do not claim
  the raw model solves full completion yet.

Weakness:
  Exact edge recovery is still weak. Rerank improves structural usability and
  degree distribution but does not make the model semantically accurate.

Next:
  Wire reranked full completion into the backend/app path if one-click
  completion is needed now. For a stronger checkpoint, train a V3 model with
  explicit bridge-edge or component-aware supervision rather than only global
  edge count/degree losses.
```

App integration, 2026-05-20:

```text
Status: wired as prototype app mode, not a final model-quality claim.

Backend:
  endpoint: POST /api/generate/graph/completion
  model key: msd_wall_full_completion_v2
  checkpoint:
    digress/outputs/2026-05-19/18-58-03-msd_wall_full_completion_v2/checkpoints/msd_wall_full_completion_v2/best.ckpt

Frontend:
  mode: graph_completion
  visible label: Graph Completion
  reused components:
    GraphModelSelector
    BubbleGraphCanvas
    GenerateButton
    HistoryBar

Runtime policy:
  target_num_nodes default: 30
  num_candidates: 8
  edge_none_logit_bias: 0.2
  rerank expected_degree: 4.2
  rerank connected_weight: 4.0
  rerank edge_count_weight: 1.0
  rerank isolated_weight: 0.25
  connectivity repair: enabled, edge type wall

Important limitation:
  The offline evaluation used ground-truth remaining room inventory as a graph
  condition. The app does not know the future room inventory, so the backend
  estimates the remaining inventory from known room counts plus an MSD room-type
  prior. This is appropriate for an app prototype, but it is not the same
  condition as the diagnostic eval setup.

Weakness:
  The app output should be structurally usable after reranking/repair, but exact
  room type and edge recovery remain weak. Keep labeling this as a prototype
  completion mode until a stronger model is trained and evaluated.
```

Graph-condition optimization plan, 2026-05-22:

```text
Status: planned. This is the next high-value model improvement before simply
scaling model size.

Why this belongs here:
  `docs/absorbing_d3pm_implementation.md` is the absorbing D3PM architecture
  and experiment notebook. Graph condition changes affect training inputs,
  evaluation setup, app inference, and checkpoint compatibility, so this is the
  right place to track the plan and later results.
```

Current V2 condition:

```text
condition vector:
  current_known_count / max_nodes
  target_total_count / max_nodes
  remaining_count / max_nodes
  remaining_room_type_inventory / target_total_count

Training/eval:
  remaining_room_type_inventory is available from the ground-truth full graph.

App:
  remaining_room_type_inventory is not known, so the backend estimates it from
  known room counts plus an MSD room-type prior.
```

Problem:

```text
The offline eval condition is stronger than the app condition. This makes V2
look cleaner in diagnostics than in real app usage.

The model also receives only room-count inventory. It does not receive enough
global graph-shape information such as target edge density, access-edge count,
or whether the visible partial graph already has multiple connected components.
For full completion, those missing global hints can cause:
  over/under connection
  weak bridge-edge selection
  poor connectedness despite reasonable avg_edges
```

V3 graph-condition goals:

```text
Use only conditions that can exist in the app, or train with explicit dropout so
the model is robust when some conditions are estimated or missing.

Separate three condition groups:
  required app conditions:
    current_known_count
    target_total_count
    remaining_count

  optional user/app conditions:
    desired room-type inventory
    desired edge density / average degree
    desired access-edge emphasis

  derived partial-graph conditions:
    known edge count
    known average degree
    known connected component count
    known isolated node count
    known edge-type histogram
```

Concrete implementation plan:

```text
1. Add graph-condition schema/versioning.
   Do not silently change old checkpoint dimensions.
   Add a new config flag such as:
     model.graph_condition.version: v3

2. Add condition dropout during training.
   Randomly hide room inventory and optional graph stats with probability
   0.3-0.5, replacing them with zeros plus an availability bit.
   This trains the model for both GT-conditioned eval and app-style inference.

3. Add availability bits.
   For every optional condition group, append a binary flag:
     room_inventory_available
     target_density_available
     edge_type_hist_available
   This avoids confusing "unknown" with a true zero value.

4. Add partial-graph structural stats.
   These are always available in the app because they come from the user's
   current graph:
     known_edges / max_possible_edges
     known_avg_degree / max_nodes
     known_components / max_nodes
     known_isolated_nodes / max_nodes
     known edge-type histogram

5. Train/eval under two modes.
   Report both:
     oracle_condition: uses GT remaining inventory
     app_condition: uses only app-available/estimated conditions
   The app-facing checkpoint should be selected by app_condition metrics, not
   by oracle_condition alone.

6. Update backend request shape only after the model supports it.
   Add optional fields:
     target_num_nodes
     desired_room_counts
     desired_avg_degree
   Keep defaults so the app can still run without user-supplied inventory.
```

V3 evaluation requirements:

```text
For every V3 checkpoint, report:
  oracle_condition metrics
  app_condition metrics
  completion_grid.png for both modes
  connected_frac
  avg_edges vs reference
  degree_js
  edge_type_js
  node_unknown_accuracy
  edge_presence_f1
  typed_edge_f1

Decision rule:
  If oracle_condition improves but app_condition does not, the model is not yet
  better for the product. It may still be useful for a controlled research
  setting, but not as the app default.
```

First V3 experiment proposal:

```text
Config name:
  msd_wall_full_completion_v3_conditioned.yaml

Start from V2 choices:
  full_completion objective
  split_presence_type edge loss
  lambda_train: [2.0, 0]
  edge_degree_loss_weight: 0.5
  known_ratio_start: 0.75
  known_ratio_end: 0.45

Add:
  graph_condition.version: v3
  graph_condition.condition_dropout: 0.4
  graph_condition.include_partial_stats: true
  graph_condition.include_availability_bits: true

Do not increase model size in the first V3 run.
Reason:
  If V3 improves with the same capacity, the bottleneck was condition mismatch,
  not model size. Only scale capacity after this is tested.
```

V3 conditioned training observation, 2026-05-22:

```text
Status: running; early training looks healthy but not enough to judge final
completion quality.

Run:
  outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned

Command:
  .\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_full_completion_v3_conditioned.yaml

Checkpoint directory:
  outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/checkpoints/msd_wall_full_completion_v3_conditioned

Current observed progress:
  latest history row: epoch 33 train
  latest validation: epoch 30
  current best checkpoint: best.ckpt from epoch 30 validation

Validation trend:
  epoch 5:  val_masked_loss=4.9381, node=1.9162, edge=1.5110
  epoch 10: val_masked_loss=4.8467, node=1.8874, edge=1.4796
  epoch 15: val_masked_loss=4.8004, node=1.8649, edge=1.4678
  epoch 20: val_masked_loss=4.7864, node=1.8582, edge=1.4641
  epoch 25: val_masked_loss=4.7430, node=1.8304, edge=1.4563
  epoch 30: val_masked_loss=4.7270, node=1.8056, edge=1.4607

Training trend:
  train_weighted fell from 5.886 at epoch 1 to about 4.41 by epoch 33.
  train_node_ce fell from 1.974 to about 1.708.
  train_edge_ce fell from 1.956 to about 1.349.

Interpretation:
  This is a normal early trajectory. Validation is still improving, and there
  is no clear overfitting signal yet. Edge validation improved early but is
  slightly noisy around epoch 25-30, so the run should continue at least to
  80-120 epochs before making a model-quality call.

Weakness / caution:
  V3 loss is not directly comparable to V2 loss because V3 has a larger graph
  condition vector and condition dropout. Final judgment must use both
  app_condition and oracle_condition completion evals, not train loss alone.

Next:
  Let training continue. After a stable best checkpoint, run:
    app raw eval
    app rerank/repair eval
    oracle raw eval
  and compare against V2.
```

V3 conditioned final result, 2026-05-22:

```text
Status: trained successfully, but not good enough to replace V2 as the app
default.

Run:
  outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned

Checkpoint:
  outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/checkpoints/msd_wall_full_completion_v3_conditioned/best.ckpt

Training:
  completed 200 epochs
  best validation at epoch 180
  best val_masked_loss=4.5272
  final epoch 200 val_masked_loss=4.6097

Training interpretation:
  The model improved steadily until about epoch 180, then regressed at the final
  validation. Use `best.ckpt`, not `last-v1.ckpt`.
```

Evaluation commands:

```powershell
.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_full_completion_v3_conditioned completion.checkpoint=outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/checkpoints/msd_wall_full_completion_v3_conditioned/best.ckpt completion.out_dir=outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/completion_eval_app_raw_128 completion.condition_mode=app completion.num_samples=128 completion.grid_samples=12 completion.repair_connectivity=false completion.rerank_candidates=1

.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_full_completion_v3_conditioned completion.checkpoint=outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/checkpoints/msd_wall_full_completion_v3_conditioned/best.ckpt completion.out_dir=outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/completion_eval_oracle_raw_128 completion.condition_mode=oracle completion.num_samples=128 completion.grid_samples=12 completion.repair_connectivity=false completion.rerank_candidates=1

.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_full_completion_v3_conditioned completion.checkpoint=outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/checkpoints/msd_wall_full_completion_v3_conditioned/best.ckpt completion.out_dir=outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/completion_eval_app_rerank8_repair_128 completion.condition_mode=app completion.num_samples=128 completion.grid_samples=12 completion.repair_connectivity=true completion.rerank_candidates=8 completion.rerank.expected_degree=4.2 completion.rerank.connected_weight=4.0 completion.rerank.edge_count_weight=1.0 completion.rerank.isolated_weight=0.25 model.edge_none_logit_bias=0.2
```

Evaluation outputs:

```text
app raw:
  outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/completion_eval_app_raw_128/completion_metrics.json
  outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/completion_eval_app_raw_128/completion_grid.png

oracle raw:
  outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/completion_eval_oracle_raw_128/completion_metrics.json
  outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/completion_eval_oracle_raw_128/completion_grid.png

app rerank/repair:
  outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/completion_eval_app_rerank8_repair_128/completion_metrics.json
  outputs/2026-05-22/14-57-20-msd_wall_full_completion_v3_conditioned/completion_eval_app_rerank8_repair_128/completion_grid.png
```

Key metrics:

```text
V3 app raw:
  node_unknown_accuracy=0.1670
  edge_present_accuracy=0.0882
  edge_presence_f1=0.1687
  typed_edge_f1=0.0784
  avg_edges=74.08 vs 62.44 reference
  connected_frac=0.906 vs 0.930 reference

V3 oracle raw:
  node_unknown_accuracy=0.2328
  edge_present_accuracy=0.0912
  edge_presence_f1=0.1662
  typed_edge_f1=0.0848
  avg_edges=69.45 vs 62.44 reference
  connected_frac=0.773 vs 0.930 reference

V3 app rerank/repair:
  node_unknown_accuracy=0.1718
  edge_present_accuracy=0.0756
  edge_presence_f1=0.1658
  typed_edge_f1=0.0761
  avg_edges=61.80 vs 62.44 reference
  connected_frac=1.000 vs 0.930 reference
  degree_js=0.00676
  edge_type_js=0.00013
```

Comparison to V2:

```text
V2 raw:
  node_unknown_accuracy=0.2524
  edge_presence_f1=0.1749
  typed_edge_f1=0.0829
  avg_edges=78.97 vs 62.44
  connected_frac=0.906 vs 0.930

V2 rerank/repair, 64 samples:
  node_unknown_accuracy=0.2334
  edge_presence_f1=0.1689
  typed_edge_f1=0.0873
  avg_edges=65.53 vs 64.27
  connected_frac=1.000 vs 0.938
```

Interpretation:

```text
V3 condition dropout improved the app/eval framing but did not improve the
model enough. App-condition reranking gives excellent structural usability
after postprocess: edge count, connectedness, degree distribution, and edge-type
distribution are all close. However, semantic recovery is worse than V2:
node_unknown_accuracy and typed_edge_f1 are both lower.

Oracle-condition improves node accuracy over app-condition but hurts
connectedness. This means stronger future inventory hints do not automatically
solve graph topology.
```

Decision:

```text
Do not replace the current app default with V3.
Keep V2 + edge_none_logit_bias=0.2 + rerank/repair as the current app prototype
unless visual inspection of V3 grids is clearly preferable for product demos.

V3 is useful as a diagnostic checkpoint showing that condition mismatch was not
the only bottleneck. The remaining problem is likely loss/objective structure:
bridge-edge selection, component-level connectivity, and semantic room recovery.
```

Next:

```text
Do not increase capacity yet based only on this result.
The next model experiment should target objective quality:
  bridge-edge / component-aware supervision
  stronger node-type auxiliary loss or class-balanced node loss
  lower condition_dropout, e.g. 0.2, only if app-condition node accuracy is the
  main target

If a larger model is tested later, keep V2/V3 configs unchanged and add a new
config name, e.g. `msd_wall_full_completion_v4_bridge.yaml`.
```

New / changed files for V3 graph condition:

```text
digress/configs/experiment/msd_wall_full_completion_v3_conditioned.yaml
  New V3 experiment config. Keeps V2 model capacity but enables graph condition
  version v3, condition dropout, availability bits, target density, and partial
  graph stats.

digress/src/diffusion/absorbing_utils.py
  Extends graph_condition_dim() with versioned v3 condition dimensions while
  preserving v2 checkpoint compatibility.

digress/src/diffusion_model_absorbing.py
  Adds v3 graph-condition construction, app/oracle availability flags,
  condition dropout, target avg-degree condition, and partial known-graph stats.

digress/scripts/graph_completion_eval.py
  Adds `completion_graph_condition(..., condition_mode="oracle"|"app")` so eval
  can compare GT-oracle conditions against realistic app conditions.

digress/scripts/test_graph_completion.py
  Adds `completion.condition_mode=oracle|app` and records the selected condition
  mode in completion_metrics.json.
```

V4 bridge-loss experiment, 2026-05-22:

```text
Status: code ready; full 200-epoch training not started yet.

Goal:
  Improve one-shot full graph completion without increasing model capacity.
  V3 showed that condition mismatch alone was not the bottleneck. The next
  hypothesis is that the loss underweights two important cases:
    rare room labels
    real edges that attach hidden completion nodes back to the known partial graph

Config:
  digress/configs/experiment/msd_wall_full_completion_v4_bridge.yaml

Training command:
  cd D:\Github\GSDiff\digress
  .\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_full_completion_v4_bridge.yaml

Expected checkpoint:
  outputs/<date>/<time>-msd_wall_full_completion_v4_bridge/checkpoints/msd_wall_full_completion_v4_bridge/best.ckpt

Important config differences from V3:
  graph_condition.version=v3
  graph_condition.condition_dropout=0.2
  graph_condition.edge_hist_dropout=0.1
  model.node_class_balance=inverse_sqrt
  model.node_class_weight_max=4.0
  model.bridge_edge_loss_weight=3.0
  model.bridge_edges_only_full_completion=true

Reasoning:
  node_class_balance makes rare room types matter more without using aggressive
  full inverse-frequency weights.
  bridge_edge_loss_weight upweights real known-to-hidden edges during the
  full-completion mask. This directly targets the observed failure mode where
  graphs have plausible global edge counts but poor attachment/connectivity to
  the visible input graph.
```

V4 validation / smoke:

```text
Syntax check:
  .\.venv\Scripts\python.exe -m py_compile digress\src\diffusion\absorbing_losses.py digress\src\diffusion_model_absorbing.py
  Result: passed.

Fast-dev smoke command:
  cd D:\Github\GSDiff\digress
  .\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_full_completion_v4_bridge.yaml general.name=debug train.batch_size=2 general.samples_to_generate=0 general.samples_to_save=0 general.final_model_samples_to_generate=0 general.final_model_samples_to_save=0

Fast-dev smoke result:
  passed one train batch and one validation batch.
  train_masked_loss=11.234
  val_masked_loss=10.7763
  masked_node_CE=2.205 on val
  masked_edge_CE=4.286 on val

Note:
  A previous full 1-epoch smoke with validation progress bar timed out on
  Windows stdout/tqdm with OSError 22 after reaching validation. That failure is
  not from the V4 loss. The debug fast-dev run confirms the model/loss path
  executes.
```

V4 evaluation commands after training:

```powershell
.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_full_completion_v4_bridge completion.checkpoint=outputs/<date>/<time>-msd_wall_full_completion_v4_bridge/checkpoints/msd_wall_full_completion_v4_bridge/best.ckpt completion.out_dir=outputs/<date>/<time>-msd_wall_full_completion_v4_bridge/completion_eval_app_rerank8_repair_128 completion.condition_mode=app completion.num_samples=128 completion.grid_samples=12 completion.repair_connectivity=true completion.rerank_candidates=8 model.edge_none_logit_bias=0.2

.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_full_completion_v4_bridge completion.checkpoint=outputs/<date>/<time>-msd_wall_full_completion_v4_bridge/checkpoints/msd_wall_full_completion_v4_bridge/best.ckpt completion.out_dir=outputs/<date>/<time>-msd_wall_full_completion_v4_bridge/completion_eval_app_raw_128 completion.condition_mode=app completion.num_samples=128 completion.grid_samples=12 completion.repair_connectivity=false completion.rerank_candidates=1
```

V4 success criteria:

```text
Must beat or match V2/V3 on:
  connected_frac after rerank/repair close to 1.0
  avg_edges close to reference
  degree_js close to V3 rerank/repair

Must improve over V3 on at least one semantic metric:
  node_unknown_accuracy
  typed_edge_f1

If V4 only improves connectedness but hurts semantic recovery further, do not
promote it to the app default. Keep V2 rerank/repair as the app prototype and
consider either a larger model or a separate semantic auxiliary objective.
```

New / changed files for V4 bridge loss:

```text
digress/src/diffusion/absorbing_losses.py
  Adds `_node_class_weights()` for class-balanced masked node CE.
  Adds `_known_to_hidden_bridge_mask()` to identify real masked edges between
  visible partial nodes and hidden completion nodes.
  Adds weighted presence/type edge loss for those bridge edges.

digress/configs/experiment/msd_wall_full_completion_v4_bridge.yaml
  New full-completion experiment config. Keeps V3 graph condition and model
  capacity, lowers condition dropout, and enables node/bridge loss weighting.

docs/absorbing_d3pm_implementation.md
  Records the V4 hypothesis, commands, smoke result, expected evaluation, and
  files changed.
```

V4 bridge-loss final result, 2026-05-22:

```text
Status: trained successfully, but not good enough to replace V2 as the app
default.

Run:
  outputs/2026-05-22/19-48-45-msd_wall_full_completion_v4_bridge

Checkpoint:
  outputs/2026-05-22/19-48-45-msd_wall_full_completion_v4_bridge/checkpoints/msd_wall_full_completion_v4_bridge/best.ckpt

Training:
  completed 200 epochs
  best validation at epoch 180
  best val_masked_loss=5.3272
  final epoch 200 val_masked_loss=5.3372

Training interpretation:
  The run is stable and did not collapse. It improves from val=5.7913 at epoch
  5 to best val=5.3272 at epoch 180, then plateaus. Because V4 adds
  class-balanced node loss and bridge-edge weighting, this validation number is
  not directly comparable to V2/V3 loss values. Task metrics are required.
```

Evaluation commands:

```powershell
.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_full_completion_v4_bridge completion.checkpoint=outputs/2026-05-22/19-48-45-msd_wall_full_completion_v4_bridge/checkpoints/msd_wall_full_completion_v4_bridge/best.ckpt completion.out_dir=outputs/2026-05-22/19-48-45-msd_wall_full_completion_v4_bridge/completion_eval_app_rerank8_repair_128 completion.condition_mode=app completion.num_samples=128 completion.grid_samples=12 completion.repair_connectivity=true completion.rerank_candidates=8 model.edge_none_logit_bias=0.2

.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_full_completion_v4_bridge completion.checkpoint=outputs/2026-05-22/19-48-45-msd_wall_full_completion_v4_bridge/checkpoints/msd_wall_full_completion_v4_bridge/best.ckpt completion.out_dir=outputs/2026-05-22/19-48-45-msd_wall_full_completion_v4_bridge/completion_eval_app_raw_128 completion.condition_mode=app completion.num_samples=128 completion.grid_samples=12 completion.repair_connectivity=false completion.rerank_candidates=1
```

Evaluation outputs:

```text
app raw:
  outputs/2026-05-22/19-48-45-msd_wall_full_completion_v4_bridge/completion_eval_app_raw_128/completion_metrics.json
  outputs/2026-05-22/19-48-45-msd_wall_full_completion_v4_bridge/completion_eval_app_raw_128/completion_grid.png

app rerank/repair:
  outputs/2026-05-22/19-48-45-msd_wall_full_completion_v4_bridge/completion_eval_app_rerank8_repair_128/completion_metrics.json
  outputs/2026-05-22/19-48-45-msd_wall_full_completion_v4_bridge/completion_eval_app_rerank8_repair_128/completion_grid.png
```

Key metrics:

```text
V4 app raw:
  node_unknown_accuracy=0.1771
  edge_present_accuracy=0.0941
  edge_presence_f1=0.1707
  typed_edge_f1=0.0817
  avg_edges=76.50 vs 62.44 reference
  connected_frac=0.969 vs 0.930 reference
  degree_js=0.03091
  edge_type_js=0.00099

V4 app rerank/repair:
  raw first-candidate node_unknown_accuracy=0.1877
  raw first-candidate edge_presence_f1=0.1637
  raw first-candidate avg_edges=63.93 vs 62.44 reference
  raw first-candidate connected_frac=0.781 vs 0.930 reference

  final node_unknown_accuracy=0.1744
  final edge_present_accuracy=0.0791
  final edge_presence_f1=0.1647
  final typed_edge_f1=0.0785
  final avg_edges=63.18 vs 62.44 reference
  final connected_frac=1.000 vs 0.930 reference
  final degree_js=0.00593
  final edge_type_js=0.00008
  inference=69.42s / 128 samples = 0.542s per sample
```

Comparison:

```text
Against V3 app rerank/repair:
  V3 node_unknown_accuracy=0.1718
  V4 node_unknown_accuracy=0.1744  slightly higher, but not meaningful enough

  V3 edge_presence_f1=0.1658
  V4 edge_presence_f1=0.1647  slightly worse

  V3 typed_edge_f1=0.0761
  V4 typed_edge_f1=0.0785  slightly higher, but still weak

  V3 avg_edges=61.80 vs 62.44
  V4 avg_edges=63.18 vs 62.44

  V3 connected_frac=1.000 after repair
  V4 connected_frac=1.000 after repair

Against V2 rerank/repair:
  V2 node_unknown_accuracy=0.2334
  V4 node_unknown_accuracy=0.1744

  V2 edge_presence_f1=0.1689
  V4 edge_presence_f1=0.1647

  V2 typed_edge_f1=0.0873
  V4 typed_edge_f1=0.0785
```

Interpretation:

```text
V4 bridge weighting did not solve the core full-completion problem. It can
produce structurally usable graphs after rerank/repair, but semantic room-type
recovery and true edge recovery remain weak.

Raw V4 is still over-connected: avg_edges=76.50 vs 62.44. Rerank/repair fixes
global structure, but the model itself has not learned significantly better
GT-aligned edge selection.

The bridge loss may have improved attachment pressure, but it did not improve
the exact hidden graph reconstruction enough to justify replacing V2.
```

Decision:

```text
Do not promote V4 to app default.
Keep V2 + edge_none_logit_bias=0.2 + rerank/repair as the current one-click
full-completion prototype unless visual inspection strongly favors another
checkpoint.

Use V4 as a negative/diagnostic result:
  loss shaping alone helped structure after inference-time selection, but did
  not materially improve semantic or typed-edge accuracy.
```

Next:

```text
If continuing one-shot full completion:
  Try a larger model only as a controlled V5 experiment, not as a guaranteed fix.
  Also add stronger semantic supervision, e.g. explicit node-type auxiliary
  objective on hidden nodes or candidate reranking that scores room-type prior
  consistency.

If prioritizing app usefulness:
  Prefer the next-node / graph-policy route, because exact one-shot GT recovery
  is too ambiguous and the current one-shot checkpoints mostly need reranking
  and repair to become usable.
```

Historical specialized config choices, now superseded by Graph Policy V2:

```text
next_node:
  random=0.05, full_completion=0.10, next_node=0.85
  eval_strategy=next_node
  lambda_train=[1.5, 0]
  edge_present_loss_weight=3.0

full_completion:
  random=0.05, full_completion=0.90, next_node=0.05
  eval_strategy=full_completion
  lambda_train=[2.0, 0]
  edge_present_loss_weight=3.0
  known_ratio_start=0.7, known_ratio_end=0.5, curriculum_epochs=80
```

Why no task conditioning here:

```text
Task conditioning is most useful when one unified checkpoint must handle many
tasks equally well. With task-specific checkpoints, the task is already defined
by the checkpoint and the training distribution, so task conditioning is not
the highest-priority change.
```

### Edge-Present Weighted Loss

The specialized configs enable `edge_present_loss_weight`:

```yaml
model:
  edge_present_loss_weight: 3.0
```

This modifies only absorbing edge CE:

```text
true edge class == 0 ("none")        -> CE weight 1.0
true edge class > 0 (real relation)  -> CE weight edge_present_loss_weight
```

Why this is needed:

```text
MSD graphs have many more none edge slots than real edge slots. Without
reweighting, a model can get good all-slot edge accuracy by predicting many
none edges while still missing the true wall / door / passage / entrance
connections. That failure hurts next-node connection F1 and full-completion
connectedness.
```

Important:

```text
This is a training loss change. Old checkpoints are unchanged. Configs that do
not set edge_present_loss_weight default to 1.0 and keep the old behavior.
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
