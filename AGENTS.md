# Agent Notes for GSDiff

This file records project context and collaboration preferences for future
agents working in this repository.

## Project Goal

GSDiff is moving toward an app-driven architectural generation workflow:

1. Generate or edit a room-relation graph / bubble diagram.
2. Use the graph as the conditioning input for GSDiff floorplan generation.
3. Support interactive graph editing in the app, especially adding one room at
   a time and predicting its connections.

The current graph-modeling work is under `digress/`, which is intentionally a
separate DiGress-based environment from GRAN and the main GSDiff app.

## Current Modeling Direction

The project uses absorbing-state discrete graph diffusion for MSD graph tasks.
The important implementation files are:

- `digress/src/diffusion_model_absorbing.py`
- `digress/src/diffusion/absorbing_transition.py`
- `digress/src/diffusion/absorbing_utils.py`
- `digress/scripts/test_graph_generation.py`
- `digress/scripts/test_graph_completion.py`
- `digress/scripts/test_next_node_completion.py`
- `docs/absorbing_d3pm_implementation.md`

The current strategy is multi-checkpoint, not one checkpoint for all tasks.
Do not add task-type conditioning unless the project returns to a unified
single-checkpoint paper-style claim.

Expected checkpoints:

- `msd_wall_absorbing_v2`: unconditional graph-generation baseline.
- `msd_wall_absorbing_completion`: unified research baseline for full
  completion plus next-node completion.
- `msd_wall_absorbing_next_node`: app interaction checkpoint for "add one
  room" and predict its target-to-known edge types.
- `msd_wall_absorbing_full_completion`: one-click completion from partial graph
  to full graph.

The app-facing priority is next-node completion. Full graph completion is useful
for one-click auto-complete but is harder and more ambiguous.

## Key Commands

From `D:\Github\GSDiff\digress`:

```powershell
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_absorbing_next_node.yaml
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall_absorbing_full_completion.yaml
```

Test next-node:

```powershell
.\.venv\Scripts\python.exe scripts\test_next_node_completion.py msd_wall_absorbing_next_node next_node.checkpoint=outputs/.../checkpoints/msd_wall_absorbing_next_node/best.ckpt
```

Test full completion:

```powershell
.\.venv\Scripts\python.exe scripts\test_graph_completion.py msd_wall_absorbing_full_completion completion.checkpoint=outputs/.../checkpoints/msd_wall_absorbing_full_completion/best.ckpt
```

Test unconditional generation:

```powershell
.\.venv\Scripts\python.exe scripts\test_graph_generation.py msd_wall_absorbing_v2 test.checkpoint=outputs/.../checkpoints/msd_wall_absorbing_v2/best.ckpt
```

## Evaluation Priorities

For full completion, do not rely only on CE loss. Also inspect:

- `completion_grid.png`
- connected fraction
- edge-present accuracy
- average edge count versus reference
- graph distribution deltas

For next-node, the important app-facing metrics are:

- target node type accuracy
- target edge-present accuracy
- connection precision / recall / F1
- predicted target degree versus reference target degree
- predicted target connected-to-known fraction

The previous unified completion checkpoint showed:

- Full completion had reasonable edge count but poor connectivity.
- Next-node was more promising for the app but still weak on exact room type
  and true connection recovery.

## Experiment Tracking Requirement

Model work must be tracked continuously in docs. Do not only say "trained" or
"tested". Record what was run, what happened, what is still weak, and what
hypothesis should be tested next.

For every meaningful training or evaluation run, update the relevant doc,
usually `docs/absorbing_d3pm_implementation.md`, with:

- experiment/config name
- exact command or important overrides
- run directory
- checkpoint path, especially `best.ckpt`
- epoch count and whether training completed or was interrupted
- key train/validation losses
- task-specific metrics, not just loss
- visual output paths such as `completion_grid.png` or `next_node_grid.png`
- what improved compared with previous checkpoints
- what is still not good enough
- plausible causes of failure
- concrete next experiment or parameter change

Use docs as a lab notebook. The user values clear experiment history because it
prevents repeating bad ideas and makes model decisions defensible.

Preferred result language:

```text
Status: usable / promising / not good enough / failed
Evidence: concrete metrics and visual file paths
Weakness: exact observed failure mode
Next: one or two specific changes to test
```

Avoid vague claims like "looks better" without metrics or image paths. If a
metric is misleading, say why and propose a better one.

## Architecture Guidance

Prefer small, testable changes before major architecture rewrites.

High-value next improvements:

- edge-present weighted loss for real non-none edges
- optional connectivity postprocess / reranking at inference time
- task-specific checkpoint selection using task-specific metrics
- possible larger-capacity sweep only after task/loss issues are understood
- geometry features only as a separate experiment, because they change data
  schema and model inputs

Do not assume bigger model capacity is the first fix. If train loss drops while
validation/task metrics lag, first check task distribution, edge imbalance,
and evaluation mismatch.

## App Direction

The app should eventually expose graph-generation modes explicitly:

- random graph generation
- next-node suggestion
- full graph completion
- model/GPU loading status

The preferred interaction is not one deterministic answer only. For next-node,
the app should ideally return top-k candidates so the user can accept or edit
one suggestion.

## User Collaboration Preferences

The user prefers:

- concise Chinese communication
- direct, pragmatic engineering judgment
- implementation over long abstract planning
- config-driven commands rather than long argument lists
- documentation updates whenever architecture or experiments change
- clear distinction between baseline, current best, and not-yet-good-enough
- honest assessment: say when a model is not good enough
- preserving old checkpoints/configs instead of overwriting baselines
- commit changes as meaningful standalone commits

When explaining model results, include concrete metrics and file paths. Avoid
overclaiming based on loss alone; visual output and task-specific metrics matter.

## Git / File Hygiene

- Do not revert user changes unless explicitly asked.
- Outputs and datasets should remain ignored.
- Keep generated experiment outputs out of commits.
- Prefer adding new experiment configs over mutating a known baseline config.
- If adding or changing model behavior, update
  `docs/absorbing_d3pm_implementation.md`.
