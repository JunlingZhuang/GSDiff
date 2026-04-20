
# GRAN (v2 fork for RPLAN floorplan bubble diagrams)

This fork extends the original GRAN with:

- **Node attribute prediction** — the `output_attr` head predicts per-node
  class labels (e.g. room type for floorplans) jointly with edges.
- **GATv2 backbone** — optional `torch_geometric.nn.GATv2Conv` replacement for
  the original GRU-based message passing.
- **Partial-graph conditioning** — `GRANv2.expand_graph()` continues autoregressive
  generation from a user-supplied partial graph; `predict_attr_with_edges()`
  predicts only the attribute of a new node given its connection pattern.
- **Standalone RPLAN pipeline** — `dataset/rplan_preprocessing/` reads raw
  RPLAN PNGs and produces GRAN-ready graphs with room types + edge types
  (wall vs door).

All 22 unit tests pass (`pytest tests/`). See `docs/v2-progress.md` for
per-task file changes and test results.

---

## Quick start (RPLAN workflow)

### 0. Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA 11.8 (or change `device` to `cpu` in configs for CPU-only)
- RPLAN raw PNGs (80,788 files from http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/).
  Place them anywhere; you'll point the preprocessor at that directory.

### 1. Set up the environment

This fork piggybacks on the parent GSDiff venv at `D:/Github/GSDiff/.venv`
(torch 2.0.1+cu118 lives there). Install the extras GRAN needs on top:

```bash
# Activate the parent venv first (or set VIRTUAL_ENV)
VIRTUAL_ENV=D:/Github/GSDiff/.venv uv pip install \
    torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

VIRTUAL_ENV=D:/Github/GSDiff/.venv uv pip install \
    torch-geometric==2.7.0 \
    networkx==2.8.8 \
    scipy==1.10.1 \
    pyemd==0.5.1 \
    opencv-python \
    shapely \
    scikit-image \
    matplotlib tqdm tensorboardX pyyaml easydict pytest
```

Verify:

```bash
D:/Github/GSDiff/.venv/Scripts/python.exe -c \
    "import torch, torch_geometric, pyemd; print(torch.__version__, torch_geometric.__version__)"
# expect: 2.0.1+cu118 2.7.0
```

### 2. Preprocess RPLAN

Convert raw 4-channel PNGs into GRAN-ready bubble diagram graphs
(nodes = rooms with `attr` class 0..6; edges = adjacency with `edge_type`
1=wall / 2=door).

```bash
cd D:/Github/GSDiff/GRAN

# Start small to verify the pipeline (takes ~1 min):
D:/Github/GSDiff/.venv/Scripts/python.exe dataset/rplan_preprocessing/preprocess.py \
    --raw_dir <path-to-RPLAN-raw-png-directory> \
    --out_dir data/rplan \
    --max_samples 2000 \
    --visualize 10

# Outputs:
#   data/rplan/graphs.p       -- networkx pickle (GRAN loads this)
#   data/rplan/tu/            -- TU-format text files (alternative loader)
#   data/rplan/vis/           -- 10 sample bubble diagram visualizations
```

For a larger training set, rerun without `--max_samples` (or set e.g.
`--max_samples 10000`). The pickle is overwritten each time.

### 3. Train

Two ready-made configs:

```bash
# Debug run: 5 epochs, small model, ~20 seconds on RTX 3090.
# Confirms the pipeline works end-to-end before a real run.
D:/Github/GSDiff/.venv/Scripts/python.exe run_exp.py \
    -c config/gran_v2_rplan_debug.yaml

# Full run: 300 epochs, bigger model, tuned for 40k preprocessed graphs.
# ~30-60 minutes on RTX 3090.
D:/Github/GSDiff/.venv/Scripts/python.exe run_exp.py \
    -c config/gran_v2_rplan.yaml
```

#### Training output

Epochs without validation emit one line; epochs with validation emit two
aligned lines so you can compare train / val at a glance:

```
ep 0001/0300  lr=2.00e-04  train: edge=0.6573  attr=1.3499  total=2.0072
ep 0002/0300  lr=2.00e-04  train: edge=0.4889  attr=0.8651  total=1.3539
ep 0003/0300  lr=2.00e-04  train: edge=0.3971  attr=0.7337  total=1.1308
ep 0004/0300  lr=2.00e-04  train: edge=0.3494  attr=0.6908  total=1.0402
ep 0005/0300  lr=2.00e-04  train: edge=0.3156  attr=0.6723  total=0.9879
                            val: edge=0.3312  attr=0.7102  total=1.0237
ep 0006/0300  lr=2.00e-04  train: edge=0.2902  attr=0.6570  total=0.9472
...
INFO | saving snapshot @ epoch 0025
```

A tqdm progress bar also rides on top showing the most recent per-batch
losses and ETA:

```
train: 5%|█▌  | 15/300 [01:32<29:10, edge=0.290, attr=0.660, total=0.950]
```

#### Reading the losses

| Column | Meaning | Healthy range |
|--------|---------|---------------|
| `edge` | BCE-with-logits edge loss | starts near `ln 2 ≈ 0.69`, drops toward ~0.2 |
| `attr` | Cross-entropy node-attribute loss | starts near `ln K` (K = `num_attr_classes`; for RPLAN K=7 → 1.95), drops toward ~0.4-0.6 |
| `total` | `edge + lambda_attr * attr` | the quantity actually optimized |

If `val edge` diverges upward from `train edge`, the model is overfitting
— consider more data, lower `lr`, or higher `wd`.

#### Train / dev / test split

The v2 runner uses **disjoint** train/dev/test partitions (the original GRAN
code had `graphs_dev` as a subset of `graphs_train`, which made val loss
meaningless):

```
total_graphs
├── train: [0, train_ratio * total)              → used for backward passes
├── dev:   [train, train + dev_ratio * total)   → used ONLY for val loss
└── test:  [train + dev, total)                  → used only by `-t` mode
```

Control via config (`dataset.train_ratio`, `dataset.dev_ratio`).

#### Artifacts

Each run produces a directory at `exp/GRANv2_rplan/<run_id>/`:

| File | Content |
|------|---------|
| `model_snapshot_<epoch>.pth` | checkpoint saved every `train.snapshot_epoch` epochs (default 25). Each holds model weights + optimizer + scheduler state, so you can resume or evaluate any of them. |
| `config.yaml` | config snapshot. `test.test_model_dir` / `test.test_model_name` are auto-updated at each snapshot so `-t` mode picks up the latest checkpoint automatically. |
| `events.out.tfevents.*` | TensorBoard logs (per-iter + per-epoch + val scalars). |
| `log_exp_*.txt` | plain-text training log (same as stdout, minus the tqdm bar). |
| `train_stats.p` | pickle of `{train_loss, edge_loss, attr_loss, train_step, val_*}` lists at training end. |

With `snapshot_epoch: 25` and `max_epoch: 300` that's 12 checkpoints, each
~15-20 MB for the 256-dim / 6-layer model (smaller models are proportionally
smaller).

#### Resume from a snapshot

```yaml
train:
  is_resume: true
  resume_dir: exp/GRANv2_rplan/<run_id>
  resume_model: model_snapshot_0000150.pth
  resume_epoch: 150         # where to pick up the epoch counter
```

### 4. Test / sample graphs

After training, the snapshot path is auto-recorded in
`exp/GRANv2_rplan/<run_id>/config.yaml`. Run the test mode against it:

```bash
D:/Github/GSDiff/.venv/Scripts/python.exe run_exp.py \
    -c exp/GRANv2_rplan/<run_id>/config.yaml -t
```

Outputs:
- Generated graphs visualized as bubble diagrams under the same `<run_id>`
- MMD scores (degree, clustering, spectral) printed to stdout

### 5. View TensorBoard

```bash
D:/Github/GSDiff/.venv/Scripts/python.exe -m tensorboard.main \
    --logdir=exp/GRANv2_rplan
```

Open http://localhost:6006. Scalars shown:
- `edge_loss`, `attr_loss`, `train_loss` (per iteration)
- `epoch/edge_loss`, `epoch/attr_loss`, `epoch/total_loss` (per epoch means)

### 6. Configure training knobs

All tuning happens in the YAML config — no code changes needed. The
three sections you'll touch most:

```yaml
# config/gran_v2_rplan.yaml
train:
  max_epoch: 500            # training length
  batch_size: 32            # RPLAN graphs are tiny, so batches can be large
  lr: 1.0e-4                # learning rate
  lambda_attr: 1.0          # weight of attr CE loss in total_loss
  lr_decay_epoch: [300, 400]

model:
  use_gatv2: true           # false = original GRU-based GNN
  gatv2_num_heads: 4
  num_attr_classes: 7       # matches GSDiff's room taxonomy
  hidden_dim: 128
  num_GNN_layers: 4

dataset:
  train_ratio: 0.9          # fraction of graphs for training
```

See inline comments inside the YAML for every knob.

### 7. Run unit tests

```bash
cd D:/Github/GSDiff/GRAN
D:/Github/GSDiff/.venv/Scripts/python.exe -m pytest tests/ -v
# expect: 22 passed
```

---

## Extended inference modes (v2 only)

After loading a trained `GRANv2` in Python:

```python
# Mode A: expand a partial graph
A, attrs = model.expand_graph(
    partial_A=partial_A,          # (B, n_partial, n_partial) known adjacency
    partial_attrs=partial_attrs,  # (B, n_partial) known room types
    num_target_nodes=6,
)

# Mode B: predict a new node's type given its connections
fixed_edges = torch.zeros(1, 5); fixed_edges[:, 0] = 1; fixed_edges[:, 2] = 1
attr_logits = model.predict_attr_with_edges(partial_A, partial_attrs, fixed_edges)
new_node_type = attr_logits.argmax(dim=-1)
```

---

## Upstream (original GRAN)

Everything below this line is the original README from the upstream GRAN
repository, kept verbatim for reference.

---

# GRAN

This is the official PyTorch implementation of [Efficient Graph Generation with Graph Recurrent Attention Networks](https://arxiv.org/abs/1910.00760) as described in the following NeurIPS 2019 paper:

```
@inproceedings{liao2019gran,
  title={Efficient Graph Generation with Graph Recurrent Attention Networks}, 
  author={Liao, Renjie and Li, Yujia and Song, Yang and Wang, Shenlong and Nash, Charlie and Hamilton, William L. and Duvenaud, David and Urtasun, Raquel and Zemel, Richard}, 
  booktitle={NeurIPS},
  year={2019}
}
```

## Visualization

### Generation of GRAN per step:
![](http://www.cs.toronto.edu/~rjliao/imgs/gran_model.gif)


### Overall generation process:
<img src="http://www.cs.toronto.edu/~rjliao/imgs/gran_generation.gif" height="400px" width="550px" />


## Dependencies
Python 3, PyTorch(1.2.0)

Other dependencies can be installed via 

  ```pip install -r requirements.txt```


## Run Demos

### Train
* To run the training of experiment ```X``` where ```X``` is one of {```gran_grid```, ```gran_DD```, ```gran_DB```, ```gran_lobster```}:

  ```python run_exp.py -c config/X.yaml```
  

**Note**:

* Please check the folder ```config``` for a full list of configuration yaml files.
* Most hyperparameters in the configuration yaml file are self-explanatory.

### Test

* After training, you can specify the ```test_model``` field of the configuration yaml file with the path of your best model snapshot, e.g.,

  ```test_model: exp/gran_grid/xxx/model_snapshot_best.pth```	

* To run the test of experiments ```X```:

  ```python run_exp.py -c config/X.yaml -t```

**Note**:

* Please check the [evaluation](https://github.com/JiaxuanYou/graph-generation) to set up.

### Trained Models
* You could use our trained model for comparisons. Please make sure you are using the same split of the dataset. Running the following script will download the trained model:

	```./download_model.sh```	

## Sampled Graphs from GRAN

* Proteins Graphs from Training Set:
![](http://www.cs.toronto.edu/~rjliao/imgs/protein_train.png)

* Proteins Graphs Sampled from GRAN:
![](http://www.cs.toronto.edu/~rjliao/imgs/protein_sample.png)

## Cite
Please cite our paper if you use this code in your research work.

## Questions/Bugs
Please submit a Github issue or contact rjliao@cs.toronto.edu if you have any questions or find any bugs.
