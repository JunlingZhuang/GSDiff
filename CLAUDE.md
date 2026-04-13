# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GSDiff (AAAI 2025) synthesizes vector floorplans as structural graphs using a two-stage diffusion pipeline. Rooms are represented as graph cycles where nodes are corner points with coordinates + semantic labels, and edges represent walls.

## Environment Setup

```bash
# Python 3.10, PyTorch 2.0.1 + CUDA 11.8
uv venv --python 3.10 .venv
source .venv/Scripts/activate  # Windows (bash)
uv pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install networkx==3.1 numpy==1.26.0 opencv-python==4.9.0.80 scipy==1.10.1 \
  scikit-image==0.24.0 scikit-learn==1.4.1.post1 shapely==2.0.6 \
  tensorboardx==2.6.2.2 tqdm==4.65.0 matplotlib==3.9.2 pytorch-fid==0.3.0 pillow==10.0.1
```

Set `UV_CACHE_DIR` to a path on a drive with sufficient space (PyTorch ~2.4GB).

## Running Scripts

All training/testing scripts must be run from `scripts/` directory due to relative `../datasets/` paths.

```bash
cd scripts

# Testing (requires pretrained weights in outputs/ + preprocessed RPLAN data)
python test_main.py          # Unconstrained generation
python test_topo.py          # Topology-constrained
python test_boun.py          # Boundary-constrained

# Training
python trainval_main_unconstrained.py       # Stage 1: unconstrained nodes
python trainval_main_edge_unconstrained.py  # Stage 2: unconstrained edges
python trainval_main_topo.py                # Stage 1: topology-constrained
python trainval_main_edge_topo.py           # Stage 2: topology-constrained
python trainval_main_boun.py                # Stage 1: boundary-constrained
python trainval_main_edge_boun.py           # Stage 2: boundary-constrained

# Autoencoders (train before constrained generation)
python train-TopoTransformer-autoe.py       # Topology encoder (phase 1)
python train-TopoTransformer-autoe-final.py # Topology encoder (phase 2, finetune)
python train-CNN-autoe-final.py             # Boundary CNN encoder

# Evaluation (from project root)
python evalmetric-no-constrain-fid-kid.py
python evalmetric-no-constrain-geometry-topological-metrics.py
```

## Architecture

### Two-Stage Pipeline

```
Stage 1: Noise → Denoising Diffusion Transformer → Corner coordinates + room semantics
Stage 2: Corners → Edge Prediction Transformer → Wall connectivity → Rendered floorplan
```

Stage 2 constructs its own attention masks from Stage 1 output — it does not depend on the dataset. Stage 1 requires `global_attn_matrix` (53×53 bool) from the dataset to know per-sample node counts.

### Three Generation Modes

| Mode | Stage 1 Model | Condition Encoder | Condition Input |
|------|--------------|-------------------|-----------------|
| Unconstrained | `house_nn1.HeterHouseModel` | None | None |
| Topology | `heterhouse_80_106_2.TopoHeterHouseModel` | `bubble_diagram_57_9.TopoGraphModel` | Room types + adjacency matrix |
| Boundary | `heterhouse_81_106_3.BoundHeterHouseModel` | `boundary_78_10.BoundaryModel` | 256×256 boundary image |

Edge prediction models (`house_nn2.EdgeModel`, `heterhouse_56_31.TopoEdgeModel`, `heterhouse_56_32.BoundEdgeModel`) share the same I/O interface across all modes.

### Key Tensor Shapes

- Node features: `(batch, 53, 9)` — 2 coords (normalized [-1,1]) + 7 one-hot semantics
- With padding flag: `(batch, 53, 10)` — appended during diffusion
- Attention mask: `(batch, 53, 53)` bool — True for real nodes, False for padding
- Edge output: `(batch, 2809, 2)` — 53×53 flattened, binary [not-connected, connected]

### Pretrained Model Directory Layout

```
scripts/outputs/
├── unconst-node-ddpm/    # Unconstrained Stage 1
├── unconst-edge/         # Unconstrained Stage 2
├── topo-ae/              # Topology Transformer autoencoder
├── topo-node-ddpm/       # Topology-constrained Stage 1
├── topo-edge/            # Topology-constrained Stage 2
├── boun-cnn-ae/          # Boundary CNN autoencoder
├── boun-node-ddpm/       # Boundary-constrained Stage 1
└── boun-edge/            # Boundary-constrained Stage 2
```

See `docs/pretrained-models.md` for detailed I/O specs per model.

## Data Pipeline

RPLAN data (80,788 PNG images) must be obtained from http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html and preprocessed:

```bash
cd datasets
python rplan-extract.py          # Extract structural graphs (→ 71,763 valid samples)
python rplan-process{1..4}.py    # Process to rplang-v3-withsemantics/
python rplan-process{5..7}.py    # Boundary data → rplang-v3-withsemantics-withboundary/
python rplan-process{8..10}.py   # Bubble diagrams → rplang-v3-bubble-diagram/
python move.py                   # Split into train(65763)/val(3000)/test(3000)
```

Expected dataset structure: `datasets/rplang-v3-{withsemantics,withsemantics-withboundary,bubble-diagram}/{train,val,test}/*.npy`

## Git

- Do NOT add `Co-Authored-By: Claude` or any Claude attribution to commit messages.

## Conventions

- **Path config**: Each script has `sys.path.append()` lines at the top that must match local installation path.
- **Device config**: `device = 'cuda:0'` hardcoded near top of each script (marked "Modify it yourself").
- **Diffusion**: 1000 steps, cosine beta schedule, AdamW optimizer, lr=1e-4, batch_size=256 (train).
- **Room semantics**: 7 categories — 0:Living, 1:Bedroom, 2:Bathroom, 3:Kitchen, 4:Balcony, 5:Storage, 6:ExternalWall.
- **Rendering**: `gsdiff/utils.py` contains all visualization and graph geometry functions. Cycle detection converts edge graphs to room polygons for rendering.
- **LIFULL dataset**: Alternate dataset with separate scripts (`*-lifull.py`) and loaders (`datasets/lifull*.py`).
