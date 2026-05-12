# GSDiff

Official implementation of the AAAI 2025 paper: "GSDiff: Synthesizing Vector Floorplans via Geometry-enhanced Structural Graph Generation"

## Installation

Requires Python 3.10 and CUDA 11.8+. We recommend [uv](https://docs.astral.sh/uv/) for fast dependency management.

```bash
# Create virtual environment
uv venv --python 3.10 .venv

# Activate (choose your platform)
source .venv/bin/activate        # Linux / macOS
source .venv/Scripts/activate    # Windows (Git Bash)

# Install PyTorch with CUDA 11.8
uv pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
uv pip install networkx==3.1 numpy==1.26.0 opencv-python==4.9.0.80 scipy==1.10.1 \
  scikit-image==0.24.0 scikit-learn==1.4.1.post1 shapely==2.0.6 \
  tensorboardx==2.6.2.2 tqdm==4.65.0 matplotlib==3.9.2 pytorch-fid==0.3.0 pillow==10.0.1
```

> **Note:** PyTorch is ~2.4GB. If your default drive is low on space, set `UV_CACHE_DIR` to a path on another drive before installing.

## Data Preparation

### RPLAN Dataset

1. Create folder `datasets/rplandata/Data`.
2. Download the 80,788 RPLAN dataset (http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html). It contains a `floorplan_dataset` folder. Place this `floorplan_dataset` folder under `datasets/rplandata/Data`.
3. Run the preprocessing pipeline from the `datasets/` directory:

```bash
cd datasets

# Extract structural graphs (80,788 → 71,763 valid samples)
uv run python rplan-extract.py        # original version
# or
uv run python rplan-extract-fast.py   # optimized version (~10x faster, same output)

# Process node/semantic data → rplang-v3-withsemantics/
uv run python rplan-process1.py
uv run python rplan-process2.py
uv run python rplan-process3.py
uv run python rplan-process4.py

# Process boundary data → rplang-v3-withsemantics-withboundary/
uv run python rplan-process5.py
uv run python rplan-process6.py
uv run python rplan-process7.py

# Process topology/bubble diagrams → rplang-v3-bubble-diagram/
uv run python rplan-process8.py
uv run python rplan-process9.py
uv run python rplan-process10.py

# Split into train(65,763) / val(3,000) / test(3,000)
uv run python move.py
```

**Which steps do I need?**
- **Unconstrained generation:** Steps 3-4 (extract + process1-4) + move.py
- **Topology-constrained:** Above + process8-10
- **Boundary-constrained:** Above + process5-7

After `rplan-extract.py`, intermediate folders (`1-channel-semantics-256`, `3-channel-semantics-256`, `bin_imgs`, `e_imgs`, etc.) are created under `datasets/rplandata/Data`. You can remove them after all processing is complete.

> **Note on topology data:** The semantics of bubble diagram GT involve randomness. Due to RPLAN dataset terms, we cannot release any part of it. The bubble diagram GT semantics extracted using the provided scripts may differ slightly from our experimental results. However, given the large data scale, the bias should be minor. Alternatively, you can use `get_cycle_basis_and_semantic_3_semansimplified` instead of `get_cycle_basis_and_semantic_2_semansimplified` in `rplan-process8/9/10.py` for deterministic room semantics extraction.

### LIFULL Dataset

Create path `datasets/lifulldata` and follow the data request process of [Raster-to-Graph](https://github.com/SizheHu/Raster-to-Graph) to place the data there. The data contains 10,804 images and corresponding annotations.

### MSD Dataset for DiGress

The DiGress MSD baseline uses the Kaggle Modified Swiss Dwellings dataset and trains an unconditional graph generator.

Authenticate Kaggle first, then run from `digress/`:

```powershell
cd digress

# Quick start: prepare the wall-augmented graph dataset.
# Defaults: wall_contact_eps=0.01, wall_min_contact_length=0.02, wall_segment_gap=0.45.
.\.venv\Scripts\python.exe scripts\download_msd_dataset.py --prepare --add-wall-edges

# Train the wall-augmented unconditional DiGress graph model.
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall.yaml

# Resume training from a checkpoint if needed.
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall.yaml general.resume="outputs/<run>/checkpoints/msd_wall/last.ckpt"

# Generate wall-augmented test graphs and the combined sample PNG.
.\.venv\Scripts\python.exe scripts\test_graph_generation.py msd_wall
```

Outputs:
- Raw Kaggle data: `datasets/msd/raw/modified-swiss-dwellings-v2/`
- Wall-augmented graph pickle: `digress/data/msd_wall/graphs.p`
- Dataset statistics: `digress/data/msd_wall/dataset_stats.json`
- Dataset visual checks: `digress/data/msd_wall/sample_graphs.png` and `digress/data/msd_wall/vis/`
- Training checkpoints and curves: `digress/outputs/.../checkpoints/msd_wall/`

The wall preprocessing command overwrites `digress/data/msd_wall/*` and clears its processed cache. It does not delete `datasets/msd/raw/*`.

## Pretrained Weights

Download and extract into `scripts/outputs/`:

| Model | Link |
|-------|------|
| Unconstrained (node + edge) | [Google Drive](https://drive.google.com/file/d/15gM0GtW2GwHmlpz0r-rpvo-k-BlNy_gu/view?usp=sharing) |
| Topology-constrained (node + edge) | [Google Drive](https://drive.google.com/file/d/1pk7SmvLZ8ON3OUL3SNxPRu73ndVKru0z/view?usp=sharing) |
| Boundary-constrained (node + edge) | [Google Drive](https://drive.google.com/file/d/1puqxXIW4Y7AeQHFuC76PlYpWQm6MD8PS/view?usp=sharing) |
| Boundary CNN autoencoder | [Google Drive](https://drive.google.com/file/d/1l6QRpfX5Jtucg3R995HajlwRG8SewUJW/view?usp=sharing) |
| Topology Transformer autoencoder | [Google Drive](https://drive.google.com/file/d/1tExX8LdrFpJfBQH5y2emC6BltBwf9tHx/view?usp=sharing) |

**LIFULL weights:**

| Model | Link |
|-------|------|
| Node | [Google Drive](https://drive.google.com/file/d/1k_q9-vQXbs3PDzLxvz-tQRvO3j0DzlPN/view?usp=sharing) |
| Edge | [Google Drive](https://drive.google.com/file/d/1XkoMZAMOeBPTteUTVDukgc4BNoEEJSXS/view?usp=sharing) |

Expected directory layout after extraction:

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

See [docs/pretrained-models.md](docs/pretrained-models.md) for detailed I/O specifications per model.

## Usage

### Configuration

Before running any script, update two things at the top of each file:

1. **`sys.path.append()`** lines — change to your local project path
2. **`device = 'cuda:0'`** — change to match your GPU

### Testing (Inference)

All test scripts run from `scripts/`:

```bash
cd scripts

# Unconstrained generation (3,000 samples, run 5x for averaging)
uv run python test_main.py

# Topology-constrained (757 samples)
uv run python test_topo.py

# Boundary-constrained (378 samples)
uv run python test_boun.py
```

**Evaluation details:**
- **Unconstrained:** Uses the original 3,000 test results, run 5 times to get the average.
- **Topology-constrained:** Intersection of 3,000 results with HouseDiffusion and House-GAN++ test sets → 757 samples. Sample numbers are in line 183 of `evalmetric-topoconstrain-ged-roomnumber.py`.
- **Boundary-constrained:** Same intersection → 378 samples. Sample numbers are in line 9 of `evalmetric-boun-constrain-fid-kid.py`.

### Training

```bash
cd scripts

# Unconstrained
uv run python trainval_main_unconstrained.py       # Stage 1: node generation
uv run python trainval_main_edge_unconstrained.py  # Stage 2: edge prediction

# Topology-constrained (train autoencoders first)
uv run python train-TopoTransformer-autoe.py       # Topology encoder phase 1
uv run python train-TopoTransformer-autoe-final.py # Topology encoder phase 2
uv run python trainval_main_topo.py                # Stage 1
uv run python trainval_main_edge_topo.py           # Stage 2

# Boundary-constrained (train autoencoder first)
uv run python train-CNN-autoe-final.py             # Boundary CNN encoder
uv run python trainval_main_boun.py                # Stage 1
uv run python trainval_main_edge_boun.py           # Stage 2
```

### Evaluation

Run from project root:

```bash
uv run python evalmetric-no-constrain-fid-kid.py
uv run python evalmetric-no-constrain-geometry-topological-metrics.py
uv run python evalmetric-topoconstrain-ged-roomnumber.py
uv run python evalmetric-boun-constrain-fid-kid.py
```

### LIFULL Dataset

All LIFULL training and testing scripts have `lifull` in their file names. The purpose of each script is stated at the top of the file.

## Web App (Demo)

A web-based demo for interactive floorplan generation. No dataset required — uses pretrained models directly.

### Setup

```bash
# Frontend dependencies (from app/ directory)
cd app
npm install

# Backend dependencies (uses the project .venv)
cd app/backend
source ../../.venv/Scripts/activate    # Windows (Git Bash)
# source ../../.venv/bin/activate      # Linux / macOS
uv pip install fastapi uvicorn
```

### Running

```bash
cd app

# Start both frontend + backend together
npm run dev:all

# Or separately:
npm run dev        # Frontend only (port 3000)
npm run backend    # Backend only (port 8000)
```

Open http://localhost:3000 in your browser.

### Features

- **Unconstrained mode:** Generate random floorplans with one click
- **Topology mode:** Define room types (4-8 rooms) and adjacency relationships, then generate a matching floorplan
- Generation takes ~30 seconds per floorplan on a 3090
