# RPLAN to GRAN Dataset Conversion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Scope note:** This pipeline is **standalone and lives entirely inside `GRAN/`**. It has no dependency on any other project's preprocessing. Raw RPLAN PNG files are read directly and converted to GRAN's training format in one self-contained pipeline.

**Goal:** Build a standalone RPLAN-to-GRAN data pipeline that reads raw RPLAN PNG images and produces GRAN training data, **preserving edge type information** (wall vs door) so future GRAN variants can predict edge types without re-processing the raw images again.

**Architecture:** Single-pass pipeline that reads raw RPLAN PNGs, extracts (1) room polygons with semantic labels, (2) room adjacency, and (3) **edge types** (wall-only vs wall-with-door). Output is saved as a **3-value adjacency matrix** (0=no-adj, 1=wall, 2=door) plus room types. Initial GRAN training uses the binary form (`matrix > 0`) with zero model changes; future v4 (edge type prediction) upgrades to full 3-class without touching the data pipeline again.

**Tech Stack:** Python 3.10, numpy, networkx, opencv-python, shapely, scikit-image

**Why bake in edge types now:** We only want to touch RPLAN raw data once. Building the pipeline without edge types and later re-running it over 80,788 raw images is wasteful. Extracting edge types during the first pass costs <10% extra compute and future-proofs the data.

---

## Source Data: Raw RPLAN PNG Images

**Location:** `GRAN/data/rplan_raw/*.png` (user places ~80,788 files here before running the pipeline)

Each PNG has **4 channels** (loaded via `cv2.imread(path, -1)`):

| Channel | Content | Pixel value meaning |
|---------|---------|--------------------|
| 0 (B) | Room category | 0-12 = room types; 13 = front door; 14-16 = walls/structure |
| 1 (G) | Room instance ID | 0-N, different rooms of same type get different IDs |
| 2 (R) | Inside/outside mask | 255 = inside, 0 = outside |
| 3 (A) | Front door location | Non-zero only at entrance |

**Room category mapping** (channel 0, values 0-12 — reduced to 7 classes for GRAN):
```
0  Living room        -> GRAN class 0 (Living)
1  Master bedroom     -> GRAN class 1 (Bedroom)
2  Kitchen            -> GRAN class 3 (Kitchen)
3  Bathroom           -> GRAN class 2 (Bathroom)
4  Dining room        -> GRAN class 0 (merged with Living)
5  Child room         -> GRAN class 1 (merged with Bedroom)
6  Study room         -> GRAN class 1 (merged with Bedroom)
7  Second room        -> GRAN class 1 (merged with Bedroom)
8  Guest room         -> GRAN class 1 (merged with Bedroom)
9  Balcony            -> GRAN class 4 (Balcony)
10 Entrance           -> GRAN class 0 (merged with Living)
11 Storage            -> GRAN class 5 (Storage)
12 Wall-in closet     -> GRAN class 5 (merged with Storage)
13 Front door marker  -> used for entrance detection only
14-16 Walls/structure -> used for wall mask, NOT a room
```

### How edge types are detected from pixels

For each pair of adjacent rooms `(A, B)`:

1. **Find shared boundary**: Use `scipy.ndimage.binary_dilation` on each room mask, intersect them to get the shared wall pixels
2. **Check for door opening**: A door is a **gap in the wall mask** between two rooms. Specifically:
   - Sample points along the shared boundary
   - At each point, check if the pixel in channel 0 is a **wall pixel (14-16)** or a **room pixel (0-12)**
   - If >threshold% of the boundary is wall → edge type = 1 (wall only)
   - If some portion is room pixels (meaning the rooms "bleed into" each other at the doorway) → edge type = 2 (door)
3. **No shared boundary** → no adjacency (edge type 0)

**Heuristic threshold:** If along the shared boundary of length L pixels, ≥ 3 consecutive pixels belong to a room (not wall), consider it a door. Tunable constant `DOOR_GAP_MIN_PX = 3`.

### Room type distribution (after merging)

7 final classes for GRAN:

| GRAN class | Label | Expected frequency |
|-----------|-------|-------------------|
| 0 | Living (incl. dining, entrance) | ~20% |
| 1 | Bedroom (incl. master, child, study, guest, second) | ~40% |
| 2 | Bathroom | ~15% |
| 3 | Kitchen | ~12% |
| 4 | Balcony | ~10% |
| 5 | Storage (incl. closet) | ~3% |
| 6 | ExternalWall (reserved, not used for rooms) | 0% |

### Graph statistics (expected after processing all 80,788 images)

- Valid floorplans: ~71,000 (some raw PNGs have parsing errors)
- Nodes per graph: 4-12 (most 5-8)
- Edges per graph: 4-14 (most 5-9)
- Edge type distribution: ~60% wall, ~40% door (doors are roughly between functional areas)

## Target Data: GRAN TU-Dataset Format

GRAN loads graph datasets via `graph_load_batch()` which expects TU-format text files:

```
data/{DATASET_NAME}/
├── {NAME}_A.txt              # Edge list: "src, dst\n" (1-indexed)
├── {NAME}_node_labels.txt    # Node label per line (1 per node)
├── {NAME}_graph_indicator.txt # Which graph each node belongs to (1-indexed)
├── {NAME}_graph_labels.txt   # Graph-level label per line
└── {NAME}_node_attributes.txt # (optional) Node feature vector per line
```

Alternatively, GRAN can load a pickle of `List[nx.Graph]` via `save_graph_list()` / `load_graph_list()`.

**We'll use both:** TU-format for compatibility, plus a pickle for direct loading.

---

## Task 1: Conversion Script

**Files:**
- Create: `GRAN/scripts/convert_rplan_to_gran.py`
- Create: `GRAN/data/RPLAN/` (output directory)

- [ ] **Step 1: Write the conversion script**

```python
# scripts/convert_rplan_to_gran.py
"""Convert RPLAN bubble diagrams to GRAN TU-dataset format.

Usage:
    cd GRAN
    uv run python scripts/convert_rplan_to_gran.py \
        --rplan_dir ../datasets/rplang-v3-bubble-diagram \
        --output_dir data/RPLAN \
        --pickle_path data/rplan_graphs.p

Reads: {rplan_dir}/{train,val,test}/*.npy
Writes: TU-format files in output_dir + networkx pickle
"""
import os
import sys
import argparse
import numpy as np
import networkx as nx
import pickle
from glob import glob
from tqdm import tqdm


def load_bubble_diagram(npy_path):
    """Load a single RPLAN bubble diagram .npy file.

    Returns:
        nx.Graph with node attrs: 'attr' (room type int), 'centroid' (x, y)
        or None if invalid
    """
    data = np.load(npy_path, allow_pickle=True).item()

    adj = data['adjacency_matrix']
    semantics = data['semantics']
    centroids = data.get('centroids', None)

    n_rooms = adj.shape[0]
    if n_rooms < 2:
        return None

    G = nx.from_numpy_array(adj)

    # Add node attributes
    for i in range(n_rooms):
        G.nodes[i]['attr'] = int(semantics[i])
        if centroids is not None:
            G.nodes[i]['centroid'] = centroids[i].tolist()

    # Remove self-loops if any
    G.remove_edges_from(nx.selfloop_edges(G))

    # Only keep largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)

    return G


def save_tu_format(graphs, output_dir, name='RPLAN'):
    """Save list of nx.Graph objects in TU-dataset format."""
    os.makedirs(output_dir, exist_ok=True)

    edge_list = []        # (src, dst) 1-indexed global
    node_labels = []      # one per node
    node_attrs = []       # centroid features (optional)
    graph_indicators = [] # which graph each node belongs to
    graph_labels = []     # graph-level labels (dummy: number of rooms)

    node_offset = 0
    for g_idx, G in enumerate(graphs):
        n = G.number_of_nodes()
        graph_labels.append(n)  # use room count as graph label

        for node in range(n):
            node_labels.append(G.nodes[node].get('attr', 0))
            graph_indicators.append(g_idx + 1)  # 1-indexed

            centroid = G.nodes[node].get('centroid', [0.0, 0.0])
            node_attrs.append(centroid)

        for u, v in G.edges():
            # TU format: 1-indexed, both directions for undirected
            edge_list.append((u + node_offset + 1, v + node_offset + 1))
            edge_list.append((v + node_offset + 1, u + node_offset + 1))

        node_offset += n

    # Write files
    with open(os.path.join(output_dir, f'{name}_A.txt'), 'w') as f:
        for src, dst in edge_list:
            f.write(f'{src}, {dst}\n')

    with open(os.path.join(output_dir, f'{name}_node_labels.txt'), 'w') as f:
        for label in node_labels:
            f.write(f'{label}\n')

    with open(os.path.join(output_dir, f'{name}_graph_indicator.txt'), 'w') as f:
        for ind in graph_indicators:
            f.write(f'{ind}\n')

    with open(os.path.join(output_dir, f'{name}_graph_labels.txt'), 'w') as f:
        for label in graph_labels:
            f.write(f'{label}\n')

    with open(os.path.join(output_dir, f'{name}_node_attributes.txt'), 'w') as f:
        for attr in node_attrs:
            f.write(', '.join(f'{x:.6f}' for x in attr) + '\n')

    print(f'Saved TU-format to {output_dir}/')
    print(f'  Nodes: {len(node_labels)}')
    print(f'  Edges: {len(edge_list) // 2} (undirected)')
    print(f'  Graphs: {len(graph_labels)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rplan_dir', type=str,
                        default='../datasets/rplang-v3-bubble-diagram',
                        help='Path to RPLAN bubble diagram directory')
    parser.add_argument('--output_dir', type=str, default='data/RPLAN',
                        help='Output directory for TU-format files')
    parser.add_argument('--pickle_path', type=str, default='data/rplan_graphs.p',
                        help='Output path for networkx pickle')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'val', 'test'],
                        help='Which splits to convert')
    parser.add_argument('--min_nodes', type=int, default=3,
                        help='Minimum number of rooms')
    parser.add_argument('--max_nodes', type=int, default=20,
                        help='Maximum number of rooms')
    args = parser.parse_args()

    all_graphs = []
    split_counts = {}

    for split in args.splits:
        split_dir = os.path.join(args.rplan_dir, split)
        if not os.path.isdir(split_dir):
            print(f'Warning: {split_dir} not found, skipping')
            continue

        npy_files = sorted(glob(os.path.join(split_dir, '*.npy')))
        print(f'Processing {split}: {len(npy_files)} files')

        count = 0
        for f in tqdm(npy_files, desc=split):
            G = load_bubble_diagram(f)
            if G is None:
                continue
            n = G.number_of_nodes()
            if n < args.min_nodes or n > args.max_nodes:
                continue
            all_graphs.append(G)
            count += 1

        split_counts[split] = count

    print(f'\nTotal valid graphs: {len(all_graphs)}')
    for split, count in split_counts.items():
        print(f'  {split}: {count}')

    # Statistics
    num_nodes = [G.number_of_nodes() for G in all_graphs]
    num_edges = [G.number_of_edges() for G in all_graphs]
    print(f'\nNode stats: min={min(num_nodes)}, max={max(num_nodes)}, '
          f'mean={np.mean(num_nodes):.1f}')
    print(f'Edge stats: min={min(num_edges)}, max={max(num_edges)}, '
          f'mean={np.mean(num_edges):.1f}')

    # Semantic distribution
    all_attrs = []
    for G in all_graphs:
        for n in G.nodes():
            all_attrs.append(G.nodes[n].get('attr', 0))
    unique, counts = np.unique(all_attrs, return_counts=True)
    print('\nRoom type distribution:')
    room_names = ['Living', 'Bedroom', 'Bathroom', 'Kitchen',
                  'Balcony', 'Storage', 'ExternalWall']
    for u, c in zip(unique, counts):
        name = room_names[u] if u < len(room_names) else f'Type{u}'
        print(f'  {name} ({u}): {c} ({100*c/sum(counts):.1f}%)')

    # Save TU format
    save_tu_format(all_graphs, args.output_dir, name='RPLAN')

    # Save pickle (for direct GRAN loading)
    os.makedirs(os.path.dirname(args.pickle_path), exist_ok=True)
    with open(args.pickle_path, 'wb') as f:
        pickle.dump(all_graphs, f)
    print(f'\nSaved pickle to {args.pickle_path}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the conversion**

```bash
cd D:/Github/GSDiff/GRAN
uv run python scripts/convert_rplan_to_gran.py \
    --rplan_dir ../datasets/rplang-v3-bubble-diagram \
    --output_dir data/RPLAN \
    --pickle_path data/rplan_graphs.p
```

Expected output:
```
Processing train: 65763 files
Processing val: 3000 files
Processing test: 3000 files
Total valid graphs: ~70000
Node stats: min=3, max=8, mean=~6
Room type distribution:
  Living (0): ... (15%)
  Bedroom (1): ... (38%)
  ...
Saved TU-format to data/RPLAN/
Saved pickle to data/rplan_graphs.p
```

- [ ] **Step 3: Commit**

```bash
git add scripts/convert_rplan_to_gran.py
git commit -m "feat: add RPLAN bubble diagram to GRAN format converter"
```

---

## Task 2: Register RPLAN in GRAN's Data Loader

**Files:**
- Modify: `utils/data_helper.py`

- [ ] **Step 1: Add RPLAN to create_graphs()**

Add the following elif block in `create_graphs()` after the FIRSTMM_DB block:

```python
  elif graph_type == 'RPLAN':
    # Load from pickle (converted bubble diagrams)
    pickle_path = os.path.join(data_dir, 'rplan_graphs.p')
    if os.path.exists(pickle_path):
      with open(pickle_path, 'rb') as f:
        graphs = pickle.load(f)
      # Ensure integer node labels and no self-loops
      for i in range(len(graphs)):
        graphs[i] = nx.convert_node_labels_to_integers(graphs[i])
        graphs[i].remove_edges_from(nx.selfloop_edges(graphs[i]))
    else:
      # Fallback: load TU format
      graphs = graph_load_batch(
          data_dir,
          min_num_nodes=3,
          max_num_nodes=20,
          name='RPLAN',
          node_attributes=True,
          graph_labels=True)
```

- [ ] **Step 2: Commit**

```bash
git add utils/data_helper.py
git commit -m "feat: register RPLAN dataset in GRAN data loader"
```

---

## Task 3: RPLAN Config Files

**Files:**
- Create: `config/gran_rplan.yaml` (original GRAN on RPLAN)
- Create: `config/gran_v2_rplan.yaml` (GRANv2 with room attributes on RPLAN)

- [ ] **Step 1: Create configs**

```yaml
# config/gran_rplan.yaml
---
exp_name: GRAN
exp_dir: exp/GRAN
runner: GranRunner
use_horovod: false
use_gpu: true
device: cuda:0
gpus: [0]
seed: 1234
dataset:
  loader_name: GRANData
  name: RPLAN
  data_path: data/
  node_order: DFS
  train_ratio: 0.8
  dev_ratio: 0.2
  num_subgraph_batch: 50
  num_fwd_pass: 1
  has_node_feat: false
  is_save_split: false
  is_sample_subgraph: true
  is_overwrite_precompute: false
model:
  name: GRANMixtureBernoulli
  num_mix_component: 20
  is_sym: true
  block_size: 1
  sample_stride: 1
  max_num_nodes: 20   # RPLAN rooms max ~8, pad to 20 for safety
  hidden_dim: 128
  embedding_dim: 128
  num_GNN_layers: 4   # smaller graphs need fewer layers
  num_GNN_prop: 1
  num_canonical_order: 1
  dimension_reduce: true
  has_attention: true
  edge_weight: 1.0e+0
train:
  optimizer: Adam
  lr_decay: 0.3
  lr_decay_epoch: [100000000]
  num_workers: 4
  max_epoch: 3000
  batch_size: 32
  display_iter: 10
  snapshot_epoch: 100
  valid_epoch: 50
  lr: 1.0e-4
  wd: 0.0e-4
  momentum: 0.9
  shuffle: true
  is_resume: false
  resume_epoch: 5000
  resume_dir:
  resume_model: model_snapshot_0005000.pth
test:
  batch_size: 100
  num_workers: 0
  num_test_gen: 3000
  is_vis: true
  is_single_plot: false
  is_test_ER: false
  num_vis: 20
  vis_num_row: 5
  better_vis: true
  test_model_dir: snapshot_model
  test_model_name: gran_rplan.pth
```

```yaml
# config/gran_v2_rplan.yaml
---
exp_name: GRANv2
exp_dir: exp/GRANv2
runner: GranRunnerV2
use_horovod: false
use_gpu: true
device: cuda:0
gpus: [0]
seed: 1234
dataset:
  loader_name: GRANDataV2
  name: RPLAN
  data_path: data/
  node_order: DFS
  train_ratio: 0.8
  dev_ratio: 0.2
  num_subgraph_batch: 50
  num_fwd_pass: 1
  has_node_feat: false
  is_save_split: false
  is_sample_subgraph: true
  is_overwrite_precompute: false
model:
  name: GRANv2
  num_mix_component: 20
  is_sym: true
  block_size: 1
  sample_stride: 1
  max_num_nodes: 20
  hidden_dim: 128
  embedding_dim: 128
  num_GNN_layers: 4
  num_GNN_prop: 1
  num_canonical_order: 1
  dimension_reduce: true
  has_attention: true
  edge_weight: 1.0e+0
  num_attr_classes: 7    # 7 room types
  use_gatv2: true
  gatv2_num_heads: 4
train:
  optimizer: Adam
  lr_decay: 0.3
  lr_decay_epoch: [100000000]
  num_workers: 4
  max_epoch: 3000
  batch_size: 32
  display_iter: 10
  snapshot_epoch: 100
  valid_epoch: 50
  lr: 1.0e-4
  wd: 0.0e-4
  momentum: 0.9
  shuffle: true
  is_resume: false
  resume_epoch: 5000
  resume_dir:
  resume_model: model_snapshot_0005000.pth
  lambda_attr: 1.0
test:
  batch_size: 100
  num_workers: 0
  num_test_gen: 3000
  is_vis: true
  is_single_plot: false
  is_test_ER: false
  num_vis: 20
  vis_num_row: 5
  better_vis: true
  test_model_dir: snapshot_model
  test_model_name: gran_v2_rplan.pth
```

- [ ] **Step 2: Commit**

```bash
git add config/gran_rplan.yaml config/gran_v2_rplan.yaml
git commit -m "feat: add RPLAN training configs for GRAN and GRANv2"
```

---

## Task 4: Validation — Verify Converted Data Loads

- [ ] **Step 1: Quick smoke test**

```bash
cd D:/Github/GSDiff/GRAN
uv run python -c "
from utils.data_helper import create_graphs
graphs = create_graphs('RPLAN', data_dir='data/')
print(f'Loaded {len(graphs)} graphs')
print(f'Sample graph: {graphs[0].number_of_nodes()} nodes, {graphs[0].number_of_edges()} edges')
print(f'Node attrs: {[graphs[0].nodes[n].get(\"attr\", -1) for n in graphs[0].nodes()]}')
"
```

Expected:
```
Loaded ~70000 graphs
Sample graph: 6 nodes, 8 edges
Node attrs: [0, 1, 1, 3, 4, 5]
```

- [ ] **Step 2: Verify training can start (1 epoch)**

```bash
cd D:/Github/GSDiff/GRAN
uv run python run_exp.py -c config/gran_rplan.yaml
# Ctrl+C after first epoch displays loss
```

- [ ] **Step 3: Commit any fixes**

---

## Summary

| Step | What | Output |
|------|------|--------|
| Task 1 | Conversion script | `data/RPLAN/` TU files + `data/rplan_graphs.p` pickle |
| Task 2 | Register in data loader | `create_graphs('RPLAN')` works |
| Task 3 | Config files | `gran_rplan.yaml`, `gran_v2_rplan.yaml` |
| Task 4 | Smoke test | Verify loading + 1 epoch training |

### RPLAN vs Existing GRAN Datasets

| Property | RPLAN | Grid | DD | FIRSTMM_DB |
|----------|-------|------|----|------------|
| Nodes/graph | 4-8 | 100-400 | 100-500 | 50-5000 |
| Node attributes | 7 room types | None | None | Optional |
| Graph count | ~70,000 | 100 | 1,178 | 41 |
| Domain | Floorplan topology | Synthetic | Protein | Molecular |

RPLAN graphs are **much smaller but far more numerous** than existing datasets, so training should be fast with large batch sizes.
