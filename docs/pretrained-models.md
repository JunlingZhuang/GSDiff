# GSDiff Pretrained Models Reference

## Architecture Overview

GSDiff uses a **two-stage pipeline** for floorplan generation:

```
Stage 1 (Node Generation)          Stage 2 (Edge Prediction)
Noise → Denoise → Corners + Sem  →  Corners → Predict Edges → Full Floorplan
```

- **Stage 1** generates room corner coordinates and semantic labels via a denoising diffusion Transformer.
- **Stage 2** predicts wall connectivity (edges) between corners via a separate Transformer with random self-supervision.

Both stages are required to produce a complete floorplan. Stage 2 takes Stage 1's output directly and constructs its own attention masks — it does not depend on the dataset.

Three generation modes share this two-stage structure, differing only in whether Stage 1 receives an additional condition encoding.

---

## Model Inventory

### Unconstrained Generation

No external condition. Generates floorplans from pure noise.

| Directory | File | Stage | Role |
|-----------|------|-------|------|
| `unconst-node-ddpm/` | `model1000000.pt` | 1 | Node denoising diffusion Transformer |
| `unconst-edge/` | `model_stage2_best_061000.pt` | 2 | Edge prediction Transformer |
| `unconst-edge-selfsup/` | `model_stage2_best_010300.pt` | 2 | Edge prediction variant (simplified self-supervision) |

### Topology-Constrained Generation

Conditioned on a **bubble diagram** (room types + adjacency graph).

| Directory | File | Stage | Role |
|-----------|------|-------|------|
| `topo-ae/` | `model_stage0_best_006000.pt` | Encoder | Topology Transformer autoencoder |
| `topo-node-ddpm/` | `model1000000.pt` | 1 | Topology-conditioned node diffusion Transformer |
| `topo-edge/` | `model_stage2_best_076000.pt` | 2 | Topology-conditioned edge prediction Transformer |

### Boundary-Constrained Generation

Conditioned on a **boundary image** (building outline).

| Directory | File | Stage | Role |
|-----------|------|-------|------|
| `boun-cnn-ae/` | `model_stage0_best_006700.pt` | Encoder | Boundary CNN autoencoder |
| `boun-node-ddpm/` | `model1000000.pt` | 1 | Boundary-conditioned node diffusion Transformer |
| `boun-edge/` | `model_stage2_best_065000.pt` | 2 | Boundary-conditioned edge prediction Transformer |

### Shared / Auxiliary

| Directory | File | Role |
|-----------|------|------|
| `shared-edge-base/` | `model_stage2_best_010300.pt` | Base edge predictor used during training validation |
| `topo-ae-ckpt/` | `model011000.pt` | Intermediate checkpoint of topology autoencoder training |

---

## Detailed I/O Specification

### Stage 1: Node Generation Models

All Stage 1 models share the same output format. They differ in input.

**Output (all modes):**
- Corner coordinates: `(batch, 53, 2)` — normalized to `[-1, 1]`
- Semantic labels: `(batch, 53, 7)` — one-hot over 7 room categories
- Padding flag: `(batch, 53, 1)` — `0` = real node, `1` = padding

**Common input:**
- `global_attn_matrix`: `(batch, 53, 53)` bool — attention mask indicating which nodes are real (from dataset during training/testing)
- `t`: `(batch,)` int — diffusion timestep

#### `unconst-node-ddpm` — HeterHouseModel

```
Input:
  x_t:               (batch, 53, 10)    Noisy node features [coords(2) + semantics(7) + padding_flag(1)]
  global_attn_matrix: (batch, 53, 53)   Attention mask (from dataset)
  t:                  (batch,)          Diffusion timestep

Output:
  noise_coords:       (batch, 53, 2)    Predicted noise for coordinates
  noise_semantics:    (batch, 53, 8)    Predicted noise for semantics + padding flag

Constraint: None
Dataset dependency: Needs global_attn_matrix from preprocessed data to know per-sample node count
```

#### `topo-node-ddpm` — TopoHeterHouseModel

```
Input:
  x_t:               (batch, 53, 10)    Noisy node features
  global_attn_matrix: (batch, 53, 53)   Attention mask (from dataset)
  t:                  (batch,)          Diffusion timestep
  topo_latent:        (batch, D)        Latent from topo-ae encoder

Output:
  noise_coords:       (batch, 53, 2)    Predicted noise for coordinates
  noise_semantics:    (batch, 53, 8)    Predicted noise for semantics + padding flag

Constraint: Bubble diagram → encoded by topo-ae
Dataset dependency: Needs bubble diagram .npy files (room semantics + adjacency matrix)
```

#### `boun-node-ddpm` — BoundHeterHouseModel

```
Input:
  x_t:               (batch, 53, 10)    Noisy node features
  global_attn_matrix: (batch, 53, 53)   Attention mask (from dataset)
  t:                  (batch,)          Diffusion timestep
  boun_feature:       (batch, 1024, 16, 16)  Feature map from boun-cnn-ae encoder

Output:
  noise_coords:       (batch, 53, 2)    Predicted noise for coordinates
  noise_semantics:    (batch, 53, 8)    Predicted noise for semantics + padding flag

Constraint: Boundary image → encoded by boun-cnn-ae
Dataset dependency: Needs boundary images + pre-extracted CNN feature maps
```

### Stage 2: Edge Prediction Models

All Stage 2 models take Stage 1 output as input. They construct their own masks from Stage 1 results and do **not** depend on the dataset.

#### `unconst-edge` — EdgeModel

```
Input:
  corners:            (batch, 53, 2)    Corner coordinates (from Stage 1, re-padded to 53)
  global_attn_matrix: (batch, 53, 53)   Constructed from Stage 1 output node count
  padding_mask:       (batch, 53, 1)    Constructed from Stage 1 output node count
  semantics:          (batch, 53, 7)    Semantic labels (from Stage 1)

Output:
  edges:              (batch, 53*53, 2) Edge predictions [not-connected, connected] logits
  (+ 2 auxiliary outputs used during training only)

Constraint: None
Dataset dependency: None — uses Stage 1 output only
```

#### `topo-edge` — TopoEdgeModel

```
Input:  Same as unconst-edge
Output: Same as unconst-edge

Constraint: Implicitly constrained (operates on topology-conditioned Stage 1 output)
Dataset dependency: None — uses Stage 1 output only
```

#### `boun-edge` — BoundEdgeModel

```
Input:  Same as unconst-edge
Output: Same as unconst-edge

Constraint: Implicitly constrained (operates on boundary-conditioned Stage 1 output)
Dataset dependency: None — uses Stage 1 output only
```

### Autoencoders (Condition Encoders)

These are frozen during generation. They encode external constraints into latent representations consumed by Stage 1.

#### `topo-ae` — TopoGraphModel

```
Input:
  semantics:          (max_rooms, 7)    One-hot room type labels (padded to 8 rooms)
  adjacency_matrix:   (max_rooms, max_rooms)  Binary adjacency matrix

Output:
  latent:             (batch, D)        Latent representation of the bubble diagram

Source data: rplang-v3-bubble-diagram/{split}/*.npy
  Each .npy contains: { 'semantics': [int list], 'adjacency_matrix': [[int]] }
```

#### `boun-cnn-ae` — Boundary CNN Autoencoder

```
Input:
  boundary_image:     (batch, 1, 256, 256)  Grayscale boundary image

Output (encoder only, used for generation):
  feature_map_16:     (batch, 1024, 16, 16)  Bottleneck feature map

Source data: rplang-v3-withsemantics-withboundary/{split}/*.npy
  Pre-extracted features cached in: prerunning_cnn_featuremaps/*.npy
```

---

## Inference Pipeline Summary

### Unconstrained

```
                    ┌─────────────────────┐
   Noise ──────────►│  unconst-node-ddpm  │──── corners + semantics ────►┌──────────────┐
   attn_mask ──────►│  (1000 steps)       │                              │ unconst-edge │──► edges ──► render
                    └─────────────────────┘                              └──────────────┘
```

**Dataset needed for:** `global_attn_matrix` (Stage 1 only)

### Topology-Constrained

```
   Bubble diagram ──►┌─────────┐
   (rooms + adj)     │ topo-ae │──── topo_latent ────┐
                     └─────────┘                     │
                    ┌────────────────────────────┐   │
   Noise ──────────►│  topo-node-ddpm            │◄──┘  corners + sem ──►┌────────────┐
   attn_mask ──────►│  (1000 steps, conditioned) │─────────────────────►│ topo-edge  │──► edges ──► render
                    └────────────────────────────┘                      └────────────┘
```

**Dataset needed for:** bubble diagram .npy (encoder input) + `global_attn_matrix` (Stage 1)

### Boundary-Constrained

```
   Boundary image ──►┌─────────────┐
   (256x256)         │ boun-cnn-ae │──── feature_map ────┐
                     └─────────────┘                     │
                    ┌────────────────────────────┐       │
   Noise ──────────►│  boun-node-ddpm            │◄──────┘  corners + sem ──►┌────────────┐
   attn_mask ──────►│  (1000 steps, conditioned) │────────────────────────►│ boun-edge  │──► edges ──► render
                    └────────────────────────────┘                         └────────────┘
```

**Dataset needed for:** boundary images / pre-extracted CNN features + `global_attn_matrix` (Stage 1)

---

## Room Semantic Categories

| Index | Category |
|-------|----------|
| 0 | Living Room / Dining Room / Function Room |
| 1 | Bedroom / Study / Children's Room / Guest Room |
| 2 | Bathroom / Toilet |
| 3 | Kitchen |
| 4 | Balcony |
| 5 | Storage |
| 6 | External Wall |

## Rendering Color Scheme (RGB)

| Semantic | Color |
|----------|-------|
| 0 — Living | (244, 241, 222) |
| 1 — Bedroom | (234, 182, 159) |
| 2 — Bathroom | (107, 112, 92) |
| 3 — Kitchen | (224, 122, 95) |
| 4 — Balcony | (95, 121, 123) |
| 5 — Storage | (242, 204, 143) |
| 6 — External | (0, 0, 0) |
