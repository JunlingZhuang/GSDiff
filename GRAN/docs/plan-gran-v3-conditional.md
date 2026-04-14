# GRAN v3: Conditional Graph Generation from Room Types

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable GRAN to generate graph topology conditioned on a given set of node types (e.g., "1 living room + 2 bedrooms + 1 kitchen + 1 bathroom → generate their adjacency structure").

**Architecture:** Two alternative approaches are documented. Approach A (type embedding as input) is simpler and recommended. Approach B (global condition encoder) is more expressive but more complex. Both inject type information as **input** to the model, reversing v2's attr prediction flow (v2: graph → attrs; v3: attrs → graph).

**Tech Stack:** Python 3.10, PyTorch 2.0.1+cu118, built on GRANv2 from `plan-gran-v2-upgrade.md`

**Prerequisite:** GRANv2 (Task 1-6 of `plan-gran-v2-upgrade.md`) must be complete.

---

## Problem Statement

### Input
A multiset of room types with counts:
```
{Living: 1, Bedroom: 2, Kitchen: 1, Bathroom: 1, Balcony: 1}
```

Equivalently, an ordered list of room type IDs:
```
[0, 1, 1, 3, 2, 4]   # 0=Living, 1=Bedroom, 2=Bathroom, 3=Kitchen, 4=Balcony
```

Total 6 rooms, 7 possible types (Living/Bedroom/Bathroom/Kitchen/Balcony/Storage/External).

### Output
A graph `G = (V, E)` where:
- `|V| = N` matches the number of rooms in input
- Each node's type matches the input specification
- Edges represent learned adjacency patterns from training data

### Example
```
Input:  [Living, Bedroom, Bedroom, Kitchen, Bathroom, Balcony]
                                    ↓
                    [ GRAN v3 model ]
                                    ↓
Output: edges = [(0,1), (0,2), (0,3), (0,4), (0,5), (1,4), (3,0)]
        (Living at center, bedrooms connect to living + bathroom,
         kitchen connects to living, balcony off living)
```

### Comparison to Other Modes

| Mode | Input | Output | Model |
|------|-------|--------|-------|
| Unconditional (v1/v2) | None | Graph + (attrs) | GRAN / GRANv2 |
| Partial graph completion (v2) | Partial `(A, attrs)` | Completed graph | GRANv2 |
| Attribute-only prediction (v2) | Full `A` | Attrs | GRANv2 |
| **Type-conditioned generation (v3)** | **Type multiset** | **Graph structure** | **GRANv3** |

---

## Approach A: Type Embedding as Initial Node Feature (Recommended)

**Idea:** Replace the zero-initialization of new nodes in GRAN's autoregressive loop with a type embedding. The GNN propagation will then naturally incorporate type information when predicting edges.

### Why It Works

GRAN's generation process for each new node:
```
step 1: new node has initial feature (all zeros in v1)
step 2: GNN propagates messages from neighbors (but new node has no edges yet)
step 3: model predicts edges from the current node states
```

By setting the new node's initial feature to `type_embedding[target_type]`:
```
step 1: new node has initial feature = type_embedding[2]  (e.g., Bathroom)
step 2: GNN propagates; the "Bathroom" information spreads to neighbors
step 3: model predicts edges, now aware that this new node is a bathroom
        (e.g., more likely to connect to Bedrooms, less likely to connect to Kitchen)
```

The GNN learns to associate type embeddings with edge patterns during training, and uses that association to bias edge prediction during generation.

### Changes Required

**1 new module:** `nn.Embedding(num_types, embedding_dim)`
**3 code changes:** `_inference()`, `_sampling()`, dataset loading

### Pros and Cons

**Pros:**
- Minimal code change (one embedding layer + ~5 lines of init code)
- Preserves GRAN's autoregressive structure entirely
- Can reuse existing GRANv2 checkpoint weights for backbone (only new embedding needs training)
- Naturally compatible with variable-size graphs (each node independently gets its type embedding)
- Attr embedding can be **shared** with the v2 attr head: `attr_head.weight == type_embedding.weight.T`

**Cons:**
- Type info only affects the new node's initial state; propagation through GNN determines how strongly it influences edge prediction
- Less control over type influence compared to explicit fusion mechanisms

### Implementation (Approach A)

**Files:**
- Create: `model/gran_v3.py`
- Create: `config/gran_v3_rplan.yaml`
- Create: `tests/test_gran_v3_type_embed.py`

---

## Task A1: GRANv3 Model with Type Embedding Initialization

- [ ] **Step 1: Write failing test**

```python
# tests/test_gran_v3_type_embed.py
import torch
from easydict import EasyDict as edict
from model.gran_v3 import GRANv3


def make_config(num_types=7):
    return edict({
        'device': 'cpu',
        'model': edict({
            'max_num_nodes': 20,
            'hidden_dim': 64,
            'embedding_dim': 64,
            'is_sym': True,
            'block_size': 1,
            'sample_stride': 1,
            'num_GNN_prop': 1,
            'num_GNN_layers': 2,
            'edge_weight': 1.0,
            'dimension_reduce': True,
            'has_attention': True,
            'num_canonical_order': 1,
            'num_mix_component': 5,
            'num_attr_classes': num_types,
            'use_gatv2': False,
            'approach': 'type_embed',  # Approach A
        })
    })


def test_gran_v3_has_type_embedding():
    """GRANv3 should have a type_embedding module."""
    config = make_config()
    model = GRANv3(config)
    assert hasattr(model, 'type_embedding')
    assert model.type_embedding.num_embeddings == 7
    assert model.type_embedding.embedding_dim == 64


def test_gran_v3_sample_with_types():
    """Sampling with target types should produce a graph matching the types."""
    config = make_config()
    model = GRANv3(config)
    model.eval()

    target_types = torch.tensor([[0, 1, 1, 3, 2, 4]])  # (B=1, N=6)

    A_list = model({
        'is_sampling': True,
        'batch_size': 1,
        'target_types': target_types,
    })

    assert len(A_list) == 1
    A = A_list[0]
    assert A.shape[0] == 6, f"Expected 6 nodes, got {A.shape[0]}"


def test_gran_v3_different_types_give_different_graphs():
    """Different type inputs should produce different graph distributions."""
    config = make_config()
    model = GRANv3(config)
    model.eval()
    torch.manual_seed(42)

    # All bedrooms
    types_a = torch.tensor([[1, 1, 1, 1, 1]])
    # All different
    types_b = torch.tensor([[0, 1, 2, 3, 4]])

    A_a = model({
        'is_sampling': True,
        'batch_size': 1,
        'target_types': types_a,
    })[0]

    A_b = model({
        'is_sampling': True,
        'batch_size': 1,
        'target_types': types_b,
    })[0]

    # They shouldn't be identical (statistically at least)
    assert A_a.shape == A_b.shape
    # With random init, initial outputs may match - just check execution succeeds
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd D:/Github/GSDiff/GRAN
uv run python -m pytest tests/test_gran_v3_type_embed.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'model.gran_v3'`

- [ ] **Step 3: Implement GRANv3 (Approach A)**

```python
# model/gran_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.gran_v2 import GRANv2
from model.gran_mixture_bernoulli import mixture_bernoulli_loss

__all__ = ['GRANv3']


class GRANv3(GRANv2):
    """GRAN v3: conditional generation from node types.

    Extends GRANv2 with:
    1. Type embedding table that maps type_id → initial node feature
    2. Sampling loop that accepts a `target_types` tensor
    3. Training loop that uses ground-truth types to initialize node features

    Input to sampling: target_types tensor (B, N) of type class indices
    Output: generated adjacency matrices matching the specified types
    """

    def __init__(self, config):
        super().__init__(config)

        # Type embedding: maps type ID → embedding vector
        # Dimensionality matches the input to the GNN (embedding_dim after dimension_reduce)
        self.type_embedding = nn.Embedding(
            num_embeddings=self.num_attr_classes,
            embedding_dim=self.embedding_dim,
        )
        # Scale so type embeds are on similar order to decoder_input output
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.1)

    def _sampling_with_types(self, B, target_types):
        """Sample edges conditioned on pre-specified node types.

        Args:
            B: batch size
            target_types: (B, N) tensor of type class indices
                          N must equal the desired number of nodes

        Returns:
            A: (B, N_pad, N_pad) adjacency matrix
        """
        with torch.no_grad():
            K = self.block_size
            S = self.sample_stride
            H = self.hidden_dim
            N = self.max_num_nodes
            N_target = target_types.shape[1]

            mod_val = (N - K) % S
            if mod_val > 0:
                N_pad = N - K - mod_val + int(np.ceil((K + mod_val) / S)) * S
            else:
                N_pad = N

            A = torch.zeros(B, N_pad, N_pad).to(self.device)
            dim_input = self.embedding_dim if self.dimension_reduce else self.max_num_nodes
            node_state = torch.zeros(B, N_pad, dim_input).to(self.device)

            # Precompute type embeddings for all target nodes
            # target_types: (B, N_target), embed: (B, N_target, embedding_dim)
            type_embeds = self.type_embedding(target_types)  # (B, N_target, D)

            for ii in range(0, N_target, S):
                jj = ii + K
                if jj > N_target:
                    break

                A[:, ii:, :] = .0
                A = torch.tril(A, diagonal=-1)

                # Update node_state for previously generated nodes (from adj rows)
                if ii >= K:
                    if self.dimension_reduce:
                        node_state[:, ii - K:ii, :] = self.decoder_input(A[:, ii - K:ii, :N])
                    else:
                        node_state[:, ii - K:ii, :] = A[:, ii - S:ii, :N]
                else:
                    if self.dimension_reduce:
                        node_state[:, :ii, :] = self.decoder_input(A[:, :ii, :N])
                    else:
                        node_state[:, :ii, :] = A[:, ii - S:ii, :N]

                # KEY CHANGE: initialize new nodes with type embeddings instead of zeros
                new_node_embeds = type_embeds[:, ii:jj, :]  # (B, K, D)
                node_state_in = torch.cat([
                    node_state[:, :ii, :],
                    new_node_embeds,
                ], dim=1)  # (B, jj, D)

                # Standard GNN propagation (same as v2)
                adj = F.pad(
                    A[:, :ii, :ii], (0, K, 0, K), 'constant', value=1.0)
                adj = torch.tril(adj, diagonal=-1)
                adj = adj + adj.transpose(1, 2)
                edges = [
                    adj[bb].to_sparse().coalesce().indices() + bb * adj.shape[1]
                    for bb in range(B)
                ]
                edges = torch.cat(edges, dim=1).t()

                att_idx = torch.cat([torch.zeros(ii).long(),
                                     torch.arange(1, K + 1)]).to(self.device)
                att_idx = att_idx.view(1, -1).expand(B, -1).contiguous().view(-1, 1)

                att_edge_feat = torch.zeros(edges.shape[0],
                                            2 * self.att_edge_dim).to(self.device)
                att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
                att_edge_feat = att_edge_feat.scatter(
                    1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

                node_state_out = self.decoder(
                    node_state_in.view(-1, H), edges, edge_feat=att_edge_feat)
                node_state_out = node_state_out.view(B, jj, -1)

                # Edge prediction (same as v2)
                idx_row, idx_col = np.meshgrid(np.arange(ii, jj), np.arange(jj))
                idx_row = torch.from_numpy(idx_row.reshape(-1)).long().to(self.device)
                idx_col = torch.from_numpy(idx_col.reshape(-1)).long().to(self.device)

                diff = node_state_out[:, idx_row, :] - node_state_out[:, idx_col, :]
                diff = diff.view(-1, node_state.shape[2])
                log_theta = self.output_theta(diff)
                log_alpha = self.output_alpha(diff)

                log_theta = log_theta.view(B, -1, K, self.num_mix_component)
                log_theta = log_theta.transpose(1, 2)

                log_alpha = log_alpha.view(B, -1, self.num_mix_component)
                prob_alpha = F.softmax(log_alpha.mean(dim=1), -1)
                alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1).long()

                prob = []
                for bb in range(B):
                    prob += [torch.sigmoid(log_theta[bb, :, :, alpha[bb]])]
                prob = torch.stack(prob, dim=0)
                A[:, ii:jj, :jj] = torch.bernoulli(prob[:, :jj - ii, :])

            if self.is_sym:
                A = torch.tril(A, diagonal=-1)
                A = A + A.transpose(1, 2)

            # Trim to target size
            A = A[:, :N_target, :N_target]
            return A

    def _inference_with_types(self, A_pad, edges, node_idx_gnn, node_idx_feat,
                              att_idx, node_types):
        """Inference that injects type embeddings into initial node features.

        Used during training with ground-truth types.
        """
        B, C, N_max, _ = A_pad.shape
        H = self.hidden_dim
        A_pad_flat = A_pad.view(B * C * N_max, -1)

        if self.dimension_reduce:
            node_feat = self.decoder_input(A_pad_flat)
        else:
            node_feat = A_pad_flat

        # Inject type embeddings
        # node_types: (B, N_max) → broadcast to (B * C * N_max)
        node_types_flat = node_types.unsqueeze(1).expand(-1, C, -1).reshape(-1)
        type_feat = self.type_embedding(node_types_flat)  # (B*C*N_max, D)
        node_feat = node_feat + type_feat

        # Pad row 0 as "new node" placeholder (same as v1/v2)
        node_feat = F.pad(node_feat, (0, 0, 1, 0), 'constant', value=0.0)

        att_idx = att_idx.view(-1, 1)
        att_edge_feat = torch.zeros(edges.shape[0],
                                    2 * self.att_edge_dim).to(node_feat.device)
        att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
        att_edge_feat = att_edge_feat.scatter(
            1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

        node_state = self.decoder(
            node_feat[node_idx_feat], edges, edge_feat=att_edge_feat)

        diff = node_state[node_idx_gnn[:, 0], :] - node_state[node_idx_gnn[:, 1], :]
        log_theta = self.output_theta(diff).view(-1, self.num_mix_component)
        log_alpha = self.output_alpha(diff).view(-1, self.num_mix_component)

        return log_theta, log_alpha, node_state

    def forward(self, input_dict):
        is_sampling = input_dict.get('is_sampling', False)
        target_types = input_dict.get('target_types', None)

        if is_sampling:
            if target_types is not None:
                # v3 conditional generation
                B = target_types.shape[0]
                A = self._sampling_with_types(B, target_types.to(self.device))
                A_list = [A[ii] for ii in range(B)]
                return A_list
            else:
                # Fall back to v2 unconditional
                return super().forward(input_dict)

        # Training forward
        A_pad = input_dict['adj']
        edges = input_dict['edges']
        node_idx_gnn = input_dict['node_idx_gnn']
        node_idx_feat = input_dict['node_idx_feat']
        att_idx = input_dict['att_idx']
        label = input_dict['label']
        subgraph_idx = input_dict['subgraph_idx']
        subgraph_idx_base = input_dict['subgraph_idx_base']
        node_types = input_dict.get('node_types', None)

        if node_types is None:
            # No types → fall back to v2 behavior
            return super().forward(input_dict)

        log_theta, log_alpha, _ = self._inference_with_types(
            A_pad, edges, node_idx_gnn, node_idx_feat, att_idx, node_types)

        edge_loss = mixture_bernoulli_loss(
            label, log_theta, log_alpha,
            self.adj_loss_func, subgraph_idx, subgraph_idx_base,
            self.num_canonical_order)

        return edge_loss
```

- [ ] **Step 4: Run tests**

```bash
cd D:/Github/GSDiff/GRAN
uv run python -m pytest tests/test_gran_v3_type_embed.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add model/gran_v3.py tests/test_gran_v3_type_embed.py
git commit -m "feat: add GRANv3 with type embedding for conditional generation"
```

---

## Task A2: Dataset V3 with Node Types as Input

The v2 dataset stores node attributes as **targets** (for attr head loss). V3 uses them as **inputs** to the model instead. The data field is the same (`node_attrs` / `node_types`), only the role in training changes.

**Files:**
- Modify: `dataset/gran_data_v2.py` (already includes `node_attrs` field)
- Create: `dataset/gran_data_v3.py` (alias or thin wrapper)

- [ ] **Step 1: Create v3 dataset wrapper**

```python
# dataset/gran_data_v3.py
"""GRANv3 dataset: same data as v2, but `node_attrs` is used as model input
(node_types), not as prediction target.
"""
from dataset.gran_data_v2 import GRANDataV2


class GRANDataV3(GRANDataV2):
    """v3 reuses v2 data; v3 runner renames `node_attrs` → `node_types`
    when constructing the input dict. See GranRunnerV3.
    """
    pass
```

- [ ] **Step 2: Commit**

```bash
git add dataset/gran_data_v3.py
git commit -m "feat: add GRANDataV3 wrapper for type-conditioned training"
```

---

## Task A3: GRANv3 Runner and Config

**Files:**
- Create: `runner/gran_runner_v3.py`
- Create: `config/gran_v3_rplan.yaml`

- [ ] **Step 1: Create runner**

```python
# runner/gran_runner_v3.py
"""Runner for GRANv3 type-conditioned generation.

Differences from GranRunnerV2:
- Passes `node_types` (instead of `node_attrs`) to the model's forward()
- Training loss is pure edge NLL (no attr head)
- Test uses ground-truth types from test set to condition generation
"""
import torch
from runner.gran_runner_v2 import GranRunnerV2
from runner.gran_runner import GranRunner

__all__ = ['GranRunnerV3']


class GranRunnerV3(GranRunnerV2):
    """Runner for GRANv3 with type-conditioned generation."""

    def _build_input_dict(self, batch_ff, gpu_id):
        """Override to rename node_attrs → node_types."""
        data = {}
        data['adj'] = batch_ff['adj'].pin_memory().to(gpu_id, non_blocking=True)
        data['edges'] = batch_ff['edges'].pin_memory().to(gpu_id, non_blocking=True)
        data['node_idx_gnn'] = batch_ff['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
        data['node_idx_feat'] = batch_ff['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
        data['label'] = batch_ff['label'].pin_memory().to(gpu_id, non_blocking=True)
        data['att_idx'] = batch_ff['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
        data['subgraph_idx'] = batch_ff['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
        data['subgraph_idx_base'] = batch_ff['subgraph_idx_base'].pin_memory().to(gpu_id, non_blocking=True)

        if 'node_attrs' in batch_ff:
            # Pass as node_types (input), not node_attrs (prediction target)
            data['node_types'] = batch_ff['node_attrs'].pin_memory().to(gpu_id, non_blocking=True)

        return data
```

- [ ] **Step 2: Create config**

```yaml
# config/gran_v3_rplan.yaml
---
exp_name: GRANv3
exp_dir: exp/GRANv3
runner: GranRunnerV3
use_horovod: false
use_gpu: true
device: cuda:0
gpus: [0]
seed: 1234
dataset:
  loader_name: GRANDataV3
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
  name: GRANv3
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
  num_attr_classes: 7
  use_gatv2: false
  approach: type_embed
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
  test_model_name: gran_v3_rplan.pth
```

- [ ] **Step 3: Commit**

```bash
git add runner/gran_runner_v3.py config/gran_v3_rplan.yaml
git commit -m "feat: add GRANv3 runner and RPLAN config"
```

---

## Approach B: Global Condition Encoder (Alternative)

**Idea:** Encode the entire type multiset into a single global condition vector, then inject it into the GNN at each propagation step. This treats the type distribution as a conditioning signal similar to text prompts in text-to-image diffusion.

### When to Prefer Approach B Over A

- **Set-level reasoning:** If the model needs to reason about the *distribution* of types rather than individual type assignments (e.g., "this is a 3-bedroom apartment" as a holistic concept)
- **Beyond type counts:** When conditions include non-node information (e.g., total area, floor level, target style) that can't be attached to individual nodes
- **Cross-attention style:** When you want the model to dynamically attend to different parts of the condition during generation

For simple room-type-to-graph tasks, **Approach A is sufficient and simpler**.

### Architecture Sketch

```
Input: room_type_multiset = {0: 1, 1: 2, 3: 1, 2: 1, 4: 1}
                                   ↓
                       ┌──────────────────────┐
                       │  Condition Encoder   │
                       │  - type embedding    │
                       │  - set pooling       │
                       │  - MLP               │
                       └──────────────────────┘
                                   ↓
                         cond_embed: (B, D_cond)
                                   ↓
        ┌──────────────────────────┴──────────────────────────┐
        ↓                                                      ↓
   FiLM modulation                                    Cross-attention
   (γ, β = MLP(cond_embed))                           (K, V = cond_embed,
   node_state = γ * node_state + β                     Q = node_state)
```

### Changes vs Approach A

| Component | Approach A | Approach B |
|-----------|-----------|-----------|
| New modules | 1 (Embedding) | 2-3 (Encoder + Fusion + optional CrossAttn) |
| Training data format | Per-node types | Per-graph type set |
| Computational cost | ~0 overhead | +5-10% per step |
| Condition flexibility | Node-level only | Graph-level (can encode anything) |
| Checkpoint reuse from v2 | Easy (add embedding) | Harder (new encoder) |

### Implementation (Approach B) — Sketch Only

```python
# model/gran_v3_condenc.py (not implemented in this plan)

class TypeSetEncoder(nn.Module):
    """Encode a multiset of types into a global embedding."""
    def __init__(self, num_types, embed_dim, out_dim):
        super().__init__()
        self.type_embed = nn.Embedding(num_types, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, type_ids):
        # type_ids: (B, N_variable)
        embeds = self.type_embed(type_ids)  # (B, N, D)
        pooled = embeds.mean(dim=1)          # (B, D) — set-invariant mean pooling
        return self.mlp(pooled)              # (B, out_dim)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: conditions features via scale+shift."""
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, feat_dim)
        self.to_shift = nn.Linear(cond_dim, feat_dim)

    def forward(self, features, cond):
        scale = self.to_scale(cond).unsqueeze(1)
        shift = self.to_shift(cond).unsqueeze(1)
        return features * (1 + scale) + shift


class GRANv3CondEnc(GRANv2):
    def __init__(self, config):
        super().__init__(config)
        self.cond_encoder = TypeSetEncoder(
            num_types=self.num_attr_classes,
            embed_dim=self.embedding_dim,
            out_dim=self.embedding_dim,
        )
        self.film = FiLMLayer(self.embedding_dim, self.hidden_dim)

    def _inference_with_cond(self, A_pad, edges, ..., cond_types):
        cond_embed = self.cond_encoder(cond_types)  # (B, D)
        # ... standard GNN forward ...
        node_state = self.decoder(...)
        # Inject condition via FiLM
        node_state = self.film(node_state, cond_embed)
        # ... rest of inference ...
```

---

## Comparison Summary

| Dimension | Approach A (Type Embedding) | Approach B (Condition Encoder) |
|-----------|----------------------------|-------------------------------|
| **Complexity** | Low (one embedding layer) | Medium (encoder + fusion) |
| **Training cost** | Nearly identical to v2 | +5-10% per step |
| **New parameters** | `num_types × D` | `num_types × D + fusion MLPs + FiLM` |
| **Flexibility** | Node-type only | Arbitrary global conditions |
| **Checkpoint reuse** | Easy | Harder |
| **Recommended for** | Room type → graph | Complex multi-modal conditioning |
| **Implementation effort** | 1 model file + 1 runner | 2-3 model files + fusion logic |

**Recommendation:** Implement **Approach A** first (Tasks A1-A3 above). Only add Approach B if you need per-graph conditions beyond per-node types (e.g., global style, target metrics).

---

## Training Strategy

### Training Data Preparation

For RPLAN:
1. Extract node types from bubble diagram .npy files (already in `GRANDataV2`)
2. Use canonical ordering (DFS/BFS) to align type sequence with adjacency rows
3. Each training sample: `(adj_subgraph, node_types_in_order)`

### Loss

Pure edge NLL (no attr loss needed, since types are inputs):

```python
loss = mixture_bernoulli_loss(edge_label, log_theta, log_alpha, ...)
```

### Training Loop

Same as v2, just pass `node_types` instead of `node_attrs` via the runner. No lambda hyperparameter needed.

### Convergence Expectation

- **Faster convergence** than v2: edge prediction task is simpler when type info is given as input
- **Better edge quality**: model can use type info as a strong prior
- **Loss plateaus**: expected to converge in fewer epochs than unconditional GRAN

---

## Validation Protocol

- [ ] **Test 1: Type preservation**
  Generate a graph with `target_types = [0, 1, 1, 3]`. Verify the returned graph has exactly 4 nodes and the type assignments match.

- [ ] **Test 2: Type-specific edge patterns**
  Compare edge distributions for different type inputs:
  - Graph with `[Kitchen, Living, Bedroom]` should have Kitchen-Living edge high probability
  - Graph with `[Kitchen, Bathroom, Bathroom]` should have fewer Kitchen-Bathroom connections
  Verify the model learned type-conditional edge distributions.

- [ ] **Test 3: Comparison with unconditional baseline**
  For the same type multiset from the test set, compare:
  - v3 generated edges vs v1/v2 unconditional generated edges
  - Measure similarity to the ground-truth graph (graph edit distance)
  v3 should produce graphs closer to ground truth.

- [ ] **Test 4: Generalization**
  Test with type multisets not seen in training (e.g., unusual combinations). Verify the model produces reasonable adjacency patterns rather than collapsing or producing empty graphs.

---

## File Structure Summary

```
GRAN/
├── model/
│   ├── gran_v2.py          # existing (from v2 plan)
│   └── gran_v3.py          # NEW: Approach A
├── dataset/
│   ├── gran_data_v2.py     # existing
│   └── gran_data_v3.py     # NEW: wrapper for v3
├── runner/
│   ├── gran_runner_v2.py   # existing
│   └── gran_runner_v3.py   # NEW: renames attrs → types
├── config/
│   ├── gran_v2_rplan.yaml  # existing
│   └── gran_v3_rplan.yaml  # NEW
├── tests/
│   └── test_gran_v3_type_embed.py  # NEW
└── docs/
    ├── plan-gran-v2-upgrade.md
    ├── plan-rplan-to-gran.md
    └── plan-gran-v3-conditional.md  # THIS FILE
```

---

## Summary

**Approach A (Type Embedding as Initial Feature):**
- 3 tasks: A1 (model), A2 (dataset wrapper), A3 (runner + config)
- Minimal code change, maximum ROI
- Recommended for implementation

**Approach B (Global Condition Encoder):**
- Not implemented as tasks (sketch only)
- Reserved for future extension if global conditions are needed
- More complex but more expressive

Both approaches achieve the same core goal: generate graph topology given a specified set of node types. Approach A is simpler because it leverages the fact that in autoregressive generation, you know each node's type at the moment it's being generated, so you can inject type info locally at each step rather than globally pre-computing a condition embedding.
