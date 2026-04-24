# Absorbing D3PM for MSD Bubble Diagram Generation — Implementation Plan

## 1. Problem Statement

### 1.1 Target
Generate architectural bubble diagrams for multi-apartment building complexes using the MSD (Modified Swiss Dwellings, ECCV 2024) dataset.

### 1.2 Three required tasks (all in one model)
| Task | Description | Inference input | Inference output |
|------|-------------|-----------------|------------------|
| A. Unconditional generation | Generate a complete bubble diagram from scratch | Target graph size n | Full (X, E) with n nodes |
| B. Partial graph completion | Given k known nodes and their edges, predict the next node(s) with connections | Partial (X_known, E_known), k < n | Completed (X, E) |
| C. Node attribute prediction | Given all edges and a new node's connections to existing nodes, predict the new node type | (X_known, E_full), one node masked | Predicted node type distribution |

### 1.3 Data status
- **Node attributes**: room types (categorical)
- **Edge attributes**: connection types (categorical, with "no edge" as one class)
- **Data extraction pipeline is complete.** The downstream implementer should inspect the existing pipeline output to determine:
  - Exact node type vocabulary (class count and labels)
  - Exact edge type vocabulary (class count and labels)
  - Graph size distribution
  - Train/val/test splits
  - File format used (torch_geometric.Data, networkx, pickle, etc.)
- The implementer writes a DiGress-compatible dataset loader that converts the existing pipeline output into `(X, E, y, node_mask)` tensors expected by DiGress's `AbstractDataModule`.

### 1.4 Known failure modes from prior work
- GRAN on RPLAN: failed to learn "living room is in center" structural prior due to BFS-order + autoregressive + separate node-type/topology heads
- Attribute-topology coupling is broken when heads are independent
- These failures motivate the joint-modeling approach of Absorbing D3PM

---

## 2. Method: Absorbing-State Discrete Graph Diffusion

### 2.1 Why this method

The method is a D3PM (Austin et al., NeurIPS 2021) with absorbing transition, applied to graphs with the DiGress architecture (Vignac et al., ICLR 2023) as backbone.

**Key property**: Absorbing D3PM is mathematically equivalent to BERT-style masked modeling and to any-order autoregressive models (Ou et al., 2024). This equivalence directly gives us all three tasks as special cases of one trained model, differing only in inference-time initialization.

**Why not vanilla DiGress**: DiGress uses marginal transition. Its training distribution never contains partially-observed graphs, so partial completion and attribute prediction require post-hoc hacks (scaffold extension, RePaint resampling) that suffer from train/test mismatch and long-range inconsistency.

**Why not HSpectre / GruM**: Both are excellent for unconditional generation only. HSpectre grows a graph from a single node (incompatible with a given partial graph). GruM predicts destinations from noise (incompatible with partially-observed inputs).

**Why not Latent Graph Diffusion (LGD)**: LGD's "unified" scope is over task types (generation/regression/classification), not over partial-observation ratios. Partial completion is not a native mode.

### 2.2 Unified formulation

All three tasks use the same trained model. They differ only in how the input graph is initialized at inference:

```
Mode A (unconditional): All nodes and edges initialized to [MASK]
Mode B (partial completion): Known nodes/edges keep their values; unknown = [MASK]
Mode C (attribute prediction): Almost everything known; target node type = [MASK]
```

The model runs the same reverse diffusion loop in all three cases, only unmasking positions that are currently [MASK]. Known positions act as anchors and are never modified.

### 2.3 Mathematical setup

#### Vocabulary extension
Extend node type space from `a` classes to `a+1` classes (add [MASK] as last class).  
Extend edge type space from `b` classes to `b+1` classes (add [MASK] as last class).

If MSD extraction yields `a` room types and `b` edge types, the model's internal vocabularies become `a+1` and `b+1`. The extra class represents "not yet determined".

#### Transition matrix (absorbing)
Let `e_mask_X` be the one-hot vector for [MASK] in node space (shape `(a+1,)`), similarly for edges.

At diffusion step t:
```
Q_t_X = α_t · I + β_t · 1 · e_mask_X^T
Q_t_E = α_t · I + β_t · 1 · e_mask_E^T
```

where `α_t` follows the standard cosine schedule and `β_t = 1 - α_t`.

The cumulative transition after t steps:
```
Q̄_t_X = ᾱ_t · I + β̄_t · 1 · e_mask_X^T
```

Interpretation: at step t, each position independently either stays unchanged (probability ᾱ_t) or becomes [MASK] (probability β̄_t = 1 - ᾱ_t). As t → T, ᾱ_t → 0, so all positions become [MASK].

#### Forward process
For a clean graph G = (X, E):
```
q(X_t | X) = X · Q̄_t_X    # categorical per node
q(E_t | E) = E · Q̄_t_E    # categorical per edge
```

Sampling G_t from G: each node is independently either kept at its clean value or set to [MASK], with mask probability β̄_t. Same for edges.

#### Reverse process
Train a denoising network `φ_θ(G_t, t, features) → (p_X, p_E)` that predicts, for each position, the distribution over the original (non-mask) categories.

Loss (compute only on masked positions; non-mask positions carry no gradient):
```
L = E_t [ Σ_{i: x_i_t = [MASK]}  CE(p_X_i, x_i)
       + λ · Σ_{i,j: e_ij_t = [MASK]}  CE(p_E_ij, e_ij) ]
```

This is the standard absorbing D3PM loss — identical in structure to BERT's masked language modeling loss, just over graph positions.

### 2.4 Denoising network architecture

Start from DiGress's Graph Transformer backbone and add two enhancements:

1. **Keep DiGress structural features**: cycle counts (3-5 cycles), spectral features (Laplacian eigenvalues, eigenvectors), node/edge marginal priors. These are computed from the currently-observed (non-mask) portion of the graph at each step.

2. **Add Graphormer-style centrality encoding** (key for "living room in center" prior):
   - Degree centrality as learnable positional embedding per node
   - Eigenvector centrality as additional node feature
   - These are concatenated to node features before the transformer layers

3. **Spatial encoding via attention bias** (optional, incremental):
   - Compute shortest path distance between node pairs on the known subgraph
   - Inject as bias term in self-attention: `Attn(i,j) += b(SPD(i,j))`
   - Helps the model reason about spatial proximity

The network outputs per-position categorical distributions over the original (non-mask) categories. The [MASK] class is excluded from the output (a classes out, not a+1).

### 2.5 Training procedure

```
For each training step:
  Sample G = (X, E) from training split
  Sample t ~ Uniform(1, T)
  Sample G_t by independently masking each node with prob β̄_t
                                and each edge with prob β̄_t
  Compute structural features on G_t (observed portion only)
  Predict p_X, p_E = φ_θ(G_t, t, features)
  Loss = cross-entropy on masked positions only
  Backprop
```

Key hyperparameters:
- T = 500 (can reduce to 50-100 with MaskGIT sampling)
- Cosine schedule for α_t
- λ = 5 (edge loss weight, matches DiGress convention)
- Batch size depends on max graph size (start with batch=8)

### 2.6 Inference procedures

#### Mode A: Unconditional generation
```
Input: target graph size n (sampled from training size distribution)
Initialize: X_T[i] = [MASK] for all i in 1..n
            E_T[i,j] = [MASK] for all i,j
For t = T down to 1:
    z = structural_features(G_t)
    p_X, p_E = φ_θ(G_t, t, z)
    For each currently-masked position:
        Sample replacement from its predicted distribution
        (MaskGIT-style: unmask top-k highest-confidence positions per step)
    Already-unmasked positions stay fixed
Return G_0
```

#### Mode B: Partial completion
```
Input: partial graph (X_known, E_known) with k known nodes
       target total size n
Initialize: 
    X_T[i] = X_known[i] for i in 1..k
    X_T[i] = [MASK] for i in k+1..n
    E_T[i,j] = E_known[i,j] if both i,j in 1..k, else [MASK]
Run the same reverse loop as Mode A
Known positions are anchors and never re-masked
Return G_0
```

#### Mode C: Attribute prediction
```
Input: full graph with one target node's type unknown
       X[target] = [MASK]; X[others] = known; E = fully known
Single forward pass (or 2-3 steps):
    p_X, _ = φ_θ(G, t=1, features)
Return p_X[target]  # distribution over room types
```

### 2.7 MaskGIT-style parallel decoding (recommended)

Instead of standard 500-step D3PM sampling, use MaskGIT's approach for faster and better-quality generation:

```
For t in a short schedule (10-20 steps):
    Predict p_X, p_E at all currently-masked positions
    Compute confidence (max probability) at each masked position
    Select top-k most confident positions to unmask (k depends on step, follows a cosine schedule)
    Sample new values for those positions
    The remaining masked positions stay [MASK] for the next step
```

Schedule example (T=12 steps for n=50 nodes):
- Unmask counts per step follow a front-loaded schedule:
  `[5, 8, 10, 10, 8, 5, 2, 1, 1, 1, 1, remaining]`

This reduces inference from ~500 steps to ~10-20 steps with minimal quality loss (established in MaskGIT for images, MDLM for text, MELD for molecules).

---

## 3. Implementation Steps

### 3.1 Base code
Fork DiGress official repo: `https://github.com/cvignac/DiGress`

Key files to modify:
- `src/datasets/` — add MSD dataset class that reads from the existing extraction pipeline output
- `src/diffusion/noise_schedule.py` — change transition matrix from marginal to absorbing
- `src/diffusion_model_discrete.py` — adapt loss and add new sampling modes
- `src/models/transformer_model.py` — add Graphormer centrality encoding

### 3.2 MSD dataset loader
Data extraction is already complete. The implementer needs to:

1. Inspect the existing pipeline output format (likely `torch_geometric.data.Data` or similar)
2. Write a DiGress-compatible `MSDDataset` class that inherits from DiGress's dataset abstractions
3. Implement `DatasetInfos` class with:
   - `num_node_types` = (observed classes) + 1 for [MASK]
   - `num_edge_types` = (observed classes) + 1 for [MASK]
   - `node_types_marginal` — distribution over non-mask node classes
   - `edge_types_marginal` — distribution over non-mask edge classes
   - `n_nodes_distribution` — distribution over graph sizes for sampling
4. Set up the `MSDDataModule` with train/val/test split loading

Reference: DiGress's `moses_dataset.py` and `spectre_datasets.py` are the canonical examples.

### 3.3 Transition matrix change (the core modification)

In DiGress's noise schedule module, find where `Q_t_X` and `Q_t_E` are constructed (typically in a `MarginalTransition` class).

Current code (marginal transition):
```python
u_X = self.u_X  # shape (a,), marginal distribution of node types
Q_t_X = alpha_t * I + beta_t * ones @ u_X.unsqueeze(0)
```

Replace with absorbing transition:
```python
# Mask one-hot vector, shape (a+1,) with 1.0 at mask index (last position)
mask_idx_X = a  # [MASK] is the (a+1)-th class, index a
u_X_absorb = torch.zeros(a + 1)
u_X_absorb[mask_idx_X] = 1.0

Q_t_X = alpha_t * I_{a+1} + beta_t * ones @ u_X_absorb.unsqueeze(0)
```

Same pattern for `Q_t_E` with edge mask index = b.

### 3.4 Loss modification

Locate the loss computation in `diffusion_model_discrete.py`. DiGress currently computes cross-entropy at all positions.

Change to compute only at currently-masked positions:
```python
# After forward noising, identify masked positions
mask_positions_X = (X_t.argmax(-1) == mask_idx_X)  # boolean, shape (batch, n)
mask_positions_E = (E_t.argmax(-1) == mask_idx_E)  # boolean, shape (batch, n, n)

# Compute loss only at those positions
loss_X = F.cross_entropy(
    pred_X[mask_positions_X], 
    true_X[mask_positions_X].argmax(-1)
)
loss_E = F.cross_entropy(
    pred_E[mask_positions_E],
    true_E[mask_positions_E].argmax(-1)
)
total_loss = loss_X + lambda_E * loss_E
```

Model output dimension: `a` for nodes (not `a+1`) — we do not predict [MASK] as an output class.

### 3.5 Sampling modification

Add dedicated inference functions alongside DiGress's existing `sample_batch`:

```python
def partial_completion_sample(self, X_known, E_known, n_total, n_known):
    # Initialize
    X = torch.full((n_total,), self.mask_idx_X)
    X[:n_known] = X_known
    E = torch.full((n_total, n_total), self.mask_idx_E)
    E[:n_known, :n_known] = E_known
    
    # Anchor mask: positions that are given and should never change
    anchor_X = torch.zeros(n_total, dtype=bool)
    anchor_X[:n_known] = True
    anchor_E = torch.zeros(n_total, n_total, dtype=bool)
    anchor_E[:n_known, :n_known] = True
    
    # Reverse diffusion (MaskGIT style)
    for step in range(n_maskgit_steps):
        features = self.compute_structural_features(X, E)
        p_X, p_E = self.denoise_network(X, E, step, features)
        
        # Unmask top-k non-anchor positions that are currently [MASK]
        X, E = self.maskgit_step_unmask(
            X, E, p_X, p_E, 
            anchor_X, anchor_E, 
            step, n_maskgit_steps
        )
    
    return X, E

def attribute_prediction(self, X_known, E_full, target_idx):
    X = X_known.clone()
    X[target_idx] = self.mask_idx_X
    features = self.compute_structural_features(X, E_full)
    p_X, _ = self.denoise_network(X, E_full, t=1, features=features)
    return p_X[target_idx]  # distribution over a room types
```

For Mode A, reuse the partial_completion_sample function with `n_known=0`.

### 3.6 Graphormer centrality encoding

In the denoising network's node feature construction, add:
```python
# Binary adjacency from known (non-mask) edges
observed_adj = (E.argmax(-1) != mask_idx_E) & (E.argmax(-1) != 0)  # exclude mask and "no edge"
observed_adj = observed_adj.float()

# Degree centrality as learnable embedding
degree = observed_adj.sum(dim=-1).long()  # (batch, n)
degree = degree.clamp(max=max_degree)
degree_emb = self.degree_embedding(degree)  # (batch, n, d_deg)

# Eigenvector centrality via power iteration (10 iters, differentiable)
eigvec_cent = power_iteration(observed_adj, n_iters=10)  # (batch, n)

# Concatenate to existing node features
node_features = torch.cat([
    X_emb,           # original DiGress node embeddings
    degree_emb,      # new: degree centrality embedding
    eigvec_cent.unsqueeze(-1),  # new: eigenvector centrality scalar
], dim=-1)
```

### 3.7 Training configuration

```yaml
model:
  transition: absorbing  # was: marginal
  n_layers: 8            # same as DiGress molecular config
  hidden_dims:
    dx: 256
    de: 64
    dy: 64
  extra_features: 
    - cycles           # keep from DiGress
    - spectral         # keep from DiGress
    - centrality       # new: Graphormer-style

train:
  batch_size: 8         # tune based on max graph size
  lr: 2e-4
  n_epochs: 500
  diffusion_steps: 500
  mask_schedule: cosine
  edge_loss_weight: 5.0
  check_val_every_n_epochs: 10

sample:
  method: maskgit       # or "standard" for 500-step D3PM
  n_steps_maskgit: 12
  # Cosine-like schedule for positions unmasked per step:
  # later steps unmask fewer to focus model on hard decisions
  top_k_schedule: cosine_front_loaded

dataset:
  name: msd
  data_dir: <path to extracted pipeline output>
  max_n_nodes: 100       # filter out buildings larger than this
```

### 3.8 Evaluation metrics

**Unconditional generation**:
- Graph statistics MMD (degree distribution, clustering, orbit counts, spectrum)
- Building validity: connected, has entrance, has living room
- Centrality prior: fraction of generated graphs where living room is in top-20% by eigenvector centrality

**Partial completion**:
- From test set graphs, mask last 20% of nodes, run completion
- Predicted node type accuracy at masked positions
- Predicted edge type accuracy at masked positions
- Overall graph validity after completion

**Attribute prediction**:
- From test set, mask one node's type at a time
- Top-1 accuracy, confusion matrix per room type

---

## 4. Expected Outcomes

### 4.1 Performance expectations
- Unconditional generation: comparable to or slightly better than DiGress baseline
- Partial completion: significantly better than DiGress + scaffold extension (no train/test mismatch)
- Attribute prediction: competitive with a dedicated Graph Transformer classifier, without separate training
- Inference speed: 30-50x faster than standard DiGress with MaskGIT sampling (~10-20 steps vs 500)

### 4.2 Architectural priors to verify
- Living room should consistently appear in high-centrality positions
- Room-type degree distributions should match training statistics
- Common adjacencies (living-kitchen, bedroom-bathroom) should be overrepresented

### 4.3 Failure modes to watch for
- **High mask ratio (Mode A) produces invalid graphs**: may need longer training or stronger structural features
- **Low mask ratio (Mode C) is inaccurate**: may need auxiliary fine-tuning with attribute-only loss
- **Large building complexes cause OOM**: fall back to SparseDiff-style sparse attention on `E` tensor
- **"Living room in center" prior not learned**: add explicit centrality conditioning or condition on apartment-level structure

---

## 5. Time Estimate

Data extraction pipeline is already complete. Remaining work:

- Week 1: DiGress fork setup, MSD dataset loader integration with existing pipeline output, vanilla DiGress sanity-check run
- Week 2-3: Absorbing transition implementation, training works end-to-end, Mode A unconditional generation working
- Week 4: MaskGIT sampler implementation
- Week 5: Graphormer centrality encoding, full training with all enhancements
- Week 6-7: Modes B and C inference implementations, evaluation pipeline
- Week 8-9: Hyperparameter tuning, building-specific structural feature experiments
- Week 10: Final evaluation, results consolidation

Total: ~10 weeks, unchanged — data pipeline saving is offset by the risk of MSD-specific adjustments discovered during training.

---

## 6. Key References

- Austin et al. "Structured Denoising Diffusion Models in Discrete State-Spaces." NeurIPS 2021 (D3PM, absorbing transition)
- Vignac et al. "DiGress: Discrete Denoising Diffusion for Graph Generation." ICLR 2023 (base architecture)
- Ying et al. "Do Transformers Really Perform Bad for Graph Representation?" NeurIPS 2021 (Graphormer, centrality encoding)
- Chang et al. "MaskGIT: Masked Generative Image Transformer." CVPR 2022 (parallel decoding)
- Sahoo et al. "Simple and Effective Masked Diffusion Language Models." NeurIPS 2024 (MDLM, training stability)
- Ou et al. "Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data." 2024 (theoretical equivalence to any-order AR)
- van Engelenburg et al. "MSD: A Benchmark Dataset for Floor Plan Generation of Building Complexes." ECCV 2024 (dataset)

---

## 7. Repository Pointers

- DiGress (base): https://github.com/cvignac/DiGress
- SparseDiff (fallback for large graphs): https://github.com/qym7/SparseDiff
- MSD dataset: https://github.com/caspervanengelenburg/msd
- MaskGIT reference: https://github.com/google-research/maskgit
- Graphormer reference: https://github.com/microsoft/Graphormer
