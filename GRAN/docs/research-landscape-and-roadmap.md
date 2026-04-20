# Bubble Diagram Generation: Research Landscape & Project Roadmap

> Why we're doing this, how it relates to current state-of-the-art, and
> what to do next. Focused on the specific task: **generate residential
> bubble diagrams (room-adjacency graphs with room types)** from RPLAN-scale
> data.

---

## 1. Why bubble diagram generation matters

### The architectural design pipeline

```
          user intent ("I want 3BR2BA with a balcony")
                             │
                             ▼
   ┌───────────────────────────────────────────────┐
   │  Stage 1:  BUBBLE DIAGRAM (this project)      │
   │  - nodes   = room types                       │
   │  - edges   = adjacency (which rooms touch)    │
   │  - no spatial coordinates yet                 │
   └───────────────────────────────────────────────┘
                             │
                             ▼
   ┌───────────────────────────────────────────────┐
   │  Stage 2:  VECTOR FLOORPLAN (downstream)      │
   │  - place rooms with actual coordinates        │
   │  - draw walls, doors, windows                 │
   │  (handled by GSDiff / HouseDiffusion / etc.)  │
   └───────────────────────────────────────────────┘
```

Bubble diagrams are the **intermediate topology** that every downstream
floorplan layout engine consumes. Reliable, diverse bubble-diagram
generation is thus the foundation of automated residential design.

### Who already uses this abstraction

- **GSDiff** (AAAI 2025) — takes a bubble diagram as topology constraint,
  diffuses the spatial layout
- **HouseDiffusion** (CVPR 2023) — same: bubble diagram → vector floorplan
- **House-GAN++** (CVPR 2021) — same
- **Graph2Plan** (SIGGRAPH 2020) — same

**All of these assume the bubble diagram is given as input.** Our task is
to *generate* the bubble diagram itself, so these downstream methods don't
need a human-supplied graph.

### Why this is a hard problem

1. **Topology correctness** — rooms must be laid out in plausible patterns
   (Living usually at center, bedrooms grouped, bathroom near bedroom, ...)
2. **Diversity** — must not always output the same "mode" of floorplan
3. **Constraint handling** — users may want partial control (fix 3 rooms,
   generate the rest; or fix room counts, generate connectivity)
4. **Scale mismatch** — RPLAN graphs are **tiny** (4–8 nodes) compared to
   most graph-generation benchmarks (100–5000 nodes), so standard
   architectures often under-perform

---

## 2. State of the art (2024–2025)

### Directly comparable work

| Paper | Venue | Year | Approach | Notes |
|-------|-------|------|----------|-------|
| [DBGNN + VGAE](https://www.mdpi.com/2076-3417/15/8/4490) | Applied Sciences | 2025 | Dual-branch GNN + VAE on bubble graphs | Closest to our task: dedicated bubble diagram gen |
| [HouseDiffusion](https://openaccess.thecvf.com/content/CVPR2023/papers/Shabani_HouseDiffusion_Vector_Floorplan_Generation_via_a_Diffusion_Model_With_Discrete_CVPR_2023_paper.pdf) | CVPR | 2023 | Graph-conditioned diffusion, discrete+continuous denoising | Generates floorplan, not bubble diagram; uses bubble diagram as input |
| [GSDiff](https://wutomwu.github.io/publications/2025-GSDiff/paper.pdf) | AAAI | 2025 | Diffusion on structural graph | Consumes bubble diagrams |
| [ChatHouseDiffusion](https://arxiv.org/html/2410.11908v1) | 2024 | Prompt-guided diffusion | Text → floorplan |
| [DStruct2Design](https://arxiv.org/html/2407.15723v1) | 2024 | Data-structure driven generation | Handles numerical + topological constraints |
| [HouseTune](https://arxiv.org/html/2411.12279) | 2024 | Two-stage with LLM | LLM + diffusion pipeline |

### Dominant trends

1. **Diffusion is the new default**. Since 2023 most new floorplan papers use
   diffusion rather than autoregressive / GAN. Reason: diffusion naturally
   handles joint node+edge generation without the ordering problem.
2. **Graph constraints are first-class**. Bubble diagrams are the standard
   conditioning signal for downstream floorplan generators.
3. **Autoregressive (GRAN-style) is "solid baseline"**. Still cited; not
   SOTA but reliable and fast to train on small graphs.

### What no one has fully solved

- **Handling partial constraints** — users often want "fix these 3 rooms,
  fill in the rest". Most diffusion models can do unconstrained or
  fully-constrained but not smooth inpainting.
- **Room-type imbalance** — RPLAN's 38% bedroom / 3% storage skew makes
  rare classes underrepresented in generation.
- **Global structural priors** — enforcing "Living Room at center" without
  hard-coded post-processing remains an open problem.

---

## 3. What we built: GRAN v2

### Why GRAN as the backbone

- **Mature, simple, fast** — 5 epochs train on RPLAN in ~2 minutes
- **Good baseline for the Diffusion-vs-Autoregressive comparison** — lets
  us quantify how much the autoregressive ordering inductive bias hurts
- **Production-ready** — no research-grade exotic parts

### What v2 adds on top of original GRAN (2019)

| Extension | What it enables | Cost |
|-----------|----------------|------|
| **Node attribute head** | Generate room types alongside edges | +CE loss, +1 MLP head |
| **GATv2 backbone** (optional) | Better attention over neighbors vs original GRU-gate | torch_geometric dep |
| **Partial graph sampling** | `expand_graph()` API: continue from user prefix | inference-only change |
| **Attr with fixed edges** | `predict_attr_with_edges()`: given connections, predict type | inference-only change |
| **Multinomial attr sampling** | Diverse room types instead of argmax collapse | one-line change |

Plus production polish:
- Disjoint train/dev/test split (parent had dev = train prefix — a bug)
- Val loss + best-model saving
- Side-by-side viz, attr-colored bubble plots
- Config-driven training (every knob in YAML)

See `v2-progress.md` for full per-task changelog.

### Current limitations (acknowledged)

1. **Autoregressive ordering bias** — model's predictions depend on the
   canonical node ordering; learns "position 0 ≈ Living" which is fragile.
2. **Attr-structure coupling is weak** — model treats attrs as a per-node
   side task, not intertwined with structure. Generated bubbles sometimes
   have no Living Room, or 2 Living Rooms.
3. **No spatial info** — pure topology only. Cannot place rooms.
4. **Small graph regime** — performance saturates quickly (loss plateaus by
   epoch 5 on RPLAN) because the task is simpler than the model's capacity.

---

## 4. Roadmap

### Near-term (within the GRAN v2 framework)

Ordered by expected ROI:

**A. Multiple canonical orderings** (config change only)
```yaml
dataset:
  node_order: 'all'          # DFS + BFS + k-core + degree_desc + default
model:
  num_canonical_order: 5
```
Forces the model to learn structural invariants rather than memorizing the
ordering. Training cost: ~5x per epoch, but RPLAN is fast so fine.

**B. Structural node features** (small model change)
Add the **current degree** or **current-rank-by-degree** of each node as an
extra input embedding. This lets the model learn "highest-degree node is
probably Living" directly, independent of ordering.

**C. Partial graph masking at training time** (runner change)
During training, occasionally hand the model a random prefix + ask it to
continue. This closes the train/inference gap for `expand_graph()` — right
now the model sees only full graphs during training but is asked for
partial completions at inference.

**D. Train on 40k–70k data**
Current cap is 20k. RPLAN has 70k available. Each 2x data roughly buys
0.01 loss improvement. Probably only worth it for publication.

### Longer-term (beyond GRAN v2)

**E. Switch to diffusion** (architectural pivot)

Benchmark our GRAN v2 bubble diagrams against a DiGress-style approach
trained on the same RPLAN data. Modern SOTA (HouseDiffusion, DBGNN+VGAE)
uses diffusion/VAE because they sidestep the ordering problem entirely.

**F. Attach to GSDiff end-to-end**

Pipeline: GRAN v2 generates bubble diagram → GSDiff Stage 1/2 generates
spatial floorplan. The user interacts with the app at the bubble diagram
level ("edit rooms and connections") and gets a full floorplan back.

**G. Constraint-aware generation**

Following DStruct2Design: users specify *any subset* of
```
{ room_types, room_counts, room_areas, required_adjacencies, forbidden_adjacencies }
```
and the model generates a satisfying bubble diagram. This is a research
direction, not a quick win.

### Not recommended

- **Position embedding alone** — fragile (see Section 3 limitation #1).
  Only helpful when combined with structural features.
- **Hand-rolled post-processing** (e.g., force highest-degree node to be
  Living) — cheating, doesn't generalize.
- **Scaling model capacity blindly** — already plateaued at hidden_dim=256
  with 20k data. More capacity won't help without more data or better
  features.

---

## 5. Concrete next action

If we continue this track:

1. Do **A + B** (multi-ordering + structural features). ~1 day of work,
   should visibly improve the "Living-at-center" pattern.
2. Re-train on 40k data with those changes.
3. Re-compute MMD and visually inspect `gen_grid.png`.
4. If quality is acceptable, connect to the GSDiff app backend.

If we pivot to diffusion:

1. Prototype DiGress-on-RPLAN as a sibling of GRAN v2.
2. Compare MMD and visual quality head-to-head.
3. Use the winner (or both) in the app.

---

## 6. References

- [Automated Residential Bubble Diagram Generation Based on Dual-Branch Graph Neural Network and Variational Encoding](https://www.mdpi.com/2076-3417/15/8/4490) — closest 2025 work to our task
- [HouseDiffusion (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Shabani_HouseDiffusion_Vector_Floorplan_Generation_via_a_Diffusion_Model_With_Discrete_CVPR_2023_paper.pdf)
- [GSDiff (AAAI 2025)](https://wutomwu.github.io/publications/2025-GSDiff/paper.pdf) — our upstream sister project
- [ChatHouseDiffusion](https://arxiv.org/html/2410.11908v1) — prompt-guided floorplan generation
- [DStruct2Design](https://arxiv.org/html/2407.15723v1) — data-structure driven constraints
- [HouseTune](https://arxiv.org/html/2411.12279) — two-stage LLM + diffusion
- [FloorplanDiffusion (ICMR 2025)](https://dl.acm.org/doi/10.1145/3731715.3733343) — latent diffusion for floorplans
- [GRAN (NeurIPS 2019)](https://arxiv.org/abs/1910.00760) — our autoregressive backbone
- [GATv2 (ICLR 2022)](https://arxiv.org/abs/2105.14491) — backbone upgrade in v2
