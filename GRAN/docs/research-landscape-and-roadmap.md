# Bubble Diagram Generation: Research Landscape & Project Roadmap

> **Scope correction (2026-04):** earlier drafts of this doc conflated
> **floorplan generation** (spatial 2D layout output) with
> **bubble-diagram generation** (graph-level output: nodes + labels +
> edges, no geometry). They are NOT the same problem. Most famous
> "floorplan-graph" papers (HouseGAN++, HouseDiffusion, Graph2Plan,
> GSDiff, ChatHouseDiffusion) take a bubble diagram **as input** and
> emit a spatial layout; they do not generate, complete, or label
> bubble diagrams. This doc is about the graph-level task only.

---

## TL;DR — 一页总结

### 我们到底在做什么

- **输入**：无 / 部分图 / 完整无标签图
- **输出**：bubble diagram — `(adjacency, room_labels)`，无坐标、无像素
- **数据**：RPLAN 4-8 节点（住宅），未来扩展到 MSD (~50) 和医疗 (30-200+)

### v2 对外的 3 核心任务 + 2 辅助能力

见 `GRAN/model/gran_v2.py`（`_sampling`、`expand_graph`、
`predict_attr_with_edges`、`forward` 里的 `attr_only` 路径）。

**3 个核心任务（用户面卖点）：**

| # | 任务 | 输入 | 输出 |
|---|------|------|------|
| **T1** Unconditional | 无 | 完整 `(A, attrs)` |
| **T2** Partial completion | partial `(A_K, attrs_K)` + 要加的节点数 N | N 个新节点的 **attrs + 与已有节点的连边** |
| **T5** Attr + fixed edges | partial `(A_K, attrs_K)` + 用户指定"新节点连到哪些已有节点" | 新节点的 **attr**（结构由用户固定） |

**2 个辅助能力（合并 / 诊断用）：**

| # | 定位 | 说明 |
|---|------|-----|
| **T3** Expand (≡ T2 alias) | Thin wrapper | T2 用 `target_size`，T3 用 `add_n = target_size − K`，**同一份代码** |
| **T4** Attr-only | **诊断探针** | 给完整 `A`，预测所有 attrs。用于 probe 模型是否学到 structure→attr 耦合，不作产品卖点 |

### 核心判断

- **Floorplan 专业论文（GSDiff、HouseDiffusion、HouseGAN++、Graph2Plan、
  ChatHouseDiffusion）对我们的 backbone 选型没有直接帮助** — 它们消费
  bubble diagram，不生成 bubble diagram。保留作为下游接入方。
- **真正相关的 baselines 是通用图生成**：GRAN、GraphRNN、DeepGMG
  （自回归）；DiGress、GDSS、EDGE、SPECTRE（扩散 / score-based）；
  Graphormer、GRIT（graph transformer，通常是判别而非生成）。
- **没有任何一个公开 backbone 天然覆盖 T1/T2/T5 三个核心任务**。需要
  做一些 glue：
  - GRAN / DeepGMG：T1/T2 原生（T3 = T2 alias），T5 需要把 edge
    sampling 固定成用户输入；T4（诊断）把 attr head 当独立 classifier
    再跑一次。
  - DiGress / GDSS：T1/T4/T5 干净（inpainting mask）；**T2/T3 是核心
    坑**（插节点要扩 adjacency 维度，不是 mask 能搞定的）。
  - Graphormer / GRIT：T4/T5 原生（node classification），T1-T2
    不支持，需要外挂自回归或 diffusion 的生成头。

### 三阶段 roadmap（按建筑体量，不按论文年份）

| 阶段 | 节点数 | 推荐 backbone | 核心任务覆盖 | 工期 |
|------|--------|-------------|------------|------|
| **P1（当前）住宅 demo** | ≤ 15 | **GRAN v2 + GATv2**（现状） | T1/T2 ✅ / T5 ⚠ via attr head rerun | 进行中 |
| **P2 中型商业/办公** | ≤ 30 | **GRAN v2 + Graph Transformer backbone** (Graphormer / GRIT) | 同 P1（框架不变） | 2-3 周 |
| **P3 医疗/大型建筑** | 30-500 | **Discrete Diffusion (DiGress)** + inpainting | T1/T5 干净 / **T2 需要 node-insertion schedule（研究工作）** | 6-10 周 + 数据集 |

### 我们知道会卡在哪里

- **T2 (partial completion, 含 T3 alias) 在 diffusion 上是 hard**：
  "在中间插新节点"需要扩 adjacency 维度，纯 mask 搞不定。纯 DiGress
  scaffold extension demo 只演示了"加新边到已有节点"，不是"插节点"。
  要自己写增量式 inpainting schedule 或用 EDGE 风格稀疏扩散。**这是
  Phase 3 的核心研究坑。**
- **T5 (predict_attr_with_edges) 在扩散上是 trivial**：固定全部 edges
  + 部分 attrs，只 denoise 剩余 attrs 维度。
- **T4 (attr-only, 诊断用) 对 autoregressive 不自然**：GRAN 一次一行，
  拿到完整 `A` 后再预测 attrs 相当于"把 attr head 当成一个图神经
  分类器再跑一遍"；GRAN v2 目前就是这么做的。

---

## 1. 问题定义

### Bubble diagram generation ≠ floorplan generation

```
                   ┌────────────────────────────────┐
                   │  Bubble diagram generator      │   ← 本项目 (GRAN v2)
                   │  IN:  nothing / partial graph  │
                   │  OUT: (A, room_labels)         │
                   └────────────────────────────────┘
                                     │
                                     ▼  （下游消费）
                   ┌────────────────────────────────┐
                   │  Floorplan layout generator    │
                   │  IN:  bubble diagram + 约束     │
                   │  OUT: coords / pixels / walls  │
                   │  e.g. GSDiff, HouseDiffusion,  │
                   │       HouseGAN++, Graph2Plan   │
                   └────────────────────────────────┘
```

**为什么容易搞混**：上述下游论文自身的 abstract 里都画 bubble diagram，
但他们假设 bubble diagram 是 user-supplied。我们的任务是 `P(bubble)`，
他们的任务是 `P(layout | bubble)`。

### 5 个推理任务的严格定义

参考 `GRAN/model/gran_v2.py`：

- **T1** `_sampling(B)` → `(A, attrs)`
- **T2** `_sampling(B, partial_A, partial_attrs, start_idx=k)` → 补全
- **T3** `expand_graph(partial_A, partial_attrs, num_target_nodes=N')`
- **T4** `forward({'attr_only': True, 'A': A})` →  per-node logits
- **T5** `predict_attr_with_edges(partial_A, partial_attrs, fixed_edges)`
  → 新节点 attr

这 5 个是 app 的对外 API，任何新 backbone 必须全部支持。

---

## 2. 文献审计：谁是真的 graph-level generator？

### 2.1 三分类

| 类别 | 输出是图 | 输出是空间 | 可双用 |
|------|---------|-----------|-------|
| GRAPH-LEVEL generative | ✅ | ❌ | — |
| FLOORPLAN-LEVEL generative | ❌ | ✅ | — |
| MIXED / 可切换 | ✅ | ✅ | ⚠ |

### 2.2 本文移除的 floorplan-only references（之前的稿子误算了）

这些**不是** bubble-diagram 生成器，而是 bubble-diagram → layout
映射器。不应作为 backbone 候选，只保留作为下游消费方：

| 论文 | 会议 | 为什么不是 bubble gen | 我们怎么用 |
|------|------|---------------------|----------|
| [HouseGAN++](https://arxiv.org/abs/2103.02574) | CVPR 2021 | relational GAN 从 bubble → raster floorplan | 下游 layout |
| [Graph2Plan](https://arxiv.org/abs/2004.13204) | SIGGRAPH 2020 | GNN+CNN 从 bubble+boundary → floorplan | 下游 layout |
| [HouseDiffusion](https://arxiv.org/abs/2211.13287) | CVPR 2023 | bubble+种子 → vector floorplan diffusion | 下游 layout |
| [GSDiff](https://github.com/SizheHu/GSDiff) | AAAI 2025 | bubble → corners + walls (本仓库上游) | 下游 layout |
| [ChatHouseDiffusion](https://arxiv.org/html/2410.11908v1) | 2024 | text → floorplan | 下游 |
| [HouseTune](https://arxiv.org/html/2411.12279) | 2024 | text → LLM plan → diffusion | 下游 |
| [DStruct2Design](https://arxiv.org/html/2407.15723v1) | 2024 | 约束 → floorplan | 下游 |
| [Nursing Unit Stable Diffusion](https://www.mdpi.com/2075-5309/14/9/2601) | 2024 | 图像空间扩散出护理单元像素 | 医疗下游参考 |

### 2.3 MIXED — 模糊地带，需要说清楚

| 论文 | 混在哪里 | 对我们有用吗 |
|------|---------|------------|
| [DBGNN + VGAE](https://www.mdpi.com/2076-3417/15/8/4490) (Applied Sciences 2025) | 摘要里主打 "bubble diagram generation"，VAE 一侧确实输出图；但联合训练里还是走向 floorplan。**最接近**我们任务的唯一一篇 | 值得读透、复现 baseline。注意验证它是否真的做 T1-T5，极可能只做 T1 |
| [MSD benchmark](https://caspervanengelenburg.github.io/msd-eccv24-page/) (ECCV 2024) | 既有 bubble diagram 也有 spatial layout，benchmark 本身不是 generator | 未来大图数据集 |

### 2.4 真正的 GRAPH-LEVEL 候选 backbones

| 族 | 代表 | 会议 | repo |
|----|------|------|------|
| **自回归 GNN** | [GRAN](https://arxiv.org/abs/1910.00760) | NeurIPS 2019 | [lrjconan/GRAN](https://github.com/lrjconan/GRAN) |
| | [GraphRNN](https://arxiv.org/abs/1802.08773) | ICML 2018 | [JiaxuanYou/graph-generation](https://github.com/JiaxuanYou/graph-generation) |
| | [DeepGMG (DGMG)](https://arxiv.org/abs/1803.03324) | ICML 2018 | [DGL example](https://github.com/awslabs/dgl-lifesci/tree/master/examples/generative_models/dgmg) |
| | [GraphAF](https://arxiv.org/abs/2001.09382) (flow) | ICLR 2020 | [DeepGraphLearning/GraphAF](https://github.com/DeepGraphLearning/GraphAF) |
| **离散扩散** | [DiGress](https://arxiv.org/abs/2209.14734) | ICLR 2023 | [cvignac/DiGress](https://github.com/cvignac/DiGress) |
| | [EDGE](https://arxiv.org/abs/2305.04111) | ICML 2023 | [tufts-ml/graph-generation-EDGE](https://github.com/tufts-ml/graph-generation-EDGE) |
| **连续 SDE** | [GDSS](https://arxiv.org/abs/2202.02514) | ICML 2022 | [harryjo97/GDSS](https://github.com/harryjo97/GDSS) |
| **一次性 / 谱** | [SPECTRE](https://arxiv.org/abs/2204.01613) | ICML 2022 | [KarolisMart/SPECTRE](https://github.com/KarolisMart/SPECTRE) |
| **Graph Transformer**（判别，非生成） | [Graphormer](https://arxiv.org/abs/2106.05234) | NeurIPS 2021 | [microsoft/Graphormer](https://github.com/microsoft/Graphormer) |
| | [GRIT](https://arxiv.org/abs/2305.17589) | ICML 2023 | [LiamMa/GRIT](https://github.com/LiamMa/GRIT) |
| | [SAN](https://arxiv.org/abs/2106.03893) | NeurIPS 2021 | [DevinKreuzer/SAN](https://github.com/DevinKreuzer/SAN) |

---

## 3. 5-Task 覆盖矩阵（核心表）

图例：**N** = native（原生，论文 demo 过或 trivially 支持）；
**P** = possible with adapter（需要写一个不 trivial 但已知的扩展，
如 inpainting mask、attr head）；**X** = no / 论文范式不支持。

每个 cell 后面跟一句话解释和 **limitations**（论文做不了什么）。

### 3.1 GRAN (NeurIPS 2019) — 我们现在的 backbone

| 任务 | 评级 | 说明 |
|------|-----|------|
| T1 uncond | **N** | 按节点顺序自回归生成行 mixture-of-Bernoulli，是论文主任务 |
| T2 partial | **N** | 只需把 partial 前缀塞进 A 后让 loop 从 `start_idx` 启动（v2 已实现） |
| T3 expand | **N** | 同 T2；继续跑 loop 到目标 N' |
| T4 attr-only | **P** | GRAN 原文没有 attr head，v2 后装了一个 MLP。给定完整 A 做 classification 需要跑一次 GNN 前向，不是采样路径 |
| T5 fixed-edges | **P** | v2 `predict_attr_with_edges` 专门写了一条分支：skip Bernoulli sample，用 user-specified edges 覆盖 A，然后吃 attr head |

**Limitations（GRAN 做不好什么）：**
- Ordering bias：对 canonical ordering 敏感，训练时见的是 DFS/BFS，
  推理时拿到"其他"ordering 直接崩
- O(N³) 采样复杂度 — 30+ 节点体感差
- Attr 和 structure 解耦 — 生成的图经常 0 个 Living 或 2 个 Living
- 没法做"在中间插节点"—必须严格从节点 k+1 往后续

### 3.2 GraphRNN (ICML 2018)

| 任务 | 评级 | 说明 |
|------|-----|------|
| T1 | **N** | 两个 RNN 级联生成图，标准任务 |
| T2 | **P** | 原论文没有公开的 partial-conditioning 接口；需要在 graph-level RNN 的 hidden state 里塞入前缀的 one-hot encoding |
| T3 | **P** | 同 T2 |
| T4 | **X** | 原论文几乎不处理 node labels；appendix 才提到 attribute 扩展且"没用 attr 质量的评估指标" |
| T5 | **X** | 同 T4 |

**Limitations：**
- 结构纯净，attrs 是事后加的，没有官方的 labeled-graph 实现
- BFS ordering 硬编码
- 显存开销对大图不友好（edge-level RNN 是 O(N²) 序列）

### 3.3 DeepGMG / DGMG (ICML 2018)

| 任务 | 评级 | 说明 |
|------|-----|------|
| T1 | **N** | 原文就是 "generate node + edges sequentially"，paper 明确包含 attribute 生成 |
| T2 | **P** | 决策序列可以从任意中间状态 resume，但需要写 state deserialization |
| T3 | **P** | 同 T2 |
| T4 | **P** | DGMG 在每步都输出 node type 分布，给完整 A 可以再跑一次取每个 node 的 classification；但非官方路径 |
| T5 | **P** | 同 T4，需要固定 edge 决策序列 |

**Limitations：**
- 训练慢（多次小决策，每步都要 GNN propagate）
- 论文作者自己说它更适合 molecular graphs（几十节点内）
- 没有大规模 repo 维护，DGL 里是 example code

### 3.4 DiGress (ICLR 2023) — 我们的 P3 首选

| 任务 | 评级 | 说明 |
|------|-----|------|
| T1 | **N** | 离散扩散生图，node + edge categorical，核心任务 |
| T2 | **P** | 论文 "scaffold extension" 就是类似机制：把已知节点/边冻结，只 denoise 剩余部分。不是 out-of-the-box 接口，需要写 mask |
| T3 | **P** | 同 T2，但要小心：插入新节点意味着扩 adjacency 维度，而扩散模型通常在固定节点数上运行。要么预分配 N_max 槽位，要么用 EDGE 风格的 sparse 扩散 |
| T4 | **N** | 固定所有 edges 的 categorical，只 denoise node categorical。原生支持，本质是 conditional node classification via denoising |
| T5 | **N** | 固定大部分 edges + 部分 attrs，denoise 新节点 attr。Trivial mask setup |

**Limitations：**
- O(N²) 每个 timestep（全 attention），5000+ 节点需要 EDGE 这种
  稀疏变体
- 训练数据要求 > GRAN（大量 noise levels × 全 graph 预测）
- "插中间节点" 需要 pre-allocation；否则邻接维度变化不可微
- 没有 ordering — 对我们反而是优点

### 3.5 EDGE (ICML 2023)

| 任务 | 评级 | 说明 |
|------|-----|------|
| T1 | **N** | 稀疏离散扩散，扩到 1000+ 节点 |
| T2 | **P** | 理论上和 DiGress 一样可以 inpainting，但 repo 没 demo，需要自己写 |
| T3 | **P** | EDGE 每步只改 O(edges) 条，插节点比 DiGress 简单 |
| T4 | **P** | EDGE 专注结构 + 度约束，node attr head 需要自己加 |
| T5 | **P** | 同 T4 |

**Limitations：**
- **EDGE 原文强调 degree-guided，不原生建模 node categorical attrs** —
  node label 要自己接
- 稀疏实现比 DiGress 复杂，调试成本高
- 医院图几百节点正好是它的甜点，住宅图 6 节点反而 overkill

### 3.6 GDSS (ICML 2022)

| 任务 | 评级 | 说明 |
|------|-----|------|
| T1 | **N** | 联合 SDE on (node feat, adj)，文中主任务 |
| T2 | **P** | 同 DiGress，inpainting mask 可行但需自己写 |
| T3 | **P** | 连续 SDE 插节点尤其麻烦（维度变化要么 pad 要么重 SDE） |
| T4 | **P** | 固定 adj 的 SDE 只驱动 node feat — 离散 label 需要 softmax 化连续输出 |
| T5 | **P** | 同 T4 + partial attr masking |

**Limitations：**
- Node attrs 是连续 feature vector，不原生离散类别 → 做 room type
  会 awkward（要么 soft one-hot，要么外挂 categorical head）
- 收敛慢，score matching loss 不稳定

### 3.7 SPECTRE (ICML 2022)

| 任务 | 评级 | 说明 |
|------|-----|------|
| T1 | **N** | 一次性生成谱 + eigenvector，再 reconstruct graph |
| T2 | **P** | 可以把 partial 当做 observed eigenspectrum 约束，但非官方 |
| T3 | **X** | 一次性出全图；插节点意味着重做整个谱 |
| T4 | **X** | 模型不输出 node labels |
| T5 | **X** | 同 T4 |

**Limitations：**
- 不是 attributed graph 模型
- 一次性生成不适合 T2/T3/T5 这种"增量式"任务
- 对我们的产品价值 = 0，除非做纯拓扑 baseline

### 3.8 Graphormer / GRIT (graph transformers)

| 任务 | 评级 | 说明 |
|------|-----|------|
| T1 | **X** | 判别模型，默认没有 generative head |
| T2 | **X** | 同 T1 |
| T3 | **X** | 同 T1 |
| T4 | **N** | 把它当 encoder + node classification head 就是 T4 |
| T5 | **N** | 同 T4 |

**Limitations：**
- **它们不是 generative models**，直接拿来做 T1-T3 等于重写
- 作为 GRAN 的 backbone 替换 backbone（保留自回归外壳）才是正确用法
  — 这是 P2 的路子

### 3.9 GraphAF (ICLR 2020, flow-based)

| 任务 | 评级 | 说明 |
|------|-----|------|
| T1 | **N** | Autoregressive flow 生成节点 + 边 |
| T2 | **P** | 和 GRAN 同样原理，flow 可 resume |
| T3 | **P** | 同 T2 |
| T4 | **P** | 有 atom-type head；要改成 room-type |
| T5 | **P** | 需要把 edge flow 在对应步 clamp |

**Limitations：**
- 主打分子，floorplan domain 没有公开 port
- 可逆约束让架构设计受限
- 速度介于 GRAN 和 DiGress 之间，没有特别优势

### 3.10 总表（供速查）

| Backbone | T1 | T2 | T3 | T4 | T5 | 对我们是否候选 |
|---------|----|----|----|----|----|-------------|
| GRAN (v2 当前) | N | N | N | P | P | ✅ P1 |
| GraphRNN | N | P | P | X | X | ❌ attr 支持差 |
| DeepGMG | N | P | P | P | P | ⚠ 慢、无维护 |
| GraphAF | N | P | P | P | P | ⚠ 分子向 |
| DiGress | N | P | P | N | N | ✅ P3 |
| EDGE | N | P | P | P | P | ⚠ P3 大图变体 |
| GDSS | N | P | P | P | P | ⚠ 连续 attr awkward |
| SPECTRE | N | P | X | X | X | ❌ 无 attr |
| Graphormer/GRIT | X | X | X | N | N | ⚠ 只能当 backbone（P2） |

---

## 4. 三阶段 Roadmap（新版）

### Phase 1：住宅 demo（当前，≤ 15 节点）— 留在 GRAN v2

**推荐 backbone：** GRAN v2 + GATv2（已落地）

**5-task 覆盖：** T1/T2/T3 ✅；T4/T5 通过 attr head + v2 的 Mode B 分支
支持，但耦合较弱（见 §3.1 limitations）。

**继续改进（按 ROI）：**
- A. 多 canonical ordering（已加入 v2）
- B. Degree-rank 结构 embedding（已加入 v2）
- C. 训练时随机 partial mask — 让 T2/T3 训练分布匹配推理分布
- D. 扩数据到 40-70k（publication 级别才有意义）

### Phase 2：中型（≤ 30 节点）— GRAN v2 + Graph Transformer backbone

**推荐 backbone：** `GRANv2` 外壳不变，把 GATv2 替换为
[Graphormer](https://github.com/microsoft/Graphormer) 或
[GRIT](https://github.com/LiamMa/GRIT)（Laplacian / random-walk PE）

**为什么不直接跳 P3：**
- 自回归 loop 不变 → T1-T5 全部继续工作，不需要重写 app 的推理层
- Graph transformer 的 Laplacian PE 解决 GRAN 的 ordering bias
- 30 节点还在 O(N³) 可容忍的范围内

**新增工作量：**
- `model.backbone: 'gatv2' | 'graphormer' | 'grit'` 配置开关
- SPD / Laplacian PE 计算（离线即可）
- MSD 数据集接入（或自爬商业建筑图）

### Phase 3：医疗/大型（30-500 节点）— 切到 DiGress

**推荐 backbone：** [DiGress](https://github.com/cvignac/DiGress) 离散
扩散，带 inpainting mask 覆盖 T1-T5。超过 ~200 节点再 fallback 到
[EDGE](https://github.com/tufts-ml/graph-generation-EDGE)（sparse 变体）。

**5-task 覆盖（honest 版本）：**

| 任务 | 实现策略 | 已知问题 |
|------|---------|---------|
| T1 | 原生 | 数据够的话没事 |
| T2 | Inpainting mask：fix 前 k 个节点的 node+edge categorical，denoise 剩余 | 需要 pre-allocate N_max 槽位；未出现节点的 "empty" categorical 要单独列 |
| T3 | 同 T2，但扩 N_max；或者在 N_max 上做 padding 并用 special "absent" 类 | **最危险的一个** — DiGress scaffold extension demo 的是"加边"，不是"加节点"；需要我们自己写 schedule |
| T4 | 固定所有 edge categorical，把 node categorical 设为全噪声，做若干步去噪 | 其实是 conditional denoising，原生支持，和 Bubble→Attr 任务等价 |
| T5 | 固定 edges + 部分 attrs，denoise 其余 attrs 维度 | 最 trivial 的 mask 配置 |

**为什么 DiGress 而不是 GSDiff/HouseDiffusion：**
- GSDiff/HouseDiffusion 都是 bubble → **空间 layout** 的模型，**它们
  不生成 bubble diagram**，只消费 bubble diagram。前一稿把它们列为
  backbone 候选是 scope 错误。
- DiGress 就是 "discrete diffusion on labeled graphs"，和我们的
  数学任务 1:1 对齐

**新增工作量：**
- Fork DiGress，改 node / edge categorical taxonomy 到房间类别
- 写 T2/T3 的 node-insertion schedule（不是 out-of-the-box）
- 新 denoiser = graph transformer（可复用 P2 的实现）
- 自建医疗数据集（最大未知项）

### 不推荐

- **SPECTRE** — 无 attr，T3/T4/T5 基本做不了
- **GraphRNN** — attr 支持只在 appendix，无官方 evaluation
- **纯 Graphormer** — 判别模型，不做生成。除非作为 P2 的 GRAN
  backbone 替换件
- **GSDiff/HouseDiffusion/HouseGAN++** — 已在 §2.2 移除，下游消费
  方而非 backbone 候选

---

## 5. 具体下一步

### 如果继续走 GRAN 轨道
1. 做 Task §4 里 P1-C（训练时 partial masking）
2. 在 MSD（20-50 节点）上验证 P2 的 Graph Transformer backbone
3. 评测 5 个任务都跑通

### 如果要为 P3 降风险（现在就能做）
1. 保持 `GRANDataV2` 的 `(adj, attrs)` 抽象 — DiGress 直接复用
2. 7 类 → 30+ 类 taxonomy 分版本存，不要合并
3. MMD 加 Laplacian 谱特征 — 大图上 degree 没用
4. 把 `expand_graph` 和 `predict_attr_with_edges` 的单元测试固化，
   这是跨 backbone 的 API 合同

### 如果直接开 P3
1. Fork DiGress，接 RPLAN dataloader（先验证 T1 等价 MMD）
2. 写 mask-based T4 / T5，做 apples-to-apples 比较
3. 写 T2 / T3 的 node-insertion mask schedule — 这是新研究工作，
   不是 engineering

---

## 6. 参考文献

### Graph-level generative（真正的 backbone 候选）
- [GRAN (NeurIPS 2019)](https://arxiv.org/abs/1910.00760) ([code](https://github.com/lrjconan/GRAN))
- [GraphRNN (ICML 2018)](https://arxiv.org/abs/1802.08773) ([code](https://github.com/JiaxuanYou/graph-generation))
- [DeepGMG / DGMG (ICML 2018)](https://arxiv.org/abs/1803.03324)
- [GraphAF (ICLR 2020)](https://arxiv.org/abs/2001.09382) ([code](https://github.com/DeepGraphLearning/GraphAF))
- [DiGress (ICLR 2023)](https://arxiv.org/abs/2209.14734) ([code](https://github.com/cvignac/DiGress))
- [EDGE (ICML 2023)](https://arxiv.org/abs/2305.04111) ([code](https://github.com/tufts-ml/graph-generation-EDGE))
- [GDSS (ICML 2022)](https://arxiv.org/abs/2202.02514) ([code](https://github.com/harryjo97/GDSS))
- [SPECTRE (ICML 2022)](https://arxiv.org/abs/2204.01613) ([code](https://github.com/KarolisMart/SPECTRE))

### Graph transformer（backbone 替换 / 判别头）
- [Graphormer (NeurIPS 2021)](https://arxiv.org/abs/2106.05234) ([code](https://github.com/microsoft/Graphormer))
- [GRIT (ICML 2023)](https://arxiv.org/abs/2305.17589) ([code](https://github.com/LiamMa/GRIT))
- [SAN (NeurIPS 2021)](https://arxiv.org/abs/2106.03893)
- [GATv2 (ICLR 2022)](https://arxiv.org/abs/2105.14491) — 我们 v2 的 backbone

### MIXED — 部分相关
- [DBGNN + VGAE (Applied Sciences 2025)](https://www.mdpi.com/2076-3417/15/8/4490) — 最接近"bubble 生成器"的公开论文，待复现验证
- [MSD benchmark (ECCV 2024)](https://caspervanengelenburg.github.io/msd-eccv24-page/) ([code](https://github.com/caspervanengelenburg/msd)) — 未来大图数据集

### Floorplan downstream（只作为 bubble 消费方，不是 backbone 候选）
- [HouseGAN++ (CVPR 2021)](https://arxiv.org/abs/2103.02574)
- [Graph2Plan (SIGGRAPH 2020)](https://arxiv.org/abs/2004.13204)
- [HouseDiffusion (CVPR 2023)](https://arxiv.org/abs/2211.13287)
- [GSDiff (AAAI 2025)](https://github.com/SizheHu/GSDiff) — 本仓库上游
- [ChatHouseDiffusion (2024)](https://arxiv.org/html/2410.11908v1)
- [HouseTune (2024)](https://arxiv.org/html/2411.12279)
- [DStruct2Design (2024)](https://arxiv.org/html/2407.15723v1)

Sources:
- [DiGress: Discrete Denoising diffusion for graph generation (arXiv 2209.14734)](https://arxiv.org/abs/2209.14734)
- [GRAN (arXiv 1910.00760)](https://arxiv.org/abs/1910.00760)
- [GraphRNN (arXiv 1802.08773)](https://arxiv.org/abs/1802.08773)
- [DeepGMG (arXiv 1803.03324)](https://arxiv.org/abs/1803.03324)
- [EDGE (arXiv 2305.04111)](https://arxiv.org/abs/2305.04111)
- [GDSS (arXiv 2202.02514)](https://arxiv.org/abs/2202.02514)
