# 计划：从住宅（4-8 节点）扩展到医疗场景（30+ 节点）

> **状态：** 策略文档。说明当 app 从 RPLAN 住宅户型（4-8 房间）扩展到医疗 /
> 大型商业平面（30-200+ 房间）时，如何演进当前 GRAN v2 技术栈。
>
> **目标：** 规划一个分阶段迁移路径，保留当前住宅 demo 能力，同时为更大规
> 模图的生成做准备。

---

## TL;DR — 一页总结

### 核心问题

- **住宅（4-8 节点）** 和 **医疗（30-200+ 节点）** 是两个数量级
- GRAN v2 的 **O(N³) 自回归复杂度** 在 30+ 节点开始成为 app 性能瓶颈
- 医疗平面的 **多 hub、层级结构、规范约束** 跟住宅完全不同，我们现在的
  "degree rank" 结构特征不够用

### V2 必须保留的 5 个能力

未来换架构不能丢的功能：

1. **T1 无条件生成** — 从噪声生成完整 floorplan
2. **T2 Partial graph completion** — 给定前 t 个节点，续生成
3. **T3 只预测属性** — 给定完整邻接，预测每个节点的房间类型
4. **T4 `expand_graph`（Mode A）** — T2 的便利包装
5. **T5 `predict_attr_with_edges`（Mode B）** — 给定连接模式，预测新节点类型

### 三阶段 roadmap

| 阶段 | 目标节点数 | 架构 | 工期 |
|------|-----------|------|------|
| **Phase 1（现在）** | ≤ 15 | GRAN v2 + GATv2 + 多 ordering + degree rank | 进行中 |
| **Phase 2（中期）** | ≤ 30 | GRAN v2 + **Graph Transformer backbone**（换 GNN，保留自回归） | 2-3 周 |
| **Phase 3（长期）** | 30-500 | **Diffusion**（DiGress / EDGE）全面重构 | 6-10 周 + 数据集 |

### 关键判断

- ✅ **Phase 2 最安全**：只换 backbone，其他全部保留，5 个任务都能跑
- ✅ **Phase 3 最干净**：Diffusion 用 **inpainting 一套 mask** 统一实现全部 5 个任务，代码量比 GRAN v2 少很多
- ❌ **纯 Graphormer 做不了**：它是分类模型不是生成模型
- ⚠️ **医疗数据集要自己搞**：公开数据里 MSD（ECCV 2024，瑞士多单元建筑）最接近但还不够

### 现在就能做的"降低 Phase 3 风险"的事

1. 数据加载器保持 `(adj, attrs)` 抽象 — diffusion 能直接复用
2. 类别 taxonomy 分版本维护 — 7 类 → 30+ 类时不改历史文件
3. `max_num_nodes` 参数化 — 已经做了
4. MMD 用 Laplacian 谱特征 — 大图上这比 degree 更有用
5. 关注 MSD benchmark，2025 年底会有公开 baseline 可对比

---

## 详细版本

## 1. 医疗平面 vs 住宅：结构差异

| 维度 | 住宅（RPLAN） | 医疗 |
|------|---------------|------|
| 每张图节点数 | 4-8 | 30-200+ |
| Hub 数量 | 通常 1 个（客厅） | **多个**（护士站、电梯厅、接待、食堂……） |
| 拓扑结构 | 扁平树状 | **层级**（区 ↔ 走廊 ↔ 房间） |
| 类别数 | 7 种房间 | **20-40 种**（病房、手术室、ICU、实验室、药房、办公、后勤……） |
| 核心 pattern | 客厅周围环绕房间 | 长**走廊主干**，房间挂在走廊两侧 |
| 规范约束 | 美学 / 生活习惯 | **规范合规**（消防出口、清洁区 vs 污染区分流、最小走廊宽度） |
| 数据集 | RPLAN ~7 万、LIFULL ~12 万 | 稀缺；[MSD](https://caspervanengelenburg.github.io/msd-eccv24-page/)（ECCV 2024, ~5k 瑞士多单元）最接近但仍不是医疗 |

**关键影响**：我们现在 degree-rank embedding 隐含的"Living 在中心"启发式
**不适用**医疗场景。医院平面是多 hub 结构。

---

## 2. GRAN v2 在大图上的瓶颈

现有 pipeline（`GRANv2` + GATv2 + 多 canonical ordering + degree-rank）在
30+ 节点上会撞墙，原因：

### 2.1 复杂度墙

自回归生成是 **O(N³)**：
- N 个自回归步骤
- 每步 GNN 传播需要 O(N²) 时间（节点 × 邻居）

```
8 节点   → ~1 秒/张          （app 体验好）
30 节点  → ~30+ 秒/张        （用户体验变差）
100 节点 → ~10+ 分钟/张      （app 基本不可用）
500 节点 → 实际上跑不动
```

### 2.2 顺序偏差

即使用多 canonical ordering，自回归模型对**长程结构**力不从心（例如
"B 走廊最远端必须有紧急出口"）。GNN 深度受自回归 context 限制，每步只能
看到前缀。

### 2.3 结构特征不够用

Degree rank 在"1 个 hub"的情况下是好信号。医院有 5 个并列 hub 时，
rank-0 信息量太少。需要补充：

- **Clustering coefficient**（邻居之间是否互连）
- **Shortest-path distance to multiple hubs**
- **Community / partition membership**（Louvain、谱聚类）
- **Hierarchical positional encoding**（属于哪个区 / 楼层）

### 2.4 类别爆炸

7 类房间是小 softmax。30+ 类长尾分布（"消毒间"可能 1/1000）是更难的
分类任务，需要更多数据。

---

## 3. 研究现状：大图生成业界怎么做

### 3.1 数据集

| 数据集 | 节点数 | 领域 | 年份 |
|--------|-------|------|------|
| [RPLAN](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) | 4-8 | 中式住宅 | 2019 |
| [LIFULL](https://www.nii.ac.jp/dsc/idr/lifull/) | 4-10 | 日式住宅 | — |
| [HouseGAN++ 派生](https://github.com/ennauata/houseganpp) | 5-15 | 西式住宅 | 2021+ |
| [**MSD**](https://caspervanengelenburg.github.io/msd-eccv24-page/)（ECCV 2024）([code](https://github.com/caspervanengelenburg/msd)) | **up to ~50** | 瑞士多单元建筑 | 2024 |
| 医院数据集 | 30-200+ | 领域专有，多为私有 | — |

**医疗方向几乎肯定需要自建数据集。** MSD 是最接近的公开 benchmark，
但仍是住宅 / 多单元，不是医院。

### 3.2 架构按可扩展性排序

**1. 自回归 GNN（GRAN, GraphRNN）** — 现在用的
- 20 节点以内性能好
- 50+ 因为 O(N³) 复杂度崩溃

**2. Graph Transformer**（全局注意力）

| 方法 | 会议 | 特点 | 代码 |
|------|------|------|------|
| [Graphormer](https://arxiv.org/abs/2106.05234) | NeurIPS 2021 | 用 SPD bias 作为空间编码 | [microsoft/Graphormer](https://github.com/microsoft/Graphormer) |
| [SAN](https://arxiv.org/abs/2106.03893) | NeurIPS 2021 | Laplacian PE | [DevinKreuzer/SAN](https://github.com/DevinKreuzer/SAN) |
| [GRIT](https://arxiv.org/abs/2305.17589) | ICML 2023 | Random-walk PE，无需 message passing，graph transformer 当前 SOTA | [LiamMa/GRIT](https://github.com/LiamMa/GRIT) |
| [Graph Transformer (Dwivedi)](https://arxiv.org/abs/2012.09699) | DLG-AAAI 2021 | 最简参考实现 | [graphdeeplearning/graphtransformer](https://github.com/graphdeeplearning/graphtransformer) |

- 需要 Laplacian / SPD positional encoding 才有竞争力
- 内存 O(N²)，用稀疏变体能处理 ~500 节点

**3. 离散扩散（DiGress ICLR 2023）**
- [DiGress](https://arxiv.org/abs/2209.14734)（[code](https://github.com/cvignac/DiGress)）
- 对**离散节点 + 边属性**的图，目前最可扩展的生成模型
- 用 graph transformer + Laplacian PE 作 denoiser
- **扩展到 GuacaMol 1.3M 分子**，不用分子专用技巧
- **天然支持 partial graph inpainting**（通过 mask 条件）
- **最适合医疗场景**

**4. 连续 score-based 图生成**

| 方法 | 会议 | 特点 | 代码 |
|------|------|------|------|
| [GDSS](https://arxiv.org/abs/2202.02514) | ICML 2022 | 联合对节点 + 邻接做 SDE | [harryjo97/GDSS](https://github.com/harryjo97/GDSS) |
| [SPECTRE](https://arxiv.org/abs/2204.01613) | ICML 2022 | 一次性谱条件生成，大图表现强 | [KarolisMart/SPECTRE](https://github.com/KarolisMart/SPECTRE) |
| [GraphAF](https://arxiv.org/abs/2001.09382) | ICLR 2020 | Flow-based 自回归 baseline | [DeepGraphLearning/GraphAF](https://github.com/DeepGraphLearning/GraphAF) |

**5. 高效 / 可扩展图扩散（后 DiGress）**

| 方法 | 会议 | 特点 | 代码 |
|------|------|------|------|
| [EDGE](https://arxiv.org/abs/2305.04111) | ICML 2023 | **稀疏**离散扩散，扩展到上千节点，显式度建模。**医院图超过 100 节点时直接相关** | [tufts-ml/graph-generation-EDGE](https://github.com/tufts-ml/graph-generation-EDGE) |
| [Iterative Local Expansion](https://arxiv.org/abs/2312.11529) | ICLR 2024 | 稀疏 Markov transition，层级 coarse-to-fine denoising | [AndreasBergmeister/graph-generation](https://github.com/AndreasBergmeister/graph-generation) |

N > 500 基本必须用这类方法。

**6. 3D / 几何参考（非 floorplan 但相关）**
- [EDM](https://arxiv.org/abs/2203.17003)（[code](https://github.com/ehoogeboom/e3_diffusion_for_molecules)）ICML 2022 — E(3) 等变扩散，同时处理连续坐标 + 离散类型。如果 Stage-2 几何放置改成 diffusion，这是模板

### 3.3 Floorplan 专门工作

| 论文 | 会议 | 做什么 | 代码 | 对我们有用吗 |
|------|------|--------|------|------------|
| [HouseGAN++](https://arxiv.org/abs/2103.02574) | CVPR 2021 | Relational GAN + layout refinement，bubble diagram 条件 | [ennauata/houseganpp](https://github.com/ennauata/houseganpp) | ≤15 房间强 baseline |
| [Graph2Plan](https://arxiv.org/abs/2004.13204) | SIGGRAPH 2020 | GNN + CNN，layout graph + boundary → 光栅 floorplan | [HanHan55/Graph2plan](https://github.com/HanHan55/Graph2plan) | 条件生成的前身 |
| [HouseDiffusion](https://openaccess.thecvf.com/content/CVPR2023/papers/Shabani_HouseDiffusion_Vector_Floorplan_Generation_via_a_Diffusion_Model_With_Discrete_CVPR_2023_paper.pdf) | CVPR 2023 | 离散 + 连续去噪，bubble diagram 条件 | [aminshabani/house_diffusion](https://github.com/aminshabani/house_diffusion) | Vector floorplan head 参考 |
| [GSDiff](https://wutomwu.github.io/publications/2025-GSDiff/paper.pdf) | AAAI 2025 | 两阶段：扩散节点 → 扩散边 | [SizheHu/GSDiff](https://github.com/SizheHu/GSDiff) | 上游姊妹项目，可直接复用 |
| [DStruct2Design](https://arxiv.org/html/2407.15723v1) | 2024 | 数据结构驱动的 constraint 满足，基于 LLaMA-3 | [plstory/DS2D](https://github.com/plstory/DS2D) | Partial-constraint 生成模板 |
| [HouseTune](https://arxiv.org/html/2411.12279) | 2024 | 两阶段 LLM + diffusion | 暂无公开代码 | 自然语言交互参考 |
| [ChatHouseDiffusion](https://arxiv.org/html/2410.11908v1) | 2024 | Text-conditioned floorplan diffusion + editing | [ChatHouseDiffusion/chathousediffusion](https://github.com/ChatHouseDiffusion/chathousediffusion) | Text-to-floorplan 交互模板 |
| [DBGNN+VGAE](https://www.mdpi.com/2076-3417/15/8/4490) | Applied Sciences 2025 | Dual-branch GNN + VAE 生成 bubble diagram | — | 同我们这个 niche |
| [Nursing Unit via Stable Diffusion](https://www.mdpi.com/2075-5309/14/9/2601) | 2024 | 图像空间扩散生成医院护理单元 | — | **第一篇医疗扩散论文** |
| [MSD benchmark](https://caspervanengelenburg.github.io/msd-eccv24-page/) | ECCV 2024 | 5k 瑞士多单元建筑数据集 + benchmark | [caspervanengelenburg/msd](https://github.com/caspervanengelenburg/msd) | 采用为标准 benchmark |

---

## 3.5 V2 任务覆盖矩阵 — 换架构后 5 个能力还能做吗

**这是最关键的一节**：v2（见 `docs/plan-gran-v2-upgrade.md` Tasks 1-7）给
app 提供的 **5 种推理模式**，未来迁移架构**必须全部保留**，否则产品
回退。

| V2 任务 | 作用 | API |
|--------|------|-----|
| **T1** 无条件生成 | 从零采样完整 floorplan | `forward({'is_sampling': True})` |
| **T2** Partial graph completion | 给定前缀 `(A_partial, attrs_partial)`，续生成 | `_sampling(..., partial_A, partial_attrs, start_idx)` |
| **T3** 只预测属性 | 给定完整图，预测每个节点类型 | `_sampling(..., partial_A=full, start_idx=N)` |
| **T4** `expand_graph` (Mode A) | T2 的便利包装，扩到 `num_target_nodes` | `expand_graph(partial_A, partial_attrs, num_target_nodes)` |
| **T5** `predict_attr_with_edges` (Mode B) | 给定 partial + 下一个节点的连接，预测属性 | `predict_attr_with_edges(partial_A, partial_attrs, fixed_edges)` |

### 架构对这 5 个任务的支持情况

| 架构 | T1 | T2 | T3 | T4 | T5 | 备注 |
|------|----|----|----|----|----|------|
| **GRAN v2 + GATv2（当前）** | ✅ | ✅ | ✅ | ✅ | ✅ | 已实现并测试 |
| **GRAN + Graph Transformer**（[Graphormer](https://github.com/microsoft/Graphormer) / [GRIT](https://github.com/LiamMa/GRIT) 作 backbone） | ✅ | ✅ | ✅ | ✅ | ✅ | 自回归循环不变，只换 backbone |
| **纯 Graph Transformer**（[Graphormer](https://github.com/microsoft/Graphormer)） | ❌ | ❌ | ⚠️ | ❌ | ⚠️ | 是分类/回归框架，不是生成模型 |
| **Graph Diffusion**（[DiGress](https://github.com/cvignac/DiGress), [EDGE](https://github.com/tufts-ml/graph-generation-EDGE)） | ✅ | ✅ | ✅ | ✅ | ✅ | 用 inpainting-style mask 全覆盖 |
| **Continuous SDE**（[GDSS](https://github.com/harryjo97/GDSS)） | ✅ | ✅ | ✅ | ✅ | ⚠️ | 同上，但 T5（fixed edges）需离散风格 mask |
| **一次性谱生成**（[SPECTRE](https://github.com/KarolisMart/SPECTRE)） | ✅ | ⚠️ | ⚠️ | ⚠️ | ❌ | 为无条件一次性生成设计 |
| **自回归 flow**（[GraphAF](https://github.com/DeepGraphLearning/GraphAF)） | ✅ | ✅ | ✅ | ✅ | ✅ | 跟 GRAN 类似，自回归天然支持 |

图例：✅ 原生支持 / ⚠️ 能做但需额外设计 / ❌ 不适合

### 为什么 Diffusion 特别适合我们的任务

Graph diffusion 用**一套 inpainting 机制**覆盖所有 5 个任务：

```python
# 所有 v2 任务共用一套代码
def sample(mask, known_values):
    x = pure_noise()
    for t in reversed(range(T)):
        x = denoise_step(x, t)
        x[mask] = known_values[mask]   # 固定已知部分
    return x
```

不同任务只是不同的 `mask` 配置：

| 任务 | Mask |
|------|------|
| T1 无条件 | 全未知 |
| T2 partial completion | 前 t 个节点的 edges + attrs 固定 |
| T3 attr-only | 所有 edges 固定，所有 attrs 未知 |
| T4 expand_graph | 同 T2 |
| T5 attr-with-fixed-edges | 所有已有 edges + 用户指定的新边固定，新节点 attr 未知 |

对比 GRAN：T2、T3、T4、T5 每个都需要 `_sampling()` 里写专门分支（Task 7
光是这 2 个模式就加了 ~200 行分支逻辑）。Diffusion 压缩成"设 mask 然后
运行"。

**结论**：
- Phase 2（只换 backbone）是最安全迁移 — 同框架、同测试
- Phase 3（diffusion）是最大重构，但给出**最干净**的长期 API

---

## 4. 三阶段 Roadmap

### Phase 1（当前 — 住宅 demo）：保持 GRAN v2

- 数据集：RPLAN 2-4 万子集
- 架构：`GRANv2` + GATv2 + 多 ordering + degree rank
- 交付：MVP app 带 `expand_graph` / `predict_attr_with_edges`
- 时间：进行中

### Phase 2（中期 — ≤30 节点建筑）：换成 Graph Transformer backbone

适用办公、小型商业建筑、过渡规模。**自回归框架保留**，5 个推理模式
（T1-T5）全部继续工作。

**具体任务：**

1. 加 `model.backbone: 'gatv2' | 'graph_transformer'` config 开关
2. 实现 `model/graph_transformer.py`（Graphormer 风格：full attention +
   Laplacian PE + SPD bias）
3. `GRANv2._sampling()` 和 `_inference()` **不动** — 只依赖 backbone 的
   `forward(node_feat, edge, edge_feat)` 接口
4. 在 MSD（或爬取的商业数据集）上重训，`num_canonical_order = 3,
   use_degree_feature: true` 已启用

**保留**：训练循环、loss、5 个推理模式、attr head、可视化、MMD 评估
**改变**：只有 GNN backbone

**预算**：2-3 周

### Phase 3（长期 — 30+ 节点医疗）：切到 Graph Diffusion

当自回归成为瓶颈（速度或质量）时**必需**。DiGress 是参考实现。

**具体任务：**

1. Fork [DiGress](https://github.com/cvignac/DiGress) 或重实现：离散扩散
   于图，支持离散节点 + 边属性
2. 用 graph transformer 做 denoiser（可复用 Phase 2 的实现）
3. 加 Laplacian PE（归一化邻接矩阵的特征向量）作为节点特征
4. Conditional generation（bubble diagram、partial graph、fixed attrs）
   用 inpainting mask — **一个框架覆盖所有 5 个 v2 任务**
5. 改造 RPLAN pipeline（`dataset/rplan_preprocessing/`）产出 diffusion-
   ready 数据，bubble diagram 格式基本不变
6. 自建 / 授权医院 floorplan 数据集（**最大未知项** — 目前没有 RPLAN
   规模的公开 benchmark）
7. 扩展 20-40 类医疗房间的 attribute mapping
8. 在 30+ 节点布局上验证 MMD + 视觉质量

**保留**：数据预处理概念、房间类别 taxonomy 扩展、可视化、MMD 评估
**新增**：训练循环（DDPM / 离散扩散）、采样器（denoising chain）、loss
（categorical CE + noise schedule）

**预算**：6-10 周 + 数据集收集

---

## 5. 现在就可以做的事（降低 Phase 3 风险）

即使在跑住宅 demo 时，也可以让未来医疗迁移更便宜：

1. **数据 loader 保持抽象**：`GRANDataV2` 已经产出 `(adj, attrs)` 元组，
   与消费方解耦。Diffusion 可以用同一份数据，改个 collator 即可
2. **类别 taxonomy 提前扩展**：现在 `RPLAN_TO_GRAN_CLASS` 把 13 类 RPLAN
   合并成 7 类 GRAN。保留 7 类版本，新开一份医疗 taxonomy，不要合并
3. **`max_num_nodes` 参数化**：已完成，未来只需改 config
4. **MMD 用 Laplacian 谱**：已在 safe MMD pipeline 里。切到大图后，谱
   MMD 是主要质量信号（degree / clustering 在层级图上信息量低）
5. **关注 MSD benchmark 的采用情况**：MSD 2024 刚发布，2025 年底会有
   公开 baseline 可对比

---

## 6. 决策树

```
"下一批目标数据是否 ≤ 15 节点？"
  ├── 是 → 保留 GRAN v2，加多 ordering + Laplacian PE 即可
  └── 否
      │
      "≤ 30 节点且要快？"
      ├── 是 → Phase 2（GRAN v2 + Graph Transformer backbone）
      └── 否 → Phase 3（图扩散，如 DiGress）
```

**医疗场景直接 target Phase 3**。不要试图把 GRAN 自回归硬撑到 30+ 节点，
O(N³) 是硬墙。

---

## 7. 如果立即切 Phase，从哪个 repo fork 最划算

### Phase 2（Graph Transformer backbone）首选

| 优先级 | Repo | 理由 |
|-------|------|------|
| 🥇 | [microsoft/Graphormer](https://github.com/microsoft/Graphormer) | 工业级实现、文档齐全、PyTorch 原生，SPD bias 成熟 |
| 🥈 | [LiamMa/GRIT](https://github.com/LiamMa/GRIT) | Graph transformer 当前 SOTA，random-walk PE，小图效果比 Graphormer 更好 |
| 🥉 | [graphdeeplearning/graphtransformer](https://github.com/graphdeeplearning/graphtransformer) | 最简参考实现，便于理解和定制 |

### Phase 3（Graph Diffusion）首选

| 优先级 | Repo | 理由 |
|-------|------|------|
| 🥇 | [cvignac/DiGress](https://github.com/cvignac/DiGress) | 离散图扩散标杆实现，直接支持 categorical node/edge attributes，扩展到 100 节点有实证 |
| 🥈 | [tufts-ml/graph-generation-EDGE](https://github.com/tufts-ml/graph-generation-EDGE) | **稀疏**离散扩散，节点数 > 100 时首选 |
| 🥉 | [SizheHu/GSDiff](https://github.com/SizheHu/GSDiff) | 上游姊妹项目，floorplan 专门，但规模目前只到住宅 |

### 快速对比：Fork DiGress 还是 GSDiff 做 Phase 3

| | DiGress | GSDiff |
|---|---------|--------|
| 图节点规模 | 100-1000+ | < 20 |
| Floorplan-specific | ❌ 通用图 | ✅ |
| 已实现 attr + edge | ✅ 同时 | ✅ |
| 代码成熟度 | 高（ICLR 2023 + 广泛采用） | 中（2025 AAAI 新发） |
| 医院场景适配 | **需适配 domain 知识** | **需 scale up** |

**结论**：
- 小规模（< 20）延用 GSDiff 更省事
- **大规模（30+ 医疗）→ Fork DiGress 起步**，加上 floorplan 特定 loss
  和 taxonomy

---

## 8. 参考文献速查

### 核心架构
- [DiGress](https://arxiv.org/abs/2209.14734) ICLR 2023 ([code](https://github.com/cvignac/DiGress))
- [Graphormer](https://arxiv.org/abs/2106.05234) NeurIPS 2021 ([code](https://github.com/microsoft/Graphormer))
- [GATv2](https://arxiv.org/abs/2105.14491) ICLR 2022
- [GRAN](https://arxiv.org/abs/1910.00760) NeurIPS 2019 ([code](https://github.com/lrjconan/GRAN))
- [GRIT](https://arxiv.org/abs/2305.17589) ICML 2023 ([code](https://github.com/LiamMa/GRIT))
- [GDSS](https://arxiv.org/abs/2202.02514) ICML 2022 ([code](https://github.com/harryjo97/GDSS))
- [SPECTRE](https://arxiv.org/abs/2204.01613) ICML 2022 ([code](https://github.com/KarolisMart/SPECTRE))
- [EDGE](https://arxiv.org/abs/2305.04111) ICML 2023 ([code](https://github.com/tufts-ml/graph-generation-EDGE))
- [GraphAF](https://arxiv.org/abs/2001.09382) ICLR 2020 ([code](https://github.com/DeepGraphLearning/GraphAF))

### Floorplan 专门
- [HouseGAN++](https://arxiv.org/abs/2103.02574) CVPR 2021 ([code](https://github.com/ennauata/houseganpp))
- [Graph2Plan](https://arxiv.org/abs/2004.13204) SIGGRAPH 2020 ([code](https://github.com/HanHan55/Graph2plan))
- [HouseDiffusion](https://openaccess.thecvf.com/content/CVPR2023/papers/Shabani_HouseDiffusion_Vector_Floorplan_Generation_via_a_Diffusion_Model_With_Discrete_CVPR_2023_paper.pdf) CVPR 2023 ([code](https://github.com/aminshabani/house_diffusion))
- [GSDiff](https://wutomwu.github.io/publications/2025-GSDiff/paper.pdf) AAAI 2025 ([code](https://github.com/SizheHu/GSDiff))
- [MSD](https://caspervanengelenburg.github.io/msd-eccv24-page/) ECCV 2024 ([code](https://github.com/caspervanengelenburg/msd))
- [Nursing Unit Stable Diffusion](https://www.mdpi.com/2075-5309/14/9/2601) 2024
- [DStruct2Design](https://arxiv.org/html/2407.15723v1) 2024 ([code](https://github.com/plstory/DS2D))
- [HouseTune](https://arxiv.org/html/2411.12279) 2024
- [ChatHouseDiffusion](https://arxiv.org/html/2410.11908v1) 2024 ([code](https://github.com/ChatHouseDiffusion/chathousediffusion))
- [DBGNN+VGAE](https://www.mdpi.com/2076-3417/15/8/4490) Applied Sciences 2025

### 可扩展性参考
- [ICLR 2024: Efficient and Scalable Graph Generation](https://proceedings.iclr.cc/paper_files/paper/2024/file/7565f036ceb20a2c74d341bfaa9fffad-Paper-Conference.pdf)
- [EDM](https://arxiv.org/abs/2203.17003) ICML 2022 ([code](https://github.com/ehoogeboom/e3_diffusion_for_molecules)) — 3D 几何扩散参考
