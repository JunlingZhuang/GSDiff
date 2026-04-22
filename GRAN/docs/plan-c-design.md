# Plan C — Attr-Conditioned Edge Head + Structure-Aware Attr Head

> **状态**：设计文档，尚未实现。
> **触发**：Plan A（class-weighted CE）证伪后。见 `experiments.md` Exp 5。
> **目标**：让 GRAN v2 学到**结构-属性耦合**，修复以下 bug（attr-aware
> 指标量化）：
> - Living 不是 hub（ref deg 5.20，当前 ~2.5）
> - Balcony 反而是 hub（ref deg 1.63，当前 ~4.9）
> - Living-Living 相邻 +17% 过多
> - Bedroom-Bathroom 主卧套间 -11% 缺失
> - 每图多个 Living（ref 97.7% 为 1 个，当前只 16-22%）

---

## 1. 问题诊断

### 1.1 当前架构（baseline / plan A）

```
      node_embedding h_u, h_v
         │          │
         ▼          ▼
     ┌──────┐   ┌────────┐
     │ attr │   │ edge   │   ← 两个 head 独立
     │ head │   │ head   │
     └──────┘   └────────┘
         │          │
         ▼          ▼
       attr_u     P(edge u-v)
```

**attr head** 只看 `h_u`（node embedding）→ 预测 attr。
**edge head** 只看 `(h_u, h_v)` → 预测 Bernoulli mixture for edge。

**两个 head 不交换信息**。

### 1.2 这为什么导致结构-语义脱节

模型训练时看的数据：
- 真实 Living 节点的 `h_u` 会有较高 degree（因为训练中真实 degree=5）
- 但模型自己**在生成时**不用 degree 做决策——它只用 `h_u` 的 pattern
- 真实 Balcony 节点的 `h_u` 有低 degree，但 `h_u` 包含的其他 pattern 可能
  和 Living 相似（都是"旁边有 bedroom"）
- 模型**没有显式激励去把 attr 和 degree/结构绑在一起**

更严重：生成时模型不知道"这张图已经生成了一个 Living，所以这个新
节点不该再是 Living"——因为 attr head 只看当前 node embedding，**不看
图里已经有哪些 attr**。

---

## 2. Plan C 设计

两个改动，独立可测，建议**分两步**做：

### 2.1 Step C-1：Attr-Conditioned Edge Head（最小改动）

**最小可行改动**：让 edge head 看到端点的 attr embedding。

```
  node emb  attr emb
  h_u  h_v  a_u  a_v
   │    │    │    │
   └────┴────┴────┘
        │
        ▼
    ┌────────┐
    │ edge   │
    │ head   │
    └────────┘
        │
        ▼
   P(edge u-v)
```

#### 实现

```python
# 新增：attr embedding table
self.attr_embedding = nn.Embedding(
    num_attr_classes + 1,      # +1 for "unknown/not yet sampled"
    config.model.attr_embedding_dim,  # 新 config, e.g. 32
)
nn.init.normal_(self.attr_embedding.weight, std=0.1)

# 修改：edge head MLP 的输入维度
# before: edge_mlp(concat[h_u, h_v]) -> edge_dim = 2 * hidden
# after:  edge_mlp(concat[h_u, h_v, a_u, a_v]) -> edge_dim = 2 * (hidden + attr_emb)
```

#### 训练时（teacher forcing）

Edge prediction 的 attr 输入使用**真实 attr**：

```python
# 对每条候选边 (u, v)：
a_u_emb = self.attr_embedding(labels[u])  # ground truth attr
a_v_emb = self.attr_embedding(labels[v])
edge_input = torch.cat([h_u, h_v, a_u_emb, a_v_emb], dim=-1)
edge_logits = self.edge_head(edge_input)
```

#### 推理时（autoregressive）

关键：在 GRAN 的生成顺序里，**节点 k 的 attr 在决定节点 k 的边之前
已经采样**。所以当决策 edge (k, j) 时：

- `a_k`：刚采样出来（attr head 在 edge head 之前跑）
- `a_j`（for j < k）：之前已采样，存在某个 list 里

两个 attr 都有，直接 embed 后 concat 进 edge head：

```python
# _sampling loop at step k:
attr_logits_k = self.attr_head(h_k)
sampled_attr_k = torch.multinomial(softmax(attr_logits_k), 1)
attrs_sampled.append(sampled_attr_k)

for j in range(k):
    a_u_emb = self.attr_embedding(sampled_attr_k)
    a_v_emb = self.attr_embedding(attrs_sampled[j])
    edge_input = torch.cat([h_k, h_j, a_u_emb, a_v_emb])
    ...
```

**保持 AR 因果顺序**：不看未来 attrs。

#### 预期效果

- Living-Living 相邻大幅降低（学到 "P(edge | Living, Living) ≈ 0"）
- Bedroom-Bathroom 相邻回升（学到 "P(edge | Bedroom, Bathroom) 高"）
- **不修复** Living hub 问题（attr head 还没结构信号）
- **不修复** Living 数量（attr head 不看图里其他已有 attrs）

### 2.2 Step C-2：Structure-Aware Attr Head（第二步）

**进一步改动**：让 attr head 看到节点的**结构上下文**（当前 degree、
邻居的 attr 分布）。

关键问题：在 AR 生成里，节点 k 的 attr **在它的边决定之前**采样，所以
attr head 看不到 "node k 的 degree"（还没有边）。

**两种解决方案**：

#### 方案 A：逆转 AR 顺序（先 edges 后 attr）

改成：先决定 node k 连到哪些已有节点，再预测 node k 的 attr。

```
step k:
  for j < k:
    sample edge (k, j) ← edge head 只看 h_k, h_j (no attr for k yet)
  compute neighbor_attr_distribution[k] from decided edges
  sample attr_k ← attr head sees h_k + neighbor_attr_dist
```

**优点**：attr head 能看到 "我连了 2 个 bedroom + 1 个 kitchen"，自然
会选 Living。

**缺点**：破坏了 C-1 的前提（edge head 要知道 attr_k）。两者冲突。

#### 方案 B：迭代式两步（C-1 + 轻量 attr 修正）

保持 C-1 的顺序（先 attr 后 edges），但增加 attr head 的结构输入：

```python
# attr head 输入：
# 1. node embedding h_k
# 2. 当前已决定节点的 attr 直方图（多少个 Living / Bedroom / ...）
# 3. Node k 的 ordering 位置

attr_head_input = torch.cat([
    h_k,
    current_graph_attr_histogram,  # (num_classes,) counts so far
    ordering_position_emb,
])
attr_logits = self.attr_head(attr_head_input)
```

**优点**：attr head 至少知道"图里已经有 2 个 Living 了，别再选 Living"，
直接压 Living count bloat。

**缺点**：不知道自己会连接到什么节点（信号较弱）。

#### 方案 C：不做 C-2，只做 C-1

赌 C-1 能把 Living-Living 问题打掉后，Living count 自然会降（因为 Living
连接少 → Living 作为中心出现的必要性降低）。实验驱动判断。

**建议：先做 C-1，看 attr-aware 指标是否显著改善。如果 Living count
仍然爆，再做 C-2（方案 B）**。

---

## 3. 实现步骤

### 3.1 Step C-1 具体任务清单

1. **配置**：
   ```yaml
   # config/gran_v2_rplan_planc.yaml (新)
   model:
     use_attr_conditioned_edge: true
     attr_embedding_dim: 32
     # 其他和 baseline 一样
   ```

2. **Model 改动**（`model/gran_v2.py`）：
   - `__init__`：新增 `nn.Embedding(num_attr_classes + 1, attr_emb_dim)`；
     edge head MLP 输入维度改为 `2 * (hidden + attr_emb_dim)`
   - `_inference`（training forward）：在 edge head 调用前，用
     ground-truth labels 查 attr_embedding，concat 进 edge head input
   - `_sampling`（inference）：保持"先 attr 后 edge"的顺序，edge 决策
     前从已采样的 attrs list 查 attr_embedding

3. **向后兼容**：
   - `use_attr_conditioned_edge=False` 时走旧路径
   - Load_state_dict 用 `strict=False`（已修）兼容旧 checkpoint

4. **测试**（`tests/test_attr_cond_edge.py`，新文件）：
   - `test_attr_embedding_registered_when_flag_on`
   - `test_edge_head_input_dim_adjusts`
   - `test_forward_pass_with_flag_on`
   - `test_forward_pass_with_flag_off_matches_baseline`
     （flag off 时和旧模型完全等价）
   - `test_sampling_uses_sampled_attrs_for_edge_head`
   - `test_attr_embedding_gradient_flows`（grad 能回传）

5. **训练 + 评估**：
   ```bash
   D:/Github/GSDiff/.venv/Scripts/python.exe run_exp.py \
       -c config/gran_v2_rplan_planc.yaml
   ```
   50 epoch，同 baseline 规模。

6. **判定指标**（重点看 attr-aware，不看 MMD）：
   - endpoint_attr_pair_kl（目标：< 0.6，好于 baseline）
   - `0-0` pair（Living-Living）：目标 < 5%
   - `1-2` pair（Bedroom-Bathroom）：目标 > 10%
   - living_count_kl（目标：显著下降）
   - 每图 1 Living 比例：目标 > 50%
   - MMD 不变坏即可（结构不重要）

### 3.2 Step C-2（如果 C-1 不够）

基于 C-1 结果判断：
- 如果 C-1 后 Living count 仍爆 → 做 C-2 方案 B（attr head 加结构上下文）
- 如果 C-1 后 Living deg 仍低 → 考虑 degree feature 重开（配合 ordering-id）
- 如果都 OK → 不需要 C-2

---

## 4. 风险和回退

### 4.1 风险

1. **训练不稳定**：attr embedding 随机初始化，早期 edge head 看到的
   attr 信号是噪声。**mitigation**：attr_embedding_dim 不要太大（32 足够）；
   `std=0.1` 初始化；前几个 epoch 考虑冻结 attr_embedding

2. **过拟合 attr-edge 关联**：数据集里 Bedroom-Bathroom 20% 没那么多，
   模型可能硬记。**mitigation**：dropout on attr_embedding output

3. **推理时采样 attr 错了 → 后续 edge 全错**：autoregressive 错误累积。
   **mitigation**：暂时不 mitigation，先看 baseline 有多严重

4. **checkpoint 不向后兼容**：加了新参数，旧模型 load 会缺 key。
   **mitigation**：`strict=False`（已修）

### 4.2 回退条件

如果 C-1 跑完**所有 attr-aware 指标都没改善**（误差内）：
- 说明 GRAN autoregressive 框架本身有问题，需要切 diffusion
- 不再在 GRAN 上堆 feature
- 直接启动 Phase 3 计划（见 `docs/plan-scale-to-healthcare.md`）

---

## 5. 与 paper 叙事的关系

Plan C 的贡献点（比之前的 Plan A 强很多）：

1. **架构级 contribution**：不是调 hyperparameter，是改 likelihood 分解。
   从 `P(X) · P(A|X)`（解耦）变成近似 `P(X, A)` 的联合建模。

2. **可泛化**：attr-conditioned edge head 对任何 labeled graph gen 都
   适用，不仅 floorplan。分子（碳原子 vs 氮原子的邻接偏好）、生物网络
   都可用。

3. **可量化验证**：attr-aware metric 给出清晰的前后对比（endpoint-pair
   KL、living count KL、per-class degree）。**这些 metric 本身也是
   贡献**——graph gen 领域缺这类 metric 的标准化。

4. **有 ablation**：
   - Full: GATv2 + ordering-id + attr-conditioned edge + class-weighted CE
   - w/o attr-cond edge：即 Plan A（已做）
   - w/o class-weighted CE：即 attr-cond edge only
   - w/o ordering-id：即 baseline
   - 单变量对比清晰

5. **对比实验**：
   - baseline（已有）
   - Plan A（已有）
   - Plan C-1（要做）
   - Plan C-2（可选）

这是一篇**applied-architecture paper**的素材：
- Title 方向："Structure-Semantics Coupling for Attributed Graph
  Generation: A Study on Floorplan Bubble Diagrams"
- Metric 贡献：endpoint-pair KL + living count KL + per-class degree
- 架构贡献：attr-conditioned edge head + analysis

---

## 6. 交接状态（2026-04-22）

### 已做

- ✅ Plan A 实现、训练、测试
- ✅ Plan A 结果：bedroom bloat 修但 Living bloat 出，attr-aware KL 更差
- ✅ attr-aware metric 代码接入 runner（`utils/attr_metrics.py`）
- ✅ Baseline + Plan A 重测，有完整数字（见 `experiments.md` Exp 5）
- ✅ 文档：experiments.md（含 attr-aware 对比）、plan-scale-to-healthcare.md
  （T1/T2/T5 重构）、research-landscape-and-roadmap.md（同）

### 未做（Plan C 下一步）

- [ ] Plan C 设计文档（**本文件**）
- [ ] Plan C 代码实现（model/gran_v2.py + 新 config + 测试）
- [ ] Plan C 训练 + 测试
- [ ] Plan C 结果写入 experiments.md Exp 6
- [ ] 根据 C-1 结果决定是否需要 C-2

### 建议开发顺序

1. 读 `model/gran_v2.py` 的 `_inference` 和 `_sampling`，理解 edge head
   输入的具体 tensor shape 和生成顺序
2. 写 `tests/test_attr_cond_edge.py` 先（TDD）
3. 实现 attr_embedding + edge head 输入改动
4. 跑单元测试
5. 写新 config `config/gran_v2_rplan_planc.yaml`
6. 训练（50 epoch）
7. 测试 + 看 attr-aware 指标
8. 对比 baseline / Plan A，写 Exp 6
