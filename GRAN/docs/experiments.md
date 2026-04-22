# 实验记录与模型性能对比

> 追踪 GRAN v2 在 RPLAN bubble diagram 任务上的各次实验。每个实验记录
> 完整的 config、训练曲线、MMD 分数、定性观察和结论。
>
> 新实验按时间顺序往下加；每条都完整 self-contained（config + 数据 +
> 结果 + 观察）。

---

## TL;DR 实验对比

| Run | Config | 数据量 | Epoch | best val_total | TEST/degree MMD | TEST/clustering MMD | TEST/spectral MMD | Living 在中心 | 结论 |
|-----|--------|-------|-------|---------------|-----------------|---------------------|-------------------|--------------|-----|
| ⚠️ Baseline (`24504`) | gran_v2_rplan.yaml | 20k | 300 (stop @ 30) | 0.5452 | 0.0036 | 0.0164 | 0.0079 | ❌ 只 22% 图正常 | **结构 MMD 漂亮但语义也坏**（attr-aware 指标揭示）|
| ❌ Struct (`50908`) | gran_v2_rplan_struct.yaml | 20k | 100 | 0.9631 | 0.2757 | 0.8982 | 0.2103 | ❌ 完全没学到 | 多 ordering 未做 id 条件 → 分布融合崩 |
| ❌ Degonly (`53732`) | gran_v2_rplan_degonly.yaml | 20k | 50 | **0.4118** | 0.0481 | 0.9645 | 0.1657 | ❌ 没学到 | val 低但 MMD 差——Teacher Forcing 陷阱 |
| ⚠️ Path 1 (`41540`) | gran_v2_rplan_p1.yaml | 20k | 50 | 1.1425 | 0.0357 | 0.1382 | 0.0388 | ❌ 没学到 | ordering-id 假设被证实（比 Struct 好 5-8×），但仍输 Baseline |
| ⚠️ Plan A (`28764`) | gran_v2_rplan_attrbalance.yaml | 20k | 50 | 0.5851 | **0.0021** | **0.0136** | **0.0070** | ❌ 更差 | Bedroom bloat 修好但 Living 爆炸；结构 MMD 略好但**语义指标更差** |

### Attr-aware 指标对比（Exp 5 后新增，比 MMD 有区分度）

| | Ref | Baseline | Plan A |
|---|-----|----------|--------|
| **每图 1 个 Living 的比例** | **97.7%** | 22% | **16%** |
| **endpoint-pair KL** (边语义) | 0 | 0.616 | **0.814** 🔴 |
| **living count KL** (每图客厅数) | 0 | 8.61 | **9.88** 🔴 |
| **Living 平均度数** | **5.20** (hub) | 2.46 | 2.81 |
| **Balcony 平均度数** | **1.63** (leaf) | 4.94 🔴 | 4.79 🔴 |
| **Bedroom-Bathroom 边** | 14.3% | 2.6% | 3.5% |
| **Living-Living 边**（错误） | ~0% | 9.6% | **17.3%** 🔴 |

**结论**：
- Baseline 和 Plan A 都**没学到"Living 是 hub / Balcony 是 leaf"**
- 两个模型都**把 Balcony 当 hub**（和真实相反）
- Plan A 让 Living-Living 相邻问题**变严重**（+17.3% vs baseline +9.6%）
- 都丢失"主卧-卫生间"这种经典 pattern（-10% 以上）

**关键发现 1：val_total 低 ≠ 生成质量好**。Degonly 的 val 比 Baseline 低
25%，但 MMD 差 50x。说明 autoregressive 模型在训练和推理之间有 distribution
mismatch，degree feature 放大了这个 gap。

**关键发现 2：ordering-id conditioning 修复了 Struct 的结构崩坏，但没带来增益**。
Path 1 相对 Struct：clustering 0.8982 → 0.1382（好 6.5×），spectral 0.2103 →
0.0388（好 5.4×），degree 0.2757 → 0.0357（好 7.7×）。但相对 Baseline 还是
4-26× 更差。说明多 ordering 在 RPLAN 这种小图上**没信号可学**——GRAN 原论文
的多 ordering 增益主要在大/稀疏图上。

**最终结论（诚实版，含 Plan A + attr-aware 指标）：五种 config 全部不可用。**

Attr-aware 指标（endpoint_attr_pair_kl, living_count_kl, per-class degree）
揭示：

- **Baseline 的结构 MMD 漂亮是数字错觉**：只 22% 图符合"一个客厅"，
  Living degree 2.46（应是 5.20 hub），Balcony degree 4.94（应是 1.63
  leaf）——**完全反了**
- **Plan A 让语义更差**：endpoint-pair KL 0.616 → 0.814，Living-Living
  相邻 +9.6% → **+17.3%**。loss 权重只移动 bug 位置
- **Struct / Degonly** 既丢 MMD 又没修语义
- **Path 1** 证实 ordering-id 假设但对小图无增益

**根因**：GRAN v2 的 attr head 和 edge head **完全解耦**——attr head 看
node embedding 预测类别，edge head 看 node embedding pair 预测连接，
**都不知道对方的决定**。所以模型永远学不到"Living 是 hub / Bedroom
常邻 Bathroom / 一图只能有一个 Living"这种耦合先验。

**真正的下一步**：
1. ✅ **Plan A 已证伪**：单纯调 loss 权重只是移动 bug
2. 🎯 **Plan C（架构级耦合）是唯一出路**：详细设计见
   `docs/plan-c-design.md`
3. 如果 Plan C 也不够：切 diffusion（Phase 3），见
   `plan-scale-to-healthcare.md`

**注意**：两个 run 的 val_total 数值**不可直接比较**。
- Baseline 用 `num_canonical_order=1` + 原默认 loss（logsumexp over 1 = 直接值）
- Struct 用 `num_canonical_order=3` + `sum_order_log_prob=True`（3 个 ordering 求和，所以数值 ~3 倍）

**唯一可直接比较的是 MMD 指标**（数据分布距离，跟 loss 公式无关）。

---

## 🧪 Experiment 1 — Baseline (2026-04-20)

### 目的

建立 GRANv2 在 RPLAN 上的基准表现，作为后续结构改进的对照组。

### 配置

```yaml
# config/gran_v2_rplan.yaml (at commit 98a4ed6)
dataset:
  name: RPLAN
  total_graphs: 20000
  train_ratio: 0.9     # -> train 18000
  dev_ratio: 0.05      # -> dev 1000
                       # -> test 1000
  node_order: DFS
model:
  name: GRANv2
  use_gatv2: true
  gatv2_num_heads: 8
  num_attr_classes: 7
  hidden_dim: 256
  embedding_dim: 256
  num_GNN_layers: 6
  num_mix_component: 30
  num_canonical_order: 1        # <<< 单 ordering
  # use_degree_feature: false (默认)
train:
  batch_size: 64
  lr: 2.0e-4
  lr_decay_epoch: [5, 15]
  lambda_attr: 0.5
  max_epoch: 300
```

### 训练曲线

| Epoch | train total | val total |
|-------|------------|-----------|
| 1 | 0.80 | - |
| 5 | 0.56 | 0.5557 |
| 10 | 0.55 | 0.5489 |
| 15 | 0.54 | 0.5459 |
| 25 | 0.54 | 0.5459 |
| **30 (最终保存)** | **0.54** | **0.5452** |

30 epoch 已收敛（平台），后续 ep 30-300 无实质提升，实际有效训练到
ep 30。

### MMD 结果（test 阶段）

在生成 1000 图 vs 1000 真实 test 图：

| 指标 | DEV | TEST |
|------|-----|------|
| #nodes | 0.0050 | 0.0171 |
| degree | 0.0029 | **0.0036** |
| clustering | 0.0166 | **0.0164** |
| 4-orbits | 0.0 (orca 未编译) | 0.0 |
| spectral | 0.0073 | **0.0079** |

**数字评级**：
- degree / spectral MMD < 0.01 → **SOTA 水平**
- clustering / #nodes MMD < 0.05 → **良好**

### 定性观察

查看 `gen_grid.png`：
- ✅ 节点数分布正确（4-8 节点）
- ✅ 邻接 pattern 基本合理
- ❌ **Living Room 经常不出现**（argmax 初期 collapse，后改 multinomial 也没根本解决）
- ❌ Storage 过度采样（训练集 3%，生成时 >20%）
- ❌ Living 不一定在高度数节点

### 结论

MMD 数字够好（几乎 SOTA），但视觉上**结构不合理**。问题出在：
- 模型没学到"高度数节点 = Living"
- 只依赖 DFS 顺序，"第 0 个节点"不总是 Living
- attr 分布偏移

**直接触发 Experiment 2**：多 ordering + degree rank 结构特征。

---

## 🧪 Experiment 2 — Structural Fixes (2026-04-21)

### 目的

修复 Exp1 发现的结构不合理问题。加入：
1. **多 canonical ordering**（DFS + BFS + k-core），强制模型学结构而非位置
2. **Degree-rank embedding**（给节点当前度数排名作为结构特征）

### 配置差异（vs Exp1）

```yaml
# config/gran_v2_rplan_struct.yaml (at commit bc261e8)
dataset:
  node_order: 'DFS+BFS+k_core'   # <<< 多 ordering
model:
  num_canonical_order: 3          # <<< 多 ordering
  use_degree_feature: true        # <<< 结构特征
train:
  max_epoch: 100                  # <<< 从 300 减到 100
  lr_decay_epoch: [30, 60]        # <<< 新 schedule
  snapshot_epoch: 10
  # loss 公式: sum_order_log_prob=True (commit cc2f78d)
```

### 训练曲线

| Epoch | train total | val total |
|-------|------------|-----------|
| 1 | 1.73 | - |
| 5 | 1.19 | 1.1916 |
| 10 | 1.16 | 1.1648 |
| 20 | 1.15 | 1.0968 |
| 30 | - | **0.9748** (lr decay 1) |
| 40 | - | 0.9680 |
| 50 | - | 0.9677 |
| 60 | - | **0.9640** (lr decay 2) |
| 80 | - | 0.9636 |
| 95 | - | **0.9631** ← best |
| 100 | 0.9062 | - |

收敛非常健康：
- ep 1-30 大幅下降
- ep 30/60 两次 lr_decay 各挤出一波提升
- ep 60+ 进入平台期，微调

**注意数值 scale**：Struct 的 val_total 约为 Baseline 的 2x。这是因为
`num_canonical_order=3` + `sum_order_log_prob=True` 的 loss 是**三个
ordering 的 log prob 之和**（数量级变大），与模型质量无关。

### MMD 结果

*待测（运行 `run_exp.py -t` 获取）*。

```bash
D:/Github/GSDiff/.venv/Scripts/python.exe run_exp.py \
    -c exp/GRANv2_rplan_struct/GRANv2_RPLAN_2026-Apr-21-14-24-30_50908/config.yaml \
    -t
```

### 定性观察

*待测后填写。关注点*：
- `gen_grid.png` 里 Living 是否出现在高度数节点
- 房间类型分布是否更接近训练集
- 是否还有 Storage 过度采样问题

### 假设

根据多 ordering + degree 的理论分析：
- ✅ MMD（degree / clustering / spectral）**应该持平或略好**
- ✅ 视觉合理性（Living 在中心、类别分布）**应该明显改善**
- ⚠️ val_total 数字大是 loss 公式差异，不是模型退化

### 最终结果（2026-04-21）

**MMD（TEST）严重退化**：
- #nodes: **0.9557**（baseline 0.0171，**56x 更差**）
- degree: **0.2757**（baseline 0.0036，**77x 更差**）
- clustering: **0.8982**（baseline 0.0164，**55x 更差**）
- spectral: **0.2103**（baseline 0.0079，**27x 更差**）

**视觉（gen_grid.png）**：
- 节点数偏小（大量 2-3 节点图）
- **几乎全是 Bedroom + Kitchen，Living 完全缺失**
- 简单链式结构退化

### 失败根因分析

**多 ordering 训练 + 单 ordering 推理的分布偏移。**

训练时模型看同一张图的 3 种 ordering（DFS/BFS/k-core）→ 学到的是
"融合分布"（P_avg = (P_DFS + P_BFS + P_kcore) / 3）。推理时只按一种
生成 → 采样到的是"平均 ordering 下的融合分布"，不匹配任何一个单独的
真实分布。

原版 GRAN 论文用 `sum_order_log_prob=True` 是正确做法（相当于 3 个
独立子模型），但需要**ordering-id 条件**才能真正 work，我们的实现
没有加这个 condition embedding。

### 结论

失败实验，**不 ship**。触发 Exp 3（degree-only 隔离变量）。

---

## 🧪 Experiment 3 — Degree-Only Ablation (2026-04-21)

### 目的

隔离变量，定位 Exp2 失败是**多 ordering** 还是 **degree embedding** 的锅。
保留 degree-rank feature，关闭多 ordering。

### 配置差异（vs Exp2 Struct）

```yaml
# config/gran_v2_rplan_degonly.yaml
model:
  num_canonical_order: 1          # <<< 改回 1（struct 是 3）
  use_degree_feature: true        # 保留
dataset:
  node_order: DFS                 # <<< 改回单 ordering
train:
  max_epoch: 50                   # 快速实验
  lr_decay_epoch: [15, 35]
```

### 训练曲线

| Epoch | train total | val total |
|-------|------------|-----------|
| 5 | - | 0.58 |
| 15 (lr decay 1) | 0.42 | 0.45 |
| 30 | 0.40 | **0.4159** |
| 35 (lr decay 2) | 0.39 | 0.4120 |
| **40 (best)** | - | **0.4118** |
| 50 (final) | 0.3872 | - |

Val 降到 **0.4118，比 baseline (0.5452) 低 25%**。看起来很赞。

### MMD（TEST）— 但

| 指标 | Baseline | **Degonly** | 对比 |
|------|----------|------------|------|
| #nodes | **0.0171** | 0.9557 | **56x 更差** |
| degree | **0.0036** | 0.0481 | **13x 更差** |
| clustering | **0.0164** | 0.9645 | **59x 更差** |
| spectral | **0.0079** | 0.1657 | **21x 更差** |

**val loss 降了但 MMD 大幅变差。**

### 类别分布（gen vs ref）

| 类别 | Ref | Gen | 偏差 |
|------|-----|-----|------|
| 0 Living | 15.1% | 18.5% | ✅ 合理 |
| 1 Bedroom | **36.8%** | **56.3%** | 过多 |
| 2 Bathroom | **18.0%** | **0.5%** | **几乎消失** |
| 3 Kitchen | 14.5% | 5.2% | 不足 |
| 4 Balcony | 14.7% | 19.2% | 略多 |
| 5 Storage | 1.0% | 0.2% | 略少 |

模型"偷懒"：生成最常见类（Bedroom），跳过难预测的（Bathroom, Kitchen）。

### 失败根因：Teacher Forcing 陷阱

自回归经典问题：
```
训练时：真实 prefix → 预测下一步       → loss 小
推理时：模型自己生成的 prefix → 预测    → 错误累积
```

Degree feature 放大了这个 gap：
- 训练：节点度数 = 真实图的度数
- 推理：节点度数 = 生成的（早期步度数全 0，分布偏移）

模型 overfit 到训练时看到的度数模式，推理时失真。

### 结论

**两个结构化改动都不 work**：
- 多 ordering 需要 ordering-id 条件（没实现）
- Degree feature 受 teacher forcing 拉扯（自回归固有问题）

**下一步**：放弃结构化改动，回 baseline。要根本解决"Living 结构一致性"
问题，只能切 diffusion（见 `plan-scale-to-healthcare.md` Phase 3）。

### 决定

- ❌ 不 ship 这个 config
- ✅ 保留 baseline 接入 app
- 📅 diffusion 作为长期 roadmap

---

## 🧪 Experiment 4 — Path 1: Ordering-ID Conditioning (2026-04-21)

### 目的

诊断 Exp2 (Struct) 的失败根因：多 ordering 训练 + 单 ordering 推理的分布
融合问题。假设是"模型看 3 种 ordering 但不知道当前是哪种，生成时就按
一个'平均 ordering'采样，不匹配任何单独真实分布"。

**修复**：在 GNN 输入里给每个节点加一个 `ordering_id` embedding，训练时
按当前 ordering 的 id 注入，推理时固定为 `ordering_id=0`。这样 multi-
ordering 训练相当于 3 个 id-conditioned 子模型，推理时退化成第 0 个子
模型（无融合）。

**核心差异（vs Exp2 Struct）**：隔离变量——只开 Path 1（ordering-id），
**不开** degree feature，排除 teacher-forcing 干扰。

### 配置差异（vs Exp2 Struct）

```yaml
# config/gran_v2_rplan_p1.yaml (at commit a277aab)
dataset:
  node_order: 'DFS+BFS+k_core'     # 和 Struct 一样
model:
  num_canonical_order: 3            # 和 Struct 一样
  use_ordering_id: true             # <<< NEW：Path 1 开关
  use_degree_feature: false         # <<< 和 Struct 不同，隔离变量
train:
  max_epoch: 50                     # 从 100 缩到 50
  lr_decay_epoch: [15, 35]
```

### 实现细节

- `model/gran_v2.py:__init__`: 新增 `nn.Embedding(num_canonical_order, embedding_dim)`，
  `normal_(std=0.1)` 初始化
- `_inference`: 从现有 `node_idx_feat = batch*C*N + order*N + pos + 1` 包装
  中解出 ordering id（`(flat // N) % C`），不改 dataset
- `_sampling`: 固定 `ordering_id=0`（推理时取第一个子模型）

28 个测试通过（25 旧 + 3 新: `tests/test_ordering_id.py`）。

### 训练曲线

| Epoch | val edge | val attr | val total |
|-------|----------|----------|-----------|
| 5 | 0.8405 | 0.7387 | 1.2098 |
| 15 (lr decay 1) | 0.8234 | 0.7107 | 1.1787 |
| 25 | 0.8209 | 0.7066 | 1.1742 |
| 35 (lr decay 2) | 0.7969 | 0.7043 | 1.1490 |
| 45 | 0.7915 | 0.7030 | 1.1430 |
| **50 (best)** | **0.7907** | **0.7035** | **1.1425** |

**注意**：val_total 数值不能和 Baseline (0.5452) 直接比——Path 1 用
`num_canonical_order=3` + `sum_order_log_prob=True`，loss 是 3 个 ordering
的 log prob 之和（数量级约为 Baseline 的 2x）。但可以和 Struct (0.9631)
比——同 loss 公式下 Path 1 更差，因为 Struct 额外用了 degree feature 压 loss。
**MMD 才是可比的指标。**

### MMD 结果（test 阶段）

| 指标 | Baseline | Struct ❌ | Degonly ❌ | **Path 1** | 相对 Baseline |
|------|----------|----------|-----------|------------|--------------|
| #nodes | **0.0171** | 0.9557 | 0.9557 | 0.4478 | 26× 更差 |
| degree | **0.0036** | 0.2757 | 0.0481 | 0.0357 | 10× 更差 |
| clustering | **0.0164** | 0.8982 | 0.9645 | 0.1382 | 8.4× 更差 |
| spectral | **0.0079** | 0.2103 | 0.1657 | 0.0388 | 4.9× 更差 |
| 4-orbits | 0 | 0 | 0 | 0 | 持平（orca 未编译） |

**Path 1 vs Struct**（同 config，只是多了 ordering-id）：

| 指标 | Struct | Path 1 | Path 1 好多少 |
|------|--------|--------|--------------|
| degree | 0.2757 | 0.0357 | **7.7×** |
| clustering | 0.8982 | 0.1382 | **6.5×** |
| spectral | 0.2103 | 0.0388 | **5.4×** |

### 类别分布（gen vs ref，TEST set）

| 类别 | Ref | Path 1 Gen | 偏差 |
|------|-----|-----------|------|
| 0 Living | 15.1% | 12.7% | 稍少（可接受）|
| 1 Bedroom | 36.8% | **63.0%** | **严重多** |
| 2 Bathroom | 18.0% | **2.0%** | **严重塌缩** |
| 3 Kitchen | 14.5% | 10.9% | 接近 |
| 4 Balcony | 14.7% | 9.6% | 接近 |
| 5 Storage | 1.0% | 1.7% | 接近 |
| 6 External | 0% | 0.02% | 接近 |

### Gen 图尺寸分布

| #nodes | Path 1 Gen | Ref |
|--------|-----------|-----|
| 4 | 7.5% | 0.3% |
| 5 | 15.4% | 7.9% |
| 6 | **29.5%** | 29.3% |
| 7 | 29.1% | **37.0%** |
| 8 | 13.5% | 25.5% |

生成的图**偏小**（6 节点占比最高），真实图**偏大**（7 节点主导）。这是
`#nodes MMD = 0.4478` 的来源。

### 定性观察（`vis/gen_grid.png`）

- ✅ 结构合理（无碎链、多数连通、度数分布接近真实）
- ✅ Struct 的塌缩问题消失（不再有大量 2-3 节点图）
- ❌ **一片紫色 bedroom**，Living 偶尔出现但不一定在中心
- ❌ Bathroom 几乎看不到

### 失败根因

**假设部分正确**：ordering-id conditioning 把 Struct 的结构崩坏从根本
上修好了（MMD 4-8× 回升）。但**多 ordering 本身对 RPLAN 小图没增益**。

原因（推测）：
1. RPLAN 的 bubble diagram 只有 4-8 个节点，三种 ordering 看到的图
   结构信息重叠度高——没有足够的"结构多样性"需要消化
2. 单 ordering 的 DFS 已经覆盖了大部分可学 pattern
3. 多 ordering 本质是**数据增强**。在小图上数据增强反而增加 loss
   方差，拖慢收敛

**Bedroom 过多 / Bathroom 塌缩**不是结构问题，是**属性 head 的类不平衡
先验**——三个 config 都没解决，需要单独动 attr loss（class-weighted CE
或 focal），不是 ordering/degree 能救的。

### 结论

- ✅ **ordering-id 假设被证实有效**：修复了 Struct 的实现 bug（`sum_order_log_prob=True`
  本就需要配合 id conditioning）
- ❌ **但对 app 没用**：多 ordering 对 RPLAN 小图无增益，且 bedroom bloat /
  bathroom 塌缩问题完全没动——这些是 attr head 的问题，不是结构问题
- 📝 **记入负面实验库**：未来如果切大图数据（healthcare 30+ 节点），
  Path 1 是必需的（否则 Struct 重演），可以直接复用
- 🔬 **Path 2（scheduled sampling for degree feature）优先级调低**：从
  Path 1 结果看结构改动救不了 bedroom bloat

### 决定

- ❌ 不 ship 这个 config
- ❌ Baseline 也不能直接 ship（视觉上对 app 不够）
- ✅ **下一步 pivot 到 attr loss 调整**（类平衡 CE / focal / attr-conditioned
  edge head）而不是再堆结构特征
- 📅 长期：diffusion（Phase 3）

---

## 🧪 Experiment 5 — Plan A: Class-Weighted Attr CE (2026-04-22)

### 目的

诊断是否能用 **loss 层的类平衡** 修 baseline 的 bedroom bloat / bathroom
塌缩。假设：Bedroom 63% vs ref 37% 是 CE loss 直接复制训练集频率先验的
结果，加 inverse-frequency weighting 理论上能把常见类压下去、罕见类
抬起来。

**这是隔离变量**：结构部分完全和 baseline 一样（单 ordering、无 degree
feature、无 ordering-id），只改 attr CE 的权重。

### 配置差异（vs Baseline）

```yaml
# config/gran_v2_rplan_attrbalance.yaml
model:
  num_canonical_order: 1            # 和 baseline 一样
  use_degree_feature: false         # 和 baseline 一样
  use_ordering_id: false            # 和 baseline 一样
  attr_class_weight: auto           # <<< 唯一变量
train:
  max_epoch: 50
  lr_decay_epoch: [15, 35]
```

### 实现细节

- `model/gran_v2.py:__init__`: 注册 `attr_class_weights` buffer，`auto` 模式
  初始化为 ones；新增 `set_attr_class_weights(w)` in-place setter
- `model/gran_v2.py:forward`: `F.cross_entropy(attr_logits, labels,
  weight=self.attr_class_weights)` 替换原 `self.attr_loss_func`
- `runner/gran_runner_v2.py:_compute_auto_attr_class_weights`: sklearn
  balanced 公式 `w[c] = total / (A * max(count[c], 1))`，clamp 到
  `[0.1, 10.0]`；DataParallel wrap 前注入

8 个新测试通过（`tests/test_attr_class_weight.py`），总 36 个测试绿。

### 自动计算的 weights（训练时注入）

runner 扫 `graphs_train` 算出：

| 类别 | Count | Weight | 备注 |
|------|-------|--------|------|
| 0 Living | ~1023 | **0.945** | 几乎不变 |
| 1 Bedroom | ~2500 | **0.398** | 压最狠（ref 最多）|
| 2 Bathroom | ~1221 | 0.807 | 轻度抬 |
| 3 Kitchen | ~982 | 1.018 | 几乎不变 |
| 4 Balcony | ~999 | 0.871 | 轻度压 |
| 5 Storage | ~70 | **10.0** | **顶 clamp 上限** |
| 6 External | ~1 | **10.0** | **顶 clamp 上限** |

### 训练曲线

| Epoch | val_total |
|-------|-----------|
| 5 | 0.6091 |
| 10 | 0.5976 |
| 15 (lr decay 1) | 0.5856 |
| **40 (best)** | **0.5851** |
| 50 | ~0.5851 |

Val 从 0.61 → 0.585 平稳下降。比 baseline (0.5452) 略高，合理（weighted
CE 的 loss 期望值本就比 uniform CE 高）。

### MMD 结果（test）

| 指标 | Baseline | **Plan A** | 变化 |
|------|---------|-----------|------|
| #nodes | 0.0171 | 0.0377 | 2.2× 略差 |
| degree | 0.0036 | **0.0021** | **1.7× 更好** ✨ |
| clustering | 0.0164 | **0.0136** | **1.2× 更好** ✨ |
| spectral | 0.0079 | **0.0070** | **1.1× 更好** ✨ |
| 4-orbits | 0 | 0 | 持平（orca 未编译） |

**关键发现：结构 MMD 不仅没坏，三项略好**。说明 weighted CE 对结构
学习没有破坏。

### 类别分布 — 核心发现

| 类别 | Ref | Baseline Gen | **Plan A Gen** | 诊断 |
|------|-----|-------------|---------------|------|
| 0 Living | 15.1% | ~15% | **39.3%** | 🔴 **爆炸 2.6×** |
| 1 Bedroom | 36.8% | 63.0% | **31.2%** | ✅ **修好了**（接近 ref）|
| 2 Bathroom | 18.0% | 2.0% | 3.6% | ⚠️ 略好但仍塌缩 |
| 3 Kitchen | 14.5% | 10.9% | 19.5% | ⚠️ 略高 |
| 4 Balcony | 14.7% | 9.6% | 5.1% | ⚠️ 被挤出 |
| 5 Storage | 1.0% | 1.7% | 1.2% | ✅ 持平 |
| 6 External | 0% | 0.02% | 0.02% | ✅ 持平 |

### 定性观察（gen_graphs.json 抽样）

6-7 节点图里出现大量 **2-3 个 Living** 的样本：
- `idx 1: [0,3,3,3,0,0,0,0]` → 4 个 Living
- `idx 11: [5,3,1,1,0,0,0]` → 3 个 Living
- `idx 12: [0,3,1,1,0,0,0]` → 3 个 Living

**每张 ~7 节点图 × 39% Living = 平均每图 2-3 个 Living**。
违反现实户型"一个客厅"的常识。

### 失败根因：weighted CE 的生态位转移

训练时：
- Bedroom gradient × 0.4 → 模型对 Bedroom 预测 confidence 降低
- Living weight 几乎不变 (0.95) → 仍是"强类"
- 模型在"该选常见类"时，从 Bedroom 转向 Living（Living 结构 feature 训练充分）
- Bathroom / Balcony 虽然 ref 常见但结构 feature 弱（多是 degree=1 叶
  节点，信号不稳），权重提升不足以让模型敢选

**Plan A 没解决 structure-semantics 耦合**，只是把问题从"Bedroom 吞 Living"
换成"Living 吞 Bedroom"。

### 结论

- ✅ **Bedroom bloat 修好了**（63% → 31%，接近 ref 37%）
- ✅ **结构 MMD 略好于 baseline**（degree / clustering / spectral 都小幅下降）
- ❌ **Living bloat 出现**（15% → 39%，新 bug）
- ❌ **Bathroom 仍塌缩**（2% → 3.6%，微改善不够）
- ❌ **app 可用性没改善**：从"到处 bedroom"变成"到处 living"

### 决定

- ❌ 不 ship
- 🎯 **触发 Plan C**：attr-conditioned edge head + 结构-属性联合建模
  - 让 edge head 拿到邻居 attr embedding → 学"Living 不邻接 Living"
  - 让 attr head 拿到结构 feature（degree、local cluster）→ 学
    "叶节点常是 Bathroom / Balcony，hub 是 Living"
- ⚠️ 保留 Plan A 代码路径作为对照组 / 未来 fine-tune 的起点

### Attr-aware 指标（Exp 5 后加，重测 baseline + plan A）

**每图 Living 数量分布**（真实 97.7% 为 1 个）

| 每图 Living 数 | Ref | Baseline | Plan A |
|-------------|-----|----------|--------|
| 0 | 0.0% | 10.6% | 5.1% |
| **1（正常）** | **97.7%** | 21.9% | 15.7% |
| 2 | 2.3% | 28.1% | 27.1% |
| 3 | 0.0% | 23.8% | 27.3% |
| 4+ | 0.0% | 15.6% | 24.8% |

**Per-class degree mean**（Ref 里 Living=hub, Balcony=leaf）

| 类别 | Ref | Baseline | Plan A |
|------|-----|----------|--------|
| Living | **5.20** (hub) | 2.46 | 2.81 |
| Bedroom | 2.66 | 2.51 | 2.53 |
| Bathroom | 2.55 | 4.09 | 3.82 |
| Kitchen | 2.14 | 2.71 | 2.37 |
| Balcony | **1.63** (leaf) | **4.94** 🔴 | 4.79 🔴 |
| Storage | 2.23 | 4.10 | 3.72 |

**Endpoint-pair KL + Living count KL**

| 指标 | Baseline | Plan A |
|------|----------|--------|
| endpoint_attr_pair_kl | 0.616 | **0.814** 🔴（更差） |
| living_count_kl | 8.61 | **9.88** 🔴（更差） |

**Top 5 过度使用的边 pair（gen - ref）**

| Pair | Baseline | Plan A |
|------|----------|--------|
| 0-0 (Living-Living) | +9.6% | **+17.3%** |
| 1-4 (Bedroom-Balcony) | +10.4% | - |
| 1-1 (Bedroom-Bedroom) | +7.7% | - |
| 1-3 (Bedroom-Kitchen) | - | +4.2% |
| 3-3 (Kitchen-Kitchen) | - | +2.3% |

**Top 5 缺失的边 pair（gen - ref）** — 两模型一致

| Pair | Baseline | Plan A |
|------|----------|--------|
| 1-2 (Bedroom-Bathroom) | -11.7% | -10.8% |
| 0-2 (Living-Bathroom) | -10.5% | -8.9% |
| 0-1 (Living-Bedroom) | -6.5% | -4.5% |
| 0-3 (Living-Kitchen) | -5.3% | - |
| 2-3 (Bathroom-Kitchen) | -3.2% | -1.5% |

### 重大重新判定（Exp 5 + attr-aware 后）

**之前的判断错了**：attr-aware 指标揭示 baseline 也没学到结构-语义耦合：

1. **Living 不是 hub**：两个模型都把 Living 的 degree 学到 ~2.5（ref 是 5.2）
2. **Balcony 反而是 hub**：两个模型 Balcony degree ~4.9（ref 是 1.6）—— **完全反了**
3. **Living-Living 相邻**：两个模型都大量生成（ref 几乎为 0），Plan A 更严重
4. **Bedroom-Bathroom**（主卧套间）：两个模型都几乎不生成（-11% 左右）

**Baseline 不是"当前最佳"**，只是结构 MMD 漂亮的幻觉。真实情况是**所有 5 种 config 都不可用**。

**只有 Plan C（结构-属性架构耦合）能救**。见 `docs/plan-c-design.md`。

---

## 📊 如何新增实验

下次跑新 config 时，按这个模板往下加：

```markdown
## 🧪 Experiment N — <name> (<date>)

### 目的
<1-2 句：这次实验想验证什么>

### 配置差异（vs 上一个 baseline）
<config YAML diff 片段>

### 训练曲线
<每 5-10 epoch 的 val_total 表>

### MMD 结果
| 指标 | DEV | TEST |
| ... |

### 定性观察
<gen_grid.png 看到的现象>

### 结论
<这次学到什么？是否继续这个方向？>
```

---

## 📈 评估协议

### 怎么算 MMD（可重现）

1. 训练完后 `run_exp.py -t` 自动做
2. 生成 `test.num_test_gen` 张图（默认 1000）
3. 对 dev set 和 test set 分别算 MMD
4. 5 个指标：`#nodes`, `degree`, `clustering`, `4-orbits`, `spectral`
5. **4-orbits 当前恒为 0**（orca C++ 扩展未编译，不是真实值）

### 可信度阈值（经验）

| MMD | 评级 |
|-----|------|
| < 0.01 | SOTA / 接近数据分布极限 |
| 0.01-0.05 | 良好 |
| 0.05-0.1 | 可用 |
| > 0.1 | 有明显问题 |

### 但 MMD 不是万能

**Exp1 展示了：MMD 好 ≠ 生成合理**。原因：
- MMD 测分布级别匹配
- 对**结构一致性**（Living 在 hub）不敏感
- 所以要**结合视觉评估**（`gen_grid.png`）

### 推荐的完整评估

1. **MMD 数字**（客观，可重现）
2. **视觉对比**（`vis/gen_grid.png` vs `vis/ref_grid.png`）
3. **attr class 分布**（统计 gen 中各类占比 vs 训练集 vs test）
4. **结构一致性检查**（hub 是不是 Living？是不是连通？）

---

## 📁 Run 目录速查

```
exp/GRANv2_rplan/
├── GRANv2_RPLAN_2026-Apr-20-15-43-48_24504/   ← Exp1 (baseline) 训练目录
├── GRANv2_RPLAN_2026-Apr-20-17-03-45_45408/   ← Exp1 的 test 结果
└── ...                                         ← 其他测试 run（可忽略）

exp/GRANv2_rplan_struct/
├── GRANv2_RPLAN_2026-Apr-21-13-46-10_38932/   ← Exp2 早期（loss 公式 bug，已废）
└── GRANv2_RPLAN_2026-Apr-21-14-24-30_50908/   ← Exp2 正式 run (CURRENT)
    ├── model_best.pth                          ← ★ 用这个做 test
    ├── model_snapshot_0000010~0000100.pth     ← 中间 checkpoint
    ├── config.yaml                             ← test 时自动指向 best
    ├── log_exp_*.txt                           ← 完整训练 log
    ├── train/, val/                            ← TensorBoard 分开的 writers
    └── vis/                                    ← 生成可视化（test 后才有）
```

---

## 🎯 下一步实验建议

按优先级：

### Exp 3 候选：Struct 模型测试评估

**最优先**：只需要跑一次 `-t` 就能知道 Exp2 是否成功。

### Exp 4 候选：调 lambda_attr

如果 Exp2 的 attr 生成仍然偏差，试：
```yaml
train:
  lambda_attr: 1.0    # 从 0.5 提到 1.0，让 attr 更受重视
```

### Exp 5 候选：扩数据量到 40k

```yaml
dataset:
  total_graphs: 40000
```

看数据量翻倍是否带来 MMD 或视觉提升。

### Exp 6 候选：纯 GRU（不用 GATv2）对比

验证 GATv2 是否真的比原版 GRU 好：
```yaml
model:
  use_gatv2: false
```

### Exp 7（长期）：换 Graph Transformer backbone

参考 `docs/plan-scale-to-healthcare.md` Phase 2。适合目标是 20-30 节点的
中等规模数据时启动。
