# 实验记录与模型性能对比

> 追踪 GRAN v2 在 RPLAN bubble diagram 任务上的各次实验。每个实验记录
> 完整的 config、训练曲线、MMD 分数、定性观察和结论。
>
> 新实验按时间顺序往下加；每条都完整 self-contained（config + 数据 +
> 结果 + 观察）。

---

## TL;DR 实验对比

| Run | Config | 数据量 | Epoch | best val_total | TEST/degree MMD | TEST/spectral MMD | Living 在中心 | 备注 |
|-----|--------|-------|-------|---------------|-----------------|-------------------|--------------|------|
| **Baseline** (`24504`) | gran_v2_rplan.yaml | 20k | 300 (early stop @ 30) | **0.5452** | **0.0036** | **0.0079** | ❌ 经常缺失 | 单 ordering + 无 degree feature |
| **Struct** (`50908`) | gran_v2_rplan_struct.yaml | 20k | 100 | 0.9631 | *待测* | *待测* | *待测* | 3 ordering + degree rank embedding |

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

### 评估完后待补充

等跑完 test，补：
- TEST MMD（5 个指标）
- gen_grid.png 对比截图或描述
- 与 Exp1 的视觉差异

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
