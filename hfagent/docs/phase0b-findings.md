# Phase 0-B 验证结果（2026-06-10）

> 计划文档：[docs/agent/](../../docs/agent/README.md) ｜ 0-A 报告：[phase0a-findings.md](phase0a-findings.md)
>
> ⚠️ **部分已被取代（2026-06-22）**：下文"震荡止损（stall）"纪律及对应测试 `test_oscillation_stops_early_and_keeps_best` / `stopped_early` 上报**已移除**——`max_correction_rounds` 单一上限已足够。详见 [phase1-doors-findings.md](phase1-doors-findings.md)。其余纪律（量化反馈、best-round 保留、确定性兜底）仍有效。

## 结论一句话

**纠错编排的全部纪律（§5.3）在 mock 隔离下确定性验证通过**——收敛、量化反馈、震荡止损、best-round 保留、确定性兜底，18/18 pytest（含 0-A 的 14 项），零 API 成本、可随时重跑。

## 方法（按 §7.3 的 mock 隔离原则）

[`tests/mock_vlm.py`](../tests/mock_vlm.py)：剧本化 MockVLM——内部持有一串 Plan，每轮用确定性 `render_plan` 出图喂给**真实的**纠错循环（`evaluate_one`）和**真实的** `cv_parse`。被 mock 掉的只有 Gemini 本身，所以测的是编排逻辑，不被 VLM 噪声干扰；剧本控制"模型"每轮变好还是变坏。

## 验证的纪律（[`tests/test_orchestration.py`](../tests/test_orchestration.py)）

| §5.3 纪律 | 测试 | 验证内容 |
|---|---|---|
| 外部验证器驱动 + 量化反馈 | `test_converges_when_model_improves` | 反馈文本确实含 "exam_room: drew 1, required 2" 级别的量化违规；模型改对后第 2 轮收敛 |
| 不浪费调用 | `test_perfect_first_round` | 首轮全对 → 恰好 1 次调用即停 |
| **违规数不单调下降 → 止损** | `test_oscillation_stops_early_and_keeps_best` | 剧本永远变坏：连续 2 轮无严格改善即停（5 轮上限只跑了 3 轮），不烧满配额 |
| 返回违规最少版 | 同上 | best_round=1（最优），不是最后一轮 |
| 确定性兜底 | `test_deterministic_fix_repairs_what_loop_could_not` | 循环失败的 case，`fix_room_counts` 重贴标签修复 → `final_count_exact=true` |

止损逻辑实现于 `run_phase0a.evaluate_one`（`stall` 计数器 + `stopped_early` 上报），真实管线与 mock 测试走同一份代码。

## 与 0-A 真实数据的交叉印证

0-A 实测的 clinic-mixed 4→6→6 与 hospital-floor 1→2→5（像素纠错雪崩）正是本轮 mock 剧本"oscillating"的真实原型——止损纪律不是假想防御，是已观测失败模式的对策。

## 对 Phase 1 的输入

- 编排迁移到 LangGraph 时（生成→parse→校验→条件边回退），每个节点直接包装现有 tool 函数；mock 剧本平移为 LangGraph 节点级测试。
- 仍缺（属 Phase 1+）：Reflexion 记忆（记住改过什么防横跳）、退回到"需求理解"层的多级回退、interrupt 人工接管。
