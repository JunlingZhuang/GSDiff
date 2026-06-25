# Healthcare Floor Plan Agent — 开发计划（拆分版）

> **项目定位**：一个后端 LLM agent 系统，从自然语言生成 healthcare 平面图，支持规范校验、迭代纠错，以及用户与 agent 双向的带校验编辑。前端为独立的 web CAD canvas（不在本项目范围）。
>
> **文档版本**：v1.0（由单文件 `../HEALTHCARE_FLOORPLAN_AGENT_PLAN.md` 无损拆分）｜ **范围**：通用 healthcare 起步，后细分 ｜ **后端语言**：Python
>
> 原文各章节编号（如 5.2、11.2.1）在各分篇内保留，交叉引用照旧成立。

## 核心哲学（贯穿全部分篇）

> **LLM/VLM 只在"语义和结构"上放权，所有"精确几何和合规判定"焊死在确定性代码上。**
>
> - 状态机守红线，确定性校验卡关
> - LLM 提议，验证器裁决（LLM-Modulo）
> - 外层固定管线 + 局部受约束自主
> - 概率生成必须经确定性校验 + 人签字，才算合规（healthcare 安全红线）
> - 验证精力投向不确定性，不投向重要性
> - 垂直切片敏捷迭代，每阶段可演示可测

## 文档地图

| 分篇 | 内容 | 什么时候读 |
|------|------|-----------|
| [00 系统总览](00-overview.md) | 端到端管线 + 一句话技术栈 | 第一次接触项目 |
| [01 核心架构决策](01-architecture-decisions.md) | LangGraph 主体、DeepAgents 子图、两阶段生成、外部验证器纠错、硬软规范分离、单后端 | 动架构之前必读 |
| [02 建筑数据模型](02-data-model.md) | 三层表示、关系式建模（门窗挂墙）、version 乐观锁、IFC 预留 | 写任何读写 plan 的代码之前 |
| [03 统一校验 CRUD 入口](03-unified-crud.md) | operation 原语、`apply_operation` 单一入口、审计 | 实现编辑/校验时 |
| [04 Agent 编排架构](04-agent-orchestration.md) | 状态机全图、Supervisor + 角色专精、纠错循环纪律、退回机制、多 agent 扩展策略 | 实现/调整编排时 |
| [05 Agentic UI Flow](05-agentic-ui.md) | 四类 UI flow、双输出（chat+canvas）、WebSocket 事件、三入口一套 API、reasoning 分层 | 设计前后端交互协议时 |
| [06 风险分级与验证策略](06-risk-and-validation.md) | 🔴🟡🟢 分级、两个命门（生成路线 / 编排）并行验证 | Phase 0 开工前 |
| [07 敏捷开发分期](07-phases.md) | Phase 0–5 垂直切片：目标/范围/验收/扩展点 | 排期与每个 Phase 启动时 |
| [08 可扩展性设计](08-extensibility.md) | 四个可插拔维度（tools/规则/ML模型/流程） | 加新组件时 |
| [09 测试策略](09-testing.md) | 测试金字塔、工具 I/O 契约测试、LangSmith 评估、震荡检测 | 写测试时 |
| [10 准确度优化方法](10-accuracy-playbook.md) | 生成/CV parse/纠错/规范各环节优化；门窗检测策略 C；Plan B 切换阈值 | 精度不达标时查 |
| [11 用户场景设计](11-user-scenarios.md) | 渐进式信息收集、边界输入协同、前端渲染选型（react-konva）与接口契约 | 设计用户流程/前端契约时 |
| [12 参考 Repo 与论文资源](12-resources.md) | 分层取用的 repo（KirtiJha/DeepAgents/wassim249）+ 必读论文 + 数据集 | 找起点/查证据时 |
| [13 关键决策阈值速查](13-decision-cheatsheet.md) | 全部决策点一页速查表 | 随时 |

## 推荐阅读路径

- **新人入门**：00 → 01 → 02 → 07
- **开始 Phase 0**：06 → 10（§11.2 门窗部分）→ 12
- **开始 Phase 1**：02 → 03 → 12（KirtiJha 模板参考点）
- **实现编排/纠错**：04 → 09 → 13
- **对接前端画布**：05 → 11（§12.5 接口契约）→ 02
