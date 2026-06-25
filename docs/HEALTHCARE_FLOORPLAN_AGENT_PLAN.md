# Healthcare Floor Plan Agent — 开发计划与资源文档

> 本计划已拆解为分篇文档（v1.0 内容无损迁入，章节编号保留），便于按需查阅与各篇独立演进：
>
> ## → [docs/agent/](agent/README.md)
>
> 入口为 [agent/README.md](agent/README.md)（核心哲学 + 文档地图 + 推荐阅读路径）。

| 分篇 | 内容 |
|------|------|
| [00 系统总览](agent/00-overview.md) | 端到端管线 + 技术栈 |
| [01 核心架构决策](agent/01-architecture-decisions.md) | LangGraph 主体、两阶段生成、外部验证器纠错等 |
| [02 建筑数据模型](agent/02-data-model.md) | 三层表示、门窗挂墙、version 乐观锁 |
| [03 统一校验 CRUD 入口](agent/03-unified-crud.md) | operation 原语 + `apply_operation` |
| [04 Agent 编排架构](agent/04-agent-orchestration.md) | 状态机 + 子图、角色专精、纠错纪律、退回、多 agent 扩展 |
| [05 Agentic UI Flow](agent/05-agentic-ui.md) | 四类 flow、双输出、WebSocket、三入口一套 API |
| [06 风险分级与验证策略](agent/06-risk-and-validation.md) | 两个命门并行验证 |
| [07 敏捷开发分期](agent/07-phases.md) | Phase 0–5 垂直切片 |
| [08 可扩展性设计](agent/08-extensibility.md) | 四个可插拔维度 |
| [09 测试策略](agent/09-testing.md) | 金字塔 + 工具 I/O + LangSmith |
| [10 准确度优化方法](agent/10-accuracy-playbook.md) | 各环节优化 + 门窗策略 C |
| [11 用户场景设计](agent/11-user-scenarios.md) | 渐进式输入、边界协同、前端选型与契约 |
| [12 参考 Repo 与论文资源](agent/12-resources.md) | repo 分层取用 + 必读论文 + 数据集 |
| [13 关键决策阈值速查](agent/13-decision-cheatsheet.md) | 决策点一页速查 |
