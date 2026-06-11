# 参考 Repo 与论文资源

> Healthcare Floor Plan Agent 开发计划（拆分版）｜ [← 总览与文档地图](README.md)

### 13.1 代码 Repo（分层取用，不是二选一）

```
┌─────────────────────────────────────────────┐
│ 生产基建层 ← wassim249/fastapi-langgraph-...   │
│ 鉴权/会话/限流/Langfuse/Prometheus/迁移         │
│ （Phase 5 才摘用；注意目前仅 OpenAI，Gemini需自改）│
├─────────────────────────────────────────────┤
│ 工作流编排层 ← KirtiJha/langgraph-interrupt-... │
│ 状态机/生成校验循环/interrupt退回/time-travel    │
│ /并行扇出/zero-config mock/FastAPI+Next.js      │
│ （主起点，Phase 1 起用）                         │
├─────────────────────────────────────────────┤
│ 生成子图 ← langchain-ai/deepagents             │
│ planning/subagents/filesystem，可嵌入 LangGraph │
├─────────────────────────────────────────────┤
│ 领域核心（自己写，护城河）                       │
│ VLM色块生成 / CV parse / 规范引擎 / 几何校验      │
└─────────────────────────────────────────────┘
```

| Repo | 用途 | 何时 |
|------|------|------|
| `KirtiJha/langgraph-interrupt-workflow-template` | 工作流主骨架（退回/time-travel/mock） | Phase 1 主起点 |
| `langchain-ai/deepagents` | 生成子图（planning/subagent） | Phase 4 生成环节 |
| `wassim249/fastapi-langgraph-agent-production-ready-template` | 生产基建摘用 | Phase 5 |
| `GoogleCloudPlatform/generative-ai` | Gemini+LangGraph 集成参考 | Gemini 接入 |
| `von-development/awesome-LangGraph` | 生态索引，找更新模板 | 随时查 |

**Gemini 硬伤提醒**：wassim249 目前仅支持 OpenAI，KirtiJha 多 provider。用 Gemini 都需参考 GoogleCloudPlatform 例子改 LLM registry。clone 任何模板前先看最近 commit 日期（LangGraph 已到 v1.2，别捡过时货）。

#### 13.1.1 每个 Repo 的具体参考点（打开后看哪里）

**`KirtiJha/langgraph-interrupt-workflow-template`（主起点）**
- **参考它的整体结构**作为项目骨架，把它的业务节点替换成你的（需求理解/生成/parse/校验/编辑）。
- **interrupt 机制**：看它如何用多中断点 + approve/edit/redirect 恢复 → 直接复用为"用户审阅 + 三向退回"。
- **time-travel 实现**：看它如何 `get_state_history` + 按 checkpoint_id fork → 复用为"反悔回任意草案版本"。
- **planner 扇出（Send / map-reduce）**：看它如何并发子任务 → 借鉴为按 healthcare 功能分区并行生成。
- **zero-config mock 模型**：直接用来做 Phase 0-B 的 agent 编排隔离验证（不用 API key）。
- **FastAPI + Next.js 前后端**：参考其 WebSocket/事件推送结构，对接你的 web CAD。

**`langchain-ai/deepagents`（生成子图）**
- **`create_deep_agent` 用法**：看如何传 tools/instructions/subagents → 用作你的生成子图。
- **planning 工具（write_todos）**：参考它如何让 agent 拆 todo → 用于生成环节的任务分解。
- **subagents 机制**：参考 `task` 委派 + 上下文隔离 → 按科室拆子任务。
- **"任何 CompiledStateGraph 可作 subagent"**：看 README 此说明 → 这是它无缝嵌入 LangGraph 的依据。
- **注意它是"trust the LLM"全自主**：只用作受外层状态机约束的子图，别让它当主体。

**`wassim249/fastapi-langgraph-agent-production-ready-template`（Phase 5 基建摘用）**
- **`app/services/llm/registry.py`**：LLM 循环降级 + 指数退避重试 + 超时预算 → 摘用（但要改成 Gemini）。
- **JWT 鉴权 + slowapi 限流**：`app/core/limiter.py` 等 → 摘用为生产鉴权/限流。
- **Langfuse tracing + Prometheus/Grafana**：参考其可观测接线（你用 LangSmith 可类比）。
- **Alembic 迁移 + pgvector**：参考其 DB 迁移和向量库接法 → 用于规范 RAG 持久化。
- **结构化日志（per-request context）**：参考其中间件 → 审计可追溯。
- **只摘模块、别整体起手**：它是后端基建，不含你要的生成校验循环/退回。

**`GoogleCloudPlatform/generative-ai`（Gemini 接入）**
- **`gemini/agent-engine/langgraph_human_in_the_loop.ipynb`**：Gemini + LangGraph + 中断 + time-travel 的官方集成 → 照它接 Gemini，少踩 LLM registry 改造的坑。
- **Gemini 结构化输出（response_schema）**：参考如何强制 JSON 输出 → 用于需求理解和生成的结构化 program。

**`von-development/awesome-LangGraph`（生态索引）**
- 分类索引大量模板，clone 任何模板前去翻有无更新版；找 deep-agent UI、生成式 UI 等子方向的最新样板。

### 13.2 必读论文（按顺序）

| # | 论文 | 出处 | 给你什么 |
|---|------|------|---------|
| 1 | **MedBuild AI** | arXiv 2511.11587 (2025) | 最近的 healthcare 三 agent 神经符号同类 |
| 2 | **LLM-Modulo** | ICML 2024 / arXiv 2402.01817 | 治理原则：LLM 提议，验证器裁决 |
| 3 | **Text2BIM** | ASCE JCCE 2026 / arXiv 2408.08054 | verify-fix 循环蓝图（BCF 结构化违规） |
| 4 | **Chen & Bao** | Automation in Construction 177:106331 (2025) | 代码校验工具=合规，LLM不算数学 |
| 5 | **Kamoi et al.** | TACL 2024 | 自纠错护栏：何时有效/失败 |
| 6 | **HouseTune** | **AAAI 2026**, v40(16):14059-14067 | 两阶段生成（已同行评审）；生成环节参考 + CV路线 Plan B |
| 7 | **ChatHouseDiffusion** | arXiv 2410.11908 | JSON state + 局部编辑机制 |
| 8 | CRITIC / Reflexion / ReAct | ICLR2024 / NeurIPS2023 / ICLR2023 | 循环/记忆/工具原语 |

### 13.3 数据集与方法资源

| 资源 | 用途 |
|------|------|
| Tell2Design (ACL 2023) | 80,788 文本-平面对，约束分类法 |
| RPLAN | 80k 住宅平面（注意：住宅，healthcare 需自建） |
| CubiCasa5K | CV parse 训练参考（junction heatmap） |
| Raster-to-Vector (ICCV 2017) | junctions→IP 装配，~90% 精度 |
| ResPlan | Shapely 几何清理配方 |
