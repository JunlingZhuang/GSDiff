# 敏捷开发分期

> Healthcare Floor Plan Agent 开发计划（拆分版）｜ [← 总览与文档地图](README.md)

> **垂直切片，不是水平分层。** 每个 phase 打穿全栈、越来越厚，而非按技术层一层层堆。每阶段结束有可演示成果，随时可调方向。

### 8.1 为什么敏捷不瀑布

瀑布的风险：做到 Phase4 才发现 Phase2 的色块图路线行不通，前面全废。敏捷每 phase 都端到端产出能跑、能测、能看的薄切片。**敏捷的可行性本身依赖好的可扩展架构**——两者绑定。

### 8.2 每个 Phase 的标准结构

```
Phase N
├── 目标（一句话：打通什么垂直切片）
├── 范围（做什么 / 明确不做什么 —— 防膨胀）
├── 可演示成果（结束能给人看什么）
├── 验收标准（怎么算完成）
├── 测试（这切片怎么测）
└── 扩展点（为下一迭代预留什么接口）
```

### 8.3 Phase 清单

#### Phase 0 — 验证命门（1周内，tool 级 + 单元测试，无编排架构）

> **状态**：0-A ✅ 已完成（2026-06-10，结论：路线走得通，图文报告见 [hfagent/docs/phase0a-findings.md](../../hfagent/docs/phase0a-findings.md)）；0-B ⬜ 未开始。

> **修订（v1.1）**：原版"脚本级、用完即弃"与 §10.2 的工具 I/O 契约测试自相矛盾。0-A 验证的三个步骤（生成→parse→还原渲染）**就是未来生产管线的节点**，所以从第一天起写成带 I/O 契约和单元测试的正式 tool——验证产物直接沉淀为永久回归测试。不写的仍然是：编排、状态机、前后端、数据库。

- **目标**：分别验证两个高风险命门，决定路线走不走得通；同时沉淀生产管线的前三个 tool 及其测试。
- **0-A 三个 tool（存本地 `hfagent/tools/`，每个带 pytest）**：
  ```
  ① generate_colorblock   结构化 program ──Gemini──▶ 色块平面图 PNG
  ② cv_parse              色块 PNG ──OpenCV+Shapely──▶ Plan JSON（房间多边形+类型）
  ③ render_plan           Plan JSON ──确定性渲染──▶ 还原平面图（SVG/PNG）
  ```
- **0-A 两层验证（先隔离 parser，再量 VLM）**：
  1. **Round-trip 闭环（无 VLM 噪声）**：手工合成 Plan fixture → `render_plan` → `cv_parse` → 与原 Plan 对比（逐房间 IoU、房间数、类型准确率）。先把 parser 本身测到可信——这层是确定性单元测试，进 CI。
  2. **真实 VLM 评测**：`generate_colorblock` 真调 Gemini（API key 门控的集成测试，N≥20 个 program）→ 同一个 parser parse → `render_plan` 还原 → 人眼对比 + 几何合法率统计。parser 已被第 1 层证明可信，这里量出的差距就纯粹是 VLM 图像质量，两个变量干净分离。
- **0-B**：mock VLM/CV，喂故意带违规的假平面，跑生成-校验-纠错循环，看收敛/死循环/退回判断（同样写成 pytest 可重跑，KirtiJha 模板的 zero-config mock 直接用）。
- **测试策略**：①mock API 测 prompt 构造与落盘契约；②用合成色块图 fixture 做确定性单测（ground truth 已知）；③用合成 Plan fixture 测渲染输出契约；round-trip 测试串联 ②③。
- **可演示**：round-trip 测试通过 + 指标报告；Gemini 真图 → parse JSON → 还原图三联对比；纠错循环收敛日志。
- **验收**：round-trip IoU 达标（parser 可信）；0-A 得出"VLM 色块图几何合法率是否 >70%"结论（§11.5 阈值）；0-B 得出"编排能否收敛"结论。
- **扩展点**：三个 tool 直接成为 Phase 1 管线节点；确定主路线 or 切 Plan B（HouseTune 扩散 refine）。

**代码存放（本地，本 repo）**：新建顶层 `hfagent/`（与 `app/`、`digress/`、`procedural/` 平级），自包含：

```
hfagent/
├── tools/            # generate_colorblock.py / cv_parse.py / render_plan.py
├── schema/           # Plan JSON schema（§3 数据模型，Phase 0 先用最小子集）
├── tests/            # pytest：单测 + round-trip + API-key 门控的集成测试
│   └── fixtures/     # 合成色块图 PNG + 合成 Plan JSON（ground truth）
└── README.md
```

自包含的好处：Phase 1 在它上面长出 FastAPI 与编排；若将来要独立成 repo，整体搬走即可。

#### Phase 1 — 最薄端到端骨架（能跑）
- **目标**：文本→生成→parse→数据模型→存→API 吐 JSON，全栈打通。
- **范围**：1 种房间类型，无校验，无前端。**不做**：规范、编辑、退回。
- **可演示**：调 API 输入文本，返回一个平面 JSON。
- **验收**：端到端不报错，产出合法 JSON。
- **扩展点**：数据模型 schema 定稿；tool 注册机制就位。

#### Phase 2 — 加校验 + 渲染（看得见）
- **目标**：接 1–2 条硬规范校验；前端能读 API 把平面画出来。
- **范围**：硬规范引擎雏形 + 几何校验 + 简单前端渲染验证数据模型够不够画。
- **可演示**：生成的平面在画布显示；违规被标出。
- **验收**：违规能被确定性检出；前端能渲染数据模型。
- **扩展点**：规则引擎可插拔；校验器作为 LangGraph 节点。

#### Phase 3 — 加编辑 + 实时（能交互）
- **目标**：统一 CRUD 入口；WebSocket；agent 改→画布动。
- **范围**：用户和 agent 都能改；基础 agentic UI flow（生成 + 高亮 + 预览）。
- **可演示**：用户拖墙画布更新；对 agent 说"加个洗手间"画布出现。
- **验收**：双向编辑走同一校验入口；事件正确推送渲染。
- **扩展点**：operation 类型可扩展；UI flow 事件类型可扩展。

#### Phase 4 — 加智能 + 鲁棒（能用）
- **目标**：退回/time-travel；planning；准确度优化；更多规范（RAG）；更多房间类型；用户引导场景。
- **范围**：完整纠错循环 + interrupt 退回 + 软导则 RAG + 典型库注入。
- **可演示**：复杂需求生成→自动纠错收敛→用户中途反悔回退→继续编辑。
- **验收**：纠错收敛率达标；退回干净；规范覆盖扩大。
- **扩展点**：ML 模型可插拔；多候选生成（ToT/LATS）接口预留。

#### Phase 5 — 上生产（能部署）
- **目标**：鉴权/限流/监控/迁移（摘 wassim249 模块）。
- **范围**：JWT、slowapi 限流、Langfuse/LangSmith、Alembic 迁移、Postgres checkpointer。
- **可演示**：带监控看板的可部署服务。
- **验收**：生产并发、可观测、可回溯。
- **注**：这些到这里才需要，不能更早——早背生产基建会拖慢命门验证。
