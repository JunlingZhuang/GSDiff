# 系统总览

> Healthcare Floor Plan Agent 开发计划（拆分版）｜ [← 总览与文档地图](README.md)

### 1.1 端到端管线

```
自然语言需求
   │
   ▼
① 需求理解 (LLM)  ──────────  结构化 program (房间清单/面积/邻接/类型)
   │
   ▼
② 规范检索 RAG (生成前)  ────  注入适用硬规范 + 软导则 + 健康typology库
   │
   ▼
③ 生成 (VLM/agent 子图)  ────  Gemini 出色块平面图 (粗布局，非精确几何)
   │
   ▼
④ CV Parse  ─────────────────  junctions → IP装配 → Shapely规整 → JSON
   │
   ▼
⑤ 双层校验
   ├─ 硬规范引擎 (确定性代码) ──  返回结构化违规 (带元素ID)
   └─ 软导则评审 (LLM+RAG)   ──  主观评分
   │
   ├─ 有违规 ──▶ ⑥ 纠错循环 (外部验证器驱动，有上限)
   │                └─▶ 退回③ 或 直接改坐标
   │
   └─ 通过 ──▶ ⑦ 渲染推送 + 编辑
                  │
                  ▼
            ⑧ 编辑 (用户 + agent 双向，走统一CRUD)
```

### 1.2 一句话技术栈

| 层 | 技术 |
|----|------|
| 编排骨架 | LangGraph 状态机 |
| 生成子图 | DeepAgents 风格（嵌入 LangGraph） |
| VLM | Google Gemini（图像生成 + 结构化输出） |
| CV parse | OpenCV + 整数规划装配 + Shapely 规整 |
| 几何校验 | Shapely（多边形相交/面积/包含） |
| 硬规范引擎 | 确定性 Python 规则 |
| 软导则 | RAG（向量库 + 典型知识图） |
| 后端通信 | FastAPI + WebSocket |
| 可观测/评估 | LangSmith（tracing + evaluation） |
| 持久化 | PostgreSQL（LangGraph checkpointer 用 PostgresSaver） |
