# 可扩展性设计

> Healthcare Floor Plan Agent 开发计划（拆分版）｜ [← 总览与文档地图](README.md)

> 要求：轻易扩展 RAG / 更多 tools / ML 模型 / 更改流程。可扩展性是敏捷迭代的前提。

### 9.1 四个可插拔维度

| 维度 | 机制 | 加新东西时 |
|------|------|-----------|
| Tools | LangChain `@tool` 注册表，drop-in | 放一个 @tool 函数注册即可，agent 自动识别 |
| 规范规则 | 规则引擎，每条规则独立模块 | 注册一条新规则，不改核心 |
| ML 模型 | 模型作为 LangGraph 节点/工具，统一接口 | 包一个新模型成节点接进图 |
| 流程 | LangGraph 节点+边，模块化 | 加/换节点，不动其他节点 |

### 9.2 设计原则

- **数据模型作为通用语言**：所有组件读写同一 JSON，新组件只要遵守 schema 即可接入。
- **节点即函数**：每个 LangGraph 节点是独立可测函数；换实现不影响外层。
- **子图隔离**：生成子图有独立 state，脏数据不污染外层；可整体替换（如 DeepAgents → 手写）。
- **校验入口单一**：新增 operation 类型只需在统一入口注册，校验逻辑复用。
- **VLM/CV 可换**：Gemini 和 CV parse 都是工具，路线 A/B 可切换（预留 HouseTune 扩散 Plan B 接口）。

### 9.3 ML 模型扩展示例

```python
# 把一个布局优化模型作为节点接入
def layout_optimizer_node(state):
    plan = state["floorplan"]
    optimized = my_ml_model.optimize(plan)   # 任意 ML 模型
    return {"floorplan": optimized}

graph.add_node("optimize", layout_optimizer_node)  # 一行接入
```
