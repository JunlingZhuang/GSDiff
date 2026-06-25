# 测试策略

> Healthcare Floor Plan Agent 开发计划（拆分版）｜ [← 总览与文档地图](README.md)

> 纯后端，要能测试、验证、测试工具 I/O。配合 LangSmith。

### 10.1 测试金字塔

| 层 | 测什么 | 工具 |
|----|--------|------|
| 单元测试 | 几何校验、规则引擎、CV parse 函数 | pytest |
| 工具 I/O 测试 | 每个 tool 的输入输出契约 | pytest + 固定 fixture |
| 节点测试 | 每个 LangGraph 节点（mock 依赖） | pytest |
| 集成测试 | 子图/全图端到端 | pytest + mock VLM/CV |
| 评估测试 | agent 输出质量、收敛 | LangSmith |

### 10.2 工具 I/O 测试（你特别要求）

每个工具有明确输入输出契约，用固定 fixture 测：

```python
def test_cv_parse_io():
    img = load_fixture("colorblock_clinic.png")
    result = cv_parse(img)
    assert result.is_valid_json()
    assert all(room.polygon.is_closed() for room in result.rooms)
    assert result.matches_schema(BuildingSchema)

def test_geometry_validator_io():
    plan = load_fixture("overlapping_rooms.json")
    errors = validate_geometry(plan)
    assert any(e.type == "overlap" for e in errors)  # 应检出重叠
```

### 10.3 LangSmith 评估

| 评估类型 | 用途 |
|---------|------|
| 确定性评估器（主指标） | % 硬约束满足、几何合法率、房间数/邻接匹配 |
| 收敛指标 | 迭代到合法的轮数、触顶率、每轮违规数（应单调降） |
| LLM-as-judge | 软设计质量 |
| 轨迹评估 | agent 工具调用顺序/参数对否、检测震荡 |
| 人工标注队列 | 临床/建筑专家审查 |

**回归数据集**：从真实失败轨迹构建——每个生产失败变成永久测试。每 PR 跑离线评估（pytest 集成）。

### 10.4 关键：检测震荡

轨迹评估专门抓"agent 在两个非法状态间反复横跳"和低效工具调用——纠错循环的典型病。
