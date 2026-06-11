# 建筑数据模型（地基层）

> Healthcare Floor Plan Agent 开发计划（拆分版）｜ [← 总览与文档地图](README.md)

> 这是整个项目最底层、最先定的东西。生成、CV parse、规范校验、编辑、前端渲染**全部读写同一个数据模型**。模型定错，上面每层都返工。

### 3.1 三层表示，缺一不可

| 层 | 存什么 | 服务于 |
|----|--------|--------|
| 拓扑层 | 房间邻接关系、连通性 | 生成规划、规范校验（洁污分流） |
| 几何层 | 墙线段、房间多边形、绝对坐标 | 渲染、尺寸校验 |
| 语义层 | 房间功能（手术室）、墙类型（承重） | 规范匹配、用户理解 |

healthcare 尤其需要语义+拓扑，因为其规范大量是"什么功能房间该挨着/远离什么"。

### 3.2 关系式建模（关键：门挂在墙上）

**核心原则**：门窗不存独立坐标，而是挂在墙上。墙移动时门自动跟随——这省掉大量"穿墙/门飘空"的 bug。

```json
{
  "plan_id": "p1",
  "version": 7,
  "units": "mm",
  "building_type": "healthcare_generic",
  "rooms": [
    {
      "id": "r1",
      "name": "病房 A",
      "type": "patient_room",
      "polygon": [[0,0],[4000,0],[4000,3000],[0,3000]],
      "area": 12000000,
      "zone": "ward"
    }
  ],
  "walls": [
    {"id": "w1", "start": [0,0], "end": [4000,0], "thickness": 200, "load_bearing": true}
  ],
  "doors": [
    {"id": "d1", "wall_id": "w1", "position": 0.5, "width": 1200, "type": "door"}
  ],
  "adjacency_graph": {
    "nodes": ["r1", "r2", "corridor_1"],
    "edges": [
      {"from": "r1", "to": "corridor_1", "type": "door", "via": "d1"}
    ]
  }
}
```

### 3.3 设计要点

- **每个元素有稳定 id** → agent 说"改 r1"而非"第3个房间"，前端按 id 做增量 diff。
- **version 字段** → 乐观锁，防止用户拖拽和 agent 修改并发打架。
- **门窗挂墙**（wall_id + 相对位置 0~1）→ 墙移动门自动跟随。
- **房间多边形 + 邻接图双表示** → 几何归几何、拓扑归拓扑，各自服务渲染与校验。
- **uniform wall thickness** → 统一墙厚，便于几何运算与 3D 转换（ResPlan 经验）。

### 3.4 预留转换接口

MVP 用自定义 JSON。但预留 `to_ifc()` / `from_ifc()` 等转换接口，未来对接 IFC / CAD 生态。转换层独立，不污染核心模型。
