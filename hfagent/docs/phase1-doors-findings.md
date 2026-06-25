# Phase 1 门层 & parser 加固 验证结果（2026-06-22）

> 计划文档：[docs/agent/](../../docs/agent/README.md) ｜ 架构：[agent-pipeline.md](agent-pipeline.md) ｜ 0-A：[phase0a-findings.md](phase0a-findings.md) ｜ 0-B：[phase0b-findings.md](phase0b-findings.md)

## 结论一句话

**门/邻接层落地,走"LLM 出房间连通图、确定性几何居中放门"** —— 既取代了计划 §11.2.1 的策略 C（纯规则推断），也淘汰了中途试过的"让 LLM 读门坐标"歧路。同期把墙图换成房间图，并在真实 sweep 中暴露、根治了 3 个 bug；消缝从 0-A 的固定膨胀演进为最近房间填充。所有改动经 37 项 pytest + 7-program 真实 Gemini sweep 验证；核心哲学仍是 LLM-Modulo（LLM 出结构、确定性代码裁决与定位）。

## 与计划 / 旧 findings 的差异（整合，不重写历史）

| 项 | 0-A / 0-B / 计划原状 | 本轮变更 | 理由 |
|---|---|---|---|
| 结构输出 | 墙图 `WallGraph`（`plan_to_wallgraph`） | **房间图 `RoomGraph`**（房间=节点，门=边） | 我们关心"哪两间房有门相连"，不是墙段拓扑。`merge 重合节点 / T 形插入`是否仍有价值、要否补回——**见文末"待决"** |
| 门窗 | 计划 §11.2.1 策略 C：纯几何规则推断，"本就不依赖图像" | **LLM 读真实图出连通 + 几何放门** | 色块图里门已被 pass-2 抹掉；同类多房间（如 12 间病房）纯规则无法判断哪间真有门通走廊 |
| 纠错止损 | 0-B：stall 计数器 + `stopped_early` 纪律 | **已删**，`max_correction_rounds` 单一上限即足够 | `phase0b-findings.md` 中 `test_oscillation_stops_early` 等已随之移除——该文档此点已过时 |
| 消缝 | 0-A：半墙厚固定 `dilate(6px)` | **最近房间填充**（`_fill_gaps`） | 固定 6px 对真实 16–24px 厚墙补不满，正是 0-A 遗留的缝 |
| 失败模式 #2 | 0-A："房间被门洞/墙线切碎"（已知复发） | 同源问题在 `_snap_axes` 再现并根治 | 见下 bug #3 |

## 门的方案：为什么是"LLM 出图、几何放门"

三条路对比：

1. **策略 C（计划原案）纯几何规则推断**——按"非循环房间各给一扇门通最近走廊"等规则推断。问题：是规则猜测，不反映真实设计意图；同类房间无法区分。
2. **LLM 读门坐标**（中途试过，已弃）——让模型返回每扇门的 (x,y)。**实证不可靠**：模型把 x 当比例 (0–1)、y 当像素 (0–768) 返回，`y=412.5×768` 直接飞出图外，同一模型两轴不同尺度。门全部错位。
3. **LLM 出连通 + 几何放门**（选定）——LLM 只回 `{room_a, room_b}` 连通边（无坐标），`door_placer` 在两房间真实共享墙上**居中**开门。

要点：
- **坐标交给几何，连通交给 LLM**：模型擅长"这两间通不通"，不擅长"门在第几像素"。
- **按房型匹配而非实例**：色块图丢失实例身份（同色块），"每个 patient_room–corridor 共享墙都开门"恰是意图，无需脆弱的 OCR/实例消歧。
- 真实图里房间用实例标签 `patient_room_1..N` 标注，仅供 LLM 命名门端点；pass-2 转色块时连标签一并抹掉（纯色块才好 parse，见 bug 旁注）。

## 自动调试：3 个真实 bug（均"后处理"为主）

1. **base64 图像坏档**。模型自动解析切到 `gemini-3-pro-image` 后，它把图片以 **base64 JPEG 文本**塞进 `inline_data.data`；旧 `generate_image` 直接写盘 → 坏 PNG → pass-2 报 `400 Unable to process input image`，7/7 全挂。代码没改、是环境模型变了暴露隐患。**修**：嗅探 magic bytes，需要时 base64 解码，按真实类型发 mime（`llm.py: decode_image_bytes` / `_sniff_mime`）。教训写入 CLAUDE.md：模型自动解析 → 图像字节必须格式无关，绝不直接落盘 `inline_data.data`、绝不硬编码 `image/png`。
2. **门画成斜的**。`door_placer._shared_segment` 对"绕过拐角的阶梯/L 形共享边界"取折线**首尾两点连成对角弦**。简单 program 0 斜门、inpatient-ward 77 条共享边里 18 条折线 → 正好 18 个斜门。**修**：把折线拆相邻点对，取**最长单段直边**放门。
3. **整片房型丢失**。`cv_parse._snap_axes` 用**单链聚类**（按"与上一个值之差 ≤ tol"分组）；斜墙翼产生的密集连续坐标把聚类"链爆"成跨度极大的巨簇，不同墙被吸到同一坐标 → 房间塌成退化多边形被丢。hospital-floor 因此整片丢 patient_room/nurse_station（解析出 10 间、缺 2 类），**一个斜翼污染全图**。**修**：聚类锚定到**组首**、跨度限 tol，杜绝链式蔓延。修后 8 房型全识别。

回答"门错位是 LLM 识别错还是后处理错"：**主要是后处理**（#2、#3）。LLM 侧唯一问题是它被我们"保留标签"的指令诱发把颜色 hex（`#FF6D01`）当房名写——已撤回该指令、改回纯色块。

## 消缝：固定膨胀 → 最近房间填充（0-A 半墙膨胀的演进）

- **机理**：色块图房间间是黑墙，cv_parse 把黑/白判为非房间，每个房间停在墙内沿 → 留墙宽的缝。
- **0-A 旧法**：每房间固定 `dilate(6px)` 各胀半墙。8px 合成墙能补满（测试看着完美），真实 16–24px 厚墙补不满 → 真实图漏缝。
- **新法 `_fill_gaps`**：把每个墙/缝像素分配给**最近的房间** → 两房间边界落在墙**中线**，缝归零；同色相邻房间仍是**两个独立 id**；离房间远的外部不填。
- **配套**：粗网格描轮廓后相邻边界各可偏 ≤1 网格（共 2×grid），故 `_snap_axes` 容差从 1.6×grid 提到 **2.5×grid** 桥接；`door_placer` 邻接判定改用 `a.boundary ∩ b.buffer(tol)`，对残留小缝/小重叠鲁棒。
- **实测**：health-center-large 真实色块重渲，房间-走廊白条基本消失。

## 真实 sweep 结果（7 program）

- **斜门 7/7 全部 0**（修 #2 后）。
- **色块图无文字、无 hex 标签**（撤回"保留标签"后）。
- **房间数 5~6/7 精确**（VLM 计数固有短板，确定性 `fix_room_counts` 兜底；与 0-A 结论一致）。
- hospital-floor 修 #3 后从"丢 2 房型"→ 8 房型全识别。
- 偶发：`nurse_station` 被 VLM 画太暗 `(163,20,63)`，到 storage/nurse_station 距离几乎相等（77.9 vs 79.3）边缘误分类——隐患，对策候选：拉开调色板中两色距离。

## 墙图操作的去向（已调研，结论）

跨前后端 + git 历史调研后定论：**"合并重合节点 / T 形插入"要保留其能力，但不恢复后端 `WallGraph` schema**（符合本仓"room graph, not wall graph"硬约束）。

- **两操作解决什么**：① 合并重合节点 → 相邻房间公共墙引用**同一个 node**；② T 形插入 → 角点落在别房墙中段时插入该 loop。二者都是前端 CAD 内核**拖动不撕裂**的硬前提——`kernel.ts` 的 `dragWallSeg`/`collinearChainNodes`/`moveNode` 全按共享 node id 整体平移，node 不共享就裂缝/重叠。医疗图"长走廊+两侧小房"T 形结点遍地，不是边角料。
- **`adjacency()` 不算损失**：墙图的几何邻接（隔墙=2 face）已被 `door_placer._shared_segment` 等价覆盖；`RoomGraph` 提供的是"有没有门"（几何共享墙 ≠ 有门），二者互补不重复。
- **⚠️ 当前 broken（最高优先）**：前端 `AgentWorkspace.tsx:86` 仍调 `agentToKernelGraph(r.wallgraph)`，而后端 `api.py` 现在只返回 `plan`+`room_graph`、**无 `wallgraph` 键** → Agent 模式运行时拿到 `undefined`。删墙图时漏改了前端契约。

**补法（分两段）**：
1. **第 1 段（必须，修 broken）**：后端把 `door_placer` 升级/追加 `plan_to_walls(plan)`，输出**连通的 `Plan.walls`**（房间每条边成墙、公共墙去重），不再只有逐门的孤立墙段；前端 `AgentWorkspace` 改走 Design 模式同款 `planToGraph(plan)`（`kernel.ts:40`，其 `getNode` 已**免费做合并节点**），`agent-plan.ts` 改为把后端 `Plan`(mm/px、y 朝下)适配为内核 `Plan`(米、y 朝上)。即合并节点交给前端现成逻辑，后端不再背 WallGraph。
2. **第 2 段（启用拖拽编辑时）**：在前端 `planToGraph` 合并节点后补一个 **T 形插入 pass**（逻辑移植已删 `wallgraph.py` step 2 的 `on_edge()`）；`kernel.ts` 现只有手动 `splitWall`、无自动 T 形检测，是真缺口。

## 收敛到 02-data-model（已做，2026-06-25）

后端最终输出已统一成 `docs/agent/02-data-model.md §3.2` 的权威 `Plan`（rooms + 完整 walls + 门挂墙 + adjacency_graph），不再是散落的 `fixed_plan`/`plan_with_doors`/`RoomGraph`：

- **完整墙表**：新增 `tools/plan_to_walls.py`，遍历房间多边形边、同端点去重 → `Plan.walls`。**不做** T 形插入/共享 node（留前端编辑层）。
- **门挂真实墙**：`door_placer.place_doors` 改为把门挂到 `plan_to_walls` 的墙上（`Door.wall_id`+`position`），不再各造孤立墙段；并返回 `door→(room_a,room_b)` 映射。
- **adjacency_graph + via**：`Plan` 加 `adjacency_graph{nodes,edges{from,to,type,via}}`（pydantic `from_` alias `from`）；`generate_plan` 用 door 映射把每条边连到具体 door id。`Room` 补可选 `name`/`zone`。
- **输出与返回**：写 `plan.json`（权威，取代 `doors.json`）；`generate_plan` 返回 `(report, plan.model_dump(by_alias=True), room_graph.model_dump())`。
- 验证：37 pytest 全绿；5 个真实 program 自检——每个 `Door.wall_id` 命中 `walls[]`、每条 `edges[].via` 命中真实 door id、pydantic 校验通过。
- **仍未做**：① T 形 node 共享/合并(前端 CAD 编辑层)；② 前端契约断链——`AgentWorkspace.tsx:86` 仍读 `r.wallgraph`,后端已无此键(本轮明确只动后端,前端留下一轮)。

## 待决（输入下一轮）

- 前端契约：`AgentWorkspace` 改走 `planToGraph(plan)`、`agent-plan.ts` 适配单位/y 轴（第 1 段前端部分）。
- 前端编辑期 T 形插入（第 2 段）。
- `nurse_station`/`storage` 调色板拉距（缓解暗色误分类）。

## 复现

```bash
python -m pytest hfagent/tests/ -q          # 确定性层，无需 API key
python -m hfagent.evaluate                   # 全量真实 Gemini sweep（需 GEMINI key）
```
