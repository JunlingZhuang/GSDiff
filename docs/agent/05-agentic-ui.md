# Agentic UI Flow（前端驱动）

> Healthcare Floor Plan Agent 开发计划（拆分版）｜ [← 总览与文档地图](README.md)

> agent 表面是 chatbot，但动作要投射到前端画布。机制：**agent 改后端真相源 → 后端推事件 → 前端画布按数据重绘**。agent 永不直接碰画布。

### 6.1 四类 UI Flow

```
              改数据                    不改数据(纯UI)
          ┌──────────────────┬──────────────────────┐
agent主导  │ A. 数据变更类      │ B. 视图引导类          │
          │ 生成/改/删平面     │ 高亮/聚焦/标注         │
          ├──────────────────┼──────────────────────┤
需用户参与 │ C. 确认/预览类     │ D. 交互请求类          │
          │ 幽灵预览→确认落地  │ 问澄清/给选项/等输入    │
          └──────────────────┴──────────────────────┘
```

### 6.2 A — 数据变更类（画布重绘）

| Flow | 触发 | 事件 | 画布 |
|------|------|------|------|
| 生成平面 | "生成一个诊所" | `plan_created` | 加载全新平面 |
| 增加元素 | "加个洗手间" | `room_added` | 出现新房间 |
| 修改元素 | "病房扩大1米" | `room_updated` | 房间变形 |
| 删除元素 | "去掉这道隔墙" | `element_deleted` | 元素消失 |
| 批量重构 | "病房全朝南" | `plan_patched` | 多处同时变 |
| 撤销回退 | "退回上一版" | `plan_reverted` | 回到历史版本 |

### 6.3 B — 视图引导类（不改数据，引导注意力）

healthcare 平面复杂，agent 光说"3号房洁污交叉"用户找不到。配合高亮+聚焦，一眼看到问题。

| Flow | 事件 | 画布 |
|------|------|------|
| 高亮元素 | `highlight` | 指定房间闪烁/变色 |
| 聚焦镜头 | `focus_view` | 平移+缩放到某区域 |
| 标注说明 | `annotate` | 加箭头/文字 |
| 显示尺寸 | `show_dimensions` | 叠加尺寸层 |
| 路径演示 | `draw_path` | 画疏散动线 |
| 对比展示 | `show_diff` | 改前/改后叠加 |
| 清除标注 | `clear_overlay` | 移除临时标注 |

### 6.4 C — 确认/预览类（高风险改动需用户拍板）

healthcare 关键改动不能 agent 说改就改。

| Flow | 机制 | 画布 |
|------|------|------|
| 幽灵预览 | 改动半透明显示，数据未落地 | 预览叠加层 |
| 确认落地 | 用户点"接受"，预览转正式 | 幽灵变实线 |
| 拒绝撤销 | 丢弃预览 | 恢复原状 |
| 多方案选择 | 同时给多预览供选 | 切换展示方案 |

配合 LangGraph interrupt：拆/移已验证科室等重大操作时暂停，approve/edit/reject。

### 6.5 D — 交互请求类

| Flow | 事件 | 前端 |
|------|------|------|
| 问澄清 | `ask_clarification` | 弹问题（可用按钮选项） |
| 给选项 | `present_options` | 可点选项卡 |
| 等输入 | `request_input` | 等用户输边界/参数 |

### 6.6 双输出本质

agent 一个动作 → 两个投射：**一个到画布（改了什么）+ 一个到聊天（说了什么）**。例如"生成诊所"：画布 `plan_created` 渲染 + chat"已生成，含12个房间..."并行输出。

### 6.7 传输：WebSocket

```javascript
websocket.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case "plan_created":  canvas.loadPlan(msg.payload);   break;
    case "room_updated":  canvas.updateRoom(msg.payload); break;
    case "highlight":     canvas.highlight(msg.payload);  break;
    // ...事件类型可扩展
  }
};
```

无论变化来自用户还是 agent，后端用同一通道同一事件格式推送。前端不区分来源，收到即更新。

### 6.8 交互入口与 API 设计（聊天框 + 工作区共用一套 API）

> **核心**：聊天框和工作区（按钮、画布）都调用同一套后端 API / 统一 operation 入口。这是成熟工具标配（Figma/Cursor/Notion），纯聊天是过渡形态。

#### 6.8.1 三个交互入口，一套后端

```
三个发起来源                统一入口(带校验)        数据
─────────                  ─────────────────       ────
① 聊天框(自然语言)  ─┐
② 工作区按钮         ─┼─▶  apply_operation(op)  ─▶  建筑数据模型
③ 画布直接拖拽       ─┘      ├─ 几何校验 ✓            (真相源)
                            ├─ 规范校验 ✓
                            └─ 通过才落地 + 推送
```

后端**不区分** op 来自哪个入口——只认 op 格式。因已有统一 operation 入口，加按钮入口几乎无需新后端工作。

#### 6.8.2 各入口适用场景（互补，非冗余）

| 入口 | 适合 | 走 agent reasoning？ | 速度 |
|------|------|---------------------|------|
| 聊天框 | 模糊/复合/需推理（"所有病房朝南并加隔离间"） | 是 | 慢（LLM解析） |
| 工作区按钮（直接操作类） | 明确/原子/高频（添加/删除/撤销/对齐） | 否（直接发op） | 快（毫秒） |
| 工作区按钮（触发agent类） | 一键启动agent（自动优化/检查合规/生成方案） | 是 | 秒级，有reasoning |
| 画布拖拽 | 精确空间调整（拖墙/改房间） | 否 | 快 |

**关键洞察**：不是所有操作都该走 agent。"删除房间"点按钮即可，非要打字让 LLM 理解一遍又慢又可能出错。**明确操作走按钮，模糊意图走聊天。**

#### 6.8.3 工作区按钮分两类

```
工作区按钮
├─ 直接操作类 → 发 op → 统一入口(快，无 agent)
│   [添加房间][删除][撤销][重做][对齐][复制]
│
└─ 触发 agent 类 → 启动 agent 流程(有 reasoning)
    [自动优化布局][检查合规性][生成方案][重新平衡]
```

**触发 agent 类按钮的价值**：让用户不必学"怎么对 agent 说话"就能用上 agent 能力——一个 [检查合规性] 按钮比让用户想措辞友好得多。它只是用按钮代替打字来启动已有的 agent API。

#### 6.8.4 API 形态：一个流式 API，事件类型化

生成 + reasoning 包装成**流式 API**（SSE/WebSocket），边跑边推类型化事件。LangGraph 原生支持流式（`.stream()`/`astream_events()`），所选模板（KirtiJha）已带。

```
POST /api/generate  (返回 SSE/WebSocket 流)

流中类型化事件（一个流，两类目的地）:
  对话类(→聊天区):
    {type:"reasoning", content:"正在规划科室布局..."}
    {type:"message",   content:"已生成，含12个房间"}
  行动类(→画布):
    {type:"plan_draft",  payload:{...}}
    {type:"highlight",   payload:{room_ids:[3]}}
    {type:"plan_final",  payload:{完整平面}}
    {type:"interrupt",   payload:{ask:"这版可以吗?"}}
```

前端按 `type` 分发：对话类进聊天气泡，行动类进画布。这就是"双输出"（chat + canvas）的 API 实现。

#### 6.8.5 reasoning 分层（不全部原样给用户）

| reasoning 类型 | 给前端？ | 去向 |
|---------------|---------|------|
| 面向用户的进度/解释 | ✅ | 聊天区（"正在调整洁污动线"） |
| 内部机器细节 | ❌ | LangSmith trace（tool原始参数/JSON，给你debug） |

给用户看**精炼过的 reasoning**（如 Claude 的思考折叠区），不是原始 agent 日志。

#### 6.8.6 交互范式定位

本系统**不是纯 chatbot，而是"对话式设计工具"（conversational design tool）= chat + canvas 双维度**：

| 维度 | 纯 chatbot | 本系统 |
|------|-----------|--------|
| 对话维度（说） | ✅ | ✅（一样） |
| 行动维度（做：改数据/动画布） | ❌ | ✅ |

同类产品：Cursor（chat + 代码区）、v0（chat + 预览区）、Claude artifacts。**聊天是入口之一，工作区是产出，三入口（聊天/按钮/拖拽）通向同一套 operation + 校验。**
