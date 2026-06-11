# 统一校验 CRUD 入口

> Healthcare Floor Plan Agent 开发计划（拆分版）｜ [← 总览与文档地图](README.md)

> 用户与 agent 都能增删改查，且**走同一个带校验的入口**。

### 4.1 两层 CRUD

```
第一层：原子 CRUD（底层，确定性）
   直接增删改查数据 —— "把 wall_3 坐标改成 X"

第二层：语义化 CRUD（上层，带约束）
   "把所有病房朝南" —— 翻译成一串第一层操作 → 几何+规范校验 → 才落地
```

### 4.2 统一操作原语（operation）

前端拖拽和 agent 自然语言，最终都产出同一种 op，进同一个入口：

```python
{"op": "move_wall",   "wall_id": "w1", "delta": [0, 1000]}
{"op": "resize_room", "room_id": "r1", "target_area": 15000000}
{"op": "add_door",    "wall_id": "w2", "position": 0.5, "width": 1200}
{"op": "delete_room", "room_id": "r3"}
```

### 4.3 统一入口（校验只写一处）

```python
def apply_operation(plan, op, source, base_version):
    # 0. 并发检查（乐观锁）
    if base_version != plan.version:
        return Reject(reason="version_stale", current=plan)
    # 1. 执行操作得到新状态
    new_plan = execute(plan, op)
    # 2. 几何校验（穿墙/重叠/闭合）
    geo_errors = validate_geometry(new_plan)
    # 3. 规范校验（healthcare 硬红线）
    code_errors = validate_codes(new_plan)
    if geo_errors or code_errors:
        return Reject(errors=geo_errors + code_errors)
    # 4. 落地 + 广播
    new_plan.version += 1
    log_operation(op, source)        # 审计：谁改的
    broadcast_to_frontend(new_plan)  # 推送
    return Accept(new_plan)
```

**关键**：用户删房间和 agent 删房间走同一套校验，不会出现"用户要校验、agent 绕过"的漏洞。这是 bounded generation 模式——研究共识。

### 4.4 三个好处

1. **校验只写一处，谁都绕不过**（守合规红线）。
2. **一致性**：用户改和 agent 改落同一真相源、同一 version。
3. **可追溯**：每次操作记录来源（用户第3步拖的 / agent 自动加的），healthcare 审计刚需。
