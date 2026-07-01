# 直接 Colorblock 生成 + 图像模型对比 — Findings

状态:**实验完成,结论明确,但未接入管线(parked)**。工具已写好但保持实验性。
日期:2026-06-29。

本文记录三件事:① colorblock CV 解析器的重写(已落地);② 三个图像模型在 colorblock 上的对比;③ "直接生成 colorblock"(单遍,跳过真实图)的实验与结论。

---

## 1. colorblock 解析器重写(已落地)

`hfagent/tools/image_parser.py` 的 `cv_parse`。

**之前为什么"很不准"**:旧版用 `_propagate_instances`——对整个建筑 footprint 做**无界最近种子洪水填充**(scipy `distance_transform_edt`,每个像素分给最近房间种子)。后果:
- 三房交角处出现**对角中轴线三角形**(medial-axis 斜切);
- **墙被吃掉 / 房间越过缺失的邻居膨胀**;
- 内部白洞。

**修法**:`_grow_to_centerline` —— **有界半墙中轴填充**:只把距离种子 **半个墙厚以内** 的非房间像素分给最近种子。相邻房间正好在墙中线相遇,黑墙带作为**真实分隔实体保留**(满足"墙体不应该被视为间隙"),标签/门弧的小洞被周围房间回收。半墙的距离上限同时消除了吃墙、房间膨胀、交角三角形三个问题。

其余:`_region_corners` 用全分辨率 `findContours(RETR_EXTERNAL)` + `approxPolyDP`(不再用粗 8px 网格,小房间/卫生间不再被糊掉;L/U 形房间保形);先全图 `snap_axes` 再 `_rectilinearize`(顺序反了会自交成三角形)。`cv_parse` 参数 `grid_px`+`gap_fill_px` → 改为单个 `wall_px`。

**验证**:`pytest hfagent/tests/ -q` → **50 passed**;合成 roundtrip IoU mean 0.98 / min 0.975(门槛 0.85/0.75)。离线在保存的 `_cb_*.png` 上重渲染:干净矩形、真实共享黑墙、无三角形/白洞(对比旧版的斜切+空洞)。

---

## 2. 图像模型对比(本机 key 可用)

可用图像模型:`gemini-2.5-flash-image`(原版 Nano Banana)、`gemini-3-pro-image`(Nano Banana Pro,即俗称"Nano Banana 2")、`gemini-3.1-flash-image`、`imagen-4.0-{fast,,ultra}`。

- **Imagen** 是纯文生图、不吃输入图,**做不了两遍转换的 pass-2**(需要编辑真实图);只对"直接生成"理论可用,但走的是不同 API(`generate_images`),本轮未测。
- `llm.py` 的 `IMAGE_MODEL_PREFERENCE` 里写的 `gemini-3.1-pro-image` / `gemini-3-flash-image` **本机不存在**,自动解析落到 `gemini-3-pro-image`。`gemini-3.1-flash-image` 不在偏好表里,自动解析永远选不到它。

### 2a. 两遍转换(real plan → to_colorblock)房间数误差(L1,越小越好)

eval:`hfagent/out/eval/20260629-182759-model-compare/`

| program | req | 3.1-flash | 3-pro |
|---|---|---|---|
| clinic-small | 6 | 4 | 4 |
| ward-wing | 8 | 2 | **0** |
| health-center-large | 15 | 4 | 4 |
| hospital-floor | 26 | 8 | **2** |
| outpatient-dept | 18 | **4** | 8 |
| inpatient-ward | 33 | **2** | 15 |
| clinic-mixed | 8 | 1 | 1 |
| hospital-tower-floor | 82 | 37 | 29 |

房间数:大致打平。**真正决定质量的是几何干净度**(解析器修不了的那部分):
- **3.1-flash 几何更干净**:满分辨率、色准、清晰矩形。
- **3-pro 因为死板执行"门弧填色"指令,把每个门画成走廊色 blob**,走廊变"毛毛虫"、护士站长尖刺 → 几何破碎(inpatient-ward 碎成 48 块 vs 3.1-flash 35 块)。

➡ **两遍转换路径下:3.1-flash 胜**(几何干净度,房间数可由下游 `fix_room_counts` 收敛)。

---

## 3. 直接生成 colorblock(单遍,跳过真实图)— 核心实验

动机:用户指出 flash 也不严格遵守;真实图 → colorblock 的**转换步**是门 blob、文字残留、房间数漂移的来源。**直接让模型一遍画出扁平色块图**,删掉整个转换失败面。

### 工具(已写,未接管线)
`hfagent/tools/floor_plan_generator.py`:
- `build_direct_colorblock_prompt(program, boundary=False)` —— 合并布局规则 + 色板,要求:每房一块纯色矩形;同类同色但相邻同类必须有黑墙分开成两块;统一细黑墙;铺满 footprint + 走廊脊;**不画文字、不画门/门弧、不画家具窗户**;纯白背景。
- `FloorPlanGenerator.generate_colorblock_direct()` —— 单次 `generate_image`,支持 boundary。

### 结果
eval:`hfagent/out/eval/20260629-185922-direct-colorblock/`

房间数误差(L1):

| program | req | 3-pro **直接** | 3.1-flash 直接 |
|---|---|---|---|
| clinic-small | 6 | 2 | 1 |
| ward-wing | 8 | **0** | 0 |
| health-center-large | 15 | 3 | 3 |
| hospital-floor | 26 | **4** | 11 |
| outpatient-dept | 18 | **5** | 11 |
| inpatient-ward | 33 | **8** | 13 |
| clinic-mixed | 8 | 3 | 1 |
| hospital-tower-floor | 82 | 35 | 65 |

- **`gemini-2.5-flash-image` 直接生成直接淘汰**:整类房间丢失,多个 program 输出 0/1/2 块。
- **模型反转**:两遍时 3.1-flash 赢;**直接生成时 3-pro 明显更好**(3.1-flash 从零画会过度生成,82 间画成 135 块)。原因:3-pro 更会按文字结构化指令布局,3.1-flash 更擅长图像编辑(转换)。

### 直接 vs 两遍(同为 3-pro),直接赢在三点
1. **门弧 blob 尖刺消失**——没有带门弧的真实图去转换,走廊变成干净的黄色网格(直接 inpatient-ward 对比两遍的毛毛虫)。
2. **遵守"不画文字 / 不画门"**——一遍就做到了转换时不肯做的。
3. **33 间以内房间数误差全是个位数**,且**只一次图像调用,省一半时间和成本**。

### 天花板(≥~80 间退化)
- **3-pro**:把 82 间画成**斜 X 蝴蝶形**,破坏正交(正交解析器会被搞乱);
- **3.1-flash**:保持正交但**过度碎片化**(135 块)。

这是 82 房间(1376×768 下卫生间 ~24px)的**密度上限**,两种方案都撞墙,不是直接模式独有的问题。

---

## 4. 结论与(若恢复)接管线方案

**推荐**:普通平面(≤~33–40 间)用 **直接 colorblock + `gemini-3-pro-image`**;几何更干净(无 blob)、房间数误差个位数、成本减半。

接管线时要做的(目前 **未做**):
1. 新增 `generation_mode = "direct2color"`,与 `real2color` 并列;`generate_plan` 分支调用 `generate_colorblock_direct()` 而非 `generate_real_plan()` + `to_colorblock()`。
2. **门的来源改为 `program["adjacency"]`**(直接模式没有真实图可读门),交给 `place_doors` 按房间类型挂在共享墙上——比让 VLM 读更确定。
3. 保留纠错循环(`cv_parse` → 比对房间数 → `fix_room_counts` / 重画)。
4. 更新 `hfagent/CLAUDE.md`:它当前写"不要重新引入 direct 模式",与本结论冲突,需一并修订。
5. 顺带:`build_convert_prompt`(两遍路径)里的**门弧填色**是 blob 根源,若保留两遍路径应去掉它;`IMAGE_MODEL_PREFERENCE` 把实际存在的型号(`gemini-3.1-flash-image`)纳入或调整顺序。

**未决 / 后续**:
- 82 间密度上限:考虑分块/分翼生成再拼,或接受 `fix_room_counts` 收敛。
- Imagen-4 直接生成未测(需走 `generate_images` API),理论上可能画更干净的扁平图,值得一试。
- linework 模式仍 parked(去 OCR 房型标注的 "A" 改动未做)。

### 相关产物
- 解析器:`hfagent/tools/image_parser.py`(已改)
- 直接生成工具:`hfagent/tools/floor_plan_generator.py`(`build_direct_colorblock_prompt` / `generate_colorblock_direct`,已写未接)
- eval:`20260629-181125-model-compare`(两遍,最大+最小,3 模型)、`20260629-182759-model-compare`(两遍,全 program,3.1F vs pro)、`20260629-185922-direct-colorblock`(直接,全 program,3 模型)
