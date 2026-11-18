# 文字生成场景（Text-authored scenes）

从一段文字描述直接生成可编辑的 Blender 场景，不需要照片。

这是照片流水线的**兄弟通路**，不是它的一个模式。两者共用同一套资产库、几何构建器、渲染器和哈希绑定的溯源；区别只在三处：输入是文字不是图片、验收看「是否合理」而不是「是否像照片」、结论标记为 `text_scene_built` 而不是 `accepted`（文字场景永远不是照片重建的声明）。

## 快速开始

```bash
# 1. 只用文字，一条命令出图
python -m scene_factory scenery \
  --brief "一间 5x4.2 米的北欧风客厅，灰色布艺沙发正对低矮电视柜，中间放一张木色茶几，地毯铺在沙发前，角落一盏落地灯。" \
  --config configs/weidian_gateway_gpt61sol.json \
  --catalog /path/to/production_library_20261032/library/asset_catalog.json \
  --output-dir output_living_room

# 2. 先看模型会收到什么提示词与 schema，不联网
python -m scene_factory scenery --brief "..." --output-dir /tmp/preview --dry-run

# 3. 手写布局直接构建（不调模型，零成本）
python -m scene_factory scenery \
  --layout my_layout.json --output-dir output_handmade --max-edge 1600 --samples 128
```

产物结构：

```
output_living_room/
  initial_scene.json          模型原始布局
  initial_scene_retry_1.json  定向修正后的布局（如触发）
  scene_layout.json           最终采用的布局
  logic_report.json           合理性检查报告
  result.json                 总结果（状态、映射表、是否可信）
  render/
    scene_bundle.json         V3 场景包（可复现的唯一输入）
    references/               每视角的中性占位图（标注"非照片"）
    scene.blend               可编辑 Blender 工程
    rgb_view_00.png ...       逐视角渲染
    annotations_view_00.json  逐视角相机内外参 + 物体标注
    metadata.json             渲染元数据
```

## 设计约束与取舍

### 为什么不复用 `no_review`

照片流水的 `no_review=true` 会让产出无法进入数据集（见 `dataset_export.py:128`）。文字场景不碰这条约束：它不是照片重建，本来就进不了照片数据集，所以它自带一套验收——`scene_logic` 的合理性检查——并明确标注 `quality_status: needs_human_review`。这避免了照片流水线里「强制 no_review」与「导出要求 no_review=false」的死锁重演。

### 为什么放宽约束是对的

照片流水线是 fail-closed 的：任何一个物体配不上资产，整场景判死。这对重建是正确的——缺一个冰箱就是测量失败。

对文字场景则是错的：场景是设计出来的，资产库没有某件东西只是渲染细节。所以这里改成逐物体降级，并把降级记录下来（`result.json` 的 `category_mapping.unmapped`）。绝不会有物体被静默替换成别的东西。

### 合理性检查做了什么

`scene_logic.validate_scene_logic` 是确定性的、不调模型，共 11 类检查：

| 类别 | 例子 |
| --- | --- |
| 房间比例与密度 | 2.7m 高、21m² 放 30 件家具 → overcrowded |
| 按类别的高度 | 3 米高的椅子 → `object.height_implausible` |
| 是否出墙 | 沙发探出墙 0.2m → blocking |
| 互相穿插 | 两件家具重叠 >10cm → blocking（地毯豁免） |
| 通行宽度 | 两件大件之间 <0.55m → `circulation.narrow_gap` |
| 靠墙习惯 | 衣柜离墙 1.2m → `placement.not_near_wall` |
| 座椅朝向 | 沙发背对电视 → `seating.faces_nothing` |
| 支撑关系 | 子物体比支撑物宽 → `support.child_larger_than_parent` |
| 门洞与洞口 | 椅子挡在门前 0.5m → blocking |
| 相机可用性 | 相机在衣柜内部 / 准星埋在家具里 |
| 位置重合 | 两件落地物同坐标（支撑对豁免） |

检查结果按 `blocking / major / minor` 分级。有 blocking 或 major 时，管线会自动发一次**定向修正**请求——只把最严重的几条规则连同原布局发给模型，要求只改被点名的部分。

### 程序化美化

三件不需要模型的事，管线自动做，并全部记录在案：

1. **类别映射**：把 `coffee_table` → `table`、`bookshelf` → `shelf`、`office_chair` → `chair`，让渲染器用真实的家具几何（木纹桌面、织物坐垫、金属桌腿）而不是灰盒子。映射表见 `text_scene.NATIVE_KINDS`。
2. **逐物体配色**：从文字里按**从句**取颜色词（"灰色沙发"只染沙发，不会把整间屋子染灰），没有颜色词就按类别给合理材质色。
3. **灯光归一化**：`lighting.energy` 是房间总面积光预算，模型给的值往往偏离一到五倍（900W 的 21m² 客厅会渲染成纯白）。管线按 8.5 W/m² 标定，偏离两倍以内视为刻意明暗并保留比例，超出则视为单位错误直接校正。

## 已验证与未验证

**已验证**（本机实测，Blender 5.2.0 LTS）：

- 31 项单元测试通过；全量 1006 项通过、123 项跳过，无回归
- 客厅场景：2 视角、37 网格、沙发与茶几走真实家具构建器
- 书房场景：76 网格，`basic` 回退为 0，全部家具为真实几何
- 渲染成功、逻辑检查零问题、`text_scene_built` 状态正确

**未验证**：

- **模型端到端**：本机未配置 `WEIDIAN_AI_API_KEY`，所以「文字 → 模型 → 布局 → 渲染」这条完整链路只验证了除模型调用外的全部环节。模型调用层本身（`model_api.call_model`）是被 1006 项测试覆盖的既有代码，文字模式只是传空图片列表复用它。
- **外观真实感**：目前是「几何正确 + 配色合理」，没有贴图、没有软装细节（窗帘、挂画、书籍）。比白盒子好得多，但离照片级还有距离。
- **失败时的语义修正效果**：`--revisions` 的定向修正逻辑已实现且可测，但修正后的实际改善幅度没有真实模型样本。

## 与照片流水线的关系

| | 照片流水线 | 文字场景 |
| --- | --- | --- |
| 输入 | 1–32 张照片 | 一段文字 |
| 模型角色 | 从照片推断几何 | 按设计意图设计布局 |
| 验收 | 六维视觉评分 ≥85 / realism ≥90 | 11 类确定性合理性规则 |
| 失败策略 | fail-closed，整场景判死 | 逐物体降级并记录 |
| 资产缺失 | 致命 | 保留基元几何，标注 unlisted |
| 结论标记 | `accepted` / `needs_review` / … | `text_scene_built` / `text_scene_implausible` |
| 数据集导出 | 可进入 `dataset_export` | 不进入（非重建） |

两者共用 `expand → select_and_place → generate_multiview.py` 这条构建链，所以文字场景的 `.blend`、标注、相机矩阵与照片场景格式完全一致，下游工具无需改动。
