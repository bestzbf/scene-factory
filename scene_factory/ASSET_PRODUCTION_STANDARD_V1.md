# 可复用室内物体资产生产规范 V1

2026-10-23 项目补充：对大量不同模型、静态整体调用、简单包围盒碰撞的生产用途，按 [静态整体资产交付补充规范](ASSET_STATIC_PROFILE_V1.md) 收敛要求。每包一套外观即可，不强制分区材质、多变体、部件拆分、精细碰撞或逐脚接触证据；文件契约、尺度/坐标与实际验收要求仍保留。

本规范用于批量生成、验收和登记可被 Blender 场景流水线自动调用的物体资产。
它定义的是**资产包契约**，不是某个具体 3D 生成模型的提示词。任何生成器（扫描、CAD、人工建模、Hunyuan3D 或其他服务）都必须在输出后转换为本规范。

规范版本：`asset_manifest_v1`，2026-10-21 定稿。适用范围：当前 Scene Factory 的静态、不透明、带 UV 的室内物体。固定接口用于减少返工；不承诺未来所有渲染器、资产类别或任务都无需适配。

**先完成真实模型验收，再填写质量状态，最后执行入库校验。** `validate` 检查字段、文件路径、哈希和兼容声明；它不会运行 Blender，也不会证明 `quality.geometry = passed` 的声明属实。不能把 JSON 校验通过当作模型视觉、碰撞或尺寸真实性已经通过。

本规范不能保证照片隐藏面、推断尺寸或照片级还原绝对正确。它使资产身份、尺度来源、坐标、材质和摆放约束明确，并由程序拦截已覆盖的不兼容情况。批量生产须同时遵守第 12–14 节的实物验收、能力边界和版本策略。

## 1. 目录和文件

每个几何变体使用一个不可变目录：

```text
asset_root/
  office_chair/
    geo_003/
      asset_manifest.json
      model.glb
      collision.glb
      reference/
        front.png
        side.png
      qa/
        technical_report.json
        visual_review.json
      textures/
        seat_base_color.png
        seat_normal.png
        seat_roughness.png
        frame_base_color.png
        frame_roughness.png
      previews/
        front.png
        side.png
        back.png
        three_quarter.png
        placement_test.png
```

规则：

- 生产格式统一为 glTF 2.0 Binary（`.glb`）。FBX、OBJ 等只能作为输入格式，入库前必须转换为 GLB 并重新验收。
- `model.glb` 必须能在无用户界面的 Blender 中导入和渲染。
- 贴图必须随资产包提供；不能依赖用户电脑上的绝对路径。
- 模型、贴图、碰撞代理和五张预览都记录 SHA-256。manifest 的哈希在构建 catalog 时计算并写入 catalog；不要在 manifest 内填写它自身的哈希。参考图和验收报告另随生产记录保存哈希。
- 文件移动或改名不能改变 `asset_id`；修改几何、UV 或材质时必须创建新变体。
- 资产包不允许包含可执行脚本、驱动表达式或需要外部插件才能运行的节点。

## 2. 坐标、单位和变换

所有资产必须遵守以下规范：

- 单位为米，`1 Blender unit = 1 m`。
- Z 轴向上。
- 语义正面统一为 `-Y`。如果原始模型正面不同，导出前完成旋转并把原始方向记录在来源字段中。
- 变换必须应用到网格：`location = (0,0,0)`、`rotation = (0,0,0)`、`scale = (1,1,1)`。
- 原点默认位于物体底部中心；不能使用默认导入原点代替锚点。
- 资产必须提供真实包围盒尺寸和尺寸来源。估计尺寸不得标记为实测。
- 场景调用只允许统一缩放；不能用 X/Y/Z 非均匀拉伸修正资产比例。

以上轴和包围盒均指**重新导入 Blender 后**的坐标；GLB 文件内部仍遵守 glTF 的 Y-up 约定，由导入/导出器转换。`dimensions_m` 顺序固定为 `[X宽, Y深, Z高]`，测量整个资产在规范姿态下的轴对齐包围盒。不得把 GLB 原始节点坐标直接当作 Blender 坐标。新生产包使用米制；`meters_per_import_unit` 保留给已明确量纲的迁移包，不能用于掩盖未知尺度。

必须同时记录：

```json
{
  "dimensions_m": [0.52, 0.55, 0.86],
  "scale_source": "measured",
  "scale_confidence": 1.0,
  "source_up_axis": "+Z",
  "source_front_axis": "+X",
  "canonical_up_axis": "+Z",
  "canonical_front_axis": "-Y"
}
```

`scale_source` 只允许：`measured`、`known_reference`、`reference_inferred`、`unknown`。`unknown` 资产不得进入生产库。

## 3. 身份和变体

五个 ID 的职责必须固定：

- `family_id`：可互换的物体族，例如 `office_chair`。
- `geometry_variant_id`：具体几何，例如 `office_chair_geo_003`。
- `material_variant_id`：一套与 UV/材质槽兼容的外观，例如 `office_chair_mat_dark_fabric`。
- `asset_id`：对外引用的不可变几何资产 ID，推荐等于 `family_id + geometry_variant_id`。
- `instance_id`：场景中的一次摆放，不属于资产库。

同一个 `family_id` 内的模型只有在类别、支撑方式和替换规则一致时才允许互换。相同网格换颜色是材质变体；改变网格、UV 或材质槽必须是新的几何变体。`asset_id` 与 `geometry_variant_id` 均全库唯一；推荐两者相同，且已经包含族名。

V1 每个资产包独立定义全库唯一的材质 ID，例如 `office_chair_geo_003__mat_dark_fabric`。跨几何复用相同风格时使用各包自己的材质 ID；不要重复定义同一个材质 ID。`compatible_geometry_ids` 只列实际已测试、且存在于本次 catalog 的几何 ID。每个几何的每套外观都必须独立做替换验收。

## 4. 几何要求

生产资产必须：

- 是完整可摆放物体，不能只包含照片中可见的半个物体；如果确实是局部模型，必须标记 `scope = partial`，解析器禁止把它当完整家具。
- 包含必要的支撑部件，例如椅腿、桌腿、底座、挂件或墙面安装件。
- 法线方向正确，不能存在未说明的反向面、零面积面或严重重叠。
- 不包含生成模型常见的地面薄片、背景平面、漂浮碎片或不可见的扫描残留。
- 边缘具有与真实物体相符的倒角或圆角；倒角不能用极端贴图高光替代。
- 为静态资产时不包含无关动画、骨骼或摄像机。
- 对组合资产提供部件树和稳定的部件名称；不能让解析器猜测哪个网格是桌面或桌腿。

正式入库必须提供两个文件：

- `render_mesh`：最终渲染网格。
- `collision_mesh`：低面数碰撞代理，供摆放和碰撞检查。

二者使用相同的单位、坐标和锚点。碰撞代理的面数按用途控制，不能直接假定渲染网格适合作碰撞体。V1 不强制所有装饰面封闭；真正用于物理模拟的碰撞代理须另按所用引擎验证封闭性、凸分解及碰撞稳定性。

## 5. 锚点和摆放信息

每个资产必须声明：

```json
{
  "placement": {
    "support_type": "floor",
    "anchor_type": "bottom_center",
    "anchor_point_m": [0.0, 0.0, 0.0],
    "contact_points_m": [[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0]],
    "mount_points": [],
    "clearance_m": 0.0,
    "can_rotate_z": true,
    "allow_uniform_scale": true,
    "allow_non_uniform_scale": false
  }
}
```

`support_type` 只允许：`floor`、`table`、`wall`、`ceiling`、`support`、`holder`、`free`。

桌面物体必须提供桌面接触面；墙挂物必须提供墙面锚点和安装方向；吊挂物必须提供挂点。没有锚点的资产不能自动调用。

`anchor_point_m`、`contact_points_m` 和 `mount_points` 都是资产规范姿态下的米制坐标。墙挂、吊挂资产不强制底部中心锚点，使用明确的安装锚点。V1 运行时会使所选锚点对齐场景中的位置；挂点、净空、支撑面和接触点仍需要上层摆放验收，填写这些字段不会自动完成挂接和避碰。

## 6. 材质、UV 和贴图

每个可见材质区域必须有稳定的材质槽名称，例如 `seat_fabric`、`metal_frame`、`wood_top`。

材质变体必须声明兼容的几何变体：

```json
{
  "material_variant_id": "office_chair_mat_dark_fabric",
  "compatible_geometry_ids": ["office_chair_geo_003"],
  "uv_layout_id": "office_chair_uv_003",
  "normal_convention": "OpenGL_Y+",
  "maps": {
    "base_color": {"path": "textures/seat_base_color.png", "color_space": "sRGB", "sha256": "..."},
    "normal": {"path": "textures/seat_normal.png", "color_space": "Non-Color", "sha256": "..."},
    "roughness": {"path": "textures/seat_roughness.png", "color_space": "Non-Color", "sha256": "..."},
    "metallic": {"path": null, "color_space": "Non-Color", "sha256": null}
  }
}
```

规则：

- Base Color 使用 sRGB；Normal、Roughness、Metallic、AO 使用 Non-Color。
- 法线统一使用 MikkTSpace 和 OpenGL `+Y` 约定；未知法线约定不得入库。
- 贴图不能把灯光阴影、强高光或环境反射烘焙进 Base Color。
- 同一几何变体的材质变体必须保持 UV 通道数、UV 命名、材质槽语义和切线空间一致。
- 只有颜色变化而没有纹理变化的派生版本，必须标记为参数化材质变体，不能伪装成新几何。
- 当前导入器支持 `base_color`、`normal`、`roughness`、`metallic` 和 `ao`。`ao` 只保留为节点资料，不自动乘入 Base Color。标量图使用独立灰度图；ORM 等打包图须先按通道拆分，当前接口没有通道选择器。
- 当前 V1 正式调用限不透明物体，包括普通金属、木材、塑料和织物。透明/半透明、玻璃、发光和位移材质需要专门的适配与验收；`opacity`、`transmission`、`emissive`、`height` 贴图不能进入当前正式 catalog。保存这些资产的源文件及 IOR、Transmission、厚度等制作资料，待支持后重新转换和验收，不能把它们当普通不透明材质通过。
- 当前不支持仅用数值定义整套材质；Base Color 必须有文件。纯色可使用小尺寸纯色贴图，固定 Roughness/Metallic 可使用单值灰度图。正常生产建议为每个可见区域显式提供 Roughness 和 Metallic，避免依赖渲染器默认值。
- 多材质区域使用 `slot_materials`，键必须等于 GLB 导入时的原始材质名，且包内语义唯一。没有默认 `maps` 时必须覆盖所有实际槽；有默认 `maps` 时，未单独覆盖的槽明确使用默认贴图。不能仅在描述中声称桌面和金属腿有不同材质。
- V1 替换贴图使用第一套 UV。额外 UV 可保留，但生产外观必须在第一套 UV 下测试，贴图变换需烘焙到 UV；不能依赖未实现的 UDIM、其他 UV 通道选择或 glTF 材质扩展。

## 7. 语义字段

manifest 至少包含：

```json
{
  "semantics": {
    "category": "chair",
    "description": "带靠背的办公椅",
    "tags": ["office", "fabric", "dark_gray"],
    "scope": "single_prop",
    "completeness": "complete",
    "confidence": 0.94,
    "source": "human_verified"
  }
}
```

`category`、`support_type`、`scope` 和 `completeness` 是解析器使用的字段，不能只写在自然语言描述里。GLM 可以补充标签，但关键字段必须有来源和置信度。

## 8. 质量状态和入库门禁

状态只允许按以下方向推进：

```text
draft -> technical_checked -> asset_accepted -> deprecated
                    \-> needs_repair -> technical_checked
```

`asset_accepted` 必须同时满足：

1. 文件路径、模型哈希和所有贴图哈希正确；
2. Blender 无界面导入成功；
3. 尺寸、坐标、正面、原点和锚点完整；
4. 网格、法线、UV、材质节点和贴图色彩空间通过检查；
5. 无明显地面片、碎片、漂浮部件或缺失支撑结构；
6. 正面、侧面、背面、三分之四视图通过单体检查；
7. 在至少一种典型支撑面上摆放、碰撞和渲染通过；
8. 许可证、生成来源、模型版本和输入摘要完整；
9. 所有兼容的材质变体都通过材质替换测试。

场景原图对比属于 `scene_accepted`，不作为资产晋级的前置条件。资产合格只表示可以安全复用；具体照片是否应该调用它由场景检索和场景验收决定。

## 9. 结构完整的 manifest 示例

可复制的初始文件见 [asset_manifest.template.json](asset_manifest.template.json)，它故意保留未填哈希和未知尺度，默认不能通过验收。

下面示例展示单材质资产。`...` 是由生产程序填入的真实 SHA-256，不是可直接验收的值。实际入库前必须替换所有占位哈希，并确保列出的模型、贴图、预览和来源图片都存在于资产包内。初次生成时用 `status: draft`、`production_ready: false`；示例中的 `passed` 和 `asset_accepted` 只能在真实检查完成后填写。

```json
{
  "schema_version": "asset_manifest_v1",
  "asset_id": "office_chair_geo_003",
  "family_id": "office_chair",
  "geometry_variant_id": "office_chair_geo_003",
  "status": "asset_accepted",
  "model": {
    "format": "glb",
    "path": "model.glb",
    "sha256": "...",
    "dimensions_import_units": [0.52, 0.55, 0.86],
    "dimensions_m": [0.52, 0.55, 0.86],
    "meters_per_import_unit": 1.0,
    "scale_source": "measured",
    "scale_confidence": 1.0,
    "canonical_up_axis": "+Z",
    "canonical_front_axis": "-Y",
    "triangle_count": 42000,
    "object_count": 1,
    "transform": {
      "location": [0, 0, 0],
      "rotation": [0, 0, 0],
      "scale": [1, 1, 1]
    }
  },
  "placement": {
    "support_type": "floor",
    "anchor_type": "bottom_center",
    "anchor_point_m": [0.0, 0.0, 0.0],
    "contact_points_m": [[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0]],
    "mount_points": [],
    "clearance_m": 0.0,
    "can_rotate_z": true,
    "collision_path": "collision.glb",
    "collision_sha256": "...",
    "allow_uniform_scale": true,
    "allow_non_uniform_scale": false
  },
  "semantics": {
    "category": "chair",
    "description": "带靠背的深色织物办公椅",
    "tags": ["office", "fabric", "dark_gray"],
    "scope": "single_prop",
    "completeness": "complete",
    "confidence": 0.94,
    "source": "human_verified"
  },
  "uv": {
    "layout_id": "office_chair_uv_003",
    "channels": 1,
    "normal_convention": "OpenGL_Y+",
    "texel_density_policy": "family_normalized"
  },
  "material_variant_ids": ["office_chair_mat_dark_fabric"],
  "materials": {
    "office_chair_mat_dark_fabric": {
      "material_variant_id": "office_chair_mat_dark_fabric",
      "compatible_geometry_ids": ["office_chair_geo_003"],
      "uv_layout_id": "office_chair_uv_003",
      "normal_convention": "OpenGL_Y+",
      "maps": {
        "base_color": {"path": "textures/seat_base_color.png", "color_space": "sRGB", "sha256": "..."},
        "normal": {"path": "textures/seat_normal.png", "color_space": "Non-Color", "sha256": "..."},
        "roughness": {"path": "textures/seat_roughness.png", "color_space": "Non-Color", "sha256": "..."}
      }
    }
  },
  "previews": {
    "front": {"path": "previews/front.png", "sha256": "..."},
    "side": {"path": "previews/side.png", "sha256": "..."},
    "back": {"path": "previews/back.png", "sha256": "..."},
    "three_quarter": {"path": "previews/three_quarter.png", "sha256": "..."},
    "placement_test": {"path": "previews/placement_test.png", "sha256": "..."}
  },
  "quality": {
    "file_integrity": "passed",
    "import": "passed",
    "geometry": "passed",
    "uv": "passed",
    "materials": "passed",
    "previews": "passed",
    "placement_test": "passed",
    "production_ready": true
  },
  "provenance": {
    "source_images": ["reference/front.png", "reference/side.png"],
    "generator": "hunyuan3d",
    "generator_version": "recorded_version",
    "generation_parameters_sha256": "...",
    "license": "internal",
    "created_at": "2026-10-21T00:00:00Z"
  }
}
```

## 10. 批量生产的强制规则

- 同一个生成请求使用唯一 `request_key`；相同输入和参数不得重复生成。
- 每个候选资产先写入临时目录，通过验收后再登记到正式目录。
- 正式资产不可原地覆盖；修改模型后创建新的 `geometry_variant_id` 或版本 ID。
- 失败、待审和尺度未知的资产可以保留，但解析器默认禁止调用。
- 同一场景多视角必须固定 `instance_id`、几何变体和材质变体。
- 记录每个资产的生成耗时、重试次数、验收结果和失败原因。
- 资产目录、模型、贴图和 manifest 的哈希必须写入场景输出，保证场景可复现。

## 11. 与当前 Scene Factory 的兼容说明

当前 `asset_catalog.py` 已实现 `asset_id`、`family_id`、`geometry_variant_id`、模型哈希、尺寸、单位、正面、支撑类型、状态和材质兼容关系的基本门禁。

`asset_manifest.py` 已实现 manifest 校验和 catalog 转换：

```bash
python3 -m scene_factory.asset_manifest validate /path/to/asset_root/family/geo_001 --require-production
python3 -m scene_factory.asset_manifest build-catalog /path/to/asset_root --output /path/to/asset_catalog.json
```

转换是 fail-closed 的：任意资产未通过 `asset_accepted`，整个目录构建失败。生产状态只允许 GLB；`.gltf` 可以用于技术检查，但不能进入生产 catalog。`external_assets.py` 已支持 GLB/glTF/FBX 导入、多个材质槽、显式锚点和哈希检查，并在实例化时复制网格数据以隔离材质变体；它还会拒绝模型中不存在的声明材质槽。碰撞代理当前由路径和哈希门禁保护，碰撞网格的导入、接触和穿透仍需 Blender 场景验收。场景级照片对比和 GLM 修正仍属于上层场景质量流程。

`validate` 只检查一个资产包或一个 manifest；`build-catalog` 才会递归检查全库。catalog 输出必须位于资产根目录以外。正式根目录只放准备发布的合格包，draft/失败包留在独立候选目录。构建失败返回非零退出码并保留已有 catalog；批处理必须检查退出码，不能把旧 catalog 当作新批次结果。包内路径可迁移；当前 catalog 的 `root` 为绝对路径，移动资产根目录后重新构建 catalog。

`--no-hash` 仅用于结构调试，会标记文件未验证、`production_ready: false`；不能与 `--require-production` 一起用于生产验收。

## 12. 批量生产交付清单

每个物体至少交付以下内容。表中的“报告”是资产生产环节需要生成和归档的证据，不表示 manifest 校验器会替你执行这些测量。

| 信息 | 固定位置或字段 | 取得方式与约束 |
|---|---|---|
| 版本和身份 | `schema_version`, `asset_id`, `family_id`, `geometry_variant_id` | 生产系统生成，ID 不得由检索模型临时改写 |
| 网格文件 | `model.format/path/sha256` | GLB，文件哈希由程序计算；保留生成器原始输出以便重导出 |
| 尺度 | `dimensions_import_units`, `dimensions_m`, `meters_per_import_unit`, `scale_source`, `scale_confidence` | Blender 实测模型包围盒；物理尺度另注明测量、已知参照或推断来源 |
| 坐标与变换 | `canonical_up_axis/front_axis`, `model.transform` | 重新导入后复核，不能只检查导出前的场景 |
| 网格统计 | `triangle_count`, `object_count` | 从最终 GLB 重新测量，不能使用生成前统计 |
| 摆放 | `placement` 全部字段 | 锚点、接触点、安装点、净空、旋转及缩放策略；墙挂/吊挂另留安装说明 |
| 碰撞代理 | `collision_path/collision_sha256` | 与渲染模型同坐标；碰撞可用性另外测试 |
| 语义 | `semantics.category/description/tags/scope/completeness/confidence/source` | 类别、完整性和整体/组合范围明确；模型推断必须标明来源 |
| 部件 | `parts` | 组合资产必填：至少 `part_id`, `name`, `parent_id`；另保存对象对应关系，部件图不得成环 |
| UV | `uv.layout_id/channels/normal_convention/texel_density_policy` | 实测 UV 存在性；UV 布局 ID 不可因换颜色而改名 |
| 外观变体 | `material_variant_ids`, `materials` | 每个变体包括兼容几何、UV ID、法线约定及默认/分槽贴图 |
| 贴图 | `path/color_space/sha256` | 每张图跟包保存；不得只写材质名称或一张带光照的预览 |
| 预览 | `previews` 五个固定视图 | front、side、back、three_quarter、placement_test，图像可解码且哈希正确 |
| 状态 | `status`, `quality` | 由生产质量程序写入；GLM 分数只能作为输入，不能独立签发 accepted |
| 来源 | `provenance` | 参考图、生成器及版本、参数摘要哈希、许可证标识、创建时间 |
| 验收证据 | `qa/technical_report.json`, `qa/visual_review.json` | 保留每个检查的指标、阈值、结论、失败原因、工具/模型版本、时间和所验文件哈希 |

质量报告至少绑定 `asset_id`、`geometry_variant_id`、模型和贴图哈希、Blender 版本、验收脚本版本及被测试的全部 `material_variant_id`。模型/贴图/UV 变更后旧报告失效。许可证字段的非空校验不等于权利审核；另保留来源条款或权利依据。

生产程序还应归档生成参数的原始 JSON、参考图原文件、提示词版本、生成 seed、生成器任务 ID、每次修复记录。密钥不写入资产包或参数摘要。`generation_parameters_sha256` 应能回溯到实际归档参数，而非随意填写 64 位字符。

多槽写法示例：每个槽内仍使用 `base_color`、`normal`、`roughness`、`metallic`、`ao` 的贴图记录。示例只示意结构，省略的哈希和其他字段须补齐。

```json
{
  "slot_materials": {
    "seat_fabric": {
      "base_color": {"path": "textures/seat_base_color.png", "color_space": "sRGB", "sha256": "..."},
      "roughness": {"path": "textures/seat_roughness.png", "color_space": "Non-Color", "sha256": "..."}
    },
    "metal_frame": {
      "base_color": {"path": "textures/frame_base_color.png", "color_space": "sRGB", "sha256": "..."},
      "metallic": {"path": "textures/frame_metallic.png", "color_space": "Non-Color", "sha256": "..."},
      "roughness": {"path": "textures/frame_roughness.png", "color_space": "Non-Color", "sha256": "..."}
    }
  }
}
```

`parts` 在 V1 是描述资料，导入器仍将资产作为整体实例化；需要独立摆放的桌、椅、杯子应分别生产资产包，而不是期待导入器自动拆件。抽屉、柜门等可动部件需要未来的关节接口，V1 不负责动作和物理交互。

## 13. 实际能力与验收责任

| 检查 | 当前实现 | 生产时还要执行 |
|---|---|---|
| JSON、ID、路径、哈希 | manifest 校验、catalog 构建和运行时文件校验 | 归档校验日志，失败即停止该批次发布 |
| Blender 可导入性 | 真正实例化时检查静态网格、UV 和声明材质槽 | 每个新包先执行无界面导入、全部外观替换和试渲染 |
| 尺度、坐标、锚点 | 校验声明格式、尺寸换算；严格 manifest 资产导入后比对实际包围盒（相对容差 1e-4、绝对容差 1e-5 导入单位），按锚点实例化 | 复核语义正面和锚点正确性；实测物理尺寸不能由图片置信度替代 |
| 材质正确性 | 支持上述不透明贴图；同名槽与不同实例外观隔离 | 多种光照下检查拉伸、接缝、烘焙阴影、金属和粗糙度 |
| 碰撞和支撑 | 代理文件路径和哈希；记录接触/安装数据 | 使用真实代理做接触、穿透、净空测试；当前未自动执行 |
| 语义和完整性 | 检查 complete/partial 声明 | 多视角检查背面、支撑部件、残留平面与漂浮碎片 |
| 自动选择 | 按族、状态、哈希、材质兼容与种子选变体 | 视觉匹配、目标尺寸/支撑匹配仍需上层检索；绑定表目前由外部提供 |
| 与参考照片一致 | 场景流程可以接模型进行原图/渲染审核 | 在具体场景上检查，不是资产一次合格就对所有照片合格 |

当前流程没有统一执行所有物体的几何、碰撞和视觉检测；`quality` 是上游验收结果的记录，不能通过填写 `passed` 替代实际检查。没有验收器的检查项保持未通过，不发布为正式资产。可先留候选包等待补验收，原始网格与贴图仍可保留复用。

## 14. 放量验收与减少返工

1. 先按将要批量使用的**同一生成器、导出器和验收程序**制作 20–30 个代表样本；覆盖桌椅、床柜、桌面小物、组合物及多材质区域。每种将要生产的类型至少有样本；未覆盖的特殊材质不随普通家具一同放量。
2. 完成全部单包检查、catalog 构建、重复导入、变体替换、典型摆放和实际场景渲染。每个几何至少两套材质，每个族至少两个几何，并验证相同物体的跨视角绑定固定。
3. 小批结果达到你的效果要求后，冻结生成器/导出器/Blender/验收脚本版本及类目表、尺度策略、面数与纹理预算。通过样本只代表该版本和已覆盖类别，不能外推到所有材质和所有用途。
4. 放量时每个包继续跑技术门禁；每个新几何和材质组合都留多视角视觉验收记录，另抽样检查已验收结果。记录原因码，失败候选可重试或进入修复队列，不能默认通过。
5. 升级任一关键版本先跑同一组代表样本；不原地覆盖旧资产、旧 manifest 或旧 catalog。新增不兼容需求使用新 schema/适配器版本，并保留迁移规则。

面数、贴图分辨率、误差阈值及场景内存预算需按最终输出分辨率和单场景物体数量冻结；不存在适用于所有室内场景的万能数值。它们写进 `qa` 报告的质量配置，**本 V1 校验器当前不强制这些数值**。例如照片还原与机器人仿真所需的碰撞、尺度和拓扑标准不同，后者须另建验收配置。

减少返工的关键是保留可重导出的源资产、独立的 PBR 图、UV、尺度依据和检测证据，并在放量前跑通代表样本。任何文字规范或一次单元测试都不能保证以后零问题。
