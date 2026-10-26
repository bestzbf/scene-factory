# 资产库生产与场景调用规范（设计契约）

批量生产资产时，优先遵循 [ASSET_PRODUCTION_STANDARD_V1.md](ASSET_PRODUCTION_STANDARD_V1.md)。该文件是面向资产生产程序的固定输入/输出契约；本文保留当前 `asset_catalog_v1` 运行时的兼容说明。

状态映射：manifest 通过 `asset_accepted` 门禁后，`build-catalog` 映射为运行时的 `approved`。生产包统一使用 GLB；多材质槽、显式锚点和碰撞代理会进入 catalog。导入器校验材质槽与锚点，碰撞代理目前执行路径和哈希校验；场景有部分 AABB/射线近似检查，完整网格碰撞与接触审计尚未实现。

本文定义资产批量生产与调用的运行时契约。当前已实现 manifest 校验、GLB/glTF/FBX 兼容导入、目录转换、兼容变体解析、多材质槽和哈希门禁。导入器会阻断 manifest 声明但模型中不存在的材质槽。碰撞文件目前由路径和 SHA-256 门禁保护；完整碰撞网格导入、接触模拟和穿透测试仍属于 Blender 场景验收，不应把文件哈希通过理解为碰撞模拟通过。GLM 自动候选检索、专用 3D 生成服务调度和场景级视觉修正属于上层批处理流程，不由资产目录猜测或替代。运行时使用 `asset_catalog_v1`，字段以 `asset_catalog.py` 为准。

### 已验证的基础接口

正式生产入口是 `asset_manifest_v1` -> `asset_catalog_v1`；GLB 是正式格式。旧的 FBX/glTF draft 转换器只保留给预览和迁移，不能把它的输出直接当作正式资产库。

```bash
python3 -m scene_factory.asset_manifest validate /absolute/path/to/asset_root/family/geo_001 --require-production
python3 -m scene_factory.asset_manifest build-catalog \
  /absolute/path/to/asset_root --output /absolute/path/to/asset_catalog.json
```

旧入口 `python3 -m scene_factory.asset_catalog --audit-draft ...` 只用于迁移或预览；它不替代 manifest 门禁。

`generate_multiview.py` 和 `batch_multiview.py` 增加 `--asset-catalog`、`--asset-bindings`。绑定表为
`{"schema_version":1,"instances":[{"object_id":"已有展开物体ID","asset_id":"目录ID","material_variant_id":"兼容材质ID","uniform_scale":1.0,"rotation_xyz_deg":[0,0,0]}]}`。
位置沿用场景对象，尺寸只按目录米制比例和等比缩放计算，禁止用 XYZ 拉伸去适配。无合格绑定的物体仍使用原有内置建模。

手动配置仍支持 `asset_library: {"catalog":"目录路径","bindings":"绑定表路径","preview":false}`。批量自动调用改用 `auto_bind=true`，不提供 `bindings`；固定选择器从 GLM 输出的共享场景中读取类别、尺寸和朝向，为每个场景、每轮返修生成独立绑定。生产导入器支持通过 manifest 门禁的 GLB、带 UV 的静态网格、多个材质槽、显式锚点和统一缩放。旧版 FBX/glTF 仅能进入 preview。

在 `factory.json` 增加以下字段，目录路径相对于该配置文件解析：

完整 GLM 模板为 [bigmodel_auto_assets.json](config_examples/bigmodel_auto_assets.json)。把它复制为项目的 `factory.json`，并将通过生产门禁构建的目录放到配置同级 `assets/asset_catalog.json`，或修改 `catalog` 路径。模板沿用 GLM 的分析、审核和修正三个角色；不含密钥。`category_map` 默认为空；若资产类别使用物体族名而非场景规范类别，应按下例显式映射。

```json
{
  "asset_library": {
    "catalog": "assets/asset_catalog.json",
    "auto_bind": true,
    "preview": false,
    "selection": {
      "seed": 0,
      "dimension_tolerance": 0.15,
      "min_uniform_scale": 0.8,
      "max_uniform_scale": 1.25,
      "floor_tolerance_m": 0.0001,
      "diversity": false,
      "category_map": {
        "chair": {"families": ["office_chair", "dining_chair"]}
      }
    }
  }
}
```

默认要求 catalog 类别与场景类别完全一致；上例显式映射允许 `chair` 使用两个物体族。尺寸容差是等比缩放后三轴的最大相对误差；宽深不会排序后互换，禁止非均匀拉伸。`diversity=false` 优先尺寸拟合，再用固定种子处理同等候选；增强数据时才使用 `diversity=true` 优先未用组合。当前没有照片外观检索，种子并不能使材质更像照片。

`floor_tolerance_m` 默认及上限都是 `0.0001` 米，只容忍落地 Z 值的浮点误差，不会自动把悬空物体吸附到地面。桌、台、架、柜和箱体还要求替换后包围盒高度与原目标相差不超过 `5 mm`，避免已摆好的小物仍停在旧高度。这是保守筛选，不能证明顶面可接触、桌腿全部落地或没有穿插；完整支撑图与接触审计仍待实现。

自动替换范围为可确认落地的 base objects。桌面/墙挂/吊挂及 custom 构造暂时保留程序建模。每轮输出 `asset_bindings.json` 和 `asset_selection.json`，记录绑定、比例误差、淘汰与回退原因；两张照片共用一份绑定。返修改尺寸/类别/ID 后重新选择，不保留旧实例。审核与修正 API 收到绑定及拟合证据，能区分外部资产的材质限制与可修改的程序参数。全部候选模型、材质、碰撞文件及 manifest 的内容哈希进入恢复契约。

当前会对整个候选目录做内容校验，适合先验证正确性；大库会增加启动与逐场景校验开销。扩大规模时应使用不可变资产版本、目录快照和校验索引，避免每个场景重复读取全部贴图；不能通过关闭哈希检查来优化速度。

不使用配置时，分组 runner 支持 `--asset-catalog CATALOG --asset-auto-bind`；直接 Blender renderer 仍需要解析好的 `--asset-bindings`。详细参考流程与尚未实现的能力见 [HARMONY_WORKFLOW.md](HARMONY_WORKFLOW.md)。

启用后仍从统一入口运行：`python -m scene_factory run --config /absolute/project/factory.json --input-dir /absolute/project/photos --output-dir /absolute/project/output`。独立照片各生成一个场景；同一场景多视角放在同一个一级子目录，自动推断相机，无需填写拍摄方向。照片是否属于同一场景仍由目录分组决定，不能混放不同空间。

使用 `--asset-preview` 可以检查草案资产，但未知尺度必须显式提供 `preview_height_m`；这是预览估计，不是实测。preview 场景被禁止进入 accepted。生产调用要求资产通过目录门禁；法线约定未知时也会拒绝正式实例化。模型、贴图、目录、绑定表内容哈希纳入缓存和来源记录。

真实样本产物在 `results/asset_integration_20261021/`：五个 FBX 同场景、同模型不同材质、共享网格隔离材质、底部锚点和贴图打包均已验证；尺寸为演示估计。材质变化示例只做颜色乘数，不代表已生产新的纹理或几何。

## 1. 三层职责

资产工作流应分成三层：

1. **资产生产层**：由专门的 3D 建模、扫描、重拓扑、贴图或生成服务产出 FBX/GLB、UV、材质和预览。GLM 可以理解参考图、生成结构化描述和发现缺陷，但它本身不会凭空生成可用的 Blender 网格，除非另接专门的 3D 生成服务。
2. **资产登记层**：程序测量网格、检查 UV/材质/文件哈希，GLM 根据原图和多视角预览补充语义标签、类别、风格和不确定性，最后由质量门禁决定是否可用于生产。
3. **场景调用层**：GLM 从照片中提出物体族、尺寸、朝向和摆放关系；确定性检索器从目录返回兼容的几何变体与材质变体；Blender 只负责导入、归一化、实例化、渲染和验证。

## 2. ID 关系：族、几何变体、材质变体

同一种物体不应只有一个文件名。使用下面的稳定关系：

- `family_id`：语义物体族，例如 `dining_chair`、`floor_planter`。同族模型必须能在同一类别和使用规则下互换。
- `geometry_variant_id`：同族的具体几何，例如不同靠背、腿型或尺寸。它有自己的网格、UV、包围盒和几何哈希。
- `material_variant_id`：适配某个几何变体的一组材质贴图，例如木色、黑色、织物颜色。材质变体必须声明可兼容的几何变体集合，不能因类别名称相同就强行套用。
- `asset_id`：对外引用的不可变 ID，建议由 `family_id + geometry_variant_id` 组成；文件移动、改名不能改变 ID。
- `instance_id`：场景中的一次摆放，例如 `chair_03`。同一场景每个实体有独立 `instance_id`，但所有照片视角仍绑定同一实体和同一变体。

去重使用文件内容哈希、规范化网格哈希和材质哈希，而不是只比较名称。相同几何配不同材质属于不同外观变体；小幅随机色调变化应记录为材质参数或派生变体，不能伪装成新模型。

## 3. 生产 manifest 字段

下面是生产源 `asset_manifest_v1` 的字段骨架，不是可以直接交给运行时的 catalog；运行时 `asset_catalog_v1` 必须由 `build-catalog` 生成。字段中标记的来源必须保留，不能把估计值写成实测值。

```json
{
  "schema_version": "asset_manifest_v1",
  "asset_id": "floor_planter__geo_002",
  "family_id": "floor_planter",
  "geometry_variant_id": "geo_002",
  "status": "draft",
  "model": {
    "format": "glb",
    "path": "model.glb",
    "sha256": "...",
    "dimensions_import_units": [0.76, 0.76, 0.82],
    "dimensions_m": [0.76, 0.76, 0.82],
    "meters_per_import_unit": 1.0,
    "scale_source": "measured",
    "scale_confidence": 1.0,
    "triangle_count": 50000,
    "object_count": 1,
    "canonical_up_axis": "+Z",
    "canonical_front_axis": "-Y",
    "transform": {"location": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}
  },
  "placement": {
    "support_type": "floor",
    "anchor_type": "bottom_center",
    "anchor_point_m": [0, 0, 0],
    "contact_points_m": [[0, 0, 0]],
    "mount_points": [],
    "clearance_m": 0,
    "can_rotate_z": true,
    "allow_uniform_scale": true,
    "allow_non_uniform_scale": false,
    "collision_path": "collision.glb",
    "collision_sha256": "<64 hex characters>"
  },
  "semantics": {
    "category": "planter_with_stand",
    "description": "...",
    "tags": ["floor", "cylindrical", "empty_planter"],
    "scope": "assembly",
    "completeness": "complete",
    "confidence": 0.82,
    "source": "glm_plus_blender_measurement"
  },
  "parts": [],
  "uv": {
    "layout_id": "uv_002",
    "channels": 1,
    "resolution": [4096, 4096],
    "texel_density_policy": "family_normalized",
    "normal_convention": "OpenGL_Y+"
  },
  "material_variant_ids": ["mat_light_01"],
  "materials": {
    "mat_light_01": {
      "material_variant_id": "mat_light_01",
      "compatible_geometry_ids": ["geo_002"],
      "uv_layout_id": "uv_002",
      "normal_convention": "OpenGL_Y+",
      "maps": {
        "base_color": {"path": "textures/base.png", "color_space": "sRGB", "sha256": "<64 hex characters>"},
        "roughness": {"path": "textures/roughness.png", "color_space": "Non-Color", "sha256": "<64 hex characters>"}
      }
    }
  },
  "quality": {
    "file_integrity": "passed",
    "import": "passed",
    "geometry": "passed",
    "uv": "passed",
    "materials": "passed",
    "previews": "passed",
    "placement_test": "passed",
    "production_ready": false
  },
  "previews": {
    "front": {"path": "previews/front.png", "sha256": "<64 hex characters>"},
    "side": {"path": "previews/side.png", "sha256": "<64 hex characters>"},
    "back": {"path": "previews/back.png", "sha256": "<64 hex characters>"},
    "three_quarter": {"path": "previews/three_quarter.png", "sha256": "<64 hex characters>"},
    "placement_test": {"path": "previews/placement_test.png", "sha256": "<64 hex characters>"}
  },
  "provenance": {"source_images": [], "generator": null, "license": null, "created_at": "..."}
}
```

### 必须程序自动提取的字段

程序可以可靠取得模型路径、文件哈希、对象数、三角面数、包围盒、UV 层数、材质节点连接、贴图尺寸/色彩空间、导入预览和 Blender 版本。它应同时保存导入单位和真实米制，不能把 FBX 的数值直接当成米。

### 可由 GLM 补充但必须带置信度的字段

GLM 可以根据参考图和多个角度预览提出类别、外观标签、部件名称、正面候选方向、摆放类别、完整性判断和典型尺寸估计。每个值应带 `source: glm_estimate`、`confidence` 和 `alternatives`。GLM 不得发明文件路径、哈希、UV 兼容关系或缺失的支撑几何。

### 不能仅靠描述确认的字段

真实尺寸、承重/碰撞性能、隐藏面、法线质量、权利许可、缺失的椅腿或连接件，必须由实测、建模人员或专用几何检查确认。无法确认的关键字段保留 `null`，资产进入 `draft`，不得静默进入正式数据集。

## 4. 几何变体与材质变体规则

每个几何变体至少要有一套稳定 UV。可替换材质必须满足以下条件：UV 通道数、UV 命名、切线空间、贴图坐标和材质槽语义一致；贴图分辨率和 texel density 在同一族内处于允许范围；法线约定（例如 MikkTSpace、OpenGL/Y 方向）一致。若 UV 不兼容，建立新的 `geometry_variant_id` 或明确的材质转换步骤，不能自动套材质。

组合资产应保存部件树和锚点，例如桌面、桌腿、椅子、盆体、支架分别有 `part_id`、相对变换和接触面。一个不可分割的单网格仍可作为整体资产，但必须标成 `scope: assembly`，不能让 GLM 误以为可以独立替换其中一把椅子。局部特写模型标记为 `partial`，不能自动当作完整落地物体。

尺寸变体应统一以米制规格和容差表示，例如椅座高度 `0.43m ± 0.03m`。场景绑定只允许统一缩放；若需要非均匀拉伸，应重新生成几何变体并重新验收。运行时执行锚点对齐；接触面和净空须由上层摆放验收检查，当前导入器不会自动避碰。

## 5. 调用与选择规则

GLM 输出的是候选约束，不是最终文件路径。先保存候选约束，再由确定性解析器生成运行时绑定。候选约束可以使用：

```json
{
  "instance_id": "chair_03",
  "family_id": "dining_chair",
  "geometry_variant_policy": "compatible_random",
  "material_variant_policy": "compatible_random",
  "target_dimensions_m": [0.48, 0.52, 0.84],
  "transform": {"position_m": [1.2, 0.4, 0], "rotation_z_deg": 90},
  "source_view_ids": ["photo_01", "photo_02"]
}
```

运行时绑定必须引用已经解析出的具体资产和材质变体，不能把 `family_id` 或随机策略直接交给 Blender：

```json
{
  "schema_version": 1,
  "instances": [
    {
      "object_id": "chair_03",
      "asset_id": "dining_chair__geo_002",
      "material_variant_id": "dining_chair__mat_oak",
      "uniform_scale": 1.0,
      "rotation_xyz_deg": [0, 0, 90]
    }
  ]
}
```

底层 `AssetResolver` 筛选 `family_id`、质量状态、文件哈希和材质兼容关系，保留原有种子/组合接口。新增上层 `auto_asset_binding.generate_bindings()` 从当前 V3 bundle 直接读取对象，增加正式 manifest 门禁、类别/显式映射、尺度、朝向、落地/锚点、旋转缩放规则、三轴比例和材质检查。它生成上面的具体绑定；前一段候选约束仍是设计示例，不能直接作为函数参数。没有合格候选时明确记录回退。照片外观排序、完整支撑图、网格碰撞与可见性检查尚未实现。

同一物理物体在多张照片中必须复用同一个 `instance_id`、`geometry_variant_id` 和 `material_variant_id`，否则多视角重建会出现外观跳变。照片还原模式应优先选择与参考图最接近的固定变体，不得为了去重随机换形。数据增强模式才允许在同族兼容集合中按固定随机种子换几何或材质，并记录 `augmentation_seed`。

## 6. 质量门禁与正式数据集

推荐的门禁顺序：

1. 文件和哈希完整，路径为相对路径且不越出资产目录。
2. Blender 导入、UV、法线、材质节点和贴图色彩空间检查通过。
3. 包围盒、锚点、真实尺度来源和支撑规则完整；不确定关键字段保持 `draft`。
4. 单资产至少生成正交/三分之四/背面预览；GLM 检查类别、部件和材质外观。
5. 以典型房间支撑面做实例化测试，检查接触、碰撞、相机可见性和渲染稳定性。
6. 单资产技术、视觉和典型摆放验收通过后，才标记资产 `production_ready: true`；具体场景仍须另行完成原图/渲染审核和最终验收。

正式数据集只接收最终 `accepted` 场景中引用的 `production_ready` 资产。`draft`、`needs_review`、`partial` 和 `failed` 资产保留在候选库，供修复或人工复核，不能因为 GLM 给出高置信度就跳过几何门禁。

## 7. 当前五个资产的补齐清单

`./assets/精选资产5个` 中的 5 个模型已经可以作为导入、材质连接和预览测试样本，但当前都不能直接视为生产资产：

- `008` 园艺喷枪：需要真实尺度、悬挂/支撑锚点和语义正面；不能默认落地。
- `009` 户外桌椅组合：需要拆成桌与椅的部件树，或明确只作为不可拆整体；需要真实尺寸。
- `012` 软垫凳：名称与可见凹槽几何冲突，需要确认是凳子、花盆还是容器后再入族。
- `013` 花盆：需要确认桌面盆/落地盆尺度，并拆分盆体与支架或声明整体替换规则。
- `014` 户外椅：当前是局部资产且缺少完整落地支撑，必须补建椅腿或替换为完整模型。

因此下一步不是只增加文字描述，而是为每个候选生成可验证的几何/材质变体目录，补齐尺度、锚点、部件、UV 兼容矩阵和质量状态，再接入场景候选检索。这样才能在大批量生产时既减少重复，又不牺牲同一照片场景的几何和外观一致性。

## 8. 当前实现的生产命令

```bash
python3 -m scene_factory.asset_manifest validate /path/to/asset_root/family/geo_001 --require-production
python3 -m scene_factory.asset_manifest build-catalog /path/to/asset_root --output /path/to/asset_catalog.json
```

`build-catalog` 是 fail-closed 的：任意一个包未通过 `asset_accepted`，整个 catalog 构建失败，不生成部分可信目录。`external_assets.py` 已支持 GLB/glTF/FBX 导入、UV、多个材质槽、显式锚点和哈希检查；实例化时会复制每个对象的网格数据，避免不同材质变体互相污染，并会拒绝缺失的声明材质槽。碰撞代理只做文件存在性、路径和哈希检查；碰撞网格自身的导入、接触、穿透和可用性测试仍必须由 Blender 场景验收完成。场景级照片对比、GLM 语义分析和修正仍由批处理质量流水线负责。

`validate` 只检查单个包，`build-catalog` 才递归全库，输出必须放在资产根目录之外。构建失败保留旧 catalog；调用方必须检查非零退出码。质量状态是上游真实验收的记录，manifest 校验不会执行 Blender 或视觉检查。当前正式材质范围为不透明 Base Color/Normal/Roughness/Metallic，AO 仅保留；透明、发光和位移需单独扩展导入器，不能按本版本直接放量。每包使用独立、全库唯一的材质 ID，详细约束及放量流程见生产规范第 12–14 节。
