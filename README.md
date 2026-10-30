# Scene Factory

从室内照片批量生成可编辑的 Blender 三维场景。

- **模型只负责读图。** 每个场景把照片发给多模态模型一次。模型返回一份受 JSON Schema 约束的布局：房间尺寸、相机、灯光，以及每个物体的类别、位置、尺寸、朝向和放在什么上面。
- **其余全部由本地固定程序完成。** Python/Blender 程序从资产库挑选真实的 3D 资产并摆放，然后做碰撞、支撑和边界检查，渲染每个机位，最后保存 `.blend`。
- **安全。** 模型返回的内容只当数据处理，从不当代码执行。

当前版本 **0.3.43**。主要用例是把微店的 1,648 家便利店、12,289 张监控画面，按“一家店一个场景、所有摄像头一起输入”批量建模（见[微店项目](#微店监控数据集项目)）。

```text
照片（同一空间的 1–32 个视角）
   │  预处理：去时间戳水印 · 去畸变 · 统一缩放（tools/weidian_prep，可选）
   ▼
多模态模型（gpt-6.1-sol 等）── 一次请求 ──▶ library_layout_v1 布局 JSON
   │
   ▼  本地程序，不再调用模型
资产库选资产（类别 + 尺寸 + 放置方式）→ 自动挪位避让 → 支撑面 / 碰撞 / 边界检查
   │
   ▼
Blender 渲染每个机位 → scene.blend、rgb_view_XX.png、标注 JSON、原图对比
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `scene_factory/` | 主程序（Python 包与 `scene-factory` 命令）、各流程说明文档、配置模板和测试（91 个测试文件） |
| `tools/weidian_prep/` | 监控画面预处理：去时间戳水印（OCR）、去畸变、统一缩放，按店分组输出 `groups.json` |
| `tools/batch/` | 批量运行工具：进度监控、网关探测、429 原因诊断、可平稳停止的启动器、用已付费布局离线重渲 |
| `tools/asset_library/` | 资产库构建：Poly Haven 下载入库、程序化零售资产生成、多库合并、入库校验、资产预览图 |
| `configs/` | 微店全量批次实际使用的两份配置（微店网关 / OpenAI 官方） |
| `docs/weidian/` | 微店项目的评估、试点和离线验证报告（中文） |

仓库只放代码、配置和文档。**照片、资产库（约 9 GB）、运行结果、`.blend`、渲染图和密钥都不在仓库里**，见 [不在仓库里的东西](#不在仓库里的东西)。

## 环境要求

- Python 3.10+（本机使用 3.12）
- Blender 5.2（单独安装）。可以通过配置里的 `render.blender`、命令行 `--blender` 或环境变量 `BLENDER_BIN` 指定路径。工具脚本读环境变量 `BLENDER`，默认是 macOS 的 `/Applications/Blender.app/Contents/MacOS/Blender`
- 一个支持图片输入和 JSON Schema 输出的模型 API：OpenAI Responses、OpenAI Chat 兼容接口或 Anthropic 接口都可以
- 预处理工具另需 OpenCV 和 Pillow；去时间戳要用 macOS 自带的 Apple Vision OCR（PyObjC），因此**只能在 macOS 上运行**

## 安装

```sh
git clone https://github.com/bestzbf/scene-factory.git
cd scene-factory
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
scene-factory --version          # 0.3.43
```

预处理工具的依赖单独安装：

```sh
python -m pip install "opencv-python-headless>=4.10" "numpy>=1.24" "pillow>=10" \
    "pyobjc-core>=10" "pyobjc-framework-Vision>=10" "pyobjc-framework-Quartz>=10"
```

## 快速开始（资产库模式）

```sh
scene-factory init --directory ./my-project --profile bigmodel-library-single-pass
cd my-project
# 1. 编辑 factory.json：models.analysis 填模型地址、模型名、密钥所在的环境变量名；
#    asset_library.catalog 指向资产库的 asset_catalog.json
# 2. 在本机环境变量里设置密钥（变量名见 factory.json 的 api_key_env，模板里是 BIGMODEL_API_KEY；不要写进配置文件）
export BIGMODEL_API_KEY=...

scene-factory doctor --config factory.json                     # 离线检查配置、Blender、资产库
scene-factory probe  --config factory.json                     # 发一张小图测试 API（约 150 token）
scene-factory run --config factory.json --input-dir photos --output-dir output --dry-run
scene-factory run --config factory.json --input-dir photos --output-dir output
scene-factory status output                                    # 进度与失败原因（离线）
scene-factory usage  output                                    # 已报告的请求数和 token
scene-factory report output                                    # 原图 / 渲染对比页
```

照片的放法：

- `photos/` 根目录下的每张照片单独生成一个场景。
- 同一空间的多张照片放进同一个子目录，一起生成一个场景，最多 32 张。
- 也可以不按目录组织，改用 `run --groups groups.json` 直接列出每个场景用哪些图片。微店批次用的就是这种方式。

`doctor`、`--dry-run`、`status`、`usage`、`report` 都不联网。只有 `probe` 和正式 `run` 会调用模型、消耗额度。

### 资产库

`library_layout_v1` 模式只使用资产库里已有的资产。每个资产是一个目录，里面是 `model.glb` 加 `asset.json`（类别、地面或台面放置、最长边尺寸、尺寸来源）。用下面的命令导入：

```sh
scene-factory assets-import --input-dir asset_packages --output-dir assets
```

导入时会把资产统一到标准姿态，算出实测尺寸和可以放东西的支撑面，并为每个文件记录哈希。不合格的资产会被单独隔离，不影响其他资产入库。完整规则见 [scene_factory/LIBRARY_PIPELINE.md](scene_factory/LIBRARY_PIPELINE.md)。

选资产的规则：

- 同类别、同放置方式的资产里，选能等比缩放到所需尺寸（0.8–1.25 倍、误差 ≤25%）的一个。
- 有多个候选时优先挑场景里还没用过的，避免重复。
- 只要有一个物体配不上资产，**整个场景失败**，并在 `iteration_00/asset_selection.json` 写明是哪个物体、为什么配不上。程序不会拿方盒子凑数。

### 自动挪位与大件保护（0.3.43）

模型估计的位置常有几厘米到几十厘米的重叠。可以在配置的 `asset_library.selection` 里开启自动挪位：

```json
"settle_max_shift_m": 0.5,
"settle_max_omitted_fraction": 0.1
```

- **自动挪位**：发生重叠或悬空的物体，在 0.5 m 范围内找最近的空位。台面上的小物跟着支撑它的物体一起移动。每次移动都记录在报告里。
- **省略**：实在放不下的物体可以省略，但每个场景最多省略 10%，超过就整个场景失败。
- **大件保护**：占地 ≥0.5 m²，或高 ≥1.2 m 且占地 ≥0.2 m² 的落地物体算大件，例如收银台、冷柜、货架。大件先占位置，可以挪到 1.0 m，而且**永不省略**；放不下时整个场景失败。是否算大件，按模型要求的尺寸和选中资产的尺寸分别判断，任意一个达标就算。
- 其他检查：选中资产缩放后的尺寸必须在 1 cm–30 m 之间；叠放后不能穿出天花板。

不配置这两项时，行为和旧版一样：有重叠就让场景失败。

## 微店监控数据集项目

数据：1,648 家小型便利店、12,289 张无人的监控背景帧，平均每店 7.5 个摄像头，最多 25 个，来自私有 COS 存储桶。目标：**每家店建一个三维场景，这家店所有摄像头的画面一起输入。**

### 1. 预处理（`tools/weidian_prep`）

```sh
cd tools
python -m weidian_prep.selftest                                  # 合成数据自检
python -m weidian_prep.batch --raw RAW_DIR --run-dir RUN_DIR --workers 6 --qa-every 50
```

- **去时间戳**：粗白字时间戳用“白字形 + 黑描边”识别；细白字用 Apple Vision 离线 OCR 找到日期和时间所在的行。识别后只修补笔画像素。
- **去畸变**：用单参数除法模型。估计方法是把被商品打断的同一条货架边当成一条线，求让它们最直的参数，并把线段随机分两半各估一次，结果一致才采用。单张图证据不足时，借用同店同分辨率摄像头的中位数或合并估计。
- **缩放**：长边缩到不超过 1280 像素。
- **输出**：每家店一个目录，外加 `groups.json`（场景 ID 就是店铺号，可直接交给 `scene-factory run --groups`）、`manifest.json`（每张图的来源哈希、水印框位置、畸变参数）和 QA 对比图。
- 可以断点续跑，多进程并行。全量 1,648 家约 40 分钟，0 家失败，98% 的图去掉了时间戳，89% 做了去畸变。

### 2. 资产库（`tools/asset_library`）

全量批次使用的 `production_library_20261030`：**4,106 件资产，199 个类别**。由三部分合并而成：

| 来源 | 件数 | 类别 | 构建方式 |
| --- | --- | --- | --- |
| Tripo 生成的 GLB | 337 | 26 | `scene-factory assets-import` 入库 |
| Poly Haven（CC0） | 246 | 110 | `polyhaven/build_library.py WORK_DIR`：从官方 API 下载、打包成两文件、入库 |
| 程序化零售资产 | 3,523 | 79 | `retail/plan_retail.py --write` 生成规格 → `retail/build_all.py WORK_DIR` 用 Blender 批量建模并入库 |

零售资产按试点中模型实际要求的类别和尺寸生成，包括货架、岛柜、冷柜、收银台、烟柜、整箱饮料、纸箱、杯面、收银机、电子秤等。商品用无品牌的彩色包装块表示。多个库用 `merge_libraries.py OUT NAME=LIB ...` 合并，合并时逐字节克隆资产目录，原有的哈希和入库收据都保持有效。合并结果用 `verify_library.py LIB_DIR` 校验。

### 3. 批量建模

配置在 `configs/`：

- 模型 gpt-6.1-sol，每个场景只发 1 次请求，不重试、不做视觉复审和修正。
- 开启自动挪位和大件保护。
- 渲染 1600 像素、128 采样。

```sh
export WEIDIAN_AI_API_KEY=...        # 或用 configs/openai_gpt61sol.json + OPENAI_API_KEY
python3 tools/batch/probe_gateway.py schema                      # 一次小请求，确认网关、模型、JSON Schema 和图片输入可用
scene-factory run --groups RUN_DIR/groups.json --config configs/weidian_gateway_gpt61sol.json \
    --output-dir runs/full --dry-run

# 正式运行：sigint_default_exec 让后台进程能响应 Ctrl-C/SIGINT，从而平稳停止
nohup python3 tools/batch/sigint_default_exec.py \
    scene-factory run --groups RUN_DIR/groups.json --config configs/weidian_gateway_gpt61sol.json \
    --output-dir runs/full > runs/full.log 2>&1 &
echo $! > runs/full.pid
caffeinate -is -w $(cat runs/full.pid) &                         # macOS：批次运行期间不休眠

python3 tools/batch/monitor_full_batch.py runs/full --groups RUN_DIR/groups.json
kill -INT $(cat runs/full.pid)   # 平稳停止：不再开始新场景，进行中的请求和渲染会做完；用同一条命令续跑
```

配置里的 `asset_library.catalog` 是相对路径 `../assets/production_library_20261030/...`，按配置文件所在目录解析。请把资产库放到仓库的 `assets/` 下（该目录已被忽略，不会提交），或改成实际路径。

整批的保护措施（`batch_budget`）：

- 上限：最多 2,000 次请求、2,500 万输出 token、6,000 万总 token。
- 熔断：连续 5 次服务端失败（429、5xx 或网络错误）就停止。返回 401/402/403 时立即停止。
- 磁盘：剩余空间低于 50 GB 时不再开始新场景。

已经提交的场景续跑时会直接复用，不会重复付费。一次请求如果已经发出但结果不确定，程序会阻止自动重发，以免重复扣费。

模型的布局（`initial_layout.json`）都保存在输出目录里。资产库补齐之后，可以用下面的命令离线重选资产、重新渲染，**不再调用模型**：

```sh
python3 tools/batch/render_offline.py runs/full assets/NEW_LIBRARY/library/asset_catalog.json runs/offline
```

### 当前进展（2026-10-30）

| 阶段 | 结果 |
| --- | --- |
| 12 家店试点（旧库 583 件） | 模型 12/12 成功，出图 0 家：只有 28% 的物体能配上资产 |
| 补零售资产 + 大件保护后，用已付费布局离线重跑 | **12/12 家出图**，114 个机位，没有大件被省略 |
| 全量第一批，OpenAI 官方 API | 跑了 88 家后账户余额用完（429 `insufficient_quota`），按设计自动停止 |
| 全量第二批，微店网关 | 正在跑剩下的 1,577 家。前 77 家出图率约 21%，其余主要卡在缺资产 |

现在的主要瓶颈是资产库的**长尾类别**：已统计到 113 种缺失类别（落地扇、茶盘、麻袋、电饭锅、落地空调等），每种只出现在少数几家店。计划是两批跑完后汇总缺失清单，补齐资产，再用已付费的布局离线重渲。

渲染效果和照片还有差距：商品是规则的彩色块，光照偏平，部分机位远处的墙是黑的。详见 `docs/weidian/`：

1. [数据评估](docs/weidian/01_数据评估.md)
2. [预处理试点](docs/weidian/02_预处理试点.md)
3. [第一次试点](docs/weidian/03_第一次试点.md)
4. [零售资产与离线验证](docs/weidian/04_零售资产与离线验证.md)
5. [新资产库试点与离线重跑](docs/weidian/05_新资产库试点与离线重跑.md)

## 输出与验收

每个场景一个目录：

- 模型的原始回复、布局和请求审计：`initial_bundle.*`、`initial_layout.json`
- 选资产报告：`iteration_00/asset_selection.json`
- Blender 工程和渲染：`scene.blend`、`rgb_view_XX.png`
- 标注：`annotations_view_XX.json`
- 结果：`result.json`

关闭视觉复审时，成功的场景状态是 `generated_unreviewed`：可以打开查看、继续编辑，但不会进入 `accepted.json`，`scene-factory export` 也不会导出它们。程序检查和自动验收只代表通过了程序设定的门槛，**不代表测量级几何或照片级还原**。

## 不在仓库里的东西

| 内容 | 原因 |
| --- | --- |
| 原始照片、预处理结果 | 私有数据，按数据集 README 仅限研究用途 |
| 资产库（`production_library_*`，约 9 GB） | 太大；可以用 `tools/asset_library` 的脚本重建（Tripo 原始包除外） |
| 运行输出、`.blend`、渲染图、日志 | 生成物，体积大 |
| API 密钥 | 只通过环境变量传入；配置文件里只写环境变量名 |

## 测试

```sh
PYTHONPATH=scene_factory python -m pytest -q scene_factory     # 942 passed, 123 skipped
cd tools && python -m weidian_prep.selftest                     # 预处理合成数据自检
```

跳过的测试需要本机 Blender 或真实模型 API。

## 更多文档

- [scene_factory/LIBRARY_PIPELINE.md](scene_factory/LIBRARY_PIPELINE.md)：资产库模式的完整说明
- [scene_factory/API_BATCH.md](scene_factory/API_BATCH.md)：模型接口、角色与配置项
- [scene_factory/PRODUCTION.md](scene_factory/PRODUCTION.md)：生产控制（并发、续跑、导出）
- [scene_factory/MULTIVIEW.md](scene_factory/MULTIVIEW.md)：多视角场景
- [scene_factory/TOKEN_BUDGETS.md](scene_factory/TOKEN_BUDGETS.md)：token 预算
- [SECURITY.md](SECURITY.md)：密钥与数据安全
