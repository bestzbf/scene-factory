# Scene Factory：可切换模型的照片批量建模

照片由视觉模型分析成严格的场景 JSON，固定的本地 Python/Blender 程序将 JSON 转成几何、材质、灯光和相机。每组图片生成一个 `.blend`，组内每张照片生成一个对应相机视角。更换模型不需要改 Blender 建模代码。

**0.3.3 批量运行入口见 [PRODUCTION.md](PRODUCTION.md)**：增加有界任务队列、跨阶段 API 总并发限制、进程崩溃后可恢复的本机锁、增量批次索引、状态/用量查询及仅包含最终验收合格场景的原子数据集导出。已有输出的代码指纹与新版不同，请使用新输出目录；同一版本、配置、输入可直接续跑。

0.2.0 加入独立 API 生产闭环：GLM 分阶段读图 → Blender 渲染 → GLM 对照原图 → GLM 生成修正数据 → 重渲染 → GLM 重新验收。正式 `scene-factory run --config ...` 运行时不需要 Codex、Codex CLI 或人工逐场景反馈。支持排队、断点续跑、失败隔离和原图对比。输出是估计重建，生成成功不等于几何测量准确或照片级还原。

参考 HARMONY 的源码分析、质感优化顺序及分阶段方案见 [HARMONY_WORKFLOW.md](HARMONY_WORKFLOW.md)。本轮新增可选自动资产绑定：GLM 的场景数据 → 合格目录的类别/尺寸/摆放筛选 → 每轮固定实例 → 绑定感知的视觉审核和修正。开启方法见 [ASSET_LIBRARY.md](ASSET_LIBRARY.md)；使用 `asset_library.auto_bind=true` 后无需为每个场景手写绑定表。深度/分割、照片外观检索和按需三维资产生产仍是后续模块。

## 安装与启动

需要 Python 3.10+ 和本机 Blender；当前渲染流程在 Blender 5.2 验证。以下命令在包含 `pyproject.toml` 的程序根目录执行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
scene-factory --help
scene-factory init --directory ./my-project
```

解压源码包后也可以在根目录运行 `python -m scene_factory ...`；先执行 `python -m pip install .` 安装依赖。Windows 激活命令为 `.venv\Scripts\activate`。本机 Blender 可通过配置 `render.blender`、命令行 `--blender` 或 `BLENDER_BIN` 指定。`doctor` 只检查可执行文件位置，不代表渲染兼容性已通过。

打包普通安装 wheel：

```sh
python -m pip wheel . --no-deps --wheel-dir dist
```

wheel 只包含程序和小型资产，不包含原照片、历史交付、渲染结果或密钥。

## 填写模型配置

### 智谱 BigModel / GLM-5.3-Flash（已通过在线小图测试）

使用官方 OpenAI 兼容基地址 `https://open.bigmodel.cn/api/paas/v4/`，模型 ID 为 `glm-5.3-flash`。
用户给出的 `https://open.bigmodel.cn/api/v1` 在本次实际 POST 测试中返回 HTTP 403，不能用于当前聊天请求。
已提供 `config_examples/bigmodel.json`：

```sh
scene-factory init --directory ./my-project --profile bigmodel
```

该配置使用 `provider=openai_chat`、`api_key_env=BIGMODEL_API_KEY`、`response_format=json_object`、`token_parameter=max_tokens`，实际端点为 `/api/paas/v4/chat/completions`。
在本机设置 `BIGMODEL_API_KEY` 后即可运行。密钥不写入配置或安装包。

智谱模板采用分阶段分析，规划和几何细节使用 `reasoning_effort=high`，视觉评分与修正使用 `low`；三个角色都明确配置为 `glm-5.3-flash`，开启流式请求。每次规划/细节请求都附带该组全部原图，每次视觉验收/修正请求都附带全部原图与对应渲染。渲染前的技术修复只有原图、场景 JSON 和当前检测证据，不虚构渲染或评分。技术与视觉修复合计最多两次修改，长边 1600、128 samples。每次修正后重新渲染、重新评分，只用最新渲染的验收结果决定是否通过。GLM 的高推理请求可能耗时数分钟；配置给规划/修正保留较长时限。输出被 token 上限截断时，只在既定次数和总超时内重试完整 JSON。

较早版本的整场景单次请求曾超时；之前成功的宿舍实验用过人工视觉反馈，因此不能当作独立批量系统的验证。0.2.0 将分阶段分析和自动视觉返修纳入正式程序。具体自动化实测范围与状态见发布包 `validation.json`，生成完成与视觉合格分别记录。

官方资料：[OpenAI SDK 接入](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction) · [模型与多图输入](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash)。

### 本次指定的 Z.ai / GLM-5.3-Flash

用户提供的 `https://api.z.ai/api/anthropic` 使用 **Anthropic 兼容协议**。
已提供 `config_examples/zai-anthropic.json`，可直接创建预填项目：

```sh
scene-factory init --directory ./my-project --profile zai-anthropic
```

配置采用 `provider=anthropic`、上述基地址、`model=GLM-5.3-Flash`、
`auth_scheme=bearer`、`api_key_env=ZAI_API_KEY`、`response_format=prompt`。
实际请求端点为 `/api/anthropic/v1/messages`。程序忽略响应中的独立思考块，只读取最终 JSON，仍进行本地完整校验。使用新密钥时在本机设置 `ZAI_API_KEY`，配置文件不需要修改。

Z.ai 官方资料确认 GLM-5.3-Flash 具备原生图片/多图输入能力，但具体账户与兼容端点仍需实测：
[模型说明](https://docs.z.ai/guides/vlm/glm-5.3-flash) · [Anthropic 兼容地址](https://docs.z.ai/devpack/tool/claude)。本次机器到 `api.z.ai` 的 TLS 握手中断，未获得 HTTP 鉴权/模型响应；不能将该配置标记为实网验证通过。网络恢复后运行 `scene-factory probe --config ./my-project/factory.json`。

### 其他服务商

`init` 创建的 `factory.json` 使用用户指定的模型名 `GLM-5.3-Flash`。**`base_url` 留空，必须填写你实际服务商的 API 基地址。模型名和图片支持能力仍需向服务商确认。** 不会根据密钥格式猜测厂商或发送请求。基地址应为服务文档给出的兼容路径，不要填完整 `/chat/completions` 地址，程序会追加对应端点。

```json
{
  "schema_version": 1,
  "models": {
    "analysis": {
      "provider": "openai_chat",
      "base_url": "https://YOUR-PROVIDER.example/v1",
      "model": "YOUR_VISION_MODEL",
      "api_key_env": "SCENE_API_KEY",
      "response_format": "json_object",
      "max_output_tokens": 32768,
      "timeout_seconds": 600,
      "max_retries": 2
    }
  },
  "batch": {
    "workers": 3,
    "render_workers": 1,
    "max_repairs": 0,
    "no_review": true,
    "min_free_disk_bytes": 0,
    "shared_ledger_dir": null,
    "vision_timeout": 900,
    "render_timeout": 1800
  },
  "render": {
    "max_edge": 1280,
    "samples": 96,
    "device": "AUTO",
    "appearance_policy": "photo_reconstruction",
    "tone_match": "reference"
  }
}
```

`batch.shared_ledger_dir` 默认为 `null`，表示预算和服务商限流账本保存在当前 `output-dir`。需要并行分片时，可把它设为一个共享父目录（相对路径按 `factory.json` 所在目录解析），并确保每个分片的 `output-dir` 都位于该目录之内。所有进程必须使用相同的账本目录和预算/限流配置；账本用本机 SQLite 协调并发预约。它不跨机器，也不适用于缺少可靠 SQLite 文件锁语义的网络文件系统。共享根目录会包含账本和各分片输出，请为每个分片使用独立输出目录。

密钥只从 `api_key_env` 指定的本机环境变量读取，不在配置内写 `api_key`。macOS/Linux 可以在终端使用无回显输入：

```sh
printf 'API key: '
read -rs SCENE_API_KEY
export SCENE_API_KEY
printf '\n'
```

不要把完整密钥发到聊天中，也不要写入版本控制或提交到日志。配置文件只记录环境变量名；分析、验收和修正模型可以分别使用不同的环境变量。如果服务为 `api.apikey.fun`，必须使用其专用变量 `APIKEY_FUN_API_KEY`，禁止使用其他提供商的密钥。

支持的适配器：

| provider | 自动追加端点 | 用途 |
| --- | --- | --- |
| `openai_chat` | `/chat/completions` | OpenAI 兼容的聊天与图片输入 |
| `openai_responses` | `/responses` | Responses 风格多模态输入 |
| `anthropic` | `/v1/messages`（基地址已有 `/v1` 时不重复） | Anthropic 风格输入，当前使用 `prompt` 格式 |

支持协议不代表所有模型均可用。模型需要接受图片、理解多图并有足够的 JSON 输出容量；纯文字模型不能完成照片分析。需要服务商接受 Base64 图片，且模型的上下文和最大输出足以容纳场景描述。

`response_format` 由服务能力决定：`json_schema` 请求严格结构化输出，`json_object` 请求 JSON 模式，`prompt` 仅通过提示词要求 JSON。三种模式都会经过相同的本地格式与数值检查；失败不会执行任意模型脚本，也不会静默降级到另一种模式。`temperature`、`reasoning_effort` 为可选字段，只在显式配置时发送。聊天兼容端点默认发送 `max_completion_tokens`；若服务文档要求 `max_tokens`，显式添加 `"token_parameter": "max_tokens"`。Anthropic 默认使用 `x-api-key` 认证，可通过 `"auth_scheme": "bearer"` 选择 Bearer 认证；OpenAI 兼容端点固定使用 Bearer。服务不支持这些字段时请移除。兼容 API 的参数支持仍需用 `probe` 实测。

`models.analysis` 用于照片规划和细节生成，`models.review` 用于原图/渲染对比及最终验收，`models.repair` 用于修正。省略 review 则复制 analysis，省略 repair 则复制 review。三者都可接 GLM，也可分别接不同服务。开启验收设 `batch.no_review=false`；`max_repairs` 为 1–5 时，启用视觉验收会允许技术修复和视觉修复共享此上限，关闭视觉验收时只允许针对阻断性本地预检错误的技术修复。未进行视觉验收的输出仍为 `generated_unreviewed`，不得进入正式数据集。

新增 `pipeline` 配置：

```json
{
  "analysis_mode": "staged",
  "repair_mode": "typed",
  "validation_retries": 2,
  "staged": {
    "planner_timeout": 1800,
    "detail_timeout": 1200,
    "detail_workers": 2,
    "max_groups": 8,
    "max_parts_per_group": 60,
    "validation_retries": 2,
    "planner_effort": "high",
    "detail_effort": "high"
  }
}
```

`staged` 将照片规划和各物体组细节分开，减少单次输出过长导致的失败；所有几何数值仍来自 API。最多 16 组、每组 100 部件，配置乘积不能超过 800。`pipeline.staged.detail_wire_format="native_sparse_parts_v1"` 可省略未使用的空几何数组；本地解码会恢复默认值并对完整规范场景再次验证，拓扑与数值不会由解码器生成。它保持关闭，只有显式省成本配置启用；不同模型的解释一致性尚未实测。`pipeline.staged.route_detail_images=true` 可按规划阶段声明的构件证据视角路由细节请求；只有构件和照片贴图材质映射均超过置信度门槛时才过滤图片，并把声明贴图材质的源视角一并保留，同时限制细节输出只能使用这些贴图材质。缺字段、低置信度或不一致映射会回退为发送全部原图。该选项默认关闭；置信度由模型自报，未在真实照片集上校准。`typed` 将评分与修正分开，模型输出有类型约束的数据编辑，固定程序校验并应用，不能执行模型生成的 Python。`validation_retries` 控制 JSON/语义错误纠正次数，不等同于视觉返修轮数。分阶段分析的 planner/detail_timeout 按每次请求计算；评分与修正的 vision_timeout 是该阶段全部校验重试的总时限，单次请求同时受模型 timeout_seconds 限制。旧配置默认 `analysis_mode=monolithic`，可显式保留 `repair_mode=full_bundle` 的整场景修改模式；完整 GLM 闭环请使用 bigmodel 模板。

`stream=true` 当前只支持 `openai_chat`。只保存最终有效 JSON 和数字化调用统计，不保存模型思考文本；中断或无效输出不会当作场景。`reasoning_effort` 参数须符合服务商要求；GLM 支持 low/high/max，本模板避免默认最高档。

## 照片分组

```text
my-project/
  factory.json
  photos/
    corridor.jpg             # 一张照片 → 一个场景
    dorm.jpg                 # 一张照片 → 一个场景
    computer_lab/            # 文件夹内多图 → 一个共享场景
      view_01.jpg
      view_02.jpg
    living_room/
      view_01.jpg
      detail/view_02.jpg
```

顶层每张照片各生成一个场景；每个一级子文件夹内所有照片（含更深层目录）生成一个场景。每组最多 16 张，仅放同一物理空间的照片。目录名可以为中文；输出标识会生成稳定 ASCII 名称和路径摘要，避免同名冲突。照片按相对文件名排序，每个视角的顺序可复现。隐藏文件忽略，符号链接不接受。

支持 JPEG、PNG、WebP、BMP、TIFF；进入模型之前统一处理图片方向和格式。可以先生成分组清单并检查：

```sh
scene-factory groups --input-dir ./my-project/photos --output ./my-project/groups.json
```

也可以自行维护下列清单；其中相对路径基于 `groups.json` 所在目录解析：

```json
{
  "schema_version": 1,
  "scenes": [
    {"scene_id": "computer_lab", "images": ["photos/computer_lab/view_01.jpg", "photos/computer_lab/view_02.jpg"]},
    {"scene_id": "corridor", "images": ["photos/corridor.jpg"]}
  ]
}
```

## 检查、生产与重跑

```sh
scene-factory doctor --config ./my-project/factory.json
scene-factory probe --config ./my-project/factory.json
scene-factory run --config ./my-project/factory.json --input-dir ./my-project/photos --output-dir ./my-project/output --dry-run
scene-factory run --config ./my-project/factory.json --input-dir ./my-project/photos --output-dir ./my-project/output
scene-factory report ./my-project/output
```

已有清单时，将 `--input-dir ...` 换成 `--groups ./my-project/groups.json`。`doctor` 和 `--dry-run` 不会联网，不上传照片，也不渲染。`doctor` 检查配置、密钥变量是否有效、Python 依赖和 Blender 路径；它不会输出密钥。`probe` 是显式在线操作，会向所选模型发送一张随机排列的色块小图，验证图像输入和结构化 JSON；它不代表完整复杂场景质量验收。单独测试验收或修正模型可用 `probe --role review`、`probe --role repair`。

批量模式的 `workers` 控制同时分析/验收的场景数，范围 1–8；本机 Blender 渲染会排队，避免多个大场景同时耗尽显存。`device=AUTO` 尝试可用加速设备，`CPU` 强制 CPU。`max_edge` 范围 64–8192，`samples` 范围 1–4096。`photo_reconstruction` 使用当前材质、屏幕和灯板修复；`tone_match=reference` 使用本地原图明暗统计调整共享曝光，不额外调用模型。

重跑相同命令会校验输入图片、配置、代码和已有产物的摘要，并继续或复用有效阶段。修改模型、提示词、配置、图片或代码后使用新的输出目录；不会把不同版本的结果混用。单场景失败会记录错误并继续处理其他场景。中断留下的 `.lock` 必须确认没有进程使用该场景后再移除。

通用模板默认关闭视觉验收；bigmodel 模板已开启完整闭环。分阶段分析需要 1 次规划和每个物体组 1 次细节调用，另加有界校验重试。每轮增加一次视觉验收；返修轮增加一次修正调用、重渲染和新的验收。网络错误、限流或服务端错误可能按 `max_retries` 重试并产生额外计费；每次调用的 token 使用量（服务返回时）及尝试次数记录在 `.api.json`，不会估算未知价格。超时会失败，不会把不完整输出当成功结果。

场景数 `workers` 与细节并发 `detail_workers` 相乘决定分析阶段的最高并发请求数，请按账户限流调整；bigmodel 默认 1×2。

当前 HTTP 适配器使用直接 TLS 连接，不自动读取代理环境变量，不跟随跳转。需要代理才能访问的机器须先提供可直达的服务地址或网络连接。JSON 截断、格式不符或语义校验失败会保留失败记录。分阶段分析、typed 修正和评分按配置做有界纠正，不会无限修补或悄悄换模型。达到视觉返修预算仍未达标则输出 needs_review。

## 输出

每个场景目录含 `initial_bundle.json`、来源与调用记录、迭代目录、`result.json` 和阶段进度。最终迭代目录含 `scene.blend`、`rgb_view_XX.png`、相机/对象标注和 `metadata.json`。批次根目录的 `summary.json` 列出各场景状态；批次结束自动输出 `comparison.html` 与 `comparison.jpg`，也可通过 `report` 重新生成，只展示原照片和最终效果两列。

网页包含全部已生成视角并延迟加载图片；对比总图最多拼接前 24 个视角，避免大批次生成超高图片耗尽内存。

协议实现参考：[OpenAI 图片输入](https://developers.openai.com/api/docs/guides/images-vision) 与 [结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs)。服务商未实现的参数必须在配置中显式调整，没有自动协议降级。

`review_failed` 代表已经生成可用渲染，但视觉对比或返修 API 失败，保留 `.blend` 和图片并允许重试；`failed` 是分析或渲染等技术失败。

`generated_unreviewed` 代表生成成功但未进行视觉验收；`needs_review` 代表已生成但仍存在差异；`accepted` 代表通过程序设定的自动门槛，不代表真实尺寸测量认证。物体遮挡、复杂家具、相机估计和多视角冲突仍会影响还原度。


自动验收要求每个视角的各项分数至少 85、realism 至少 90，无 major/blocking 问题、无资产能力缺口、无技术告警，并且渲染至少长边 1280 / 短边 720 / 64 samples。这是模型启发式门槛，不是摄影测量或人工质量认证。`accepted.json` 只收录通过门槛的结果；缺陷明确的输出仍可作为 needs_review 交付检查。模型评分与模型修正可能有共同偏差，大量生产的数据仍建议抽样核查。

## 0.2.0 实测范围

2026-10-20 使用官方 BigModel 端点和 `glm-5.3-flash` 完成一张宿舍照片的全自动重建：1 次规划、6 次细节生成、2 次修正、3 次原图/渲染对比，共 12 次真实 API 调用；Blender 生成三轮 1600×1200、128 samples 渲染，耗时约 37 分钟。场景数据与修正均来自 GLM，运行时没有 Codex 视觉反馈或人工几何修改。固定程序完成结构校验、材质构建、渲染及基于原图统计的曝光调整。

最终状态为 `needs_review`，未通过照片还原验收。最终 GLM 评分为构图 78、几何 65、布局 75、光照 62、材质 55、真实感 52；窗景、家具比例、衣物与材质细节仍有差异。这证明完整自动流程可运行，不证明任意室内照片都能达到照片级还原。

该线上试验保留了实际运行源码快照及来源摘要；发布前另行修复续跑与调用记录保留，并通过本地测试。发布包的确切代码未再次执行付费 GLM 全流程。四图三场景和多视角配对已通过模拟模型 API 与真实 Blender 的集成测试，不能替代多场景真实 GLM 质量验证。详细记录见发布包 `validation.json`。
