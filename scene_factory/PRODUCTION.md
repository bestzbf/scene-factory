# Scene Factory 0.3.6 批量生产

## 可直接使用的流程

每张独立照片生成一个场景；同一个子文件夹的 1–16 张照片共同生成一个场景。模型通过 API 负责分阶段照片分析、多视角联合理解、原图与渲染对比、修正和最终评分。固定本地程序校验 JSON 并操作 Blender，不需要 Codex CLI，不执行模型返回的代码。

生产配置默认开启验收，`batch.max_repairs=2` 是技术修复与视觉修复共享的候选修改次数：最多初始候选加两次修改。若一次技术修复、一次视觉修复已用尽预算，最后候选仍必须重新渲染和独立审核。`run` 保留所有任务供排查；`export` 只发布通过最新一轮验收、输入和产物完整性验证的场景。合格判定是模型启发式评分与技术检查，不是测量级几何或照片级相似度保证。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
scene-factory init --directory ./production --profile bigmodel-production
```

在本机设置 `BIGMODEL_API_KEY`，不要将密钥写入配置、源码或聊天。GLM 模板使用 `https://open.bigmodel.cn/api/paas/v4`、`glm-5.3-flash`。三个模型角色可以各自更换模型、兼容协议、URL 和密钥环境变量。使用其他服务时选择 `openai-compatible` 模板并填写实际地址；开启 `batch.no_review=false` 和 `max_repairs`，推荐沿用 GLM 模板的分阶段设置。

```text
production/
  factory.json
  photos/
    corridor.jpg                  # 一个场景
    dorm.jpg                      # 一个场景
    computer_lab/
      front.jpg                   # 同一场景的两个方向
      back.jpg
```

`init` 创建的空示例文件夹可以删除或改名。所有子目录里的照片均属于该子目录对应的一个场景，不会逐文件再拆分。

```sh
scene-factory doctor --config ./production/factory.json
scene-factory probe --config ./production/factory.json
scene-factory run --config ./production/factory.json --input-dir ./production/photos --output-dir ./production/output --dry-run
scene-factory run --config ./production/factory.json --input-dir ./production/photos --output-dir ./production/output
scene-factory status ./production/output
scene-factory usage ./production/output
scene-factory report ./production/output
scene-factory export ./production/output --destination ./production/dataset-v1
```

`doctor` 离线检查配置、凭据是否已配置、Blender 位置、Python 依赖，以及所配置资产库的结构/文件哈希。它不证明模型支持多图、不启动 Blender 渲染。`--dry-run` 只校验配置/分组和输出路径，不检查全部图片内容或联网。`probe` 会产生真实 API 消耗；用于小图和 JSON 能力测试。`run` 会持续消耗模型额度并启动实际 Blender。`status`、`usage`、`report`、`export` 均不调用模型。

## 渲染前自动修复与光照诊断

Blender 构建后先检查实际几何：严重越界的家具、相机与保留墙体的碰撞、基础窗背后的不透明保留墙、覆盖窗洞的大面积低透射照片贴图层。报告包含对象、包围盒、覆盖率以及当前 `scene_bundle.json` 的 SHA256。只有当前构建产生的阻断报告会启动技术修复，普通渲染崩溃、超时、过期报告不会被当作可修复几何问题。

生产配置下，技术修复 API 接收原照片、当前 JSON、检测证据与资产选择记录，输出受约束的 JSON 修改。程序逐项校验后在新的 `iteration_*` 中重建，重跑检查，再渲染并进行独立原图/渲染审核。不存在按房型硬编码移床、统一提高灯光功率或跳过最终验收的路径。`technical_history.json` 与逐轮 `technical_repair_provenance.json` 留存完整证据；恢复运行时验证输入、报告和修复产物哈希。

基础窗玻璃采用折射外观与透明阴影近似，使外部灯光可以进入室内；这是避免昂贵折射焦散的渲染近似。曝光校准后记录 3×3 分区亮度残差，审核/修复 API 可据此判断局部明暗与全局曝光问题。该数值受构图和材质影响，是诊断线索，不是照片还原分数。检测覆盖明确列出的情况，无法穷尽所有窗体或复杂材质的遮挡。

API 因 token 上限截断输出时不接受残缺 JSON，在现有总超时和验证重试上限内要求更简洁的完整输出。GLM 模板的修复推理强度设为 `low`，分析仍为 `high`，以避免修复请求把额度全部用于推理；这不是还原质量保证。预算耗尽、修复无变化或 API 失败的场景会被隔离，其他场景继续。

自动技术修复仅在启用审核、配置模型 API 且采用 `pipeline.repair_mode=typed` 时启动；`no_review=true`、无修改预算或旧式整包修复模式遇到阻断检查直接失败。

## 连续运行、恢复和追加照片

- 场景同时处理量由 `batch.workers` 控制，1–8；只提交相应数量的任务到执行器，不为所有场景预先创建 future。
- 同一批次进程内的所有场景、规划、细节、审核和修正请求共用 `batch.api_workers` 上限，1–32，默认 2。排队时间计入请求超时。模型服务限流可将它降为 1；`detail_workers` 不会突破此上限。多个独立进程分别计数。
- 同一批次进程内 Blender 串行渲染，避免争抢显存；不同进程之间不共享渲染锁。当前版本不是多机任务调度器；多进程或多机运行请按不同输出目录分片，并自行控制机器负载。
- 失败按场景隔离，其他任务继续；API 对网络/429/服务端错误的重试次数由模型配置 `max_retries` 控制。JSON 格式校验重试有独立上限；几何/透光技术修复与视觉修复共享 `max_repairs` 上限，不会无限返修。
- 重新运行相同 `run` 命令可恢复失败/中断任务，复用已提交且哈希一致的模型阶段与渲染。已完成且未变的场景不重复调用模型或渲染。`needs_review` 是本轮额度内未通过验收的终态，不会在每次续跑时反复花费；更换配置、修正上限或代码后使用新输出目录。
- 加入新照片或新的场景文件夹后再次执行原命令。ID 根据相对路径稳定生成；历史分组、汇总和合格索引会保留。不要重命名已有照片、改变已有场景的分组或修改正在处理的文件。
- 首次 Ctrl-C 会停止调度，运行中的 API/Blender 调用完成当前步骤后退出，并保留缓存；退出码为 130。硬退出后可重跑，未提交的步骤会重新执行。中断时未收到最终回包的 API 消耗无法保证不重复。
- 同一输出目录由操作系统文件锁保护。正常退出或进程崩溃会释放锁；保留的 `.batch.lock` / `.process.lock` 文件本身不代表仍运行，不需要删除。旧版无法确认归属的 `.lock` 不会自动删除。请使用本机磁盘；网络文件系统必须自行提供可靠的同等锁语义。
- 输入、模型配置、代码和资产文件均有指纹。变化时拒绝套用旧验收；原图、原资产应保留到本批次完成并导出。升级到 0.3.6 后请创建新的 output，保留旧结果。

输出目录中的 `summary.json`、`run_state.json` 和逐场景 `progress.json` 是原子更新的运行快照。`status` 输出最后一次快照，不是心跳检测；进程硬退出后显示 running 也可直接重跑。`status --verify` 进一步复核已接受记录的文件和输入哈希，较大资产库可能耗时。

`usage` 汇总已保存的 provider/model 调用、HTTP 尝试、失败数和服务端报告的 token；逐 HTTP 尝试求和，不重复累加最后一次的使用量。缺失回包和服务商计费调整不可推断，报告不是完整账单或费用承诺。

## 正式数据集与不合格结果

`accepted` 要求当前最终渲染对应的全部视角经过审核，realism≥90、其余维度≥85；没有 major/blocking 问题、可见资产缺口或技术警告；每视角至少 1280×720（横竖均可），且至少 64 samples。细长比例图片即使长边为 1600，短边也可能不足 720，请提高 `render.max_edge`。预览资产和未审核输出不进入正式数据集。

`needs_review` 表示本轮审核未通过或返修上限已耗尽；`generated_unreviewed` 表示关闭了审核；`failed` / `review_failed` 表示流程或审核/修正失败。这些目录保留 Blender 文件和证据，但 `export` 不会发布它们。返回码 0 表示任务没有流程异常，不表示所有照片均通过视觉验收。

`export` 从实际逐场景结果读取候选，重新验证最后一轮、审核来源、审核输入、渲染输出与文件 SHA256，不依赖手写的 `accepted.json`。篡改或缺失的合格产物会使导出失败。导出期间锁定源批次与目标，目标锁文件保留并可在进程退出后复用。目标必须是新目录，先在旁边暂存，全部成功后通过操作系统的禁止覆盖重命名原子发布；不会覆盖已有数据集。没有通过的场景时不会假造合格场景。

导出包含可编辑且贴图已打包的 `.blend`、各视角渲染/标注、原图、可搬迁的场景描述和裁剪贴图，以及必要的资产子集和验收证据。manifest 记录逐文件哈希；API 原始响应、prompt、配置和运行日志不作为正式数据集内容。源任务目录保留原始证据；导出的审核说明改写为本地相对路径，另记录原审核、来源和渲染状态文件的哈希用于追溯，明确标记路径改写，不把导出版本当成原审核输入。源文件哈希验证不是数字签名或防恶意伪造机制。

## 接入大量资产

```sh
scene-factory init --directory ./production-assets --profile bigmodel-auto-assets
```

将合格目录放到 `production-assets/assets/asset_catalog.json`，或修改配置相对路径。该模板开启自动绑定，无需每场景写绑定文件。几何和材质只能按兼容规则组合；同组照片共用同一绑定。还原任务默认 `selection.diversity=false` 先按类别和尺寸选；做数据增广时再设 true 以偏好多样组合。

当前自动绑定仅支持规则可证实的地面基类物体，采用均匀缩放、底面对齐和严格生产清单。自定义组合、桌面小物或不兼容的候选保留程序建模并记录 fallback，不会强行套错资产。没有实现图片外观向量检索、深度估计、精确网格碰撞或在线生成任意三维资产。全目录内容完整性检查在大型库上有 I/O 成本，建议为生产批次发布不可变的已筛选库；当前不是万级资产检索性能的完成证明。

资产规范见 [ASSET_PRODUCTION_STANDARD_V1.md](ASSET_PRODUCTION_STANDARD_V1.md)，调用说明见 [ASSET_LIBRARY.md](ASSET_LIBRARY.md)，HARMONY 参考分析见 [HARMONY_WORKFLOW.md](HARMONY_WORKFLOW.md)。批量稳定性、资产可调用性、照片还原质量是不同验证项。大规模投产前用自己的代表性室内图评估通过率、平均返修次数、耗时和 token；自动验收是筛选工具，不能保证每个输入都能成功还原。

## 0.3.5 真实执行后的外观校验

`render.tone_match=reference` 现在启用实际 Cycles 预览的共享灯光/环境/曝光有限搜索：最多 15 组灯光候选，全部视角共用一组物理参数，至少 2% 改善且各视角不明显退步才采用。记录原始请求和实际应用功率；最终 PNG 另行计量，模型审核仍独立决定是否达标。

视觉修正先渲染、审核，再通过 `candidate_selection.json` 比较每视角分数、主要缺陷和最终图片区域亮度。显著退步或无有效改善会保留前一最佳候选，下一次修改从它继续。失败尝试用 `failed_iteration` 标明；保留最佳不意味着通过验收，失败状态仍不可导出。

支持显式基础资产材质槽绑定、带真实褶皱的参数化布帘、窗外可见内容与采光分离。参见 [材质和布帘契约](MATERIAL_CLOTH_CONTRACT.md)、[窗外契约](EXTERIOR_CONTEXT_CONTRACT.md)、[灯光拟合契约](LIGHTING_FIT_CONTRACT.md)。窗外是有视差限制的共享 2.5D 外景平面；布帘没有仿真；材质角色识别是启发式。所有能力需要原图支持和最终审核，不能据此保证任意照片能被高质量还原。

## 0.3.5 修正前置校验

材质选择器先对未变动基础资产的实际 source_material_key 等槽位证据校验，错误在同一 API 校验循环内重试，避免占用一轮 Blender 候选预算。每个视角的逐部件投影、支撑接触和越界警告也发送给修正模型，并绑定完整 annotations 文件哈希。投影只判断视锥入画，不证明无遮挡；详见 [逐部件投影契约](CUSTOM_PART_PROJECTION_CONTRACT.md)。

## 0.3.6 光照严重残余保护

共享灯光拟合目标加入最差三分之一区域的平方误差，避免多数暗区掩盖少数严重偏暗/偏亮区域。报告记录目标版本、严重残余和需要检查非功率因素的标记，修正模型会据此检查构图、光源方向/尺寸、通光和反射率。此项改善误差权衡，不能替代照片级外观重建；不同目标版本的损失数值不可直接比较。
