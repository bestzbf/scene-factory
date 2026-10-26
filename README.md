# Scene Factory

从室内照片批量生成 Blender 场景。模型负责输出受约束的场景 JSON；固定 Python/Blender 程序负责校验、建模、渲染、比较、有限修正和合格场景导出。程序不执行模型返回的代码。

## Requirements

- Python 3.10+
- Blender 5.2 LTS（外部安装，本仓库不包含 Blender）
- 支持图片输入和 JSON 输出的模型 API；默认示例为 GLM-5.3-Flash

## Install

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
scene-factory --version
scene-factory init --directory ./production --profile bigmodel-production
```

配置 API key 时按 `production/factory.json` 中的 `api_key_env` 设置本机环境变量。示例配置不保存密钥：

```sh
read -s BIGMODEL_API_KEY
export BIGMODEL_API_KEY
```

## Batch photos

将独立照片放在 `production/photos/` 根目录；同一空间的多个视角放在同一个子目录。运行：

```sh
scene-factory doctor --config ./production/factory.json
scene-factory run --config ./production/factory.json \
  --input-dir ./production/photos --output-dir ./production/output --dry-run
scene-factory run --config ./production/factory.json \
  --input-dir ./production/photos --output-dir ./production/output
scene-factory status ./production/output
scene-factory report ./production/output
scene-factory export ./production/output --destination ./production/dataset-v1
```

`doctor` 和 `--dry-run` 不联网。`probe`、正式运行、视觉验收和自动修正会消耗模型额度。`export` 只导出最终验收通过且完整性检查通过的场景。自动验收是模型评分与程序检查，不保证摄影测量精度或照片级效果。

See [`scene_factory/PRODUCTION.md`](scene_factory/PRODUCTION.md) for production controls and [`scene_factory/API_BATCH.md`](scene_factory/API_BATCH.md) for compatible API providers and configuration.

## Tests

```sh
python -m unittest discover -s scene_factory -p 'test_*.py' -v
```

This source snapshot corresponds to version 0.3.15. Do not commit local photos, API credentials, generated scenes, renders, logs, or virtual environments.
