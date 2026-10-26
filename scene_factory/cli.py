"""User-facing commands for configurable photo-to-Blender batches."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import secrets
import shutil
import tempfile

from . import __version__, _enable_legacy_imports

_enable_legacy_imports()

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
ROOT = Path(__file__).resolve().parent


def _emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _identifier(relative, stem):
    # Always suffix a path hash: IDs stay stable when another same-named group is added.
    readable = re.sub(r'[^A-Za-z0-9_-]+', '-', stem).strip('-_')[:58] or 'scene'
    digest = hashlib.sha256(relative.encode('utf-8')).hexdigest()[:12]
    return f'{readable}-{digest}'


def discover_groups(directory):
    """Root photos are individual scenes; each immediate folder is one shared scene."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError('input-dir must be an existing photo directory')
    records, used = [], set()
    for item in sorted(directory.iterdir(), key=lambda path: path.name.casefold()):
        if item.name.startswith('.'):
            continue
        if item.is_symlink():
            raise ValueError('Input symlinks are not supported; use regular photo files/directories')
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            candidates = [item]
        elif item.is_dir():
            candidates = sorted((p for p in item.rglob('*')
                                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                                 and not any(part.startswith('.') for part in p.relative_to(item).parts)),
                                key=lambda path: path.relative_to(directory).as_posix())
            if not candidates:
                continue
        else:
            continue
        if len(candidates) > 16:
            raise ValueError(f'A scene has {len(candidates)} photos; each folder must contain 1..16 photos of one space')
        sources = []
        for candidate in candidates:
            source = candidate.resolve()
            if candidate.is_symlink() or not source.is_relative_to(directory):
                raise ValueError('Input photos must be regular files inside input-dir')
            if source in used:
                raise ValueError('A photo belongs to more than one discovered scene')
            used.add(source)
            sources.append(str(source))
        relative = item.relative_to(directory).as_posix()
        records.append({'scene_id': _identifier(relative, item.stem if item.is_file() else item.name),
                        'images': sources})
    if not records:
        raise ValueError('No supported photos found: JPEG, PNG, WebP, BMP or TIFF')
    return {'schema_version': 1, 'scenes': records}


def _write_json(path, value):
    from batch import atomic_json
    atomic_json(Path(path), value)


def init_project(directory, profile='openai-compatible'):
    if profile not in ('openai-compatible', 'zai-anthropic', 'bigmodel', 'bigmodel-production', 'bigmodel-auto-assets', 'deepseek-production'):
        raise ValueError('Unknown project profile')
    directory = Path(directory).expanduser().resolve()
    files = (directory/'factory.json', directory/'README_运行.md')
    if any(path.exists() for path in files):
        raise ValueError('Project config or README already exists; choose a new directory to avoid overwriting it')
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'photos'/'同一场景多视角').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/'config_examples'/(profile+'.json'), files[0])
    key_env = json.loads(files[0].read_text())['models']['analysis']['api_key_env']
    files[1].write_text('''# 照片批量生成 Blender 场景

1. 编辑 `factory.json`：填写服务商实际 `base_url`，确认模型支持图片；配置文件不写密钥。
2. 本机环境变量 `SCENE_API_KEY` 设置为该服务商专用密钥。
3. 每张独立照片放 `photos/`；同一房间多视角放 `photos/同一场景多视角/`，每组最多 16 张。
4. 运行：

```sh
scene-factory doctor --config factory.json
scene-factory probe --config factory.json
scene-factory run --config factory.json --input-dir photos --output-dir output --dry-run
scene-factory run --config factory.json --input-dir photos --output-dir output
scene-factory status output
scene-factory usage output
scene-factory report output
scene-factory export output --destination dataset-v1
```

`doctor` 和 `--dry-run` 不联网；`probe` 会发起一次小图测试请求。
重复运行相同 run 命令会继续未完成任务；已完成且内容未变的步骤直接复用。
每次正式 export 使用新目录，只导出通过最终审核且完整性校验通过的场景。
大批量运行说明见程序的 `scene_factory/PRODUCTION.md`。
完整说明见程序的 `scene_factory/API_BATCH.md`。
'''.replace('SCENE_API_KEY', key_env), encoding='utf-8')
    return {'directory': str(directory), 'config': str(files[0]),
            'next': f'Check base_url and model, then configure {key_env} locally.'}


def doctor(config_path, blender=None):
    from .factory_config import load_config
    from batch import discover_blender
    from model_api import preflight, ModelAPIError
    config = load_config(config_path)
    checks = []
    for role, model in config['models'].items():
        required = role == 'analysis' or (not config['batch']['no_review'] and
                    (role == 'review' or (config['batch']['max_repairs'] > 0 and config['pipeline']['repair_mode'] == 'typed')))
        try:
            preflight(model)
            configured = True
        except ModelAPIError:
            configured = False
        checks.append({'check': f'models.{role}', 'provider': model['provider'],
                       'model': model['model'], 'base_url': model['base_url'],
                       'api_key_env': model['api_key_env'], 'credential_configured': configured,
                       'required': required, 'ok': configured or not required})
    try:
        found = discover_blender(blender or config['render'].get('blender'))
        checks.append({'check': 'blender', 'path': found, 'ok': True})
    except ValueError:
        checks.append({'check': 'blender', 'ok': False,
                       'message': 'Set render.blender, --blender, or BLENDER_BIN to an installed Blender executable.'})
    for module in ('PIL', 'numpy'):
        checks.append({'check': f'python.{module}', 'ok': importlib.util.find_spec(module) is not None})
    library = config['asset_library']
    if library['catalog']:
        try:
            from asset_catalog import load_catalog
            config_directory = Path(config_path).expanduser().resolve().parent
            catalog_path = Path(library['catalog']).expanduser()
            if not catalog_path.is_absolute():
                catalog_path = config_directory/catalog_path
            catalog = load_catalog(catalog_path)
            ready = 0
            for asset in catalog.assets.values():
                if catalog.production_readiness(asset)[0]:
                    ready += 1
            material_errors = 0
            for identifier in catalog.materials:
                try:
                    catalog.verify_material_hashes(identifier)
                except (OSError, ValueError):
                    material_errors += 1
            checks.append({'check': 'asset_library.catalog', 'assets': len(catalog.assets),
                           'production_ready': ready, 'invalid_material_variants': material_errors,
                           'ok': (library['preview'] or ready > 0) and not material_errors,
                           'note': 'Structural/hash check only; automatic category, fitting and material compatibility are decided for each scene.'})
            if library['bindings']:
                bindings = Path(library['bindings']).expanduser()
                if not bindings.is_absolute():
                    bindings = config_directory/bindings
                data = json.loads(bindings.read_text(encoding='utf-8'))
                if not isinstance(data.get('instances'), list):
                    raise ValueError('Asset bindings requires instances')
                for item in data['instances']:
                    catalog.get_asset(item['asset_id'])
                    catalog.get_material_variant(item['material_variant_id'])
                checks.append({'check': 'asset_library.bindings', 'ok': True, 'instances': len(data['instances'])})
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            checks.append({'check': 'asset_library', 'ok': False,
                           'message': 'Asset catalog/bindings is missing or invalid; verify paths, hashes and production metadata.'})
    return {'offline': True, 'model_capabilities_verified': False,
            'ok': all(item['ok'] for item in checks), 'checks': checks,
            'production': {'review_enabled': not config['batch']['no_review'],
                           'automatic_repairs': config['batch']['max_repairs'],
                           'api_workers': config['batch']['api_workers'],
                           'scene_budget': config.get('budget', {}),
                           'acceptance_resolution_possible': config['render']['max_edge'] >= 1280,
                           'acceptance_samples_possible': config['render']['samples'] >= 64,
                           'preview_assets': library['preview']},
            'note': 'No network request or Blender render was made. Use probe for an explicit API image/schema test.'}


def probe(config_path, role='analysis'):
    from PIL import Image, ImageDraw
    from .factory_config import load_config
    from model_api import call_model
    config = load_config(config_path)
    model = dict(config['models'][role])
    # Reasoning-capable providers count thinking tokens in this budget too.
    model['max_output_tokens'] = min(model['max_output_tokens'], 4096)
    colors = ['red', 'green', 'blue', 'yellow']
    secrets.SystemRandom().shuffle(colors)
    schema = {'type': 'object', 'properties': {'colors': {'type': 'array',
              'items': {'type': 'string', 'enum': ['red', 'green', 'blue', 'yellow']},
              'minItems': 4, 'maxItems': 4}}, 'required': ['colors'], 'additionalProperties': False}
    with tempfile.TemporaryDirectory(prefix='scene-factory-probe-') as temp:
        folder = Path(temp)
        image = Image.new('RGB', (192, 192))
        draw = ImageDraw.Draw(image)
        palette = {'red': (230, 20, 20), 'green': (20, 190, 20), 'blue': (20, 20, 230), 'yellow': (240, 225, 20)}
        for index, color in enumerate(colors):
            x, y = index % 2 * 96, index // 2 * 96
            draw.rectangle((x, y, x+95, y+95), fill=palette[color])
        image.save(folder/'probe.png')
        response = call_model('Look at the image. Name the four quadrant colors in this order: top-left, top-right, bottom-left, bottom-right. Return the JSON colors array.',
                              [folder/'probe.png'], schema, folder/'probe.json',
                              timeout=config['batch']['vision_timeout'], config=model)
        provenance = json.loads((folder/'probe.api.json').read_text(encoding='utf-8'))
    return {'role': role, 'model': model['model'], 'schema_valid': True,
            'image_test_passed': response['colors'] == colors,
            'expected': colors, 'received': response['colors'],
            'attempts': provenance['attempts'], 'usage': provenance.get('usage', {}),
            'note': 'A small image/schema probe does not establish full scene reconstruction quality.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='照片 → 可切换模型 API → 批量 Blender 场景')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='创建独立项目及无密钥配置模板')
    init.add_argument('--directory', type=Path, required=True)
    init.add_argument('--profile', choices=('openai-compatible','zai-anthropic','bigmodel','bigmodel-production','bigmodel-auto-assets','deepseek-production'), default='openai-compatible')
    group = commands.add_parser('groups', help='按文件夹生成场景分组；同一文件夹多图共用一个场景')
    group.add_argument('--input-dir', type=Path, required=True)
    group.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run', help='调用模型分析并批量生成 Blender 文件')
    inputs = run.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-dir', type=Path)
    inputs.add_argument('--groups', type=Path)
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--dry-run', action='store_true', help='仅校验配置和分组，不联网或渲染')
    run.add_argument('--blender', help='覆盖配置中的 Blender 可执行文件')
    check = commands.add_parser('doctor', help='离线检查配置、环境变量和 Blender 路径')
    check.add_argument('--config', type=Path, required=True)
    check.add_argument('--blender')
    test = commands.add_parser('probe', help='显式调用 API，测试小图识别和 JSON 输出（会消耗配额）')
    test.add_argument('--config', type=Path, required=True)
    test.add_argument('--role', choices=('analysis', 'review', 'repair'), default='analysis')
    report = commands.add_parser('report', help='生成原照片与最终渲染两列对比')
    report.add_argument('directory', type=Path)
    status = commands.add_parser('status', help='查看任务计数、进度及失败原因，不联网')
    status.add_argument('directory', type=Path)
    status.add_argument('--verify', action='store_true', help='重新检查已验收文件和输入哈希（大型资产库可能较慢）')
    status.add_argument('--limit', type=int, default=100)
    usage = commands.add_parser('usage', help='汇总 API 已报告的调用次数及 token；不估算账单')
    usage.add_argument('directory', type=Path)
    export = commands.add_parser('export', help='仅将通过最终验收的场景原子导出为独立数据集')
    export.add_argument('directory', type=Path)
    export.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            _emit(init_project(args.directory, args.profile))
        elif args.command == 'groups':
            manifest = discover_groups(args.input_dir)
            _write_json(args.output.expanduser().resolve(), manifest)
            _emit({'manifest': str(args.output.expanduser().resolve()), 'scenes': len(manifest['scenes']),
                   'images': sum(len(record['images']) for record in manifest['scenes'])})
        elif args.command == 'run':
            from .factory_config import load_config
            load_config(args.config)
            from batch_multiview import main as batch_main
            options = ['--config', str(args.config.expanduser().resolve()), '--output-dir', str(args.output_dir.expanduser().resolve())]
            if args.dry_run:
                options.append('--dry-run')
            if args.blender:
                options += ['--blender', args.blender]
            if args.groups:
                return batch_main(['--groups', str(args.groups.expanduser().resolve()), *options])
            manifest = discover_groups(args.input_dir)
            with tempfile.TemporaryDirectory(prefix='scene-factory-groups-') as temp:
                group_path = Path(temp)/'groups.json'
                _write_json(group_path, manifest)
                return batch_main(['--groups', str(group_path), *options])
        elif args.command == 'doctor':
            result = doctor(args.config, args.blender)
            _emit(result)
            return int(not result['ok'])
        elif args.command == 'probe':
            result = probe(args.config, args.role)
            _emit(result)
            return int(not result['image_test_passed'])
        elif args.command == 'report':
            from report_multiview import report as make_report
            print(make_report(args.directory.expanduser().resolve()))
        elif args.command == 'status':
            from .diagnostics import batch_status
            _emit(batch_status(args.directory, verify=args.verify, limit=args.limit))
        elif args.command == 'usage':
            from .diagnostics import usage_report
            _emit(usage_report(args.directory))
        elif args.command == 'export':
            from .dataset_export import export_dataset
            _emit(export_dataset(args.directory, args.destination))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
