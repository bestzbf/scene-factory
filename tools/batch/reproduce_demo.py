"""Verify frozen demo inputs, then replay shared editable 3D refinement offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production-root', required=True, type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--manifest', type=Path, default=REPO/'configs/retail/demo_replay_v3.json')
    p.add_argument('--blender', default=os.environ.get('BLENDER', '/Applications/Blender.app/Contents/MacOS/Blender'))
    p.add_argument('--quality', choices=['preview', 'final'], default='final')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if manifest.get('schema_version') != 1:
        p.error('Unsupported manifest schema')
    root = a.production_root.resolve()
    errors = []
    for item in manifest['inputs']:
        path = (root/item['path']).resolve()
        if not path.is_relative_to(root):
            errors.append('Path outside production root: '+item['path'])
        elif not path.is_file():
            errors.append('Missing: '+item['path'])
        elif sha(path) != item['sha256']:
            errors.append('SHA256 mismatch: '+item['path'])
    # Source bundles retain photograph paths. Check explicitly instead of quietly
    # producing review sheets with missing references on a relocated dataset.
    for sid in manifest['shops']:
        source = root/manifest.get('source_overrides', {}).get(sid, manifest['source_root']+'/'+sid)
        bundle_path = source/'scene_bundle.json'
        if bundle_path.is_file():
            for view in json.loads(bundle_path.read_text())['views']:
                if not Path(view['source_image']).is_file():
                    errors.append('Reference path unavailable (restore original mount): '+view['source_image'])
    config = REPO/manifest['config']
    sys.path.insert(0, str(REPO/'scene_factory'))
    from retail_refinement import read_config
    try:
        parsed = read_config(config)
    except (ValueError, OSError) as e:
        errors.append(str(e))
        parsed = {}
    if errors:
        print(json.dumps({'ready': False, 'errors': errors}, ensure_ascii=False, indent=2))
        return 1
    version = subprocess.check_output([a.blender, '--version'], text=True).splitlines()[0]
    quality = manifest['quality'][a.quality]
    receipt = {'ready': True, 'manifest': str(a.manifest.resolve()), 'manifest_sha256': sha(a.manifest),
               'config_sha256': sha(config), 'blender': version, 'python': platform.python_version(),
               'platform': platform.platform(), 'quality': quality, 'shops': manifest['shops'],
               'input_files_verified': len(manifest['inputs']),
               'atlas_sha256': sha(Path(parsed['atlas'])),
               'code_sha256': {str(f.relative_to(REPO)): sha(f) for f in [
                   Path(__file__), REPO/'tools/batch/optimize_demo_batch.py', REPO/'tools/batch/optimize_retail_scene.py',
                   REPO/'scene_factory/retail_refinement.py', REPO/'scene_factory/retail_visual_detail.py',
                   REPO/'scene_factory/retail_stock_geometry.py', REPO/'scene_factory/generate_multiview.py']}}
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    if a.check_only:
        return 0
    if a.out is None:
        p.error('--out is required unless --check-only')
    out = a.out.resolve()
    if out == root or out in root.parents:
        p.error('Output must not replace production root')
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out/'replay_receipt.json'
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        for key in ('manifest_sha256', 'config_sha256', 'code_sha256', 'blender', 'quality', 'atlas_sha256'):
            if previous.get(key) != receipt[key]:
                p.error('Replay receipt changed ('+key+'); choose a new output directory')
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    cmd = [sys.executable, str(REPO/'tools/batch/optimize_demo_batch.py'), '--source-root',
           str(root/manifest['source_root']), '--out', str(out), '--config', str(config),
           '--blender', a.blender, '--edge', str(quality['edge']), '--samples', str(quality['samples']),
           '--shops', *manifest['shops']]
    for sid, path in manifest.get('source_overrides', {}).items():
        cmd += ['--source-override', sid+'='+str(root/path)]
    return subprocess.call(cmd)


if __name__ == '__main__':
    sys.exit(main())
