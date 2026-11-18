"""Compare two independently built retail outputs for repeatable shared geometry.

This does not compare PNG bytes or certify image fidelity to photographs.
"""
import argparse
import hashlib
import json
from pathlib import Path


def source_identity(source):
    """Independent prepared directories can describe the same reviewed input."""
    folder=Path(source);report_path=folder/'preparation_report.json'
    if report_path.is_file():
        report=json.loads(report_path.read_text())
        if report.get('profile_sha256') and report.get('atlas_sha256'):
            return {'profile':report['profile_sha256'],'atlas':report['atlas_sha256'],
                    'bundle':hashlib.sha256((folder/'scene_bundle.json').read_bytes()).hexdigest()}
    return {'legacy_source_path':str(folder.resolve())}


def cameras(root, verification):
    if verification.get('camera_parameters'):
        return verification['camera_parameters']
    # Older runs verify that these source cameras remain unchanged in Blender.
    bundle=json.loads((root/'scene_bundle.json').read_text())
    return {v['id']:v['camera'] for v in bundle['views']}


def compare(left, right):
    shops = {p.name for root in (left, right) for p in root.iterdir()
             if p.is_dir() and (p/'verification.json').is_file()}
    results = {}
    for sid in sorted(shops):
        try:
            a, b = [json.loads((root/sid/'verification.json').read_text()) for root in (left, right)]
            ra, rb = [json.loads((root/sid/'retail_refinement.json').read_text()) for root in (left, right)]
            configs = []
            for root in (left, right):
                c = json.loads((root/sid/'optimization_config.json').read_text())
                c.pop('_config_path', None)  # provenance path is not a refinement setting
                if c.get('atlas'):
                    c['atlas'] = hashlib.sha256(Path(c['atlas']).read_bytes()).hexdigest()
                configs.append(c)
            checks = {'both_runs_passed': a['passed'] and b['passed'],
                      'effective_settings_identical': configs[0] == configs[1],
                      'source_identical': source_identity(ra['source']) == source_identity(rb['source']),
                      'views_identical': a['checked_cameras'] == b['checked_cameras'],
                      'cameras_identical': cameras(left/sid,a) == cameras(right/sid,b),
                      'geometry_identical': a['geometry_sha256_after_views'] == b['geometry_sha256_after_views'],
                      'stock_count_identical': a['stock_units'] == b['stock_units']}
            results[sid] = {'passed': all(checks.values()), 'checks': checks,
                            'geometry_sha256': a['geometry_sha256_after_views']}
        except (OSError, KeyError, ValueError) as e:
            results[sid] = {'passed': False, 'error': str(e)}
    return {'passed': bool(results) and all(x['passed'] for x in results.values()),
            'scope': 'Independent-build shared mesh/world-transform digest; no PNG-byte or measured-geometry claim.',
            'left': str(left.resolve()), 'right': str(right.resolve()), 'shops': results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', required=True, type=Path)
    p.add_argument('--right', required=True, type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    result = compare(a.left, a.right)
    value = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out:
        a.out.write_text(value+'\n')
    print(value)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
