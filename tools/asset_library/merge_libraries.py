"""Merge already-ingested library versions into one new production library version.

    python merge_libraries.py OUT_DIR NAME=LIBRARY_DIR [NAME=LIBRARY_DIR ...]

e.g. production_library_20261030 was built as
    merge_libraries.py production_library_20261030 \
        production_library_20261029=production_library_20261029/library \
        retail_assets_20261029=retail_assets_20261029/library

Each committed asset directory is cloned byte-for-byte (APFS copy-on-write), so
every model hash and intake receipt stays valid; only the catalog root changes.
This is not a new intake run, and the source libraries are left untouched.
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

if len(sys.argv) < 3 or not all('=' in arg for arg in sys.argv[2:]):
    sys.exit(__doc__)
OUT = Path(sys.argv[1]).resolve()
SOURCES = {name: Path(path).resolve() for name, path in (arg.split('=', 1) for arg in sys.argv[2:])}
LIBRARY = OUT / 'library'
if LIBRARY.exists():
    sys.exit(f'{LIBRARY} already exists; refusing to overwrite a library version')

catalogs = {}
for name, lib in SOURCES.items():
    raw = (lib / 'asset_catalog.json').read_bytes()
    catalogs[name] = (json.loads(raw), hashlib.sha256(raw).hexdigest())

asset_ids, geometry_ids, material_ids = set(), set(), set()
for name, (catalog, _) in catalogs.items():
    for asset in catalog['assets']:
        if asset['asset_id'] in asset_ids or asset['geometry_variant_id'] in geometry_ids:
            sys.exit(f"duplicate asset or geometry id in {name}: {asset['asset_id']}")
        asset_ids.add(asset['asset_id']); geometry_ids.add(asset['geometry_variant_id'])
    for key in catalog['material_variants']:
        if key in material_ids:
            sys.exit(f'duplicate material variant id in {name}: {key}')
        material_ids.add(key)
policies = {json.dumps(c['admission_policy'], sort_keys=True) for c, _ in catalogs.values()}
if len(policies) != 1:
    sys.exit(f'admission policies differ: {policies}')

(LIBRARY / 'assets').mkdir(parents=True)
origin = {}
for name, (catalog, _) in catalogs.items():
    for asset in catalog['assets']:
        source = SOURCES[name] / 'assets' / asset['asset_id']
        # -c clones on APFS (no extra disk); elsewhere fall back to a plain copy.
        clone = ['cp', '-cR'] if sys.platform == 'darwin' else ['cp', '-R']
        subprocess.run(clone + [str(source), str(LIBRARY / 'assets' / asset['asset_id'])], check=True)
        origin[asset['asset_id']] = name

assets = sorted((a for c, _ in catalogs.values() for a in c['assets']), key=lambda a: a['asset_id'])
materials = {}
for catalog, _ in catalogs.values():
    materials.update(catalog['material_variants'])
merged = {
    'schema_version': 'asset_catalog_v1',
    'root': str(LIBRARY),
    'assets': assets,
    'material_variants': dict(sorted(materials.items())),
    'admission_policy': json.loads(next(iter(policies))),
}
temporary = LIBRARY / 'asset_catalog.json.tmp'
temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
os.replace(temporary, LIBRARY / 'asset_catalog.json')

report = {
    'schema_version': 'library_merge_report_v1',
    'created': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    'method': 'APFS copy-on-write clone of committed asset directories; catalog root rewritten. Not a new intake run.',
    'inputs': [{'name': name, 'catalog': str(SOURCES[name] / 'asset_catalog.json'), 'catalog_sha256': digest,
                'assets': len(c['assets']), 'categories': len({a['category'] for a in c['assets']})}
               for name, (c, digest) in catalogs.items()],
    'output': {'catalog': str(LIBRARY / 'asset_catalog.json'),
               'catalog_sha256': hashlib.sha256((LIBRARY / 'asset_catalog.json').read_bytes()).hexdigest(),
               'assets': len(assets), 'material_variants': len(materials),
               'categories': len({a['category'] for a in assets})},
    'checks': {'asset_id_collisions': 0, 'geometry_id_collisions': 0, 'material_id_collisions': 0,
               'admission_policies_identical': True},
    'asset_origin': origin,
}
(LIBRARY / 'merge_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

# Keep each source's own merge/build report next to the new library for provenance.
provenance = OUT / 'sources'
provenance.mkdir(parents=True, exist_ok=True)
for name, library in SOURCES.items():
    for report_name in ('merge_report.json', 'build_summary.json', 'plan_report.json'):
        for candidate in (library / report_name, library.parent / report_name):
            if candidate.is_file():
                shutil.copy2(candidate, provenance / f'{name}_{report_name}')
(OUT / 'scripts').mkdir(exist_ok=True)
shutil.copy2(Path(__file__), OUT / 'scripts' / Path(__file__).name)
print(json.dumps(report['output'], ensure_ascii=False))
print('origin:', {n: sum(1 for v in origin.values() if v == n) for n in SOURCES})
