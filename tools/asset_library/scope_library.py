"""Derive a category-scoped library from a larger one, for model prompts with a category cap.

    python scope_library.py OUT_DIR SOURCE_LIBRARY CATEGORIES.json

scenes are fed a category summary of the whole library, and `library_layout.catalog_context`
refuses to build one above 256 categories rather than silently truncating. A library that
grew past that cap through gap-filling cannot therefore drive model calls, even though it
is fine for offline re-render (selection is per object, not per prompt).

CATEGORIES.json is {"keep": [category, ...]}. Every asset in a kept category is cloned
byte-for-byte (APFS copy-on-write), so hashes and intake receipts stay valid; only the
catalog root is rewritten. This is not an intake run and the source library is untouched.
Dropping a category costs the scenes that ask for it by name, which is why the scope is
chosen from measured demand rather than asset counts.
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

if len(sys.argv) < 4:
    sys.exit(__doc__)
OUT = Path(sys.argv[1]).resolve()
SOURCE = Path(sys.argv[2]).resolve()
SCOPE = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
LIBRARY = OUT / 'library'
if LIBRARY.exists():
    sys.exit(f'{LIBRARY} already exists; refusing to overwrite a library version')

catalog_bytes = (SOURCE / 'asset_catalog.json').read_bytes()
catalog = json.loads(catalog_bytes)
keep = set(SCOPE['keep'])
if len(keep) > 256:
    sys.exit(f'scope keeps {len(keep)} categories; the model prompt accepts at most 256')
known = {a['category'] for a in catalog['assets']}
unknown = keep - known
if unknown:
    sys.exit(f'scope names categories the source lacks: {sorted(unknown)[:5]}')

assets = [a for a in catalog['assets'] if a['category'] in keep]
if not assets:
    sys.exit('scope kept no assets')
material_ids = {mid for a in assets for mid in a.get('material_variant_ids') or []}

(LIBRARY / 'assets').mkdir(parents=True)
clone = ['cp', '-cR'] if sys.platform == 'darwin' else ['cp', '-R']
for asset in assets:
    subprocess.run(clone + [str(SOURCE / 'assets' / asset['asset_id']), str(LIBRARY / 'assets' / asset['asset_id'])],
                   check=True)

materials = {k: v for k, v in catalog['material_variants'].items() if not material_ids or k in material_ids}
scoped = {
    'schema_version': catalog['schema_version'],
    'root': str(LIBRARY),
    'assets': sorted(assets, key=lambda a: a['asset_id']),
    'material_variants': dict(sorted(materials.items())),
    'admission_policy': catalog['admission_policy'],
}
temporary = LIBRARY / 'asset_catalog.json.tmp'
temporary.write_text(json.dumps(scoped, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
os.replace(temporary, LIBRARY / 'asset_catalog.json')

report = {
    'schema_version': 'library_scope_report_v1',
    'created': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    'method': 'APFS copy-on-write clone of committed asset directories; catalog root and category set rewritten. Not an intake run.',
    'source': {'library': str(SOURCE), 'catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(),
               'assets': len(catalog['assets']), 'categories': len(known)},
    'scope': {'requested_categories': len(keep), 'kept_categories': len({a['category'] for a in assets}),
              'kept_assets': len(assets), 'dropped_categories': sorted(known - keep)},
    'output': {'catalog': str(LIBRARY / 'asset_catalog.json'),
               'catalog_sha256': hashlib.sha256((LIBRARY / 'asset_catalog.json').read_bytes()).hexdigest()},
}
(LIBRARY.parent / 'scope_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'assets': len(assets), 'categories': report['scope']['kept_categories'],
                  'dropped': len(known - keep), 'catalog': report['output']['catalog']}, ensure_ascii=False))
