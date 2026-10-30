"""Read-only verification of an ingested library version.

Usage: python verify_library.py <library_dir> [<pack_reports_dir>]

Checks that the catalog loads under the real AssetCatalog gate, that every asset
passes production_readiness (receipt + hashes), how categories / placements are
distributed, how large the per-scene category hint block is, and, when pack
reports are given, how well each mesh matched its declared size.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root

from scene_factory.asset_catalog import load_catalog
from scene_factory import library_layout

library = Path(sys.argv[1])
reports = Path(sys.argv[2]) if len(sys.argv) > 2 else None
catalog = load_catalog(library / 'asset_catalog.json')

not_ready = []
for asset_id in sorted(catalog.assets):
    ok, reasons = catalog.production_readiness(asset_id)
    if not ok:
        not_ready.append({'asset_id': asset_id, 'reasons': reasons})

categories = collections.Counter(a.get('category') for a in catalog.assets.values())
placements = collections.Counter(a.get('placement') for a in catalog.assets.values())
with_surfaces = sum(1 for a in catalog.assets.values() if a.get('support_surfaces'))
context = library_layout.catalog_context(catalog)
hint_chars = len(json.dumps(context, ensure_ascii=False, separators=(',', ':')))

result = {
    'catalog': str(library / 'asset_catalog.json'),
    'catalog_sha256': catalog.source_sha256,
    'assets': len(catalog.assets),
    'production_ready': len(catalog.assets) - len(not_ready),
    'not_production_ready': not_ready,
    'category_count': len(categories),
    'within_prompt_category_cap_256': len(categories) <= 256,
    'category_hint_chars': hint_chars,
    'placement': dict(placements),
    'assets_with_support_surfaces': with_surfaces,
    'categories': dict(sorted(categories.items(), key=lambda kv: (-kv[1], kv[0]))),
}

if reports is not None and reports.is_dir():
    ratios = {}
    for path in reports.glob('*.json'):
        ratios[path.stem] = json.loads(path.read_text())['mesh_to_declared_ratio']
    off = {k: v for k, v in ratios.items() if abs(v - 1) > 0.02}
    result['pack_reports'] = len(ratios)
    result['mesh_vs_declared_size_off_by_more_than_2pct'] = off

(library.parent / 'library_verification.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
printable = {k: v for k, v in result.items() if k != 'categories'}
print(json.dumps(printable, ensure_ascii=False, indent=2))
print('largest categories:', list(result['categories'].items())[:12])
