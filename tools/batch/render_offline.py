"""Render pilot shops offline from the model layouts already paid for: no model call.

For each shop: take the saved initial_layout.json, run library selection
against a catalog (with settling), then render every camera with the normal
scene-factory Blender renderer. Writes into a new directory and never touches
the original pilot run. Category names in the old layouts are mapped to the new
library's names only for this preview; a real rerun lets the model choose them.

Usage: python render_offline.py <run_dir> <catalog.json> <out_dir> [scene_id ...]

<run_dir> is a finished scene-factory output directory (one sub-directory per scene
with initial_layout.json + initial_bundle.json). Environment: BLENDER,
NO_ALIASES=1 to keep the model's category names unchanged.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2] / 'scene_factory'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'asset_library' / 'retail'))
from asset_catalog import AssetCatalog  # noqa: E402
from batch_multiview import prepare_textures  # noqa: E402
from library_layout import expand_response  # noqa: E402
from library_selection import LibrarySelectionError, select_and_place  # noqa: E402
from plan_retail import ALIASES  # noqa: E402

import os  # noqa: E402
PILOT = None  # first argument
BLENDER = os.environ.get('BLENDER', '/Applications/Blender.app/Contents/MacOS/Blender')
OPTIONS = {'seed': 0, 'dimension_tolerance': 0.25, 'min_uniform_scale': 0.8, 'max_uniform_scale': 1.25,
           'floor_tolerance_m': 0.0001, 'diversity': True, 'category_map': {},
           'settle_max_shift_m': 0.5, 'settle_max_omitted_fraction': 0.1}


def main():
    global PILOT
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    PILOT = Path(sys.argv[1]).resolve()
    catalog_path, out = Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
    shops = sys.argv[4:] or sorted(p.name for p in PILOT.iterdir() if (p / 'initial_layout.json').is_file())
    catalog = AssetCatalog(json.loads(catalog_path.read_text()))
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for shop in shops:
        started = time.monotonic()
        folder = out / shop
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        layout = json.loads((PILOT / shop / 'initial_layout.json').read_text())
        if os.environ.get('NO_ALIASES') != '1':
            for entry in layout['objects']:
                entry['category'] = ALIASES.get(entry['category'], entry['category'])
        # The pilot's own bundle already points each view at its saved, hash-checked reference copy.
        bundle = expand_response(layout)
        original = json.loads((PILOT / shop / 'initial_bundle.json').read_text())
        sources = {view['id']: view['source_image'] for view in original['views']}
        for view in bundle['views']:
            view['source_image'] = sources[view['id']]
        try:
            bundle, bindings, report = select_and_place(bundle, layout, catalog, OPTIONS)
        except LibrarySelectionError as error:
            (folder / 'asset_selection.json').write_text(json.dumps(error.report, ensure_ascii=False, indent=1))
            summary[shop] = {'status': 'blocked', 'errors': error.report['errors'][:5]}
            print(shop, 'BLOCKED', error.report['errors'][:3], flush=True)
            continue
        spec = folder / 'scene_bundle.json'
        spec.write_text(json.dumps(bundle, ensure_ascii=False, indent=1))
        (folder / 'asset_bindings.json').write_text(json.dumps(bindings, ensure_ascii=False, indent=1))
        (folder / 'asset_selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
        manifest = prepare_textures(bundle, spec, folder / 'textures')
        command = [BLENDER, '-b', '--factory-startup', '--python-exit-code', '1', '--python', str(REPO / 'generate_multiview.py'), '--',
                   '--spec', str(spec), '--output-dir', str(folder), '--texture-manifest', str(manifest),
                   '--max-edge', '960', '--samples', '48', '--device', 'AUTO', '--appearance-policy', 'photo_reconstruction',
                   '--tone-match', 'reference', '--asset-catalog', str(catalog_path), '--asset-bindings', str(folder / 'asset_bindings.json')]
        with (folder / 'blender.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        renders = sorted(folder.glob('rgb_*.png'))
        summary[shop] = {'status': 'rendered' if result.returncode == 0 else 'render_failed', 'renders': len(renders),
                         'objects': len(bindings['instances']), 'moved': report['settling']['moved_objects'],
                         'omitted': report['settling']['omitted_objects'], 'seconds': round(time.monotonic() - started, 1)}
        print(shop, summary[shop], flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
