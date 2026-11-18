"""Demo renderer: re-render chosen shops from their paid layouts with tunable quality.

    python render_demo.py VARIANT OUT_DIR [--shops ID ...] [--edge 1920] [--samples 256]
                         [--ev 0.0] [--world 0.0] [--clamp-cameras]

Starts from each shop's saved initial_layout.json and initial_bundle.json, so no model
call is made. VARIANT names the iteration (v1, v2, ...) and is recorded in summary.json
next to the images, together with every override applied, so iterations are comparable.

Overrides, all optional and all recorded:
  --edge/--samples   render resolution and Cycles samples (default 1920 / 256)
  --ev DELTA         exposure offset added to every light in the bundle
  --world DELTA      offset added to the world strength
  --clamp-cameras    move any camera that sits at or above the room ceiling down into the
                     room; the model occasionally places security cameras above a 3 m
                     ceiling, which renders as a blank grey ceiling
  --fix-far-planes   raise the camera far clip so a view down a 15 m aisle is not cut off

Shops with a saved bundle in a model run reuse that bundle's own reference images; a shop
whose layout came from a different run (offline re-render) keeps its own sources.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'scene_factory'))
sys.path.insert(0, str(REPO / 'tools' / 'asset_library' / 'retail'))
import os  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from asset_catalog import AssetCatalog  # noqa: E402
from batch_multiview import prepare_textures  # noqa: E402
from library_layout import expand_response  # noqa: E402
from library_selection import LibrarySelectionError, select_and_place  # noqa: E402
from plan_retail import ALIASES  # noqa: E402

BLENDER = os.environ.get('BLENDER', '/Applications/Blender.app/Contents/MacOS/Blender')
CATALOG = Path(os.environ.get('CATALOG', REPO / 'assets' / 'production_library_20261032' / 'library' / 'asset_catalog.json'))
OPTIONS = {'seed': 0, 'dimension_tolerance': 0.25, 'min_uniform_scale': 0.8, 'max_uniform_scale': 1.25,
           'floor_tolerance_m': 0.0001, 'diversity': True, 'category_map': {},
           'settle_max_shift_m': 0.5, 'settle_max_omitted_fraction': 0.1}
# where each demo shop's paid layout lives
SOURCES = {}
for run in os.environ.get('RUNS', '').split(':'):
    if not run:
        continue
    root = Path(run)
    if root.is_dir():
        for scene in root.iterdir():
            if (scene / 'initial_layout.json').is_file() and scene.name not in SOURCES:
                SOURCES[scene.name] = scene


def clamp_cameras(bundle, room, inset=0.35):
    """Keep every camera inside the room. The model sometimes puts security cameras at
    4-6 m in a 3 m shop; those views render the inside of the ceiling."""
    moved = []
    ceiling = room['height'] - inset
    for view in bundle['views']:
        cam = view['camera']
        x, y, z = cam['position']
        new_z = min(z, ceiling)
        if abs(new_z - z) > 1e-6:
            cam['position'] = [x, y, round(new_z, 4)]
            # keep the sight line pointing at the same floor target
            moved.append({'view': view['id'], 'from_z': round(z, 3), 'to_z': round(new_z, 3)})
    return moved


def widen_far_plane(bundle, room):
    """A 22 mm lens down a 15 m aisle needs a far plane well past the room diagonal."""
    diag = (room['width'] ** 2 + room['depth'] ** 2 + room['height'] ** 2) ** 0.5
    far = round(max(60.0, diag * 6), 1)
    for view in bundle['views']:
        view['camera']['far_m'] = far
    return far


def apply_lighting(bundle, ev_delta, world_delta):
    """Override the two places the renderer actually reads: exposure from shared_scene.render
    and the world (ambient) strength from shared_scene.base_scene.lighting. The ceiling has no
    light above it and the area lights point down, so a low world strength is what leaves
    ceilings and upper walls black."""
    shared = bundle['shared_scene']
    render = shared.setdefault('render', {})
    lighting = shared['base_scene'].setdefault('lighting', {})
    before = {'exposure': render.get('exposure'), 'world_strength': lighting.get('world_strength'),
              'lights': [l.get('energy') for l in shared.get('lights', [])]}
    if ev_delta:
        # exposure is in EV stops; each stop doubles the light reaching the sensor
        render['exposure'] = round(float(render.get('exposure') or 0) + ev_delta, 3)
        for light in shared.get('lights', []):
            if isinstance(light.get('energy'), (int, float)):
                light['energy'] = round(light['energy'] * (2 ** ev_delta), 2)
    if world_delta:
        lighting['world_strength'] = round(float(lighting.get('world_strength') or 0.12) + world_delta, 3)
    after = {'exposure': render.get('exposure'), 'world_strength': lighting.get('world_strength'),
             'lights': [l.get('energy') for l in shared.get('lights', [])]}
    return {'before': before, 'after': after}


def render(shop, out_root, args, catalog):
    source = SOURCES[shop]
    layout = json.loads((source / 'initial_layout.json').read_text(encoding='utf-8'))
    folder = out_root / shop
    folder.mkdir(parents=True, exist_ok=True)
    for entry in layout['objects']:
        entry['category'] = ALIASES.get(entry['category'], entry['category'])
    bundle = expand_response(layout)
    original = json.loads((source / 'initial_bundle.json').read_text(encoding='utf-8'))
    sources = {view['id']: view['source_image'] for view in original['views']}
    for view in bundle['views']:
        view['source_image'] = sources[view['id']]
    room = bundle['shared_scene']['base_scene']['room']
    photo_of = {v['id']: Path(v['source_image']) for v in bundle['views']}
    record = {'variant': args.variant, 'shop': shop, 'source': str(source), 'room': room,
              'cameras': len(bundle['views']), 'overrides': {}}
    if args.lens:
        for view in bundle['views']:
            view['camera']['lens_mm'] = args.lens
        bundle['shared_scene']['base_scene'].setdefault('camera', {})['lens_mm'] = args.lens
        record['overrides']['lens_mm'] = args.lens
    if args.from_photo:
        # The layout only estimates surface colours; the photograph has them. Median of a
        # floor band / wall margin, so one stray product does not set the colour.
        first = bundle['views'][0]
        px = np.asarray(Image.open(photo_of[first['id']]).convert('RGB'), np.float32) / 255.0
        h, w = px.shape[:2]
        pick = lambda y0, y1, x0, x1: [round(float(v), 4) for v in np.median(
            px[int(h * y0):int(h * y1), int(w * x0):int(w * x1)].reshape(-1, 3), axis=0)]
        sampled = {}
        if 'floor' in args.from_photo:
            room['floor_color'] = pick(0.80, 0.95, 0.30, 0.70); sampled['floor'] = room['floor_color']
        if 'wall' in args.from_photo:
            room['wall_color'] = pick(0.25, 0.55, 0.02, 0.18); sampled['wall'] = room['wall_color']
        if 'ceiling' in args.from_photo and 'ceiling_color' in room:
            room['ceiling_color'] = pick(0.05, 0.12, 0.30, 0.70); sampled['ceiling'] = room['ceiling_color']
        record['overrides']['sampled_surfaces'] = sampled
    if args.clamp_cameras:
        record['overrides']['clamped_cameras'] = clamp_cameras(bundle, room)
    if args.fix_far_planes:
        record['overrides']['far_m'] = widen_far_plane(bundle, room)
    if args.ev or args.world:
        record['overrides']['lighting'] = apply_lighting(bundle, args.ev, args.world)
    try:
        bundle, bindings, report = select_and_place(bundle, layout, catalog, OPTIONS)
    except LibrarySelectionError as error:
        record['status'] = 'blocked'
        record['errors'] = error.report.get('errors', [])[:5]
        (folder / 'asset_selection.json').write_text(json.dumps(error.report, ensure_ascii=False, indent=1))
        return record
    spec = folder / 'scene_bundle.json'
    spec.write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding='utf-8')
    (folder / 'asset_bindings.json').write_text(json.dumps(bindings, ensure_ascii=False, indent=1))
    (folder / 'asset_selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    manifest = prepare_textures(bundle, spec, folder / 'textures')
    command = [BLENDER, '-b', '--factory-startup', '--python-exit-code', '1',
               '--python', str(REPO / 'scene_factory' / 'generate_multiview.py'), '--',
               '--spec', str(spec), '--output-dir', str(folder), '--texture-manifest', str(manifest),
               '--max-edge', str(args.edge), '--samples', str(args.samples), '--device', 'AUTO',
               '--appearance-policy', 'photo_reconstruction', '--tone-match', args.tone_match,
               '--asset-catalog', str(CATALOG), '--asset-bindings', str(folder / 'asset_bindings.json')]
    if args.retail_optimization:
        command += ['--retail-optimization', str(args.retail_optimization.resolve()),
                    '--retail-layout', str(source/'initial_layout.json')]
    if args.look:
        command += ['--agx-look', args.look]
    started = time.monotonic()
    with (folder / 'blender.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    record.update(status='rendered' if result.returncode == 0 else 'render_failed',
                  objects=len(bindings['instances']), renders=len(list(folder.glob('rgb_view_*.png'))),
                  moved=report.get('settling', {}).get('moved_objects'),
                  omitted=report.get('settling', {}).get('omitted_objects') or [],
                  seconds=round(time.monotonic() - started, 1))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant')
    parser.add_argument('out', type=Path)
    parser.add_argument('--shops', nargs='+', default=None)
    parser.add_argument('--edge', type=int, default=1920)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--ev', type=float, default=0.0)
    parser.add_argument('--world', type=float, default=0.0)
    parser.add_argument('--clamp-cameras', action='store_true')
    parser.add_argument('--fix-far-planes', action='store_true')
    parser.add_argument('--tone-match', choices=['reference', 'off'], default='reference')
    parser.add_argument('--lens', type=float, default=None)
    parser.add_argument('--from-photo', default='')
    parser.add_argument('--look', default=None)
    parser.add_argument('--retail-optimization', type=Path)
    args = parser.parse_args()
    catalog = AssetCatalog(json.loads(CATALOG.read_text(encoding='utf-8')))
    shops = args.shops or sorted(SOURCES)
    # Each iteration writes to its own variant directory; otherwise a second run silently
    # overwrites the first one's images and the comparison has nothing to compare.
    out_root = (args.out / args.variant).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {'variant': args.variant, 'settings': {'edge': args.edge, 'samples': args.samples,
               'ev': args.ev, 'world': args.world, 'clamp_cameras': args.clamp_cameras,
               'fix_far_planes': args.fix_far_planes, 'tone_match': args.tone_match}, 'shops': {}}
    for shop in shops:
        if shop not in SOURCES:
            print(shop, 'no paid layout; skipped', flush=True)
            continue
        record = render(shop, out_root, args, catalog)
        summary['shops'][shop] = record
        print(shop, record.get('status'), record.get('renders'), record.get('seconds'), flush=True)
    (out_root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
