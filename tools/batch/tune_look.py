"""Fast single-camera variant harness: try a look on one view, score it against the photo.

    python tune_look.py SHOP VARIANT [--view view_01] [--edge 960] [--samples 64] ...

Renders one camera of a demo shop with the given overrides, then reports how close the
result sits to that camera's original photo: mean luminance error, saturation error and
SSIM on a blurred pair (geometry differences between a run-down shelf and a reconstructed
one dominate raw SSIM, so the score is only meaningful between variants of the same shop).

Same overrides as render_demo.py, plus:
  --floor VALUE      multiply the room floor colour (the layout colour is often far lighter
                     than the photographed floor, which reads as a blown-out white plane)
  --world-color HEX  replace the world colour, e.g. 6f7684
  --look NAME        AgX look, e.g. 'AgX - Punchy', 'AgX - Medium High Contrast', 'None'
  --view-transform   OCIO view transform, default AgX
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'scene_factory'))
sys.path.insert(0, str(REPO / 'tools' / 'asset_library' / 'retail'))
import os  # noqa: E402
from asset_catalog import AssetCatalog  # noqa: E402
from batch_multiview import prepare_textures  # noqa: E402
from library_layout import expand_response  # noqa: E402
from library_selection import select_and_place  # noqa: E402
from plan_retail import ALIASES  # noqa: E402

BLENDER = os.environ.get('BLENDER', '/Applications/Blender.app/Contents/MacOS/Blender')
CATALOG = Path(os.environ.get('CATALOG', REPO / 'assets' / 'production_library_20261032' / 'library' / 'asset_catalog.json'))
OPTIONS = {'seed': 0, 'dimension_tolerance': 0.25, 'min_uniform_scale': 0.8, 'max_uniform_scale': 1.25,
           'floor_tolerance_m': 0.0001, 'diversity': True, 'category_map': {},
           'settle_max_shift_m': 0.5, 'settle_max_omitted_fraction': 0.1}
SOURCES = {}
for run in os.environ.get('RUNS', '').split(':'):
    if not run:
        continue
    root = Path(run)
    if root.is_dir():
        for scene in root.iterdir():
            if (scene / 'initial_layout.json').is_file() and scene.name not in SOURCES:
                SOURCES[scene.name] = scene


def hex_rgb(value):
    value = value.lstrip('#')
    return [round(int(value[i:i + 2], 16) / 255, 4) for i in (0, 2, 4)]


def rgb_to_display(rgb):
    """Room colours in the layout are already display-referred; the renderer may expect linear."""
    return [round(((c / 12.92) if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4), 5) for c in rgb]


def metrics(render, photo):
    a = np.asarray(Image.open(render).convert('RGB').resize((480, 270), Image.LANCZOS), np.float32)
    b = np.asarray(Image.open(photo).convert('RGB').resize((480, 270), Image.LANCZOS), np.float32)
    lum = lambda x: float((0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).mean())
    sat = lambda x: float(((x.max(2) - x.min(2)) / np.maximum(x.max(2), 1)).mean())
    # SSIM on heavily blurred grey pairs localised over a grid: compares layout, exposure
    # and contrast, not shelf detail. One global window is meaningless here (a giant blur
    # leaves almost no variance), so average over 4x4 tiles like the standard implementation.
    blur = lambda x: np.asarray(Image.fromarray(x.astype(np.uint8)).convert('L').filter(
        ImageFilter.GaussianBlur(6)), np.float32)
    x, y = blur(a), blur(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    scores = []
    for i in range(4):
        for j in range(4):
            xs = x[i * 67:(i + 1) * 67, j * 120:(j + 1) * 120]
            ys = y[i * 67:(i + 1) * 67, j * 120:(j + 1) * 120]
            mx, my = xs.mean(), ys.mean()
            vx, vy = xs.var(), ys.var()
            cov = ((xs - mx) * (ys - my)).mean()
            scores.append(((2 * mx * my + c1) * (2 * cov + c2)) /
                          ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))
    ssim = float(np.mean(scores))
    return {'lum_err': round(abs(lum(a) - lum(b)), 2), 'sat_err': round(abs(sat(a) - sat(b)), 4),
            'ssim': round(float(ssim), 4), 'render_lum': round(lum(a), 1), 'photo_lum': round(lum(b), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shop')
    parser.add_argument('variant')
    parser.add_argument('--view', default='view_01')
    parser.add_argument('--edge', type=int, default=960)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--ev', type=float, default=0.0)
    parser.add_argument('--world', type=float, default=0.0)
    parser.add_argument('--floor', type=float, default=1.0)
    parser.add_argument('--world-color', default=None)
    parser.add_argument('--look', default=None)
    parser.add_argument('--view-transform', default=None)
    parser.add_argument('--tone-match', choices=['reference', 'off'], default='off')
    parser.add_argument('--lens', type=float, default=None, help='override every camera lens in mm')
    parser.add_argument('--from-photo', default='', help="comma list of floor,wall,ceiling to sample from the reference photo")
    args = parser.parse_args()

    out = HERE / 'weidian_full_runs' / 'tune' / f'{args.variant}_{args.shop}' / args.view
    out.mkdir(parents=True, exist_ok=True)
    source = SOURCES[args.shop]
    layout = json.loads((source / 'initial_layout.json').read_text(encoding='utf-8'))
    for entry in layout['objects']:
        entry['category'] = ALIASES.get(entry['category'], entry['category'])
    bundle = expand_response(layout)
    original = json.loads((source / 'initial_bundle.json').read_text(encoding='utf-8'))
    sources = {v['id']: v['source_image'] for v in original['views']}
    for view in bundle['views']:
        view['source_image'] = sources[view['id']]
    # one camera only: film border style, but the renderer wants every view to build the scene
    shared = bundle['shared_scene']
    render_cfg = shared.setdefault('render', {})
    lighting = shared['base_scene'].setdefault('lighting', {})
    room = shared['base_scene']['room']
    if args.lens:
        # The model estimates a focal length per camera; forcing one value tests whether that
        # estimate is what pushes the reconstruction out of register with the photo.
        for view in bundle['views']:
            view['camera']['lens_mm'] = args.lens
        # the first view must equal the shared camera; keep them in step or validation fails
        shared_cam = bundle['shared_scene']['base_scene'].setdefault('camera', {})
        shared_cam['lens_mm'] = args.lens
    if args.ev:
        render_cfg['exposure'] = round(float(render_cfg.get('exposure') or 0) + args.ev, 3)
        for light in shared.get('lights', []):
            if isinstance(light.get('energy'), (int, float)):
                light['energy'] = round(light['energy'] * (2 ** args.ev), 2)
    if args.world:
        lighting['world_strength'] = round(float(lighting.get('world_strength') or 0.12) + args.world, 3)
    if args.world_color:
        render_cfg['world_color'] = rgb_to_display(hex_rgb(args.world_color))
    if args.from_photo:
        # The layout only estimates surface colours; the photograph has them. Sample the
        # lower-centre band for the floor and the left/right margins for the walls, using a
        # median so a stray product or sign does not set the colour.
        photo = Image.open(next(v['source_image'] for v in original['views'] if v['id'] == args.view)).convert('RGB')
        w, h = photo.size
        px = np.asarray(photo, np.float32) / 255.0
        pick = lambda y0, y1, x0, x1: [round(float(v), 4) for v in np.median(
            px[int(h*y0):int(h*y1), int(w*x0):int(w*x1)].reshape(-1, 3), axis=0)]
        if 'floor' in args.from_photo:
            room['floor_color'] = pick(0.80, 0.95, 0.30, 0.70)
        if 'wall' in args.from_photo:
            room['wall_color'] = pick(0.25, 0.55, 0.02, 0.18)
        if 'ceiling' in args.from_photo and 'ceiling_color' in room:
            room['ceiling_color'] = pick(0.05, 0.12, 0.30, 0.70)
        args.floor = args.floor if args.floor != 1.0 else 1.0
    if args.floor != 1.0:
        for key in ('floor_color',):
            if key in room:
                room[key] = [round(min(1.0, c * args.floor), 4) for c in room[key]]

    catalog = AssetCatalog(json.loads(CATALOG.read_text(encoding='utf-8')))
    bundle, bindings, report = select_and_place(bundle, layout, catalog, OPTIONS)
    spec = out / 'scene_bundle.json'
    spec.write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding='utf-8')
    (out / 'asset_bindings.json').write_text(json.dumps(bindings, ensure_ascii=False, indent=1))
    manifest = prepare_textures(bundle, spec, out / 'textures')
    command = [BLENDER, '-b', '--factory-startup', '--python-exit-code', '1',
               '--python', str(REPO / 'scene_factory' / 'generate_multiview.py'), '--',
               '--spec', str(spec), '--output-dir', str(out), '--texture-manifest', str(manifest),
               '--max-edge', str(args.edge), '--samples', str(args.samples), '--device', 'AUTO',
               '--appearance-policy', 'photo_reconstruction', '--tone-match', args.tone_match,
               '--asset-catalog', str(CATALOG), '--asset-bindings', str(out / 'asset_bindings.json')]
    if args.look:
        command += ['--agx-look', args.look]
    if args.view_transform:
        command += ['--view-transform', args.view_transform]
    started = time.monotonic()
    with (out / 'blender.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    shot = out / f'rgb_{args.view}.png'
    if result.returncode != 0 or not shot.is_file():
        print(args.variant, 'FAILED', result.returncode, flush=True)
        return
    photo = Path(next(v['source_image'] for v in original['views'] if v['id'] == args.view))
    m = metrics(shot, photo)
    m.update(variant=args.variant, shop=args.shop, view=args.view, seconds=round(time.monotonic() - started, 1),
             settings={k: v for k, v in vars(args).items() if k not in ('shop', 'variant')})
    print(json.dumps(m, ensure_ascii=False), flush=True)
    ledger = HERE / 'weidian_full_runs' / 'tune' / 'results.jsonl'
    with ledger.open('a') as stream:
        stream.write(json.dumps(m, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
