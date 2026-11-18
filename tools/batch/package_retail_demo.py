"""Publish local review images/metrics from a completed batch into a docs folder.

Does not copy large Blender files, publish to the network, or modify source renders.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageOps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--showcase-view', action='append', default=[], metavar='SHOP=VIEW')
    a = p.parse_args()
    batch, out = a.batch.resolve(), a.out.resolve()
    if batch == out or batch in out.parents or out in batch.parents:
        p.error('Package output must be independent of the batch')
    summary = json.loads((batch/'summary.json').read_text())
    if not summary or any(r['status'] != 'complete' for r in summary.values()):
        p.error('Only a fully completed batch can be packaged')
    selected = dict(v.split('=', 1) for v in a.showcase_view)
    rows = []
    for sid in summary:
        folder = batch/sid
        if not json.loads((folder/'verification.json').read_text())['passed']:
            p.error('Verification failed: '+sid)
        bundle = json.loads((folder/'scene_bundle.json').read_text())
        vid = selected.get(sid, bundle['views'][0]['id'])
        view = next(v for v in bundle['views'] if v['id'] == vid)
        row = Image.new('RGB', (1440, 430), '#202329')
        draw = ImageDraw.Draw(row)
        for col, (label, path) in enumerate([
            ('Reference photograph', Path(view['source_image'])),
            ('Shared editable 3D (estimated layout)', folder/f'rgb_{vid}.png')]):
            draw.text((col*720+10, 8), f'{sid} / {vid} / {label}', fill='white')
            with Image.open(path) as im:
                tile = ImageOps.contain(im.convert('RGB'), (708, 392))
                row.paste(tile, (col*720+6+(708-tile.width)//2, 31+(392-tile.height)//2))
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('all_shops_review.jpg', 'previous_vs_current.jpg', 'summary.json', 'initial_render_summary.json',
                 'iteration_metrics.json', 'comparison_metrics.json', 'baseline_metrics.json',
                 'replay_receipt.json', 'replay_verification.json'):
        if (batch/name).is_file():
            shutil.copy2(batch/name, out/name)
    for sid in summary:
        target = out/sid
        target.mkdir(exist_ok=True)
        for name in ('all_views_review.jpg', 'previous_vs_current.jpg', 'verification.json', 'retail_refinement.json'):
            if (batch/sid/name).is_file():
                shutil.copy2(batch/sid/name, target/name)
    gallery = Image.new('RGB', (1440, 430*len(rows)), '#202329')
    for index, row in enumerate(rows):
        gallery.paste(row, (0, 430*index))
    gallery.save(out/'demo_showcase.jpg', quality=92)
    files = {str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'package_manifest.json'}
    (out/'package_manifest.json').write_text(json.dumps({'source': str(batch), 'files_sha256': files}, indent=2)+'\n')
    print('Packaged', len(summary), 'shops at', out)


if __name__ == '__main__':
    main()
