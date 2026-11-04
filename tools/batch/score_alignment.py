"""Score how well each rendered view matches its photograph structurally, not photometrically.

    python score_alignment.py [--top 30]

Luminance and saturation errors say nothing about whether the shelves are in the right
places: a wrong layout can still average to the right brightness. This compares the
gradient field of a blurred grey pair (edge direction and strength on a coarse grid), which
responds to structure -- aisles, shelf runs, counters -- and is largely blind to colour and
to the exact shelf contents. Runs over every rendered shop already on disk; read-only.
"""
import argparse
import json
from pathlib import Path

import os

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
REVIEW = Path(os.environ.get('REVIEW', HERE / 'runs' / 'review'))


def grey(path, size=(320, 180)):
    image = Image.open(path).convert('L').resize(size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(2.5))
    return np.asarray(image, dtype=np.float32) / 255.0


def gradient(a):
    gx = np.zeros_like(a); gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def structural(a, b):
    """Correlation of gradient magnitude plus cosine agreement of gradient direction."""
    ax, ay, am = gradient(a)
    bx, by, bm = gradient(b)
    if am.std() < 1e-6 or bm.std() < 1e-6:
        return 0.0
    mag_corr = float(np.corrcoef(am.ravel(), bm.ravel())[0, 1])
    # direction agreement weighted by where both images have structure
    weight = (am * bm)
    denom = np.maximum(am * bm, 1e-9)
    cosine = ((ax * bx + ay * by) / denom)
    direction = float((np.clip(cosine, -1, 1) * weight).sum() / max(weight.sum(), 1e-9))
    return float(0.5 * mag_corr + 0.5 * direction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=25)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = REVIEW.parent / 'alignment_scores.json'
    index = json.loads((REVIEW / 'rendered_index.json').read_text(encoding='utf-8'))
    blank = json.loads((REVIEW / 'blank_views.json').read_text(encoding='utf-8'))
    rows = []
    for shop, info in index['scenes'].items():
        if shop in blank['broken']:
            continue
        folder = Path(info['folder'])
        try:
            bundle = json.loads((folder / 'scene_bundle.json').read_text(encoding='utf-8'))
            objects = len(json.loads((folder / 'asset_bindings.json').read_text(encoding='utf-8')).get('instances', []))
        except (OSError, ValueError):
            continue
        scores = []
        for view in bundle['views']:
            render = folder / f"rgb_{view['id']}.png"
            photo = Path(view['source_image'])
            if not (render.is_file() and photo.is_file()):
                continue
            scores.append(structural(grey(render), grey(photo)))
        if not scores:
            continue
        rows.append({'shop': shop, 'source': info['source'], 'objects': objects,
                     'cameras': len(bundle['views']),
                     'mean': round(float(np.mean(scores)), 4),
                     'best': round(float(np.max(scores)), 4),
                     'best_view': bundle['views'][int(np.argmax(scores))]['id']})
    rows.sort(key=lambda r: -r['mean'])
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'scored {len(rows)} shops; wrote {args.out}')
    print(f"{'shop':12s} {'source':12s} {'obj':>4s} {'cam':>4s} {'mean':>7s} {'best':>7s}")
    for r in rows[:args.top]:
        print(f"{r['shop']:12s} {r['source']:12s} {r['objects']:4d} {r['cameras']:4d} {r['mean']:7.3f} {r['best']:7.3f}")


if __name__ == '__main__':
    main()
