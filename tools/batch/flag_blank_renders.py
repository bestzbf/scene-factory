"""Flag rendered cameras that show almost nothing (a wall, the inside of an object).

    python flag_blank_renders.py

Reads review_20261102/rendered_index.json, measures every rgb_view_*.png, and writes
review_20261102/blank_views.json. A view counts as blank when its downscaled grayscale has
very low spread and almost no edges; a shop is broken when most of its views are blank.
Read-only on the runs.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / 'weidian_full_runs' / 'review_20261102'


def measure(path):
    gray = np.asarray(Image.open(path).convert('L').resize((160, 90)), dtype=np.float32)
    edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
    return float(gray.std()), float(edges)


def main():
    index = json.loads((OUT / 'rendered_index.json').read_text(encoding='utf-8'))
    shops = {}
    for shop_id, info in index['scenes'].items():
        folder = Path(info['folder'])
        views = sorted(folder.glob('rgb_view_*.png'))
        blank = []
        for view in views:
            spread, edges = measure(view)
            if spread < 12 and edges < 3:
                blank.append(view.stem.replace('rgb_', ''))
        shops[shop_id] = {'source': info['source'], 'views': len(views), 'blank': blank}
    broken = {k: v for k, v in shops.items() if v['views'] and len(v['blank']) / v['views'] >= 0.5}
    partial = {k: v for k, v in shops.items() if v['blank'] and k not in broken}
    views = sum(v['views'] for v in shops.values())
    blank_views = sum(len(v['blank']) for v in shops.values())
    by_source = {}
    for v in broken.values():
        by_source[v['source']] = by_source.get(v['source'], 0) + 1
    summary = {'shops': len(shops), 'views': views, 'blank_views': blank_views,
               'broken_shops': len(broken), 'partial_shops': len(partial),
               'broken_by_source': by_source, 'broken': broken, 'partial': partial}
    (OUT / 'blank_views.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('broken', 'partial')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
