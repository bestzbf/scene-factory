"""Photo-vs-render review sheets for every rendered Weidian shop. Read-only on the runs.

    python make_review_sheets.py

Writes weidian_full_runs/review_20261102/: rendered_index.json (every shop with a complete
render, deduplicated across runs), overview.jpg (one camera per shop for a random sample),
and shop_<id>.jpg (up to four cameras, photo left / render right). Shops are drawn at random
with a fixed seed, so the sheets show the typical result rather than the best ones.
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'weidian_full_runs'
OUT = RUNS / 'review_20261102'
# Model runs keep renders in <scene>/iteration_00/, offline re-renders in <scene>/.
MODEL_RUNS = [('openai', 'openai_gpt61sol_full_20261030'), ('gateway1', 'weidian_gpt61sol_full_20261030'),
              ('gateway2a', 'weidian_left_20261031'), ('stuck', 'weidian_stuck_rerun_20261030'),
              ('gateway2', 'weidian_restart_20261031')]
OFFLINE_RUNS = [('offline_1032', 'offline_20261032_gateway'), ('offline_1031', 'offline_20261031_gateway'),
                ('offline_1031', 'offline_20261031_openai'), ('offline_1031', 'offline_20261031_trial')]
RENDERED = {'generated_unreviewed', 'accepted', 'needs_review'}
BG, FG, DIM = (14, 16, 20), (235, 235, 235), (150, 155, 165)


def font(size):
    for path in ('/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/STHeiti Medium.ttc',
                 '/System/Library/Fonts/Hiragino Sans GB.ttc'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def complete(folder):
    bundle = folder / 'scene_bundle.json'
    if not bundle.is_file():
        return None
    views = json.loads(bundle.read_text(encoding='utf-8'))['views']
    pairs = [(v['id'], Path(v['source_image']), folder / f"rgb_{v['id']}.png") for v in views]
    return pairs if all(r.is_file() and s.is_file() for _, s, r in pairs) else None


def inventory():
    shops = {}
    for tag, name in MODEL_RUNS:
        for scene in sorted(p for p in (RUNS / name).iterdir() if p.is_dir()):
            try:
                status = json.loads((scene / 'result.json').read_text(encoding='utf-8'))['status']
            except (OSError, ValueError, KeyError):
                continue
            pairs = complete(scene / 'iteration_00') if status in RENDERED else None
            if pairs and scene.name not in shops:
                shops[scene.name] = {'source': tag, 'folder': scene / 'iteration_00', 'pairs': pairs}
    for tag, name in OFFLINE_RUNS:
        root = RUNS / name
        for scene in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
            pairs = complete(scene)
            if pairs and scene.name not in shops:
                shops[scene.name] = {'source': tag, 'folder': scene, 'pairs': pairs}
    return shops


def stats(folder):
    try:
        report = json.loads((folder / 'asset_selection.json').read_text(encoding='utf-8'))
        bindings = json.loads((folder / 'asset_bindings.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    settling = report.get('settling') or {}
    return {'objects': len(bindings.get('instances', [])), 'moved': settling.get('moved_objects'),
            'omitted': settling.get('omitted_objects') or []}


def fit(path, width):
    image = Image.open(path).convert('RGB')
    return image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)


def shop_sheet(shop_id, shop, out, width=1600, rows=4):
    half = width // 2
    info = stats(shop['folder'])
    title = (f"店铺 {shop_id}   来源 {shop['source']}   机位 {len(shop['pairs'])}   "
             f"物体 {info.get('objects', '?')}   被省略 {len(info.get('omitted', []))}")
    blocks = []
    for view_id, photo, render in shop['pairs'][:rows]:
        a, b = fit(photo, half), fit(render, half)
        row = Image.new('RGB', (width, max(a.height, b.height) + 34), BG)
        row.paste(a, (0, 34)); row.paste(b, (half, 34))
        draw = ImageDraw.Draw(row)
        draw.text((10, 6), f'{view_id}  原始照片', fill=DIM, font=font(20))
        draw.text((half + 10, 6), f'{view_id}  重建渲染', fill=DIM, font=font(20))
        blocks.append(row)
    sheet = Image.new('RGB', (width, 56 + sum(b.height for b in blocks)), BG)
    ImageDraw.Draw(sheet).text((12, 12), title, fill=FG, font=font(26))
    y = 56
    for block in blocks:
        sheet.paste(block, (0, y)); y += block.height
    sheet.save(out, quality=85)


def overview(sample, shops, out, cols=6, tile=320):
    tiles = []
    for shop_id in sample:
        _, photo, render = shops[shop_id]['pairs'][0]
        a, b = fit(photo, tile), fit(render, tile)
        cell = Image.new('RGB', (tile, a.height + b.height + 30), BG)
        cell.paste(a, (0, 30)); cell.paste(b, (0, 30 + a.height))
        ImageDraw.Draw(cell).text((6, 4), f"{shop_id} · {shops[shop_id]['source']}", fill=DIM, font=font(16))
        tiles.append(cell)
    height = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * (tile + 8) + 8, 60 + rows * (height + 8)), BG)
    ImageDraw.Draw(sheet).text((12, 14), '每格上：原始照片（第一个机位）  下：重建渲染   随机抽样，未挑选', fill=FG, font=font(24))
    for i, cell in enumerate(tiles):
        sheet.paste(cell, (8 + (i % cols) * (tile + 8), 60 + (i // cols) * (height + 8)))
    sheet.save(out, quality=85)


def main():
    OUT.mkdir(exist_ok=True)
    shops = inventory()
    by_source = {}
    for shop in shops.values():
        by_source[shop['source']] = by_source.get(shop['source'], 0) + 1
    (OUT / 'rendered_index.json').write_text(json.dumps(
        {'shops': len(shops), 'by_source': by_source,
         'scenes': {k: {'source': v['source'], 'folder': str(v['folder']), 'cameras': len(v['pairs'])}
                    for k, v in sorted(shops.items())}}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('rendered shops:', len(shops), by_source)
    rng = random.Random(20261102)
    ids = sorted(shops)
    overview(rng.sample(ids, min(30, len(ids))), shops, OUT / 'overview.jpg')
    model = [s for s in ids if not shops[s]['source'].startswith('offline') and len(shops[s]['pairs']) >= 3]
    offline = [s for s in ids if shops[s]['source'].startswith('offline') and len(shops[s]['pairs']) >= 3]
    detail = rng.sample(model, min(3, len(model))) + rng.sample(offline, min(3, len(offline)))
    for shop_id in detail:
        shop_sheet(shop_id, shops[shop_id], OUT / f'shop_{shop_id}.jpg')
    print('detail sheets:', detail)


if __name__ == '__main__':
    main()
