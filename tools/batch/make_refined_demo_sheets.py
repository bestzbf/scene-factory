"""Compare the actual rendered subset: photo / baseline / refined, without cropping.

python tools/batch/make_refined_demo_sheets.py --before demo_v3 --after demo_v4 --out sheets
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from score_alignment import grey, structural
from tune_look import metrics

BG, FG, DIM = '#101722', '#f0f3f7', '#afbbca'


def font(size):
    for path in ('/System/Library/Fonts/STHeiti Medium.ttc',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def panel(path, width, height):
    with Image.open(path) as original:
        scaled = ImageOps.contain(original.convert('RGB'), (width, height), Image.Resampling.LANCZOS)
    frame = Image.new('RGB', (width, height), BG)
    frame.paste(scaled, ((width-scaled.width)//2, (height-scaled.height)//2))
    return frame


def sheet(rows, out, width=640, pair=False):
    columns = [('photo', '原始照片'), ('before', '旧版 v3'), ('after', '新版 v4')]
    if pair:
        columns = [columns[0], columns[2]]
    gap, margin, heading, caption = 12, 20, 110, 42
    height = round(width*9/16)
    row_height = height+caption+18
    canvas = Image.new('RGB', (margin*2+len(columns)*width+(len(columns)-1)*gap,
                               heading+len(rows)*row_height+44), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), 'Scene Factory · Demo v4', font=font(27), fill=FG)
    draw.text((margin, 52), '包装印刷 / 边缘高光 / 表面细节 · 同机位对比', font=font(17), fill=DIM)
    for j, (_, title) in enumerate(columns):
        draw.text((margin+j*(width+gap), 82), title, font=font(18), fill=FG)
    for i, row in enumerate(rows):
        y = heading+i*row_height
        draw.text((margin, y+5), f'店铺 {row["shop"]}   {row["view"]}', font=font(18), fill=FG)
        for j, (key, _) in enumerate(columns):
            canvas.paste(panel(row[key], width, height), (margin+j*(width+gap), y+caption))
    draw.text((margin, canvas.height-32), '估计布局；包装为通用设计，地砖尺度为假设。未恢复真实商品身份。',
              font=font(15), fill=DIM)
    canvas.save(out, quality=93, subsampling=0)


def build(before, after, out):
    before, after, out = (Path(p).resolve() for p in (before, after, out))
    if out == before or out == after or out.is_relative_to(before) or out.is_relative_to(after):
        raise ValueError('Use a separate sheet output directory')
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for folder in sorted(after.iterdir()):
        manifest = folder/'refinement.json'
        if not folder.is_dir() or not manifest.is_file():
            continue
        report = json.loads(manifest.read_text())
        bundle = json.loads((before/folder.name/'scene_bundle.json').read_text())
        views = {v['id']: v for v in bundle['views']}
        for rendered in report['views']:
            view_id = rendered['id']
            if rendered['camera_overrides']:
                raise ValueError(f'Camera changed: {folder.name}/{view_id}')
            paths = {'photo': Path(views[view_id]['source_image']),
                     'before': before/folder.name/('rgb_'+view_id+'.png'),
                     'after': folder/('rgb_'+view_id+'.png')}
            measured = {}
            for key in ('before', 'after'):
                measured[key] = metrics(paths[key], paths['photo'])
                measured[key]['structural'] = round(structural(grey(paths[key]), grey(paths['photo'])), 4)
            with Image.open(paths['after']) as im:
                pixels = np.asarray(im.convert('RGB'), dtype=np.float32)
                measured['after']['size'] = list(im.size)
                measured['after']['rgb_std'] = round(float(pixels.std()), 3)
                if min(im.size)<64 or pixels.std()<2:
                    raise ValueError(f'Blank or invalid output: {paths["after"]}')
            records.append({'shop':folder.name, 'view':view_id,
                            **{k:str(v) for k,v in paths.items()}, 'metrics':measured})
    if not records:
        raise ValueError('No refined renders found')
    order = {'1660108397':0, '1649397835':1, '1646301141':2}
    first = sorted([r for r in records if r['view']=='view_01'], key=lambda r:order.get(r['shop'],99))
    sheet(first, out/'before_after.jpg')
    sheet(first, out/'demo_all.jpg', pair=True)
    for shop in sorted({r['shop'] for r in records}):
        sheet([r for r in records if r['shop']==shop], out/f'shop_{shop}.jpg')
    alternate = [r for r in records if r['view']!='view_01']
    if alternate:
        sheet(alternate, out/'alternate_views.jpg', pair=True)
    summary = {'rendered_views':len(records), 'shops':len({r['shop'] for r in records}),
               'camera_changes_in_final':False, 'views':records,
               'limitations':'Image gradient and appearance metrics are not calibrated geometric accuracy.'}
    for key in ('before','after'):
        subset = [r['metrics'][key] for r in first]
        means = {k:round(float(np.mean([v[k] for v in subset])),4)
                 for k in ('lum_err','sat_err','ssim','structural')}
        summary[key+'_first_view_mean'] = means
        print(key, means)
    (out/'comparison_metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Wrote', len(records), 'view comparisons to', out)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('before','after','out'):
        p.add_argument('--'+key, required=True, type=Path)
    args=p.parse_args()
    build(args.before, args.after, args.out)
