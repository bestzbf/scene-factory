"""Build a transparent photo / v4 / manual-v5 review for one reconstructed shop."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from make_refined_demo_sheets import font,panel,BG,FG,DIM
from score_alignment import grey,structural
from tune_look import metrics


def draw_sheet(rows,out,pair=False):
    width=800 if pair else 640;height=round(width*9/16);margin=20;gap=12
    columns=[('photo','原始照片'),('before','v4：材质优化'),('after','v5：单店 3D 人工修正')]
    if pair:columns=[columns[0],columns[2]]
    canvas=Image.new('RGB',(2*margin+len(columns)*width+(len(columns)-1)*gap,
                              112+len(rows)*(height+48)+56),BG)
    d=ImageDraw.Draw(canvas)
    d.text((margin,12),'Scene Factory · 店铺 1646301141',font=font(27),fill=FG)
    d.text((margin,49),'货架 / 通道 / 商品分区 / 后部堆货 · 同一 3D 场景的双机位检查',font=font(18),fill=DIM)
    for col,(_,label) in enumerate(columns):d.text((margin+col*(width+gap),82),label,font=font(18),fill=FG)
    for i,row in enumerate(rows):
        y=112+i*(height+48)
        d.text((margin,y+5),row['view'],font=font(18),fill=FG)
        for col,(key,_) in enumerate(columns):canvas.paste(panel(row[key],width,height),(margin+col*(width+gap),y+35))
    d.text((margin,canvas.height-40),'人工估计相机和尺寸；未恢复真实 SKU。仍有布局与内容差异，非自动重建质量证明。',font=font(16),fill=DIM)
    canvas.save(out,quality=94,subsampling=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['before','after','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--views',nargs='+',default=['view_01','view_08'])
    args=p.parse_args();out=args.out.resolve()
    for source in [args.before.resolve(),args.after.resolve()]:
        if out==source or out.is_relative_to(source):raise ValueError('Review output must be separate from renders')
    out.mkdir(parents=True,exist_ok=True)
    report=json.loads((args.after/'layout_refinement.json').read_text())
    if report.get('representation')!='physical_meshes_with_fixed_object_textures':
        raise ValueError('This review labels actual 3D geometry and cannot be used for layered-photo experiments')
    bundle=json.loads((args.after/'scene_bundle.json').read_text());views={v['id']:v for v in bundle['views']}
    rows=[]
    for vid in args.views:
        paths={'photo':views[vid]['source_image'],'before':str(args.before/f'rgb_{vid}.png'),
               'after':str(args.after/f'rgb_{vid}.png')}
        measured={}
        for key in ['before','after']:
            measured[key]={**metrics(paths[key],paths['photo']),
                           'structural':round(structural(grey(paths[key]),grey(paths['photo'])),4)}
            with Image.open(paths[key]) as image:
                if np.asarray(image).std()<2:raise ValueError('Blank output: '+paths[key])
                measured[key]['size']=list(image.size)
        rows.append({'view':vid,**paths,'metrics':measured})
    draw_sheet(rows,out/'before_after.jpg')
    draw_sheet(rows,out/'original_vs_v5.jpg',pair=True)
    for row in rows:draw_sheet([row],out/('pair_'+row['view']+'.jpg'),pair=True)
    record={'shop':bundle['scene_id'],'scope':'single-shop manual 3D refinement, two rendered views',
            'camera_changed':True,'texture_projection':False,'views':rows,
            'limitations':'Image similarity is affected by textures and lighting, not independently calibrated geometry.'}
    (out/'comparison_metrics.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps({'out':str(out),'metrics':{r['view']:r['metrics'] for r in rows}},ensure_ascii=False))


if __name__=='__main__':main()
