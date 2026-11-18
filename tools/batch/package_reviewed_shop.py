"""Package all reviewed views with references, two baselines and image metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw
from make_packaging_atlas import font
from score_alignment import grey, structural


def package(current, baseline, previous, out, labels, groups, preview_views):
    bundle=json.loads((current/'scene_bundle.json').read_text())
    views={v['id']:v for v in bundle['views']}
    if any(v not in views for group in list(groups.values())+[preview_views] for v in group):
        raise ValueError('Unknown comparison view')
    out.mkdir(parents=True,exist_ok=True);rows={};previous_rows={};metrics=[];paths=[]
    def save(im,name):
        path=out/name;im.save(path,quality=93);paths.append(path)
    for vid,view in views.items():
        ref=Path(view['source_image']);new=current/f'rgb_{vid}.png';old=baseline/f'rgb_{vid}.png'
        row=Image.new('RGB',(1920,400),'#202329');draw=ImageDraw.Draw(row)
        for i,(label,path) in enumerate([('原图',ref),(labels[0],old),(labels[1]+' · 真实 3D',new)]):
            with Image.open(path) as im:row.paste(im.convert('RGB').resize((628,353),Image.Resampling.LANCZOS),(i*640+6,40))
            draw.text((i*640+10,6),f'{vid}  {label}',font=font(20),fill='white')
        rows[vid]=row;reference=grey(ref)
        record={'view':vid,'baseline':structural(grey(old),reference),'current':structural(grey(new),reference)}
        if previous:
            old2=previous/f'rgb_{vid}.png';second=row.copy();d=ImageDraw.Draw(second)
            d.rectangle((640,0,1279,399),fill='#202329');d.text((650,6),f'{vid}  {labels[2]}',font=font(20),fill='white')
            with Image.open(old2) as im:second.paste(im.convert('RGB').resize((628,353),Image.Resampling.LANCZOS),(646,40))
            previous_rows[vid]=second;record['previous']=structural(grey(old2),reference)
        metrics.append(record)
        with Image.open(new) as im:save(im.convert('RGB'),vid+'.jpg')
    def sheet(selected,source,name):
        im=Image.new('RGB',(1920,400*len(selected)),'#202329')
        for i,v in enumerate(selected):im.paste(source[v],(0,400*i))
        save(im,name+'.jpg')
    sheet(list(views),rows,'all_10_comparison' if len(views)==10 else 'all_views_comparison')
    for name,selected in groups.items():sheet(selected,rows,name)
    if previous:sheet(list(views),previous_rows,'previous_vs_current')
    if preview_views:
        im=Image.new('RGB',(1280,400*len(preview_views)),'#202329')
        for i,v in enumerate(preview_views):
            im.paste(rows[v].crop((0,0,640,400)),(0,400*i));im.paste(rows[v].crop((1280,0,1920,400)),(640,400*i))
        save(im,'result_preview.jpg')
    keys=['baseline','current']+(['previous'] if previous else [])
    result={'scope':'Blurred gradient correlation/direction against photographs; not measured geometry accuracy. All views retained.',
            'labels':dict(zip(['baseline','current','previous'],labels)),
            'means':{k:statistics.mean(x[k] for x in metrics) for k in keys},
            'improved_vs_baseline':sum(x['current']>x['baseline'] for x in metrics),
            'views':metrics}
    if previous:result['improved_vs_previous']=sum(x['current']>x['previous'] for x in metrics)
    (out/'comparison_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    manifest={'current':str(current.resolve()),'baseline':str(baseline.resolve()),
              'previous':str(previous.resolve()) if previous else None,
              'images':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('current','baseline','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--previous',type=Path)
    p.add_argument('--labels',nargs=3,default=['早期版本','本轮','上一轮'])
    p.add_argument('--group',action='append',default=[],help='NAME=view_01,view_02')
    p.add_argument('--preview-views',nargs='*',default=[])
    a=p.parse_args();groups={}
    for value in a.group:
        name,ids=value.split('=',1)
        if Path(name).name!=name or name in ('.','..'):p.error('Invalid output group name')
        groups[name]=ids.split(',')
    result=package(a.current,a.baseline,a.previous,a.out,a.labels,groups,a.preview_views)
    print(json.dumps({'means':result['means'],'views':len(result['views'])},ensure_ascii=False))
