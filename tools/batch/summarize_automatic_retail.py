"""Audit every photo-group outcome and render matched comparison sheets.

Baseline renders are evaluation-only. They are never supplied to inference or
used as scene inputs. Scores are image diagnostics, not acceptance thresholds.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont, ImageOps
from score_alignment import grey, structural


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def font(size):
    for name in ('/System/Library/Fonts/STHeiti Medium.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(name).is_file():return ImageFont.truetype(name,size)
    return ImageFont.load_default(size=size)


def compare_sheet(sid, view, render, output):
    w,h=960,540
    canvas=Image.new('RGB',(2*w+120,h+170),'#101b2c')
    draw=ImageDraw.Draw(canvas)
    draw.text((40,22),sid+' / '+view['id'],font=font(30),fill='white')
    for col,(path,label) in enumerate([(Path(view['source_image']),'原始照片'),(render,'自动批量生成')]):
        x=40+col*(w+40)
        draw.text((x,75),label,font=font(26),fill='#78dbd0')
        with Image.open(path) as im:
            im=ImageOps.contain(im.convert('RGB'),(w,h))
            canvas.paste(im,(x+(w-im.width)//2,120+(h-im.height)//2))
    canvas.save(output,quality=94)


def summarize(batch, groups, out, baselines=None):
    out.mkdir(parents=True,exist_ok=True)
    records=[];views=[]
    for group in read(groups)['scenes']:
        sid=group['scene_id'];folder=batch/sid
        result=read(folder/'result.json') or read(folder/'last_error.json') or {'status':'pending'}
        iteration=result.get('iteration')
        target=folder/iteration if iteration else None
        diagnostic=False
        if target is None:
            # Keep failed and in-progress stores visible. A completed render is
            # useful diagnostic evidence even when the final quality gate fails.
            rendered=sorted(p.parent for p in folder.glob('iteration_*/render_state.json')
                            if (p.parent/'scene_bundle.json').is_file())
            if rendered:target=rendered[-1];diagnostic=True
        row={'scene_id':sid,'status':result.get('status','pending'),'input_views':len(group['images']),
             'elapsed_seconds':result.get('elapsed_seconds'),'gate':result.get('gate'),
             'error':result.get('error'),'rendered_views':0,'stock_units':0,
             'display_iteration':target.name if target else None,'diagnostic_only':diagnostic,
             'codex_requests':(read(folder/'scene_budget.json',{}) or {}).get('requests',0)}
        records.append(row)
        if target is None:continue
        bundle=read(target/'scene_bundle.json')
        if not bundle:continue
        row['scene_bundle']=str((target/'scene_bundle.json').resolve())
        row['library_layout']=str((target/'library_layout.json').resolve())
        row['stock_units']=read(target/'retail_refinement.json',{}).get('stock_units',0)
        review=read(target/'review.json',{})
        row['scores']=review.get('scores')
        for view in bundle['views']:
            render=target/('rgb_'+view['id']+'.png')
            if not render.is_file():continue
            reference=grey(view['source_image'])
            metric=structural(grey(render),reference)
            record={'scene_id':sid,'view_id':view['id'],'structure_against_photo':metric,
                    'source':view['source_image'],'render':str(render)}
            baseline=Path(baselines[sid])/render.name if baselines and sid in baselines else None
            if baseline and baseline.is_file():
                original=read(baseline.parent/'scene_bundle.json',{})
                old=next((v for v in original.get('views',[]) if v['id']==view['id']),None)
                if (old and Path(old['source_image']).is_file()
                        and hashlib.sha256(Path(old['source_image']).read_bytes()).digest()
                            ==hashlib.sha256(Path(view['source_image']).read_bytes()).digest()):
                    record['baseline_structure']=structural(grey(baseline),reference)
                    record['structure_delta']=metric-record['baseline_structure']
                    record['baseline_render']=str(baseline)
                    record['baseline_scope']='evaluation only; reference source hash verified; not a production input'
                else:
                    record['baseline_error']='Reference-photo identity differs or cannot be verified'
            sheet=out/(sid+'_'+view['id']+'.jpg')
            compare_sheet(sid,view,render,sheet);record['comparison']=sheet.name
            views.append(record);row['rendered_views']+=1
        measures=[v['structure_against_photo'] for v in views if v['scene_id']==sid]
        if measures:row['structure_mean']=statistics.mean(measures)
        deltas=[v['structure_delta'] for v in views if v['scene_id']==sid and 'structure_delta' in v]
        if deltas:row['structure_delta_against_demo']=statistics.mean(deltas)
    counts={s:sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})}
    complete=sum(r['rendered_views']==r['input_views'] and r['input_views']>0 for r in records)
    report={'scope':'All selected stores, original-photo inputs only; no store-specific scene/profile overrides. '
                    'Visual quality requires independent review; edge metrics cannot certify geometry.',
            'input_stores':len(records),'full_view_outputs':complete,'statuses':counts,
            'quality_passed':counts.get('accepted',0),'local_codex_requests':sum(r['codex_requests'] for r in records),
            'stores':records,'views':views}
    (out/'batch_quality.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    cards=''.join('<section><h2>'+html.escape(r['scene_id'])+' · '+html.escape(r['status'])+'</h2><p>'+str(r['rendered_views'])+'/'+str(r['input_views'])+' 个机位 · '+str(r['stock_units'])+' 个生成商品单元'+(' · 过程诊断图，尚非通过验收的交付结果' if r['diagnostic_only'] else '')+'</p>'+('<p>'+html.escape(r['error'])+'</p>' if r['error'] else '')+''.join('<img loading="lazy" src="'+v['comparison']+'">' for v in views if v['scene_id']==r['scene_id'])+'</section>' for r in records)
    metrics=[('输入店铺',str(len(records))),('完整机位输出',f'{complete} / {len(records)}'),
             ('质量通过',f'{report["quality_passed"]} / {len(records)}'),
             ('已输出机位',f'{len(views)} / {sum(r["input_views"] for r in records)}')]
    stats='<div class="stats">'+''.join('<div><strong>'+value+'</strong><span>'+label+'</span></div>' for label,value in metrics)+'</div>'
    page='<!doctype html><meta charset="utf-8"><title>Scene Factory 自动批量结果</title><style>body{margin:0;background:#0c1421;color:#e7edf5;font:18px system-ui}main{max-width:1440px;margin:auto;padding:56px}h1{font-size:46px}h2{margin-top:56px;color:#78dbd0}img{width:100%;margin:18px 0;border-radius:8px}p{color:#b5c6d8}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin:40px 0}.stats div{padding:26px;background:#18283b;border-radius:12px}.stats strong{display:block;font-size:38px;color:#78dbd0}.stats span{display:block;margin-top:12px;color:#b5c6d8}</style><main><h1>Scene Factory · 自动批量结果</h1><p>照片分组 → 共享场景 → 多机位输出 → 独立质量复核</p>'+stats+'<p>完整渲染与质量通过分别统计。未通过任务的图片用于检查问题，不能作为合格效果示例。</p>'+cards+'</main>'
    (out/'index.html').write_text(page)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--groups',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--baselines',type=Path)
    a=p.parse_args();r=summarize(a.batch,a.groups,a.out,read(a.baselines) if a.baselines else None)
    print(json.dumps({k:v for k,v in r.items() if k not in ('views','stores')},ensure_ascii=False))
