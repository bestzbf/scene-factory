#!/usr/bin/env python3
"""Create a local reference/render comparison from actual grouped batch outputs."""
import argparse
import html
import json
from pathlib import Path
import urllib.parse


def report(directory, max_contact_views=24):
    from PIL import Image, ImageDraw, ImageOps
    directory=Path(directory).resolve()
    summary=json.loads((directory/'summary.json').read_text())
    order={'computer_lab':0,'corridor':1,'dorm':2}
    names={'computer_lab':'机房（两图共用一个场景）','corridor':'走廊','dorm':'宿舍'}
    rows=[];cards=[];view_count=0
    if type(max_contact_views) is not int or not 1 <= max_contact_views <= 100:
        raise ValueError("max_contact_views must be 1..100")
    def link(path):
        return urllib.parse.quote(str(Path(path).relative_to(directory)),safe='/')
    for result in sorted(summary['scenes'],key=lambda r:order.get(r['scene_id'],10)):
        folder=directory/result['scene_id']
        title=names.get(result['scene_id'],result['scene_id'])
        if not result.get('iteration'):
            cards.append('<section><h2>'+html.escape(title)+'</h2><p>任务未完成：'+html.escape(result.get('error',''))+'</p></section>')
            continue
        iteration=folder/result['iteration']
        bundle=json.loads((iteration/'scene_bundle.json').read_text())
        review_path=iteration/'review.json'
        review=json.loads(review_path.read_text()) if review_path.exists() else {}
        review_text=review.get('summary','未经过自动视觉审查。')
        blocks=[]
        for view in bundle['views']:
            original=Path(view['source_image'])
            rendered=iteration/f'rgb_{view["id"]}.png'
            if not rendered.exists():continue
            row=Image.new('RGB',(1600,640),(27,30,34))
            draw=ImageDraw.Draw(row)
            for x,path,label in [(0,original,'REFERENCE'),(800,rendered,'BATCH RENDER')]:
                im=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
                im.thumbnail((790,592))
                row.paste(im,(x+(800-im.width)//2,40+(592-im.height)//2))
                draw.text((x+12,12),f'{result["scene_id"]} / {view["id"]} / {label}',fill='white')
            view_count += 1
            if len(rows) < max_contact_views:
                rows.append(row)
            initial=folder/'iteration_00'/f'rgb_{view["id"]}.png'
            blocks.append(f'''<article><h3>{html.escape(view['id'])}</h3><div class="pair">
<figure><figcaption>原照片</figcaption><a href="{link(original)}"><img loading="lazy" src="{link(original)}"></a></figure>
<figure><figcaption>批量生成 · {html.escape(result['iteration'])}</figcaption><a href="{link(rendered)}"><img loading="lazy" src="{link(rendered)}"></a></figure></div>
<div class="links"><a href="{link(iteration/f'annotations_{view["id"]}.json')}">相机与对象标注</a></div></article>''')
        issues=''.join('<li>'+html.escape(i['evidence'])+'</li>' for i in review.get('issues',[])[:6])
        cards.append(f'''<section><h2>{html.escape(title)}</h2><p class="status">状态：{html.escape(result['status'])}</p>
<p>{html.escape(review_text)}</p><div class="links"><a href="{link(iteration/'scene.blend')}">下载可编辑 Blender 文件</a><a href="{link(iteration/'scene_bundle.json')}">最终场景参数</a><a href="{link(folder/'analysis_provenance.json')}">自动分析来源记录</a></div>
{''.join(blocks)}<details><summary>自动审查发现的差异</summary><ul>{issues}</ul></details></section>''')
    if rows:
        sheet=Image.new('RGB',(1600,len(rows)*640),(27,30,34))
        for i,row in enumerate(rows):sheet.paste(row,(0,i*640))
        sheet.save(directory/'comparison.jpg',quality=94)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>批量照片重建 · 原图对照</title>
<style>body{font:16px/1.65 system-ui,sans-serif;margin:0;background:#101318;color:#e7eaf0}main{max-width:1500px;margin:auto;padding:32px}h1{font-size:30px}h2{font-size:24px;margin:0}p{max-width:1000px;color:#bec7d5}.lead{font-size:18px}.status{color:#e9be78}section{padding:26px;background:#1a202a;margin:28px 0;border-radius:12px}article{margin-top:24px}h3{margin-bottom:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0}img{width:100%;display:block;border-radius:5px}figcaption{font-size:14px;color:#a5b4c8;margin-bottom:7px}.links{display:flex;flex-wrap:wrap;gap:18px;margin:14px 0}a{color:#80c6ff}details{margin-top:20px}li{margin:8px 0}@media(max-width:760px){main{padding:16px}section{padding:16px}.pair{grid-template-columns:1fr}}</style>
<main><h1>批量场景 · 原照片与最终效果</h1><p class="lead">左侧为原照片，右侧为最终 Blender 渲染。同组照片共用一套几何、材质和灯光，每张照片保留对应相机。</p>
<p>本次使用自动读图和自动修正，无人工场景参数种子。尺寸、遮挡后结构、相机和光照均为估计。needs_review 表示仍有明显差异；生成成功不等于通过还原验收。点击图片查看原尺寸。</p>'''+''.join(cards)+'''<p><a href="summary.json">批量汇总</a> · <a href="groups.json">输入分组</a> · <a href="comparison.jpg">完整对照图</a></p></main></html>'''
    page = page.replace('</h1>', f'</h1><p>{len(summary["scenes"])} 个场景 · {view_count} 个已渲染视角。对比总图最多展示前 {max_contact_views} 个视角，网页包含全部结果。</p>', 1)
    (directory/'comparison.html').write_text(page,encoding='utf-8')
    return directory/'comparison.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(report(parser.parse_args().directory))
