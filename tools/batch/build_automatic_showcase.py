"""Build presentation images from a complete batch report, with a quality gate.

Default publication requires every store and view to pass. --diagnostic creates
explicitly labelled review material; it never presents a failed batch as final.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


BG='#101827'
FG='#f0f4fa'
ACCENT='#68dfcc'


def font(size):
    for path in ('/System/Library/Fonts/STHeiti Medium.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).is_file():return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)


def paste_fit(canvas,path,rect):
    x,y,w,h=rect
    with Image.open(path) as source:
        image=ImageOps.contain(source.convert('RGB'),(w,h))
        canvas.paste(image,(x+(w-image.width)//2,y+(h-image.height)//2))


def floorplan(store,output):
    path=Path(store['scene_bundle']);bundle=json.loads(path.read_text())
    base=bundle['shared_scene']['base_scene'];room=base['room']
    semantic={};layout={}
    layout_path=Path(store['library_layout'])
    if layout_path.is_file():
        layout=json.loads(layout_path.read_text());semantic={o['id']:o['category'] for o in layout['objects']}
    canvas=Image.new('RGB',(1600,1200),BG);draw=ImageDraw.Draw(canvas)
    draw.text((55,35),'共享布局与机位',font=font(45),fill=FG)
    draw.text((55,100),store['scene_id']+' · 同一个场景，全部相机共同使用',font=font(28),fill=ACCENT)
    scale=min(1320/room['width'],850/room['depth'])
    left=(1600-room['width']*scale)/2;top=180
    def xy(x,y):return left+(x+room['width']/2)*scale,top+(room['depth']-y)*scale
    draw.rectangle([xy(-room['width']/2,room['depth']),xy(room['width']/2,0)],fill='#1c2a3c',outline='#8096b0',width=4)
    for structure in layout.get('retail_details',{}).get('structures',[]):
        x,y,z=structure['center_m'];w,d,h=structure['size_m']
        if z-h/2>1.8:continue
        angle=math.radians(structure['rotation_z_deg'])
        corners=[xy(x+u*math.cos(angle)-v*math.sin(angle),y+u*math.sin(angle)+v*math.cos(angle)) for u,v in [(-w/2,-d/2),(w/2,-d/2),(w/2,d/2),(-w/2,d/2)]]
        draw.polygon(corners,fill='#b5c1cf')
    colors={'shelf':'#3c9d98','cold':'#65a9d0','checkout':'#d6a766','other':'#8290a5'}
    for obj in base['objects']:
        x,y,z=obj['position'];w,d,h=obj['dimensions'];angle=math.radians(obj['rotation_z_deg'])
        category=semantic.get(obj['id'],'')
        kind='cold' if any(s in category for s in ('refrigerator','freezer','chiller')) else 'checkout' if 'checkout' in category else 'shelf' if any(s in category for s in ('shelf','rack','display')) else 'other'
        corners=[xy(x+u*math.cos(angle)-v*math.sin(angle),y+u*math.sin(angle)+v*math.cos(angle)) for u,v in [(-w/2,-d/2),(w/2,-d/2),(w/2,d/2),(-w/2,d/2)]]
        draw.polygon(corners,fill=colors[kind],outline=BG)
    for i,view in enumerate(bundle['views']):
        p=view['camera']['position'];target=view['camera']['target'];x,y=xy(p[0],p[1])
        dx,dy=target[0]-p[0],target[1]-p[1];length=math.hypot(dx,dy)
        if length:
            end=xy(p[0]+dx/length*.65,p[1]+dy/length*.65)
            draw.line([(x,y),end],fill='#ffe1a0',width=4)
        draw.ellipse((x-10,y-10,x+10,y+10),fill='#ffe1a0')
        draw.text((x+12,y-17),str(i+1),font=font(24),fill=FG)
    for i,(kind,label) in enumerate([('shelf','陈列货架'),('cold','冷藏设备'),('checkout','收银台'),('other','其他物体')]):
        x=95+i*350;draw.rectangle((x,1090,x+30,1120),fill=colors[kind]);draw.text((x+44,1088),label,font=font(26),fill=FG)
    draw.text((55,1150),'由生成场景直接绘制；布局尺度为照片估计，非实测平面图。',font=font(23),fill='#9db0c9')
    canvas.save(output)
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest(),
            **({str(layout_path):hashlib.sha256(layout_path.read_bytes()).hexdigest()} if layout_path.is_file() else {})}


def build(report_path,out,diagnostic=False):
    report=json.loads(report_path.read_text())
    stores=report['stores'];views=report['views']
    if not stores or len({s['scene_id'] for s in stores})!=len(stores):
        raise ValueError('A nonempty, unique complete store denominator is required')
    if len(stores)!=report['input_stores']:raise ValueError('Store denominator differs from report')
    passed=0
    for store in stores:
        selected=[v for v in views if v['scene_id']==store['scene_id']]
        if len({v['view_id'] for v in selected})!=len(selected):raise ValueError('Duplicate view')
        good=(store['status']=='accepted' and store.get('gate',{}).get('accepted') is True
              and len(selected)==store['input_views']==store['rendered_views'] and not store.get('diagnostic_only'))
        passed+=good
        if not diagnostic and not good:raise ValueError('Final publication requires all stores and views to pass: '+store['scene_id'])
    if out.exists():raise ValueError('Use a new showcase folder; existing presentation is preserved')
    out.mkdir(parents=True)
    assets=out/'assets';assets.mkdir()
    source_hashes={};panels=[]
    lines=['# Scene Factory · 批量三维场景生产','',
           '从同店多机位照片，自动生成一份可编辑三维场景。','']
    if diagnostic:
        lines+=['**诊断材料：本批仍有任务未通过质量验收，以下图片不是合格交付结果。**','']
    lines += [f'本批输入 {len(stores)} 家店、{sum(s["input_views"] for s in stores)} 个机位；自动质量通过 {passed} 家。','']
    for store in stores:
        sid=store['scene_id'];items=[v for v in views if v['scene_id']==sid]
        lines += [f'## 店铺 {sid}','',f'输入 {store["input_views"]} 个机位，已生成 {len(items)} 个对应视角。状态：{store["status"]}。','']
        if not items:
            lines += ['该店暂无可展示渲染。',''];continue
        indices=sorted({round(i*(len(items)-1)/min(2,len(items)-1)) for i in range(min(3,len(items)))}) if len(items)>1 else [0]
        selected=[items[i] for i in indices]
        canvas=Image.new('RGB',(2160,160+640*len(selected)),BG);draw=ImageDraw.Draw(canvas)
        draw.text((45,27),f'{sid} · ONE SHARED 3D SCENE',font=font(38),fill=FG)
        draw.text((45,90),'原始照片',font=font(30),fill=ACCENT)
        draw.text((1110,90),'自动批量生成'+(' · 未通过验收' if store['status']!='accepted' else ''),font=font(30),fill=ACCENT)
        for i,view in enumerate(selected):
            top=150+i*640
            for col,key in enumerate(('source','render')):
                path=Path(view[key]);source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                paste_fit(canvas,path,(40+1080*col,top,1000,563))
            draw.text((45,top+580),view['view_id'],font=font(23),fill='#9db0c9')
        path=assets/(sid+'.jpg');canvas.save(path,quality=94)
        lines += [f'![店铺 {sid} 原图与自动场景对照](assets/{sid}.jpg)','']
        if store.get('scene_bundle'):
            source_hashes.update(floorplan(store,assets/(sid+'_layout.png')))
            lines += [f'![店铺 {sid} 共享布局与全部机位](assets/{sid}_layout.png)','']
        panels.append((sid,selected[0]['render']))
    if panels:
        cols=2;rows=(len(panels)+1)//2
        hero=Image.new('RGB',(2400,260+760*rows),BG);draw=ImageDraw.Draw(hero)
        draw.text((70,45),'SCENE FACTORY',font=font(66),fill=FG)
        draw.text((70,140),'照片 → 共享布局 → 真实几何 → 多机位渲染',font=font(37),fill=ACCENT)
        if diagnostic:draw.text((1550,65),'自动批量验证 · 未验收',font=font(31),fill='#ffbc72')
        for i,(sid,path) in enumerate(panels):
            x=60+(i%cols)*1180;y=250+(i//cols)*760
            paste_fit(hero,path,(x,y,1120,630));draw.text((x,y+650),sid,font=font(31),fill=FG)
        hero.save(assets/'batch_hero.jpg',quality=94)
        lines[4:4]=['![批量场景总览](assets/batch_hero.jpg)','']
    lines += ['## 批量生产流程','',
        '1. **组织照片**：按店铺汇集原图，保留来源、机位与图像尺寸。','',
        '2. **联合分析**：本地 Codex 识别共享房间、通道、大件与相机，输出结构化场景描述。','',
        '3. **选择资产与构建几何**：匹配资产库，按照片推断参数构建货架、冷柜及实体商品，检查支撑、碰撞和边界。','',
        '4. **输出全部机位**：共享几何只构建一次，使用各机位相机渲染并输出标注。','',
        '5. **质量复核与自动修正**：逐张对照原图，预算内修正并重建；未通过任务保留状态及原因。','',
        '6. **汇总批次**：以全部输入店铺统计完成与质量结果，同时管理并发、磁盘空间、请求预算和中断恢复。','',
        '所有店铺使用同一生产程序和通用配置。相机切换不改变场景几何；商品包装为通用近似，精确 SKU 和不可见结构不作恢复承诺。','']
    (out/'Scene Factory 批量场景生产与效果展示.md').write_text('\n'.join(lines))
    (out/'sources.json').write_text(json.dumps({'batch_report':str(report_path.resolve()),
        'batch_report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'diagnostic_only':diagnostic,'quality_passed':passed,'input_stores':len(stores),
        'images_sha256':source_hashes},ensure_ascii=False,indent=2))
    return {'directory':str(out),'quality_passed':passed,'input_stores':len(stores),'diagnostic_only':diagnostic}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--diagnostic',action='store_true')
    args=parser.parse_args()
    print(json.dumps(build(args.report,args.out,args.diagnostic),ensure_ascii=False))
