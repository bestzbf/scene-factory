"""Create a deterministic, brand-free 8x8 packaging atlas for existing retail UVs.

Uses Pillow drawing or explicitly supplied photo ROIs; no API or generated photos.
Example: python make_packaging_atlas.py atlas.png --seed 17
Optional --photo-rois JSON: [{image: path, box: [x0,y0,x1,y1]}], in pixels.
Photo crops are appearance references, never evidence of recovered product identity.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


def font(size):
    for path in ('/System/Library/Fonts/STHeiti Medium.ttc',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def printed_tile(index, seed, size):
    rng = random.Random(seed * 1009 + index)
    colours = ['#b51d27', '#d52821', '#e6b322', '#e9e0bc', '#123f2b', '#1d6937',
               '#183e69', '#262125', '#912938', '#b57125', '#db5c20', '#e3d8b9']
    base = colours[index % len(colours)]
    tile = Image.new('RGB', (size, size), base); d = ImageDraw.Draw(tile)
    gold, cream, ink = '#d5b86c', '#f3e9cd', '#252223'
    x = lambda v: round(v * size)
    # Folded seams, border rules and offset colour panels suggest printed cartons.
    d.rectangle((x(.035), x(.025), x(.965), x(.975)), outline=gold, width=max(1,x(.008)))
    d.line((x(.06), x(.08), x(.94), x(.08)), fill=cream, width=max(1,x(.006)))
    d.rectangle((x(.09), x(.12), x(.91), x(.25)), fill=cream)
    names = ['精选', '醇香', '鲜果', '清茶', '谷物', '原味', '山泉', '香脆']
    title = names[index % len(names)]
    d.text((x(.5), x(.18)), title, font=font(x(.105)), anchor='mm', fill=ink)
    subtitle = ['PREMIUM', 'ORIGINAL', 'SELECTED', 'NATURAL'][index % 4]
    d.text((x(.5),x(.29)),subtitle,font=font(x(.047)),anchor='mm',fill=cream)
    if index % 4 == 0:
        # Gift-box medallion and vertical foil lines.
        d.ellipse((x(.24),x(.36),x(.76),x(.78)),outline=gold,width=x(.014))
        d.ellipse((x(.28),x(.40),x(.72),x(.74)),outline=gold,width=x(.004))
        d.text((x(.5),x(.56)),'礼',font=font(x(.23)),anchor='mm',fill=gold)
    elif index % 4 == 1:
        # Small citrus/leaf illustration with highlights and fine segments.
        d.ellipse((x(.24),x(.39),x(.78),x(.81)),fill='#e8a527',outline='#f5d480',width=x(.012))
        for angle in range(0,360,45):
            import math
            cx,cy=x(.51),x(.60)
            d.line((cx,cy,cx+math.cos(math.radians(angle))*x(.22),
                    cy+math.sin(math.radians(angle))*x(.17)),fill='#f9db8b',width=x(.01))
        d.ellipse((x(.48),x(.34),x(.79),x(.44)),fill='#42663c')
    elif index % 4 == 2:
        # Foil snack bag: photo-like shaded crisp shapes and scattered grains.
        for _ in range(13):
            cx,cy=rng.uniform(.25,.7),rng.uniform(.44,.77)
            w,h=rng.uniform(.08,.22),rng.uniform(.04,.10)
            d.ellipse((x(cx),x(cy),x(cx+w),x(cy+h)),fill=rng.choice(['#cc8d35','#e7b961','#f2d087']),outline='#94652d')
        for y in (.04,.94):
            for a in range(12):d.line((x(.08+a*.07),x(y),x(.09+a*.07),x(y+.03)),fill=gold,width=1)
    else:
        # Beverage label with fine type rather than broad abstract colour bars.
        d.rounded_rectangle((x(.2),x(.37),x(.8),x(.79)),radius=x(.04),fill=cream,outline=gold,width=x(.007))
        d.text((x(.5),x(.51)),'优选',font=font(x(.12)),anchor='mm',fill=base)
        d.text((x(.5),x(.65)),'QUALITY',font=font(x(.046)),anchor='mm',fill=ink)
        for k in range(3):d.line((x(.28),x(.70+k*.02),x(.72),x(.70+k*.02)),fill='#aaa18b',width=1)
    d.text((x(.1),x(.84)),f'{100 + index * 7}g  /  净含量',font=font(x(.035)),fill=cream)
    # Barcode and small print are intentionally generic, with no invented real brand.
    d.rectangle((x(.60),x(.87),x(.9),x(.94)),fill=cream)
    for a in range(x(.61),x(.89),max(2,x(.009))):
        if rng.random()>.25:d.line((a,x(.875),a,x(.925)),fill=ink,width=rng.choice([1,2,3]))
    d.text((x(.1),x(.91)),'品质保证',font=font(x(.032)),fill=cream)
    pixels = np.asarray(tile,dtype=np.float32)
    yy,xx = np.mgrid[:size,:size]
    # Ink grain and a narrow folded edge, not baked directional scene lighting.
    grain = np.random.default_rng(seed+index).normal(0,.65,(size,size,1))
    fold = 1 - .12*np.exp(-((xx/size-.045)/.014)**2) - .08*np.exp(-((xx/size-.96)/.01)**2)
    pixels = pixels*fold[...,None]+grain
    return Image.fromarray(np.clip(pixels,0,255).astype('uint8'))


def make_atlas(out, seed=17, tile_size=256, rois=None):
    out = Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    atlas = Image.new('RGB',(tile_size*8,tile_size*8))
    rng = random.Random(seed); sources=[]
    for index in range(64):
        if rois:
            roi=rois[index % len(rois)]
            with Image.open(roi['image']) as original:
                box=roi['box']
                if not (0 <= box[0] < box[2] <= original.width and 0 <= box[1] < box[3] <= original.height):
                    raise ValueError(f'Invalid ROI: {roi}')
                crop=original.convert('RGB').crop(box)
            # Pad the full crop; do not stretch a wide parcel into a tall package.
            background=crop.resize((1,1)).getpixel((0,0))
            tile=Image.new('RGB',(tile_size,tile_size),background)
            inset=round(tile_size*.04)
            face=ImageOps.contain(crop,(tile_size-2*inset,tile_size-2*inset),Image.Resampling.LANCZOS)
            tile.paste(face,((tile_size-face.width)//2,(tile_size-face.height)//2))
            tile=ImageEnhance.Color(tile).enhance(rng.uniform(.96,1.04))
            sources.append({'tile':index,**roi})
        else:
            tile=printed_tile(index,seed,tile_size)
        atlas.paste(tile,((index%8)*tile_size,(index//8)*tile_size))
    atlas.save(out)
    record={'mode':'reference_crops' if rois else 'procedural_brand_free_print',
            'seed':seed,'tiles':64,'tile_size':tile_size,'source_rois':sources,
            'limitations':'UV-compatible appearance detail. Not SKU recovery or geometric reconstruction.'}
    out.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    return record


def make_carton_faces(out,seed=17):
    """Generic straight-on kraft shipping print; no photographed scene occlusions."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for index in range(5):
        rng=random.Random(seed+index);w,h=768,640
        yy,xx=np.mgrid[:h,:w]
        noise=np.random.default_rng(seed+index).normal(0,1.6,(h,w,1))
        corrugation=np.sin(xx*.8)[...,None]*.55
        pixels=np.array([157+index*4,123+index*3,83+index*2])[None,None,:]+noise+corrugation
        im=Image.fromarray(np.clip(pixels,0,255).astype('uint8'));d=ImageDraw.Draw(im)
        ink=['#63312b','#3e3d31','#395147','#304953','#6e352d'][index]
        d.rectangle((29,27,w-30,h-27),outline=ink,width=3)
        for y in [75,83,h-90,h-82]:d.line((35,y,w-35,y),fill=ink,width=2)
        d.text((w//2,170),['精选酒水','食品整箱','饮品礼盒','仓储配送','优选食品'][index],font=font(75),fill=ink,anchor='mm')
        d.text((w//2,260),'HANDLE WITH CARE',font=font(31),fill=ink,anchor='mm')
        d.text((w//2,334),['净含量：500 mL × 6','规格：12 盒 / 箱','防潮  轻放  勿倒置','整箱包装  常温保存','PRODUCT STORAGE'][index],font=font(28),fill=ink,anchor='mm')
        d.text((54,460),'↑ ↑   防潮 / 易碎',font=font(42),fill=ink)
        d.text((56,530),'批次  2026    质量检验合格',font=font(23),fill=ink)
        d.rectangle((w-208,h-158,w-48,h-101),fill='#b9a687')
        for x in range(w-202,w-54,4):
            if rng.random()>.25:d.line((x,h-154,x,h-107),fill=ink,width=rng.choice([1,2,3]))
        im.save(out/f'carton_{index:02d}.png')
    for index,colour in enumerate(['#246b29','#b32725','#22558b','#d1a824']):
        im=Image.new('RGB',(768,640),colour);d=ImageDraw.Draw(im)
        d.rectangle((30,28,738,612),outline='#e8d59b',width=4)
        d.text((384,180),['饮品整箱','精选礼品','山泉饮用水','优选饮品'][index],font=font(76),fill='#f3e9d1',anchor='mm')
        d.text((384,302),'净含量  12 瓶 / 箱',font=font(34),fill='#f3e9d1',anchor='mm')
        d.line((50,418,718,418),fill='#e8d59b',width=4)
        d.text((384,504),'轻放  防潮  常温保存',font=font(35),fill='#f3e9d1',anchor='mm')
        im.save(out/f'case_{index:02d}.png')
    record={'mode':'procedural_generic_shipping_print','seed':seed,'kraft_faces':5,'coloured_case_faces':4,
            'limitations':'Generic carton print, not recovered photographed product packaging.'}
    (out/'carton_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--tile-size',type=int,default=256)
    p.add_argument('--photo-rois',type=Path)
    p.add_argument('--carton-faces',action='store_true',help='Write five generic kraft carton faces to output directory')
    a=p.parse_args()
    if a.tile_size < 32: p.error('tile-size must be >= 32')
    result=make_carton_faces(a.out,a.seed) if a.carton_faces else make_atlas(a.out,a.seed,a.tile_size,
          json.loads(a.photo_rois.read_text()) if a.photo_rois else None)
    print(json.dumps(result,ensure_ascii=False))
