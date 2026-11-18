"""Rectify reviewed individual package faces from the single-shop demo references.

These small object crops are fixed UV textures, never full-scene camera projections.
The input photograph and quadrilaterals are recorded beside every output texture.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# TL, TR, BR, BL, manually reviewed on shop 1646301141 / view_04 (1280 x 720).
FACES=[
    [[109,113],[140,96],[163,142],[134,160]],
    [[204,61],[232,48],[247,79],[224,94]],
    [[18,151],[48,136],[81,186],[48,207]],
    [[134,340],[159,316],[193,340],[164,368]],
    [[143,412],[171,387],[197,410],[167,443]],
    [[8,448],[44,414],[75,439],[30,477]],
    [[73,594],[132,551],[162,582],[109,625]],
    [[58,348],[88,322],[118,344],[88,375]],
    [[205,265],[224,246],[248,266],[224,292]],
    [[216,375],[245,351],[267,374],[239,401]],
    [[284,189],[310,170],[335,188],[307,213]],
    [[214,468],[241,442],[264,459],[237,490]],
]


def rectify(image,quad,size=(192,256)):
    w,h=size
    output=[(0,0),(w,0),(w,h),(0,h)]
    matrix=[];values=[]
    for (x,y),(u,v) in zip(output,quad):
        matrix.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
        values.extend([u,v])
    coeff=np.linalg.solve(np.asarray(matrix,dtype=float),np.asarray(values,dtype=float))
    return image.transform(size,Image.Transform.PERSPECTIVE,coeff,Image.Resampling.BICUBIC)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--gift-reference',type=Path,help='Optional view_01 reference for corrected gold/brown gift faces')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with Image.open(args.reference) as src:
        src=src.convert('RGB')
        if src.size!=(1280,720):raise ValueError('Reviewed pixel coordinates require the original 1280 x 720 reference')
        records=[]
        for index,quad in enumerate(FACES):
            name=f'stock_face_{index:02d}.png'
            rectify(src,quad).save(args.out/name)
            records.append({'file':name,'source':str(args.reference.resolve()),'quad_px':quad})
    (args.out/'stock_face_manifest.json').write_text(json.dumps({'shop':'1646301141','view':'view_04',
        'method':'manual individual package-face rectification','faces':records,
        'limitations':'Original blur and baked illumination remain; crops do not prove exact SKU identity.'},ensure_ascii=False,indent=2))
    if args.gift_reference:
        corrections={'gold_gift':[[334,127],[353,121],[376,185],[355,195]],
                     'brown_gift':[[254,354],[280,335],[310,379],[280,399]]}
        with Image.open(args.gift_reference) as original:
            for name,quad in corrections.items():rectify(original.convert('RGB'),quad).save(args.out/(name+'.png'))
        (args.out/'gift_corrections.json').write_text(json.dumps({'source':str(args.gift_reference.resolve()),
            'quads_px':corrections,'reason':'Replace old crops which sampled empty cabinet backing rather than product faces.'},indent=2))
    print('Prepared',len(records),'individual stock faces in',args.out)


if __name__=='__main__':main()
