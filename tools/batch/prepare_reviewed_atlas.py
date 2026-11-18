"""Rectify reviewed product faces into a fixed 8x8 atlas, with source hashes.

The recipe names a base atlas and individual photo quadrilaterals in pixels.
It never places whole reference photographs into the 3D scene.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(recipe_path, source_root, out):
    recipe_path=recipe_path.resolve();recipe=json.loads(recipe_path.read_text())
    base=(recipe_path.parent/recipe['base_atlas']).resolve()
    atlas=Image.open(base).convert('RGB')
    if atlas.width!=atlas.height or atlas.width%8:
        raise ValueError('Base atlas must be square and divisible into 8x8 tiles')
    size=atlas.width//8;seen=set();records=[]
    for item in recipe['faces']:
        tile=item['tile'];quad=np.asarray(item['quad_px'],dtype=float)
        if not isinstance(tile,int) or not 0<=tile<64 or tile in seen:
            raise ValueError('Invalid or duplicate tile index')
        seen.add(tile);path=(source_root/item['image']).resolve()
        if source_root.resolve() not in path.parents:raise ValueError('Image outside source root')
        with Image.open(path) as original:
            if quad.shape!=(4,2) or not np.isfinite(quad).all() or np.any(quad<0) or np.any(quad>=[original.width,original.height]):
                raise ValueError('Invalid product quadrilateral')
            # PIL maps output coordinates to the selected physical face.
            matrix=[];rhs=[]
            for (x,y),(u,v) in zip([(0,0),(size,0),(size,size),(0,size)],quad):
                matrix.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
                rhs.extend([u,v])
            coefficients=np.linalg.solve(np.asarray(matrix),np.asarray(rhs))
            face=original.convert('RGB').transform((size,size),Image.Transform.PERSPECTIVE,coefficients,Image.Resampling.BICUBIC)
        atlas.paste(face,((tile%8)*size,(tile//8)*size))
        records.append({**item,'image_sha256':sha(path)})
    out.parent.mkdir(parents=True,exist_ok=True);atlas.save(out)
    record={'schema_version':1,'recipe_sha256':sha(recipe_path),'base_atlas_sha256':sha(base),
            'output_sha256':sha(out),'faces':records,'usage':'Fixed UV texture of individual packages; no photo backgrounds',
            'limitations':'Reviewed crop coordinates and approximate product dimensions; source blur and baked lighting remain.'}
    out.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',required=True,type=Path)
    parser.add_argument('--source-root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare(args.recipe,args.source_root,args.out),ensure_ascii=False))
