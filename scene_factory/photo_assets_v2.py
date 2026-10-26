"""Rectify per-photo texture regions. Pillow is required only for this stage."""
from pathlib import Path
import hashlib
import math


def image_size(path):
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError('V2 needs Pillow: install requirements-v2.txt in your Python environment') from exc
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).size


def render_size(path, max_edge=1600):
    w, h = image_size(path)
    ratio = min(max_edge / w, max_edge / h)
    return max(64, round(w * ratio)), max(64, round(h * ratio))


def _solve(matrix, target):
    a = [list(map(float, row)) + [float(y)] for row, y in zip(matrix, target)]
    n = len(a)
    for i in range(n):
        pivot = max(range(i, n), key=lambda j: abs(a[j][i]))
        if abs(a[pivot][i]) < 1e-10:
            raise ValueError('Degenerate photo texture quadrilateral')
        a[i], a[pivot] = a[pivot], a[i]
        value = a[i][i]
        a[i] = [v / value for v in a[i]]
        for j in range(n):
            if i != j:
                value = a[j][i]
                a[j] = [x - value * y for x, y in zip(a[j], a[i])]
    return [row[-1] for row in a]


def extract(source, textures, output):
    from PIL import Image, ImageOps
    from batch import atomic_json, file_sha256
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with Image.open(source) as original:
        photo = ImageOps.exif_transpose(original).convert('RGB')
        w, h = photo.size
        photo.save(output / 'reference.png')
        for texture in textures:
            points = [(u * (w - 1), v * (h - 1)) for u, v in texture['quad']]
            distance = lambda a,b: math.hypot(a[0]-b[0], a[1]-b[1])
            tw = max(16, round((distance(points[0],points[1])+distance(points[2],points[3])) / 2))
            th = max(16, round((distance(points[0],points[3])+distance(points[1],points[2])) / 2))
            scale = min(1, 1600 / max(tw, th)); tw, th = max(16,round(tw*scale)),max(16,round(th*scale))
            matrix, target = [], []
            for (x,y),(u,v) in zip([(0,0),(tw-1,0),(tw-1,th-1),(0,th-1)],points):
                matrix.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
                target.extend([u,v])
            coefficients = _solve(matrix,target)
            result = photo.transform((tw,th),Image.Transform.PERSPECTIVE,coefficients,Image.Resampling.BICUBIC)
            name = hashlib.sha256(texture['id'].encode()).hexdigest()[:20] + '.png'
            result.save(output/name)
            records.append({'id':texture['id'],'path':str(output/name),'sha256':file_sha256(output/name),
                            'quad':texture['quad'],'size':[tw,th]})
    manifest = {'source_image':str(Path(source).resolve()),'source_sha256':file_sha256(source),
                'reference_image':str(output/'reference.png'),'dimensions':[w,h],
                'method':'EXIF orientation; inverse homography; bicubic; source pixels only','textures':records}
    atomic_json(output/'manifest.json',manifest)
    return output/'manifest.json'
