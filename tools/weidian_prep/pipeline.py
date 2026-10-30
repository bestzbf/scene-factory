"""Turn downloaded shop folders into scene-factory multiview input, one scene per shop.

Input:  RAW/<shop_id>/<camera>.jpg            (originals, never modified)
Output: RUN/prepared/<shop_id>/<camera>.jpg   (every camera of the shop, one scene)
        RUN/groups.json                       (scene-factory group manifest, scene_id = shop_id)
        RUN/manifest.json, RUN/qa/            (provenance and visual checks)

Per image: remove OSD text -> estimate barrel distortion -> undistort -> resize
so the longest edge is at most MAX_EDGE. An image whose own lines are too weak
to estimate distortion borrows the median of the reliable estimates of the same
shop and resolution (usually the same camera model). Cameras that share no
visual overlap with any other camera of the shop are flagged, not removed.
Only on-device work: Apple Vision OCR and OpenCV. No network or model calls.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import grouping, osd, undistort

VERSION = 'weidian_prep_v2'
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
GROUP_MIN_RELIABLE = 2
GROUP_MAX_MAD = 0.06        # reliable estimates of one shop/resolution must agree this closely to be shared
FLAT_LAMBDA = 0.03          # an image whose two halves both stay within this of 0 keeps its own "no distortion"
POOLED_MIN_GAIN = 1.02
POOLED_MAX_HALF_SPLIT = 0.08
POOLED_MAX_LAMBDA = -0.05


def read(path):
    data = np.fromfile(str(path), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'cannot decode {path}')
    return image, hashlib.sha256(data.tobytes()).hexdigest()


def write(path, image, quality=95):
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f'cannot encode {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(str(path))
    return hashlib.sha256(data.tobytes()).hexdigest()


def resize(image, max_edge):
    h, w = image.shape[:2]
    if max(h, w) <= max_edge:
        return image
    scale = max_edge / max(h, w)
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _pooled(members):
    """Shop/resolution estimate from the summed line-up curves of all its cameras, if both halves agree."""
    curves = [r['distortion']['_curve'] for r in members if '_curve' in r['distortion']]
    if len(curves) < 2:
        return None
    total = np.sum(curves, 0)
    best = float(undistort.COARSE[int(np.argmax(total))])
    halves = [float(undistort.COARSE[int(np.argmax(np.sum(curves[k::2], 0)))]) for k in (0, 1)]
    gain = float(total.max() / total[int(np.argmin(np.abs(undistort.COARSE)))])
    if (best <= POOLED_MAX_LAMBDA and best > undistort.COARSE[0] + 0.011 and gain >= POOLED_MIN_GAIN
            and abs(halves[0] - halves[1]) <= POOLED_MAX_HALF_SPLIT):
        return round(best, 3)
    return None


def decide_lambdas(records):
    """Pick the correction of every image of one shop.

    Order: the image's own reliable estimate; else the median of the reliable
    estimates of the same shop and resolution (usually one camera model); else
    the estimate pooled over all those cameras' line-up curves; else none. An
    image whose own two halves both say "flat" is never corrected by its group.
    """
    groups = {}
    for record in records:
        groups.setdefault(tuple(record['source_size']), []).append(record)
    for members in groups.values():
        reliable = [r['distortion']['estimate'] for r in members if r['distortion']['applied']]
        shared, source = None, None
        if len(reliable) >= GROUP_MIN_RELIABLE:
            median = float(np.median(reliable))
            if float(np.median(np.abs(np.array(reliable) - median))) <= GROUP_MAX_MAD:
                shared, source = round(median, 3), 'shop_resolution_median'
        if shared is None:
            shared = _pooled(members)
            source = 'shop_resolution_pooled' if shared is not None else None
        for record in members:
            info = record['distortion']
            flat = all(abs(v) <= FLAT_LAMBDA for v in info.get('halves', [1]))
            if info['applied']:
                record['lambda'], record['lambda_source'] = info['estimate'], 'own'
            elif shared is not None and not flat:
                record['lambda'], record['lambda_source'] = shared, source
            else:
                record['lambda'], record['lambda_source'] = 0.0, 'none'
        for record in members:
            record['distortion'].pop('_curve', None)


def _thumb(image_bgr, width, boxes=()):
    image = image_bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), max(2, image.shape[1] // 400))
    h, w = image.shape[:2]
    small = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


def before_after_sheet(rows, path, width=520):
    """rows: (label, original_bgr, boxes, processed_bgr)."""
    font = ImageFont.truetype(FONT, 15)
    thumbs = [(label, _thumb(a, width, boxes), _thumb(b, width)) for label, a, boxes, b in rows]
    height = sum(max(a.height, b.height) + 22 for _, a, b in thumbs)
    sheet = Image.new('RGB', (2 * width + 10, height), 'white'); draw = ImageDraw.Draw(sheet); y = 0
    for label, a, b in thumbs:
        draw.text((4, y + 3), label, fill='black', font=font)
        sheet.paste(a, (0, y + 22)); sheet.paste(b, (width + 10, y + 22)); y += max(a.height, b.height) + 22
    sheet.save(path, quality=85)


def views_sheet(title, names, images, path, width=320, cols=5):
    font = ImageFont.truetype(FONT, 14)
    thumbs = [_thumb(image, width) for image in images]
    cell = max(t.height for t in thumbs) + 18
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * width, rows * cell + 26), 'white'); draw = ImageDraw.Draw(sheet)
    draw.text((4, 5), title, fill='black', font=font)
    for i, (name, thumb) in enumerate(zip(names, thumbs)):
        x, y = (i % cols) * width, 26 + (i // cols) * cell
        sheet.paste(thumb, (x, y + 18)); draw.text((x + 3, y + 2), name[:34], fill='black', font=font)
    sheet.save(path, quality=85)


def run(raw, run_dir, max_edge=1280):
    raw, run_dir = Path(raw).resolve(), Path(run_dir).resolve()
    out = run_dir / 'prepared'
    if out.exists():
        sys.exit(f'{out} already exists; use a new run directory')
    qa = run_dir / 'qa'; qa.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': VERSION, 'created': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                'raw': str(raw), 'prepared': str(out), 'max_edge': max_edge,
                'steps': ['OSD removal: OCR-anchored timestamp rows + outlined glyph rows, stroke-only inpainting',
                          'division-model undistortion; lambda from fragment line-up, else shop/resolution median',
                          f'resize longest edge to <= {max_edge} (never upscale), JPEG q95',
                          'one scene per shop with every camera; SIFT overlap only flags isolated cameras'],
                'shops': {}}
    groups = []
    for shop_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        shop = shop_dir.name
        paths = sorted(p for p in shop_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        originals, cleaned, records = [], [], []
        for path in paths:
            image, digest = read(path)
            clean, detections, inpainted = osd.remove(image)
            _, info = undistort.estimate(clean, osd.ignore_mask(clean.shape, [d['box'] for d in detections]))
            originals.append(image); cleaned.append(clean)
            records.append({'camera': path.stem, 'source': str(path), 'source_sha256': digest,
                            'source_size': [image.shape[1], image.shape[0]], 'osd': detections,
                            'osd_inpainted_fraction': round(inpainted, 5), 'distortion': info})
        decide_lambdas(records)
        processed = []
        for record, clean in zip(records, cleaned):
            final = resize(undistort.undistort(clean, record['lambda']), max_edge)
            target = out / shop / f"{record['camera']}.jpg"
            record.update(output=str(target), output_sha256=write(target, final), output_size=[final.shape[1], final.shape[0]])
            processed.append(final)
        weights = grouping.overlap_matrix(processed) if len(processed) > 1 else np.zeros((1, 1), int)
        lonely = grouping.isolated(weights)
        for i in lonely:
            records[i]['isolated_view'] = True
        manifest['shops'][shop] = {'cameras': len(paths), 'scene_id': shop,
                                   'isolated_views': [records[i]['camera'] for i in lonely],
                                   'overlap_inliers': weights.tolist(), 'images': records}
        groups.append({'scene_id': shop, 'images': [r['output'] for r in records]})
        labels = [f"{r['camera'][:30]}  lambda={r['lambda']:+.3f} ({r['lambda_source']})  osd={len(r['osd'])}"
                  + ('  ISOLATED' if r.get('isolated_view') else '') for r in records]
        before_after_sheet([(label, originals[i], [d['box'] for d in records[i]['osd']], processed[i])
                            for i, label in enumerate(labels)], qa / f'{shop}_before_after.jpg')
        views_sheet(f'{shop}: {len(paths)} cameras, one scene', [r['camera'] for r in records], processed, qa / f'{shop}_views.jpg')
        sources = [r['lambda_source'] for r in records]
        print(f"{shop}: {len(paths)} cameras | lambda own {sources.count('own')}, shop median {sources.count('shop_resolution_median')}, "
              f"shop pooled {sources.count('shop_resolution_pooled')}, "
              f"none {sources.count('none')} | osd {sum(bool(r['osd']) for r in records)}/{len(paths)} | isolated {len(lonely)}", flush=True)
    (run_dir / 'groups.json').write_text(json.dumps({'schema_version': 1, 'scenes': groups}, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', required=True); parser.add_argument('--run-dir', required=True)
    parser.add_argument('--max-edge', type=int, default=1280)
    args = parser.parse_args(argv)
    run(args.raw, args.run_dir, args.max_edge)


if __name__ == '__main__':
    main()
