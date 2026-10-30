"""Preprocess many shops in parallel, one worker process per shop, resumable.

Same per-shop steps as pipeline.run (OSD removal, undistortion, resize, overlap
flags), but each shop writes prepared/<shop>/ plus shops/<shop>.json atomically,
so an interrupted or repeated run skips finished shops. A shop is processed only
when RAW/<shop>/ is complete: its image count matches EXPECTED (a JSON map
shop -> camera count) when given. QA sheets are written for a sample of shops.

    python -m weidian_prep.batch --raw RAW --run-dir RUN [--expected counts.json]
           [--workers 8] [--qa-every 50] [--wait]

--wait keeps polling RAW for shops that are still downloading. groups.json and
manifest.json are rebuilt from shops/*.json at the end of every pass.
No network or model calls.
"""
import argparse
import datetime
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import grouping, osd, pipeline, undistort

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


def _atomic_write(path, text):
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temp.write_text(text, encoding='utf-8')
    os.replace(temp, path)


def process_shop(shop_dir, run_dir, max_edge, write_qa):
    """Process one shop; returns its manifest record. Runs in a worker process.

    Source images that do not decode (truncated in the source bucket) are skipped
    and listed in the record; a shop with no decodable image fails.
    """
    shop_dir, run_dir = Path(shop_dir), Path(run_dir)
    shop = shop_dir.name
    out = run_dir / 'prepared' / shop
    paths = sorted(p for p in shop_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    originals, cleaned, records, skipped = [], [], [], []
    for path in paths:
        try:
            image, digest = pipeline.read(path)
        except ValueError:
            skipped.append(path.name)
            continue
        clean, detections, inpainted = osd.remove(image)
        _, info = undistort.estimate(clean, osd.ignore_mask(clean.shape, [d['box'] for d in detections]))
        if write_qa:
            originals.append(image)
        cleaned.append(clean)
        records.append({'camera': path.stem, 'source': str(path), 'source_sha256': digest,
                        'source_size': [image.shape[1], image.shape[0]], 'osd': detections,
                        'osd_inpainted_fraction': round(inpainted, 5), 'distortion': info})
    if not records:
        raise ValueError(f'{shop}: no decodable source image')
    pipeline.decide_lambdas(records)
    processed = []
    for record, clean in zip(records, cleaned):
        final = pipeline.resize(undistort.undistort(clean, record['lambda']), max_edge)
        target = out / f"{record['camera']}.jpg"
        record.update(output=str(target), output_sha256=pipeline.write(target, final),
                      output_size=[final.shape[1], final.shape[0]])
        processed.append(final)
    weights = grouping.overlap_matrix(processed) if len(processed) > 1 else np.zeros((1, 1), int)
    lonely = grouping.isolated(weights)
    for i in lonely:
        records[i]['isolated_view'] = True
    if write_qa:
        qa = run_dir / 'qa'
        labels = [f"{r['camera'][:30]}  lambda={r['lambda']:+.3f} ({r['lambda_source']})  osd={len(r['osd'])}"
                  + ('  ISOLATED' if r.get('isolated_view') else '') for r in records]
        pipeline.before_after_sheet([(label, originals[i], [d['box'] for d in records[i]['osd']], processed[i])
                                     for i, label in enumerate(labels)], qa / f'{shop}_before_after.jpg')
        pipeline.views_sheet(f'{shop}: {len(records)} cameras, one scene', [r['camera'] for r in records],
                             processed, qa / f'{shop}_views.jpg')
    record = {'cameras': len(records), 'scene_id': shop, 'isolated_views': [records[i]['camera'] for i in lonely],
              'skipped_undecodable': skipped, 'overlap_inliers': weights.tolist(), 'images': records}
    _atomic_write(run_dir / 'shops' / f'{shop}.json', json.dumps(record, ensure_ascii=False) + '\n')
    return shop, record


def ready_shops(raw, run_dir, expected, attempted, settle_seconds=120):
    """Shops whose download is complete: expected image count reached and no file
    modified in the last settle_seconds (a downloader may write files in place)."""
    done = {p.stem for p in (run_dir / 'shops').glob('*.json')}
    now = time.time()
    ready = []
    for shop_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        shop = shop_dir.name
        if shop in done or shop in attempted:
            continue
        images = [p for p in shop_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        if not images:
            continue
        if expected is not None and len(images) < expected.get(shop, len(images)):
            continue                                    # still downloading
        if any(now - p.stat().st_mtime < settle_seconds for p in images):
            continue                                    # a file may still be written
        ready.append(shop_dir)
    return ready


def rebuild(run_dir, max_edge):
    records = {}
    for path in sorted((run_dir / 'shops').glob('*.json')):
        records[path.stem] = json.loads(path.read_text(encoding='utf-8'))
    groups = [{'scene_id': shop, 'images': [image['output'] for image in rec['images']]}
              for shop, rec in sorted(records.items())]
    _atomic_write(run_dir / 'groups.json', json.dumps({'schema_version': 1, 'scenes': groups}, ensure_ascii=False, indent=1) + '\n')
    manifest = {'schema_version': pipeline.VERSION, 'updated': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                'prepared': str(run_dir / 'prepared'), 'max_edge': max_edge, 'shops': records}
    _atomic_write(run_dir / 'manifest.json', json.dumps(manifest, ensure_ascii=False) + '\n')
    return len(records), sum(len(g['images']) for g in groups)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--expected', type=Path, help='JSON map shop_id -> camera count; shops are processed once complete')
    parser.add_argument('--max-edge', type=int, default=1280)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--qa-every', type=int, default=50, help='write QA sheets for every Nth shop (0 = none)')
    parser.add_argument('--wait', action='store_true', help='keep polling RAW until every expected shop is done')
    args = parser.parse_args(argv)
    raw, run_dir = args.raw.resolve(), args.run_dir.resolve()
    (run_dir / 'shops').mkdir(parents=True, exist_ok=True)
    (run_dir / 'qa').mkdir(parents=True, exist_ok=True)
    (run_dir / 'failed').mkdir(parents=True, exist_ok=True)
    expected = json.loads(args.expected.read_text()) if args.expected else None
    attempted, index, started = set(), 0, time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        while True:
            batch = ready_shops(raw, run_dir, expected, attempted)
            futures = {}
            for shop_dir in batch:
                attempted.add(shop_dir.name)
                write_qa = bool(args.qa_every) and index % args.qa_every == 0
                index += 1
                futures[pool.submit(process_shop, str(shop_dir), str(run_dir), args.max_edge, write_qa)] = shop_dir.name
            for future in as_completed(futures):
                shop = futures[future]
                try:
                    future.result()
                    (run_dir / 'failed' / f'{shop}.txt').unlink(missing_ok=True)
                except Exception:
                    (run_dir / 'failed' / f'{shop}.txt').write_text(traceback.format_exc(), encoding='utf-8')
                    print(f'FAILED {shop}', flush=True)
            shops, images = rebuild(run_dir, args.max_edge)
            failed = len(list((run_dir / 'failed').glob('*.txt')))
            print(f'{time.strftime("%H:%M:%S")} [{time.monotonic() - started:5.0f}s] prepared {shops} shops / {images} images, '
                  f'failed {failed}', flush=True)
            total = len(expected) if expected is not None else None
            if not args.wait or (total is not None and shops + failed >= total):
                break
            if not batch:
                time.sleep(30)
    return 0


if __name__ == '__main__':
    sys.exit(main())
