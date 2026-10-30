"""Synthetic self-tests: known distortion, painted timestamps, two-room camera graphs.

Run: .venv/bin/python -m weidian_prep.selftest
"""
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import grouping, osd, undistort

FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
rng = np.random.default_rng(29)
failures = []


def check(name, ok, detail=''):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def world(w, h, margin=2.0, circles=False):
    """Undistorted scene `margin` times the frame, so a wide lens still sees real content at its corners."""
    W, H = int(w * margin), int(h * margin)
    image = cv2.GaussianBlur(rng.normal(120, 18, (H, W, 3)).clip(0, 255).astype(np.uint8), (7, 7), 2)
    colour = lambda: tuple(int(c) for c in rng.integers(0, 255, 3))
    for _ in range(26):     # long straight edges: shelf fronts, tile joints, fridge frames
        centre = rng.uniform([W / 2 - w / 2, H / 2 - h / 2], [W / 2 + w / 2, H / 2 + h / 2])
        angle, half = rng.uniform(0, np.pi), rng.uniform(0.35, 0.8) * w
        step = half * np.array([np.cos(angle), np.sin(angle)])
        cv2.line(image, tuple(map(int, centre - step)), tuple(map(int, centre + step)), colour(), int(rng.integers(2, 5)), cv2.LINE_AA)
    for _ in range(8):
        x, y = rng.uniform([W / 2 - w * 0.7, H / 2 - h * 0.7], [W / 2 + w * 0.5, H / 2 + h * 0.5]).astype(int)
        cv2.rectangle(image, (x, y), (x + int(rng.uniform(150, 400)), y + int(rng.uniform(100, 300))), colour(), 3, cv2.LINE_AA)
    if circles:             # round tables and arches: curved edges that must not read as lens distortion
        image = cv2.GaussianBlur(rng.normal(120, 18, (H, W, 3)).clip(0, 255).astype(np.uint8), (7, 7), 2)
        for _ in range(14):
            centre = tuple(rng.uniform([W / 2 - w / 2, H / 2 - h / 2], [W / 2 + w / 2, H / 2 + h / 2]).astype(int))
            axes = tuple(rng.uniform(0.08, 0.35, 2).astype(float) * w)
            cv2.ellipse(image, centre, tuple(map(int, axes)), float(rng.uniform(0, 180)), 0, 360, colour(), 3, cv2.LINE_AA)
    return image


def capture(scene, w, h, lam):
    """What a w x h camera with division-model distortion lam sees of the scene centre."""
    H, W = scene.shape[:2]
    radius = np.hypot(w, h) / 2
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dx, dy = (xs - (w - 1) / 2) / radius, (ys - (h - 1) / 2) / radius
    denominator = 1 + lam * (dx * dx + dy * dy)
    map_x = ((W - 1) / 2 + dx / denominator * radius).astype(np.float32)
    map_y = ((H - 1) / 2 + dy / denominator * radius).astype(np.float32)
    return cv2.remap(scene, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def test_undistortion():
    w, h = 1280, 720
    for trial in range(3):
        scene = world(w, h)
        for true in (0.0, -0.1, -0.2, -0.3, -0.45):
            lam, info = undistort.estimate(capture(scene, w, h, true))
            detail = f"est={info['estimate']} gain={info['gain']} halves={info.get('halves')} fragments={info['fragments']} {info.get('reason', '')}"
            if true == 0:
                check(f'scene {trial} lambda=0 left alone', not info['applied'], detail)
                continue
            check(f'scene {trial} lambda={true}', info['applied'] and abs(lam - true) <= 0.03, detail)
            if info['applied']:
                again, info2 = undistort.estimate(undistort.undistort(capture(scene, w, h, true), lam))
                check(f'scene {trial} lambda={true} straight after correction', abs(info2['estimate']) <= 0.03 and not info2['applied'],
                      f"re-estimate={info2['estimate']}")
    for trial in range(2):
        lam, info = undistort.estimate(capture(world(w, h, circles=True), w, h, 0.0))
        check(f'curved objects {trial} do not fake distortion', not info['applied'],
              f"est={info['estimate']} gain={info['gain']} halves={info.get('halves')} {info.get('reason', '')}")
    for size in ((1280, 720), (1920, 1080), (704, 576), (1080, 1920)):
        map_x, map_y = undistort.maps(*size, -0.45)
        inside = map_x.min() >= -0.5 and map_y.min() >= -0.5 and map_x.max() <= size[0] - 0.5 and map_y.max() <= size[1] - 0.5
        edge = max(-map_x.min(), -map_y.min(), map_x.max() - size[0] + 1, map_y.max() - size[1] + 1)
        check(f'undistorted {size[0]}x{size[1]} samples only inside the source frame', inside and edge > -2,
              f'closest edge margin {edge:.2f}px')


def paint_timestamp(image, xy, size, outline=True):
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT, size)
    text, stroke = ('2026-05-31 23:56:08', 2) if outline else ('2026-05-31  23:57:25', 0)
    draw.text(xy, text, font=font, fill=(255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0))
    box = draw.textbbox(xy, text, font=font, stroke_width=stroke)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), box


def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx, by, bw, bh = b; bx1, by1 = bx + bw, by + bh
    inter = max(0, min(ax1, bx1) - max(ax0, bx)) * max(0, min(ay1, by1) - max(ay0, by))
    return inter / ((ax1 - ax0) * (ay1 - ay0) + bw * bh - inter)


def test_osd():
    for (w, h), outline in (((1280, 720), True), ((1920, 1080), True), ((704, 576), True), ((1280, 720), False), ((1920, 1080), False)):
        style = 'outlined' if outline else 'thin'
        base = capture(world(w, h), w, h, 0.0)
        stamped, truth = paint_timestamp(base, (int(w * 0.6), int(h * 0.05)), int(h * (0.04 if outline else 0.028)), outline)
        found = osd.detect(stamped)
        best = max((iou(truth, d['box']) for d in found), default=0)
        check(f'osd {style} found at {w}x{h}', best > 0.5, f"iou={best:.2f} via={[d['source'] for d in found]}")
        clean, _, fraction = osd.remove(stamped)
        x0, y0, x1, y1 = truth
        before = int((np.abs(stamped.astype(int) - base.astype(int)).max(2) > 60)[y0:y1, x0:x1].sum())
        after = int((np.abs(clean.astype(int) - base.astype(int)).max(2) > 60)[y0:y1, x0:x1].sum())
        check(f'osd {style} strokes removed at {w}x{h}', after < 0.25 * max(before, 1) and 0 < fraction < 0.03,
              f'changed px vs. clean frame {before} -> {after}, inpainted {fraction:.4f} of frame')
        if outline:
            check(f'no osd on clean frame {w}x{h}', not osd.detect(base))
    label = capture(world(1280, 720), 1280, 720, 0.0)
    cv2.rectangle(label, (820, 20), (1180, 70), (255, 255, 255), -1)
    cv2.putText(label, 'PRICE 12.50 SALE', (835, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    check('white price label with black print is not treated as OSD', not osd.detect(label))


def test_grouping():
    n = 25; rooms = [list(range(13)), list(range(13, 24))]
    weights = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            same = any(i in r and j in r for r in rooms)
            weights[i, j] = weights[j, i] = int(rng.integers(40, 160) if same else rng.integers(0, 12))
    groups = grouping.split(weights)
    purity = sum(max(len(set(g) & set(r)) for r in rooms) for g in groups) / 24
    check('split 25 views into groups of <= 16', all(len(g) <= 16 for g in groups) and sum(map(len, groups)) == n,
          f'sizes={[len(g) for g in groups]}')
    check('split keeps each room together', purity >= 0.9, f'purity={purity:.2f}')
    check('isolated camera flagged', grouping.isolated(weights) == [24], f'{grouping.isolated(weights)}')
    check('shop within the cap stays one scene', grouping.split(weights[:10, :10]) == [list(range(10))])


if __name__ == '__main__':
    test_undistortion(); test_osd(); test_grouping()
    print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
