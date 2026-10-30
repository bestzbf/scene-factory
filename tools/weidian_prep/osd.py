"""Find and remove burned-in CCTV on-screen text (timestamps, channel names).

Two kinds of overlay occur in this dataset, both white:
  * bold glyphs with a dark outline, found per glyph: a white blob of character
    size whose rim is mostly dark, chained into a line with similar neighbours;
  * thin glyphs without outline, found by on-device OCR (Apple Vision): a text
    fragment that is mostly digits and date/time separators anchors the row,
    and white glyph pieces on the same row extend it over characters OCR missed.
Only the top and bottom bands of the frame are searched, so signage elsewhere
is never touched. Only stroke pixels are inpainted inside each text box.
"""
import re

import cv2
import numpy as np

from . import ocr

BAND = 0.16                 # fraction of height searched at the top and at the bottom
DARK_V = 90
MIN_RIM_DARK = 0.55         # share of a glyph's rim that must be dark
MIN_GLYPHS = 3
MAX_TEXT_HEIGHT = 0.07      # OSD text is small; taller text is signage
SEPARATORS = set('-–—_:：;；/.,，年月日时分秒 ')
CIRCLED = str.maketrans('①②③④⑤⑥⑦⑧⑨⓪', '1234567890')
DATE = re.compile(r'\d{2,4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}')
YEAR = re.compile(r'(?<!\d)(19|20)\d\d(\s*[-/.年]|\s*$)')    # a fragment that starts the date
TIME = re.compile(r'\d{1,2}\s*[:：]\s*\d{2}')
TIMESTAMP_WIDTH = 20       # characters in 'YYYY-MM-DD  HH:MM:SS'
_ONE = np.ones((3, 3), np.uint8)
_RIM = np.ones((5, 5), np.uint8)


def _white_and_dark(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[..., 1], hsv[..., 2]
    return (value >= 200) & (saturation <= 60), value <= DARK_V


def _union(boxes, near):
    """Merge (x, y, w, h) boxes that `near(a, b)` links, directly or through others."""
    parent = list(range(len(boxes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if near(boxes[i], boxes[j]):
                parent[root(j)] = root(i)
    groups = {}
    for i, box in enumerate(boxes):
        groups.setdefault(root(i), []).append(box)
    return list(groups.values())


def _hull(members):
    x0 = min(x for x, _, _, _ in members); y0 = min(y for _, y, _, _ in members)
    x1 = max(x + w for x, _, w, _ in members); y1 = max(y + h for _, y, _, h in members)
    return (x0, y0, x1 - x0, y1 - y0)


def _same_row(gap_factor):
    def near(a, b):
        (xa, ya, wa, ha), (xb, yb, wb, hb) = a, b
        overlap = min(ya + ha, yb + hb) - max(ya, yb)
        gap = max(xa, xb) - min(xa + wa, xb + wb)
        return overlap >= 0.5 * min(ha, hb) and gap <= gap_factor * max(ha, hb)
    return near


def _outlined_lines(white, dark, h):
    """Lines of bold white glyphs with a dark rim, in the band's coordinates."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), connectivity=8)
    glyphs = []
    for index in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[index])
        if not 0.012 * h <= bh <= 0.08 * h or bw > 3 * bh or area > 0.75 * bw * bh:
            continue
        y0, y1 = max(0, y - 3), min(white.shape[0], y + bh + 3)
        x0, x1 = max(0, x - 3), min(white.shape[1], x + bw + 3)
        blob = labels[y0:y1, x0:x1] == index
        rim = cv2.dilate(blob.astype(np.uint8), _RIM).astype(bool) & ~white[y0:y1, x0:x1]
        if rim.any() and dark[y0:y1, x0:x1][rim].mean() >= MIN_RIM_DARK:
            glyphs.append((x, y, bw, bh))
    lines = []
    for members in _union(glyphs, lambda a, b: _same_row(1.2)(a, b) and max(a[3], b[3]) <= 1.6 * min(a[3], b[3])):
        box = _hull(members)
        if len(members) >= MIN_GLYPHS and box[2] >= 2.5 * box[3]:
            lines.append(box)
    return lines


def _digits(text):
    text = text.translate(CIRCLED)
    return sum(ch.isdigit() for ch in text), sum(ch.isdigit() or ch in SEPARATORS for ch in text), len(text.strip())


def timestamp_like(text):
    digits, allowed, length = _digits(text)
    return digits >= 4 and allowed >= 0.7 * length


def _white_pass(band_bgr):
    hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
    out = np.zeros_like(band_bgr)
    out[(hsv[..., 2] >= 190) & (hsv[..., 1] <= 70)] = 255
    return out


def _strokes(image_bgr, glyph):
    """Thin bright strokes: white top-hat at the glyph's stroke scale, so a bright backdrop does not count."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    size = max(3, int(round(0.35 * glyph)) | 1)
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))
    return (tophat >= 25) & (hsv[..., 2] >= 170) & (hsv[..., 1] <= 80)


def _fragments(band_bgr, frame_h, views):
    """OCR fragments carrying digits, from several views of the band, in band pixels."""
    h, w = band_bgr.shape[:2]
    up = lambda image: cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    images = [(band_bgr, 0, 1), (_white_pass(band_bgr), 0, 1), (up(band_bgr[:, w // 2:]), w // 2, 2),
              (up(band_bgr[:, :w // 2]), 0, 2)][:views]
    found = []
    for image, offset, scale in images:
        for text, _, (x, y, bw, bh) in ocr.read_lines(image):
            box = (x // scale + offset, y // scale, max(1, bw // scale), max(1, bh // scale))
            if box[3] <= MAX_TEXT_HEIGHT * frame_h and _digits(text)[0] >= 2:
                found.append((box, text.strip()))
    return found


def _grow(strokes, box, glyph, max_width):
    """Extend a row sideways over stroke columns OCR did not read, up to max_width."""
    x, y, w, h = box
    columns = np.flatnonzero(strokes[max(0, y):y + h].any(0))
    left, right, gap = x, x + w, 2.0 * glyph
    while True:
        step = columns[(columns < left) & (columns >= left - gap)]
        if not len(step) or right - step.min() > max_width:
            break
        left = int(step.min())
    while True:
        step = columns[(columns >= right) & (columns <= right + gap)]
        if not len(step) or step.max() + 1 - left > max_width:
            break
        right = int(step.max()) + 1
    return (left, y, right - left, h)


def _timestamp_rows(band_bgr, frame_h, views):
    rows = []
    fragments = _fragments(band_bgr, frame_h, views)
    for members in _union([f[0] for f in fragments], _same_row(3.0)):
        texts = [text for box, text in fragments if box in members]
        digits = sum(_digits(text)[0] for text in texts)
        dated = any(DATE.search(t) or YEAR.search(t) or TIME.search(t) for t in texts)
        if not any(timestamp_like(text) for text in texts) or (digits < 6 and not dated):
            continue
        box = _hull(members)
        glyph = 0.8 * float(np.median([b[3] for b in members]))
        width = float(np.median([b[2] / max(1, len(t.replace(' ', ''))) for b, t in fragments if b in members]))
        strokes = _strokes(band_bgr, glyph)
        x, y, w, h = _grow(strokes, box, glyph, 21 * width)
        missing = TIMESTAMP_WIDTH * width - w
        if missing > 3 * width:
            # Fixed layout "YYYY-MM-DD HH:MM:SS": cover the half OCR could not read.
            date, time = any(DATE.search(t) or YEAR.search(t) for t in texts), any(TIME.search(t) for t in texts)
            grow_left = missing if time and not date else 0 if date and not time else missing / 2
            x0 = max(0, int(x - grow_left)); x1 = min(band_bgr.shape[1], int(x + w + missing - grow_left))
            x, w = x0, x1 - x0
        rows.append(((x, y, w, h), glyph, ' | '.join(dict.fromkeys(texts))))
    return rows


def detect(image_bgr, use_ocr=True):
    """Return OSD detections [{'box', 'glyph_height', 'source', 'text'}] in image pixels."""
    h = image_bgr.shape[0]
    white, dark = _white_and_dark(image_bgr)
    band = int(h * BAND)
    found = []
    for start, views in ((0, 4), (h - band, 1)):
        rows = slice(start, start + band)
        found += [{'box': (x, y + start, bw, bh), 'glyph_height': float(bh), 'source': 'outlined_glyphs', 'text': ''}
                  for x, y, bw, bh in _outlined_lines(white[rows], dark[rows], h)]
        if use_ocr:
            found += [{'box': (x, y + start, bw, bh), 'glyph_height': glyph, 'source': 'ocr', 'text': text}
                      for (x, y, bw, bh), glyph, text in _timestamp_rows(image_bgr[rows], h, views)]
    merged = []
    for members in _union([d['box'] for d in found], _same_row(0.5)):
        parts = [d for d in found if d['box'] in members]
        merged.append({'box': _hull(members), 'glyph_height': round(max(d['glyph_height'] for d in parts), 1),
                       'source': '+'.join(sorted({d['source'] for d in parts})),
                       'text': ' | '.join(dict.fromkeys(d['text'] for d in parts if d['text']))})
    return merged


def stroke_mask(image_bgr, detections):
    """Thin bright strokes inside each text row plus their dark outline; never whole backgrounds."""
    h, w = image_bgr.shape[:2]
    value = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    mask = np.zeros((h, w), np.uint8)
    for detection in detections:
        (x, y, bw, bh), glyph = detection['box'], detection['glyph_height']
        margin = int(round(0.3 * glyph)) + 2
        x0, y0, x1, y1 = max(0, x - margin), max(0, y - margin), min(w, x + bw + margin), min(h, y + bh + margin)
        strokes = _strokes(image_bgr[y0:y1, x0:x1], glyph).astype(np.uint8)
        outline = cv2.dilate(strokes, _ONE, iterations=3).astype(bool) & (value[y0:y1, x0:x1] <= 110)
        mask[y0:y1, x0:x1] |= (strokes.astype(bool) | outline).astype(np.uint8)
    return cv2.dilate(mask, _ONE, iterations=1)


def remove(image_bgr, use_ocr=True):
    """Return (clean image, detections, fraction of pixels inpainted)."""
    detections = detect(image_bgr, use_ocr)
    if not detections:
        return image_bgr.copy(), [], 0.0
    mask = stroke_mask(image_bgr, detections)
    clean = cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)
    return clean, detections, np.count_nonzero(mask) / mask.size


def ignore_mask(shape, boxes, pad=0.6):
    """Boolean mask covering the boxes, for excluding text from edge analysis."""
    h, w = shape[:2]
    mask = np.zeros((h, w), bool)
    for x, y, bw, bh in boxes:
        margin = int(bh * pad)
        mask[max(0, y - margin):y + bh + margin, max(0, x - margin):x + bw + margin] = True
    return mask
