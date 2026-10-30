"""One-parameter division-model lens undistortion estimated from straight edges.

Coordinates are centred on the image and normalised by the half diagonal R:
    p_u = c + (p_d - c) / (1 + lam * r_d^2)      (lam < 0 for barrel distortion)
lam is chosen so that long edge chains (shelf edges, floor tiles, fridge frames)
become as straight as possible: the plumb-line principle. The distortion centre
is assumed to be the image centre. Evidence that is too weak leaves the image
uncorrected rather than guessing. Output keeps the source size, no black borders.
"""
import cv2
import numpy as np

COARSE = np.round(np.arange(-0.75, 0.2001, 0.01), 4)
WORK_WIDTH = 960            # edges are analysed at this width; lam is resolution independent
MIN_FRAGMENT = 0.03         # shortest edge fragment used, as a share of the diagonal
MAX_FRAGMENTS = 300         # longest fragments kept (pairwise cost grows with the square)
MIN_FRAGMENTS = 12
SIGMA_PX = 1.0              # how far (source px) a fragment may sit off a line and still count as on it
SIGMA_ANGLE = np.deg2rad(1.0)
NOISE_PX = 0.5              # edge localisation noise behind each fragment's direction uncertainty
MIN_GAIN = 1.03             # line-up score at the estimate over the score with no correction
MAX_HALF_SPLIT = 0.06       # two disjoint halves of the fragments must agree this closely
MIN_LAMBDA = -0.02          # weaker estimates are treated as "no visible distortion"
TURN_STEP = 6               # contour points between the samples used to measure turning
MAX_TURN = np.deg2rad(30)   # sharper turns are corners, junctions or line ends: chains are cut there


def _smooth_runs(points):
    """Split an ordered closed contour into runs without sharp turns."""
    ahead = np.roll(points, -TURN_STEP, 0) - points
    behind = points - np.roll(points, TURN_STEP, 0)
    cos = (ahead * behind).sum(1) / (np.linalg.norm(ahead, axis=1) * np.linalg.norm(behind, axis=1) + 1e-9)
    sharp = cos < np.cos(MAX_TURN)
    if not sharp.any():
        return [points]
    start = int(np.flatnonzero(sharp)[0])
    points, sharp = np.roll(points, -start, 0), np.roll(sharp, -start)
    cuts = np.flatnonzero(np.diff(sharp.astype(np.int8)))
    bounds = np.r_[cuts + 1, len(points)]
    return [points[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if not sharp[a]]


def edge_chains(gray, ignore=None, min_len_frac=0.05, max_bend=0.2):
    """Return long, smooth, elongated edge chains as ordered (x, y) points.

    Contours of the Canny map are traced in order and cut wherever they turn
    sharply. A one-pixel edge is traced out and back; the second pass is
    dropped because its pixels are already used.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    median = float(np.median(blurred))
    edges = cv2.Canny(blurred, max(20.0, 0.66 * median), min(250.0, max(60.0, 1.33 * median)))
    if ignore is not None:
        edges[ignore] = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_len = min_len_frac * np.hypot(w, h)
    used = np.zeros((h, w), bool)
    chains = []
    for contour in sorted(contours, key=len, reverse=True):
        if len(contour) < min_len:
            break
        for run in _smooth_runs(contour[:, 0, :]):
            if len(run) < min_len:
                continue
            xs, ys = run[:, 0], run[:, 1]
            if used[ys, xs].mean() > 0.5:
                continue
            points = run.astype(np.float64)
            low, high = np.linalg.eigvalsh(np.cov(points.T))
            if high > 0 and np.sqrt(max(low, 0.0) / high) <= max_bend:
                used[ys, xs] = True
                chains.append(points)
    return chains


def _fragment_lines(d, starts, lengths, lam):
    """Best-fit line of every fragment after undistorting with lam (normalised units).

    Returns centroid, unit tangent, angle in [0, pi), RMS residual and the local
    source-per-undistorted scale of each fragment.
    """
    denominator = 1 + lam * (d ** 2).sum(1)
    u = d / denominator[:, None]
    n = lengths
    mx, my = np.add.reduceat(u[:, 0], starts) / n, np.add.reduceat(u[:, 1], starts) / n
    a = np.add.reduceat(u[:, 0] ** 2, starts) / n - mx ** 2
    b = np.add.reduceat(u[:, 0] * u[:, 1], starts) / n - mx * my
    c = np.add.reduceat(u[:, 1] ** 2, starts) / n - my ** 2
    low = (a + c) / 2 - np.sqrt(np.maximum(((a - c) / 2) ** 2 + b ** 2, 0))
    phi = 0.5 * np.arctan2(2 * b, a - c)
    scale = np.add.reduceat(denominator, starts) / n
    return np.stack([mx, my], 1), np.stack([np.cos(phi), np.sin(phi)], 1), phi % np.pi, np.sqrt(np.maximum(low, 0)), scale


def _score(d, starts, lengths, lam, radius):
    """How well the fragments line up after undistorting with lam.

    Each fragment scores its own straightness; each pair scores how well the
    shorter one continues the longer one's line (offset and angle, with the
    tolerance widening for far-apart pairs because short lines have uncertain
    directions). Fragments must be sorted longest first.
    """
    centroid, tangent, angle, rms, scale = _fragment_lines(d, starts, lengths, lam)
    weight = lengths * np.exp(-0.5 * (rms * radius * scale / SIGMA_PX) ** 2)
    offset = (centroid[None, :, :] - centroid[:, None, :]) * radius          # [i, j]: from i to j, px
    along = np.abs((offset * tangent[:, None, :]).sum(2))
    across = np.abs(offset[..., 0] * tangent[:, None, 1] - offset[..., 1] * tangent[:, None, 0]) * scale[None, :]
    spread = NOISE_PX * np.sqrt(12.0) / lengths ** 1.5                       # direction uncertainty, rad
    tolerance = np.sqrt(SIGMA_PX ** 2 + (along * spread[:, None]) ** 2)
    turn = np.abs(angle[:, None] - angle[None, :]); turn = np.minimum(turn, np.pi - turn)
    turn_tolerance = np.sqrt(SIGMA_ANGLE ** 2 + spread[:, None] ** 2 + spread[None, :] ** 2)
    kernel = np.exp(-0.5 * ((across / tolerance) ** 2 + (turn / turn_tolerance) ** 2))
    pairs = np.triu(kernel * np.minimum(weight[:, None], weight[None, :]), 1)
    return float(weight.sum() + pairs.sum())


def _search(d, starts, lengths, radius):
    """Return (best lam, score at best, score at 0) over the coarse then fine grid."""
    coarse = np.array([_score(d, starts, lengths, lam, radius) for lam in COARSE])
    best = float(COARSE[int(np.argmax(coarse))])
    fine_grid = np.round(np.arange(best - 0.01, best + 0.0101, 0.002), 4)
    fine = np.array([_score(d, starts, lengths, lam, radius) for lam in fine_grid])
    return float(fine_grid[int(np.argmax(fine))]), float(fine.max()), float(coarse[int(np.argmin(np.abs(COARSE)))]), coarse


def _pack(chains, shape):
    h, w = shape
    centre = np.array([(w - 1) / 2, (h - 1) / 2]); radius = np.hypot(w, h) / 2
    lengths = np.array([len(c) for c in chains], float)
    starts = np.r_[0, np.cumsum(lengths)[:-1]].astype(int)
    return (np.concatenate(chains) - centre) / radius, starts, lengths, radius


def estimate(image_bgr, ignore=None):
    """Return (lam, info). lam == 0.0 means 'leave the image as it is'."""
    h, w = image_bgr.shape[:2]
    scale = WORK_WIDTH / w if w > WORK_WIDTH else 1.0
    small = cv2.resize(image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if ignore is not None and scale < 1:
        ignore = cv2.resize(ignore.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    chains = sorted(edge_chains(gray, ignore, min_len_frac=MIN_FRAGMENT), key=len, reverse=True)[:MAX_FRAGMENTS]
    info = {'fragments': len(chains), 'estimate': 0.0, 'gain': 1.0, 'applied': False}
    if len(chains) < MIN_FRAGMENTS:
        info['reason'] = 'too_few_edges'
        return 0.0, info
    d, starts, lengths, radius = _pack(chains, gray.shape)
    best, top, baseline, coarse = _search(d, starts, lengths, radius)
    halves = []
    for part in (chains[0::2], chains[1::2]):
        halves.append(_search(*_pack(part, gray.shape))[0])
    gain = top / max(baseline, 1e-9)
    info.update(estimate=round(best, 3), gain=round(gain, 3), halves=[round(v, 3) for v in halves])
    info['_curve'] = coarse / max(baseline, 1e-9)      # in-memory only, for pooling across cameras
    if best <= COARSE[0] + 0.011:
        info['reason'] = 'estimate_at_search_bound'
    elif best > MIN_LAMBDA:
        info['reason'] = 'no_visible_barrel_distortion'
    elif gain < MIN_GAIN:
        info['reason'] = 'alignment_gain_too_small'
    elif abs(halves[0] - halves[1]) > MAX_HALF_SPLIT:
        info['reason'] = 'halves_disagree'
    else:
        info['applied'] = True
        return best, info
    return 0.0, info


def _scale_for(lam, w, h):
    """Largest zoom-out that keeps every output pixel inside the source frame."""
    if lam >= 0:
        return 1.0
    radius = np.hypot(w, h) / 2
    a, b = (w / 2) / radius, (h / 2) / radius
    return min(1 / (1 + lam * a * a), 1 / (1 + lam * b * b))


def maps(w, h, lam):
    """Source pixel sampled by every output pixel under the inverse division model."""
    centre_x, centre_y = (w - 1) / 2, (h - 1) / 2
    radius = np.hypot(w, h) / 2
    k = _scale_for(lam, w, h)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    ux, uy = (xs - centre_x) * k / radius, (ys - centre_y) * k / radius
    ru = np.hypot(ux, uy)
    with np.errstate(divide='ignore', invalid='ignore'):
        rd = (1 - np.sqrt(1 - 4 * lam * ru * ru)) / (2 * lam * ru)
        factor = np.where(ru > 1e-12, rd / ru, 1.0)
    return (centre_x + ux * factor * radius).astype(np.float32), (centre_y + uy * factor * radius).astype(np.float32)


def undistort(image_bgr, lam):
    """Remap with the inverse division model; same size as the source, no borders."""
    if lam == 0:
        return image_bgr.copy()
    h, w = image_bgr.shape[:2]
    map_x, map_y = maps(w, h, lam)
    return cv2.remap(image_bgr, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def distort(image_bgr, lam):
    """Forward model, used only by the self-test to synthesise barrel distortion."""
    h, w = image_bgr.shape[:2]
    centre_x, centre_y = (w - 1) / 2, (h - 1) / 2
    radius = np.hypot(w, h) / 2
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dx, dy = (xs - centre_x) / radius, (ys - centre_y) / radius
    denominator = 1 + lam * (dx * dx + dy * dy)
    map_x = (centre_x + dx / denominator * radius).astype(np.float32)
    map_y = (centre_y + dy / denominator * radius).astype(np.float32)
    return cv2.remap(image_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
