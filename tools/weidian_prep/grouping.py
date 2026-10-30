"""Split a shop's cameras into scene groups of at most 16 views by visual overlap.

Overlap between two cameras is the number of SIFT matches that survive a ratio
test and a RANSAC fundamental-matrix check. Cameras are merged strongest pair
first, keeping clusters balanced; leftovers join the cluster they overlap most.
For shops that already fit, the same graph flags cameras that overlap with no
other camera, which may be a different area such as a back room.
"""
import math

import cv2
import numpy as np

CAP = 16
MIN_INLIERS = 25            # weaker matches are treated as "no shared view"
FEATURES = 4000


def features(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.SIFT_create(nfeatures=FEATURES).detectAndCompute(gray, None)


def inliers(first, second):
    (kp1, des1), (kp2, des2) = first, second
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0
    a = np.float32([kp1[m.queryIdx].pt for m in good])
    b = np.float32([kp2[m.trainIdx].pt for m in good])
    _, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 2.0, 0.999)
    return int(mask.sum()) if mask is not None else 0


def overlap_matrix(images_bgr):
    feats = [features(image) for image in images_bgr]
    n = len(feats)
    weights = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            weights[i, j] = weights[j, i] = inliers(feats[i], feats[j])
    return weights


def split(weights, cap=CAP, min_inliers=MIN_INLIERS):
    """Return groups (lists of view indices), each at most `cap` long."""
    n = len(weights)
    if n <= cap:
        return [list(range(n))]
    target = math.ceil(n / math.ceil(n / cap))
    parent = list(range(n)); size = [1] * n

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    edges = sorted(((weights[i, j], i, j) for i in range(n) for j in range(i + 1, n)
                    if weights[i, j] >= min_inliers), reverse=True)
    for _, i, j in edges:
        a, b = root(i), root(j)
        if a != b and size[a] + size[b] <= target:
            parent[b] = a; size[a] += size[b]
    clusters = {}
    for i in range(n):
        clusters.setdefault(root(i), []).append(i)
    groups = sorted(clusters.values(), key=len, reverse=True)
    main = [g for g in groups if len(g) > 1] or groups[:1]
    rest = [g for g in groups if g not in main]
    for small in sorted(rest, key=len):
        def overlap(group):
            return sum(weights[i, j] for i in small for j in group)
        room = [g for g in main if len(g) + len(small) <= cap]
        if room:
            best = max(room, key=overlap)
            if overlap(best) > 0 or len(room) == len(main):
                best.extend(small); continue
        main.append(small)
    return [sorted(g) for g in main]


def isolated(weights, min_inliers=MIN_INLIERS):
    """Indices of views that share no verified overlap with any other view."""
    n = len(weights)
    return [i for i in range(n) if n > 1 and all(weights[i, j] < min_inliers for j in range(n) if j != i)]
