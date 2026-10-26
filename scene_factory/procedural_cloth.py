"""Deterministic bounded hanging textile geometry; no Blender dependency."""
from __future__ import annotations

import math
import random


def curtain_mesh(size, cloth):
    """Return local XZ cloth, preserving a thickness margin inside its envelope.

    ``fold_depth`` is peak-to-peak depth. The renderer adds centred solidify
    thickness without even-offset inflation. No physical simulation is claimed.
    """
    width, _, height = size
    nx, nz = cloth['segments_x'], cloth['segments_z']
    thickness, sag = cloth['thickness'], cloth['sag']
    phase = random.Random(cloth['seed']).uniform(-math.pi, math.pi)
    vertices, faces, uv = [], [], []
    for j in range(nz + 1):
        v = j / nz
        for i in range(nx + 1):
            u = i / nx
            x = (u - .5) * (width - 2 * thickness)
            wave = 2 * math.pi * cloth['fold_count'] * u
            # Continuous long folds loosen toward the free edge. A small second
            # harmonic avoids mechanically identical cylinders without spikes.
            shift = .12 * math.sin(math.pi * v) * math.sin(2 * math.pi * u + phase)
            fold = (math.sin(wave + shift) + .13 * math.sin(2 * wave + phase + .2 * v)) / 1.13
            y = cloth['fold_depth'] * .5 * fold * (.86 + .14 * v)
            z = (.5 - v) * (height - 2 * thickness)
            z -= sag * math.sin(math.pi * u) ** 2 * (.25 + .75 * v)
            vertices.append([x, y, z])
            uv.append([u, 1 - v])
    # Centre the exact geometry in the declared envelope, with the reserved
    # margin on every axis. This bounds sag at the free edge as well as folds.
    low, high = min(p[2] for p in vertices), max(p[2] for p in vertices)
    scale = (height - 2 * thickness) / (high - low)
    for point in vertices:
        point[2] = (point[2] - (low + high) / 2) * scale
    for j in range(nz):
        for i in range(nx):
            a = j * (nx + 1) + i
            # Front faces -Y as for image_plane; top row runs left to right.
            faces.append([a, a + nx + 1, a + nx + 2, a + 1])
    return vertices, faces, uv
