"""Procedural convenience-store assets, built from the 12-shop pilot's requests.

Run inside Blender:
    Blender -b --factory-startup --python build_retail_asset.py -- <spec.json> <package_dir>

<spec.json> holds {"asset_id", "category", "archetype", "size_m": [W, D, H],
"placement": "floor"|"surface", "seed", "params": {...}}. The result is
<package_dir>/model.glb + asset.json in the scene_asset_input_v1 two-file
protocol, so the normal asset intake measures and checks it. Geometry is
already in metres with -Y as the front and the bottom centre at the origin.

Merchandise is modelled as colourful product blocks with a packaging-style
texture atlas (no brands, no text), because the photographs are dominated by
stocked shelves. Glass is a dark, glossy opaque material: the intake profile
admits only opaque materials. Nothing is downloaded and no model API is used.
"""
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np

ARGS = sys.argv[sys.argv.index('--') + 1:]
SPEC = json.loads(Path(ARGS[0]).read_text(encoding='utf-8'))
OUT = Path(ARGS[1])
W, D, H = SPEC['size_m']
P = SPEC.get('params', {})
ORIENT_Y = P.get('orient') == 'y'
if ORIENT_Y:
    W, D = D, W
RNG = np.random.default_rng(SPEC['seed'])

bpy.ops.wm.read_factory_settings(use_empty=True)
MESH = bmesh.new()
UV = MESH.loops.layers.uv.new('UVMap')
MATERIALS = []
MATERIAL_INDEX = {}

# ---------------------------------------------------------------- materials

PALETTE = {
    'white_metal': ((0.86, 0.87, 0.86), 0.25, 0.35), 'grey_metal': ((0.55, 0.57, 0.58), 0.6, 0.35),
    'dark_metal': ((0.12, 0.12, 0.13), 0.5, 0.4), 'chrome': ((0.78, 0.79, 0.8), 1.0, 0.18),
    'black_plastic': ((0.035, 0.035, 0.04), 0.0, 0.45), 'grey_plastic': ((0.42, 0.43, 0.45), 0.0, 0.5),
    'white_plastic': ((0.88, 0.88, 0.86), 0.0, 0.4), 'glass': ((0.6, 0.68, 0.72), 0.0, 0.05),
    'shelf_board': ((0.82, 0.83, 0.82), 0.3, 0.35), 'counter_top': ((0.9, 0.9, 0.88), 0.0, 0.3),
    'wood': ((0.55, 0.38, 0.22), 0.0, 0.55), 'light_wood': ((0.74, 0.6, 0.42), 0.0, 0.5),
    'cardboard': ((0.62, 0.45, 0.27), 0.0, 0.85), 'cardboard_tape': ((0.55, 0.4, 0.23), 0.0, 0.5),
    'screen': ((0.02, 0.025, 0.03), 0.0, 0.08), 'rubber': ((0.05, 0.05, 0.05), 0.0, 0.8),
    'red_accent': ((0.72, 0.06, 0.05), 0.0, 0.45), 'blue_accent': ((0.08, 0.25, 0.6), 0.0, 0.45),
    'green_accent': ((0.1, 0.45, 0.2), 0.0, 0.45), 'yellow_accent': ((0.9, 0.7, 0.08), 0.0, 0.45),
    'water_plastic': ((0.55, 0.72, 0.85), 0.0, 0.15), 'cap_blue': ((0.1, 0.3, 0.75), 0.0, 0.4),
    'cooler_interior': ((0.9, 0.92, 0.94), 0.0, 0.3), 'fire_red': ((0.75, 0.05, 0.04), 0.1, 0.3),
}


def material(name):
    if name in MATERIAL_INDEX:
        return MATERIAL_INDEX[name]
    if name == 'products':
        mat = _atlas_material()
    elif name == 'cardboard':
        mat = _cardboard_material()
    else:
        colour, metallic, roughness = PALETTE[name]
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Base Color'].default_value = (*colour, 1.0)
        bsdf.inputs['Metallic'].default_value = metallic
        bsdf.inputs['Roughness'].default_value = roughness
    MATERIALS.append(mat)
    MATERIAL_INDEX[name] = len(MATERIALS) - 1
    return MATERIAL_INDEX[name]


ATLAS_TILES = 8          # 8 x 8 packaging designs in one texture


def _cardboard_material():
    """Kraft cardboard with a darker tape stripe across the middle and faint fibre noise."""
    size = 128
    rng = np.random.default_rng(5)
    base = np.array([0.62, 0.45, 0.27], np.float32)
    pixels = np.ones((size, size, 4), np.float32)
    pixels[..., :3] = base * rng.uniform(0.93, 1.03, (size, size, 1)).astype(np.float32)
    pixels[size // 2 - 6:size // 2 + 6, :, :3] = np.array([0.52, 0.37, 0.21], np.float32)
    image = bpy.data.images.new('cardboard', size, size)
    image.pixels.foreach_set(pixels.ravel())
    image.pack()
    mat = bpy.data.materials.new('cardboard')
    mat.use_nodes = True
    texture = mat.node_tree.nodes.new('ShaderNodeTexImage'); texture.image = image
    bsdf = mat.node_tree.nodes['Principled BSDF']
    mat.node_tree.links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.85
    return mat


def _atlas_material():
    """A 512 px atlas of 64 brand-free packaging designs: colour bands, label panels, print lines."""
    size, tile = 512, 512 // ATLAS_TILES
    rng = np.random.default_rng(SPEC['seed'] * 7 + 3)
    palette = np.array([[0.70, 0.06, 0.05], [0.85, 0.12, 0.08], [0.93, 0.92, 0.88], [0.97, 0.96, 0.93],
                        [0.95, 0.55, 0.08], [0.95, 0.80, 0.15], [0.12, 0.45, 0.20], [0.05, 0.28, 0.12],
                        [0.10, 0.30, 0.65], [0.05, 0.12, 0.35], [0.45, 0.28, 0.14], [0.10, 0.10, 0.10],
                        [0.55, 0.20, 0.55], [0.90, 0.45, 0.60], [0.65, 0.68, 0.70], [0.05, 0.55, 0.65]], np.float32)
    weights = np.array([3, 2, 3, 2, 1.5, 1.5, 1.2, 0.8, 1.5, 1, 1.2, 1, 0.6, 0.8, 0.8, 0.6]); weights /= weights.sum()
    pixels = np.zeros((size, size, 4), np.float32)
    pixels[..., 3] = 1
    yy, xx = np.mgrid[:tile, :tile]
    for ty in range(ATLAS_TILES):
        for tx in range(ATLAS_TILES):
            base, band, label, ink = palette[rng.choice(len(palette), 4, replace=False, p=weights)]
            block = np.empty((tile, tile, 3), np.float32)
            block[:] = base
            top = int(tile * rng.uniform(0.06, 0.25)); block[:top] = band
            if rng.random() < 0.5:
                block[tile - int(tile * rng.uniform(0.05, 0.15)):] = band
            ly, lh = int(tile * rng.uniform(0.3, 0.5)), int(tile * rng.uniform(0.18, 0.32))
            lx0, lx1 = int(tile * rng.uniform(0.05, 0.2)), int(tile * rng.uniform(0.8, 0.95))
            block[ly:ly + lh, lx0:lx1] = label
            for line in range(int(rng.integers(1, 4))):              # print lines inside the label
                yl = ly + int(lh * (0.25 + 0.25 * line))
                if yl < ly + lh - 1:
                    block[yl:yl + 2, lx0 + 3:lx0 + 3 + int((lx1 - lx0 - 6) * rng.uniform(0.4, 1.0))] = ink
            if rng.random() < 0.5:                                     # product photo blob
                cx, cy, r = tile * rng.uniform(0.3, 0.7), tile * rng.uniform(0.7, 0.9), tile * rng.uniform(0.08, 0.16)
                block[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = palette[rng.choice(len(palette), p=weights)]
            block *= np.linspace(1.0, 0.86, tile, dtype=np.float32)[:, None, None]   # soft top-down shading
            pixels[ty * tile:(ty + 1) * tile, tx * tile:(tx + 1) * tile, :3] = block
    image = bpy.data.images.new('product_atlas', size, size)
    image.pixels.foreach_set(pixels.ravel())
    image.pack()
    mat = bpy.data.materials.new('products')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes['Principled BSDF']
    texture = nodes.new('ShaderNodeTexImage'); texture.image = image
    links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.42
    return mat


def atlas_tile(index=None):
    index = int(RNG.integers(ATLAS_TILES * ATLAS_TILES)) if index is None else index % (ATLAS_TILES * ATLAS_TILES)
    tx, ty = index % ATLAS_TILES, index // ATLAS_TILES
    step = 1.0 / ATLAS_TILES
    inset = step * 0.04
    # Blender image rows start at the bottom; atlas rows were written top-down.
    v0 = 1.0 - (ty + 1) * step
    return (tx * step + inset, v0 + inset, (tx + 1) * step - inset, v0 + step - inset)


# ---------------------------------------------------------------- geometry

def box(x0, y0, z0, x1, y1, z1, mat, tile=None, skip=(), top_uv=None, peak=0.0):
    """Axis-aligned box; `tile` maps an atlas tile on its front (-Y) face, plain UVs elsewhere."""
    if x1 - x0 < 1e-4 or y1 - y0 < 1e-4 or z1 - z0 < 1e-4:
        return
    index = material(mat)
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
               (x0, y0, z1), (x1, y0, z1), (x1, y1, z1 + peak), (x0, y1, z1 + peak)]
    verts = [MESH.verts.new(c) for c in corners]
    faces = {'bottom': (0, 3, 2, 1), 'top': (4, 5, 6, 7), 'front': (0, 1, 5, 4),
             'back': (2, 3, 7, 6), 'left': (3, 0, 4, 7), 'right': (1, 2, 6, 5)}
    for name, order in faces.items():
        if name in skip:
            continue
        face = MESH.faces.new([verts[i] for i in order])
        face.material_index = index
        if tile and name in ('front', 'back', 'left', 'right'):
            u0, v0, u1, v1 = tile
            uvs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        elif name == 'top' and top_uv:
            u0, v0, u1, v1 = top_uv
            uvs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        elif tile:
            u0, v0, u1, v1 = tile
            uvs = [(u0, v1 - (v1 - v0) * 0.2), (u1, v1 - (v1 - v0) * 0.2), (u1, v1), (u0, v1)]
        else:
            uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for loop, uv in zip(face.loops, uvs):
            loop[UV].uv = uv


def cylinder(cx, cy, z0, radius, height, mat, segments=16, tile=None, peak=0.0):
    index = material(mat)
    bottom, top = [], []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        bottom.append(MESH.verts.new((cx + radius * math.cos(a), cy + radius * math.sin(a), z0)))
        top.append(MESH.verts.new((cx + radius * math.cos(a), cy + radius * math.sin(a), z0 + height)))
    for i in range(segments):
        j = (i + 1) % segments
        face = MESH.faces.new([bottom[i], bottom[j], top[j], top[i]])
        face.material_index = index
        if tile:
            u0, v0, u1, v1 = tile
            uvs = [(u0 + (u1 - u0) * i / segments, v0), (u0 + (u1 - u0) * (i + 1) / segments, v0),
                   (u0 + (u1 - u0) * (i + 1) / segments, v1), (u0 + (u1 - u0) * i / segments, v1)]
        else:
            uvs = [(i / segments, 0), ((i + 1) / segments, 0), ((i + 1) / segments, 1), (i / segments, 1)]
        for loop, uv in zip(face.loops, uvs):
            loop[UV].uv = uv
    caps = [(bottom, True)] + ([] if peak > 0 else [(top, False)])
    for ring, reverse in caps:
        face = MESH.faces.new(list(reversed(ring)) if reverse else ring)
        face.material_index = index
        for loop in face.loops:
            co = loop.vert.co
            loop[UV].uv = (0.5 + (co.x - cx) / (2 * radius) * 0.9, 0.5 + (co.y - cy) / (2 * radius) * 0.9)
    if peak > 0:                                  # shallow cone: no flat top that could hold anything
        apex = MESH.verts.new((cx, cy, z0 + height + peak))
        for i in range(segments):
            face = MESH.faces.new([top[i], top[(i + 1) % segments], apex])
            face.material_index = index
            for loop in face.loops:
                co = loop.vert.co
                loop[UV].uv = (0.5 + (co.x - cx) / (2 * radius) * 0.9, 0.5 + (co.y - cy) / (2 * radius) * 0.9)


# ---------------------------------------------------------------- merchandise

def stock_level(x0, x1, y0, y1, z0, clear, front_gap=0.01):
    """Fill one shelf level with rows of product blocks; returns nothing, adds geometry."""
    if clear < 0.06 or x1 - x0 < 0.08 or y1 - y0 < 0.05:
        return
    fill = P.get('fill', 0.85)
    x = x0 + 0.005
    while x < x1 - 0.03:
        width = float(RNG.uniform(0.05, 0.16))
        if x + width > x1 - 0.005:
            width = x1 - 0.005 - x
            if width < 0.03:
                break
        if RNG.random() > fill:                          # a gap on the shelf
            x += width + 0.004
            continue
        height = float(min(clear - 0.012, clear * RNG.uniform(0.55, 0.92)))
        depth = float((y1 - y0) * RNG.uniform(0.75, 0.97))
        tile = atlas_tile()
        if RNG.random() < 0.25 and width < 0.11:          # bottles and cans as cylinders
            r = min(width, depth) / 2 * 0.95
            for row in range(max(1, int(depth / (2 * r + 0.004)))):
                cy = y0 + front_gap + r + row * (2 * r + 0.004)
                if cy + r > y1:
                    break
                cylinder(x + width / 2, cy, z0, r, height, 'products', 10, tile, peak=0.004)
        else:
            box(x, y0 + front_gap, z0, x + width, y0 + front_gap + depth, z0 + height, 'products', tile, peak=0.004)
        x += width + 0.004


def stocked_bay(x0, x1, y_front, depth, z_bottom, z_top, levels, board='shelf_board', facing=-1, stocked=True):
    """Shelf boards (and stock) between z_bottom and z_top, facing -Y (facing=-1) or +Y (+1)."""
    thickness = 0.02
    zs = np.linspace(z_bottom, z_top, levels + 1)[:-1]
    gap = (z_top - z_bottom) / levels
    for z in zs:
        if facing < 0:
            box(x0, y_front, z, x1, y_front + depth, z + thickness, board)
            if stocked:
                stock_level(x0, x1, y_front, y_front + depth - 0.01, z + thickness, gap - thickness - 0.03)
        else:
            box(x0, y_front - depth, z, x1, y_front, z + thickness, board)
            if stocked:
                _stock_backwards(x0, x1, y_front - depth + 0.01, y_front, z + thickness, gap - thickness - 0.03)


def _stock_backwards(x0, x1, y0, y1, z0, clear):
    """Stock for the +Y face of a double-sided gondola: product fronts point +Y."""
    if clear < 0.06:
        return
    x = x0 + 0.005
    while x < x1 - 0.03:
        width = float(RNG.uniform(0.05, 0.16))
        if x + width > x1 - 0.005:
            width = x1 - 0.005 - x
            if width < 0.03:
                break
        if RNG.random() > P.get('fill', 0.85):
            x += width + 0.004; continue
        height = float(min(clear - 0.012, clear * RNG.uniform(0.55, 0.92)))
        depth = float((y1 - y0) * RNG.uniform(0.75, 0.97))
        tile = atlas_tile()
        box(x, y1 - 0.01 - depth, z0, x + width, y1 - 0.01, z0 + height, 'products', tile, peak=0.004)
        x += width + 0.004


# ---------------------------------------------------------------- archetypes

def shelving(bay_dividers):
    """Shelving stocked on both long faces (the back face is simply hidden when it stands
    against a wall). The top board carries stock along both edges; the clean strip
    between them is the support surface for items the model places on top."""
    levels = int(P.get('levels', max(3, min(6, round(H / 0.36)))))
    frame = P.get('frame', 'white_metal')
    half = D / 2
    spine = 0.012 if D < 0.5 else 0.02
    stock_on_top = P.get('top_stock', True)
    top_stock = min(0.28, H * 0.15) if stock_on_top else 0.0
    body = H - top_stock                                 # top board surface height
    board = 0.025
    box(-W / 2 + 0.03, -spine, 0, W / 2 - 0.03, spine, body - board, frame)       # spine
    box(-W / 2 + 0.03, -half, 0, W / 2 - 0.03, half, 0.1, frame)                  # plinth
    box(-W / 2, -half, 0, -W / 2 + 0.03, half, body - board, frame)               # end panels
    box(W / 2 - 0.03, -half, 0, W / 2, half, body - board, frame)
    edges = np.linspace(-W / 2 + 0.03, W / 2 - 0.03, max(1, round(W / 1.0)) + 1)
    if bay_dividers:
        for x in edges[1:-1]:
            box(x - 0.012, -half, 0.1, x + 0.012, half, body - board, frame)
    shelf_top = body - board
    if P.get('header', False) and H > 1.4:
        for sign in (-1, 1):
            y0, y1 = sorted((sign * half, sign * (half - 0.03)))
            box(-W / 2 + 0.03, y0, shelf_top - 0.1, W / 2 - 0.03, y1, shelf_top, P.get('header_colour', 'red_accent'))
        shelf_top -= 0.1
    face_depth = half - spine - 0.005
    for a, b in zip(edges[:-1], edges[1:]):
        x0, x1 = a + (0.012 if bay_dividers else 0), b - (0.012 if bay_dividers else 0)
        stocked_bay(x0, x1, -half, face_depth, 0.1, shelf_top, levels - 1, frame, -1, True)
        stocked_bay(x0, x1, half, face_depth, 0.1, shelf_top, levels - 1, frame, +1, True)
    if not stock_on_top:
        box(-W / 2, -half, body - board, W / 2, half, body, frame)                 # clear top board
        return
    edge = min(0.28, max(0.06, D * 0.32))
    # Top board: its upper face is three coplanar strips. Stock stands on the two edge
    # strips; the clean centre strip is a separate face that intake measures exactly.
    box(-W / 2, -half, body - board, W / 2, half, body, frame, skip=('top',))
    for y0, y1 in ((-half, -half + edge), (-half + edge, half - edge), (half - edge, half)):
        flat_top(-W / 2, y0, W / 2, y1, body, frame)
    clear = top_stock - 0.005
    stock_level(-W / 2 + 0.02, W / 2 - 0.02, -half, -half + edge, body, clear, front_gap=0.005)
    _stock_backwards(-W / 2 + 0.02, W / 2 - 0.02, half - edge, half + 0.005, body, clear)


def wall_shelf():
    shelving(bay_dividers=True)


def gondola():
    shelving(bay_dividers=False)


def glass_door_cooler():
    """Upright beverage cooler: cabinet, lit-looking interior, stocked shelves, glass doors."""
    doors = int(P.get('doors', max(1, min(4, round(W / 0.62)))))
    body = P.get('body', 'black_plastic')
    wall, top_h, base_h = 0.05, max(0.12, H * 0.08), 0.12
    box(-W / 2, -D / 2 + 0.06, 0, W / 2, D / 2, base_h, body)                  # base
    box(-W / 2, -D / 2, H - top_h, W / 2, D / 2, H, P.get('canopy', body))     # canopy
    box(-W / 2, D / 2 - wall, base_h, W / 2, D / 2, H - top_h, 'cooler_interior')
    box(-W / 2, -D / 2, base_h, -W / 2 + wall, D / 2, H - top_h, body)
    box(W / 2 - wall, -D / 2, base_h, W / 2, D / 2, H - top_h, body)
    inner_x0, inner_x1 = -W / 2 + wall, W / 2 - wall
    edges = np.linspace(inner_x0, inner_x1, doors + 1)
    for x in edges[1:-1]:
        box(x - 0.02, -D / 2, base_h, x + 0.02, D / 2 - wall, H - top_h, body)  # mullions
    levels = int(P.get('levels', 5))
    for a, b in zip(edges[:-1], edges[1:]):
        stocked_bay(a + 0.02, b - 0.02, -D / 2 + 0.06, D - 0.06 - wall - 0.01, base_h, H - top_h - 0.02,
                    levels, 'chrome', -1, True)
    # Glass doors are drawn as frames only: the intake admits opaque materials, and an opaque
    # pane would hide the stocked interior that dominates these cameras' view of a cooler.
    for a, b in zip(edges[:-1], edges[1:]):
        box(a + 0.01, -D / 2 - 0.02, base_h + 0.02, b - 0.01, -D / 2, base_h + 0.05, body)
        box(a + 0.01, -D / 2 - 0.02, H - top_h - 0.05, b - 0.01, -D / 2, H - top_h - 0.02, body)
        box(a + 0.01, -D / 2 - 0.02, base_h + 0.02, a + 0.035, -D / 2, H - top_h - 0.02, body)
        box(b - 0.035, -D / 2 - 0.02, base_h + 0.02, b - 0.01, -D / 2, H - top_h - 0.02, body)
        handle_x = b - 0.07 if (b - a) > 0.3 else (a + b) / 2
        box(handle_x - 0.012, -D / 2 - 0.05, H * 0.35, handle_x + 0.012, -D / 2 - 0.02, H * 0.65, 'chrome')


def open_cooler():
    """Open-front multideck chiller: no doors, stocked shelves, a front sill."""
    body = P.get('body', 'white_metal')
    top_h, base_h = max(0.18, H * 0.1), 0.25
    box(-W / 2, -D / 2, 0, W / 2, D / 2, base_h, body)
    box(-W / 2, -D / 2 + 0.15, H - top_h, W / 2, D / 2, H, body)
    box(-W / 2, D / 2 - 0.05, base_h, W / 2, D / 2, H - top_h, 'cooler_interior')
    box(-W / 2, -D / 2, base_h, -W / 2 + 0.05, D / 2, H - top_h, body)
    box(W / 2 - 0.05, -D / 2, base_h, W / 2, D / 2, H - top_h, body)
    stocked_bay(-W / 2 + 0.05, W / 2 - 0.05, -D / 2 + 0.12, D - 0.2, base_h, H - top_h - 0.02,
                int(P.get('levels', 5)), 'chrome', -1, True)
    box(-W / 2, -D / 2, base_h, W / 2, -D / 2 + 0.06, base_h + 0.12, P.get('accent', 'red_accent'))


def chest_freezer():
    """Chest freezer: insulated tub with an open top (lids are clear glass in the photos), stocked inside."""
    body = P.get('body', 'white_metal')
    well = min(0.3, H * 0.4)
    wall = 0.05
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H - well, body)                       # lower body, its top is the well floor
    box(-W / 2, -D / 2, H - well, W / 2, -D / 2 + wall, H, body)               # rim walls
    box(-W / 2, D / 2 - wall, H - well, W / 2, D / 2, H, body)
    box(-W / 2, -D / 2 + wall, H - well, -W / 2 + wall, D / 2 - wall, H, body)
    box(W / 2 - wall, -D / 2 + wall, H - well, W / 2, D / 2 - wall, H, body)
    stock_level(-W / 2 + wall + 0.01, W / 2 - wall - 0.01, -D / 2 + wall + 0.01, D / 2 - wall - 0.01, H - well, well - 0.04)
    if P.get('accent'):
        box(-W / 2, -D / 2 - 0.004, H * 0.3, W / 2, -D / 2, H * 0.5, P['accent'])


def checkout_counter():
    """Cashier counter: closed body, flat worktop with a measurable support surface, kick plinth."""
    body = P.get('body', 'white_plastic')
    top = P.get('top', 'counter_top')
    box(-W / 2 + 0.03, -D / 2 + 0.03, 0, W / 2 - 0.03, D / 2 - 0.03, 0.08, 'black_plastic')
    box(-W / 2, -D / 2, 0.08, W / 2, D / 2, H - 0.04, body)
    box(-W / 2 - 0.02, -D / 2 - 0.02, H - 0.04, W / 2 + 0.02, D / 2 + 0.02, H, top)
    if P.get('front_panel'):
        box(-W / 2 + 0.05, -D / 2 - 0.005, 0.2, W / 2 - 0.05, -D / 2, H - 0.12, P['front_panel'])
    if P.get('display_front', False):                                          # glass display front
        box(-W / 2 + 0.06, -D / 2 - 0.008, 0.25, W / 2 - 0.06, -D / 2, H - 0.1, 'glass')


def display_counter():
    """Glass display / tobacco counter: glass box on a cabinet base, stocked inside."""
    body = P.get('body', 'light_wood')
    base_h = min(0.35, H * 0.4)
    box(-W / 2, -D / 2, 0, W / 2, D / 2, base_h, body)
    box(-W / 2, -D / 2, base_h, -W / 2 + 0.03, D / 2, H, 'chrome')
    box(W / 2 - 0.03, -D / 2, base_h, W / 2, D / 2, H, 'chrome')
    levels = max(1, int((H - base_h) / 0.22))
    stocked_bay(-W / 2 + 0.04, W / 2 - 0.04, -D / 2 + 0.03, D - 0.06, base_h, H - 0.03, levels, 'chrome', -1, True)
    box(-W / 2 + 0.03, -D / 2 - 0.006, H - 0.05, W / 2 - 0.03, -D / 2, H - 0.02, 'chrome')   # front rail, no pane
    box(-W / 2, -D / 2, H - 0.02, W / 2, D / 2, H, 'glass')                    # top glass (a support surface)


def tobacco_cabinet():
    """Tall back-of-counter cabinet: open stocked shelves above a closed base."""
    body = P.get('body', 'light_wood')
    base_h = min(0.9, H * 0.4)
    box(-W / 2, -D / 2, 0, W / 2, D / 2, base_h, body)
    box(-W / 2, D / 2 - 0.02, base_h, W / 2, D / 2, H, body)
    box(-W / 2, -D / 2, base_h, -W / 2 + 0.02, D / 2, H, body)
    box(W / 2 - 0.02, -D / 2, base_h, W / 2, D / 2, H, body)
    box(-W / 2, -D / 2, H - 0.03, W / 2, D / 2, H, body)
    levels = max(2, int((H - base_h) / 0.22))
    stocked_bay(-W / 2 + 0.02, W / 2 - 0.02, -D / 2, D - 0.03, base_h, H - 0.03, levels, body, -1, True)


def storage_cabinet():
    """Closed cabinet with doors and a flat top."""
    body = P.get('body', 'light_wood')
    box(-W / 2, -D / 2, 0.05, W / 2, D / 2, H - 0.02, body)
    box(-W / 2 + 0.02, -D / 2 + 0.02, 0, W / 2 - 0.02, D / 2 - 0.02, 0.05, 'black_plastic')
    box(-W / 2 - 0.01, -D / 2 - 0.01, H - 0.02, W / 2 + 0.01, D / 2 + 0.01, H, P.get('top', body))
    doors = max(1, round(W / 0.5))
    for x in np.linspace(-W / 2, W / 2, doors + 1)[1:-1]:
        box(x - 0.003, -D / 2 - 0.004, 0.08, x + 0.003, -D / 2, H - 0.05, 'dark_metal')
    for a, b in zip(np.linspace(-W / 2, W / 2, doors + 1)[:-1], np.linspace(-W / 2, W / 2, doors + 1)[1:]):
        hx = b - 0.06 if doors > 1 else W / 2 - 0.08
        box(hx - 0.008, -D / 2 - 0.025, H * 0.55, hx + 0.008, -D / 2 - 0.004, H * 0.75, 'chrome')


def stacked_cartons():
    """A carton: plain cardboard (tape drawn on as a texture band) or printed packaging.

    The top is one flat face, so the whole lid measures as a support surface
    and cartons can be stacked.
    """
    if P.get('plain', True):
        # One closed box (consistent outward normals); sides use the plain lower part of the
        # texture and the top shows the tape stripe.
        box(-W / 2, -D / 2, 0, W / 2, D / 2, H, 'cardboard', tile=(0.0, 0.0, 1.0, 0.4), top_uv=(0.0, 0.0, 1.0, 1.0))
    else:
        box(-W / 2, -D / 2, 0, W / 2, D / 2, H, 'products', atlas_tile())


def flat_top(x0, y0, x1, y1, z, mat):
    index = material(mat)
    verts = [MESH.verts.new(c) for c in ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))]
    face = MESH.faces.new(verts)
    face.material_index = index
    for loop, uv in zip(face.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop[UV].uv = uv


def bottle_multipack():
    """Shrink-wrapped pack of water bottles, laid out as a grid of cylinders on a tray."""
    cols, rows = int(P.get('cols', 4)), int(P.get('rows', 3))
    r = min(W / cols, D / rows) / 2 * 0.95
    box(-W / 2, -D / 2, 0, W / 2, D / 2, 0.01, 'cardboard')
    for i in range(cols):
        for j in range(rows):
            cx = -W / 2 + (i + 0.5) * W / cols; cy = -D / 2 + (j + 0.5) * D / rows
            cylinder(cx, cy, 0.01, r, H - 0.04, 'water_plastic', 12)
            cylinder(cx, cy, H - 0.035, r * 0.35, 0.03, 'cap_blue', 8)
    box(-W / 2, -D / 2, H - 0.004, W / 2, D / 2, H, 'water_plastic')          # shrink-wrap top


def bottle():
    cylinder(0, 0, 0, W / 2, H * 0.68, 'products', 16, atlas_tile())
    cylinder(0, 0, H * 0.68, W / 2 * 0.7, H * 0.14, P.get('glass', 'water_plastic'), 16)
    cylinder(0, 0, H * 0.82, W / 2 * 0.32, H * 0.12, P.get('glass', 'water_plastic'), 12)
    cylinder(0, 0, H * 0.94, W / 2 * 0.36, H * 0.06, P.get('cap', 'red_accent'), 12)


def can():
    cylinder(0, 0, 0, W / 2, H, 'products', 16, atlas_tile())
    cylinder(0, 0, H, W / 2 * 0.9, 0.004, 'chrome', 16)


def cup_noodle():
    cylinder(0, 0, 0, W / 2 * 0.78, H * 0.96, 'products', 18, atlas_tile())
    cylinder(0, 0, H * 0.96, W / 2, H * 0.04, 'white_plastic', 18)


def snack_bag():
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H * 0.9, 'products', atlas_tile())
    box(-W / 2, -D / 2 * 0.3, H * 0.9, W / 2, D / 2 * 0.3, H, 'products', atlas_tile())


def water_jug():
    cylinder(0, 0, 0, W / 2, H * 0.78, 'water_plastic', 18)
    cylinder(0, 0, H * 0.78, W / 2 * 0.6, H * 0.12, 'water_plastic', 18)
    cylinder(0, 0, H * 0.9, W / 2 * 0.25, H * 0.1, 'cap_blue', 12)


def pos_terminal():
    """Cash register / POS: base unit (with optional drawer) under a tilted screen, filling W x D x H."""
    base_h = H * 0.3
    box(-W / 2, -D / 2, 0, W / 2, D / 2, base_h * (0.75 if P.get('drawer', True) else 1), 'black_plastic')
    if P.get('drawer', True):
        box(-W / 2, -D / 2, base_h * 0.75, W / 2, D / 2, base_h, 'grey_plastic')
    box(-W / 2 * 0.25, -D / 2 * 0.1, base_h, W / 2 * 0.25, D / 2 * 0.3, H * 0.5, 'black_plastic')
    box(-W / 2 * 0.95, -D / 2 * 0.15, H * 0.5, W / 2 * 0.95, D / 2 * 0.05, H, 'black_plastic')
    box(-W / 2 * 0.88, -D / 2 * 0.15 - 0.003, H * 0.55, W / 2 * 0.88, -D / 2 * 0.15, H * 0.95, 'screen')


def payment_terminal():
    """Card / QR payment device on a stand."""
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H * 0.12, 'black_plastic')
    box(-W / 2 * 0.3, -D / 2 * 0.2, H * 0.12, W / 2 * 0.3, D / 2 * 0.2, H * 0.4, 'grey_plastic')
    box(-W / 2 * 0.9, -D / 2 * 0.25, H * 0.4, W / 2 * 0.9, D / 2 * 0.25, H, 'black_plastic')
    box(-W / 2 * 0.8, -D / 2 * 0.25 - 0.002, H * 0.48, W / 2 * 0.8, -D / 2 * 0.25, H * 0.92, 'screen')


def barcode_scanner():
    """Presentation scanner: weighted base and an angled head."""
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H * 0.25, 'black_plastic')
    box(-W / 2 * 0.6, -D / 2 * 0.2, H * 0.25, W / 2 * 0.6, D / 2 * 0.4, H, 'black_plastic')
    box(-W / 2 * 0.5, -D / 2 * 0.2 - 0.002, H * 0.45, W / 2 * 0.5, -D / 2 * 0.2, H * 0.9, P.get('window', 'red_accent'))


def monitor():
    box(-W / 2 * 0.3, -D / 2, 0, W / 2 * 0.3, D / 2, 0.015, 'black_plastic')
    box(-0.02, -0.02, 0.015, 0.02, 0.02, H * 0.3, 'black_plastic')
    box(-W / 2, -0.02, H * 0.2, W / 2, 0.01, H, 'black_plastic')
    box(-W / 2 + 0.01, -0.023, H * 0.2 + 0.01, W / 2 - 0.01, -0.02, H - 0.01, 'screen')


def scale():
    """Retail scale: body with a stainless weighing platform on top and a front display."""
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H * 0.82, 'white_plastic')
    box(-W / 2 * 0.98, -D / 2 * 0.98, H * 0.82, W / 2 * 0.98, D / 2 * 0.98, H, 'chrome')
    box(-W / 2 * 0.6, -D / 2 - 0.004, H * 0.15, W / 2 * 0.6, -D / 2, H * 0.65, 'screen')


def basket():
    """Shopping basket: open plastic tub with handles."""
    colour = P.get('colour', 'red_accent')
    t = 0.008
    box(-W / 2, -D / 2, 0, W / 2, D / 2, t, colour)
    box(-W / 2, -D / 2, 0, W / 2, -D / 2 + t, H, colour)
    box(-W / 2, D / 2 - t, 0, W / 2, D / 2, H, colour)
    box(-W / 2, -D / 2, 0, -W / 2 + t, D / 2, H, colour)
    box(W / 2 - t, -D / 2, 0, W / 2, D / 2, H, colour)


def basket_stack():
    colour = P.get('colour', 'red_accent')
    count = max(2, int(H / 0.07))
    for i in range(count):
        z = i * (H - 0.2) / max(1, count - 1)
        box(-W / 2, -D / 2, z, W / 2, D / 2, z + 0.02, colour)
    box(-W / 2, -D / 2, 0, -W / 2 + 0.01, D / 2, H, colour)
    box(W / 2 - 0.01, -D / 2, 0, W / 2, D / 2, H, colour)
    box(-W / 2, -D / 2, 0, W / 2, -D / 2 + 0.01, H, colour)
    box(-W / 2, D / 2 - 0.01, 0, W / 2, D / 2, H, colour)


def plastic_crate():
    """Stackable crate: open-sided look with a flat lid that measures as a support surface."""
    colour = P.get('colour', 'blue_accent')
    box(-W / 2, -D / 2, 0, W / 2, D / 2, 0.02, colour)
    for x0, x1, y0, y1 in ((-W / 2, W / 2, -D / 2, -D / 2 + 0.015), (-W / 2, W / 2, D / 2 - 0.015, D / 2),
                           (-W / 2, -W / 2 + 0.015, -D / 2, D / 2), (W / 2 - 0.015, W / 2, -D / 2, D / 2)):
        box(x0, y0, 0, x1, y1, H - 0.012, colour)
    if P.get('stocked', False):
        stock_level(-W / 2 + 0.02, W / 2 - 0.02, -D / 2 + 0.02, D / 2 - 0.02, 0.02, H - 0.05)
    box(-W / 2, -D / 2, H - 0.012, W / 2, D / 2, H, colour)


def wire_rack():
    """Freestanding wire display rack with stocked tiers."""
    tiers = int(P.get('levels', max(3, round(H / 0.35))))
    for x in (-W / 2, W / 2 - 0.02):
        for y in (-D / 2, D / 2 - 0.02):
            box(x, y, 0, x + 0.02, y + 0.02, H, 'chrome')
    stocked_bay(-W / 2 + 0.02, W / 2 - 0.02, -D / 2, D, 0.12, H - 0.02, tiers, 'chrome', -1, True)


def countertop_display():
    """Stepped merchandise display: side panels, back panel and stocked steps (counter or floor)."""
    steps = int(P.get('steps', 3 if H < 0.8 else 4))
    body = P.get('body', 'chrome')
    box(-W / 2, D / 2 - 0.015, 0, W / 2, D / 2, H, body)                       # back panel
    box(-W / 2, -D / 2, 0, -W / 2 + 0.015, D / 2 - 0.015, H, body)             # side panels
    box(W / 2 - 0.015, -D / 2, 0, W / 2, D / 2 - 0.015, H, body)
    # The stocked version keeps the front half of each step free for items the model puts
    # there; the clear twin (top_stock=False) leaves every step empty.
    for i in range(steps):
        z = H * i / steps; y0 = -D / 2 + (D - 0.015) * i / steps * 0.6
        box(-W / 2 + 0.015, y0, z, W / 2 - 0.015, D / 2 - 0.015, z + 0.012, body)
        if P.get('top_stock', True):
            stock_from = y0 + (D / 2 - 0.015 - y0) * 0.5
            stock_level(-W / 2 + 0.02, W / 2 - 0.02, stock_from, D / 2 - 0.02, z + 0.012, H / steps - 0.02)


def produce_display():
    """Low angled produce table filled with produce-coloured blocks."""
    body = P.get('body', 'wood')
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H * 0.75, body)
    stock_level(-W / 2 + 0.03, W / 2 - 0.03, -D / 2 + 0.03, D / 2 - 0.03, H * 0.75, H * 0.25)


def water_dispenser():
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H * 0.75, 'white_plastic')
    box(-W / 2 * 0.5, -D / 2 - 0.03, H * 0.42, W / 2 * 0.5, -D / 2, H * 0.47, 'grey_plastic')
    cylinder(0, 0, H * 0.75, min(W, D) / 2 * 0.85, H * 0.25, 'water_plastic', 18)


def fire_extinguisher():
    cylinder(0, 0, 0, W / 2 * 0.85, H * 0.82, 'fire_red', 16)
    cylinder(0, 0, H * 0.82, W / 2 * 0.3, H * 0.1, 'black_plastic', 10)
    box(-W / 2, -0.01, H * 0.9, W / 2 * 0.2, 0.01, H, 'black_plastic')


def bin_():
    cylinder(0, 0, 0, W / 2, H * 0.94, P.get('colour', 'grey_plastic'), 18)
    cylinder(0, 0, H * 0.94, W / 2 * 1.02, H * 0.06, P.get('lid', 'dark_metal'), 18)


def kiosk():
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H * 0.08, 'dark_metal')
    box(-W / 2 * 0.6, -D / 2 * 0.6, H * 0.08, W / 2 * 0.6, D / 2 * 0.6, H * 0.55, 'white_metal')
    box(-W / 2, -D / 2, H * 0.55, W / 2, D / 2, H, 'white_metal')
    box(-W / 2 * 0.85, -D / 2 - 0.003, H * 0.6, W / 2 * 0.85, -D / 2, H * 0.95, 'screen')


def appliance_box(colour='white_plastic', window=True):
    """Generic small appliance (microwave, printer, steamer, kettle base)."""
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H, colour)
    if window:
        box(-W / 2 * 0.85, -D / 2 - 0.003, H * 0.15, W / 2 * 0.45, -D / 2, H * 0.85, 'screen')


def sink():
    """Freestanding stainless sink: cabinet frame, basin top with a measurable drainboard."""
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H - 0.04, 'chrome')
    basin_w = min(W * 0.55, W - 0.1)
    box(-W / 2, -D / 2, H - 0.04, W / 2 - basin_w - 0.02, D / 2, H, 'chrome')            # drainboard
    box(W / 2 - 0.02, -D / 2, H - 0.04, W / 2, D / 2, H, 'chrome')
    box(W / 2 - basin_w - 0.02, -D / 2, H - 0.04, W / 2 - 0.02, -D / 2 + 0.04, H, 'chrome')
    box(W / 2 - basin_w - 0.02, D / 2 - 0.04, H - 0.04, W / 2 - 0.02, D / 2, H, 'chrome')
    box(W / 2 - basin_w - 0.02, -D / 2 + 0.04, H - 0.2, W / 2 - 0.02, D / 2 - 0.04, H - 0.18, 'grey_metal')


def tub():
    """Open tub or basin (wash basin, bucket, wire basket): walls and a floor, no lid."""
    colour = P.get('colour', 'grey_plastic')
    t = max(0.006, min(W, D) * 0.03)
    box(-W / 2, -D / 2, 0, W / 2, D / 2, t, colour)
    box(-W / 2, -D / 2, 0, W / 2, -D / 2 + t, H, colour)
    box(-W / 2, D / 2 - t, 0, W / 2, D / 2, H, colour)
    box(-W / 2, -D / 2, 0, -W / 2 + t, D / 2, H, colour)
    box(W / 2 - t, -D / 2, 0, W / 2, D / 2, H, colour)


def sack():
    """Filled sack: a slightly tapered printed block with a tied neck."""
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H * 0.85, 'products', atlas_tile())
    box(-W / 2 * 0.35, -D / 2 * 0.35, H * 0.85, W / 2 * 0.35, D / 2 * 0.35, H, P.get('colour', 'light_wood'))


def plant():
    """Potted plant: pot plus leaf blocks reaching the full footprint."""
    pot_h = H * 0.38
    cylinder(0, 0, 0, min(W, D) / 2 * 0.62, pot_h, P.get('pot', 'white_plastic'), 16)
    for i in range(8):
        a = 2 * math.pi * i / 8
        cx, cy = math.cos(a) * W * 0.3, math.sin(a) * D * 0.3
        box(max(-W / 2, cx - W * 0.2), max(-D / 2, cy - D * 0.2), pot_h, min(W / 2, cx + W * 0.2), min(D / 2, cy + D * 0.2),
            H * float(RNG.uniform(0.78, 1.0)), 'green_accent')
    box(-W * 0.12, -D * 0.12, pot_h, W * 0.12, D * 0.12, H, 'green_accent')


def flask():
    cylinder(0, 0, 0, W / 2, H * 0.88, P.get('colour', 'chrome'), 18)
    cylinder(0, 0, H * 0.88, W / 2 * 0.55, H * 0.12, 'black_plastic', 14)


def fan():
    cylinder(0, 0, 0, W / 2 * 0.6, H * 0.06, 'white_plastic', 16)
    box(-0.02, -0.02, H * 0.06, 0.02, 0.02, H * 0.5, 'white_plastic')
    cylinder(0, 0, H * 0.5, W / 2 * 0.25, H * 0.1, 'white_plastic', 12)
    box(-W / 2, -D / 2, H * 0.3, W / 2, -D / 2 + 0.02, H, P.get('colour', 'grey_plastic'))


def mop():
    box(-W / 2, -D / 2, 0, W / 2, D / 2, 0.05, 'grey_plastic')
    cylinder(0, 0, 0.05, 0.012, H - 0.05, P.get('colour', 'blue_accent'), 8)


def ladder():
    for x in (-W / 2, W / 2 - 0.03):
        box(x, -D / 2, 0, x + 0.03, -D / 2 + 0.04, H, 'grey_metal')
        box(x, D / 2 - 0.04, 0, x + 0.03, D / 2, H * 0.92, 'grey_metal')
    for z in np.linspace(H * 0.25, H * 0.95, 3):
        box(-W / 2, -D / 2, z - 0.02, W / 2, -D / 2 + 0.12, z, 'grey_metal')


def desk():
    """Simple office desk / work table with a flat top."""
    top = P.get('top', 'light_wood')
    for x in (-W / 2 + 0.03, W / 2 - 0.06):
        for y in (-D / 2 + 0.03, D / 2 - 0.06):
            box(x, y, 0, x + 0.03, y + 0.03, H - 0.03, 'dark_metal')
    box(-W / 2, -D / 2, H - 0.03, W / 2, D / 2, H, top)


def cooler_box():
    """Portable cooler / ice box: body plus a slightly wider flat lid."""
    body = P.get('body', 'blue_accent')
    box(-W / 2 + 0.01, -D / 2 + 0.01, 0, W / 2 - 0.01, D / 2 - 0.01, H - 0.05, body)
    box(-W / 2, -D / 2, H - 0.05, W / 2, D / 2, H, 'white_plastic')


def cushion():
    box(-W / 2, -D / 2, 0, W / 2, D / 2, H, P.get('colour', 'blue_accent'))


ARCHETYPES = {
    'wall_shelf': wall_shelf, 'gondola': gondola, 'glass_door_cooler': glass_door_cooler,
    'open_cooler': open_cooler, 'chest_freezer': chest_freezer, 'checkout_counter': checkout_counter,
    'display_counter': display_counter, 'tobacco_cabinet': tobacco_cabinet, 'storage_cabinet': storage_cabinet,
    'carton': stacked_cartons, 'bottle_multipack': bottle_multipack, 'bottle': bottle, 'can': can,
    'cup_noodle': cup_noodle, 'snack_bag': snack_bag, 'water_jug': water_jug, 'pos_terminal': pos_terminal,
    'payment_terminal': payment_terminal, 'barcode_scanner': barcode_scanner, 'monitor': monitor,
    'scale': scale, 'basket': basket, 'basket_stack': basket_stack, 'plastic_crate': plastic_crate,
    'wire_rack': wire_rack, 'countertop_display': countertop_display, 'produce_display': produce_display,
    'water_dispenser': water_dispenser, 'fire_extinguisher': fire_extinguisher, 'bin': bin_, 'kiosk': kiosk,
    'appliance': lambda: appliance_box(P.get('colour', 'white_plastic'), P.get('window', True)),
    'sink': sink, 'tub': tub, 'sack': sack, 'plant': plant, 'flask': flask, 'fan': fan, 'mop': mop,
    'ladder': ladder, 'desk': desk, 'cushion': cushion, 'cooler_box': cooler_box,
}

ARCHETYPES[SPEC['archetype']]()
if ORIENT_Y:
    bmesh.ops.rotate(MESH, verts=MESH.verts, cent=(0, 0, 0),
                     matrix=__import__('mathutils').Matrix.Rotation(math.radians(90), 3, 'Z'))

# Snap to the planned size so the library offers exactly the sizes that were planned. Small
# protrusions (handles, rims) get absorbed; a large mismatch means an archetype bug.
coords = np.array([v.co[:] for v in MESH.verts])
low, high = coords.min(0), coords.max(0)
target = np.array(SPEC['size_m'], float)
factor = target / np.maximum(high - low, 1e-9)
if np.any(factor < 0.75) or np.any(factor > 1.34):
    raise SystemExit(f"archetype {SPEC['archetype']} built {np.round(high - low, 3).tolist()} for planned {SPEC['size_m']}")
centre = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
for v in MESH.verts:
    v.co = __import__('mathutils').Vector(((np.array(v.co[:]) - centre) * factor).tolist())

# Every face now exists; build one object, recalc normals outward, and export.
mesh = bpy.data.meshes.new(SPEC['asset_id'])
MESH.to_mesh(mesh)
MESH.free()
for mat in MATERIALS:
    mesh.materials.append(mat)
obj = bpy.data.objects.new(SPEC['asset_id'], mesh)
bpy.context.scene.collection.objects.link(obj)
coords = np.array([v.co[:] for v in mesh.vertices])
low, high = coords.min(0), coords.max(0)
size = high - low
OUT.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action='DESELECT'); obj.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(OUT / 'model.glb'), export_format='GLB', use_selection=True,
                          export_yup=True, export_cameras=False, export_lights=False, export_animations=False,
                          export_image_format='JPEG')
metadata = {'schema_version': 'scene_asset_input_v1', 'category': SPEC['category'], 'placement': SPEC['placement'],
            'longest_dimension_m': round(float(size.max()), 6), 'scale_source': 'measured'}
(OUT / 'asset.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(OUT / 'build_report.json').write_text(json.dumps({
    'spec': SPEC, 'built_size_m': [round(float(v), 4) for v in size], 'snap_factor': [round(float(v), 4) for v in factor],
    'bottom_z': round(float(low[2]), 5),
    'centre_xy': [round(float(v), 4) for v in (low[:2] + high[:2]) / 2], 'triangles': sum(len(p.vertices) - 2 for p in mesh.polygons),
    'materials': [m.name for m in MATERIALS]}, ensure_ascii=False) + '\n', encoding='utf-8')
print('BUILT', SPEC['asset_id'], [round(float(v), 3) for v in size])
