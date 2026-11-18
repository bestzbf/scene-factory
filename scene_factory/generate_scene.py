"""Deterministic, data-only scene specifications to editable Blender scenes.

Run with Blender: blender -b --python generate_scene.py -- --spec input.json
--output-dir output --width 640 --height 480 --samples 16

Coordinates are metres, Z up; object positions anchor their bottom centre.
Furniture fronts face local -Y. Geometry is a reusable approximation, not a
measurement or an assertion about hidden surfaces in the reference photograph.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector
try:
    from .assets_lib.material_roles import source_role
except ImportError:
    from assets_lib.material_roles import source_role


MATERIALS = {}
MESHES = {}


def material(name, color, roughness=.55, metallic=0, emission=0):
    key = (tuple(color), round(roughness, 4), metallic, emission)
    if key in MATERIALS:
        return MATERIALS[key]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    shader.inputs['Base Color'].default_value = (*color[:3], 1)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Metallic'].default_value = metallic
    if emission:
        shader.inputs['Emission Color'].default_value = (*color[:3], 1)
        shader.inputs['Emission Strength'].default_value = emission
    MATERIALS[key] = mat
    return mat


def window_glass_material():
    """Architectural glazing with efficient, explicit shadow transmission.

    Ordinary rays see dielectric glass. Shadow rays use a neutral transparent
    approximation, so an exterior area light can illuminate the interior
    without requiring difficult refractive-caustic sampling. Two panel faces
    each pass 96% of the shadow-ray energy (about 92% combined). This is not
    measured glass photometry and does not generate refractive caustics.
    """
    key = ('architectural_window_glass', 1)
    if key in MATERIALS:
        return MATERIALS[key]
    mat = bpy.data.materials.new('WindowGlass')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = next(node for node in nodes if node.type == 'BSDF_PRINCIPLED')
    shader.inputs['Base Color'].default_value = (.98, .98, .98, 1)
    shader.inputs['Roughness'].default_value = .025
    shader.inputs['Metallic'].default_value = 0
    shader.inputs['Transmission Weight'].default_value = 1
    shader.inputs['IOR'].default_value = 1.45
    shader.inputs['Emission Strength'].default_value = 0
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    transparent.inputs['Color'].default_value = (.96, .96, .96, 1)
    ray = nodes.new('ShaderNodeLightPath')
    mix = nodes.new('ShaderNodeMixShader')
    mix.name = 'ArchitecturalShadowTransmission'
    links.new(ray.outputs['Is Shadow Ray'], mix.inputs[0])
    links.new(shader.outputs['BSDF'], mix.inputs[1])
    links.new(transparent.outputs['BSDF'], mix.inputs[2])
    output = next(node for node in nodes if node.type == 'OUTPUT_MATERIAL')
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    mat['glazing_transport'] = 'dielectric_camera_rays_transparent_shadow_approximation'
    mat['shadow_transmission_per_face'] = .96
    mat['photometry_measured'] = False
    mat['transport_limitations'] = 'No refractive caustics; neutral clear architectural glazing approximation.'
    MATERIALS[key] = mat
    return mat


def mesh_object(name, mesh, parent, pos, mat):
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    obj.location = pos
    if not mesh.materials:
        mesh.materials.append(mat)
    obj.material_slots[0].link = 'OBJECT'
    obj.material_slots[0].material = mat
    return obj


def box(root, name, pos, size, mat, bevel=.012):
    size = tuple(max(.0001, float(x)) for x in size)
    bevel = min(bevel, min(size) * .24)
    key = ('box', size, bevel)
    if key not in MESHES:
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1)
        for v in bm.verts:
            v.co.x *= size[0]
            v.co.y *= size[1]
            v.co.z *= size[2]
        if bevel:
            bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel,
                            segments=2, affect='EDGES')
        bm.normal_update()
        mesh = bpy.data.meshes.new(name + '_Mesh')
        bm.to_mesh(mesh)
        bm.free()
        MESHES[key] = mesh
    return mesh_object(name, MESHES[key], root, pos, mat)


def sphere(root, name, pos, size, mat):
    key = ('sphere',)
    if key not in MESHES:
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=.5)
        mesh = bpy.data.meshes.new('SharedIcosphere')
        bm.to_mesh(mesh)
        bm.free()
        for poly in mesh.polygons:
            poly.use_smooth = True
        MESHES[key] = mesh
    obj = mesh_object(name, MESHES[key], root, pos, mat)
    obj.scale = size
    return obj


def rod(root, name, a, b, radius, mat, vertices=12):
    key = ('cylinder', vertices)
    if key not in MESHES:
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=vertices,
                             radius1=1, radius2=1, depth=1)
        mesh = bpy.data.meshes.new('SharedCylinder')
        bm.to_mesh(mesh)
        bm.free()
        for poly in mesh.polygons:
            poly.use_smooth = len(poly.vertices) == 4
        MESHES[key] = mesh
    a, b = Vector(a), Vector(b)
    obj = mesh_object(name, MESHES[key], root, (a + b) / 2, mat)
    obj.scale = (radius, radius, (b - a).length)
    obj.rotation_euler = (b - a).to_track_quat('Z', 'Y').to_euler()
    return obj


def root_object(name):
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)
    root.empty_display_type = 'PLAIN_AXES'
    root.empty_display_size = .15
    return root


# Flat props lose all relief when authored in a unit cube and scaled down; these
# are built directly in metres instead, with an identity root scale.
METRIC_KINDS = frozenset({'rug', 'painting', 'book', 'book_row', 'tray',
                          'cushion', 'lamp', 'vase', 'pendant_light',
                          'bowl', 'plate', 'mug', 'pitcher', 'basket',
                          'candlestick', 'sculpture', 'ottoman', 'planter_box',
                          'radio', 'board_game'})


def _metric_details(root, kind, entry, box, rod, sphere, *, base, darker, pale, metal, black):
    """Real-size detail for flat or small objects, in metres, origin at bottom-centre."""
    w, d, h = (max(0.001, float(v)) for v in entry['dimensions'])

    if kind == 'rug':
        # A woven rug: the field slightly proud of the floor, a contrasting border
        # laid on top of it (not buried inside), and a fringe at both short ends.
        thickness = max(0.006, h)
        box(root, 'RugField', (0, 0, thickness / 2), (w, d, thickness), base, bevel=0.002)
        inset = min(0.05, min(w, d) * 0.06)
        for name, pos, size in (
                ('BorderN', (0, d / 2 - inset / 2, thickness + 0.0006), (w, inset, 0.0012)),
                ('BorderS', (0, -d / 2 + inset / 2, thickness + 0.0006), (w, inset, 0.0012)),
                ('BorderW', (-w / 2 + inset / 2, 0, thickness + 0.0006), (inset, d - 2 * inset, 0.0012)),
                ('BorderE', (w / 2 - inset / 2, 0, thickness + 0.0006), (inset, d - 2 * inset, 0.0012))):
            box(root, name, pos, size, darker, bevel=0.0)
        fringe = min(0.03, d * 0.03)
        for index in range(12):
            x = -w / 2 + (index + 0.5) * (w / 12)
            for y in (-d / 2 - fringe / 2, d / 2 + fringe / 2):
                rod(root, 'Fringe', (x, y, 0.001), (x, y, thickness * 0.6), 0.0012, pale)

    elif kind == 'painting':
        # Four moulding bars, a mat and a recessed canvas. A single bar across the
        # middle was what turned framed art into three white sticks on the wall.
        frame = min(0.045, min(w, h) * 0.08)
        depth = max(0.02, d)
        box(root, 'FrameTop', (0, -depth / 2, h - frame / 2), (w, depth, frame), darker, bevel=0.003)
        box(root, 'FrameBottom', (0, -depth / 2, frame / 2), (w, depth, frame), darker, bevel=0.003)
        box(root, 'FrameLeft', (-w / 2 + frame / 2, -depth / 2, h / 2),
            (frame, depth, h - 2 * frame), darker, bevel=0.003)
        box(root, 'FrameRight', (w / 2 - frame / 2, -depth / 2, h / 2),
            (frame, depth, h - 2 * frame), darker, bevel=0.003)
        # The frame's front face is at -depth/2, and more negative Y is closer to
        # the room. The mat must sit BEHIND the canvas, otherwise the pale mat
        # sheet covers the picture and the frame reads as an empty white panel.
        box(root, 'Mat', (0, -depth / 2 + 0.008, h / 2),
            (w - 2 * frame, 0.003, h - 2 * frame), pale, bevel=0.0)
        cw = max(0.05, w - 2 * frame - 0.06)
        ch = max(0.05, h - 2 * frame - 0.06)
        box(root, 'Canvas', (0, -depth / 2 + 0.004, h / 2), (cw, 0.003, ch), base, bevel=0.0)
        # A single flat field reads as a blank panel from across the room. Two
        # blocks in the frame's own two tones give the canvas a composition, so
        # it reads as art at the distance the stills are taken from.
        box(root, 'CanvasBand', (0, -depth / 2 + 0.0022, h * 0.36),
            (cw * 0.82, 0.002, ch * 0.26), darker, bevel=0.0)
        box(root, 'CanvasBlock', (cw * 0.16, -depth / 2 + 0.0014, h * 0.66),
            (cw * 0.30, 0.002, ch * 0.22), pale, bevel=0.0)

    elif kind == 'book':
        # Orientation matters: a shelf book stands upright (tall and thin) while a
        # coffee-table book lies flat. Building every book as a flat one turned a
        # bookshelf into a row of white slabs, because only the page block showed.
        if h > 1.8 * w:
            # Upright: two vertical covers, a page block between them, a spine on
            # one side and a visible head of pages at the top.
            board = max(0.003, w * 0.06)
            box(root, 'CoverFront', (0, -d / 2 + board / 2, h / 2), (w, board, h), base, bevel=0.001)
            box(root, 'CoverBack', (0, d / 2 - board / 2, h / 2), (w, board, h), base, bevel=0.001)
            box(root, 'Pages', (0, 0, h / 2), (w * 0.9, d - 2 * board, h * 0.94), pale, bevel=0.001)
            box(root, 'Spine', (-w / 2 + board / 2, 0, h / 2), (board * 2, d, h), darker, bevel=0.0015)
        else:
            pages = max(0.001, h * 0.7)
            box(root, 'Pages', (0, 0, h * 0.15 + pages / 2), (w * 0.96, d * 0.96, pages), pale, bevel=0.001)
            box(root, 'CoverTop', (0, 0, h - 0.0015), (w, d, 0.003), base, bevel=0.0015)
            box(root, 'CoverBottom', (0, 0, h * 0.15 - 0.0015), (w, d, 0.003), base, bevel=0.0015)
            box(root, 'Spine', (-w / 2 + 0.004, 0, h * 0.5), (0.008, d, h * 0.62), darker, bevel=0.002)

    elif kind == 'book_row':
        # A written brief says "a row of books", which the library can only offer
        # as one solid block. Built here it is what it claims to be: separate
        # volumes of varied width, height and colour, leaning against each other
        # along the board, with a couple that lean out of true.
        spines = ((0.42, 0.13, 0.11), (0.12, 0.20, 0.31), (0.52, 0.44, 0.27),
                  (0.16, 0.26, 0.20), (0.31, 0.16, 0.25), (0.10, 0.10, 0.11),
                  (0.62, 0.55, 0.42), (0.66, 0.35, 0.14), (0.34, 0.37, 0.40))
        thickness = 0.042
        count = max(3, min(26, int(w / thickness)))
        gap = 0.0015
        pitch = (w - gap) / count
        seed = int(abs(entry.get('dimensions', [1])[0] * 1000)) % 97
        cursor = -w / 2 + gap
        for index in range(count):
            wobble = ((seed + index * 37) % 11) / 10.0          # 0.0 .. 1.0
            spine_w = pitch * (0.78 + 0.2 * wobble) - gap
            tall = h * (0.84 + 0.16 * (((seed + index * 53) % 7) / 6.0))
            depth = d * (0.86 + 0.12 * (((seed + index * 29) % 5) / 4.0))
            tone = spines[(seed + index * 3) % len(spines)]
            cover = source_role(material('BookCover_%d' % index, tone, 0.52), 'body')
            pages = source_role(material('BookPages_%d' % index,
                                         (0.88, 0.86, 0.80), 0.74), 'body')
            centre_x = cursor + spine_w / 2
            board = max(0.0025, spine_w * 0.1)
            box(root, 'BookCoverA_%d' % index,
                (centre_x - spine_w / 2 + board / 2, 0, tall / 2),
                (board, depth, tall), cover, bevel=0.0012)
            box(root, 'BookCoverB_%d' % index,
                (centre_x + spine_w / 2 - board / 2, 0, tall / 2),
                (board, depth, tall), cover, bevel=0.0012)
            box(root, 'BookPages_%d' % index, (centre_x, 0, tall / 2),
                (max(0.004, spine_w - 2 * board), depth * 0.94, tall * 0.95),
                pages, bevel=0.0008)
            box(root, 'BookSpine_%d' % index,
                (centre_x, -depth / 2 + board / 2, tall / 2),
                (spine_w, board, tall * 0.995), cover, bevel=0.0012)
            cursor += pitch

    elif kind == 'bowl':
        # A bowl is a wall and a floor, not a solid lump: without the inner
        # surface it reads as a dome and a fruit bowl looks like a paperweight.
        radius = min(w, d) / 2
        rod(root, 'BowlFoot', (0, 0, 0), (0, 0, h * 0.10), radius * 0.46, darker)
        rod(root, 'BowlBody', (0, 0, h * 0.06), (0, 0, h), radius, base)
        rod(root, 'BowlWell', (0, 0, h * 0.26), (0, 0, h * 1.02), radius * 0.86, pale)

    elif kind == 'plate':
        radius = min(w, d) / 2
        rod(root, 'PlateFoot', (0, 0, 0), (0, 0, h * 0.22), radius * 0.52, darker)
        rod(root, 'PlateBody', (0, 0, h * 0.18), (0, 0, h), radius, base)
        rod(root, 'PlateWell', (0, 0, h * 0.62), (0, 0, h * 1.02), radius * 0.72, pale)

    elif kind == 'mug':
        radius = min(w, d) / 2
        rod(root, 'MugBody', (0, 0, 0), (0, 0, h), radius * 0.82, base)
        rod(root, 'MugWell', (0, 0, h * 0.14), (0, 0, h * 1.02), radius * 0.68, pale)
        # A handle: two stubs and a bar, which is enough at this size.
        for z in (h * 0.34, h * 0.72):
            box(root, 'MugHandleArm', (radius * 0.95, 0, z),
                (radius * 0.5, radius * 0.22, h * 0.1), base, bevel=0.002)
        box(root, 'MugHandleBar', (radius * 1.18, 0, h * 0.53),
            (radius * 0.2, radius * 0.22, h * 0.42), base, bevel=0.002)

    elif kind == 'pitcher':
        radius = min(w, d) / 2
        rod(root, 'PitcherBody', (0, 0, 0), (0, 0, h * 0.68), radius * 0.92, base)
        rod(root, 'PitcherShoulder', (0, 0, h * 0.62), (0, 0, h * 0.86), radius * 0.66, base)
        rod(root, 'PitcherNeck', (0, 0, h * 0.84), (0, 0, h), radius * 0.52, base)
        rod(root, 'PitcherRim', (0, 0, h * 0.96), (0, 0, h * 1.02), radius * 0.58, darker)
        box(root, 'PitcherSpout', (-radius * 0.72, 0, h * 0.94),
            (radius * 0.4, radius * 0.4, h * 0.1), base, bevel=0.004)
        box(root, 'PitcherHandle', (radius * 0.98, 0, h * 0.6),
            (radius * 0.22, radius * 0.26, h * 0.5), darker, bevel=0.006)

    elif kind == 'basket':
        # A closed vessel with the weave suggested by two inset bands. Separate
        # staves let daylight through and read as a slatted bin; bands that stood
        # proud of the body fought with its surface and lit up white.
        radius = min(w, d) / 2
        rod(root, 'BasketBody', (0, 0, 0), (0, 0, h * 0.96), radius, base)
        rod(root, 'BasketLowerBand', (0, 0, h * 0.14), (0, 0, h * 0.2), radius * 0.995, darker)
        rod(root, 'BasketUpperBand', (0, 0, h * 0.72), (0, 0, h * 0.78), radius * 0.995, darker)
        rod(root, 'BasketRim', (0, 0, h * 0.96), (0, 0, h), radius * 1.03, darker)
        # The mouth: a shallow recess so it does not read as a solid drum.
        rod(root, 'BasketMouth', (0, 0, h * 0.86), (0, 0, h * 0.965), radius * 0.9, black)

    elif kind == 'ottoman':
        # An upholstered pouf: a plump cushion body on short feet, not a crate.
        foot = min(0.06, h * 0.14)
        for sx in (-1, 1):
            for sy in (-1, 1):
                rod(root, 'OttomanFoot', (sx * w * 0.36, sy * d * 0.36, 0),
                    (sx * w * 0.36, sy * d * 0.36, foot), 0.018, black)
        body = box(root, 'OttomanBody', (0, 0, foot + (h - foot) / 2),
                   (w * 0.97, d * 0.97, h - foot), base, bevel=min(w, d, h) * 0.3)
        box(root, 'OttomanSeam', (0, 0, foot + (h - foot) * 0.55),
            (w * 0.99, d * 0.99, (h - foot) * 0.05), darker, bevel=0.005)

    elif kind == 'planter_box':
        # A trough with soil and a shrub in it. Left as a box it read as a grey
        # crate standing against the wall.
        lip = min(0.03, h * 0.1)
        box(root, 'PlanterBody', (0, 0, (h - lip) * 0.5), (w, d, h - lip), base,
            bevel=min(w, d) * 0.06)
        box(root, 'PlanterLip', (0, 0, h - lip * 0.5), (w * 1.04, d * 1.04, lip),
            darker, bevel=0.006)
        box(root, 'PlanterSoil', (0, 0, h - lip * 0.55),
            (w * 0.9, d * 0.9, lip * 0.35), black, bevel=0)
        leaf = source_role(material('PlanterLeaf', (.11, .26, .09), .72), 'body')
        # A low dense shrub. Each clump is a sphere of one radius in all three
        # axes; scaling the axes independently is what flattened them into a row
        # of green pucks sitting on the soil.
        clump = min(w, d) * 0.16
        tiers = ((0.30, 0.10), (0.62, 0.16), (0.88, 0.10))
        for tier, (lift, shrink) in enumerate(tiers):
            radius = clump * (1.0 - 0.18 * tier)
            count = max(4, int(w / (radius * 1.7)))
            for index in range(count):
                u = (index + 0.5) / count
                fx = -w * 0.42 + w * 0.84 * u
                fy = d * 0.18 * math.sin(index * 2.3 + tier * 1.1)
                fz = h * lift + radius * (0.7 + shrink)
                sphere(root, 'PlanterLeaf_%d_%d' % (tier, index), (fx, fy, fz),
                       (radius * 2.0, radius * 1.7, radius * 1.5), leaf)

    elif kind == 'radio':
        # A small table radio: cabinet, dial and speaker grille.
        box(root, 'RadioBody', (0, 0, h / 2), (w, d, h), base, bevel=min(w, h) * 0.08)
        box(root, 'RadioDial', (0, -d / 2 + 0.003, h * 0.66),
            (w * 0.5, 0.005, h * 0.16), pale, bevel=0)
        box(root, 'RadioGrille', (0, -d / 2 + 0.003, h * 0.34),
            (w * 0.62, 0.005, h * 0.34), darker, bevel=0)
        for side in (-1, 1):
            rod(root, 'RadioKnob', (side * w * 0.34, -d / 2, h * 0.18),
                (side * w * 0.34, -d / 2 - 0.012, h * 0.18), min(0.012, h * 0.09), black)

    elif kind == 'board_game':
        # A boxed game lying flat with pieces stacked on the lid.
        box(root, 'GameBox', (0, 0, h * 0.42), (w, d, h * 0.84), base,
            bevel=min(w, d) * 0.03)
        box(root, 'GameLid', (0, 0, h * 0.92), (w * 0.98, d * 0.98, h * 0.12),
            darker, bevel=0.002)
        for index in range(5):
            px = -w * 0.3 + w * 0.15 * index
            rod(root, 'GamePiece_%d' % index, (px, d * 0.1, h * 0.98),
                (px, d * 0.1, h * 1.06), min(0.016, w * 0.06), pale)

    elif kind == 'candlestick':
        radius = min(w, d) / 2
        rod(root, 'CandleFoot', (0, 0, 0), (0, 0, h * 0.06), radius, metal)
        rod(root, 'CandleStem', (0, 0, h * 0.05), (0, 0, h * 0.52), radius * 0.26, metal)
        rod(root, 'CandleKnop', (0, 0, h * 0.3), (0, 0, h * 0.38), radius * 0.44, metal)
        rod(root, 'CandleCup', (0, 0, h * 0.5), (0, 0, h * 0.58), radius * 0.5, metal)
        rod(root, 'Candle', (0, 0, h * 0.56), (0, 0, h * 0.96), radius * 0.3, pale)
        flame = source_role(material('CandleFlame', (1, .78, .42), .3, emission=14), 'body')
        sphere(root, 'CandleFlame', (0, 0, h * 0.99), (radius * 0.2, radius * 0.2, h * 0.08), flame)

    elif kind == 'sculpture':
        # An abstract piece: a plinth and two offset masses. Deliberately not a
        # figure — a box labelled "sculpture" is honest, a bad human is not.
        box(root, 'SculpturePlinth', (0, 0, h * 0.06), (w * 0.72, d * 0.72, h * 0.12),
            darker, bevel=0.004)
        sphere(root, 'SculptureMass', (0, 0, h * 0.44),
               (w * 0.66, d * 0.66, h * 0.52), base)
        box(root, 'SculptureBlade', (w * 0.1, 0, h * 0.76),
            (w * 0.22, d * 0.3, h * 0.44), base, bevel=0.01)

    elif kind == 'tray':
        box(root, 'TrayFloor', (0, 0, h / 2), (w, d, h), darker, bevel=0.003)
        rim = min(0.014, w * 0.05)
        for name, pos, size in (
                ('RimX0', (-w / 2 + rim / 2, 0, h * 0.75), (rim, d, h * 0.5)),
                ('RimX1', (w / 2 - rim / 2, 0, h * 0.75), (rim, d, h * 0.5)),
                ('RimY0', (0, -d / 2 + rim / 2, h * 0.75), (w, rim, h * 0.5)),
                ('RimY1', (0, d / 2 - rim / 2, h * 0.75), (w, rim, h * 0.5))):
            box(root, name, pos, size, base, bevel=0.002)

    elif kind == 'cushion':
        # A cushion is a rounded rectangle that leans. A box is right in
        # silhouette but the helper clamps its bevel to a fraction of the
        # smallest side, so it reads sharp; an ellipsoid is soft but becomes a
        # ball at these proportions. Two boxes of different depth, the inner one
        # narrower and proud, give the squashed silhouette with a soft shoulder.
        lean = math.radians(13.0)
        thick = min(max(d, 0.10), w * 0.44)
        outer = box(root, 'CushionBody', (0, 0, h * 0.5), (w, thick, h), base,
                    bevel=min(w, h) * 0.14)
        outer.rotation_euler.x = -lean
        inner = box(root, 'CushionSwell', (0, -thick * 0.42, h * 0.5),
                    (w * 0.90, thick * 0.62, h * 0.88), base,
                    bevel=min(w, h) * 0.13)
        inner.rotation_euler.x = -lean
        seam = box(root, 'CushionSeam', (0, 0, h * 0.5),
                   (w * 1.004, thick * 1.02, h * 0.035), darker, bevel=0.004)
        seam.rotation_euler.x = -lean

    elif kind == 'lamp':
        # A shade with a real profile: base disc, slim stem, tapered drum. The
        # unit-cube version put a table-lamp shade on a 1.6 m floor lamp.
        radius = min(w, d) / 2
        if h > 1.0:
            rod(root, 'LampBase', (0, 0, 0.0), (0, 0, 0.03), radius, black)
            rod(root, 'LampStem', (0, 0, 0.03), (0, 0, h * 0.72), radius * 0.09, metal)
            shade_h = h * 0.28
            rod(root, 'LampShade', (0, 0, h - shade_h), (0, 0, h), radius * 0.62, base)
            rod(root, 'LampShadeFoot', (0, 0, h - shade_h), (0, 0, h - shade_h + 0.006),
                radius * 0.64, darker)
        else:
            rod(root, 'LampBase', (0, 0, 0.0), (0, 0, h * 0.06), radius, darker)
            rod(root, 'LampStem', (0, 0, h * 0.06), (0, 0, h * 0.52), radius * 0.10, metal)
            rod(root, 'LampShade', (0, 0, h * 0.5), (0, 0, h), radius * 0.75, base)

    elif kind == 'vase':
        radius = min(w, d) / 2
        rod(root, 'VaseBody', (0, 0, 0.02), (0, 0, h * 0.62), radius * 0.85, base)
        rod(root, 'VaseBelly', (0, 0, h * 0.28), (0, 0, h * 0.44), radius, base)
        rod(root, 'VaseNeck', (0, 0, h * 0.62), (0, 0, h * 0.94), radius * 0.42, base)
        rod(root, 'VaseRim', (0, 0, h * 0.94), (0, 0, h), radius * 0.5, darker)

    elif kind == 'pendant_light':
        rod(root, 'PendantCable', (0, 0, h), (0, 0, h * 0.55), 0.004, black)
        radius = min(w, d) / 2
        rod(root, 'PendantDrum', (0, 0, h * 0.20), (0, 0, h * 0.55), radius, base)
        rod(root, 'PendantDiffuser', (0, 0, h * 0.18), (0, 0, h * 0.20), radius * 0.9, pale)


def asset(entry):
    """Build in a unit bounding volume, then scale to requested dimensions."""
    root = root_object(entry['id'])
    kind = entry['kind']
    # Tag isolated copies after parameter-cache lookup; source role is not a
    # property of the shader tuple shared by MATERIALS.
    base = source_role(material(kind + '_Surface', entry.get('color', [.5, .5, .5]),
                                entry.get('roughness', .55)), 'body')
    black = source_role(material('Charcoal', (.022, .028, .033), .46), 'body')
    metal = source_role(material('Steel', (.23, .26, .28), .29, .75), 'metal')
    pale = source_role(material('WarmWhite', (.83, .83, .79), .62), 'body')
    darker = source_role(material(kind + '_Shade', [v * .68 for v in entry.get('color', [.5] * 3)], .65), 'body')
    screen = source_role(material('Screen', (.035, .16, .27), .3, emission=.35), 'body')
    B = lambda name, pos, size, mat=base, bevel=.012: box(root, name, pos, size, mat, bevel)
    R = lambda name, a, b, radius, mat=metal: rod(root, name, a, b, radius, mat)
    S = lambda name, pos, size, mat=base: sphere(root, name, pos, size, mat)

    if kind in ('desk', 'table'):
        B('Tabletop', (0, 0, .953), (1, 1, .094), bevel=.025)
        for x in (-.435, .435):
            for y in (-.405, .405):
                B('Leg', (x, y, .453), (.055, .065, .906), metal)
        B('Underframe', (0, .38, .84), (.88, .045, .09), metal)
        if kind == 'desk':
            B('CableTray', (0, .36, .78), (.52, .15, .045), black)
    elif kind == 'chair':
        B('SeatCushion', (0, -.05, .455), (.82, .72, .12), bevel=.04)
        B('BackCushion', (0, .32, .77), (.8, .15, .46), bevel=.045)
        B('BackSupport', (0, .39, .64), (.12, .075, .43), black)
        R('GasLift', (0, 0, .08), (0, 0, .42), .038)
        for n in range(5):
            angle = math.tau * n / 5
            end = (.42 * math.cos(angle), .42 * math.sin(angle), .07)
            R('BaseSpoke', (0, 0, .12), end, .025, black)
            S('Caster', (end[0], end[1], .036), (.11, .085, .072), black)
        for x in (-.46, .46):
            R('ArmPost', (x, .1, .45), (x, .1, .63), .022, black)
            B('ArmPad', (x, -.02, .65), (.08, .42, .045), black)
    elif kind == 'monitor':
        B('Foot', (0, -.015, .025), (.52, .94, .05), black)
        B('Stand', (0, .14, .25), (.09, .18, .46), metal)
        B('DisplayBody', (0, 0, .665), (1, .18, .67), black)
        B('DisplayGlass', (0, -.095, .677), (.947, .015, .599), screen, 0)
        B('ScreenHeader', (0, -.104, .952), (.92, .004, .028), pale, 0)
        B('ScreenSidebar', (-.365, -.104, .655), (.19, .004, .48), darker, 0)
        for z in (.81, .75, .69, .63):
            B('ScreenContent', (.08, -.104, z), (.53, .004, .014), pale, 0)
    elif kind == 'computer':
        B('Tower', (0, 0, .5), (1, 1, 1), bevel=.025)
        B('FrontInset', (0, -.504, .49), (.85, .015, .89), black)
        for z in (.65, .69, .73, .77):
            B('Vent', (0, -.514, z), (.65, .012, .012), metal, 0)
        S('PowerButton', (.29, -.52, .91), (.045, .018, .035), pale)
    elif kind == 'sofa':
        for x in (-.4, .4):
            for y in (-.38, .38):
                B('SofaFoot', (x, y, .055), (.06, .06, .11), metal)
        B('SofaBase', (0, 0, .265), (1, .96, .31), darker, .04)
        B('SofaBack', (0, .375, .66), (.98, .25, .68), bevel=.06)
        for x in (-.44, .44):
            B('SofaArm', (x, -.04, .49), (.12, .91, .5), bevel=.04)
        for x in (-.26, 0, .26):
            B('SeatCushion', (x, -.115, .49), (.25, .65, .15), bevel=.04)
            B('BackCushion', (x, .21, .73), (.25, .17, .43), bevel=.04)
    elif kind == 'bed':
        B('BedBase', (0, 0, .22), (.98, .98, .32), darker, .025)
        B('Mattress', (0, -.035, .475), (.94, .91, .21), pale, .04)
        B('Duvet', (0, -.16, .596), (.96, .63, .055), base, .025)
        B('Headboard', (0, .455, .55), (1, .09, .9), base, .035)
        for x in (-.23, .23):
            B('Pillow', (x, .282, .62), (.39, .19, .095), pale, .045)
    elif kind in ('cabinet', 'shelf'):
        B('BackPanel', (0, .475, .5), (1, .05, 1), darker)
        for x in (-.475, .475):
            B('SidePanel', (x, 0, .5), (.05, 1, 1))
        for z in (.025, .26, .5, .74, .975):
            B('Shelf', (0, 0, z), (.95, 1, .05))
        if kind == 'cabinet':
            for x in (-.244, .244):
                B('DoorPanel', (x, -.478, .5), (.478, .044, .94))
                B('Handle', (x * .22, -.506, .53), (.018, .015, .12), metal)
        else:
            for index in range(12):
                x = -.32 + (index % 4) * .2
                z = .31 + (index // 4) * .24
                bookmat = source_role(material('Book', ((index % 3 + 1) * .18, .19 + index % 2 * .2, .24), .7), 'body')
                B('Book', (x, .03, z + .06), (.10, .46, .18), bookmat, .002)
    elif kind == 'plant':
        pot = source_role(material('Terracotta', (.37, .22, .12), .85), 'body')
        soil = source_role(material('Soil', (.045, .03, .02), 1), 'body')
        trunk = source_role(material('Bark', (.15, .09, .045), .85), 'body')
        R('Pot', (0, 0, .025), (0, 0, .27), .24, pot)
        R('Soil', (0, 0, .265), (0, 0, .275), .222, soil)
        R('Trunk', (0, 0, .25), (0, 0, .86), .025, trunk)
        # Foliage as many small clusters in tiers, not a few flat plates. The
        # plates sat in one plane and read as a stack of green discs from any
        # camera near eye height, which is exactly where the stills are taken.
        rng = random.Random(entry['id'])
        for i in range(54):
            theta = i * 2.39996
            tier = i % 12
            z = .40 + tier * .040 + rng.random() * .022
            # wider low down, tighter at the crown
            spread = (.30 - .016 * tier) * (0.55 + 0.45 * rng.random())
            x, y = math.cos(theta) * spread, math.sin(theta) * spread
            if spread > .09:
                R('Branch', (0, 0, max(.30, z - .10)), (x * .72, y * .72, z), .006, trunk)
            size = .085 + rng.random() * .045
            S('Foliage', (x, y, z), (size, size, size * 0.82),
              base if i % 3 else darker)
    elif kind in ('window', 'door'):
        # Local -Y is the room-facing surface; panels need no wall booleans.
        panel = source_role(window_glass_material(), 'glass') if kind == 'window' else base
        B('Panel', (0, 0, .5), (.91, .6, .93), panel, 0)
        for x in (-.475, .475):
            B('FrameVertical', (x, -.03, .5), (.05, 1, 1), pale)
        for z in (.025, .975):
            B('FrameHorizontal', (0, -.03, z), (1, 1, .05), pale)
        if kind == 'window':
            B('Mullion', (0, -.06, .5), (.025, .85, 1), metal)
            B('Transom', (0, -.06, .35), (1, .85, .025), metal)
        else:
            B('DoorHandle', (.34, -.42, .49), (.15, .12, .025), metal)
    elif kind == 'curtain':
        for i in range(16):
            x = -.46875 + i / 16
            y = -.13 if i % 2 else .13
            B('FabricPleat', (x, y, .49), (.075, .65, .98), base if i % 2 else darker, .02)
        R('CurtainRail', (-.49, 0, .985), (.49, 0, .985), .015)
    elif kind == 'pendant_light':
        R('Cable', (0, 0, .64), (0, 0, 1), .012, black)
        R('Shade', (0, 0, .05), (0, 0, .66), .5, base)
        glow = source_role(material('Luminaire', (1, .94, .82), .3, emission=5), 'body')
        R('Diffuser', (0, 0, 0), (0, 0, .035), .465, glow)
    elif kind == 'linear_light':
        B('Housing', (0, 0, .56), (1, 1, .88), base)
        glow = source_role(material('Luminaire', (1, .94, .82), .3, emission=5), 'body')
        B('Diffuser', (0, 0, .06), (.97, .93, .12), glow)
    elif kind == 'rug':
        B('Rug', (0, 0, .5), (1, 1, 1), bevel=.01)
        for x in (-.46, .46):
            B('WovenBorder', (x, 0, .995), (.025, .94, .006), darker, 0)
    elif kind == 'lamp':
        # A readable lamp silhouette matters more than its exact fitting: the
        # shade is an emissive diffuser so the object reads as a light source
        # in the render, not as another grey block.
        shade = source_role(material('LampShade', [min(1, v * 1.55 + .18) for v in entry.get('color', [.8, .74, .6])], .62, emission=.55), 'body')
        B('LampBase', (0, 0, .025), (.44, .44, .05), darker, .012)
        R('LampStem', (0, 0, .04), (0, 0, .58), .022, metal)
        if entry.get('is_floor_lamp'):
            pass
        B('ShadeBody', (0, 0, .74), (.86, .86, .38), shade, .02)
        S('ShadeTop', (0, 0, .92), (.5, .5, .06), shade)
        B('ShadeBottom', (0, 0, .56), (.8, .8, .02), shade, .01)
    elif kind == 'tv':
        B('TvPanel', (0, 0, .56), (.96, .07, .56), black, .006)
        B('TvScreen', (0, -.04, .56), (.93, .004, .52), screen, 0)
        B('TvNeck', (0, .02, .25), (.16, .1, .2), black)
        B('TvFoot', (0, 0, .05), (.42, .26, .045), black, .008)
        B('TvBadge', (0, -.045, .31), (.06, .003, .012), metal, 0)
    elif kind == 'book':
        # A short stacked pile reads as books at any angle, unlike a single slab.
        for index in range(3):
            shade = source_role(material('BookSpine' + str(index),
                                         [.18 + index * .10, .16 + index * .05, .22], .72), 'body')
            B('Book' + str(index), (0, 0, .18 + index * .26), (.94, .62 + index * .08, .24),
              shade, .004)
    elif kind == 'painting':
        # A framed picture, not a slab: a thin canvas set into a raised frame,
        # with a mat border so it reads as art from a few metres away.
        B('Canvas', (0, .012, .5), (.90, .016, .86), bevel=.002)
        mat = source_role(material('PictureMat', (.93, .92, .89), .8), 'body')
        B('MatBorder', (0, -.002, .5), (.94, .012, .90), mat, .002)
        for x in (-.478, .478):
            B('FrameSide', (x, -.02, .5), (.044, .05, 1), darker, .004)
        for z in (.022, .978):
            B('FrameBar', (0, -.02, z), (.956, .05, .044), darker, .004)
    elif kind == 'cushion':
        # Two stacked soft boxes with a rounded edge and a slight tilt is enough
        # for a throw pillow; a single bevel box reads as a plank.
        B('CushionBody', (0, 0, .5), (.96, .30, .88), base, .10)
        B('CushionSeam', (0, 0, .5), (.90, .32, .06), darker, .02)
        B('CushionFold', (0, 0, .74), (.84, .33, .05), darker, .02)
    elif kind == 'tray':
        B('TrayFloor', (0, 0, .18), (.94, .94, .07), darker, .012)
        for x in (-.46, .46):
            B('TrayRimX', (x, 0, .5), (.075, 1, .96), base, .012)
        for y in (-.46, .46):
            B('TrayRimY', (0, y, .5), (.95, .075, .96), base, .012)
    elif kind == 'vase':
        R('VaseBody', (0, 0, .04), (0, 0, .58), .26, base)
        R('VaseNeck', (0, 0, .58), (0, 0, .82), .12, base)
        B('VaseRim', (0, 0, .84), (.3, .3, .05), darker, .012)
    elif kind == 'decor':
        S('Decor', (0, 0, .5), (1, 1, 1))
    else:  # column, beam, box (and forward-compatible fallback categories)
        B(kind.title(), (0, 0, .5), (1, 1, 1), bevel=.006)

    root.location = entry['position']
    root.scale = entry['dimensions']
    if kind in METRIC_KINDS:
        # These were being authored inside a unit cube and then scaled by the
        # object's real dimensions, so every small detail was multiplied by a
        # very small number: a 6 mm rug border on a 12 mm rug became 0.07 mm and
        # vanished. They are rebuilt below in metres with an identity scale.
        for child in list(root.children):
            bpy.data.objects.remove(child, do_unlink=True)
        root.scale = (1.0, 1.0, 1.0)
        _metric_details(root, kind, entry, box, rod, sphere,
                        base=base, darker=darker, pale=pale, metal=metal, black=black)
    root.rotation_euler = [math.radians(v) for v in entry.get('rotation_degrees', [0, 0, 0])]
    for key in ('id', 'kind', 'observed', 'confidence', 'notes'):
        root[key] = entry.get(key, {'observed': True, 'confidence': .5, 'notes': ''}.get(key, ''))
    root['category'] = kind
    root['anchor'] = 'bottom_center'
    root['geometry_source'] = 'procedural_approximation'
    return root


def build_room(room):
    w, d, h = (room[k] for k in ('width', 'depth', 'height'))
    root = root_object('RoomShell')
    root['category'] = 'room_structure'
    wall = material('WallPaint', room.get('wall_color', [.75] * 3), .82)
    floor = material('Floor', room.get('floor_color', [.34, .3, .25]), .67)
    ceiling = material('Ceiling', room.get('ceiling_color', [.85] * 3), .85)
    box(root, 'Floor', (0, d / 2, -.075), (w + .3, d + .3, .15), floor, 0)
    box(root, 'Ceiling', (0, d / 2, h + .075), (w + .3, d + .3, .15), ceiling, 0)
    for x in (-w / 2 - .075, w / 2 + .075):
        box(root, 'SideWall', (x, d / 2, h / 2), (.15, d + .3, h), wall, 0)
    for y in (d + .075,):
        box(root, 'EndWall', (0, y, h / 2), (w, .15, h), wall, 0)
    skirting = material('Skirting', [.48] * 3, .66)
    for x in (-w / 2 + .012, w / 2 - .012):
        box(root, 'Skirting', (x, d / 2, .055), (.024, d, .11), skirting, .003)
    for y in (.012, d - .012):
        box(root, 'Skirting', (0, y, .055), (w, .024, .11), skirting, .003)
    return root


def fixture_light_plan(spec):
    """Place downward emitters at the undersides of expanded fixture instances.

    The image-derived energy is a room-wide area-light budget, divided in
    proportion to diffuser area (equal estimated luminance). This is an
    approximation, not recovered lamp wattage or an IES distribution.
    """
    light = spec.get('lighting', {})
    color = light.get('color', [1, .94, .85])
    total_energy = light.get('energy', 1500)
    emitters = []
    for entry in spec['objects']:
        kind = entry.get('kind', entry.get('category'))
        if kind not in ('linear_light', 'pendant_light'):
            continue
        width, depth, _ = entry['dimensions']
        position = list(entry['position'])
        # Fixtures use a bottom-centre anchor. A small gap keeps the light
        # outside the diffuser, avoiding self-occlusion by its closed mesh.
        position[2] -= .003
        if kind == 'linear_light':
            size, size_y = width * .97, depth * .93
            shape, area = 'RECTANGLE', size * size_y
        else:
            # The procedural pendant may be elliptical. Inscribe a disk so
            # emitted light does not originate outside its visible diffuser.
            size, size_y = .93 * min(width, depth), None
            shape, area = 'DISK', math.pi * (size / 2) ** 2
        yaw = entry.get('rotation_z_deg', entry.get('rotation_degrees', [0, 0, 0])[2])
        emitters.append({'name': 'FixtureEmitter_' + entry['id'],
                         'source_fixture_id': entry['id'],
                         'source': 'photo_inferred_fixture',
                         'confidence': entry.get('confidence', .5),
                         'position_m': position, 'rotation_z_deg': yaw,
                         'shape': shape, 'size_m': size, 'size_y_m': size_y,
                         'emitter_area_m2': area, 'color': list(color)})
    if not emitters:
        room = spec['room']
        w, d, h = (room[k] for k in ('width', 'depth', 'height'))
        for index, (x, y) in enumerate(((-w * .23, d * .3), (w * .23, d * .3),
                                       (-w * .23, d * .74), (w * .23, d * .74))):
            emitters.append({'name': 'InferredCeilingEmitter_' + str(index),
                             'source_fixture_id': None,
                             'source': 'unobserved_ceiling_grid_fallback',
                             'confidence': 0,
                             'position_m': [x, y, h - .02], 'rotation_z_deg': 0,
                             'shape': 'RECTANGLE', 'size_m': .6, 'size_y_m': .6,
                             'emitter_area_m2': .36, 'color': list(color)})
    total_area = sum(item['emitter_area_m2'] for item in emitters)
    for item in emitters:
        item['energy_w'] = total_energy * item['emitter_area_m2'] / total_area
    return {'method': 'fixture_underside_area_lights',
            'energy_distribution': 'proportional_to_diffuser_area',
            'total_area_light_energy_w': total_energy,
            'fallback_inferred': all(item['source_fixture_id'] is None for item in emitters),
            'camera_fill': False,
            'limitations': ('Fixture dimensions, emitted colour and power are inferred; no measured IES profiles. '
                            'Area-light budget excludes low-power visible diffuser mesh emission and world illumination.'),
            'emitters': emitters}


def add_light(name, position, target, energy, color, size, shape='DISK', size_y=None, yaw_degrees=0):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.color = color
    data.shape = shape
    data.size = size
    if shape == 'RECTANGLE':
        data.size_y = size_y
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()
    obj.rotation_euler.rotate_axis('Z', math.radians(yaw_degrees))
    return obj


def aabb(root):
    points = [obj.matrix_world @ Vector(corner)
              for obj in root.children_recursive if obj.type == 'MESH'
              for corner in obj.bound_box]
    if not points:
        raise ValueError('Asset has no mesh: ' + root.name)
    low = [min(p[i] for p in points) for i in range(3)]
    high = [max(p[i] for p in points) for i in range(3)]
    if not all(math.isfinite(v) for v in low + high):
        raise ValueError('Non-finite asset bounds: ' + root.name)
    return {'min': low, 'max': high}


def annotations(scene, spec, roots):
    bpy.context.view_layer.update()
    camera = scene.camera
    objects, warnings = [], []
    room = spec['room']
    wall_assets = {'window', 'door', 'curtain', 'column', 'beam', 'pendant_light', 'linear_light'}
    roomlo = [-room['width'] / 2, 0, 0]
    roomhi = [room['width'] / 2, room['depth'], room['height']]
    for entry, root in zip(spec['objects'], roots):
        bounds = aabb(root)
        objects.append({**entry, 'world_aabb_m': bounds, 'matrix_world': [list(row) for row in root.matrix_world],
                        'mesh_part_count': sum(o.type == 'MESH' for o in root.children_recursive),
                        'geometry_source': root.get('geometry_source',root.get('asset_source','basic_procedural_asset'))})
        if entry['kind'] not in wall_assets and any(bounds['min'][i] < roomlo[i] - .06 or bounds['max'][i] > roomhi[i] + .06 for i in range(3)):
            warnings.append({'type': 'approximate_room_violation', 'object_ids': [entry['id']]})
    # Only flag major volumetric overlaps between solid furniture; desk/monitor,
    # chair/desk, rug/furniture and wall-mounted objects have intentional contact.
    solid = {'sofa', 'bed', 'cabinet', 'computer', 'box', 'plant'}
    for i, obj in enumerate(objects):
        if obj['kind'] not in solid:
            continue
        a = obj['world_aabb_m']
        for other in objects[i + 1:]:
            if other['kind'] not in solid:
                continue
            b = other['world_aabb_m']
            overlap = [max(0, min(a['max'][k], b['max'][k]) - max(a['min'][k], b['min'][k])) for k in range(3)]
            volume = math.prod(overlap)
            smaller = min(math.prod(a['max'][k] - a['min'][k] for k in range(3)),
                          math.prod(b['max'][k] - b['min'][k] for k in range(3)))
            if smaller > 0 and volume / smaller > .2:
                warnings.append({'type': 'approximate_aabb_overlap', 'object_ids': [obj['id'], other['id']],
                                 'overlap_fraction_of_smaller_bbox': volume / smaller})
    projection = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(),
                                            x=scene.render.resolution_x, y=scene.render.resolution_y,
                                            scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y)
    return {
        'schema_version': 1,
        'coordinate_convention': {'units': 'metres', 'handedness': 'right', 'up_axis': '+Z',
                                  'room_x': '[-width/2, width/2]', 'room_y': '[0, depth]',
                                  'object_anchor': 'bottom_center', 'asset_front': 'local -Y',
                                  'camera_forward': 'local -Z', 'camera_up': 'local +Y',
                                  'matrix_storage': 'row-major nested arrays',
                                  'projection': 'Blender/OpenGL clip coordinates; NDC x,y in [-1,1]'},
        'geometry_status': 'approximate_single_image_reconstruction',
        'camera': {**spec['camera'], 'matrix_world': [list(row) for row in camera.matrix_world],
                   'world_to_camera': [list(row) for row in camera.matrix_world.inverted()],
                   'projection_matrix': [list(row) for row in projection],
                   'sensor_width_mm': camera.data.sensor_width,
                   'resolution_px': [scene.render.resolution_x, scene.render.resolution_y]},
        'objects': objects, 'warnings': warnings,
        'validation_scope': 'Finite world-space AABBs, approximate room bounds and selected major AABB overlaps; no exact collision or reference-fidelity verification.',
        'assumptions': spec.get('assumptions', [])
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--asset-quality', choices=['basic','detailed'], default='detailed')
    parser.add_argument('--desk-details', action='store_true', help='Add inferred supported tabletop accessories; annotated separately')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if not (64 <= args.width <= 8192 and 64 <= args.height <= 8192 and 1 <= args.samples <= 4096):
        parser.error('Resolution must be 64..8192 and samples 1..4096')
    started = time.monotonic()
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import expand_repeat_positions, validate
    validate(spec)
    original_spec = spec
    expanded = []
    for entry in spec['objects']:
        for i, j, position in expand_repeat_positions(
                entry['position'], entry['repeat'], entry['spacing'], f"scene.objects[{entry['id']}]"):
            item = dict(entry)
            item['id'] = entry['id'] + ('__%d_%d' % (i,j) if entry['repeat'] != [1,1] else '')
            item['kind'] = entry['category']
            item['position'] = position
            item['rotation_degrees'] = [0,0,entry['rotation_z_deg']]
            expanded.append(item)
    spec = {**spec,'objects':expanded}
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1
    scene['source_spec'] = str(args.spec.resolve())
    scene['reconstruction_method'] = 'Structured reference interpretation + reusable procedural assets'
    build_room(spec['room'])
    detailed_build = None
    if args.asset_quality == 'detailed':
        from detailed_assets import build as detailed_build
    roots = []
    detailed_count = 0
    for entry in spec['objects']:
        detailed = detailed_build and not (entry['kind'] == 'chair' and spec['scene_type'] not in ('office','classroom'))
        root = detailed_build(entry, seed=spec['seed']) if detailed else None
        if root is None:
            root = asset(entry)
        else:
            detailed_count += 1
        roots.append(root)
    bpy.context.view_layer.update()
    detail_entries = []
    if args.desk_details:
        from desk_details import populate
        detail_entries, detail_roots = populate(spec['objects'], roots, globals(), spec['seed'])
        spec = {**spec, 'objects': spec['objects'] + detail_entries}
        roots.extend(detail_roots)
    scene['asset_quality'] = args.asset_quality
    scene['desk_details'] = args.desk_details

    camdata = bpy.data.cameras.new('ReferenceCamera')
    cam = bpy.data.objects.new('ReferenceCamera', camdata)
    bpy.context.collection.objects.link(cam)
    cam.location = spec['camera']['position']
    delta = Vector(spec['camera']['target']) - cam.location
    if delta.length < .001:
        raise ValueError('Camera position and target must differ')
    cam.rotation_euler = delta.to_track_quat('-Z', 'Y').to_euler()
    cam.rotation_euler.rotate_axis('Z', math.radians(spec['camera'].get('roll_degrees', 0)))
    camdata.lens = spec['camera'].get('lens_mm', 24)
    camdata.sensor_width = 36
    camdata.clip_start = .03
    camdata.clip_end = 500
    scene.camera = cam
    light = spec.get('lighting', {})
    lighting_provenance = fixture_light_plan(spec)
    for emitter in lighting_provenance['emitters']:
        position = emitter['position_m']
        obj = add_light(emitter['name'], position,
                        [position[0], position[1], position[2] - 1],
                        emitter['energy_w'], emitter['color'], emitter['size_m'],
                        emitter['shape'], emitter['size_y_m'], emitter['rotation_z_deg'])
        obj['source_fixture_id'] = emitter['source_fixture_id'] or ''
        obj['lighting_source'] = emitter['source']
    scene['lighting_method'] = lighting_provenance['method']
    scene['lighting_fallback_inferred'] = lighting_provenance['fallback_inferred']
    scene.world = bpy.data.worlds.new('InteriorWorld')
    scene.world.use_nodes = True
    background = next(n for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND')
    background.inputs['Color'].default_value = (.3, .35, .4, 1)
    background.inputs['Strength'].default_value = light.get('world_strength', .25)
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    scene.render.filepath = str(out / 'rgb.png')
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.exposure = .25
    for screen_ui in bpy.data.screens:
        for area in screen_ui.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
                area.spaces.active.clip_end = 500
    data = annotations(scene, spec, roots)
    data['lighting'] = lighting_provenance
    (out / 'annotations.json').write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    # Store a complete self-contained specification inside the editable blend.
    textblock = bpy.data.texts.new('scene_spec.json')
    textblock.write(json.dumps(original_spec, indent=2, ensure_ascii=False))
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'scene.blend'))
    bpy.ops.render.render(write_still=True)
    metadata = {'status': 'completed', 'schema_version': 1, 'scene_type': spec.get('scene_type', 'interior'),
                'spec_path': str(args.spec.resolve()), 'blender_version': bpy.app.version_string,
                'outputs': ['scene.blend', 'rgb.png', 'annotations.json'],
                'asset_quality': args.asset_quality, 'detailed_asset_count': detailed_count,
                'inferred_detail_count': len(detail_entries), 'desk_details': args.desk_details,
                'lighting': lighting_provenance,
                'asset_count': len(roots), 'mesh_count': sum(o.type == 'MESH' for o in scene.objects),
                'render': {'engine': 'CYCLES', 'device': 'CPU', 'samples': args.samples,
                           'width': args.width, 'height': args.height},
                'elapsed_seconds': round(time.monotonic() - started, 3),
                'warning_count': len(data['warnings']),
                'quality_status': 'needs_review' if data['warnings'] else 'unreviewed',
                'limitations': 'Single-photo scale, occluded geometry and reusable asset shapes are estimated; not surveyed geometry.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print('SCENE_FACTORY_COMPLETED ' + str(out))


if __name__ == '__main__':
    main()
