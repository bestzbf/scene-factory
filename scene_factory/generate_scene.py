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
        rng = random.Random(entry['id'])
        for i in range(36):
            theta = i * 2.39996
            z = .48 + (i % 9) * .052
            radius = .10 + rng.random() * .24
            x, y = math.cos(theta) * radius, math.sin(theta) * radius
            R('Branch', (0, 0, z - .13), (x, y, z), .008, trunk)
            leaf = S('Foliage', (x, y, z), (.24, .22, .16), base if i % 3 else darker)
            leaf.rotation_euler.z = theta
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
    elif kind == 'decor':
        S('Decor', (0, 0, .5), (1, 1, 1))
    else:  # column, beam, box (and forward-compatible fallback categories)
        B(kind.title(), (0, 0, .5), (1, 1, 1), bevel=.006)

    root.location = entry['position']
    root.scale = entry['dimensions']
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
