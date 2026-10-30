"""Render a QA contact sheet of ingested assets: one Cycles-lit three-quarter view each.

Usage: Blender -b --factory-startup --python render_asset_sheet.py -- <catalog.json> <out.png> [asset_id ...]
Each tile is labelled with category and measured W x D x H. Support surfaces are
outlined in green so shelf/counter tops that can hold products are visible.
"""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
catalog_path, out_path, wanted = Path(argv[0]).resolve(), Path(argv[1]).resolve(), set(argv[2:])
catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
root = Path(catalog['root'])
assets = [a for a in catalog['assets'] if not wanted or a['asset_id'] in wanted]
assets.sort(key=lambda a: (a['category'], a['asset_id']))
tile, cols = 360, 6
tiles_dir = out_path.parent / (out_path.stem + '_tiles')
tiles_dir.mkdir(parents=True, exist_ok=True)

for asset in assets:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    bpy.ops.import_scene.gltf(filepath=str(root / asset['model_file']))
    meshes = [o for o in scene.objects if o.type == 'MESH']
    points = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    low = Vector([min(p[i] for p in points) for i in range(3)]); high = Vector([max(p[i] for p in points) for i in range(3)])
    centre, radius = (low + high) / 2, max((high - low).length / 2, 0.02)
    bpy.ops.mesh.primitive_plane_add(size=radius * 12, location=(centre.x, centre.y, low.z))
    floor = bpy.context.active_object
    floor_mat = bpy.data.materials.new('floor'); floor_mat.use_nodes = True
    floor_mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.62, 0.62, 0.6, 1)
    floor.data.materials.append(floor_mat)
    for surface in asset.get('support_surfaces', [])[:24]:
        poly = surface['polygon_xy']
        curve = bpy.data.curves.new('s', 'CURVE'); curve.dimensions = '3D'; curve.bevel_depth = radius * 0.006
        spline = curve.splines.new('POLY'); spline.points.add(len(poly)); spline.use_cyclic_u = True
        for i, (x, y) in enumerate(poly + poly[:1]):
            spline.points[i].co = (x, y, surface['z'] + 0.003, 1)
        mark = bpy.data.objects.new('s', curve); scene.collection.objects.link(mark)
        green = bpy.data.materials.new('g'); green.use_nodes = True
        green.node_tree.nodes['Principled BSDF'].inputs['Emission Color'].default_value = (0, 1, 0.2, 1)
        green.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 3
        curve.materials.append(green)
    camera = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); scene.collection.objects.link(camera)
    camera.data.lens = 40
    camera.location = centre + Vector((-0.75, -1.6, 0.85)).normalized() * radius * 3.0
    camera.rotation_euler = (centre - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.clip_end = radius * 50
    scene.camera = camera
    sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); scene.collection.objects.link(sun)
    sun.data.energy = 3.5; sun.rotation_euler = (math.radians(50), 0, math.radians(-30))
    world = bpy.data.worlds.new('w'); scene.world = world; world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.85, 0.87, 0.9, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.9
    scene.render.engine = 'CYCLES'; scene.cycles.samples = 24; scene.cycles.use_denoising = True
    scene.cycles.device = 'CPU'
    scene.render.resolution_x = scene.render.resolution_y = tile
    scene.render.filepath = str(tiles_dir / f"{asset['asset_id']}.png")
    bpy.ops.render.render(write_still=True)
    print('TILE', asset['asset_id'], flush=True)

# Compose with Blender's image API (no PIL inside Blender).
import numpy as np
rows = math.ceil(len(assets) / cols)
sheet = np.ones((rows * (tile + 30), cols * tile, 4), np.float32)
for i, asset in enumerate(assets):
    image = bpy.data.images.load(str(tiles_dir / f"{asset['asset_id']}.png"))
    pixels = np.array(image.pixels[:], np.float32).reshape(tile, tile, 4)
    r, c = divmod(i, cols)
    y0 = sheet.shape[0] - (r + 1) * (tile + 30)
    sheet[y0:y0 + tile, c * tile:(c + 1) * tile] = pixels
out = bpy.data.images.new('sheet', sheet.shape[1], sheet.shape[0])
out.pixels.foreach_set(sheet.ravel())
out.filepath_raw = str(out_path); out.file_format = 'PNG'; out.save()
labels = [{'asset_id': a['asset_id'], 'category': a['category'],
           'size_m': [round(v * a['meters_per_unit'], 2) for v in a['dimensions_import_units']],
           'support_surfaces': len(a.get('support_surfaces', []))} for a in assets]
(out_path.with_suffix('.json')).write_text(json.dumps({'cols': cols, 'tile': tile, 'label_px': 30, 'assets': labels},
                                                      ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('SHEET', out_path, len(assets))
