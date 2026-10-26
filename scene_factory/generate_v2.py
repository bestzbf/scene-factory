"""Render bounded V2 descriptions as editable Blender scenes.

The JSON supplies geometry data, never executable Python. Run using Blender:
blender -b --factory-startup --python-exit-code 1 --python generate_v2.py --
  --spec scene_spec.json --output-dir output --texture-manifest textures.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_scene as g
from v2_materials import assign_material, build_materials, load_texture_manifest, planar_uv
from base_material_binding import apply_base_material_overrides


def expand_objects(base):
    expanded = []
    for entry in base['objects']:
        for i in range(entry['repeat'][0]):
            for j in range(entry['repeat'][1]):
                expanded.append({**entry,
                    'id': entry['id'] + ('__%d_%d' % (i, j) if entry['repeat'] != [1, 1] else ''),
                    'kind': entry['category'],
                    'position': [entry['position'][0] + i * entry['spacing'][0],
                                 entry['position'][1] + j * entry['spacing'][1], entry['position'][2]],
                    'rotation_degrees': [0, 0, entry['rotation_z_deg']]})
    return {**base, 'objects': expanded}


def override_room(root, overrides, materials):
    for obj in root.children:
        surface = ('floor' if obj.name.startswith('Floor') else
                   'ceiling' if obj.name.startswith('Ceiling') else
                   'left' if obj.name.startswith('SideWall') and obj.location.x < 0 else
                   'right' if obj.name.startswith('SideWall') else
                   'back' if obj.name.startswith('EndWall') else None)
        if surface and overrides[surface]:
            assign_material(obj, materials[overrides[surface]])
            planar_uv(obj)
            obj['surface_id'] = surface


def custom_mesh(name, vertices, faces, root, position, material):
    mesh = bpy.data.meshes.new(name + '_Mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # Closed meshes have an unambiguous exterior. Open surfaces keep the
    # supplied winding, which determines their intended visible side.
    if bm.edges and all(edge.is_manifold for edge in bm.edges):
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        if bm.calc_volume(signed=True) < 0:
            bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return g.mesh_object(name, mesh, root, position, material)


def build_custom(entry, materials):
    root = g.root_object(entry['id'])
    root.location = entry['position']
    root.rotation_euler = [math.radians(value) for value in entry['rotation_degrees']]
    root['category'] = 'custom_geometry'
    root['geometry_source'] = 'declarative_photo_specific_parts'
    root['observed'] = entry['observed']
    root['confidence'] = entry['confidence']
    root['support_id'] = entry['support_id']
    default = materials.get(entry['material_id']) or g.material('CustomDefault', (.5, .5, .5))
    for index, part in enumerate(entry['parts']):
        name = entry['id'] + '_part_%03d' % index
        material = materials.get(part['material_id']) or default
        kind, pos, size = part['type'], part['position'], part['size']
        if kind == 'box':
            obj = g.box(root, name, pos, size, material, bevel=min(.008, min(size) * .08))
        elif kind == 'sphere':
            dimensions = size if min(size) > 0 else [2 * part['radius']] * 3
            obj = g.sphere(root, name, pos, dimensions, material)
        elif kind == 'cylinder':
            obj = g.rod(root, name, (pos[0], pos[1], pos[2] - size[2] / 2),
                        (pos[0], pos[1], pos[2] + size[2] / 2), part['radius'], material, vertices=24)
        elif kind == 'curve':
            data = bpy.data.curves.new(name + '_Curve', 'CURVE')
            data.dimensions = '3D'
            data.resolution_u = 12
            data.bevel_depth = part['radius']
            data.bevel_resolution = 3
            data.use_fill_caps = True
            spline = data.splines.new('BEZIER')
            spline.bezier_points.add(len(part['points']) - 1)
            for point, xyz in zip(spline.bezier_points, part['points']):
                point.co = xyz
                point.handle_left_type = point.handle_right_type = 'AUTO'
            obj = bpy.data.objects.new(name, data)
            bpy.context.collection.objects.link(obj)
            obj.parent = root
            obj.location = pos
            data.materials.append(material)
        elif kind == 'mesh':
            obj = custom_mesh(name, part['vertices'], part['faces'], root, pos, material)
        elif kind == 'curtain':
            from procedural_cloth import curtain_mesh
            cloth = part['cloth'][0]
            vertices, faces, uvs = curtain_mesh(size, cloth)
            obj = custom_mesh(name, vertices, faces, root, pos, material)
            layer = obj.data.uv_layers.new(name='TextileUV')
            for loop in obj.data.loops:
                layer.data[loop.index].uv = uvs[loop.vertex_index]
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
            solidify = obj.modifiers.new('TextileThickness', 'SOLIDIFY')
            solidify.thickness = cloth['thickness']
            solidify.offset = 0
            solidify.use_even_offset = False
            obj['geometry_source'] = 'bounded_seeded_hanging_textile'
            obj['cloth_parameters_json'] = json.dumps(cloth, sort_keys=True)
        elif kind == 'image_plane':
            w, h = size[0], size[2]
            obj = custom_mesh(name, [(-w / 2, 0, -h / 2), (w / 2, 0, -h / 2),
                                    (w / 2, 0, h / 2), (-w / 2, 0, h / 2)],
                              [(0, 1, 2, 3)], root, pos, material)
            layer = obj.data.uv_layers.new(name='PhotoUV')
            for loop, uv in zip(layer.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
                loop.uv = uv
        else:
            raise ValueError('Unsupported custom part: ' + kind)
        obj.rotation_euler = [math.radians(value) for value in part['rotation_degrees']]
        obj['part_type'] = kind
        obj['custom_group_id'] = entry['id']
        obj['custom_part_index'] = index
        if obj.type == 'MESH':
            planar_uv(obj)
    return root


def evaluated_bounds(root):
    """Bounds include curve bevels and evaluated modifiers, not only mesh boxes."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in root.children_recursive:
        if obj.type not in {'MESH', 'CURVE', 'SURFACE', 'FONT'}:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            if mesh:
                points.extend(evaluated.matrix_world @ vertex.co for vertex in mesh.vertices)
        finally:
            evaluated.to_mesh_clear()
    if not points:
        raise ValueError('Asset has no evaluated geometry: ' + root.name)
    low = [min(point[i] for point in points) for i in range(3)]
    high = [max(point[i] for point in points) for i in range(3)]
    if not all(math.isfinite(value) for value in low + high):
        raise ValueError('Non-finite bounds: ' + root.name)
    return {'min': low, 'max': high}


def support_surface(root, bounds):
    """Sample evaluated upward surfaces under the footprint, including seat/shelf levels.

    A furniture group's highest AABB face is often a backrest or headboard and
    must not be mistaken for its only support surface. This remains a bounded
    approximation: nine downward rays per mesh, not full contact simulation.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    lo, hi = bounds['min'], bounds['max']
    samples = [(lo[0] + (hi[0] - lo[0]) * u, lo[1] + (hi[1] - lo[1]) * v)
               for u in (.15, .5, .85) for v in (.15, .5, .85)]
    candidates = []
    for obj in root.children_recursive:
        if obj.type not in {'MESH', 'CURVE', 'SURFACE', 'FONT'}:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            if not mesh or not mesh.polygons:
                continue
            vertices = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
            tree = BVHTree.FromPolygons(vertices, [list(face.vertices) for face in mesh.polygons])
            top = max(vertex.z for vertex in vertices) + 1
            for x, y in samples:
                location, normal, _, _ = tree.ray_cast(Vector((x, y, top)), Vector((0, 0, -1)), 200)
                if location is not None and normal.z > .35:
                    candidates.append(location.z)
        finally:
            evaluated.to_mesh_clear()
    return min(candidates, key=lambda z: abs(lo[2] - z)) if candidates else None


def make_annotations(scene, spec, base, roots, custom_roots):
    data = g.annotations(scene, base, roots)
    data['schema_version'] = 2
    data['scene_type'] = spec['scene_type']
    data['coordinate_convention']['custom_group_anchor'] = 'explicit origin; part centres are local to group'
    data['reference_image'] = spec.get('reference_image', spec.get('image'))
    from part_projection import custom_part_projections
    data['custom_part_projection'] = custom_part_projections(scene, spec['custom_objects'], custom_roots,
        view_id=scene.camera.get('reference_view_id', data['reference_image'].get('source_id', '')))
    if scene.get('base_material_binding_report'):
        data['material_binding'] = json.loads(scene['base_material_binding_report'])
    if scene.get('exterior_context_report'):
        data['exterior_context'] = json.loads(scene['exterior_context_report'])
    for obj, root in zip(data['objects'], roots):
        obj['world_aabb_m'] = evaluated_bounds(root)
        obj['observed'] = None
        obj['evidence_status'] = 'base_v1_has_no_observed_flag; confidence_is_estimated'
        obj['support_check'] = {'status': 'not_specified'}
    by_id = {obj['id']: obj for obj in data['objects']}
    root_by_id = {entry['id']: root for entry, root in zip(base['objects'], roots)}
    for entry, root in zip(spec['custom_objects'], custom_roots):
        item = {**entry, 'kind': 'custom_geometry', 'world_aabb_m': evaluated_bounds(root),
                'matrix_world': [list(row) for row in root.matrix_world],
                'mesh_part_count': sum(obj.type == 'MESH' for obj in root.children_recursive),
                'curve_part_count': sum(obj.type == 'CURVE' for obj in root.children_recursive),
                'geometry_source': root['geometry_source'],
                'evidence_status': 'visible_interpretation' if entry['observed'] else 'inferred_completion'}
        data['objects'].append(item)
        by_id[item['id']] = item
        root_by_id[item['id']] = root
        roomlo = [-base['room']['width'] / 2, 0, 0]
        roomhi = [base['room']['width'] / 2, base['room']['depth'], base['room']['height']]
        bounds = item['world_aabb_m']
        if any(bounds['min'][i] < roomlo[i] - .06 or bounds['max'][i] > roomhi[i] + .06 for i in range(3)):
            data['warnings'].append({'type': 'approximate_room_violation', 'object_ids': [entry['id']]})
    for item in data['objects']:
        if item['kind'] != 'custom_geometry':
            continue
        support_id = item['support_id']
        bounds = item['world_aabb_m']
        if not support_id:
            item['support_check'] = {'status': 'not_specified'}
            continue
        if support_id == 'floor':
            w, d = base['room']['width'], base['room']['depth']
            support_bounds = {'min': [-w / 2, 0, -.15], 'max': [w / 2, d, 0]}
        elif support_id in by_id:
            support_bounds = by_id[support_id]['world_aabb_m']
        else:
            raise ValueError('Unknown support_id: ' + support_id)
        surface_z = 0 if support_id == 'floor' else support_surface(root_by_id[support_id], bounds)
        top_z = support_bounds['max'][2] if surface_z is None else surface_z
        gap = bounds['min'][2] - top_z
        overlap = [max(0, min(bounds['max'][i], support_bounds['max'][i]) -
                       max(bounds['min'][i], support_bounds['min'][i])) for i in range(2)]
        footprint = math.prod(max(bounds['max'][i] - bounds['min'][i], 1e-9) for i in range(2))
        fraction = min(1, math.prod(overlap) / footprint)
        okay = surface_z is not None and abs(gap) <= .025 and fraction > 0
        item['support_check'] = {'status': 'approximate_contact' if okay else 'needs_review',
                                 'support_id': support_id, 'support_top_z_m': top_z,
                                 'support_aabb_top_z_m': support_bounds['max'][2],
                                 'sampled_surface_found': surface_z is not None,
                                 'bottom_gap_m': gap, 'footprint_overlap_fraction': fraction,
                                 'method': '9 footprint samples against evaluated upward support surfaces; AABB footprint; no full collision simulation'}
        if not okay:
            data['warnings'].append({'type': 'support_contact_mismatch',
                'object_ids': [item['id'], support_id], 'bottom_gap_m': gap,
                'footprint_overlap_fraction': fraction})
    data['validation_scope'] += ' Custom evaluated curve/mesh bounds and declared support sampled upward surfaces/footprints; no full collision simulation.'
    return data


def add_lights(scene, spec, base):
    plan = None
    if not spec['render'].get('replace_base_lights', False):
        plan = g.fixture_light_plan(base)
        for emitter in plan['emitters']:
            pos = emitter['position_m']
            obj = g.add_light(emitter['name'], pos, [pos[0], pos[1], pos[2] - 1],
                emitter['energy_w'], emitter['color'], emitter['size_m'], emitter['shape'],
                emitter['size_y_m'], emitter['rotation_z_deg'])
            obj['lighting_source'] = emitter['source']
        if not spec['lights']:
            return plan
    for entry in spec['lights']:
        data = bpy.data.lights.new(entry['id'], entry['type'])
        data.energy, data.color = entry['energy'], entry['color']
        if entry['type'] == 'AREA':
            data.shape, data.size = entry['shape'], entry['size']
            if data.shape == 'RECTANGLE':
                data.size_y = entry['size_y']
            data.spread = math.radians(entry['spread_degrees'])
        else:
            data.shadow_soft_size = entry['size'] / 2
        obj = bpy.data.objects.new(entry['id'], data)
        bpy.context.collection.objects.link(obj)
        obj.location = entry['position']
        delta = Vector(entry['target']) - obj.location
        if delta.length > 1e-8:
            obj.rotation_euler = delta.to_track_quat('-Z', 'Y').to_euler()
        obj['lighting_source'] = 'photo_interpretation_unmeasured'
    return {'method': 'base_and_explicit_v2_lights' if plan else 'explicit_v2_lights',
            'fallback_inferred': plan['fallback_inferred'] if plan else False,
            'camera_fill': False, 'emitters': spec['lights'], 'base_lighting': plan,
            'limitations': 'Positions, powers and colours are estimates, not calibrated photometry.'}


def configure_device(scene, requested):
    scene.cycles.device = 'CPU'
    if requested == 'CPU':
        return 'CPU'
    preferences = bpy.context.preferences.addons['cycles'].preferences
    for backend in ('METAL', 'OPTIX', 'CUDA', 'HIP', 'ONEAPI'):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            devices = [device for device in preferences.devices if device.type == backend]
            if devices:
                for device in preferences.devices:
                    device.use = device.type == backend
                scene.cycles.device = 'GPU'
                return backend
        except (TypeError, ValueError, RuntimeError):
            continue
    return 'CPU'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--source-image', type=Path)
    parser.add_argument('--texture-manifest', type=Path)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--device', choices=['CPU', 'AUTO'], default='AUTO')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    started = time.monotonic()
    from spec_v2 import validate
    spec = validate(json.loads(args.spec.read_text(encoding='utf-8')))
    if spec['schema_version'] != 2:
        raise ValueError('generate_v2.py requires schema_version=2; use upgrade_v1 first')
    base = expand_objects(spec['base_scene'])
    ref = spec.get('reference_image', spec.get('image'))
    width = args.width or (round(args.height * ref['width'] / ref['height']) if args.height else min(1280, ref['width']))
    height = args.height or round(width * ref['height'] / ref['width'])
    if not (64 <= width <= 8192 and 64 <= height <= 8192 and 1 <= args.samples <= 4096):
        parser.error('Resolution must be 64..8192 and samples 1..4096')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metadata.json').unlink(missing_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1
    scene['source_spec'] = str(args.spec.resolve())
    scene['reconstruction_method'] = 'V2 structured interpretation + reusable assets + declarative per-photo geometry'
    images = load_texture_manifest(args.texture_manifest)
    missing = {item['id'] for item in spec['textures']} - set(images)
    if missing:
        raise ValueError('Missing rectified texture crops: ' + ', '.join(sorted(missing)))
    materials = build_materials(spec['materials'], images)
    room = g.build_room(base['room'])
    override_room(room, spec['surface_overrides'], materials)
    from detailed_assets import build as detailed_build
    from interior_assets import build as interior_build
    from schema import CATEGORIES
    roots, counts = [], {'interior': 0, 'office_detailed': 0, 'basic': 0}
    for entry in base['objects']:
        if entry['kind'] not in CATEGORIES:
            raise ValueError('Unsupported asset category: ' + entry['kind'])
        root = interior_build(entry, scene_type=spec['scene_type'], seed=base['seed'])
        origin = 'interior'
        if root is None:
            root = detailed_build(entry, seed=base['seed'])
            origin = 'office_detailed'
        if root is None:
            root = g.asset(entry)
            origin = 'basic'
        counts[origin] += 1
        roots.append(root)
    material_binding = apply_base_material_overrides(base['objects'], roots,
        spec.get('base_material_overrides', []), materials)
    (out / 'material_binding.json').write_text(json.dumps(material_binding, indent=2), encoding='utf-8')
    if material_binding['status'] != 'passed':
        raise ValueError('Base material overrides failed; see material_binding.json')
    scene['base_material_binding_report'] = json.dumps(material_binding, sort_keys=True)
    from exterior_context import build_exterior_contexts
    exterior_context = build_exterior_contexts(base, spec.get('exterior_contexts', []), images,
                                             [base['camera']['position']])
    (out / 'exterior_context.json').write_text(json.dumps(exterior_context, indent=2), encoding='utf-8')
    if exterior_context['status'] != 'passed':
        raise ValueError('Exterior context attachment failed; see exterior_context.json')
    scene['exterior_context_report'] = json.dumps(exterior_context, sort_keys=True)
    custom_roots = [build_custom(entry, materials) for entry in spec['custom_objects']]
    camdata = bpy.data.cameras.new('ReferenceCamera')
    camera = bpy.data.objects.new('ReferenceCamera', camdata)
    bpy.context.collection.objects.link(camera)
    camera.location = base['camera']['position']
    camera.rotation_euler = (Vector(base['camera']['target']) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.rotation_euler.rotate_axis('Z', math.radians(spec['camera']['roll_degrees']))
    camdata.lens, camdata.sensor_width = base['camera']['lens_mm'], 36
    camdata.clip_start, camdata.clip_end = .02, 500
    scene.camera = camera
    if args.source_image:
        reference = bpy.data.images.load(str(args.source_image.resolve()), check_existing=True)
        reference.pack()
        reference['role'] = 'original_reference_not_rendered'
    lighting = add_lights(scene, spec, base)
    scene.world = bpy.data.worlds.new('V2InteriorWorld')
    scene.world.use_nodes = True
    background = next(node for node in scene.world.node_tree.nodes if node.type == 'BACKGROUND')
    background.inputs['Color'].default_value = (*spec['render']['world_color'], 1)
    background.inputs['Strength'].default_value = base['lighting']['world_strength']
    scene.render.engine = 'CYCLES'
    device = configure_device(scene, args.device)
    scene.cycles.samples = args.samples
    scene.cycles.preview_samples = min(args.samples, 64)
    scene.cycles.use_denoising = True
    scene.cycles.use_preview_denoising = True
    scene.cycles.max_bounces = 10
    scene.cycles.diffuse_bounces = 5
    scene.cycles.glossy_bounces = 5
    scene.cycles.transmission_bounces = 8
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    scene.render.filepath = str(out / 'rgb.png')
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.exposure = spec['render']['exposure']
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
                area.spaces.active.clip_end = 500
    bpy.context.view_layer.update()
    data = make_annotations(scene, spec, base, roots, custom_roots)
    data['camera']['roll_degrees'] = spec['camera']['roll_degrees']
    data['lighting'] = lighting
    (out / 'annotations.json').write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    embedded = bpy.data.texts.new('scene_spec_v2.json')
    embedded.write(json.dumps(spec, indent=2, ensure_ascii=False))
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'scene.blend'))
    bpy.ops.render.render(write_still=True)
    metadata = {'status': 'completed', 'schema_version': 2, 'scene_type': spec['scene_type'],
        'spec_path': str(args.spec.resolve()), 'blender_version': bpy.app.version_string,
        'outputs': ['scene.blend', 'rgb.png', 'annotations.json', 'material_binding.json', 'exterior_context.json'],
        'asset_quality': 'v2_mixed_detailed', 'asset_count': len(roots) + len(custom_roots),
        'asset_builder_counts': counts, 'custom_group_count': len(custom_roots),
        'detailed_asset_count': counts['interior'] + counts['office_detailed'],
        'texture_count': len(images), 'lighting': lighting, 'material_binding': material_binding,
        'exterior_context': exterior_context,
        'mesh_count': sum(obj.type == 'MESH' for obj in scene.objects),
        'curve_count': sum(obj.type == 'CURVE' for obj in scene.objects),
        'render': {'engine': 'CYCLES', 'device': device, 'samples': args.samples, 'width': width, 'height': height},
        'elapsed_seconds': round(time.monotonic() - started, 3),
        'warning_count': len(data['warnings']),
        'quality_status': 'needs_review' if data['warnings'] else 'unreviewed',
        'limitations': 'Single-photo geometry and photometry are estimated. Custom geometry is bounded declarative data; no arbitrary generated code is executed.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print('SCENE_FACTORY_COMPLETED ' + str(out))


if __name__ == '__main__':
    main()
