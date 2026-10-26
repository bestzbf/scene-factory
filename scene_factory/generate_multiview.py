"""Build one editable Blender scene from a grouped photo interpretation.

blender -b --factory-startup --python-exit-code 1 --python generate_multiview.py
  -- --spec bundle.json --output-dir output --max-edge 1280 --samples 64

All geometry is constructed once. Every reference gets a named camera and
matched render, and all originals/crops/data are packed in the same .blend.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_v2 as v2
from multiview_spec import file_sha256, load, render_dimensions, resolve_sources, texture_view_map, validate


def surface_for_object(obj, room):
    if obj.name.startswith('Floor'):
        return 'floor'
    if obj.name.startswith('Ceiling'):
        return 'ceiling'
    if obj.name.startswith('SideWall'):
        return 'left' if obj.location.x < 0 else 'right'
    if obj.name.startswith('EndWall'):
        return 'back'
    if obj.name.startswith('FrontWall'):
        return 'front'
    if obj.name.startswith('Skirting'):
        if abs(obj.location.x) > room['width'] / 2 - .05:
            return 'left' if obj.location.x < 0 else 'right'
        return 'front' if obj.location.y < room['depth'] / 2 else 'back'
    return None


def build_room(base, architecture, overrides, materials):
    """Omitted walls and their skirting are actually absent, not hidden planes."""
    room = v2.g.build_room(base['room'])
    v2.override_room(room, overrides, materials)
    for obj in list(room.children):
        surface = surface_for_object(obj, base['room'])
        if surface in architecture['omit_surfaces'] or (surface == 'front' and not architecture['front_wall']):
            bpy.data.objects.remove(obj, do_unlink=True)
        elif surface:
            obj['surface_id'] = surface
    if architecture['front_wall']:
        dims = base['room']
        wall = v2.g.material('FrontWallSurface', dims['wall_color'], .82)
        obj = v2.g.box(room, 'FrontWall', (0, -.075, dims['height']/2),
                       (dims['width'], .15, dims['height']), wall, 0)
        obj['surface_id'] = 'front'
    return room


def add_camera(view):
    name = 'ReferenceCamera_' + view['id']
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    values = view['camera']
    obj.location = values['position']
    obj.rotation_euler = (Vector(values['target']) - obj.location).to_track_quat('-Z', 'Y').to_euler()
    obj.rotation_euler.rotate_axis('Z', math.radians(values['roll_degrees']))
    data.lens, data.sensor_width = values['lens_mm'], 36
    data.clip_start, data.clip_end = .02, 500
    obj['reference_view_id'] = view['id']
    obj['source_image'] = view['source_image']
    obj['estimated_camera_not_calibrated'] = True
    return obj


def build_scene(bundle, texture_manifest=None, appearance_policy='photo_reconstruction',
                asset_catalog=None, asset_bindings=None, asset_preview=False):
    """Construct shared geometry exactly once; returns handles for all views."""
    validate(bundle)
    spec = bundle['shared_scene']
    base = v2.expand_objects(spec['base_scene'])
    importer, bindings = None, {}
    if (asset_catalog is None) != (asset_bindings is None):
        raise ValueError('asset_catalog and asset_bindings must be supplied together')
    if asset_catalog is not None:
        from external_assets import ExternalAssetImporter
        if not isinstance(asset_bindings, dict) or set(asset_bindings) != {'schema_version', 'instances'} or asset_bindings['schema_version'] != 1:
            raise ValueError('Asset bindings require schema_version=1 and instances')
        instances = asset_bindings['instances']
        if not isinstance(instances, list):
            raise ValueError('Asset instances must be a list')
        bindings = {item['object_id']: item for item in instances}
        if len(bindings) != len(instances) or set(bindings) - {o['id'] for o in base['objects']}:
            raise ValueError('Duplicate or unknown expanded asset object IDs')
        importer = ExternalAssetImporter(asset_catalog, preview=asset_preview)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Shared caches contain Blender datablock handles and should not survive
    # an independently requested scene build in the same Python process.
    v2.g.MATERIALS.clear()
    v2.g.MESHES.clear()
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1
    scene['scene_id'] = bundle['scene_id']
    scene['reconstruction_method'] = 'one shared scene from supplied photo interpretations; multiple estimated cameras'
    images = v2.load_texture_manifest(texture_manifest)
    missing = {item['id'] for item in spec['textures']} - set(images)
    if missing:
        raise ValueError('Missing preprocessed crops: ' + ', '.join(sorted(missing)))
    materials = v2.build_materials(spec['materials'], images)
    room = build_room(base, bundle['architecture'], spec['surface_overrides'], materials)
    from detailed_assets import build as detailed_build
    from interior_assets import build as interior_build
    roots, counts = [], {'interior': 0, 'office_detailed': 0, 'basic': 0}
    if importer:
        counts['external'] = 0
    for entry in base['objects']:
        if entry['id'] in bindings:
            root = importer.instantiate(entry, bindings[entry['id']])
            origin = 'external'
        else:
            root = interior_build(entry, scene_type=spec['scene_type'], seed=base['seed'])
            origin = 'interior'
        if root is None:
            root = detailed_build(entry, seed=base['seed'], appearance_policy=appearance_policy)
            origin = 'office_detailed'
        if root is None:
            root, origin = v2.g.asset(entry), 'basic'
        roots.append(root)
        counts[origin] += 1
    material_binding = v2.apply_base_material_overrides(base['objects'], roots,
        spec.get('base_material_overrides', []), materials)
    scene['base_material_binding_report'] = json.dumps(material_binding, sort_keys=True)
    from exterior_context import build_exterior_contexts
    exterior_context = build_exterior_contexts(base, spec.get('exterior_contexts', []), images,
        [view['camera']['position'] for view in bundle['views']])
    scene['exterior_context_report'] = json.dumps(exterior_context, sort_keys=True)
    custom_roots = [v2.build_custom(entry, materials) for entry in spec['custom_objects']]
    cameras = {view['id']: add_camera(view) for view in bundle['views']}
    scene.camera = cameras[bundle['views'][0]['id']]
    lighting = v2.add_lights(scene, spec, base)
    # An exterior area emitter models incoming daylight, not a physical white
    # rectangle outside the window. Cycles hides area lights from camera rays
    # by default but still shows them through transmissive glazing. Keep their
    # illumination while hiding these exterior proxies from transmission rays.
    exterior_proxies = []
    width, depth, height = (base['room'][key] for key in ('width','depth','height'))
    for obj in scene.objects:
        if obj.type != 'LIGHT' or obj.data.type != 'AREA':
            continue
        x,y,z = obj.location
        if x < -width/2 or x > width/2 or y < 0 or y > depth or z > height:
            obj.visible_transmission = False
            obj['exterior_illumination_proxy'] = True
            exterior_proxies.append(obj.name)
    lighting['exterior_proxy_transmission_hidden'] = exterior_proxies
    lighting['glazing_transport_approximations'] = [
        {'object': obj.name, 'material': slot.material.name,
         'method': slot.material['glazing_transport'],
         'shadow_transmission_per_face': slot.material.get('shadow_transmission_per_face'),
         'limitations': slot.material.get('transport_limitations')}
        for obj in scene.objects for slot in obj.material_slots
        if slot.material and slot.material.get('glazing_transport')]
    if appearance_policy == 'photo_reconstruction':
        from appearance_lighting import add_fixture_faces
        lighting['visible_fixture_faces'] = add_fixture_faces(scene)
    scene['appearance_policy'] = appearance_policy
    scene.world = bpy.data.worlds.new('SharedInteriorWorld')
    scene.world.use_nodes = True
    background = next(node for node in scene.world.node_tree.nodes if node.type == 'BACKGROUND')
    background.inputs['Color'].default_value = (*spec['render']['world_color'], 1)
    background.inputs['Strength'].default_value = base['lighting']['world_strength']
    bpy.context.view_layer.update()
    return {'scene': scene, 'base': base, 'room': room, 'roots': roots,
            'custom_roots': custom_roots, 'cameras': cameras, 'counts': counts,
            'lighting': lighting, 'texture_count': len(images),
            'external_assets': importer.records if importer else [],
            'material_binding': material_binding, 'exterior_context': exterior_context}


def configure_render(scene, spec, samples, device):
    scene.render.engine = 'CYCLES'
    backend = v2.configure_device(scene, device)
    scene.cycles.samples = samples
    scene.cycles.preview_samples = max(32, min(samples, 128))
    scene.cycles.use_denoising = True
    scene.cycles.use_preview_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = .025
    scene.cycles.max_bounces = 10
    scene.cycles.diffuse_bounces = 5
    scene.cycles.glossy_bounces = 5
    scene.cycles.transmission_bounces = 8
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = ('AgX - Medium High Contrast' if
        scene.get('appearance_policy') == 'photo_reconstruction' else 'None')
    scene.view_settings.exposure = spec['render']['exposure']
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.region_3d.view_perspective = 'CAMERA'
                space.clip_end = 500
                space.shading.type = 'MATERIAL'
                space.shading.use_scene_lights = True
                space.shading.use_scene_world = True
                space.shading.use_scene_lights_render = True
                space.shading.use_scene_world_render = True
    return backend


def set_view(scene, camera, view, max_edge):
    scene.camera = camera
    width, height = render_dimensions(view, max_edge)
    scene.render.resolution_x, scene.render.resolution_y = width, height
    bpy.context.view_layer.update()
    return width, height


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def _daylight_material_facts(obj):
    """Inspect actual object slots; linked/complex transmission stays unknown."""
    facts = []
    for slot in obj.material_slots:
        material = slot.material
        if material is None or not material.use_nodes:
            continue
        nodes = material.node_tree.nodes
        shaders = [node for node in nodes if node.type == 'BSDF_PRINCIPLED']
        if len(shaders) != 1:
            continue
        shader = shaders[0]
        direct_surface = any(link.to_node.type == 'OUTPUT_MATERIAL' and link.to_socket.name == 'Surface'
                             for link in shader.outputs['BSDF'].links)
        if not direct_surface and material.get('glazing_transport') != 'dielectric_camera_rays_transparent_shadow_approximation':
            continue
        transmission = shader.inputs.get('Transmission Weight')
        if transmission is None or transmission.is_linked:
            continue
        facts.append({'material': material.name, 'transmission': float(transmission.default_value),
                      'photo_texture': any(link.from_node.type == 'TEX_IMAGE' and link.from_node.image is not None
                                           for link in shader.inputs['Base Color'].links),
                      'shadow_transport': material.get('glazing_transport', 'material_defined')})
    return facts


def _daylight_mesh_bounds(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    corners = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    return evaluated, corners, {'min': [min(p[i] for p in corners) for i in range(3)],
                                'max': [max(p[i] for p in corners) for i in range(3)]}


def _daylight_hits(obj, start, end, depsgraph):
    """Object-local ray tests avoid first-hit glass hiding a wall behind it."""
    evaluated = obj.evaluated_get(depsgraph)
    inverse = evaluated.matrix_world.inverted_safe()
    origin, target = inverse @ start, inverse @ end
    delta = target - origin
    if delta.length < 1e-8:
        return False
    return bool(evaluated.ray_cast(origin, delta.normalized(), distance=delta.length)[0])


def daylight_preflight(bundle, handles):
    """High-confidence aperture blockers, with no scene-specific corrections.

    An actual clear pane in a declared base window defines each sampled
    aperture. Only a retained opaque room wall or a union of nearby parallel
    textured partial-transmission custom planes can block this gate. Ordinary
    curtains, frame bars, untextured multiple glazing and remote objects are
    not classified as these faults. This is geometric coverage, not a light
    transport simulation or an estimate of lost lux.
    """
    scene = handles['scene']
    depsgraph = bpy.context.evaluated_depsgraph_get()
    errors, warnings, records = [], [], []
    custom_objects = [(entry['id'], obj)
                      for entry, root in zip(bundle['shared_scene']['custom_objects'], handles['custom_roots'])
                      for obj in root.children_recursive if obj.type == 'MESH' and not obj.hide_render]
    shell_objects = [obj for obj in handles['room'].children if obj.type == 'MESH' and
                     obj.get('surface_id') in {'left', 'right', 'front', 'back'} and not obj.hide_render]
    room = handles['base']['room']
    def exterior(position):
        return (abs(position.x) > room['width'] / 2 or position.y < 0 or
                position.y > room['depth'] or position.z < 0 or position.z > room['height'])
    for entry, root in zip(handles['base']['objects'], handles['roots']):
        if entry['kind'] != 'window':
            continue
        axes = [root.matrix_world.to_3x3().col[i].normalized() for i in range(3)]
        xaxis, normal, zaxis = axes
        panes = []
        for obj in root.children_recursive:
            if obj.type != 'MESH' or obj.hide_render:
                continue
            facts = _daylight_material_facts(obj)
            if not facts or not all(f['transmission'] >= .9 and not f['photo_texture'] for f in facts):
                continue
            _, corners, bounds = _daylight_mesh_bounds(obj, depsgraph)
            intervals = [(min(p.dot(axis) for p in corners), max(p.dot(axis) for p in corners)) for axis in axes]
            sizes = [b - a for a, b in intervals]
            if sizes[1] <= .15 and min(sizes[0], sizes[2]) >= .2:
                panes.append((sizes[0] * sizes[2], obj, intervals, bounds, facts))
        if not panes:
            # Complex/imported windows without an inspectable pane are outside
            # this narrow detector, not evidence that their transport is wrong.
            continue
        _, pane, intervals, bounds, facts = max(panes, key=lambda item: item[0])
        centre = sum((axis * ((a + b) / 2) for axis, (a, b) in zip(axes, intervals)), Vector())
        width, height = intervals[0][1] - intervals[0][0], intervals[2][1] - intervals[2][0]
        samples = [centre + xaxis * ((i + .5) / 7 - .5) * width * .9 +
                   zaxis * ((j + .5) / 7 - .5) * height * .9 for i in range(7) for j in range(7)]
        # Only associate emitters on the window's outward side whose actual
        # forward hemisphere includes its centre. Other exterior lights are
        # recorded nowhere and cannot cause a false rejection for this window.
        lights = []
        for light in scene.objects:
            if light.type != 'LIGHT' or light.data.type != 'AREA' or light.data.energy <= 0:
                continue
            location = light.matrix_world.translation
            to_pane = centre - location
            if (not exterior(location) or (location - centre).dot(normal) <= .02 or
                    to_pane.length < 1e-8):
                continue
            forward = light.matrix_world.to_3x3() @ Vector((0, 0, -1))
            if forward.normalized().dot(to_pane.normalized()) <= .1:
                continue
            lights.append(light)
        wall_counts = {}
        for wall in shell_objects:
            material_facts = _daylight_material_facts(wall)
            if not material_facts or not all(f['transmission'] <= .01 for f in material_facts):
                continue
            hit_indices = [i for i, point in enumerate(samples)
                           if _daylight_hits(wall, point - normal * .25, point + normal * .35, depsgraph)]
            if hit_indices:
                wall_counts[wall.name] = {'object': wall.name, 'surface_id': wall.get('surface_id'),
                    'covered_sample_indices': hit_indices, 'materials': material_facts,
                    'world_aabb_m': _daylight_mesh_bounds(wall, depsgraph)[2]}
        wall_coverage = set(i for record in wall_counts.values() for i in record['covered_sample_indices'])
        overlaps, photo_coverage = [], set()
        for owner_id, obj in custom_objects:
            material_facts = _daylight_material_facts(obj)
            if not material_facts or not all(f['photo_texture'] and f['transmission'] < .9 for f in material_facts):
                continue
            evaluated, corners, obj_bounds = _daylight_mesh_bounds(obj, depsgraph)
            distances = [(p - centre).dot(normal) for p in corners]
            if max(distances) - min(distances) > .025 or abs((max(distances) + min(distances)) / 2) > .2:
                continue
            if not evaluated.data.polygons:
                continue
            largest = max(evaluated.data.polygons, key=lambda poly: poly.area)
            face_normal = (evaluated.matrix_world.to_3x3().inverted_safe().transposed() @ largest.normal).normalized()
            if abs(face_normal.dot(normal)) < .98:
                continue
            hit_indices = [i for i, point in enumerate(samples)
                           if _daylight_hits(obj, point - normal * .25, point + normal * .25, depsgraph)]
            if hit_indices:
                photo_coverage.update(hit_indices)
                overlaps.append({'object_id': owner_id, 'part_object': obj.name,
                    'world_aabb_m': obj_bounds, 'materials': material_facts, 'covered_sample_indices': hit_indices})
        # Actual light-to-aperture segments corroborate retained-wall blocking.
        # These rays are not used to call arbitrary furniture a blocking fault.
        light_paths = []
        for light in lights:
            counts = {name: sum(_daylight_hits(bpy.data.objects[name], light.matrix_world.translation,
                        point - normal * .25, depsgraph) for point in samples) for name in wall_counts}
            light_paths.append({'light_id': light.name, 'position_m': list(light.matrix_world.translation),
                                'retained_wall_hit_samples': counts, 'sample_count': len(samples)})
        record = {'window_id': entry['id'], 'pane_object': pane.name, 'pane_world_aabb_m': bounds,
                  'pane_materials': facts, 'sample_count': len(samples),
                  'sample_method': '7x7 grid over central 90 percent of actual pane width/height',
                  'opaque_wall_coverage': len(wall_coverage) / len(samples),
                  'textured_duplicate_coverage': len(photo_coverage) / len(samples),
                  'retained_walls': list(wall_counts.values()), 'overlapping_planes': overlaps,
                  'associated_exterior_area_light_paths': light_paths}
        records.append(record)
        if len(wall_coverage) / len(samples) > .5:
            errors.append({'code': 'daylight_window_backed_by_opaque_wall', 'object_id': entry['id'],
                'surface_ids': sorted({v['surface_id'] for v in wall_counts.values()}),
                'light_ids': [light.name for light in lights], 'coverage_fraction': record['opaque_wall_coverage'],
                'evidence': record,
                'action': 'Create a real opening in the retained opaque wall using omitted shell plus surrounding custom wall segments; preserve room boundaries elsewhere. The declared window cannot transmit daylight through an opaque wall. Fix the opening before increasing light power or exposure.'})
        if len(photo_coverage) / len(samples) > .5:
            errors.append({'code': 'daylight_duplicate_textured_glazing', 'object_id': entry['id'],
                'overlapping_object_ids': sorted({v['object_id'] for v in overlaps}),
                'light_ids': [light.name for light in lights], 'coverage_fraction': record['textured_duplicate_coverage'],
                'evidence': record,
                'action': 'Resolve duplicated glazing: this base window already has clear glass, but nearby parallel photo-textured custom planes with transmission below 0.9 cover most of its aperture. Keep one coherent glazing construction. Do not paste photographed window brightness onto additional transmitting surfaces; preserve exterior context only with a deliberate physically coherent representation. Inspect the original photos before choosing geometry edits.'})
        elif overlaps:
            warnings.append({'code': 'daylight_partial_textured_overlap', 'object_id': entry['id'],
                'coverage_fraction': record['textured_duplicate_coverage'], 'evidence': record,
                'action': 'Inspect partial photo-textured overlap at window; no automatic geometry change was made.'})
    return {'errors': errors, 'warnings': warnings, 'windows': records,
            'scope': 'High-confidence retained opaque wall and duplicated textured partial-transmission aperture coverage only; largest inspectable clear pane per declared base window. Not a photometric simulation or exhaustive obstruction detector.'}


def geometry_preflight(bundle, handles):
    """Reject gross physical mistakes before calibration or production renders.

    Bounds come from evaluated geometry, including imported assets and local
    custom parts. Only preserved room boundaries constrain furniture. Thin
    boundary structures and ceiling trim are excluded conservatively; these
    tests are a render gate, not a full collision or visual-quality certificate.
    """
    bpy.context.view_layer.update()
    room = handles['base']['room']
    low = [-room['width'] / 2, 0, 0]
    high = [room['width'] / 2, room['depth'], room['height']]
    omitted = set(bundle['architecture']['omit_surfaces'])
    all_boundaries = [('left', 0, -1, low[0]), ('right', 0, 1, high[0]),
                      ('front', 1, -1, 0), ('back', 1, 1, high[1]),
                      ('floor', 2, -1, 0), ('ceiling', 2, 1, high[2])]
    boundaries = [b for b in all_boundaries if b[0] not in omitted and
                  (b[0] != 'front' or bundle['architecture']['front_wall'])]
    errors, warnings, camera_checks, objects = [], [], [], []
    clearance = .025
    # The actual shell retains only non-omitted surfaces. Its axis-aligned boxes
    # make distance-to-box exact, including cameras just inside a wall surface.
    shell = []
    for obj in handles['room'].children:
        if obj.type != 'MESH' or not obj.get('surface_id') or obj.name.startswith('Skirting'):
            continue
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        shell.append((obj, [min(p[i] for p in corners) for i in range(3)],
                      [max(p[i] for p in corners) for i in range(3)]))
    for view in bundle['views']:
        position = list(handles['cameras'][view['id']].matrix_world.translation)
        nearby = []
        for obj, wall_low, wall_high in shell:
            distance = math.sqrt(sum(max(wall_low[i] - position[i], 0,
                                         position[i] - wall_high[i]) ** 2 for i in range(3)))
            if distance <= clearance:
                nearby.append({'surface': obj['surface_id'], 'object': obj.name,
                               'distance_m': round(distance, 6)})
        camera_checks.append({'view_id': view['id'], 'position_m': position,
                              'minimum_required_clearance_m': clearance, 'nearby_shell': nearby})
        if nearby:
            errors.append({'code': 'camera_room_shell_collision', 'view_id': view['id'],
                           'position_m': position, 'surfaces': nearby,
                           'action': 'Move the camera clear of the preserved room shell. Check the X origin: room interior is [-width/2,+width/2], not [0,width]. Fix geometry before increasing lighting.'})
    wall_categories = {'window', 'door', 'curtain', 'column', 'beam', 'pendant_light', 'linear_light'}
    entries = [(entry, root, False) for entry, root in zip(handles['base']['objects'], handles['roots'])]
    entries += [(entry, root, True) for entry, root in zip(bundle['shared_scene']['custom_objects'], handles['custom_roots'])]
    for entry, root, custom in entries:
        bounds = v2.evaluated_bounds(root)
        size = [bounds['max'][i] - bounds['min'][i] for i in range(3)]
        # Structure is identified by geometry, never by model-chosen object IDs.
        # A panel centered on a room boundary may straddle it legitimately.
        near_boundary_panel = custom and any(
            surface != 'floor' and size[axis] <= .4 and
            min(size[i] for i in range(3) if i != axis) >= .5 and
            abs((bounds['min'][axis] + bounds['max'][axis]) / 2 - limit) <= .25
            for surface, axis, _, limit in all_boundaries)
        ceiling_trim = custom and size[2] <= .6 and bounds['min'][2] >= high[2] - .6
        structural = (not custom and entry['kind'] in wall_categories) or near_boundary_panel or ceiling_trim
        outside = []
        for surface, axis, sign, limit in boundaries:
            excess = (bounds['max'][axis] - limit) if sign > 0 else (limit - bounds['min'][axis])
            # Tiny bevels and trim intersections are not a reason to discard a
            # reconstruction. Custom groups also have arbitrary explicit origins.
            threshold = max(.25, .3 * size[axis]) if custom else .15
            if excess > threshold:
                outside.append({'surface': surface, 'distance_m': round(excess, 6),
                                'extent_m': round(size[axis], 6), 'threshold_m': round(threshold, 6)})
        objects.append({'object_id': entry['id'], 'world_aabb_m': bounds,
                        'structural_exemption': bool(structural), 'severe_boundary_violations': outside})
        if outside and not structural:
            errors.append({'code': 'geometry_outside_preserved_room', 'object_id': entry['id'],
                           'world_aabb_m': bounds, 'violations': outside,
                           'action': 'Correct object/group world placement or the declared room dimensions. Group parts remain local; do not shift local coordinates with world coordinates.'})
        elif outside:
            warnings.append({'code': 'structural_bounds_need_review', 'object_id': entry['id'], 'violations': outside})
    daylight = daylight_preflight(bundle, handles)
    errors.extend(daylight['errors'])
    warnings.extend(daylight['warnings'])
    material_binding = handles.get('material_binding', {'status': 'passed', 'errors': []})
    errors.extend(material_binding['errors'])
    exterior_context = handles.get('exterior_context', {'status': 'passed', 'errors': []})
    errors.extend(exterior_context['errors'])
    return {'schema_version': 1, 'status': 'rejected' if errors else 'passed',
            'room_interior_aabb_m': {'min': low, 'max': high},
            'errors': errors, 'warnings': warnings, 'camera_checks': camera_checks, 'objects': objects,
            'daylight_checks': daylight, 'material_binding': material_binding,
            'exterior_context': exterior_context,
            'scope': 'Evaluated asset bounds against preserved room surfaces and camera clearance from actual shell boxes. Open boundaries, lights and thin structural groups are exempt from furniture bounds. No full mesh collision or photo fidelity guarantee.'}


def enforce_geometry_preflight(bundle, handles, output, *, spec_sha256):
    """Persist diagnostics bound to the exact saved candidate before failing."""
    if (not isinstance(spec_sha256, str) or len(spec_sha256) != 64 or
            any(character not in '0123456789abcdef' for character in spec_sha256)):
        raise ValueError('Geometry preflight requires a valid SHA256 binding to the candidate spec')
    report = geometry_preflight(bundle, handles)
    report['spec_sha256'] = spec_sha256
    write_json(Path(output) / 'geometry_preflight.json', report)
    if report['status'] != 'passed':
        raise ValueError('Geometry preflight rejected the scene before rendering; fix camera/shell collisions, out-of-room objects and daylight aperture blockers using geometry_preflight.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--texture-manifest', type=Path)
    parser.add_argument('--asset-catalog', type=Path)
    parser.add_argument('--asset-bindings', type=Path)
    parser.add_argument('--asset-preview', action='store_true', help='Allow draft assets only for preview, never production acceptance')
    parser.add_argument('--max-edge', type=int, default=1280)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--device', choices=['CPU', 'AUTO'], default='AUTO')
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--appearance-policy', choices=['photo_reconstruction', 'legacy'], default='photo_reconstruction')
    parser.add_argument('--tone-match', choices=['reference', 'off'], default='reference')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    if bool(args.asset_catalog) != bool(args.asset_bindings):
        parser.error('--asset-catalog and --asset-bindings are required together')
    if not 1 <= args.samples <= 4096:
        parser.error('samples must be 1..4096')
    started = time.monotonic()
    bundle = load(args.spec)
    sources = resolve_sources(bundle, args.spec)
    source_hashes = {key: file_sha256(path) for key, path in sources.items()}
    spec_hash = file_sha256(args.spec)
    for view in bundle['views']:
        render_dimensions(view, args.max_edge)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metadata.json').unlink(missing_ok=True)
    catalog, bindings, asset_inputs = None, None, {}
    if args.asset_catalog:
        from asset_catalog import load_catalog
        catalog = load_catalog(args.asset_catalog)
        bindings = json.loads(args.asset_bindings.read_text(encoding='utf-8'))
        paths = [args.asset_catalog.resolve(), args.asset_bindings.resolve()]
        for binding in bindings['instances']:
            asset = catalog.get_asset(binding['asset_id'])
            material = catalog.get_material_variant(binding['material_variant_id'])
            paths.append(catalog.resolve_path(asset['model_file']))
            paths.extend(catalog.resolve_path(m['path']) for m in material['maps'].values() if m.get('path'))
            for slot_maps in (material.get('slot_materials') or {}).values():
                paths.extend(catalog.resolve_path(m['path']) for m in slot_maps.values() if m.get('path'))
            if asset.get('collision_file'):
                paths.append(catalog.resolve_path(asset['collision_file']))
        asset_inputs = {str(p): file_sha256(p) for p in set(paths)}
    handles = build_scene(bundle, args.texture_manifest, args.appearance_policy, catalog, bindings, args.asset_preview)
    scene, base, spec = handles['scene'], handles['base'], bundle['shared_scene']
    scene['source_bundle'] = str(args.spec.resolve())
    backend = configure_render(scene, spec, args.samples, args.device)
    if spec_hash != file_sha256(args.spec):
        raise RuntimeError('Candidate spec changed during construction; no preflight report was published')
    geometry_gate = enforce_geometry_preflight(bundle, handles, out, spec_sha256=spec_hash)
    tone = {'status': 'disabled', 'mode': args.tone_match}
    if args.tone_match == 'reference' and not args.no_render:
        from calibrate_reference import calibrate
        tone = calibrate(scene, bundle['views'], handles['cameras'], sources, set_view)
        handles['lighting']['applied_after_reference_fit'] = tone.get('lighting_fit', {}).get('applied_settings', {})
    elif args.tone_match == 'reference':
        tone = {'status': 'skipped_no_render', 'mode': args.tone_match}
    scene['reference_tone_calibration'] = json.dumps(tone, ensure_ascii=False)
    for view in bundle['views']:
        reference = bpy.data.images.load(str(sources[view['id']]), check_existing=True)
        reference.name = 'OriginalReference_' + view['id']
        reference.pack()
        reference['role'] = 'original_reference_not_rendered'
        reference['view_id'] = view['id']
        reference['source_sha256'] = source_hashes[view['id']]
        camera = handles['cameras'][view['id']]
        background = camera.data.background_images.new()
        background.image = reference
        background.alpha = .35
        camera.data.show_background_images = False
    provenance = {'interpretation': 'Supplied structured photo interpretation; dimensions, hidden geometry and cameras are estimated, not measured.',
                  'generation': 'Deterministic shared geometry; switching reference camera does not rebuild objects.',
                  'unattended_photo_analysis_verified': False,
                  'source_images': [{'view_id': key, 'path': str(path), 'sha256': source_hashes[key]} for key, path in sources.items()],
                  'texture_sources': texture_view_map(bundle), 'spec_sha256': spec_hash,
                  'appearance_policy': args.appearance_policy, 'tone_calibration': tone}
    provenance['external_assets'] = handles['external_assets']
    provenance['external_asset_inputs_sha256'] = asset_inputs
    provenance['asset_preview_only'] = args.asset_preview
    provenance['material_binding'] = handles['material_binding']
    provenance['exterior_context'] = handles['exterior_context']
    for name, value in [('scene_bundle_v3.json', bundle), ('source_provenance.json', provenance)]:
        block = bpy.data.texts.new(name)
        block.write(json.dumps(value, indent=2, ensure_ascii=False))
    outputs, view_records = ['scene.blend', 'source_provenance.json', 'tone_calibration.json', 'geometry_preflight.json', 'material_binding.json', 'exterior_context.json'], []
    write_json(out/'material_binding.json', handles['material_binding'])
    write_json(out/'exterior_context.json', handles['exterior_context'])
    write_json(out/'tone_calibration.json', tone)
    write_json(out/'source_provenance.json', provenance)
    first = bundle['views'][0]
    set_view(scene, handles['cameras'][first['id']], first, args.max_edge)
    scene.render.filepath = str(out / ('rgb_' + first['id'] + '.png'))
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    for view in bundle['views']:
        width, height = set_view(scene, handles['cameras'][view['id']], view, args.max_edge)
        viewbase = copy.deepcopy(base)
        viewbase['camera'] = {key: view['camera'][key] for key in ('position', 'target', 'lens_mm')}
        viewspec = copy.deepcopy(spec)
        viewspec['image'] = {'width': view['width'], 'height': view['height'], 'source_id': view['id']}
        annotation = v2.make_annotations(scene, viewspec, viewbase, handles['roots'], handles['custom_roots'])
        annotation['view_id'] = view['id']
        annotation['scene_id'] = bundle['scene_id']
        annotation['camera']['roll_degrees'] = view['camera']['roll_degrees']
        annotation['lighting'] = handles['lighting']
        annotation['architecture'] = bundle['architecture']
        annotation['external_assets'] = handles['external_assets']
        annotation['source_image_sha256'] = source_hashes[view['id']]
        name = 'annotations_' + view['id'] + '.json'
        write_json(out/name, annotation)
        outputs.append(name)
        record = {'id': view['id'], 'camera': handles['cameras'][view['id']].name,
                  'width': width, 'height': height, 'annotation': name,
                  'source_image': str(sources[view['id']]), 'source_sha256': source_hashes[view['id']],
                  'warning_count': len(annotation['warnings'])}
        if not args.no_render:
            name = 'rgb_' + view['id'] + '.png'
            scene.render.filepath = str(out/name)
            bpy.ops.render.render(write_still=True)
            outputs.append(name)
            record['render'] = name
        view_records.append(record)
    shutil.copyfile(out/('annotations_'+first['id']+'.json'), out/'annotations.json')
    outputs.append('annotations.json')
    if not args.no_render:
        shutil.copyfile(out/('rgb_'+first['id']+'.png'), out/'rgb.png')
        outputs.append('rgb.png')
        if args.tone_match == 'reference':
            from calibrate_reference import final_render_diagnostics
            tone['final_spatial_diagnostics'] = final_render_diagnostics(sources,
                {view['id']: out/('rgb_'+view['id']+'.png') for view in bundle['views']},
                at_correction_bound=tone.get('at_correction_bound', False))
            write_json(out/'tone_calibration.json', tone)
            write_json(out/'source_provenance.json', provenance)
    if spec_hash != file_sha256(args.spec) or any(source_hashes[key] != file_sha256(path) for key, path in sources.items()):
        raise RuntimeError('An input changed during generation; output is not marked complete')
    if any(file_sha256(path) != digest for path, digest in asset_inputs.items()):
        raise RuntimeError('An asset input changed during generation')
    metadata = {'status': 'built' if args.no_render else 'completed', 'schema_version': 3,
        'scene_id': bundle['scene_id'], 'scene_type': spec['scene_type'], 'views': view_records,
        'blender_version': bpy.app.version_string, 'geometry_build_count': 1,
        'camera_count': len(handles['cameras']), 'asset_count': len(handles['roots'])+len(handles['custom_roots']),
        'asset_builder_counts': handles['counts'], 'custom_group_count': len(handles['custom_roots']),
        'mesh_count': sum(obj.type == 'MESH' for obj in scene.objects),
        'curve_count': sum(obj.type == 'CURVE' for obj in scene.objects),
        'texture_count': handles['texture_count'], 'architecture': bundle['architecture'],
        'appearance_policy': args.appearance_policy,
        'geometry_preflight': geometry_gate,
        'material_binding': handles['material_binding'],
        'exterior_context': handles['exterior_context'],
        'tone_calibration': tone,
        'lighting': handles['lighting'], 'render': {'engine': 'CYCLES', 'device': backend, 'samples': args.samples, 'max_edge': args.max_edge,
            'view_transform': scene.view_settings.view_transform, 'look': scene.view_settings.look,
            'exposure': scene.view_settings.exposure, 'tone_match': args.tone_match},
        'provenance': provenance, 'quality_status': 'needs_visual_review',
        'elapsed_seconds': round(time.monotonic()-started, 3), 'outputs': outputs,
        'output_sha256': {name: file_sha256(out/name) for name in outputs}}
    write_json(out/'metadata.json', metadata)
    print('MULTIVIEW_COMPLETED ' + str(out), flush=True)


if __name__ == '__main__':
    main()
