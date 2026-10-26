"""Shared 2.5D window scenery, explicitly separated from illumination."""
from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector


def exterior_material(identifier, image, brightness_ev):
    material = bpy.data.materials.new('ExteriorContext_' + identifier)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    texture = nodes.new('ShaderNodeTexImage'); texture.image = image
    texture.extension = 'EXTEND'
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 2 ** brightness_ev
    links.new(texture.outputs['Color'], emission.inputs['Color'])
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    path = nodes.new('ShaderNodeLightPath')
    visible = nodes.new('ShaderNodeMath'); visible.operation = 'MAXIMUM'
    links.new(path.outputs['Is Camera Ray'], visible.inputs[0])
    links.new(path.outputs['Is Glossy Ray'], visible.inputs[1])
    transmitted = nodes.new('ShaderNodeMath'); transmitted.operation = 'MAXIMUM'
    links.new(visible.outputs[0], transmitted.inputs[0])
    links.new(path.outputs['Is Transmission Ray'], transmitted.inputs[1])
    # A transmitted path that previously scattered diffusely must not receive
    # this radiance. Object visibility alone is insufficient after a glass bounce.
    diffuse_depth = nodes.new('ShaderNodeMath'); diffuse_depth.operation = 'LESS_THAN'
    links.new(path.outputs['Diffuse Depth'], diffuse_depth.inputs[0]); diffuse_depth.inputs[1].default_value = .5
    mask = nodes.new('ShaderNodeMath'); mask.operation = 'MULTIPLY'
    links.new(transmitted.outputs[0], mask.inputs[0]); links.new(diffuse_depth.outputs[0], mask.inputs[1])
    no_shadow = nodes.new('ShaderNodeMath'); no_shadow.operation = 'SUBTRACT'
    no_shadow.inputs[0].default_value = 1; links.new(path.outputs['Is Shadow Ray'], no_shadow.inputs[1])
    final = nodes.new('ShaderNodeMath'); final.operation = 'MULTIPLY'
    links.new(mask.outputs[0], final.inputs[0]); links.new(no_shadow.outputs[0], final.inputs[1])
    mix = nodes.new('ShaderNodeMixShader')
    links.new(final.outputs[0], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1]); links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs['Surface'])
    material['exterior_context_only'] = True
    material['illumination_source'] = False
    if hasattr(material, 'cycles') and hasattr(material.cycles, 'sample_as_light'):
        material.cycles.sample_as_light = False
    if hasattr(material, 'cycles') and hasattr(material.cycles, 'emission_sampling'):
        material.cycles.emission_sampling = 'NONE'
    return material


def build_exterior_contexts(base, contexts, images, cameras):
    entries = {entry['id']: entry for entry in base['objects']}
    records, errors = [], []
    room = base['room']
    for context in contexts:
        entry = entries[context['window_id']]
        width, depth, height = entry['dimensions']
        angle = math.radians(entry.get('rotation_z_deg', 0))
        rotation = Matrix.Rotation(angle, 3, 'Z')
        xaxis, outward, zaxis = [rotation @ Vector(v) for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        center = Vector(entry['position']) + Vector((0, 0, height / 2))
        boundaries = [(Vector((-room['width'] / 2, center.y, center.z)), Vector((-1, 0, 0))),
                      (Vector((room['width'] / 2, center.y, center.z)), Vector((1, 0, 0))),
                      (Vector((center.x, 0, center.z)), Vector((0, -1, 0))),
                      (Vector((center.x, room['depth'], center.z)), Vector((0, 1, 0)))]
        attached = any((center - point).length <= max(.3, depth) and outward.dot(normal) > .8
                       for point, normal in boundaries)
        if not attached:
            errors.append({'code': 'exterior_context_invalid_window_attachment', 'window_id': entry['id'],
                           'action': 'Place the declared window near the room exterior boundary with local +Y pointing outside; never use an interior backdrop.'})
            continue
        distance = depth / 2 + context['depth_m']
        # Fixed plane spans aperture projection from every input view. The
        # crop/plane never changes with active render camera.
        projected = []
        for camera in cameras:
            eye = Vector(camera)
            denominator = (center - eye).dot(outward)
            if denominator <= .025:
                errors.append({'code': 'exterior_context_camera_outside_window', 'window_id': entry['id'],
                               'action': 'Camera must be on the interior (-Y) side of this window.'})
                continue
            factor = (denominator + distance) / denominator
            for u, v in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)):
                corner = center + width * u * xaxis + height * v * zaxis
                point = eye + factor * (corner - eye) - center - distance * outward
                projected.append((point.dot(xaxis), point.dot(zaxis)))
        if not projected:
            continue
        low = [min(p[i] for p in projected) for i in range(2)]
        high = [max(p[i] for p in projected) for i in range(2)]
        if high[0] - low[0] > width * 12 or high[1] - low[1] > height * 12:
            errors.append({'code': 'exterior_context_projection_too_large', 'window_id': entry['id'],
                           'action': 'Reduce depth_m; exterior plane projection is limited to 12 times the associated window dimensions.'})
            continue
        corners = [(low[0], low[1]), (high[0], low[1]), (high[0], high[1]), (low[0], high[1])]
        vertices = [center + distance * outward + x * xaxis + z * zaxis for x, z in corners]
        mesh = bpy.data.meshes.new('ExteriorContextMesh_' + entry['id'])
        mesh.from_pydata(vertices, [], [(0, 1, 2, 3)]); mesh.update()
        obj = bpy.data.objects.new('ExteriorContext_' + entry['id'], mesh)
        bpy.context.collection.objects.link(obj)
        uv = mesh.uv_layers.new(name='ExteriorCropUV')
        for loop, point in zip(uv.data, ((0, 0), (1, 0), (1, 1), (0, 1))):
            loop.uv = point
        material = exterior_material(entry['id'], images[context['texture_id']], context['brightness_ev'])
        mesh.materials.append(material)
        obj.visible_diffuse = False; obj.visible_shadow = False
        obj.visible_volume_scatter = False
        obj['exterior_context_window_id'] = entry['id']
        records.append({**context, 'object_name': obj.name, 'material': material.name,
                        'plane_world_vertices_m': [list(v) for v in vertices],
                        'diffuse_illumination': False, 'shadow_occlusion': False,
                        'shared_across_views': True, 'method': 'window_attached_2_5d_exterior_crop',
                        'limitations': 'Flat exterior approximation; no reconstructed building depth or correct large-baseline parallax. Crop semantics require independent visual review.'})
    return {'schema_version': 1, 'status': 'rejected' if errors else 'passed',
            'contexts': records, 'errors': errors,
            'scope': 'Visible window exterior is separate from physical incident daylight.'}
