"""Blender materials for the declarative V2 renderer; no downloads or code execution."""
from __future__ import annotations

import json
from pathlib import Path

import bpy


def load_texture_manifest(path):
    """Read pre-rectified local crops. Accept a mapping or a list with id/path."""
    if path is None:
        return {}
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding='utf-8'))
    records = payload.get('textures', payload) if isinstance(payload, dict) else payload
    if isinstance(records, dict):
        records = [dict(value, id=key) if isinstance(value, dict)
                   else {'id': key, 'path': value} for key, value in records.items()]
    if not isinstance(records, list):
        raise ValueError('Texture manifest must contain a textures mapping or list')
    images = {}
    for record in records:
        texture_id = record['id']
        if texture_id in images:
            raise ValueError('Duplicate texture manifest id: ' + texture_id)
        filename = record.get('path', record.get('file', record.get('crop_path')))
        if not isinstance(filename, str):
            raise ValueError('Missing crop path for texture: ' + texture_id)
        image_path = Path(filename)
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = bpy.data.images.load(str(image_path.resolve()), check_existing=True)
        image.pack()
        images[texture_id] = image
    return images


def build_materials(definitions, images):
    result = {}
    for definition in definitions:
        name = definition['id']
        mat = bpy.data.materials.new('V2_' + name)
        mat.use_nodes = True
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        shader = next(node for node in nodes if node.type == 'BSDF_PRINCIPLED')
        shader.inputs['Base Color'].default_value = (*definition['color'], 1)
        shader.inputs['Roughness'].default_value = definition['roughness']
        shader.inputs['Metallic'].default_value = definition['metallic']
        shader.inputs['Transmission Weight'].default_value = definition['transmission']
        texture_id = definition['texture_id']
        if texture_id:
            if texture_id not in images:
                raise ValueError('No preprocessed crop for texture: ' + texture_id)
            texture = nodes.new('ShaderNodeTexImage')
            texture.name = 'PhotoCrop_' + texture_id
            texture.image = images[texture_id]
            links.new(texture.outputs['Color'], shader.inputs['Base Color'])
            coord = nodes.new('ShaderNodeTexCoord')
            links.new(coord.outputs['UV'], texture.inputs['Vector'])
        if definition['noise_scale'] > 0 and definition['bump_strength'] > 0:
            noise = nodes.new('ShaderNodeTexNoise')
            noise.inputs['Scale'].default_value = definition['noise_scale']
            noise.inputs['Detail'].default_value = 2
            coord = nodes.new('ShaderNodeTexCoord')
            links.new(coord.outputs['Object'], noise.inputs['Vector'])
            bump = nodes.new('ShaderNodeBump')
            bump.inputs['Strength'].default_value = definition['bump_strength']
            bump.inputs['Distance'].default_value = definition['bump_distance']
            links.new(noise.outputs['Fac'], bump.inputs['Height'])
            links.new(bump.outputs['Normal'], shader.inputs['Normal'])
        mat['scene_material_id'] = name
        mat['parameter_source'] = 'image_interpretation_and_estimation_not_measured_BRDF'
        result[name] = mat
    return result


def assign_material(obj, material):
    if len(obj.data.materials) == 0:
        obj.data.materials.append(material)
    obj.material_slots[0].link = 'OBJECT'
    obj.material_slots[0].material = material


def planar_uv(obj):
    """Give generated mesh faces a deterministic planar UV rather than (0, 0)."""
    mesh = obj.data
    if mesh.uv_layers:
        return
    uv = mesh.uv_layers.new(name='V2_UV')
    for polygon in mesh.polygons:
        axis = max(range(3), key=lambda index: abs(polygon.normal[index]))
        axes = [index for index in range(3) if index != axis]
        points = [mesh.vertices[index].co for index in polygon.vertices]
        low = [min(p[index] for p in points) for index in axes]
        span = [max(p[index] for p in points) - low[i] for i, index in enumerate(axes)]
        for loop_index in polygon.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv.data[loop_index].uv = tuple((co[index] - low[i]) / max(span[i], 1e-9)
                                           for i, index in enumerate(axes))
