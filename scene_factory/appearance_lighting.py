"""Visible fixture faces for nearby, geometry-backed AREA lights.

AREA lights supply the illumination. These faces only supply their visible
appearance and reflections; they do not add a second diffuse emitter.
No object names or scene categories are used to infer fixtures.
"""
import math

import bpy
from mathutils import Vector


def _appearance_emission(face):
    """Recognize only our tagged, non-illuminating generated fixture graph."""
    if (face.type != 'MESH' or not face.get('fixture_appearance_only')
            or face.visible_diffuse or face.visible_shadow or face.visible_volume_scatter
            or len(face.data.materials) != 1):
        return None
    material = face.material_slots[0].material
    if not material or material.library or not material.use_nodes:
        return None
    nodes = material.node_tree.nodes
    emissions = [node for node in nodes if node.type == 'EMISSION']
    # Do not edit a user-modified/foreign material on the basis of a name.
    if (len(emissions) != 1 or not any(node.type == 'LIGHT_PATH' for node in nodes)
            or not any(node.type == 'BSDF_TRANSPARENT' for node in nodes)
            or not any(node.type == 'MIX_SHADER' for node in nodes)
            or emissions[0].inputs['Strength'].is_linked):
        return None
    return emissions[0]


def bind_fixture_power_drivers(scene):
    """Bind generated appearance strength to its actual AREA Light datablock.

    The driver survives saving and follows shared light data and power rollback.
    It only changes the existing appearance-only emission socket; no ray graph,
    visibility flag or diffuse emitter is added. Older generated faces can be
    recognized by their explicit tags and original source-light reference.
    Linked materials and foreign drivers remain untouched.
    """
    bpy.context.view_layer.update()
    records = []
    for face in list(scene.objects):
        emission = _appearance_emission(face)
        if emission is None:
            if face.get('fixture_appearance_only'):
                records.append({'face': face.name, 'status': 'skipped',
                                'reason': 'not_recognized_editable_appearance_graph'})
            continue
        lamp = face.get('source_light_object') or scene.objects.get(face.get('source_light', ''))
        if not isinstance(lamp, bpy.types.Object) or lamp.type != 'LIGHT' or lamp.data.type != 'AREA':
            records.append({'face': face.name, 'status': 'skipped', 'reason': 'source_light_missing_or_not_area'})
            continue
        area = face.get('fixture_emitting_area_m2')
        if area is None:
            # Legacy faces were generated in lamp local XY; their world mesh
            # area is the actual appearance surface even for a rotated lamp.
            face.data.calc_loop_triangles()
            area = 0.0
            for triangle in face.data.loop_triangles:
                a, b, c = [face.matrix_world @ face.data.vertices[index].co
                           for index in triangle.vertices]
                area += (b - a).cross(c - a).length / 2
        if not isinstance(area, (int, float)) or not math.isfinite(area) or area <= 0:
            records.append({'face': face.name, 'status': 'skipped', 'reason': 'invalid_emitting_area'})
            continue
        tree = emission.id_data
        animation = tree.animation_data
        if animation and (animation.action or animation.nla_tracks):
            records.append({'face': face.name, 'status': 'skipped',
                            'reason': 'animated_appearance_graph'})
            continue
        path = emission.inputs['Strength'].path_from_id('default_value')
        existing = next((curve for curve in tree.animation_data.drivers
                         if curve.data_path == path), None) if tree.animation_data else None
        version = face.get('fixture_power_binding_version')
        if existing:
            variable_matches = (len(existing.driver.variables) == 1
                and existing.driver.variables[0].name == 'watts'
                and existing.driver.variables[0].targets[0].id == lamp.data
                and existing.driver.variables[0].targets[0].data_path == 'energy')
            expected_expression = 'min(500.0,max(0.0,watts)/' + repr(float(area) * math.pi) + ')'
            if version != 1 or not variable_matches or existing.driver.expression != expected_expression:
                records.append({'face': face.name, 'status': 'skipped', 'reason': 'foreign_strength_driver'})
                continue
        material = face.material_slots[0].material
        if material.users > 1 or face.data.users > 1:
            # Object binding isolates faces sharing a mesh as well as material.
            material = material.copy()
            face.material_slots[0].link = 'OBJECT'
            face.material_slots[0].material = material
            emission = next(node for node in material.node_tree.nodes if node.type == 'EMISSION')
            tree = emission.id_data
            existing = next((curve for curve in tree.animation_data.drivers
                             if curve.data_path == path), None) if tree.animation_data else None
        curve = existing or emission.inputs['Strength'].driver_add('default_value')
        driver = curve.driver
        if not existing:
            driver.type = 'SCRIPTED'
            variable = driver.variables.new()
            variable.name = 'watts'
            variable.type = 'SINGLE_PROP'
            variable.targets[0].id_type = 'LIGHT'
            variable.targets[0].id = lamp.data
            variable.targets[0].data_path = 'energy'
            # Zero power extinguishes the visible face. No diffuse emitter is
            # added; the existing ray-type mix remains exactly as constructed.
            driver.expression = 'min(500.0,max(0.0,watts)/' + repr(float(area) * math.pi) + ')'
        face['source_light_object'] = lamp
        face['fixture_emitting_area_m2'] = float(area)
        face['fixture_power_binding_version'] = 1
        records.append({'light': lamp.name, 'face': face.name, 'status': 'bound',
                        'light_datablock': lamp.data.name, 'emitting_area_m2': float(area),
                        'method': 'direct_light_energy_driver',
                        'additional_diffuse_emission': False})
    bpy.context.view_layer.update()
    return records


def add_fixture_faces(scene):
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    records = []
    for lamp in list(scene.objects):
        if lamp.type != 'LIGHT' or lamp.data.type != 'AREA' or lamp.get('exterior_illumination_proxy'):
            continue
        if lamp.data.energy <= 0 or lamp.data.shape not in {'RECTANGLE', 'SQUARE', 'DISK'}:
            continue
        normal = lamp.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        hit, point, face_normal, _, backing, _ = scene.ray_cast(
            depsgraph, lamp.location - normal * .0001, -normal, distance=.12)
        if not hit or backing.type != 'MESH' or abs(face_normal.dot(normal)) < .95:
            continue
        inverse = lamp.matrix_world.inverted()
        corners = [inverse @ (backing.matrix_world @ Vector(corner)) for corner in backing.bound_box]
        spans = [max(c[i] for c in corners)-min(c[i] for c in corners) for i in range(3)]
        sx = lamp.data.size
        sy = lamp.data.size_y if lamp.data.shape == 'RECTANGLE' else sx
        # Do not turn a ceiling, wall or oversized softbox proxy into a fixture.
        if spans[2] > .15 or not (.65*sx <= spans[0] <= 1.6*sx and .65*sy <= spans[1] <= 1.6*sy):
            continue
        material = bpy.data.materials.new('FixtureAppearance_' + lamp.name)
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = (*lamp.data.color, 1)
        area = sx*sy*(math.pi/4 if lamp.data.shape == 'DISK' else 1)
        emission.inputs['Strength'].default_value = min(500, max(0, lamp.data.energy/(area*math.pi)))
        transparent = nodes.new('ShaderNodeBsdfTransparent')
        ray = nodes.new('ShaderNodeLightPath')
        visible = nodes.new('ShaderNodeMath'); visible.operation = 'MAXIMUM'
        links.new(ray.outputs['Is Camera Ray'], visible.inputs[0])
        links.new(ray.outputs['Is Glossy Ray'], visible.inputs[1])
        through_glass = nodes.new('ShaderNodeMath'); through_glass.operation = 'MAXIMUM'
        links.new(visible.outputs[0], through_glass.inputs[0])
        links.new(ray.outputs['Is Transmission Ray'], through_glass.inputs[1])
        # A glossy/transmission segment can follow a diffuse bounce. Exclude
        # the entire path once that has happened, or it adds indirect energy
        # even when object.visible_diffuse is disabled on the current segment.
        no_diffuse = nodes.new('ShaderNodeMath'); no_diffuse.operation = 'LESS_THAN'
        no_diffuse.inputs[1].default_value = .5
        links.new(ray.outputs['Diffuse Depth'], no_diffuse.inputs[0])
        appearance_only = nodes.new('ShaderNodeMath'); appearance_only.operation = 'MULTIPLY'
        links.new(through_glass.outputs[0], appearance_only.inputs[0])
        links.new(no_diffuse.outputs[0], appearance_only.inputs[1])
        mix = nodes.new('ShaderNodeMixShader')
        links.new(appearance_only.outputs[0], mix.inputs[0])
        links.new(transparent.outputs[0], mix.inputs[1])
        links.new(emission.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], output.inputs['Surface'])
        if hasattr(material, 'cycles') and hasattr(material.cycles, 'emission_sampling'):
            material.cycles.emission_sampling = 'NONE'
        if lamp.data.shape == 'DISK':
            vertices = [(math.cos(i*math.tau/48)*sx/2, math.sin(i*math.tau/48)*sy/2, 0) for i in range(48)]
        else:
            vertices = [(-sx/2,-sy/2,0),(-sx/2,sy/2,0),(sx/2,sy/2,0),(sx/2,-sy/2,0)]
        mesh = bpy.data.meshes.new('FixtureFace_' + lamp.name)
        mesh.from_pydata(vertices, [], [tuple(range(len(vertices)))])
        mesh.update()
        face = bpy.data.objects.new('FixtureFace_' + lamp.name, mesh)
        scene.collection.objects.link(face)
        face.matrix_world = lamp.matrix_world.copy()
        face.location = point + normal*.0005
        mesh.materials.append(material)
        face.visible_diffuse = False
        face.visible_shadow = False
        face.visible_volume_scatter = False
        face['fixture_appearance_only'] = True
        face['source_light'] = lamp.name
        face['backing_geometry'] = backing.name
        face['fixture_emitting_area_m2'] = area
        records.append({'light': lamp.name, 'face': face.name, 'backing': backing.name,
                        'additional_diffuse_emission': False})
    bind_fixture_power_drivers(scene)
    return records
