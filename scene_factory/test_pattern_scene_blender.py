"""Actual Blender equivalence of declared arrays and explicit custom parts.

These small authored fixtures compare evaluated geometry and material binding.
They do not measure photograph fidelity or claim a successful reconstruction.
Run with blender -b --factory-startup --python-exit-code 1 --python this_file.
"""
import copy
import math
from pathlib import Path
import sys
import unittest

try:
    import bpy
    from mathutils import Euler, Vector
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class PatternSceneBlenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import compact_scene
        import generate_multiview
        import pattern_scene
        from spec_v2 import material, part
        from test_multiview import fixture
        cls.compact = compact_scene
        cls.renderer = generate_multiview
        cls.codec = pattern_scene
        cls.material = staticmethod(material)
        cls.part = staticmethod(part)
        cls.fixture = staticmethod(fixture)

    def bundle(self, group):
        bundle = self.fixture()
        scene = bundle['shared_scene']
        scene['scene_type'] = scene['base_scene']['scene_type'] = 'office'
        scene['base_scene']['objects'] = [{
            'id': 'reference_box', 'category': 'box', 'position': [0, 4, 0],
            'dimensions': [.2, .2, .2], 'rotation_z_deg': 0,
            'color': [.3, .3, .3], 'repeat': [1, 1], 'spacing': [0, 0],
            'confidence': .9,
        }]
        for key in ('textures', 'lights', 'base_material_overrides', 'exterior_contexts'):
            scene[key] = []
        scene['surface_overrides'] = dict.fromkeys(scene['surface_overrides'], '')
        scene['materials'] = [
            self.material('wood', [.32, .15, .06], roughness=.62),
            self.material('metal', [.1, .13, .17], roughness=.24, metallic=.85),
            self.material('paint', [.7, .73, .76], roughness=.82),
        ]
        scene['custom_objects'] = [copy.deepcopy(group)]
        scene['render']['replace_base_lights'] = True
        first = bundle['views'][0]
        scene['image'] = {'width': first['width'], 'height': first['height'],
                          'source_id': first['id']}
        scene['camera'] = {'roll_degrees': first['camera']['roll_degrees']}
        scene['base_scene']['camera'] = {key: copy.deepcopy(first['camera'][key])
                                         for key in ('position', 'target', 'lens_mm')}
        return bundle

    @staticmethod
    def group(identifier, parts, rotation=None):
        return {'id': identifier, 'position': [.5, 2.25, .125],
                'rotation_degrees': [11, -7, 31] if rotation is None else rotation,
                'material_id': 'wood', 'observed': True, 'confidence': .9,
                'support_id': '', 'parts': parts}

    @staticmethod
    def at_positions(source, positions):
        """Explicit fixture positions, intentionally not an array decoder."""
        result = []
        for position in positions:
            item = copy.deepcopy(source)
            item['position'] = list(position)
            result.append(item)
        return result

    @staticmethod
    def rounded(values):
        return tuple(round(float(value), 6) for value in values)

    def signature(self, canonical):
        handles = self.renderer.build_scene(canonical, appearance_policy='legacy')
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        result = {}
        for root in handles['custom_roots']:
            for obj in root.children_recursive:
                key = (obj['custom_group_id'], int(obj['custom_part_index']))
                evaluated = obj.evaluated_get(depsgraph)
                mesh = evaluated.to_mesh()
                try:
                    self.assertIsNotNone(mesh)
                    self.assertGreater(len(mesh.vertices), 0)
                    slots = []
                    for slot in obj.material_slots:
                        material = slot.material
                        shader = next(node for node in material.node_tree.nodes
                                      if node.type == 'BSDF_PRINCIPLED')
                        slots.append((material['scene_material_id'],
                                      self.rounded(shader.inputs['Base Color'].default_value),
                                      round(shader.inputs['Roughness'].default_value, 6),
                                      round(shader.inputs['Metallic'].default_value, 6)))
                    result[key] = {
                        'type': obj.type,
                        'part_type': obj['part_type'],
                        'matrix_world': tuple(self.rounded(row) for row in obj.matrix_world),
                        'world_origin': self.rounded(obj.matrix_world.translation),
                        'vertices_world': tuple(self.rounded(evaluated.matrix_world @ vertex.co)
                                                for vertex in mesh.vertices),
                        'edges': tuple(tuple(edge.vertices) for edge in mesh.edges),
                        'polygons': tuple((tuple(face.vertices), face.material_index,
                                           face.use_smooth) for face in mesh.polygons),
                        'uvs': tuple(tuple(self.rounded(loop.uv) for loop in layer.data)
                                     for layer in mesh.uv_layers),
                        'materials': tuple(slots),
                        'evaluated_material_ids': tuple(material.get('scene_material_id')
                                                        for material in mesh.materials),
                    }
                finally:
                    evaluated.to_mesh_clear()
        return result

    def assert_equivalent(self, group, arrays, explicit_parts, expected_materials):
        original = self.bundle(group)
        raw = self.compact.compact_response(original)
        raw['wire_format'] = self.codec.FORMAT_VERSION
        for item in raw['shared_scene']['base_scene']['objects']:
            item['placement'] = {'mode': 'world', 'position': item.pop('position')}
        raw['shared_scene']['custom_objects'][0]['part_arrays'] = copy.deepcopy(arrays)
        raw_before = copy.deepcopy(raw)
        expanded = self.codec.expand_response(raw)
        explicit = copy.deepcopy(original)
        explicit['shared_scene']['custom_objects'][0]['parts'] = explicit_parts

        actual = self.signature(expanded)
        expected = self.signature(explicit)
        self.assertEqual(actual, expected)
        self.assertEqual(raw, raw_before)
        self.assertEqual(len(actual), len(explicit_parts))
        ordered = [actual[(group['id'], index)] for index in range(len(explicit_parts))]
        self.assertEqual([record['materials'][0][0] for record in ordered], expected_materials)
        self.assertTrue(all(record['evaluated_material_ids'] == (material,)
                            for record, material in zip(ordered, expected_materials)))
        # Independently check part origins in group coordinates; the part's
        # own rotation must not rotate the repetition translation.
        rotation = Euler(tuple(math.radians(value) for value in group['rotation_degrees']), 'XYZ').to_matrix()
        for record, item in zip(ordered, explicit_parts):
            expected_origin = Vector(group['position']) + rotation @ Vector(item['position'])
            for actual_axis, expected_axis in zip(record['world_origin'], expected_origin):
                self.assertAlmostEqual(actual_axis, expected_axis, places=5)
        receipt = self.codec.pattern_report(raw, expanded)
        self.assertEqual(receipt['counts']['expanded_parts'], len(explicit_parts))
        return ordered

    def test_ceiling_grid_and_singleton_have_identical_evaluated_geometry(self):
        beam = self.part('box', size=[3, .06, .04], position=[0, -.75, 2.5],
                         rotation_degrees=[0, 0, 17], material_id='metal')
        rail = self.part('box', size=[.05, 2, .03], position=[-1.25, 0, 2.55],
                         rotation_degrees=[2, 0, -9], material_id='paint')
        marker = self.part('box', size=[.1, .1, .1], position=[.125, .25, 2.4])
        explicit = (self.at_positions(beam, [(0, -.75, 2.5), (0, 0, 2.5),
                                             (0, .75, 2.5), (0, 1.5, 2.5)])
                    + self.at_positions(rail, [(-1.25, 0, 2.55), (0, 0, 2.55),
                                               (1.25, 0, 2.55)])
                    + [copy.deepcopy(marker)])
        self.assert_equivalent(self.group('ceiling_grid', [beam, rail, marker]), [
            {'part_indices': [0], 'count': [1, 4, 1], 'spacing': [0, .75, 0]},
            {'part_indices': [1], 'count': [3, 1, 1], 'spacing': [1.25, 0, 0]},
        ], explicit, ['metal'] * 4 + ['paint'] * 3 + ['wood'])

    def test_shelf_negative_vertical_array_preserves_inherited_and_override_materials(self):
        board = self.part('box', size=[.8, .4, .04], position=[0, 0, 1.5],
                          rotation_degrees=[3, 2, -4])
        upright = self.part('box', size=[.04, .4, 1.5], position=[-.375, 0, .75],
                            rotation_degrees=[0, 0, 6], material_id='metal')
        back = self.part('box', size=[.8, .025, 1.5], position=[0, .2, .75],
                         material_id='paint')
        explicit = (self.at_positions(board, [(0, 0, 1.5), (0, 0, 1.125),
                                              (0, 0, .75), (0, 0, .375)])
                    + self.at_positions(upright, [(-.375, 0, .75), (.375, 0, .75)])
                    + [copy.deepcopy(back)])
        self.assert_equivalent(self.group('shelf', [board, upright, back], [-8, 5, 63]), [
            {'part_indices': [0], 'count': [1, 1, 4], 'spacing': [0, 0, -.375]},
            {'part_indices': [1], 'count': [2, 1, 1], 'spacing': [.75, 0, 0]},
        ], explicit, ['wood'] * 4 + ['metal'] * 2 + ['paint'])

    def test_multi_prototype_mesh_and_polyline_arrays_preserve_evaluated_topology(self):
        mesh = self.part('mesh', position=[-.5, 0, .5], rotation_degrees=[13, 21, -37],
                         material_id='paint',
                         vertices=[[0, 0, 0], [.3, 0, 0], [0, .2, 0], [0, 0, .4]],
                         faces=[[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
        rail = self.part('polyline', position=[-.5, .125, .25], radius=.015,
                         rotation_degrees=[-11, 19, 28], material_id='metal',
                         points=[[-.2, 0, 0], [-.2, 0, .45], [.2, 0, .45], [.2, 0, 0]])
        explicit = (self.at_positions(mesh, [(-.5, 0, .5), (-.5, .5, .5),
                                             (.25, 0, .5), (.25, .5, .5)])
                    + self.at_positions(rail, [(-.5, .125, .25), (-.5, .625, .25),
                                               (.25, .125, .25), (.25, .625, .25)]))
        records = self.assert_equivalent(self.group('equipment', [mesh, rail], [17, -14, 39]), [
            {'part_indices': [0, 1], 'count': [2, 2, 1], 'spacing': [.75, .5, 0]},
        ], explicit, ['paint'] * 4 + ['metal'] * 4)
        self.assertEqual([record['type'] for record in records], ['MESH'] * 4 + ['CURVE'] * 4)
        self.assertTrue(all(len(record['vertices_world']) == 4 for record in records[:4]))
        self.assertTrue(all(len(record['vertices_world']) > 4 for record in records[4:]))


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PatternSceneBlenderTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
