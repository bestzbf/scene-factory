"""Blender contract tests for explicit built-in source material roles.

These tests inspect used material slots (the same evidence exposed by
material_binding), so an unused material created by a builder cannot satisfy a
role claim. They intentionally cover dispatch branches rather than claiming
all arbitrary external assets.
"""
import sys
import unittest
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class BuiltinMaterialRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import detailed_assets
        import interior_assets
        from base_material_binding import _source_role
        cls.detailed = detailed_assets
        cls.interior = interior_assets
        cls.source_role = staticmethod(_source_role)

    def tearDown(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

    def roles(self, root):
        result = set()
        for obj in [root] + list(root.children_recursive):
            if obj.type != 'MESH':
                continue
            used = {poly.material_index for poly in obj.data.polygons}
            for index, slot in enumerate(obj.material_slots):
                if index in used and slot.material is not None:
                    role = slot.material.get('asset_material_role')
                    self.assertIn(role, {'body', 'metal', 'glass', 'fabric'},
                                  (obj.name, slot.material.name, role))
                    self.assertEqual(role, self.source_role(slot.material))
                    result.add(role)
        return result

    def entry(self, kind, scene_type='other', color=(.34, .38, .42)):
        dimensions = {
            'bed': (2, 2, .9), 'sofa': (2, .9, .9), 'cabinet': (1, .45, 1.8),
            'shelf': (1, .35, 1.8), 'chair': (.65, .65, 1.1),
            'desk': (1.4, .75, .76), 'table': (1.4, .75, .76),
            'monitor': (.65, .23, .48), 'computer': (.25, .45, .6),
            'plant': (.6, .6, 1.7),
        }
        return {'id': 'role_' + kind + '_' + scene_type, 'category': kind,
                'color': list(color), 'dimensions': dimensions[kind],
                'position': [0, 0, 0]}

    def test_interior_roles_are_explicit_and_used(self):
        expected = {
            'bed': {'body', 'fabric'}, 'sofa': {'body', 'fabric'},
            'cabinet': {'body', 'metal'}, 'shelf': {'body'},
            'chair': {'body', 'fabric'},
        }
        for kind, wanted in expected.items():
            with self.subTest(kind=kind):
                root = self.interior.build(self.entry(kind), scene_type='bedroom')
                self.assertTrue(wanted <= self.roles(root))

    def test_detailed_roles_cover_office_assets_and_screen_id(self):
        expected = {
            'desk': {'body', 'metal'}, 'table': {'body'},
            'chair': {'body', 'fabric', 'metal'}, 'monitor': {'body', 'metal'},
            'computer': {'body', 'metal'}, 'plant': {'body'},
        }
        for policy in ('photo_reconstruction', 'legacy'):
            for kind, wanted in expected.items():
                for color in ((.34, .38, .42), (.61, .43, .23)):
                    with self.subTest(kind=kind, policy=policy, color=color):
                        root = self.detailed.build_asset(self.entry(kind, color=color), appearance_policy=policy)
                        self.assertTrue(wanted <= self.roles(root))
                        if kind == 'monitor':
                            ids = {slot.material.get('scene_material_id')
                                   for obj in [root] + list(root.children_recursive)
                                   if obj.type == 'MESH'
                                   for slot in obj.material_slots if slot.material}
                            self.assertIn('monitor_screen', ids)

    def test_role_cache_does_not_cross_contaminate_shared_shader_materials(self):
        import generate_scene as g
        from assets_lib.material_roles import source_role
        source = g.material('RoleCacheCollisionBody', (.153, .257, .359), .537)
        # The parameter cache deliberately aliases these different names.
        self.assertIs(source, g.material('RoleCacheCollisionFabric', (.153, .257, .359), .537))
        body = source_role(source, 'body')
        fabric = source_role(source, 'fabric')
        self.assertIsNot(body, fabric)
        self.assertNotIn('asset_material_role', source)
        self.assertEqual(self.source_role(body), 'body')
        self.assertEqual(self.source_role(fabric), 'fabric')
        self.assertIs(source_role(source, 'body'), body)
        self.assertEqual(body.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value[:],
                         fabric.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value[:])

    def test_role_copy_cache_rebuilds_removed_datablock(self):
        from assets_lib.material_roles import source_role
        source = bpy.data.materials.new('RoleCacheOwnershipSource')
        tagged = source_role(source, 'fabric')
        bpy.data.materials.remove(tagged)
        rebuilt = source_role(source, 'fabric')
        self.assertIs(bpy.data.materials.get(rebuilt.name), rebuilt)
        self.assertEqual(rebuilt.get('asset_material_role'), 'fabric')
        bpy.data.materials.remove(source)
        replacement = bpy.data.materials.new('RoleCacheOwnershipSource')
        replacement_tag = source_role(replacement, 'metal')
        self.assertEqual(replacement_tag.get('asset_material_role'), 'metal')
        self.assertNotIn('asset_material_role', replacement)

    def test_dispatch_roles_survive_sequential_scene_rebuilds(self):
        from generate_multiview import build_scene
        from test_multiview import fixture
        for scene_type, wanted, origin in (
                ('office', {'body', 'fabric', 'metal'}, 'office_detailed'),
                ('bedroom', {'body', 'fabric'}, 'interior'),
                ('classroom', {'body', 'fabric', 'metal'}, 'office_detailed')):
            with self.subTest(scene_type=scene_type):
                bundle = fixture()
                scene = bundle['shared_scene']
                scene['scene_type'] = scene['base_scene']['scene_type'] = scene_type
                entry = scene['base_scene']['objects'][0]
                entry.update(category='chair', color=[.153, .257, .359], repeat=[1, 1],
                             dimensions=[.65, .65, 1.1], position=[0, 2, 0])
                scene['base_scene']['objects'] = [entry]
                scene['custom_objects'] = []
                scene['textures'] = []
                scene['base_material_overrides'] = []
                scene['surface_overrides'] = dict.fromkeys(scene['surface_overrides'], '')
                handles = build_scene(bundle, appearance_policy='photo_reconstruction')
                self.assertEqual(handles['counts'][origin], 1)
                self.assertTrue(wanted <= self.roles(handles['roots'][0]))
                inventory = handles['material_binding']['available_slots']
                self.assertTrue(wanted <= {slot['role'] for slot in inventory})

    def test_legacy_table_covers_both_seed_branches(self):
        for parity in (0, 1):
            entry = self.entry('table')
            seed = next(candidate for candidate in range(20)
                        if self.detailed._choice(entry, candidate) % 2 == parity)
            root = self.detailed.build_asset(entry, seed=seed, appearance_policy='legacy')
            roles = self.roles(root)
            self.assertIn('body', roles)
            self.assertEqual('metal' in roles, parity == 0)

    def test_metallic_target_binds_shelf_body_without_inventing_source_metal(self):
        from generate_multiview import build_scene
        from spec_v2 import material
        from test_multiview import fixture
        bundle = fixture()
        scene = bundle['shared_scene']
        entry = scene['base_scene']['objects'][0]
        entry.update(id='steel_shelf', category='shelf', color=[.4, .4, .4],
                     repeat=[1, 1], dimensions=[1, .4, 1.8], position=[0, 2, 0])
        scene['base_scene']['objects'] = [entry]
        scene['custom_objects'] = []
        scene['textures'] = []
        scene['surface_overrides'] = dict.fromkeys(scene['surface_overrides'], '')
        replacement = material('steel', [.25, .27, .28])
        replacement['metallic'] = 1
        scene['materials'] = [replacement]
        binding = {'object_id': 'steel_shelf', 'material_id': 'steel',
                   'selector': {'role': 'body', 'material_name': '', 'slot_index': -1}}
        scene['base_material_overrides'] = [binding]
        for _ in range(2):
            handles = build_scene(bundle)
            report = handles['material_binding']
            self.assertEqual(report['status'], 'passed')
            self.assertGreater(report['applied_slot_count'], 0)
            self.assertEqual({slot['role'] for slot in report['available_slots']}, {'body'})
            for slot in report['bindings'][0]['matched_slots']:
                actual = bpy.data.objects[slot['object_name']].material_slots[slot['slot_index']].material
                shader = next(node for node in actual.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
                self.assertEqual(shader.inputs['Metallic'].default_value, 1)
        binding['selector']['role'] = 'metal'
        rejected = build_scene(bundle)['material_binding']
        self.assertEqual(rejected['status'], 'rejected')
        self.assertEqual(rejected['applied_slot_count'], 0)
        self.assertEqual(rejected['errors'][0]['code'], 'base_material_override_no_matches')

if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(BuiltinMaterialRoleTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
