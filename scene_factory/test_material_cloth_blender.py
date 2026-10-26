"""Real Blender evidence for instance materials and bounded folded cloth."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class MaterialClothBlenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate_multiview
        from spec_v2 import part, material
        from test_multiview import fixture
        cls.renderer = generate_multiview
        cls.part = staticmethod(part)
        cls.material = staticmethod(material)
        cls.fixture = staticmethod(fixture)

    def scene_fixture(self):
        bundle = self.fixture()
        spec = bundle['shared_scene']
        source = copy.deepcopy(spec['base_scene']['objects'][0])
        source.update(id='cabinet', category='cabinet', position=[-1, 2, 0],
                      dimensions=[.6, .5, 1], repeat=[2, 1], spacing=[2, 0])
        spec['base_scene']['objects'] = [source]
        spec['materials'] = [self.material('replacement', [.5, .2, .08])]
        spec['textures'] = []
        spec['custom_objects'] = []
        spec['surface_overrides'] = dict.fromkeys(spec['surface_overrides'], '')
        spec['base_material_overrides'] = [{'object_id': 'cabinet__0_0', 'material_id': 'replacement',
            'selector': {'role': 'body', 'material_name': '', 'slot_index': -1}}]
        return bundle

    def test_partial_binding_applies_to_exact_instance_preserves_hardware(self):
        handles = self.renderer.build_scene(self.scene_fixture())
        report = handles['material_binding']
        self.assertEqual(report['status'], 'passed')
        self.assertGreater(report['applied_slot_count'], 0)
        self.assertTrue(any(slot['role'] == 'metal' for slot in report['available_slots']))
        for slot in report['available_slots']:
            actual = bpy.data.objects[slot['object_name']].material_slots[slot['slot_index']].material
            if slot['object_id'] == 'cabinet__0_0' and slot['role'] == 'body':
                self.assertEqual(actual['scene_material_id'], 'replacement')
                self.assertEqual(bpy.data.objects[slot['object_name']].material_slots[slot['slot_index']].link, 'OBJECT')
            else:
                self.assertEqual(actual.name, slot['source_material'])

    def test_semantic_monitor_screen_selector_preserves_casing_and_other_instance(self):
        bundle = self.scene_fixture()
        entry = bundle['shared_scene']['base_scene']['objects'][0]
        entry.update(id='display', category='monitor', dimensions=[.65, .23, .48])
        bundle['shared_scene']['base_material_overrides'] = [{
            'object_id': 'display__0_0', 'material_id': 'replacement',
            'selector': {'role': 'all', 'material_name': 'monitor_screen', 'slot_index': -1}}]
        handles = self.renderer.build_scene(bundle, appearance_policy='photo_reconstruction')
        report = handles['material_binding']
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['applied_slot_count'], 1)
        screens = [s for s in report['available_slots'] if s['source_material_id'] == 'monitor_screen']
        self.assertEqual(len(screens), 2)
        for slot in report['available_slots']:
            actual = bpy.data.objects[slot['object_name']].material_slots[slot['slot_index']].material
            if slot['object_id'] == 'display__0_0' and slot['source_material_id'] == 'monitor_screen':
                self.assertEqual(actual['scene_material_id'], 'replacement')
            else:
                self.assertEqual(actual.name, slot['source_material'])

    def test_unmatched_or_conflicting_selector_rejects_before_render(self):
        for conflict in (False, True):
            bundle = self.scene_fixture()
            spec = bundle['shared_scene']
            if conflict:
                spec['materials'].append(self.material('other', [.1, .1, .1]))
                second = copy.deepcopy(spec['base_material_overrides'][0]); second['material_id'] = 'other'
                spec['base_material_overrides'].append(second)
                code = 'base_material_override_conflict'
            else:
                spec['base_material_overrides'][0]['selector']['material_name'] = 'UnknownSlot'
                code = 'base_material_override_no_matches'
            handles = self.renderer.build_scene(bundle)
            self.assertEqual(handles['material_binding']['applied_slot_count'], 0)
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, 'preflight rejected'):
                    self.renderer.enforce_geometry_preflight(bundle, handles, temporary, spec_sha256='a' * 64)
                report = json.loads((Path(temporary) / 'geometry_preflight.json').read_text())
            self.assertTrue(any(error['code'] == code for error in report['errors']))
            self.assertTrue(report['material_binding']['available_slots'])

    def test_stable_source_name_selector_survives_rebuild(self):
        bundle = self.scene_fixture()
        handles = self.renderer.build_scene(bundle)
        source = next(slot for slot in handles['material_binding']['available_slots']
                      if slot['object_id'] == 'cabinet__0_0' and slot['role'] == 'body')
        bundle['shared_scene']['base_material_overrides'][0]['selector'].update(
            material_name=source['source_material_key'], slot_index=source['slot_index'])
        rebuilt = self.renderer.build_scene(bundle)
        self.assertEqual(rebuilt['material_binding']['status'], 'passed')
        self.assertGreater(rebuilt['material_binding']['applied_slot_count'], 0)

    def test_cloth_evaluated_mesh_has_folds_thickness_uv_and_controlled_bounds(self):
        bundle = self.scene_fixture()
        part = self.part('curtain', size=[1.2, .18, 2])
        bundle['shared_scene']['custom_objects'] = [{'id': 'hanging_textile',
            'position': [0, 3, 1.2], 'rotation_degrees': [0, 0, 0],
            'material_id': 'replacement', 'confidence': .8, 'observed': True,
            'support_id': '', 'parts': [part]}]
        handles = self.renderer.build_scene(bundle)
        root = handles['custom_roots'][0]
        obj = root.children[0]
        self.assertEqual(len(obj.data.vertices), 65 * 25)
        self.assertTrue(all(polygon.use_smooth for polygon in obj.data.polygons))
        self.assertEqual(obj.modifiers[0].type, 'SOLIDIFY')
        self.assertAlmostEqual(obj.modifiers[0].thickness, .002)
        self.assertEqual(obj.data.uv_layers.active.name, 'TextileUV')
        bounds = self.renderer.v2.evaluated_bounds(root)
        for axis, center in enumerate([0, 3, 1.2]):
            self.assertGreaterEqual(bounds['min'][axis], center - part['size'][axis] / 2 - 1e-6)
            self.assertLessEqual(bounds['max'][axis], center + part['size'][axis] / 2 + 1e-6)
        self.assertGreater(bounds['max'][1] - bounds['min'][1], .09)
        self.assertEqual(self.renderer.geometry_preflight(bundle, handles)['status'], 'passed')

    def test_production_cli_saves_binding_report_and_actual_cycles_render(self):
        bundle = self.scene_fixture()
        bundle['views'] = bundle['views'][:1]
        bundle['views'][0].update(width=64, height=64)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image = bpy.data.images.new('BindingProductionInput', width=64, height=64)
            image.generated_color = (.2, .2, .2, 1)
            image.filepath_raw = str(directory / 'reference.png')
            image.file_format = 'PNG'; image.save()
            bundle['views'][0]['source_image'] = str(directory / 'reference.png')
            candidate = directory / 'bundle.json'
            candidate.write_text(json.dumps(bundle))
            output = directory / 'output'
            previous = sys.argv
            try:
                sys.argv = ['blender', '--', '--spec', str(candidate), '--output-dir', str(output),
                            '--max-edge', '64', '--samples', '1', '--device', 'CPU', '--tone-match', 'off']
                self.renderer.main()
            finally:
                sys.argv = previous
            metadata = json.loads((output / 'metadata.json').read_text())
            report = json.loads((output / 'material_binding.json').read_text())
            self.assertEqual(metadata['status'], 'completed')
            self.assertEqual(report['status'], 'passed')
            self.assertGreater(report['applied_slot_count'], 0)
            self.assertIn('material_binding.json', metadata['output_sha256'])
            self.assertTrue((output / 'rgb.png').is_file())
            self.assertTrue((output / 'scene.blend').is_file())


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MaterialClothBlenderTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
