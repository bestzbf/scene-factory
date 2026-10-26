"""Real Blender aperture diagnostics for unattended reconstruction."""
from pathlib import Path
import sys
import unittest

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class DaylightPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate_multiview
        from test_multiview import fixture
        cls.renderer = generate_multiview
        cls.fixture = staticmethod(fixture)

    def bundle(self):
        bundle = self.fixture()
        spec = bundle['shared_scene']
        spec['textures'] = []
        bundle['texture_sources'] = []
        spec['custom_objects'] = []
        base = spec['base_scene']
        depth = base['room']['depth']
        base['objects'] = [{**base['objects'][0], 'id': 'test_window', 'category': 'window',
            'position': [0, depth, .8], 'dimensions': [2, .1, 1.4], 'rotation_z_deg': 0,
            'repeat': [1, 1], 'spacing': [0, 0]}]
        bundle['architecture'] = {'omit_surfaces': ['back'], 'front_wall': False}
        from make_v2_examples import area
        spec['lights'] = [area('outdoor', [0, depth + .8, 1.6], [0, depth - 1, 1], 300)]
        spec['render']['replace_base_lights'] = True
        return bundle

    def build(self, bundle):
        return self.renderer.build_scene(bundle)

    def add_plane(self, bundle, textured=True, transmission=.5, size=(1.9, .002, 1.3), x=0):
        from make_v2_examples import group
        from spec_v2 import material, part
        depth = bundle['shared_scene']['base_scene']['room']['depth']
        bundle['shared_scene']['materials'].append(material('additional_pane', [.9, .9, .9], transmission=transmission))
        bundle['shared_scene']['custom_objects'].append(group('extra_glazing', [x, depth-.02, 1.5],
            [part(position=[0,0,0], size=list(size), material_id='additional_pane')], material_id='additional_pane'))
        handles = self.build(bundle)
        if textured:
            mat = handles['custom_roots'][0].children[0].material_slots[0].material
            node = mat.node_tree.nodes.new('ShaderNodeTexImage')
            node.image = bpy.data.images.new('test_local_photo_crop', width=2, height=2)
            shader = next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
            mat.node_tree.links.new(node.outputs['Color'], shader.inputs['Base Color'])
        bpy.context.view_layer.update()
        return handles

    def test_clear_open_window_and_frame_bars_pass(self):
        bundle = self.bundle()
        report = self.renderer.daylight_preflight(bundle, self.build(bundle))
        self.assertEqual(report['errors'], [])
        self.assertEqual(report['warnings'], [])
        self.assertEqual(len(report['windows']), 1)
        self.assertEqual(len(report['windows'][0]['associated_exterior_area_light_paths']), 1)

    def test_actual_opaque_retained_wall_behind_window_is_blocking(self):
        bundle = self.bundle()
        bundle['architecture']['omit_surfaces'] = []
        report = self.renderer.daylight_preflight(bundle, self.build(bundle))
        issue = next(i for i in report['errors'] if i['code']=='daylight_window_backed_by_opaque_wall')
        self.assertGreater(issue['coverage_fraction'], .9)
        self.assertEqual(issue['surface_ids'], ['back'])
        paths = issue['evidence']['associated_exterior_area_light_paths']
        self.assertTrue(any(count > 40 for count in paths[0]['retained_wall_hit_samples'].values()))

    def test_overlapping_textured_partial_transmission_plane_is_blocking(self):
        bundle = self.bundle()
        report = self.renderer.geometry_preflight(bundle, self.add_plane(bundle))
        self.assertEqual(report['status'], 'rejected')
        issue = next(i for i in report['errors'] if i['code']=='daylight_duplicate_textured_glazing')
        self.assertEqual(issue['overlapping_object_ids'], ['extra_glazing'])
        self.assertGreater(issue['coverage_fraction'], .9)

    def test_plain_clear_double_glazing_does_not_trigger_duplicate_fault(self):
        bundle = self.bundle()
        report = self.renderer.daylight_preflight(bundle, self.add_plane(bundle, textured=False, transmission=1))
        self.assertEqual(report['errors'], [])
        self.assertEqual(report['warnings'], [])

    def test_partial_edge_photo_plane_is_warning_not_blocking(self):
        bundle = self.bundle()
        report = self.renderer.daylight_preflight(bundle,
            self.add_plane(bundle, size=(.3,.002,1.3), x=-.7))
        self.assertEqual(report['errors'], [])
        self.assertEqual(report['warnings'][0]['code'], 'daylight_partial_textured_overlap')

    def test_unrelated_exterior_light_does_not_associate_with_window(self):
        bundle = self.bundle()
        light = bundle['shared_scene']['lights'][0]
        light['position'] = [10, 3, 1.6]
        light['target'] = [0, 3, 1.6]
        report = self.renderer.daylight_preflight(bundle, self.build(bundle))
        self.assertEqual(report['errors'], [])
        self.assertEqual(report['windows'][0]['associated_exterior_area_light_paths'], [])

    def test_rotated_side_window_uses_its_actual_aperture_frame(self):
        bundle = self.bundle()
        spec = bundle['shared_scene']
        spec['base_scene']['objects'][0].update(position=[2.5,3,.8], rotation_z_deg=-90)
        bundle['architecture']['omit_surfaces'] = ['right']
        spec['lights'][0].update(position=[3.3,3,1.6], target=[1.5,3,1.1])
        report = self.renderer.daylight_preflight(bundle, self.build(bundle))
        self.assertEqual(report['errors'], [])
        self.assertEqual(len(report['windows'][0]['associated_exterior_area_light_paths']), 1)
        bundle['architecture']['omit_surfaces'] = []
        report = self.renderer.daylight_preflight(bundle, self.build(bundle))
        issue = next(i for i in report['errors'] if i['code']=='daylight_window_backed_by_opaque_wall')
        self.assertEqual(issue['surface_ids'], ['right'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DaylightPreflightTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
