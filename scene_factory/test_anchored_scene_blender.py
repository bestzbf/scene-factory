"""Check placement decoding against actual built-in Blender asset bounds.

These generic fixtures explicitly declare contacts; they are not predictions
from photographs and do not measure reconstruction fidelity.
"""
import copy
from pathlib import Path
import sys
import unittest

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class AnchoredPlacementBlenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import anchored_scene
        import compact_scene
        import generate_multiview
        from test_multiview import fixture
        cls.codec = anchored_scene
        cls.compact = compact_scene
        cls.renderer = generate_multiview
        cls.fixture = staticmethod(fixture)

    def bundle(self):
        bundle = self.fixture()
        scene = bundle['shared_scene']
        scene['scene_type'] = scene['base_scene']['scene_type'] = 'office'
        for key in ('textures', 'materials', 'custom_objects', 'lights',
                    'base_material_overrides', 'exterior_contexts'):
            scene[key] = []
        scene['surface_overrides'] = dict.fromkeys(scene['surface_overrides'], '')
        first = bundle['views'][0]
        scene['image'] = {key: first[key] for key in ('width', 'height')}
        scene['image']['source_id'] = first['id']
        scene['camera'] = {'roll_degrees': first['camera']['roll_degrees']}
        scene['base_scene']['camera'] = {key: copy.deepcopy(first['camera'][key])
                                         for key in ('position', 'target', 'lens_mm')}
        return bundle

    def object(self, identifier, category, position, dimensions, **extra):
        return {'id': identifier, 'category': category, 'position': position,
                'dimensions': dimensions, 'rotation_z_deg': 0,
                'color': [.25, .3, .35], 'repeat': [1, 1],
                'spacing': [0, 0], 'confidence': .9, **extra}

    def wire(self, bundle, placements):
        wire = self.compact.compact_response(bundle)
        wire['wire_format'] = self.codec.FORMAT_VERSION
        for item in wire['shared_scene']['base_scene']['objects']:
            item['placement'] = copy.deepcopy(placements[item['id']])
            del item['position']
        return wire

    def build(self, raw):
        bundle = self.codec.expand_response(raw)
        handles = self.renderer.build_scene(bundle, appearance_policy='photo_reconstruction')
        bounds = {entry['id']: self.renderer.v2.evaluated_bounds(root)
                  for entry, root in zip(handles['base']['objects'], handles['roots'])}
        return bundle, handles, bounds

    def test_floor_declaration_removes_centre_height_ambiguity_without_moving_xy(self):
        bundle = self.bundle()
        bundle['shared_scene']['base_scene']['objects'] = [
            self.object('desk', 'desk', [0, 2, .72], [1.4, .75, .74]),
            self.object('chair', 'chair', [0, 1.2, .5], [.6, .6, .95])]
        before = copy.deepcopy(bundle)
        raw = self.wire(bundle, {
            'desk': {'mode': 'floor', 'xy': [0, 2]},
            'chair': {'mode': 'floor', 'xy': [0, 1.2]}})
        decoded, handles, bounds = self.build(raw)
        self.assertEqual(bundle, before)
        self.assertAlmostEqual(bounds['desk']['min'][2], 0, places=5)
        self.assertAlmostEqual(bounds['desk']['max'][2], .74, places=4)
        self.assertAlmostEqual(bounds['chair']['min'][2], 0, places=5)
        self.assertAlmostEqual(bounds['chair']['max'][2], .95, places=4)
        self.assertEqual(decoded['shared_scene']['base_scene']['objects'][0]['position'], [0, 2, 0])
        self.assertEqual(self.renderer.geometry_preflight(decoded, handles)['status'], 'passed')

    def test_on_top_monitor_contacts_actual_rotated_desk_top(self):
        bundle = self.bundle()
        bundle['shared_scene']['base_scene']['objects'] = [
            self.object('monitor', 'monitor', [0, 2, .74], [.55, .2, .45]),
            self.object('desk', 'desk', [0, 2, 0], [1.4, .75, .74], rotation_z_deg=37)]
        raw = self.wire(bundle, {
            'monitor': {'mode': 'on_top', 'xy': [0, 2], 'support_id': 'desk', 'clearance': .01},
            'desk': {'mode': 'floor', 'xy': [0, 2]}})
        _, _, bounds = self.build(raw)
        self.assertAlmostEqual(bounds['monitor']['min'][2] - bounds['desk']['max'][2], .01, places=4)
        self.assertAlmostEqual(bounds['monitor']['max'][2], 1.20, places=4)

    def test_exact_repeated_support_and_explicit_raised_cabinet(self):
        bundle = self.bundle()
        bundle['shared_scene']['base_scene']['objects'] = [
            self.object('desk', 'desk', [-.8, 2, 0], [1.2, .7, .74],
                        repeat=[2, 1], spacing=[1.6, 0]),
            self.object('monitor', 'monitor', [.8, 2, .74], [.5, .2, .4]),
            self.object('wall_cabinet', 'cabinet', [0, 4, 1.1], [1, .4, .8])]
        raw = self.wire(bundle, {
            'desk': {'mode': 'floor', 'xy': [-.8, 2]},
            'monitor': {'mode': 'on_top', 'xy': [.8, 2], 'support_id': 'desk__1_0', 'clearance': 0},
            'wall_cabinet': {'mode': 'world', 'position': [0, 4, 1.1]}})
        _, _, bounds = self.build(raw)
        self.assertAlmostEqual(bounds['monitor']['min'][2], bounds['desk__1_0']['max'][2], places=4)
        self.assertAlmostEqual(bounds['wall_cabinet']['min'][2], 1.1, places=4)
        self.assertAlmostEqual(bounds['wall_cabinet']['max'][2], 1.9, places=4)

    def test_world_mode_preserves_legacy_bottom_height(self):
        bundle = self.bundle()
        bundle['shared_scene']['base_scene']['objects'] = [
            self.object('raised_desk', 'desk', [0, 2, .72], [1.4, .75, .74])]
        raw = self.wire(bundle, {
            'raised_desk': {'mode': 'world', 'position': [0, 2, .72]}})
        _, _, bounds = self.build(raw)
        self.assertAlmostEqual(bounds['raised_desk']['min'][2], .72, places=4)
        self.assertAlmostEqual(bounds['raised_desk']['max'][2], 1.46, places=4)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AnchoredPlacementBlenderTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
