"""The plan's classification must reach actual Blender asset dispatch."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staged_analysis
from spec_v2 import SCENE_TYPES

try:
    import bpy
except ImportError:
    bpy = None


def fixture(scene_type):
    views = [{'id': 'a', 'source_image': '/fixture/unused.png', 'width': 640, 'height': 480}]
    plan = {
        'scene_type': scene_type, 'room': {'d': [5, 6, 3], 'c': [[.7, .7, .7]] * 3},
        'cameras': [{'id': 'a', 'd': [0, .5, 1.6, 0, 3, 1, 28, 0]}],
        'materials': [], 'assets': [{'id': 'seat', 'category': 'chair',
            'd': [0, 3, 0, .68, .68, 1.02, 0, .03, .03, .03, 1, 1, 0, 0, .9]}],
        'base_material_overrides': [], 'exterior_contexts': [],
        'lights': [{'id': 'light', 'type': 'AREA', 'shape': 'RECTANGLE',
                    'd': [0, 3, 2.8, 0, 3, 0, 1, 1, 1, 100, 1, 1, 120]}],
        'render': [0, .1, .5, .5, .5], 'surfaces': ['', '', '', '', ''],
        'omit': [], 'front_wall': False, 'textures': [], 'groups': [],
        'assumptions': ['Synthetic dispatch fixture, not a photo reconstruction.']}
    return plan, views


class SceneTypeTests(unittest.TestCase):
    def test_every_supported_class_is_retained_and_only_v1_shop_is_mapped(self):
        for scene_type in SCENE_TYPES:
            with self.subTest(scene_type=scene_type):
                plan, views = fixture(scene_type)
                before = copy.deepcopy(plan)
                bundle, _ = staged_analysis.to_bundle(plan, views, 'fixture', {})
                self.assertEqual(bundle['shared_scene']['scene_type'], scene_type)
                self.assertEqual(bundle['shared_scene']['base_scene']['scene_type'],
                                 'other' if scene_type == 'shop' else scene_type)
                self.assertEqual(plan, before)

    def test_missing_or_unsupported_class_fails_instead_of_guessing(self):
        for value in (None, 'Office', 'bedroom; execute text'):
            plan, views = fixture(value)
            if value is None:
                plan.pop('scene_type')
            with self.subTest(value=value), self.assertRaises(ValueError):
                staged_analysis.validate_plan(plan, views, 'fixture')


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class SceneTypeBlenderTests(unittest.TestCase):
    def test_native_classification_routes_office_and_bedroom_to_distinct_chairs(self):
        from generate_multiview import build_scene
        actual = {}
        for scene_type in ('office', 'bedroom', 'classroom'):
            plan, views = fixture(scene_type)
            bundle, _ = staged_analysis.to_bundle(plan, views, 'fixture', {})
            handles = build_scene(bundle, appearance_policy='photo_reconstruction')
            root = handles['roots'][0]
            actual[scene_type] = {'counts': dict(handles['counts']),
                                  'parts': len(root.children_recursive),
                                  'names': [child.name for child in root.children_recursive]}
        for scene_type in ('office', 'classroom'):
            self.assertEqual(actual[scene_type]['counts']['office_detailed'], 1)
            self.assertTrue(any('Caster' in name for name in actual[scene_type]['names']))
        self.assertEqual(actual['bedroom']['counts']['interior'], 1)
        self.assertFalse(any('Caster' in name for name in actual['bedroom']['names']))
        self.assertGreater(actual['office']['parts'], actual['bedroom']['parts'])


if __name__ == '__main__':
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(SceneTypeTests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(SceneTypeBlenderTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
