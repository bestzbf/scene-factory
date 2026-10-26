"""Validate emitter placement using Blender, without rendering.

blender -b --python scene_factory/test_fixture_lighting.py
The normal Python suite skips this Blender-specific check.
"""
import math
from pathlib import Path
import sys
import unittest

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class FixtureLightingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from generate_scene import add_light, fixture_light_plan
        cls.add_light = staticmethod(add_light)
        cls.plan = staticmethod(fixture_light_plan)

    def spec(self, objects):
        return {'room': {'width': 6, 'depth': 8, 'height': 3},
                'lighting': {'energy': 1000, 'color': [1, .9, .8]},
                'objects': objects}

    def test_fixture_geometry_and_power(self):
        fixtures = [
            {'id': 'strip_1', 'kind': 'linear_light', 'position': [1, 2, 2.8],
             'dimensions': [2, .12, .08], 'rotation_z_deg': 90, 'confidence': .9},
            {'id': 'pendant_1', 'kind': 'pendant_light', 'position': [-1, 4, 2.5],
             'dimensions': [.4, .4, .3], 'rotation_z_deg': 20, 'confidence': .8}]
        plan = self.plan(self.spec(fixtures))
        self.assertFalse(plan['fallback_inferred'])
        self.assertFalse(plan['camera_fill'])
        strip, pendant = plan['emitters']
        self.assertEqual(len(plan['emitters']), 2)
        self.assertEqual(strip['shape'], 'RECTANGLE')
        self.assertEqual(pendant['shape'], 'DISK')
        self.assertEqual(strip['source_fixture_id'], 'strip_1')
        self.assertEqual(strip['rotation_z_deg'], 90)
        self.assertAlmostEqual(strip['position_m'][2], 2.797)
        self.assertAlmostEqual(pendant['position_m'][2], 2.497)
        self.assertAlmostEqual(sum(e['energy_w'] for e in plan['emitters']), 1000)
        self.assertAlmostEqual(strip['energy_w'] / pendant['energy_w'],
                               strip['emitter_area_m2'] / pendant['emitter_area_m2'])

    def test_blender_rectangle_follows_fixture_yaw_and_points_down(self):
        from mathutils import Vector
        obj = self.add_light('TestFixture', [1, 2, 2.797], [1, 2, 1.797],
                             100, [1, .9, .8], 1.94, 'RECTANGLE', .1116, 90)
        try:
            bpy.context.view_layer.update()
            frame = obj.matrix_world.to_3x3()
            forward = (frame @ Vector((0, 0, -1))).normalized()
            long_axis = (frame @ Vector((1, 0, 0))).normalized()
            self.assertLess((forward - Vector((0, 0, -1))).length, 1e-6)
            self.assertLess((long_axis - Vector((0, 1, 0))).length, 1e-6)
            self.assertEqual(obj.data.shape, 'RECTANGLE')
            self.assertAlmostEqual(obj.data.size, 1.94, places=5)
            self.assertAlmostEqual(obj.data.size_y, .1116, places=5)
        finally:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.lights.remove(data)

    def test_fallback_is_explicit_and_conserves_budget(self):
        plan = self.plan(self.spec([]))
        self.assertTrue(plan['fallback_inferred'])
        self.assertEqual(len(plan['emitters']), 4)
        self.assertFalse(plan['camera_fill'])
        for emitter in plan['emitters']:
            self.assertIsNone(emitter['source_fixture_id'])
            self.assertEqual(emitter['source'], 'unobserved_ceiling_grid_fallback')
            self.assertEqual(emitter['confidence'], 0)
            self.assertAlmostEqual(emitter['energy_w'], 250)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(FixtureLightingTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
