"""Focused Blender checks for rigid ``polyline`` custom parts.

Run directly with Blender, for example::

    blender -b --factory-startup --python-exit-code 1 \
      --python scene_factory/test_polyline_renderer_blender.py

The test intentionally exercises the renderer rather than the JSON validator;
the latter is covered by the normal Python test suite.
"""
import math
from pathlib import Path
import random
import sys
import unittest

try:
    import bpy
    from mathutils import Vector
except ImportError:  # pragma: no cover - imported only by Blender
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class PolylineRendererBlenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate_v2
        from spec_v2 import part
        cls.renderer = generate_v2
        import staged_analysis
        cls.staged = staged_analysis
        cls.part = staticmethod(part)

    def setUp(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

    @staticmethod
    def group(identifier, parts):
        return {
            'id': identifier,
            'position': [1.2, -0.7, 2.1],
            'rotation_degrees': [17, -11, 33],
            'material_id': '',
            'observed': True,
            'confidence': .95,
            'support_id': '',
            'parts': parts,
        }

    def test_polyline_is_rigid_and_evaluated_bounds_do_not_overshoot(self):
        # Door/frame-like U route: the top centerline is exactly z=1.48. A
        # Bezier AUTO spline can overshoot between these points; POLY cannot.
        points = [[-1.97, 0, .75], [-1.97, 0, 1.48],
                  [1.97, 0, 1.48], [1.97, 0, .75]]
        radius = .018
        rng = random.Random(20261026)
        part_rotation = [rng.uniform(-25, 25) for _ in range(3)]
        group_rotation = [rng.uniform(-35, 35) for _ in range(3)]
        item = self.part('polyline', radius=radius, points=points,
                         position=[-.31, .23, .12],
                         rotation_degrees=part_rotation)
        group = self.group('rigid_frame', [item])
        group['rotation_degrees'] = group_rotation
        root = self.renderer.build_custom(group, {})
        bpy.context.view_layer.update()

        obj = root.children[0]
        spline = obj.data.splines[0]
        self.assertEqual(spline.type, 'POLY')
        self.assertAlmostEqual(obj.data.bevel_depth, radius)
        self.assertTrue(obj.data.use_fill_caps)
        self.assertEqual(len(spline.points), len(points))
        for actual, expected in zip(spline.points, points):
            for coordinate, expected_coordinate in zip(actual.co[:3], expected):
                self.assertAlmostEqual(coordinate, expected_coordinate, places=6)

        # Evaluate the world-space centerline with both the part and group
        # transforms. Circular bevel geometry may extend by radius in any
        # direction, but must never exceed the transformed vertex envelope by
        # more than that radius (plus a small polygonal bevel tolerance).
        center_world = [obj.matrix_world @ Vector(point) for point in points]
        expected_max_z = max(point.z for point in center_world) + radius + .003
        expected_min_z = min(point.z for point in center_world) - radius - .003
        bounds = self.renderer.evaluated_bounds(root)
        self.assertLessEqual(bounds['max'][2], expected_max_z)
        self.assertGreaterEqual(bounds['min'][2], expected_min_z)

        detail_part = {'type': 'polyline',
                       'data': item['position'] + item['size'] + item['rotation_degrees'] + [radius],
                       'points': points, 'vertices': [], 'faces': [], 'cloth': []}
        staged_low, staged_high = self.staged._detail_part_world_bounds(
            detail_part, {'p': root.location[:],
                          'r': [math.degrees(angle) for angle in root.rotation_euler]})
        for axis in range(3):
            self.assertLessEqual(staged_low[axis] - 1e-5, bounds['min'][axis])
            self.assertGreaterEqual(staged_high[axis] + 1e-5, bounds['max'][axis])

        # A rigid route has a genuine corner at the top-left; its centerline
        # must include that vertex exactly rather than an interpolated arc.
        self.assertTrue(any((obj.matrix_world @ Vector(point) -
                             obj.matrix_world @ Vector(points[1])).length < 1e-6
                            for point in points))

    def test_legacy_curve_remains_bezier_auto(self):
        item = self.part('curve', radius=.025,
                         points=[[-.4, 0, 0], [0, 0, .5], [.4, 0, 0]])
        root = self.renderer.build_custom(self.group('legacy_smooth_wire', [item]), {})
        spline = root.children[0].data.splines[0]
        self.assertEqual(spline.type, 'BEZIER')
        self.assertTrue(all(point.handle_left_type == 'AUTO'
                            and point.handle_right_type == 'AUTO'
                            for point in spline.bezier_points))


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PolylineRendererBlenderTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
