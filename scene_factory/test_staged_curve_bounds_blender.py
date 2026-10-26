"""Compare staged AUTO-Bezier bounds with evaluated Blender curve meshes."""
import math
from pathlib import Path
import random
import sys
import unittest

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class StagedCurveBoundsBlenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import staged_analysis
        cls.staged = staged_analysis

    def actual_bounds(self, points, part_position, part_rotation, group_position, group_rotation, radius):
        curve = bpy.data.curves.new('StagedCurveBoundsTest', 'CURVE')
        curve.dimensions = '3D'
        curve.resolution_u = 12
        curve.bevel_depth = radius
        curve.bevel_resolution = 3
        curve.use_fill_caps = True
        spline = curve.splines.new('BEZIER')
        spline.bezier_points.add(len(points) - 1)
        for point, coordinates in zip(spline.bezier_points, points):
            point.co = coordinates
            point.handle_left_type = point.handle_right_type = 'AUTO'
        parent = bpy.data.objects.new('StagedCurveBoundsParentTest', None)
        bpy.context.collection.objects.link(parent)
        parent.location = group_position
        parent.rotation_euler = [math.radians(value) for value in group_rotation]
        obj = bpy.data.objects.new('StagedCurveBoundsObjectTest', curve)
        bpy.context.collection.objects.link(obj)
        obj.parent = parent
        obj.location = part_position
        obj.rotation_euler = [math.radians(value) for value in part_rotation]
        try:
            bpy.context.view_layer.update()
            evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = evaluated.to_mesh()
            try:
                world = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
                return ([min(point[axis] for point in world) for axis in range(3)],
                        [max(point[axis] for point in world) for axis in range(3)])
            finally:
                evaluated.to_mesh_clear()
        finally:
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.objects.remove(parent, do_unlink=True)
            bpy.data.curves.remove(curve)

    def test_random_auto_curves_fit_analytic_bounds(self):
        randomizer = random.Random(20261024)
        for case in range(64):
            count = randomizer.randint(2, 7)
            points = [[randomizer.uniform(-2, 2) for _ in range(3)] for _ in range(count)]
            part_position = [randomizer.uniform(-.5, .5) for _ in range(3)]
            part_rotation = [randomizer.uniform(-35, 35) for _ in range(3)]
            group_position = [randomizer.uniform(-3, 3) for _ in range(3)]
            group_rotation = [randomizer.uniform(-35, 35) for _ in range(3)]
            radius = randomizer.uniform(.001, .1)
            part = {'type': 'curve', 'data': part_position + [0, 0, 0] + part_rotation + [radius],
                    'points': points, 'vertices': [], 'faces': [], 'cloth': []}
            group = {'p': group_position, 'r': group_rotation}
            expected_low, expected_high = self.staged._detail_part_world_bounds(part, group)
            actual_low, actual_high = self.actual_bounds(
                points, part_position, part_rotation, group_position, group_rotation, radius)
            with self.subTest(case=case):
                for axis in range(3):
                    self.assertLessEqual(expected_low[axis] - 2e-4, actual_low[axis])
                    self.assertGreaterEqual(expected_high[axis] + 2e-4, actual_high[axis])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(StagedCurveBoundsBlenderTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
