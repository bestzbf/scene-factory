"""Real evaluated Blender geometry and camera projection regression tests."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class PartProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate_v2
        import generate_multiview
        from spec_v2 import part
        from part_projection import custom_part_projections
        cls.v2, cls.multi = generate_v2, generate_multiview
        cls.part = staticmethod(part)
        cls.project = staticmethod(custom_part_projections)

    def setUp(self):
        bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
        self.v2.g.MATERIALS.clear(); self.v2.g.MESHES.clear()
        self.scene = bpy.context.scene
        self.scene.render.resolution_x = self.scene.render.resolution_y = 100
        self.scene.render.resolution_percentage = 100
        self.scene.render.pixel_aspect_x = self.scene.render.pixel_aspect_y = 1
        self.camera = self.multi.add_camera({'id': 'projection', 'source_image': 'unused.png',
            'camera': {'position': [0, 0, 1], 'target': [0, 4, 1], 'lens_mm': 50, 'roll_degrees': 0}})
        self.scene.camera = self.camera

    def group(self, identifier, position, parts=None, rotation=None):
        return {'id': identifier, 'position': position, 'rotation_degrees': rotation or [0, 0, 0],
                'material_id': '', 'observed': True, 'confidence': .8, 'support_id': '',
                'parts': parts or [self.part('box', size=[1, .2, 1])]}

    def report(self, groups):
        roots = [self.v2.build_custom(group, {}) for group in groups]
        before = {obj.as_pointer() for obj in self.scene.objects}
        report = self.project(self.scene, groups, roots, 'projection')
        self.assertEqual(before, {obj.as_pointer() for obj in self.scene.objects})
        self.assertFalse(report['occlusion_tested'])
        json.dumps(report, allow_nan=False)
        return report, roots

    def test_in_frame_off_frame_behind_and_partial_are_distinct(self):
        groups = [self.group('inside', [0, 4, 1]), self.group('outside', [4, 4, 1]),
                  self.group('behind', [0, -4, 1]), self.group('partial', [1.4, 4, 1])]
        report, roots = self.report(groups)
        records = {item['group_id']: item for item in report['parts']}
        self.assertEqual([records[g['id']]['part_id'] for g in groups], [g['id'] + '/0' for g in groups])
        self.assertEqual(records['inside']['status'], 'in_frame')
        self.assertEqual(records['inside']['projected_bbox_in_frame_fraction'], 1)
        self.assertTrue(records['inside']['intersects_frame_before_occlusion'])
        self.assertEqual(records['outside']['status'], 'outside_frame')
        self.assertEqual(records['outside']['frame_bbox_coverage_fraction'], 0)
        self.assertEqual(records['behind']['status'], 'behind_camera')
        self.assertEqual(records['behind']['front_vertex_fraction'], 0)
        self.assertIsNone(records['behind']['projected_bbox_normalized'])
        self.assertEqual(records['partial']['status'], 'partially_in_frame')
        self.assertGreater(records['partial']['projected_bbox_in_frame_fraction'], 0)
        self.assertLess(records['partial']['projected_bbox_in_frame_fraction'], 1)

    def test_group_and_part_rotation_use_evaluated_bounds_and_camera_roll(self):
        parts = [self.part('box', size=[1.4, .2, .35], position=[.5, 0, 0], rotation_degrees=[0, 15, 25])]
        group = self.group('rotated', [0, 4, 1], parts, [0, 0, 90])
        report, roots = self.report([group]); result = report['parts'][0]
        bounds = self.v2.evaluated_bounds(roots[0])
        for side in ('min', 'max'):
            for actual, expected in zip(result['world_aabb_m'][side], bounds[side]):
                self.assertAlmostEqual(actual, expected, places=6)
        bbox = result['projected_bbox_normalized']
        self.camera.rotation_euler.rotate_axis('Z', math.pi / 2)
        rolled = self.project(self.scene, [group], roots)['parts'][0]['projected_bbox_normalized']
        self.assertAlmostEqual(bbox[2] - bbox[0], rolled[3] - rolled[1], places=5)
        self.assertAlmostEqual(bbox[3] - bbox[1], rolled[2] - rolled[0], places=5)

    def test_cloth_thickness_and_curve_bevel_are_evaluated(self):
        groups = [self.group('cloth', [0, 4, 1], [self.part('curtain', size=[1, .2, 1.5])]),
                  self.group('wire', [0, 3, 1], [self.part('curve', radius=.08,
                      points=[[-.5, 0, 0], [.5, 0, 0]])])]
        report, _ = self.report(groups)
        cloth, wire = report['parts']
        self.assertEqual(cloth['part_type'], 'curtain')
        self.assertGreater(cloth['world_aabb_m']['max'][1] - cloth['world_aabb_m']['min'][1], .09)
        self.assertGreater(wire['world_aabb_m']['max'][2] - wire['world_aabb_m']['min'][2], .15)
        self.assertTrue(wire['intersects_frame_before_occlusion'])

    def test_triangle_clip_avoids_bbox_false_positive_and_handles_enclosing_polygon(self):
        # This upper-right triangle's bbox overlaps the image, but its diagonal
        # edge remains beyond the corner. Do not call it in view from AABB alone.
        triangle = self.part('mesh', vertices=[[0, 0, 4], [4, 0, 0], [4, 0, 4]], faces=[[0, 1, 2]])
        groups = [self.group('corner', [0, 4, 1], [triangle]),
                  self.group('enclosing', [0, 4, 1], [self.part('box', size=[10, .1, 10])])]
        report, _ = self.report(groups)
        corner, enclosing = report['parts']
        self.assertGreater(corner['frame_bbox_coverage_fraction'], 0)
        self.assertFalse(corner['intersects_frame_before_occlusion'])
        self.assertEqual(corner['status'], 'outside_frame')
        self.assertTrue(enclosing['intersects_frame_before_occlusion'])
        self.assertEqual(enclosing['frame_bbox_coverage_fraction'], 1)

    def test_near_plane_crossing_has_finite_box_and_clipped_depth(self):
        report, _ = self.report([self.group('near', [0, .03, 1], [self.part('box', size=[.2, .2, .2])])])
        result = report['parts'][0]
        self.assertGreater(result['front_vertex_fraction'], 0)
        self.assertLess(result['front_vertex_fraction'], 1)
        self.assertTrue(result['intersects_frame_before_occlusion'])
        self.assertEqual(result['status'], 'partially_in_frame')
        self.assertTrue(all(math.isfinite(x) for x in result['projected_bbox_normalized']))

    def test_sixteen_production_views_save_individual_diagnostics(self):
        from test_multiview import fixture
        bundle = fixture(); spec = bundle['shared_scene']
        spec['materials'] = []; spec['textures'] = []; bundle['texture_sources'] = []
        spec['surface_overrides'] = dict.fromkeys(spec['surface_overrides'], '')
        spec['custom_objects'] = [self.group('visible_box', [0, 3, 1])]
        entry = copy.deepcopy(spec['base_scene']['objects'][0])
        entry.update(id='fixture_cabinet', category='cabinet', position=[1, 3, 0],
                     dimensions=[.5, .5, 1], repeat=[1, 1], spacing=[0, 0])
        spec['base_scene']['objects'] = [entry]
        view = copy.deepcopy(bundle['views'][0])
        view.update(width=64, height=64)
        view['camera'].update(position=[0, 1, 1], target=[0, 3, 1], lens_mm=35, roll_degrees=0)
        bundle['views'] = [{**copy.deepcopy(view), 'id': 'view_' + str(i)} for i in range(16)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image = bpy.data.images.new('ProjectionSource', width=64, height=64)
            image.filepath_raw = str(directory / 'reference.png'); image.file_format = 'PNG'; image.save()
            for view in bundle['views']:
                view['source_image'] = image.filepath_raw
            candidate = directory / 'bundle.json'; candidate.write_text(json.dumps(bundle))
            output = directory / 'output'; previous = sys.argv
            try:
                sys.argv = ['blender', '--', '--spec', str(candidate), '--output-dir', str(output),
                            '--max-edge', '64', '--samples', '1', '--device', 'CPU', '--tone-match', 'off', '--no-render']
                self.multi.main()
            finally:
                sys.argv = previous
            names = set()
            for view in bundle['views']:
                annotation = json.loads((output / ('annotations_' + view['id'] + '.json')).read_text())
                report = annotation['custom_part_projection']
                self.assertEqual(report['view_id'], view['id'])
                self.assertEqual(report['part_count'], 1)
                self.assertEqual(report['parts'][0]['status'], 'in_frame')
                names.add(report['parts'][0]['object_name'])
            self.assertEqual(len(names), 1)
            metadata = json.loads((output / 'metadata.json').read_text())
            self.assertEqual(metadata['camera_count'], 16)
            self.assertIn('annotations_view_15.json', metadata['output_sha256'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PartProjectionTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
