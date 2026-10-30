"""Pre-detail spatial regression tests; no model service or Blender required."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import staged_analysis as staged
from batch import atomic_json, file_sha256, read_json
from model_api import public_config
from test_staged_analysis import fixture_plan, fixture_detail

try:
    import bpy
except ImportError:
    bpy = None


class PlanGeometryTests(unittest.TestCase):
    def setUp(self):
        self.views = [{'id': 'view_0', 'source_image': '/unused/reference.png', 'width': 64, 'height': 48}]
        self.plan = fixture_plan(self.views, group_count=1)
        self.plan['front_wall'] = False
        self.plan['omit'] = []

    def validate(self, plan=None):
        return staged.validate_plan(plan or self.plan, self.views, 'fixture_scene')

    def test_real_trial_offset_regression_rejected_without_mutation(self):
        plan = copy.deepcopy(self.plan)
        plan['room']['d'] = [3.4, 5.4, 2.8]
        plan['cameras'][0]['d'] = [1.7, .7, 1.5, 1.55, 4.6, 1.15, 20, 0]
        plan['assets'][0]['d'] = [1.55, 4.85, 0, 1.3, .5, .76, 0, .3, .2, .1, 1, 1, 0, 0, .8]
        plan['groups'][0]['p'] = [2.78, 2.15, .9]
        plan['groups'][0]['d'] = [1.1, 2.3, 1.95]
        original = copy.deepcopy(plan)
        with self.assertRaises(staged.PlanGeometryError) as caught:
            self.validate(plan)
        message = staged._feedback(caught.exception)
        self.assertIn('cameras[0]', message)
        self.assertIn('retained right surface', message)
        self.assertIn('groups[0]', message)
        self.assertIn('assets[0]', message)
        self.assertIn('X=[-1.700,1.700]', message)
        self.assertIn('camera/light targets', message)
        self.assertEqual(plan, original)

    def test_open_front_camera_and_exterior_lights_valid(self):
        self.plan['cameras'][0]['d'][:3] = [0, -1, 1.6]
        self.plan['lights'][0]['d'][:6] = [20, 25, 9, 0, 3, 1]
        self.validate()

    def test_exterior_camera_cannot_look_through_retained_wall(self):
        positions = {'left': [-4, 3, 1.6], 'right': [4, 3, 1.6],
                     'front': [0, -1, 1.6], 'back': [0, 9, 1.6],
                     'floor': [0, 3, -1], 'ceiling': [0, 3, 4]}
        for face, position in positions.items():
            with self.subTest(face=face):
                plan = copy.deepcopy(self.plan)
                plan['front_wall'] = True
                plan['cameras'][0]['d'][:6] = position + [0, 3, 1.5]
                with self.assertRaisesRegex(staged.PlanGeometryError, 'first enters through retained ' + face):
                    self.validate(plan)

    def test_exterior_camera_may_enter_through_any_omitted_face(self):
        positions = {'left': [-4, 3, 1.6], 'right': [4, 3, 1.6],
                     'back': [0, 9, 1.6], 'floor': [0, 3, -1], 'ceiling': [0, 3, 4]}
        for face, position in positions.items():
            with self.subTest(face=face):
                plan = copy.deepcopy(self.plan)
                plan['omit'] = [face]
                plan['cameras'][0]['d'][:6] = position + [0, 3, 1.5]
                self.validate(plan)

    def test_oblique_exterior_camera_uses_first_entry_face(self):
        # Both X and Y begin outside; the ray reaches open front after passing
        # the right-wall X plane outside the wall's finite depth.
        self.plan['cameras'][0]['d'][:6] = [4, -3, 1.6, 0, 3, 1.5]
        self.validate()
        # A closer Y starting position instead enters through the closed right.
        self.plan['cameras'][0]['d'][:3] = [4, -1, 1.6]
        with self.assertRaisesRegex(staged.PlanGeometryError, 'first enters through retained right'):
            self.validate()

    def test_open_front_camera_must_face_room(self):
        self.plan['cameras'][0]['d'][:6] = [0, -1, 1.6, 0, -2, 1.6]
        with self.assertRaisesRegex(staged.PlanGeometryError, 'does not enter room volume'):
            self.validate()

    def test_exterior_parallel_and_grazing_rays_rejected(self):
        w = self.plan['room']['d'][0]
        for position, target in (([4, -1, 1.6], [4, 3, 1.6]),
                                 ([w / 2, -1, 1.6], [w / 2, 3, 1.6]),
                                 ([4, -1, 4], [0, 3, 4])):
            with self.subTest(position=position):
                self.plan['cameras'][0]['d'][:6] = position + target
                with self.assertRaisesRegex(staged.PlanGeometryError, 'does not enter room volume'):
                    self.validate()

    def test_view_target_is_direction_not_segment_endpoint(self):
        # Target remains outside the room, but the same forward ray enters it.
        self.plan['cameras'][0]['d'][:6] = [0, -2, 1.6, 0, -1, 1.6]
        self.validate()

    def test_corner_entry_requires_both_faces_open(self):
        w = self.plan['room']['d'][0]
        # Equal X/Y delta reaches the front-right corner simultaneously.
        self.plan['cameras'][0]['d'][:6] = [w / 2 + 1, -1, 1.6, w / 2 - 1, 1, 1.6]
        with self.assertRaisesRegex(staged.PlanGeometryError, 'first enters through retained right'):
            self.validate()
        self.plan['omit'] = ['right']
        self.validate()

    def test_camera_contacts_each_retained_surface(self):
        w, d, h = self.plan['room']['d']
        positions = {'left': [-w / 2, 3, 1.6], 'right': [w / 2 + .08, 3, 1.6],
                     'back': [0, d, 1.6], 'front': [0, -.08, 1.6],
                     'floor': [0, 3, .025], 'ceiling': [0, 3, h - .025]}
        for surface, position in positions.items():
            with self.subTest(surface=surface):
                plan = copy.deepcopy(self.plan)
                plan['front_wall'] = True
                plan['cameras'][0]['d'][:3] = position
                with self.assertRaisesRegex(staged.PlanGeometryError, 'retained ' + surface):
                    self.validate(plan)

    def test_omitted_surface_does_not_reject_camera_at_opening(self):
        w, d, h = self.plan['room']['d']
        positions = {'left': [-w / 2, 3, 1.6], 'right': [w / 2, 3, 1.6],
                     'back': [0, d, 1.6], 'floor': [0, 3, 0], 'ceiling': [0, 3, h]}
        for surface, position in positions.items():
            with self.subTest(surface=surface):
                plan = copy.deepcopy(self.plan)
                plan['omit'] = [surface]
                plan['cameras'][0]['d'][:3] = position
                self.validate(plan)

    def test_rotated_base_asset_extents_checked(self):
        # At 0 degrees this footprint fits; at 90 its long side crosses right wall.
        values = self.plan['assets'][0]['d']
        values[:7] = [2.4, 3, 0, .5, 1.6, .75, 0]
        self.validate()
        values[6] = 90
        with self.assertRaisesRegex(staged.PlanGeometryError, 'rotated bounds outside room'):
            self.validate()

    def test_repeated_instances_checked_not_only_first(self):
        values = self.plan['assets'][0]['d']
        values[:7] = [0, 3, 0, .8, .6, .75, 0]
        values[10:14] = [3, 1, 1.5, 0]
        with self.assertRaisesRegex(staged.PlanGeometryError, r'instance\[2,0\]'):
            self.validate()

    def test_negative_repeat_spacing_checked(self):
        values = self.plan['assets'][0]['d']
        values[:7] = [0, 3, 0, .8, .6, .75, 0]
        values[10:14] = [3, 1, -1.5, 0]
        with self.assertRaisesRegex(staged.PlanGeometryError, r'instance\[2,0\]'):
            self.validate()

    def test_architecture_may_occupy_shell_thickness(self):
        asset = self.plan['assets'][0]
        asset['category'] = 'window'
        asset['d'][:7] = [0, 7.9, .9, 2, .15, 1.4, 0]
        self.validate()
        asset['category'] = 'desk'
        with self.assertRaises(staged.PlanGeometryError):
            self.validate()

    def test_group_origin_does_not_imply_geometric_centre(self):
        # d is advisory: a wall anchored at its lower edge is legitimate.
        group = self.plan['groups'][0]
        group['p'] = [-2.85, 7.9, 0]
        group['r'] = [45, 0, 90]
        group['d'] = [5.7, .15, 3.1]
        self.validate()

    def test_group_anchor_outside_room_rejected_even_if_wall_omitted(self):
        self.plan['omit'] = ['right']
        self.plan['groups'][0]['p'] = [4, 3, 0]
        with self.assertRaisesRegex(staged.PlanGeometryError, 'origin outside room envelope'):
            self.validate()

    def test_custom_part_bounds_are_checked_before_bundle_assembly(self):
        group = self.plan['groups'][0]
        detail = fixture_detail(group['id'])
        # The group's origin is valid, but the rotated/local part crosses the
        # right room boundary.  This is the failure mode that previously made
        # Blender render an object outside the room before preflight noticed.
        group['p'] = [2.75, 3, 0]
        detail['parts'][0]['data'] = [0, 0, .5, 1.0, .8, 1.0, 0, 0, 0, 0]
        with self.assertRaisesRegex(staged.PlanGeometryError, r'groups\[0\]\.parts\[0\]'):
            staged.validate_detail(detail, group, self.plan, self.views, 'fixture_scene')

    def test_custom_part_below_floor_is_rejected(self):
        group = self.plan['groups'][0]
        detail = fixture_detail(group['id'])
        detail['parts'][0]['data'] = [0, 0, -.3, .6, .6, .5, 0, 0, 0, 0]
        with self.assertRaisesRegex(staged.PlanGeometryError, 'world bounds'):
            staged.validate_detail(detail, group, self.plan, self.views, 'fixture_scene')

    def test_custom_architecture_part_may_use_shell_tolerance(self):
        group = self.plan['groups'][0]
        group['p'] = [-2.79, 3, 0]
        group['r'] = [0, 0, 0]
        group['instructions'] = 'Observed wall panel close to the window opening.'
        detail = fixture_detail(group['id'])
        detail['parts'][0]['data'] = [0, 0, 1.55, .15, 1.0, 3.1, 0, 0, 0, 0]
        # A wall-thickness-sized part may touch the shell tolerance, while it
        # still must remain inside the declared room envelope plus 0.15m.
        staged.validate_detail(detail, group, self.plan, self.views, 'fixture_scene')

    def test_curve_auto_handle_overshoot_is_caught_before_detail_acceptance(self):
        group = self.plan['groups'][0]
        group['p'] = [0, 2, .8]
        group['r'] = [0, 0, 0]
        detail = fixture_detail(group['id'], 'curve')
        detail['parts'][0]['points'] = [[1, 2, 1], [1, 2, 2], [-1, -1, 2]]
        detail['parts'][0]['data'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, .01]

        # Blender's AUTO handles on this short, uneven spline reach above 3.0m,
        # although the control points themselves end at 2.0m. The local gate
        # must reject the transformed curve before another detail group is sent.
        low, high = staged._detail_part_world_bounds(detail['parts'][0], group)
        self.assertGreater(high[2], 3.1)
        with self.assertRaisesRegex(staged.PlanGeometryError, r'groups\[0\]\.parts\[0\]'):
            staged.validate_detail(detail, group, self.plan, self.views, 'fixture_scene')

    def test_two_point_auto_curve_remains_a_tight_straight_bound(self):
        group = self.plan['groups'][0]
        group['p'] = [0, 2, 0]
        group['r'] = [0, 0, 0]
        detail = fixture_detail(group['id'], 'curve')
        detail['parts'][0]['points'] = [[-.5, 0, .8], [.5, 0, .8]]
        detail['parts'][0]['data'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, .02]
        low, high = staged._detail_part_world_bounds(detail['parts'][0], group)
        for axis, expected in enumerate(([-.52, .52], [1.98, 2.02], [.78, .82])):
            self.assertAlmostEqual(low[axis], expected[0], places=6)
            self.assertAlmostEqual(high[axis], expected[1], places=6)

    def test_transformed_rigid_polyline_bounds_gate_room_placement(self):
        group = self.plan['groups'][0]
        group['p'] = [0, 2, .2]
        group['r'] = [0, 0, 30]
        detail = fixture_detail(group['id'], 'polyline')
        detail['parts'][0]['points'] = [[-.5, 0, .8], [-.5, 0, 1.4],
                                        [.5, 0, 1.4], [.5, 0, .8]]
        detail['parts'][0]['data'] = [.12, -.03, 0, 0, 0, 0, 9, -14, 21, .018]
        low, high = staged._detail_part_world_bounds(detail['parts'][0], group)
        self.assertLessEqual(low[0], high[0])
        self.assertLessEqual(low[1], high[1])
        self.assertLessEqual(low[2], high[2])
        staged.validate_detail(detail, group, self.plan, self.views, 'fixture_scene')

        # The transformed right rail plus its radius protrudes beyond the
        # room's right wall. The staged validator must catch the actual local
        # polyline bounds before issuing or accepting a detail result.
        group['p'] = [2.78, 2, .2]
        with self.assertRaisesRegex(staged.PlanGeometryError, r'groups\[0\]\.parts\[0\]'):
            staged.validate_detail(detail, group, self.plan, self.views, 'fixture_scene')

    @unittest.skipUnless(bpy is not None, 'Requires Blender Python')
    def test_auto_curve_bound_contains_blender_evaluated_mesh(self):
        group = {'p': [.2, 2, .4], 'r': [13, -8, 24]}
        detail = fixture_detail('curve_bound', 'curve')
        part = detail['parts'][0]
        part['points'] = [[.258252, -.048788, .249822], [1.301288, -1.300675, -1.745683],
                          [-.510683, -1.063499, 1.958901], [1.940362, .150766, .286302],
                          [-1.49593, -1.317154, -.910403], [-.31232, 1.078145, -.65761]]
        part['data'] = [.03, -.02, .1, 0, 0, 0, 7, 11, -3, .01]
        expected_low, expected_high = staged._detail_part_world_bounds(part, group)

        curve = bpy.data.curves.new('AutoCurveBoundsTest', 'CURVE')
        curve.dimensions = '3D'
        curve.resolution_u = 12
        curve.bevel_depth = part['data'][9]
        curve.bevel_resolution = 3
        curve.use_fill_caps = True
        spline = curve.splines.new('BEZIER')
        spline.bezier_points.add(len(part['points']) - 1)
        for point, coordinates in zip(spline.bezier_points, part['points']):
            point.co = coordinates
            point.handle_left_type = point.handle_right_type = 'AUTO'
        parent = bpy.data.objects.new('AutoCurveBoundsParentTest', None)
        bpy.context.collection.objects.link(parent)
        parent.location = group['p']
        parent.rotation_euler = [math.radians(value) for value in group['r']]
        obj = bpy.data.objects.new('AutoCurveBoundsObjectTest', curve)
        bpy.context.collection.objects.link(obj)
        obj.parent = parent
        obj.location = part['data'][:3]
        obj.rotation_euler = [math.radians(value) for value in part['data'][6:9]]
        try:
            bpy.context.view_layer.update()
            evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = evaluated.to_mesh()
            try:
                world = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
                actual_low = [min(point[axis] for point in world) for axis in range(3)]
                actual_high = [max(point[axis] for point in world) for axis in range(3)]
            finally:
                evaluated.to_mesh_clear()
            for axis in range(3):
                self.assertLessEqual(expected_low[axis] - 1e-5, actual_low[axis])
                self.assertGreaterEqual(expected_high[axis] + 1e-5, actual_high[axis])
        finally:
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.objects.remove(parent, do_unlink=True)
            bpy.data.curves.remove(curve)

    def test_spatial_feedback_contains_indices_not_untrusted_labels(self):
        self.plan['groups'][0]['id'] = 'UNTRUSTED_LABEL'
        self.plan['groups'][0]['p'] = [4, 3, 0]
        with self.assertRaises(staged.PlanGeometryError) as caught:
            self.validate()
        feedback = staged._feedback(caught.exception)
        self.assertIn('groups[0]', feedback)
        self.assertNotIn('UNTRUSTED_LABEL', feedback)

    def test_retry_stops_bad_plan_before_any_detail_request(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            photo = root / 'photo.png'
            Image.new('RGB', (64, 48), (100, 90, 80)).save(photo)
            views = [{**self.views[0], 'source_image': str(photo)}]
            config = {'provider': 'openai_chat', 'model': 'fixture',
                      'base_url': 'http://127.0.0.1:1/v1', 'api_key_env': '',
                      'response_format': 'json_schema', 'max_retries': 0}
            calls = []
            corrected = copy.deepcopy(self.plan)
            invalid = copy.deepcopy(corrected)
            invalid['cameras'][0]['d'][0] = invalid['room']['d'][0] / 2
            invalid['cameras'][0]['d'][1] = 1

            def model(prompt, images, schema, output, timeout, effort, *, config):
                stage = schema['properties'].get('group_id', {}).get('enum', ['plan'])[0]
                calls.append(stage)
                value = (copy.deepcopy(invalid if calls.count('plan') == 1 else corrected)
                         if stage == 'plan' else fixture_detail(stage))
                if stage == 'plan' and 'groups' in schema['properties']:
                    group_properties = schema['properties']['groups']['items']['properties']
                    if 'evidence_view_ids' not in group_properties:
                        for group in value.get('groups', []):
                            for key in ('evidence_view_ids', 'evidence_confidence',
                                        'texture_material_ids', 'texture_material_confidence'):
                                group.pop(key, None)
                atomic_json(Path(output).with_suffix('.raw.json'), value)
                atomic_json(Path(output).with_suffix('.api.json'), {
                    'status': 'completed', 'config': public_config(config),
                    'prompt_sha256': staged.hashlib.sha256(prompt.encode()).hexdigest(),
                    'schema_sha256': staged._hash(schema),
                    'image_sha256': [file_sha256(path) for path in images]})
                return value

            output = root / 'bundle.json'
            with patch.object(staged, 'call_model', side_effect=model):
                result = staged.analyze('fixture_scene', views, output, model_config=config,
                                        options={'validation_retries': 1})
            self.assertEqual(calls, ['plan', 'plan', 'group_0'])
            self.assertEqual(result['views'][0]['camera']['position'][0], -.3)
            states = output.with_name('bundle_stages')
            self.assertEqual(read_json(states / 'plan_attempt_00.validation.json')['status'], 'failed')
            retry_prompt = (states / 'plan_attempt_01.prompt.txt').read_text()
            self.assertIn('retained right surface', retry_prompt)
            self.assertIn('Correct the complete native JSON', retry_prompt)

    def test_exhausted_geometry_retries_never_request_detail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            photo = root / 'photo.png'
            Image.new('RGB', (64, 48)).save(photo)
            views = [{**self.views[0], 'source_image': str(photo)}]
            invalid = copy.deepcopy(self.plan)
            invalid['groups'][0]['p'] = [4, 3, 0]
            config = {'provider': 'openai_chat', 'model': 'fixture',
                      'base_url': 'http://127.0.0.1:1/v1', 'api_key_env': ''}

            def invalid_plan(prompt, images, schema, output, timeout, effort, *, config):
                value = copy.deepcopy(invalid)
                if 'groups' in schema['properties']:
                    group_properties = schema['properties']['groups']['items']['properties']
                    if 'evidence_view_ids' not in group_properties:
                        for group in value.get('groups', []):
                            for key in ('evidence_view_ids', 'evidence_confidence',
                                        'texture_material_ids', 'texture_material_confidence'):
                                group.pop(key, None)
                atomic_json(Path(output).with_suffix('.raw.json'), value)
                atomic_json(Path(output).with_suffix('.api.json'), {
                    'status': 'completed', 'config': public_config(config),
                    'prompt_sha256': staged.hashlib.sha256(prompt.encode()).hexdigest(),
                    'schema_sha256': staged._hash(schema),
                    'image_sha256': [file_sha256(path) for path in images]})
                return value

            with patch.object(staged, 'call_model', side_effect=invalid_plan) as model:
                with self.assertRaisesRegex(staged.StagedAnalysisError, 'exhausted validation attempts'):
                    staged.analyze('fixture_scene', views, root / 'bundle.json', model_config=config,
                                   options={'validation_retries': 1})
            self.assertEqual(model.call_count, 2)
            self.assertTrue(all('room' in call.args[2]['properties'] for call in model.call_args_list))
            self.assertFalse((root / 'bundle.json').exists())


if __name__ == '__main__':
    unittest.main()
