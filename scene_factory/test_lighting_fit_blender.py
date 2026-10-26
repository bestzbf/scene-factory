"""Actual Cycles power-fit tests: overbright input, multi-view fit and rollback."""
from pathlib import Path
import sys
import unittest

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class LightingFitBlenderTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_reference_calibration_blender import ReferenceCalibrationBlenderTests
        ReferenceCalibrationBlenderTests.setUpClass()
        self.fixture = ReferenceCalibrationBlenderTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.scene = self.fixture.scene
        self.fixture.plane(emission=False)
        data = bpy.data.lights.new('PowerFitLight', 'AREA')
        data.energy, data.size = 20., 2.
        light = bpy.data.objects.new('PowerFitLight', data)
        self.scene.collection.objects.link(light)
        light.location = (0, 0, 3)
        self.light = light
        self.camera = self.fixture.camera('PowerFitCamera', (0, -2, 2), (0, 0, 0))
        self.scene.camera = self.camera
        self.scene.cycles.samples = 16
        self.scene.view_settings.exposure = 0
        self.scene.render.resolution_percentage = 100
        _, self.reference = self.fixture.render_rgb('known_power_reference')
        self.light.data.energy = 1280.

    def tiny_view(self, scene, camera, view, max_edge):
        scene.camera = camera
        scene.render.resolution_x, scene.render.resolution_y = 24, 16

    def test_rendered_fit_improves_extreme_power_and_restores_transient_settings(self):
        scene = self.scene
        original = (scene.camera, scene.cycles.samples, scene.cycles.seed,
                    scene.render.resolution_x, scene.render.resolution_y,
                    scene.render.resolution_percentage, scene.render.filepath,
                    scene.cycles.use_denoising, scene.cycles.use_adaptive_sampling)
        geometry = {obj.as_pointer() for obj in scene.objects}
        from calibrate_reference import calibrate
        report = calibrate(scene, [{'id': 'a'}, {'id': 'b'}],
            {'a': self.camera, 'b': self.camera}, {'a': self.reference, 'b': self.reference}, self.tiny_view)
        fit = report['lighting_fit']
        self.assertEqual(fit['status'], 'selected')
        self.assertLess(fit['selected_spatial_loss'], fit['baseline_spatial_loss'] * .2)
        self.assertLess(self.light.data.energy, 1280.)
        self.assertGreater(self.light.data.energy, 0)
        self.assertLessEqual(fit['candidate_count'], fit['max_candidate_count'])
        self.assertEqual(fit['render_count'], fit['candidate_count'] * 2)
        self.assertEqual(original, (scene.camera, scene.cycles.samples, scene.cycles.seed,
                    scene.render.resolution_x, scene.render.resolution_y,
                    scene.render.resolution_percentage, scene.render.filepath,
                    scene.cycles.use_denoising, scene.cycles.use_adaptive_sampling))
        self.assertEqual(geometry, {obj.as_pointer() for obj in scene.objects})
        self.assertEqual(report['final_exposure'], scene.view_settings.exposure)
        self.assertTrue(fit['requires_independent_visual_review'])
        self.assertEqual(len(fit['reference_sha256']['a']), 64)

    def test_final_saved_render_diagnostics_records_exact_images(self):
        from calibrate_reference import final_render_diagnostics
        result = final_render_diagnostics({'a': self.reference}, {'a': self.reference})
        self.assertEqual(result['measurement_stage'], 'final_saved_render')
        self.assertEqual(len(result['render_sha256']['a']), 64)
        self.assertEqual(result['per_view']['a']['median_absolute_error_stops'], 0)

    def test_mid_search_exception_rolls_back_light_power_and_settings(self):
        from lighting_fit import fit_rendered_lighting
        from calibrate_reference import read_display_rgb
        from reference_tone import compute_spatial_stats
        rgb = read_display_rgb(self.reference)
        reference = {'a': compute_spatial_stats(rgb, bottom_up=True)}
        from appearance_lighting import add_fixture_faces
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 3.06))
        bpy.context.object.scale = (2., 2., .1)
        fixture_record, = add_fixture_faces(self.scene)
        face = bpy.data.objects[fixture_record['face']]
        strength = next(node for node in face.material_slots[0].material.node_tree.nodes
                        if node.type == 'EMISSION').inputs['Strength']
        bpy.context.view_layer.update()
        original_strength = strength.default_value
        original = (self.light.data.energy, self.scene.view_settings.exposure, self.scene.cycles.seed,
                    self.scene.cycles.samples, self.scene.render.resolution_percentage)
        calls = []
        def fail_second(scene, camera, view, max_edge):
            calls.append(1)
            self.tiny_view(scene, camera, view, max_edge)
            bpy.context.view_layer.update()
            if len(calls) > 1:
                self.assertLess(strength.default_value, original_strength)
                raise RuntimeError('failed candidate render')
        with self.assertRaisesRegex(RuntimeError, 'failed candidate'):
            fit_rendered_lighting(self.scene, [{'id': 'a'}], {'a': self.camera}, {'a': self.reference},
                fail_second, reference, -1.5, 0., read_display_rgb)
        self.assertEqual(original, (self.light.data.energy, self.scene.view_settings.exposure,
            self.scene.cycles.seed, self.scene.cycles.samples, self.scene.render.resolution_percentage))
        self.assertAlmostEqual(strength.default_value, original_strength, places=4)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LightingFitBlenderTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
