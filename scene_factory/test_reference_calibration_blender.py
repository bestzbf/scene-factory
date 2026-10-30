"""Tiny Blender integration tests for display metering and fixture appearance."""
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib
from types import SimpleNamespace
from unittest.mock import patch

try:
    import bpy
    import numpy as np
    from mathutils import Vector
except ImportError:
    bpy = None


def write_rgb_png(path, rgb, width=3, height=2):
    """Write literal sRGB-encoded 8-bit pixels, with no external image library."""
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind + payload) & 0xffffffff)
    row = bytes([0]) + bytes(rgb) * width
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' +
        chunk(b'IHDR', struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0)) +
        chunk(b'sRGB', b'\x00') + chunk(b'IDAT', zlib.compress(row * height)) + chunk(b'IEND', b''))


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class ReferenceCalibrationBlenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import calibrate_reference
        import appearance_lighting
        cls.calibration = calibrate_reference
        cls.fixtures = appearance_lighting

    def setUp(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.scene = bpy.context.scene
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.device = 'CPU'
        self.scene.cycles.samples = 4
        self.scene.cycles.use_adaptive_sampling = False
        self.scene.cycles.use_denoising = False
        self.scene.cycles.seed = 41
        self.scene.cycles.use_animated_seed = False
        self.scene.render.use_border = False
        self.scene.render.resolution_x = 24
        self.scene.render.resolution_y = 16
        self.scene.render.resolution_percentage = 100
        self.scene.render.image_settings.file_format = 'PNG'
        self.scene.render.image_settings.color_mode = 'RGB'
        self.scene.render.image_settings.color_depth = '8'
        self.scene.render.film_transparent = False
        self.scene.view_settings.view_transform = 'Standard'
        self.scene.view_settings.look = 'None'
        self.scene.view_settings.exposure = .25
        self.scene.view_settings.gamma = 1
        self.scene.world = bpy.data.worlds.new('CalibrationTestWorld')
        self.scene.world.use_nodes = True
        background = next(node for node in self.scene.world.node_tree.nodes if node.type == 'BACKGROUND')
        background.inputs['Strength'].default_value = 0

    def camera(self, name, location, target):
        data = bpy.data.cameras.new(name)
        data.type = 'ORTHO'
        data.ortho_scale = 2
        obj = bpy.data.objects.new(name, data)
        self.scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()
        return obj

    def plane(self, emission=False):
        bpy.ops.mesh.primitive_plane_add(size=20)
        plane = bpy.context.object
        mat = bpy.data.materials.new('CalibrationTarget')
        mat.use_nodes = True
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        shader = nodes.new('ShaderNodeEmission' if emission else 'ShaderNodeBsdfDiffuse')
        shader.inputs['Color'].default_value = (.2, .2, .2, 1)
        if emission:
            shader.inputs['Strength'].default_value = 1
        links.new(shader.outputs[0], output.inputs['Surface'])
        plane.data.materials.append(mat)
        return plane

    def render_rgb(self, name):
        bpy.ops.render.render()
        path = self.directory / (name + '.png')
        bpy.data.images['Render Result'].save_render(str(path), scene=self.scene)
        return self.calibration.read_display_rgb(path), path

    def preview_settings(self):
        return (self.scene.cycles.seed, self.scene.cycles.use_animated_seed,
                self.scene.cycles.use_adaptive_sampling, self.scene.cycles.use_denoising,
                self.scene.render.resolution_percentage, self.scene.render.use_border)

    def test_exposure_preview_overrides_view_settings_and_restores_them(self):
        self.plane(emission=True)
        camera = self.camera('DeterministicCamera', (0, 0, 2), (0, 0, 0))
        self.scene.camera = camera
        _, reference_path = self.render_rgb('deterministic_reference')
        self.scene.cycles.seed = 97
        self.scene.cycles.use_animated_seed = True
        self.scene.cycles.use_adaptive_sampling = True
        self.scene.render.resolution_percentage = 50
        self.scene.render.use_border = True
        original = self.preview_settings()
        calls = []
        render = bpy.ops.render.render
        def render_checked(*args, **kwargs):
            calls.append(self.preview_settings())
            self.assertEqual(calls[-1], (41, False, False, True, 100, False))
            return render(*args, **kwargs)
        def tiny_view(scene, selected_camera, view, max_edge):
            scene.camera = selected_camera
            scene.render.resolution_x, scene.render.resolution_y = 24, 16
            scene.cycles.seed = 123
            scene.cycles.use_animated_seed = True
            scene.render.use_border = True
            scene.render.resolution_percentage = 25
        # bpy.ops dynamically constructs operator wrappers on attribute access;
        # patch the module binding, keeping actual Blender data and rendering.
        checked_bpy = SimpleNamespace(data=bpy.data,
            ops=SimpleNamespace(render=SimpleNamespace(render=render_checked)))
        with patch('calibrate_reference.bpy', checked_bpy):
            first = self.calibration.calibrate(self.scene, [{'id': 'a'}], {'a': camera},
                {'a': reference_path}, tiny_view, fit_lighting=False)
            self.assertEqual(self.preview_settings(), original)
            second = self.calibration.calibrate(self.scene, [{'id': 'a'}], {'a': camera},
                {'a': reference_path}, tiny_view, fit_lighting=False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(first['correction_ev'], 0)
        self.assertEqual(first['selected_preview_stats'], second['selected_preview_stats'])
        self.assertEqual(self.preview_settings(), original)

    def test_encoded_rgb_is_not_double_decoded_and_image_is_released(self):
        path = self.directory / 'literal.png'
        write_rgb_png(path, (128, 64, 192))
        before = set(bpy.data.images)
        values = self.calibration.read_display_rgb(path)
        self.assertEqual(before, set(bpy.data.images))
        np.testing.assert_allclose(values, np.broadcast_to(np.array([128, 64, 192]) / 255, (2, 3, 3)), atol=1e-7)
        # This must be ~.216 linear luminance for a gray 128 reference, not the
        # ~.038 result from mistakenly decoding already scene-linear pixels.
        write_rgb_png(path, (128, 128, 128))
        stats = self.calibration.compute_stats(self.calibration.read_display_rgb(path))
        self.assertAlmostEqual(stats['quantiles']['p50'], .2158605, places=6)

    def test_shared_calibration_selects_known_offset_and_restores_settings(self):
        self.plane(emission=True)
        cameras = {'a': self.camera('MeterA', (0, 0, 2), (0, 0, 0)),
                   'b': self.camera('MeterB', (.3, 0, 2), (.3, 0, 0))}
        self.scene.camera = cameras['a']
        initial_exposure = self.scene.view_settings.exposure
        self.scene.view_settings.exposure = initial_exposure - 1
        _, reference_path = self.render_rgb('reference')
        self.scene.view_settings.exposure = initial_exposure
        self.scene.render.resolution_x, self.scene.render.resolution_y = 27, 19
        self.scene.render.filepath = str(self.directory / 'intended_final.png')
        original = (self.scene.camera, self.scene.cycles.samples, self.scene.render.resolution_x,
                    self.scene.render.resolution_y, self.scene.render.filepath)
        views = [{'id': 'a'}, {'id': 'b'}]
        geometry = {o.as_pointer() for o in self.scene.objects}
        calls = []
        def tiny_view(scene, camera, view, max_edge):
            calls.append((view['id'], max_edge))
            scene.camera = camera
            scene.render.resolution_x, scene.render.resolution_y = 24, 16
        report = self.calibration.calibrate(self.scene, views, cameras,
            {'a': reference_path, 'b': reference_path}, tiny_view)
        self.assertEqual(report['correction_ev'], -1)
        self.assertAlmostEqual(self.scene.view_settings.exposure, initial_exposure - 1)
        self.assertEqual(report['shared_across_views'], ['a', 'b'])
        self.assertEqual(calls, [('a', 384), ('b', 384)])
        self.assertEqual(original, (self.scene.camera, self.scene.cycles.samples, self.scene.render.resolution_x,
                                   self.scene.render.resolution_y, self.scene.render.filepath))
        self.assertEqual(geometry, {o.as_pointer() for o in self.scene.objects})
        self.assertEqual(self.scene.view_settings.gamma, 1)

    def test_failed_calibration_restores_original_settings(self):
        camera = self.camera('FailureCamera', (0, 0, 2), (0, 0, 0))
        self.scene.camera = camera
        path = self.directory / 'reference.png'
        write_rgb_png(path, (128, 128, 128))
        original = (self.scene.cycles.samples, self.scene.render.resolution_x,
                    self.scene.render.resolution_y, self.scene.view_settings.exposure)
        original_preview = self.preview_settings()
        def fail_after_change(scene, camera, view, max_edge):
            scene.render.resolution_x = 12
            scene.view_settings.exposure = -1.5
            scene.cycles.seed = 93
            scene.cycles.use_animated_seed = True
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.use_denoising = True
            scene.render.resolution_percentage = 25
            scene.render.use_border = True
            raise RuntimeError('Synthetic set_view failure')
        with self.assertRaisesRegex(RuntimeError, 'Synthetic'):
            self.calibration.calibrate(self.scene, [{'id': 'a'}], {'a': camera}, {'a': path}, fail_after_change)
        self.assertEqual(original, (self.scene.cycles.samples, self.scene.render.resolution_x,
                                   self.scene.render.resolution_y, self.scene.view_settings.exposure))
        self.assertEqual(original_preview, self.preview_settings())

    def test_render_failure_restores_preview_settings(self):
        camera = self.camera('RenderFailureCamera', (0, 0, 2), (0, 0, 0))
        self.scene.camera = camera
        path = self.directory / 'reference.png'
        write_rgb_png(path, (128, 128, 128))
        original = self.preview_settings()
        exposure = self.scene.view_settings.exposure
        def failed_render():
            raise RuntimeError('Render failed')
        failed_bpy = SimpleNamespace(data=bpy.data,
            ops=SimpleNamespace(render=SimpleNamespace(render=failed_render)))
        with patch('calibrate_reference.bpy', failed_bpy):
            with self.assertRaisesRegex(RuntimeError, 'Render failed'):
                self.calibration.calibrate(self.scene, [{'id': 'a'}], {'a': camera},
                    {'a': path}, lambda *args: None, fit_lighting=False)
        self.assertEqual(self.preview_settings(), original)
        self.assertEqual(self.scene.view_settings.exposure, exposure)

    def test_selected_lighting_fit_preview_is_recorded_and_diagnosed(self):
        """Use the lighting-fit result schema when committing its selected candidate."""
        self.plane(emission=True)
        camera = self.camera('SelectedFitCamera', (0, -2, 2), (0, 0, 0))
        self.scene.camera = camera
        _, reference_path = self.render_rgb('selected_fit_reference')
        rgb = self.calibration.read_display_rgb(reference_path)
        preview_stats = self.calibration.compute_stats(rgb)
        preview_grid = self.calibration.compute_spatial_stats(rgb, bottom_up=True)
        fit = {'status': 'selected', 'selected_index': 1, 'method': 'test-fit',
               'baseline_spatial_loss': 0.2, 'selected_spatial_loss': 0.1,
               'candidates': [], 'preview_max_edge': 256, 'preview_samples': 4,
               'limitations': []}
        def tiny_view(scene, selected_camera, view, max_edge):
            scene.camera = selected_camera
            scene.render.resolution_x, scene.render.resolution_y = 24, 16

        # Fit candidates and fail-closed rollback candidates use distinct
        # persisted field names; both can become the selected final result.
        for stats_field, grid_field in (('preview_stats', 'preview_grid'),
                                        ('views', 'grid_views')):
            with self.subTest(stats_field=stats_field):
                self.scene.view_settings.exposure = 0.25
                selected = {'ev': 0.25, stats_field: {'a': preview_stats},
                            grid_field: {'a': preview_grid}}
                with patch('lighting_fit.fit_rendered_lighting', return_value=(fit, selected)):
                    report = self.calibration.calibrate(self.scene, [{'id': 'a'}], {'a': camera},
                        {'a': reference_path}, tiny_view)

                self.assertEqual(report['selected_preview_stats'], {'a': preview_stats})
                self.assertEqual(report['spatial_diagnostics'], self.calibration.compare_spatial_stats(
                    {'a': preview_grid}, {'a': preview_grid}, at_correction_bound=False))
                self.assertEqual(report['final_exposure'], 0.5)

    def test_fixture_face_does_not_change_diffuse_floor_illumination(self):
        self.plane(emission=False)
        self.scene.camera = self.camera('DiffuseCheck', (0, 0, 1), (0, 0, 0))
        self.scene.cycles.samples = 32
        self.scene.view_settings.exposure = 0
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 2))
        backing = bpy.context.object
        backing.name = 'SmallFixtureBacking'
        backing.scale = (.8, .8, .05)
        lamp_data = bpy.data.lights.new('PhysicalArea', 'AREA')
        lamp_data.shape = 'SQUARE'
        lamp_data.size = .8
        lamp_data.energy = 100
        lamp = bpy.data.objects.new('PhysicalArea', lamp_data)
        self.scene.collection.objects.link(lamp)
        lamp.location = (0, 0, 1.96)
        before, _ = self.render_rgb('without_face')
        records = self.fixtures.add_fixture_faces(self.scene)
        self.assertEqual(len(records), 1)
        face = bpy.data.objects[records[0]['face']]
        self.assertFalse(face.visible_diffuse)
        self.assertFalse(face.visible_shadow)
        self.assertEqual(lamp.data.energy, 100)
        after, _ = self.render_rgb('with_face')
        # One output quantization step tolerates platform-level color rounding.
        np.testing.assert_allclose(before, after, atol=1 / 255 + 1e-7, rtol=0)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceCalibrationBlenderTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
