"""Actual PNG/Blender pixel ordering regression for spatial light diagnostics."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class SpatialToneBlenderTests(unittest.TestCase):
    def test_calibration_publishes_only_selected_grid_and_keeps_selection(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_reference_calibration_blender import ReferenceCalibrationBlenderTests
        # Reuse the small real Cycles fixture, not mocked image statistics.
        ReferenceCalibrationBlenderTests.setUpClass()
        fixture = ReferenceCalibrationBlenderTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.plane(emission=True)
        camera = fixture.camera('SpatialMeter', (0, 0, 2), (0, 0, 0))
        fixture.scene.camera = camera
        original_exposure = fixture.scene.view_settings.exposure
        fixture.scene.view_settings.exposure = original_exposure - 1
        _, reference = fixture.render_rgb('reference_spatial')
        fixture.scene.view_settings.exposure = original_exposure
        def set_view(scene, camera, view, max_edge):
            scene.camera = camera
            scene.render.resolution_x, scene.render.resolution_y = 24, 16
        report = fixture.calibration.calibrate(fixture.scene, [{'id': 'a'}],
            {'a': camera}, {'a': reference}, set_view)
        self.assertEqual(report['correction_ev'], -1)
        spatial = report['spatial_diagnostics']
        self.assertEqual(spatial['coordinate_origin'], 'top_left')
        self.assertFalse(spatial['requires_inspection'])
        self.assertEqual(len(spatial['per_view']['a']['regions']), 9)
        self.assertTrue(all('grid_views' not in row for row in report['candidates']))
        self.assertNotIn('regions', report['selected_preview_stats']['a'])

    def test_png_upper_rows_remain_upper_in_meter(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from calibrate_reference import read_display_rgb
        from reference_tone import compute_spatial_stats
        def chunk(kind, payload):
            return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind + payload) & 0xffffffff)
        # PNG scanlines are top-down; actual Blender pixels are bottom-up.
        raw = b''.join((b'\x00' + bytes([v, v, v]) * 9) * 3 for v in [210, 128, 40])
        png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', 9, 9, 8, 2, 0, 0, 0)) +
               chunk(b'sRGB', b'\x00') + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vertical_pattern.png'
            path.write_bytes(png)
            pixels = read_display_rgb(path)
            self.assertLess(float(pixels[0, 0, 0]), float(pixels[-1, 0, 0]))
            report = compute_spatial_stats(pixels, bottom_up=True)
            self.assertEqual(report['regions'][0]['id'], 'upper_left')
            self.assertGreater(report['regions'][0]['p65'], report['regions'][-1]['p65'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SpatialToneBlenderTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
