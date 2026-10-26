"""Real evaluated-geometry regression checks for the pre-render safety gate."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class GeometryPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate_multiview
        from test_multiview import fixture
        cls.renderer = generate_multiview
        cls.fixture = staticmethod(fixture)

    def bundle(self):
        bundle = self.fixture()
        bundle['shared_scene']['custom_objects'] = []
        bundle['shared_scene']['base_scene']['objects'] = [bundle['shared_scene']['base_scene']['objects'][0]]
        return bundle

    def preflight(self, bundle):
        return self.renderer.geometry_preflight(bundle, self.renderer.build_scene(bundle))

    def test_valid_scene_passes(self):
        self.assertEqual(self.preflight(self.bundle())['status'], 'passed')

    def test_camera_on_preserved_wall_rejected(self):
        bundle = self.bundle()
        bundle['views'][0]['camera']['position'][0] = 2.5
        report = self.preflight(bundle)
        self.assertEqual(report['status'], 'rejected')
        error = next(e for e in report['errors'] if e['code'] == 'camera_room_shell_collision')
        self.assertEqual(error['view_id'], 'a')
        self.assertEqual(error['surfaces'][0]['surface'], 'right')

    def test_omitted_wall_does_not_reject_camera_or_furniture(self):
        bundle = self.bundle()
        bundle['architecture']['omit_surfaces'].append('right')
        bundle['views'][0]['camera']['position'][0] = 2.5
        bundle['shared_scene']['base_scene']['objects'][0]['position'][0] = 3.2
        self.assertEqual(self.preflight(bundle)['status'], 'passed')

    def test_real_furniture_geometry_outside_room_rejected(self):
        bundle = self.bundle()
        bundle['shared_scene']['base_scene']['objects'][0]['position'][0] = 3.2
        report = self.preflight(bundle)
        error = next(e for e in report['errors'] if e['code'] == 'geometry_outside_preserved_room')
        self.assertEqual(error['object_id'], 'bed')
        self.assertGreater(error['world_aabb_m']['min'][0], 2.0)

    def test_custom_parts_use_actual_bounds_not_only_group_origin(self):
        bundle = self.bundle()
        from make_v2_examples import group
        from spec_v2 import part
        bundle['shared_scene']['custom_objects'] = [group('custom_bed', [0, 2, 0], [part(position=[3.2, 0, .7], size=[1, 2, 1.4])])]
        report = self.preflight(bundle)
        self.assertTrue(any(e.get('object_id') == 'custom_bed' for e in report['errors']))

    def test_structural_panel_and_exterior_light_are_not_furniture(self):
        bundle = self.bundle()
        from make_v2_examples import group, area
        from spec_v2 import part
        bundle['shared_scene']['custom_objects'] = [group('wall_panel', [2.5, 3, 1.4], [part(position=[0, 0, 0], size=[.2, 5.8, 2.8])])]
        bundle['shared_scene']['lights'] = [area('daylight', [5, 6, 3], [0, 3, 1], 300)]
        report = self.preflight(bundle)
        self.assertEqual(report['status'], 'passed')
        self.assertTrue(next(o for o in report['objects'] if o['object_id'] == 'wall_panel')['structural_exemption'])

    def test_main_stops_before_calibration_render_or_scene_publication(self):
        bundle = self.bundle()
        bundle['views'][0]['camera']['position'][0] = 2.5
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for view in bundle['views']:
                (folder / view['source_image']).write_bytes(b'not loaded before geometry gate')
            spec = folder / 'bundle.json'
            spec.write_text(json.dumps(bundle))
            out = folder / 'output'
            args = ['blender', '--', '--spec', str(spec), '--output-dir', str(out)]
            with patch.object(sys, 'argv', args), self.assertRaisesRegex(ValueError, 'Geometry preflight rejected'):
                self.renderer.main()
            self.assertTrue((out / 'geometry_preflight.json').is_file())
            report = json.loads((out / 'geometry_preflight.json').read_text())
            self.assertEqual(report['spec_sha256'], hashlib.sha256(spec.read_bytes()).hexdigest())
            self.assertEqual(report['status'], 'rejected')
            self.assertFalse((out / 'tone_calibration.json').exists())
            self.assertFalse((out / 'scene.blend').exists())
            self.assertFalse((out / 'rgb.png').exists())
            self.assertFalse((out / 'metadata.json').exists())

    def test_passed_report_retains_exact_spec_hash(self):
        bundle = self.bundle()
        # Noncanonical whitespace deliberately distinguishes file bytes from
        # a canonical serialization of the parsed bundle.
        raw = ('\n' + json.dumps(bundle, indent=3) + '\n\n').encode()
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            report = self.renderer.enforce_geometry_preflight(
                bundle, self.renderer.build_scene(bundle), folder, spec_sha256=digest)
            persisted = json.loads((Path(folder) / 'geometry_preflight.json').read_text())
            self.assertEqual(report, persisted)
            self.assertEqual(report['spec_sha256'], digest)
            self.assertEqual(report['status'], 'passed')

    def test_invalid_hash_never_publishes_report(self):
        bundle = self.bundle()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'SHA256 binding'):
                self.renderer.enforce_geometry_preflight(
                    bundle, self.renderer.build_scene(bundle), folder, spec_sha256='stale')
            self.assertFalse((Path(folder) / 'geometry_preflight.json').exists())


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryPreflightTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
