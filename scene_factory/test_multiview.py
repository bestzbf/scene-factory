"""Grouped input validation tests; normal Python (no Blender dependency)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from multiview_spec import load, render_dimensions, resolve_sources, texture_view_map, validate


def fixture():
    spec = json.loads((Path(__file__).parent/'examples_v2/bedroom.json').read_text())
    bundle = {'schema_version': 3, 'scene_id': 'room', 'shared_scene': spec,
            'architecture': {'omit_surfaces': ['back'], 'front_wall': True},
            'views': [{'id': 'a', 'source_image': 'a.png', 'width': 1280, 'height': 960,
                       'camera': {'position': [0, .3, 1.6], 'target': [0, 5, 1.4], 'lens_mm': 24, 'roll_degrees': -2}},
                      {'id': 'b', 'source_image': 'b.png', 'width': 960, 'height': 1280,
                       'camera': {'position': [0, 5, 1.6], 'target': [0, 1, 1.4], 'lens_mm': 28, 'roll_degrees': 1}}],
            'texture_sources': []}
    first = bundle['views'][0]['camera']
    bundle['shared_scene']['base_scene']['camera'] = {
        key: first[key] for key in ('position', 'target', 'lens_mm')}
    bundle['shared_scene']['camera']['roll_degrees'] = first['roll_degrees']
    return bundle


class MultiviewSpecTests(unittest.TestCase):
    def test_valid_shared_geometry_and_aspect(self):
        bundle = fixture()
        self.assertIs(validate(bundle), bundle)
        self.assertEqual(render_dimensions(bundle['views'][0], 800), (800, 600))
        self.assertEqual(render_dimensions(bundle['views'][1], 800), (600, 800))
        self.assertNotEqual(bundle['views'][0]['camera'], bundle['views'][1]['camera'])

    def test_ids_cannot_escape_output_directory_or_duplicate(self):
        for bad in ('../view', 'a/b', 'a\\b', '', 'a.b', ' '):
            bundle = fixture()
            bundle['views'][0]['id'] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                validate(bundle)
        bundle = fixture()
        bundle['views'][1]['id'] = 'a'
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate(bundle)

    def test_camera_validation(self):
        for value in (float('nan'), float('inf'), 0, True, 11, 86):
            bundle = fixture()
            bundle['views'][0]['camera']['lens_mm'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(bundle)
        for value in (12, 85):
            bundle = fixture()
            bundle['views'][0]['camera']['lens_mm'] = value
            bundle['shared_scene']['base_scene']['camera']['lens_mm'] = value
            self.assertIs(validate(bundle), bundle)
        bundle = fixture()
        bundle['views'][0]['camera']['target'] = bundle['views'][0]['camera']['position'][:]
        with self.assertRaisesRegex(ValueError, 'differ'):
            validate(bundle)

    def test_primary_camera_and_roll_must_match_legacy_fields(self):
        for field, value in (('position', [1, 2, 3]), ('target', [1, 2, 3]), ('lens_mm', 35)):
            bundle = fixture()
            bundle['shared_scene']['base_scene']['camera'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'First view camera'):
                validate(bundle)
        bundle = fixture()
        bundle['shared_scene']['camera']['roll_degrees'] = 5
        with self.assertRaisesRegex(ValueError, 'First view roll_degrees'):
            validate(bundle)

    def test_texture_source_maps_explicit_second_view(self):
        bundle = fixture()
        bundle['shared_scene']['textures'] = [{'id': 'screen', 'quad': [[0,0],[1,0],[1,1],[0,1]]}]
        self.assertEqual(texture_view_map(validate(bundle)), {'screen': 'a'})
        bundle['texture_sources'] = [{'texture_id': 'screen', 'view_id': 'b'}]
        self.assertEqual(texture_view_map(validate(bundle)), {'screen': 'b'})
        bundle['texture_sources'].append(bundle['texture_sources'][0].copy())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate(bundle)
        bundle['texture_sources'] = [{'texture_id': 'screen', 'view_id': 'missing'}]
        with self.assertRaisesRegex(ValueError, 'unknown view'):
            validate(bundle)

    def test_omitted_surfaces_validated(self):
        for bad in (['front'], ['back', 'back'], ['unknown'], 'back'):
            bundle = fixture()
            bundle['architecture']['omit_surfaces'] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                validate(bundle)

    def test_sources_relative_to_bundle_not_current_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle = fixture()
            for name in ('a.png', 'b.png'):
                (directory/name).write_bytes(b'placeholder')
            path = directory/'bundle.json'
            path.write_text(json.dumps(bundle))
            self.assertEqual(resolve_sources(load(path), path)['a'], (directory/'a.png').resolve())
            (directory/'b.png').unlink()
            with self.assertRaises(FileNotFoundError):
                resolve_sources(bundle, path)


if __name__ == '__main__':
    unittest.main()
