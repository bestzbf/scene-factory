"""Library layout contract tests, including loopback transport recovery."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from batch import file_sha256, read_json
import library_layout as layout
import model_api
from multiview_spec import validate
from test_model_api import FixtureServer, Handler, PNG, chat


def fixture():
    return {
        'schema_version': 'library_layout_v1', 'scene_id': 'room', 'scene_type': 'living_room',
        'description': 'Synthetic local contract test, not photo reconstruction.', 'seed': 42,
        'room': {'width': 5, 'depth': 6, 'height': 3, 'wall_color': [.6, .6, .6],
                 'floor_color': [.3, .3, .3], 'ceiling_color': [.8, .8, .8]},
        'views': [{'id': 'view_01', 'source_image': 'input:view_01', 'width': 1, 'height': 1,
                   'camera': {'position': [0, .5, 1.6], 'target': [0, 4, 1],
                              'lens_mm': 24, 'roll_degrees': 0}}],
        'architecture': {'omit_surfaces': [], 'front_wall': False},
        'lighting': {'energy': 500, 'color': [1, .95, .9], 'world_strength': .1},
        'lights': [], 'exposure': 0, 'world_color': [.3, .35, .4],
        'objects': [
            {'id': 'remote', 'category': 'remote_control', 'xy': [0, 3],
             'target_dimensions_m': [.05, .18, .02], 'rotation_z_deg': 0,
             'support_id': 'table', 'clearance_m': .001},
            {'id': 'table', 'category': 'table', 'xy': [0, 3],
             'target_dimensions_m': [1.5, .8, .75], 'rotation_z_deg': 0,
             'support_id': 'floor', 'clearance_m': 0}],
        'assumptions': ['Source geometry and physical scale are not measured here.']}


def catalog_fixture(count=1):
    return {'schema_version': 'asset_catalog_v1', 'root': '/not-sent-to-model',
            'assets': [{'asset_id': 'secret_asset_' + str(i), 'category': 'table',
                        'status': 'technical_ready', 'placement': 'floor',
                        'dimensions_import_units': [1.5, .8, .75], 'meters_per_unit': 1,
                        'model_file': 'private/model_' + str(i) + '.glb',
                        'support_surfaces': [{'z': .75}]}
                       for i in range(count)]}


class LibraryLayoutTests(unittest.TestCase):
    def test_expand_is_immutable_and_resolves_out_of_order_graph_provisionally(self):
        raw = fixture()
        before = copy.deepcopy(raw)
        result = layout.expand_response(raw)
        validate(result)
        self.assertEqual(raw, before)
        objects = result['shared_scene']['base_scene']['objects']
        self.assertEqual(objects[0]['position'], [0, 3, .751])
        self.assertEqual(objects[1]['position'], [0, 3, 0])
        self.assertEqual([obj['category'] for obj in objects], ['box', 'box'])
        self.assertEqual(result['shared_scene']['lights'], [])
        self.assertFalse(result['shared_scene']['render']['replace_base_lights'])

    def test_scene_and_views_bound_to_trusted_identity(self):
        raw = fixture()
        views = [{'id': 'view_01', 'source_image': '/tmp/a.png', 'width': 1, 'height': 1}]
        layout.validate_layout(raw, views, 'room')
        with self.assertRaisesRegex(ValueError, 'scene identity'):
            layout.validate_layout(raw, views, 'another_room')
        for key, value in [('id', 'other'), ('width', 2), ('height', 2), ('source_image', 'a.png')]:
            wrong = copy.deepcopy(raw)
            wrong['views'][0][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                layout.validate_layout(wrong, views, 'room')

    def test_multiview_reordering_fails_even_with_valid_ids(self):
        raw = fixture()
        second = copy.deepcopy(raw['views'][0])
        second.update(id='view_02', source_image='input:view_02')
        raw['views'].append(second)
        expected = copy.deepcopy(raw['views'])
        raw['views'].reverse()
        with self.assertRaisesRegex(ValueError, 'view order'):
            layout.validate_layout(raw, expected)

    def test_graph_missing_cycle_self_duplicate_and_reserved_id_fail(self):
        for case in ('missing', 'cycle', 'self', 'duplicate', 'reserved', 'path'):
            raw = fixture()
            if case == 'missing': raw['objects'][0]['support_id'] = 'absent'
            elif case == 'cycle': raw['objects'][1]['support_id'] = 'remote'
            elif case == 'self': raw['objects'][0]['support_id'] = 'remote'
            elif case == 'duplicate': raw['objects'][1]['id'] = 'remote'
            elif case == 'reserved': raw['objects'][1]['id'] = 'floor'
            else: raw['objects'][0]['id'] = '../remote'
            with self.subTest(case=case), self.assertRaises(ValueError):
                layout.validate_layout(raw)

    def test_floor_omission_and_floating_floor_declaration_fail(self):
        for omitted, clearance in [(['floor'], 0), ([], .01)]:
            raw = fixture()
            raw['architecture']['omit_surfaces'] = omitted
            raw['objects'][1]['clearance_m'] = clearance
            with self.assertRaises(ValueError):
                layout.validate_layout(raw)

    def test_empty_extra_nonfinite_and_bad_categories_rejected(self):
        for case in ('empty', 'extra', 'nan', 'category'):
            raw = fixture()
            if case == 'empty': raw['objects'] = []
            elif case == 'extra': raw['objects'][0]['asset_id'] = 'guessed'
            elif case == 'nan': raw['objects'][0]['target_dimensions_m'][0] = float('nan')
            else: raw['objects'][0]['category'] = 'remote control'
            with self.subTest(case=case), self.assertRaises(ValueError):
                layout.validate_layout(raw)

    def test_unknown_library_category_can_be_reported_as_gap(self):
        raw = fixture()
        raw['objects'][0]['category'] = 'rare_lab_instrument'
        self.assertIs(layout.validate_layout(raw), raw)

    def test_context_is_category_bounded_not_asset_inventory(self):
        small = layout.catalog_context(catalog_fixture(1))
        large = layout.catalog_context(catalog_fixture(10000))
        encoded = json.dumps(large)
        self.assertEqual(large['categories'][0]['count'], 10000)
        self.assertEqual(large['categories'][0]['dimension_min_m'], [1.5, .8, .75])
        self.assertLess(len(encoded), len(json.dumps(small)) + 20)
        for forbidden in ('secret_asset_', 'private/', 'not-sent-to-model', 'model_file'):
            self.assertNotIn(forbidden, encoded)

    def test_context_category_overflow_and_bad_scale_fail_explicitly(self):
        cat = catalog_fixture(2)
        cat['assets'][1]['category'] = 'chair'
        with self.assertRaisesRegex(ValueError, 'exceeds prompt limit'):
            layout.catalog_context(cat, max_categories=1)
        cat['assets'][1]['meters_per_unit'] = None
        with self.assertRaisesRegex(ValueError, 'metric dimensions'):
            layout.catalog_context(cat)
        self.assertEqual(len(layout.catalog_context({'assets': [{**cat['assets'][0], 'status': 'draft'},
            {**cat['assets'][0], 'status': 'approved'}]})['categories']), 1)

    def test_pinned_context_scans_once_preserves_limits_and_returns_independent_values(self):
        data = catalog_fixture(2)
        data['assets'][1]['category'] = 'chair'
        catalog = SimpleNamespace(assets={a['asset_id']: a for a in data['assets']},
            _batch_catalog_sha256='immutable-batch', _snapshot_file_hashes={},
            _snapshot_lock=threading.RLock())
        with patch.object(layout, '_catalog_context', wraps=layout._catalog_context) as summarize:
            with self.assertRaisesRegex(ValueError, 'exceeds prompt limit'):
                layout.catalog_context(catalog, max_categories=1)
            first = layout.catalog_context(catalog)
            first['categories'][0]['count'] = 999
            second = layout.catalog_context(catalog, max_categories=2)
            self.assertEqual(second['categories'][0]['count'], 1)
            self.assertEqual(summarize.call_count, 1)

    def test_unpinned_dictionary_context_tracks_mutations(self):
        catalog = catalog_fixture()
        self.assertEqual(layout.catalog_context(catalog)['categories'][0]['category'], 'table')
        catalog['assets'][0]['category'] = 'chair'
        self.assertEqual(layout.catalog_context(catalog)['categories'][0]['category'], 'chair')


class LibraryLayoutAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FixtureServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.image = self.directory / 'photo.png'
        self.image.write_bytes(PNG)
        self.views = [{'id': 'view_01', 'source_image': str(self.image), 'width': 1, 'height': 1}]
        self.raw = fixture()
        self.output = self.directory / 'initial_bundle.json'
        self.catalog = catalog_fixture()
        self.model = {'provider': 'openai_chat', 'model': 'library-fixture',
                      'base_url': f'http://127.0.0.1:{self.server.server_port}/v1',
                      'api_key_env': '', 'max_retries': 0, 'max_output_tokens': 12288,
                      'reasoning_effort': 'low'}
        self.server.requests, self.server.responses = [], []

    def queue(self):
        self.server.responses.append((200, {}, chat(json.dumps(self.raw))))

    def analyze(self, **kwargs):
        return layout.analyze('room', self.views, self.output, self.catalog,
                              model_config=self.model, **kwargs)

    def test_one_request_saves_original_raw_layout_and_committed_provenance(self):
        self.queue()
        result = self.analyze()
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(read_json(self.output.with_suffix('.raw.json')), self.raw)
        self.assertEqual(read_json(self.directory / 'initial_layout.json'), self.raw)
        self.assertEqual(read_json(self.output), result)
        self.assertEqual(result['views'][0]['source_image'], str(self.image))
        prov = read_json(self.directory / 'analysis_provenance.json')
        self.assertEqual(prov['bundle_sha256'], file_sha256(self.output))
        self.assertEqual(prov['layout_sha256'], file_sha256(self.directory / 'initial_layout.json'))
        self.assertEqual(prov['reasoning_effort'], 'low')
        prompt = self.output.with_suffix('.prompt.txt').read_text()
        self.assertIn('LIBRARY CATEGORY HINTS', prompt)
        self.assertNotIn('secret_asset_', prompt)
        self.assertNotIn('/not-sent-to-model', prompt)

    def test_semantic_failure_preserves_response_and_resume_never_reposts(self):
        self.raw['objects'][0]['support_id'] = 'missing'
        self.queue()
        before = None
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'Unknown support_id'):
                self.analyze()
            self.assertEqual(len(self.server.requests), 1)
            self.assertFalse((self.directory / 'analysis_provenance.json').exists())
            current = {s: self.output.with_suffix(s).read_bytes()
                       for s in ('.raw.json', '.api.json', '.prompt.txt')}
            if before is not None: self.assertEqual(current, before)
            before = current

    def test_commit_interruption_recovers_transport_without_another_request(self):
        self.queue()
        write = layout.atomic_json
        def interrupt(path, data):
            if Path(path).name == 'analysis_provenance.json':
                raise OSError('synthetic interruption')
            return write(path, data)
        with patch.object(layout, 'atomic_json', side_effect=interrupt):
            with self.assertRaisesRegex(OSError, 'synthetic interruption'):
                self.analyze()
        before = {s: self.output.with_suffix(s).read_bytes()
                  for s in ('.raw.json', '.api.json', '.prompt.txt')}
        with patch.object(model_api, '_request', side_effect=AssertionError('No second POST')):
            self.analyze()
        self.assertEqual(len(self.server.requests), 1)
        for suffix, contents in before.items():
            self.assertEqual(self.output.with_suffix(suffix).read_bytes(), contents)
        prov = read_json(self.directory / 'analysis_provenance.json')
        self.assertEqual(prov['raw_model_response_cache']['method'], 'same_stage_completed_api_response')

    def test_changed_catalog_context_blocks_paid_replay(self):
        self.queue(); self.analyze()
        self.catalog['assets'][0]['dimensions_import_units'][0] = 2
        with patch.object(model_api, '_request', side_effect=AssertionError('No second POST')):
            with self.assertRaisesRegex(ValueError, 'cannot be safely reused'):
                self.analyze()

    def test_cross_batch_raw_cache_is_explicitly_unsupported_without_request(self):
        with self.assertRaisesRegex(ValueError, 'does not support cross-batch'):
            self.analyze(raw_cache=self.directory)
        self.assertEqual(self.server.requests, [])


if __name__ == '__main__':
    unittest.main()
