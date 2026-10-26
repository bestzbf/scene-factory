"""Tests for packaged input grouping, configuration, and offline commands."""
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scene_factory.cli import discover_groups, doctor, init_project, main, probe
from scene_factory.factory_config import load_config, normalize_config


BASE = {'schema_version': 1, 'models': {'analysis': {
    'provider': 'openai_chat', 'base_url': 'https://models.example.test/v1',
    'model': 'vision-model', 'api_key_env': 'SCENE_TEST_KEY', 'response_format': 'json_object'}}}


class ConfigTests(unittest.TestCase):
    def test_scene_budget_is_optional_strict_and_idempotent(self):
        self.assertNotIn('budget', normalize_config(BASE))
        policy = {'max_requests': 24, 'max_output_tokens': 393216, 'max_reported_tokens': 600000}
        result = normalize_config({**BASE, 'budget': policy})
        self.assertEqual(result['budget'], policy)
        self.assertEqual(normalize_config(result), result)
        for bad in (None, [], {'max_requests': True}, {'max_requests': -1},
                    {'max_output_tokens': 1.5}, {'total_dollars': 3}):
            with self.subTest(budget=bad), self.assertRaises(ValueError):
                normalize_config({**BASE, 'budget': bad})

    def test_asset_library_requires_paired_paths_and_explicit_preview(self):
        result = normalize_config({**BASE, 'asset_library': {'catalog': 'catalog.json', 'bindings': 'bindings.json'}})
        self.assertFalse(result['asset_library']['preview'])
        for bad in ({'catalog': 'catalog.json'}, {'preview': 'false'}, {'catalog': 12, 'bindings': 'b.json'}):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                normalize_config({**BASE, 'asset_library': bad})

    def test_automatic_asset_selection_config_and_legacy_compatibility(self):
        result = normalize_config({**BASE, 'asset_library': {
            'catalog': 'catalog.json', 'auto_bind': True,
            'selection': {'seed': 42, 'dimension_tolerance': .12}}})
        self.assertIsNone(result['asset_library']['bindings'])
        self.assertFalse(result['asset_library']['preview'])
        self.assertEqual(result['asset_library']['selection']['seed'], 42)
        self.assertFalse(result['asset_library']['selection']['diversity'])
        self.assertEqual(normalize_config(result), result)
        cases = [
            {'auto_bind': True},
            {'auto_bind': True, 'catalog': 'c.json', 'bindings': 'b.json'},
            {'catalog': 'c.json', 'auto_bind': 'true'},
            {'selection': {'seed': 1}},
            {'catalog': 'c.json', 'auto_bind': True, 'selection': {'allow_preview': True}},
            {'catalog': 'c.json', 'auto_bind': True, 'selection': {'dimension_tolerance': float('nan')}},
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_config({**BASE, 'asset_library': value})

    def test_staged_roles_and_limits(self):
        source = copy.deepcopy(BASE)
        source['models']['review'] = {**source['models']['analysis'], 'model': 'reviewer'}
        source['pipeline'] = {'analysis_mode': 'staged', 'staged': {'detail_workers': 2}}
        result = normalize_config(source)
        self.assertEqual(result['models']['repair']['model'], 'reviewer')
        result['models']['repair']['model'] = 'repairer'
        self.assertEqual(result['models']['review']['model'], 'reviewer')
        self.assertEqual(result['pipeline']['staged']['planner_timeout'], 900)
        self.assertEqual(result['pipeline']['repair_stagnation_patience'], 2)
        for bad in ({'analysis_mode': 'codex'}, {'repair_mode': 'python'},
                    {'validation_retries': True}, {'staged': {'max_groups': 16, 'max_parts_per_group': 100}},
                    {'repair_stagnation_patience': True}, {'repair_stagnation_patience': -1},
                    {'repair_stagnation_patience': 6}, {'repair_stagnation_patience': 1.5},
                    {'staged': {'api_key': 'forbidden'}}):
            with self.subTest(pipeline=bad), self.assertRaises(ValueError):
                normalize_config({**BASE, 'pipeline': bad})

    def test_repair_stagnation_can_be_disabled_or_tuned_explicitly(self):
        for patience in (0, 1, 2, 5):
            with self.subTest(patience=patience):
                result = normalize_config({**BASE, 'pipeline': {'repair_stagnation_patience': patience}})
                self.assertEqual(result['pipeline']['repair_stagnation_patience'], patience)
                self.assertEqual(normalize_config(result), result)

    def test_defaults_and_independent_review(self):
        source = copy.deepcopy(BASE)
        result = normalize_config(source)
        self.assertTrue(result['batch']['no_review'])
        self.assertEqual(result['batch']['max_repairs'], 0)
        self.assertEqual(result['render']['samples'], 96)
        result['models']['review']['model'] = 'other'
        self.assertEqual(result['models']['analysis']['model'], 'vision-model')
        self.assertEqual(source, BASE)

    def test_rejects_secrets_unknown_fields_and_implicit_endpoint(self):
        for location in ('top', 'model', 'batch', 'render'):
            value = copy.deepcopy(BASE)
            target = value if location == 'top' else (value['models']['analysis'] if location == 'model' else value.setdefault(location, {}))
            target['api_key'] = 'do-not-record-this'
            with self.assertRaises(ValueError) as caught:
                normalize_config(value)
            self.assertNotIn('do-not-record-this', str(caught.exception))
        value = copy.deepcopy(BASE)
        del value['models']['analysis']['base_url']
        with self.assertRaisesRegex(ValueError, 'base_url'):
            normalize_config(value)

    def test_rejects_invalid_scalar_and_contradictory_review_options(self):
        cases = [('batch', 'workers', True), ('batch', 'workers', 0),
                 ('batch', 'no_review', 'false'), ('batch', 'max_repairs', 1),
                 ('render', 'max_edge', 12), ('render', 'samples', 1.5),
                 ('render', 'tone_match', 'auto'), ('render', 'device', 'GPU')]
        for section, key, entry in cases:
            with self.subTest(section=section, key=key, entry=entry):
                value = copy.deepcopy(BASE)
                value[section] = {key: entry}
                with self.assertRaises(ValueError):
                    normalize_config(value)

    def test_load_rejects_duplicate_fields_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config.json'
            for text in ('{"schema_version":1,"schema_version":1}', '{"value":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(path)


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def photo(self, path):
        path = self.root/path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'grouping does not decode photos')

    def test_groups_shared_multiview_and_stable_ids(self):
        for path in ('single.JPG', 'single.png', '卧室/view02.jpg', '卧室/other/view01.jpg',
                     'empty/readme.txt', '.hidden.jpg', '卧室/.ignore.jpg'):
            self.photo(path)
        first = discover_groups(self.root)
        self.assertEqual(len(first['scenes']), 3)
        self.assertEqual(sorted(len(record['images']) for record in first['scenes']), [1, 1, 2])
        self.assertEqual(len({record['scene_id'] for record in first['scenes']}), 3)
        self.assertTrue(all(record['scene_id'].isascii() for record in first['scenes']))
        self.photo('new.jpg')
        second = discover_groups(self.root)
        previous = {tuple(record['images']): record['scene_id'] for record in first['scenes']}
        for record in second['scenes']:
            if tuple(record['images']) in previous:
                self.assertEqual(record['scene_id'], previous[tuple(record['images'])])

    def test_limits_group_size(self):
        for index in range(17):
            self.photo(f'one_room/view{index}.jpg')
        with self.assertRaisesRegex(ValueError, '1..16'):
            discover_groups(self.root)

    def test_rejects_symlink_photo(self):
        self.photo('one.jpg')
        (self.root/'alias.jpg').symlink_to(self.root/'one.jpg')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            discover_groups(self.root)

    def test_no_photos_fails(self):
        with self.assertRaisesRegex(ValueError, 'No supported photos'):
            discover_groups(self.root)


class CommandTests(unittest.TestCase):
    def test_production_profiles_enable_review_repair_and_api_backpressure(self):
        for profile in ('bigmodel-production', 'bigmodel-auto-assets'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temp:
                result = init_project(Path(temp)/'project', profile)
                config = load_config(result['config'])
                self.assertFalse(config['batch']['no_review'])
                self.assertGreaterEqual(config['batch']['max_repairs'], 1)
                self.assertEqual(config['batch']['api_workers'], 2)
                self.assertEqual(config['pipeline']['analysis_mode'], 'staged')
                self.assertEqual(config['pipeline']['repair_mode'], 'typed')
                self.assertGreaterEqual(config['render']['samples'], 64)
                self.assertEqual(config['asset_library']['auto_bind'], profile.endswith('auto-assets'))

    def test_api_worker_limit_rejects_invalid_values(self):
        self.assertEqual(normalize_config(BASE)['batch']['api_workers'], 2)
        for value in (True, 0, 33, 1.5, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_config({**BASE, 'batch': {'api_workers': value}})

    def test_offline_operational_commands_dispatch_without_model_calls(self):
        commands = (
            (['status', 'output', '--verify', '--limit', '5'],
             'scene_factory.diagnostics.batch_status', (Path('output'),), {'verify': True, 'limit': 5}),
            (['usage', 'output'], 'scene_factory.diagnostics.usage_report', (Path('output'),), {}),
            (['export', 'output', '--destination', 'dataset-v1'],
             'scene_factory.dataset_export.export_dataset', (Path('output'), Path('dataset-v1')), {}),
        )
        for argv, target, expected_args, expected_kwargs in commands:
            with self.subTest(command=argv[0]), redirect_stdout(io.StringIO()), \
                 patch(target, return_value={'offline': True}) as operation, \
                 patch('model_api._request', side_effect=AssertionError('network called')):
                self.assertEqual(main(argv), 0)
                operation.assert_called_once_with(*expected_args, **expected_kwargs)

    def test_api_adapter_can_be_imported_without_cli_initialization(self):
        package_root = str(Path(__file__).resolve().parent.parent)
        with tempfile.TemporaryDirectory() as temp:
            env = {**os.environ, 'PYTHONPATH': package_root}
            result = subprocess.run(
                [sys.executable, '-c', 'from scene_factory.model_api import call_model, ModelAPIError'],
                cwd=temp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_named_profiles_have_explicit_endpoints_and_no_credentials(self):
        for profile, provider, base, env in (
            ('bigmodel', 'openai_chat', 'https://open.bigmodel.cn/api/paas/v4', 'BIGMODEL_API_KEY'),
            ('zai-anthropic', 'anthropic', 'https://api.z.ai/api/anthropic', 'ZAI_API_KEY'),
        ):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temp:
                result = init_project(Path(temp)/'project', profile)
                config = load_config(result['config'])
                model = config['models']['analysis']
                self.assertEqual((model['provider'], model['base_url'], model['api_key_env']),
                                 (provider, base, env))
                self.assertNotIn('api_key', model)
                self.assertIn(env, (Path(result['directory'])/'README_运行.md').read_text())

    def test_init_preserves_existing_config_and_requires_provider_url(self):
        with tempfile.TemporaryDirectory() as temp:
            result = init_project(Path(temp)/'project')
            config = Path(result['config'])
            original = config.read_bytes()
            with self.assertRaisesRegex(ValueError, 'base_url'):
                load_config(config)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                init_project(config.parent)
            self.assertEqual(config.read_bytes(), original)

    def test_doctor_does_not_call_network_or_print_key(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp)/'factory.json'
            config.write_text(json.dumps(BASE))
            with patch.dict(os.environ, {'SCENE_TEST_KEY': 'test-secret-value'}), \
                 patch('model_api._request', side_effect=AssertionError('network called')):
                result = doctor(config, blender='/definitely/no/blender')
            serialized = json.dumps(result)
            self.assertNotIn('test-secret-value', serialized)
            self.assertTrue(result['offline'])
            self.assertFalse(result['model_capabilities_verified'])
            self.assertTrue(result['checks'][0]['credential_configured'])
            self.assertFalse(result['ok'])

    def test_probe_validates_image_input_without_real_endpoint(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp)/'factory.json'
            config.write_text(json.dumps(BASE))
            def fake_call(prompt, images, schema, output, **kwargs):
                self.assertIn('top-left', prompt)
                self.assertEqual(schema['required'], ['colors'])
                with Image.open(images[0]) as image:
                    palette = {(230, 20, 20): 'red', (20, 190, 20): 'green',
                               (20, 20, 230): 'blue', (240, 225, 20): 'yellow'}
                    colors = [palette[image.getpixel(position)] for position in ((20, 20), (120, 20), (20, 120), (120, 120))]
                Path(output).with_name('probe.api.json').write_text(json.dumps({'attempts': 1, 'usage': {}}))
                self.assertEqual(kwargs['config']['max_output_tokens'], 4096)
                return {'colors': colors}
            with patch('model_api.call_model', side_effect=fake_call):
                result = probe(config)
            self.assertTrue(result['image_test_passed'])
            self.assertTrue(result['schema_valid'])

    def test_run_forwards_generated_manifest_and_no_environment_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config = folder/'factory.json'
            config.write_text(json.dumps(BASE))
            photos = folder/'photos'
            photos.mkdir()
            (photos/'one.jpg').write_bytes(b'photo')
            captured = {}
            def batch(argv):
                captured['argv'] = argv
                captured['manifest'] = json.loads(Path(argv[argv.index('--groups')+1]).read_text())
                return 0
            original = dict(os.environ)
            with patch('batch_multiview.main', side_effect=batch):
                result = main(['run', '--config', str(config), '--input-dir', str(photos),
                               '--output-dir', str(folder/'output'), '--dry-run'])
            self.assertEqual(result, 0)
            self.assertIn('--config', captured['argv'])
            self.assertIn('--dry-run', captured['argv'])
            self.assertEqual(len(captured['manifest']['scenes']), 1)
            self.assertEqual(dict(os.environ), original)
            self.assertFalse((folder/'output').exists())


if __name__ == '__main__':
    unittest.main()
