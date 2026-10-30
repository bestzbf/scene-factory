"""Anchored placement transport and batch recovery, using loopback fixtures only.

The renderer is deliberately mocked: these tests establish API cost, identity,
placement evidence and resume contracts, not visual fidelity or Blender contact.
"""
import base64
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from anchored_scene import FORMAT_VERSION, expand_response, placement_report
from batch import read_json
import batch_multiview as runner
from factory_config import normalize_config
from multiview_spec import validate
from scene_factory.cli import init_project, main as cli_main
from scene_factory.factory_config import load_config
from test_anchored_scene import base_object, top, wire
import test_batch_multiview as batch_fixtures
from test_compact_scene_api import complete_bundle
from test_model_api import FixtureServer, Handler, chat
import vision_multiview as vision


class AnchoredSceneAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FixtureServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.image = self.directory / 'reference.png'
        # A source larger than common thumbnail limits proves the API payload
        # contains the original file, independently of render resolution.
        Image.new('RGB', (1601, 1201), (80, 110, 140)).save(self.image)
        self.model = {'provider': 'openai_chat', 'model': 'anchored-fixture',
                      'base_url': f'http://127.0.0.1:{self.server.server_port}/v1',
                      'api_key_env': '', 'max_retries': 0, 'max_output_tokens': 12288}
        self.server.requests, self.server.responses = [], []
        self.bundle = complete_bundle()
        self.bundle['shared_scene']['base_scene']['objects'] = [
            base_object(position=[0, 2, .72]),
            base_object('screen', 'monitor', dimensions=[.6, .2, .45])]
        for view in self.bundle['views']:
            view['width'], view['height'] = 1601, 1201
        first = self.bundle['views'][0]
        self.bundle['shared_scene']['image'] = {
            'width': first['width'], 'height': first['height'], 'source_id': first['id']}
        self.views = [{**{key: view[key] for key in ('id', 'width', 'height')},
                       'source_image': str(self.image)} for view in self.bundle['views']]
        self.raw = wire(self.bundle, {'desk': {'mode': 'floor', 'xy': [0, 2]},
                                     'screen': top(clearance=.01)})
        self.output = self.directory / 'initial_bundle.json'

    def queue(self, response=None):
        self.server.responses.append((200, {}, chat(json.dumps(self.raw if response is None else response))))

    def analyze(self, wire_format=FORMAT_VERSION):
        return vision.analyze('room', self.views, self.output, model_config=self.model,
                              wire_format=wire_format)

    def test_one_post_preserves_raw_expands_canonical_and_records_placement(self):
        self.queue()
        result = self.analyze()
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(read_json(self.output.with_suffix('.raw.json')), self.raw)
        self.assertEqual(read_json(self.output), result)
        self.assertNotIn('wire_format', result)
        desk, screen = result['shared_scene']['base_scene']['objects']
        self.assertEqual(desk['position'], [0, 2, 0])
        self.assertEqual(screen['position'], [0, 2, .75])
        self.assertNotIn('placement', desk)
        validate(result)
        provenance = read_json(self.directory / 'analysis_provenance.json')
        encoding = provenance['scene_wire_encoding']
        self.assertEqual(encoding['format'], FORMAT_VERSION)
        self.assertEqual(encoding['placement_report'], placement_report(self.raw, expand_response(self.raw)))
        self.assertEqual(encoding['placement_report']['instances'][1]['support_top_z'], .74)
        self.assertEqual(encoding['placement_report']['instances'][1]['resolved_bottom_position'], [0, 2, .75])
        self.assertEqual(encoding['implementation_sha256'], vision.file_sha256(
            Path(vision.__file__).with_name('anchored_scene.py')))
        self.assertEqual(encoding['dependencies_sha256']['compact_scene.py'], vision.file_sha256(
            Path(vision.__file__).with_name('compact_scene.py')))
        self.assertFalse(provenance['manual_scene_seed'])
        request = self.server.requests[0]['body']
        content = request['messages'][0]['content']
        image_urls = [item['image_url']['url'] for item in content if item['type'] == 'image_url']
        self.assertEqual(len(image_urls), len(self.views))
        for url in image_urls:
            self.assertEqual(base64.b64decode(url.split(',', 1)[1]), self.image.read_bytes())
        self.assertIn(FORMAT_VERSION, content[0]['text'])

    def test_invalid_support_keeps_completed_response_without_fallback_or_repost(self):
        self.raw['shared_scene']['base_scene']['objects'][1]['placement']['support_id'] = 'unknown_support'
        self.queue()
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'exact existing base support instance'):
                self.analyze()
            self.assertEqual(len(self.server.requests), 1)
            self.assertFalse(self.output.exists())
            self.assertFalse((self.directory / 'analysis_provenance.json').exists())
            self.assertEqual(read_json(self.output.with_suffix('.raw.json')), self.raw)
        audit = read_json(self.output.with_suffix('.api.json'))
        self.assertEqual(audit['status'], 'completed')
        self.assertEqual(audit['attempts'], 1)

    def test_completed_api_response_recovers_after_local_commit_failure_without_post(self):
        self.queue()
        with patch.object(vision, 'normalize_response', side_effect=RuntimeError('local commit interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'local commit interrupted'):
                self.analyze()
        self.assertEqual(len(self.server.requests), 1)
        self.assertFalse(self.output.exists())
        raw_bytes = self.output.with_suffix('.raw.json').read_bytes()
        audit_bytes = self.output.with_suffix('.api.json').read_bytes()
        result = self.analyze()
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(result['shared_scene']['base_scene']['objects'][0]['position'][2], 0)
        self.assertEqual(self.output.with_suffix('.raw.json').read_bytes(), raw_bytes)
        self.assertEqual(self.output.with_suffix('.api.json').read_bytes(), audit_bytes)
        receipt = read_json(self.directory / 'analysis_provenance.json')['raw_model_response_cache']
        self.assertEqual(receipt['method'], 'same_stage_completed_api_response')
        self.assertEqual(receipt['raw_sha256'], vision.file_sha256(self.output.with_suffix('.raw.json')))

    def test_cross_wire_resume_is_rejected_without_mutating_completed_evidence(self):
        self.queue()
        self.analyze()
        original = {suffix: self.output.with_suffix(suffix).read_bytes()
                    for suffix in ('.raw.json', '.api.json', '.prompt.txt')}
        for wire_format in ('canonical', 'typed_parts_v1'):
            with self.subTest(wire_format=wire_format), self.assertRaises(ValueError):
                self.analyze(wire_format)
            self.assertEqual(len(self.server.requests), 1)
            for suffix, data in original.items():
                self.assertEqual(self.output.with_suffix(suffix).read_bytes(), data)

    def run_batch(self, config, groups, output, render):
        config_path = self.directory / 'factory.json'
        runner.atomic_json(config_path, config)
        command = ['--config', str(config_path), '--groups', str(groups), '--output-dir', str(output)]
        with patch.object(vision, 'call_model', side_effect=AssertionError('CLI fallback must not run')), \
             patch.object(runner, 'discover_blender', return_value='fixture-blender'), \
             patch.object(runner, 'run_stage', side_effect=render), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(command)

    def batch_inputs(self):
        groups = self.directory / 'groups.json'
        runner.atomic_json(groups, {'schema_version': 1, 'scenes': [
            {'scene_id': 'room', 'images': [str(self.image)]}]})
        # Batch identity is established from the supplied one-image group.
        raw = copy.deepcopy(self.raw)
        raw['views'] = [copy.deepcopy(raw['views'][0])]
        raw['views'][0]['id'] = 'view_01'
        raw['views'][0]['source_image'] = 'input:view_01'
        raw['shared_scene']['image']['source_id'] = 'view_01'
        config = {'schema_version': 1, 'models': {'analysis': self.model},
                  'pipeline': {'analysis_mode': 'monolithic', 'scene_wire_format': FORMAT_VERSION,
                               'validation_retries': 0},
                  'batch': {'workers': 1, 'max_repairs': 0, 'no_review': True},
                  'budget': {'max_requests': 1, 'max_output_tokens': 12288, 'max_reported_tokens': 0},
                  'render': {'max_edge': 128, 'samples': 16, 'tone_match': 'off', 'device': 'CPU'}}
        return groups, raw, config

    def test_batch_one_scene_renders_and_resumes_without_new_post_or_acceptance(self):
        groups, raw, config = self.batch_inputs()
        output = self.directory / 'output'
        self.queue(raw)
        renders = []
        def render(*args):
            renders.append(args)
            return batch_fixtures.GroupedBatchTests.fake_render(self, *args)
        self.assertEqual(self.run_batch(config, groups, output, render), 0)
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(len(renders), 1)
        scene_dir = output / 'room'
        raw_bytes = (scene_dir / 'initial_bundle.raw.json').read_bytes()
        self.assertEqual(read_json(scene_dir / 'initial_bundle.raw.json'), raw)
        self.assertEqual(read_json(output / 'accepted.json'), [])
        self.assertEqual(read_json(output / 'summary.json')['scenes'][0]['status'], 'generated_unreviewed')
        self.assertEqual(read_json(scene_dir / 'initial_bundle.json')['shared_scene']['base_scene']['objects'][1]['position'][2], .75)
        self.assertIn('placement_report', read_json(scene_dir / 'analysis_provenance.json')['scene_wire_encoding'])
        self.assertEqual(self.run_batch(config, groups, output, render), 0)
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(len(renders), 1)
        self.assertEqual((scene_dir / 'initial_bundle.raw.json').read_bytes(), raw_bytes)

    def test_batch_invalid_support_never_invokes_renderer_or_second_post(self):
        groups, raw, config = self.batch_inputs()
        raw['shared_scene']['base_scene']['objects'][1]['placement']['support_id'] = 'missing_support'
        self.queue(raw)
        output = self.directory / 'output'
        render = unittest.mock.Mock(side_effect=AssertionError('Invalid support must not render'))
        self.assertEqual(self.run_batch(config, groups, output, render), 1)
        self.assertEqual(self.run_batch(config, groups, output, render), 1)
        render.assert_not_called()
        self.assertEqual(len(self.server.requests), 1)
        self.assertFalse((output / 'room' / 'initial_bundle.json').exists())
        self.assertEqual(read_json(output / 'room' / 'initial_bundle.raw.json'), raw)
        self.assertEqual(read_json(output / 'summary.json')['scenes'][0]['status'], 'failed')


class AnchoredProfileTests(unittest.TestCase):
    def test_explicit_profile_disables_extra_model_calls_and_documents_status(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'project'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(['init', '--directory', str(project), '--profile', 'bigmodel-single-pass-anchored']), 0)
            config = load_config(project / 'factory.json')
            text = (project / 'README_运行.md').read_text()
        self.assertEqual(config['pipeline']['analysis_mode'], 'monolithic')
        self.assertEqual(config['pipeline']['scene_wire_format'], FORMAT_VERSION)
        self.assertEqual(config['pipeline']['validation_retries'], 0)
        self.assertTrue(config['batch']['no_review'])
        self.assertEqual(config['batch']['max_repairs'], 0)
        self.assertEqual(config['budget']['max_requests'], 1)
        self.assertEqual(config['budget']['max_output_tokens'], 12288)
        model = config['models']['analysis']
        self.assertEqual(model['model'], 'glm-5.3-flash')
        self.assertEqual(model['reasoning_effort'], 'low')
        self.assertEqual(model['max_retries'], 0)
        self.assertEqual(model['max_output_tokens'], 12288)
        self.assertIn('原始分辨率', text)
        self.assertIn('generated_unreviewed', text)
        self.assertIn('内置资产', text)
        self.assertIn('外部资产库会被拒绝', text)

    def test_external_asset_catalog_is_rejected_only_for_anchored_wire(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = init_project(Path(temp) / 'profile', 'bigmodel-single-pass-anchored')
            config = read_json(profile['config'])
        for library in ({'catalog': 'assets/catalog.json', 'bindings': 'bindings.json'},
                        {'catalog': 'assets/catalog.json', 'auto_bind': True},
                        {'catalog': 'assets/catalog.json', 'bindings': 'bindings.json', 'preview': True}):
            with self.subTest(library=library), self.assertRaisesRegex(ValueError, 'built-in assets only'):
                normalize_config({**config, 'asset_library': library})
        unchanged = {**config, 'pipeline': {**config['pipeline'], 'scene_wire_format': 'canonical'},
                     'asset_library': {'catalog': 'assets/catalog.json', 'bindings': 'bindings.json'}}
        self.assertEqual(normalize_config(unchanged)['asset_library']['catalog'], 'assets/catalog.json')
        for pipeline in ({'analysis_mode': 'staged', 'scene_wire_format': FORMAT_VERSION},
                         {'scene_wire_format': 'unknown'}):
            with self.subTest(pipeline=pipeline), self.assertRaises(ValueError):
                normalize_config({**config, 'pipeline': pipeline})


if __name__ == '__main__':
    unittest.main()
