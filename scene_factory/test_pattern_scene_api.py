"""Pattern-wire single-request, recovery and resume contracts over loopback HTTP.

No provider or real Blender is invoked. Synthetic API responses and mocked
render artifacts establish protocol, identity, budget and resume behavior;
they are deliberately not evidence of photo fidelity or actual token savings.
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
from unittest.mock import Mock, patch

from PIL import Image

from batch import read_json
import batch_multiview as runner
from factory_config import normalize_config
import model_api
from multiview_spec import validate
import pattern_scene
from scene_factory.cli import init_project, main as cli_main
from scene_factory.factory_config import load_config
from spec_v2 import part
from test_anchored_scene import base_object, top, wire
import test_batch_multiview as batch_fixtures
from test_compact_scene_api import complete_bundle
from test_model_api import FixtureServer, Handler, chat
import vision_multiview as vision


FORMAT_VERSION = pattern_scene.FORMAT_VERSION


class PatternSceneAPITests(unittest.TestCase):
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
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.image = self.directory / 'reference.png'
        # Deliberately above common 1024/1600 thumbnail cutoffs. Uploaded bytes
        # must equal this file, independent of the tiny mocked render size.
        Image.new('RGB', (1601, 1201), (82, 106, 143)).save(self.image)
        self.model = {'provider': 'openai_chat', 'model': 'pattern-fixture',
                      'base_url': f'http://127.0.0.1:{self.server.server_port}/v1',
                      'api_key_env': '', 'max_retries': 0, 'max_output_tokens': 12288}
        self.server.requests, self.server.responses = [], []
        self.bundle = complete_bundle()
        scene = self.bundle['shared_scene']
        scene['base_scene']['objects'] = [base_object(position=[0, 2, .72]),
            base_object('screen', 'monitor', dimensions=[.6, .2, .45])]
        scene['custom_objects'][0]['parts'] = [
            part('box', size=[1, .3, .04], position=[0, 0, .4]),
            part('box', size=[.04, .3, 1.2], position=[-.48, 0, .6]),
            part('mesh', position=[0, 0, 0],
                 vertices=[[0, 0, 0], [.3, 0, 0], [0, .2, 0]], faces=[[0, 1, 2]])]
        for view in self.bundle['views']:
            view['width'], view['height'] = 1601, 1201
        first = self.bundle['views'][0]
        scene['image'] = {'width': first['width'], 'height': first['height'],
                          'source_id': first['id']}
        self.views = [{**{key: view[key] for key in ('id', 'width', 'height')},
                       'source_image': str(self.image)} for view in self.bundle['views']]
        self.raw = wire(self.bundle, {'desk': {'mode': 'floor', 'xy': [0, 2]},
                                     'screen': top(clearance=.01)})
        self.raw['wire_format'] = FORMAT_VERSION
        self.raw['shared_scene']['custom_objects'][0]['part_arrays'] = [
            {'part_indices': [0], 'count': [1, 1, 3], 'spacing': [0, 0, .4]},
            {'part_indices': [1], 'count': [2, 1, 1], 'spacing': [.96, 0, 0]}]
        self.output = self.directory / 'initial_bundle.json'

    def queue(self, response=None):
        self.server.responses.append((200, {}, chat(json.dumps(
            self.raw if response is None else response))))

    def analyze(self, wire_format=FORMAT_VERSION):
        return vision.analyze('room', self.views, self.output, model_config=self.model,
                              wire_format=wire_format)

    def test_one_post_keeps_original_image_bytes_and_expands_array_provenance(self):
        self.queue()
        result = self.analyze()
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(read_json(self.output.with_suffix('.raw.json')), self.raw)
        self.assertEqual(read_json(self.output), result)
        self.assertNotIn('wire_format', result)
        parts = result['shared_scene']['custom_objects'][0]['parts']
        self.assertEqual(len(parts), 6)
        self.assertEqual([p['position'] for p in parts[:3]],
                         [[0, 0, .4], [0, 0, .8], [0, 0, 1.2000000000000002]])
        self.assertEqual([p['position'] for p in parts[3:5]], [[-.48, 0, .6], [.48, 0, .6]])
        self.assertEqual(parts[5]['faces'], [[0, 1, 2]])
        self.assertNotIn('part_arrays', result['shared_scene']['custom_objects'][0])
        desk, screen = result['shared_scene']['base_scene']['objects']
        self.assertEqual(desk['position'], [0, 2, 0])
        self.assertEqual(screen['position'], [0, 2, .75])
        validate(result)

        provenance = read_json(self.directory / 'analysis_provenance.json')
        encoding = provenance['scene_wire_encoding']
        expanded = pattern_scene.expand_response(self.raw)
        self.assertEqual(encoding['format'], FORMAT_VERSION)
        self.assertEqual(encoding['pattern_report'], pattern_scene.pattern_report(self.raw, expanded))
        self.assertEqual(encoding['placement_report'], pattern_scene.placement_report(self.raw, expanded))
        self.assertEqual(encoding['pattern_report']['counts']['source_parts'], 3)
        self.assertEqual(encoding['pattern_report']['counts']['expanded_parts'], 6)
        self.assertEqual(encoding['implementation_sha256'], vision.file_sha256(
            Path(vision.__file__).with_name('pattern_scene.py')))
        for name in ('anchored_scene.py', 'compact_scene.py', 'pattern_prompt.py'):
            self.assertEqual(encoding['dependencies_sha256'][name], vision.file_sha256(
                Path(vision.__file__).with_name(name)))
        self.assertFalse(provenance['manual_scene_seed'])
        self.assertIn('not provider tokens', encoding['size_measurement'])
        self.assertEqual(provenance, read_json(self.directory / 'initial_bundle_provenance.json'))

        request = self.server.requests[0]['body']
        content = request['messages'][0]['content']
        urls = [item['image_url']['url'] for item in content if item['type'] == 'image_url']
        self.assertEqual(len(urls), len(self.views))
        for url in urls:
            self.assertEqual(base64.b64decode(url.split(',', 1)[1]), self.image.read_bytes())
        self.assertIn(FORMAT_VERSION, content[0]['text'])
        schema = request['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['wire_format']['enum'], [FORMAT_VERSION])
        group_rule = schema['properties']['shared_scene']['properties']['custom_objects']['items']
        self.assertIn('part_arrays', group_rule['required'])

    def test_analysis_provenance_records_api_effort_without_cli_default(self):
        for configured in ('low', None):
            self.output = self.directory / str(configured) / 'initial_bundle.json'
            self.model.pop('reasoning_effort', None)
            if configured is not None:
                self.model['reasoning_effort'] = configured
            self.queue()
            self.analyze()
            provenance = read_json(self.output.parent / 'analysis_provenance.json')
            self.assertEqual(provenance['reasoning_effort'], configured)
            sent = self.server.requests[-1]['body']
            self.assertEqual(sent.get('reasoning_effort'), configured)

    def test_invalid_array_retains_completed_evidence_and_never_reposts(self):
        arrays = self.raw['shared_scene']['custom_objects'][0]['part_arrays']
        arrays.append(copy.deepcopy(arrays[0]))
        self.queue()
        evidence = None
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'selected more than once'):
                self.analyze()
            self.assertEqual(len(self.server.requests), 1)
            self.assertFalse(self.output.exists())
            self.assertFalse((self.directory / 'analysis_provenance.json').exists())
            self.assertEqual(read_json(self.output.with_suffix('.raw.json')), self.raw)
            current = {suffix: self.output.with_suffix(suffix).read_bytes()
                       for suffix in ('.raw.json', '.api.json', '.prompt.txt')}
            if evidence is not None:
                self.assertEqual(current, evidence)
            evidence = current
        audit = read_json(self.output.with_suffix('.api.json'))
        self.assertEqual(audit['status'], 'completed')
        self.assertEqual(audit['attempts'], 1)

    def test_completed_api_recovers_after_canonical_commit_failure_without_new_post(self):
        self.queue()
        original_write = vision.atomic_json
        def interrupted_write(path, value):
            if Path(path) == self.output:
                raise OSError('local canonical commit interrupted')
            return original_write(path, value)
        with patch.object(vision, 'atomic_json', side_effect=interrupted_write):
            with self.assertRaisesRegex(OSError, 'local canonical commit interrupted'):
                self.analyze()
        self.assertEqual(len(self.server.requests), 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(read_json(self.output.with_suffix('.api.json'))['status'], 'completed')
        before = {suffix: self.output.with_suffix(suffix).read_bytes()
                  for suffix in ('.raw.json', '.api.json', '.prompt.txt')}
        with patch.object(model_api, '_request', side_effect=AssertionError('Paid request must not repeat')):
            result = self.analyze()
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(len(result['shared_scene']['custom_objects'][0]['parts']), 6)
        for suffix, contents in before.items():
            self.assertEqual(self.output.with_suffix(suffix).read_bytes(), contents)
        provenance = read_json(self.directory / 'analysis_provenance.json')
        receipt = provenance['raw_model_response_cache']
        self.assertEqual(receipt['method'], 'same_stage_completed_api_response')
        self.assertEqual(receipt['raw_sha256'], vision.file_sha256(self.output.with_suffix('.raw.json')))
        self.assertIn('pattern_report', provenance['scene_wire_encoding'])

    def test_cross_wire_reuse_fails_without_post_or_mutating_evidence(self):
        self.queue()
        self.analyze()
        before = {suffix: self.output.with_suffix(suffix).read_bytes()
                  for suffix in ('.raw.json', '.api.json', '.prompt.txt', '.json')}
        for wire_format in ('canonical', 'typed_parts_v1', 'anchored_parts_v1'):
            with self.subTest(wire_format=wire_format), self.assertRaises(ValueError):
                self.analyze(wire_format)
            self.assertEqual(len(self.server.requests), 1)
            for suffix, contents in before.items():
                self.assertEqual(self.output.with_suffix(suffix).read_bytes(), contents)

    def test_custom_support_normalization_still_rejects_unknown_and_cyclic_refs(self):
        original = copy.deepcopy(self.raw)
        for case in ('unknown', 'cycle'):
            self.raw = copy.deepcopy(original)
            group = self.raw['shared_scene']['custom_objects'][0]
            group['support_id'] = 'missing' if case == 'unknown' else group['id']
            self.output = self.directory / case / 'initial_bundle.json'
            self.queue()
            requests_before = len(self.server.requests)
            for _ in range(2):
                with self.subTest(case=case), self.assertRaises(ValueError):
                    self.analyze()
                self.assertFalse(self.output.exists())
                self.assertEqual(len(self.server.requests), requests_before + 1)

    def batch_inputs(self):
        groups = self.directory / 'groups.json'
        runner.atomic_json(groups, {'schema_version': 1, 'scenes': [
            {'scene_id': 'room', 'images': [str(self.image)]}]})
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

    def run_batch(self, config, groups, output, render):
        config_path = self.directory / 'factory.json'
        runner.atomic_json(config_path, config)
        command = ['--config', str(config_path), '--groups', str(groups), '--output-dir', str(output)]
        with patch.object(vision, 'call_model', side_effect=AssertionError('CLI fallback must not run')), \
             patch.object(runner, 'discover_blender', return_value='fixture-blender'), \
             patch.object(runner, 'run_stage', side_effect=render), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(command)

    def test_batch_completes_then_resume_uses_no_api_no_render_and_stays_unreviewed(self):
        groups, raw, config = self.batch_inputs()
        output = self.directory / 'output'
        self.queue(raw)
        render = Mock(side_effect=lambda *args: batch_fixtures.GroupedBatchTests.fake_render(self, *args))
        self.assertEqual(self.run_batch(config, groups, output, render), 0)
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(render.call_count, 1)
        scene_dir = output / 'room'
        result = read_json(scene_dir / 'result.json')
        self.assertEqual(result['status'], 'generated_unreviewed')
        self.assertEqual(read_json(output / 'accepted.json'), [])
        self.assertEqual(read_json(scene_dir / 'initial_bundle.raw.json'), raw)
        self.assertIn('pattern_report', read_json(scene_dir / 'analysis_provenance.json')['scene_wire_encoding'])
        immutable = {name: (scene_dir / name).read_bytes()
                     for name in set(result['artifacts']) | {'result.json', 'scene_budget.json'}}
        no_render = Mock(side_effect=AssertionError('Completed scene must not rerender'))
        with patch.object(model_api, 'call_model', side_effect=AssertionError('Completed scene must not call model')), \
             patch.object(model_api, '_request', side_effect=AssertionError('Completed scene must not POST')):
            self.assertEqual(self.run_batch(config, groups, output, no_render), 0)
        no_render.assert_not_called()
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(read_json(output / 'accepted.json'), [])
        self.assertTrue(read_json(output / 'summary.json')['scenes'][0]['resumed'])
        for name, contents in immutable.items():
            self.assertEqual((scene_dir / name).read_bytes(), contents)

    def test_batch_invalid_array_never_renders_or_sends_second_post(self):
        groups, raw, config = self.batch_inputs()
        raw['shared_scene']['custom_objects'][0]['part_arrays'][0]['spacing'] = [0, 0, 0]
        self.queue(raw)
        output = self.directory / 'output'
        render = Mock(side_effect=AssertionError('Invalid array must not render'))
        self.assertEqual(self.run_batch(config, groups, output, render), 1)
        self.assertEqual(self.run_batch(config, groups, output, render), 1)
        self.assertEqual(len(self.server.requests), 1)
        render.assert_not_called()
        self.assertFalse((output / 'room' / 'initial_bundle.json').exists())
        self.assertEqual(read_json(output / 'room' / 'initial_bundle.raw.json'), raw)
        self.assertEqual(read_json(output / 'summary.json')['scenes'][0]['status'], 'failed')
        self.assertEqual(read_json(output / 'accepted.json'), [])

    def test_pattern_or_prompt_fingerprint_change_blocks_completed_batch_reuse(self):
        groups, raw, config = self.batch_inputs()
        output = self.directory / 'output'
        self.queue(raw)
        render = lambda *args: batch_fixtures.GroupedBatchTests.fake_render(self, *args)
        self.assertEqual(self.run_batch(config, groups, output, render), 0)
        original = runner.pipeline_hashes()
        scene_dir = output / 'room'
        saved_result = (scene_dir / 'result.json').read_bytes()
        saved_raw = (scene_dir / 'initial_bundle.raw.json').read_bytes()
        for name in ('pattern_scene.py', 'pattern_prompt.py'):
            changed = {**original, name: '0' * 64}
            forbidden_render = Mock(side_effect=AssertionError('Changed implementation must not render'))
            with self.subTest(name=name), patch.object(runner, 'pipeline_hashes', return_value=changed), \
                 patch.object(model_api, '_request', side_effect=AssertionError('Changed implementation must not POST')):
                self.assertEqual(self.run_batch(config, groups, output, forbidden_render), 1)
            forbidden_render.assert_not_called()
            self.assertEqual(len(self.server.requests), 1)
            self.assertEqual((scene_dir / 'result.json').read_bytes(), saved_result)
            self.assertEqual((scene_dir / 'initial_bundle.raw.json').read_bytes(), saved_raw)
            self.assertIn('Input, code or options changed', (scene_dir / 'last_error.json').read_text())


class PatternProfileTests(unittest.TestCase):
    def test_explicit_profile_disables_extra_calls_preserves_resolution_and_documents_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / 'project'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(['init', '--directory', str(project),
                                           '--profile', 'bigmodel-single-pass-patterns']), 0)
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
        self.assertIn('part_arrays', text)
        self.assertIn('外部资产库会被拒绝', text)
        self.assertIn('没有同模型对照证明', text)

    def test_mode_is_explicit_monolithic_and_built_in_assets_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = init_project(Path(temporary) / 'project', 'bigmodel-single-pass-patterns')
            config = read_json(profile['config'])
        for library in ({'catalog': 'assets/catalog.json', 'bindings': 'bindings.json'},
                        {'catalog': 'assets/catalog.json', 'auto_bind': True}):
            with self.subTest(library=library), self.assertRaisesRegex(ValueError, 'built-in assets only'):
                normalize_config({**config, 'asset_library': library})
        with self.assertRaisesRegex(ValueError, 'requires monolithic'):
            normalize_config({**config, 'pipeline': {**config['pipeline'], 'analysis_mode': 'staged'}})
        defaults = normalize_config({'schema_version': 1, 'models': config['models']})
        self.assertEqual(defaults['pipeline']['scene_wire_format'], 'canonical')

    def test_pipeline_fingerprints_cover_pattern_decoder_prompt_and_dependency_files(self):
        hashes = runner.pipeline_hashes()
        for name in ('pattern_scene.py', 'pattern_prompt.py', 'anchored_scene.py', 'compact_scene.py'):
            self.assertIn(name, hashes)
            self.assertEqual(hashes[name], vision.file_sha256(Path(runner.__file__).with_name(name)))


if __name__ == '__main__':
    unittest.main()
