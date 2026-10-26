"""Synthetic HTTP API -> staged analysis -> Blender -> paired review/edit tests.

These fixtures test autonomous orchestration, not remote GLM reconstruction
quality. Set SCENE_FACTORY_TEST_BLENDER to use a real Blender executable;
otherwise render artifacts are deterministic mocks with distinct iterations.
"""
from __future__ import annotations

import base64
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

import automatic_repair
import batch_multiview as runner
from batch import atomic_json, file_sha256, read_json
import vision_multiview as vision
from test_staged_analysis import fixture_plan, fixture_detail


class AutonomousPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for index in range(4):
            Image.new('RGB', (64, 48), (40 + index * 40, 110, 130)).save(self.root / f'{index}.png')
        self.groups = self.root / 'groups.json'
        atomic_json(self.groups, {'schema_version': 1, 'scenes': [
            {'scene_id': 'two_views', 'images': ['0.png', '1.png']},
            {'scene_id': 'single_a', 'images': ['2.png']},
            {'scene_id': 'single_b', 'images': ['3.png']}]})
        self.requests = []
        self.calls = {}
        self.plan_fail_once = False
        self.review_fail_once = False
        self.fail_final_review = False
        self.final_accept = False
        self.render_count = 0
        self.lock = threading.Lock()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                content = body['messages'][0]['content']
                prompt = content[0]['text']
                props = body['response_format']['json_schema']['schema']['properties']
                if 'room' in props and 'cameras' in props and 'groups' in props and 'assumptions' in props:
                    stage = 'plan'
                    data = owner.payload(prompt, 'TRUSTED INPUT IDENTITY (images in exactly this order):\n')
                    scene_id = data['scene_id']
                    result = fixture_plan(data['views'], group_count=1)
                    # Simple closed room for fast real Blender fixture renders.
                    result['omit'] = []
                    result['front_wall'] = False
                elif 'group_id' in props:
                    stage = 'detail'
                    data = owner.payload(prompt, 'TRUSTED INPUT IDENTITY (images in exactly this order):\n')
                    scene_id = data['scene_id']
                    result = fixture_detail(props['group_id']['enum'][0])
                elif 'verdict' in props:
                    stage = 'review'
                    data = owner.payload(prompt, 'UNTRUSTED CURRENT SCENE DATA (only identity/pairing is fixed):\n')
                    scene_id = data['scene_bundle']['scene_id']
                    changed = data['scene_bundle']['shared_scene']['render']['exposure'] > -.4
                    scores = {key: (95 if changed and owner.final_accept else 45 if changed else 68) for key in vision.DIMENSIONS}
                    result = {'schema_version': 3, 'verdict': 'recommend_accept' if changed and owner.final_accept else 'repair',
                        'summary': 'Synthetic second render assessment.' if changed else 'Synthetic first render assessment.',
                        'scores': scores, 'per_view_scores': [{'view_id': view['id'], 'scores': scores, 'summary': 'Synthetic fixture.'}
                                                           for view in data['scene_bundle']['views']],
                        'issues': [], 'asset_gaps': [], 'reference_limitations': ['Synthetic data; not a fidelity measurement.']}
                else:
                    stage = 'repair'
                    data = owner.payload(prompt, 'UNTRUSTED SCENE AND ASSESSMENT DATA:\n')
                    scene_id = data['scene_bundle']['scene_id']
                    result = automatic_repair.empty_edits('Synthetic API-authored exposure adjustment.')
                    render = copy.deepcopy(data['scene_bundle']['shared_scene']['render'])
                    render['exposure'] += .5
                    result['render'] = [render]
                with owner.lock:
                    key = (scene_id, stage)
                    owner.calls[key] = owner.calls.get(key, 0) + 1
                    attempt = owner.calls[key]
                    owner.requests.append({'body': body, 'stage': stage, 'scene_id': scene_id, 'payload': data,
                                           'image_sha256': [owner.image_hash(item) for item in content if item['type'] == 'image_url']})
                status = 200
                if stage == 'plan' and owner.plan_fail_once and attempt == 1:
                    result['cameras'].reverse()
                if stage == 'review' and ((owner.review_fail_once and attempt == 1) or (owner.fail_final_review and changed)):
                    status = 503
                response = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}],
                                       'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        model = {'provider': 'openai_chat', 'model': 'fixture-analysis',
                 'base_url': f'http://127.0.0.1:{self.server.server_port}/v1', 'api_key_env': '',
                 'response_format': 'json_schema', 'max_retries': 0, 'timeout_seconds': 20}
        self.config = {'schema_version': 1, 'models': {'analysis': model,
            'review': {**model, 'model': 'fixture-review'}, 'repair': {**model, 'model': 'fixture-repair'}},
            'batch': {'workers': 2, 'max_repairs': 1, 'no_review': False, 'vision_timeout': 20, 'render_timeout': 120},
            'render': {'max_edge': 128, 'samples': 8, 'tone_match': 'off', 'device': 'CPU'},
            'pipeline': {'analysis_mode': 'staged', 'repair_mode': 'typed', 'validation_retries': 1,
                         'staged': {'planner_timeout': 20, 'detail_timeout': 20, 'detail_workers': 2, 'validation_retries': 1}}}
        self.config_path = self.root / 'config.json'
        self.out = self.root / 'output'

    @staticmethod
    def payload(prompt, marker):
        return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]

    @staticmethod
    def image_hash(item):
        return hashlib.sha256(base64.b64decode(item['image_url']['url'].split(',', 1)[1])).hexdigest()

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def fake_render(self, command, log, timeout):
        self.render_count += 1
        out = Path(command[command.index('--output-dir') + 1])
        bundle = read_json(Path(command[command.index('--spec') + 1]))
        exposure = bundle['shared_scene']['render']['exposure']
        (out / 'scene.blend').write_bytes(b'BLENDER-synthetic-shared-scene' + str(exposure).encode() + b'\0' * 1400)
        outputs, views = ['scene.blend'], []
        max_edge = int(command[command.index('--max-edge') + 1])
        samples = int(command[command.index('--samples') + 1])
        width, height = max_edge, round(max_edge * .75)
        for view in bundle['views']:
            name, annotation = 'rgb_' + view['id'] + '.png', 'annotations_' + view['id'] + '.json'
            Image.new('RGB', (width, height), (80 + round(exposure * 30), 105, 120)).save(out / name)
            atomic_json(out / annotation, {'warnings': []})
            outputs.extend([name, annotation])
            views.append({'id': view['id'], 'render': name, 'annotation': annotation,
                          'width': width, 'height': height, 'warning_count': 0})
        atomic_json(out / 'metadata.json', {'status': 'completed', 'geometry_build_count': 1,
            'camera_count': len(views), 'views': views, 'outputs': outputs,
            'output_sha256': runner.artifact_hashes(out, outputs), 'render': {'samples': samples}})

    def run_batch(self, *, force_mock=False):
        atomic_json(self.config_path, self.config)
        command = ['--config', str(self.config_path), '--groups', str(self.groups), '--output-dir', str(self.out)]
        blender = os.environ.get('SCENE_FACTORY_TEST_BLENDER') if not force_mock else None
        with patch.object(vision, 'call_model', side_effect=AssertionError('Codex CLI forbidden')):
            if blender:
                return runner.main(command + ['--blender', blender])
            with patch.object(runner, 'discover_blender', return_value='synthetic-blender'), \
                 patch.object(runner, 'run_stage', side_effect=self.fake_render):
                return runner.main(command)

    def single_scene(self):
        atomic_json(self.groups, {'schema_version': 1, 'scenes': [{'scene_id': 'two_views', 'images': ['0.png', '1.png']}]})

    def results(self):
        return {item['scene_id']: item for item in read_json(self.out / 'summary.json')['scenes']}

    def test_grouped_three_scenes_two_iterations_pairing_latest_review_and_resume(self):
        self.plan_fail_once = True
        self.final_accept = True
        self.assertEqual(self.run_batch(), 0)
        results = self.results()
        self.assertEqual(set(results), {'two_views', 'single_a', 'single_b'})
        self.assertTrue(all(item['status'] == 'needs_review' for item in results.values()))
        self.assertEqual(len(list(self.out.rglob('scene.blend'))), 6)
        counts = {'two_views': 2, 'single_a': 1, 'single_b': 1}
        for scene_id, count in counts.items():
            item = results[scene_id]
            self.assertEqual(item['iteration'], 'iteration_01')
            self.assertEqual(item['final_review_iteration'], 'iteration_01')
            self.assertEqual(len(item['history']), 2)
            self.assertEqual(item['history'][-1]['scores']['realism'], 95)
            self.assertFalse(item['gate']['accepted'])
            self.assertTrue(any('preview_resolution_or_sampling_only' in reason for reason in item['gate']['reasons']))
            initial = read_json(self.out / scene_id / 'iteration_00/scene_bundle.json')
            final = read_json(self.out / scene_id / 'iteration_01/scene_bundle.json')
            self.assertEqual(len(final['views']), count)
            self.assertEqual(final['shared_scene']['render']['exposure'], initial['shared_scene']['render']['exposure'] + .5)
            for index in (0, 1):
                directory = self.out / scene_id / f'iteration_{index:02d}'
                prov = read_json(directory / 'review_provenance.json')
                self.assertEqual(prov['model']['model'], 'fixture-review')
                self.assertEqual(prov['input_hashes']['scene.blend'], file_sha256(directory / 'scene.blend'))
                self.assertEqual(prov['input_hashes']['rgb_view_01.png'], file_sha256(directory / 'rgb_view_01.png'))
                self.assertEqual(len(prov['model_image_inputs']), count)
                for view_index, image_input in enumerate(prov['model_image_inputs']):
                    render = directory / f'rgb_view_{view_index + 1:02d}.png'
                    self.assertEqual(image_input['render_sha256'], file_sha256(render))
                    self.assertEqual(prov['input_hashes'][render.name], file_sha256(render))
                    model_input = Path(image_input['model_input_path'])
                    self.assertEqual(image_input['model_input_sha256'], file_sha256(model_input))
                    self.assertEqual(image_input['model_input_dimensions_px'], [64, 48])
                self.assertTrue(prov['final_assessment_only'])
            repair_prov = read_json(self.out / scene_id / 'iteration_00/repair_provenance.json')
            self.assertEqual(repair_prov['model']['model'], 'fixture-repair')
            self.assertFalse(repair_prov['manual_geometry_edits'])
            self.assertFalse(repair_prov['manual_visual_feedback'])
        for request in self.requests:
            count = counts[request['scene_id']]
            paired = request['stage'] in ('review', 'repair')
            self.assertEqual(len(request['image_sha256']), count * (2 if paired else 1))
            self.assertEqual(request['body']['model'], {'plan': 'fixture-analysis', 'detail': 'fixture-analysis',
                                                       'review': 'fixture-review', 'repair': 'fixture-repair'}[request['stage']])
            if paired:
                data = request['payload']
                self.assertEqual(data['image_pairs'], [{'view_id': f'view_{index + 1:02d}',
                    'original_image_number': index * 2 + 1, 'render_image_number': index * 2 + 2} for index in range(count)])
                exposure = data['scene_bundle']['shared_scene']['render']['exposure']
                iteration = 'iteration_01' if exposure > -.4 else 'iteration_00'
                for index in range(count):
                    source = self.out / request['scene_id'] / 'sources' / f'view_{index + 1:02d}' / 'reference.png'
                    render_directory = self.out / request['scene_id'] / iteration
                    render = render_directory / f'rgb_view_{index + 1:02d}.png'
                    proxy = render_directory / 'model_input_images' / render.name
                    evidence_name = 'review_provenance.json' if request['stage'] == 'review' else 'repair_provenance.json'
                    provenance = read_json(render_directory / evidence_name)
                    image_input = provenance['model_image_inputs'][index]
                    self.assertEqual(request['image_sha256'][2 * index], file_sha256(source))
                    self.assertEqual(image_input['render_sha256'], file_sha256(render))
                    self.assertEqual(image_input['model_input_sha256'], file_sha256(proxy))
                    self.assertEqual(Path(image_input['model_input_path']), proxy.resolve())
                    self.assertEqual(request['image_sha256'][2 * index + 1], file_sha256(proxy))
        self.assertEqual(self.calls[('two_views', 'plan')], 2)
        request_count = len(self.requests)
        render_count = self.render_count
        self.assertEqual(self.run_batch(), 0)
        self.assertEqual(len(self.requests), request_count)
        self.assertEqual(self.render_count, render_count)
        self.assertTrue(all(item.get('resumed') for item in self.results().values()))

    def test_review_failure_preserves_blend_then_resume_retries_only_review(self):
        self.single_scene()
        self.config['batch']['max_repairs'] = 0
        self.review_fail_once = True
        self.assertEqual(self.run_batch(), 1)
        item = self.results()['two_views']
        self.assertEqual(item['status'], 'review_failed')
        self.assertFalse(item['gate']['accepted'])
        self.assertEqual(item['failed_stage'], 'joint_reference_review')
        blend = self.out / 'two_views/iteration_00/scene.blend'
        self.assertGreater(blend.stat().st_size, 1000)
        digest, count = file_sha256(blend), self.render_count
        self.assertEqual(self.run_batch(), 0)
        self.assertEqual(file_sha256(blend), digest)
        self.assertEqual(self.render_count, count)
        self.assertEqual(self.calls[('two_views', 'plan')], 1)
        self.assertEqual(self.calls[('two_views', 'detail')], 1)
        self.assertEqual(self.calls[('two_views', 'review')], 2)
        self.assertEqual(self.results()['two_views']['status'], 'needs_review')

    def test_final_review_failure_keeps_reviewed_best_but_never_inherits_acceptance(self):
        self.single_scene()
        self.fail_final_review = True
        self.assertEqual(self.run_batch(), 1)
        item = self.results()['two_views']
        self.assertEqual(item['status'], 'review_failed')
        # The best independently reviewed artifact remains visible, while the
        # failed current attempt is explicit and cannot inherit acceptance.
        self.assertEqual(item['iteration'], 'iteration_00')
        self.assertEqual(item['failed_iteration'], 'iteration_01')
        self.assertFalse(item['gate']['accepted'])
        self.assertEqual(item['final_review_iteration'], 'iteration_00')
        self.assertEqual(item['summary'], item['history'][0]['summary'])
        self.assertEqual(item['gate']['reasons'], ['pipeline_error:joint_reference_review'])
        self.assertEqual(len(item['history']), 1)
        self.assertEqual(len(list(self.out.rglob('scene.blend'))), 2)
        self.fail_final_review = False
        count = self.render_count
        self.assertEqual(self.run_batch(), 0)
        item = self.results()['two_views']
        # The second review completes on resume, but its score 45 regresses
        # from 68. Retain the reviewed incumbent with its own matching review.
        self.assertEqual(item['iteration'], 'iteration_00')
        self.assertEqual(item['final_review_iteration'], 'iteration_00')
        self.assertEqual(item['history'][-1]['scores']['realism'], 45)
        self.assertFalse(item['history'][-1]['selected_as_best'])
        self.assertFalse(item['gate']['accepted'])
        self.assertEqual(self.calls[('two_views', 'repair')], 1)
        self.assertEqual(self.render_count, count)

    def test_acceptance_requires_current_render_and_adequate_inspection_resolution(self):
        self.single_scene()
        self.config['render'].update(max_edge=1280, samples=64)
        self.final_accept = True
        self.assertEqual(self.run_batch(force_mock=True), 0)
        item = self.results()['two_views']
        self.assertEqual(item['status'], 'accepted')
        self.assertEqual(item['iteration'], 'iteration_01')
        self.assertEqual(item['final_review_iteration'], 'iteration_01')
        self.assertEqual(self.calls[('two_views', 'review')], 2)
        self.assertEqual(self.calls[('two_views', 'repair')], 1)
        self.assertFalse(item['history'][0]['gate']['accepted'])
        self.assertTrue(item['history'][1]['gate']['accepted'])


if __name__ == '__main__':
    unittest.main()
