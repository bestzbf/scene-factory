"""Synthetic HTTP model -> rejected geometry -> typed API repair -> review.

Default Blender output is explicitly mocked. SCENE_FACTORY_TEST_BLENDER enables
real geometry gates and renders; SCENE_FACTORY_PREFLIGHT_TEST_OUTPUT retains
evidence under a NEW local directory. These tests never prove GLM photo quality.
"""
from __future__ import annotations

import base64
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from multiview_spec import validate
from staged_analysis import _expand
from verify_production import fixture_plan, fixture_detail, _mock_render
import vision_multiview as vision


class PreflightAPIIntegrationTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get('SCENE_FACTORY_PREFLIGHT_TEST_OUTPUT')
        if retained:
            self.root = Path(retained).resolve() / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name).resolve()
        self.retained = bool(retained)
        for index in range(3):
            Image.new('RGB', (64, 48), (80 + index * 20, 105, 125)).save(self.root / f'{index}.png')
        self.groups_path, self.out = self.root / 'groups.json', self.root / 'output'
        self.scene_groups = [{'scene_id': 'synthetic_accepted', 'images': ['0.png', '1.png']},
                             {'scene_id': 'synthetic_rejected', 'images': ['2.png']}]
        self.requests, self.server_errors, self.builds = [], [], []
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                try:
                    if self.path != '/v1/chat/completions':
                        raise AssertionError('Unexpected local fixture endpoint')
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    answer, record = owner.reply(body)
                    with owner.lock:
                        owner.requests.append(record)
                    payload = json.dumps({'choices': [{'finish_reason': 'stop',
                        'message': {'content': json.dumps(answer)}}],
                        'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as exc:
                    with owner.lock:
                        owner.server_errors.append(str(exc))
                    self.send_error(500, 'Synthetic API fixture contract error')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        model = {'provider': 'openai_chat', 'base_url': f'http://127.0.0.1:{self.server.server_port}/v1',
                 'model': 'synthetic-analysis', 'api_key_env': '', 'response_format': 'json_schema',
                 'max_retries': 0, 'timeout_seconds': 30}
        self.config = {'schema_version': 1,
            'models': {role: {**model, 'model': 'synthetic-' + role} for role in ('analysis', 'repair', 'review')},
            'batch': {'workers': 1, 'max_repairs': 1, 'no_review': False, 'vision_timeout': 30,
                      'render_timeout': 2400},
            'render': {'max_edge': 1280, 'samples': 64, 'tone_match': 'off', 'device': 'AUTO'},
            'pipeline': {'analysis_mode': 'monolithic', 'repair_mode': 'typed', 'validation_retries': 0}}
        self.config_path = self.root / 'factory.json'
        self.original_run_stage = runner.run_stage

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @staticmethod
    def payload(prompt, marker):
        return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]

    def reply(self, body):
        content = body['messages'][0]['content']
        prompt = content[0]['text']
        if 'UNTRUSTED TECHNICAL REPAIR DATA:\n' in prompt:
            stage = 'technical_repair'
            data = self.payload(prompt, 'UNTRUSTED TECHNICAL REPAIR DATA:\n')
            bundle = data['scene_bundle']
            scene_id, views = bundle['scene_id'], bundle['views']
            reports = data['technical_preflight']
            self.assertTrue(any(r['status'] == 'rejected' and r['errors'] for r in reports.values()))
            edits = automatic_repair.empty_edits('Synthetic API geometry correction; independent review still required.')
            for index, view in enumerate(views):
                camera = copy.deepcopy(view['camera'])
                camera['position'] = [-.45 + index * .9, .4, 1.65]
                edits['cameras'].append({'view_id': view['id'], 'camera': camera})
            if scene_id != 'synthetic_budget_exhausted':
                desk = copy.deepcopy(bundle['shared_scene']['base_scene']['objects'][0])
                desk['position'] = [-1, 3, 0]
                edits['base_objects'] = [desk]
            answer = edits
            self.assertEqual(data['original_images'], [{'view_id': v['id'], 'image_number': i + 1}
                                                       for i, v in enumerate(views)])
        elif 'UNTRUSTED CURRENT SCENE DATA (only identity/pairing is fixed):\n' in prompt:
            stage = 'review'
            data = self.payload(prompt, 'UNTRUSTED CURRENT SCENE DATA (only identity/pairing is fixed):\n')
            bundle = data['scene_bundle']
            scene_id, views = bundle['scene_id'], bundle['views']
            accepted = scene_id == 'synthetic_accepted'
            scores = {key: 95 if accepted else 40 for key in vision.DIMENSIONS}
            answer = {'schema_version': 3, 'verdict': 'recommend_accept' if accepted else 'asset_limited',
                'summary': 'Synthetic API fixture verdict; not photograph fidelity evidence.',
                'scores': scores, 'per_view_scores': [{'view_id': view['id'], 'scores': scores.copy(),
                    'summary': 'Synthetic review score.'} for view in views], 'issues': [],
                'asset_gaps': [] if accepted else ['Synthetic deliberate rejection.'],
                'reference_limitations': ['Local test images and model outputs are synthetic, not GLM outputs.']}
            self.assertEqual(data['image_pairs'], [{'view_id': view['id'], 'original_image_number': 2*i + 1,
                'render_image_number': 2*i + 2} for i, view in enumerate(views)])
        else:
            stage = 'analysis'
            data = self.payload(prompt, 'TRUSTED INPUT IDENTITY (images in exactly this order):\n')
            scene_id, views = data['scene_id'], data['views']
            plan = fixture_plan(views)
            # Schema-valid model output that is physically invalid. Monolithic
            # analysis must be protected by the same real Blender preflight.
            for camera in plan['cameras']:
                camera['d'][0] = plan['room']['d'][0] / 2
            plan['assets'][0]['d'][0] = 4.0
            answer = validate(_expand(plan, views, scene_id, {'cabinet': fixture_detail('cabinet')}))
        image_urls = [item['image_url']['url'] for item in content if item['type'] == 'image_url']
        hashes = [hashlib.sha256(base64.b64decode(value.split(',', 1)[1], validate=True)).hexdigest()
                  for value in image_urls]
        expected, render_inputs = [], []
        for view in views:
            expected.append(file_sha256(self.out / scene_id / 'sources' / view['id'] / 'reference.png'))
            if stage == 'review':
                iteration = self.out / scene_id / 'iteration_01'
                render = iteration / f'rgb_{view["id"]}.png'
                proxy = iteration / 'model_input_images' / render.name
                model_input = proxy if proxy.is_file() else render
                expected.append(file_sha256(model_input))
                render_inputs.append({'render_path': str(render.resolve()),
                    'render_sha256': file_sha256(render),
                    'model_input_path': str(model_input.resolve()),
                    'model_input_sha256': file_sha256(model_input)})
        self.assertEqual(hashes, expected)
        self.assertEqual(body['model'], 'synthetic-' + ('repair' if stage == 'technical_repair' else stage))
        return answer, {'scene_id': scene_id, 'stage': stage, 'model': body['model'],
                        'image_count': len(hashes), 'ordered_image_sha256': hashes,
                        'model_image_inputs': render_inputs}

    def build(self, command, log, timeout):
        spec_path = Path(command[command.index('--spec') + 1])
        directory = Path(command[command.index('--output-dir') + 1])
        bundle = read_json(spec_path)
        self.builds.append(str(directory.relative_to(self.root)))
        if os.environ.get('SCENE_FACTORY_TEST_BLENDER'):
            return self.original_run_stage(command, log, timeout)
        # The default fixture emits the exact runtime report contract and then
        # raises just as Blender does. It never pretends to test geometry math.
        invalid_camera = any(v['camera']['position'][0] >= 2.8 for v in bundle['views'])
        invalid_desk = bundle['shared_scene']['base_scene']['objects'][0]['position'][0] > 3
        if invalid_camera or invalid_desk:
            errors = []
            if invalid_camera:
                errors.append({'code': 'camera_room_shell_collision', 'view_id': bundle['views'][0]['id']})
            if invalid_desk:
                errors.append({'code': 'geometry_outside_preserved_room', 'object_id': 'desk'})
            atomic_json(directory / 'geometry_preflight.json', {'schema_version': 1, 'status': 'rejected',
                'spec_sha256': file_sha256(spec_path), 'errors': errors, 'warnings': [], 'synthetic_mock_gate': True})
            Path(log).write_text('Synthetic mocked Blender preflight failure; no renderer ran.\n')
            raise RuntimeError('Synthetic Blender preflight rejection')
        return _mock_render(command, log, timeout)

    def run_batch(self):
        atomic_json(self.groups_path, {'schema_version': 1, 'scenes': self.scene_groups})
        atomic_json(self.config_path, self.config)
        args = ['--config', str(self.config_path), '--groups', str(self.groups_path), '--output-dir', str(self.out)]
        blender = os.environ.get('SCENE_FACTORY_TEST_BLENDER')
        if blender:
            args += ['--blender', blender]
        with patch.object(vision, 'call_model', side_effect=AssertionError('Codex CLI must never be used')), \
             patch.object(runner, 'run_stage', side_effect=self.build):
            if blender:
                code = runner.main(args)
            else:
                with patch.object(runner, 'discover_blender', return_value='synthetic-blender'):
                    code = runner.main(args)
        self.assertEqual(self.server_errors, [])
        atomic_json(self.root / 'http_requests.json', self.requests)
        atomic_json(self.root / 'build_invocations.json', self.builds)
        return code

    def results(self):
        return {item['scene_id']: item for item in read_json(self.out / 'summary.json')['scenes']}

    def evidence(self, **values):
        atomic_json(self.root / 'validation.json', {'status': 'passed', 'synthetic_fixture': True,
            'real_blender': bool(os.environ.get('SCENE_FACTORY_TEST_BLENDER')),
            'external_provider_calls': 0, 'photograph_reconstruction_validated': False,
            'real_glm_visual_quality_validated': False, 'formal_dataset_release': False,
            'scope': 'Synthetic HTTP API orchestration, repair budget and accepted-only isolation.', **values})

    def test_api_preflight_repair_rebuild_independent_review_and_resume(self):
        self.assertEqual(self.run_batch(), 0)
        results = self.results()
        self.assertEqual({k: v['status'] for k, v in results.items()},
                         {'synthetic_accepted': 'accepted', 'synthetic_rejected': 'needs_review'})
        self.assertEqual([r['scene_id'] for r in read_json(self.out / 'accepted.json')], ['synthetic_accepted'])
        self.assertEqual(len(self.builds), 4)
        for scene_id in results:
            self.assertEqual([r['stage'] for r in self.requests if r['scene_id'] == scene_id],
                             ['analysis', 'technical_repair', 'review'])
            folder = self.out / scene_id
            self.assertFalse((folder / 'iteration_00/scene.blend').exists())
            self.assertFalse((folder / 'iteration_00/review.json').exists())
            self.assertTrue((folder / 'iteration_01/scene.blend').is_file())
            self.assertEqual(results[scene_id]['final_review_iteration'], 'iteration_01')
            prov = read_json(folder / 'iteration_00/technical_repair_provenance.json')
            self.assertEqual(prov['input_renders'], 0)
            self.assertTrue(prov['candidate_requires_new_render_and_review'])
            self.assertFalse(prov['manual_geometry_edits'])
            self.assertEqual(prov['model']['model'], 'synthetic-repair')
            review = read_json(folder / 'iteration_01/review_provenance.json')
            self.assertEqual(review['model']['model'], 'synthetic-review')
            self.assertEqual(review['input_hashes']['scene.blend'], file_sha256(folder / 'iteration_01/scene.blend'))
            for image_input in review['model_image_inputs']:
                render = folder / 'iteration_01' / f'rgb_{image_input["view_id"]}.png'
                model_input = Path(image_input['model_input_path'])
                self.assertEqual(image_input['render_sha256'], file_sha256(render))
                self.assertEqual(review['input_hashes'][render.name], file_sha256(render))
                self.assertEqual(image_input['model_input_sha256'], file_sha256(model_input))
                self.assertEqual(image_input['model_input_dimensions_px'], [64, 48])
        calls, builds = len(self.requests), len(self.builds)
        self.assertEqual(self.run_batch(), 0)
        self.assertEqual((len(self.requests), len(self.builds)), (calls, builds))
        self.evidence(api_requests=calls, build_invocations=builds, resume_api_requests=0,
                      resume_build_invocations=0, accepted_only_isolation=True, ordered_image_hashes_verified=True)

    def test_shared_budget_stops_persistently_invalid_candidate_without_review(self):
        self.scene_groups = [{'scene_id': 'synthetic_budget_exhausted', 'images': ['0.png']}]
        self.assertEqual(self.run_batch(), 1)
        result = self.results()['synthetic_budget_exhausted']
        self.assertEqual(result['status'], 'failed')
        self.assertIn('budget', result['error'])
        self.assertEqual([r['stage'] for r in self.requests], ['analysis', 'technical_repair'])
        self.assertEqual(len(self.builds), 2)
        self.assertEqual(read_json(self.out / 'accepted.json'), [])
        self.assertFalse(list(self.out.rglob('scene.blend')))
        self.assertFalse(list(self.out.rglob('review.json')))
        self.assertFalse((self.out / 'synthetic_budget_exhausted/iteration_02').exists())
        self.evidence(api_requests=2, build_invocations=2, repair_budget=1, independent_review_calls=0,
                      failure_isolated=True, accepted_only_isolation=True)

    def test_zero_budget_never_calls_repair_model(self):
        self.scene_groups = [{'scene_id': 'synthetic_accepted', 'images': ['0.png']}]
        self.config['batch']['max_repairs'] = 0
        self.assertEqual(self.run_batch(), 1)
        self.assertEqual([r['stage'] for r in self.requests], ['analysis'])
        self.assertEqual(len(self.builds), 1)
        self.assertEqual(read_json(self.out / 'accepted.json'), [])
        self.evidence(api_requests=1, build_invocations=1, repair_budget=0, failure_isolated=True)

    def test_corrupt_committed_technical_evidence_never_reenters_api(self):
        self.scene_groups = [{'scene_id': 'synthetic_budget_exhausted', 'images': ['0.png']}]
        self.assertEqual(self.run_batch(), 1)
        scene = self.out / 'synthetic_budget_exhausted'
        diagnostic = scene / 'iteration_00/geometry_preflight.json'
        tampered = read_json(diagnostic)
        tampered['spec_sha256'] = '0' * 64
        atomic_json(diagnostic, tampered)
        requests, builds = len(self.requests), len(self.builds)
        self.assertEqual(self.run_batch(), 1)
        self.assertEqual((len(self.requests), len(self.builds)), (requests, builds))
        self.assertIn('stale or not bound', self.results()['synthetic_budget_exhausted']['error'])
        self.assertEqual(read_json(self.out / 'accepted.json'), [])
        self.evidence(api_requests=requests, build_invocations=builds,
                      tampered_resume_api_requests=0, tampered_resume_builds=0, failure_isolated=True)


if __name__ == '__main__':
    unittest.main()
