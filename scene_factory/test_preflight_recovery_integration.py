"""Opt-in real Blender rejection -> local HTTP repair -> render -> HTTP review.

Only synthetic scenes and a local fixture model are used. This verifies the
unattended protocol, not external-model photo fidelity.
"""
import base64
import copy
from contextlib import nullcontext, redirect_stdout
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import automatic_repair
import batch_multiview as runner
from batch import atomic_json, file_sha256, read_json
from compact_context import decode_record_tables
from spec_v2 import part
from test_multiview import fixture
import vision_multiview as vision


@unittest.skipUnless(os.environ.get('SCENE_FACTORY_TEST_BLENDER'), 'Set SCENE_FACTORY_TEST_BLENDER for real Blender integration')
class RealPreflightRecoveryTests(unittest.TestCase):
    def test_actual_preflight_recovery_and_independent_review(self):
        artifact_root = os.environ.get('SCENE_FACTORY_TEST_ARTIFACTS')
        if artifact_root:
            Path(artifact_root).mkdir(parents=True, exist_ok=False)
        workspace = nullcontext(artifact_root) if artifact_root else tempfile.TemporaryDirectory()
        with workspace as temporary:
            root = Path(temporary).resolve()
            inputs = []
            for index in range(2):
                path = root / f'photo_{index}.png'
                Image.new('RGB', (64, 48), (60 + index * 20, 95, 115)).save(path)
                inputs.append(path)
            records = []
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args): pass
                def do_POST(self):
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    content = request['messages'][0]['content']
                    prompt = content[0]['text']
                    original_only = 'UNTRUSTED TECHNICAL REPAIR DATA:\n' in prompt
                    marker = ('UNTRUSTED TECHNICAL REPAIR DATA:\n' if original_only else
                              'UNTRUSTED CURRENT SCENE DATA (only identity/pairing is fixed):\n')
                    payload = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]
                    image_hashes = [hashlib.sha256(base64.b64decode(item['image_url']['url'].split(',', 1)[1])).hexdigest()
                                    for item in content if item['type'] == 'image_url']
                    records.append({'stage': 'technical' if original_only else 'review', 'images': image_hashes,
                                    'payload': payload, 'model': request['model']})
                    if original_only:
                        result = automatic_repair.empty_edits('Synthetic camera collision correction from measured evidence.')
                        for view in payload['scene_bundle']['views']:
                            camera = copy.deepcopy(view['camera'])
                            camera['position'][0] = 0
                            result['cameras'].append({'view_id': view['id'], 'camera': camera})
                    else:
                        scores = {name: 60 for name in vision.DIMENSIONS}
                        result = {'schema_version': 3, 'verdict': 'repair', 'summary': 'Synthetic independent final review.',
                                  'scores': scores, 'per_view_scores': [{'view_id': v['id'], 'scores': scores, 'summary': 'Synthetic preview.'}
                                      for v in payload['scene_bundle']['views']], 'issues': [], 'asset_gaps': [],
                                  'reference_limitations': ['Synthetic fixture does not measure photo fidelity.']}
                    response = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}],
                                           'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(response)))
                    self.end_headers(); self.wfile.write(response)
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                config = {'provider': 'openai_chat', 'model': 'fixture-repair', 'base_url': f'http://127.0.0.1:{server.server_port}/v1',
                          'api_key_env': '', 'response_format': 'json_schema', 'max_retries': 0, 'timeout_seconds': 30}
                args = SimpleNamespace(output_dir=root / 'output', max_edge=128, samples=8, device='CPU', max_repairs=1,
                                       effort='low', vision_timeout=30, render_timeout=120, no_review=False, tone_match='off',
                                       blender=os.environ['SCENE_FACTORY_TEST_BLENDER'],
                                       model_configs={'analysis': config, 'repair': config, 'review': {**config, 'model': 'fixture-review'}},
                                       pipeline_options={'repair_mode': 'typed', 'validation_retries': 0})
                def analyze(scene_id, views, output, **kwargs):
                    bundle = fixture()
                    bundle['scene_id'] = scene_id
                    # Repetitive native geometry exercises the lossless wire
                    # encoding through HTTP repair and the real Blender rebuild.
                    group = copy.deepcopy(bundle['shared_scene']['custom_objects'][0])
                    group.update(id='wire_geometry', position=[-2, 1.5, .05], support_id='floor',
                                 parts=[part('box', size=[.04, .04, .04], position=[0, i * .01, 0])
                                        for i in range(30)])
                    bundle['shared_scene']['custom_objects'] = [group]
                    bundle['shared_scene']['base_scene']['objects'] = bundle['shared_scene']['base_scene']['objects'][:1]
                    cameras = [copy.deepcopy(v['camera']) for v in bundle['views']]
                    cameras[0]['position'][0] = bundle['shared_scene']['base_scene']['room']['width'] / 2
                    bundle['views'] = [{**view, 'camera': camera} for view, camera in zip(views, cameras)]
                    bundle['shared_scene']['image'] = {'width': views[0]['width'], 'height': views[0]['height'], 'source_id': views[0]['id']}
                    atomic_json(output, bundle)
                    atomic_json(output.parent / 'analysis_provenance.json', {'manual_scene_seed': False,
                                'bundle_sha256': file_sha256(output), 'source_sha256': {v['id']: file_sha256(v['source_image']) for v in views},
                                'test_fixture': True})
                    return bundle
                with patch.object(vision, 'analyze', side_effect=analyze), \
                     patch.object(vision, 'call_model', side_effect=AssertionError('Legacy CLI forbidden')), \
                     redirect_stdout(io.StringIO()):
                    result = runner.run_job({'scene_id': 'synthetic_room', 'images': inputs}, args, runner.pipeline_hashes())
                self.assertEqual(result['status'], 'needs_review', result.get('error'))
                self.assertEqual(result['final_review_iteration'], 'iteration_01')
                self.assertFalse(result['gate']['accepted'])
                folder = Path(result['directory'])
                first, last = folder / 'iteration_00', folder / 'iteration_01'
                self.assertFalse((first / 'scene.blend').exists())
                self.assertTrue((last / 'scene.blend').is_file())
                report = read_json(first / 'geometry_preflight.json')
                self.assertEqual(report['spec_sha256'], file_sha256(first / 'scene_bundle.json'))
                self.assertTrue(any(error['code'] == 'camera_room_shell_collision' for error in report['errors']))
                self.assertEqual(read_json(last / 'geometry_preflight.json')['status'], 'passed')
                self.assertEqual([record['stage'] for record in records], ['technical', 'review'])
                bundle = read_json(last / 'scene_bundle.json')
                originals = [file_sha256(v['source_image']) for v in bundle['views']]
                self.assertEqual(records[0]['images'], originals)
                wire_bundle = records[0]['payload']['scene_bundle']
                native_bundle = read_json(first / 'scene_bundle.json')
                for view in native_bundle['views']:
                    view['source_image'] = 'input:' + view['id']
                self.assertEqual(decode_record_tables(wire_bundle), native_bundle)
                compaction = read_json(first / 'technical_repair_provenance.json')['scene_bundle_compaction']
                self.assertGreater(compaction['encoded_arrays'], 0)
                self.assertEqual(bundle['shared_scene']['custom_objects'],
                                 native_bundle['shared_scene']['custom_objects'])
                self.assertEqual(records[1]['images'], [digest for v, original in zip(bundle['views'], originals)
                                                       for digest in (original, file_sha256(last / ('rgb_' + v['id'] + '.png')))])
                self.assertEqual(records[1]['model'], 'fixture-review')
                self.assertTrue((last / 'review_provenance.json').is_file())
                self.assertEqual([h['iteration'] for h in result['history']], [1])
                self.assertEqual([h['iteration'] for h in result['technical_history']], [0])
                from dataset_export import DatasetExportError, export_dataset
                with self.assertRaisesRegex(DatasetExportError, 'No accepted'):
                    export_dataset(args.output_dir, root / 'dataset')
                self.assertFalse((root / 'dataset').exists())
                if artifact_root:
                    atomic_json(root / 'integration_evidence.json', {
                        'test_fixture': True, 'uses_remote_api': False, 'uses_real_blender': True,
                        'claims_photo_fidelity': False, 'requests': records, 'result': result,
                        'checks': ['real_geometry_rejection_before_render', 'original_only_http_repair',
                                   'new_candidate_real_render', 'independent_http_paired_review',
                                   'no_fabricated_preflight_scores', 'failed_review_stays_out_of_dataset']})
            finally:
                server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()
