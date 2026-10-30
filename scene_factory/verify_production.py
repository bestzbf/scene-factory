"""Retained end-to-end production smoke fixture, with no provider calls.

    python -m scene_factory.verify_production --output-dir /absolute/new/folder \
      --blender /Applications/Blender.app/Contents/MacOS/Blender

A real loopback HTTP API supplies deliberately synthetic plan, detail, review
and typed-repair replies. Blender is real unless --mock-render is explicitly
selected. This verifies orchestration and acceptance isolation; it never
establishes photograph reconstruction quality or real model acceptance.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from unittest.mock import patch

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

from PIL import Image, ImageDraw

from automatic_repair import empty_edits
from batch import atomic_json, file_sha256, read_json
import batch_multiview as runner
from multiview_spec import validate
from vision_multiview import DIMENSIONS


def _decode_after(prompt, marker):
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]


def fixture_plan(views):
    """One floor-supported prop and a desk, with one shared physical light."""
    return {
        'scene_type': 'office',
        'base_material_overrides': [], 'exterior_contexts': [],
        'room': {'d': [5.7, 7.9, 3.1],
                 'c': [[.55, .52, .48], [.18, .16, .13], [.75, .73, .70]]},
        'cameras': [{'id': view['id'],
                     'd': [-.45 + index * .9, .4, 1.65, 0, 3.3, 1.0, 28, 0]}
                    for index, view in enumerate(views)],
        'materials': [{'id': 'wood', 'texture_id': '',
                       'd': [.24, .12, .055, .7, 0, 0, 0, 0, 0]}],
        'assets': [{'id': 'desk', 'category': 'desk',
                    'd': [-1, 3, 0, 1.4, .6, .75, 0, .24, .12, .055, 1, 1, 0, 0, .9]}],
        'lights': [{'id': 'ceiling_light', 'type': 'AREA', 'shape': 'RECTANGLE',
                    'd': [0, 3.3, 2.95, 0, 3.3, 0, 1, .94, .87, 400, .8, .5, 140]}],
        'render': [0, .15, .4, .45, .5], 'surfaces': ['', '', '', '', ''],
        'omit': [], 'front_wall': False, 'textures': [],
        'groups': [{'id': 'cabinet', 'p': [1, 4, 0], 'r': [0, 0, 0], 'm': 'wood',
                    'support': 'floor', 'd': [.8, .5, 1], 'confidence': .9,
                    'evidence_view_ids': [view['id'] for view in views],
                    'evidence_confidence': .9,
                    'texture_material_ids': [], 'texture_material_confidence': .9,
                    'observed': True, 'instructions': 'Synthetic floor-supported box fixture.'}],
        'assumptions': ['Synthetic integration fixture. No photograph interpretation or realism claim.'],
    }


def fixture_detail(group_id):
    return {'group_id': group_id, 'parts': [{
        'type': 'box', 'material_id': '', 'data': [0, 0, .5, .8, .5, 1, 0, 0, 0, 0],
        'points': [], 'vertices': [], 'faces': [], 'cloth': [],
    }]}


class FixtureAPI:
    """Only the real HTTP adapter is used: model answers are known test data."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.requests = []
        self.errors = []
        self.guard = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                try:
                    if self.path != '/v1/chat/completions':
                        raise ValueError('Unexpected fixture endpoint')
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    result, record = owner.reply(body)
                    with owner.guard:
                        owner.requests.append(record)
                    response = json.dumps({
                        'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}],
                        'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
                    }).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                except Exception as exc:
                    with owner.guard:
                        owner.errors.append(type(exc).__name__)
                    self.send_error(500, 'Synthetic fixture contract failed')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def base_url(self):
        return f'http://127.0.0.1:{self.server.server_port}/v1'

    def reply(self, body):
        content = body['messages'][0]['content']
        prompt = content[0]['text']
        properties = body['response_format']['json_schema']['schema']['properties']
        if 'group_id' in properties:
            stage = 'detail'
            identity = _decode_after(prompt, 'TRUSTED INPUT IDENTITY (images in exactly this order):\n')
            scene_id, views = identity['scene_id'], identity['views']
            result = fixture_detail(properties['group_id']['enum'][0])
            if 'wire_format' in properties:
                from compact_staged_detail import compact_detail
                result = compact_detail(result)
        elif 'groups' in properties and 'schema_version' not in properties:
            stage = 'plan'
            identity = _decode_after(prompt, 'TRUSTED INPUT IDENTITY (images in exactly this order):\n')
            scene_id, views = identity['scene_id'], identity['views']
            result = fixture_plan(views)
            group_properties = properties['groups']['items']['properties']
            if 'evidence_view_ids' not in group_properties:
                for group in result['groups']:
                    for key in ('evidence_view_ids', 'evidence_confidence',
                                'texture_material_ids', 'texture_material_confidence'):
                        group.pop(key, None)
        elif 'part_updates' in properties:
            stage = 'repair'
            data = _decode_after(prompt, 'UNTRUSTED SCENE AND ASSESSMENT DATA:\n')
            bundle = data['scene_bundle']
            scene_id, views = bundle['scene_id'], bundle['views']
            result = empty_edits('Synthetic exposure edit; requires a new render and review.')
            result['render'] = [{**bundle['shared_scene']['render'], 'exposure': -.35}]
        else:
            stage = 'review'
            data = _decode_after(prompt, 'UNTRUSTED CURRENT SCENE DATA (only identity/pairing is fixed):\n')
            bundle = data['scene_bundle']
            scene_id, views = bundle['scene_id'], bundle['views']
            rejected = scene_id == 'single_rejected'
            repairable = scene_id == 'two_views' and bundle['shared_scene']['render']['exposure'] == 0
            score = 50 if rejected or repairable else 95
            scores = {dimension: score for dimension in DIMENSIONS}
            result = {
                'schema_version': 3,
                'verdict': 'asset_limited' if rejected else ('repair' if repairable else 'recommend_accept'),
                'summary': 'Synthetic fixture verdict, not real visual quality assessment.',
                'scores': scores,
                'per_view_scores': [{'view_id': view['id'], 'scores': scores.copy(),
                                     'summary': 'Synthetic score.'} for view in views],
                'issues': [],
                'asset_gaps': ['Deliberately rejected synthetic scene.'] if rejected else [],
                'reference_limitations': ['All photos and API replies are synthetic test data.'],
            }
        expected = len(views) * (2 if stage in ('review', 'repair') else 1)
        images = [item['image_url']['url'] for item in content if item['type'] == 'image_url']
        image_count = len(images)
        if image_count != expected:
            raise AssertionError('Every stage must receive all relevant matched photographs/renders')
        expected_paths, model_image_inputs = [], []
        scene_directory = self.directory / 'output' / scene_id
        for view in views:
            expected_paths.append(scene_directory / 'sources' / view['id'] / 'reference.png')
            if stage in ('review', 'repair'):
                iteration = 'iteration_00' if bundle['shared_scene']['render']['exposure'] == 0 else 'iteration_01'
                render = scene_directory / iteration / ('rgb_' + view['id'] + '.png')
                proxy = render.parent / 'model_input_images' / render.name
                model_input = proxy if proxy.is_file() else render
                expected_paths.append(model_input)
                model_image_inputs.append({'view_id': view['id'],
                    'render_path': str(render.resolve()), 'render_sha256': file_sha256(render),
                    'model_input_path': str(model_input.resolve()),
                    'model_input_sha256': file_sha256(model_input)})
        uploaded_hashes = [hashlib.sha256(base64.b64decode(url.split(',', 1)[1], validate=True)).hexdigest()
                           for url in images]
        if uploaded_hashes != [file_sha256(path) for path in expected_paths]:
            raise AssertionError('Originals and renders must match the prompt pairing in exact input order')
        if stage in ('review', 'repair') and data['image_pairs'] != [
                {'view_id': view['id'], 'original_image_number': 2 * index + 1,
                 'render_image_number': 2 * index + 2} for index, view in enumerate(views)]:
            raise AssertionError('The pairing metadata must identify the uploaded photograph/render order')
        return result, {'scene_id': scene_id, 'stage': stage, 'image_count': image_count,
                        'view_count': len(views), 'model': body['model'],
                        'ordered_image_sha256': uploaded_hashes,
                        'model_image_inputs': model_image_inputs}


def _mock_render(command, _log, _timeout):
    """Explicit fast test mode: these bytes are not a valid Blender document."""
    directory = Path(command[command.index('--output-dir') + 1])
    bundle = validate(read_json(command[command.index('--spec') + 1]))
    samples = int(command[command.index('--samples') + 1])
    (directory / 'scene.blend').write_bytes(b'SYNTHETIC-NOT-A-BLENDER-FILE\0' * 60)
    outputs, views = ['scene.blend'], []
    for view in bundle['views']:
        render, annotation = f'rgb_{view["id"]}.png', f'annotations_{view["id"]}.json'
        Image.new('RGB', (1280, 960), (73, 89, 101)).save(directory / render)
        atomic_json(directory / annotation, {'warnings': [], 'synthetic_mock_render': True})
        views.append({'id': view['id'], 'render': render, 'annotation': annotation,
                      'width': 1280, 'height': 960, 'warning_count': 0})
        outputs.extend([render, annotation])
    atomic_json(directory / 'metadata.json', {
        'status': 'completed', 'geometry_build_count': 1, 'camera_count': len(views),
        'views': views, 'outputs': outputs, 'render': {'samples': samples},
        'synthetic_mock_render': True,
        'output_sha256': {name: file_sha256(directory / name) for name in outputs},
    })


def verify(output_dir, blender=None, mock_render=False, *, detail_wire_format='canonical'):
    """Run a new retained fixture and return its written verification record."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    photo_directory = directory / 'photos'
    photo_directory.mkdir()
    for index in range(4):
        image = Image.new('RGB', (64, 48), (80 + index * 15, 110, 135))
        ImageDraw.Draw(image).rectangle((index + 2, 8, 31, 33), fill=(180, 90, index * 40))
        image.save(photo_directory / f'{index}.png')
    groups = directory / 'groups.json'
    atomic_json(groups, {'schema_version': 1, 'scenes': [
        {'scene_id': 'two_views', 'images': ['photos/0.png', 'photos/1.png']},
        {'scene_id': 'single_accepted', 'images': ['photos/2.png']},
        {'scene_id': 'single_rejected', 'images': ['photos/3.png']},
    ]})
    calls = []
    original_run_stage = runner.run_stage

    def render(command, log, timeout):
        calls.append(str(log))
        return (_mock_render if mock_render else original_run_stage)(command, log, timeout)

    with FixtureAPI(directory) as api:
        model = {'provider': 'openai_chat', 'base_url': api.base_url, 'api_key_env': '',
                 'response_format': 'json_schema', 'max_retries': 0, 'timeout_seconds': 60}
        config = {
            'schema_version': 1,
            'models': {role: {**model, 'model': 'synthetic-' + role}
                       for role in ('analysis', 'review', 'repair')},
            'batch': {'workers': 2, 'max_repairs': 1, 'no_review': False,
                      'vision_timeout': 60, 'render_timeout': 2400},
            'render': {'max_edge': 1280, 'samples': 64, 'tone_match': 'off', 'device': 'CPU'},
            'pipeline': {'analysis_mode': 'staged', 'repair_mode': 'typed', 'validation_retries': 0,
                         'staged': {'planner_timeout': 60, 'detail_timeout': 60, 'detail_workers': 1,
                                    'max_groups': 2, 'max_parts_per_group': 10, 'validation_retries': 0,
                                    'detail_wire_format': detail_wire_format}},
        }
        config_path = directory / 'factory.json'
        atomic_json(config_path, config)
        command = ['--config', str(config_path), '--groups', str(groups),
                   '--output-dir', str(directory / 'output')]
        if blender:
            command += ['--blender', str(blender)]
        # The API path must never fall through to the old Codex CLI adapter.
        with patch.object(runner, 'run_stage', side_effect=render), \
             patch('vision_multiview.call_model', side_effect=AssertionError('Legacy model CLI was called')):
            source_hashes = runner.pipeline_hashes()
            if mock_render:
                with patch.object(runner, 'discover_blender', return_value='synthetic-blender'):
                    exit_code = runner.main(command)
            else:
                exit_code = runner.main(command)
            if exit_code != 0:
                raise AssertionError('First production batch failed; inspect retained output/summary.json')
            initial_requests, initial_renders = len(api.requests), len(calls)
            artifact_before = {str(path.relative_to(directory)): file_sha256(path)
                               for pattern in ('*.blend', 'rgb_*.png', 'metadata.json')
                               for path in (directory / 'output').rglob(pattern)}
            if source_hashes != runner.pipeline_hashes():
                raise AssertionError('Pipeline source changed during this smoke run; rerun in a new output directory after editing finishes')
            if mock_render:
                with patch.object(runner, 'discover_blender', return_value='synthetic-blender'):
                    resume_exit = runner.main(command)
            else:
                resume_exit = runner.main(command)
        if resume_exit != 0 or len(api.requests) != initial_requests or len(calls) != initial_renders:
            raise AssertionError('An unchanged second batch must reuse completed API and render artifacts')
        if any(file_sha256(directory / path) != digest for path, digest in artifact_before.items()):
            raise AssertionError('A completed Blender/render artifact changed during unchanged resume')
        if api.errors:
            raise AssertionError('Loopback fixture server reported errors')
        requests = copy.deepcopy(api.requests)
    summary = read_json(directory / 'output' / 'summary.json')
    results = {result['scene_id']: result for result in summary['scenes']}
    statuses = {scene_id: result['status'] for scene_id, result in results.items()}
    if statuses != {'two_views': 'accepted', 'single_accepted': 'accepted', 'single_rejected': 'needs_review'}:
        raise AssertionError('Unexpected final fixture statuses: ' + json.dumps(statuses))
    accepted = sorted(result['scene_id'] for result in read_json(directory / 'output' / 'accepted.json'))
    if accepted != ['single_accepted', 'two_views']:
        raise AssertionError('Rejected fixture must never enter accepted.json')
    shared = directory / 'output' / 'two_views'
    repair_provenance = read_json(shared / 'iteration_00' / 'repair_provenance.json')
    if not repair_provenance['candidate_requires_new_render_and_review']:
        raise AssertionError('Typed repair was not followed by a separately reviewed candidate')
    if results['two_views'].get('final_review_iteration') != 'iteration_01':
        raise AssertionError('First iteration review must not certify a repaired scene')
    for iteration_name in ('iteration_00', 'iteration_01'):
        metadata = read_json(shared / iteration_name / 'metadata.json')
        if metadata['geometry_build_count'] != 1 or metadata['camera_count'] != 2:
            raise AssertionError('The two-photo group must share one geometry build with two cameras')
    if initial_requests != 11 or initial_renders != 4:
        raise AssertionError(f'Unexpected fixture call counts: {initial_requests} API / {initial_renders} Blender')
    if any(row['model'] != 'synthetic-' + ('analysis' if row['stage'] in ('plan', 'detail') else row['stage'])
           for row in requests):
        raise AssertionError('Analysis, review and repair must use their separately configured API models')
    validation = {
        'status': 'passed', 'synthetic_fixture': True,
        'external_provider_calls': 0, 'paid_model_calls': 0,
        'detail_wire_format': detail_wire_format,
        'real_blender': not mock_render, 'photograph_reconstruction_validated': False,
        'real_model_visual_acceptance_validated': False, 'formal_dataset_release': False,
        'scope': 'Technical batch orchestration only; synthetic verdicts and photos are not evidence of reconstruction quality.',
        'statuses': statuses, 'accepted_fixture_scene_ids': accepted,
        'api_requests': initial_requests, 'api_requests_on_resume': len(requests) - initial_requests,
        'blender_build_invocations': initial_renders, 'blender_build_invocations_on_resume': len(calls) - initial_renders,
        'rendered_views': sum(len(result['views']) * (2 if scene_id == 'two_views' else 1)
                              for scene_id, result in results.items()),
        'all_originals_sent_to_every_analysis_stage': True,
        'all_original_render_pairs_sent_to_every_review_and_repair': True,
        'ordered_image_hashes_verified': True,
        'separately_configured_api_models_verified': True,
        'separate_final_review_after_typed_repair': True, 'rejected_scene_excluded': True,
        'unchanged_resume_keeps_render_artifacts': True,
        'request_evidence': requests,
        'summary': 'output/summary.json', 'comparison': 'output/comparison.html',
        'elapsed_seconds': round(time.monotonic() - started, 3),
    }
    atomic_json(directory / 'validation.json', validation)
    return validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New retained fixture directory; never overwritten')
    parser.add_argument('--blender', help='Installed Blender executable for real rendering')
    parser.add_argument('--mock-render', action='store_true', help='Fast orchestration check; produces invalid synthetic .blend bytes')
    from compact_staged_detail import FORMAT
    parser.add_argument('--detail-wire-format', choices=('canonical', FORMAT), default='canonical')
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.output_dir, args.blender, args.mock_render,
                           detail_wire_format=args.detail_wire_format), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
