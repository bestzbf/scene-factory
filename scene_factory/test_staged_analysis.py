"""Native staged API tests; loopback fixtures are synthetic, not GLM fidelity tests."""
from __future__ import annotations

import copy
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

import staged_analysis as staged
import compact_staged_detail as compact_detail
from batch import atomic_json, read_json
from multiview_spec import validate
from model_api import ModelAPIError


def fixture_plan(views, group_count=2):
    return {'scene_type': 'office', 'base_material_overrides': [], 'exterior_contexts': [], 'room': {'d': [5.7, 7.9, 3.1], 'c': [[.6, .5, .4], [.2, .3, .4], [.8, .7, .6]]},
        'cameras': [{'id': v['id'], 'd': [i - .3, .4 + i * 6, 1.6, .1, 3.5, 1.4, 28 + i, 0]} for i, v in enumerate(views)],
        'materials': [{'id': 'wood', 'texture_id': '', 'd': [.35, .22, .11, .7, .1, 0, 20, .1, .01]},
                      {'id': 'photo_mat', 'texture_id': 'crop', 'd': [.5, .5, .5, .4, 0, 0, 0, 0, 0]}],
        'assets': [{'id': 'desk', 'category': 'desk', 'd': [-1, 3, 0, 1.4, .6, .75, 12, .3, .2, .1, 1, 1, 0, 0, .83]}],
        'lights': [{'id': 'key_light', 'type': 'AREA', 'shape': 'RECTANGLE', 'd': [-1, 3, 2.9, 0, 2, 0, 1, .9, .8, 290, .8, .5, 140]}],
        'render': [-.4, .12, .3, .35, .4], 'surfaces': ['wood', '', '', '', ''], 'omit': ['back'], 'front_wall': True,
        'textures': [{'id': 'crop', 'view_id': views[-1]['id'], 'quad': [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]}],
        'groups': [{'id': 'group_' + str(i), 'p': [i * 2, 2, 0], 'r': [0, 0, 15], 'm': 'wood',
                    'support': 'floor', 'd': [1.2, .6, 1.8], 'confidence': .9,
                    'evidence_confidence': .95, 'observed': True,
                    'instructions': 'Synthetic fixture cabinet.',
                    'texture_material_ids': ['photo_mat'], 'texture_material_confidence': .95,
                    'evidence_view_ids': [v['id'] for index, v in enumerate(views) if index == i % len(views)]}
                   for i in range(group_count)],
        'assumptions': ['Synthetic fixture, not an actual photo interpretation.']}


def fixture_detail(group_id, kind='box'):
    return {'group_id': group_id, 'parts': [{'type': kind, 'material_id': '',
        'data': [.12, -.03, .9, 1.2, .6, 1.8, 0, 0, 3, 0], 'points': [], 'vertices': [], 'faces': [], 'cloth': []}]}


def restore_detail_context(context, evidence_source=None):
    """Independent test decoder for lossless fields plus deliberate evidence elision."""
    result = copy.deepcopy(context)
    result.pop('context_encoding')
    target = result.pop('requested_group')
    index = result.pop('requested_group_index')
    def records(value):
        if isinstance(value, list):
            return value
        return [dict(zip(value['columns'], row)) for row in value['rows']]
    result['groups'] = records(result.pop('other_groups'))
    result['groups'].insert(index, target)
    if evidence_source is not None:
        for group, original in zip(result['groups'], evidence_source['groups']):
            for key in ('evidence_view_ids', 'evidence_confidence', 'texture_material_ids',
                        'texture_material_confidence'):
                if key in original:
                    group[key] = copy.deepcopy(original[key])
    for key in ('cameras', 'materials', 'assets', 'lights', 'textures',
                'base_material_overrides', 'exterior_contexts'):
        if key in result:
            result[key] = records(result[key])
    return result


class DetailContextTests(unittest.TestCase):
    def setUp(self):
        self.plan = fixture_plan([{'id': 'left'}, {'id': 'right'}], group_count=3)
        self.plan['groups'][0]['instructions'] = 'Desk top is at Z=0.74; aisle at X=0.'
        self.plan['groups'][1]['support'] = 'group_0'
        self.plan['groups'][1]['instructions'] = 'Place hutch on the declared desk top.'
        self.plan['groups'][2]['instructions'] = 'Leave an actual window opening; no duplicate glazing.'
        self.plan['assets'][0]['d'][10:14] = [2, 3, 1.5, .8]
        self.plan['base_material_overrides'] = [{'object_id': 'desk__0_0', 'material_id': 'wood',
            'selector': {'role': 'body', 'material_name': '', 'slot_index': -1}}]
        self.plan['exterior_contexts'] = [{'window_id': 'window', 'texture_id': 'crop',
                                          'depth_m': 2, 'brightness_ev': 1}]

    def test_quality_first_planner_wire_omits_unused_routing_contract(self):
        views = [{'id': 'left'}, {'id': 'right'}]
        routed = staged.native_schema('fixture_scene', ['left', 'right'], include_routing=True)
        quality_first = staged.native_schema('fixture_scene', ['left', 'right'], include_routing=False)
        routed_group = routed['properties']['groups']['items']['properties']
        quality_first_group = quality_first['properties']['groups']['items']['properties']
        routing_fields = {'evidence_view_ids', 'evidence_confidence',
                          'texture_material_ids', 'texture_material_confidence'}
        self.assertTrue(routing_fields <= set(routed_group))
        self.assertFalse(routing_fields & set(quality_first_group))
        encoded_size = lambda value: len(json.dumps(
            value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        self.assertLess(encoded_size(quality_first), encoded_size(routed))
        self.assertEqual(staged.detail_input_views(
            self.plan['groups'][0], self.plan, views,
            {**staged.DEFAULT_OPTIONS, 'route_detail_images': False}), views)

    def test_every_target_keeps_its_evidence_and_other_groups_keep_physical_context(self):
        before = copy.deepcopy(self.plan)
        for index, group in enumerate(self.plan['groups']):
            with self.subTest(group=group['id']):
                context = staged.detail_plan_context(self.plan, group['id'])
                self.assertEqual(context['requested_group'], group)
                self.assertEqual(context['requested_group_index'], index)
                projected = restore_detail_context(context, before)
                self.assertEqual(projected, before)
                self.assertEqual(projected['groups'][index], group)
                for other_index, other in enumerate(projected['groups']):
                    if other_index != index:
                        self.assertEqual(other, before['groups'][other_index])
        self.assertEqual(self.plan, before)

    def test_non_target_evidence_fields_are_removed_from_detail_context_only(self):
        plan = fixture_plan([{'id': 'left'}, {'id': 'right'}], group_count=16)
        before = copy.deepcopy(plan)
        context = staged.detail_plan_context(plan, 'group_0')
        self.assertEqual(context['requested_group'], plan['groups'][0])
        columns = context['other_groups']['columns']
        for key in ('evidence_view_ids', 'evidence_confidence', 'texture_material_ids',
                    'texture_material_confidence'):
            self.assertNotIn(key, columns)
        groups_without_evidence = [
            {key: value for key, value in group.items()
             if key not in ('evidence_view_ids', 'evidence_confidence', 'texture_material_ids',
                            'texture_material_confidence')}
            for group in plan['groups'][1:]
        ]
        self.assertEqual(restore_detail_context(context, before)['groups'][1:],
                         plan['groups'][1:])
        full_other_table = {
            'columns': sorted(plan['groups'][1]),
            'rows': [[group[key] for key in sorted(plan['groups'][1])]
                     for group in plan['groups'][1:]],
        }
        serialized_size = lambda value: len(json.dumps(
            value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        self.assertEqual(restore_detail_context(context, before)['groups'][1:],
                         [{**group, 'evidence_view_ids': original['evidence_view_ids'],
                           'evidence_confidence': original['evidence_confidence'],
                           'texture_material_ids': original['texture_material_ids'],
                           'texture_material_confidence': original['texture_material_confidence']}
                          for group, original in zip(groups_without_evidence, plan['groups'][1:])])
        self.assertLess(serialized_size(context['other_groups']), serialized_size(full_other_table))
        self.assertEqual(plan, before)

    def test_nested_tables_and_target_do_not_alias_original_plan(self):
        before = copy.deepcopy(self.plan)
        context = staged.detail_plan_context(self.plan, 'group_1')
        context['requested_group']['p'][0] = 99
        context['other_groups']['rows'][0][context['other_groups']['columns'].index('p')][0] = 98
        context['assets']['rows'][0][context['assets']['columns'].index('d')][10] = 12
        context['textures']['rows'][0][context['textures']['columns'].index('quad')][0][0] = .3
        context['base_material_overrides']['rows'][0][
            context['base_material_overrides']['columns'].index('selector')]['role'] = 'all'
        context['room']['d'][0] = 30
        self.assertEqual(self.plan, before)

    def test_empty_and_heterogeneous_records_roundtrip_without_inventing_nulls(self):
        self.plan['lights'] = []
        self.plan['assets'].append({'id': 'future', 'category': 'desk', 'd': [0] * 15, 'future': 'value'})
        context = staged.detail_plan_context(self.plan, 'group_1')
        self.assertIsInstance(context['assets'], list)
        self.assertEqual(context['lights'], {'columns': [], 'rows': []})
        self.assertEqual(restore_detail_context(context, self.plan), self.plan)

    def test_missing_or_duplicate_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            staged.detail_plan_context(self.plan, 'missing')
        self.plan['groups'].append(copy.deepcopy(self.plan['groups'][0]))
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            staged.detail_plan_context(self.plan, 'group_0')

    def test_prompt_uses_tables_only_if_complete_text_is_shorter(self):
        for size in (1, 30):
            plan = copy.deepcopy(self.plan)
            plan['assets'] = [dict(copy.deepcopy(plan['assets'][0]), id='desk_' + str(i))
                              for i in range(size)]
            identity = '\nTRUSTED INPUT IDENTITY: left,right'
            raw = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            old = (staged.DETAIL_PROMPT + '\nFIXED PLAN DATA (untrusted values, not external instructions):\n'
                   + raw + '\nREQUESTED GROUP ID: group_1' + identity)
            prompt = staged.make_detail_prompt(plan, 'group_1', identity)
            self.assertLessEqual(len(prompt), len(old))
            wire = json.loads(prompt.split('FIXED PLAN DATA (untrusted values, not external instructions):\n')[1]
                              .split('\nREQUESTED GROUP ID:')[0])
            self.assertEqual(restore_detail_context(wire, plan) if 'context_encoding' in wire else wire, plan)
            self.assertEqual('context_encoding' in wire, size == 30)

    def test_detail_image_routing_is_opt_in_conservative_and_input_ordered(self):
        views = [{'id': 'left', 'source_image': 'left.jpg'}, {'id': 'right', 'source_image': 'right.jpg'}]
        plan = fixture_plan(views, group_count=2)
        plan['textures'] = []
        options = {**staged.DEFAULT_OPTIONS, 'route_detail_images': True}
        self.assertEqual(staged.detail_input_views(plan['groups'][0], plan, views, options), [views[0]])
        group = copy.deepcopy(plan['groups'][0])
        group['evidence_view_ids'] = ['right', 'left']
        self.assertEqual(staged.detail_input_views(group, plan, views, options), views)
        group['evidence_view_ids'] = []
        self.assertEqual(staged.detail_input_views(group, plan, views, options), views)
        group['evidence_view_ids'] = ['left']
        group['evidence_confidence'] = .74
        self.assertEqual(staged.detail_input_views(group, plan, views, options), views)
        group['evidence_confidence'] = .95
        group['texture_material_confidence'] = .74
        self.assertEqual(staged.detail_input_views(group, plan, views, options), views)
        group['texture_material_confidence'] = .95
        group['evidence_confidence'] = .9
        group['observed'] = False
        self.assertEqual(staged.detail_input_views(group, plan, views, options), views)

    def test_high_confidence_view_routing_reduces_detail_image_attachments(self):
        views = [{'id': 'left', 'source_image': 'left.jpg'}, {'id': 'right', 'source_image': 'right.jpg'}]
        plan = fixture_plan(views, group_count=8)
        plan['textures'] = []
        for index, group in enumerate(plan['groups']):
            group['evidence_view_ids'] = [views[index % 2]['id']]
        options = {**staged.DEFAULT_OPTIONS, 'route_detail_images': True}
        baseline_attachments = sum(len(views) for _ in plan['groups'])
        routed_attachments = sum(len(staged.detail_input_views(group, plan, views, options))
                                 for group in plan['groups'])
        self.assertEqual(baseline_attachments, 16)
        self.assertEqual(routed_attachments, 8)
        self.assertFalse(staged.DEFAULT_OPTIONS['route_detail_images'])

    def test_detail_image_routing_keeps_any_photo_texture_evidence(self):
        views = [{'id': 'left', 'source_image': 'left.jpg'}, {'id': 'right', 'source_image': 'right.jpg'}]
        plan = fixture_plan(views, group_count=2)
        options = {**staged.DEFAULT_OPTIONS, 'route_detail_images': True}
        self.assertEqual(staged.detail_input_views(plan['groups'][0], plan, views, options), views)

    def test_texture_routing_is_per_group_and_detail_schema_restricts_omitted_materials(self):
        views = [{'id': name, 'source_image': name + '.jpg'} for name in ('geometry', 'texture_a', 'texture_b')]
        plan = fixture_plan(views, group_count=2)
        plan['textures'] = [dict(plan['textures'][0], view_id='texture_a')]
        plan['groups'][0]['evidence_view_ids'] = ['geometry']
        plan['groups'][1]['evidence_view_ids'] = ['geometry']
        plan['groups'][0]['texture_material_ids'] = ['photo_mat']
        plan['groups'][1]['texture_material_ids'] = []
        options = {**staged.DEFAULT_OPTIONS, 'route_detail_images': True, 'detail_view_min_confidence': .9}
        self.assertEqual([view['id'] for view in staged.detail_input_views(plan['groups'][0], plan, views, options)],
                         ['geometry', 'texture_a'])
        self.assertEqual([view['id'] for view in staged.detail_input_views(plan['groups'][1], plan, views, options)],
                         ['geometry'])
        all_images_baseline = len(views) * len(plan['groups'])
        routed = sum(len(staged.detail_input_views(group, plan, views, options)) for group in plan['groups'])
        self.assertEqual((all_images_baseline, routed), (6, 3))
        allowed = staged.detail_texture_scope(plan['groups'][1], plan, options)[1]
        schema = staged.detail_schema('group_1', 60, allowed)
        good = fixture_detail('group_1')
        staged._validate_wire(good, schema)
        bad = fixture_detail('group_1')
        bad['parts'][0]['material_id'] = 'photo_mat'
        with self.assertRaisesRegex(ValueError, 'unsupported value'):
            staged._validate_wire(bad, schema)

    def test_texture_scope_validation_rejects_dangling_duplicate_and_nontexture_ids(self):
        views = [{'id': 'left', 'width': 64, 'height': 48, 'source_image': 'left.jpg'},
                 {'id': 'right', 'width': 64, 'height': 48, 'source_image': 'right.jpg'}]
        options = {**staged.DEFAULT_OPTIONS, 'route_detail_images': True}
        for values, error in ((['missing'], 'photo-textured'),
                              (['photo_mat', 'photo_mat'], 'unique'),
                              (['wood'], 'photo-textured')):
            plan = fixture_plan(views)
            plan['groups'][0]['texture_material_ids'] = values
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, error):
                staged.validate_plan(plan, views, 'fixture_scene', options)
        plan = fixture_plan(views)
        plan['groups'][0]['m'] = 'photo_mat'
        plan['groups'][0]['texture_material_ids'] = []
        with self.assertRaisesRegex(ValueError, 'include its textured default'):
            staged.validate_plan(plan, views, 'fixture_scene', options)


class StagedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.views = []
        for i in range(2):
            path = self.root / f'reference_{i}.png'
            Image.new('RGB', (64 + i * 8, 48), (50 + i * 100, 80, 90)).save(path)
            self.views.append({'id': 'view_' + str(i), 'source_image': str(path), 'width': 64 + i * 8, 'height': 48})
        self.plan = fixture_plan(self.views)
        self.requests = []
        self.counter = {}
        self.replies = self.default_reply
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append(body)
                prompt = body['messages'][0]['content'][0]['text']
                schema = body['response_format']['json_schema']['schema']
                stage = schema['properties'].get('group_id', {}).get('enum', ['plan'])[0]
                owner.counter[stage] = owner.counter.get(stage, 0) + 1
                result = owner.replies(stage, owner.counter[stage], prompt)
                if stage == 'plan' and 'groups' in schema['properties'] and isinstance(result, dict):
                    group_properties = schema['properties']['groups']['items']['properties']
                    if 'evidence_view_ids' not in group_properties:
                        for group in result.get('groups', []):
                            for key in ('evidence_view_ids', 'evidence_confidence',
                                        'texture_material_ids', 'texture_material_confidence'):
                                group.pop(key, None)
                content = result if isinstance(result, str) else json.dumps(result)
                response = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': content}}],
                                       'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.config = {'provider': 'openai_chat', 'model': 'any-vision-model',
            'base_url': f'http://127.0.0.1:{self.server.server_port}/v1', 'api_key_env': '',
            'response_format': 'json_schema', 'max_retries': 0, 'timeout_seconds': 10}
        self.output = self.root / 'scene' / 'initial_bundle.json'
        self.options = {'planner_timeout': 10, 'detail_timeout': 10, 'detail_workers': 2, 'validation_retries': 1}

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def default_reply(self, stage, attempt, prompt):
        return copy.deepcopy(self.plan) if stage == 'plan' else fixture_detail(stage)

    def analyze(self, **kwargs):
        args = dict(scene_id='fixture_scene', views=self.views, output=self.output,
                    model_config=self.config, options=self.options)
        args.update(kwargs)
        # Any legacy CLI call would be a test failure.
        with patch('vision_multiview.call_model', side_effect=AssertionError('Codex CLI forbidden')):
            return staged.analyze(**args)

    def test_multiview_roundtrip_every_api_gets_both_photos(self):
        result = validate(self.analyze())
        self.assertEqual(len(self.requests), 3)
        self.assertEqual([v['id'] for v in result['views']], [v['id'] for v in self.views])
        self.assertEqual([v['source_image'] for v in result['views']], [v['source_image'] for v in self.views])
        for request in self.requests:
            content = request['messages'][0]['content']
            self.assertEqual(len([v for v in content if v['type'] == 'image_url']), 2)
            self.assertIn('input:view_0', content[0]['text'])
            self.assertLess(content[0]['text'].rindex('input:view_0'), content[0]['text'].rindex('input:view_1'))
            schema = request['response_format']['json_schema']['schema']
            if 'groups' in schema['properties']:
                group_properties = schema['properties']['groups']['items']['properties']
                self.assertNotIn('evidence_view_ids', group_properties)
                self.assertNotIn('texture_material_ids', group_properties)
                self.assertNotIn('evidence_view_ids', content[0]['text'])
                self.assertNotIn('texture_material_ids', content[0]['text'])
        scene = result['shared_scene']
        self.assertEqual(scene['scene_type'], 'office')
        self.assertEqual(scene['base_scene']['scene_type'], 'office')
        self.assertEqual(scene['base_scene']['room']['width'], 5.7)
        self.assertEqual(scene['base_scene']['objects'][0]['rotation_z_deg'], 12)
        self.assertEqual(scene['custom_objects'][0]['parts'][0]['position'], [.12, -.03, .9])
        self.assertEqual(scene['lights'][0]['energy'], 290)
        self.assertEqual(scene['materials'][0]['noise_scale'], 20)
        self.assertEqual(result['texture_sources'], [{'texture_id': 'crop', 'view_id': 'view_1'}])
        self.assertTrue(scene['render']['replace_base_lights'])
        provenance = read_json(self.output.parent / 'analysis_provenance.json')
        self.assertEqual(provenance['model']['model'], 'any-vision-model')
        self.assertEqual(provenance['bundle_sha256'], staged.file_sha256(self.output))
        self.assertFalse(provenance['manual_scene_seed'])
        for stage in provenance['stages']:
            for key in ('raw', 'api', 'prompt', 'schema'):
                path = Path(stage['artifacts'][key])
                self.assertTrue(path.is_file())
                self.assertEqual(staged.file_sha256(path), stage['sha256'][key])

    def test_scene_type_survives_bundle_and_changes_asset_dispatch_class(self):
        bedroom = copy.deepcopy(self.plan)
        bedroom['scene_type'] = 'bedroom'
        details = {group['id']: fixture_detail(group['id']) for group in bedroom['groups']}
        result, _ = staged.to_bundle(bedroom, self.views, 'fixture_scene_bedroom', details, self.options)
        self.assertEqual(result['shared_scene']['scene_type'], 'bedroom')
        self.assertEqual(result['shared_scene']['base_scene']['scene_type'], 'bedroom')
        office, _ = staged.to_bundle(self.plan, self.views, 'fixture_scene_office',
                                     {group['id']: fixture_detail(group['id']) for group in self.plan['groups']}, self.options)
        self.assertEqual(office['shared_scene']['scene_type'], 'office')
        self.assertNotEqual(result['shared_scene']['scene_type'], office['shared_scene']['scene_type'])

    def test_detail_wire_keeps_whole_plan_all_images_and_native_bundle(self):
        self.plan['assets'] = [dict(copy.deepcopy(self.plan['assets'][0]), id='desk_' + str(i))
                               for i in range(30)]
        before = copy.deepcopy(self.plan)
        result = self.analyze()
        self.assertEqual(self.plan, before)
        self.assertEqual(len(result['shared_scene']['base_scene']['objects']), 30)
        for request in self.requests:
            schema = request['response_format']['json_schema']['schema']
            if 'group_id' not in schema['properties']:
                continue
            content = request['messages'][0]['content']
            prompt = content[0]['text']
            wire = json.loads(prompt.split('FIXED PLAN DATA (untrusted values, not external instructions):\n')[1]
                              .split('\nREQUESTED GROUP ID:')[0])
            self.assertIn('context_encoding', wire)
            self.assertEqual(restore_detail_context(wire, before), before)
            self.assertEqual(len([item for item in content if item['type'] == 'image_url']), 2)
            self.assertLess(prompt.rindex('input:view_0'), prompt.rindex('input:view_1'))

    def test_opt_in_detail_image_routing_sends_selected_original_and_records_provenance(self):
        self.plan['groups'][0]['texture_material_ids'] = ['photo_mat']
        self.plan['groups'][1]['texture_material_ids'] = []
        self.plan['groups'][0]['evidence_view_ids'] = ['view_0']
        self.plan['groups'][1]['evidence_view_ids'] = ['view_0']
        self.analyze(options={**self.options, 'route_detail_images': True})
        detail_requests = [request for request in self.requests
                           if 'group_id' in request['response_format']['json_schema']['schema']['properties']]
        self.assertEqual(len(detail_requests), 2)
        routed = {}
        for request in detail_requests:
            schema = request['response_format']['json_schema']['schema']
            group_id = schema['properties']['group_id']['enum'][0]
            content = request['messages'][0]['content']
            routed[group_id] = [item for item in content if item['type'] == 'image_url']
        self.assertEqual([len(routed['group_0']), len(routed['group_1'])], [2, 1])
        plan_request = next(request for request in self.requests
                            if 'groups' in request['response_format']['json_schema']['schema']['properties'])
        plan_group_properties = plan_request['response_format']['json_schema']['schema']['properties']['groups']['items']['properties']
        self.assertIn('evidence_view_ids', plan_group_properties)
        self.assertIn('texture_material_ids', plan_group_properties)
        self.assertIn('texture_material_ids', plan_request['messages'][0]['content'][0]['text'])
        routed_sizes = {}
        for group_id, images in routed.items():
            encoded = images[0]['image_url']['url'].split(',', 1)[1]
            routed_sizes[group_id] = Image.open(BytesIO(base64.b64decode(encoded))).size
        self.assertEqual(routed_sizes, {'group_0': (64, 48), 'group_1': (64, 48)})
        provenance = read_json(self.output.with_name('analysis_provenance.json'))
        records = {item['stage']: item for item in provenance['stages']}
        self.assertEqual(records['detail_group_0']['input_view_ids'], ['view_0', 'view_1'])
        self.assertEqual(records['detail_group_1']['input_view_ids'], ['view_0'])

    def test_opt_in_sparse_detail_wire_is_cached_as_validated_canonical_data(self):
        def sparse_reply(stage, attempt, prompt):
            data = self.default_reply(stage, attempt, prompt)
            if stage != 'plan':
                return compact_detail.compact_detail(data)
            return data
        self.replies = sparse_reply
        self.plan['groups'][1]['texture_material_ids'] = []
        options = {**self.options, 'route_detail_images': True,
                   'detail_wire_format': compact_detail.FORMAT}
        result = validate(self.analyze(options=options))
        detail_requests = [request for request in self.requests
                           if 'group_id' in request['response_format']['json_schema']['schema']['properties']]
        self.assertEqual(len(detail_requests), 2)
        for request in detail_requests:
            content = request['messages'][0]['content']
            schema = request['response_format']['json_schema']['schema']
            prompt = content[0]['text']
            self.assertEqual(schema['properties']['wire_format']['enum'], [compact_detail.FORMAT])
            self.assertIn(compact_detail.FORMAT, prompt)
            group_id = schema['properties']['group_id']['enum'][0]
            material_enum = schema['properties']['parts']['items']['properties']['material_id']['enum']
            if group_id == 'group_1':
                self.assertEqual(material_enum, ['', 'wood'])
                self.assertEqual(len([item for item in content if item['type'] == 'image_url']), 1)
        stage_dir = self.output.with_name('initial_bundle_stages')
        for stage in ('detail_group_0', 'detail_group_1'):
            raw = read_json(stage_dir / f'{stage}_attempt_00.raw.json')
            canonical = read_json(stage_dir / f'{stage}_attempt_00.canonical.json')
            self.assertEqual(raw['wire_format'], compact_detail.FORMAT)
            self.assertNotIn('wire_format', canonical)
            self.assertEqual(canonical, fixture_detail(stage.removeprefix('detail_')))
        self.assertEqual(result['shared_scene']['custom_objects'][0]['parts'][0]['type'], 'box')
        provenance = read_json(self.output.with_name('analysis_provenance.json'))
        self.assertTrue(all('detail_wire_encoding' in stage for stage in provenance['stages'][1:]))
        # Recovery must validate and reuse both immutable wire and canonical
        # artifacts instead of charging another request.
        count = len(self.requests)
        self.assertEqual(self.analyze(options=options), result)
        self.assertEqual(len(self.requests), count)
        canonical_path = stage_dir / 'detail_group_0_attempt_00.canonical.json'
        tampered = read_json(canonical_path)
        tampered['parts'][0]['data'][0] += .1
        atomic_json(canonical_path, tampered)
        with self.assertRaises(staged.StagedAnalysisError):
            self.analyze(options=options)
        self.assertEqual(len(self.requests), count)

    def test_invalid_compact_detail_is_rejected_before_bundle_publication(self):
        def bad_reply(stage, attempt, prompt):
            data = self.default_reply(stage, attempt, prompt)
            if stage != 'plan':
                wire = compact_detail.compact_detail(data)
                # Valid transport schema, but no topology for a declared mesh:
                # the decoder must reject it before whole-scene assembly.
                wire['parts'][0]['type'] = 'mesh'
                return wire
            return data
        self.replies = bad_reply
        with self.assertRaises(staged.StagedAnalysisError):
            self.analyze(options={**self.options, 'detail_wire_format': compact_detail.FORMAT})
        self.assertFalse(self.output.exists())

    def test_staged_options_bound_detail_image_routing_and_threshold(self):
        for value in (0, 1, .5):
            options = staged.normalize_options({'route_detail_images': True,
                                                'detail_view_min_confidence': value})
            self.assertEqual(options['detail_view_min_confidence'], float(value))
        for value in (-.01, 1.01, True, 'high'):
            with self.assertRaises(ValueError):
                staged.normalize_options({'detail_view_min_confidence': value})
        with self.assertRaises(ValueError):
            staged.normalize_options({'route_detail_images': 1})

    def test_historic_plan_without_evidence_map_keeps_all_views_without_mutation(self):
        for group in self.plan['groups']:
            group.pop('evidence_view_ids')
            group.pop('evidence_confidence')
            group.pop('texture_material_ids')
            group.pop('texture_material_confidence')
        before = copy.deepcopy(self.plan)
        for enabled in (False, True):
            options = {**self.options, 'route_detail_images': enabled}
            staged.validate_plan(self.plan, self.views, 'fixture_scene', options)
            for group in self.plan['groups']:
                self.assertEqual(staged.detail_input_views(group, self.plan, self.views, options), self.views)
        self.assertEqual(self.plan, before)
        # Local import compatibility must not weaken the fresh API contract.
        with self.assertRaises(ValueError):
            staged._validate_wire(self.plan, staged.native_schema('fixture_scene', ['view_0', 'view_1']))

    def test_context_implementation_hash_change_refuses_cached_job(self):
        self.analyze()
        count = len(self.requests)
        real_hash = staged.file_sha256
        with patch.object(staged, 'file_sha256', side_effect=lambda path:
                          'f' * 64 if Path(path).name == 'staged_analysis.py' else real_hash(path)):
            with self.assertRaisesRegex(staged.StagedAnalysisError, 'changed'):
                self.analyze()
        self.assertEqual(len(self.requests), count)

    def test_camera_reorder_rejected_then_retry_feedback(self):
        def reply(stage, attempt, prompt):
            result = self.default_reply(stage, attempt, prompt)
            if stage == 'plan' and attempt == 1:
                result['cameras'].reverse()
            return result
        self.replies = reply
        result = self.analyze()
        self.assertEqual(self.counter['plan'], 2)
        self.assertEqual(result['views'][0]['id'], 'view_0')
        prompt = (self.output.parent / 'initial_bundle_stages/plan_attempt_01.prompt.txt').read_text()
        self.assertIn('camera order or identity', prompt)

    def test_identity_and_bounds_rejected_without_silent_normalization(self):
        for mutation in ('duplicate_camera', 'camera_target', 'repeat_float', 'support_cycle', 'undefined_material',
                         'duplicate_evidence_view', 'reverse_evidence_views', 'unknown_evidence_view'):
            plan = copy.deepcopy(self.plan)
            if mutation == 'duplicate_camera': plan['cameras'][1]['id'] = plan['cameras'][0]['id']
            elif mutation == 'camera_target': plan['cameras'][0]['d'][3:6] = plan['cameras'][0]['d'][:3]
            elif mutation == 'repeat_float': plan['assets'][0]['d'][10] = 1.0
            elif mutation == 'support_cycle':
                plan['groups'][0]['support'] = 'group_1'; plan['groups'][1]['support'] = 'group_0'
            elif mutation == 'duplicate_evidence_view':
                plan['groups'][0]['evidence_view_ids'] = ['view_0', 'view_0']
            elif mutation == 'reverse_evidence_views':
                plan['groups'][0]['evidence_view_ids'] = ['view_1', 'view_0']
            elif mutation == 'unknown_evidence_view':
                plan['groups'][0]['evidence_view_ids'] = ['not_a_view']
            else: plan['groups'][0]['m'] = 'not_declared'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                staged.validate_plan(plan, self.views, 'fixture_scene',
                    {**staged.DEFAULT_OPTIONS, 'route_detail_images': True})

    def test_real_dorm_origin_regression_is_rejected_without_translation(self):
        # Reduced geometry from the failed GLM dorm plan: everything was placed
        # around X=width/2 while the actual room shell is centred at X=0.
        plan = copy.deepcopy(self.plan)
        plan['room']['d'] = [3.4, 5.4, 2.8]
        plan['front_wall'] = False
        for camera in plan['cameras']:
            camera['d'] = [1.7, .7, 1.5, 1.55, 4.6, 1.15, 20, 0]
        plan['assets'][0]['d'][:7] = [1.55, 4.85, 0, 1.3, .5, .76, 0]
        plan['groups'][0].update(p=[.62, 2.15, .9], d=[1.1, 2.3, 1.95])
        plan['groups'][1].update(p=[2.78, 2.15, .9], d=[1.1, 2.3, 1.95])
        before = copy.deepcopy(plan)
        with self.assertRaises(staged.PlanGeometryError) as raised:
            staged.validate_plan(plan, self.views, 'fixture_scene')
        message = staged._feedback(raised.exception)
        self.assertIn('right', message)
        self.assertIn('groups[1]', message)
        self.assertIn('X=[-1.700,1.700]', message)
        self.assertEqual(plan, before)

    def test_camera_clearance_rejects_retained_shell_and_permits_openings(self):
        width, depth, height = self.plan['room']['d']
        poses = {'left': [-width / 2, 2, 1.6], 'right': [width / 2 - .02, 2, 1.6],
                 'back': [0, depth, 1.6], 'front': [0, 0, 1.6],
                 'floor': [0, 2, .02], 'ceiling': [0, 2, height - .02]}
        for surface, position in poses.items():
            plan = copy.deepcopy(self.plan)
            plan['omit'] = []
            plan['cameras'][0]['d'][:3] = position
            with self.subTest(surface=surface), self.assertRaises(staged.PlanGeometryError):
                staged.validate_plan(plan, self.views, 'fixture_scene')
            if surface == 'front':
                plan['front_wall'] = False
            else:
                plan['omit'] = [surface]
            staged.validate_plan(plan, self.views, 'fixture_scene')

    def test_rotated_and_repeated_furniture_bounds_are_checked(self):
        for rotated in (False, True):
            plan = copy.deepcopy(self.plan)
            dimensions = plan['assets'][0]['d']
            if rotated:
                dimensions[:7] = [2.4, 3, 0, 1.2, 1.2, .75, 45]
            else:
                dimensions[10:14] = [2, 1, 4, 0]
            with self.subTest(rotated=rotated), self.assertRaisesRegex(staged.PlanGeometryError, 'assets\\[0\\]'):
                staged.validate_plan(plan, self.views, 'fixture_scene')

    def test_external_lights_and_legitimate_shell_attachments_are_allowed(self):
        plan = copy.deepcopy(self.plan)
        width, depth, height = plan['room']['d']
        plan['lights'][0]['d'][:6] = [0, depth + 2, height + 1, 0, 3, 1]
        plan['assets'].append({'id': 'wall_window', 'category': 'window',
            'd': [0, depth, 1, 2, .2, 1.2, 0, .6, .6, .6, 1, 1, 0, 0, .8]})
        plan['groups'][0].update(p=[width / 2 + .075, 2, 1.4], d=[.15, 2, 2.8])
        before = copy.deepcopy(plan)
        self.assertEqual(staged.validate_plan(plan, self.views, 'fixture_scene'), before)
        # Group origins are explicit pivots, not bounding-box centres. A floor
        # pivot and desired full height must not imply below-floor geometry.
        plan['groups'][0].update(p=[0, 2, 0], d=[2, 2, height])
        staged.validate_plan(plan, self.views, 'fixture_scene')

    def test_plan_geometry_feedback_replans_before_any_detail_call(self):
        def reply(stage, attempt, prompt):
            if stage != 'plan':
                self.assertEqual(self.counter['plan'], 2)
            result = self.default_reply(stage, attempt, prompt)
            if stage == 'plan' and attempt == 1:
                result['cameras'][0]['d'][0] = result['room']['d'][0] / 2
            return result
        self.replies = reply
        self.analyze()
        self.assertEqual(self.counter['plan'], 2)
        self.assertEqual(self.counter['group_0'], 1)
        feedback = (self.output.parent / 'initial_bundle_stages/plan_attempt_01.prompt.txt').read_text()
        self.assertIn('Plan geometry invalid:', feedback)
        self.assertIn('room centre X=0', feedback)

    def geometry_patch_reply(self, draft, value=-1, field='position.x'):
        from plan_geometry_patch import build_request
        # The loopback provider emulates the request schema by omitting
        # optional routing metadata when routing is disabled. Geometry patch
        # hashes must bind to that exact wire candidate, not the richer source
        # fixture used to seed the provider response.
        draft = self.wire_plan(draft)
        _, schema = build_request(draft, ['desk'], 'fixture geometry error', 'fixture contract')
        return {'base_sha256': schema['properties']['base_sha256']['enum'][0],
                'edits': [{'asset_id': 'desk', 'field': field, 'value': value}]}

    def wire_plan(self, plan):
        result = copy.deepcopy(plan)
        if not self.options.get('route_detail_images', False):
            for group in result.get('groups', []):
                for key in ('evidence_view_ids', 'evidence_confidence',
                            'texture_material_ids', 'texture_material_confidence'):
                    group.pop(key, None)
        return result

    def test_asset_geometry_retry_preserves_complete_plan_and_resumes_patch(self):
        draft = copy.deepcopy(self.plan)
        draft['assets'][0]['d'][0] = 2.8
        def reply(stage, attempt, prompt):
            if stage == 'plan':
                return draft if attempt == 1 else self.geometry_patch_reply(draft)
            return self.default_reply(stage, attempt, prompt)
        self.replies = reply
        first = self.analyze()
        self.assertEqual(self.counter['plan'], 2)
        folder = self.output.with_name('initial_bundle_stages')
        self.assertEqual(read_json(folder / 'plan_attempt_01.canonical.json'), self.wire_plan(self.plan))
        correction = self.requests[1]
        self.assertIn('base_sha256', correction['response_format']['json_schema']['schema']['properties'])
        patch_prompt = correction['messages'][0]['content'][0]['text']
        self.assertIn('desk', patch_prompt)
        self.assertIn('native_plan', patch_prompt)
        self.assertEqual(len([v for v in correction['messages'][0]['content'] if v['type'] == 'image_url']), 2)
        state = read_json(folder / 'plan_attempt_01.validation.json')
        self.assertEqual(state['request_kind'], 'asset_geometry_patch')
        self.assertIn('patch_base', state['artifacts'])
        count = len(self.requests)
        self.assertEqual(self.analyze(), first)
        self.assertEqual(len(self.requests), count)

    def test_failed_geometry_candidate_resumes_with_patch_without_regeneration(self):
        draft = copy.deepcopy(self.plan)
        draft['assets'][0]['d'][0] = 2.8
        self.options['validation_retries'] = 0
        self.replies = lambda stage, attempt, prompt: (draft if attempt == 1 else
            self.geometry_patch_reply(draft)) if stage == 'plan' else fixture_detail(stage)
        with self.assertRaises(staged.StagedAnalysisError):
            self.analyze()
        self.assertEqual(self.counter, {'plan': 1})
        self.analyze()
        self.assertEqual(self.counter['plan'], 2)
        self.assertIn('base_sha256', self.requests[1]['response_format']['json_schema']['schema']['properties'])

    def test_geometry_patch_still_outside_room_requires_another_validated_patch(self):
        draft = copy.deepcopy(self.plan)
        draft['assets'][0]['d'][0] = 2.8
        intermediate = copy.deepcopy(draft)
        intermediate['assets'][0]['d'][0] = 2.7
        self.options['validation_retries'] = 0
        def reply(stage, attempt, prompt):
            if stage == 'plan':
                if attempt == 1: return draft
                return self.geometry_patch_reply(draft, 2.7) if attempt == 2 else self.geometry_patch_reply(intermediate)
            return fixture_detail(stage)
        self.replies = reply
        for _ in range(2):
            with self.assertRaises(staged.StagedAnalysisError):
                self.analyze()
        self.assertEqual(self.counter, {'plan': 2})
        self.analyze()
        self.assertEqual(self.counter['plan'], 3)
        self.assertEqual(read_json(self.output.with_name('initial_bundle_stages') / 'plan_attempt_02.canonical.json'), self.wire_plan(self.plan))

    def test_invalid_patch_keeps_original_bound_base_for_retry(self):
        draft = copy.deepcopy(self.plan)
        draft['assets'][0]['d'][0] = 2.8
        self.options['validation_retries'] = 2
        def reply(stage, attempt, prompt):
            if stage == 'plan':
                if attempt == 1: return draft
                return self.geometry_patch_reply(draft, 2.8 if attempt == 2 else -1)
            return fixture_detail(stage)
        self.replies = reply
        self.analyze()
        self.assertEqual(self.counter['plan'], 3)
        folder = self.output.with_name('initial_bundle_stages')
        self.assertEqual(read_json(folder / 'plan_attempt_02.patch_base.json'), self.wire_plan(draft))
        self.assertEqual(read_json(folder / 'plan_attempt_02.canonical.json'), self.wire_plan(self.plan))

    def test_tampered_rejected_candidate_never_triggers_patch_call(self):
        draft = copy.deepcopy(self.plan)
        draft['assets'][0]['d'][0] = 2.8
        self.options['validation_retries'] = 0
        self.replies = lambda stage, attempt, prompt: draft
        with self.assertRaises(staged.StagedAnalysisError):
            self.analyze()
        path = self.output.with_name('initial_bundle_stages') / 'plan_attempt_00.raw.json'
        changed = read_json(path)
        changed['assets'][0]['d'][0] = 2.7
        atomic_json(path, changed)
        with self.assertRaisesRegex(staged.StagedAnalysisError, 'artifact changed'):
            self.analyze()
        self.assertEqual(self.counter, {'plan': 1})

    def test_geometry_candidate_provenance_is_verified_before_retry(self):
        draft = copy.deepcopy(self.plan)
        draft['assets'][0]['d'][0] = 2.8
        self.replies = lambda stage, attempt, prompt: draft
        original = staged.call_model
        def wrong_provenance(prompt, images, schema, output, *args, **kwargs):
            data = original(prompt, images, schema, output, *args, **kwargs)
            path = Path(output).with_suffix('.api.json')
            api = read_json(path)
            api['prompt_sha256'] = '0' * 64
            atomic_json(path, api)
            return data
        with patch.object(staged, 'call_model', side_effect=wrong_provenance):
            with self.assertRaisesRegex(staged.StagedAnalysisError, 'provenance'):
                self.analyze()
        self.assertEqual(self.counter, {'plan': 1})

    def test_invalid_patch_response_resume_keeps_bound_candidate(self):
        draft = copy.deepcopy(self.plan)
        draft['assets'][0]['d'][0] = 2.8
        def reply(stage, attempt, prompt):
            if stage == 'plan':
                if attempt == 1: return draft
                if attempt == 2: return '{"edits":'
                return self.geometry_patch_reply(draft)
            return fixture_detail(stage)
        self.replies = reply
        with self.assertRaises(staged.StagedAnalysisError):
            self.analyze()
        self.assertEqual(self.counter, {'plan': 2})
        self.analyze()
        self.assertEqual(self.counter['plan'], 3)
        self.assertIn('base_sha256', self.requests[2]['response_format']['json_schema']['schema']['properties'])
        self.assertEqual(read_json(self.output.with_name('initial_bundle_stages') / 'plan_attempt_02.canonical.json'), self.wire_plan(self.plan))

    def test_geometry_patch_respects_scene_budget_before_transport(self):
        from scene_budget import scene_budget_session, budget_snapshot
        draft = copy.deepcopy(self.plan)
        draft['assets'][0]['d'][0] = 2.8
        self.replies = lambda stage, attempt, prompt: draft
        with scene_budget_session(self.output.parent, {'max_requests': 1}):
            with self.assertRaises(ModelAPIError) as caught:
                self.analyze()
        self.assertEqual(caught.exception.code, 'scene_budget_exceeded')
        self.assertEqual(self.counter, {'plan': 1})
        state = read_json(self.output.with_name('initial_bundle_stages') / 'plan_attempt_01.validation.json')
        self.assertEqual(state['request_kind'], 'asset_geometry_patch')
        self.assertEqual(state['status'], 'failed')

    def test_invalid_plan_never_spends_detail_budget_after_retry_limit(self):
        self.plan['cameras'][0]['d'][0] = self.plan['room']['d'][0] / 2
        with self.assertRaisesRegex(staged.StagedAnalysisError, 'Plan geometry invalid'):
            self.analyze()
        self.assertEqual(self.counter, {'plan': 2})
        self.assertFalse(self.output.exists())

    def test_semantic_retry_then_exact_cache_reuse(self):
        def reply(stage, attempt, prompt):
            result = self.default_reply(stage, attempt, prompt)
            if stage == 'group_0' and attempt == 1:
                result['parts'][0]['data'][3] = 0
            return result
        self.replies = reply
        first = self.analyze()
        self.assertEqual(self.counter['group_0'], 2)
        self.assertEqual(self.counter['group_1'], 1)
        count = len(self.requests)
        second = self.analyze()
        self.assertEqual(first, second)
        self.assertEqual(len(self.requests), count)
        prompt = (self.output.parent / 'initial_bundle_stages/detail_group_0_attempt_01.prompt.txt').read_text()
        self.assertIn('box: size must be positive', prompt)

    def test_invalid_json_and_schema_retry(self):
        for failure in ('json', 'schema'):
            self.counter.clear()
            def reply(stage, attempt, prompt):
                if stage == 'plan' and attempt == 1:
                    return '{"room":' if failure == 'json' else {'room': 'wrong'}
                return self.default_reply(stage, attempt, prompt)
            self.replies = reply
            with self.subTest(failure=failure):
                self.analyze(output=self.root / failure / 'initial_bundle.json')
                self.assertEqual(self.counter['plan'], 2)

    def test_model_photo_options_changes_refuse_cached_stage(self):
        self.analyze()
        count = len(self.requests)
        with self.assertRaisesRegex(staged.StagedAnalysisError, 'changed'):
            self.analyze(model_config={**self.config, 'model': 'another-model'})
        with self.assertRaisesRegex(staged.StagedAnalysisError, 'changed'):
            self.analyze(options={**self.options, 'max_groups': 4})
        Image.new('RGB', (64, 48), (4, 8, 12)).save(self.views[0]['source_image'])
        with self.assertRaisesRegex(staged.StagedAnalysisError, 'changed'):
            self.analyze()
        self.assertEqual(len(self.requests), count)

    def test_tampered_cache_fails_without_becoming_accepted(self):
        self.analyze()
        raw = self.output.parent / 'initial_bundle_stages/detail_group_0_attempt_00.raw.json'
        value = read_json(raw)
        value['parts'][0]['data'][3] = 0
        atomic_json(raw, value)
        count = len(self.requests)
        with self.assertRaisesRegex(staged.StagedAnalysisError, 'Detail groups failed'):
            self.analyze()
        failures = read_json(raw.parent / 'failures.json')
        self.assertIn('artifact changed', failures['failed_groups']['group_0'])
        self.assertEqual(len(self.requests), count)

    def test_all_failed_groups_reported_no_incomplete_scene(self):
        def reply(stage, attempt, prompt):
            result = self.default_reply(stage, attempt, prompt)
            if stage != 'plan':
                result['parts'][0]['data'][3] = 0
            return result
        self.replies = reply
        with self.assertRaisesRegex(staged.StagedAnalysisError, 'group_0, group_1'):
            self.analyze()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.counter['group_0'], 2)
        self.assertEqual(self.counter['group_1'], 2)
        self.assertEqual(set(read_json(self.output.parent / 'initial_bundle_stages/failures.json')['failed_groups']), {'group_0', 'group_1'})
        with self.assertRaises(staged.StagedAnalysisError):
            self.analyze()
        self.assertEqual(self.counter['group_0'], 4)
        self.assertEqual(self.counter['group_1'], 4)
        self.assertTrue((self.output.parent / 'initial_bundle_stages/detail_group_0_attempt_03.validation.json').is_file())

    def test_transient_network_failure_can_resume_without_repeating_plan(self):
        original = staged.call_model
        failed = False
        def fail_one_detail(prompt, images, schema, output, *args, **kwargs):
            nonlocal failed
            if schema['properties'].get('group_id', {}).get('enum') == ['group_0'] and not failed:
                failed = True
                raise ModelAPIError('timeout', 'Synthetic transient timeout')
            return original(prompt, images, schema, output, *args, **kwargs)
        with patch.object(staged, 'call_model', side_effect=fail_one_detail):
            with self.assertRaisesRegex(staged.StagedAnalysisError, 'group_0'):
                self.analyze()
        state = self.output.parent / 'initial_bundle_stages/detail_group_0_attempt_00.validation.json'
        before = staged.file_sha256(state)
        result = self.analyze()
        self.assertEqual(result['scene_id'], 'fixture_scene')
        self.assertEqual(self.counter['plan'], 1)
        self.assertEqual(self.counter['group_1'], 1)
        self.assertEqual(staged.file_sha256(state), before)
        self.assertTrue((state.parent / 'detail_group_0_attempt_01.validation.json').is_file())

    def test_interrupted_attempt_is_preserved_and_new_invocation_can_resume(self):
        with patch.object(staged, 'call_model', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.analyze()
        state = self.output.parent / 'initial_bundle_stages/plan_attempt_00.validation.json'
        self.assertEqual(read_json(state)['status'], 'running')
        before = staged.file_sha256(state)
        self.analyze()
        self.assertEqual(staged.file_sha256(state), before)
        self.assertTrue((state.parent / 'plan_attempt_01.validation.json').is_file())
        self.assertEqual(self.counter['plan'], 1)

    def test_missing_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'All planned groups'):
            staged.to_bundle(self.plan, self.views, 'fixture_scene', {'group_0': fixture_detail('group_0')})

    def test_geometry_and_reference_validation_precedes_final_assembly(self):
        cases = []
        mesh = fixture_detail('group_0', 'mesh')
        mesh['parts'][0].update(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 7]])
        cases.append(mesh)
        curve = fixture_detail('group_0', 'curve')
        curve['parts'][0].update(points=[[0, 0, 0], [0, 0, 0]])
        curve['parts'][0]['data'][9] = .1
        cases.append(curve)
        unknown_material = fixture_detail('group_0')
        unknown_material['parts'][0]['material_id'] = 'not_in_plan'
        cases.append(unknown_material)
        for value in cases:
            with self.subTest(value=value['parts'][0]['type']), self.assertRaises(ValueError):
                staged.validate_detail(value, self.plan['groups'][0], self.plan, self.views, 'fixture_scene')

    def test_sanitized_feedback_never_echoes_arbitrary_reference(self):
        marker = 'UNTRUSTED_SECRET_VALUE'
        self.assertNotIn(marker, staged._feedback(ValueError('parts.material_id: unknown reference ' + marker)))
        self.assertNotIn(marker, staged._feedback(ValueError('native: missing or extra fields {' + marker + '}')))
        self.assertNotIn(marker, staged._feedback(ValueError(marker)))

    def test_explicit_effort_and_no_api_fallback(self):
        self.analyze(options={**self.options, 'planner_effort': 'high', 'detail_effort': 'low'})
        for req in self.requests:
            schema = req['response_format']['json_schema']['schema']
            self.assertEqual(req['reasoning_effort'], 'low' if 'group_id' in schema['properties'] else 'high')
        with self.assertRaisesRegex(ValueError, 'explicit API'):
            self.analyze(model_config=None)
        with self.assertRaisesRegex(ValueError, 'part budget'):
            staged.normalize_options({'max_groups': 16, 'max_parts_per_group': 100})
        self.assertEqual(staged.normalize_options({'planner_effort': 'max'})['planner_effort'], 'max')


if __name__ == '__main__':
    unittest.main()
