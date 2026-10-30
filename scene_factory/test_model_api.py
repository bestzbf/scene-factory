"""Provider contract tests using only a loopback fixture server (no paid calls)."""
import base64
import copy
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

import model_api as api
import vision_multiview
from batch import atomic_json, file_sha256, read_json


SCHEMA = {'type': 'object', 'properties': {
    'scene': {'type': 'string', 'maxLength': 20},
    'number': {'type': 'integer', 'minimum': 1, 'maximum': 5}},
    'required': ['scene', 'number'], 'additionalProperties': False}
RESULT = {'scene': 'fixture', 'number': 2}
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC')


def chat(content=None, finish='stop', **message_extra):
    return {'choices': [{'finish_reason': finish, 'message': {
        'content': json.dumps(RESULT) if content is None else content, **message_extra}}],
        'usage': {'prompt_tokens': 12, 'completion_tokens': 7, 'total_tokens': 19}}


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.server.requests.append({'path': self.path, 'headers': dict(self.headers), 'body': json.loads(body)})
        status, headers, response = self.server.responses.pop(0)
        if callable(response):
            response = response(self)
        data = response if isinstance(response, bytes) else json.dumps(response).encode()
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


class ModelAPITests(unittest.TestCase):
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
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.image = self.folder / 'one.png'
        self.image.write_bytes(PNG)
        self.jpeg = self.folder / 'two.jpg'
        jpeg = io.BytesIO()
        Image.new('RGB', (1, 1), (0, 255, 0)).save(jpeg, format='JPEG')
        self.jpeg.write_bytes(jpeg.getvalue())
        self.output = self.folder / 'analysis.json'
        self.server.requests, self.server.responses = [], []
        self.config = {'provider': 'openai_chat', 'model': 'fixture-model-a',
                       'base_url': f'http://127.0.0.1:{self.server.server_port}/v1',
                       'api_key_env': '', 'max_retries': 0}

    def queue(self, body=None, status=200, **headers):
        self.server.responses.append((status, headers, chat() if body is None else body))

    def call(self, **overrides):
        config = {**self.config, **overrides}
        return api.call_model('Analyse these photos in supplied order.', [self.image, self.jpeg], SCHEMA, self.output, config=config)

    def record(self):
        return json.loads(self.output.with_name('analysis.api.json').read_text())

    def test_chat_schema_images_order_model_and_output(self):
        self.queue()
        self.assertEqual(self.call(), RESULT)
        req = self.server.requests[0]
        self.assertEqual(req['path'], '/v1/chat/completions')
        payload = req['body']
        self.assertEqual(payload['model'], 'fixture-model-a')
        self.assertEqual(payload['max_completion_tokens'], 32768)
        self.assertNotIn('reasoning_effort', payload)
        self.assertNotIn('Authorization', req['headers'])
        self.assertEqual(payload['response_format']['json_schema']['schema'], SCHEMA)
        self.assertTrue(payload['response_format']['json_schema']['strict'])
        content = payload['messages'][0]['content']
        self.assertTrue(content[0]['text'].startswith('Analyse these photos'))
        self.assertEqual(content[1]['text'], 'Image 1 (input order):')
        self.assertTrue(content[2]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertEqual(content[3]['text'], 'Image 2 (input order):')
        self.assertTrue(content[4]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(json.loads(self.output.with_name('analysis.raw.json').read_text()), RESULT)
        record = self.record()
        self.assertEqual(record['status'], 'completed')
        self.assertNotIn('schema_validation', record)
        self.assertEqual(record['usage']['total_tokens'], 19)
        self.assertEqual(len(record['image_sha256']), 2)
        metrics = record['request_metrics']
        self.assertEqual(metrics['image_count'], 2)
        self.assertEqual(metrics['image_source_bytes_total'], sum(item['source_bytes'] for item in metrics['images']))
        self.assertEqual(metrics['image_base64_bytes_total'], sum(item['base64_bytes'] for item in metrics['images']))
        self.assertEqual(metrics['serialized_request_body_utf8_bytes'],
                         len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')))
        self.assertEqual(record['output_text_utf8_bytes'], len(json.dumps(RESULT).encode('utf-8')))
        self.assertGreater(record['transport_attempts'][0]['response_bytes'], 0)
        self.assertIn('not provider token counts', record['request_metrics_semantics'])
        self.assertNotIn('data:image', json.dumps(record))

    def analysis_fixture(self):
        from test_vision_multiview import JointVisionTests
        _, views, raw = JointVisionTests().prepare(self.folder.resolve())
        return views, raw

    def analyze(self, views, **kwargs):
        return vision_multiview.analyze('room', views, self.output,
                                       model_config=kwargs.pop('config', self.config), **kwargs)

    def test_analysis_semantic_failure_resume_reuses_one_http_response_and_usage(self):
        views, raw = self.analysis_fixture()
        raw['views'][0]['camera']['target'] = raw['views'][0]['camera']['position'][:]
        self.queue(chat(json.dumps(raw)))
        for invocation in range(2):
            with self.subTest(invocation=invocation), self.assertRaisesRegex(ValueError, 'coincides'):
                self.analyze(views)
            saved = {p.name: file_sha256(p) for p in self.folder.glob('analysis.*') if p.is_file()}
            if invocation == 0:
                original = saved
            else:
                self.assertEqual(saved, original)
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.record()['usage']['total_tokens'], 19)
        self.assertFalse((self.folder/'analysis_provenance.json').exists())
        self.assertEqual(len(list(self.folder.glob('*.api.json'))), 1)

    def test_analysis_commit_gap_resume_reuses_one_http_response(self):
        views, raw = self.analysis_fixture()
        self.queue(chat(json.dumps(raw)))
        def fail_commit(path, value):
            if Path(path).name == 'analysis_provenance.json':
                raise OSError('Synthetic interruption before provenance commit')
            atomic_json(path, value)
        with patch.object(vision_multiview, 'atomic_json', side_effect=fail_commit):
            with self.assertRaisesRegex(OSError, 'Synthetic interruption'):
                self.analyze(views)
        saved = {p.name: file_sha256(p) for p in self.folder.glob('analysis.*') if p.is_file()}
        result = self.analyze(views)
        self.assertEqual(result['scene_id'], 'room')
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(saved, {self.folder.joinpath(name).name: file_sha256(self.folder/name) for name in saved})
        provenance = read_json(self.folder/'analysis_provenance.json')
        self.assertEqual(provenance['raw_model_response_cache']['method'], 'same_stage_completed_api_response')

    def test_analysis_completed_resume_checks_all_request_and_transport_evidence(self):
        views, raw = self.analysis_fixture()
        raw['views'][0]['camera']['target'] = raw['views'][0]['camera']['position'][:]
        self.queue(chat(json.dumps(raw)))
        with self.assertRaises(ValueError):
            self.analyze(views)
        api_path = self.output.with_suffix('.api.json')
        original_api = api_path.read_bytes()
        raw_path = self.output.with_suffix('.raw.json')
        original_raw = raw_path.read_bytes()
        prompt_path = self.output.with_suffix('.prompt.txt')
        original_prompt = prompt_path.read_bytes()
        def audit_change(key, value):
            record = json.loads(original_api)
            record[key] = value
            atomic_json(api_path, record)
        actions = [
            lambda: audit_change('prompt_sha256', '0'*64),
            lambda: audit_change('schema_sha256', '0'*64),
            lambda: audit_change('image_sha256', list(reversed(self.record()['image_sha256']))),
            lambda: audit_change('config', {**self.record()['config'], 'model':'changed'}),
            lambda: audit_change('raw_response_sha256', '0'*64),
            lambda: audit_change('attempts', 3),
            lambda: audit_change('transport_attempts', []),
            lambda: audit_change('failure_code', 'incomplete'),
            lambda: audit_change('finish_reason', 'length'),
            lambda: raw_path.write_bytes(original_raw + b' '),
            lambda: prompt_path.write_bytes(original_prompt + b'changed'),
        ]
        for action in actions:
            api_path.write_bytes(original_api)
            raw_path.write_bytes(original_raw)
            prompt_path.write_bytes(original_prompt)
            action()
            before = {p.name:file_sha256(p) for p in self.folder.glob('analysis.*') if p.is_file()}
            with self.assertRaisesRegex(ValueError, 'automatic model replay was blocked'):
                self.analyze(views)
            self.assertEqual(before, {p.name:file_sha256(p) for p in self.folder.glob('analysis.*') if p.is_file()})
        self.assertEqual(len(self.server.requests), 1)

    def test_analysis_explicit_http_rejection_creates_new_immutable_audit(self):
        views, raw = self.analysis_fixture()
        self.queue({'error':{'message':'synthetic quota'}}, status=402)
        with self.assertRaises(api.ModelAPIError):
            self.analyze(views)
        old = {p.name:file_sha256(p) for p in self.folder.glob('analysis.*') if p.is_file()}
        self.queue(chat(json.dumps(raw)))
        result = self.analyze(views)
        self.assertEqual(result['scene_id'], 'room')
        self.assertEqual(len(self.server.requests), 2)
        self.assertTrue((self.folder/'analysis_retry_1.api.json').is_file())
        self.assertTrue((self.folder/'analysis_retry_1.raw.json').is_file())
        self.assertTrue((self.folder/'analysis_retry_1.prompt.txt').is_file())
        for name, digest in old.items():
            self.assertEqual(file_sha256(self.folder/name), digest)
        self.assertEqual(read_json(self.folder/'analysis_retry_1.api.json')['usage']['total_tokens'], 19)

    def test_analysis_failed_audit_conflicts_never_send_or_rewrite(self):
        views, _ = self.analysis_fixture()
        self.queue({'error': {'message': 'synthetic quota'}}, status=402)
        with self.assertRaises(api.ModelAPIError):
            self.analyze(views)
        api_path = self.output.with_suffix('.api.json')
        original = self.record()
        cases = []
        for changes in ({'attempts': True}, {'attempts': 2},
                        {'attempts': -1}, {'attempts': 0, 'transport_attempts': []}):
            cases.append({**copy.deepcopy(original), **changes})
        for changes in ({'attempt': True}, {'attempt': 2}, {'request_started': False},
                        {'request_sent': False}, {'phase': 'connecting'},
                        {'http_status': 200}, {'finish_reason': 'stop'},
                        {'usage': {'total_tokens': 9}}, {'completion_marker_missing': True}):
            record = copy.deepcopy(original)
            record['transport_attempts'][0].update(changes)
            cases.append(record)
        unsent = copy.deepcopy(original)
        unsent.pop('http_status')
        unsent['failure_code'] = 'network_error'
        unsent['transport_attempts'] = [{'attempt': 1, 'phase': 'connecting',
            'request_started': False, 'request_sent': False}]
        for changes in ({'phase': 'response_complete'}, {'http_status': 200},
                        {'response_bytes': 80}, {'usage': {'total_tokens': 9}},
                        {'response_status': 'completed'}, {'finish_reason': 'stop'},
                        {'completion_marker_missing': True}, {'usage_invalid': True}):
            record = copy.deepcopy(unsent)
            record['transport_attempts'][0].update(changes)
            cases.append(record)
        # Even if the individual attempt claims not to have been sent, the
        # aggregate record must not contain contradictory response evidence.
        for changes in ({'http_status': 200}, {'usage': {'total_tokens': 9}},
                        {'response_status': 'completed'}, {'finish_reason': 'stop'}):
            cases.append({**copy.deepcopy(unsent), **changes})
        for record in cases:
            with self.subTest(record=record):
                atomic_json(api_path, record)
                before = {p.name: file_sha256(p) for p in self.folder.glob('analysis*') if p.is_file()}
                with patch.object(api, '_request', side_effect=AssertionError('No new POST')) as request:
                    with self.assertRaisesRegex(ValueError, 'automatic model replay was blocked'):
                        self.analyze(views)
                    request.assert_not_called()
                self.assertEqual(before, {p.name: file_sha256(p) for p in self.folder.glob('analysis*') if p.is_file()})
        self.assertEqual(len(self.server.requests), 1)

    def test_analysis_proven_unsent_failure_allows_one_new_immutable_attempt(self):
        views, raw = self.analysis_fixture()
        def refuse(config, body, key, deadline, *, transport, stop_event):
            transport.update(phase='connecting', request_started=False, request_sent=False)
            raise ConnectionRefusedError('synthetic loopback refusal')
        with patch.object(api, '_request', side_effect=refuse), self.assertRaises(api.ModelAPIError):
            self.analyze(views)
        before = {p.name: file_sha256(p) for p in self.folder.glob('analysis.*') if p.is_file()}
        self.queue(chat(json.dumps(raw)))
        self.assertEqual(self.analyze(views)['scene_id'], 'room')
        self.assertEqual(len(self.server.requests), 1)
        self.assertTrue((self.folder/'analysis_retry_1.api.json').is_file())
        self.assertEqual(before, {name: file_sha256(self.folder/name) for name in before})

    def test_analysis_successful_http_with_invalid_json_is_not_automatically_replayed(self):
        views, _ = self.analysis_fixture()
        self.queue(chat('{'))
        with self.assertRaises(api.ModelAPIError):
            self.analyze(views)
        old = {p.name:file_sha256(p) for p in self.folder.glob('analysis.*') if p.is_file()}
        with self.assertRaisesRegex(ValueError, 'automatic model replay was blocked'):
            self.analyze(views)
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(old, {p.name:file_sha256(p) for p in self.folder.glob('analysis.*') if p.is_file()})
        self.assertFalse(self.output.exists())

    def test_structured_json_schema_mode_does_not_repeat_generic_output_instruction(self):
        self.queue()
        self.call()
        body = self.server.requests[0]['body']
        prompt = body['messages'][0]['content'][0]['text']
        self.assertEqual(prompt, 'Analyse these photos in supplied order.')
        self.assertNotIn('Return exactly one JSON object', prompt)
        self.assertEqual(body['response_format']['json_schema']['schema'], SCHEMA)

    def test_unstructured_modes_keep_explicit_schema_instruction(self):
        self.queue()
        self.call(response_format='prompt')
        prompt = self.server.requests[0]['body']['messages'][0]['content'][0]['text']
        self.assertIn('Return exactly one JSON object matching the supplied schema.', prompt)
        self.assertIn('JSON SCHEMA:', prompt)

    def test_all_adapters_upload_full_resolution_source_bytes_in_order(self):
        from PIL import Image
        images = [self.folder / 'large.png', self.folder / 'portrait.jpg']
        sizes = [(2048, 1536), (1200, 1800)]
        for path, size in zip(images, sizes):
            Image.new('RGB', size, (40, 90, 130)).save(path)
        originals = [path.read_bytes() for path in images]
        replies = {
            'openai_chat': chat(),
            'openai_responses': {'status': 'completed', 'output': [
                {'type': 'message', 'status': 'completed', 'content': [
                    {'type': 'output_text', 'text': json.dumps(RESULT)}]}]},
            'anthropic': {'type': 'message', 'stop_reason': 'end_turn', 'content': [
                {'type': 'text', 'text': json.dumps(RESULT)}]},
        }
        for provider, response in replies.items():
            with self.subTest(provider=provider):
                self.queue(response)
                api.call_model('Preserve all original views.', images, SCHEMA,
                               self.folder / (provider + '.json'),
                               config={**self.config, 'provider': provider})
                body = self.server.requests[-1]['body']
                content = (body['input'] if provider == 'openai_responses' else body['messages'])[0]['content']
                transmitted = []
                for item in content:
                    if item['type'] == 'image_url':
                        transmitted.append(base64.b64decode(item['image_url']['url'].split(',', 1)[1]))
                    elif item['type'] == 'input_image':
                        transmitted.append(base64.b64decode(item['image_url'].split(',', 1)[1]))
                    elif item['type'] == 'image':
                        transmitted.append(base64.b64decode(item['source']['data']))
                self.assertEqual(transmitted, originals)
                for data, size in zip(transmitted, sizes):
                    with Image.open(io.BytesIO(data)) as image:
                        self.assertEqual(image.size, size)
        self.assertEqual([path.read_bytes() for path in images], originals)

    def test_different_chat_model_and_prompt_mode(self):
        self.queue()
        self.call(model='other-vendor/vision-b', response_format='prompt', temperature=.3)
        body = self.server.requests[0]['body']
        self.assertEqual(body['model'], 'other-vendor/vision-b')
        self.assertEqual(body['temperature'], .3)
        self.assertNotIn('response_format', body)
        self.assertIn('JSON SCHEMA:', body['messages'][0]['content'][0]['text'])

    def test_chat_json_object_mode(self):
        self.queue()
        self.call(response_format='json_object', reasoning_effort='low')
        payload = self.server.requests[0]['body']
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertEqual(payload['reasoning_effort'], 'low')
        self.assertIn('JSON SCHEMA:', payload['messages'][0]['content'][0]['text'])

    def test_glm_max_effort_and_explicit_nonstream(self):
        self.queue()
        self.assertEqual(self.call(reasoning_effort='max', stream=False), RESULT)
        payload = self.server.requests[0]['body']
        self.assertEqual(payload['reasoning_effort'], 'max')
        self.assertIs(payload['stream'], False)

    def test_responses_contract(self):
        self.queue({'status': 'completed', 'output': [{'type': 'reasoning'},
            {'type': 'message', 'status': 'completed', 'content': [{'type': 'output_text', 'text': json.dumps(RESULT)}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30}})
        self.assertEqual(self.call(provider='openai_responses', model='other-responses-model', reasoning_effort='high'), RESULT)
        req = self.server.requests[0]
        self.assertEqual(req['path'], '/v1/responses')
        body = req['body']
        self.assertEqual(body['model'], 'other-responses-model')
        self.assertEqual(body['reasoning'], {'effort': 'high'})
        self.assertEqual(body['text']['format']['schema'], SCHEMA)
        self.assertFalse(body['store'])
        self.assertEqual(body['input'][0]['content'][0]['type'], 'input_text')
        self.assertEqual(body['input'][0]['content'][2]['type'], 'input_image')

    def test_anthropic_contract(self):
        self.queue({'type': 'message', 'stop_reason': 'end_turn', 'content': [
            {'type': 'text', 'text': json.dumps(RESULT)}], 'usage': {'input_tokens': 5, 'output_tokens': 9}})
        with patch.dict(os.environ, {'FIXTURE_MODEL_KEY': 'fixture-secret'}):
            self.assertEqual(self.call(provider='anthropic', model='claude-fixture', api_key_env='FIXTURE_MODEL_KEY'), RESULT)
        req = self.server.requests[0]
        self.assertEqual(req['path'], '/v1/messages')
        self.assertEqual(req['headers']['x-api-key'], 'fixture-secret')
        self.assertEqual(req['headers']['anthropic-version'], '2023-06-01')
        self.assertNotIn('Authorization', req['headers'])
        self.assertEqual(req['body']['max_tokens'], 32768)
        self.assertIn('JSON SCHEMA:', req['body']['messages'][0]['content'][0]['text'])
        source = req['body']['messages'][0]['content'][2]['source']
        self.assertEqual(source['media_type'], 'image/png')
        self.assertEqual(base64.b64decode(source['data']), PNG)
        self.assertNotIn('fixture-secret', self.output.with_name('analysis.api.json').read_text())

    def test_anthropic_root_endpoint_bearer_and_thinking(self):
        self.queue({'type': 'message', 'stop_reason': 'end_turn', 'content': [
            {'type': 'thinking', 'thinking': 'private reasoning'},
            {'type': 'redacted_thinking', 'data': 'private'},
            {'type': 'text', 'text': json.dumps(RESULT)}]})
        with patch.dict(os.environ, {'FIXTURE_MODEL_KEY': 'fixture-secret'}):
            self.assertEqual(self.call(provider='anthropic', model='glm-5.3-flash',
                base_url=f'http://127.0.0.1:{self.server.server_port}/api/anthropic',
                api_key_env='FIXTURE_MODEL_KEY', auth_scheme='bearer'), RESULT)
        req = self.server.requests[0]
        self.assertEqual(req['path'], '/api/anthropic/v1/messages')
        self.assertEqual(req['headers']['Authorization'], 'Bearer fixture-secret')
        self.assertNotIn('x-api-key', req['headers'])
        self.assertNotIn('private', self.output.with_name('analysis.api.json').read_text())

    def test_chat_explicit_legacy_token_parameter(self):
        self.queue()
        self.call(token_parameter='max_tokens', max_output_tokens=8192)
        payload = self.server.requests[0]['body']
        self.assertEqual(payload['max_tokens'], 8192)
        self.assertNotIn('max_completion_tokens', payload)

    def test_actual_multiview_response_schema(self):
        views = [{'id': 'view_01', 'width': 1280, 'height': 960, 'source_image': 'input:view_01'}]
        schema = vision_multiview.response_schema(views, 'fixture_scene')

        def example(rule):
            if 'enum' in rule:
                return rule['enum'][0]
            if rule['type'] == 'object':
                return {key: example(child) for key, child in rule['properties'].items()}
            if rule['type'] == 'array':
                return [example(rule['items']) for _ in range(rule.get('minItems', 0))]
            if rule['type'] == 'string':
                return 'fixture'
            if rule['type'] == 'boolean':
                return False
            return rule.get('minimum', 0)

        expected = example(schema)
        self.queue(chat(json.dumps(expected)))
        result = api.call_model('Analyze scene', [self.image], schema, self.output, config=self.config)
        self.assertEqual(result, expected)
        self.assertEqual(self.server.requests[0]['body']['response_format']['json_schema']['schema'], schema)

    def test_string_pattern_and_minlength_enforced_locally(self):
        schema = {'type': 'object', 'properties': {'id': {'type': 'string', 'minLength': 2, 'maxLength': 10, 'pattern': r'^[A-Za-z0-9_-]+$'}},
                  'required': ['id'], 'additionalProperties': False}
        for identifier in ('a', '../escape', 'id with space'):
            self.queue(chat(json.dumps({'id': identifier})))
            with self.subTest(identifier=identifier), self.assertRaises(api.ModelAPIError) as exc:
                api.call_model('Analyze scene', [self.image], schema, self.output, config=self.config)
            self.assertEqual(exc.exception.code, 'schema_mismatch')

    def test_exact_fenced_json_accepted(self):
        self.queue(chat('```json\n' + json.dumps(RESULT) + '\n```'))
        self.assertEqual(self.call(), RESULT)

    def test_schema_rejects_extra_missing_wrong_range_bool_nan_duplicate(self):
        malformed = [json.dumps({**RESULT, 'extra': 'secret'}), '{"scene":"fixture"}',
                     '{"scene":"fixture","number":10}', '{"scene":"fixture","number":true}',
                     '{"scene":"fixture","number":NaN}',
                     '{"scene":"fixture","number":2,"number":3}']
        for content in malformed:
            with self.subTest(content=content):
                self.queue(chat(content))
                with self.assertRaises(api.ModelAPIError):
                    self.call()
                self.assertEqual(self.record()['status'], 'failed')

    def test_prose_code_and_partial_json_rejected(self):
        for content in ('Here is JSON: ' + json.dumps(RESULT), '```python\nprint(1)\n```', '{"scene":', '[]'):
            with self.subTest(content=content):
                self.queue(chat(content))
                with self.assertRaises(api.ModelAPIError) as exc:
                    self.call()
                self.assertEqual(exc.exception.code, 'invalid_json')

    def test_usage_and_finish_are_saved_when_json_is_rejected(self):
        self.queue(chat('{"invalid":true}'))
        with self.assertRaises(api.ModelAPIError):
            self.call()
        record = self.record()
        self.assertEqual(record['usage']['total_tokens'], 19)
        self.assertEqual(record['finish_reason'], 'stop')
        self.assertEqual(record['transport_attempts'][0]['usage'], record['usage'])
        self.assertFalse(self.output.with_name('analysis.raw.json').exists())

    def test_schema_mismatch_records_safe_first_known_field_path_and_usage(self):
        self.queue(chat('{"scene":"fixture","number":99}'))
        with self.assertRaises(api.ModelAPIError) as caught:
            self.call()
        self.assertEqual(caught.exception.code, 'schema_mismatch')
        record = self.record()
        self.assertEqual(record['schema_validation'],
                         {'path': ['number'], 'constraint': 'maximum'})
        self.assertEqual(record['usage']['total_tokens'], 19)
        self.assertEqual(record['transport_attempts'][0]['usage'], record['usage'])
        self.assertFalse(self.output.with_name('analysis.raw.json').exists())

    def test_schema_mismatch_never_records_unknown_malicious_field_name_or_value(self):
        canary = 'CREDENTIAL_CANARY_DO_NOT_PERSIST'
        self.queue(chat(json.dumps({**RESULT, canary: 'provider-text-' + canary})))
        with self.assertRaises(api.ModelAPIError):
            self.call()
        record_text = self.output.with_name('analysis.api.json').read_text()
        record = json.loads(record_text)
        self.assertEqual(record['schema_validation'],
                         {'path': [], 'constraint': 'additional_properties'})
        self.assertNotIn(canary, record_text)
        self.assertNotIn('provider-text-', record_text)
        self.assertFalse(self.output.with_name('analysis.raw.json').exists())

    def test_schema_validation_diagnostic_handles_canonical_additive_omissions(self):
        # Stored V2/V3 canonical data may omit these additive extension fields.
        schema = {'type': 'object', 'properties': {
            'schema_version': {'type': 'integer'},
            'base_scene': {'type': 'object', 'properties': {'name': {'type': 'string'}},
                           'required': ['name'], 'additionalProperties': False},
            'surface_overrides': {'type': 'object', 'properties': {}, 'required': [],
                                  'additionalProperties': False},
            'custom_objects': {'type': 'array', 'items': {'type': 'string'}},
            'base_material_overrides': {'type': 'array', 'items': {'type': 'string'}},
            'exterior_contexts': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['schema_version', 'base_scene', 'surface_overrides', 'custom_objects',
                         'base_material_overrides', 'exterior_contexts'],
            'additionalProperties': False}
        legacy = {'schema_version': 2, 'base_scene': {'name': 'fixture'},
                  'surface_overrides': {}, 'custom_objects': []}
        self.assertEqual(api._extract_json(json.dumps(legacy), schema), legacy)

    def test_union_diagnostic_uses_schema_discriminator_and_redacted_path(self):
        schema = {'type': 'object', 'properties': {'part': {'anyOf': [
            {'type': 'object', 'properties': {
                'type': {'type': 'string', 'enum': ['box']},
                'size': {'type': 'number'}},
             'required': ['type', 'size'], 'additionalProperties': False},
            {'type': 'object', 'properties': {
                'type': {'type': 'string', 'enum': ['sphere']},
                'radius': {'type': 'number'}},
             'required': ['type', 'radius'], 'additionalProperties': False}]}},
            'required': ['part'], 'additionalProperties': False}
        with self.assertRaises(api.ModelAPIError) as caught:
            api._extract_json('{"part":{"type":"box","size":"UNTRUSTED_VALUE"}}', schema)
        self.assertEqual(caught.exception.code, 'schema_mismatch')
        self.assertEqual(caught.exception.schema_validation,
                         {'path': ['part', 'size'], 'constraint': 'type'})
        self.assertNotIn('UNTRUSTED_VALUE', str(caught.exception))

    def test_chat_refusal_and_truncation_rejected(self):
        for response, code in [(chat(refusal='declined'), 'refusal'), (chat(finish='length'), 'output_truncated'),
                               (chat(finish='content_filter'), 'refusal'), (chat(finish='tool_calls'), 'incomplete')]:
            self.queue(response)
            with self.assertRaises(api.ModelAPIError) as exc:
                self.call()
            self.assertEqual(exc.exception.code, code)

    def test_responses_refusal_and_truncation_rejected(self):
        for response, code in [({'status': 'incomplete', 'output': [], 'incomplete_details': {'reason': 'max_output_tokens'}}, 'output_truncated'),
                               ({'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'no'}]}]}, 'refusal')]:
            self.queue(response)
            with self.assertRaises(api.ModelAPIError) as exc:
                self.call(provider='openai_responses')
            self.assertEqual(exc.exception.code, code)

    def test_anthropic_truncation_rejected(self):
        self.queue({'stop_reason': 'max_tokens', 'content': [{'type': 'text', 'text': json.dumps(RESULT)}]})
        with self.assertRaises(api.ModelAPIError) as exc:
            self.call(provider='anthropic')
        self.assertEqual(exc.exception.code, 'output_truncated')

    def test_429_and_500_retry_then_success(self):
        self.queue({'error': 'secret echo'}, status=429, **{'Retry-After': '0'})
        self.queue({'error': 'temporary'}, status=503)
        self.queue()
        with patch.object(api.time, 'sleep') as sleep:
            self.assertEqual(self.call(max_retries=2), RESULT)
        self.assertEqual(len(self.server.requests), 3)
        self.assertEqual(self.record()['attempts'], 3)
        self.assertEqual(sleep.call_count, 2)

    def test_400_does_not_retry_and_sanitizes_provider_error(self):
        secret = 'do-not-record-this-credential'
        self.queue({'error': secret, 'images': 'data:image/png;base64,secret'}, status=400)
        with patch.dict(os.environ, {'FIXTURE_MODEL_KEY': secret}):
            with self.assertRaises(api.ModelAPIError) as exc:
                self.call(api_key_env='FIXTURE_MODEL_KEY', max_retries=4)
        self.assertNotIn(secret, str(exc.exception))
        self.assertNotIn(secret, self.output.with_name('analysis.api.json').read_text())
        self.assertNotIn('data:image', self.output.with_name('analysis.api.json').read_text())
        self.assertEqual(len(self.server.requests), 1)
        self.assertFalse(self.output.with_name('analysis.raw.json').exists())

    def test_redirect_never_followed(self):
        self.queue(b'', status=307, Location=f'http://127.0.0.1:{self.server.server_port}/leak')
        with self.assertRaises(api.ModelAPIError) as exc:
            self.call()
        self.assertEqual(exc.exception.code, 'redirect_blocked')
        self.assertEqual(len(self.server.requests), 1)

    def test_retry_obeys_total_timeout(self):
        self.queue({}, status=429, **{'Retry-After': '30'})
        started = time.monotonic()
        with self.assertRaises(api.ModelAPIError) as exc:
            self.call(max_retries=2, timeout_seconds=.2)
        self.assertEqual(exc.exception.code, 'timeout')
        self.assertLess(time.monotonic() - started, 1)
        # Explicit rejection plus retry backoff expiry is a known outcome.
        self.assertNotIn('automatic_replay_blocked', self.record()['transport_attempts'][0])
        api.guard_ambiguous_model_resume([self.output.with_name('analysis.api.json')])

    def test_long_retry_after_cannot_be_shortened_to_fit_request_deadline(self):
        self.queue({}, status=429, **{'Retry-After': '120'})
        with patch.object(api.time, 'sleep') as sleep:
            with self.assertRaises(api.ModelAPIError) as exc:
                self.call(max_retries=2, timeout_seconds=45)
        self.assertEqual(exc.exception.code, 'timeout')
        sleep.assert_not_called()
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.record()['transport_attempts'][0]['retry_after_seconds'], 120)
        api.guard_ambiguous_model_resume([self.output.with_name('analysis.api.json')])

    def test_retry_after_retains_numeric_and_http_date_minimum_wait(self):
        for header, expected in (('120', 120), ('172800', 172800),
                                 ('-1', 0), ('nan', 0), ('inf', 0), ('invalid', 0), (None, 0)):
            with self.subTest(header=header):
                self.assertEqual(api._retry_after(header), expected)
        header = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120), usegmt=True)
        self.assertGreater(api._retry_after(header), 118)
        self.assertLessEqual(api._retry_after(header), 120)

    def test_slow_response_obeys_total_timeout(self):
        def slow(_handler):
            time.sleep(.3)
            return chat()
        self.queue(slow)
        started = time.monotonic()
        with self.assertRaises(api.ModelAPIError) as exc:
            self.call(timeout_seconds=.1, max_retries=2)
        self.assertEqual(exc.exception.code, 'timeout')
        self.assertLess(time.monotonic() - started, .8)

    def test_response_size_is_bounded(self):
        self.queue(chat())
        with patch.object(api, 'MAX_RESPONSE_BYTES', 50):
            with self.assertRaises(api.ModelAPIError) as exc:
                self.call()
        self.assertEqual(exc.exception.code, 'response_too_large')

    def test_invalid_envelope_sanitized(self):
        self.queue(b'gateway says SECRET')
        with self.assertRaises(api.ModelAPIError) as exc:
            self.call()
        self.assertNotIn('SECRET', str(exc.exception))
        self.assertEqual(exc.exception.code, 'invalid_envelope')

    def test_missing_explicit_key_never_falls_back(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'must-not-use'}, clear=True):
            with self.assertRaises(api.ModelAPIError) as exc:
                api.preflight({**self.config, 'api_key_env': 'OTHER_PROVIDER_KEY'})
        self.assertEqual(exc.exception.code, 'missing_credential')
        self.assertEqual(self.server.requests, [])

    def test_missing_credential_preserves_prior_completed_audit_and_usage(self):
        self.queue()
        with patch.dict(os.environ, {'FIXTURE_MODEL_KEY': 'fixture-secret'}, clear=True):
            self.assertEqual(self.call(api_key_env='FIXTURE_MODEL_KEY'), RESULT)
        api_path = self.output.with_name('analysis.api.json')
        raw_path = self.output.with_name('analysis.raw.json')
        saved = {path: path.read_bytes() for path in (api_path, raw_path)}
        self.assertEqual(self.record()['usage']['total_tokens'], 19)
        with patch.dict(os.environ, {}, clear=True), patch.object(api, '_request') as request:
            with self.assertRaises(api.ModelAPIError) as exc:
                self.call(api_key_env='FIXTURE_MODEL_KEY')
        self.assertEqual(exc.exception.code, 'missing_credential')
        request.assert_not_called()
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual({path: path.read_bytes() for path in saved}, saved)
        self.assertEqual(self.record()['status'], 'completed')
        self.assertEqual(self.record()['usage']['total_tokens'], 19)

    def test_public_config_and_preflight(self):
        config = {**self.config, 'api_key_env': 'FIXTURE_MODEL_KEY'}
        with patch.dict(os.environ, {}, clear=True):
            public = api.public_config(config)
            self.assertEqual(public, api.normalize_config(config))
            with self.assertRaises(api.ModelAPIError):
                api.preflight(config)
        with patch.dict(os.environ, {'FIXTURE_MODEL_KEY': 'fixture-secret'}, clear=True):
            self.assertEqual(api.preflight(config), public)
        self.assertEqual(self.server.requests, [])

    def test_third_party_key_policy(self):
        for host in ('api.apikey.fun', 'api.apikey.fun.'):
            with self.assertRaises(ValueError):
                api.normalize_config({**self.config, 'base_url': f'https://{host}/v1', 'api_key_env': 'OPENAI_API_KEY'})
        good = api.normalize_config({**self.config, 'base_url': 'https://api.apikey.fun/v1', 'api_key_env': 'APIKEY_FUN_API_KEY'})
        self.assertEqual(good['api_key_env'], 'APIKEY_FUN_API_KEY')

    def test_malformed_config(self):
        cases = [None, [], {}, {**self.config, 'provider': 'unknown'}, {**self.config, 'provider': []}, {**self.config, 'model': ''},
                 {**self.config, 'api_key': 'secret'}, {**self.config, 'max_retries': True},
                 {**self.config, 'max_output_tokens': 0}, {**self.config, 'timeout_seconds': float('nan')},
                 {**self.config, 'temperature': 3}, {**self.config, 'response_format': 'auto'}, {**self.config, 'response_format': []},
                 {**self.config, 'auth_scheme': 'basic'}, {**self.config, 'token_parameter': 'tokens'},
                 {**self.config, 'provider': 'anthropic', 'token_parameter': 'max_tokens'},
                 {**self.config, 'provider': 'anthropic', 'response_format': 'json_schema'},
                 {**self.config, 'provider': 'anthropic', 'reasoning_effort': 'high'},
                 {**self.config, 'stream': 1}, {**self.config, 'stream': None},
                 {**self.config, 'provider': 'openai_responses', 'stream': True},
                 {**self.config, 'provider': 'anthropic', 'stream': False},
                 {**self.config, 'api_key_env': 'KEY=secret'}]
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                api.normalize_config(config)

    def test_unsafe_urls(self):
        for url in ('http://example.com/v1', 'https://user:pass@example.com/v1', 'https://example.com/v1?key=secret',
                    'https://example.com/v1#secret', 'file:///tmp/api', 'https://example.com:99999',
                    'https://example.com/../api', 'https://example.com/\napi'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                api.normalize_config({**self.config, 'base_url': url, 'api_key_env': 'KEY'})

    def test_proxy_config(self):
        https = {**self.config, 'provider': 'openai_responses', 'base_url': 'https://api.openai.com/v1', 'api_key_env': 'OPENAI_API_KEY'}
        self.assertNotIn('proxy', api.normalize_config(https))
        for value, expected in (('http://127.0.0.1:7897', 'http://127.0.0.1:7897'), ('http://localhost:8080/', 'http://localhost:8080'),
                                ('http://[::1]:7897', 'http://[::1]:7897')):
            with self.subTest(proxy=value):
                self.assertEqual(api.normalize_config({**https, 'proxy': value})['proxy'], expected)
        for value in (None, '', 'https://127.0.0.1:7897', 'socks5://127.0.0.1:7897', 'http://127.0.0.1',
                      'http://user:pass@127.0.0.1:7897', 'http://10.0.0.5:7897', 'http://proxy.example.com:7897',
                      'http://127.0.0.1:7897/path', 'http://127.0.0.1:7897?x=1', 'http://127.0.0.1:99999', 7897):
            with self.subTest(proxy=value), self.assertRaises(ValueError):
                api.normalize_config({**https, 'proxy': value})
        with self.assertRaisesRegex(ValueError, 'HTTPS endpoints only'):
            api.normalize_config({**self.config, 'proxy': 'http://127.0.0.1:7897'})

    def test_proxy_is_a_connect_tunnel_with_end_to_end_tls(self):
        config = api.normalize_config({**self.config, 'provider': 'openai_responses', 'base_url': 'https://api.openai.com/v1',
                                       'api_key_env': 'OPENAI_API_KEY', 'proxy': 'http://127.0.0.1:7897'})
        seen = {}

        class Response:
            status = 200
            fp = None
            def __init__(self):
                self.body = io.BytesIO(b'{"ok": true}')
            def getheader(self, name):
                return None
            def read1(self, size):
                return self.body.read(size)
            def close(self):
                pass

        class Connection:
            def __init__(self, host, port, timeout=None):
                seen['connect_to'] = (host, port)
                self.sock = type('Sock', (), {'settimeout': lambda self, value: None})()
            def set_tunnel(self, host, port=None, headers=None):
                seen['tunnel'] = (host, port, headers)
            def connect(self):
                pass
            def request(self, method, path, body=None, headers=None):
                seen['request'] = (method, path, headers)
            def getresponse(self):
                return Response()
            def close(self):
                pass

        with patch.object(api.http.client, 'HTTPSConnection', Connection):
            status, _, raw = api._request(config, b'{}', 'fixture-secret', time.monotonic() + 30)
        self.assertEqual((status, raw), (200, b'{"ok": true}'))
        self.assertEqual(seen['connect_to'], ('127.0.0.1', 7897))
        self.assertEqual(seen['tunnel'], ('api.openai.com', 443, None))
        method, path, headers = seen['request']
        self.assertEqual((method, path, headers['Authorization']), ('POST', '/v1/responses', 'Bearer fixture-secret'))

    def test_unsupported_schema_fails_before_network(self):
        schema = {**SCHEMA, 'oneOf': []}
        with self.assertRaises(ValueError):
            api.call_model('prompt', [self.image], schema, self.output, config=self.config)
        self.assertEqual(self.server.requests, [])

    def test_unsupported_image_fails_before_network(self):
        image = self.folder / 'unsupported.gif'
        image.write_bytes(b'GIF89a')
        with self.assertRaises(ValueError):
            api.call_model('prompt', [image], SCHEMA, self.output, config=self.config)
        self.assertEqual(self.server.requests, [])


if __name__ == '__main__':
    unittest.main()
