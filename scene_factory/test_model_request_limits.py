"""Offline request size guards and diagnostic preservation; no paid API calls."""
import base64
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import model_api as api
from test_model_api import PNG, RESULT, SCHEMA, chat


class RequestLimitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.image = self.folder / 'private-reference.png'
        self.image.write_bytes(PNG)
        self.output = self.folder / 'analysis.json'
        self.config = {'provider': 'openai_chat', 'model': 'fixture',
                       'base_url': 'http://127.0.0.1:1/v1', 'api_key_env': '',
                       'max_retries': 0, 'max_output_tokens': 512}

    def record(self):
        return json.loads(self.output.with_name('analysis.api.json').read_text())

    def call(self, prompt='场景参考', **config):
        return api.call_model(prompt, [self.image], SCHEMA, self.output,
                              config={**self.config, **config})

    @staticmethod
    def response(envelope=None):
        return 200, 0, json.dumps(chat() if envelope is None else envelope).encode()

    def test_optional_default_preserves_public_config_and_validates_strictly(self):
        original = api.public_config(self.config)
        self.assertNotIn('max_prompt_characters', original)
        self.assertEqual(api.public_config(original), original)
        for limit in (0, 1, 2000000):
            with self.subTest(limit=limit):
                normalized = api.public_config({**self.config, 'max_prompt_characters': limit})
                self.assertEqual(normalized['max_prompt_characters'], limit)
                self.assertEqual(api.normalize_config(normalized), normalized)
        for limit in (-1, 2000001, True, 4.0, '100', None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                api.normalize_config({**self.config, 'max_prompt_characters': limit})

    def test_oversize_guard_precedes_credentials_image_encoding_and_network(self):
        # A missing local file and missing key would fail if either were read.
        prompt = 'PRIVATE_PROMPT_房间'
        with patch.object(api, '_image_payloads') as encode, \
                patch.object(api, '_credential') as credential, \
                patch.object(api, '_request') as request:
            with self.assertRaises(api.ModelAPIError) as failure:
                api.call_model(prompt, [self.folder / 'missing-private-photo.png'], SCHEMA,
                               self.output, config={**self.config, 'api_key_env': 'MISSING_TEST_KEY',
                                                    'max_prompt_characters': 5, 'max_retries': 4})
        self.assertEqual(failure.exception.code, 'input_budget_exceeded')
        for fn in (encode, credential, request):
            fn.assert_not_called()
        record = self.record()
        self.assertEqual(record['failure_code'], 'input_budget_exceeded')
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['attempts'], 0)
        self.assertEqual(record['transport_attempts'], [])
        self.assertEqual(record['image_sha256'], [])
        self.assertEqual(record['request_metrics']['prompt_characters'], len(prompt))
        self.assertEqual(record['request_metrics']['prompt_utf8_bytes'], len(prompt.encode()))
        self.assertEqual(record['request_metrics']['image_count'], 1)
        saved = json.dumps(record) + str(failure.exception)
        for private in ('PRIVATE_PROMPT_', 'missing-private-photo.png', str(self.folder)):
            self.assertNotIn(private, saved)
        self.assertFalse(self.output.with_name('analysis.raw.json').exists())
        self.assertNotIn('usage', record)  # No invented zero-token billing claim.

    def test_unicode_character_limit_is_inclusive_and_not_a_byte_limit(self):
        prompt = '房间🙂'
        schema = {**SCHEMA, 'description': '多视图场景'}
        with patch.object(api, '_request', return_value=self.response()) as request:
            result = api.call_model(prompt, [self.image], schema, self.output,
                                    config={**self.config, 'max_prompt_characters': len(prompt)})
        self.assertEqual(result, RESULT)
        metrics = self.record()['request_metrics']
        self.assertEqual(metrics, {
            'prompt_characters': 3, 'prompt_utf8_bytes': 10,
            'schema_utf8_bytes': len(json.dumps(schema, ensure_ascii=False, separators=(',', ':')).encode()),
            'image_count': 1, 'max_output_tokens': 512,
            'images': [{'media_type': 'image/png', 'source_bytes': len(PNG),
                        'base64_bytes': len(base64.b64encode(PNG)),
                        'width_px': 1, 'height_px': 1}],
            'image_source_bytes_total': len(PNG),
            'image_base64_bytes_total': len(base64.b64encode(PNG)),
            'serialized_request_body_utf8_bytes': len(request.call_args.args[1])})
        self.assertIn('Byte fields describe local payload sizes only',
                      self.record()['request_metrics_semantics'])
        self.assertEqual(self.record()['transport_attempts'][0]['request_metrics'], metrics)
        # This is a local guard, never an unsupported provider API parameter.
        self.assertNotIn('max_prompt_characters', json.loads(request.call_args.args[1]))

    def test_zero_and_omitted_limit_leave_requests_enabled(self):
        for override in ({}, {'max_prompt_characters': 0}):
            with self.subTest(override=override), \
                    patch.object(api, '_request', return_value=self.response()) as request:
                self.assertEqual(self.call('x' * 5000, **override), RESULT)
                self.assertEqual(request.call_count, 1)

    def test_retry_metrics_are_saved_on_each_attempt_and_usage_is_unchanged(self):
        with patch.object(api, '_request', side_effect=[(429, 0, None), (503, 0, None),
                                                       self.response()]) as request, \
                patch.object(api.time, 'sleep'):
            self.assertEqual(self.call(max_retries=2), RESULT)
        record = self.record()
        self.assertEqual(request.call_count, 3)
        self.assertEqual(record['attempts'], 3)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['usage'], {'prompt_tokens': 12, 'completion_tokens': 7, 'total_tokens': 19})
        for attempt in record['transport_attempts']:
            self.assertEqual(attempt['request_metrics'], record['request_metrics'])
        self.assertNotIn('usage', record['transport_attempts'][0])
        self.assertEqual(record['transport_attempts'][-1]['usage'], record['usage'])

    def test_nonretryable_http_failure_saves_sizes_without_payload_or_credentials(self):
        with patch.dict(os.environ, {'FIXTURE_MODEL_KEY': 'credential-not-for-logs'}), \
                patch.object(api, '_request', return_value=(402, 0, None)) as request:
            with self.assertRaises(api.ModelAPIError):
                self.call('PRIVATE_PROMPT', api_key_env='FIXTURE_MODEL_KEY', max_retries=4)
        self.assertEqual(request.call_count, 1)
        record = self.record()
        self.assertEqual(record['failure_code'], 'http_error')
        self.assertEqual(record['request_metrics']['prompt_characters'], 14)
        self.assertNotIn('usage', record)
        self.assertEqual(record['transport_attempts'][0]['request_metrics'], record['request_metrics'])
        for private in ('credential-not-for-logs', 'PRIVATE_PROMPT', self.image.name, 'data:image'):
            self.assertNotIn(private, json.dumps(record))

    def test_invalid_result_keeps_reported_usage_with_request_sizes(self):
        with patch.object(api, '_request', return_value=self.response(chat('{"wrong":1}'))):
            with self.assertRaises(api.ModelAPIError) as failure:
                self.call()
        self.assertEqual(failure.exception.code, 'schema_mismatch')
        record = self.record()
        self.assertEqual(record['usage']['total_tokens'], 19)
        self.assertEqual(record['request_metrics']['max_output_tokens'], 512)
        self.assertEqual(record['output_text_utf8_bytes'], len('{"wrong":1}'.encode('utf-8')))
        self.assertEqual(record['transport_attempts'][0]['request_metrics'], record['request_metrics'])
        self.assertFalse(self.output.with_name('analysis.raw.json').exists())

    def test_missing_credential_is_also_recorded_without_network(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(api, '_request') as request:
            with self.assertRaises(api.ModelAPIError) as failure:
                self.call(api_key_env='MISSING_TEST_KEY')
        self.assertEqual(failure.exception.code, 'missing_credential')
        request.assert_not_called()
        self.assertEqual(self.record()['failure_code'], 'missing_credential')
        self.assertEqual(self.record()['attempts'], 0)
        self.assertEqual(self.record()['request_metrics']['image_count'], 1)


if __name__ == '__main__':
    unittest.main()
