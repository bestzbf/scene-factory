"""OpenAI-compatible SSE transport tests, entirely offline / loopback."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import model_api as api
from test_model_api import PNG, RESULT, SCHEMA


def event(value, eol=b'\n'):
    return b'data: ' + json.dumps(value, ensure_ascii=False).encode() + eol + eol


def chunk(content=None, finish=None, **delta):
    if content is not None:
        delta['content'] = content
    return {'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}


def stream(content=None, finish='stop', done=True):
    content = json.dumps(RESULT) if content is None else content
    return (event(chunk(role='assistant'))
            + event(chunk(reasoning_content='PRIVATE_REASONING_MUST_NEVER_BE_SAVED'))
            + event(chunk(content=content[:5])) + event(chunk(content=content[5:]))
            + event(chunk(finish=finish))
            + event({'choices': [], 'usage': {'prompt_tokens': 12, 'completion_tokens': 7,
                                              'total_tokens': 19, 'unknown': 'SECRET'}})
            + (b'data: [DONE]\n\n' if done else b''))


class StreamHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.requests.append({'body': json.loads(self.rfile.read(int(self.headers['Content-Length']))),
                                     'headers': dict(self.headers)})
        response = self.server.response
        self.send_response(200)
        self.send_header('Content-Type', response.get('mime', 'text/event-stream; charset=utf-8'))
        self.end_headers()
        for data in response['chunks']:
            time.sleep(response.get('delay', 0))
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, *args):
        pass


class StreamingAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), StreamHandler)
        cls.server.daemon_threads = True
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
        self.image = self.folder/'one.png'
        self.image.write_bytes(PNG)
        self.output = self.folder/'analysis.json'
        self.config = {'provider': 'openai_chat', 'model': 'fixture-stream',
                       'base_url': f'http://127.0.0.1:{self.server.server_port}/v1',
                       'api_key_env': '', 'max_retries': 0, 'stream': True}
        self.server.requests = []
        self.server.response = {'chunks': [stream()]}

    def call(self, **overrides):
        return api.call_model('Describe the input photo.', [self.image], SCHEMA, self.output,
                              config={**self.config, **overrides})

    def record(self):
        return json.loads(self.output.with_name('analysis.api.json').read_text())

    def test_streamed_json_and_metadata_no_reasoning_persistence(self):
        self.assertEqual(self.call(), RESULT)
        request = self.server.requests[0]
        self.assertIs(request['body']['stream'], True)
        self.assertEqual(request['headers']['Accept'], 'text/event-stream')
        record = self.record()
        self.assertEqual(record['usage']['total_tokens'], 19)
        self.assertEqual(record['finish_reason'], 'stop')
        stats = record['transport_attempts'][0]
        self.assertTrue(stats['done'])
        self.assertGreater(stats['reasoning_characters'], 0)
        self.assertEqual(stats['content_characters'], len(json.dumps(RESULT)))
        saved = ''.join(p.read_text() for p in self.folder.glob('*.json'))
        self.assertNotIn('PRIVATE_REASONING', saved)
        self.assertNotIn('SECRET', saved)
        self.assertEqual({p.name for p in self.folder.glob('*')},
                         {'one.png', 'analysis.api.json', 'analysis.raw.json'})

    def test_sse_crlf_bom_comments_multiline_and_utf8_split_bytes(self):
        content = json.dumps({'scene': '房间', 'number': 2}, ensure_ascii=False)
        envelope = json.dumps(chunk(content), ensure_ascii=False).encode()
        # A newline between JSON tokens is legal, and must be joined by SSE.
        boundary = envelope.index(b'[{') + 1
        data = (b'\xef\xbb\xbf: heartbeat\r\n\r\n'
                + b'event: message\r\nid: ignored\r\nretry: 100\r\n'
                + b'data: ' + envelope[:boundary] + b'\r\n'
                + b'data: ' + envelope[boundary:] + b'\r\n\r\n'
                + event(chunk(finish='stop'), b'\r') + b'data: [DONE]\r\n\r\n')
        stats = {}
        result = api._read_chat_stream((bytes([b]) for b in data), stats)
        self.assertEqual(json.loads(result['choices'][0]['message']['content']), {'scene': '房间', 'number': 2})
        self.assertTrue(stats['done'])

    def test_missing_done_does_not_accept_valid_json_or_discard_usage(self):
        self.server.response = {'chunks': [stream(done=False)]}
        with self.assertRaises(api.ModelAPIError) as exc:
            self.call()
        self.assertEqual(exc.exception.code, 'incomplete')
        self.assertFalse(self.output.with_name('analysis.raw.json').exists())
        self.assertEqual(self.record()['usage']['total_tokens'], 19)
        self.assertEqual(self.record()['finish_reason'], 'stop')
        self.assertEqual(self.record()['http_status'], 200)

    def test_unterminated_done_event_is_incomplete(self):
        self.server.response = {'chunks': [stream(done=False) + b'data: [DONE]\n']}
        with self.assertRaises(api.ModelAPIError) as exc:
            self.call()
        self.assertEqual(exc.exception.code, 'incomplete')

    def test_length_missing_finish_and_refusal_rejected(self):
        for finish, code in (('length', 'output_truncated'), (None, 'incomplete'), ('content_filter', 'refusal')):
            with self.subTest(finish=finish):
                self.server.response = {'chunks': [stream(finish=finish)]}
                with self.assertRaises(api.ModelAPIError) as exc:
                    self.call()
                self.assertEqual(exc.exception.code, code)
                self.assertEqual(self.record()['usage']['total_tokens'], 19)
        self.server.response = {'chunks': [event(chunk(refusal='private provider refusal')) + stream()]}
        with self.assertRaises(api.ModelAPIError) as exc:
            self.call()
        self.assertEqual(exc.exception.code, 'refusal')

    def test_schema_and_json_failures_preserve_usage_and_finish(self):
        for content, code in (('{"wrong":1}', 'schema_mismatch'), ('{unfinished', 'invalid_json')):
            with self.subTest(code=code):
                self.server.response = {'chunks': [stream(content)]}
                with self.assertRaises(api.ModelAPIError) as exc:
                    self.call()
                self.assertEqual(exc.exception.code, code)
                self.assertEqual(self.record()['usage']['total_tokens'], 19)
                self.assertEqual(self.record()['finish_reason'], 'stop')

    def test_stream_error_and_malformed_event_are_sanitized(self):
        for body in (b'event: error\ndata: CREDENTIAL_ECHO\n\n',
                     event({'error': 'CREDENTIAL_ECHO'}),
                     b'data: CREDENTIAL_ECHO\n\n',
                     b'data: {"choices":[],"choices":"CREDENTIAL_ECHO"}\n\n'):
            self.server.response = {'chunks': [body]}
            with self.assertRaises(api.ModelAPIError) as exc:
                self.call()
            self.assertEqual(exc.exception.code, 'invalid_envelope')
            self.assertNotIn('CREDENTIAL_ECHO', str(exc.exception))
            self.assertNotIn('CREDENTIAL_ECHO', json.dumps(self.record()))

    def test_wrong_mime_fails_without_protocol_fallback(self):
        self.server.response = {'mime': 'application/json', 'chunks': [stream()]}
        with self.assertRaises(api.ModelAPIError) as exc:
            self.call(max_retries=2)
        self.assertEqual(exc.exception.code, 'invalid_envelope')
        self.assertEqual(len(self.server.requests), 1)

    def test_stream_obeys_deadline_while_bytes_keep_arriving(self):
        self.server.response = {'chunks': [b':'] * 100, 'delay': .01}
        started = time.monotonic()
        with self.assertRaises(api.ModelAPIError) as exc:
            self.call(timeout_seconds=.1)
        self.assertEqual(exc.exception.code, 'timeout')
        self.assertLess(time.monotonic() - started, .5)
        self.assertEqual(self.record()['http_status'], 200)

    def test_stream_whole_response_and_individual_event_bounds(self):
        for field, bound in (('MAX_RESPONSE_BYTES', 100), ('MAX_SSE_EVENT_BYTES', 40)):
            with self.subTest(field=field), patch.object(api, field, bound):
                with self.assertRaises(api.ModelAPIError) as exc:
                    self.call()
                self.assertEqual(exc.exception.code, 'response_too_large')

    def test_tool_calls_and_multiple_completions_rejected(self):
        cases = [(chunk(tool_calls=[{'id': 'unexpected'}]), 'incomplete'),
                 ({'choices': [{'index': 1, 'delta': {}}]}, 'invalid_envelope'),
                 ({'choices': [{}, {}]}, 'invalid_envelope')]
        for body, code in cases:
            with self.subTest(code=code):
                self.server.response = {'chunks': [event(body)]}
                with self.assertRaises(api.ModelAPIError) as exc:
                    self.call()
                self.assertEqual(exc.exception.code, code)

    def test_unknown_finish_reason_not_copied_to_record(self):
        self.server.response = {'chunks': [stream(finish='SECRET_PROVIDER_VALUE')]}
        with self.assertRaises(api.ModelAPIError):
            self.call()
        self.assertEqual(self.record()['finish_reason'], 'unknown')
        self.assertNotIn('SECRET_PROVIDER_VALUE', json.dumps(self.record()))


if __name__ == '__main__':
    unittest.main()
