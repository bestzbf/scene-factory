"""Offline POST replay safety, including real loopback connection failures."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import model_api as api
from test_model_api import PNG, RESULT, SCHEMA, chat
from test_model_api_streaming import chunk, event, stream


class ReplayHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        # Discard request data; the fixture records only completed POST counts.
        self.rfile.read(int(self.headers['Content-Length']))
        self.server.posts += 1
        mode = self.server.mode if self.server.posts == 1 else 'success'
        if mode == 'close_before_headers':
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        if mode == 'timeout_before_headers':
            time.sleep(.3)
            self.close_connection = True
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if self.server.streaming else 'application/json')
        self.end_headers()
        if mode in ('reset_after_content', 'reset_after_json'):
            data = (event(chunk(content=json.dumps(RESULT)))
                    + event(chunk(finish='stop'))
                    + event({'choices': [], 'usage': {'prompt_tokens': 12,
                        'completion_tokens': 7, 'total_tokens': 19}}))
            if mode == 'reset_after_json':
                data = json.dumps(chat()).encode()
            self.wfile.write(data)
            self.wfile.flush()
            # Deliver the complete events before aborting the TCP connection.
            time.sleep(.03)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            self.connection.close()
            self.close_connection = True
            return
        data = stream(done=mode != 'missing_done') if self.server.streaming else json.dumps(chat()).encode()
        self.wfile.write(data)
        self.wfile.flush()


class ReplaySafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / 'private-image.png'
        self.image.write_bytes(PNG)
        self.output = self.root / 'analysis.json'
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), ReplayHandler)
        self.server.daemon_threads = True
        self.server.mode, self.server.streaming, self.server.posts = 'success', True, 0
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.config = {'provider': 'openai_chat', 'model': 'fixture',
            'base_url': f'http://127.0.0.1:{self.server.server_port}/v1',
            'api_key_env': 'REPLAY_TEST_KEY', 'max_retries': 3, 'stream': True}

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def call(self, **overrides):
        with patch.dict(os.environ, {'REPLAY_TEST_KEY': 'PRIVATE_CREDENTIAL_MUST_NOT_BE_SAVED'}):
            return api.call_model('PRIVATE_PROMPT_MUST_NOT_BE_SAVED', [self.image], SCHEMA,
                                  self.output, config={**self.config, **overrides})

    def record(self):
        return json.loads(self.output.with_name('analysis.api.json').read_text())

    def assert_safe_failure(self, code):
        record = self.record()
        self.assertEqual(record['failure_code'], code)
        self.assertEqual(record['attempts'], 1)
        self.assertEqual(self.server.posts, 1)
        self.assertEqual(record['status'], 'failed')
        self.assertFalse(self.output.with_name('analysis.raw.json').exists())
        for secret in ('PRIVATE_CREDENTIAL', 'PRIVATE_PROMPT', self.image.name, 'data:image'):
            self.assertNotIn(secret, json.dumps(record))
        return record['transport_attempts'][0]

    def test_tcp_reset_after_success_content_and_usage_does_not_replay(self):
        self.server.mode = 'reset_after_content'
        with self.assertRaises(api.ModelAPIError) as failure, patch.object(api.time, 'sleep', wraps=time.sleep) as sleep:
            self.call()
        self.assertEqual(failure.exception.code, 'response_interrupted')
        stats = self.assert_safe_failure('response_interrupted')
        self.assertEqual(stats['phase'], 'receiving_response')
        self.assertTrue(stats['request_started'])
        self.assertTrue(stats['request_sent'])
        self.assertTrue(stats['automatic_replay_blocked'])
        self.assertEqual(stats['http_status'], 200)
        self.assertGreater(stats['response_bytes'], 0)
        self.assertEqual(stats['content_characters'], len(json.dumps(RESULT)))
        self.assertEqual(stats['finish_reason'], 'stop')
        self.assertFalse(stats['done'])
        self.assertEqual(stats['usage']['total_tokens'], 19)
        self.assertEqual(self.record()['usage']['total_tokens'], 19)
        # The server's short delivery delay is allowed, retry backoff is not.
        self.assertTrue(all(call.args[0] == .03 for call in sleep.call_args_list))

    def test_complete_post_then_close_before_headers_does_not_replay(self):
        self.server.mode = 'close_before_headers'
        with self.assertRaises(api.ModelAPIError) as failure:
            self.call()
        self.assertEqual(failure.exception.code, 'request_outcome_unknown')
        stats = self.assert_safe_failure('request_outcome_unknown')
        self.assertEqual(stats['phase'], 'waiting_response')
        self.assertTrue(stats['request_sent'])
        self.assertTrue(stats['automatic_replay_blocked'])
        self.assertNotIn('http_status', stats)
        self.assertNotIn('usage', self.record())

    def test_nonstream_response_reset_does_not_replay_or_accept_json(self):
        self.server.mode, self.server.streaming = 'reset_after_json', False
        with self.assertRaises(api.ModelAPIError) as failure:
            self.call(stream=False)
        self.assertEqual(failure.exception.code, 'response_interrupted')
        stats = self.assert_safe_failure('response_interrupted')
        self.assertEqual(stats['phase'], 'receiving_response')
        self.assertEqual(stats['http_status'], 200)
        self.assertGreater(stats['response_bytes'], 0)
        self.assertTrue(stats['automatic_replay_blocked'])
        # An unvalidated envelope is not a source of known billed usage.
        self.assertNotIn('usage', self.record())

    def test_stream_missing_completion_marker_is_not_accepted_or_replayed(self):
        self.server.mode = 'missing_done'
        with self.assertRaises(api.ModelAPIError) as failure:
            self.call()
        self.assertEqual(failure.exception.code, 'incomplete')
        stats = self.assert_safe_failure('incomplete')
        self.assertFalse(stats['done'])
        self.assertEqual(stats['finish_reason'], 'stop')
        self.assertEqual(self.record()['usage']['total_tokens'], 19)
        self.assertTrue(stats['completion_marker_missing'])
        self.assertTrue(stats['automatic_replay_blocked'])
        before = self.output.with_name('analysis.api.json').read_bytes()
        with patch.object(api, '_request', side_effect=AssertionError('No resumed POST')) as request, \
                self.assertRaises(api.ModelAPIError) as resumed:
            self.call()
        self.assertEqual(resumed.exception.code, 'ambiguous_previous_request')
        request.assert_not_called()
        self.assertEqual(self.server.posts, 1)
        self.assertEqual(self.output.with_name('analysis.api.json').read_bytes(), before)

    def test_post_timeout_stays_bounded_and_does_not_replay(self):
        self.server.mode = 'timeout_before_headers'
        started = time.monotonic()
        with self.assertRaises(api.ModelAPIError) as failure:
            self.call(timeout_seconds=.1)
        self.assertEqual(failure.exception.code, 'timeout')
        self.assertLess(time.monotonic() - started, .5)
        stats = self.assert_safe_failure('timeout')
        self.assertEqual(stats['phase'], 'waiting_response')
        self.assertTrue(stats['automatic_replay_blocked'])

    def test_real_connect_refusal_before_post_can_retry_and_recover(self):
        # Binding without listen gives a real, exclusively reserved refused port.
        with socket.socket() as refused:
            refused.bind(('127.0.0.1', 0))
            port = refused.getsockname()[1]
            real_connect = api.http.client.HTTPConnection.connect
            calls = []
            def connect(connection):
                calls.append(None)
                if len(calls) == 1:
                    with socket.create_connection(('127.0.0.1', port), timeout=connection.timeout):
                        self.fail('Reserved non-listening port unexpectedly accepted a connection')
                return real_connect(connection)
            with patch.object(api.http.client.HTTPConnection, 'connect', connect), patch.object(api.time, 'sleep'):
                self.assertEqual(self.call(), RESULT)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.server.posts, 1)
        record = self.record()
        self.assertEqual(record['attempts'], 2)
        first, second = record['transport_attempts']
        self.assertEqual(first['phase'], 'connecting')
        self.assertFalse(first['request_started'])
        self.assertFalse(first['request_sent'])
        self.assertNotIn('automatic_replay_blocked', first)
        self.assertEqual(second['phase'], 'response_complete')
        self.assertTrue(second['request_sent'])

    def test_deadline_after_post_send_is_ambiguous_on_next_invocation(self):
        real_request = api.http.client.HTTPConnection.request
        def delayed_request(connection, *args, **kwargs):
            real_request(connection, *args, **kwargs)
            time.sleep(.15)
        with patch.object(api.http.client.HTTPConnection, 'request', delayed_request), \
                self.assertRaises(api.ModelAPIError) as failure:
            self.call(timeout_seconds=.1)
        self.assertEqual(failure.exception.code, 'timeout')
        stats = self.assert_safe_failure('timeout')
        self.assertEqual(stats['phase'], 'waiting_response')
        self.assertTrue(stats['request_sent'])
        self.assertTrue(stats['automatic_replay_blocked'])
        with patch.object(api, '_request', side_effect=AssertionError('No resumed network')) as request, \
                self.assertRaises(api.ModelAPIError) as resumed:
            self.call()
        self.assertEqual(resumed.exception.code, 'ambiguous_previous_request')
        request.assert_not_called()

    def test_failure_during_post_write_is_unknown_and_not_retried(self):
        with patch.object(api.http.client.HTTPConnection, 'request',
                          side_effect=ConnectionResetError('PRIVATE_PROVIDER_EXCEPTION')), \
                self.assertRaises(api.ModelAPIError) as failure:
            self.call()
        self.assertEqual(failure.exception.code, 'request_outcome_unknown')
        record = self.record()
        self.assertEqual(record['attempts'], 1)
        self.assertEqual(self.server.posts, 0)
        stats = record['transport_attempts'][0]
        self.assertEqual(stats['phase'], 'sending_request')
        self.assertTrue(stats['request_started'])
        self.assertFalse(stats['request_sent'])
        self.assertTrue(stats['automatic_replay_blocked'])
        self.assertNotIn('PRIVATE_PROVIDER_EXCEPTION', json.dumps(record) + str(failure.exception))


if __name__ == '__main__':
    unittest.main()
