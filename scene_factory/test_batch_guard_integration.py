"""Real loopback transport + persistent batch stop/recovery; no paid API."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from batch import atomic_json
from batch_guard import batch_session, snapshot
from scene_budget import scene_budget_session
import test_model_api as transport_fixture
from test_model_api import SCHEMA
import model_api as api
import batch_multiview as runner
from diagnostics import batch_status
from batch_index import append_result, EVENTS_FILE
from provider_limits import provider_limit_session
from runtime_controls import api_session, api_slot


class GuardTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        transport_fixture.ModelAPITests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        transport_fixture.ModelAPITests.tearDownClass()

    def setUp(self):
        self.fixture = transport_fixture.ModelAPITests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.folder.resolve()
        self.fixture.output = self.root / 'scene' / 'analysis.json'
        self.fixture.output.parent.mkdir()

    def test_fatal_http_stops_other_scenes_and_requires_explicit_reopen(self):
        f = self.fixture
        f.queue({}, status=402)
        with batch_session(self.root, {'cooldown_seconds': 0}):
            with self.assertRaises(api.ModelAPIError):
                f.call()
            with self.assertRaisesRegex(api.ModelAPIError, 'stopped'):
                f.call()
        self.assertEqual(len(f.server.requests), 1)
        self.assertEqual(snapshot(self.root)['summary']['stop_reason'], 'http_402')
        with batch_session(self.root, {'cooldown_seconds': 0}):
            with self.assertRaises(api.ModelAPIError):
                f.call()
        f.queue()
        with batch_session(self.root, {'cooldown_seconds': 0}, resume=True):
            f.call()
        self.assertEqual(len(f.server.requests), 2)
        self.assertEqual(snapshot(self.root)['summary']['requests'], 2)

    def test_retry_failure_threshold_caps_http_attempts(self):
        f = self.fixture
        f.queue({}, status=503)
        f.queue({}, status=503)
        with batch_session(self.root, {'cooldown_seconds': 0, 'failure_threshold': 2}):
            with self.assertRaises(api.ModelAPIError):
                f.call(max_retries=3)
        self.assertEqual(len(f.server.requests), 2)
        self.assertEqual(snapshot(self.root)['summary']['stop_reason'], 'consecutive_provider_failures')

    def test_scene_limit_releases_unsent_batch_output_reservation(self):
        f = self.fixture
        f.queue()
        with batch_session(self.root), scene_budget_session(f.output.parent, {'max_requests': 1}):
            f.call(max_output_tokens=100)
            with self.assertRaises(api.ModelAPIError):
                f.call(max_output_tokens=100)
        state = snapshot(self.root)['summary']
        self.assertEqual(len(f.server.requests), 1)
        self.assertEqual(state['output_tokens_accounted'], 7)
        self.assertEqual(state['pending_attempts'], 0)
        self.assertEqual(state['unknown_usage_attempts'], 0)
        self.assertEqual(state['consecutive_failures'], 0)

    def test_batch_budget_blocked_resume_preserves_completed_api_provenance(self):
        f = self.fixture
        f.queue()
        with batch_session(self.root, {'max_requests': 1, 'cooldown_seconds': 0}):
            self.assertEqual(f.call(), transport_fixture.RESULT)
            with self.assertRaises(api.ModelAPIError) as caught:
                f.call()

        self.assertEqual(caught.exception.code, 'batch_budget_exceeded')
        self.assertEqual(len(f.server.requests), 1)
        record = f.record()
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(len(record['transport_attempts']), 1)
        self.assertEqual(record['usage']['total_tokens'], 19)

    def test_batch_budget_blocked_resume_preserves_rejected_response_usage(self):
        f = self.fixture
        f.queue(transport_fixture.chat(content='{"scene":"fixture","number":9}'))
        with batch_session(self.root, {'max_requests': 1, 'cooldown_seconds': 0}):
            with self.assertRaises(api.ModelAPIError) as rejected:
                f.call()
            self.assertEqual(rejected.exception.code, 'schema_mismatch')
            with self.assertRaises(api.ModelAPIError) as blocked:
                f.call()

        self.assertEqual(blocked.exception.code, 'batch_budget_exceeded')
        self.assertEqual(len(f.server.requests), 1)
        record = f.record()
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['failure_code'], 'schema_mismatch')
        self.assertEqual(len(record['transport_attempts']), 1)
        self.assertEqual(record['usage']['total_tokens'], 19)

    def test_queued_calls_observe_real_session_stop_without_preparation_or_reservations(self):
        # Hold the only API slot throughout cancellation. This exercises the
        # real session registry, without releasing a running request first.
        f = self.fixture
        for kind in ('batch', 'provider'):
            with self.subTest(session=kind):
                stopped = threading.Event()
                queued = threading.Event()
                config = {**f.config, 'rate_limit': {'requests_per_minute': 1}}
                session = (batch_session(self.root, stop_event=stopped) if kind == 'batch'
                           else provider_limit_session(self.root, [config], stop_event=stopped))

                @contextmanager
                def observed_slot(timeout, check_cancelled=None):
                    def observe():
                        queued.set()
                        check_cancelled()
                    with api_slot(timeout, observe) as waited:
                        yield waited

                with session as control, api_session(1), api_slot(1), \
                        patch('runtime_controls.api_slot', observed_slot), \
                        patch.object(api, '_credential') as credential, \
                        patch.object(api, '_image_payloads') as images, \
                        ThreadPoolExecutor(1) as pool:
                    future = pool.submit(api.call_model, 'fixture', [f.image], SCHEMA,
                                         f.output, 30, config=config)
                    self.assertTrue(queued.wait(2))
                    stopped.set()
                    with self.assertRaises(api.ModelAPIError) as caught:
                        future.result(timeout=2)
                    self.assertEqual(caught.exception.code, 'batch_stopped')
                    credential.assert_not_called()
                    images.assert_not_called()
                    self.assertEqual(f.server.requests, [])
                    self.assertFalse(f.output.with_suffix('.api.json').exists())
                    if kind == 'batch':
                        state = snapshot(self.root)['summary']
                        self.assertEqual(state['requests'], 0)
                        self.assertEqual(state['pending_attempts'], 0)
                    else:
                        state = control.snapshot()
                        self.assertEqual(state['pending_count'], 0)
                        self.assertEqual(sum(scope['requests'] for scope in state['scopes']), 0)

    def test_coordinator_stops_scheduling_after_batch_limit(self):
        f = self.fixture
        for _ in range(2):
            f.queue()
        out = self.root / 'batch'
        args = SimpleNamespace(output_dir=out, workers=1, model_configs={'analysis': f.config},
                               batch_budget_policy={'max_requests': 2, 'cooldown_seconds': 0})
        groups = [{'scene_id': f's_{i}', 'images': [f.image]} for i in range(100)]
        def work(group, args, hashes):
            folder = out / group['scene_id']
            folder.mkdir()
            api.call_model('test', [f.image], SCHEMA, folder/'analysis.json', config=f.config)
            result = {'scene_id': group['scene_id'], 'status': 'generated_unreviewed'}
            atomic_json(folder/'result.json', result)
            return result
        with patch.object(runner, 'run_job', side_effect=work), patch('report_multiview.report'), redirect_stdout(io.StringIO()):
            self.assertEqual(runner._coordinate(groups, args, {}), 1)
        self.assertEqual(len(f.server.requests), 2)
        status = batch_status(out)
        self.assertEqual(status['run_state'], 'stopped')
        self.assertEqual(status['counts'], {'generated_unreviewed': 2, 'failed': 1, 'interrupted': 97})

    def test_stop_interrupts_retry_backoff_without_second_request_or_reservation(self):
        f = self.fixture
        for kind in ('batch', 'provider'):
            with self.subTest(session=kind):
                waiting = threading.Event()

                class ObservedStop(threading.Event):
                    def wait(self, timeout=None):
                        if timeout == 30:
                            waiting.set()
                        return super().wait(timeout)

                stopped = ObservedStop()
                root = self.root/kind
                output = root/'scene/analysis.json'
                config = {**f.config, 'max_retries': 2, 'timeout_seconds': 60}
                if kind == 'provider':
                    config['rate_limit'] = {'requests_per_minute': 5}
                session = (batch_session(root, {'cooldown_seconds': 0}, stop_event=stopped)
                           if kind == 'batch' else provider_limit_session(root, [config], stop_event=stopped))
                f.server.requests.clear()
                f.queue({}, status=429, **{'Retry-After': '30'})
                with session as control, \
                        patch.object(api.time, 'sleep', side_effect=AssertionError('Backoff must be interruptible')), \
                        ThreadPoolExecutor(1) as pool:
                    future = pool.submit(api.call_model, 'fixture', [f.image], SCHEMA, output, config=config)
                    try:
                        self.assertTrue(waiting.wait(2), 'Request did not enter cancellable backoff')
                    finally:
                        stopped.set()
                    with self.assertRaises(api.ModelAPIError) as caught:
                        future.result(timeout=2)
                    self.assertEqual(caught.exception.code, 'batch_stopped')
                    self.assertEqual(len(f.server.requests), 1)
                    record = json.loads(output.with_suffix('.api.json').read_text())
                    self.assertEqual(record['attempts'], 1)
                    self.assertEqual(len(record['transport_attempts']), 1)
                    self.assertEqual(record['transport_attempts'][0]['http_status'], 429)
                    api.guard_ambiguous_model_resume([output.with_suffix('.api.json')])
                    if kind == 'batch':
                        state = snapshot(root)['summary']
                        self.assertEqual(state['requests'], 1)
                        self.assertEqual(state['pending_attempts'], 0)
                    else:
                        state = control.snapshot()
                        self.assertEqual(state['pending_count'], 0)
                        self.assertEqual(state['scopes'][0]['requests'], 1)

    def test_coordinator_rate_session_blocks_second_transport_within_deadline(self):
        f = self.fixture
        f.queue()
        config = {**f.config, 'max_retries': 0, 'rate_limit': {'requests_per_minute': 1}}
        out = self.root/'rate_batch'
        args = SimpleNamespace(output_dir=out, workers=1, no_review=True,
                               model_configs={'analysis': config}, render_workers=2,
                               batch_budget_policy={'cooldown_seconds': 0})
        groups = [{'scene_id': f's_{i}', 'images': [f.image]} for i in range(2)]

        def work(group, args, hashes):
            folder = out/group['scene_id']
            folder.mkdir()
            # The first request completes; the second reaches a real local
            # rate queue and expires without another HTTP POST.
            timeout = 2 if group['scene_id'] == 's_0' else .1
            api.call_model('test', [f.image], SCHEMA, folder/'analysis.json', timeout, config=config)
            result = {'scene_id': group['scene_id'], 'status': 'generated_unreviewed'}
            atomic_json(folder/'result.json', result)
            return result

        with patch.object(runner, 'run_job', side_effect=work), patch('report_multiview.report'), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.coordinate(groups, args, {}), 1)
        self.assertEqual(len(f.server.requests), 1)
        state = json.loads((out/'run_state.json').read_text())
        self.assertEqual(state['provider_rate_control']['pending_count'], 0)
        self.assertEqual(state['provider_rate_control']['scopes'][0]['requests'], 1)
        self.assertEqual(snapshot(out)['summary']['pending_attempts'], 0)
        self.assertEqual(batch_status(out)['counts'], {'generated_unreviewed': 1, 'failed': 1})


class LiveIndexTests(unittest.TestCase):
    def test_status_reads_shared_ledger_checkpoint_from_shard_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch_control = {'policy': {'max_requests': 5},
                             'summary': {'requests': 2, 'pending_attempts': 0}}
            provider_control = {'scopes': [{'requests': 2, 'tokens': 40}],
                                'bucket_count': 1, 'pending_count': 0}
            atomic_json(root/'summary.json', {'run_id': 'shared', 'run_state': 'running', 'total': 1,
                'incremental_events': EVENTS_FILE,
                'scenes': [{'scene_id': 'a', 'status': 'pending'}],
                'batch_control': batch_control, 'provider_rate_control': provider_control})
            atomic_json(root/'run_state.json', {'run_id': 'shared', 'state': 'running',
                'batch_control': batch_control, 'provider_rate_control': provider_control})
            value = batch_status(root)
            self.assertEqual(value['batch_control'], batch_control)
            self.assertEqual(value['provider_rate_control'], provider_control)

    def test_live_diagnostics_merge_only_current_run_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root/'summary.json', {'run_id': 'current', 'run_state': 'running', 'total': 2,
                'incremental_events': EVENTS_FILE, 'scenes': [
                    {'scene_id': 'a', 'status': 'pending'}, {'scene_id': 'b', 'status': 'pending'}]})
            append_result(root, 'previous', 1, {'scene_id': 'b', 'status': 'accepted'})
            append_result(root, 'current', 1, {'scene_id': 'a', 'status': 'generated_unreviewed'})
            atomic_json(root/'run_state.json', {'run_id': 'current', 'state': 'stopping', 'updated_at': 'now'})
            value = batch_status(root)
            self.assertEqual(value['counts'], {'generated_unreviewed': 1, 'pending': 1})
            self.assertEqual(value['run_state'], 'stopping')
            self.assertEqual(value['updated_at'], 'now')
            self.assertFalse(value['accepted_artifacts_verified'])


if __name__ == '__main__':
    unittest.main()
