"""Thread backpressure and offline operational evidence under batch load."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scene_factory.diagnostics import batch_status, usage_report
from batch import atomic_json
import model_api
from runtime_controls import api_session, api_slot, render_session, render_slot

MODEL = {'provider': 'openai_chat', 'model': 'fixture', 'base_url': 'http://localhost:1/v1', 'api_key_env': ''}


class BackpressureTests(unittest.TestCase):
    def test_nested_scene_and_detail_threads_share_one_limit(self):
        lock = threading.Lock()
        release = threading.Event()
        saturated = threading.Event()
        active = peak = calls = 0
        def invoke(*args, **kwargs):
            nonlocal active, peak, calls
            with lock:
                active += 1
                calls += 1
                peak = max(peak, active)
                if active == 2:
                    saturated.set()
            try:
                if not release.wait(5):
                    raise AssertionError('test release timed out')
                return {'ok': True}
            finally:
                with lock:
                    active -= 1
        with api_session(2), patch.object(model_api, '_call_model', side_effect=invoke), ThreadPoolExecutor(8) as pool:
            futures = [pool.submit(model_api.call_model, 'fixture', [], {}, Path('unused'), 10, config=MODEL) for _ in range(8)]
            try:
                self.assertTrue(saturated.wait(4))
                with lock:
                    self.assertEqual(calls, 2)
            finally:
                release.set()
            self.assertTrue(all(future.result()['ok'] for future in futures))
        self.assertEqual((calls, peak), (8, 2))

    def test_queued_call_can_timeout_without_sending_request(self):
        with api_session(1), api_slot(1), patch.object(model_api, '_call_model') as invoke:
            started = time.monotonic()
            with self.assertRaises(model_api.ModelAPIError) as caught:
                model_api.call_model('fixture', [], {}, Path('unused'), .1, config=MODEL)
            self.assertEqual(caught.exception.code, 'timeout')
            self.assertLess(time.monotonic()-started, 2)
            invoke.assert_not_called()

    def test_queue_time_is_deducted_from_request_timeout(self):
        @contextmanager
        def waited_slot(timeout, check_cancelled=None):
            self.assertEqual(timeout, 5)
            yield 2

        with patch('runtime_controls.api_slot', waited_slot), \
                patch.object(model_api, '_call_model', return_value={'ok': True}) as invoke:
            model_api.call_model('fixture', [], {}, Path('unused'), 5, config=MODEL)
        self.assertEqual(invoke.call_args.kwargs['timeout'], 3)

    def test_exception_releases_slot_and_nested_batches_fail_clearly(self):
        with api_session(1):
            with self.assertRaises(RuntimeError):
                with api_session(1):
                    pass
            with self.assertRaises(ValueError):
                with api_slot(1):
                    raise ValueError('fixture')
            with api_slot(.1):
                pass
        with api_session(1):
            pass

    def test_queued_api_stops_without_waiting_for_the_active_request(self):
        stopped = threading.Event()
        queued = threading.Event()
        entered = threading.Event()

        def check_cancelled():
            queued.set()
            if stopped.is_set():
                raise RuntimeError('fixture interrupted')

        def call():
            with api_slot(30, check_cancelled):
                entered.set()

        with api_session(1), api_slot(1), ThreadPoolExecutor(1) as pool:
            future = pool.submit(call)
            self.assertTrue(queued.wait(2))
            stopped.set()
            with self.assertRaisesRegex(RuntimeError, 'fixture interrupted'):
                future.result(timeout=2)
            self.assertFalse(entered.is_set())

    def test_post_acquire_cancellation_releases_api_slot(self):
        checks = 0

        def check_cancelled():
            nonlocal checks
            checks += 1
            if checks == 2:
                raise RuntimeError('cancelled after acquire')

        with api_session(1):
            with self.assertRaisesRegex(RuntimeError, 'cancelled after acquire'):
                with api_slot(1, check_cancelled):
                    self.fail('cancelled API entered')
            with api_slot(.1):
                pass

    def test_cancellation_without_api_session_never_prepares_request(self):
        stopped = threading.Event()
        stopped.set()
        with patch.object(model_api, '_current_stop_event', return_value=stopped), \
                patch.object(model_api, '_call_model') as invoke:
            with self.assertRaises(model_api.ModelAPIError) as caught:
                model_api.call_model('fixture', [], {}, Path('unused'), 30, config=MODEL)
        self.assertEqual(caught.exception.code, 'batch_stopped')
        invoke.assert_not_called()


class RenderBackpressureTests(unittest.TestCase):
    def test_direct_and_batch_callers_share_slots_across_session_boundary(self):
        release = threading.Event()
        entered_direct = threading.Event()
        entered_batch = threading.Event()
        lock = threading.Lock()
        active = peak = 0

        def direct():
            nonlocal active, peak
            with render_slot():
                with lock:
                    active += 1
                    peak = max(peak, active)
                entered_direct.set()
                if not release.wait(5):
                    raise AssertionError('direct render release timed out')
                with lock:
                    active -= 1

        def batch():
            nonlocal active, peak
            with render_slot():
                with lock:
                    active += 1
                    peak = max(peak, active)
                    active -= 1
                entered_batch.set()

        with ThreadPoolExecutor(2) as pool:
            future = pool.submit(direct)
            self.assertTrue(entered_direct.wait(2))
            with render_session(1):
                batch_future = pool.submit(batch)
                try:
                    self.assertFalse(entered_batch.wait(.2))
                finally:
                    release.set()
                self.assertTrue(entered_batch.wait(2))
                batch_future.result(timeout=2)
            future.result(timeout=2)
        self.assertEqual(peak, 1)

    def test_actual_worker_overlap_obeys_default_and_explicit_slots(self):
        for limit in (1, 2):
            with self.subTest(limit=limit):
                lock = threading.Lock()
                release = threading.Event()
                saturated = threading.Event()
                active = peak = calls = 0

                def render():
                    nonlocal active, peak, calls
                    with render_slot() as queue_seconds:
                        self.assertGreaterEqual(queue_seconds, 0)
                        with lock:
                            active += 1
                            calls += 1
                            peak = max(peak, active)
                            if active == limit:
                                saturated.set()
                        try:
                            if not release.wait(5):
                                raise AssertionError('render release timed out')
                        finally:
                            with lock:
                                active -= 1

                with render_session(limit), ThreadPoolExecutor(6) as pool:
                    futures = [pool.submit(render) for _ in range(6)]
                    try:
                        self.assertTrue(saturated.wait(4))
                        with lock:
                            self.assertEqual(calls, limit)
                    finally:
                        release.set()
                    for future in futures:
                        future.result(timeout=5)
                self.assertEqual((calls, peak), (6, limit))

    def test_queued_render_stops_before_entering_blender(self):
        stopped = threading.Event()
        queued = threading.Event()
        entered = threading.Event()

        def check_cancelled():
            queued.set()
            if stopped.is_set():
                raise RuntimeError('fixture interrupted')

        def render():
            with render_slot(check_cancelled):
                entered.set()

        with render_session(), render_slot(), ThreadPoolExecutor(1) as pool:
            future = pool.submit(render)
            self.assertTrue(queued.wait(2))
            stopped.set()
            with self.assertRaisesRegex(RuntimeError, 'fixture interrupted'):
                future.result(timeout=2)
            self.assertFalse(entered.is_set())

    def test_exception_and_post_acquire_cancellation_release_slots(self):
        with render_session():
            with self.assertRaisesRegex(ValueError, 'render failed'):
                with render_slot():
                    raise ValueError('render failed')
            checks = 0

            def check_cancelled():
                nonlocal checks
                checks += 1
                if checks == 2:
                    raise RuntimeError('cancelled after acquire')

            with self.assertRaisesRegex(RuntimeError, 'cancelled after acquire'):
                with render_slot(check_cancelled):
                    self.fail('cancelled render entered')
            with ThreadPoolExecutor(1) as pool:
                def next_render():
                    with render_slot():
                        return True
                self.assertTrue(pool.submit(next_render).result(timeout=2))

    def test_invalid_and_nested_sessions_fail_and_api_limit_is_independent(self):
        for value in (True, 0, 9, 1.5, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                with render_session(value):
                    pass
        with render_session(), api_session():
            with self.assertRaisesRegex(RuntimeError, 'Blender batch'):
                with render_session():
                    pass
            with api_slot(.1), render_slot():
                pass
        with render_session():
            pass


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_usage_counts_each_transport_without_duplicating_final_usage(self):
        for index, record in enumerate([
            {'status': 'completed', 'attempts': 2, 'usage': {'total_tokens': 10},
             'transport_attempts': [{'usage': {'total_tokens': 3}}, {'usage': {'total_tokens': 10}}]},
            {'status': 'failed', 'attempts': 1, 'usage': {'total_tokens': 7, 'arbitrary': 'ignored'}},
            {'status': 'failed', 'attempts': 1},
        ]):
            atomic_json(self.root/f'call_{index}.api.json', {**record, 'config': {'provider': 'fixture', 'model': 'test'}})
        (self.root/'broken.api.json').write_text('{')
        result = usage_report(self.root)
        self.assertEqual((result['logical_calls'], result['transport_attempts']), (3, 4))
        self.assertEqual(result['models'][0]['tokens'], {'total_tokens': 20})
        self.assertEqual(result['models'][0]['attempts_with_reported_usage'], 3)
        self.assertEqual(result['models'][0]['failed_calls'], 2)
        self.assertEqual(result['unreadable_records'], 1)
        self.assertFalse(result['invoice_complete'])

    def test_status_is_bounded_and_explicitly_unverified_by_default(self):
        atomic_json(self.root/'summary.json', {'scenes': [{'scene_id': 'a', 'status': 'accepted'},
                    {'scene_id': 'b', 'status': 'failed'}], 'total': 2, 'run_state': 'completed'})
        atomic_json(self.root/'a/progress.json', {'stage': 'complete', 'status': 'accepted', 'at': 'fixture'})
        result = batch_status(self.root, limit=1)
        self.assertEqual(result['counts'], {'accepted': 1, 'failed': 1})
        self.assertEqual(result['omitted_scenes'], 1)
        self.assertFalse(result['accepted_artifacts_verified'])
        self.assertEqual(result['scenes'][0]['last_progress']['stage'], 'complete')
        with patch('batch_multiview.recorded_results', return_value={'a': {'scene_id': 'a', 'status': 'failed'}}):
            self.assertEqual(batch_status(self.root, verify=True)['counts'], {'failed': 1})


if __name__ == '__main__':
    unittest.main()
