"""Thread backpressure and offline operational evidence under batch load."""
from concurrent.futures import ThreadPoolExecutor
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
from runtime_controls import api_session, api_slot

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
