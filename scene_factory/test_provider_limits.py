"""Local-only quota admission tests; fake time avoids minute-long sleeps."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import model_api
from PIL import Image
from provider_limits import (LEDGER_NAME, ProviderLimitError, ProviderLimiter,
                             normalize_rate_limit, provider_limit_session, reserve_request)


class Clock:
    def __init__(self):
        self.now = 1000.25
        self.waits = []

    def time(self):
        return self.now

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds
        return False


def config(**rate):
    return model_api.normalize_config({'provider': 'openai_chat', 'model': 'fixture',
        'base_url': 'http://127.0.0.1:29183/v1', 'api_key_env': '',
        'max_output_tokens': 60, 'max_retries': 0,
        'rate_limit': rate})


def complete(input_tokens=10, output_tokens=20, **extra):
    return {'phase': 'response_complete', 'request_started': True, 'request_sent': True,
            'http_status': 200, 'usage': {'prompt_tokens': input_tokens,
                'completion_tokens': output_tokens, 'total_tokens': input_tokens+output_tokens}, **extra}


class ProviderLimitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = Clock()
        self.image = self.root/'source.png'
        Image.new('RGB', (4, 4), (32, 64, 96)).save(self.image)

    def limiter(self, configs, **kwargs):
        limiter = ProviderLimiter(self.root, configs, clock=self.clock.time,
            monotonic=self.clock.time, waiter=self.clock.wait, **kwargs)
        limiter.initialize()
        self.addCleanup(limiter.close)
        return limiter

    def test_strict_config_validation_and_absent_default(self):
        base = config()
        self.assertEqual(base['rate_limit'], dict.fromkeys(('requests_per_minute', 'tokens_per_minute', 'input_tokens_per_request'), 0))
        absent = dict(base)
        del absent['rate_limit']
        self.assertNotIn('rate_limit', model_api.normalize_config(absent))
        for value in (None, [], True, {'unexpected': 1}, {'requests_per_minute': True},
                      {'tokens_per_minute': -1}, {'requests_per_minute': 1.5},
                      {'tokens_per_minute': 100},
                      {'tokens_per_minute': 90, 'input_tokens_per_request': 40}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model_api.normalize_config({**base, 'rate_limit': value})
        self.assertEqual(normalize_rate_limit({'tokens_per_minute': 100,
            'input_tokens_per_request': 40}, max_output_tokens=60)['tokens_per_minute'], 100)

    def test_default_no_session_no_ledger_and_enabled_fail_closed(self):
        base = config()
        self.assertIsNone(reserve_request(self.root/'a.json', base, deadline=1100))
        self.assertFalse((self.root/LEDGER_NAME).exists())
        with self.assertRaises(ProviderLimitError) as caught:
            reserve_request(self.root/'a.json', config(requests_per_minute=1), deadline=1100)
        self.assertEqual(caught.exception.code, 'provider_rate_session_required')

    def test_rpm_wait_and_conservative_window_boundary(self):
        cfg = config(requests_per_minute=1)
        limiter = self.limiter([cfg])
        first = limiter.reserve(cfg, 1100)
        first.finish(complete())
        second = limiter.reserve(cfg, 1100)
        self.assertGreaterEqual(self.clock.now, 1061)
        self.assertAlmostEqual(second.waited_seconds, self.clock.now-1000.25)
        self.assertTrue(self.clock.waits)
        self.assertEqual(limiter.snapshot()['scopes'][0]['requests'], 1)

    def test_complete_usage_releases_tpm_without_counting_reasoning_twice(self):
        cfg = config(tokens_per_minute=100, input_tokens_per_request=40)
        limiter = self.limiter([cfg])
        ticket = limiter.reserve(cfg, 1100)
        result = complete(10, 20)
        result['usage']['completion_tokens_details'] = {'reasoning_tokens': 12}
        ticket.finish(result)
        self.assertEqual(limiter.snapshot()['scopes'][0]['tokens'], 30)
        smaller = {**cfg, 'max_output_tokens': 30}
        limiter.reserve(smaller, 1100)
        self.assertEqual(self.clock.waits, [])
        self.assertEqual(limiter.snapshot()['scopes'][0]['tokens'], 100)

    def test_missing_partial_invalid_inconsistent_usage_is_conservative(self):
        cfg = config(tokens_per_minute=100, input_tokens_per_request=40)
        variants = [
            {'phase': 'response_complete', 'http_status': 200},
            {**complete(), 'usage': {'completion_tokens': 20}},
            {**complete(), 'usage_invalid': True},
            {**complete(), 'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 21}},
            {**complete(), 'automatic_replay_blocked': True},
            {**complete(), 'phase': 'receiving_response'},
        ]
        for index, transport in enumerate(variants):
            with self.subTest(index=index):
                ticket = self.limiter_for_variant(index, cfg).reserve(cfg, 1100)
                ticket.finish(transport)
                self.assertEqual(ticket._limiter.snapshot()['scopes'][0]['tokens'], 100)

    def limiter_for_variant(self, index, cfg):
        limiter = ProviderLimiter(self.root/str(index), [cfg], clock=self.clock.time,
            monotonic=self.clock.time, waiter=self.clock.wait)
        limiter.initialize()
        self.addCleanup(limiter.close)
        return limiter

    def test_reported_usage_above_reservation_is_counted(self):
        cfg = config(tokens_per_minute=100, input_tokens_per_request=40)
        limiter = self.limiter([cfg])
        ticket = limiter.reserve(cfg, 1100)
        ticket.finish(complete(150, 20))
        self.assertEqual(limiter.snapshot()['scopes'][0]['tokens'], 170)
        limiter.reserve(cfg, 1100)
        self.assertGreaterEqual(self.clock.now, 1061)

    def test_only_proven_unsent_refunds_rpm_and_tpm(self):
        cfg = config(requests_per_minute=1, tokens_per_minute=100, input_tokens_per_request=40)
        limiter = self.limiter([cfg])
        for phase in ('not_sent', 'connecting'):
            ticket = limiter.reserve(cfg, 1100)
            ticket.finish({'phase': phase, 'request_started': False, 'request_sent': False})
            self.assertEqual(limiter.snapshot()['scopes'][0]['requests'], 0)
            self.assertEqual(limiter.snapshot()['scopes'][0]['tokens'], 0)
        unknown = limiter.reserve(cfg, 1100)
        unknown.finish({'phase': 'sending_request', 'request_started': True, 'request_sent': False})
        self.assertEqual(limiter.snapshot()['scopes'][0]['requests'], 1)
        self.assertEqual(limiter.snapshot()['scopes'][0]['tokens'], 100)
        self.assertEqual(self.clock.waits, [])

    def test_retry_each_attempt_has_its_own_reservation(self):
        cfg = config(requests_per_minute=10)
        limiter = self.limiter([cfg])
        limiter.reserve(cfg, 1100).finish({'phase': 'receiving_response',
            'request_started': True, 'request_sent': True, 'http_status': 429})
        limiter.reserve(cfg, 1100).finish(complete())
        snapshot = limiter.snapshot()
        self.assertEqual(snapshot['scopes'][0]['requests'], 2)
        self.assertEqual(snapshot['pending_count'], 0)

    def test_pending_and_settled_expire_and_buckets_remain_bounded(self):
        cfg = config(requests_per_minute=1000)
        limiter = self.limiter([cfg])
        old_ticket = limiter.reserve(cfg, 1100)
        for _ in range(100):
            limiter.reserve(cfg, self.clock.now+10).finish(complete())
            self.clock.now += 1
        snapshot = limiter.snapshot()
        self.assertLessEqual(snapshot['bucket_count'], 61)
        self.assertEqual(snapshot['pending_count'], 0)
        old_ticket.finish(complete())
        self.clock.now += 62
        self.assertEqual(limiter.snapshot()['bucket_count'], 0)

    def test_scope_shared_across_models_and_credentials_strictest_quota(self):
        a = config(requests_per_minute=3)
        b = {**a, 'model': 'other', 'api_key_env': 'OTHER_KEY',
             'rate_limit': {**a['rate_limit'], 'requests_per_minute': 1}}
        limiter = self.limiter([a, b])
        limiter.reserve(a, 1100).finish(complete())
        limiter.reserve(b, 1100)
        self.assertGreaterEqual(self.clock.now, 1061)
        self.assertEqual(len(limiter.snapshot()['scopes']), 1)
        serialized = (self.root/LEDGER_NAME).read_bytes()
        self.assertNotIn(b'OTHER_KEY', serialized)
        self.assertNotIn(b'127.0.0.1', serialized)

    def test_shared_tpm_requires_all_models_have_input_allowance(self):
        a = config(tokens_per_minute=100, input_tokens_per_request=40)
        b = {**config(), 'model': 'other'}
        with self.assertRaises(ValueError):
            ProviderLimiter(self.root, [a, b])

    def test_concurrent_admission_never_exceeds_rpm(self):
        cfg = config(requests_per_minute=4)
        limiter = self.limiter([cfg])
        # Constant fake time, and a waiter that consumes the caller deadline,
        # lets excess callers fail immediately without advancing the window.
        limiter.waiter = lambda _: (_ for _ in ()).throw(ProviderLimitError('timeout', 'fixture timeout'))
        def reserve(_):
            try:
                return limiter.reserve(cfg, 1100)
            except ProviderLimitError:
                return None
        with ThreadPoolExecutor(max_workers=12) as pool:
            tickets = list(pool.map(reserve, range(12)))
        self.assertEqual(sum(ticket is not None for ticket in tickets), 4)
        self.assertEqual(limiter.snapshot()['scopes'][0]['requests'], 4)

    def test_persistence_and_second_connection_share_quota(self):
        cfg = config(requests_per_minute=1)
        one = self.limiter([cfg])
        one.reserve(cfg, 1100).finish(complete())
        two = self.limiter([cfg])
        two.reserve(cfg, 1100)
        self.assertGreaterEqual(self.clock.now, 1061)
        self.assertEqual(two.snapshot()['scopes'][0]['requests'], 1)

    def test_deadline_and_stop_event_prevent_post_admission(self):
        cfg = config(requests_per_minute=1)
        event = threading.Event()
        limiter = self.limiter([cfg], stop_event=event)
        limiter.reserve(cfg, 1100)
        with self.assertRaises(ProviderLimitError) as caught:
            limiter.reserve(cfg, 1001)
        self.assertEqual(caught.exception.code, 'timeout')
        event.set()
        with self.assertRaises(ProviderLimitError) as caught:
            limiter.reserve(cfg, 1100)
        self.assertEqual(caught.exception.code, 'batch_stopped')

    def test_policy_change_with_live_buckets_rejected(self):
        cfg = config(requests_per_minute=1)
        limiter = self.limiter([cfg])
        limiter.reserve(cfg, 1100)
        with self.assertRaises(ProviderLimitError) as caught:
            self.limiter([config(requests_per_minute=2)])
        self.assertEqual(caught.exception.code, 'provider_rate_policy_changed')

    def test_session_binds_output_tree_and_rejects_overlap(self):
        cfg = config(requests_per_minute=2)
        with provider_limit_session(self.root, [cfg]) as limiter:
            ticket = reserve_request(self.root/'nested'/'a.json', cfg,
                                     deadline=__import__('time').monotonic()+2)
            self.assertIsNotNone(ticket)
            ticket.finish(complete())
            self.assertEqual(limiter.snapshot()['scopes'][0]['requests'], 1)
            with self.assertRaises(ProviderLimitError):
                with provider_limit_session(self.root/'nested', [cfg]):
                    pass

    def test_separate_output_shards_share_provider_quota_across_processes(self):
        shared = self.root/'shared-ledger'
        shard_a = shared/'shard-a'
        shard_b = shared/'shard-b'
        cfg = config(requests_per_minute=1)
        with provider_limit_session(shared, [cfg]) as limiter:
            ticket = reserve_request(shard_a/'scene/analysis.json', cfg,
                                     deadline=__import__('time').monotonic()+2)
            ticket.finish(complete())
            self.assertEqual(limiter.snapshot()['scopes'][0]['requests'], 1)

        code = '''
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from model_api import normalize_config
from provider_limits import ProviderLimitError, provider_limit_session, reserve_request
shared, output = Path(sys.argv[2]), Path(sys.argv[3])
cfg = normalize_config({'provider':'openai_chat','model':'fixture',
    'base_url':'http://127.0.0.1:29183/v1','api_key_env':'',
    'max_output_tokens':60,'max_retries':0,
    'rate_limit':{'requests_per_minute':1}})
with provider_limit_session(shared, [cfg]):
    try:
        reserve_request(output/'scene/analysis.json', cfg, deadline=time.monotonic()+.08)
    except ProviderLimitError as exc:
        if exc.code != 'timeout':
            raise
    else:
        raise SystemExit('shared provider quota admitted a second request')
'''
        process = subprocess.run([sys.executable, '-c', code,
                                  str(Path(__file__).parent), str(shared), str(shard_b)],
                                 capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        with provider_limit_session(shared, [cfg]) as control:
            snapshot = control.snapshot()
            self.assertEqual(snapshot['scopes'][0]['requests'], 1)
            self.assertEqual(snapshot['pending_count'], 0)

    def test_transport_integration_retry_and_usage(self):
        cfg = config(requests_per_minute=10, tokens_per_minute=1000, input_tokens_per_request=40)
        cfg['max_retries'] = 1
        calls = []
        def request(config, body, key, deadline, *, transport, stop_event=None):
            calls.append(transport['provider_rate_limit_request_id'])
            transport.update(phase='response_complete', request_started=True, request_sent=True)
            if len(calls) == 1:
                return 429, 0, None
            return 200, 0, b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"ok\\":true}"}}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}'
        schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
                  'required': ['ok'], 'additionalProperties': False}
        with provider_limit_session(self.root, [cfg]) as limiter, patch.object(model_api, '_request', side_effect=request), patch.object(model_api.time, 'sleep'):
            result = model_api.call_model('fixture', [self.image], schema, self.root/'result.json', config=cfg)
            self.assertEqual(result, {'ok': True})
            self.assertEqual(len(set(calls)), 2)
            snapshot = limiter.snapshot()
            self.assertEqual(snapshot['scopes'][0]['requests'], 2)
            self.assertEqual(snapshot['scopes'][0]['tokens'], 130)
            self.assertEqual(snapshot['pending_count'], 0)

    def test_missing_session_fails_before_transport(self):
        cfg = config(requests_per_minute=1)
        schema = {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}
        with patch.object(model_api, '_request') as request, self.assertRaises(model_api.ModelAPIError) as caught:
            model_api.call_model('fixture', [self.image], schema, self.root/'result.json', config=cfg)
        request.assert_not_called()
        self.assertEqual(caught.exception.code, 'provider_rate_session_required')


if __name__ == '__main__':
    unittest.main()
