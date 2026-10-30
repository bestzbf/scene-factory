"""Batch ledger concurrency, accounting and provider circuit tests; no network."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from batch import atomic_json
from batch_guard import (BatchBudgetError, LEDGER_NAME, batch_session, current_stop,
                         normalize_policy, reserve_request, snapshot)
from scene_budget import reserve_request as scene_reserve, scene_budget_session

CONFIG = {'provider': 'openai_chat', 'base_url': 'http://localhost:1234/v1', 'timeout_seconds': 2}
COMPLETE = {'phase': 'response_complete', 'http_status': 200,
            'usage': {'input_tokens': 12, 'output_tokens': 7, 'total_tokens': 19}}


class BatchGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def reserve(self, name='scene/stage.json', cap=20, **kwargs):
        return reserve_request(self.root/name, cap, kwargs.pop('config', CONFIG), **kwargs)

    def test_policy_validation(self):
        self.assertEqual(normalize_policy()['cooldown_seconds'], 10)
        for invalid in ({'max_requests': True}, {'max_output_tokens': -1},
                        {'failure_threshold': 0}, {'cooldown_seconds': float('nan')}, {'unknown': 1}):
            with self.subTest(invalid=invalid), self.assertRaises(BatchBudgetError):
                normalize_policy(invalid)

    def test_request_allowance_shared_across_threads_and_preserved_on_resume(self):
        policy = {'max_requests': 37, 'cooldown_seconds': 0}
        event = threading.Event()
        def attempt(i):
            try:
                ticket = self.reserve(f'scene_{i}/stage.json')
                ticket.finish(COMPLETE)
                return True
            except BatchBudgetError as exc:
                self.assertEqual(exc.code, 'batch_budget_exceeded')
                return False
        with batch_session(self.root, policy, stop_event=event):
            with ThreadPoolExecutor(max_workers=16) as pool:
                self.assertEqual(sum(pool.map(attempt, range(100))), 37)
            self.assertTrue(event.is_set())
            self.assertEqual(current_stop(), 'max_requests')
        counts = snapshot(self.root)['summary']
        self.assertEqual(counts['requests'], 37)
        self.assertEqual(counts['output_tokens_accounted'], 37*7)
        self.assertEqual(counts['reported_tokens'], 37*19)
        self.assertEqual(counts['pending_attempts'], 0)
        with batch_session(self.root, policy, resume=True):
            self.assertEqual(current_stop(), 'max_requests')
            with self.assertRaises(BatchBudgetError):
                self.reserve('later.json')
        self.assertEqual(snapshot(self.root)['summary']['requests'], 37)

    def test_unknown_usage_keeps_full_reservation_and_complete_usage_releases_it(self):
        policy = {'max_output_tokens': 47, 'cooldown_seconds': 0}
        with batch_session(self.root, policy):
            first = self.reserve()
            first.finish(COMPLETE)
            first.finish(COMPLETE)  # Settlement is idempotent.
            self.reserve('scene/two.json').finish({'http_status': 200, 'phase': 'response_complete'})
            self.reserve('scene/three.json').finish({'http_status': 400})
            with self.assertRaises(BatchBudgetError) as failure:
                self.reserve('scene/four.json', cap=1)
            self.assertEqual(failure.exception.code, 'batch_budget_exceeded')
        counts = snapshot(self.root)['summary']
        self.assertEqual(counts['requests'], 3)
        self.assertEqual(counts['output_tokens_accounted'], 47)
        self.assertEqual(counts['unknown_usage_attempts'], 2)

    def test_inflight_output_capacity_waits_without_latching_then_recovers(self):
        with batch_session(self.root, {'max_output_tokens': 100}):
            first = self.reserve('first.json', cap=60)
            with ThreadPoolExecutor(max_workers=1) as pool:
                second = pool.submit(self.reserve, 'second.json', cap=60)
                time.sleep(.08)
                self.assertFalse(second.done())
                self.assertIsNone(current_stop())
                self.assertEqual(snapshot(self.root)['summary']['requests'], 1)
                first.finish(COMPLETE)
                second.result(timeout=2).finish(COMPLETE)
            self.reserve('third.json', cap=60).finish(COMPLETE)
        self.assertEqual(snapshot(self.root)['summary']['output_tokens_accounted'], 21)

    def test_inflight_capacity_deadline_does_not_latch_or_consume_capacity(self):
        with batch_session(self.root, {'max_output_tokens': 100}):
            first = self.reserve('first.json', cap=60)
            with self.assertRaises(BatchBudgetError) as failure:
                self.reserve('second.json', cap=60, deadline=time.monotonic()+.03)
            self.assertEqual(failure.exception.code, 'timeout')
            self.assertIsNone(current_stop())
            self.assertEqual(snapshot(self.root)['summary']['requests'], 1)
            first.finish(COMPLETE)
            self.reserve('third.json', cap=60).finish(COMPLETE)

    def test_inflight_unknown_settlement_can_latch_fixed_output_exhaustion(self):
        with batch_session(self.root, {'max_output_tokens': 100}):
            first = self.reserve('first.json', cap=60)
            with ThreadPoolExecutor(max_workers=1) as pool:
                second = pool.submit(self.reserve, 'second.json', cap=60)
                time.sleep(.05)
                first.finish({'http_status': 200, 'phase': 'response_complete'})
                with self.assertRaises(BatchBudgetError) as failure:
                    second.result(timeout=2)
            self.assertEqual(failure.exception.code, 'batch_budget_exceeded')
            self.assertEqual(current_stop(), 'max_output_tokens')
        self.assertEqual(snapshot(self.root)['summary']['output_tokens_accounted'], 60)

    def test_proven_unsent_attempt_refunds_request_slot_exactly_once(self):
        with batch_session(self.root, {'max_requests': 1}):
            ticket = self.reserve('cancelled.json')
            unsent = {'phase': 'not_sent', 'request_started': False}
            ticket.finish(unsent)
            ticket.finish(unsent)
            self.assertEqual(snapshot(self.root)['summary']['requests'], 0)
            self.reserve('real.json').finish(COMPLETE)
        counts = snapshot(self.root)['summary']
        self.assertEqual(counts['requests'], 1)
        self.assertEqual(counts['output_tokens_accounted'], 7)
        self.assertEqual(counts['pending_attempts'], 0)

    def test_connect_failure_refunds_capacity_but_preserves_provider_circuit(self):
        with batch_session(self.root, {'max_requests': 1, 'max_output_tokens': 20,
                                      'failure_threshold': 2, 'cooldown_seconds': 0}):
            ticket = self.reserve('connection_failed.json')
            transport = {'phase': 'connecting', 'request_started': False,
                         'request_sent': False, 'network_error_type': 'ConnectionRefusedError'}
            ticket.finish(transport)
            ticket.finish(transport)
            counts = snapshot(self.root)['summary']
            self.assertEqual(counts['requests'], 0)
            self.assertEqual(counts['output_tokens_accounted'], 0)
            self.assertEqual(counts['unknown_usage_attempts'], 0)
            self.assertEqual(counts['consecutive_failures'], 1)
            self.assertIsNone(current_stop())
            self.reserve('recovered.json').finish(COMPLETE)
        self.assertEqual(snapshot(self.root)['summary']['requests'], 1)

    def test_repeated_unsent_connection_failures_open_circuit_without_spending_budget(self):
        with batch_session(self.root, {'failure_threshold': 2, 'cooldown_seconds': 0}):
            for index in range(2):
                self.reserve(f'connection_{index}.json').finish({
                    'phase': 'connecting', 'request_started': False, 'request_sent': False,
                    'network_error_type': 'ConnectionRefusedError'})
            self.assertEqual(current_stop(), 'consecutive_provider_failures')
            with self.assertRaises(BatchBudgetError) as error:
                self.reserve('later.json')
            self.assertEqual(error.exception.code, 'batch_circuit_open')
        counts = snapshot(self.root)['summary']
        self.assertEqual(counts['requests'], 0)
        self.assertEqual(counts['output_tokens_accounted'], 0)
        self.assertEqual(counts['consecutive_failures'], 2)

    def test_ambiguous_connect_state_never_refunds_possible_sent_request(self):
        evidence = ({}, {'request_started': True}, {'request_sent': True},
                    {'request_sent': ''}, {'http_status': 200}, {'response_bytes': 1},
                    {'usage': {'total_tokens': 1}}, {'automatic_replay_blocked': True},
                    {'usage_invalid': True})
        for index, extra in enumerate(evidence):
            transport = {'phase': 'connecting', 'request_started': False, 'request_sent': False}
            if not extra:
                del transport['request_started']
            transport.update(extra)
            root = self.root/str(index)
            with self.subTest(transport=transport), batch_session(root, {'cooldown_seconds': 0}):
                reserve_request(root/'unknown.json', 20, CONFIG).finish(transport)
            counts = snapshot(root)['summary']
            self.assertEqual(counts['requests'], 1)
            self.assertEqual(counts['output_tokens_accounted'], 20)
            self.assertEqual(counts['unknown_usage_attempts'], 1)

    def test_api_connect_retry_succeeds_with_one_request_budget(self):
        import model_api
        from test_model_api import PNG, RESULT, SCHEMA, chat

        attempts = []
        def request(config, body, key, deadline, *, transport, stop_event):
            attempts.append(transport)
            if len(attempts) == 1:
                transport.update(phase='connecting', request_started=False, request_sent=False)
                raise ConnectionRefusedError('offline fixture')
            transport.update(phase='response_complete', request_started=True,
                             request_sent=True, http_status=200)
            return 200, 0, json.dumps(chat()).encode()

        config = {**CONFIG, 'model': 'fixture', 'api_key_env': '',
                  'max_output_tokens': 20, 'max_retries': 1}
        image = self.root/'fixture.png'
        image.write_bytes(PNG)
        with batch_session(self.root, {'max_requests': 1, 'max_output_tokens': 20,
                                      'failure_threshold': 2, 'cooldown_seconds': 0}), \
                scene_budget_session(self.root/'scene', {'max_requests': 1, 'max_output_tokens': 20}), \
                patch.object(model_api, '_request', side_effect=request), \
                patch.object(model_api.time, 'sleep'):
            result = model_api.call_model('offline fixture', [image], SCHEMA,
                                          self.root/'scene/analysis.json', config=config)
            self.assertEqual(result, RESULT)
            self.assertIsNone(current_stop())
        counts = snapshot(self.root)['summary']
        self.assertEqual(len(attempts), 2)
        self.assertEqual(counts['requests'], 1)
        self.assertEqual(counts['output_tokens_accounted'], 7)
        self.assertEqual(counts['unknown_usage_attempts'], 0)
        self.assertEqual(counts['pending_attempts'], 0)
        scene_ledger = json.loads((self.root/'scene/scene_budget.json').read_text())
        self.assertEqual(scene_ledger['requests'], 1)
        self.assertEqual(scene_ledger['output_tokens_accounted'], 7)

    def test_inflight_request_slot_waits_for_unsent_refund(self):
        with batch_session(self.root, {'max_requests': 1}):
            first = self.reserve('cancelled.json')
            with ThreadPoolExecutor(max_workers=1) as pool:
                second = pool.submit(self.reserve, 'real.json')
                time.sleep(.05)
                self.assertFalse(second.done())
                self.assertIsNone(current_stop())
                first.finish({'phase': 'not_sent', 'request_started': False})
                second.result(timeout=2).finish(COMPLETE)
        self.assertEqual(snapshot(self.root)['summary']['requests'], 1)

    def test_ambiguous_not_sent_records_never_refund_requests_or_output(self):
        invalid = [
            {'phase': 'not_sent'}, {'phase': 'not_sent', 'request_started': True},
            {'phase': 'not_sent', 'request_started': False, 'http_status': 200},
            {'phase': 'not_sent', 'request_started': False, 'response_bytes': 1},
            {'phase': 'not_sent', 'request_started': False, 'usage': {'total_tokens': 1}},
            {'phase': 'not_sent', 'request_started': False, 'request_sent': True},
            {'phase': 'not_sent', 'request_started': False, 'automatic_replay_blocked': True},
            {'phase': 'not_sent', 'request_started': False, 'usage_invalid': True},
            {'phase': 'not_sent', 'request_started': False, 'usage': ''},
            {'phase': 'not_sent', 'request_started': False, 'response_bytes': ''},
        ]
        for index, transport in enumerate(invalid):
            root = self.root/str(index)
            with self.subTest(transport=transport), batch_session(root, {'cooldown_seconds': 0}):
                reserve_request(root/'stage.json', 20, CONFIG).finish(transport)
            counts = snapshot(root)['summary']
            self.assertEqual(counts['requests'], 1)
            self.assertEqual(counts['output_tokens_accounted'], 20)
            self.assertEqual(counts['unknown_usage_attempts'], 1)

    def test_unsent_does_not_reset_provider_failure_streak(self):
        with batch_session(self.root, {'failure_threshold': 2, 'cooldown_seconds': 0}):
            self.reserve('first.json').finish({'http_status': 503})
            self.reserve('cancelled.json').finish({'phase': 'not_sent', 'request_started': False})
            self.assertEqual(snapshot(self.root)['summary']['consecutive_failures'], 1)
            self.reserve('second.json').finish({'http_status': 503})
            self.assertEqual(current_stop(), 'consecutive_provider_failures')
        self.assertEqual(snapshot(self.root)['summary']['requests'], 2)

    def test_settled_v1_ledger_migrates_without_scanning_attempts(self):
        with batch_session(self.root):
            self.reserve('first.json').finish(COMPLETE)
        with sqlite3.connect(self.root/LEDGER_NAME) as db:
            db.execute('ALTER TABLE state DROP COLUMN pending_output_tokens')
            db.execute('UPDATE state SET version=1')
        with batch_session(self.root):
            self.reserve('second.json').finish(COMPLETE)
        self.assertEqual(snapshot(self.root)['summary']['requests'], 2)

    def test_pending_v1_ledger_blocks_resume_before_migration(self):
        with batch_session(self.root):
            self.reserve('first.json')
        with sqlite3.connect(self.root/LEDGER_NAME) as db:
            db.execute('ALTER TABLE state DROP COLUMN pending_output_tokens')
            db.execute('UPDATE state SET version=1')
        with self.assertRaises(BatchBudgetError) as failure:
            with batch_session(self.root, resume=True):
                pass
        self.assertEqual(failure.exception.code, 'ambiguous_previous_request')
        self.assertEqual(snapshot(self.root)['summary']['pending_attempts'], 1)

    def test_legacy_shared_pending_without_owner_generation_stays_ambiguous(self):
        policy = {'max_requests': 3, 'cooldown_seconds': 0}
        with batch_session(self.root, policy):
            self.reserve('legacy_pending.json')
        # Recreate the v2 shared-ledger schema, whose pending attempts did not
        # record an invocation owner. Do not infer liveness from the shard lock
        # reacquired by this new coordinator.
        with sqlite3.connect(self.root/LEDGER_NAME) as db:
            db.execute('ALTER TABLE attempts DROP COLUMN owner_generation')
            db.execute('UPDATE state SET version=2')
        shard = self.root/'shard'
        shard.mkdir()
        from batch_multiview import process_lock
        with process_lock(shard/'.batch.lock'):
            with self.assertRaises(BatchBudgetError) as failure:
                with batch_session(self.root, policy, shared=True):
                    self.reserve('later.json')
        self.assertEqual(failure.exception.code, 'ambiguous_previous_request')
        counts = snapshot(self.root)['summary']
        self.assertEqual(counts['requests'], 1)
        self.assertEqual(counts['pending_attempts'], 1)

    def test_reported_tokens_stop_subsequent_calls_without_claiming_unknown_input_cap(self):
        with batch_session(self.root, {'max_reported_tokens': 10}):
            self.reserve().finish(COMPLETE)
            with self.assertRaises(BatchBudgetError) as failure:
                self.reserve('next.json')
            self.assertEqual(failure.exception.code, 'batch_budget_exceeded')
        self.assertEqual(snapshot(self.root)['summary']['reported_tokens'], 19)

    def test_pending_reservation_after_abrupt_exit_blocks_even_explicit_resume(self):
        code = '''
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from batch_guard import batch_session, reserve_request
root = Path(sys.argv[2])
with batch_session(root):
    reserve_request(root/'scene/stage.json', 20, {'provider':'fixture', 'base_url':'https://example.com'})
    os._exit(17)
'''
        process = subprocess.run([sys.executable, '-c', code,
            str(Path(__file__).parent), str(self.root)], capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 17, process.stderr.decode())
        self.assertEqual(snapshot(self.root)['summary']['pending_attempts'], 1)
        for explicit in (False, True):
            with self.subTest(explicit=explicit), self.assertRaises(BatchBudgetError) as failure:
                with batch_session(self.root, resume=explicit):
                    pass
            self.assertEqual(failure.exception.code, 'ambiguous_previous_request')
        self.assertEqual(snapshot(self.root)['summary']['output_tokens_accounted'], 20)

    def test_fatal_status_latches_and_explicit_resume_only_clears_service_stop(self):
        for status in (401, 402, 403):
            root = self.root/str(status)
            with batch_session(root) as budget:
                reserve_request(root/'scene/stage.json', 20, CONFIG).finish({'http_status': status})
                self.assertEqual(current_stop(), 'http_'+str(status))
                self.assertTrue(budget.stop_event.is_set())
            with batch_session(root):
                with self.assertRaises(BatchBudgetError) as failure:
                    reserve_request(root/'scene/next.json', 20, CONFIG)
                self.assertEqual(failure.exception.code, 'batch_circuit_open')
            with batch_session(root, resume=True):
                self.assertIsNone(current_stop())
                reserve_request(root/'scene/next.json', 20, CONFIG).finish(COMPLETE)
            counts = snapshot(root)['summary']
            self.assertEqual(counts['requests'], 2)
            self.assertEqual(counts['output_tokens_accounted'], 27)
            self.assertEqual(counts['unknown_usage_attempts'], 1)

    def test_service_failure_counts_are_independent_and_success_resets(self):
        policy = {'failure_threshold': 2, 'cooldown_seconds': 0}
        other = {**CONFIG, 'base_url': 'https://another.example/v1'}
        with batch_session(self.root, policy):
            self.reserve('one.json').finish({'http_status': 503})
            self.reserve('two.json', config=other).finish({'http_status': 429})
            self.assertIsNone(current_stop())
            self.reserve('three.json').finish(COMPLETE)
            self.reserve('four.json').finish({'http_status': 503})
            self.assertIsNone(current_stop())
            self.reserve('five.json').finish({'network_error_type': 'ConnectionResetError'})
            self.assertEqual(current_stop(), 'consecutive_provider_failures')

    def test_shared_cooldown_blocks_short_deadline_without_consuming_capacity(self):
        with batch_session(self.root, {'cooldown_seconds': 1}):
            self.reserve('first.json').finish({'http_status': 429, 'retry_after_seconds': 2})
            with self.assertRaises(BatchBudgetError) as failure:
                self.reserve('same_service.json', deadline=time.monotonic()+.02)
            self.assertEqual(failure.exception.code, 'timeout')
            self.assertEqual(snapshot(self.root)['summary']['requests'], 1)
            other = {**CONFIG, 'base_url': 'https://other.example/v1'}
            self.reserve('other_service.json', config=other).finish(COMPLETE)
            self.assertEqual(snapshot(self.root)['summary']['requests'], 2)

    def test_short_shared_cooldown_waits_and_then_allows_request(self):
        with batch_session(self.root, {'cooldown_seconds': .03}):
            self.reserve('first.json').finish({'http_status': 503})
            started = time.monotonic()
            self.reserve('second.json').finish(COMPLETE)
            self.assertGreaterEqual(time.monotonic()-started, .02)

    def test_provider_cooldown_longer_than_one_day_is_preserved_on_resume(self):
        policy = {'cooldown_seconds': 0}
        before = time.time()
        with batch_session(self.root, policy):
            self.reserve('first.json').finish({'http_status': 429, 'retry_after_seconds': 172800})
        self.assertGreaterEqual(snapshot(self.root)['summary']['cooldown_until'], before+172800)
        with batch_session(self.root, policy, resume=True):
            with self.assertRaises(BatchBudgetError) as failure:
                self.reserve('later.json', deadline=time.monotonic()+90000)
            self.assertEqual(failure.exception.code, 'timeout')
            self.assertEqual(snapshot(self.root)['summary']['requests'], 1)

    def test_missing_ledger_with_existing_audits_and_policy_changes_fail_closed(self):
        atomic_json(self.root/'scene/stage.api.json', {'attempts': 1})
        with self.assertRaises(BatchBudgetError):
            with batch_session(self.root):
                pass
        (self.root/'scene/stage.api.json').unlink()
        with batch_session(self.root, {'max_requests': 2}):
            self.reserve().finish(COMPLETE)
        with self.assertRaises(BatchBudgetError):
            with batch_session(self.root, {'max_requests': 3}):
                pass

    def test_scene_and_batch_limits_both_remain_accounted(self):
        scene = self.root/'scene'
        with batch_session(self.root), scene_budget_session(scene, {'max_requests': 2}):
            batch = self.reserve()
            local = scene_reserve(scene/'stage.json', 20)
            batch.finish(COMPLETE)
            local.finish(COMPLETE)
        self.assertEqual(snapshot(self.root)['summary']['requests'], 1)
        ledger = json.loads((scene/'scene_budget.json').read_text())
        self.assertEqual(ledger['requests'], 1)
        self.assertEqual(ledger['output_tokens_accounted'], 7)

    def test_disjoint_thread_scope_and_outside_requests(self):
        other_root = self.root/'other'
        first_root = self.root/'first'
        with batch_session(first_root), batch_session(other_root):
            self.assertIsNone(reserve_request(self.root/'outside.json', 20, CONFIG))
            reserve_request(first_root/'stage.json', 20, CONFIG).finish({'http_status': 401})
            self.assertEqual(current_stop(first_root), 'http_401')
            self.assertIsNone(current_stop(other_root))
            reserve_request(other_root/'stage.json', 20, CONFIG).finish(COMPLETE)

    def test_second_process_reserves_from_same_persisted_counters(self):
        policy = {'max_requests': 2}
        with batch_session(self.root, policy):
            self.reserve().finish(COMPLETE)
        code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from batch_guard import batch_session, reserve_request
root = Path(sys.argv[2])
with batch_session(root, {'max_requests':2}):
    reserve_request(root/'second.json', 20, {'provider':'openai_chat','base_url':'http://localhost:1234/v1'}).finish(
        {'http_status':200,'phase':'response_complete','usage':{'input_tokens':12,'output_tokens':7,'total_tokens':19}})
'''
        process = subprocess.run([sys.executable, '-c', code, str(Path(__file__).parent), str(self.root)],
                                 capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        with batch_session(self.root, policy):
            with self.assertRaises(BatchBudgetError):
                self.reserve('third.json')
        self.assertEqual(snapshot(self.root)['summary']['requests'], 2)

    def test_separate_output_shards_use_one_shared_budget_ledger(self):
        shared = self.root/'shared-ledger'
        shard_a = shared/'shard-a'
        shard_b = shared/'shard-b'
        policy = {'max_requests': 1, 'cooldown_seconds': 0}
        with batch_session(shared, policy):
            reserve_request(shard_a/'scene/stage.json', 20, CONFIG).finish(COMPLETE)

        # A separate process models the independent shard coordinator. It must
        # observe the same durable counter even though its output directory is
        # different from the first shard.
        code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from batch_guard import BatchBudgetError, batch_session, reserve_request
shared = Path(sys.argv[2])
output = Path(sys.argv[3])
try:
    with batch_session(shared, {'max_requests': 1, 'cooldown_seconds': 0}):
        reserve_request(output/'scene/stage.json', 20,
                        {'provider':'openai_chat','base_url':'http://localhost:1234/v1'})
except BatchBudgetError as exc:
    if exc.code != 'batch_budget_exceeded':
        raise
else:
    raise SystemExit('shared budget was not enforced')
'''
        process = subprocess.run([sys.executable, '-c', code,
                                  str(Path(__file__).parent), str(shared), str(shard_b)],
                                 capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        self.assertEqual(snapshot(shared)['summary']['requests'], 1)

    def test_live_pending_request_in_another_shard_does_not_block_new_process(self):
        shared = self.root/'shared-ledger'
        shard_a, shard_b = shared/'shard-a', shared/'shard-b'
        shard_a.mkdir(parents=True)
        shard_b.mkdir(parents=True)
        ready, release = self.root/'ready', self.root/'release'
        policy = {'max_requests': 10, 'cooldown_seconds': 0}
        code = '''
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from batch_guard import batch_session, reserve_request
from batch_multiview import process_lock
shared, output, ready, release = map(Path, sys.argv[2:])
with process_lock(output/'.batch.lock'):
    with batch_session(shared, {'max_requests':10,'cooldown_seconds':0}, shared=True):
        ticket = reserve_request(output/'scene/stage.json', 20,
            {'provider':'openai_chat','base_url':'http://localhost:1234/v1'})
        ready.write_text('pending')
        deadline = time.monotonic()+10
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        if not release.exists():
            raise SystemExit('test release timeout')
        ticket.finish({'http_status':200,'phase':'response_complete',
            'usage':{'input_tokens':12,'output_tokens':7,'total_tokens':19}})
'''
        process = subprocess.Popen([sys.executable, '-c', code,
            str(Path(__file__).parent), str(shared), str(shard_a), str(ready), str(release)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic()+5
        while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(.01)
        self.assertTrue(ready.exists(), 'first shard did not establish its pending request')
        try:
            # The first request is still pending, but its process holds the
            # correct output lock. The second shard must be allowed to join.
            with batch_session(shared, policy, shared=True):
                reserve_request(shard_b/'scene/stage.json', 20, CONFIG).finish(COMPLETE)
        finally:
            release.touch()
        stdout, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 0, (stdout+stderr).decode())
        self.assertEqual(snapshot(shared)['summary']['requests'], 2)

    def test_crashed_same_shard_generation_still_blocks_reacquired_lock(self):
        """A replacement may reacquire the shard lock, but not an old request owner."""
        from batch_multiview import process_lock
        shared = self.root/'shared-ledger'
        shard = shared/'shard-a'
        shard.mkdir(parents=True)
        code = '''
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from batch_guard import batch_session, reserve_request
from batch_multiview import process_lock
shared, shard = map(Path, sys.argv[2:])
with process_lock(shard/'.batch.lock'):
    with batch_session(shared, {'max_requests':10,'cooldown_seconds':0}, shared=True):
        reserve_request(shard/'scene/stage.json', 20,
            {'provider':'fixture','base_url':'https://example.com'})
        os._exit(17)
'''
        process = subprocess.run([sys.executable, '-c', code,
            str(Path(__file__).parent), str(shared), str(shard)],
            capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 17, process.stderr.decode())
        before = snapshot(shared)['summary']
        self.assertEqual(before['pending_attempts'], 1)
        # The kernel shard lock was released by the crash. Reacquiring it must
        # not prove that the old provider request is still in flight.
        with process_lock(shard/'.batch.lock'):
            with self.assertRaises(BatchBudgetError) as failure:
                with batch_session(shared, {'max_requests':10,'cooldown_seconds':0}, shared=True):
                    reserve_request(shard/'later.json', 20,
                        {'provider':'fixture','base_url':'https://example.com'})
        self.assertEqual(failure.exception.code, 'ambiguous_previous_request')
        after = snapshot(shared)['summary']
        self.assertEqual(after, before)  # No request, usage or reservation was lost.

    def test_ten_thousand_reservations_update_bounded_rows_and_store_each_attempt_once(self):
        started = time.monotonic()
        with batch_session(self.root):
            for index in range(10_000):
                self.reserve(f'scene_{index}/stage.json').finish(COMPLETE)
        counts = snapshot(self.root)['summary']
        self.assertEqual(counts['requests'], 10_000)
        self.assertEqual(counts['reported_tokens'], 190_000)
        self.assertEqual(counts['pending_attempts'], 0)
        with sqlite3.connect(self.root/LEDGER_NAME) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0], 10_000)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM state').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM services').fetchone()[0], 1)
        print(f'10,000 reserve/settle pairs: {time.monotonic()-started:.2f}s; '
              f'ledger={(self.root/LEDGER_NAME).stat().st_size:,} bytes')


if __name__ == '__main__':
    unittest.main()
