"""Offline durable scene-budget limits, accounting and concurrent reservations."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import scene_budget as budget


def complete(usage, **kwargs):
    return {'phase': 'response_complete', 'http_status': 200, 'usage': usage, **kwargs}


class SceneBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / 'scene'

    def reserve(self, cap=100, name='plan.json'):
        return budget.reserve_request(self.root / name, cap)

    def assert_code(self, expected, action):
        with self.assertRaises(budget.BudgetError) as error:
            action()
        self.assertEqual(error.exception.code, expected)

    def ledger(self):
        return json.loads((self.root / budget.LEDGER_NAME).read_text())

    def test_strict_policy_and_disabled_default(self):
        expected = {'max_requests': 0, 'max_output_tokens': 0, 'max_reported_tokens': 0}
        self.assertEqual(budget.normalize_policy(), expected)
        self.assertEqual(budget.normalize_policy({}), expected)
        for value in (False, [], '', {'unknown': 2}, {'max_requests': True},
                      {'max_output_tokens': 1.5}, {'max_reported_tokens': -1},
                      {'max_requests': '3'}, {'max_requests': None}):
            with self.subTest(value=value):
                self.assert_code('invalid_scene_budget', lambda: budget.normalize_policy(value))
        with budget.scene_budget_session(self.root) as session:
            self.assertIsNone(self.reserve())
            self.assertIsNone(session.snapshot())
        self.assertFalse(self.root.exists())
        self.assertIsNone(budget.reserve_request(self.root / 'out.json', 100))

    def test_full_cap_is_reserved_and_not_reduced(self):
        with budget.scene_budget_session(self.root, {'max_output_tokens': 100}) as session:
            self.assert_code('scene_budget_exceeded', lambda: self.reserve(101))
            self.assertEqual(session.snapshot()['requests'], 0)
            ticket = self.reserve(100)
            self.assertRegex(ticket.request_id, r'^[0-9a-f]{32}$')
            saved = session.snapshot()
            self.assertEqual(saved['output_tokens_accounted'], 100)
            self.assertEqual(saved['unknown_usage_attempts'], 1)
            self.assertEqual(saved['pending_attempts'], 1)
            self.assert_code('scene_budget_exceeded', lambda: self.reserve(1))
            ticket.finish(complete({'prompt_tokens': 100, 'completion_tokens': 25, 'total_tokens': 125}))
            self.assertEqual(session.snapshot()['output_tokens_accounted'], 25)
            self.assertEqual(session.snapshot()['reported_tokens'], 125)
            self.assertEqual(session.snapshot()['unknown_usage_attempts'], 0)
            self.assert_code('scene_budget_exceeded', lambda: self.reserve(76))
            self.reserve(75)
            self.assertEqual(session.snapshot()['requests'], 2)

    def test_all_network_attempts_count_even_before_send_or_rejection(self):
        with budget.scene_budget_session(self.root, {'max_requests': 3}) as session:
            transports = ({'phase': 'connecting', 'request_started': False},
                          {'phase': 'response_complete', 'http_status': 429},
                          {'phase': 'waiting_response', 'request_started': True,
                           'automatic_replay_blocked': True})
            for transport in transports:
                self.reserve().finish(transport)
            data = session.snapshot()
            self.assertEqual(data['requests'], 3)
            self.assertEqual(data['output_tokens_accounted'], 300)
            self.assertEqual(data['unknown_usage_attempts'], 3)
            self.assertEqual(data['pending_attempts'], 0)
            self.assert_code('scene_budget_exceeded', self.reserve)

    def test_complete_provider_usage_and_idempotent_finish(self):
        samples = [({'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}, 10, 3, 13),
                   ({'input_tokens': 11, 'output_tokens': 4, 'total_tokens': 15}, 11, 4, 15),
                   ({'input_tokens': 2, 'cache_creation_input_tokens': 13,
                     'cache_read_input_tokens': 17, 'output_tokens': 5}, 32, 5, 37),
                   ({'input_tokens': 0, 'output_tokens': 0}, 0, 0, 0),
                   ({'prompt_tokens': 7, 'input_tokens': 7, 'completion_tokens': 9,
                     'output_tokens': 9, 'total_tokens': 16}, 7, 9, 16)]
        with budget.scene_budget_session(self.root, {'max_requests': 10}) as session:
            for usage, input_count, output_count, total in samples:
                ticket = self.reserve()
                ticket.finish(complete(usage))
                before = (self.root / budget.LEDGER_NAME).read_bytes()
                ticket.finish({})
                self.assertEqual((self.root / budget.LEDGER_NAME).read_bytes(), before)
                entry = session.snapshot()['attempts'][-1]
                self.assertEqual(entry['reported_input_tokens'], input_count)
                self.assertEqual(entry['reported_output_tokens'], output_count)
                self.assertEqual(entry['reported_tokens'], total)
                self.assertTrue(entry['usage_complete'])
            self.assertEqual(session.snapshot()['reported_tokens'], 81)
            self.assertEqual(session.snapshot()['output_tokens_accounted'], 21)

    def test_partial_invalid_inconsistent_or_interrupted_usage_keeps_reservation(self):
        transports = [complete({}), complete({'completion_tokens': 3}),
                      complete({'prompt_tokens': 8}), complete({'total_tokens': 19}),
                      complete({'input_tokens': 2, 'output_tokens': True}),
                      complete({'input_tokens': 2, 'output_tokens': 3, 'total_tokens': 6}),
                      complete({'input_tokens': 2, 'prompt_tokens': 4, 'output_tokens': 3}),
                      complete({'input_tokens': 2, 'output_tokens': 3}, completion_marker_missing=True),
                      complete({'input_tokens': 2, 'output_tokens': 3}, usage_invalid=True),
                      complete({'input_tokens': 2, 'output_tokens': 3}, automatic_replay_blocked=True),
                      {'phase': 'receiving_response', 'http_status': 200,
                       'usage': {'input_tokens': 2, 'output_tokens': 3}},
                      complete({'input_tokens': 2, 'output_tokens': 3}, http_status=500),
                      complete({'cache_creation_input_tokens': 13, 'cache_read_input_tokens': 17})]
        with budget.scene_budget_session(self.root, {'max_requests': 20}) as session:
            for transport in transports:
                self.reserve().finish(transport)
                self.assertFalse(session.snapshot()['attempts'][-1]['usage_complete'])
            data = session.snapshot()
            self.assertEqual(data['output_tokens_accounted'], 100 * len(transports))
            self.assertEqual(data['unknown_usage_attempts'], len(transports))
            self.assertGreater(data['reported_tokens'], 0)

    def test_actual_over_cap_counts_in_full_and_prevents_next_request(self):
        with budget.scene_budget_session(self.root, {'max_output_tokens': 100}) as session:
            self.reserve().finish(complete({'input_tokens': 1, 'output_tokens': 125}))
            self.assertEqual(session.snapshot()['output_tokens_accounted'], 125)
            self.assert_code('scene_budget_exceeded', lambda: self.reserve(1))
        root2 = self.root.parent / 'partial'
        with budget.scene_budget_session(root2, {'max_output_tokens': 100}) as session:
            budget.reserve_request(root2 / 'plan.json', 100).finish({'usage': {'output_tokens': 125}})
            self.assertEqual(session.snapshot()['output_tokens_accounted'], 125)

    def test_reported_threshold_stops_next_request_not_in_flight(self):
        with budget.scene_budget_session(self.root, {'max_reported_tokens': 50}) as session:
            first = self.reserve()
            second = self.reserve()
            first.finish(complete({'input_tokens': 45, 'output_tokens': 5}))
            self.assert_code('scene_budget_exceeded', self.reserve)
            second.finish(complete({'input_tokens': 60, 'output_tokens': 10}))
            self.assertEqual(session.snapshot()['reported_tokens'], 120)
            self.assertEqual(session.snapshot()['requests'], 2)

    def test_partial_reported_total_reaches_threshold_without_releasing_output(self):
        with budget.scene_budget_session(self.root, {'max_reported_tokens': 50}) as session:
            self.reserve().finish({'usage': {'total_tokens': 50}})
            self.assert_code('scene_budget_exceeded', self.reserve)
            self.assertEqual(session.snapshot()['unknown_usage_attempts'], 1)
            self.assertEqual(session.snapshot()['output_tokens_accounted'], 100)

    def test_restart_preserves_pending_and_settled_reservations(self):
        policy = {'max_requests': 2, 'max_output_tokens': 101}
        with budget.scene_budget_session(self.root, policy):
            self.reserve().finish(complete({'input_tokens': 2, 'output_tokens': 1}))
            self.reserve()
        snapshot = budget.budget_snapshot(self.root)['summary']
        self.assertEqual(snapshot['requests'], 2)
        self.assertEqual(snapshot['pending_attempts'], 1)
        self.assertEqual(snapshot['output_tokens_accounted'], 101)
        self.assert_code('ambiguous_previous_request', lambda: budget.scene_budget_session(self.root, policy).__enter__())

    def test_settled_restart_reuses_ledger_and_same_active_session_allows_pending(self):
        policy = {'max_requests': 3}
        with budget.scene_budget_session(self.root, policy) as session:
            first = self.reserve()
            with budget.scene_budget_session(self.root, policy) as nested:
                self.assertIs(nested, session)
                second = self.reserve()
            first.finish(complete({'input_tokens': 1, 'output_tokens': 1}))
            second.finish(complete({'input_tokens': 1, 'output_tokens': 1}))
        with budget.scene_budget_session(self.root, policy) as session:
            self.assertEqual(session.snapshot()['requests'], 2)
            self.reserve()

    def test_root_resolution_shared_threads_and_disjoint_scenes(self):
        other = self.root.parent / 'scene_other'
        with budget.scene_budget_session(self.root, {'max_requests': 1}) as first, \
                budget.scene_budget_session(other, {'max_requests': 2}) as second:
            with ThreadPoolExecutor(max_workers=2) as executor:
                ticket = executor.submit(budget.reserve_request, self.root / 'details' / 'a.json', 20).result()
            self.assertIsNotNone(ticket)
            with budget.scene_budget_session(self.root / '.' / 'subdir' / '..', {'max_requests': 1}) as nested:
                self.assertIs(first, nested)
                self.assert_code('scene_budget_exceeded', self.reserve)
            budget.reserve_request(other / 'a.json', 20)
            self.assertEqual(first.snapshot()['requests'], 1)
            self.assertEqual(second.snapshot()['requests'], 1)
            self.assertIsNone(budget.reserve_request(self.root.parent / 'scene_extra' / 'a.json', 20))
            alias = self.root.parent / 'alias'
            alias.symlink_to(self.root, target_is_directory=True)
            self.assert_code('scene_budget_exceeded', lambda: budget.reserve_request(alias / 'a.json', 20))

    def test_no_concurrent_oversubscription(self):
        with budget.scene_budget_session(self.root, {'max_requests': 7, 'max_output_tokens': 700}) as session:
            def reserve_one(index):
                try:
                    return self.reserve(name=f'details/{index}.json')
                except budget.BudgetError as exc:
                    self.assertEqual(exc.code, 'scene_budget_exceeded')
                    return None
            with ThreadPoolExecutor(max_workers=20) as executor:
                tickets = list(executor.map(reserve_one, range(40)))
            self.assertEqual(sum(ticket is not None for ticket in tickets), 7)
            self.assertEqual(session.snapshot()['requests'], 7)
            self.assertEqual(session.snapshot()['output_tokens_accounted'], 700)

    def test_changed_or_overlapping_sessions_rejected(self):
        with budget.scene_budget_session(self.root, {'max_requests': 1}):
            for root, policy in ((self.root, {'max_requests': 2}), (self.root, {}),
                                 (self.root / 'details', {}), (self.root.parent, {})):
                with self.subTest(root=root, policy=policy):
                    self.assert_code('invalid_scene_budget', lambda: budget.scene_budget_session(root, policy).__enter__())
        for policy in ({}, {'max_requests': 2}, {'max_requests': 1, 'max_output_tokens': 1}):
            self.assert_code('invalid_scene_budget', lambda: budget.scene_budget_session(self.root, policy).__enter__())

    def test_deleted_corrupt_or_changed_schema_ledger_never_resets(self):
        with budget.scene_budget_session(self.root, {'max_requests': 1}):
            self.reserve()
        path = self.root / budget.LEDGER_NAME
        original = path.read_bytes()
        for replacement in (b'{"private":"SECRET"', b'[]', b'{"schema_version":1,"schema_version":1}'):
            path.write_bytes(replacement)
            with self.assertRaises(budget.BudgetError) as error:
                with budget.scene_budget_session(self.root, {'max_requests': 1}):
                    pass
            self.assertEqual(error.exception.code, 'invalid_scene_budget')
            self.assertNotIn('SECRET', str(error.exception))
            self.assertEqual(path.read_bytes(), replacement)
        value = json.loads(original)
        value['schema_version'] = 2
        path.write_text(json.dumps(value))
        self.assert_code('invalid_scene_budget', lambda: budget.scene_budget_session(self.root, {'max_requests': 1}).__enter__())
        path.unlink()
        self.assert_code('invalid_scene_budget', lambda: budget.scene_budget_session(self.root, {'max_requests': 1}).__enter__())

    def test_audit_attempts_without_ledger_refuse_initialization(self):
        for index, record in enumerate(({'attempts': 1}, {'transport_attempts': [{'attempt': 1}]},
                                        {'attempts': -1}, {'attempts': True}, {'transport_attempts': {}})):
            root = self.root / str(index)
            root.mkdir(parents=True)
            (root / 'plan.api.json').write_text(json.dumps(record))
            self.assert_code('invalid_scene_budget', lambda: budget.scene_budget_session(root, {'max_requests': 1}).__enter__())
            self.assertFalse((root / budget.LEDGER_NAME).exists())
        clean = self.root / 'no-network'
        clean.mkdir()
        (clean / 'plan.api.json').write_text(json.dumps({'attempts': 0, 'transport_attempts': []}))
        with budget.scene_budget_session(clean, {'max_requests': 1}) as session:
            self.assertEqual(session.snapshot()['requests'], 0)

    def test_readonly_snapshot_without_session_or_writes(self):
        self.assertIsNone(budget.budget_snapshot(self.root))
        self.assertFalse(self.root.exists())
        with budget.scene_budget_session(self.root, {'max_requests': 2}):
            self.reserve().finish(complete({'input_tokens': 9, 'output_tokens': 3}))
        before = {str(path): path.read_bytes() for path in self.root.iterdir()}
        snapshot = budget.budget_snapshot(self.root)
        self.assertEqual(snapshot['policy']['max_requests'], 2)
        self.assertEqual(snapshot['summary'], {'requests': 1, 'output_tokens_accounted': 3,
            'reported_tokens': 12, 'unknown_usage_attempts': 0, 'pending_attempts': 0})
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.root.iterdir()})
        (self.root / budget.LEDGER_NAME).write_text('{invalid')
        self.assert_code('invalid_scene_budget', lambda: budget.budget_snapshot(self.root))

    def test_invalid_cap_and_unavailable_lock_fail_before_reservation(self):
        with budget.scene_budget_session(self.root, {'max_requests': 1}) as session:
            for cap in (True, 0, -1, 1.0, '1', None):
                self.assert_code('invalid_scene_budget', lambda: self.reserve(cap))
            self.assertEqual(session.snapshot()['requests'], 0)
        with patch.object(budget, 'fcntl', None):
            self.assert_code('invalid_scene_budget', lambda: budget.scene_budget_session(self.root, {'max_requests': 1}).__enter__())

    def test_failed_durable_write_prevents_ticket(self):
        with budget.scene_budget_session(self.root, {'max_requests': 1}) as session:
            with patch.object(budget.os, 'replace', side_effect=OSError('PRIVATE')):
                self.assert_code('invalid_scene_budget', self.reserve)
            self.assertEqual(session.snapshot()['requests'], 0)
            self.assertFalse(list(self.root.glob('*.tmp')))

    def test_process_crash_leaves_persisted_reservation(self):
        code = '''import os, sys
from pathlib import Path
from scene_budget import scene_budget_session, reserve_request
root = Path(sys.argv[1])
with scene_budget_session(root, {'max_requests': 1}):
    reserve_request(root / 'plan.json', 100)
    os._exit(17)
'''
        result = subprocess.run([sys.executable, '-c', code, str(self.root)],
            env=dict(os.environ, PYTHONPATH=str(Path(budget.__file__).parent)),
            capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(budget.budget_snapshot(self.root)['summary']['pending_attempts'], 1)
        self.assertEqual(budget.budget_snapshot(self.root)['summary']['output_tokens_accounted'], 100)
        self.assert_code('ambiguous_previous_request', lambda: budget.scene_budget_session(self.root, {'max_requests': 1}).__enter__())

    def test_invalid_ledger_totals_and_usage_fail_closed(self):
        with budget.scene_budget_session(self.root, {'max_requests': 2}):
            self.reserve().finish(complete({'input_tokens': 9, 'output_tokens': 3}))
        original = self.ledger()
        path = self.root / budget.LEDGER_NAME
        mutations = [lambda data: data.update(requests=0),
                     lambda data: data.update(requests=True),
                     lambda data: data['policy'].update(max_requests=True),
                     lambda data: data['attempts'][0].update(reported_tokens=0),
                     lambda data: data['attempts'][0].update(reported_output_tokens=None),
                     lambda data: data['attempts'][0].update(state='reserved'),
                     lambda data: data['attempts'][0].update(output='../escape.json')]
        for mutate in mutations:
            value = json.loads(json.dumps(original))
            mutate(value)
            path.write_text(json.dumps(value))
            self.assert_code('invalid_scene_budget', lambda: budget.budget_snapshot(self.root))

    def test_snapshot_does_not_recreate_missing_lock(self):
        with budget.scene_budget_session(self.root, {'max_requests': 1}):
            self.reserve()
        lock = self.root / budget.LOCK_NAME
        lock.unlink()
        self.assertEqual(budget.budget_snapshot(self.root)['summary']['requests'], 1)
        self.assertFalse(lock.exists())

    def test_subprocess_restart_and_cross_process_atomic_limits(self):
        code = '''import sys, time
from pathlib import Path
from scene_budget import scene_budget_session, reserve_request, BudgetError
root = Path(sys.argv[1])
with scene_budget_session(root, {'max_requests': 9, 'max_output_tokens': 900}):
    (root.parent / ('ready_' + sys.argv[2])).write_text('ready')
    deadline = time.monotonic() + 10
    while len(list(root.parent.glob('ready_*'))) < 3:
        assert time.monotonic() < deadline
        time.sleep(.01)
    for i in range(10):
        try:
            reserve_request(root / ('child_' + str(i) + '.json'), 100)
        except BudgetError as exc:
            assert exc.code == 'scene_budget_exceeded'
'''
        env = dict(os.environ, PYTHONPATH=str(Path(budget.__file__).parent))
        processes = [subprocess.Popen([sys.executable, '-c', code, str(self.root), str(index)],
                                     env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for index in range(3)]
        for process in processes:
            output, errors = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, errors)
        snapshot = budget.budget_snapshot(self.root)['summary']
        self.assertEqual(snapshot['requests'], 9)
        self.assertEqual(snapshot['pending_attempts'], 9)
        self.assertEqual(snapshot['output_tokens_accounted'], 900)
        self.assert_code('ambiguous_previous_request', lambda: budget.scene_budget_session(
            self.root, {'max_requests': 9, 'max_output_tokens': 900}).__enter__())


if __name__ == '__main__':
    unittest.main()
