"""Offline accounting must expose unknowns instead of inventing saved tokens."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scene_factory.diagnostics import usage_report
from batch import atomic_json


class UsageStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def record(self, relative, **changes):
        value = {'config': {'provider': 'fixture', 'model': 'test'},
                 'status': 'completed', 'attempts': 1,
                 'usage': {'prompt_tokens': 6, 'completion_tokens': 4, 'total_tokens': 10},
                 'prompt_sha256': 'a' * 64, 'elapsed_seconds': 2.5}
        value.update(changes)
        path = self.root / relative
        atomic_json(path, value)
        return path

    def validation(self, path, **changes):
        value = {'attempt': 1, 'actual_prompt_sha256': 'a' * 64, 'status': 'failed'}
        value.update(changes)
        atomic_json(path.with_name(path.name.replace('.api.json', '.validation.json')), value)

    def stages(self):
        return {row['stage']: row for row in usage_report(self.root)['stages']}

    def test_roles_are_exact_output_contracts_and_unknowns_survive(self):
        cases = {
            'scene/initial_bundle_stages/plan_attempt_00.api.json': 'analysis.plan',
            'scene/initial_bundle_stages/detail_table_attempt_02.api.json': 'analysis.detail',
            'scene/initial_bundle.api.json': 'analysis.monolithic',
            'scene/iteration_00/review_retry_2.api.json': 'review',
            'scene/iteration_00/repair_from_best/repair_attempt_01.api.json': 'repair.visual',
            'scene/iteration_01/technical_repair_attempt_00.api.json': 'repair.technical',
            'probe/response.api.json': 'unknown',
            'probe/plan_attempt_00.api.json': 'unknown',
            'random/repair_attempt_00.api.json': 'unknown',
            'other/initial_bundle.api.json': 'unknown',
        }
        atomic_json(self.root / 'scene/job.json', {})
        for name in cases:
            self.record(name)
        result = usage_report(self.root)
        stages = {row['stage']: row for row in result['stages']}
        self.assertEqual(stages['unknown']['logical_calls'], 4)
        self.assertEqual(sum(row['logical_calls'] for row in stages.values()), len(cases))
        self.assertEqual(sum(row['tokens']['total_tokens'] for row in stages.values()), 100)
        self.assertEqual(result['models'][0]['tokens']['total_tokens'], 100)
        self.assertEqual(result['models'][0]['stages'], result['stages'])
        self.assertIn('CLI', result['scope'])
        self.assertIn('not batch wall time', result['elapsed_seconds_semantics'])

    def test_monolithic_retry_attempts_share_stage_and_sum_transport_usage(self):
        atomic_json(self.root / 'scene' / 'job.json', {})
        self.record('scene/initial_bundle.api.json', attempts=1,
                    transport_attempts=[{'phase': 'response_complete', 'http_status': 200,
                                         'usage': {'total_tokens': 3}}])
        self.record('scene/initial_bundle_retry_1.api.json', attempts=2,
                    transport_attempts=[{'phase': 'response_complete', 'http_status': 200,
                                         'usage': {'total_tokens': 4}},
                                        {'phase': 'response_complete', 'http_status': 200,
                                         'usage': {'total_tokens': 5}}])
        stage = self.stages()['analysis.monolithic']
        self.assertEqual(stage['logical_calls'], 2)
        self.assertEqual(stage['transport_attempts'], 3)
        self.assertEqual(stage['attempts_with_reported_usage'], 3)
        self.assertEqual(stage['tokens']['total_tokens'], 12)

    def test_unsent_connecting_and_reservation_cancellation_are_not_requests(self):
        self.record('connection.api.json', status='failed', usage={}, attempts=2,
                    transport_attempts=[{'phase': 'connecting', 'request_started': False},
                                        {'phase': 'not_sent', 'request_started': False}])
        # Reservation failed before incrementing the record's attempt counter.
        self.record('cancel.api.json', status='failed', usage={}, attempts=0,
                    transport_attempts=[{'phase': 'not_sent', 'request_started': False,
                                         'reservation_cancelled': True}])
        result = usage_report(self.root)
        self.assertEqual(result['logical_calls'], 2)
        self.assertEqual(result['transport_attempts'], 0)
        self.assertEqual(result['proven_unsent_attempts'], 3)
        self.assertEqual(result['attempts_without_reported_usage'], 0)
        self.assertEqual(result['models'][0]['tokens'], {})

    def test_contradictory_unsent_and_incomplete_legacy_attempts_remain_counted(self):
        self.record('mixed.api.json', status='failed', attempts=4, usage={'total_tokens': 999},
                    transport_attempts=[{'phase': 'not_sent', 'request_started': False},
                        {'phase': 'not_sent', 'request_started': False, 'http_status': 200,
                         'usage': {'total_tokens': 7}}])
        result = usage_report(self.root)
        model = result['models'][0]
        self.assertEqual(model['transport_attempts'], 3)
        self.assertEqual(model['proven_unsent_attempts'], 1)
        self.assertEqual(model['attempts_without_reported_usage'], 2)
        self.assertEqual(model['tokens'], {'total_tokens': 7})
        self.assertEqual(model['failed_call_tokens'], {'total_tokens': 7})

    def test_legacy_top_usage_is_not_copied_to_each_attempt_or_added_to_transport(self):
        self.record('legacy.api.json', attempts=3)
        self.record('modern.api.json', attempts=2, transport_attempts=[
            {'usage': {'total_tokens': 3}}, {'usage': {'total_tokens': 10}}])
        model = usage_report(self.root)['models'][0]
        self.assertEqual(model['transport_attempts'], 5)
        self.assertEqual(model['attempts_with_reported_usage'], 3)
        self.assertEqual(model['attempts_without_reported_usage'], 2)
        self.assertEqual(model['tokens']['total_tokens'], 23)

    def test_bound_failed_validation_counts_completed_api_cost_separately(self):
        path = self.record('s/initial_bundle_stages/plan_attempt_00.api.json')
        self.validation(path)
        row = self.stages()['analysis.plan']
        self.assertEqual(row['failed_calls'], 0)
        self.assertEqual(row['failed_call_tokens'], {})
        self.assertEqual(row['local_validation']['rejected_calls'], 1)
        self.assertEqual(row['local_validation']['rejected_tokens']['total_tokens'], 10)

    def test_stale_unbound_malformed_and_api_failed_validation_are_unknown(self):
        for index, changes in enumerate([{'actual_prompt_sha256': 'b' * 64},
                                         {'attempt': 7}, {'attempt': True}]):
            path = self.record(f's{index}/initial_bundle_stages/plan_attempt_00.api.json')
            self.validation(path, **changes)
        path = self.record('bad/initial_bundle_stages/plan_attempt_00.api.json')
        self.validation(path, status='completed', artifacts={'api': 'wrong'})
        path = self.record('api/initial_bundle_stages/plan_attempt_00.api.json',
                           status='failed', failure_code='schema_mismatch')
        self.validation(path)
        row = self.stages()['analysis.plan']
        self.assertEqual(row['local_validation']['unknown_calls'], 5)
        self.assertEqual(row['local_validation']['rejected_tokens'], {})
        self.assertEqual(row['failed_call_tokens']['total_tokens'], 10)

    def test_successful_validation_requires_api_hash_binding_and_symlinks_are_ignored(self):
        path = self.record('s/initial_bundle_stages/detail_desk_attempt_00.api.json')
        self.validation(path, status='completed', artifacts={
            'api': hashlib.sha256(path.read_bytes()).hexdigest()})
        row = self.stages()['analysis.detail']
        self.assertEqual(row['local_validation']['validated_calls'], 1)
        state = path.with_name(path.name.replace('.api.json', '.validation.json'))
        other = self.root / 'other.json'
        state.rename(other)
        state.symlink_to(other)
        row = self.stages()['analysis.detail']
        self.assertEqual(row['local_validation']['unknown_calls'], 1)

    def test_review_rejection_and_repair_attempt_status_are_not_assumed_invalid_calls(self):
        self.record('s/iteration_00/review.api.json')
        atomic_json(self.root / 's/iteration_00/review.json', {'verdict': 'reject'})
        self.record('s/iteration_00/repair_attempt_00.api.json')
        atomic_json(self.root / 's/iteration_00/repair_attempts.json', {
            'attempts': [{'attempt': 0, 'status': 'rejected'}]})
        for row in self.stages().values():
            self.assertEqual(row['local_validation']['unknown_calls'], 1)
            self.assertEqual(row['local_validation']['rejected_tokens'], {})

    def test_bad_records_do_not_partially_mutate_totals_and_invalid_usage_is_missing(self):
        self.record('broken.api.json', attempts=-1)
        self.record('bad_attempt.api.json', transport_attempts=['invalid'])
        self.record('invalid_usage.api.json', usage={'total_tokens': True, 'completion_tokens': -1})
        result = usage_report(self.root)
        self.assertEqual(result['unreadable_records'], 2)
        self.assertEqual(result['logical_calls'], 1)
        self.assertEqual(result['attempts_without_reported_usage'], 1)
        self.assertEqual(result['models'][0]['tokens'], {})

    def test_one_directory_walk_discovers_api_records_and_sorted_budgets(self):
        self.record('z/iteration_00/review.api.json')
        self.record('a/initial_bundle_stages/plan_attempt_00.api.json')
        for scene in ('z', 'a'):
            atomic_json(self.root / scene / 'scene_budget.json', {})
        actual_walk, actual_scandir = os.walk, os.scandir
        snapshot = {'policy': {}, 'summary': {}}
        with patch('scene_factory.diagnostics.os.walk', wraps=actual_walk) as walked, \
                patch('scene_factory.diagnostics.os.scandir', wraps=actual_scandir) as scanned, \
                patch.object(Path, 'rglob', side_effect=AssertionError('No second recursive scan')), \
                patch('scene_budget.budget_snapshot', return_value=snapshot) as budgets:
            result = usage_report(self.root)
        walked.assert_called_once_with(self.root.resolve(), followlinks=False)
        self.assertEqual(len(scanned.call_args_list), 5)
        self.assertEqual(len({str(call.args[0]) for call in scanned.call_args_list}), 5)
        self.assertEqual(result['logical_calls'], 2)
        self.assertEqual(result['models'][0]['tokens']['total_tokens'], 20)
        self.assertEqual([row['scene'] for row in result['scene_budgets']], ['a', 'z'])
        self.assertEqual([call.args[0] for call in budgets.call_args_list],
                         [self.root.resolve() / 'a', self.root.resolve() / 'z'])

    def test_usage_records_are_consumed_before_next_path_is_discovered(self):
        events = []
        first = self.root / 'first' / 'call.api.json'
        second = self.root / 'second' / 'call.api.json'
        records = {first: {'config': {'provider': 'fixture', 'model': 'first'},
                           'status': 'completed', 'attempts': 1,
                           'usage': {'total_tokens': 1}},
                   second: {'config': {'provider': 'fixture', 'model': 'second'},
                            'status': 'completed', 'attempts': 1,
                            'usage': {'total_tokens': 2}}}

        def walk(*args, **kwargs):
            events.append('discover_first')
            yield str(first.parent), [], [first.name]
            # An eager path list would advance here before accounting for
            # the first record, retaining the whole batch in memory.
            self.assertIn('read_first', events)
            events.append('discover_second')
            yield str(second.parent), [], [second.name]

        def read(path):
            path = Path(path)
            events.append('read_' + path.parent.name)
            return records[path]

        with patch('scene_factory.diagnostics.os.walk', side_effect=walk), \
                patch('scene_factory.diagnostics.read_json', side_effect=read):
            result = usage_report(self.root)
        self.assertEqual(events, ['discover_first', 'read_first', 'discover_second', 'read_second'])
        self.assertEqual(result['logical_calls'], 2)
        self.assertEqual(result['models'][0]['tokens']['total_tokens'], 1)
        self.assertEqual(result['models'][1]['tokens']['total_tokens'], 2)

    def test_payload_metrics_are_reported_as_bytes_without_token_inference(self):
        atomic_json(self.root / 's' / 'initial_bundle.api.json', {
            'config': {'provider': 'fixture', 'model': 'metrics'},
            'status': 'completed', 'attempts': 2,
            'request_metrics': {
                'serialized_request_body_utf8_bytes': 1200,
                'images': [{'source_bytes': 10, 'base64_bytes': 16},
                           {'source_bytes': 20, 'base64_bytes': 28}]},
            'output_text_utf8_bytes': 42,
            'transport_attempts': [
                {'request_metrics': {'serialized_request_body_utf8_bytes': 1200},
                 'response_bytes': 300},
                {'request_metrics': {'serialized_request_body_utf8_bytes': 1200},
                 'response_bytes': 301}],
        })
        result = usage_report(self.root)
        metrics = result['payload_metrics']
        self.assertEqual(metrics['calls_with_metrics'], 1)
        self.assertEqual(metrics['request_body_utf8_bytes'], 1200)
        self.assertEqual(metrics['attempt_request_body_utf8_bytes'], 2400)
        self.assertEqual(metrics['input_image_count'], 2)
        self.assertEqual(metrics['input_image_source_bytes'], 30)
        self.assertEqual(metrics['input_image_base64_bytes'], 44)
        self.assertEqual(metrics['response_bytes'], 601)
        self.assertEqual(metrics['output_text_utf8_bytes'], 42)
        self.assertIn('not provider token estimates', result['payload_metrics_semantics'])
        self.assertEqual(result['models'][0]['tokens'], {})

    def test_directory_links_are_not_traversed_and_record_link_rules_survive(self):
        self.record('local/iteration_00/review.api.json')
        with tempfile.TemporaryDirectory() as other:
            external = Path(other)
            atomic_json(external / 'extra.api.json', {
                'config': {'provider': 'fixture', 'model': 'external'},
                'status': 'completed', 'attempts': 1, 'usage': {'total_tokens': 999}})
            atomic_json(external / 'scene_budget.json', {})
            (self.root / 'linked_directory').symlink_to(external, target_is_directory=True)
            (self.root / 'linked.api.json').symlink_to(external / 'extra.api.json')
            (self.root / 'scene_budget.json').symlink_to(external / 'scene_budget.json')
            actual_scandir = os.scandir
            with patch('scene_factory.diagnostics.os.scandir', wraps=actual_scandir) as scanned, \
                    patch('scene_budget.budget_snapshot') as budgets:
                result = usage_report(self.root)
            scanned_paths = {Path(call.args[0]) for call in scanned.call_args_list}
            self.assertEqual(len(scanned_paths), 3)
            self.assertNotIn(external.resolve(), scanned_paths)
            self.assertNotIn(self.root.resolve() / 'linked_directory', scanned_paths)
            budgets.assert_not_called()
        self.assertEqual(result['logical_calls'], 1)
        self.assertEqual(result['models'][0]['model'], 'test')
        self.assertEqual(result['models'][0]['tokens']['total_tokens'], 10)
        self.assertEqual(result['invalid_budget_records'], 1)
        self.assertEqual(result['unreadable_records'], 0)

    def test_matching_directories_remain_invalid_records(self):
        (self.root / 'invalid.api.json').mkdir()
        (self.root / 'scene_budget.json').mkdir()
        result = usage_report(self.root)
        self.assertEqual(result['logical_calls'], 0)
        self.assertEqual(result['unreadable_records'], 1)
        self.assertEqual(result['invalid_budget_records'], 1)


if __name__ == '__main__':
    unittest.main()
