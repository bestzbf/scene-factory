"""Persistent per-scene budgets through real loopback model API transports.

All model responses are local synthetic fixtures. Set
SCENE_FACTORY_TEST_BLENDER to exercise the batch check with actual Blender;
neither mode measures reconstruction quality or incurs provider charges.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import model_api as api
from scene_budget import BudgetError, reserve_request, scene_budget_session
import test_api_batch_integration as batch_fixture
import test_staged_analysis as staged_fixture
from test_model_api import FixtureServer, Handler, PNG, RESULT, SCHEMA, chat
from test_model_api_streaming import chunk, event, stream


class SceneBudgetAPIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FixtureServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root / 'photo.png'
        self.image.write_bytes(PNG)
        self.server.requests, self.server.responses = [], []
        self.config = {'provider': 'openai_chat', 'model': 'fixture-budget',
            'base_url': f'http://127.0.0.1:{self.server.server_port}/v1',
            'api_key_env': '', 'max_retries': 0, 'max_output_tokens': 20}

    def queue(self, envelope=None, status=200):
        self.server.responses.append((status, {}, chat() if envelope is None else envelope))

    def call(self, root, name='analysis', **overrides):
        return api.call_model('Synthetic budget fixture.', [self.image], SCHEMA,
                              root / (name + '.json'), config={**self.config, **overrides})

    def assert_budget_failure(self, operation):
        with self.assertRaises(api.ModelAPIError) as failure:
            operation()
        self.assertEqual(failure.exception.code, 'scene_budget_exceeded')

    def test_each_http_retry_consumes_request_allowance(self):
        scene = self.root / 'retry'
        self.queue({'error': {'message': 'synthetic unavailable'}}, status=503)
        self.queue({'error': {'message': 'synthetic limited'}}, status=429)
        self.queue()
        with scene_budget_session(scene, {'max_requests': 2}), patch.object(api.time, 'sleep'):
            self.assert_budget_failure(lambda: self.call(scene, max_retries=3))
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual(len(self.server.responses), 1)
        record = json.loads((scene / 'analysis.api.json').read_text())
        self.assertEqual(record['failure_code'], 'scene_budget_exceeded')
        ledger = json.loads((scene / 'scene_budget.json').read_text())
        self.assertEqual(ledger['requests'], 2)
        self.assertEqual(ledger['unknown_usage_attempts'], 2)
        self.assertEqual(ledger['output_tokens_accounted'], 40)

    def test_invalid_json_and_schema_still_consume_requests_and_reported_tokens(self):
        for label, content, expected in (
                ('json', '{broken', 'invalid_json'),
                ('schema', json.dumps({'scene': 'fixture', 'number': 99}), 'schema_mismatch')):
            with self.subTest(label=label):
                scene = self.root / label
                self.queue(chat(content=content))
                before = len(self.server.requests)
                policy = {'max_reported_tokens': 19}
                with scene_budget_session(scene, policy):
                    with self.assertRaises(api.ModelAPIError) as invalid:
                        self.call(scene)
                    self.assertEqual(invalid.exception.code, expected)
                    self.assert_budget_failure(lambda: self.call(scene, 'repair'))
                self.assertEqual(len(self.server.requests), before + 1)
                audit = json.loads((scene / 'analysis.api.json').read_text())
                self.assertEqual(audit['usage']['total_tokens'], 19)
                self.assertFalse((scene / 'analysis.raw.json').exists())
                with scene_budget_session(scene, policy):
                    self.assert_budget_failure(lambda: self.call(scene, 'resumed_repair'))
                self.assertEqual(len(self.server.requests), before + 1)

    def test_session_restart_does_not_reset_request_allowance(self):
        scene = self.root / 'resume'
        self.queue()
        with scene_budget_session(scene, {'max_requests': 1}):
            self.assertEqual(self.call(scene), RESULT)
        ledger = json.loads((scene / 'scene_budget.json').read_text())
        self.assertTrue(ledger)
        with scene_budget_session(scene, {'max_requests': 1}):
            self.assert_budget_failure(lambda: self.call(scene, 'review'))
        self.assertEqual(len(self.server.requests), 1)
        self.assertTrue((scene / 'analysis.raw.json').is_file())

    def test_missing_usage_retains_output_reservation_after_restart(self):
        scene = self.root / 'missing_usage'
        envelope = chat()
        envelope.pop('usage')
        self.queue(envelope)
        policy = {'max_output_tokens': 20}
        with scene_budget_session(scene, policy):
            self.assertEqual(self.call(scene), RESULT)
        with scene_budget_session(scene, policy):
            self.assert_budget_failure(lambda: self.call(scene, 'review'))
        self.assertEqual(len(self.server.requests), 1)
        ledger = json.loads((scene / 'scene_budget.json').read_text())
        self.assertEqual(ledger['output_tokens_accounted'], 20)
        self.assertEqual(ledger['unknown_usage_attempts'], 1)
        self.assertEqual(ledger['reported_tokens'], 0)

    def test_reported_output_reconciles_reservation_before_next_request(self):
        scene = self.root / 'known_usage'
        self.queue()
        self.queue()
        # Each request reserves 20; the first reports only seven output tokens.
        # The second fits (7 + 20 <= 30), but a third does not (14 + 20 > 30).
        with scene_budget_session(scene, {'max_output_tokens': 30}):
            self.assertEqual(self.call(scene), RESULT)
            self.assertEqual(self.call(scene, 'review'), RESULT)
            self.assert_budget_failure(lambda: self.call(scene, 'repair'))
        self.assertEqual(len(self.server.requests), 2)
        ledger = json.loads((scene / 'scene_budget.json').read_text())
        self.assertEqual(ledger['output_tokens_accounted'], 14)
        self.assertEqual(ledger['reported_tokens'], 38)
        self.assertEqual(ledger['unknown_usage_attempts'], 0)

    def test_invalid_usage_is_sanitized_and_keeps_reservation_for_json_and_sse(self):
        invalid_usage = {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 'bad'}
        envelope = chat()
        envelope['usage'] = invalid_usage
        sse = (event(chunk(content=json.dumps(RESULT))) + event(chunk(finish='stop'))
               + event({'choices': [], 'usage': invalid_usage}) + b'data: [DONE]\n\n')
        for name, config, headers, response in (
                ('json_invalid_usage', {}, {}, envelope),
                ('sse_invalid_usage', {'stream': True}, {'Content-Type': 'text/event-stream'}, sse)):
            with self.subTest(transport=name):
                scene = self.root / name
                self.server.responses.append((200, headers, response))
                before = len(self.server.requests)
                policy = {'max_output_tokens': 20}
                with scene_budget_session(scene, policy):
                    self.assertEqual(self.call(scene, **config), RESULT)
                audit_text = (scene / 'analysis.api.json').read_text()
                audit = json.loads(audit_text)
                self.assertTrue(audit['usage_invalid'])
                self.assertTrue(audit['transport_attempts'][0]['usage_invalid'])
                self.assertEqual(audit['usage'], {'prompt_tokens': 10, 'completion_tokens': 3})
                self.assertNotIn('"bad"', audit_text)
                ledger = json.loads((scene / 'scene_budget.json').read_text())
                self.assertEqual(ledger['output_tokens_accounted'], 20)
                self.assertEqual(ledger['reported_tokens'], 13)
                self.assertEqual(ledger['unknown_usage_attempts'], 1)
                self.assertFalse(ledger['attempts'][0]['usage_complete'])
                self.assertNotIn('"bad"', json.dumps(ledger))
                with scene_budget_session(scene, policy):
                    self.assert_budget_failure(lambda: self.call(scene, 'review', **config))
                self.assertEqual(len(self.server.requests), before + 1)

    def test_pending_reservation_blocks_new_session_before_any_additional_http(self):
        scene = self.root / 'pending_resume'
        policy = {'max_requests': 10, 'max_output_tokens': 100}
        self.queue()
        with scene_budget_session(scene, policy):
            # Simulate a process stopping after durable reservation and before
            # settlement. A different request may still proceed in this active
            # session, as parallel stages normally can.
            pending = reserve_request(scene / 'interrupted.json', 20)
            self.assertIsNotNone(pending)
            self.assertEqual(self.call(scene, 'concurrent'), RESULT)
        self.assertEqual(len(self.server.requests), 1)
        ledger_path = scene / 'scene_budget.json'
        previous = ledger_path.read_bytes()
        ledger = json.loads(previous)
        self.assertEqual(ledger['pending_attempts'], 1)
        self.assertEqual(ledger['requests'], 2)
        self.assertEqual(ledger['output_tokens_accounted'], 27)
        with self.assertRaises(BudgetError) as failure:
            with scene_budget_session(scene, policy):
                self.call(scene, 'after_restart')
        self.assertEqual(failure.exception.code, 'ambiguous_previous_request')
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(ledger_path.read_bytes(), previous)
        self.assertFalse((scene / 'after_restart.api.json').exists())

    def test_api_audit_write_failure_leaves_pending_and_blocks_resumed_http(self):
        scene = self.root / 'audit_write_failure'
        policy = {'max_requests': 10, 'max_output_tokens': 100}
        self.queue()
        real_write = api.atomic_json
        def fail_audit(path, value):
            if Path(path).name == 'analysis.api.json':
                raise OSError('Synthetic audit persistence failure')
            return real_write(path, value)
        with scene_budget_session(scene, policy), patch.object(api, 'atomic_json', side_effect=fail_audit):
            with self.assertRaisesRegex(OSError, 'Synthetic audit persistence failure'):
                self.call(scene)
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(json.loads((scene / 'analysis.raw.json').read_text()), RESULT)
        self.assertFalse((scene / 'analysis.api.json').exists())
        ledger_path = scene / 'scene_budget.json'
        previous = ledger_path.read_bytes()
        ledger = json.loads(previous)
        self.assertEqual(ledger['requests'], 1)
        self.assertEqual(ledger['pending_attempts'], 1)
        self.assertEqual(ledger['output_tokens_accounted'], 20)
        self.assertEqual(ledger['reported_tokens'], 0)
        with self.assertRaises(BudgetError) as failure:
            with scene_budget_session(scene, policy):
                self.call(scene)
        self.assertEqual(failure.exception.code, 'ambiguous_previous_request')
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(ledger_path.read_bytes(), previous)

    def test_reported_total_stops_next_call_and_is_independent_per_scene(self):
        for name in ('first_scene', 'second_scene'):
            scene = self.root / name
            self.queue()
            with scene_budget_session(scene, {'max_reported_tokens': 19}):
                self.assertEqual(self.call(scene), RESULT)
                self.assert_budget_failure(lambda: self.call(scene, 'review'))
        self.assertEqual(len(self.server.requests), 2)
        for name in ('first_scene', 'second_scene'):
            self.assertTrue((self.root / name / 'scene_budget.json').is_file())

    def test_streaming_responses_and_anthropic_share_budget_enforcement(self):
        fixtures = [
            ('streaming', {'stream': True}, '/v1/chat/completions',
             {'Content-Type': 'text/event-stream'}, stream(), 19),
            ('responses', {'provider': 'openai_responses'}, '/v1/responses', {},
             {'status': 'completed', 'output': [{'type': 'message', 'status': 'completed',
                 'content': [{'type': 'output_text', 'text': json.dumps(RESULT)}]}],
              'usage': {'input_tokens': 10, 'output_tokens': 7, 'total_tokens': 17}}, 17),
            ('anthropic', {'provider': 'anthropic'}, '/v1/messages', {},
             {'type': 'message', 'stop_reason': 'end_turn',
              'content': [{'type': 'text', 'text': json.dumps(RESULT)}],
              'usage': {'input_tokens': 5, 'output_tokens': 7,
                        'cache_creation_input_tokens': 3, 'cache_read_input_tokens': 4}}, 19)]
        for name, config, endpoint, headers, envelope, total in fixtures:
            with self.subTest(provider=name):
                scene = self.root / name
                self.server.responses.append((200, headers, envelope))
                before = len(self.server.requests)
                with scene_budget_session(scene, {'max_requests': 1}):
                    self.assertEqual(self.call(scene, **config), RESULT)
                    self.assert_budget_failure(lambda: self.call(scene, 'review', **config))
                self.assertEqual(len(self.server.requests), before + 1)
                self.assertEqual(self.server.requests[-1]['path'], endpoint)
                ledger = json.loads((scene / 'scene_budget.json').read_text())
                self.assertEqual(ledger['requests'], 1)
                self.assertEqual(ledger['output_tokens_accounted'], 7)
                self.assertEqual(ledger['reported_tokens'], total)
                self.assertEqual(ledger['unknown_usage_attempts'], 0)


class StagedSceneBudgetIntegrationTests(unittest.TestCase):
    setUp = staged_fixture.StagedTests.setUp
    stop_server = staged_fixture.StagedTests.stop_server
    default_reply = staged_fixture.StagedTests.default_reply
    analyze = staged_fixture.StagedTests.analyze

    def test_planner_budget_error_is_propagated_without_validation_retries(self):
        staged = staged_fixture.staged
        self.options['validation_retries'] = 3
        self.config['max_output_tokens'] = 20
        with scene_budget_session(self.output.parent, {'max_output_tokens': 19}), \
                patch.object(staged, 'call_model', wraps=staged.call_model) as call:
            with self.assertRaises(api.ModelAPIError) as failure:
                self.analyze()
        self.assertEqual(failure.exception.code, 'scene_budget_exceeded')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(self.requests, [])
        stages = self.output.parent / 'initial_bundle_stages'
        self.assertEqual(len(list(stages.glob('plan_attempt_*.validation.json'))), 1)
        self.assertEqual(list(stages.glob('detail_*')), [])
        self.assertFalse(self.output.exists())
        self.assertFalse((self.output.parent / 'analysis_provenance.json').exists())

    def test_detail_budget_error_is_propagated_without_validation_retries(self):
        staged = staged_fixture.staged
        self.options['validation_retries'] = 3
        with scene_budget_session(self.output.parent, {'max_requests': 1}), \
                patch.object(staged, 'call_model', wraps=staged.call_model) as call:
            with self.assertRaises(api.ModelAPIError) as failure:
                self.analyze()
        self.assertEqual(failure.exception.code, 'scene_budget_exceeded')
        self.assertEqual(self.counter, {'plan': 1})
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(call.call_count, 1 + len(self.plan['groups']))
        stages = self.output.parent / 'initial_bundle_stages'
        for group in self.plan['groups']:
            self.assertEqual(len(list(stages.glob('detail_' + group['id'] + '_attempt_*.validation.json'))), 1)
        failures = json.loads((stages / 'failures.json').read_text())
        self.assertEqual(set(failures['failed_groups']), {group['id'] for group in self.plan['groups']})
        self.assertEqual(failures['completed_groups'], [])
        self.assertFalse(self.output.exists())
        self.assertFalse((self.output.parent / 'analysis_provenance.json').exists())


class SceneBudgetBatchIntegrationTests(unittest.TestCase):
    # Reuse fixture construction and optional real Blender without inheriting
    # and rerunning that class's unrelated grouped-model tests.
    setUp = batch_fixture.APIBatchIntegrationTests.setUp
    stop_server = batch_fixture.APIBatchIntegrationTests.stop_server
    run_batch = batch_fixture.APIBatchIntegrationTests.run_batch

    def test_review_budget_stop_preserves_render_blocks_export_and_resumes_without_calls(self):
        runner = batch_fixture.runner
        self.config['batch']['no_review'] = False
        self.config['budget'] = {'max_requests': 1}
        runner.atomic_json(self.groups, {'schema_version': 1, 'scenes': [
            {'scene_id': 'two_views', 'images': ['0.png', '1.png']}]})
        with redirect_stdout(io.StringIO()):
            self.run_batch()
        self.assertEqual(len(self.requests), 1)
        scene = self.out / 'two_views'
        result = runner.read_json(scene / 'result.json')
        self.assertEqual(result['status'], 'needs_review', result.get('error'))
        self.assertEqual(result['stop_reason'], 'scene_budget_exceeded')
        self.assertFalse(result['gate']['accepted'])
        render = scene / 'iteration_00'
        self.assertTrue((render / 'scene.blend').is_file())
        self.assertEqual(len(list(render.glob('rgb_*.png'))), 2)
        self.assertFalse((render / 'review_provenance.json').exists())
        self.assertEqual(runner.read_json(self.out / 'accepted.json'), [])
        from dataset_export import DatasetExportError, export_dataset
        with self.assertRaisesRegex(DatasetExportError, 'No accepted'):
            export_dataset(self.out, self.root / 'dataset')
        self.assertFalse((self.root / 'dataset').exists())
        before_ledger = (scene / 'scene_budget.json').read_bytes()
        before_artifacts = {path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in [render / 'scene.blend', *render.glob('rgb_*.png')]}
        with redirect_stdout(io.StringIO()):
            self.run_batch()
        self.assertEqual(len(self.requests), 1)
        self.assertEqual((scene / 'scene_budget.json').read_bytes(), before_ledger)
        for path, (contents, modified) in before_artifacts.items():
            self.assertEqual(path.read_bytes(), contents)
            self.assertEqual(path.stat().st_mtime_ns, modified)
        summary = runner.read_json(self.out / 'summary.json')
        self.assertTrue(summary['scenes'][0]['resumed'])
        self.assertEqual(runner.read_json(self.out / 'accepted.json'), [])


if __name__ == '__main__':
    unittest.main()
