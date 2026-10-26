"""Offline cross-invocation guards for model requests that may have executed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import automatic_repair as repair
from batch import atomic_json, file_sha256
import model_api as api
import staged_analysis as staged
import vision_multiview as vision
from test_automatic_repair import CONFIG, prepare
from test_model_api import PNG, RESULT, SCHEMA
from test_technical_repair import preflight_fixture


def failed_call(code, blocked=False):
    def fail(prompt, images, schema, output, *args, **kwargs):
        path = Path(output)
        atomic_json(path.with_name(path.stem + '.api.json'), {
            'status': 'failed', 'failure_code': code,
            'usage': {'total_tokens': 19},
            'transport_attempts': [{'automatic_replay_blocked': blocked}]})
        raise api.ModelAPIError(code, 'Sanitized fixture failure')
    return fail


class ModelResumeGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.image = self.root / 'photo.png'
        self.image.write_bytes(PNG)
        self.events = []

    def snapshot(self, directory=None):
        return {str(path.relative_to(self.root)): file_sha256(path)
                for path in (directory or self.root).rglob('*') if path.is_file()}

    def assert_guarded(self, action, invoke):
        before = self.snapshot()
        with patch.object(*invoke, side_effect=AssertionError('No replay allowed')) as model:
            with self.assertRaises(api.ModelAPIError) as error:
                action()
        self.assertEqual(error.exception.code, 'ambiguous_previous_request')
        self.assertIn('may already have consumed tokens', str(error.exception))
        self.assertIn('new output directory', str(error.exception))
        self.assertEqual(model.call_count, 0)
        self.assertEqual(self.snapshot(), before)

    def test_helper_checks_exact_failure_or_explicit_boolean_without_credentials(self):
        path = self.root / 'stage.api.json'
        for record in ({'failure_code': 'response_interrupted'},
                       {'failure_code': 'request_outcome_unknown'},
                       {'failure_code': 'timeout', 'transport_attempts': [{'automatic_replay_blocked': True}]}):
            with self.subTest(record=record):
                atomic_json(path, {**record, 'secret_provider_field': 'PRIVATE_MUST_NOT_APPEAR'})
                with self.assertRaises(api.ModelAPIError) as failure:
                    api.guard_ambiguous_model_resume([path])
                self.assertEqual(failure.exception.code, 'ambiguous_previous_request')
                self.assertNotIn('PRIVATE_', str(failure.exception))
        for record in ({'failure_code': 'network_error', 'transport_attempts': [{'request_started': False}]},
                       {'failure_code': 'http_error', 'http_status': 402},
                       {'transport_attempts': [{'automatic_replay_blocked': 'true'}]},
                       {'status': 'completed'}):
            atomic_json(path, record)
            api.guard_ambiguous_model_resume([path])
        api.guard_ambiguous_model_resume([self.root / 'missing.api.json'])

    def test_direct_same_output_resume_is_blocked_before_network_or_audit_overwrite(self):
        output = self.root / 'direct.json'
        record_path = output.with_name('direct.api.json')
        atomic_json(record_path, {'failure_code': 'response_interrupted',
                                 'usage': {'total_tokens': 19}})
        before = file_sha256(record_path)
        with patch.object(api, '_request', side_effect=AssertionError('No network')) as request, \
                self.assertRaises(api.ModelAPIError) as failure:
            api.call_model('photo', [self.image], SCHEMA, output, config=CONFIG)
        self.assertEqual(failure.exception.code, 'ambiguous_previous_request')
        request.assert_not_called()
        self.assertEqual(file_sha256(record_path), before)

    def staged_call(self):
        return staged._api_stage('prompt', [self.image], SCHEMA, self.root, 'plan',
                                 10, 'low', CONFIG, 2, lambda value: None, None, {})

    def staged_success(self, prompt, images, schema, output, *args, **kwargs):
        path = Path(output)
        atomic_json(path.with_name(path.stem + '.raw.json'), RESULT)
        atomic_json(path.with_name(path.stem + '.api.json'), {
            'status': 'completed', 'config': api.public_config(CONFIG),
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'schema_sha256': staged._hash(schema), 'image_sha256': [file_sha256(p) for p in images]})
        return RESULT

    def test_staged_ambiguous_failures_never_retry_now_or_on_resume(self):
        for code, blocked in (('response_interrupted', False), ('request_outcome_unknown', False), ('timeout', True)):
            with self.subTest(code=code):
                for path in self.root.glob('plan*'):
                    path.unlink()
                with patch.object(staged, 'call_model', side_effect=failed_call(code, blocked)) as model:
                    with self.assertRaises(staged.StagedAnalysisError):
                        self.staged_call()
                    self.assertEqual(model.call_count, 1)
                self.assert_guarded(self.staged_call, (staged, 'call_model'))

    def test_staged_explicit_failures_resume_and_completed_stage_reuses(self):
        for code in ('network_error', 'http_error'):
            with self.subTest(code=code):
                for path in self.root.glob('plan*'):
                    path.unlink()
                with patch.object(staged, 'call_model', side_effect=failed_call(code)):
                    with self.assertRaises(staged.StagedAnalysisError):
                        self.staged_call()
                with patch.object(staged, 'call_model', side_effect=self.staged_success) as model:
                    self.assertEqual(self.staged_call()[0], RESULT)
                    self.assertEqual(model.call_count, 1)
                # A completed stage is already authoritative; no new request.
                with patch.object(staged, 'call_model', side_effect=AssertionError('Use cache')) as model:
                    self.assertEqual(self.staged_call()[0], RESULT)
                    model.assert_not_called()

    def test_staged_guard_does_not_consume_other_groups_records(self):
        atomic_json(self.root / 'plan_other_attempt_00.api.json', {'failure_code': 'response_interrupted'})
        with patch.object(staged, 'call_model', side_effect=self.staged_success) as model:
            self.assertEqual(self.staged_call()[0], RESULT)
        self.assertEqual(model.call_count, 1)

    def test_committed_stage_and_repairs_reuse_despite_older_ambiguous_audit(self):
        with patch.object(staged, 'call_model', side_effect=self.staged_success):
            expected = self.staged_call()[0]
        atomic_json(self.root / 'plan_attempt_99.api.json', {'failure_code': 'response_interrupted'})
        with patch.object(staged, 'call_model', side_effect=AssertionError('Use completed stage')) as model:
            self.assertEqual(self.staged_call()[0], expected)
        model.assert_not_called()
        for technical in (False, True):
            directory = self.root / ('committed_' + str(technical))
            directory.mkdir()
            if technical:
                bundle, report, edits = preflight_fixture(directory)
                action = lambda: repair.technical_repair(bundle, directory, report, CONFIG)
                prefix = 'technical_repair'
            else:
                bundle, report = prepare(directory)
                edits = repair.empty_edits()
                action = lambda: repair.repair(bundle, directory, report, CONFIG)
                prefix = 'repair'
            with patch.object(repair, 'call_model', return_value=edits):
                expected = action()
            atomic_json(directory / (prefix + '_attempt_99.api.json'), {'failure_code': 'request_outcome_unknown'})
            with patch.object(repair, 'call_model', side_effect=AssertionError('Use completed repair')) as model:
                self.assertEqual(action(), expected)
            model.assert_not_called()

    def test_visual_and_technical_repair_resume_guards_and_clear_failures(self):
        for technical in (False, True):
            for code, blocked in (('response_interrupted', False), ('request_outcome_unknown', False),
                                  ('timeout', True), ('network_error', False), ('http_error', False)):
                with self.subTest(technical=technical, code=code):
                    directory = self.root / ('repair_' + str(technical) + '_' + code)
                    directory.mkdir()
                    if technical:
                        bundle, report, success = preflight_fixture(directory)
                        action = lambda: repair.technical_repair(bundle, directory, report, CONFIG, validation_retries=2)
                    else:
                        bundle, report = prepare(directory)
                        success = repair.empty_edits()
                        action = lambda: repair.repair(bundle, directory, report, CONFIG, validation_retries=2)
                    with patch.object(repair, 'call_model', side_effect=failed_call(code, blocked)) as model:
                        with self.assertRaises(api.ModelAPIError):
                            action()
                        self.assertEqual(model.call_count, 1)
                    if code in ('response_interrupted', 'request_outcome_unknown') or blocked:
                        self.assert_guarded(action, (repair, 'call_model'))
                    else:
                        with patch.object(repair, 'call_model', return_value=success) as model:
                            action()
                        self.assertEqual(model.call_count, 1)
                        with patch.object(repair, 'call_model', side_effect=AssertionError('Use commit')) as model:
                            action()
                        model.assert_not_called()

    def review_fixture(self, directory):
        bundle, report = prepare(directory)
        (directory / 'scene.blend').write_bytes(b'local fixture')
        atomic_json(directory / 'metadata.json', {'status': 'completed', 'views': []})
        for view in bundle['views']:
            atomic_json(directory / ('annotations_' + view['id'] + '.json'), {'warnings': []})
        report.pop('revised_bundle')
        return bundle, report

    def test_review_resume_guard_and_explicit_failures(self):
        for code, blocked in (('response_interrupted', False), ('request_outcome_unknown', False),
                              ('timeout', True), ('network_error', False), ('http_error', False)):
            with self.subTest(code=code):
                directory = self.root / ('review_' + code)
                directory.mkdir()
                bundle, report = self.review_fixture(directory)
                action = lambda: vision.review(bundle, directory, final_only=True, model_config=CONFIG, validation_retries=2)
                with patch.object(vision, '_invoke_model', side_effect=failed_call(code, blocked)) as model:
                    with self.assertRaises(api.ModelAPIError):
                        action()
                    self.assertEqual(model.call_count, 1)
                if code in ('response_interrupted', 'request_outcome_unknown') or blocked:
                    self.assert_guarded(action, (vision, '_invoke_model'))
                else:
                    with patch.object(vision, '_invoke_model', return_value=report) as model:
                        action()
                    self.assertEqual(model.call_count, 1)

    def test_review_completed_commit_reuses_even_when_older_attempt_is_ambiguous(self):
        directory = self.root / 'review_committed'
        directory.mkdir()
        bundle, report = self.review_fixture(directory)
        action = lambda: vision.review(bundle, directory, final_only=True, model_config=CONFIG)
        with patch.object(vision, '_invoke_model', return_value=report) as model:
            expected = action()
        self.assertEqual(model.call_count, 1)
        atomic_json(directory / 'review_retry_99.api.json', {'failure_code': 'response_interrupted'})
        with patch.object(vision, '_invoke_model', side_effect=AssertionError('Use review commit')) as model:
            self.assertEqual(vision.load_review(directory, bundle), expected)
        model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
