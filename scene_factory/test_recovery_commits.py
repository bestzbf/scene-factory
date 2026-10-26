"""Recovery from interrupted artifact commits and bounded review retries.

Only synthetic images, mocked API replies and mocked rendering are used.
"""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from batch import atomic_json, file_sha256, read_json
import batch_multiview as runner
from model_api import ModelAPIError
import test_batch_multiview as batch_fixtures
import test_vision_multiview as vision_fixtures
import vision_multiview as vision


CONFIG = {'provider': 'openai_chat', 'model': 'fixture-vision',
          'base_url': 'http://localhost:1/v1', 'api_key_env': ''}


def assessment(bundle):
    scores = {key: 50 for key in vision.DIMENSIONS}
    return {'schema_version': 3, 'verdict': 'repair', 'summary': 'Synthetic mismatch',
            'scores': scores, 'per_view_scores': [
                {'view_id': v['id'], 'scores': scores.copy(), 'summary': 'Mismatch'}
                for v in bundle['views']],
            'issues': [], 'asset_gaps': [], 'reference_limitations': []}


def review_fixture(folder):
    bundle, views, _ = vision_fixtures.JointVisionTests().prepare(folder)
    metadata = {'status': 'completed', 'render': {'samples': 128}, 'views': []}
    for index, view in enumerate(views):
        name = view['id']
        # Match the renderer/reference aspect ratio while deliberately making
        # the synthetic render larger, so paired review exercises the proxy.
        Image.new('RGB', (2560, 1920), (30 + index * 20, 90, 120)).save(folder / ('rgb_' + name + '.png'))
        atomic_json(folder / ('annotations_' + name + '.json'), {'warnings': []})
        metadata['views'].append({'id': name, 'width': 1280, 'height': 960, 'warning_count': 0})
    atomic_json(folder/'metadata.json', metadata)
    (folder/'scene.blend').write_bytes(b'fixture')
    return bundle


@unittest.skipIf(batch_fixtures.Image is None, 'Pillow required')
class RecoveryCommitTests(unittest.TestCase):
    def setUp(self):
        self.case = batch_fixtures.GroupedBatchTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.group = runner.load_groups(self.case.groups_path)[0]

    def run_job(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run_job(self.group, self.case.args, self.case.pipeline)

    def test_initial_payload_without_commit_reenters_staged_assembly(self):
        self.case.args.pipeline_options = {'analysis_mode': 'staged'}
        self.case.args.model_configs = {'analysis': CONFIG}
        import staged_analysis
        calls = []
        def analyze(scene_id, views, output, **kwargs):
            calls.append(str(output))
            bundle = self.case.fake_analyze(scene_id, views, output, **kwargs)
            if len(calls) == 1:
                (output.parent/'analysis_provenance.json').unlink()
                raise RuntimeError('interrupted before provenance commit')
            return bundle
        with patch.object(staged_analysis, 'analyze', side_effect=analyze), \
             patch.object(runner, 'run_stage', side_effect=self.case.fake_render) as render:
            first = self.run_job()
            second = self.run_job()
        self.assertEqual(first['status'], 'failed')
        self.assertEqual(second['status'], 'generated_unreviewed')
        self.assertEqual(len(calls), 2)
        self.assertEqual(render.call_count, 1)

    def test_committed_initial_missing_or_changed_payload_fails_closed(self):
        def analyze(*args, **kwargs):
            self.case.fake_analyze(*args, **kwargs)
            raise RuntimeError('stop after committed analysis')
        with patch.object(vision, 'analyze', side_effect=analyze):
            self.run_job()
        initial = self.case.args.output_dir/'lab/initial_bundle.json'
        saved = initial.read_bytes()
        for missing in (True, False):
            if missing:
                initial.unlink()
            else:
                # Change a valid numeric value so validation succeeds but hash differs.
                initial.write_bytes(saved)
                value = read_json(initial)
                value['views'][0]['camera']['lens_mm'] += 1
                atomic_json(initial, value)
            with patch.object(vision, 'analyze', side_effect=AssertionError('Committed cache must fail closed')) as analyze:
                result = self.run_job()
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(analyze.call_count, 0)

    def test_review_commit_gap_reassesses_and_reuses_completed_render(self):
        self.case.args.no_review = False
        self.case.args.model_configs = {'analysis': CONFIG, 'review': CONFIG}
        def broken_commit(path, value):
            if Path(path).name == 'review_provenance.json':
                raise OSError('interrupted before review provenance commit')
            atomic_json(path, value)
        def reply(*args, **kwargs):
            bundle = read_json(self.case.args.output_dir/'lab/iteration_00/scene_bundle.json')
            return assessment(bundle)
        with patch.object(vision, 'analyze', side_effect=self.case.fake_analyze), \
             patch.object(runner, 'run_stage', side_effect=self.case.fake_render) as render, \
             patch.object(vision, '_invoke_model', side_effect=reply) as model:
            with patch.object(vision, 'atomic_json', side_effect=broken_commit):
                first = self.run_job()
            second = self.run_job()
        self.assertEqual(first['status'], 'review_failed')
        self.assertEqual(second['status'], 'needs_review')
        self.assertEqual(render.call_count, 1)
        self.assertEqual(model.call_count, 2)
        self.assertEqual([c.args[3].name for c in model.call_args_list], ['review.json', 'review_retry_1.json'])
        folder = self.case.args.output_dir/'lab/iteration_00'
        vision.load_review(folder, read_json(folder/'scene_bundle.json'))

    def test_committed_review_with_missing_payload_does_not_reassess(self):
        self.case.args.no_review = False
        self.case.args.model_configs = {'analysis': CONFIG, 'review': CONFIG}
        def reply(*args, **kwargs):
            bundle = read_json(self.case.args.output_dir/'lab/iteration_00/scene_bundle.json')
            return assessment(bundle)
        with patch.object(vision, 'analyze', side_effect=self.case.fake_analyze), \
             patch.object(runner, 'run_stage', side_effect=self.case.fake_render), \
             patch.object(vision, '_invoke_model', side_effect=reply):
            self.run_job()
        folder = self.case.args.output_dir/'lab'
        (folder/'result.json').unlink()  # Simulate interruption before final scene commit.
        (folder/'iteration_00/review.json').unlink()
        with patch.object(vision, '_invoke_model', side_effect=AssertionError('Do not replace committed review')) as model:
            resumed = self.run_job()
        self.assertEqual(resumed['status'], 'review_failed')
        self.assertEqual(model.call_count, 0)


class ReviewAttemptTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name).resolve()
        self.bundle = review_fixture(self.folder)

    def call(self, **kwargs):
        return vision.review(self.bundle, self.folder, final_only=True, model_config=CONFIG, **kwargs)

    def test_failed_usage_raw_and_prompt_survive_new_invocation(self):
        records = []
        def first(prompt, images, schema, path, *args, **kwargs):
            atomic_json(path.with_suffix('.api.json'), {'status': 'failed', 'usage': {'total_tokens': 99}})
            atomic_json(path.with_suffix('.raw.json'), {'uncommitted': 'old response'})
            records.extend([path.with_suffix(s) for s in ('.api.json', '.raw.json', '.prompt.txt')])
            raise ModelAPIError('timeout', 'fixture timeout')
        with patch.object(vision, '_invoke_model', side_effect=first):
            with self.assertRaises(ModelAPIError):
                self.call()
        old = {p.name: file_sha256(p) for p in records}
        def second(prompt, images, schema, path, *args, **kwargs):
            self.assertEqual(path.name, 'review_retry_1.json')
            atomic_json(path.with_suffix('.api.json'), {'status': 'completed', 'usage': {'total_tokens': 5}})
            return assessment(self.bundle)
        with patch.object(vision, '_invoke_model', side_effect=second):
            self.call()
        self.assertEqual(old, {p.name: file_sha256(p) for p in records})
        self.assertEqual(sum(read_json(p)['usage']['total_tokens'] for p in self.folder.glob('*.api.json')), 104)
        self.assertEqual(read_json(self.folder/'review_provenance.json')['attempt_outputs'], ['review_retry_1.json'])

    def test_prompt_only_interruption_reserves_suffix(self):
        (self.folder/'review.prompt.txt').write_text('original prompt')
        (self.folder/'review_retry_7.prompt.txt').write_text('interrupted prompt')
        with patch.object(vision, '_invoke_model', return_value=assessment(self.bundle)) as model:
            self.call()
        self.assertEqual(model.call_args.args[3].name, 'review_retry_8.json')
        self.assertEqual((self.folder/'review_retry_7.prompt.txt').read_text(), 'interrupted prompt')

    def test_validation_retries_pass_remaining_total_timeout(self):
        now = [100.0]
        durations = []
        def reply(prompt, images, schema, path, timeout, *args, **kwargs):
            durations.append(timeout)
            if len(durations) == 1:
                now[0] += 8
                invalid = assessment(self.bundle)
                invalid['scores']['geometry'] = 80  # Worse per-view score must not be hidden.
                return invalid
            return assessment(self.bundle)
        with patch.object(vision.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(vision, '_invoke_model', side_effect=reply):
            self.call(timeout=10, validation_retries=1)
        self.assertEqual(durations, [10, 2])

    def test_timeout_exhaustion_stops_validation_retry(self):
        now = [100.0]
        def reply(*args, **kwargs):
            now[0] += 10
            raise ModelAPIError('invalid_json', 'fixture invalid JSON')
        with patch.object(vision.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(vision, '_invoke_model', side_effect=reply) as model:
            with self.assertRaises(ModelAPIError) as raised:
                self.call(timeout=10, validation_retries=3)
        self.assertEqual(raised.exception.code, 'timeout')
        self.assertEqual(model.call_count, 1)
        self.assertFalse((self.folder/'review_provenance.json').exists())


if __name__ == '__main__':
    unittest.main()
