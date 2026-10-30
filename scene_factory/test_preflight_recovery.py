"""Batch orchestration regressions for shared preflight/visual repair budgets."""
import copy
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

import automatic_repair
import batch_multiview as runner
from batch import atomic_json, file_sha256, read_json
from model_api import ModelAPIError
from dataset_export import DatasetExportError, export_dataset
import test_batch_multiview as fixtures
from test_automatic_repair import CONFIG
import vision_multiview as vision


class PreflightRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixtures.GroupedBatchTests.setUp(self)
        self.args.no_review = False
        self.args.max_repairs = 1
        self.args.model_configs = {role: copy.deepcopy(CONFIG) for role in ('analysis', 'repair', 'review')}
        self.args.pipeline_options = {'repair_mode': 'typed', 'validation_retries': 0}
        self.group = runner.load_groups(self.groups_path)[0]
        self.events = []

    def analyze(self, *args, **kwargs):
        return fixtures.GroupedBatchTests.fake_analyze(self, *args, **kwargs)

    def render(self, command, log, timeout):
        out = Path(command[command.index('--output-dir') + 1])
        self.events.append(('render', out.name))
        fixtures.GroupedBatchTests.fake_render(self, command, log, timeout)
        metadata = read_json(out / 'metadata.json')
        for view in metadata['views']:
            Image.new('RGB', (1280, 720), (91, 103, 120)).save(out / view['render'])
            view.update(width=1280, height=720)
        metadata['render']['samples'] = 64
        metadata['output_sha256'] = runner.artifact_hashes(out, metadata['outputs'])
        atomic_json(out / 'metadata.json', metadata)

    def reject(self, command, log, timeout, *, spec_hash=None):
        out = Path(command[command.index('--output-dir') + 1])
        spec = Path(command[command.index('--spec') + 1])
        self.events.append(('preflight', out.name))
        atomic_json(out / 'geometry_preflight.json', {
            'schema_version': 1, 'status': 'rejected', 'spec_sha256': spec_hash or file_sha256(spec),
            'errors': [{'code': 'daylight_aperture_blocked', 'object_ids': ['pane']}], 'warnings': []})
        raise RuntimeError('Blender rejected preflight')

    def reject_first(self, command, log, timeout):
        if Path(command[command.index('--output-dir') + 1]).name == 'iteration_00':
            return self.reject(command, log, timeout)
        return self.render(command, log, timeout)

    def model_edits(self, prompt, images, schema, output, *args, **kwargs):
        self.events.append(('technical_model', output.parent.name))
        self.assertEqual(len(images), len(self.group['images']))
        payload = json.JSONDecoder().raw_decode(prompt.split('UNTRUSTED TECHNICAL REPAIR DATA:\n')[1])[0]
        edits = automatic_repair.empty_edits('Synthetic measured camera correction.')
        for view in payload['scene_bundle']['views']:
            camera = copy.deepcopy(view['camera'])
            camera['position'][1] += .1
            edits['cameras'].append({'view_id': view['id'], 'camera': camera})
        return edits

    def review(self, bundle, directory, **kwargs):
        self.events.append(('review', directory.name))
        self.assertTrue(kwargs['final_only'])
        self.assertTrue((directory / 'render_state.json').is_file())
        self.assertEqual(kwargs['model_config'], self.args.model_configs['review'])
        return self.report(bundle)

    def report(self, bundle, verdict='recommend_accept'):
        scores = {name: (95 if verdict == 'recommend_accept' else 60) for name in vision.DIMENSIONS}
        return {'schema_version': 3, 'verdict': verdict, 'summary': 'Synthetic current-render review.',
                'scores': scores, 'per_view_scores': [{'view_id': v['id'], 'scores': scores, 'summary': 'Synthetic.'} for v in bundle['views']],
                'issues': [], 'asset_gaps': [], 'reference_limitations': [], 'revised_bundle': copy.deepcopy(bundle)}

    def run_job(self, render=None, review=None, model=None):
        with patch.object(vision, 'analyze', side_effect=self.analyze), \
             patch.object(runner, 'run_stage', side_effect=render or self.reject_first), \
             patch.object(vision, 'review', side_effect=review or self.review), \
             patch.object(automatic_repair, 'call_model', side_effect=model or self.model_edits), \
             redirect_stdout(io.StringIO()):
            return runner.run_job(self.group, self.args, self.pipeline)

    def test_preflight_repair_requires_new_render_and_current_final_review(self):
        result = self.run_job()
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['iteration'], 'iteration_01')
        self.assertEqual(result['final_review_iteration'], 'iteration_01')
        self.assertEqual(self.events, [('preflight', 'iteration_00'), ('technical_model', 'iteration_00'),
                                       ('render', 'iteration_01'), ('review', 'iteration_01')])
        self.assertEqual([h['iteration'] for h in result['history']], [1])
        self.assertEqual([h['iteration'] for h in result['technical_history']], [0])
        first = self.args.output_dir / self.group['scene_id'] / 'iteration_00'
        self.assertFalse((first / 'scene.blend').exists())
        self.assertFalse((first / 'review.json').exists())
        self.assertTrue(result['gate']['accepted'])
        self.assertIn('technical_history.json', result['artifacts'])

    def test_exhausted_or_disabled_recovery_never_calls_model(self):
        for setting in ('max_repairs', 'no_model', 'not_typed'):
            with self.subTest(setting=setting):
                self.args.output_dir = self.root / setting
                self.args.max_repairs = 0 if setting == 'max_repairs' else 1
                self.args.no_review = setting == 'no_review'
                self.args.model_configs = None if setting == 'no_model' else {r: copy.deepcopy(CONFIG) for r in ('analysis', 'review', 'repair')}
                self.args.pipeline_options['repair_mode'] = 'legacy' if setting == 'not_typed' else 'typed'
                result = self.run_job(model=AssertionError('No technical model allowed'))
                self.assertEqual(result['status'], 'failed')
                self.assertIn('no configured automatic repair budget', result['error'])
                self.assertNotIn('history', result)

    def test_no_review_allows_bounded_preflight_repair_but_blocks_export(self):
        self.args.output_dir = self.root / 'no_review_repair'
        self.args.no_review = True

        result = self.run_job()

        self.assertEqual(result['status'], 'generated_unreviewed')
        self.assertEqual(result['iteration'], 'iteration_01')
        self.assertEqual(self.events, [('preflight', 'iteration_00'),
                                       ('technical_model', 'iteration_00'),
                                       ('render', 'iteration_01')])
        self.assertEqual(len(result['technical_history']), 1)
        self.assertNotIn('history', result)
        self.assertNotIn('gate', result)
        with self.assertRaisesRegex(DatasetExportError, 'No accepted scenes'):
            export_dataset(self.args.output_dir, self.root / 'formal_dataset')
        self.assertFalse((self.root / 'formal_dataset').exists())

    def test_repeated_preflight_failure_consumes_shared_candidate_limit(self):
        result = self.run_job(render=self.reject)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(sum(kind == 'technical_model' for kind, _ in self.events), 1)
        self.assertEqual(sum(kind == 'preflight' for kind, _ in self.events), 2)
        self.assertFalse(any(kind == 'review' for kind, _ in self.events))

    def test_visual_then_technical_failure_cannot_exceed_shared_budget(self):
        def render(command, log, timeout):
            return (self.render if Path(command[command.index('--output-dir') + 1]).name == 'iteration_00'
                    else self.reject)(command, log, timeout)
        def revise(bundle, *args, **kwargs):
            edits = automatic_repair.empty_edits()
            edits['render'] = [{**bundle['shared_scene']['render'], 'exposure': -.2}]
            return automatic_repair.apply_edits(bundle, edits)
        with patch.object(automatic_repair, 'repair', side_effect=revise) as visual:
            result = self.run_job(render=render, review=lambda b, d, **k: self.report(b, 'repair'),
                                  model=AssertionError('No second repair budget'))
        self.assertEqual(visual.call_count, 1)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['gate']['accepted'])
        self.assertEqual(len(result['history']), 1)

    def test_technical_then_visual_failure_cannot_exceed_shared_budget(self):
        with patch.object(automatic_repair, 'repair', side_effect=AssertionError('No second repair budget')):
            result = self.run_job(review=lambda b, d, **k: self.report(b, 'repair'))
        self.assertEqual(result['status'], 'needs_review')
        self.assertFalse(result['gate']['accepted'])
        self.assertEqual(result['final_review_iteration'], 'iteration_01')
        self.assertEqual(len(result['technical_history']), 1)

    def test_committed_technical_repair_resume_does_not_relaunch_blender(self):
        first = self.run_job(review=RuntimeError('Review transport failed'))
        self.assertEqual(first['status'], 'review_failed')
        self.events.clear()
        second = self.run_job(render=AssertionError('Both iterations are committed'),
                              model=AssertionError('Technical commit must be cached'))
        self.assertEqual(second['status'], 'accepted')
        self.assertEqual(self.events, [('review', 'iteration_01')])
        self.assertEqual(second['final_review_iteration'], 'iteration_01')

    def test_stale_report_is_deleted_before_unrelated_renderer_crash(self):
        first = self.run_job(model=ModelAPIError('http_error', 'Unavailable'))
        self.assertEqual(first['status'], 'failed')
        def crash(command, log, timeout):
            directory = Path(command[command.index('--output-dir') + 1])
            self.assertFalse((directory / 'geometry_preflight.json').exists())
            raise RuntimeError('Unrelated crash')
        second = self.run_job(render=crash, model=AssertionError('Stale error cannot trigger API'))
        self.assertEqual(second['status'], 'failed')
        self.assertEqual(second['error'], 'Unrelated crash')

    def test_resume_preserves_spec_bytes_and_rejects_tampered_commit(self):
        first = self.run_job(review=RuntimeError('Review unavailable'))
        self.assertEqual(first['status'], 'review_failed')
        spec = Path(first['directory']) / 'iteration_00' / 'scene_bundle.json'
        spec.write_bytes(spec.read_bytes() + b'\n')
        digest = file_sha256(spec)
        second = self.run_job(render=AssertionError('Tampered commit must fail before render'),
                              model=AssertionError('Tampered commit must fail before API'))
        self.assertEqual(second['status'], 'failed')
        self.assertIn('stale or not bound', second['error'])
        self.assertEqual(file_sha256(spec), digest)

    def test_mismatched_report_hash_and_noop_repair_fail_closed(self):
        result = self.run_job(render=lambda *a: self.reject(*a, spec_hash='0' * 64),
                              model=AssertionError('Unbound report cannot trigger API'))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('stale or not bound', result['error'])
        self.args.output_dir = self.root / 'noop'
        result = self.run_job(model=lambda *a, **k: automatic_repair.empty_edits())
        self.assertEqual(result['status'], 'failed')
        self.assertIn('no change', result['error'])

    def test_final_review_error_never_accepts_technical_candidate(self):
        result = self.run_job(review=RuntimeError('No independent review available'))
        self.assertEqual(result['status'], 'review_failed')
        self.assertFalse(result['gate']['accepted'])
        self.assertNotIn('final_review_iteration', result)
        self.assertTrue((self.args.output_dir / self.group['scene_id'] / 'iteration_01' / 'scene.blend').is_file())

    def test_technical_recovery_exports_only_after_real_review_provenance(self):
        import test_dataset_export as export_fixtures
        from dataset_export import export_dataset
        with patch.object(vision, 'analyze', side_effect=lambda *a, **k: export_fixtures.DatasetExportTests.analyze(self, *a, **k)), \
             patch.object(runner, 'run_stage', side_effect=self.reject_first), \
             patch.object(automatic_repair, 'call_model', side_effect=self.model_edits), \
             patch.object(vision, '_invoke_model', side_effect=lambda *a, **k: export_fixtures.DatasetExportTests.review_response(self, *a, **k)), \
             redirect_stdout(io.StringIO()):
            result = runner.run_job(self.group, self.args, self.pipeline)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['final_review_iteration'], 'iteration_01')
        self.assertTrue((Path(result['directory']) / 'iteration_01' / 'review_provenance.json').is_file())
        manifest = export_dataset(self.args.output_dir, self.root / 'dataset')
        self.assertEqual(manifest['scene_count'], 1)
        self.assertEqual(manifest['view_count'], 2)


if __name__ == '__main__':
    unittest.main()
