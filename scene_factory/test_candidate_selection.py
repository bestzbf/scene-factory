"""Regression protection for reviewed automatic-repair candidates."""
import copy
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import automatic_repair
import batch_multiview as runner
from batch import atomic_json, file_sha256, read_json
from candidate_selection import candidate_metrics, compare_candidates
import test_batch_multiview as fixtures
from test_automatic_repair import CONFIG
import vision_multiview as vision


def report_scores(score=70, **overrides):
    scores = {key: score for key in vision.DIMENSIONS}
    scores.update(overrides)
    return {'scores': scores, 'per_view_scores': [{'view_id': 'view_01', 'scores': scores}],
            'issues': [], 'asset_gaps': []}


def measured(median, spread=1):
    return {'final_spatial_diagnostics': {'spatial_diagnostics': {'per_view': {
        'view_01': {'median_absolute_error_stops': median,
                    'p90_minus_p10_error_stops': spread}}}}}


class CandidateSelectionTests(unittest.TestCase):
    def metrics(self, score=70, median=None, spread=1, **overrides):
        return candidate_metrics(report_scores(score, **overrides), {'accepted': False},
                                 measured(median, spread) if median is not None else {})

    def test_regressed_lighting_rejected_despite_higher_other_scores(self):
        baseline = self.metrics(70, median=1)
        candidate = self.metrics(80, lighting=65, median=2)
        decision = compare_candidates(baseline, candidate)
        self.assertFalse(decision['selected'])
        self.assertEqual(decision['reason'], 'dimension_regression')

    def test_measured_final_brightness_regression_overrides_subjective_improvement(self):
        decision = compare_candidates(self.metrics(60, median=1), self.metrics(65, median=2))
        self.assertFalse(decision['selected'])
        self.assertEqual(decision['reason'], 'photometric_regression')

    def test_consistent_improvement_wins_and_ties_keep_original(self):
        baseline = self.metrics(60, median=2)
        self.assertTrue(compare_candidates(baseline, self.metrics(65, median=1))['selected'])
        self.assertFalse(compare_candidates(baseline, baseline)['selected'])

    def test_photometric_improvement_can_break_subjective_tie(self):
        decision = compare_candidates(self.metrics(60, median=2), self.metrics(60, median=1))
        self.assertTrue(decision['selected'])
        self.assertEqual(decision['reason'], 'photometric_improvement')

    def test_preview_measurements_do_not_count_as_final_render_evidence(self):
        report = report_scores(70)
        preview = {'tone_calibration': measured(4)['final_spatial_diagnostics']}
        self.assertIsNone(candidate_metrics(report, {}, preview)['spatial']['median_absolute_error_stops'])

    def test_worse_second_view_cannot_hide_behind_global_minimum(self):
        first = report_scores(60)
        first['per_view_scores'].append({'view_id': 'view_02', 'scores': {k: 90 for k in vision.DIMENSIONS}})
        second = copy.deepcopy(first)
        second['per_view_scores'][0]['scores'] = {k: 70 for k in vision.DIMENSIONS}
        second['per_view_scores'][1]['scores']['layout'] = 80
        self.assertFalse(compare_candidates(candidate_metrics(first), candidate_metrics(second))['selected'])

    def test_acceptance_remains_independent_terminal_gate(self):
        baseline = self.metrics(60)
        accepted = candidate_metrics(report_scores(95), {'accepted': True})
        self.assertTrue(compare_candidates(baseline, accepted)['selected'])
        self.assertFalse(compare_candidates(accepted, baseline)['selected'])

    def test_missing_invalid_or_different_views_fail_closed(self):
        with self.assertRaises(ValueError):
            candidate_metrics({})
        report = report_scores(70)
        report['per_view_scores'][0]['scores']['lighting'] = float('nan')
        with self.assertRaises(ValueError):
            candidate_metrics(report)
        first = self.metrics(70)
        second = copy.deepcopy(first)
        second['view_ids'] = ['other']
        with self.assertRaises(ValueError):
            compare_candidates(first, second)

    def test_new_major_issue_prevents_selection(self):
        first = candidate_metrics(report_scores(60))
        second_report = report_scores(70)
        second_report['issues'] = [{'severity': 'major'}]
        self.assertFalse(compare_candidates(first, candidate_metrics(second_report))['selected'])


class CandidateBatchTests(unittest.TestCase):
    def setUp(self):
        fixtures.GroupedBatchTests.setUp(self)
        self.args.no_review = False
        self.args.max_repairs = 2
        self.args.model_configs = {role: copy.deepcopy(CONFIG) for role in ('analysis', 'repair', 'review')}
        self.args.pipeline_options = {'repair_mode': 'typed', 'validation_retries': 0}
        self.group = runner.load_groups(self.groups_path)[0]
        self.repair_calls = []
        self.reviewed = []
        self.scores = [70, 50, 75]

    def analyze(self, *args, **kwargs):
        return fixtures.GroupedBatchTests.fake_analyze(self, *args, **kwargs)

    def render(self, *args, **kwargs):
        return fixtures.GroupedBatchTests.fake_render(self, *args, **kwargs)

    def review(self, bundle, directory, **kwargs):
        self.reviewed.append(directory.name)
        score = self.scores[int(directory.name.split('_')[-1])]
        scores = {name: score for name in vision.DIMENSIONS}
        report = {'schema_version': 3, 'verdict': 'repair', 'summary': 'Synthetic independent review.',
                  'scores': scores, 'per_view_scores': [{'view_id': v['id'], 'scores': scores, 'summary': 'Synthetic.'}
                                                      for v in bundle['views']],
                  'issues': [], 'asset_gaps': [], 'reference_limitations': [],
                  'revised_bundle': copy.deepcopy(bundle)}
        atomic_json(directory/'review.json', report)
        return report

    def repair(self, bundle, directory, report, **kwargs):
        self.repair_calls.append((bundle['shared_scene']['render']['exposure'], directory))
        if directory.name == 'repair_from_best':
            self.assertEqual(directory.parent.name, 'iteration_01')
            evidence = read_json(directory/'candidate_feedback.json')
            self.assertEqual(evidence['selected_iteration'], 'iteration_00')
            self.assertEqual(evidence['candidates'][-1]['reason'], 'dimension_regression')
            copied = read_json(directory/'scene_bundle.json')
            self.assertEqual(copied, bundle)
        revised = copy.deepcopy(bundle)
        revised['shared_scene']['render']['exposure'] += .1 * len(self.repair_calls)
        return revised

    def run_job(self):
        with patch.object(vision, 'analyze', side_effect=self.analyze), \
             patch.object(runner, 'run_stage', side_effect=self.render), \
             patch.object(vision, 'review', side_effect=self.review), \
             patch.object(automatic_repair, 'repair', side_effect=self.repair), \
             redirect_stdout(io.StringIO()):
            return runner.run_job(self.group, self.args, self.pipeline)

    def test_regression_rolls_back_next_repair_source_and_preserves_all_history(self):
        result = self.run_job()
        self.assertEqual(result['status'], 'needs_review')
        self.assertFalse(result['gate']['accepted'])
        self.assertEqual(result['iteration'], 'iteration_02')
        self.assertEqual(result['final_review_iteration'], 'iteration_02')
        self.assertEqual(self.repair_calls[0][0], self.repair_calls[1][0])
        self.assertEqual([h['selected_as_best'] for h in result['history']], [True, False, True])
        self.assertEqual(len(result['history']), 3)
        self.assertIn('candidate_selection.json', result['artifacts'])
        self.assertIn('iteration_01/repair_from_best/repair_source.json', result['artifacts'])

    def test_last_bad_candidate_does_not_replace_best_final_output_or_review(self):
        self.args.max_repairs = 1
        result = self.run_job()
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(result['iteration'], 'iteration_00')
        self.assertEqual(result['final_review_iteration'], 'iteration_00')
        self.assertEqual(result['history'][-1]['scores']['lighting'], 50)
        self.assertEqual(result['summary'], result['history'][0]['summary'])
        self.assertFalse(result['gate']['accepted'])
        self.assertEqual(result['candidate_selection']['candidate_count'], 2)
        self.assertTrue((Path(result['directory'])/'iteration_01'/'scene.blend').exists())

    def test_completed_resume_reuses_selection_without_new_calls(self):
        self.args.max_repairs = 1
        first = self.run_job()
        with patch.object(runner, 'run_stage', side_effect=AssertionError('No fresh render')), \
             patch.object(vision, 'review', side_effect=AssertionError('No fresh review')), \
             patch.object(automatic_repair, 'repair', side_effect=AssertionError('No fresh repair')), \
             redirect_stdout(io.StringIO()):
            result = runner.run_job(self.group, self.args, self.pipeline)
        self.assertTrue(result['resumed'])
        self.assertEqual(result['iteration'], first['iteration'])

    def test_interrupted_later_review_keeps_best_and_resume_reuses_committed_renders(self):
        original_review = self.review
        def fail_last(bundle, directory, **kwargs):
            if directory.name == 'iteration_02':
                raise RuntimeError('Synthetic review interruption')
            return original_review(bundle, directory, **kwargs)
        self.review = fail_last
        first = self.run_job()
        self.assertEqual(first['status'], 'review_failed')
        self.assertEqual(first['iteration'], 'iteration_00')
        self.assertEqual(first['final_review_iteration'], 'iteration_00')
        self.assertFalse(first['gate']['accepted'])
        self.review = original_review
        # A deterministic fake repair stands in for automatic_repair's
        # hash-bound committed-edit cache on resume.
        self.repair_calls.clear()
        self.render = lambda *a, **k: self.fail('Committed candidate must not rerender')
        resumed = self.run_job()
        self.assertEqual(resumed['status'], 'needs_review')
        self.assertEqual(resumed['iteration'], 'iteration_02')
        self.assertEqual([h['selected_as_best'] for h in resumed['history']], [True, False, True])

    def test_rollback_snapshot_tampering_is_not_silently_overwritten(self):
        self.run_job()
        folder = self.args.output_dir/self.group['scene_id']
        source = folder/'iteration_00'
        best = {'directory': source, 'metadata': read_json(source/'metadata.json')}
        feedback = read_json(folder/'iteration_01'/'repair_from_best'/'candidate_feedback.json')
        (folder/'iteration_01'/'repair_from_best'/'scene_bundle.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            runner.prepare_best_repair_source(best, folder/'iteration_01', feedback)

    def test_final_meter_is_bound_to_saved_final_png(self):
        self.args.max_repairs = 0
        result = self.run_job()
        folder = Path(result['directory'])/'iteration_00'
        metrics = read_json(folder/'final_spatial_diagnostics.json')
        self.assertEqual(metrics['measurement'], 'saved_final_render_png')
        for view_id, hashes in metrics['image_sha256'].items():
            self.assertEqual(hashes['render_sha256'], file_sha256(folder/('rgb_'+view_id+'.png')))


class FullBundleRollbackTests(unittest.TestCase):
    """Rejected legacy proposals must not be rendered again after rollback."""
    def setUp(self):
        fixtures.GroupedBatchTests.setUp(self)
        self.args.no_review = False
        self.args.max_repairs = 4
        self.args.model_configs = None
        self.args.pipeline_options = {}
        self.group = runner.load_groups(self.groups_path)[0]
        self.rendered = []
        self.reviewed = []
        self.fail_second_review = False

    def analyze(self, *args, **kwargs):
        return fixtures.GroupedBatchTests.fake_analyze(self, *args, **kwargs)

    def render(self, command, *args):
        self.rendered.append(Path(command[command.index('--spec')+1]).parent.name)
        return fixtures.GroupedBatchTests.fake_render(self, command, *args)

    def review(self, bundle, directory, **kwargs):
        index = int(directory.name.split('_')[-1])
        self.reviewed.append(directory.name)
        if index == 1 and self.fail_second_review:
            raise RuntimeError('Synthetic interruption before second review')
        score = 70 if index == 0 else 50
        scores = {name: score for name in vision.DIMENSIONS}
        revised = copy.deepcopy(bundle)
        if index == 0:
            revised['shared_scene']['render']['exposure'] += .1
        report = {'schema_version': 3, 'verdict': 'repair', 'summary': f'Synthetic candidate {index}.',
                  'scores': scores, 'per_view_scores': [{'view_id': v['id'], 'scores': scores, 'summary': 'Synthetic.'}
                                                      for v in bundle['views']],
                  'issues': [], 'asset_gaps': [], 'reference_limitations': [], 'revised_bundle': revised}
        atomic_json(directory/'review.json', report)
        # The real reviewer writes hash-bound provenance. Cache handling is
        # tested separately; this fixture commits a marker and loads its JSON.
        atomic_json(directory/'review_provenance.json', {'synthetic': True})
        return report

    def run_job(self):
        with patch.object(vision, 'analyze', side_effect=self.analyze), \
             patch.object(runner, 'run_stage', side_effect=self.render), \
             patch.object(vision, 'review', side_effect=self.review), \
             patch.object(vision, 'load_review', side_effect=lambda directory, bundle: read_json(directory/'review.json')), \
             redirect_stdout(io.StringIO()):
            return runner.run_job(self.group, self.args, self.pipeline)

    def assert_stopped_at_best(self, result):
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(result['stop_reason'], 'repeated_repair')
        self.assertEqual(result['iteration'], 'iteration_00')
        self.assertEqual(result['final_review_iteration'], 'iteration_00')
        self.assertEqual(result['summary'], 'Synthetic candidate 0.')
        self.assertFalse(result['gate']['accepted'])
        self.assertEqual(result['candidate_selection']['candidate_count'], 2)
        self.assertEqual([h['selected_as_best'] for h in result['history']], [True, False])
        self.assertEqual(self.rendered, ['iteration_00', 'iteration_01'])
        self.assertFalse((Path(result['directory'])/'iteration_02').exists())

    def test_rejected_proposal_stops_before_duplicate_render_and_preserves_best(self):
        result = self.run_job()
        self.assert_stopped_at_best(result)
        self.assertEqual(self.reviewed, ['iteration_00', 'iteration_01'])
        runner.verify_hashes(Path(result['directory']), result['artifacts'])

    def test_completed_resume_reuses_terminal_result_without_model_or_blender(self):
        first = self.run_job()
        with patch.object(vision, 'analyze', side_effect=AssertionError('No fresh analysis')), \
             patch.object(runner, 'run_stage', side_effect=AssertionError('No fresh render')), \
             patch.object(vision, 'review', side_effect=AssertionError('No fresh review')), \
             redirect_stdout(io.StringIO()):
            resumed = runner.run_job(self.group, self.args, self.pipeline)
        self.assertTrue(resumed['resumed'])
        self.assert_stopped_at_best(resumed)
        self.assertEqual(first['artifacts'], resumed['artifacts'])

    def test_interrupted_resume_rebuilds_attempt_history_without_rerendering(self):
        self.fail_second_review = True
        first = self.run_job()
        self.assertEqual(first['status'], 'review_failed')
        self.assertEqual(first['iteration'], 'iteration_00')
        self.fail_second_review = False
        resumed = self.run_job()
        self.assert_stopped_at_best(resumed)
        self.assertEqual(self.reviewed, ['iteration_00', 'iteration_01', 'iteration_01'])

    def test_same_guard_applies_to_configured_full_bundle_path(self):
        self.args.model_configs = {role: copy.deepcopy(CONFIG) for role in ('analysis', 'repair', 'review')}
        self.args.pipeline_options = {'repair_mode': 'full_bundle'}
        self.assert_stopped_at_best(self.run_job())

    def test_exhausted_budget_does_not_attempt_another_repair(self):
        self.args.max_repairs = 1
        result = self.run_job()
        self.assertEqual(result['status'], 'needs_review')
        self.assertNotIn('stop_reason', result)
        self.assertEqual(self.rendered, ['iteration_00', 'iteration_01'])
        self.assertEqual(result['iteration'], 'iteration_00')


if __name__ == '__main__':
    unittest.main()
