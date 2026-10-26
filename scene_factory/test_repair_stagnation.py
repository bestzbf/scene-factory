"""Stop repeated unproductive repairs while preserving the reviewed winner."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

import automatic_repair
import batch_multiview as runner
from batch import atomic_json, read_json
from test_automatic_repair import CONFIG
import test_batch_multiview as fixtures
import vision_multiview as vision


class RepairStagnationTests(unittest.TestCase):
    def setUp(self):
        fixtures.GroupedBatchTests.setUp(self)
        self.args.no_review = False
        self.args.max_repairs = 4
        self.args.model_configs = {role: copy.deepcopy(CONFIG) for role in ('analysis', 'repair', 'review')}
        self.args.pipeline_options = {'repair_mode': 'typed', 'validation_retries': 0}
        self.group = runner.load_groups(self.groups_path)[0]
        self.scores = [70, 50, 55, 60, 80]
        self.rendered, self.reviewed, self.repaired = [], [], []
        self.interrupt_review = None

    def analyze(self, *args, **kwargs):
        return fixtures.GroupedBatchTests.fake_analyze(self, *args, **kwargs)

    def render(self, command, *args):
        self.rendered.append(Path(command[command.index('--spec') + 1]).parent.name)
        return fixtures.GroupedBatchTests.fake_render(self, command, *args)

    def review(self, bundle, directory, **kwargs):
        index = int(directory.name.rsplit('_', 1)[-1])
        self.reviewed.append(directory.name)
        if index == self.interrupt_review:
            raise RuntimeError('Synthetic review interruption')
        scores = {name: self.scores[index] for name in vision.DIMENSIONS}
        report = {'schema_version': 3, 'verdict': 'repair', 'summary': f'Candidate {index}.',
                  'scores': scores,
                  'per_view_scores': [{'view_id': view['id'], 'scores': scores, 'summary': 'Synthetic.'}
                                      for view in bundle['views']],
                  'issues': [], 'asset_gaps': [], 'reference_limitations': [],
                  'revised_bundle': copy.deepcopy(bundle)}
        atomic_json(directory / 'review.json', report)
        # Real provenance verification is covered by vision tests. This marker
        # exercises replay of committed reviews, not repeated model sampling.
        atomic_json(directory / 'review_provenance.json', {'synthetic': True})
        return report

    def repair(self, bundle, directory, report, **kwargs):
        attempt = directory.parent if directory.name == 'repair_from_best' else directory
        self.repaired.append(attempt.name)
        revised = copy.deepcopy(bundle)
        revised['shared_scene']['render']['exposure'] += .01 * (int(attempt.name.rsplit('_', 1)[-1]) + 1)
        return revised

    def run_job(self):
        with patch.object(vision, 'analyze', side_effect=self.analyze), \
             patch.object(runner, 'run_stage', side_effect=self.render), \
             patch.object(vision, 'review', side_effect=self.review), \
             patch.object(vision, 'load_review', side_effect=lambda directory, bundle: read_json(directory / 'review.json')), \
             patch.object(automatic_repair, 'repair', side_effect=self.repair), \
             redirect_stdout(io.StringIO()):
            return runner.run_job(self.group, self.args, self.pipeline)

    def assert_stagnated(self, result, selected='iteration_00', count=3):
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(result['stop_reason'], 'repair_stagnation')
        self.assertFalse(result['gate']['accepted'])
        self.assertEqual(result['iteration'], selected)
        self.assertEqual(result['final_review_iteration'], selected)
        self.assertEqual(result['candidate_selection']['candidate_count'], count)
        self.assertFalse((Path(result['directory']) / f'iteration_{count:02d}').exists())
        runner.verify_hashes(Path(result['directory']), result['artifacts'])

    def test_default_stops_after_two_rejected_candidates_before_third_repair(self):
        result = self.run_job()
        self.assert_stagnated(result)
        self.assertEqual(self.repaired, ['iteration_00', 'iteration_01'])
        self.assertEqual(self.rendered, ['iteration_00', 'iteration_01', 'iteration_02'])
        selection = read_json(Path(result['directory']) / 'candidate_selection.json')
        self.assertEqual(selection['repair_stagnation'], {'patience': 2, 'consecutive_nonimproving': 2})
        self.assertEqual([row['consecutive_nonimproving'] for row in selection['candidates']], [0, 1, 2])

    def test_equal_scores_are_nonimprovement_and_do_not_buy_endless_new_geometry(self):
        self.scores = [70] * 5
        result = self.run_job()
        self.assert_stagnated(result)
        self.assertEqual(len(self.repaired), 2)

    def test_improvement_resets_streak_and_allows_another_failed_attempt(self):
        self.args.max_repairs = 5
        self.scores = [70, 50, 75, 60, 60, 80]
        result = self.run_job()
        self.assert_stagnated(result, selected='iteration_02', count=5)
        selection = read_json(Path(result['directory']) / 'candidate_selection.json')
        self.assertEqual([row['consecutive_nonimproving'] for row in selection['candidates']], [0, 1, 0, 1, 2])

    def test_one_stops_on_first_rejected_repair(self):
        self.args.pipeline_options['repair_stagnation_patience'] = 1
        result = self.run_job()
        self.assert_stagnated(result, count=2)
        self.assertEqual(self.repaired, ['iteration_00'])

    def test_zero_disables_stagnation_until_iteration_budget(self):
        self.args.pipeline_options['repair_stagnation_patience'] = 0
        result = self.run_job()
        self.assertEqual(result['status'], 'needs_review')
        self.assertNotIn('stop_reason', result)
        self.assertEqual(result['iteration'], 'iteration_04')
        self.assertEqual(len(self.repaired), 4)

    def test_resume_reconstructs_streak_from_cached_reviews_without_rerendering(self):
        self.interrupt_review = 2
        first = self.run_job()
        self.assertEqual(first['status'], 'review_failed')
        self.assertEqual(first['iteration'], 'iteration_00')
        self.interrupt_review = None
        self.repaired.clear()
        resumed = self.run_job()
        self.assert_stagnated(resumed)
        self.assertEqual(self.rendered, ['iteration_00', 'iteration_01', 'iteration_02'])
        self.assertEqual(self.reviewed, ['iteration_00', 'iteration_01', 'iteration_02', 'iteration_02'])
        self.assertEqual(self.repaired, ['iteration_00', 'iteration_01'])

    def test_completed_stop_resumes_without_api_or_render_work(self):
        first = self.run_job()
        with patch.object(vision, 'analyze', side_effect=AssertionError('No new analysis')), \
             patch.object(runner, 'run_stage', side_effect=AssertionError('No new render')), \
             patch.object(vision, 'review', side_effect=AssertionError('No new review')), \
             patch.object(automatic_repair, 'repair', side_effect=AssertionError('No new repair')), \
             redirect_stdout(io.StringIO()):
            resumed = runner.run_job(self.group, self.args, self.pipeline)
        self.assertTrue(resumed['resumed'])
        self.assert_stagnated(resumed)
        self.assertEqual(first['artifacts'], resumed['artifacts'])

    def test_full_bundle_path_respects_same_stagnation_guard(self):
        self.args.pipeline_options['repair_mode'] = 'full_bundle'
        self.args.pipeline_options['repair_stagnation_patience'] = 1
        original = self.review
        def full_review(bundle, directory, **kwargs):
            report = original(bundle, directory, **kwargs)
            report['revised_bundle']['shared_scene']['render']['exposure'] += .1
            atomic_json(directory / 'review.json', report)
            return report
        self.review = full_review
        result = self.run_job()
        self.assert_stagnated(result, count=2)
        self.assertEqual(self.repaired, [])


if __name__ == '__main__':
    unittest.main()
