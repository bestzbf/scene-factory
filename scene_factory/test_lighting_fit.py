"""Deterministic safeguards for bounded shared lighting fitting."""
import copy
import json
import unittest

from lighting_fit import (MIN_MULTIPLIER, MAX_LIGHTING_CANDIDATES,
                          apply_lighting_controls, candidate_multipliers,
                          score_spatial_candidate, select_lighting_candidate,
                          select_lighting_candidate_fail_closed,
                          select_feasible_lighting_candidate,
                          calibration_status_after_fit)


def grid(value=.2):
    return {'status': 'available', 'regions': [{'id': str(index), 'p65': value} for index in range(9)]}


class LightingFitTests(unittest.TestCase):
    def test_equal_view_score_is_zero_and_rejects_missing_cells(self):
        reference = {'front': grid(.2), 'side': grid(.4)}
        self.assertEqual(score_spatial_candidate(reference, copy.deepcopy(reference))['loss'], 0)
        bad = {'front': grid(.2), 'side': {'status': 'available', 'regions': []}}
        with self.assertRaises(ValueError):
            score_spatial_candidate(reference, bad)

    def test_flat_brightness_match_still_penalizes_missing_contrast(self):
        reference = {'a': grid(.1)}
        reference['a']['regions'][4]['p65'] = .8
        flat = {'a': grid(.1)}
        score = score_spatial_candidate(reference, flat)
        self.assertGreater(score['per_view']['a']['contrast_loss'], 0)
        self.assertGreater(score['loss'], score['per_view']['a']['brightness_loss'])

    def test_worst_third_prevents_sacrificing_large_dark_or_bright_regions(self):
        # Six exact cells must not outweigh three cells four stops wrong when
        # a candidate can keep all nine within two stops. Use both signs so
        # the fix never encodes a bright-window/dark-room assumption.
        for sign in (-1, 1):
            reference = {'a': grid(.025)}
            sacrificed, balanced = {'a': grid(.025)}, {'a': grid(.025 * 2 ** (sign * 2))}
            for index in range(3):
                sacrificed['a']['regions'][index]['p65'] *= 2 ** (sign * 4)
            poor = score_spatial_candidate(reference, sacrificed)
            fair = score_spatial_candidate(reference, balanced)
            self.assertLess(fair['loss'], poor['loss'])
            self.assertEqual(poor['per_view']['a']['worst_third_squared_error'], 16.)
            self.assertEqual(len(poor['per_view']['a']['severe_residual_region_ids']), 3)

    def test_uninformative_view_prevents_shared_fit(self):
        reference = {'a': grid(0.), 'b': grid(.2)}
        measurement = score_spatial_candidate(reference, {'a': grid(.3), 'b': grid(.2)})
        self.assertIsNone(measurement['loss'])
        self.assertEqual(select_lighting_candidate([{'measurement': measurement}]), 0)

    def test_shared_selection_requires_improvement_and_view_guard(self):
        baseline = {'measurement': {'loss': 1., 'per_view': {'a': {'loss': 1.}, 'b': {'loss': 1.}}}}
        small = {'measurement': {'loss': .99, 'per_view': {'a': {'loss': .99}, 'b': {'loss': .99}}}}
        good = {'measurement': {'loss': .8, 'per_view': {'a': {'loss': .8}, 'b': {'loss': .8}}}}
        self.assertEqual(select_lighting_candidate([baseline, small]), 0)
        self.assertEqual(select_lighting_candidate([baseline, good]), 1)
        regression = {'measurement': {'loss': .5, 'per_view': {'a': {'loss': .5}, 'b': {'loss': 1.2}}}}
        self.assertEqual(select_lighting_candidate([baseline, regression]), 0)

    def test_fail_closed_restores_baseline_when_power_limit_and_severe_residual_remain(self):
        baseline = {'multipliers': {'interior': 1., 'world': 1.}, 'correction_ev': 0.,
                    'measurement': {'loss': 10., 'per_view': {'front': {
                        'loss': 10., 'severe_residual_region_ids': ['upper_left']}}}}
        saturated = {'multipliers': {'interior': 16., 'world': 2.}, 'correction_ev': 0.,
                     'measurement': {'loss': 8., 'per_view': {'front': {
                         'loss': 8., 'severe_residual_region_ids': ['upper_left']}}}}
        selected, reason = select_lighting_candidate_fail_closed([
            {'measurement': baseline['measurement'], **baseline},
            {'measurement': saturated['measurement'], **saturated}])
        self.assertEqual(selected, 0)
        self.assertEqual(reason['code'], 'boundary_saturated_with_severe_spatial_residuals')
        self.assertEqual(reason['action'], 'restored_original_lighting_and_exposure')
        self.assertEqual(reason['severe_residual_region_ids_by_view'], {'front': ['upper_left']})
        self.assertEqual(reason['saturated_controls'][0]['control'], 'interior')

    def test_fail_closed_uses_scored_two_stop_residual_threshold(self):
        reference = {'front': grid(.4)}
        baseline_stats = {'front': grid(.004)}
        improved_but_severe_stats = {'front': grid(.025)}
        baseline_score = score_spatial_candidate(reference, baseline_stats)
        bounded_score = score_spatial_candidate(reference, improved_but_severe_stats)
        self.assertLess(bounded_score['loss'], baseline_score['loss'])
        self.assertTrue(bounded_score['per_view']['front']['severe_residual_region_ids'])
        selected, reason = select_lighting_candidate_fail_closed([
            {'measurement': baseline_score, 'multipliers': {'interior': 1.}, 'correction_ev': 0.},
            {'measurement': bounded_score, 'multipliers': {'interior': 16.}, 'correction_ev': 0.}])
        self.assertEqual(selected, 0)
        self.assertEqual(reason['threshold_stops'], 2.0)

    def test_fail_closed_also_catches_exposure_limit(self):
        candidate = {'multipliers': {'interior': 2.}, 'correction_ev': 1.5,
                     'measurement': {'loss': 8., 'per_view': {'front': {
                         'loss': 8., 'severe_residual_region_ids': ['middle_right']}}}}
        selected, reason = select_lighting_candidate_fail_closed([candidate])
        self.assertEqual(selected, 0)
        self.assertEqual(reason['saturated_controls'][0]['control'], 'exposure_correction_ev')

    def test_fail_closed_keeps_saturated_fit_if_no_severe_residual_remains(self):
        baseline = {'multipliers': {'interior': 1.}, 'correction_ev': 0.,
                    'measurement': {'loss': 10., 'per_view': {'front': {
                        'loss': 10., 'severe_residual_region_ids': ['upper_left']}}}}
        good = {'multipliers': {'interior': 16.}, 'correction_ev': 1.5,
                'measurement': {'loss': 1., 'per_view': {'front': {
                    'loss': 1., 'severe_residual_region_ids': []}}}}
        selected, reason = select_lighting_candidate_fail_closed([
            {'measurement': baseline['measurement'], **baseline},
            {'measurement': good['measurement'], **good}])
        self.assertEqual(selected, 1)
        self.assertIsNone(reason)

    def test_fail_closed_keeps_non_saturated_fit_and_saturated_fit_without_severe_error(self):
        base_measurement = {'loss': 10., 'per_view': {'front': {
            'loss': 10., 'severe_residual_region_ids': ['upper_left']}}}
        baseline = {'measurement': base_measurement}
        non_saturated = {'multipliers': {'interior': 4.}, 'correction_ev': .5,
                         'measurement': {'loss': 8., 'per_view': {'front': {
                             'loss': 8., 'severe_residual_region_ids': ['upper_left']}}}}
        selected, reason = select_lighting_candidate_fail_closed([baseline, non_saturated])
        self.assertEqual(selected, 1)
        self.assertIsNone(reason)

        bounded_no_severe = {'multipliers': {'interior': 16.}, 'correction_ev': 0.,
                             'measurement': {'loss': 7., 'per_view': {'front': {
                                 'loss': 7., 'severe_residual_region_ids': []}}}}
        selected, reason = select_lighting_candidate_fail_closed([baseline, bounded_no_severe])
        self.assertEqual(selected, 1)
        self.assertIsNone(reason)

    def test_boundary_rejection_selects_measured_fallback_against_original_scene(self):
        def measured(loss, ev, powers=1.):
            return {'multipliers': {'interior': powers}, 'correction_ev': ev,
                    'measurement': {'loss': loss, 'per_view': {'front': {
                        'loss': loss, 'severe_residual_region_ids': ['lower_left']}}}}
        # Original is worse than the exposure-only baseline. A safe fallback
        # may be slightly worse than that rejected, boundary-limited baseline.
        baseline = measured(.90, -1.5)
        baseline['_initial_exposure_variant'] = measured(1.4, 0.)
        fallback, worse = measured(.94, -.75, .5), measured(.96, .5, .25)
        candidates = [baseline, worse, fallback]
        unchanged = copy.deepcopy(candidates)
        selected, reason = select_lighting_candidate_fail_closed(candidates)
        self.assertEqual(selected, 2)
        self.assertEqual(reason['fallback_selection']['baseline'], 'original_powers_and_exposure')
        self.assertEqual(reason['fallback_selection']['baseline_loss'], 1.4)
        self.assertFalse(reason['fallback_selection']['per_view_degradation_allowed'])
        self.assertEqual(reason['original_rejection_reason']['action'],
                         'restored_original_lighting_and_exposure')
        self.assertEqual(reason['severe_residual_region_ids_by_view'], {'front': ['lower_left']})
        self.assertEqual(candidates, unchanged)
        json.dumps(reason, allow_nan=False)

    def test_fallback_rejects_view_regression_even_when_aggregate_improves(self):
        original = {'measurement': {'loss': 1., 'per_view': {'a': {'loss': 1.}, 'b': {'loss': 1.}}}}
        candidate = {'multipliers': {'interior': 1.}, 'correction_ev': .5,
                     'measurement': {'loss': .6, 'per_view': {'a': {'loss': .19}, 'b': {'loss': 1.01}}}}
        self.assertIsNone(select_feasible_lighting_candidate([candidate], original))
        candidate['measurement']['per_view']['b']['loss'] = 1.
        self.assertEqual(select_feasible_lighting_candidate([candidate], original), 0)

    def test_fallback_requires_two_percent_improvement_and_complete_finite_measurement(self):
        original = {'measurement': {'loss': 1., 'per_view': {'front': {'loss': 1.}}}}
        candidate = {'multipliers': {'interior': 1.}, 'correction_ev': .5,
                     'measurement': {'loss': .98, 'per_view': {'front': {'loss': .98}}}}
        self.assertEqual(select_feasible_lighting_candidate([candidate], original), 0)
        for loss in (.98001, None, float('nan'), float('inf'), True):
            value = copy.deepcopy(candidate)
            value['measurement']['loss'] = loss
            self.assertIsNone(select_feasible_lighting_candidate([value], original))
        value = copy.deepcopy(candidate)
        value['measurement']['per_view'] = {'other': {'loss': .1}}
        self.assertIsNone(select_feasible_lighting_candidate([value], original))
        self.assertIsNone(select_feasible_lighting_candidate([candidate], None))
        self.assertIsNone(select_feasible_lighting_candidate([candidate], {'measurement': None}))
        value = copy.deepcopy(candidate)
        value['measurement'] = None
        self.assertIsNone(select_feasible_lighting_candidate([value], original))

    def test_fallback_requires_every_baseline_light_control(self):
        original = {'measurement': {'loss': 1., 'per_view': {'front': {'loss': 1.}}}}
        baseline = {'multipliers': {'interior': 1., 'world': 1.}, 'correction_ev': -1.5,
                    'measurement': {'loss': .5, 'per_view': {'front': {'loss': .5}}}}
        for powers in ({'interior': .5}, {'interior': .5, 'world': 1., 'extra': 1.}):
            candidate = copy.deepcopy(baseline)
            candidate['multipliers'], candidate['correction_ev'] = powers, 0.
            self.assertIsNone(select_feasible_lighting_candidate([baseline, candidate], original))

    def test_fallback_excludes_all_power_and_exposure_bounds(self):
        original = {'measurement': {'loss': 1., 'per_view': {'front': {'loss': 1.}}}}
        candidate = {'multipliers': {'interior': 1.}, 'correction_ev': .5,
                     'measurement': {'loss': .5, 'per_view': {'front': {'loss': .5}}}}
        for ev, power in ((-1.5, 1.), (1.5, 1.), (0., MIN_MULTIPLIER), (0., 16.),
                          (0., 17.), (float('nan'), 1.), (0., float('nan'))):
            value = copy.deepcopy(candidate)
            value['correction_ev'], value['multipliers']['interior'] = ev, power
            self.assertIsNone(select_feasible_lighting_candidate([value], original))

    def test_boundary_rejection_without_safe_fallback_preserves_original_rollback(self):
        original = {'measurement': {'loss': 1., 'per_view': {'front': {'loss': 1.}}}}
        boundary = {'multipliers': {'interior': 16.}, 'correction_ev': 0.,
                    'measurement': {'loss': .5, 'per_view': {'front': {
                        'loss': .5, 'severe_residual_region_ids': ['upper_left']}}},
                    '_initial_exposure_variant': original}
        selected, reason = select_lighting_candidate_fail_closed([boundary])
        self.assertEqual(selected, 0)
        self.assertEqual(reason['action'], 'restored_original_lighting_and_exposure')
        self.assertNotIn('fallback_selection', reason)

    def test_proposals_are_bounded_deterministic_and_do_not_mutate(self):
        current = {'interior': 1., 'world': 1.}
        proposals = list(candidate_multipliers(['interior', 'world'], current, global_stage=True))
        self.assertEqual(current, {'interior': 1., 'world': 1.})
        self.assertEqual(len(proposals), 4)
        for proposal in proposals:
            self.assertTrue(all(MIN_MULTIPLIER <= value <= 16 for value in proposal.values()))
        self.assertEqual(proposals, list(candidate_multipliers(['interior', 'world'], current, global_stage=True)))

    def test_control_application_records_and_rolls_back_values(self):
        class Data:
            def __init__(self, energy): self.energy, self.type = energy, "AREA"
        light = Data(400.)
        class Socket:
            def __init__(self, value): self.default_value = value
        world = Socket(.2)
        controls = {'interior': [{'kind': 'light', 'name': 'ceiling', 'data': light, 'initial': 400.}],
                    'world': [{'kind': 'world', 'name': 'Background', 'socket': world, 'initial': .2}]}
        original = (light.energy, world.default_value)
        report = apply_lighting_controls(controls, {'interior': .25, 'world': 2.})
        self.assertEqual((light.energy, world.default_value), (100., .4))
        self.assertEqual(len(report), 2)
        apply_lighting_controls(controls, {'interior': 1., 'world': 1.})
        self.assertEqual((light.energy, world.default_value), original)
        json.dumps(report, allow_nan=False)

    def test_candidate_budget_is_explicit(self):
        self.assertLessEqual(MAX_LIGHTING_CANDIDATES, 20)

    def test_calibration_status_exposes_safe_revert(self):
        self.assertEqual(calibration_status_after_fit('selected', 'selected'), 'lighting_selected')
        self.assertEqual(calibration_status_after_fit('selected', 'reverted_unresolved'),
                         'lighting_reverted_unresolved')
        self.assertEqual(calibration_status_after_fit('selected', 'selected_feasible_unresolved'),
                         'lighting_selected_feasible_unresolved')
        self.assertEqual(calibration_status_after_fit('unchanged', 'unchanged'), 'unchanged')


if __name__ == '__main__':
    unittest.main()
