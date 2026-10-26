"""Numerical and contract tests for shared reference exposure metering."""
import copy
import json
import unittest

import numpy as np

from reference_tone import choose_shared_exposure, compute_stats


def stats(values):
    return {'quantiles': dict(zip(('p50', 'p65', 'p80'), values))}


def candidate(ev, **views):
    return {'ev': ev, 'views': {key: stats(value) for key, value in views.items()}}


class ReferenceToneTests(unittest.TestCase):
    def test_rgb_decodes_display_transfer_before_luminance(self):
        result = compute_stats(np.full((2, 3, 3), .5, dtype=np.float64))
        self.assertAlmostEqual(result['quantiles']['p50'], .21404114048223255)
        self.assertEqual(result['pixel_count'], 6)
        self.assertEqual((result['width'], result['height']), (3, 2))
        red = compute_stats(np.array([[[1., 0., 0.]]]))
        self.assertAlmostEqual(red['quantiles']['p65'], .2126)
        json.dumps(result, allow_nan=False)

    def test_uint8_and_normalized_float_have_identical_statistics(self):
        image = np.arange(54, dtype=np.uint8).reshape(2, 9, 3) * 4
        self.assertEqual(compute_stats(image), compute_stats(image.astype(float) / 255))

    def test_rejects_ambiguous_or_invalid_rgb(self):
        cases = [np.zeros((0, 2, 3)), np.zeros((2, 2, 4)), np.zeros((2, 2)),
                 np.full((2, 2, 3), 255.), np.full((2, 2, 3), np.nan),
                 np.full((2, 2, 3), -.1), np.zeros((2, 2, 3), dtype=np.uint16)]
        for case in cases:
            with self.subTest(shape=case.shape, dtype=case.dtype), self.assertRaises(ValueError):
                compute_stats(case)

    def test_exact_multiview_match_selects_one_shared_offset(self):
        reference = {'a': stats([.1, .2, .3]), 'b': stats([.15, .25, .4])}
        candidates = [candidate(0, a=[.2, .4, .6], b=[.3, .5, .8]),
                      candidate(-1, a=[.1, .2, .3], b=[.15, .25, .4]),
                      candidate(-1.5, a=[.07, .14, .21], b=[.1, .17, .28])]
        original = copy.deepcopy(candidates)
        result = choose_shared_exposure(candidates, reference)
        self.assertEqual(result['correction_ev'], -1)
        self.assertEqual(result['selected_index'], 1)
        self.assertEqual(result['shared_across_views'], ['a', 'b'])
        self.assertEqual(result['selected_robust_loss'], 0)
        self.assertGreater(result['baseline_robust_loss'], 0)
        self.assertEqual(candidates, original)  # Metering never edits the input.
        json.dumps(result, allow_nan=False)

    def test_conflicting_views_keep_shared_compromise(self):
        reference = {'a': stats([.1, .1, .1]), 'b': stats([.1, .1, .1])}
        candidates = [candidate(-1, a=[.1]*3, b=[.025]*3),
                      candidate(0, a=[.2]*3, b=[.05]*3),
                      candidate(1, a=[.4]*3, b=[.1]*3)]
        result = choose_shared_exposure(candidates, reference)
        self.assertEqual(result['correction_ev'], 0)
        self.assertEqual(result['status'], 'unchanged')
        # Reference pixel counts must not let one higher-resolution view win.
        reference['a']['pixel_count'] = 10000000
        reference['b']['pixel_count'] = 100
        self.assertEqual(choose_shared_exposure(candidates, reference)['correction_ev'], 0)

    def test_window_highlight_and_black_tail_do_not_dominate(self):
        reference = np.full((10, 10, 3), 128, dtype=np.uint8)
        reference[:1] = 255
        altered_tail = np.full((10, 10, 3), 128, dtype=np.uint8)
        altered_tail[:1] = 0
        a, b = compute_stats(reference), compute_stats(altered_tail)
        self.assertNotEqual(a['mean_luminance'], b['mean_luminance'])
        self.assertEqual(a['quantiles'], b['quantiles'])
        result = choose_shared_exposure([{'ev': 0, 'views': {'a': b}},
                                         {'ev': -1, 'views': {'a': a}}], {'a': a})
        self.assertEqual(result['correction_ev'], 0)

    def test_clipped_or_dark_view_prevents_unjustified_adjustment(self):
        for value in (0., 1.):
            reference = {'a': stats([value]*3), 'b': stats([.1, .2, .3])}
            candidates = [candidate(0, a=[value]*3, b=[.2, .4, .6]),
                          candidate(-1, a=[value]*3, b=[.1, .2, .3])]
            result = choose_shared_exposure(candidates, reference)
            self.assertEqual(result['correction_ev'], 0)
            self.assertEqual(result['status'], 'insufficient_information')
            self.assertIsNone(result['selected_robust_loss'])
            json.dumps(result, allow_nan=False)

    def test_candidate_clipping_is_penalized_not_excluded(self):
        reference = {'a': stats([.15, .25, .4])}
        result = choose_shared_exposure([candidate(0, a=[.16, .26, .41]),
                                         candidate(1, a=[1, 1, 1]),
                                         candidate(-1, a=[0, 0, 0])], reference)
        self.assertEqual(result['correction_ev'], 0)
        self.assertTrue(all(item['robust_loss'] is not None for item in result['candidates']))

    def test_correction_bound_excludes_even_a_better_candidate(self):
        reference = {'a': stats([.1, .2, .3])}
        candidates = [candidate(0, a=[.4, .6, .8]),
                      candidate(-1.5, a=[.15, .25, .35]),
                      candidate(-2, a=[.1, .2, .3])]
        result = choose_shared_exposure(candidates, reference)
        self.assertEqual(result['correction_ev'], -1.5)
        self.assertFalse(result['candidates'][2]['eligible'])
        self.assertEqual(choose_shared_exposure(candidates, reference, max_correction_ev=0)['correction_ev'], 0)
        with self.assertRaises(ValueError):
            choose_shared_exposure(candidates, reference, max_correction_ev=2)

    def test_requires_complete_views_baseline_and_valid_quantiles(self):
        reference = {'a': stats([.1, .2, .3]), 'b': stats([.2, .3, .4])}
        invalid = [[], [candidate(-1, a=[.1, .2, .3], b=[.2, .3, .4])],
                   [candidate(0, a=[.1, .2, .3])],
                   [candidate(0, a=[.3, .2, .1], b=[.2, .3, .4])],
                   [candidate(0, a=[.1, .2, float('nan')], b=[.2, .3, .4])],
                   [candidate(0, a=[.1, .2, .3], b=[.2, .3, .4])]*2]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                choose_shared_exposure(records, reference)


if __name__ == '__main__':
    unittest.main()
