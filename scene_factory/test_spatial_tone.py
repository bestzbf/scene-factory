"""Local brightness evidence that whole-image exposure statistics cannot see."""
import copy
import json
import unittest

import numpy as np

from reference_tone import (compute_spatial_stats, compare_spatial_stats,
                            compute_stats, choose_shared_exposure)


def image_from_rows(values, block=4):
    rows = [np.full((block, block * 3, 3), v, dtype=np.uint8) for v in values]
    return np.concatenate(rows)


class SpatialToneTests(unittest.TestCase):
    def test_swapped_regions_have_same_histogram_but_different_lighting(self):
        reference = image_from_rows([210, 128, 40])
        rendered = reference[::-1].copy()
        global_reference, global_render = compute_stats(reference), compute_stats(rendered)
        self.assertEqual(global_reference['quantiles'], global_render['quantiles'])
        global_report = choose_shared_exposure([{'ev': 0, 'views': {'a': global_render}}], {'a': global_reference})
        self.assertEqual(global_report['selected_robust_loss'], 0)
        before = rendered.copy()
        ref_grid, render_grid = compute_spatial_stats(reference), compute_spatial_stats(rendered)
        original_stats = copy.deepcopy(render_grid)
        report = compare_spatial_stats({'a': ref_grid}, {'a': render_grid})
        self.assertTrue(report['requires_inspection'])
        self.assertIn('a:nonuniform_spatial_brightness', report['reasons'])
        rows = report['per_view']['a']['regions']
        self.assertLess(rows[0]['log2_luminance_error'], -1)
        self.assertGreater(rows[-1]['log2_luminance_error'], 1)
        np.testing.assert_array_equal(rendered, before)
        self.assertEqual(render_grid, original_stats)
        json.dumps(report, allow_nan=False)

    def test_normalized_regions_accept_different_resolutions(self):
        reference = image_from_rows([210, 128, 40], block=2)
        rendered = image_from_rows([210, 128, 40], block=9)
        report = compare_spatial_stats({'a': compute_spatial_stats(reference)},
                                       {'a': compute_spatial_stats(rendered)})
        self.assertFalse(report['requires_inspection'])
        self.assertEqual(report['per_view']['a']['median_absolute_error_stops'], 0)

    def test_bottom_up_buffer_keeps_top_left_region_labels(self):
        top_down = image_from_rows([210, 128, 40])
        self.assertEqual(compute_spatial_stats(top_down),
                         compute_spatial_stats(top_down[::-1], bottom_up=True))
        grid = compute_spatial_stats(top_down[::-1], bottom_up=True)
        self.assertEqual(grid['regions'][0]['id'], 'upper_left')
        self.assertGreater(grid['regions'][0]['p65'], grid['regions'][-1]['p65'])

    def test_tiny_images_skip_without_empty_quantile_errors(self):
        a, b = compute_spatial_stats(np.zeros((2, 3, 3))), compute_spatial_stats(np.ones((12, 12, 3)))
        self.assertEqual(a['status'], 'skipped_small_image')
        report = compare_spatial_stats({'a': a}, {'a': b})
        self.assertEqual(report['per_view']['a']['status'], 'skipped_small_image')
        self.assertFalse(report['requires_inspection'])

    def test_bound_requires_inspection_even_for_uniform_grid(self):
        grid = compute_spatial_stats(image_from_rows([128, 128, 128]))
        report = compare_spatial_stats({'a': grid}, {'a': grid}, at_correction_bound=True)
        self.assertEqual(report['reasons'], ['exposure_correction_at_bound'])
        self.assertTrue(report['requires_inspection'])

    def test_global_residual_not_misdiagnosed_as_nonuniform(self):
        a = compute_spatial_stats(image_from_rows([128, 128, 128]))
        b = compute_spatial_stats(image_from_rows([40, 40, 40]))
        report = compare_spatial_stats({'a': a}, {'a': b})
        self.assertFalse(report['per_view']['a']['nonuniform_residual'])
        self.assertEqual(report['reasons'], ['a:large_spatial_brightness_residual'])

    def test_black_and_clipped_reference_cells_are_explicitly_unusable(self):
        for value in (0, 255):
            a = compute_spatial_stats(image_from_rows([value] * 3))
            b = compute_spatial_stats(image_from_rows([128] * 3))
            report = compare_spatial_stats({'a': a}, {'a': b})
            self.assertEqual(report['per_view']['a']['status'], 'insufficient_information')
            self.assertEqual(report['per_view']['a']['usable_region_count'], 0)
            self.assertIsNone(report['per_view']['a']['regions'][0]['log2_luminance_error'])
            json.dumps(report, allow_nan=False)

    def test_invalid_view_sets_or_pixels_fail(self):
        grid = compute_spatial_stats(image_from_rows([128] * 3))
        with self.assertRaises(ValueError):
            compare_spatial_stats({'a': grid}, {'b': grid})
        with self.assertRaises(ValueError):
            compute_spatial_stats(np.ones((12, 12, 3)) * np.nan)
        with self.assertRaises(ValueError):
            compute_spatial_stats(np.ones((12, 12, 3)), bottom_up='yes')


if __name__ == '__main__':
    unittest.main()
