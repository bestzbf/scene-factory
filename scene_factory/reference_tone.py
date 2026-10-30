"""Conservative shared exposure selection from displayed reference/render RGB.

This module does not render, modify images or use Blender. Each candidate must
be produced by the SAME render/view transform with one scene-wide exposure EV
offset applied to every view. It selects an already-rendered candidate, never
predicts a different gamma, changes materials or assigns per-camera exposure.

``compute_stats(np.asarray(Image.open(path).convert('RGB')))`` reads uint8 sRGB
or normalized floating-point display RGB. ``choose_shared_exposure`` accepts
``[{'ev': -0.5, 'views': {'view_01': stats, ...}}, ...]`` and a reference stats
mapping with the same view IDs. Include an ev=0 candidate. EV is a correction
relative to the existing scene exposure, NOT an absolute exposure setting.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


QUANTILES = ('p50', 'p65', 'p80')
MAX_CORRECTION_EV = 1.5
SEVERE_SPATIAL_RESIDUAL_STOPS = 2.0
_LOW = .0001
_HIGH = .98
_HUBER_DELTA = .5
LIMITATIONS = [
    'Global tone matching only; this is not a reconstruction fidelity or visual acceptance score.',
    'Geometry, framing, object coverage and materials can bias whole-image luminance quantiles.',
    'References are assumed to be sRGB display images; their camera exposure, white balance and tone curve are unknown.',
    'One bounded correction is shared by all views; local lighting, gamma and materials are unchanged.',
]


def _linear_rgb(rgb):
    """Validate display pixels once and decode their sRGB transfer function."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
        raise ValueError('Expected a non-empty H x W x 3 display RGB array')
    if array.dtype == np.uint8:
        encoded = array.astype(np.float64) / 255
    elif np.issubdtype(array.dtype, np.floating):
        encoded = array.astype(np.float64, copy=False)
        if not np.all(np.isfinite(encoded)) or np.any(encoded < 0) or np.any(encoded > 1):
            raise ValueError('Floating display RGB must be finite and normalized to [0, 1]')
    else:
        raise ValueError('Display RGB must be uint8 or normalized floating point')
    return np.where(encoded <= .04045, encoded / 12.92,
                    ((encoded + .055) / 1.055) ** 2.4)


def compute_stats(rgb):
    """Return JSON-safe linear luminance statistics of sRGB display pixels.

    uint8 arrays are divided by 255. Floating arrays must already be in [0, 1].
    Alpha is intentionally not silently discarded: callers must composite or
    explicitly convert RGBA first. No bright-pixel mask removes a real window;
    only central quantiles contribute to exposure fitting.
    """
    linear = _linear_rgb(rgb)
    luminance = linear @ np.array([.2126, .7152, .0722])
    values = np.quantile(luminance, [.50, .65, .80])
    return {
        'quantiles': {key: float(value) for key, value in zip(QUANTILES, values)},
        'pixel_count': int(luminance.size),
        'width': int(linear.shape[1]), 'height': int(linear.shape[0]),
        'mean_luminance': float(luminance.mean()),
        'shadow_fraction': float(np.mean(luminance <= _LOW)),
        'highlight_fraction': float(np.mean(luminance >= _HIGH)),
        'luminance_space': 'sRGB EOTF decoded; Rec.709 luminance coefficients',
    }


def compute_spatial_stats(rgb, *, bottom_up=False):
    """Meter a fixed 3x3 grid, with normalized coordinates from the TOP left.

    These are diagnostics only, not exposure fitting inputs. Blender image
    pixels start at the bottom; callers using that buffer must set bottom_up.
    Different source/render resolutions are compared in normalized regions,
    never by matching individual pixels. Very small images skip this meter.
    """
    if type(bottom_up) is not bool:
        raise ValueError('bottom_up must be a boolean')
    linear = _linear_rgb(rgb)
    height, width = linear.shape[:2]
    result = {'status': 'available', 'grid': [3, 3], 'width': width, 'height': height,
              'coordinate_origin': 'top_left', 'statistic': 'p65_linear_luminance', 'regions': []}
    if min(height, width) < 6:
        result.update(status='skipped_small_image', minimum_dimension=6)
        return result
    if bottom_up:
        linear = linear[::-1]
    luminance = linear @ np.array([.2126, .7152, .0722])
    for row, vertical in enumerate(('upper', 'middle', 'lower')):
        y0, y1 = row * height // 3, (row + 1) * height // 3
        for column, horizontal in enumerate(('left', 'center', 'right')):
            x0, x1 = column * width // 3, (column + 1) * width // 3
            values = luminance[y0:y1, x0:x1]
            result['regions'].append({'id': vertical + '_' + horizontal,
                'normalized_bounds_xyxy': [column / 3, row / 3, (column + 1) / 3, (row + 1) / 3],
                'p65': float(np.quantile(values, .65)), 'pixel_count': int(values.size)})
    return result


def compare_spatial_stats(reference, rendered, *, at_correction_bound=False):
    """Explain local brightness residuals without changing render parameters.

    Positive log2 errors mean rendered regions are brighter. A large spread
    means one exposure shift cannot align all regions. The warning can also
    be caused by geometry, framing or materials, so it never diagnoses a light
    power error alone and never grants visual acceptance.
    """
    if not isinstance(reference, Mapping) or not isinstance(rendered, Mapping) or set(reference) != set(rendered) or not reference:
        raise ValueError('Spatial comparisons need the same nonempty view IDs')
    if type(at_correction_bound) is not bool:
        raise ValueError('at_correction_bound must be a boolean')
    per_view, reasons = {}, []
    if at_correction_bound:
        reasons.append('exposure_correction_at_bound')
    for view_id in sorted(reference):
        ref, render = reference[view_id], rendered[view_id]
        if ref['status'] != 'available' or render['status'] != 'available':
            per_view[view_id] = {'status': 'skipped_small_image', 'regions': []}
            continue
        ref_regions = {r['id']: r for r in ref['regions']}
        rendered_regions = {r['id']: r for r in render['regions']}
        if len(ref_regions) != 9 or set(ref_regions) != set(rendered_regions):
            raise ValueError('Spatial grids must contain the same nine regions')
        regions, errors = [], []
        for region in ref['regions']:
            reference_value, rendered_value = region['p65'], rendered_regions[region['id']]['p65']
            if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                   for v in (reference_value, rendered_value)):
                raise ValueError('Spatial luminance must be finite in [0, 1]')
            usable = _LOW < reference_value < _HIGH
            error = math.log2(max(rendered_value, _LOW) / reference_value) if usable else None
            if usable:
                errors.append(error)
            regions.append({'id': region['id'], 'normalized_bounds_xyxy': region['normalized_bounds_xyxy'],
                            'reference_p65': reference_value, 'render_p65': rendered_value,
                            'log2_luminance_error': error,
                            'status': 'compared' if usable else 'reference_near_black_or_clipped'})
        spread = float(np.quantile(errors, .9) - np.quantile(errors, .1)) if errors else None
        median_abs = float(np.median(np.abs(errors))) if errors else None
        nonuniform = len(errors) >= 3 and spread >= 1.5 and max(abs(e) for e in errors) >= 1.0
        large_residual = len(errors) >= 3 and median_abs >= 1.0
        severe_regions = [region['id'] for region in regions
                          if region['log2_luminance_error'] is not None
                          and abs(region['log2_luminance_error']) > SEVERE_SPATIAL_RESIDUAL_STOPS]
        if nonuniform:
            reasons.append(view_id + ':nonuniform_spatial_brightness')
        if large_residual:
            reasons.append(view_id + ':large_spatial_brightness_residual')
        if severe_regions:
            reasons.append(view_id + ':severe_regional_brightness_residual')
        per_view[view_id] = {'status': 'compared' if errors else 'insufficient_information',
                            'regions': regions, 'usable_region_count': len(errors),
                            'p90_minus_p10_error_stops': spread,
                            'median_absolute_error_stops': median_abs,
                            'max_absolute_region_error_stops': max(map(abs, errors)) if errors else None,
                            'severe_residual_region_ids': severe_regions,
                            'nonuniform_residual': nonuniform}
    return {'method': '3x3 normalized image grid; p65 sRGB-decoded linear luminance; log2(render/reference)',
            'coordinate_origin': 'top_left', 'positive_error_means': 'render_brighter_than_reference',
            'requires_inspection': bool(reasons), 'reasons': reasons, 'per_view': per_view,
            'thresholds': {'nonuniform_spread_stops': 1.5, 'large_region_error_stops': 1.0,
                           'severe_region_error_stops': SEVERE_SPATIAL_RESIDUAL_STOPS,
                           'large_median_absolute_error_stops': 1.0, 'minimum_usable_regions': 3},
            'guidance': 'Inspect window transmission, duplicate/occluding surfaces, light direction, material reflectance and framing before changing exposure or adding fill. A bounded global exposure adjustment cannot correct spatially different errors.',
            'limitations': 'Diagnostic only: grid regions are not semantic object masks or aligned geometry; framing/material differences can cause warnings. No pixel edits, lighting changes or visual acceptance are performed.'}


def _quantiles(stats, label):
    if not isinstance(stats, Mapping) or not isinstance(stats.get('quantiles'), Mapping):
        raise ValueError(f'{label}: missing quantiles mapping')
    result = {}
    for key in QUANTILES:
        value = stats['quantiles'].get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{label}: {key} must be a finite luminance in [0, 1]')
        result[key] = float(value)
    if list(result.values()) != sorted(result.values()):
        raise ValueError(f'{label}: luminance quantiles must be monotonic')
    return result


def _huber(error):
    magnitude = abs(error)
    return .5 * magnitude * magnitude if magnitude <= _HUBER_DELTA else _HUBER_DELTA * (magnitude - .5 * _HUBER_DELTA)


def choose_shared_exposure(candidates, reference_stats, max_correction_ev=MAX_CORRECTION_EV):
    """Choose one bounded exposure correction with equal weighting per view.

    Score the p50/p65/p80 reference and candidate quantiles in log2 luminance,
    using Huber loss (delta .5 EV) so one mismatched region has limited influence.
    Reference quantiles near black or clipped white are excluded. A candidate's
    black/white values are still penalized, never removed to improve its score.
    A reference view with no informative quantiles results in no correction.

    Every candidate must contain exactly the reference view IDs. Missing views,
    duplicate EVs and invalid numeric values raise instead of silently fitting
    another scene. Offsets outside the requested bound (never >1.5 EV) are
    recorded but not eligible. Exact score ties prefer the smallest correction.
    """
    if (isinstance(max_correction_ev, bool) or not isinstance(max_correction_ev, (int, float))
            or not math.isfinite(max_correction_ev) or not 0 <= max_correction_ev <= MAX_CORRECTION_EV):
        raise ValueError('max_correction_ev must be within [0, 1.5]')
    if not isinstance(reference_stats, Mapping) or not reference_stats:
        raise ValueError('Reference statistics need at least one named view')
    if not all(isinstance(key, str) and key for key in reference_stats):
        raise ValueError('Reference view IDs must be non-empty strings')
    view_ids = sorted(reference_stats)
    reference = {view: _quantiles(reference_stats[view], 'Reference ' + view) for view in view_ids}
    usable = {view: [key for key in QUANTILES if _LOW < reference[view][key] < _HIGH] for view in view_ids}
    rows, seen, zero_index = [], set(), None
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError('Each exposure candidate must be a mapping')
        ev = candidate.get('ev')
        if isinstance(ev, bool) or not isinstance(ev, (int, float)) or not math.isfinite(ev):
            raise ValueError('Candidate ev must be a finite correction in stops')
        ev = float(ev)
        if ev in seen:
            raise ValueError('Duplicate candidate EV: ' + str(ev))
        seen.add(ev)
        if ev == 0:
            zero_index = index
        if not isinstance(candidate.get('views'), Mapping) or set(candidate['views']) != set(view_ids):
            raise ValueError('Each candidate must contain exactly the reference view IDs')
        values = {view: _quantiles(candidate['views'][view], f'Candidate {ev} {view}') for view in view_ids}
        per_view = {}
        for view in view_ids:
            errors = [math.log2(max(values[view][key], _LOW) / reference[view][key]) for key in usable[view]]
            per_view[view] = {
                'quantiles_used': usable[view],
                'robust_loss': sum(_huber(error) for error in errors) / len(errors) if errors else None,
                'median_log2_luminance_error': float(np.median(errors)) if errors else None,
            }
        losses = [per_view[view]['robust_loss'] for view in view_ids]
        score = sum(losses) / len(losses) if all(loss is not None for loss in losses) else None
        rows.append({'candidate_index': index, 'ev': ev, 'eligible': abs(ev) <= max_correction_ev,
                     'robust_loss': score, 'per_view': per_view})
    if zero_index is None:
        raise ValueError('An ev=0 candidate is required for a safe unchanged baseline')
    eligible = [row for row in rows if row['eligible'] and row['robust_loss'] is not None]
    baseline = next(row for row in rows if row['candidate_index'] == zero_index)
    if eligible:
        best_score = min(row['robust_loss'] for row in eligible)
        tied = [row for row in eligible if abs(row['robust_loss'] - best_score) <= 1e-12]
        selected = min(tied, key=lambda row: (abs(row['ev']), row['ev']))
        status = 'unchanged' if selected['ev'] == 0 else 'selected'
    else:
        selected, status = baseline, 'insufficient_information'
    return {
        'correction_ev': selected['ev'],
        'selected_index': selected['candidate_index'],
        'status': status,
        'shared_across_views': view_ids,
        'max_correction_ev': float(max_correction_ev),
        'method': 'equal-view mean Huber loss of log2 p50/p65/p80 linear luminance; delta=0.5 stops',
        'baseline_robust_loss': baseline['robust_loss'],
        'selected_robust_loss': selected['robust_loss'],
        'candidates': rows,
        'limitations': LIMITATIONS[:],
    }
