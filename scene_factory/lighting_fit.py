"""Bounded, shared-scene lighting search using rendered display measurements.

The objective is a brightness/contrast fitting aid, not a reconstruction score.
No reference pixels are projected into the scene and no per-camera lights or
exposure values are created. Pure search/scoring helpers are usable without
Blender, including tests for rollback and multi-view fairness.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

MIN_MULTIPLIER = 1 / 64
MAX_MULTIPLIER = 16.0
MAX_LIGHTING_CANDIDATES = 15
MIN_RELATIVE_IMPROVEMENT = .02
MAX_VIEW_LOSS_INCREASE = .05
# Huber alone lets many correct cells outweigh a few seriously wrong cells.
# The worst-third quadratic term adds a spatial fairness cost without knowing
# which cells contain windows, furniture or walkable floor.
TAIL_LOSS_WEIGHT = .2
OBJECTIVE_VERSION = 'spatial_log_huber_contrast_tail_v2'
_LOW = .0001
_HIGH = .98
MAX_EXPOSURE_CORRECTION_EV = 1.5
SEVERE_RESIDUAL_STOPS = 2.0


def _huber(value, delta=.5):
    value = abs(value)
    return .5 * value * value if value <= delta else delta * (value - .5 * delta)


def _regions(stats):
    if not isinstance(stats, Mapping) or stats.get('status') != 'available':
        return None
    regions = stats.get('regions')
    if not isinstance(regions, list) or len(regions) != 9:
        raise ValueError('Lighting fitting requires nine normalized image regions')
    result = {}
    for region in regions:
        key, value = region.get('id'), region.get('p65')
        if (not isinstance(key, str) or key in result or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Lighting regions require unique IDs and finite luminance in [0, 1]')
        result[key] = float(value)
    return result


def score_spatial_candidate(reference, rendered):
    """Equal-view log luminance + spatial contrast loss; no image alteration.

    A mismatch in geometry/materials can also cause this loss. Every view needs
    at least three informative regions or no candidate may replace baseline.
    Relative-region contrasts prevent a flat image from winning solely by its
    median brightness. A quadratic penalty on the worst third of region errors
    prevents bounded-influence Huber fitting from sacrificing a substantial
    area to improve the majority. This treats bright and dark cells equally;
    no scene semantics or reference-position assumptions are used. Extreme
    reference black/white cells are unmetered.
    """
    if not isinstance(reference, Mapping) or not reference or set(reference) != set(rendered):
        raise ValueError('Lighting fitting needs the same nonempty view IDs')
    per_view = {}
    for view_id in sorted(reference):
        ref, image = _regions(reference[view_id]), _regions(rendered[view_id])
        if ref is None or image is None:
            per_view[view_id] = {'loss': None, 'usable_regions': 0}
            continue
        if set(ref) != set(image):
            raise ValueError('Reference/render lighting region IDs differ')
        usable = [key for key in ref if _LOW < ref[key] < _HIGH]
        if len(usable) < 3:
            per_view[view_id] = {'loss': None, 'usable_regions': len(usable)}
            continue
        errors = {key: math.log2(max(image[key], _LOW) / ref[key]) for key in usable}
        brightness = sum(_huber(value) for value in errors.values()) / len(errors)
        # Error differences are the error in regional brightness contrasts.
        # Use all region pairs so orientation and camera conventions do not
        # privilege any specific window/object location.
        contrast = [_huber(errors[a] - errors[b]) for i, a in enumerate(usable) for b in usable[i+1:]]
        contrast_loss = sum(contrast) / len(contrast)
        tail_count = max(1, math.ceil(len(errors) / 3))
        worst_errors = sorted((abs(value) for value in errors.values()), reverse=True)[:tail_count]
        tail_loss = sum(value * value for value in worst_errors) / tail_count
        per_view[view_id] = {'loss': brightness + .35 * contrast_loss + TAIL_LOSS_WEIGHT * tail_loss,
                             'brightness_loss': brightness, 'contrast_loss': contrast_loss,
                             'worst_third_squared_error': tail_loss,
                             'worst_third_region_count': tail_count,
                             'max_absolute_region_error_stops': worst_errors[0],
                             'severe_residual_region_ids': [key for key, value in errors.items()
                                                            if abs(value) > SEVERE_RESIDUAL_STOPS],
                             'usable_regions': len(usable), 'region_error_stops': errors}
    losses = [value['loss'] for value in per_view.values()]
    loss = sum(losses) / len(losses) if all(value is not None for value in losses) else None
    return {'objective_version': OBJECTIVE_VERSION, 'loss': loss, 'per_view': per_view}


def select_lighting_candidate(candidates):
    """Keep baseline unless a complete candidate makes a material improvement.

    Each candidate carries ``measurement`` from score_spatial_candidate. The
    baseline is index zero (its already-fitted exposure is retained). No view
    may get materially worse for an aggregate gain from another camera.
    """
    if not candidates:
        raise ValueError('An unchanged lighting baseline is required')
    baseline = candidates[0]['measurement']
    base_loss = baseline['loss']
    best = 0
    if base_loss is not None:
        for index, candidate in enumerate(candidates[1:], 1):
            measurement = candidate['measurement']
            loss = measurement['loss']
            if loss is None or set(measurement['per_view']) != set(baseline['per_view']):
                continue
            if any(value['loss'] is None or value['loss'] > baseline['per_view'][view]['loss'] * 1.05 + .005
                   for view, value in measurement['per_view'].items()):
                continue
            if loss > base_loss - max(.002, base_loss * MIN_RELATIVE_IMPROVEMENT):
                continue
            if loss < candidates[best]['measurement']['loss'] - 1e-12:
                best = index
    return best


def unresolved_saturated_fit_reason(candidate):
    """Explain when a bounded fit hit a limit but still has severe residuals.

    Such a candidate is evidence that global power/exposure cannot explain the
    image's spatial luminance pattern. Applying it can make a visibly broken
    scene worse, so the caller should restore the original powers and exposure.
    This pure helper also keeps the stop condition straightforward to test.
    """
    multipliers = candidate.get('multipliers', {})
    saturated_controls = [
        {'control': name, 'value': value, 'bounds': [MIN_MULTIPLIER, MAX_MULTIPLIER]}
        for name, value in sorted(multipliers.items())
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and (math.isclose(value, MIN_MULTIPLIER, rel_tol=0., abs_tol=1e-12)
             or math.isclose(value, MAX_MULTIPLIER, rel_tol=0., abs_tol=1e-12))
    ]
    correction_ev = candidate.get('correction_ev')
    if (isinstance(correction_ev, (int, float)) and not isinstance(correction_ev, bool)
            and math.isclose(abs(correction_ev), MAX_EXPOSURE_CORRECTION_EV,
                             rel_tol=0., abs_tol=1e-12)):
        saturated_controls.append({'control': 'exposure_correction_ev', 'value': correction_ev,
                                   'bounds': [-MAX_EXPOSURE_CORRECTION_EV,
                                              MAX_EXPOSURE_CORRECTION_EV]})

    per_view = candidate.get('measurement', {}).get('per_view', {})
    severe_regions = {
        view_id: sorted(value.get('severe_residual_region_ids', []))
        for view_id, value in sorted(per_view.items())
        if value.get('severe_residual_region_ids')
    }
    if not saturated_controls or not severe_regions:
        return None
    return {
        'code': 'boundary_saturated_with_severe_spatial_residuals',
        'message': ('The bounded global light/exposure fit reached a power or exposure limit '
                    'while one or more image regions remained more than two stops from reference. '
                    'The original light powers and exposure were restored.'),
        'saturated_controls': saturated_controls,
        'severe_residual_region_ids_by_view': severe_regions,
        'threshold_stops': SEVERE_RESIDUAL_STOPS,
        'action': 'restored_original_lighting_and_exposure',
    }


def select_lighting_candidate_fail_closed(candidates):
    """Reject unsafe fits, then try measured interior-of-bound alternatives.

    Fallback is allowed only against an explicitly retained measurement of
    the original scene, never the exposure-only baseline. It remains an
    unresolved result requiring independent review, not an acceptance gate.
    """
    proposed = select_lighting_candidate(candidates)
    reason = unresolved_saturated_fit_reason(candidates[proposed])
    if reason:
        original = candidates[0].get('_initial_exposure_variant')
        selected = select_feasible_lighting_candidate(candidates, original)
        if selected is not None:
            original_loss = original['measurement']['loss']
            return selected, {**reason,
                'original_rejection_reason': reason,
                'action': 'applied_measured_non_boundary_candidate_pending_review',
                'message': ('The boundary-saturated proposal was rejected. A previously measured '
                            'non-boundary candidate improves the original scene without degrading '
                            'any view. Remaining regional errors still require independent review.'),
                'fallback_selection': {'candidate_index': selected,
                    'baseline': 'original_powers_and_exposure',
                    'baseline_loss': original_loss,
                    'selected_loss': candidates[selected]['measurement']['loss'],
                    'minimum_relative_improvement': MIN_RELATIVE_IMPROVEMENT,
                    'per_view_degradation_allowed': False}}
    return (0 if reason else proposed), reason


def select_feasible_lighting_candidate(candidates, original):
    """Return the best already-measured, non-boundary improvement or None."""
    def finite_loss(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0)

    if (not isinstance(original, Mapping) or not candidates
            or not isinstance(candidates[0], Mapping)):
        return None
    baseline = original.get('measurement', {})
    expected_powers = candidates[0].get('multipliers')
    if (not isinstance(baseline, Mapping) or not isinstance(expected_powers, Mapping)
            or not expected_powers):
        return None
    base_loss, base_views = baseline.get('loss'), baseline.get('per_view')
    if (not finite_loss(base_loss) or base_loss <= 0
            or not isinstance(base_views, Mapping) or not base_views
            or any(not isinstance(view, Mapping) or not finite_loss(view.get('loss'))
                   for view in base_views.values())):
        return None
    best = None
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        powers, ev = candidate.get('multipliers'), candidate.get('correction_ev')
        if (not isinstance(powers, Mapping) or set(powers) != set(expected_powers)
                or not isinstance(ev, (int, float)) or isinstance(ev, bool)
                or not math.isfinite(ev) or abs(ev) >= MAX_EXPOSURE_CORRECTION_EV
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(value) or not MIN_MULTIPLIER < value < MAX_MULTIPLIER
                       for value in powers.values())):
            continue
        measurement = candidate.get('measurement', {})
        if not isinstance(measurement, Mapping):
            continue
        loss, per_view = measurement.get('loss'), measurement.get('per_view')
        if (not finite_loss(loss) or loss > base_loss * (1 - MIN_RELATIVE_IMPROVEMENT)
                or not isinstance(per_view, Mapping) or set(per_view) != set(base_views)
                or any(not isinstance(value, Mapping) or not finite_loss(value.get('loss'))
                       or value['loss'] > base_views[view_id]['loss'] + 1e-12
                       for view_id, value in per_view.items())):
            continue
        if best is None or loss < candidates[best]['measurement']['loss'] - 1e-12:
            best = index
    return best


def calibration_status_after_fit(exposure_status, fit_status):
    """Expose whether lighting fitting selected a fit or rejected it safely."""
    if fit_status == 'selected':
        return 'lighting_selected'
    if fit_status == 'reverted_unresolved':
        return 'lighting_reverted_unresolved'
    if fit_status == 'selected_feasible_unresolved':
        return 'lighting_selected_feasible_unresolved'
    return exposure_status


def candidate_multipliers(parameters, current, *, global_stage=False):
    """Yield deterministic bounded proposals, excluding the current settings."""
    if (not parameters or len(parameters) != len(set(parameters))
            or set(current) != set(parameters)):
        raise ValueError('Unique lighting parameter IDs must match multiplier values')
    for value in current.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not MIN_MULTIPLIER <= value <= MAX_MULTIPLIER:
            raise ValueError('Lighting multipliers are outside the bounded range')
    proposals = []
    if global_stage:
        for scale in (1/16, 1/4, 4, 16):
            proposals.append({key: max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, current[key]*scale)) for key in parameters})
    else:
        for key in parameters:
            for scale in (.25, 4):
                trial = dict(current)
                trial[key] = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, trial[key]*scale))
                proposals.append(trial)
    seen = {tuple(current[key] for key in parameters)}
    for trial in proposals:
        signature = tuple(trial[key] for key in parameters)
        if signature not in seen:
            seen.add(signature)
            yield trial


def collect_lighting_controls(scene):
    """Discover editable actual emitter powers; never infer from scene IDs.

    Group geometry-backed indoor fixtures separately from exterior proxies.
    Keep sun/directional lights separate from local sources. Linked/animated
    values remain unchanged instead of silently overriding a node graph.
    """
    from appearance_lighting import bind_fixture_power_drivers
    # Upgrade safely recognized legacy generated faces before any preview.
    # Drivers then follow every candidate and exception rollback directly.
    fixture_bindings = bind_fixture_power_drivers(scene)
    groups, ignored = {}, [dict(item, kind='fixture_appearance')
                          for item in fixture_bindings if item['status'] == 'skipped']
    seen = set()
    for obj in sorted(scene.objects, key=lambda item: item.name):
        if obj.type != 'LIGHT' or obj.hide_render or obj.data.energy <= 0:
            continue
        data = obj.data
        if data.as_pointer() in seen:
            continue
        seen.add(data.as_pointer())
        if data.library or data.animation_data:
            ignored.append({'light': obj.name, 'reason': 'linked_or_animated_power'})
            continue
        if data.use_nodes and data.node_tree and (data.node_tree.library or data.node_tree.animation_data):
            ignored.append({'light': obj.name, 'reason': 'linked_or_animated_light_node_tree'})
            continue
        if data.use_nodes and data.node_tree and any(node.type == 'EMISSION' and node.inputs['Strength'].is_linked for node in data.node_tree.nodes):
            ignored.append({'light': obj.name, 'reason': 'linked_emission_strength'})
            continue
        group = ('exterior' if obj.get('exterior_illumination_proxy') else
                 'directional' if data.type == 'SUN' else 'interior')
        groups.setdefault(group, []).append({'kind': 'light', 'name': obj.name,
            'data': data, 'initial': float(data.energy)})
    world = scene.world
    if world and world.use_nodes and not world.library and not world.animation_data:
        if world.node_tree.library or world.node_tree.animation_data:
            ignored.append({'world': world.name, 'reason': 'linked_or_animated_world_node_tree'})
            return groups, ignored
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND' and node.inputs['Strength'].default_value > 0:
                socket = node.inputs['Strength']
                if not socket.is_linked:
                    groups.setdefault('world', []).append({'kind': 'world', 'name': node.name,
                        'socket': socket, 'initial': float(socket.default_value)})
    return groups, ignored


def apply_lighting_controls(groups, multipliers):
    if set(groups) != set(multipliers):
        raise ValueError('Lighting control groups do not match multipliers')
    applied = []
    for group in sorted(groups):
        factor = multipliers[group]
        if not MIN_MULTIPLIER <= factor <= MAX_MULTIPLIER or not math.isfinite(factor):
            raise ValueError('Lighting multiplier outside bounds')
        for control in groups[group]:
            value = control['initial'] * factor
            if control['kind'] == 'light':
                control['data'].energy = value
                value = float(control['data'].energy)
            else:
                control['socket'].default_value = value
                value = float(control['socket'].default_value)
            applied.append({'group': group, 'kind': control['kind'], 'name': control['name'],
                'initial_value': control['initial'], 'multiplier': factor, 'applied_value': value,
                'unit': 'W' if control['kind'] == 'light' and control['data'].type != 'SUN' else 'strength'})
    return applied


def fit_rendered_lighting(scene, views, cameras, sources, set_view, reference_grid,
                          baseline_exposure_ev, initial_exposure, read_display_rgb):
    """Render and select bounded shared powers, restoring all preview settings.

    ``baseline_exposure_ev`` is the conservative exposure-only fit. Every
    changed lighting setting is actually rendered in Cycles for every view;
    only inexpensive color-management EV variants reuse its Render Result.
    At most MAX_LIGHTING_CANDIDATES settings (including baseline) are rendered.
    The final render still requires independent visual review.
    """
    import hashlib
    import tempfile
    from pathlib import Path
    import bpy
    from reference_tone import compute_stats, compute_spatial_stats

    groups, ignored = collect_lighting_controls(scene)
    limitations = [
        'Lighting fit only; no visual acceptance or geometry/material correctness is implied.',
        'Spatial cells are not semantic masks; framing and reflectance can bias the brightness objective.',
        'Light directions, colors, emitter sizes, geometry and material reflectance remain unchanged.',
        'Visible exterior appearance is separate from incident illumination; raising power does not add exterior scenery.',
        'World background strength may affect both visible environment and indirect illumination.',
    ]
    if not groups:
        return {'status': 'skipped_no_editable_emitters', 'ignored_controls': ignored,
                'candidate_count': 0, 'applied_settings': [], 'limitations': limitations}, None
    original = {'camera': scene.camera, 'samples': scene.cycles.samples,
                'x': scene.render.resolution_x, 'y': scene.render.resolution_y,
                'percentage': scene.render.resolution_percentage,
                'exposure': scene.view_settings.exposure, 'filepath': scene.render.filepath,
                'seed': scene.cycles.seed, 'animated_seed': scene.cycles.use_animated_seed,
                'adaptive': scene.cycles.use_adaptive_sampling,
                'denoising': scene.cycles.use_denoising,
                'border': scene.render.use_border}
    parameters = sorted(groups)
    unchanged = {key: 1.0 for key in parameters}
    records, measurements = [], []
    seen = set()
    completed = False
    selected = 0
    applied = []
    offsets = [round(i * .25, 2) for i in range(-6, 7)]
    try:
        scene.cycles.samples = min(16, original['samples'])
        scene.cycles.seed = 41
        scene.cycles.use_animated_seed = False
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = True
        scene.render.resolution_percentage = 100
        scene.render.use_border = False
        with tempfile.TemporaryDirectory(prefix='scene-light-fit-') as folder:
            path = str(Path(folder) / 'preview.png')
            def evaluate(multipliers, baseline=False):
                apply_lighting_controls(groups, multipliers)
                exposures = list(dict.fromkeys([baseline_exposure_ev, 0.0])) if baseline else offsets
                candidates = [{'ev': ev, 'views': {}, 'grid_views': {}} for ev in exposures]
                for view in views:
                    set_view(scene, cameras[view['id']], view, 256)
                    scene.view_settings.exposure = initial_exposure
                    bpy.ops.render.render()
                    result = bpy.data.images['Render Result']
                    for candidate in candidates:
                        scene.view_settings.exposure = initial_exposure + candidate['ev']
                        result.save_render(path, scene=scene)
                        rgb = read_display_rgb(path)
                        candidate['views'][view['id']] = compute_stats(rgb)
                        candidate['grid_views'][view['id']] = compute_spatial_stats(rgb, bottom_up=True)
                for candidate in candidates:
                    candidate['measurement'] = score_spatial_candidate(reference_grid, candidate['grid_views'])
                valid = [candidate for candidate in candidates if candidate['measurement']['loss'] is not None]
                if baseline:
                    # Candidate zero is the conservative exposure-only fit
                    # used as the lighting-search baseline. The extra
                    # initial-exposure render is retained only for an exact
                    # fail-closed rollback and must not change selection.
                    best = next((candidate for candidate in candidates
                                 if candidate['ev'] == baseline_exposure_ev), candidates[0])
                else:
                    best = min(valid, key=lambda candidate: (candidate['measurement']['loss'],
                        abs(candidate['ev'] - baseline_exposure_ev), abs(candidate['ev']))) if valid else candidates[0]
                record = {'candidate_index': len(records), 'multipliers': dict(multipliers),
                    'correction_ev': best['ev'], 'ev': best['ev'],
                    'measurement': best['measurement'],
                    'preview_stats': best['views'], 'preview_grid': best['grid_views'],
                    'exposure_candidates': [{'ev': candidate['ev'], 'spatial_loss': candidate['measurement']['loss']}
                                            for candidate in candidates]}
                if baseline:
                    # Retain the exact initial-exposure measurement for a
                    # possible fail-closed rollback after the search completes.
                    record['_initial_exposure_variant'] = next(
                        candidate for candidate in candidates if candidate['ev'] == 0.0)
                records.append(record)
                measurements.append(best)
                seen.add(tuple(multipliers[key] for key in parameters))
                return record

            evaluate(unchanged, baseline=True)
            for proposal in candidate_multipliers(parameters, unchanged, global_stage=True):
                evaluate(proposal)
            selected = select_lighting_candidate(records)
            for stage in ('coarse', 'fine'):
                current = dict(records[selected]['multipliers'])
                proposals = list(candidate_multipliers(parameters, current)) if stage == 'coarse' else []
                if stage == 'fine':
                    for key in parameters:
                        for factor in (.5, 2):
                            trial = dict(current)
                            trial[key] = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, current[key] * factor))
                            proposals.append(trial)
                for proposal in proposals:
                    if len(records) >= MAX_LIGHTING_CANDIDATES:
                        break
                    signature = tuple(proposal[key] for key in parameters)
                    if signature in seen:
                        continue
                    evaluate(proposal)
                selected = select_lighting_candidate(records)
        searched_selected = selected
        selected, unresolved_reason = select_lighting_candidate_fail_closed(records)
        feasible_fallback = bool(unresolved_reason and unresolved_reason.get('fallback_selection'))
        if unresolved_reason and not feasible_fallback:
            # A power/exposure limit with severe remaining regional error is
            # not a successful global lighting fit. Restore the exact starting
            # scene state, including the original exposure rather than the
            # intermediate exposure-only correction.
            selected = 0
            applied_candidate = dict(records[0]['_initial_exposure_variant'])
            applied_candidate['multipliers'] = dict(unchanged)
            applied_candidate['candidate_index'] = 0
            applied = apply_lighting_controls(groups, unchanged)
            scene.view_settings.exposure = initial_exposure
        else:
            applied_candidate = records[selected]
            applied = apply_lighting_controls(groups, applied_candidate['multipliers'])
            scene.view_settings.exposure = initial_exposure + applied_candidate.get(
                'ev', applied_candidate['correction_ev'])
        completed = True
    finally:
        scene.camera = original['camera']
        scene.cycles.samples = original['samples']
        scene.cycles.seed = original['seed']
        scene.cycles.use_animated_seed = original['animated_seed']
        scene.cycles.use_adaptive_sampling = original['adaptive']
        scene.cycles.use_denoising = original['denoising']
        scene.render.resolution_x, scene.render.resolution_y = original['x'], original['y']
        scene.render.resolution_percentage = original['percentage']
        scene.render.use_border = original['border']
        scene.render.filepath = original['filepath']
        if not completed:
            apply_lighting_controls(groups, unchanged)
            scene.view_settings.exposure = original['exposure']
        # Evaluate the direct fixture-energy drivers after the selected power
        # or exception rollback, including before the .blend is saved.
        bpy.context.view_layer.update()
    baseline_loss = (records[0]['_initial_exposure_variant']['measurement']['loss']
                     if unresolved_reason else records[0]['measurement']['loss'])
    final_loss = applied_candidate['measurement']['loss']
    report = {'status': ('selected_feasible_unresolved' if feasible_fallback else
                         'reverted_unresolved' if unresolved_reason else
                         'selected' if selected else 'unchanged'),
              'method': 'actual Cycles render search; equal-view 3x3 log luminance Huber + .35 regional contrast + .2 worst-third squared residual',
              'objective_version': OBJECTIVE_VERSION,
              'tail_loss_weight': TAIL_LOSS_WEIGHT,
              'severe_region_residuals': {view_id: value.get('severe_residual_region_ids', [])
                  for view_id, value in applied_candidate['measurement']['per_view'].items()},
              'selected_power_at_bound': any(value in (MIN_MULTIPLIER, MAX_MULTIPLIER)
                  for value in applied_candidate['multipliers'].values()),
              'requires_non_power_inspection': any(value.get('severe_residual_region_ids')
                  for value in applied_candidate['measurement']['per_view'].values()),
              'unresolved_reason': unresolved_reason,
              'feasible_fallback_selected': feasible_fallback,
              'residual_guidance': 'Severe residuals after fitting are not evidence for another global power increase. Inspect camera framing/visible coverage, light direction and size, aperture transmission and reflectance. Grouped powers cannot fix a spatial distribution unsupported by the scene.',
              'shared_across_views': sorted(reference_grid),
              'selected_index': selected, 'searched_selected_index': searched_selected,
              'reverted_from_candidate_index': searched_selected if unresolved_reason and not feasible_fallback else None,
              'rejected_candidate_index': searched_selected if unresolved_reason else None,
              'applied_exposure_correction_ev': applied_candidate.get('ev', applied_candidate.get('correction_ev')),
              'candidate_count': len(records),
              'max_candidate_count': MAX_LIGHTING_CANDIDATES,
              'render_count': len(records)*len(views),
              'preview_max_edge': 256, 'preview_samples': min(16, original['samples']),
              'fixed_seed': 41, 'baseline_spatial_loss': baseline_loss,
              'selected_spatial_loss': final_loss,
              'relative_loss_reduction': (baseline_loss-final_loss)/baseline_loss if baseline_loss else 0.0,
              'multiplier_bounds': [MIN_MULTIPLIER, MAX_MULTIPLIER],
              'exposure_correction_bounds': [-MAX_EXPOSURE_CORRECTION_EV, MAX_EXPOSURE_CORRECTION_EV],
              'minimum_relative_improvement': MIN_RELATIVE_IMPROVEMENT,
              'per_view_degradation_guard': {'relative': .05, 'absolute': .005},
              'applied_settings': applied, 'ignored_controls': ignored,
              'candidates': [{key: value for key, value in row.items()
                              if key not in ('preview_grid', 'preview_stats', '_initial_exposure_variant')}
                             for row in records],
              'reference_sha256': {view_id: hashlib.sha256(Path(path).read_bytes()).hexdigest() for view_id, path in sorted(sources.items())},
              'geometry_changed': False, 'material_changed': False,
              'material_change_scope': 'Surface reflectance and material ray graphs unchanged; eligible generated fixture appearance emission follows actual light power through direct datablock drivers.',
              'per_camera_parameters': False, 'additional_model_calls': 0,
              'requires_independent_visual_review': True, 'limitations': limitations}
    return report, applied_candidate
