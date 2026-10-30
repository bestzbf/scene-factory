"""Bounded shared exposure and light-power fitting from actual Cycles previews.

Fits one physical scene across every view. No model call, per-camera lighting,
scene-description rewrite or reference photograph projection is involved.
"""
from pathlib import Path
import tempfile

import bpy
import numpy as np

from reference_tone import compute_stats, choose_shared_exposure, compute_spatial_stats, compare_spatial_stats


def read_display_rgb(path):
    """Return display RGB in Blender's bottom-to-top image buffer order."""
    image = bpy.data.images.load(str(Path(path).resolve()), check_existing=False)
    try:
        # Without Non-Color, Blender decodes sRGB pixels to scene-linear.
        # compute_stats explicitly accepts encoded display RGB.
        image.colorspace_settings.name = 'Non-Color'
        width, height = image.size
        pixels = np.empty(width*height*4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        return np.clip(pixels.reshape(height,width,4)[:,:,:3], 0, 1)
    finally:
        bpy.data.images.remove(image)


def calibrate(scene, views, cameras, sources, set_view, *, fit_lighting=True):
    original = {'camera': scene.camera, 'samples': scene.cycles.samples,
                'x': scene.render.resolution_x, 'y': scene.render.resolution_y,
                'percentage': scene.render.resolution_percentage,
                'seed': scene.cycles.seed, 'animated_seed': scene.cycles.use_animated_seed,
                'adaptive': scene.cycles.use_adaptive_sampling,
                'denoising': scene.cycles.use_denoising,
                'border': scene.render.use_border,
                'exposure': scene.view_settings.exposure, 'filepath': scene.render.filepath}
    reference, reference_grid = {}, {}
    for view in views:
        rgb = read_display_rgb(sources[view['id']])
        reference[view['id']] = compute_stats(rgb)
        reference_grid[view['id']] = compute_spatial_stats(rgb, bottom_up=True)
    offsets = [round(i*.25, 2) for i in range(-6, 7)]
    candidates = [{'ev': offset, 'views': {}, 'grid_views': {}} for offset in offsets]
    try:
        with tempfile.TemporaryDirectory(prefix='scene-tone-') as folder:
            path = str(Path(folder)/'candidate.png')
            for view in views:
                set_view(scene, cameras[view['id']], view, 384)
                # Apply after set_view so camera setup cannot accidentally
                # reintroduce cropped, scaled or stochastic metering previews.
                scene.cycles.samples = min(32, original['samples'])
                scene.cycles.seed = 41
                scene.cycles.use_animated_seed = False
                scene.cycles.use_adaptive_sampling = False
                scene.cycles.use_denoising = True
                scene.render.resolution_percentage = 100
                scene.render.use_border = False
                scene.view_settings.exposure = original['exposure']
                bpy.ops.render.render()
                result = bpy.data.images['Render Result']
                for candidate in candidates:
                    scene.view_settings.exposure = original['exposure'] + candidate['ev']
                    result.save_render(path, scene=scene)
                    rgb = read_display_rgb(path)
                    candidate['views'][view['id']] = compute_stats(rgb)
                    candidate['grid_views'][view['id']] = compute_spatial_stats(rgb, bottom_up=True)
        report = choose_shared_exposure(candidates, reference)
        report.update({'initial_exposure': original['exposure'],
                       'final_exposure': original['exposure']+report['correction_ev'],
                       'view_transform': scene.view_settings.view_transform,
                       'look': scene.view_settings.look, 'reference_stats': reference,
                       'selected_preview_stats': candidates[report['selected_index']]['views'],
                       'preview_max_edge': 384, 'preview_samples': min(32, original['samples']),
                       'exposure_preview_settings': {'seed': 41, 'animated_seed': False,
                           'adaptive_sampling': False, 'denoising': True,
                           'resolution_percentage': 100, 'render_border': False},
                       'additional_model_calls': 0,
                       'at_correction_bound': abs(report['correction_ev']) == report['max_correction_ev']})
        report['spatial_diagnostics'] = compare_spatial_stats(reference_grid,
            candidates[report['selected_index']]['grid_views'], at_correction_bound=report['at_correction_bound'])
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
        scene.view_settings.exposure = original['exposure']
    scene.view_settings.exposure = report['final_exposure']
    if fit_lighting:
        from lighting_fit import fit_rendered_lighting
        try:
            fit, selected = fit_rendered_lighting(scene, views, cameras, sources, set_view,
                reference_grid, report['correction_ev'], original['exposure'], read_display_rgb)
        except Exception:
            scene.view_settings.exposure = original['exposure']
            raise
        report['lighting_fit'] = fit
        if selected is not None:
            report['exposure_only_correction_ev'] = report['correction_ev']
            report['exposure_only_selected_index'] = report['selected_index']
            report['exposure_only_spatial_diagnostics'] = report['spatial_diagnostics']
            report['exposure_only_fit'] = {key: report[key] for key in
                ('selected_index', 'method', 'baseline_robust_loss', 'selected_robust_loss', 'candidates')}
            report['selected_index'] = fit['selected_index']
            report['method'] = fit['method']
            report['baseline_robust_loss'] = fit['baseline_spatial_loss']
            report['selected_robust_loss'] = fit['selected_spatial_loss']
            report['candidates'] = fit['candidates']
            report['correction_ev'] = selected['ev']
            report['final_exposure'] = original['exposure'] + selected['ev']
            report['at_correction_bound'] = abs(selected['ev']) == report['max_correction_ev']
            # Normal fit records use preview_stats/preview_grid. Fail-closed
            # rollback selects the unchanged exposure candidate, which still
            # uses the exposure-only views/grid_views names.
            preview_stats = selected.get('preview_stats', selected.get('views'))
            preview_grid = selected.get('preview_grid', selected.get('grid_views'))
            if not isinstance(preview_stats, dict) or not isinstance(preview_grid, dict):
                raise ValueError('Selected lighting candidate lacks preview diagnostics')
            report['selected_preview_stats'] = preview_stats
            report['spatial_diagnostics'] = compare_spatial_stats(reference_grid,
                preview_grid, at_correction_bound=report['at_correction_bound'])
            from lighting_fit import calibration_status_after_fit
            # A bounded fit that hit a power/exposure limit while severe
            # regional residuals remained was rolled back. Keep the unresolved
            # status visible instead of inheriting the exposure-only label.
            report['status'] = calibration_status_after_fit(report['status'], fit['status'])
            report['preview_max_edge'] = fit['preview_max_edge']
            report['preview_samples'] = fit['preview_samples']
            report['limitations'] = fit['limitations']
    else:
        report['lighting_fit'] = {'status': 'disabled'}
    return report


def final_render_diagnostics(sources, render_paths, *, at_correction_bound=False):
    """Meter final saved PNGs independently of small preview/search estimates."""
    if not sources or set(sources) != set(render_paths):
        raise ValueError('Final diagnostics need exactly the source view IDs')
    reference, rendered = {}, {}
    for view_id in sorted(sources):
        reference[view_id] = compute_spatial_stats(read_display_rgb(sources[view_id]), bottom_up=True)
        rendered[view_id] = compute_spatial_stats(read_display_rgb(render_paths[view_id]), bottom_up=True)
    report = compare_spatial_stats(reference, rendered, at_correction_bound=at_correction_bound)
    report['measurement_stage'] = 'final_saved_render'
    report['render_sha256'] = {view_id: __import__('hashlib').sha256(Path(path).read_bytes()).hexdigest()
                               for view_id, path in sorted(render_paths.items())}
    return report
