# Shared automatic light-power fitting

`calibrate(scene, views, cameras, sources, set_view, *, fit_lighting=True)` remains backward-compatible. The default adds actual Cycles previews that search shared world/exterior/interior/directional power multipliers and the existing ±1.5 EV exposure interval. Light groups are inferred from actual Blender emitters and the existing exterior-proxy flag, never scene names or dorm-specific values.

- Bounded cost: at most 15 lighting settings × number of views, 256-pixel previews, up to 16 samples, fixed seed 41. Each actual lighting render is reused only for color-management exposure variants.
- Exposure-only previews use a 384-pixel maximum edge and up to 32 samples.
  After each camera setup they enforce seed 41, animated seed off, adaptive
  sampling off, denoising on, 100% resolution and render border off. This avoids
  metering a cropped or unexpectedly downscaled image. These transient settings
  are restored on success and failure; only a selected exposure is retained.
  The report records them under `exposure_preview_settings`. Reproducibility
  applies to a fixed Blender/device setup, not bitwise equality across hardware.
- Bounded changes: powers remain between 1/64 and 16 times their input value; source positions, sizes, colors, materials and geometry remain fixed. One power/exposure setting serves every view.
- Selection: equal-view mean of 3×3 log-luminance Huber loss plus 0.35 times regional contrast loss and 0.2 times the mean squared error of the worst third of cells. At least three informative reference cells in every view are required. Preserve the exposure-only baseline unless loss improves by at least 2% and 0.002 absolute; no view may degrade beyond 5% plus 0.005 loss.
- Fail-closed boundary guard: if the selected power or ±1.5 EV correction reaches a bound while any view still has a regional residual above 2 stops, reject that proposal. Among the existing measured candidate records, a fallback may be applied only when all its powers and exposure are strictly inside their bounds, its loss improves the measured original powers/exposure by at least 2%, and no view's loss increases. Choose the lowest-loss eligible record; do not compare against the rejected exposure-only baseline. This adds no model requests or candidate renders. Report `status='selected_feasible_unresolved'`, retaining the original rejection reason and remaining severe-cell IDs. If no such record exists, restore the original powers and exposure with `status='reverted_unresolved'`. Neither status grants visual acceptance.
- Final spatial diagnostics flag every informative cell with an absolute error above 2 stops, even when the median and p90–p10 statistics are small. The report includes maximum regional error and severe-cell IDs; a single extreme region cannot be hidden by eight better regions.
- Reproducible reporting: `tone_calibration.lighting_fit` includes every candidate measurement, selected index, actual applied powers, multiplier bounds, preview settings, source SHA256 values and limitations. The original exposure-only measurements remain under `exposure_only_fit`.
- On failure or a boundary rejection without a feasible measured fallback, restore original powers and all transient render settings. Otherwise keep selected powers/exposure, restoring camera, sampling, seed, resolution, border settings and output path.
- This is a brightness optimization aid, not a fidelity score or acceptance. Existing independent visual review is still mandatory.

## Renderer integration

Add `lighting_fit.py` to the immutable pipeline snapshot and fingerprint. `generate_multiview.py` already calls `calibrate` under `--tone-match reference`, so no new UI switch is necessary. `off` skips both exposure and power fitting.

Effective final powers are in `tone['lighting_fit']['applied_settings']`; preserve these alongside the model's requested lighting parameters to avoid presenting requested powers as applied powers.

After all final PNGs are written, calculate final-resolution evidence:

```python
from calibrate_reference import final_render_diagnostics
tone['final_spatial_diagnostics'] = final_render_diagnostics(
    sources,
    {view['id']: out / ('rgb_' + view['id'] + '.png') for view in bundle['views']},
    at_correction_bound=tone.get('at_correction_bound', False),
)
```

Rewrite `tone_calibration.json` before output hashes are taken. This helper includes exact final render SHA256 values and `measurement_stage='final_saved_render'`. The existing `spatial_diagnostics` is the selected **post-exposure** preview measurement.

## Verification and limits

25 numerical/contract tests (lighting + prior tone/spatial suites), 2 real Cycles power fitting/rollback tests, and 6 prior Blender exposure/spatial tests passed. A power-only replay of untouched prior GLM dorm output took approximately 20 seconds CPU and reduced preview spatial loss from 2.065 to 1.087 (47%). This does not prove visual acceptance: the prior scene still lacks bright exterior scenery and has incorrect shapes/materials. Evidence is in `production_local/lighting_fit_v034_regression/`; no scene-specific correction was supplied to the fit.


## Source-only v2 objective audit (2026-10-23)

The v1 Huber loss has bounded influence beyond 0.5 EV. Seven roughly correct cells can therefore outweigh two cells that are 3–4 EV too dark. The contrast term depends on differences of log errors and is invariant to a shared log-brightness shift. It does not prevent this majority-cell tradeoff. This was observed in real GLM replay iteration 03: chosen exterior power reached 1/64 while upper/lower center errors were −3.77 and −3.12 EV.

The v2 objective (`spatial_log_huber_contrast_tail_v2`) adds a symmetric worst-third squared-error term. It knows neither window semantics nor positions, and penalizes severe under- and overexposure equally. Reports additionally contain the objective version, worst-third error, severe-cell IDs and whether selected powers reached the allowed bound. Scores from different objective versions must not be directly compared.

A source-only, new-output Cycles replay with unchanged scene geometry/materials and original model powers selected exterior 1/32 and world 1/16. Final-render worst-cell error fell 3.77→2.38 EV and worst-three RMS fell 2.85→1.98 EV, but median absolute error rose 0.35→1.42 EV. This is a more balanced compromise, not a demonstrated reconstruction-quality improvement. The actual lit scene cannot match the reference spatial distribution by grouped powers alone; wrong framing/visible coverage and emitter directions remain. New evidence is in `production_local/optimization_v035_validation/lighting_tail_v2_source_replay/`. Running installed 0.3.5 and historical artifacts were not changed.

26 numerical tests and 3 real Cycles tests pass with v2. Do not bypass independent review or promote this fixture to the dataset. Persisting opposite-sign severe residuals should guide model repair toward actual light direction, aperture transport, camera framing and material reflectance; repeated power-only increases will be counteracted by fitting and cannot resolve the structural mismatch.


## Version 0.3.10: fixture appearance follows actual power

Generated geometry-backed light faces now bind their appearance-only emission
strength directly to the actual AREA Light datablock energy through a Blender
driver. Every fitting candidate, selected setting, and failure rollback therefore
updates camera-visible and reflection-visible light appearance as well as the
real emitter. Zero power also extinguishes the generated face. Drivers survive
renaming and saving/reloading and support multiple objects sharing Light data.
No second diffuse emitter is added; the existing ray graph is preserved.

`bind_fixture_power_drivers(scene)` can upgrade explicitly tagged older generated
faces. It measures legacy face area in world space, isolates shared materials
per object, and skips linked/unrecognized graphs, authored animation and foreign
drivers instead of overwriting them. Skips appear in the fit's `ignored_controls`.
Animated or linked Light/World node trees are also excluded from power controls.
Surface reflectance stays fixed; the report's `material_change_scope` explicitly
distinguishes it from the generated fixture emission following emitter power.

Read-only checks on the saved office confirmed that all eight faces in iterations
01 and 02 retained their initial brightness while actual lamp powers had risen
by factors of 4 and 16. New-output Cycles replays at identical camera, geometry,
exposure and light powers demonstrate the coupling fix. Whole-image differences
are small; this is a rendering-consistency fix, not photographic acceptance or
evidence that missing objects, shapes and textures have been reconstructed.
Historical source files remain unchanged.
