# One photo group → one shared Blender scene

## Automatic grouped batch entry point

`batch_multiview.py` accepts original photos grouped by physical scene. It does not accept manual scene seeds. Camera directions, metric estimates, furniture placement, crops and lights are inferred automatically. The caller only groups images that belong to the same scene; automatic scene clustering is not implemented.

```json
{"schema_version":1,"scenes":[
  {"scene_id":"computer_lab","images":["lab/view_01.jpg","lab/view_02.jpg"]},
  {"scene_id":"corridor","images":["corridor/view_01.jpg"]},
  {"scene_id":"dorm","images":["dorm/view_01.jpg"]}
]}
```

Paths resolve relative to this manifest. Run with Python containing Pillow, a configured Codex CLI, and Blender:

```sh
python3 scene_factory/batch_multiview.py --groups scene_factory/inputs/2026-10-20/groups.json --output-dir scene_factory/deliveries/my_batch --workers 3 --max-edge 1280 --samples 96 --max-repairs 1
python3 scene_factory/report_multiview.py scene_factory/deliveries/my_batch
```

Analysis and review run concurrently across jobs. Blender jobs serialize on this machine to avoid competing for GPU memory. All photographs in one group go into one joint analysis. Each repair is reviewed against **all** matching original/render pairs; all views must satisfy the quality gate. Raw model responses, prompts, normalized descriptions, source hashes and every iteration are retained. Returning to a completed output verifies recorded hashes before resuming; changed input/code/options requires a new output directory. `--dry-run` lists groups without analysis. `--no-review` cannot be accepted.

Open `comparison.html` for side-by-side results, `summary.json` for outcomes, and each job's `iteration_XX/scene.blend`. `initial_bundle.raw.json` preserves the model response. `analysis_provenance.json` records the absence of manual seeds. Source path restoration and repeated support-ID resolution are recorded separately and do not move geometry. `needs_review` means usable artifacts still have matching/realism defects. Earlier manual explorations in other sample folders are excluded from the automatic batch.

This is approximate, editable reconstruction from ordinary photographs. The model can be wrong about shared geometry, cameras and dimensions. Higher render samples remove noise but do not repair those errors.

### Optional automatic asset library

The API configuration accepts `asset_library: {"catalog":"assets/asset_catalog.json","auto_bind":true,"preview":false,"selection":{"seed":0,"diversity":false}}`. Each scene and repair iteration receives its own `asset_bindings.json` and `asset_selection.json`; all cameras share those instances. The selector filters production gates, categories, scale, dimensions, floor support, anchors and compatible materials. Unsupported objects retain procedural geometry with explicit reasons. Reference-image appearance retrieval and exact mesh collision are not implemented. Review and typed repair receive the actual binding evidence, and a changed scene is selected again before rendering. Candidate file hashes bind resume to the library snapshot; preview output cannot become accepted.

The legacy runner also accepts `--asset-catalog CATALOG --asset-auto-bind`. The direct renderer requires concrete bindings. See [ASSET_LIBRARY.md](ASSET_LIBRARY.md) for configuration and [HARMONY_WORKFLOW.md](HARMONY_WORKFLOW.md) for the source analysis and remaining stages.

### Photo appearance defaults (2026-10-20 update)

Both the grouped batch runner and direct V3 renderer now default to
`--appearance-policy photo_reconstruction --tone-match reference`:

- Neutral office tabletops use laminate microfinish instead of invented oak grain.
- Standard monitors use conservative dark glass when no actual screen crop was supplied; random demo screenshots are disabled. Explicit photo crop planes remain unchanged. This fallback does not recover the missing screen content.
- Office chairs no longer randomly receive polished metal bases. Upholstery still follows the interpreted color; the current chair silhouette library remains limited.
- An AREA light backed by a nearby flat object of matching size gets a visible fixture face. The face adds camera/reflection appearance only; the existing AREA light supplies illumination. Walls, large ceilings and exterior daylight proxies are excluded geometrically.
- AgX Medium High Contrast and bounded reference exposure fitting replace the flat default appearance. Each view gets one 384-pixel preview; 13 exposure transforms are evaluated without repeating ray tracing. One shared correction in [-1.5, +1.5] EV is selected from all source images. It costs no additional model calls and does not change geometry, light positions or watts. NumPy is required in Blender's Python (included in the tested Blender 5.2 installation).

`tone_calibration.json` records the exposure, reference statistics, candidates and limitations. These whole-image statistics are a tone aid, **not a fidelity score**: wrong framing, object coverage and materials can bias them. An adjustment reaching its bound is reported. `--no-review` skips model visual assessment but retains this deterministic exposure step; use `--tone-match off` to disable it. `--no-render` skips calibration. To reproduce older V3 appearance, pass `--appearance-policy legacy --tone-match off`.

The saved `.blend` contains the selected exposure and look, so reopening it uses the same settings. Original scene-description colors are still interpreted as scene-linear material values; photo textures are sRGB and decoded once by Blender. No global gamma conversion is applied to old descriptions.

If a metadata validation bug interrupted a prior run after the model had returned, `--reuse-analysis-from PATH_TO_PRIOR_BATCH` can recover the unchanged raw responses into a new output directory. It checks the exact prompt and all normalized image hashes, records the old raw-response hash and pipeline, and applies only declared identity normalization. It cannot be used to supply a hand-authored scene description. The current normalizer accepts an exact `input:<view_id>` spelling for the primary source identity and an exact `__0_0` alias on a singleton support; geometry is unchanged.

After a generic renderer correction, `rerender_multiview_batch.py --batch PRIOR_BATCH --output-dir NEW_BATCH` verifies the previous automatic output hashes and re-renders the final automatic descriptions unchanged. It then performs a fresh joint visual assessment. This is useful for renderer defects that affect many scenes. Exterior area-light proxies are now hidden from transmission rays so that their white surfaces do not obscure outdoor geometry through windows; their illumination remains active. The prior renders and reviews remain available.

## Scene description and direct renderer

The schema version 3 wrapper contains exactly:

```json
{
  "schema_version": 3,
  "scene_id": "computer_lab",
  "shared_scene": {"schema_version": 2},
  "architecture": {"omit_surfaces": ["back"], "front_wall": true},
  "views": [
    {"id": "view_01", "source_image": "photos/one.jpg", "width": 1280, "height": 960,
     "camera": {"position": [0, 0.3, 1.6], "target": [0, 5, 1.4], "lens_mm": 24, "roll_degrees": 0}},
    {"id": "view_02", "source_image": "photos/two.jpg", "width": 1280, "height": 960,
     "camera": {"position": [0, 5, 1.6], "target": [0, 1, 1.4], "lens_mm": 24, "roll_degrees": 0}}
  ],
  "texture_sources": [{"texture_id": "window_view", "view_id": "view_02"}]
}
```

`shared_scene` must be a **complete valid V2 document**, omitted above for brevity. Geometry is built once. The V2 camera is superseded by the view cameras; no separate rooms or duplicate furniture are created per input photo. Relative source paths are resolved from the bundle file. Every texture crop belongs to the first reference unless overridden in `texture_sources`; preprocessing must use the selected original photograph.

Every view focal length is bounded to 12–85 mm. The first view's position, target and focal length must match `shared_scene.base_scene.camera`, and its roll must match `shared_scene.camera.roll_degrees`. The normalizer copies those first-view fields into the duplicated legacy camera and records the synchronization in provenance before validation. `shared_scene.scene_type` also controls which stock furniture families and material slots are built: classify by the room's dominant use, so a work area with rows of desks and task chairs is `office` even when it contains storage shelves.

`omit_surfaces` removes selected full base surfaces **and their skirting**. Supply custom wall segments in V2 for actual door/window apertures. A photo plane placed over an opaque wall does not create an opening. `front_wall` closes the normally absent Y=0 wall when opposing cameras need it. Its colour follows `base_scene.room.wall_color`.

Prepare crops using `photo_assets_v2.extract` grouped by `texture_view_map(bundle)` and merge those records into a manifest containing `textures: [{id,path}]`. Then invoke:

```sh
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python-exit-code 1 --python scene_factory/generate_multiview.py -- --spec path/to/bundle.json --output-dir path/to/output --texture-manifest path/to/textures.json --max-edge 1280 --samples 128 --device AUTO
```

Omit `--texture-manifest` when the specification has no photo crops. `--max-edge 800 --samples 32` produces a faster visual check. `--no-render` builds and annotates the editable file without image rendering. Source dimensions use the EXIF-oriented photograph; the batch preprocessor must validate/normalize them.

Output: one `scene.blend`, `ReferenceCamera_<view id>` cameras, `rgb_<view id>.png`, and `annotations_<view id>.json`. `rgb.png`/`annotations.json` alias the first view. Originals, texture crops, bundle data and provenance are packed into the Blender file. `metadata.json` records view names, one geometry build, image and artifact hashes, render settings and review status. A completed render is **not** automatically accepted as a faithful reconstruction.

The renderer also accepts manually supplied bundles; these must not be represented as automated photo analysis. The grouped batch runner instead calls `vision_multiview.analyze(scene_id, views, output)` with all original images and no manual scene seed, then calls `review(bundle, directory)` with every original/render pair. Raw model JSON, exact prompts and hash-bound provenance are retained. The reviewer returns per-view scores and a full revised shared bundle. `quality_gate(report, metadata)` fails unless every view meets the thresholds; scores remain uncalibrated. `load_review(directory, bundle)` rejects cached reviews after input/render/report changes. Single-view hidden structures and absolute metric scale remain estimates; successful execution alone does not demonstrate reconstruction fidelity.

Verification:

```sh
cd scene_factory
python3 -m unittest test_multiview
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python-exit-code 1 --python test_multiview_renderer.py
```
