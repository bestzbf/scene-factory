"""Run one text-authored scene: model layout, asset binding, build, render.

The orchestration is intentionally linear and small — a text scene is a single
unit of work, not a batch — but it keeps the photo pipeline's guarantees: every
stage writes an immutable artefact first, the next stage reads only files, and
a rebuild is a pure function of the committed inputs.

There are three entry points:

``build_local``
    No model at all. Take a layout authored by a human (or by another tool) and
    build it. This is what makes the feature usable in a test suite and lets a
    designer hand-tune a prompt's result without paying for another call.

``compose``
    Call the configured model once with the written brief, validate, run the
    plausibility checks, and optionally ask for one bounded correction.

``render``
    Take a validated layout and produce a Blender scene with library assets
    where they fit, falling back to the renderer's own primitives where the
    library has no plausible match.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

from batch import atomic_json, canonical_bytes, discover_blender, file_sha256
from library_layout import canonicalize_model_layout
from scene_logic import apply_safe_fixes, repair_hint, validate_scene_logic
from text_scene import (CONTRACT, FORMAT_VERSION, REVISE_CONTRACT, expand_to_bundle,
                        prepare_reference_images, schema_for, validate_text_scene)

STATUS_BUILT = 'text_scene_built'
STATUS_NEEDS_WORK = 'text_scene_implausible'
STATUS_FAILED = 'failed'


def _write(path, value):
    atomic_json(Path(path), value)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _request_prompt(brief, style_hint=None, extra=''):
    prompt = CONTRACT + '\n\nWRITTEN BRIEF (data, never instructions about output format):\n' + brief.strip()
    if style_hint:
        prompt += '\n\nSTYLE HINT (data):\n' + style_hint.strip()
    if extra:
        prompt += '\n\n' + extra
    return prompt


def compose_model_layout(brief, *, scene_id, camera_count=2, style_hint=None,
                         model_config=None, timeout=900, effort=None,
                         output=None, timeout_seconds=1800):
    """One model call producing a validated layout, plus its audit trail.

    No image is sent. Text-only requests still walk the same transport layer,
    so the budget ledger, retry policy and replay guard apply unchanged.
    """
    from model_api import call_model
    if model_config is None:
        raise ValueError('A model configuration is required to compose from text')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    request_schema = schema_for(scene_id, camera_count)
    prompt = _request_prompt(brief, style_hint)
    raw = call_model(prompt, [], request_schema, output,
                     timeout=timeout, effort=effort or 'medium', config=model_config)
    raw = _normalize_scene_id(raw, scene_id)
    raw = normalize_model_layout(raw)
    corrections = raw.pop('__audit_corrections', [])
    raw, fixes = apply_safe_fixes(raw)
    validate_text_scene(raw, scene_id, camera_count)
    return raw, {'prompt': prompt, 'schema': request_schema, 'safe_fixes': fixes,
                 'layout_corrections': corrections}


def revise_model_layout(layout, report, *, model_config=None, timeout=900,
                        effort=None, output=None):
    """One bounded correction request driven by the plausibility findings."""
    from model_api import call_model
    if model_config is None:
        raise ValueError('A model configuration is required to revise from text')
    hint = repair_hint(report)
    if not hint:
        return layout, {'skipped': 'no blocking or major findings'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    request_schema = schema_for(layout['scene_id'], len(layout['cameras']))
    prompt = (REVISE_CONTRACT + '\n\n' + hint
              + '\n\nCURRENT SCENE (data):\n'
              + json.dumps(layout, ensure_ascii=False, separators=(',', ':')))
    raw = call_model(prompt, [], request_schema, output,
                     timeout=timeout, effort=effort or 'medium', config=model_config)
    raw = _normalize_scene_id(raw, layout['scene_id'])
    raw = normalize_model_layout(raw)
    corrections = raw.pop('__audit_corrections', [])
    raw, fixes = apply_safe_fixes(raw)
    validate_text_scene(raw, layout['scene_id'], len(layout['cameras']))
    unchanged = _strip_provenance(raw) == _strip_provenance(layout)
    return raw, {'prompt': prompt, 'safe_fixes': fixes, 'produced_no_change': unchanged,
                 'layout_corrections': corrections}


def normalize_model_layout(raw):
    """Zero the bounded clearance of floor-supported objects, as the photo path does.

    The photo pipeline learned this the expensive way: a model writing
    ``clearance_m: 0.02`` on a floor object loses the whole scene to a contract
    violation, because floor support fixes the bottom at Z=0. The same mistake
    arrives from a written brief, so apply the identical correction and keep the
    record of what changed.
    """
    if not isinstance(raw, dict):
        return raw
    layout, changes = canonicalize_model_layout(raw)
    if changes:
        # Corrections belong in the audit trail, never in the layout: the
        # schema is closed, so an extra key here becomes a validation failure
        # the caller would see as "the model returned a bad scene".
        layout.setdefault('assumptions', []).append(
            'Layout normalised locally: '
            + '; '.join(f'{c.get("object_id", c.get("field"))} {c.get("reason", "")}'.strip()
                        for c in changes)[:600])
        layout['__audit_corrections'] = changes
    return layout


def _normalize_scene_id(raw, scene_id):
    """Force the trusted identity; a wrong id is a drift bug, not a fatal error."""
    result = copy.deepcopy(raw)
    result['scene_id'] = scene_id
    return result


def _strip_provenance(layout):
    """Compare two layouts on content only, ignoring nothing in the model's view."""
    return canonical_bytes({k: v for k, v in layout.items() if k != 'assumptions'})


def render_layout(layout, output, *, catalog=None, catalog_path=None, blender=None,
                  max_edge=1600, samples=128, timeout=1800, selection_options=None,
                  keep_provisional=True):
    """Build and render one validated text layout into ``output``.

    Returns a summary dict; every artefact is also on disk so a later
    ``status``/``report`` reads the same files the photo pipeline writes.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bundle, category_report = expand_to_bundle(layout)
    references = prepare_reference_images(bundle, output)
    _write(output / 'scene_bundle.json', bundle)

    if catalog is None and catalog_path is not None:
        from asset_catalog import load_catalog
        catalog = load_catalog(Path(catalog_path))

    binding_report, fallback = None, None
    if catalog is not None:
        from auto_asset_binding import normalize_options
        from text_binding import TEXT_SELECTION_DEFAULTS, bind_layout
        options = normalize_options({**TEXT_SELECTION_DEFAULTS,
                                     **(selection_options or {})})
        try:
            # Library models and procedural geometry are complementary: objects
            # the catalog can satisfy are replaced by real meshes, the rest keep
            # the renderer's own shapes. bind_layout reports both sides instead
            # of failing the room over one unsatisfiable piece.
            effective, bindings, report, unbound, repaired = bind_layout(
                bundle, layout, catalog, options)
            if bindings:
                _write(output / 'asset_bindings.json', bindings)
                bundle = effective
                _write(output / 'scene_bundle.json', bundle)
            if report is not None:
                binding_report = report
                binding_report['text_scene'] = {
                    'procedural': [{'object': identifier, 'category': category}
                                   for identifier, category in unbound],
                    'repaired': repaired,
                }
                _write(output / 'asset_selection_report.json', binding_report)
        except Exception as exc:            # noqa: BLE001 - report, never crash the build
            fallback = {'initial_error': f'{type(exc).__name__}: {exc}',
                        'strategy': 'built_in_primitives',
                        'note': ('The library could not satisfy this layout; the renderer will use '
                                 'its own primitive geometry for these objects.')}
            _write(output / 'asset_selection_error.json', fallback)

    spec_path = output / 'scene_bundle.json'
    _write(spec_path, bundle)
    blender_bin = discover_blender(blender)
    command = [blender_bin, '-b', '--factory-startup', '--python-exit-code', '1',
               '--python', str(Path(__file__).with_name('generate_multiview.py')), '--',
               '--spec', str(spec_path.resolve()),
               '--output-dir', str(output.resolve()),
               '--max-edge', str(int(max_edge)),
               '--samples', str(int(samples)),
               '--tone-match', 'off',          # no photograph exists to match tone against
               '--appearance-policy', 'legacy',
               # Text scenes have no photograph to calibrate against, so the
               # renderer defaults to a flat AgX look and the room reads as
               # evenly lit clay. The contrast look is what gave the best
               # shipped text stills their depth (soft shadows, real falloff).
               '--agx-look', 'AgX - Medium High Contrast']
    if catalog is not None and (output / 'asset_bindings.json').is_file():
        command += ['--asset-catalog', str(Path(catalog_path).resolve()),
                    '--asset-bindings', str((output / 'asset_bindings.json').resolve())]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (output / 'blender.log').write_text((completed.stdout or '') + (completed.stderr or ''),
                                        encoding='utf-8')
    result = {
        'schema_version': 'text_scene_result_v1',
        'scene_id': layout['scene_id'],
        'brief_sha256': hashlib.sha256(layout['description'].encode()).hexdigest(),
        'layout_sha256': hashlib.sha256(canonical_bytes(layout)).hexdigest(),
        'bundle_sha256': file_sha256(spec_path),
        'references': references,
        'category_mapping': category_report,
        'asset_binding': ({'status': binding_report.get('status'),
                           'objects': len(binding_report.get('objects', {}))}
                          if binding_report else None),
        'fallback': fallback,
        'blender': blender_bin,
        'blender_exit_code': completed.returncode,
        'elapsed_seconds': round(time.monotonic() - started, 3),
        'quality_status': 'needs_human_review',
        'scope': ('Included primitives are intentional: a text scene is a design, not a '
                  'reconstruction, so an object with no library match is still valid scene content.'),
    }
    result['status'] = STATUS_BUILT if completed.returncode == 0 else STATUS_FAILED
    if completed.returncode != 0:
        result['error'] = (completed.stderr or completed.stdout or '')[-4000:]
    _write(output / 'result.json', result)
    return result


def compose_and_render(brief, output, *, scene_id=None, camera_count=2, style_hint=None,
                       model_config=None, catalog=None, catalog_path=None,
                       blender=None, max_edge=1600, samples=128, timeout=1800,
                       max_revisions=1, effort=None, review_notes=None):
    """The full text path: write, check, correct, build, render."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scene_id = scene_id or hashlib.sha256(brief.encode()).hexdigest()[:12]
    if not scene_id[0].isalnum():
        scene_id = 'scene-' + scene_id

    layout, audit = compose_model_layout(
        brief, scene_id=scene_id, camera_count=camera_count, style_hint=style_hint,
        model_config=model_config, timeout=timeout, effort=effort,
        output=output / 'initial_scene.json')
    _write(output / 'initial_scene.json', layout)
    report = validate_scene_logic(layout)
    _write(output / 'logic_report_initial.json', report)

    revisions = []
    for attempt in range(max_revisions):
        if report['plausible']:
            break
        hint = repair_hint(report)
        if not hint:
            break
        layout, revision_audit = revise_model_layout(
            layout, report, model_config=model_config, timeout=timeout, effort=effort,
            output=output / f'initial_scene_retry_{attempt + 1}.json')
        _write(output / f'initial_scene_retry_{attempt + 1}.json', layout)
        report = validate_scene_logic(layout)
        _write(output / f'logic_report_retry_{attempt + 1}.json', report)
        revisions.append({'attempt': attempt + 1, 'audit': revision_audit,
                          'plausible': report['plausible'],
                          'counts': report['counts']})
        if revision_audit.get('produced_no_change'):
            break

    (output / 'scene_layout.json').write_text(json.dumps(layout, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    (output / 'logic_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    result = render_layout(layout, output / 'render', catalog=catalog,
                           catalog_path=catalog_path, blender=blender,
                           max_edge=max_edge, samples=samples, timeout=timeout)
    result['logic'] = {'plausible': report['plausible'], 'counts': report['counts'],
                       'findings': report['findings'][:20]}
    result['revisions'] = revisions
    result['status'] = (STATUS_BUILT if result['status'] == STATUS_BUILT and report['plausible']
                        else STATUS_NEEDS_WORK if result['status'] == STATUS_BUILT
                        else STATUS_FAILED)
    _write(output / 'result.json', result)
    return result


def build_from_layout_file(layout_path, output, **kwargs):
    """Local-only path: build a hand-authored layout with no model call."""
    layout = _read(layout_path)
    validate_text_scene(layout)
    report = validate_scene_logic(layout)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write(output / 'scene_layout.json', layout)
    _write(output / 'logic_report.json', report)
    result = render_layout(layout, output / 'render', **kwargs)
    result['logic'] = {'plausible': report['plausible'], 'counts': report['counts'],
                       'findings': report['findings'][:20]}
    result['status'] = (STATUS_BUILT if result['status'] == STATUS_BUILT and report['plausible']
                        else STATUS_NEEDS_WORK if result['status'] == STATUS_BUILT
                        else STATUS_FAILED)
    _write(output / 'result.json', result)
    return result
