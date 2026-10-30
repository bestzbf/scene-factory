"""Strict, credential-free configuration for the batch factory."""
from __future__ import annotations

import copy
import json
from pathlib import Path

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

from model_api import normalize_config as normalize_model
from model_api import public_config as public_model

BATCH_DEFAULTS = {'workers': 3, 'api_workers': 2, 'render_workers': 1, 'max_repairs': 0, 'no_review': True,
                  'vision_timeout': 900, 'render_timeout': 1800, 'min_free_disk_bytes': 0,
                  'shared_ledger_dir': None}
RENDER_DEFAULTS = {'max_edge': 1280, 'samples': 96, 'device': 'AUTO',
                   'appearance_policy': 'photo_reconstruction', 'tone_match': 'reference'}
PIPELINE_DEFAULTS = {'analysis_mode': 'monolithic', 'repair_mode': 'typed',
                     'validation_retries': 1, 'repair_stagnation_patience': 2, 'staged': {},
                     'scene_wire_format': 'canonical'}
ASSET_LIBRARY_DEFAULTS = {'catalog': None, 'bindings': None, 'preview': False,
                          'auto_bind': False, 'selection': {}}


def _object(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a JSON object')
    if set(value) - set(allowed):
        raise ValueError(f'{label} contains unknown fields; credentials belong in environment variables, not config')
    return value


def _integer(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label} must be an integer between {low} and {high}')
    return value


def normalize_config(data):
    """Return a normalized independent config; never inspect environment values."""
    _object(data, ('schema_version', 'models', 'batch', 'render', 'pipeline', 'asset_library', 'budget', 'batch_budget'), 'config')
    if type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError('config requires schema_version=1')
    models = _object(data.get('models'), ('analysis', 'review', 'repair'), 'models')
    if 'analysis' not in models:
        raise ValueError('models.analysis is required')
    normalized_models = {}
    for role, value in models.items():
        if not isinstance(value, dict) or not value.get('base_url'):
            raise ValueError(f'models.{role}.base_url must be explicitly configured with your provider URL')
        normalized_models[role] = normalize_model(value)
    normalized_models.setdefault('review', copy.deepcopy(normalized_models['analysis']))
    normalized_models.setdefault('repair', copy.deepcopy(normalized_models['review']))
    batch = {**BATCH_DEFAULTS, **_object(data.get('batch', {}), BATCH_DEFAULTS, 'batch')}
    for key, low, high in (('workers', 1, 8), ('api_workers', 1, 32), ('render_workers', 1, 8), ('max_repairs', 0, 5),
                           ('vision_timeout', 1, 7200), ('render_timeout', 1, 14400),
                           ('min_free_disk_bytes', 0, 10**15)):
        _integer(batch[key], f'batch.{key}', low, high)
    ledger_dir = batch['shared_ledger_dir']
    if ledger_dir is not None and (not isinstance(ledger_dir, str) or not ledger_dir.strip() or '\x00' in ledger_dir):
        raise ValueError('batch.shared_ledger_dir must be null or a nonempty path')
    if type(batch['no_review']) is not bool:
        raise ValueError('batch.no_review must be true or false')
    # With visual review disabled, max_repairs still budgets typed repairs for
    # blocking local preflight failures (for example, furniture outside the
    # room). The repaired scene remains generated_unreviewed and cannot enter
    # the accepted dataset without a later visual review.
    render = {**RENDER_DEFAULTS, **_object(data.get('render', {}), (*RENDER_DEFAULTS, 'blender'), 'render')}
    for key, low, high in (('max_edge', 64, 8192), ('samples', 1, 4096)):
        _integer(render[key], f'render.{key}', low, high)
    for key, allowed in (('device', ('AUTO', 'CPU')),
                         ('appearance_policy', ('photo_reconstruction', 'legacy')),
                         ('tone_match', ('reference', 'off'))):
        if render[key] not in allowed:
            raise ValueError(f'render.{key} must be one of: {", ".join(allowed)}')
    if 'blender' in render:
        if not isinstance(render['blender'], str) or not render['blender'].strip() or '\x00' in render['blender']:
            raise ValueError('render.blender must be a nonempty executable path or command')
    pipeline = {**PIPELINE_DEFAULTS, **_object(data.get('pipeline', {}), PIPELINE_DEFAULTS, 'pipeline')}
    if pipeline['analysis_mode'] not in ('monolithic', 'staged'):
        raise ValueError('pipeline.analysis_mode must be monolithic or staged')
    if pipeline['repair_mode'] not in ('typed', 'full_bundle'):
        raise ValueError('pipeline.repair_mode must be typed or full_bundle')
    if pipeline['scene_wire_format'] not in ('canonical', 'typed_parts_v1', 'anchored_parts_v1', 'anchored_patterns_v1', 'library_layout_v1'):
        raise ValueError('pipeline.scene_wire_format must be canonical, typed_parts_v1, anchored_parts_v1, anchored_patterns_v1 or library_layout_v1')
    if pipeline['analysis_mode'] != 'monolithic' and pipeline['scene_wire_format'] != 'canonical':
        raise ValueError('pipeline.scene_wire_format=' + pipeline['scene_wire_format'] + ' requires monolithic analysis')
    _integer(pipeline['validation_retries'], 'pipeline.validation_retries', 0, 3)
    _integer(pipeline['repair_stagnation_patience'], 'pipeline.repair_stagnation_patience', 0, 5)
    from staged_analysis import normalize_options
    pipeline['staged'] = normalize_options(pipeline['staged'], timeout=batch['vision_timeout'])
    asset_library = {**copy.deepcopy(ASSET_LIBRARY_DEFAULTS), **_object(data.get('asset_library', {}), ASSET_LIBRARY_DEFAULTS, 'asset_library')}
    for key in ('preview', 'auto_bind'):
        if type(asset_library[key]) is not bool:
            raise ValueError(f'asset_library.{key} must be true or false')
    if asset_library['auto_bind']:
        if asset_library['catalog'] is None or asset_library['bindings'] is not None:
            raise ValueError('asset_library.auto_bind requires catalog and forbids manual bindings')
    elif (asset_library['catalog'] is None) != (asset_library['bindings'] is None):
        raise ValueError('asset_library.catalog and asset_library.bindings must be supplied together')
    for key in ('catalog', 'bindings'):
        if asset_library[key] is not None and (not isinstance(asset_library[key], str) or not asset_library[key].strip() or '\x00' in asset_library[key]):
            raise ValueError(f'asset_library.{key} must be a nonempty path')
    if pipeline['scene_wire_format'] in ('anchored_parts_v1', 'anchored_patterns_v1') and asset_library['catalog'] is not None:
        raise ValueError('pipeline.scene_wire_format=' + pipeline['scene_wire_format'] + ' currently supports built-in assets only; '
                         'external asset contact surfaces have not been verified')
    if pipeline['scene_wire_format'] == 'library_layout_v1':
        if not asset_library['catalog'] or not asset_library['auto_bind'] or asset_library['preview']:
            raise ValueError('library_layout_v1 requires a catalog, auto_bind=true and preview=false')
        if not batch['no_review'] or batch['max_repairs'] != 0:
            raise ValueError('library_layout_v1 currently requires no_review=true and max_repairs=0; canonical repairs cannot preserve library intent')
    if not isinstance(asset_library['selection'], dict):
        raise ValueError('asset_library.selection must be a JSON object')
    if asset_library['auto_bind']:
        from auto_asset_binding import normalize_options as normalize_selection
        asset_library['selection'] = normalize_selection(asset_library['selection'])
    elif asset_library['selection']:
        raise ValueError('asset_library.selection requires auto_bind=true')
    result = {'schema_version': 1, 'models': normalized_models, 'batch': batch, 'render': render, 'pipeline': pipeline,
              'asset_library': asset_library}
    if 'budget' in data:
        from scene_budget import normalize_policy, BudgetError
        try:
            result['budget'] = normalize_policy(_object(data['budget'],
                ('max_requests', 'max_output_tokens', 'max_reported_tokens'), 'budget'))
        except BudgetError as exc:
            raise ValueError(str(exc)) from None
    if pipeline['scene_wire_format'] == 'library_layout_v1':
        if (normalized_models['analysis']['max_retries'] != 0
                or pipeline['validation_retries'] != 0
                or result.get('budget', {}).get('max_requests') != 1):
            raise ValueError('library_layout_v1 requires max_retries=0, validation_retries=0 and budget.max_requests=1')
        if asset_library['selection'].get('category_map'):
            raise ValueError('library_layout_v1 uses exact library categories; category_map must be empty')
    if 'batch_budget' in data:
        from batch_guard import normalize_policy, BatchBudgetError
        try:
            result['batch_budget'] = normalize_policy(data['batch_budget'])
        except BatchBudgetError as exc:
            raise ValueError(str(exc)) from None
    return result


def public_config(config):
    result = normalize_config(config)
    result['models'] = {role: public_model(value) for role, value in result['models'].items()}
    return result


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Config contains a duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError('Config contains a non-finite number')


def load_config(path):
    path = Path(path).expanduser().resolve()
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Config exceeds the 1 MiB size limit')
    try:
        data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_no_duplicates,
                          parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid config JSON at line {exc.lineno}, column {exc.colno}') from None
    return normalize_config(data)
