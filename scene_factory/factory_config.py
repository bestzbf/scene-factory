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

BATCH_DEFAULTS = {'workers': 3, 'api_workers': 2, 'max_repairs': 0, 'no_review': True,
                  'vision_timeout': 900, 'render_timeout': 1800}
RENDER_DEFAULTS = {'max_edge': 1280, 'samples': 96, 'device': 'AUTO',
                   'appearance_policy': 'photo_reconstruction', 'tone_match': 'reference'}
PIPELINE_DEFAULTS = {'analysis_mode': 'monolithic', 'repair_mode': 'typed',
                     'validation_retries': 1, 'repair_stagnation_patience': 2, 'staged': {}}
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
    _object(data, ('schema_version', 'models', 'batch', 'render', 'pipeline', 'asset_library', 'budget'), 'config')
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
    for key, low, high in (('workers', 1, 8), ('api_workers', 1, 32), ('max_repairs', 0, 5),
                           ('vision_timeout', 1, 7200), ('render_timeout', 1, 14400)):
        _integer(batch[key], f'batch.{key}', low, high)
    if type(batch['no_review']) is not bool:
        raise ValueError('batch.no_review must be true or false')
    if batch['no_review'] and batch['max_repairs']:
        raise ValueError('batch.max_repairs must be 0 when batch.no_review is true')
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
