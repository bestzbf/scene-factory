"""Shared retail rules accepted by unattended photo-to-scene production.

Store-specific geometry/camera profiles belong to reviewed rebuilds. This
adapter deliberately accepts only rules inferred uniformly for every fixture.
"""
from pathlib import Path

from batch import file_sha256
from retail_refinement import read_config


def production_inputs(path):
    path = Path(path).expanduser().resolve()
    config = read_config(path)
    if set(config) - {'schema_version', 'seed', 'atlas', 'defaults', 'scenes', '_config_path'}:
        raise ValueError('Unattended retail profile contains unsupported fields')
    if config.get('scenes'):
        raise ValueError('Unattended production forbids per-store retail overrides')
    defaults = config.get('defaults', {})
    if set(defaults) - {'fixtures', 'fixture_defaults'} or defaults.get('fixtures'):
        raise ValueError('Unattended production accepts shared fixture_defaults only, not object IDs or geometry overrides')
    common = defaults.get('fixture_defaults', {})
    if set(common) - {'enabled', 'top_stock'}:
        raise ValueError('Unattended fixture rules only support enabled and top_stock')
    if not config.get('atlas'):
        raise ValueError('Unattended retail profile requires a shared packaging atlas')
    inputs = {str(p): file_sha256(p) for p in (path, Path(config['atlas']))}
    return {'policy': 'shared_rules_only', 'config': str(path), 'input_sha256': inputs}


def verify_inputs(record):
    for path, digest in record.get('input_sha256', {}).items():
        if file_sha256(path) != digest:
            raise ValueError('Shared retail profile or packaging atlas changed during production')
