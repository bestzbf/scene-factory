"""A photo group describes one shared V2 scene and one camera per reference.

This module has no Blender/Pillow dependency. Scene descriptions are supplied
interpretations, not a claim of automatic or metrically exact reconstruction.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from spec_v2 import validate as validate_v2

SURFACES = {'floor', 'ceiling', 'left', 'right', 'back'}
SAFE_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$')
MAX_VIEWS = 32             # reference photos per shared scene, e.g. every camera of one shop


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(label + ' must contain exactly: ' + ', '.join(sorted(expected)))


def _identifier(value, label):
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(label + ' must be a safe 1..80 character alphanumeric, dash or underscore identifier')


def _number(value, low, high, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(label + ' is outside the finite allowed range')


def _vector(value, label):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(label + ' must contain three coordinates')
    for entry in value:
        _number(entry, -1000, 1000, label)


def validate(bundle):
    _keys(bundle, {'schema_version', 'scene_id', 'shared_scene', 'architecture', 'views', 'texture_sources'}, 'bundle')
    if bundle['schema_version'] != 3:
        raise ValueError('Multiview bundle requires schema_version=3')
    _identifier(bundle['scene_id'], 'scene_id')
    validate_v2(bundle['shared_scene'])
    architecture = bundle['architecture']
    _keys(architecture, {'omit_surfaces', 'front_wall'}, 'architecture')
    omitted = architecture['omit_surfaces']
    if not isinstance(omitted, list) or any(not isinstance(x, str) or x not in SURFACES for x in omitted) or len(set(omitted)) != len(omitted):
        raise ValueError('omit_surfaces must be a unique list of named room surfaces')
    if not isinstance(architecture['front_wall'], bool):
        raise ValueError('front_wall must be boolean')
    if not isinstance(bundle['views'], list) or not 1 <= len(bundle['views']) <= MAX_VIEWS:
        raise ValueError(f'A shared scene requires 1..{MAX_VIEWS} reference views')
    ids = set()
    for view in bundle['views']:
        _keys(view, {'id', 'source_image', 'width', 'height', 'camera'}, 'view')
        _identifier(view['id'], 'view id')
        if view['id'] in ids:
            raise ValueError('Duplicate reference view: ' + view['id'])
        ids.add(view['id'])
        if not isinstance(view['source_image'], str) or not view['source_image'] or '\x00' in view['source_image']:
            raise ValueError('source_image must be a local file path')
        for axis in ('width', 'height'):
            if type(view[axis]) is not int or not 1 <= view[axis] <= 16384:
                raise ValueError('Reference dimensions must be integers in 1..16384')
        camera = view['camera']
        _keys(camera, {'position', 'target', 'lens_mm', 'roll_degrees'}, 'camera')
        _vector(camera['position'], 'camera position')
        _vector(camera['target'], 'camera target')
        if sum((a-b)**2 for a,b in zip(camera['position'], camera['target'])) < 1e-10:
            raise ValueError('Camera position and target must differ')
        _number(camera['lens_mm'], 12, 85, 'lens_mm')
        _number(camera['roll_degrees'], -180, 180, 'roll_degrees')
    primary_camera = bundle['views'][0]['camera']
    legacy_camera = bundle['shared_scene']['base_scene']['camera']
    stored_roll = bundle['shared_scene']['camera']['roll_degrees']
    if any(primary_camera[key] != legacy_camera[key]
           for key in ('position', 'target', 'lens_mm')):
        raise ValueError('First view camera must exactly match shared_scene.base_scene.camera')
    if primary_camera['roll_degrees'] != stored_roll:
        raise ValueError('First view roll_degrees must exactly match shared_scene.camera.roll_degrees')
    if not isinstance(bundle['texture_sources'], list):
        raise ValueError('texture_sources must be a list')
    texture_ids = {item['id'] for item in bundle['shared_scene']['textures']}
    mapped = set()
    for entry in bundle['texture_sources']:
        _keys(entry, {'texture_id', 'view_id'}, 'texture source')
        if not isinstance(entry['texture_id'], str) or entry['texture_id'] not in texture_ids:
            raise ValueError('Texture source references an unknown texture')
        if not isinstance(entry['view_id'], str) or entry['view_id'] not in ids:
            raise ValueError('Texture source references an unknown view')
        if entry['texture_id'] in mapped:
            raise ValueError('Duplicate texture source mapping: ' + entry['texture_id'])
        mapped.add(entry['texture_id'])
    return bundle


def load(path):
    return validate(json.loads(Path(path).read_text(encoding='utf-8')))


def texture_view_map(bundle):
    """Unmapped texture regions belong to the first input view."""
    result = {item['id']: bundle['views'][0]['id'] for item in bundle['shared_scene']['textures']}
    result.update({item['texture_id']: item['view_id'] for item in bundle['texture_sources']})
    return result


def resolve_sources(bundle, spec_path):
    parent = Path(spec_path).resolve().parent
    result = {}
    for view in bundle['views']:
        source = Path(view['source_image']).expanduser()
        source = (parent / source).resolve() if not source.is_absolute() else source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        result[view['id']] = source
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def render_dimensions(view, max_edge):
    if type(max_edge) is not int or not 64 <= max_edge <= 8192:
        raise ValueError('max_edge must be an integer in 64..8192')
    scale = max_edge / max(view['width'], view['height'])
    return max(1, round(view['width'] * scale)), max(1, round(view['height'] * scale))
