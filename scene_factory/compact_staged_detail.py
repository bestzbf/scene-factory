"""Opt-in sparse typed geometry for staged detail; no visible information removed."""
from __future__ import annotations

import copy

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

from compact_scene import FIELDS_BY_TYPE
from model_api import _walk_wire
from schema import obj
from spec_v2 import PART_SCHEMA

FORMAT = 'native_sparse_parts_v1'
GEOMETRY_GUIDANCE = '''WIRE native_sparse_parts_v1: return wire_format="native_sparse_parts_v1". Each part contains only type,material_id,data; data remains EXACT10numbers=[positionXYZ,sizeXYZ,rotationXYZdeg,radius]. Put topology in sparse geometry records {part_index,points,vertices,faces,cloth}; part_index is zero-based, with one complete record for every curve/polyline, mesh or curtain. Other parts need no record. The decoder restores unused arrays only; it cannot add geometry. Boxes require positive XYZ dimensions; cylinders positive radius/height; spheres positive radius or ellipsoid dimensions; curves/polylines need 2+ distinct points and positive radius; meshes require valid explicit faces; image planes need positive width/height and photo-textured material. Use rigid polylines for frames/rails/trim corners, smooth curves for bends. Curtains need one cloth record with fold_count,fold_depth,sag,thickness,segments_x,segments_z,seed; fold_depth+2*thickness<=size.y, 2*sag<size.z-2*thickness, segments_x>=4*fold_count. Use observed folds and economical resolution, not flat panels. Preserve all photo-supported shapes, dimensions, materials, coordinate transforms and room bounds from the rest of this prompt.'''
ARRAYS = ('points', 'vertices', 'faces', 'cloth')


def compact_detail_prompt(canonical_prompt):
    """Replace, rather than append to, the canonical per-part-array guidance."""
    if not isinstance(canonical_prompt, str):
        raise TypeError('Expected a text detail prompt')
    lines = canonical_prompt.split('\n', 2)
    if len(lines) != 3 or not lines[0].startswith("Return ONLY the requested custom group's parts"):
        raise ValueError('Expected a staged detail prompt with a structured-geometry paragraph')
    # Keep the original spatial/coordinate contract verbatim. Only the
    # per-part wire-format paragraph is replaced; dropping those bounds would
    # save text at the cost of geometry fidelity.
    tail = lines[1].split(' Positions+-50m', 1)
    if len(tail) != 2:
        raise ValueError('Staged detail spatial guidance changed; update sparse prompt adapter')
    replacement = lines[0] + '\n' + GEOMETRY_GUIDANCE + ' Positions+-50m' + tail[1]
    return replacement + '\n' + lines[2]


def _arrays(kind):
    if kind not in FIELDS_BY_TYPE:
        raise ValueError('Unknown compact detail part type')
    return tuple(key for key in FIELDS_BY_TYPE[kind] if key in ARRAYS)


def detail_schema(group_id, max_parts=60, allowed_material_ids=None):
    data = {'type': 'array', 'items': {'type': 'number', 'minimum': -10000, 'maximum': 10000},
            'minItems': 10, 'maxItems': 10}
    geometry = obj({'part_index': {'type': 'integer', 'minimum': 0, 'maximum': max_parts - 1},
        **{key: copy.deepcopy(PART_SCHEMA['properties'][key]) for key in ARRAYS}})
    material_id = copy.deepcopy(PART_SCHEMA['properties']['material_id'])
    if allowed_material_ids is not None:
        material_id['enum'] = ['', *allowed_material_ids]
    return obj({'wire_format': {'type': 'string', 'enum': [FORMAT]},
        'group_id': {'type': 'string', 'enum': [group_id]},
        'parts': {'type': 'array', 'minItems': 1, 'maxItems': max_parts, 'items': obj({
            'type': copy.deepcopy(PART_SCHEMA['properties']['type']),
            'material_id': material_id, 'data': data})},
        'geometry': {'type': 'array', 'maxItems': max_parts, 'items': geometry}})


def compact_detail(detail, max_parts=60, allowed_material_ids=None):
    from staged_analysis import detail_schema as canonical_schema
    _walk_wire(detail, canonical_schema(detail['group_id'], max_parts, allowed_material_ids))
    result = {'wire_format': FORMAT, 'group_id': detail['group_id'], 'parts': [], 'geometry': []}
    for index, item in enumerate(detail['parts']):
        fields = _arrays(item['type'])
        if any(item[key] != [] for key in ARRAYS if key not in fields):
            raise ValueError('Refusing to discard meaningful native detail array')
        result['parts'].append({key: copy.deepcopy(item[key]) for key in ('type', 'material_id', 'data')})
        if fields:
            result['geometry'].append({'part_index': index, **{key: copy.deepcopy(item[key]) for key in ARRAYS}})
    _walk_wire(result, detail_schema(detail['group_id'], max_parts, allowed_material_ids))
    return result


def expand_detail(raw, group_id, max_parts=60, allowed_material_ids=None):
    _walk_wire(raw, detail_schema(group_id, max_parts, allowed_material_ids))
    payloads = {}
    for item in raw['geometry']:
        index = item['part_index']
        if index >= len(raw['parts']) or index in payloads:
            raise ValueError('Duplicate or out-of-range compact geometry part_index')
        fields = _arrays(raw['parts'][index]['type'])
        if not fields or any(item[key] != [] for key in ARRAYS if key not in fields):
            raise ValueError('Compact geometry fields do not match part type')
        if any(len(item[key]) < {'points': 2, 'vertices': 3, 'faces': 1, 'cloth': 1}[key] for key in fields):
            raise ValueError('Compact detail is missing required geometry')
        payloads[index] = item
    result = {'group_id': group_id, 'parts': []}
    for index, item in enumerate(raw['parts']):
        if _arrays(item['type']) and index not in payloads:
            raise ValueError('Compact detail is missing required geometry')
        payload = payloads.get(index, {})
        result['parts'].append({**copy.deepcopy(item),
            **{key: copy.deepcopy(payload.get(key, [])) for key in ARRAYS}})
    return result
