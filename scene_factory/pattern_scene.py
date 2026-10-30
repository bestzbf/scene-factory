"""Bounded, explicit part arrays layered on the anchored typed-parts wire.

An array is a transport instruction, not a procedural furniture generator. Its
source indices, counts and translations are supplied by the model; no geometry,
material, support or detail is inferred. Expansion changes only part positions
in the custom group's local axes. The canonical renderer remains unchanged.
"""
from __future__ import annotations

import copy
import math
import struct

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

import anchored_scene
from schema import obj
from spec_v2 import (MAX_CURVE_POINTS, MAX_FACE_INDICES, MAX_PARTS, MAX_VERTICES,
                     PART_SCHEMA, _walk)

FORMAT_FIELD = anchored_scene.FORMAT_FIELD
FORMAT_VERSION = 'anchored_patterns_v1'
MAX_GROUP_PARTS = 100

CONTRACT = (anchored_scene.CONTRACT.replace(anchored_scene.FORMAT_VERSION,
                                           FORMAT_VERSION) + """

PART ARRAYS: Every custom_objects group also requires part_arrays (use [] when
there are no exact repeats). Each entry has ONLY part_indices, count, spacing:
{"part_indices":[0,2],"count":[nx,ny,nz],"spacing":[sx,sy,sz]}.
part_indices are zero-based indices into that group's ORIGINAL typed parts.
Select each source index at most once across ALL arrays in that group. Each
count is an integer 1..100, its product must be >=2, and every axis with count>1
must have a nonzero spacing in metres. Negative spacing is valid. The decoder
keeps source part order; at each selected source slot it emits count_X *
count_Y * count_Z copies with X outermost, Y middle, Z innermost, including the
original [0,0,0] slot exactly once. Unselected parts are retained exactly once.
For each copy, position = source.position + repeat_index * spacing along the
GROUP'S LOCAL XYZ axes, independent of the part's own rotation. Rotations,
points, vertices, faces, sizes, materials and cloth parameters remain unchanged.
Do not both list repeated parts explicitly and array them. Arrays may select
multiple different prototypes for structures such as two shelf uprights or a
ceiling rail pair, but they must be exact translated copies, not approximate
variations. Return distinct parts for any visible difference. Expanded geometry
must stay within canonical bounds, <=100 parts per group and <=800 parts total,
including all repeated mesh/curve/cloth topology and existing scene budgets.
This encoding never creates unobserved detail or reduces image resolution.
""").strip()


def _full_schema():
    from vision_multiview import SCHEMA
    return SCHEMA


def compact_response_schema(full_schema):
    """Copy the strict anchored schema, preserving caller identity constraints."""
    result = anchored_scene.compact_response_schema(full_schema)
    try:
        group = result['properties']['shared_scene']['properties']['custom_objects']['items']
        position = group['properties']['position']
        if (position.get('type') != 'array' or position.get('minItems') != 3
                or position.get('maxItems') != 3
                or position.get('items', {}).get('type') != 'number'
                or 'part_arrays' in group['properties']):
            raise ValueError('Expected known custom group position schema')
        group['properties']['part_arrays'] = {
            'type': 'array', 'maxItems': MAX_GROUP_PARTS,
            'items': obj({
                'part_indices': {'type': 'array', 'minItems': 1,
                                 'maxItems': MAX_GROUP_PARTS,
                                 'items': {'type': 'integer', 'minimum': 0,
                                           'maximum': MAX_GROUP_PARTS - 1}},
                'count': {'type': 'array', 'minItems': 3, 'maxItems': 3,
                          'items': {'type': 'integer', 'minimum': 1,
                                    'maximum': MAX_GROUP_PARTS}},
                'spacing': copy.deepcopy(position),
            }),
        }
        group['required'].append('part_arrays')
    except (KeyError, TypeError):
        raise ValueError('Expected known canonical custom-object schema') from None
    result['properties'][FORMAT_FIELD]['enum'] = [FORMAT_VERSION]
    return result


response_schema = compact_response_schema


def _geometry_cost(item):
    """Mirror spec_v2.validate's existing aggregate topology accounting.

    These costs are only an allocation preflight. The final anchored decoder
    still invokes the authoritative canonical structural/geometry validators.
    """
    vertices = len(item.get('vertices', []))
    indices = sum(len(face) for face in item.get('faces', []))
    points = len(item.get('points', []))
    if item['type'] == 'curtain':
        cloth = item['cloth'][0]
        nx, nz = cloth['segments_x'], cloth['segments_z']
        vertices += (nx + 1) * (nz + 1) * 2
        indices += nx * nz * 8 + (nx + nz) * 8
    return vertices, indices, points


def _float32(value):
    """Blender object coordinates are single precision in the render stage."""
    return struct.unpack('f', struct.pack('f', value))[0]


def _preflight(raw):
    """Validate and count every expansion before allocating a copied part."""
    if not isinstance(raw, dict) or raw.get(FORMAT_FIELD) != FORMAT_VERSION:
        raise ValueError('Expected explicit wire_format=' + FORMAT_VERSION)
    # Reuse the same private strict-union walker as anchored_scene._decode.
    # The endpoint check below uses spec_v2._walk with the canonical position
    # rule; it does not replace the final anchored/canonical validation pass.
    from model_api import _walk_wire
    try:
        _walk_wire(raw, compact_response_schema(_full_schema()))
    except OverflowError:
        raise ValueError('Wire number exceeds the finite allowed range') from None

    totals = {'groups': 0, 'source_parts': 0, 'part_arrays': 0,
              'expanded_parts': 0, 'vertices': 0, 'face_indices': 0,
              'curve_points': 0}
    plans = []
    for group in raw['shared_scene']['custom_objects']:
        parts = group['parts']
        selected = {}
        group_count = len(parts)
        for array_index, array in enumerate(group['part_arrays']):
            copies = math.prod(array['count'])
            if copies < 2:
                raise ValueError('part_arrays: count product must be at least two')
            if any(count > 1 and step == 0
                   for count, step in zip(array['count'], array['spacing'])):
                raise ValueError('part_arrays: every active axis needs nonzero spacing')
            for source_index in array['part_indices']:
                if source_index >= len(parts):
                    raise ValueError('part_arrays: source part index is out of range')
                if source_index in selected:
                    raise ValueError('part_arrays: source part index selected more than once')
                selected[source_index] = (array_index, array)
            group_count += len(array['part_indices']) * (copies - 1)
            if group_count > MAX_GROUP_PARTS:
                raise ValueError('part_arrays: expanded group exceeds 100 parts')
        totals['groups'] += 1
        totals['source_parts'] += len(parts)
        totals['part_arrays'] += len(group['part_arrays'])
        totals['expanded_parts'] += group_count
        if totals['expanded_parts'] > MAX_PARTS:
            raise ValueError('part_arrays: expanded scene exceeds total custom geometry budget')
        for source_index, item in enumerate(parts):
            selection = selected.get(source_index)
            copies = math.prod(selection[1]['count']) if selection else 1
            if selection:
                array = selection[1]
                # Every coordinate varies monotonically along its own group
                # axis. Check its <=100 scalar coordinates before allocation:
                # floating point can round distinct nonzero offsets back onto
                # the same origin (for example 1.0 + 1e-20 == 1.0). This uses
                # the same exact-coordinate uniqueness rule as the existing
                # schema repeat expander, scoped to each prototype so separate
                # authored prototypes may intentionally coincide.
                endpoint = []
                for position, count, step in zip(item['position'], array['count'], array['spacing']):
                    coordinates = [position + index * step for index in range(count)]
                    if len(set(coordinates)) != len(coordinates):
                        raise ValueError('part_arrays: repeat/spacing produces duplicate expanded positions')
                    # A JSON coordinate can be distinct in Python but collapse
                    # when assigned to Blender's float32 transform fields.
                    render_coordinates = [_float32(coordinate) for coordinate in coordinates]
                    if len(set(render_coordinates)) != len(render_coordinates):
                        raise ValueError('part_arrays: repeat/spacing produces duplicate expanded positions')
                    endpoint.append(coordinates[-1])
                _walk(endpoint, PART_SCHEMA['properties']['position'],
                      'part_arrays.expanded.position')
            vertices, indices, points = _geometry_cost(item)
            totals['vertices'] += vertices * copies
            totals['face_indices'] += indices * copies
            totals['curve_points'] += points * copies
            if (totals['vertices'] > MAX_VERTICES
                    or totals['face_indices'] > MAX_FACE_INDICES
                    or totals['curve_points'] > MAX_CURVE_POINTS):
                raise ValueError('part_arrays: expanded topology exceeds total custom geometry budget')
        plans.append(selected)
    return plans, totals


def _anchored_wire(raw):
    plans, totals = _preflight(raw)
    result = copy.deepcopy(raw)
    result[FORMAT_FIELD] = anchored_scene.FORMAT_VERSION
    report = {'format': FORMAT_VERSION, 'counts': totals, 'groups': []}
    for group, selected in zip(result['shared_scene']['custom_objects'], plans):
        sources = group['parts']
        parts = []
        evidence = []
        for source_index, source in enumerate(sources):
            selection = selected.get(source_index)
            array_index = selection[0] if selection else None
            counts = selection[1]['count'] if selection else [1, 1, 1]
            spacing = selection[1]['spacing'] if selection else [0, 0, 0]
            for i in range(counts[0]):
                for j in range(counts[1]):
                    for k in range(counts[2]):
                        repeat = [i, j, k]
                        item = copy.deepcopy(source)
                        item['position'] = [position + offset * step
                                            for position, offset, step
                                            in zip(source['position'], repeat, spacing)]
                        evidence.append({'part_index': len(parts),
                                         'source_part_index': source_index,
                                         'array_index': array_index,
                                         'repeat_index': repeat})
                        parts.append(item)
        group['parts'] = parts
        del group['part_arrays']
        report['groups'].append({'group_id': group['id'],
                                 'source_part_count': len(sources),
                                 'expanded_part_count': len(parts),
                                 'part_array_count': len(set(value[0] for value in selected.values())),
                                 'parts': evidence})
    return result, report


def expand_response(raw):
    """Expand declared translations, then use the existing anchored decoder."""
    converted, _ = _anchored_wire(raw)
    return anchored_scene.expand_response(converted)


def placement_report(raw, expanded=None):
    """Forward exact base-placement evidence using the converted anchored wire."""
    converted, _ = _anchored_wire(raw)
    return anchored_scene.placement_report(converted, expanded)


def pattern_report(raw, expanded=None):
    """Return deterministic output-part to source-part/repeat-index provenance.

    Always run canonical validation, even if the caller supplies an expansion.
    An unrelated or modified bundle cannot be attached to a valid receipt.
    """
    converted, report = _anchored_wire(raw)
    canonical = anchored_scene.expand_response(converted)
    if expanded is not None and expanded != canonical:
        raise ValueError('Pattern evidence expansion does not match the raw wire response')
    return report
