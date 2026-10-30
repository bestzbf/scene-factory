"""Versioned, opt-in typed-parts wire encoding for V3 scene responses.

Only unused per-part fields are omitted. This is semantic compaction, not a
byte-for-byte archive encoding: expansion restores canonical zero/empty fields.
In particular ``generate_v2.build_custom`` ignores size for curves/meshes,
radius for boxes/meshes/image planes, and uses XYZ sphere size in preference to
radius when every axis is positive. All fields used by rendering are retained.
Cylinder size.x/y and image_plane size.y are kept verbatim, despite being unused,
because the whole size vector is already required. No topology, shape,
transforms, materials, view identities or scene data are guessed or repaired.

The compact format has an explicit top-level marker; ordinary V3 data must not
be guessed to be compact. Old persisted bundles remain ordinary V3. Callers
must still apply the existing whole-scene semantic validation/normalization and
trusted view-identity checks after decoding, exactly as for ordinary responses.
"""
from __future__ import annotations

import copy
import json

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

from schema import obj
from spec_v2 import PART_SCHEMA, _walk, part as canonical_part

FORMAT_FIELD = 'wire_format'
FORMAT_VERSION = 'typed_parts_v1'
COMMON = ('type', 'position', 'rotation_degrees', 'material_id')
FIELDS_BY_TYPE = {
    'box': ('size',),
    'sphere': ('radius',),
    'cylinder': ('size', 'radius'),
    'curve': ('points', 'radius'),
    'polyline': ('points', 'radius'),
    'mesh': ('vertices', 'faces'),
    'image_plane': ('size',),
    'curtain': ('size', 'cloth'),
}
# Radius-only avoids a zero size vector for balls. The second variant preserves
# both size and radius exactly for ellipsoids and legacy partial-size spheres.
# The canonical renderer chooses size only when ALL three axes are positive.
FIELD_VARIANTS = {**{kind: (fields,) for kind, fields in FIELDS_BY_TYPE.items()},
                  'sphere': (('radius',), ('size', 'radius'))}
# Only these non-default values may be discarded: both the semantic validator
# and geometry builder disregard them for the listed type. All other omitted
# values must equal canonical_part's empty/zero default or compaction fails.
IGNORED_NONDEFAULT = {
    'box': {'radius'}, 'sphere': set(), 'cylinder': set(),
    'curve': {'size'}, 'polyline': {'size'}, 'mesh': {'size', 'radius'},
    'image_plane': {'radius'}, 'curtain': set(),
}
CONTRACT = """
WIRE FORMAT typed_parts_v1: Return wire_format="typed_parts_v1" with the V3
wrapper. Only custom_objects[].parts change: every part requires type, position,
rotation_degrees and material_id, followed by ONLY these type-specific fields:
box size; sphere radius OR size+radius; cylinder size+radius; curve or polyline points+radius;
mesh vertices+faces; image_plane size; curtain size+cloth. Omit all other part
fields, including unused empty arrays. This wire rule replaces full-part field
requirements in the general scene contract. The decoder only restores unused
zero/empty defaults. Keep all other scene fields and supplied view identities.
Sphere positive XYZ size means ellipsoid and overrides radius (use radius=0);
otherwise use positive radius alone for a ball.
Mesh needs >=3 vertices and >=1 nondegenerate
face with distinct valid vertex indices; never output vertices without faces.
Curve and polyline need >=2 distinct points and positive radius; use polyline for rigid edges and exact corners. Curtain needs exactly one
cloth record and the existing fold/thickness/size constraints. No topology or
geometry is supplied by the decoder; output complete meaningful geometry.
""".strip()


def _part_rule(full_part_schema):
    """Build strict disjoint type branches, preserving every supplied bound."""
    if (not isinstance(full_part_schema, dict)
            or full_part_schema.get('type') != 'object'
            or full_part_schema.get('additionalProperties') is not False
            or set(full_part_schema.get('properties', {})) != set(PART_SCHEMA['properties'])
            or set(full_part_schema.get('required', [])) != set(PART_SCHEMA['required'])):
        raise ValueError('Unsupported full part schema; compact codec requires the known V2 part fields')
    fields = full_part_schema['properties']
    if set(fields.get('type', {}).get('enum', [])) != set(FIELDS_BY_TYPE):
        raise ValueError('Unsupported part types for typed_parts_v1')
    branches = []
    for kind, variants in FIELD_VARIANTS.items():
        if kind == 'polyline':
            # Curves and polylines share the same coordinate payload; only
            # Blender interpolation differs. One strict branch keeps the
            # encoded schema within the provider-compatible branch limit.
            continue
        for geometry in variants:
            kept = set(COMMON + geometry)
            properties = {key: copy.deepcopy(value) for key, value in fields.items() if key in kept}
            properties['type']['enum'] = ['curve', 'polyline'] if kind == 'curve' else [kind]
            if kind == 'mesh':
                properties['vertices']['minItems'] = max(3, properties['vertices'].get('minItems', 0))
                properties['faces']['minItems'] = max(1, properties['faces'].get('minItems', 0))
            elif kind in ('curve', 'polyline'):
                properties['points']['minItems'] = max(2, properties['points'].get('minItems', 0))
            elif kind == 'curtain':
                properties['cloth']['minItems'] = 1
            branches.append(obj(properties))
    return {'anyOf': _merge_equivalent_type_branches(branches)}


def _merge_equivalent_type_branches(branches):
    """Merge strict object branches differing only in a type enum.

    This reduces schema transport size without weakening constraints: two
    branches are equivalent only when their complete schemas are identical
    after removing ``properties.type.enum``. The first branch's property order
    is retained and enum values are appended in their original stable order.
    """
    merged = []
    positions = {}
    for branch in branches:
        key_branch = copy.deepcopy(branch)
        try:
            type_rule = key_branch['properties']['type']
            enum = type_rule.pop('enum')
        except (KeyError, TypeError):
            # _part_rule constructs the input branches internally, so this is
            # defensive. Let schema validation reject malformed structures.
            merged.append(copy.deepcopy(branch))
            continue
        key = json.dumps(key_branch, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False)
        existing_index = positions.get(key)
        if existing_index is None:
            positions[key] = len(merged)
            merged.append(copy.deepcopy(branch))
            continue
        existing_enum = merged[existing_index]['properties']['type']['enum']
        for value in enum:
            if value not in existing_enum:
                existing_enum.append(value)
    return merged


def compact_response_schema(full_schema):
    """Copy an ordinary V3 response schema; retain all identity restrictions."""
    result = copy.deepcopy(full_schema)
    try:
        if (result['type'] != 'object' or result['additionalProperties'] is not False
                or set(result['required']) != set(result['properties'])
                or FORMAT_FIELD in result['properties']
                or result['properties']['schema_version'].get('enum') != [3]):
            raise ValueError('Expected an ordinary strict V3 response schema')
        parts = result['properties']['shared_scene']['properties']['custom_objects']['items']['properties']['parts']
        parts['items'] = _part_rule(parts['items'])
    except (KeyError, TypeError):
        raise ValueError('Expected an ordinary strict V3 response schema') from None
    result['properties'][FORMAT_FIELD] = {'type': 'string', 'enum': [FORMAT_VERSION]}
    result['required'].append(FORMAT_FIELD)
    return result


def _full_schema():
    # Lazy import avoids a cycle when vision_multiview opts in to this codec.
    from vision_multiview import SCHEMA
    return SCHEMA


_PART_BRANCHES = _part_rule(PART_SCHEMA)['anyOf']


def _branch(kind, geometry):
    if not isinstance(kind, str) or kind not in FIELDS_BY_TYPE:
        raise ValueError('Unknown compact part type')
    kept = set(COMMON + tuple(geometry))
    return next(rule for rule in _PART_BRANCHES
                if kind in rule['properties']['type']['enum']
                and set(rule['properties']) == kept)


def _validate_geometry(bundle):
    """Use canonical geometry/material checks on a private normalized view.

    The normalizer permits repeated-object support aliases which do not yet
    identify a single instance. Clear supports only in this private validation
    view so decoding does not reject those before the existing normalizer runs.
    Likewise expand exact repeated-group material aliases only in that private
    view. Preserve caller-visible aliases so the ordinary response normalizer
    can expand them and record its evidence. Unknown aliases still fail the
    canonical checks. Final multiview validation must check the original
    support graph and trusted source identities.
    """
    from spec_v2 import validate
    from vision_v2 import expand_repeated_base_material_overrides
    scene = bundle['shared_scene']
    validation_view = {**scene, 'custom_objects': [
        {**group, 'support_id': ''} for group in scene['custom_objects']]}
    validation_view, _ = expand_repeated_base_material_overrides(validation_view)
    validate(validation_view)


def _groups(bundle):
    if not isinstance(bundle, dict) or bundle.get('schema_version') != 3:
        raise ValueError('Expected version-3 scene response')
    try:
        groups = bundle['shared_scene']['custom_objects']
    except (KeyError, TypeError):
        raise ValueError('Expected V3 shared_scene.custom_objects') from None
    if not isinstance(groups, list):
        raise ValueError('custom_objects must be an array')
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get('parts'), list):
            raise ValueError('Custom object parts must be an array')
    return groups


def compact_response(bundle):
    """Encode a syntactically valid ordinary V3 bundle without mutating it.

    Known legacy additive omissions (cloth and V2 extension arrays) are allowed
    as in the canonical validator. Every meaningful value remains unchanged.
    Support aliases and the complete view contract are validated by the caller.
    """
    _walk(bundle, _full_schema(), 'scene')
    _validate_geometry(bundle)
    result = copy.deepcopy(bundle)
    for group in _groups(result):
        for index, item in enumerate(group['parts']):
            kind = item['type']
            geometry = FIELDS_BY_TYPE[kind]
            if kind == 'sphere':
                size = item['size']
                if size != [0, 0, 0]:
                    geometry = ('size', 'radius')
            keep = set(COMMON + geometry)
            defaults = canonical_part(kind)
            for key, value in item.items():
                if key not in keep and value != defaults[key] and key not in IGNORED_NONDEFAULT[kind]:
                    raise ValueError('Refusing to discard meaningful part field: ' + key)
            encoded = {key: value for key, value in item.items() if key in keep}
            _walk(encoded, _branch(kind, geometry), 'compact_part')
            group['parts'][index] = encoded
    result[FORMAT_FIELD] = FORMAT_VERSION
    return result


def expand_response(raw):
    """Decode only explicitly marked compact V3; reject extra/malformed data.

    No mesh faces, points, cloth, meaningful sizes or references are invented.
    The raw model response and all other data are preserved by deep copy.
    """
    if not isinstance(raw, dict) or raw.get(FORMAT_FIELD) != FORMAT_VERSION:
        raise ValueError('Expected explicit wire_format=' + FORMAT_VERSION)
    result = copy.deepcopy(raw)
    del result[FORMAT_FIELD]
    for group in _groups(result):
        for index, item in enumerate(group['parts']):
            if not isinstance(item, dict):
                raise ValueError('Compact part must be an object')
            kind = item.get('type')
            if not isinstance(kind, str) or kind not in FIELD_VARIANTS:
                raise ValueError('Unknown compact part type')
            geometry = tuple(keys for keys in FIELD_VARIANTS[kind]
                             if set(COMMON + keys) == set(item))
            if len(geometry) != 1:
                raise ValueError('Compact part has ambiguous geometry fields')
            geometry = geometry[0]
            _walk(item, _branch(kind, geometry), 'compact_part')
            expanded = canonical_part(kind, **{key: value for key, value in item.items() if key != 'type'})
            group['parts'][index] = expanded
    _walk(result, _full_schema(), 'scene')
    _validate_geometry(result)
    return result
