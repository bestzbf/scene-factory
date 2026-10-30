"""Explicit base-object placement plus the existing typed-parts wire codec.

This opt-in transport format removes a common bottom-versus-centre Z ambiguity.
It never infers which objects belong on the floor: the model must declare floor,
an exact flat support instance, or ordinary world bottom-centre coordinates.
Identical child/support repeat grids can bind corresponding declared instances;
the decoder never moves their XY coordinates to make a contact valid.
Saved bundles remain canonical V3; placement evidence is a separate receipt.
"""
from __future__ import annotations

import copy
import math

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

import compact_scene
from schema import expand_repeat_positions, obj
from spec_v2 import _walk

FORMAT_FIELD = compact_scene.FORMAT_FIELD
FORMAT_VERSION = 'anchored_parts_v1'
FLAT_TOP_CATEGORIES = frozenset(('desk', 'table', 'cabinet', 'shelf', 'box'))
CONTACT_TOLERANCE_METRES = .02
MAX_CLEARANCE_METRES = .03

CONTRACT = (compact_scene.CONTRACT
    .replace(compact_scene.FORMAT_VERSION, FORMAT_VERSION)
    .replace('Only custom_objects[].parts change:',
             'Custom parts use typed fields; base objects also use BASE PLACEMENT below:')
    .replace('Keep all other scene fields and supplied view identities.',
             'Keep unrelated scene fields and supplied view identities.') + """

BASE PLACEMENT: In shared_scene.base_scene.objects, replace position with ONE
placement object. Choose {"mode":"floor","xy":[x,y]} for an object whose bottom
rests on the room floor; the decoder sets bottom Z=0. Choose
{"mode":"on_top","xy":[x,y],"support_id":"exact_instance_id","clearance":0}
for bottom contact with the OUTER TOP of a flat desk, table, cabinet, shelf or
box. The decoder sets bottom Z=support bottom Z+support height+clearance.
Clearance must be 0..0.03 metres; zero means contact. This does not describe an
internal shelf, seat, bed mattress, sloped surface or an arbitrary custom mesh.
Use {"mode":"world","position":[x,y,z]} for wall/ceiling mounting, a raised
floor/platform, non-flat or unknown supports, and all other explicit placement.
World position is always BOTTOM CENTRE, never centre or tabletop height.

on_top support_id must identify an exact BASE instance: a singleton uses its id;
a repeated group uses id__i_j with zero-based X/Y indices. A repeated group root
is ambiguous and forbidden. When child and support have exactly identical
repeat AND spacing arrays, both are repeated, and support_id is the support's
id__0_0, child copy [i,j] binds to support id__i_j. Otherwise every child copy
must fit the one declared exact support instance. The decoder checks each child
bottom-centre XY against its resolved support's yaw-rotated top footprint. A
repeat mismatch never searches for another support or moves XY. Supports may
occur in any output order, but no cycles
are allowed. floor placement is forbidden when architecture.omit_surfaces
contains floor; use explicit world positioning for custom/raised floors.
Keep all other canonical fields unchanged and use typed custom-part fields.
""").strip()


def _full_schema():
    from vision_multiview import SCHEMA
    return SCHEMA


def _placement_rule(position_rule):
    if (not isinstance(position_rule, dict) or position_rule.get('type') != 'array'
            or position_rule.get('minItems') != 3
            or position_rule.get('maxItems') != 3
            or position_rule.get('items', {}).get('type') != 'number'):
        raise ValueError('Expected bounded canonical base-object position schema')
    xy = copy.deepcopy(position_rule)
    xy['minItems'] = xy['maxItems'] = 2
    return {'anyOf': [
        obj({'mode': {'type': 'string', 'enum': ['floor']}, 'xy': copy.deepcopy(xy)}),
        obj({'mode': {'type': 'string', 'enum': ['on_top']}, 'xy': copy.deepcopy(xy),
             'support_id': {'type': 'string', 'minLength': 1, 'maxLength': 12000},
             'clearance': {'type': 'number', 'minimum': 0, 'maximum': MAX_CLEARANCE_METRES}}),
        obj({'mode': {'type': 'string', 'enum': ['world']},
             'position': copy.deepcopy(position_rule)}),
    ]}


def compact_response_schema(full_schema):
    """Retain every supplied schema/identity constraint in a strict wire copy."""
    result = compact_scene.compact_response_schema(full_schema)
    try:
        rule = result['properties']['shared_scene']['properties']['base_scene']['properties']['objects']['items']
        position = rule['properties'].pop('position')
        rule['properties']['placement'] = _placement_rule(position)
        rule['required'] = ['placement' if field == 'position' else field
                            for field in rule['required']]
    except (KeyError, TypeError):
        raise ValueError('Expected known canonical base-object schema') from None
    result['properties'][FORMAT_FIELD]['enum'] = [FORMAT_VERSION]
    return result


# A short alias is useful for callers which name their schema factory uniformly.
response_schema = compact_response_schema


def _instance_index(objects, scene):
    """Index exact instances, rejecting root/expanded namespace ambiguity."""
    roots = {}
    instances = {}
    for index, item in enumerate(objects):
        identifier = item['id']
        if not identifier.strip() or identifier in roots:
            raise ValueError('base_scene.objects: empty or duplicate ids')
        roots[identifier] = index
        slots = expand_repeat_positions([0, 0, 0], item['repeat'], item['spacing'],
                                        'base_scene.objects')
        for i, j, _ in slots:
            exact_id = identifier if item['repeat'] == [1, 1] else f'{identifier}__{i}_{j}'
            if exact_id in instances:
                raise ValueError('Expanded base object ids collide')
            instances[exact_id] = (index, i, j)
    for identifier, index in roots.items():
        if identifier in instances and instances[identifier][0] != index:
            raise ValueError('Base root and expanded instance namespaces collide')
    other_ids = [item['id'] for key in ('custom_objects', 'lights') for item in scene[key]]
    if (len(other_ids) != len(set(other_ids))
            or set(other_ids) & (set(roots) | set(instances))
            or 'floor' in set(other_ids) | set(roots) | set(instances)):
        raise ValueError('Object/light/floor namespaces collide')
    return roots, instances


def _on_support(xy, parent_position, parent):
    """Check declared contact centre; no relocation or overhang inference."""
    angle = math.radians(parent['rotation_z_deg'])
    dx, dy = xy[0] - parent_position[0], xy[1] - parent_position[1]
    local_x = math.cos(angle) * dx + math.sin(angle) * dy
    local_y = -math.sin(angle) * dx + math.cos(angle) * dy
    tolerance = CONTACT_TOLERANCE_METRES + 1e-12
    if (abs(local_x) > parent['dimensions'][0] / 2 + tolerance
            or abs(local_y) > parent['dimensions'][1] / 2 + tolerance):
        raise ValueError('on_top bottom-centre XY is outside the support top footprint')


def _decode(raw):
    if not isinstance(raw, dict) or raw.get(FORMAT_FIELD) != FORMAT_VERSION:
        raise ValueError('Expected explicit wire_format=' + FORMAT_VERSION)
    # The canonical walker has no union support; use the strict transport
    # validator before transforming any placement or typed geometry branches.
    from model_api import _walk_wire
    try:
        _walk_wire(raw, compact_response_schema(_full_schema()))
    except OverflowError:
        raise ValueError('Wire number exceeds the finite allowed range') from None
    result = copy.deepcopy(raw)
    scene = result['shared_scene']
    objects = scene['base_scene']['objects']
    roots, instances = _instance_index(objects, scene)
    position_rule = _full_schema()['properties']['shared_scene']['properties']['base_scene']['properties']['objects']['items']['properties']['position']
    placements = [copy.deepcopy(item['placement']) for item in objects]
    resolved = {}
    resolving = set()
    support_tops = {}
    resolved_supports = {}
    support_bindings = {}

    def position_for_instance(exact_id):
        index, i, j = instances[exact_id]
        root_position = resolve(index)
        item = objects[index]
        position = [root_position[0] + i * item['spacing'][0],
                    root_position[1] + j * item['spacing'][1], root_position[2]]
        _walk(position, position_rule, 'base_scene.objects.expanded.position')
        return position

    def resolve(index):
        if index in resolved:
            return resolved[index]
        if index in resolving:
            raise ValueError('Cyclic base-object placement support graph')
        resolving.add(index)
        item, placement = objects[index], placements[index]
        mode = placement['mode']
        if mode == 'world':
            position = placement['position'][:]
        elif mode == 'floor':
            if 'floor' in result['architecture']['omit_surfaces']:
                raise ValueError('floor placement requires the canonical room floor')
            position = placement['xy'] + [0]
        else:
            support_id = placement['support_id']
            if support_id in roots and objects[roots[support_id]]['repeat'] != [1, 1]:
                raise ValueError('on_top support is an ambiguous repeated group root; use exact instance id')
            if support_id not in instances:
                raise ValueError('on_top requires an exact existing base support instance')
            parent_index, support_i, support_j = instances[support_id]
            parent = objects[parent_index]
            if parent['category'] not in FLAT_TOP_CATEGORIES:
                raise ValueError('on_top support category has no supported flat outer top')
            corresponding = (item['repeat'] != [1, 1]
                             and item['repeat'] == parent['repeat']
                             and item['spacing'] == parent['spacing']
                             and (support_i, support_j) == (0, 0))
            support_bindings[index] = ('matching_repeat_index' if corresponding
                                       else 'exact_instance')
            for child_i, child_j, _ in expand_repeat_positions(
                    [0, 0, 0], item['repeat'], item['spacing'],
                    'base_scene.objects.on_top.repeat'):
                exact_support = (f'{parent["id"]}__{child_i}_{child_j}'
                                 if corresponding else support_id)
                parent_position = position_for_instance(exact_support)
                child_xy = [placement['xy'][0] + child_i * item['spacing'][0],
                            placement['xy'][1] + child_j * item['spacing'][1]]
                try:
                    _on_support(child_xy, parent_position, parent)
                except ValueError as error:
                    child_id = (item['id'] if item['repeat'] == [1, 1]
                                else f'{item["id"]}__{child_i}_{child_j}')
                    raise ValueError(f'{child_id} on support {exact_support}: {error}') from None
                key = (index, child_i, child_j)
                resolved_supports[key] = exact_support
                support_tops[key] = parent_position[2] + parent['dimensions'][2]
            # Canonical base repeats share one Z. Parent repetitions only
            # translate in XY, so every checked corresponding top has this Z.
            top = support_tops[index, 0, 0]
            position = placement['xy'] + [top + placement['clearance']]
        _walk(position, position_rule, 'base_scene.objects.position')
        resolved[index] = position
        resolving.remove(index)
        return position

    for index, item in enumerate(objects):
        item['position'] = resolve(index)
        del item['placement']
    # Validate every instance, including repeated XY offsets, rather than only
    # the root positions that happen to be stored in canonical JSON.
    evidence = []
    for exact_id, (index, i, j) in instances.items():
        placement = placements[index]
        evidence.append({
            'object_id': exact_id, 'base_object_id': objects[index]['id'],
            'repeat_index': [i, j], 'placement': copy.deepcopy(placement),
            'resolved_bottom_position': position_for_instance(exact_id),
            'support_id': resolved_supports.get((index, i, j), ''),
            'support_top_z': support_tops.get((index, i, j)),
            'support_binding': support_bindings.get(index, ''),
        })
    result[FORMAT_FIELD] = compact_scene.FORMAT_VERSION
    canonical = compact_scene.expand_response(result)
    return canonical, {'format': FORMAT_VERSION, 'instances': evidence}


def expand_response(raw):
    """Resolve explicit placements into a fresh canonical V3 scene bundle."""
    return _decode(raw)[0]


def placement_report(raw, expanded=None):
    """Return deterministic per-instance placement evidence, without mutation.

    An optional caller-supplied expansion must match a fresh decode exactly;
    it cannot substitute unchecked coordinates into the evidence receipt.
    """
    canonical, report = _decode(raw)
    if expanded is not None and expanded != canonical:
        raise ValueError('Placement evidence expansion does not match the raw wire response')
    return report
