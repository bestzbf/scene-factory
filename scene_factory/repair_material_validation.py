"""Early, pure-Python material-selector checks against a rendered inventory.

This is a conservative precheck of unchanged source objects, not a substitute
for Blender's binding preflight. No bpy imports or material names from errors
are needed: the caller already supplies the full inventory in the repair
payload. All raised messages are fixed strings plus bounded numeric indices,
so model-supplied names, paths or credentials cannot be echoed into feedback.
"""
from __future__ import annotations

from collections.abc import Mapping


_PLACEMENT_FIELDS = frozenset({'position', 'confidence'})
_ROLES = frozenset({'body', 'metal', 'glass', 'fabric'})
_ZERO_MATCH = (
    'Material selector matches no observed source slot. Use exact source_material, '
    'source_material_key or source_material_id from material_binding.available_slots; '
    'object_name is a mesh/component label and is not a material selector. '
    'Role, material_name and slot_index filters intersect. Use role=body/metal/glass/fabric '
    'with material_name="" and slot_index=-1 when the intended source role is sufficient.')
_CONFLICT = (
    'Material selectors assign different target materials to the same observed source slot. '
    'Remove the overlapping binding or narrow role/material_name/slot_index using '
    'material_binding.available_slots; each source slot may receive only one target material.')


def _source_signature(record):
    return {key: value for key, value in record.items() if key not in _PLACEMENT_FIELDS}


def _base(bundle):
    if not isinstance(bundle, Mapping):
        return None, None
    scene = bundle.get('shared_scene')
    if not isinstance(scene, Mapping) or not isinstance(scene.get('base_scene'), Mapping):
        return None, None
    return scene, scene['base_scene']


def _expanded_records(base):
    """Map expanded IDs to source definitions; never parse suffixes heuristically."""
    entries = base.get('objects')
    if not isinstance(entries, list):
        return None
    expanded = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get('id'), str):
            return None
        repeat = entry.get('repeat')
        if (not isinstance(repeat, list) or len(repeat) != 2
                or any(type(value) is not int or not 1 <= value <= 100 for value in repeat)):
            return None
        for i in range(repeat[0]):
            for j in range(repeat[1]):
                identifier = entry['id'] + (f'__{i}_{j}' if repeat != [1, 1] else '')
                if identifier in expanded:
                    return None
                expanded[identifier] = entry
    return expanded


def _matches(slot, selector):
    """Exact equivalent of base_material_binding._matches, without bpy import."""
    role = selector.get('role', 'all') or 'all'
    name = selector.get('material_name', '')
    index = selector.get('slot_index', -1)
    return ((role == 'all' or role == slot['role'])
            and (not name or name in {slot.get('source_material', ''),
                                     slot.get('source_material_id', ''), slot.get('source_material_key', '')})
            and (index < 0 or index == slot['slot_index']))


def _valid_slot(slot):
    return (isinstance(slot, Mapping) and isinstance(slot.get('object_id'), str)
            and isinstance(slot.get('object_name'), str)
            and type(slot.get('slot_index')) is int and slot['slot_index'] >= 0
            and slot.get('role') in _ROLES
            and all(isinstance(slot.get(key, ''), str)
                    for key in ('source_material', 'source_material_key', 'source_material_id')))


def validate_candidate_material_bindings(original, candidate, material_binding, external_assets=None):
    """Reject provably invalid selectors for unchanged rendered base instances.

    Args:
        original/candidate: validated V3 scene bundles.
        material_binding: current metadata.material_binding report containing
            ``available_slots`` captured BEFORE the current render's overrides.
        external_assets: optional current metadata.provenance.external_assets.

    Returns None on success or when the old inventory is not applicable. Raises
    ValueError with fixed, bounded, safe feedback on zero matches or conflicting
    target assignments. This does not grant acceptance or guarantee rebuilds.

    Geometry/material input changes skip the affected object. Changes to the
    scene seed/type skip every object. External library instances additionally
    skip when any source object definition/order changes (diversity selection
    can cascade), or any object's bottom height changes (selection eligibility).
    XY motion/confidence remain checkable. Rotation or bottom-height changes
    can alter library eligibility, even for an old procedural fallback, and
    defer to Blender's new inventory.
    """
    old_scene, old_base = _base(original)
    new_scene, new_base = _base(candidate)
    if old_base is None or new_base is None or not isinstance(material_binding, Mapping):
        return None
    slots = material_binding.get('available_slots')
    if not isinstance(slots, list):
        return None  # Older metadata does not carry this optional evidence.
    if any(old_scene.get(key) != new_scene.get(key) for key in ('scene_type', 'asset_library', 'asset_bindings')):
        return None
    if any(old_base.get(key) != new_base.get(key) for key in ('seed', 'scene_type', 'asset_library', 'asset_bindings')):
        return None
    if any(original.get(key) != candidate.get(key) for key in ('scene_id', 'asset_library', 'asset_bindings')):
        return None
    old_records, new_records = _expanded_records(old_base), _expanded_records(new_base)
    if old_records is None or new_records is None:
        return None
    if external_assets is None:
        external_ids = set()
    elif isinstance(external_assets, list) and all(isinstance(item, Mapping)
                                                 and isinstance(item.get('object_id'), str) for item in external_assets):
        external_ids = {item['object_id'] for item in external_assets}
    else:
        return None  # Unusable library provenance cannot justify trusting slots.
    old_order = [_source_signature(item) for item in old_base['objects']]
    new_order = [_source_signature(item) for item in new_base['objects']]
    library_reselection = old_order != new_order
    for identifier in old_records.keys() & new_records.keys():
        before_position = old_records[identifier].get('position')
        after_position = new_records[identifier].get('position')
        if (not isinstance(before_position, list) or len(before_position) != 3
                or not isinstance(after_position, list) or len(after_position) != 3
                or before_position[2] != after_position[2]):
            library_reselection = True
    valid_slots = {}
    invalid_objects = set()
    for slot in slots:
        if not _valid_slot(slot):
            if isinstance(slot, Mapping) and isinstance(slot.get('object_id'), str):
                invalid_objects.add(slot['object_id'])
            else:
                return None
            continue
        valid_slots.setdefault(slot['object_id'], []).append(slot)
    plans = {}
    overrides = new_scene.get('base_material_overrides', [])
    if not isinstance(overrides, list):
        return None  # Caller validates schema before invoking this precheck.
    for index, override in enumerate(overrides):
        if not isinstance(override, Mapping):
            return None
        identifier = override.get('object_id')
        if (not isinstance(identifier, str) or identifier not in old_records or identifier not in new_records
                or identifier in invalid_objects
                or _source_signature(old_records[identifier]) != _source_signature(new_records[identifier])
                or old_records[identifier]['position'][2] != new_records[identifier]['position'][2]
                or (identifier in external_ids and library_reselection)):
            continue
        selector = override.get('selector', {'role': 'all', 'material_name': '', 'slot_index': -1})
        if (not isinstance(selector, Mapping) or selector.get('role', 'all') not in _ROLES | {'all', ''}
                or not isinstance(selector.get('material_name', ''), str)
                or type(selector.get('slot_index', -1)) is not int):
            raise ValueError('Material selector has invalid role/name/index types; use the typed selector schema.')
        matches = [slot for slot in valid_slots.get(identifier, []) if _matches(slot, selector)]
        if not matches:
            raise ValueError(f'base_material_overrides[{min(index, 999)}]: ' + _ZERO_MATCH)
        for slot in matches:
            identity = (identifier, slot['object_name'], slot['slot_index'])
            target = override.get('material_id')
            if identity in plans and plans[identity] != target:
                raise ValueError(f'base_material_overrides[{min(index, 999)}]: ' + _CONFLICT)
            plans[identity] = target
    return None
