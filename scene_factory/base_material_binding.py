"""Explicit per-instance material bindings with evidence of actual assignments.

    The inventory is captured before any changes. Partial selectors preserve
    unrelated source slots, and OBJECT links isolate shared mesh instances.
"""
from __future__ import annotations

import re

from v2_materials import planar_uv


def _source_role(material):
    if material is None:
        return 'body'
    declared = material.get('asset_material_role') or material.get('material_role')
    if declared in {'body', 'metal', 'glass', 'fabric'}:
        return declared
    nodes = material.node_tree.nodes if material.use_nodes and material.node_tree else []
    shader = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
    if shader is not None:
        transmission = shader.inputs.get('Transmission Weight')
        metallic = shader.inputs.get('Metallic')
        if transmission and not transmission.is_linked and transmission.default_value > .5:
            return 'glass'
        if metallic and not metallic.is_linked and metallic.default_value > .45:
            return 'metal'
    if re.search(r'glass|glazing', material.name, re.I):
        return 'glass'
    if re.search(r'fabric|linen|upholster|textile|quilt|woven|cloth', material.name, re.I):
        return 'fabric'
    return 'body'


def _matches(slot, selector):
    role = selector.get('role', 'all') or 'all'
    name = selector.get('material_name', '')
    index = selector.get('slot_index', -1)
    return ((role == 'all' or role == slot['role'])
            and (not name or name in {slot['source_material'], slot['source_material_id'], slot['source_material_key']})
            and (index < 0 or index == slot['slot_index']))


def apply_base_material_overrides(entries, roots, overrides, materials):
    """Return inventory, actual changes and blocking errors for renderer preflight.

    The function does not throw on a zero-match selector: multi-view preflight
    persists that error so the model can repair its exact selector from evidence.
    Schema reference/type validation still happens before scene construction.
    """
    inventory, runtime = [], {}
    for entry, root in zip(entries, roots):
        root['base_instance_id'] = entry['id']
        objects = [root] + list(root.children_recursive)
        for obj in objects:
            if obj.type not in {'MESH', 'CURVE', 'SURFACE', 'FONT'}:
                continue
            # Unused mesh slots are not evidence that the visible asset changed.
            used_slots = ({face.material_index for face in obj.data.polygons}
                          if obj.type == 'MESH' else None)
            for index, slot in enumerate(obj.material_slots):
                if used_slots is not None and index not in used_slots:
                    continue
                material = slot.material
                record = {'object_id': entry['id'], 'object_name': obj.name,
                          'slot_index': index,
                          'source_material': material.name if material else '',
                          # Blender may suffix duplicate datablock names on a
                          # rebuild. Expose a stable key for model selectors.
                          'source_material_key': re.sub(r'\.\d{3,}$', '', material.name) if material else '',
                          'source_material_id': material.get('scene_material_id', '') if material else '',
                          'role': _source_role(material)}
                inventory.append(record)
                runtime[(obj.name, index)] = obj
    plans, errors, bindings = {}, [], []
    for index, override in enumerate(overrides):
        object_id, material_id = override['object_id'], override['material_id']
        selector = override.get('selector', {'role': 'all', 'material_name': '', 'slot_index': -1})
        matches = [slot for slot in inventory if slot['object_id'] == object_id and _matches(slot, selector)]
        binding = {'override_index': index, 'object_id': object_id, 'material_id': material_id,
                   'selector': selector, 'matched_slots': [], 'status': 'pending'}
        if not matches:
            binding['status'] = 'rejected'
            errors.append({'code': 'base_material_override_no_matches', 'object_id': object_id,
                           'override_index': index, 'selector': selector,
                           'action': 'Use available source slot names/roles in material_binding.available_slots. Creating a material does not bind it.'})
        if material_id not in materials:
            binding['status'] = 'rejected'
            errors.append({'code': 'base_material_override_missing_material', 'object_id': object_id,
                           'override_index': index, 'material_id': material_id})
        for slot in matches:
            key = (slot['object_name'], slot['slot_index'])
            if key in plans and plans[key] != material_id:
                errors.append({'code': 'base_material_override_conflict', 'object_id': object_id,
                               'object_name': key[0], 'slot_index': key[1], 'override_index': index,
                               'action': 'Remove overlapping bindings that assign different materials to one source slot.'})
            plans[key] = material_id
            binding['matched_slots'].append({**slot, 'applied_material_id': None})
        bindings.append(binding)
    # Make the whole transaction fail closed: evidence cannot report successful
    # assignments for a half-applied set of conflicting or unresolved bindings.
    if not errors:
        for key, material_id in plans.items():
            obj = runtime[key]
            slot = obj.material_slots[key[1]]
            slot.link = 'OBJECT'
            slot.material = materials[material_id]
            if obj.type == 'MESH':
                planar_uv(obj)
            if slot.material != materials[material_id]:
                raise RuntimeError('Blender material binding failed: ' + repr(key))
        for binding in bindings:
            binding['status'] = 'applied'
            for slot in binding['matched_slots']:
                slot['applied_material_id'] = binding['material_id']
                slot['applied_material'] = materials[binding['material_id']].name
    else:
        for binding in bindings:
            if binding['status'] == 'pending':
                binding['status'] = 'blocked'
    return {'schema_version': 1, 'status': 'rejected' if errors else 'passed',
            'requested_count': len(overrides), 'applied_slot_count': len(plans) if not errors else 0,
            'bindings': bindings, 'available_slots': inventory, 'errors': errors,
            'role_source': 'explicit source material role when available; otherwise shader transmission/metallic and fabric material-name heuristics'}
