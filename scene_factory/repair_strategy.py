"""Deterministic repair focus and partial scene context, never an acceptance gate."""
from __future__ import annotations

import copy


STRATEGY_VERSION = 4
_SEVERITY = {'blocking': 0, 'major': 1, 'minor': 2}
_DIMENSION = {'framing': 0, 'geometry': 1, 'layout': 2, 'materials': 3, 'lighting': 4, 'realism': 5}


def _referenced_materials(materials, identifiers):
    """Resolve named materials and materials using explicitly named textures."""
    return {material['id'] for material in materials
            if material['id'] in identifiers
            or (material.get('texture_id') and material['texture_id'] in identifiers)}


def repair_context(bundle, report):
    """Focus one coherent issue family while retaining the whole scene inventory.

    Unknown/unlocalized issues fall back to full geometry. No warnings, scores,
    source views, materials, lights or object identities are removed. Partial
    geometry is explicitly marked and may not be rewritten by a patch.
    """
    issues = report.get('issues', [])
    ranked = sorted(range(len(issues)), key=lambda i: (
        _SEVERITY.get(issues[i].get('severity'), 3),
        _DIMENSION.get(issues[i].get('dimension'), 6), i))
    focused = ranked[:1]
    if focused:
        first = issues[focused[0]]
        identifiers = set(first.get('object_ids', []))
        for index in ranked[1:]:
            issue = issues[index]
            if (issue.get('dimension') == first.get('dimension')
                    and identifiers & set(issue.get('object_ids', []))):
                focused.append(index)
                break
    identifiers = {value for index in focused for value in issues[index].get('object_ids', [])}
    groups = bundle['shared_scene']['custom_objects']
    known_groups = {group['id'] for group in groups}
    scene = bundle['shared_scene']
    dependencies = {identifier: set() for identifier in known_groups}
    for item in scene['base_scene']['objects']:
        root = item['id']
        dependencies.setdefault(root, set())
        repeat = item.get('repeat', [1, 1])
        if repeat != [1, 1]:
            # A base edit replaces the complete repeat record. Even an issue
            # naming one instance can require a family-wide placement change.
            # Use the actual repeat mapping, not string-prefix matching.
            for i in range(repeat[0]):
                for j in range(repeat[1]):
                    instance = root + f'__{i}_{j}'
                    dependencies[root].add(instance)
                    dependencies.setdefault(instance, set()).add(root)
    known_ids = set(dependencies)
    known_ids.update(item['id'] for key in ('materials', 'lights', 'textures') for item in scene[key])
    related = identifiers & dependencies.keys()
    material_identifiers = _referenced_materials(scene['materials'], identifiers)
    # Include groups using a referenced material and groups physically supported
    # by referenced objects. Walk supports in both directions, including through
    # base assets, so a local fix cannot lose contact with related geometry.
    for group in groups:
        materials = {group.get('material_id')} | {part.get('material_id') for part in group['parts']}
        if materials & material_identifiers:
            related.add(group['id'])
        support = group.get('support_id')
        if support in dependencies:
            dependencies[group['id']].add(support)
            dependencies[support].add(group['id'])
    for binding in scene.get('base_material_overrides', []):
        if binding['material_id'] in material_identifiers and binding['object_id'] in dependencies:
            related.add(binding['object_id'])
    pending = list(related)
    while pending:
        for identifier in dependencies[pending.pop()] - related:
            related.add(identifier)
            pending.append(identifier)
    detailed = related & known_groups
    # Global issues and unresolved references anywhere in the assessment need
    # all geometry. They may not be the highest-ranked family, but a partial
    # context would prevent the model from seeing evidence needed to address
    # them (or from discovering which referenced ID is actually unknown).
    all_identifiers = {value for issue in issues for value in issue.get('object_ids', [])}
    all_material_identifiers = _referenced_materials(scene['materials'], all_identifiers)
    global_or_unresolved = (
        any(issue.get('dimension') in ('framing', 'lighting', 'realism') for issue in issues)
        or bool(all_identifiers - known_ids)
        # Architecture reflectance and exterior appearance affect the whole
        # scene, even when their texture issue was not the selected family.
        or bool(all_material_identifiers & set(scene['surface_overrides'].values()))
        or any(context['texture_id'] in all_identifiers
               for context in scene.get('exterior_contexts', [])))
    if not detailed or global_or_unresolved:
        detailed = known_groups
    context = copy.deepcopy(bundle)
    for view in context['views']:
        view['source_image'] = 'input:' + view['id']
    for group in context['shared_scene']['custom_objects']:
        if group['id'] not in detailed:
            parts = group.pop('parts')
            group['part_count'] = len(parts)
            group['part_types'] = sorted({part['type'] for part in parts})
    focus = {'strategy_version': STRATEGY_VERSION,
             'issue_indices': focused,
             'detailed_group_ids': sorted(detailed),
             'inventory_only_group_ids': sorted(known_groups - detailed),
             'instructions': (
                 'Address the selected issue indices as one coherent physical correction. '
                 'Other issues remain unresolved, not waived. Inspect images and measured evidence; '
                 'the issue ranking is heuristic, not proof of cause. Preserve unrelated content. '
                 'All shared materials, lights, base assets and group transforms are supplied. '
                 'Only detailed groups include parts. Never invent omitted part indices or rewrite '
                 'inventory-only geometry. Return a no-op with a concrete explanation if the '
                 'evidenced fix cannot be expressed from this context. A new render and independent '
                 'review are always required.')}
    return context, focus


def validate_edit_scope(edits, focus):
    """Reject partial-geometry edits that the model could not inspect."""
    hidden = set(focus['inventory_only_group_ids'])
    touched = {item['group_id'] for field in ('part_updates', 'remove_parts', 'add_parts')
               for item in edits[field]}
    touched.update(item['id'] for item in edits['replace_groups'])
    touched.update(edits['remove_groups'])
    if hidden & touched:
        raise ValueError('Repair attempted to rewrite geometry omitted from the focused context')
