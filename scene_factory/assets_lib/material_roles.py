"""Stable source-slot roles without mutating shared appearance materials.

Roles describe existing asset components, never the desired override material.
The role participates in the copy cache so identical shader parameters can
serve body, fabric or metal components without semantic cross-contamination.
"""
from __future__ import annotations

import bpy


SOURCE_ROLES = frozenset({'body', 'metal', 'glass', 'fabric'})
_ROLE_MATERIALS = {}


def source_role(material, role):
    """Return a role-tagged copy, leaving the caller's shared source intact.

    Blender can remove/reload datablocks while this module remains imported.
    Check both source and copy ownership before reusing a cached RNA handle;
    pointer reuse must not make a new source inherit an old cached copy.
    """
    if role not in SOURCE_ROLES:
        raise ValueError('Unknown built-in source material role')
    if material.get('asset_material_role') == role:
        return material
    key = (material.as_pointer(), role)
    cached = _ROLE_MATERIALS.get(key)
    if cached is not None:
        source, tagged = cached
        try:
            if (source is material and bpy.data.materials.get(source.name) is source
                    and bpy.data.materials.get(tagged.name) is tagged
                    and tagged.get('asset_material_role') == role):
                return tagged
        except ReferenceError:
            pass
        _ROLE_MATERIALS.pop(key, None)
    tagged = material.copy()
    tagged['asset_material_role'] = role
    _ROLE_MATERIALS[key] = (material, tagged)
    return tagged
