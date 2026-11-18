"""Bind a text-authored layout to models from the asset library.

A text scene and a photo scene describe the same physical thing, but the two
paths hand their layout to different code, and three details differ:

1. **Container shape.** ``library_selection.validate_layout`` requires
   ``library_layout_v1``: ``views[]`` with id/source_image/width/height and an
   ``architecture`` block. A text layout carries ``cameras[]`` and ``room``.
   :func:`to_library_layout` translates one into the other.

2. **Category vocabulary.** ``select_and_place`` looks categories up with a
   plain ``index.get(req['category'])`` and never reads
   ``options['category_map']`` — only the photo path's ``generate_bindings``
   honours that map. So a written brief's words ("bookshelf", "armchair",
   "table") have to be rewritten to the catalog's own names before the call.

3. **Placement.** A floor-supported object needs a floor-placed asset: a desk
   lamp cannot stand in for a floor lamp. Categories with no floor alternative
   are left out instead of being forced into a mismatch.

What this module deliberately does **not** do is silently invent geometry. An
object is either bound to a real library model or reported as unbound, and the
caller keeps the renderer's procedural shape for it — the same contract the
photo path documents as "a generic box is never silently rendered".

The selection is all-or-nothing inside ``select_and_place``, so a single
unsatisfiable child (a tray on a table with no measured support region) would
sink the whole room. :func:`bind_layout` therefore drops the failing object and
retries a bounded number of times, mirroring the photo pipeline's rule of
repairing only the failing part.
"""
from __future__ import annotations

import copy
import math

# Written-brief vocabulary -> catalog categories, most specific first. The first
# name present in the catalog wins.
CATEGORY_REWRITE = {
    'table': ['dining_table', 'coffee_table', 'side_table', 'console_table',
              'work_table', 'covered_display_table'],
    'bookshelf': ['shelf'],
    'armchair': ['reclining_armchair', 'folding_lounge_chair', 'chair'],
    'lamp': ['desk_lamp'],
    # A planter box is an empty vessel: binding a 1.4 m fiddle-leaf fig to a
    # wooden trough leaves a bare crate in the corner, which reads worse than
    # the procedural plant (pot, soil, trunk and 36 leaves). Only assets that
    # actually carry foliage may stand in for a plant.
    'plant': ['potted_plant', 'potted_foliage_plant', 'houseplant', 'potted_tree'],
    'decor': ['vase', 'tray'],
    'box': ['cardboard_box', 'plastic_crate', 'bucket'],
    # No catalog equivalent: these stay procedural rather than becoming
    # something they are not.
    'curtain': [],
    'rug': [],
}

# The selector may scale an asset up, so a piece placed flush against a wall can
# end up outside it. Reserve that headroom before selection.
MAX_SCALE = 1.25
# One withdrawal per attempt, and a dressed room has many small props whose
# footprint may miss a fragmented support surface: the catalog's tabletops are
# detected as several small polygons, so a mug 20 cm off centre fails to seat.
# Four attempts was enough for a six-object room and gave up on a dressed one,
# which dropped the whole scene to primitives over a handful of props.
MAX_REPAIRS = 24

# A written brief gives target sizes in round, human numbers, while the catalog
# holds whatever proportions each model was scanned at: a 1.6 m table candidate
# beside a 1.2 m request is the same kind of furniture, not a mismatch. The photo
# path can afford the default 0.15 tolerance because a reconstruction should not
# stretch furniture to fit a bad estimate; a text scene is a design, so it may.
# Without this the strict defaults reject most furniture and the room falls back
# to primitives.
TEXT_SELECTION_DEFAULTS = {
    'dimension_tolerance': 0.45,
    'min_uniform_scale': 0.55,
    'max_uniform_scale': 1.8,
}


def _catalog_categories(catalog):
    return {asset['category'] for asset in catalog.assets.values()}


def _floor_categories(catalog):
    return {asset['category'] for asset in catalog.assets.values()
            if asset.get('placement') == 'floor'}


def _target_category(category, on_floor, available):
    """Map a written category onto a catalog name usable at this placement."""
    if category in available and (not on_floor or category in available):
        return category
    for candidate in CATEGORY_REWRITE.get(category, ()):
        if candidate in available:
            return candidate
    return None


def _clamp_to_room(xy, dimensions, rotation_deg, room):
    """Keep a footprint inside the room contract: X in -w/2..w/2, Y in 0..d."""
    angle = math.radians(rotation_deg)
    cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
    half_x = (dimensions[0] * cos + dimensions[1] * sin) / 2 * MAX_SCALE
    half_y = (dimensions[0] * sin + dimensions[1] * cos) / 2 * MAX_SCALE
    x_lo, x_hi = -room['width'] / 2 + half_x, room['width'] / 2 - half_x
    y_lo, y_hi = half_y, room['depth'] - half_y
    x = min(max(xy[0], x_lo), x_hi) if x_lo <= x_hi else 0.0
    y = min(max(xy[1], y_lo), y_hi) if y_lo <= y_hi else room['depth'] / 2
    return [x, y]


def to_library_layout(layout, catalog, *, width=1600, height=900):
    """Translate ``text_scene_v1`` into ``library_layout_v1``.

    Returns ``(library_layout, dropped)`` where ``dropped`` lists
    ``(object_id, written_category)`` for objects the catalog cannot represent.

    The renderer's own category rewrite is applied first, so both sides of the
    pipeline judge the same object the same way. Without it the renderer builds a
    "book set" as a row of separate books while the binder still sees a catalog
    category, binds the one solid block it has, fails to seat it on a shelf whose
    asset exposes only its top face, and takes the whole room down to primitives
    over one object.
    """
    from text_scene import compile_native_categories
    layout, _ = compile_native_categories(layout)
    available = _catalog_categories(catalog)
    floor_available = _floor_categories(catalog)
    room = layout['room']

    views = []
    for index, camera in enumerate(layout['cameras']):
        views.append({'id': f'view_{index:02d}',
                      'source_image': f'input:view_{index:02d}',
                      'width': width, 'height': height,
                      'camera': copy.deepcopy(camera)})

    objects, dropped = [], []
    for entry in layout['objects']:
        on_floor = entry['support_id'] == 'floor'
        pool = floor_available if on_floor else available
        category = _target_category(entry['category'], on_floor, pool)
        if category is None:
            dropped.append((entry['id'], entry['category']))
            continue
        xy = list(entry['xy'])
        if on_floor:
            xy = _clamp_to_room(xy, entry['target_dimensions_m'],
                                entry['rotation_z_deg'], room)
        objects.append({'id': entry['id'],
                        'category': category,
                        'xy': xy,
                        'target_dimensions_m': list(entry['target_dimensions_m']),
                        'rotation_z_deg': entry['rotation_z_deg'],
                        'support_id': entry['support_id'],
                        'clearance_m': entry.get('clearance_m', 0)})

    translated = {
        'schema_version': 'library_layout_v1',
        'scene_id': layout['scene_id'],
        'scene_type': (layout['scene_type'] if layout['scene_type'] in
                       {'office', 'living_room', 'bedroom', 'classroom', 'kitchen', 'shop', 'other'}
                       else 'other'),
        'description': layout['description'],
        'seed': layout['seed'],
        'room': {key: room[key] for key in
                 ('width', 'depth', 'height', 'wall_color', 'floor_color', 'ceiling_color')},
        'views': views,
        'architecture': {'omit_surfaces': [], 'front_wall': False},
        'lighting': copy.deepcopy(layout['lighting']),
        'lights': copy.deepcopy(layout['lights']),
        'exposure': layout['exposure'],
        'world_color': list(layout['world_color']),
        'objects': objects,
        'assumptions': list(layout.get('assumptions', ())),
    }
    # An object the catalog cannot represent takes whatever rests on it with it:
    # a sideboard that stays procedural leaves its lamp and its bowl pointing at
    # a support the library layout no longer contains, and placement stops with
    # "unknown support_id" — which cost the whole room its library models.
    present = {item['id'] for item in translated['objects']}
    spreading = True
    while spreading:
        spreading = False
        for item in list(translated['objects']):
            support = item['support_id']
            if support != 'floor' and support not in present:
                translated['objects'].remove(item)
                present.discard(item['id'])
                dropped.append((item['id'], item['category']))
                spreading = True
    return translated, dropped


def _trim(bundle, keep):
    trimmed = copy.deepcopy(bundle)
    trimmed['shared_scene']['base_scene']['objects'] = [
        entry for entry in trimmed['shared_scene']['base_scene']['objects']
        if entry['id'] in keep]
    return trimmed


def _merge(full, bound):
    """Combine the bound subset back into the complete bundle.

    Library models and procedural geometry are complementary, not exclusive: an
    object the catalog can satisfy is replaced by the real mesh, and every other
    object keeps the renderer's own shape. Dropping the unbound ones instead
    would delete the curtains, the rug and the floor lamp from the picture.
    """
    bound_by_id = {entry['id']: entry
                   for entry in bound['shared_scene']['base_scene']['objects']}
    merged = copy.deepcopy(full)
    merged['shared_scene']['base_scene']['objects'] = [
        bound_by_id.get(entry['id'], entry)
        for entry in full['shared_scene']['base_scene']['objects']]
    return merged


def bind_layout(bundle, layout, catalog, options):
    """Bind library models onto ``bundle``; never raise for missing assets.

    Returns ``(bundle, bindings, report, dropped, repaired)``. ``bindings`` and
    ``report`` are ``None`` when no single attempt could satisfy the scene.
    """
    from library_selection import LibrarySelectionError, select_and_place

    translated, dropped = to_library_layout(layout, catalog)
    keep = {entry['id'] for entry in bundle['shared_scene']['base_scene']['objects']}
    translated['objects'] = [o for o in translated['objects'] if o['id'] in keep]
    requested = {o['id'] for o in translated['objects']}
    working = _trim(bundle, requested)

    if not requested:
        return bundle, None, None, dropped, []

    repaired = []
    last_report = None
    for _attempt in range(MAX_REPAIRS):
        try:
            effective, bindings, report = select_and_place(
                working, translated, catalog, options)
            return _merge(bundle, effective), bindings, report, dropped, repaired
        except LibrarySelectionError as error:
            last_report = error.report
            removable = _withdrawable(last_report)
            if not removable:
                break
            # Withdrawing a table has to take what stands on it: the plates and
            # the vase would otherwise keep pointing at a support that is no
            # longer in the layout, and placement stops at "unknown support_id"
            # with the whole room falling back to primitives. The children are
            # not failures of their own, so they are recorded as carried.
            going = dict(removable)
            children = {o['id']: o.get('support_id') for o in translated['objects']}
            spreading = True
            while spreading:
                spreading = False
                for identifier, support in children.items():
                    if support in going and identifier not in going:
                        going[identifier] = f'support withdrawn ({support})'
                        spreading = True
            for name, reason in removable.items():
                repaired.append({'object': name, 'reason': reason})
            for name, reason in going.items():
                if name not in removable:
                    repaired.append({'object': name, 'reason': reason})
            keep_now = {o['id'] for o in translated['objects']} - set(going)
            translated['objects'] = [o for o in translated['objects']
                                     if o['id'] in keep_now]
            working = _trim(working, keep_now)
            if not translated['objects']:
                break
    return bundle, None, last_report, dropped, repaired


def _withdrawable(report):
    """Objects to take out of the request, with the reason, for one retry round.

    A text scene has no reviewed source to be faithful to, so an object the
    library cannot place is not a failure of the room — the renderer builds it
    from its own geometry instead. Only the offending objects are withdrawn;
    everything that can bind still binds. Category misses go first because they
    are certain, whereas a collision may clear once the other piece is placed
    differently.
    """
    missing, single, pairs = {}, {}, []
    for entry in report.get('errors', ()):
        code = entry.get('code')
        ids = entry.get('object_ids') or ([entry['object_id']] if entry.get('object_id') else [])
        if code == 'missing_compatible_asset':
            for identifier in ids:
                missing[identifier] = 'no_compatible_asset'
        elif len(ids) == 1:
            single[ids[0]] = code or 'placement_failed'
        elif len(ids) >= 2:
            pairs.append(ids)
    if missing:
        return missing
    if single:
        return single
    # Collisions: withdraw the later object of each pair so the earlier one keeps
    # the clash-free spot, then let the next round re-place what remains.
    withdrawn = {}
    for ids in pairs:
        for identifier in ids[1:]:
            withdrawn.setdefault(identifier, 'collision_with_' + ids[0])
    return withdrawn
