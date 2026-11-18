"""Plausibility checks for text-authored scenes.

Photo reconstruction is judged against a reference image: the question is
"does this match the photograph". A scene written from text has no reference,
so the only quality signal available locally is whether the result is
*physically and socially plausible*: furniture standing at believable heights,
seats facing something, a walkable path, nothing buried in a doorway, a room
neither empty nor crammed.

Everything here is deterministic and model-free. It returns findings rather
than raising, so a caller can decide whether a defect is fatal (geometry that
cannot render) or merely implausible (a chair facing a blank wall). Repair is
driven by the highest-severity findings only; see ``repair_hint``.
"""
from __future__ import annotations

import math

# Category -> plausible (min, max) height in metres for the *whole object*.
# Ranges are deliberately wide: they catch unit mistakes (a 3 m chair) and
# swapped axes, not taste. Values follow common residential/commercial
# furniture rather than any single manufacturer.
HEIGHT_RANGES = {
    'bed':            (0.35, 1.30),
    'bunk_bed':       (1.40, 2.20),
    'sofa':           (0.60, 1.10),
    'armchair':       (0.60, 1.15),
    'chair':          (0.70, 1.15),
    'office_chair':   (0.85, 1.35),
    'stool':          (0.40, 0.85),
    'bench':          (0.40, 0.95),
    'table':          (0.35, 0.85),
    'dining_table':   (0.55, 0.85),
    'coffee_table':   (0.25, 0.55),
    'side_table':     (0.35, 0.75),
    'console_table':  (0.35, 0.90),
    'desk':           (0.60, 0.82),
    'counter':        (0.85, 1.15),
    'checkout_counter': (0.85, 1.15),
    # Low storage is a large family: a media console and a TV stand sit well
    # below the height of a wardrobe, so the lower bound has to admit them.
    'cabinet':        (0.30, 2.30),
    'media_console':  (0.30, 0.80),
    'tv_stand':       (0.25, 0.80),
    'sideboard':      (0.45, 1.10),
    'wardrobe':       (1.60, 2.45),
    'shelf':          (0.60, 2.45),
    'bookshelf':      (0.60, 2.45),
    'shelf_unit':     (0.60, 2.45),
    'drawer_unit':    (0.55, 1.40),
    'tv':             (0.30, 1.20),
    'monitor':        (0.15, 0.70),
    'computer':       (0.15, 0.70),
    'fridge':         (1.30, 2.20),
    'freezer':        (0.70, 2.10),
    'microwave':      (0.20, 0.50),
    'plant':          (0.15, 2.20),
    'potted_plant':   (0.15, 2.20),
    'lamp':           (0.30, 1.85),
    'floor_lamp':     (1.20, 1.85),
    'desk_lamp':      (0.30, 0.70),
    'rug':            (0.005, 0.06),
    'box':            (0.10, 1.20),
    'cardboard_box':  (0.10, 1.20),
    'door':           (1.80, 2.40),
    'window':         (0.40, 2.40),
    'column':         (2.00, 8.00),
    'beam':           (0.10, 0.80),
}

# Categories that are conventionally placed against a wall.
WALL_HUGGING = {
    'wardrobe', 'cabinet', 'bookshelf', 'bookshelf_unit', 'shelf_unit',
    'drawer_unit', 'bed', 'bunk_bed', 'sofa', 'counter', 'checkout_counter',
    'desk', 'tv', 'fridge',
}

# Categories that must have something to look at (seating semantics).
SEATING = {'sofa', 'armchair', 'chair', 'office_chair', 'bench', 'stool'}

# Categories a seating object plausibly faces.
FOCAL = {'tv', 'coffee_table', 'dining_table', 'table', 'desk', 'monitor',
         'computer', 'counter', 'checkout_counter', 'shelf', 'shelves',
         'bookshelf', 'fireplace', 'window'}

# Objects whose footprint a person must be able to walk past.
CIRCULATION_BLOCKERS = {
    'wardrobe', 'cabinet', 'bookshelf', 'bookshelf_unit', 'shelf_unit',
    'sofa', 'bed', 'bunk_bed', 'counter', 'checkout_counter', 'fridge',
    'dining_table', 'desk', 'table',
}

MIN_WALK_CLEARANCE_M = 0.55   # one narrow person passing
MIN_DOOR_CLEARANCE_M = 0.75   # swing/approach room in front of a door

# Flat floor coverings are laid *under* other furniture by design, so they are
# exempt from the interpenetration rule. Anything with real height still counts.
FLOOR_COVERINGS = {'rug', 'carpet', 'mat', 'doormat', 'floor_mat', 'floor_carpet'}
FLOOR_COVERING_MAX_HEIGHT_M = 0.06

SEVERITY_ORDER = {'blocking': 0, 'major': 1, 'minor': 2, 'note': 3}


def _finding(severity, code, message, object_ids=(), fixable=True):
    return {'severity': severity, 'code': code, 'message': message,
            'object_ids': list(object_ids), 'auto_fixable': fixable}


def _yaw_radians(degrees):
    return math.radians(float(degrees))


def _dimensions(entry):
    """Accept both the model-facing and the expanded layout spelling.

    ``library_layout_v1`` objects carry ``target_dimensions_m``; an expanded
    canonical object carries ``dimensions``. Supporting both keeps the checks
    usable before and after asset selection.
    """
    for key in ('target_dimensions_m', 'dimensions'):
        value = entry.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return [float(v) for v in value]
    raise ValueError(f'Object {entry.get("id")!r} has no 3-element dimensions')


def footprint(entry):
    """Return the four world XY corners of an object's rotated footprint."""
    dims = _dimensions(entry)
    width, depth = dims[0], dims[1]
    cx, cy = float(entry['xy'][0]), float(entry['xy'][1])
    theta = _yaw_radians(entry.get('rotation_z_deg', 0))
    cos, sin = math.cos(theta), math.sin(theta)
    corners = []
    for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = dx * width / 2, dy * depth / 2
        corners.append((cx + lx * cos - ly * sin, cy + lx * sin + ly * cos))
    return corners


def aabb(entry):
    """Axis-aligned XY bounds of the rotated footprint."""
    corners = footprint(entry)
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_depth(a, b):
    """How deeply two axis-aligned rectangles overlap; 0 when they do not."""
    return (min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1]))


def _rect_overlaps(a, b, tolerance=0.0):
    dx, dy = _overlap_depth(a, b)
    return dx > tolerance and dy > tolerance


def _distance_between(a, b):
    """Shortest gap between two axis-aligned rectangles; negative when overlapping."""
    dx = max(b[0] - a[2], a[0] - b[2])
    dy = max(b[1] - a[3], a[1] - b[3])
    if dx < 0 and dy < 0:
        return max(dx, dy)          # interpenetrating: report the deeper penetration
    return math.hypot(max(dx, 0.0), max(dy, 0.0))


def _facing_target_yaw(source_xy, target_xy):
    """Yaw in degrees that points the object's local -Y at the target."""
    dx = target_xy[0] - source_xy[0]
    dy = target_xy[1] - source_xy[1]
    return math.degrees(math.atan2(dx, -dy))


def _angle_gap(a, b):
    """Smallest absolute difference between two yaw angles, in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _support_resolves_bottom(entry, objects, room_height):
    """Bottom Z implied by the support chain, without trusting the model."""
    parent = entry.get('support_id', 'floor')
    if parent == 'floor':
        return 0.0
    chain, seen = [], set()
    while parent and parent != 'floor':
        if parent in seen:
            return None
        seen.add(parent)
        node = objects.get(parent)
        if node is None:
            return None
        chain.append(node)
        parent = node.get('support_id', 'floor')
    bottom = 0.0
    for node in reversed(chain):
        bottom += _dimensions(node)[2]
    return bottom


def validate_scene_logic(layout, *, strict=False):
    """Return plausibility findings for a text-authored layout.

    ``layout`` is the model-facing form: room, objects with ``xy``,
    ``target_dimensions_m``, ``rotation_z_deg``, ``support_id``. Findings never
    raise; callers decide what is fatal. ``strict`` promotes the
    geometry-derivable findings to blocking so a batch can reject weak output.
    """
    findings = []
    room = layout['room']
    width, depth, height = float(room['width']), float(room['depth']), float(room['height'])
    objects = {entry['id']: entry for entry in layout['objects']}

    dims = _dimensions

    # ---- 1. Room proportions and object density -------------------------
    area = width * depth
    if height < 2.0 or height > 5.0:
        findings.append(_finding('major', 'room.height_implausible',
                                 f'Room height {height:.2f} m is outside a believable interior range.'))
    if area > 0:
        density = len(objects) / area
        if density > 1.2:
            findings.append(_finding('major', 'room.overcrowded',
                                     f'{len(objects)} objects in {area:.1f} m2 is implausibly dense.'))
        elif len(objects) >= 3 and density < 0.04 and area > 12:
            findings.append(_finding('minor', 'room.underfurnished',
                                     f'{len(objects)} objects in {area:.1f} m2 leaves the room bare.'))

    # ---- 2. Category heights --------------------------------------------
    for entry in objects.values():
        category = str(entry.get('category', '')).lower()
        obj_height = dims(entry)[2]
        limits = HEIGHT_RANGES.get(category)
        if limits and not (limits[0] <= obj_height <= limits[1]):
            findings.append(_finding(
                'major' if obj_height > limits[1] * 1.6 or obj_height < limits[0] * 0.4 else 'minor',
                'object.height_implausible',
                f'{entry["id"]} ({category}) is {obj_height:.2f} m tall; '
                f'believable range is {limits[0]:.2f}-{limits[1]:.2f} m.', [entry['id']]))

    # ---- 3. Objects fit inside the room ---------------------------------
    for entry in objects.values():
        left, bottom, right, top = aabb(entry)
        out = max(-left - width / 2, -bottom, right - width / 2, top - depth)
        if out > 0.02:
            findings.append(_finding('blocking', 'object.outside_room',
                                     f'{entry["id"]} extends {out:.2f} m outside the room.',
                                     [entry['id']]))

    # ---- 4. Interpenetration --------------------------------------------
    # Ignore pairs linked by a support chain: a lamp standing on a desk shares
    # footprint with it by design.
    def related(a, b):
        for x, y in ((a, b), (b, a)):
            parent = x.get('support_id')
            guard = 0
            while parent and parent != 'floor' and guard < 16:
                if parent == y.get('id'):
                    return True
                parent = objects.get(parent, {}).get('support_id')
                guard += 1
        return False

    entries = list(objects.values())

    def is_floor_covering(entry):
        category = str(entry.get('category', '')).lower()
        if category in FLOOR_COVERINGS:
            return True
        # A low object spread over a wide area is a rug whatever it is called.
        dims = _dimensions(entry)
        return (dims[2] <= FLOOR_COVERING_MAX_HEIGHT_M
                and min(dims[0], dims[1]) >= 0.8)

    # A shelf unit is many boards, and a written brief names the shelf, not the
    # board: "five rows of books, a sculpture and a basket on the shelf" is a
    # complete description that leaves every one of them at the same xy. The
    # renderer is what distributes them over the boards, so two things sharing a
    # shelf cannot be shown to collide from the plan alone. Anything resting on a
    # single surface (a tabletop) is still checked: there, overlap is a real error.
    BOARDED = {'shelf', 'bookshelf', 'bookcase', 'shelf_unit', 'bookshelf_unit',
               'shelving', 'shelving_unit', 'display_shelf'}

    def boarded_siblings(a, b):
        support = a.get('support_id')
        if not support or support == 'floor' or support != b.get('support_id'):
            return False
        parent = objects.get(support)
        if parent is None:
            return False
        if str(parent.get('category', '')).lower() not in BOARDED:
            return False
        # Only when the parent really is tall enough to have separate boards.
        return _dimensions(parent)[2] >= 0.8

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i], entries[j]
            if related(a, b):
                continue
            if boarded_siblings(a, b):
                continue
            # A rug under a sofa is correct design, not a collision.
            if is_floor_covering(a) or is_floor_covering(b):
                thinner = a if _dimensions(a)[2] <= _dimensions(b)[2] else b
                other = b if thinner is a else a
                if _dimensions(thinner)[2] <= FLOOR_COVERING_MAX_HEIGHT_M:
                    continue
            dx, dy = _overlap_depth(aabb(a), aabb(b))
            if dx > 0.02 and dy > 0.02:
                penetration = min(dx, dy)
                findings.append(_finding(
                    'blocking' if penetration > 0.10 else 'major',
                    'object.interpenetrates',
                    f'{a["id"]} and {b["id"]} overlap by {penetration:.2f} m in plan.',
                    [a['id'], b['id']]))

    # ---- 5. Circulation between large neighbours ------------------------
    blockers = [e for e in entries if str(e.get('category', '')).lower() in CIRCULATION_BLOCKERS]
    for i in range(len(blockers)):
        for j in range(i + 1, len(blockers)):
            a, b = blockers[i], blockers[j]
            if related(a, b):
                continue
            gap = _distance_between(aabb(a), aabb(b))
            if 0.0 <= gap < MIN_WALK_CLEARANCE_M:
                # Only a problem when both are solid enough to form a barrier.
                if min(dims(a)[2], dims(b)[2]) > 0.7:
                    findings.append(_finding(
                        'minor', 'circulation.narrow_gap',
                        f'Only {gap:.2f} m between {a["id"]} and {b["id"]}; '
                        f'{MIN_WALK_CLEARANCE_M:.2f} m is a comfortable passing width.',
                        [a['id'], b['id']]))

    # ---- 6. Wall-hugging furniture actually touches a wall --------------
    for entry in entries:
        category = str(entry.get('category', '')).lower()
        if category not in WALL_HUGGING:
            continue
        left, bottom, right, top = aabb(entry)
        gaps = [left + width / 2, bottom, width / 2 - right, depth - top]
        nearest = min(gaps)
        if nearest > 0.45:
            findings.append(_finding(
                'minor', 'placement.not_near_wall',
                f'{entry["id"]} ({category}) stands {nearest:.2f} m from the nearest wall; '
                'this category is normally placed against one.', [entry['id']]))

    # ---- 7. Seating faces something -------------------------------------
    for entry in entries:
        category = str(entry.get('category', '')).lower()
        if category not in SEATING:
            continue
        chair_xy = entry['xy']
        chair_yaw = float(entry.get('rotation_z_deg', 0))
        best = None
        for other in entries:
            if other['id'] == entry['id']:
                continue
            other_category = str(other.get('category', '')).lower()
            if other_category not in FOCAL:
                continue
            distance = _distance_between(aabb(entry), aabb(other))
            if distance > 3.5:
                continue
            gap = _angle_gap(chair_yaw, _facing_target_yaw(chair_xy, other['xy']))
            if gap <= 50.0 and (best is None or gap < best[0]):
                best = (gap, other['id'])
        if best is None:
            findings.append(_finding(
                'minor', 'seating.faces_nothing',
                f'{entry["id"]} ({category}) at yaw {chair_yaw:.0f} faces no plausible focal object '
                'within 3.5 m.', [entry['id']]))

    # ---- 8. Support graph sanity ----------------------------------------
    for entry in entries:
        parent_id = entry.get('support_id', 'floor')
        if parent_id == 'floor':
            continue
        parent = objects.get(parent_id)
        if parent is None:
            findings.append(_finding('blocking', 'support.missing_parent',
                                     f'{entry["id"]} rests on unknown object {parent_id}.', [entry['id']]))
            continue
        if not parent.get('supports_objects', True):
            findings.append(_finding('major', 'support.unsupported_surface',
                                     f'{parent_id} is not a supporting surface, but {entry["id"]} rests on it.',
                                     [entry['id'], parent_id]))
        child_w, child_d = dims(entry)[0], dims(entry)[1]
        parent_w, parent_d = dims(parent)[0], dims(parent)[1]
        if child_w > parent_w + 0.05 or child_d > parent_d + 0.05:
            findings.append(_finding(
                'major', 'support.child_larger_than_parent',
                f'{entry["id"]} ({child_w:.2f}x{child_d:.2f} m) is wider than its support '
                f'{parent_id} ({parent_w:.2f}x{parent_d:.2f} m).', [entry['id'], parent_id]))
        bottom_z = _support_resolves_bottom(entry, objects, height)
        if bottom_z is not None and height > 0 and bottom_z + dims(entry)[2] > height * 0.95:
            findings.append(_finding('major', 'support.exceeds_ceiling',
                                     f'{entry["id"]} stacks to {bottom_z + dims(entry)[2]:.2f} m '
                                     f'under a {height:.2f} m ceiling.', [entry['id']]))

    # ---- 9. Doors and windows -------------------------------------------
    for entry in entries:
        category = str(entry.get('category', '')).lower()
        if category not in ('door', 'doorway'):
            continue
        door_box = aabb(entry)
        for other in entries:
            if other['id'] == entry['id'] or str(other.get('category', '')).lower() in ('door', 'doorway'):
                continue
            if related(entry, other):
                continue
            gap = _distance_between(door_box, aabb(other))
            if 0.0 <= gap < MIN_DOOR_CLEARANCE_M:
                findings.append(_finding(
                    'major', 'circulation.blocks_doorway',
                    f'{other["id"]} sits {gap:.2f} m from the doorway {entry["id"]}, '
                    'blocking the approach.', [entry['id'], other['id']]))

    for entry in entries:
        category = str(entry.get('category', '')).lower()
        if category not in ('window', 'door'):
            continue
        if dims(entry)[2] > height:
            findings.append(_finding('major', 'opening.exceeds_wall_height',
                                     f'{entry["id"]} is {dims(entry)[2]:.2f} m tall '
                                     f'in a {height:.2f} m room.', [entry['id']]))

    # ---- 10. Cameras are usable viewpoints -------------------------------
    # A camera inside a wardrobe renders the inside of a box, and one thinner
    # than arm's length in front of a wall renders nothing but that wall. Both
    # are cheap to detect in plan and expensive to discover after a render.
    for index, camera in enumerate(layout.get('cameras', [])):
        position = camera['position']
        label = f'camera {index}'
        if not (1.0 <= position[2] <= 1.9) and position[2] <= height:
            findings.append(_finding(
                'minor', 'camera.height_unusual',
                f'{label} sits at {position[2]:.2f} m; interior views are normally 1.0-1.9 m.'))
        cx, cy = float(position[0]), float(position[1])
        for entry in entries:
            left, bottom, right, top = aabb(entry)
            if left <= cx <= right and bottom <= cy <= top:
                entry_bottom = _support_resolves_bottom(entry, objects, height) or 0.0
                entry_height = _dimensions(entry)[2]
                if entry_bottom <= position[2] <= entry_bottom + entry_height:
                    findings.append(_finding(
                        'blocking', 'camera.inside_object',
                        f'{label} is inside {entry["id"]} ({entry.get("category")}).',
                        [entry['id']]))
        # A target buried inside a solid object frames an occluded surface.
        target = camera['target']
        for entry in entries:
            left, bottom, right, top = aabb(entry)
            if not (left <= target[0] <= right and bottom <= target[1] <= top):
                continue
            entry_bottom = _support_resolves_bottom(entry, objects, height) or 0.0
            if entry_bottom <= target[2] <= entry_bottom + _dimensions(entry)[2]:
                findings.append(_finding(
                    'minor', 'camera.targets_object_interior',
                    f'{label} aims inside {entry["id"]} ({entry.get("category")}); '
                    'frame its surface or the room instead.', [entry['id']]))
                break

    # ---- 11. Duplicate instances at one spot ----------------------------
    # An object resting on another legitimately shares its XY, so compare only
    # entries that are not linked by a support chain.
    seen_positions = {}
    for entry in entries:
        key = (round(float(entry['xy'][0]), 2), round(float(entry['xy'][1]), 2))
        seen_positions.setdefault(key, []).append(entry)
    for key, group in seen_positions.items():
        if len(group) < 2:
            continue
        unrelated = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if not related(group[i], group[j]):
                    unrelated.append((group[i]['id'], group[j]['id']))
        if unrelated:
            ids = sorted({name for pair in unrelated for name in pair})
            findings.append(_finding('major', 'object.coincident',
                                     f'{", ".join(ids)} share the same XY position {key} '
                                     'without a support relationship.', ids))

    if strict:
        for finding in findings:
            if finding['code'] in ('object.interpenetrates', 'object.outside_room',
                                   'support.missing_parent', 'object.coincident'):
                finding['severity'] = 'blocking'

    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f['severity'], 9), f['code']))
    counts = {}
    for finding in findings:
        counts[finding['severity']] = counts.get(finding['severity'], 0) + 1
    return {
        'schema_version': 'scene_logic_report_v1',
        'findings': findings,
        'counts': counts,
        'blocking': counts.get('blocking', 0),
        'plausible': counts.get('blocking', 0) == 0 and counts.get('major', 0) == 0,
        'scope': ('Deterministic local plausibility checks. No reference image exists for a '
                  'text-authored scene, so these rules test physical and habitual coherence, '
                  'not visual fidelity to any photograph.'),
    }


def repair_hint(report, limit=3):
    """Condense the worst findings into one instruction for a repair request."""
    ranked = [f for f in report['findings'] if f['severity'] in ('blocking', 'major')]
    if not ranked:
        return None
    lines = [f'- {f["message"]}' for f in ranked[:limit]]
    return ('The layout violates these plausibility rules; fix only what they name and keep '
            'everything else unchanged:\n' + '\n'.join(lines))


def apply_safe_fixes(layout):
    """Apply the repairs that need no model judgement; report what changed.

    Three corrections are unconditional enough to do locally: pull objects back
    inside the room, push coincident objects apart, and re-seat contents whose
    stack would cross the ceiling. All three are pure geometry and reversible,
    and all three are recorded.

    The ceiling correction matters most for books: a model asked for a
    bookshelf "full of books" tends to put a stack *on top* of a 2.1 m
    bookshelf, producing a 3.9 m pile inside a 2.9 m room. Moving the content
    inside the furniture is what the brief actually described.
    """
    import copy
    result = copy.deepcopy(layout)
    room = result['room']
    width, depth = float(room['width']), float(room['depth'])
    ceiling = float(room['height'])
    changes = []
    for entry in result['objects']:
        left, bottom, right, top = aabb(entry)
        dx = dy = 0.0
        if left < -width / 2:
            dx = -width / 2 - left
        elif right > width / 2:
            dx = width / 2 - right
        if bottom < 0:
            dy = -bottom
        elif top > depth:
            dy = depth - top
        if dx or dy:
            before = list(entry['xy'])
            entry['xy'] = [float(entry['xy'][0]) + dx, float(entry['xy'][1]) + dy]
            changes.append({'object_id': entry['id'], 'action': 'moved_inside_room',
                            'from': before, 'to': list(entry['xy']),
                            'reason': 'footprint extended past a room boundary'})
    occupied = {}
    for entry in result['objects']:
        key = (round(float(entry['xy'][0]), 2), round(float(entry['xy'][1]), 2))
        if key in occupied and entry.get('support_id', 'floor') == 'floor':
            before = list(entry['xy'])
            entry['xy'] = [float(entry['xy'][0]) + 0.35, float(entry['xy'][1]) + 0.35]
            changes.append({'object_id': entry['id'], 'action': 'separated_coincident',
                            'from': before, 'to': list(entry['xy']),
                            'reason': f'coincident with {occupied[key]}'})
            key = (round(float(entry['xy'][0]), 2), round(float(entry['xy'][1]), 2))
        occupied.setdefault(key, entry['id'])

    # Re-seat contents whose stack would cross the ceiling. A model writing
    # "a bookshelf full of books" commonly places the stack on top of the
    # shelf, which is geometrically valid but physically impossible in the
    # room — the brief meant inside it.
    by_id = {entry['id']: entry for entry in result['objects']}
    for entry in result['objects']:
        parent_id = entry.get('support_id', 'floor')
        if parent_id == 'floor':
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        parent_height = _dimensions(parent)[2]
        child_height = _dimensions(entry)[2]
        stack_top = parent_height + child_height
        if stack_top <= ceiling - 0.05:
            continue
        # Only containers with internal volume can absorb their contents.
        category = str(parent.get('category', '')).lower()
        container = category in ('shelf', 'bookshelf', 'shelf_unit', 'bookcase',
                                 'cabinet', 'wardrobe', 'sideboard', 'drawer_unit')
        if not container:
            continue
        # Keep the support relationship: the content really does belong to this
        # container, so re-parenting it to the floor would both lose that
        # meaning and create a spurious overlap with the shelf it sits inside.
        # Shrinking the content to the container's usable interior is the
        # correction that matches the brief.
        interior = max(0.05, min(child_height, (ceiling - 0.05) - parent_height))
        # The validator's own headroom rule is 95% of the ceiling, not the raw
        # wall top: keep a margin so a fitted stack is not reported again.
        interior = min(interior, max(0.03, ceiling * 0.90 - parent_height))
        entry['target_dimensions_m'][2] = round(interior, 3)
        changes.append({'object_id': entry['id'], 'action': 'shrunk_to_fit_container',
                        'support_id': parent_id, 'from_height_m': child_height,
                        'to_height_m': round(interior, 3), 'reason': (
                            f'stack on {parent_id} reached {stack_top:.2f} m under a '
                            f'{ceiling:.2f} m ceiling; content belongs inside the '
                            f'{category}, so it was fitted to the interior')})
    # Re-measure after fitting: the loops above change heights, and a check
    # against the pre-correction numbers would leave an impossible stack
    # reported as a residual defect.
    by_id = {entry['id']: entry for entry in result['objects']}
    for entry in result['objects']:
        parent_id = entry.get('support_id', 'floor')
        if parent_id == 'floor':
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        category = str(parent.get('category', '')).lower()
        if category not in ('shelf', 'bookshelf', 'shelf_unit', 'bookcase',
                            'cabinet', 'wardrobe', 'sideboard', 'drawer_unit'):
            continue
        overflow = (_dimensions(parent)[2] + _dimensions(entry)[2]) - (ceiling - 0.05)
        if overflow > 0:
            entry['target_dimensions_m'][2] = round(
                max(0.03, _dimensions(entry)[2] - overflow), 3)
    return result, changes
