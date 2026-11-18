"""Text-authored scenes: describe a room in words, get a buildable Blender scene.

This is a sibling of the photo pipeline, not a mode of it. The contract is
deliberately looser than ``library_layout_v1``: a text scene has no photograph
to be faithful to, so there is nothing to reconstruct and no reference for a
visual reviewer to measure against. What replaces "does it match the photo" is
"is it coherent" — see :mod:`scene_logic`.

Two properties are kept from the photo path because they are cheap and they
carry the whole audit story:

* the same V3 bundle, asset library, deterministic builder and renderer;
* the same hash-bound provenance, so a text scene can be reproduced and its
  inputs audited exactly like a photo scene.

The differences are confined to three places: the prompt (words instead of
images), the review stage (logic findings instead of visual scoring), and the
acceptance label (``text_scene_built`` rather than ``accepted`` — a text scene
is never a photographic reconstruction claim).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

from schema import SCHEMA as BASE_SCHEMA, obj, vector
from spec_v2 import SCENE_TYPES, SCHEMA as V2_SCHEMA
from multiview_spec import SAFE_ID, validate as validate_bundle

FORMAT_VERSION = 'text_scene_v1'

ID = {'type': 'string', 'minLength': 1, 'maxLength': 80, 'pattern': SAFE_ID.pattern}
CATEGORY = {'type': 'string', 'minLength': 1, 'maxLength': 64,
            'pattern': r'^[a-z][a-z0-9_]{0,63}$'}
TEXT = {'type': 'string', 'maxLength': 12000}
CAMERA = obj({'position': vector(3), 'target': vector(3),
              'lens_mm': {'type': 'number', 'minimum': 12, 'maximum': 85},
              'roll_degrees': {'type': 'number', 'minimum': -180, 'maximum': 180}})

# Deliberately sparse. Every field the model must supply is required, and
# nothing else is: a text scene is allowed to be approximate, but it is not
# allowed to be underspecified in a way the builder cannot render.
SCHEMA = obj({
    'schema_version': {'type': 'string', 'enum': [FORMAT_VERSION]},
    'scene_id': copy.deepcopy(ID),
    'scene_type': {'type': 'string', 'enum': SCENE_TYPES},
    'description': TEXT,
    'seed': {'type': 'integer', 'minimum': 0, 'maximum': 2147483647},
    'room': obj({
        'width': {'type': 'number', 'minimum': 2, 'maximum': 30},
        'depth': {'type': 'number', 'minimum': 2, 'maximum': 40},
        'height': {'type': 'number', 'minimum': 2, 'maximum': 8},
        'wall_color': vector(3, 0, 1),
        'floor_color': vector(3, 0, 1),
        'ceiling_color': vector(3, 0, 1),
        'style': {'type': 'string', 'maxLength': 120},
    }),
    'lighting': {
        'type': 'object',
        'properties': {
            'energy': {'type': 'number', 'minimum': 50, 'maximum': 4000},
            'color': vector(3, 0, 1),
            'world_strength': {'type': 'number', 'minimum': 0, 'maximum': 2},
        },
        'required': ['energy', 'color', 'world_strength'],
        'additionalProperties': False,
    },
    'lights': copy.deepcopy(V2_SCHEMA['properties']['lights']),
    'exposure': copy.deepcopy(V2_SCHEMA['properties']['render']['properties']['exposure']),
    'world_color': vector(3, 0, 1),
    'cameras': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'items': copy.deepcopy(CAMERA)},
    'objects': {'type': 'array', 'minItems': 1, 'maxItems': 160, 'items': obj({
        'id': copy.deepcopy(ID),
        'category': copy.deepcopy(CATEGORY),
        'xy': vector(2),
        'target_dimensions_m': vector(3, .01, 30),
        'rotation_z_deg': {'type': 'number', 'minimum': -360, 'maximum': 360},
        'support_id': copy.deepcopy(ID),
        'clearance_m': {'type': 'number', 'minimum': 0, 'maximum': .05},
        'supports_objects': {'type': 'boolean'},
        'note': {'type': 'string', 'maxLength': 200},
    })},
    'assumptions': {'type': 'array', 'items': TEXT, 'maxItems': 30},
})
schema = SCHEMA

CONTRACT = '''Design ONE coherent interior scene from the written brief. Return only JSON matching text_scene_v1. There is no photograph: the brief is the only evidence, and the result is judged on whether it would look like a real, habitable space rather than on fidelity to any image. Never output mesh parts, code, commands, asset IDs or file paths.

scene_type is REQUIRED and must be exactly one of: office, living_room, bedroom, classroom, kitchen, shop, other.

Design in this order, and let each step constrain the next:
1. Choose a room size that suits the stated use. A bedroom is not 12 m wide; a classroom is not 3 m deep. Give room.style a short phrase only if the brief implies one (for example "north european minimal"); otherwise use an empty string.
2. Lay out circulation FIRST: decide where a person enters and walks. Keep a clear path of at least 0.8 m between large furniture groups, and never put furniture in front of a door.
3. Place large furniture next. Anything normally set against a wall (bed, wardrobe, cabinet, bookshelf, sofa, desk, counter) must actually touch a wall, within about 0.1 m.
4. Then seating. Every seat faces something worth facing: a sofa faces the television or coffee table, dining chairs face the table, a desk chair faces the desk. Do not leave a chair aimed at a blank wall.
5. Then surfaces and small objects: a lamp or a monitor rests on a table via support_id, and must be smaller than the object supporting it. Never place a support object above the ceiling or stack more than about 2.2 m high.
6. Only then add atmosphere: rug, plants, wall art, ceiling or floor lamps.
7. Then furnish the room as a lived-in space, not a showroom. A room is too empty
   when a visitor could walk a straight line across it without passing anything,
   or when a camera can frame a third of the image as bare floor or bare wall.
   Work to these densities in a room of this size:
   - 70 to 95 objects in total. Fewer than 60 always reads as a display suite.
   - Floor furniture (excluding rugs) should cover 45-60% of the floor in plan.
     Leave real circulation: a 0.9 m path from the door to each seating group.
   - No square patch of open floor larger than about 2.0 x 2.0 m anywhere except
     on a circulation path. Fill such a patch with a side table, a stool, an
     ottoman, a floor plant, a basket, a storage chest or a floor lamp.
   - Every horizontal surface carries 2 to 5 things. A bare tabletop, a bare
     sideboard top and a bare shelf board are all mistakes. A coffee table holds a
     tray, books, a bowl or a mug; a dining table holds a centrepiece plus plates,
     bowls, mugs or a pitcher; a sideboard holds a lamp, a sculpture, a framed
     picture, a basket or a radio; a desk holds a computer, a mug and books.
   - A bookshelf is filled board by board: 5 to 9 books per board on most boards,
     plus a sculpture, a bust, a basket, a board game or a plant on one or two
     boards. Half-empty boards are what make a shelf read as a prop. A shelf unit
     has several boards, so things sharing one shelf may share an xy; the renderer
     distributes them over its boards.
   - On a single surface, though, two objects must not overlap in plan: give each
     its own xy within the surface. If one object genuinely rests on another (a
     bowl on a plate, a book on a stack), make the lower one its support_id
     instead of placing both on the table at the same spot.
   - 3 to 6 framed pictures spread over at least three different walls, at a mix
     of sizes. One lonely picture on one wall leaves the other walls bare.
   - Soft furnishing everywhere it belongs: 3 to 5 cushions across the seating,
     a second rug under a secondary group, curtains at each window.
   Richness comes from many small, correctly supported things, not from bigger
   furniture. Prefer several small objects on a surface over one large one.

8. Prefer these category names: the asset library can supply a real model for
   them, and anything else falls back to a simple shape. Floor: sofa, chair,
   stool, armchair, bench, ottoman, coffee_table, side_table, console_table,
   dining_table, desk, cabinet, sideboard, dresser, chest, shelf, bookshelf, bed,
   room_divider, planter_box, basket, storage_bin. Surfaces: vase, mug, bowl,
   plate, pitcher, basket, fruit, book, book_set, tray, candlestick, sculpture,
   bust, statue, animal_figurine, board_game, radio, terracotta_pot, plant,
   potted_plant, television, computer, monitor. Lighting and textiles: lamp,
   floor_lamp, desk_lamp, pendant_lamp, cushion, throw_pillow, rug, curtain,
   picture, painting. Do not use industrial, workshop, military or medical props
   in a home: no axe, crowbar, jerrycan, oil drum, gas cylinder, toolbox or
   gas mask, even though the library happens to contain them.

Coordinates are metres. X runs -width/2..width/2, Y runs 0..depth, Z is up from the floor. Each object gives its WORLD xy at the footprint centre, target_dimensions_m as [width X, depth Y, height Z] BEFORE yaw, rotation_z_deg (0 means the object's front faces -Y), a support_id that is either floor or the exact id of the object beneath it, and clearance_m (0 for floor objects). Objects resting on furniture must use that furniture's id as support_id and must be no larger than it in plan.

Choose category names in lowercase snake_case, and prefer names an asset library is likely to contain (bed, sofa, table, chair, desk, cabinet, shelf, wardrobe, rug, lamp, plant, tv, monitor, fridge, counter). Use one category per distinct physical kind. Do not invent an exotic category when a common one describes the same object.

Give 1 to 4 cameras. Each camera is a believable viewing position inside the room, aimed at a focal part of the scene, with lens_mm between 16 and 50 for interiors. A camera must not be inside furniture, inside a wall, or aimed at a wall closer than 1 m away. Vary them: at least one stands at 1.4-1.7 m eye height across the room showing the main seating group; one may sit lower or higher (0.9-2.3 m) to read as a design view over the whole room, and one should look along a secondary axis so the three views do not all face the same wall. Every view must have furniture in the foreground and middle distance: a frame filled mostly by bare wall wastes the render.

Use lighting.energy and lighting.world_strength for a plausible interior base exposure, and add explicit lights only where a real room would have a visible fixture worth modelling. A dark room with one window is a design choice; a room with no light source is a mistake.

Record in assumptions anything the brief left open that you had to decide, and any object you wanted but could not size confidently. A coherent room with believable proportions beats both a bare room and an overcrowded one: fill the space with things a household actually owns, keep the circulation clear, and never invent an object you cannot place plausibly.'''

REVISE_CONTRACT = '''Revise the supplied scene so it satisfies the plausibility findings. Change only what the findings name; keep every other object, its id, its position and its category. Do not add objects the findings did not ask for and do not remove objects that are not implicated. Return the complete corrected JSON matching text_scene_v1.'''


def schema_for(scene_id, camera_count):
    """Bind request identity and camera count so the model cannot drift."""
    if not isinstance(scene_id, str) or not SAFE_ID.fullmatch(scene_id):
        raise ValueError('Invalid scene_id')
    if type(camera_count) is not int or not 1 <= camera_count <= 8:
        raise ValueError('camera_count must be an integer in 1..8')
    result = copy.deepcopy(SCHEMA)
    result['properties']['scene_id']['enum'] = [scene_id]
    rule = result['properties']['cameras']
    rule['minItems'] = rule['maxItems'] = camera_count
    return result


def validate_text_scene(raw, scene_id=None, camera_count=None):
    """Strict syntax and referential checks; no plausibility judgement here."""
    from spec_v2 import _walk
    _walk(raw, SCHEMA, 'scene')
    if scene_id is not None and raw['scene_id'] != scene_id:
        raise ValueError('Model changed scene identity')
    if camera_count is not None and len(raw['cameras']) != camera_count:
        raise ValueError('Model changed the camera count')
    room = raw['room']
    width, depth, height = room['width'], room['depth'], room['height']
    objects = {}
    for entry in raw['objects']:
        identifier = entry['id']
        if identifier == 'floor' or identifier in objects:
            raise ValueError('Duplicate or reserved object id: ' + identifier)
        objects[identifier] = entry
    for entry in raw['objects']:
        support = entry['support_id']
        if support == 'floor':
            if entry['clearance_m'] != 0:
                raise ValueError('Floor placement requires zero clearance')
        elif support not in objects:
            raise ValueError('Unknown support_id: ' + support)
        if support == entry['id']:
            raise ValueError('Object cannot support itself: ' + entry['id'])
        # Round-trip the plan into the room the model declared, so a scene that
        # cannot be built never reaches Blender.
        half_w, half_d = width / 2, depth / 2
        x, y = entry['xy']
        if not (-half_w - 0.3 <= x <= half_w + 0.3 and -0.3 <= y <= depth + 0.3):
            raise ValueError(f'{entry["id"]} centre lies outside the room')
        if entry['target_dimensions_m'][2] > height:
            raise ValueError(f'{entry["id"]} is taller than the room')
    visiting, done = set(), set()

    def visit(identifier):
        if identifier in visiting:
            raise ValueError('Cyclic support graph at ' + identifier)
        if identifier in done:
            return
        visiting.add(identifier)
        parent = objects[identifier]['support_id']
        if parent != 'floor':
            visit(parent)
        visiting.remove(identifier)
        done.add(identifier)

    for identifier in objects:
        visit(identifier)
    for camera in raw['cameras']:
        position = camera['position']
        if not (0 <= position[2] <= height + 0.5):
            raise ValueError('Camera is outside the room vertically')
        if abs(position[0]) > width / 2 + 0.5 or not (-0.5 <= position[1] <= depth + 0.5):
            raise ValueError('Camera is outside the room footprint')
        if sum((a - b) ** 2 for a, b in zip(position, camera['target'])) < .01:
            raise ValueError('Camera position and target must differ')
    return raw


def _cameras_to_views(cameras, width, height):
    """Give each camera a view record; text scenes have no source photographs."""
    views = []
    for index, camera in enumerate(cameras):
        views.append({'id': f'view_{index:02d}',
                      'source_image': f'references/view_{index:02d}.png',
                      'width': width, 'height': height,
                      'camera': copy.deepcopy(camera)})
    return views


def expand_to_bundle(raw, image_size=(1600, 900)):
    """Turn a text layout into the same V3 bundle the photo pipeline renders.

    Objects become base-scene records whose category is rewritten to a kind the
    renderer models natively (sofa, table, chair, …) and whose colour comes from
    the brief or the category default. Asset selection may later replace them
    with library models; when it does not, the renderer still produces furniture
    rather than a grey block.
    """
    validate_text_scene(raw)
    compiled, category_report = compile_native_categories(raw)
    compiled, lighting_changes = normalize_lighting(compiled)
    category_report['lighting'] = lighting_changes
    styled = styling_for(compiled)
    objects = {entry['id']: entry for entry in styled['objects']}
    bottoms = {}

    room_height = float(styled['room']['height'])

    def bottom(identifier):
        if identifier not in bottoms:
            entry = objects[identifier]
            category = str(entry.get('category', '')).lower()
            height = entry['target_dimensions_m'][2]
            parent = entry['support_id']
            if category in CEILING_MOUNTED:
                # A pendant hangs from the ceiling: its top sits at the slab.
                bottoms[identifier] = max(0.0, room_height - height)
            elif category in WALL_MOUNTED:
                # Pictures hang at eye level on the wall they straddle.
                bottoms[identifier] = max(0.0, min(room_height - height,
                                                    WALL_MOUNT_HEIGHT - height / 2))
            elif parent == 'floor':
                bottoms[identifier] = 0.0
            else:
                parent_entry = objects[parent]
                parent_kind = str(parent_entry.get('category', '')).lower()
                parent_height = float(parent_entry['target_dimensions_m'][2])
                # A cushion belongs on the seat, not on top of the backrest: the
                # parent's full height would float it above the sofa.
                if category in ('cushion', 'pillow') and parent_kind in (
                        'sofa', 'chair', 'armchair', 'bed', 'bench', 'stool'):
                    bottoms[identifier] = (bottom(parent) + parent_height * 0.42
                                           + entry['clearance_m'])
                else:
                    bottoms[identifier] = (bottom(parent) + parent_height
                                           + entry['clearance_m'])
        return bottoms[identifier]

    # Books on a shelf: the brief names the shelf, not which board, so every book
    # was landing on top of the cabinet as one row. Spread them over the shelf's
    # own boards instead, which is what the object means.
    shelved = {}
    for entry in styled['objects']:
        parent = objects.get(entry['support_id'])
        if parent is None:
            continue
        if str(parent.get('category', '')).lower() in ('shelf', 'bookshelf', 'bookcase', 'cabinet'):
            before = bottoms.get(entry['id'])
            shelved.setdefault(entry['support_id'], []).append(entry)
    for parent_id, children in shelved.items():
        parent = objects[parent_id]
        parent_bottom = bottoms.get(parent_id, 0.0)
        parent_height = float(parent['target_dimensions_m'][2])
        rows = max(2, min(5, int(round(parent_height / 0.36))))
        board = parent_height / rows
        span_x, span_y = (float(parent['target_dimensions_m'][0]),
                          float(parent['target_dimensions_m'][1]))
        yaw = float(parent.get('rotation_z_deg', 0)) % 180
        along_y = abs(yaw - 90) < 45
        span = span_y if along_y else span_x
        centre = float(parent['xy'][1] if along_y else parent['xy'][0])
        usable = max(0.05, span - 0.06)
        lean = 0.006

        # Boards are filled by what actually fits. A brief may describe a shelf as
        # five rows of books where each "row" is a metre wide, and packing three
        # of those onto a 1.35 m board pushed them past the shelf's ends; the
        # binder then refused the whole scene for want of a support surface, and
        # every object in the room fell back to a white primitive.
        # Widest first, each onto the board with the most room left: a plain
        # left-to-right fill strands the small props after the wide runs have
        # taken a board each.
        def row_used(group):
            return (sum(max(0.02, float(e['target_dimensions_m'][0])) for e in group)
                    + lean * max(0, len(group) - 1))

        groups = [[] for _ in range(rows)]
        order = sorted(children,
                       key=lambda e: -max(0.02, float(e['target_dimensions_m'][0])))
        for entry in order:
            width = max(0.02, float(entry['target_dimensions_m'][0]))
            best, best_room = None, -1.0
            for group in groups:
                room = usable - row_used(group) - (lean if group else 0.0)
                if room >= width and room > best_room:
                    best, best_room = group, room
            if best is None:
                best = min(groups, key=row_used)
            best.append(entry)
        # Keep the brief's own order within a board, and drop boards left empty.
        position = {id(entry): index for index, entry in enumerate(children)}
        groups = [sorted(group, key=lambda e: position[id(e)])
                  for group in groups if group]

        # The brief lists shelf contents as one run of xy, so without re-packing
        # every board holds the same short huddle and the rest reads as bare.
        # Books lean against each other, and the slack becomes one or two gaps
        # between clusters rather than even air between every item.
        for row, group in enumerate(groups):
            widths = [max(0.02, float(e['target_dimensions_m'][0])) for e in group]
            packed = sum(widths) + lean * max(0, len(group) - 1)
            slack = max(0.0, usable - packed)
            splits = [] if slack <= 0.12 or len(group) < 3 else (
                [len(group) // 2] if len(group) < 6 else
                [len(group) // 3, 2 * len(group) // 3])
            gap = (slack * 0.7 / len(splits)) if splits else 0.0
            cursor = centre - usable / 2 + (slack - gap * len(splits)) / 2
            for index, (entry, width) in enumerate(zip(group, widths)):
                if index and index in splits:
                    cursor += gap
                position = cursor + width / 2
                if along_y:
                    entry['xy'] = [float(entry['xy'][0]), float(position)]
                else:
                    entry['xy'] = [float(position), float(entry['xy'][1])]
                cursor += width + lean
            for entry in group:
                item_height = float(entry['target_dimensions_m'][2])
                level = parent_bottom + board * (row + 1) - item_height
                bottoms[entry['id']] = max(parent_bottom, level)

    width, height = image_size
    views = _cameras_to_views(styled['cameras'], width, height)
    first = views[0]
    base = {
        'schema_version': 1,
        'scene_type': 'other' if styled['scene_type'] == 'shop' else styled['scene_type'],
        'description': styled['description'],
        'seed': styled['seed'],
        'room': {key: copy.deepcopy(styled['room'][key])
                 for key in ('width', 'depth', 'height', 'wall_color', 'floor_color', 'ceiling_color')},
        'camera': {key: copy.deepcopy(first['camera'][key]) for key in ('position', 'target', 'lens_mm')},
        'lighting': copy.deepcopy(styled['lighting']),
        'objects': [{
            'id': entry['id'],
            # The base schema constrains category to its own enum, so the
            # modelled kind travels as the renderer's `kind` field instead.
            'category': _base_category(entry['category']),
            'position': [float(entry['xy'][0]), float(entry['xy'][1]), bottom(entry['id'])],
            'dimensions': copy.deepcopy(entry['target_dimensions_m']),
            'rotation_z_deg': entry['rotation_z_deg'],
            'color': list(entry['color']),
            'repeat': [1, 1], 'spacing': [0, 0], 'confidence': .5,
        } for entry in styled['objects']],
        'assumptions': list(styled['assumptions']),
    }
    from spec_v2 import upgrade_v1
    shared = upgrade_v1(base, first['width'], first['height'], first['id'])

    # Hanging textiles are the one part the primitive box cannot fake: two grey
    # slabs beside a window read as pillars. The renderer already has a bounded
    # cloth mesh with real folds, so a curtain is carried as custom geometry
    # instead of a base-schema object (whose category enum has no curtain).
    soft = [entry for entry in styled['objects'] if entry['category'].lower() == 'curtain']
    if soft:
        base['objects'] = [entry for entry in base['objects']
                           if entry['id'] not in {item['id'] for item in soft}]
        shared['base_scene']['objects'] = copy.deepcopy(base['objects'])
        shared['custom_objects'] = []
        for entry in soft:
            width, depth, height = entry['target_dimensions_m']
            has_top = bottom(entry['id'])
            material_id = 'textile_' + entry['id']
            shared['materials'].append({
                'id': material_id,
                'color': list(entry['color']),
                'roughness': .92, 'metallic': 0, 'transmission': 0,
                'noise_scale': 0, 'bump_strength': 0, 'bump_distance': 0,
                'texture_id': '',
            })
            # A described curtain is "drawn aside to leave the middle open", and a
            # single slab spanning the whole wall is what made the render look like
            # a closed accordion blind. Wide runs are built as two gathered panels
            # with daylight between them; a narrow run is already one panel.
            panels = [(0.0, width)]
            if width > 2.2:
                panel = width * 0.30
                panels = [(-(width - panel) / 2.0, panel), ((width - panel) / 2.0, panel)]
            obj = {
                'id': entry['id'],
                'position': [float(entry['xy'][0]), float(entry['xy'][1]),
                             float(has_top) + height / 2],
                'rotation_degrees': [0, 0, float(entry['rotation_z_deg'])],
                'material_id': material_id,
                'confidence': .5,
                'observed': False,
                'support_id': entry['support_id'],
                'parts': [],
            }
            for index, (offset, panel) in enumerate(panels):
                folds = max(4, min(14, int(round(panel / 0.12))))
                obj['parts'].append({
                    'type': 'curtain',
                    'position': [float(offset), 0, 0],
                    'size': [float(panel), float(depth), float(height)],
                    'rotation_degrees': [0, 0, 0],
                    'radius': 0, 'points': [], 'vertices': [], 'faces': [],
                    'material_id': material_id,
                    'cloth': [{
                        'fold_count': folds,
                        'fold_depth': min(0.10, max(0.03, depth * 0.55)),
                        'sag': 0.02,
                        'thickness': 0.003,
                        'segments_x': max(32, folds * 6),
                        'segments_z': 20,
                        'seed': (int(styled.get('seed') or 0) + index * 7) % 997,
                    }],
                })
            shared['custom_objects'].append(obj)
    shared['scene_type'] = styled['scene_type']
    shared['camera']['roll_degrees'] = first['camera']['roll_degrees']
    # A floor lamp is both an object (the body you can see) and a light (what it
    # emits), and the model naturally gives both the same name. That is a sensible
    # description, not a mistake, but the scene spec needs one id per thing, so the
    # emitter is renamed rather than failing the whole scene.
    taken = {entry['id'] for entry in styled['objects']}
    lights = copy.deepcopy(styled['lights'])
    for light in lights:
        if light.get('id') in taken:
            candidate = str(light['id']) + '_emitter'
            suffix = 2
            while candidate in taken:
                candidate = f"{light['id']}_emitter_{suffix}"
                suffix += 1
            light['id'] = candidate
        taken.add(light.get('id'))
    shared['lights'] = lights
    shared['render'] = {'exposure': styled['exposure'],
                        'world_color': list(styled['world_color']),
                        # A photograph can show that a room really is lit by one
                        # lamp, so the photo path lets explicit lights replace
                        # the inferred ceiling grid. A written brief cannot:
                        # "a floor lamp in the corner" describes an addition, not
                        # the absence of ceiling light. Keep the base fixtures
                        # and layer the explicit lights on top, otherwise every
                        # text scene renders as a single dim point source.
                        'replace_base_lights': False}
    bundle = {
        'schema_version': 3,
        'scene_id': styled['scene_id'],
        'shared_scene': shared,
        'architecture': {'omit_surfaces': [], 'front_wall': False},
        'views': views,
        'texture_sources': [],
    }
    validate_bundle(bundle)
    return bundle, category_report


def _base_category(kind, requested=None):
    """Map a modelled kind onto the base schema's category enum.

    ``generate_v2.expand_objects`` derives the builder's ``kind`` from
    ``entry['category']``, so the base category *is* the modelling instruction.
    Its enum already contains sofa, bed, table, chair, desk, cabinet, shelf,
    plant, monitor and computer — the ten shapes the renderer can build — so
    the mapped kind can generally be passed straight through. Anything outside
    that enum (a wardrobe, a coffee table, a floor lamp) keeps the honest
    primitive box, coloured to read as the right material.
    """
    from schema import CATEGORIES
    if kind in CATEGORIES:
        return kind
    return 'box'


def prepare_reference_images(bundle, directory):
    """Write a neutral placeholder for each view's reference path.

    The renderer resolves ``source_image`` and packs it as a camera background
    guide; it also hashes it into provenance. A text scene has no photograph,
    so the honest artefact is a blank frame that says so — not a fabricated
    image. It is always paired with ``tone_match=off``, so it never drives
    exposure or colour decisions.
    """
    from PIL import Image, ImageDraw
    directory = Path(directory)
    written = []
    for view in bundle['views']:
        target = (directory / view['source_image']).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new('RGB', (view['width'], view['height']), (32, 32, 36))
        draw = ImageDraw.Draw(image)
        label = f'NO SOURCE PHOTOGRAPH - text scene - {view["id"]}'
        draw.text((24, 24), label, fill=(190, 190, 196))
        draw.rectangle((8, 8, view['width'] - 9, view['height'] - 9), outline=(90, 90, 98))
        image.save(target)
        written.append({'view_id': view['id'], 'path': view['source_image'],
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'role': 'neutral_placeholder_not_a_photograph'})
    return written


def _selection_options(base_options, overrides=None):
    """Loosen size matching slightly for invented scenes.

    A written brief names kinds of object, not measured antiques: "a low coffee
    table" has no exact depth. The default 15% tolerance is calibrated for
    photo reconstruction where the photograph constrains the size. For a text
    scene a wider tolerance plus bounded settle is what makes an invented
    layout actually buildable, without letting a sofa become a stool.
    """
    options = copy.deepcopy(base_options or {})
    options.setdefault('dimension_tolerance', 0.30)
    options.setdefault('min_uniform_scale', 0.70)
    options.setdefault('max_uniform_scale', 1.40)
    options.setdefault('settle_max_shift_m', 0.35)
    options.setdefault('settle_max_omitted_fraction', 0.10)
    options.setdefault('allow_portable_floor_props_on_support', True)
    if overrides:
        options.update(overrides)
    return options


# ---------------------------------------------------------------------------
# Native geometry mapping
# ---------------------------------------------------------------------------
# The renderer already knows how to build believable furniture from dimensions
# and a colour: ``interior_assets`` models beds, sofas, cabinets, shelving and
# chairs with wood/fabric/metal materials, and ``detailed_assets`` models desks,
# tables, chairs, plants, monitors and towers. Everything else falls through to
# a plain box.
#
# A text scene must therefore compile its vocabulary into those two builders.
# Writing ``category='box'`` for a sofa is what made the first render a room of
# white cubes: the base scene carries only a category, so the renderer had no
# way to know a sofa was intended.
NATIVE_KINDS = {
    'bed': 'bed', 'bunk_bed': 'bed', 'single_bed': 'bed', 'double_bed': 'bed',
    'sofa': 'sofa', 'couch': 'sofa', 'loveseat': 'sofa', 'armchair': 'chair',
    'cabinet': 'cabinet', 'wardrobe': 'cabinet', 'sideboard': 'cabinet',
    'dresser': 'cabinet', 'drawer_unit': 'cabinet', 'locker': 'cabinet',
    'cupboard': 'cabinet', 'fridge': 'cabinet', 'refrigerator': 'cabinet',
    'bookshelf': 'shelf', 'shelf': 'shelf', 'shelves': 'shelf',
    'shelf_unit': 'shelf', 'bookcase': 'shelf', 'rack': 'shelf',
    'chair': 'chair', 'dining_chair': 'chair', 'stool': 'chair', 'bench': 'chair',
    'office_chair': 'chair', 'desk_chair': 'chair', 'computer_chair': 'chair',
    'task_chair': 'chair', 'accent_chair': 'chair', 'kitchen_chair': 'chair',
    'desk': 'desk', 'office_desk': 'desk', 'workstation': 'desk',
    'table': 'table', 'dining_table': 'table', 'kitchen_table': 'table',
    'coffee_table': 'table', 'side_table': 'table', 'console_table': 'table',
    'tv_stand': 'table', 'tv_unit': 'table', 'nightstand': 'table',
    'plant': 'plant', 'potted_plant': 'plant', 'houseplant': 'plant',
    'monitor': 'monitor', 'screen': 'monitor', 'display': 'monitor',
    'computer': 'computer', 'desktop': 'computer', 'pc_tower': 'computer',
    # Small props the renderer can shape convincingly; without these a lamp or
    # a television renders as a bare box and the whole room reads as unbuilt.
    'lamp': 'lamp', 'table_lamp': 'lamp', 'desk_lamp': 'lamp',
    'floor_lamp': 'lamp', 'reading_lamp': 'lamp', 'bedside_lamp': 'lamp',
    'tv': 'tv', 'television': 'tv', 'tv_console': 'cabinet',
    'book': 'book', 'books': 'book',
    # A library "book set" asset is one solid block, and the shelf assets in the
    # catalog expose only their top face as a support surface, so a bound row of
    # books cannot be placed on a board at all. Built here instead, a row becomes
    # real separate books standing on the board the distributor gives it.
    'book_set': 'book_row', 'book_row': 'book_row', 'books_row': 'book_row',
    'book_stack': 'book', 'book_pile': 'book',
    # Tableware and shelf dressing. Without a builder each of these is a plain
    # box, and a laid dining table reads as a row of white blocks — the thing
    # that makes a correctly dressed room still look unbuilt.
    'bowl': 'bowl', 'serving_bowl': 'bowl', 'fruit_bowl': 'bowl',
    'plate': 'plate', 'dinner_plate': 'plate', 'saucer': 'plate',
    'mug': 'mug', 'cup': 'mug', 'teacup': 'mug', 'coffee_cup': 'mug',
    'glass': 'mug', 'tumbler': 'mug',
    'pitcher': 'pitcher', 'jug': 'pitcher', 'carafe': 'pitcher',
    'teapot': 'pitcher', 'kettle': 'pitcher',
    'basket': 'basket', 'wicker_basket': 'basket', 'storage_basket': 'basket',
    'candlestick': 'candlestick', 'candle': 'candlestick',
    'candle_holder': 'candlestick', 'candelabra': 'candlestick',
    'sculpture': 'sculpture', 'statue': 'sculpture', 'figurine': 'sculpture',
    'bust': 'sculpture', 'ornament': 'sculpture', 'animal_figurine': 'sculpture',
    'ottoman': 'ottoman', 'pouffe': 'ottoman', 'footstool': 'ottoman',
    'planter_box': 'planter_box', 'planter': 'planter_box',
    'radio': 'radio', 'bookshelf_speaker': 'radio',
    'board_game': 'board_game', 'game_box': 'board_game',
    'decor': 'decor', 'vase': 'vase',
    # Wall dressing and soft furnishing the renderer now models in its own
    # builders; leaving them unmapped turned each one into a plain box.
    'painting': 'painting', 'picture': 'painting', 'art': 'painting',
    'wall_art': 'painting', 'artwork': 'painting', 'poster': 'painting',
    'cushion': 'cushion', 'pillow': 'cushion', 'throw_pillow': 'cushion',
    'tray': 'tray',
    # Hanging fixtures: a pendant is not a box on the floor, and it is one of
    # the few props that can carry a room's lighting design.
    'pendant_lamp': 'pendant_light', 'pendant': 'pendant_light',
    'ceiling_lamp': 'pendant_light', 'chandelier': 'pendant_light',
    'ceiling_light': 'pendant_light', 'downlight': 'linear_light',
}

# Wall-mounted and ceiling-mounted categories: the model marks their support as
# the floor, so without a rule they end up lying on the ground.
WALL_MOUNTED = frozenset({'painting', 'picture', 'art', 'wall_art', 'artwork', 'poster',
                          'mirror', 'wall_clock', 'clock', 'tv'})
CEILING_MOUNTED = frozenset({'pendant_light', 'chandelier', 'ceiling_light',
                             'linear_light', 'downlight'})
WALL_MOUNT_HEIGHT = 1.35     # centre-line of a picture hung at eye level

# A written brief cannot prove an opening in a wall, and the renderer has no
# wall boolean to cut one. Declaring a window would therefore put a glass panel
# against solid wall paint, which the geometry preflight correctly rejects as a
# daylight aperture that cannot transmit. Such objects are kept out of the
# scene and recorded as assumptions instead of being silently dropped.
UNPROVABLE_OPENINGS = {'window', 'door', 'doorway', 'skylight', 'french_door',
                       'sliding_door', 'archway'}

# Colours used when the brief does not name one. Values are scene-linear
# diffuse reflectance, chosen to read as the material rather than as a paint
# swatch under typical interior exposure.
CATEGORY_COLORS = {
    'bed': (.62, .60, .56), 'sofa': (.30, .31, .33), 'cabinet': (.46, .38, .30),
    'shelf': (.47, .39, .31), 'chair': (.34, .30, .27), 'desk': (.52, .43, .33),
    'table': (.52, .43, .33), 'plant': (.10, .24, .07), 'monitor': (.07, .07, .08),
    'computer': (.10, .10, .11),
    # Soft furnishing and textiles: these were missing, so a rug or a curtain
    # with no colour word in the brief came out the same neutral grey as a box.
    'rug': (.68, .63, .55), 'curtain': (.80, .77, .71), 'cushion': (.115, .100, .085),
    'lamp': (.86, .82, .72), 'tv': (.06, .06, .07), 'book': (.24, .20, .18),
    'decor': (.72, .68, .60), 'carpet': (.68, .63, .55),
    # Tableware and dressing. Each of these fell to the neutral default grey,
    # which is why a wicker basket came out the colour of a filing cabinet and
    # a ceramic bowl the colour of concrete. These are the material's own tone,
    # so a brief that names no colour still gets wicker, china and brass.
    'basket': (.64, .47, .26), 'bowl': (.86, .84, .78), 'plate': (.90, .89, .85),
    'mug': (.84, .83, .80), 'pitcher': (.78, .76, .71),
    'candlestick': (.66, .55, .26), 'sculpture': (.74, .71, .64),
    'tray': (.42, .34, .26), 'vase': (.80, .78, .72),
    'painting': (.38, .36, .40), 'book_row': (.32, .25, .20),
}

# Colours that read as a material family when the brief mentions it.
COLOR_WORDS = (
    # A brief may be written in Chinese, and the object notes are then Chinese
    # too: without these terms every object fell back to its category default
    # and a "浅米色羊毛地毯" rendered mid-grey.
    (('white', 'ivory', 'cream', '白色', '奶白', '象牙', '米白'), (.88, .87, .84)),
    (('black', 'charcoal', 'graphite', '黑色', '炭黑', '墨黑'), (.07, .07, .075)),
    (('grey', 'gray', 'slate', 'concrete', '灰色', '浅灰', '深灰', '水泥'), (.36, .37, .38)),
    (('beige', 'sand', 'taupe', 'linen', '米色', '浅米', '米黄', '亚麻', '驼色'), (.70, .66, .58)),
    (('brown', 'walnut', 'oak', 'wood', 'timber', 'teak',
      '棕色', '胡桃', '橡木', '实木', '木质', '原木', '木色'), (.42, .30, .19)),
    (('navy', 'blue', '蓝色', '深蓝', '藏青'), (.12, .18, .34)),
    (('green', 'olive', 'sage', '绿色', '墨绿', '橄榄'), (.22, .32, .20)),
    (('red', 'burgundy', 'terracotta', 'rust', '红色', '赭', '砖红', '陶土'), (.40, .16, .12)),
    (('yellow', 'mustard', 'ochre', '黄色', '芥末'), (.62, .48, .16)),
)


def _clauses(text):
    """Split a brief into clauses so a colour word stays with its own object."""
    import re
    parts = re.split(r'[.;:,]|\band\b|\bwith\b|\bplus\b', (text or '').lower())
    return [part.strip() for part in parts if part.strip()]


def _clauses(text):
    """Split a brief into clauses so a colour word stays with its own object."""
    import re
    parts = re.split(r'[.;:,]|\band\b|\bwith\b|\bplus\b', (text or '').lower())
    return [part.strip() for part in parts if part.strip()]


def normalize_lighting(raw):
    """Bring a stated light budget into a range that reads as a lit interior.

    ``lighting.energy`` is a room-wide area-light budget in watts, and it has an
    honest physical meaning: the same 900 W that lights a 21 m2 room evenly
    blows out a 6 m2 room and barely reaches across a 60 m2 hall. A language
    model choosing that number has no way to know that, and in practice it
    picks a plausible-sounding constant — the first text scene rendered here at
    900 W came out almost pure white.

    So treat the model's value as intent rather than measurement: keep its
    relative brightness, rescale it to a per-square-metre target, and record the
    change. An explicitly small or large value still reads as dim or bright,
    because the ratio is preserved.
    """
    result = copy.deepcopy(raw)
    room = result['room']
    area = float(room['width']) * float(room['depth'])
    if area <= 0:
        return result, []
    stated = float(result['lighting']['energy'])
    # Calibrated against this renderer, not against photography: a 21 m2 room
    # under AgX reads correctly at roughly 180 W total across four ceiling
    # emitters, i.e. about 8-9 W/m2. 900 W in the same room renders almost pure
    # white, which is the mistake a model makes by picking a round number.
    target = 8.5 * area
    # A stated value within a factor of two of the calibrated target is read as
    # intent (a dim bedroom, a bright showroom) and its ratio is kept. Beyond
    # that it is a unit mistake — a language model choosing a round number has
    # no sense of watts per square metre — so the calibrated target wins.
    ratio = stated / target
    ratio = ratio if 0.5 <= ratio <= 2.0 else 1.0
    corrected = round(target * ratio, 1)
    changes = []
    if abs(corrected - stated) > max(1.0, 0.02 * stated):
        changes.append({'field': 'lighting.energy', 'from': stated, 'to': corrected,
                        'reason': (f'{area:.1f} m2 room; rescaled to ~40 W/m2 while preserving '
                                   f'the stated relative brightness (x{ratio:.2f})')})
        result['lighting']['energy'] = corrected
    world = float(result['lighting'].get('world_strength', 0))
    if world > 0.6:
        changes.append({'field': 'lighting.world_strength', 'from': world, 'to': 0.6,
                        'reason': 'ambient above 0.6 flattens interior contrast'})
        result['lighting']['world_strength'] = 0.6
    # Named fixtures are added to the base budget, not substituted for it, so a
    # brief that asks for two pendants and a floor lamp arrives as the calibrated
    # ceiling budget PLUS whatever the model costed those fixtures at. Left alone
    # that lands a 39 m2 room at 29 W/m2 against a calibrated 8.5 and the render
    # comes back white. The fixtures keep their relative strengths and the group
    # is scaled so the whole room stays inside one budget.
    explicit = [light for light in (result.get('lights') or [])
                if float(light.get('energy') or 0) > 0]
    fixtures = sum(float(light['energy']) for light in explicit)
    if explicit and fixtures > 0 and not result.get('replace_base_lights'):
        base_energy = float(result['lighting']['energy'])
        combined = base_energy + fixtures
        # One room, one budget. A little over the calibrated figure is what a
        # room with its own visible fixtures should be, so the ceiling is 1.15x;
        # past that, the base grid and the fixtures are scaled together, which
        # keeps the lamp a lamp instead of leaving it a dark prop while the
        # ceiling does all the work.
        ceiling = target * 1.15
        if combined > ceiling:
            scale = ceiling / combined
            result['lighting']['energy'] = round(base_energy * scale, 1)
            for light in explicit:
                light['energy'] = round(float(light['energy']) * scale, 1)
            changes.append({'field': 'lighting.energy + lights[].energy',
                            'from': round(combined, 1), 'to': round(ceiling, 1),
                            'reason': (f'{area:.1f} m2 room: base grid and {len(explicit)} named '
                                       f'fixture(s) are one budget, scaled x{scale:.2f} to '
                                       f'~{ceiling / area:.1f} W/m2')})
    exposure = float(result.get('exposure', 0))
    if abs(exposure) > 1.0:
        changes.append({'field': 'exposure', 'from': exposure, 'to': max(-1.0, min(1.0, exposure)),
                        'reason': 'exposure clamped to +/-1 stop; larger values saturate highlights'})
        result['exposure'] = max(-1.0, min(1.0, exposure))
    return result, changes


def color_for(category, description='', note=''):
    """Pick a believable colour for one object.

    Colour words in a brief describe *one* object ("a grey sofa"). A fixed-size
    window around the category name still leaks across clauses — "a grey sofa
    facing a low TV unit, a coffee table between them" would tint the table grey
    too. Splitting into clauses first keeps each colour with its own noun, and
    the object's own note takes priority when present.
    """
    kind = NATIVE_KINDS.get(str(category).lower(), str(category).lower())
    synonyms = {str(category).lower(), str(category).lower().replace('_', ' '),
                kind, kind.replace('_', ' ')}
    scopes = [note or '']
    for clause in _clauses(description):
        if any(synonym in clause for synonym in synonyms):
            scopes.append(clause)
    for scope in scopes:
        for words, value in COLOR_WORDS:
            if any(word in scope for word in words):
                return list(value)
    return list(CATEGORY_COLORS.get(kind, (.35, .35, .36)))


def compile_native_categories(raw):
    """Rewrite categories so the renderer builds real furniture where it can.

    Returns the layout unchanged in structure plus a report of what was mapped,
    so a reader can always tell which objects are modelled and which are
    placeholders. Unknown categories are deliberately left alone rather than
    guessed into an unrelated shape — a box labelled as a bench would be worse
    than an honest box.
    """
    result = copy.deepcopy(raw)
    mapped, unmapped, openings = [], [], []
    kept = []
    for entry in result['objects']:
        source = str(entry.get('category', '')).lower()
        if source in UNPROVABLE_OPENINGS:
            openings.append({'object_id': entry['id'], 'category': source})
            continue
        # Children of a removed opening fall back to the floor rather than
        # dangling from an object that no longer exists.
        kept.append(entry)
        native = NATIVE_KINDS.get(source)
        if native:
            entry['category'] = native
            if native == 'lamp' and ('floor' in source or entry['target_dimensions_m'][2] > 1.0):
                # A floor lamp is a lamp scaled tall; the renderer's silhouette
                # is proportional, so the flag only drives the audit record.
                entry['category'] = 'lamp'
            mapped.append({'object_id': entry['id'], 'from': source, 'to': native})
        else:
            unmapped.append({'object_id': entry['id'], 'category': source})
    if openings:
        surviving = {entry['id'] for entry in kept}
        for entry in kept:
            if entry['support_id'] != 'floor' and entry['support_id'] not in surviving:
                entry['support_id'] = 'floor'
                entry['clearance_m'] = 0
        result['objects'] = kept
        result.setdefault('assumptions', []).append(
            'Wall openings were described but not modelled: a text brief cannot establish a real '
            'aperture, and a glass panel against solid wall would block its own daylight. '
            'Recorded: ' + ', '.join(f'{o["object_id"]} ({o["category"]})' for o in openings) + '.')
    return result, {'mapped': mapped, 'unmapped': unmapped, 'unmodelled_openings': openings,
                    'note': ('Mapped categories use the renderer\'s modelled furniture; '
                             'unmapped ones keep the renderer\'s default primitive.')}


# A shelf of eighteen books in one brown reads as a single grey mass. Real
# shelves are a spread of muted spines, so books get a palette and cycle through
# it by their position in the run; everything else keeps the described colour.
# Textiles that read as different fabrics on one sofa, in linear albedo. The
# single default was mid-grey, which AgX at +0.7 exposure lifts to cream, so a
# row of five cushions came out as five identical pale cards.
CUSHION_TONES = (
    (0.115, 0.100, 0.085), (0.055, 0.070, 0.095), (0.210, 0.150, 0.090),
    (0.090, 0.115, 0.090), (0.150, 0.125, 0.120),
)

BOOK_SPINES = (
    (0.42, 0.13, 0.11), (0.12, 0.20, 0.31), (0.52, 0.44, 0.27),
    (0.16, 0.26, 0.20), (0.31, 0.16, 0.25), (0.62, 0.55, 0.42),
    (0.10, 0.10, 0.11), (0.66, 0.35, 0.14), (0.34, 0.37, 0.40),
)


def styling_for(raw):
    """Apply per-object colour so the scene reads as designed, not as grey clay."""
    result = copy.deepcopy(raw)
    description = result.get('description', '')
    book = 0
    cushion = 0
    for entry in result['objects']:
        category = str(entry.get('category', '')).lower()
        color = color_for(entry.get('category'), description)
        if category == 'book':
            color = list(BOOK_SPINES[book % len(BOOK_SPINES)])
            book += 1
        elif category == 'cushion':
            # Keep a colour the brief actually named; otherwise give each its own.
            if color == list(CATEGORY_COLORS.get('cushion', ())):
                color = list(CUSHION_TONES[cushion % len(CUSHION_TONES)])
            cushion += 1
        entry['color'] = color
    return result

