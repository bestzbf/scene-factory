"""Bounded data-only per-photo scene extension; v1 remains unchanged.

V2 top-level fields: schema_version, scene_type, base_scene, image, materials,
textures, surface_overrides, custom_objects, lights, camera, render, and additive
base_material_overrides/exterior_contexts. Old saved JSON may omit extensions.
Custom group position is its local origin in world metres (not bottom centre).
Part position is its centre in group space. Curves/meshes use local points and
vertices, then part rotation/position, then group rotation/position. Local
image-plane front is -Y, width=size.x and height=size.z. Box texture front=-Y.
All rotations are XYZ Euler degrees; reference photo crop order UL UR LR LL.
Sphere uses positive XYZ size (ellipsoid) or positive radius (sphere).
Supports refer to custom IDs, expanded base IDs (id__i_j), 'floor', or empty.
For shelf interiors/soft seat surfaces the AABB-top test is not a surface solver;
leave support unspecified when no reliable top contact can be expressed.
Only image metadata lives here; the CLI supplies the actual original photo path.
"""
import copy
import json
import math
from pathlib import Path

from schema import SCHEMA as V1_SCHEMA, obj, vector, color, validate as validate_v1

TEXT = {'type': 'string', 'maxLength': 12000}
ID = {'type': 'string', 'maxLength': 128}
ANGLE = vector(3, -360, 360)
SCENE_TYPES = ['office', 'living_room', 'bedroom', 'classroom', 'kitchen', 'shop', 'other']
MATERIAL_SCHEMA = obj({
    'id': ID, 'color': color,
    'roughness': {'type': 'number', 'minimum': 0, 'maximum': 1},
    'metallic': {'type': 'number', 'minimum': 0, 'maximum': 1},
    'transmission': {'type': 'number', 'minimum': 0, 'maximum': 1},
    'noise_scale': {'type': 'number', 'minimum': 0, 'maximum': 1000},
    'bump_strength': {'type': 'number', 'minimum': 0, 'maximum': 1},
    'bump_distance': {'type': 'number', 'minimum': 0, 'maximum': .1},
    'texture_id': ID,
})
CURTAIN_SCHEMA = obj({
    'fold_count': {'type': 'integer', 'minimum': 2, 'maximum': 32},
    'fold_depth': {'type': 'number', 'minimum': .002, 'maximum': 1},
    'sag': {'type': 'number', 'minimum': 0, 'maximum': 1},
    'thickness': {'type': 'number', 'minimum': .0005, 'maximum': .05},
    'segments_x': {'type': 'integer', 'minimum': 8, 'maximum': 128},
    'segments_z': {'type': 'integer', 'minimum': 4, 'maximum': 64},
    'seed': {'type': 'integer', 'minimum': 0, 'maximum': 2147483647},
})
PART_SCHEMA = obj({
    'type': {'type': 'string', 'enum': ['box', 'cylinder', 'sphere', 'curve', 'mesh', 'image_plane', 'curtain']},
    'position': vector(3), 'size': vector(3, 0, 30), 'rotation_degrees': ANGLE,
    'radius': {'type': 'number', 'minimum': 0, 'maximum': 5},
    'points': {'type': 'array', 'items': vector(3), 'maxItems': 256},
    'vertices': {'type': 'array', 'items': vector(3), 'maxItems': 2048},
    'faces': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 2047}, 'minItems': 3, 'maxItems': 16}, 'maxItems': 4096},
    'material_id': ID,
})
PART_SCHEMA['properties']['cloth'] = {'type': 'array', 'items': CURTAIN_SCHEMA, 'maxItems': 1}
PART_SCHEMA['required'].append('cloth')
BASE_MATERIAL_OVERRIDE_SCHEMA = obj({
    'object_id': ID,
    'material_id': ID,
    'selector': obj({
        'role': {'type': 'string', 'enum': ['', 'body', 'metal', 'fabric', 'glass', 'all']},
        'material_name': {'type': 'string', 'maxLength': 128},
        'slot_index': {'type': 'integer', 'minimum': -1, 'maximum': 63},
    }),
})
EXTERIOR_CONTEXT_SCHEMA = obj({
    'window_id': ID, 'texture_id': ID,
    'depth_m': {'type': 'number', 'minimum': .2, 'maximum': 20},
    'brightness_ev': {'type': 'number', 'minimum': -3, 'maximum': 6},
})
V2_SCHEMA = obj({
    'schema_version': {'type': 'integer', 'enum': [2]},
    'scene_type': {'type': 'string', 'enum': SCENE_TYPES},
    'base_scene': V1_SCHEMA,
    'image': obj({'width': {'type': 'integer', 'minimum': 1, 'maximum': 16384},
                  'height': {'type': 'integer', 'minimum': 1, 'maximum': 16384}, 'source_id': ID}),
    'materials': {'type': 'array', 'items': MATERIAL_SCHEMA, 'maxItems': 128},
    # Optional fields preserve old saved specs byte-for-byte during validation.
    'base_material_overrides': {'type': 'array', 'items': BASE_MATERIAL_OVERRIDE_SCHEMA, 'maxItems': 256},
    'exterior_contexts': {'type': 'array', 'items': EXTERIOR_CONTEXT_SCHEMA, 'maxItems': 32},
    'textures': {'type': 'array', 'maxItems': 32, 'items': obj({'id': ID, 'quad': {'type': 'array', 'items': vector(2, 0, 1), 'minItems': 4, 'maxItems': 4}})},
    'surface_overrides': obj({key: ID for key in ('floor', 'ceiling', 'left', 'right', 'back')}),
    'custom_objects': {'type': 'array', 'maxItems': 80, 'items': obj({
        'id': ID, 'position': vector(3), 'rotation_degrees': ANGLE, 'material_id': ID,
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'observed': {'type': 'boolean'}, 'support_id': ID,
        'parts': {'type': 'array', 'items': PART_SCHEMA, 'minItems': 1, 'maxItems': 100},
    })},
    'lights': {'type': 'array', 'maxItems': 80, 'items': obj({
        'id': ID, 'type': {'type': 'string', 'enum': ['AREA', 'POINT']},
        'position': vector(3), 'target': vector(3), 'color': color,
        'energy': {'type': 'number', 'minimum': 0, 'maximum': 10000},
        'size': {'type': 'number', 'minimum': .001, 'maximum': 20},
        'size_y': {'type': 'number', 'minimum': .001, 'maximum': 20},
        'shape': {'type': 'string', 'enum': ['RECTANGLE', 'DISK']},
        'spread_degrees': {'type': 'number', 'minimum': 1, 'maximum': 180},
    })},
    'camera': obj({'roll_degrees': {'type': 'number', 'minimum': -180, 'maximum': 180}}),
    'render': obj({'exposure': {'type': 'number', 'minimum': -5, 'maximum': 5},
                   'world_color': color, 'replace_base_lights': {'type': 'boolean'}}),
})
SCHEMA = V2_SCHEMA
MAX_PARTS = 800
MAX_VERTICES = 100000
MAX_FACE_INDICES = 300000
MAX_CURVE_POINTS = 16000


def upgrade_v1(spec, width=1600, height=1200, source_id=''):
    """Wrap validated v1 without modifying it; exact v1 default render behavior."""
    validate_v1(spec)
    return validate({
        'schema_version': 2, 'scene_type': spec['scene_type'], 'base_scene': copy.deepcopy(spec),
        'image': {'width': width, 'height': height, 'source_id': source_id},
        'materials': [], 'textures': [],
        'surface_overrides': {key: '' for key in ('floor', 'ceiling', 'left', 'right', 'back')},
        'custom_objects': [], 'lights': [], 'camera': {'roll_degrees': 0},
        'render': {'exposure': .25, 'world_color': [.3, .35, .4], 'replace_base_lights': False},
    })


def part(kind='box', **values):
    """All required fields, including empty unused arrays; callers supply dimensions."""
    result = {'type': kind, 'position': [0, 0, 0], 'size': [0, 0, 0],
              'rotation_degrees': [0, 0, 0], 'radius': 0, 'points': [],
              'vertices': [], 'faces': [], 'material_id': '', 'cloth': []}
    if kind == 'curtain':
        result['cloth'] = [{'fold_count': 8, 'fold_depth': .12, 'sag': .025,
                           'thickness': .002, 'segments_x': 64, 'segments_z': 24,
                           'seed': 23}]
    result.update(values)
    return result


def material(identifier, rgb, **values):
    result = {'id': identifier, 'color': rgb, 'roughness': .5, 'metallic': 0,
              'transmission': 0, 'noise_scale': 0, 'bump_strength': 0,
              'bump_distance': 0, 'texture_id': ''}
    result.update(values)
    return result


def _walk(value, rule, path):
    typ = rule['type']
    accepted = {'object': isinstance(value, dict), 'array': isinstance(value, list),
                'string': isinstance(value, str), 'number': type(value) in (int, float),
                'integer': type(value) is int, 'boolean': type(value) is bool}
    if not accepted.get(typ, False):
        raise ValueError(f'{path}: expected {typ}')
    if 'enum' in rule and value not in rule['enum']:
        raise ValueError(f'{path}: unsupported value')
    if typ == 'object':
        required = set(rule.get('required', rule['properties']))
        # The API wire schemas still require every field (provider strict JSON
        # mode), while the local validator accepts only these additive
        # omissions in stored V2 scenes/parts and older repair records. Avoid
        # normalizing in place: independent review checks exact scene identity.
        properties = rule['properties']
        if {'schema_version', 'base_scene', 'surface_overrides', 'custom_objects'} <= set(properties):
            required.discard('base_material_overrides')
            required.discard('exterior_contexts')
        if {'type', 'size', 'radius', 'points', 'vertices', 'faces', 'material_id'} <= set(properties):
            required.discard('cloth')
        if set(properties) == {'object_id', 'material_id', 'selector'}:
            required.discard('selector')
        if not required <= set(value) or set(value) - set(rule['properties']):
            missing_or_extra = (required - set(value)) | (set(value) - set(rule['properties']))
            raise ValueError(f'{path}: missing or extra fields {missing_or_extra}')
        for key, child in rule['properties'].items():
            if key in value:
                _walk(value[key], child, path + '.' + key)
    elif typ == 'array':
        if not rule.get('minItems', 0) <= len(value) <= rule.get('maxItems', 100000):
            raise ValueError(f'{path}: invalid array length')
        for i, item in enumerate(value):
            _walk(item, rule['items'], f'{path}[{i}]')
    elif typ in ('number', 'integer'):
        try:
            valid = math.isfinite(value) and rule.get('minimum', -math.inf) <= value <= rule.get('maximum', math.inf)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f'{path}: nonfinite or out of bounds')
    elif typ == 'string' and len(value) > rule.get('maxLength', 12000):
        raise ValueError(f'{path}: string too long')


def _ids(items, label):
    result = [item['id'] for item in items]
    if any(not identifier.strip() for identifier in result) or len(set(result)) != len(result):
        raise ValueError(label + ': empty or duplicate ids')
    return set(result)


def _ref(identifier, identifiers, label, allow_empty=True):
    if (identifier or not allow_empty) and identifier not in identifiers:
        raise ValueError(label + ': unknown reference ' + identifier)


def _validate_quad(quad):
    # Convex ordered source quadrilateral prevents singular perspective crops.
    cross = []
    for i in range(4):
        a, b, c = quad[i], quad[(i+1) % 4], quad[(i+2) % 4]
        cross.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
    if not all(value > 1e-8 for value in cross):
        raise ValueError('textures: quad must be convex UL UR LR LL in image coordinates')


def validate(spec):
    """Validate v1 or v2, rejecting code/paths, dangling references and unbounded data."""
    if isinstance(spec, dict) and spec.get('schema_version') == 1:
        return validate_v1(spec)
    # Runtime compatibility is separate from the strict API wire schema.
    # _walk accepts additive omissions without mutating data/provenance.
    _walk(spec, V2_SCHEMA, 'scene')
    base = validate_v1(spec['base_scene'])
    materials = _ids(spec['materials'], 'materials')
    textures = _ids(spec['textures'], 'textures')
    groups = _ids(spec['custom_objects'], 'custom_objects')
    base_ids = _ids(base['objects'], 'base_scene.objects')
    lights = _ids(spec['lights'], 'lights')
    expanded_ids = {item['id'] + ('__%d_%d' % (i, j) if item['repeat'] != [1, 1] else '')
                    for item in base['objects'] for i in range(item['repeat'][0]) for j in range(item['repeat'][1])}
    if len(expanded_ids) != sum(item['repeat'][0] * item['repeat'][1] for item in base['objects']):
        raise ValueError('expanded base object ids collide')
    if 'floor' in groups | base_ids | expanded_ids | lights:
        raise ValueError('floor is reserved for the floor support sentinel')
    if groups & (base_ids | expanded_ids) or lights & (groups | base_ids | expanded_ids):
        raise ValueError('object and light ids must be unique across the scene')
    for item in spec['materials']:
        _ref(item['texture_id'], textures, 'materials.texture_id')
    material_overrides = spec.get('base_material_overrides', [])
    override_signatures = set()
    for item in material_overrides:
        _ref(item['object_id'], expanded_ids, 'base_material_overrides.object_id', allow_empty=False)
        _ref(item['material_id'], materials, 'base_material_overrides.material_id', allow_empty=False)
        selector = item.get('selector')
        if selector is not None and not (selector['role'] or selector['material_name'] or selector['slot_index'] >= 0):
            raise ValueError('base_material_overrides.selector: provide a role, material_name or nonnegative slot_index')
        signature = (item['object_id'], item['material_id'],
                     json.dumps(selector, sort_keys=True) if selector else 'all')
        if signature in override_signatures:
            raise ValueError('Duplicate base material override')
        override_signatures.add(signature)
    for item in spec['textures']:
        _validate_quad(item['quad'])
    window_ids = {item['id'] + ('__%d_%d' % (i, j) if item['repeat'] != [1, 1] else '')
                  for item in base['objects'] if item['category'] == 'window'
                  for i in range(item['repeat'][0]) for j in range(item['repeat'][1])}
    exterior_ids = set()
    texture_quads = {item['id']: item['quad'] for item in spec['textures']}
    for context in spec.get('exterior_contexts', []):
        _ref(context['window_id'], window_ids, 'exterior_contexts.window_id', allow_empty=False)
        _ref(context['texture_id'], textures, 'exterior_contexts.texture_id', allow_empty=False)
        if context['window_id'] in exterior_ids:
            raise ValueError('exterior_contexts: duplicate window attachment')
        exterior_ids.add(context['window_id'])
        quad = texture_quads[context['texture_id']]
        area = abs(sum(quad[i][0] * quad[(i+1) % 4][1] - quad[(i+1) % 4][0] * quad[i][1]
                       for i in range(4))) / 2
        if area > .5:
            raise ValueError('exterior_contexts: crop exceeds half the photo; use only exterior content seen inside its window')
    for identifier in spec['surface_overrides'].values():
        _ref(identifier, materials, 'surface_overrides')
    material_map = {item['id']: item for item in spec['materials']}
    part_count = vertex_count = index_count = point_count = 0
    support_map = {}
    for group in spec['custom_objects']:
        _ref(group['material_id'], materials, 'custom_objects.material_id')
        _ref(group['support_id'], groups | expanded_ids | {'floor'}, 'custom_objects.support_id')
        support_map[group['id']] = group['support_id']
        for item in group['parts']:
            part_count += 1
            _ref(item['material_id'], materials, 'parts.material_id')
            effective_material = item['material_id'] or group['material_id']
            kind = item['type']
            cloths = item.get('cloth', [])
            cloth = cloths[0] if cloths else None
            if kind == 'curtain':
                if not cloth:
                    raise ValueError('curtain: cloth parameters required')
                if min(item['size']) <= cloth['thickness'] * 2:
                    raise ValueError('curtain: every size axis must exceed twice cloth thickness')
                if cloth['segments_x'] < cloth['fold_count'] * 4:
                    raise ValueError('curtain: segments_x must be at least fold_count*4')
                if (cloth['fold_depth'] + 2 * cloth['thickness'] > item['size'][1]
                        or cloth['sag'] * 2 >= item['size'][2] - 2 * cloth['thickness']):
                    raise ValueError('curtain: fold depth or sag exceeds the declared size envelope')
                if item['vertices'] or item['faces'] or item['radius']:
                    raise ValueError('curtain: vertices and faces must be empty and radius must be zero')
            elif cloths:
                raise ValueError('non-curtain: cloth parameters are not allowed')
            if kind == 'box' and min(item['size']) <= 0:
                raise ValueError('box: size must be positive on every axis')
            if kind == 'cylinder' and (item['radius'] <= 0 or item['size'][2] <= 0):
                raise ValueError('cylinder: positive radius and size.z required')
            if kind == 'sphere' and item['radius'] <= 0 and min(item['size']) <= 0:
                raise ValueError('sphere: positive radius or positive XYZ dimensions required')
            if kind == 'image_plane':
                if item['size'][0] <= 0 or item['size'][2] <= 0:
                    raise ValueError('image_plane: positive size.x and size.z required')
                if not effective_material or not material_map[effective_material]['texture_id']:
                    raise ValueError('image_plane: material must reference a photo texture')
            if kind == 'curve':
                if len(item['points']) < 2 or item['radius'] <= 0:
                    raise ValueError('curve: at least two points and positive radius required')
                if not any(sum((a-b)**2 for a,b in zip(item['points'][0], point)) > 1e-10 for point in item['points'][1:]):
                    raise ValueError('curve: coincident points')
            elif item['points']:
                raise ValueError('non-curve: points must be empty')
            if kind == 'mesh':
                if len(item['vertices']) < 3 or not item['faces']:
                    raise ValueError('mesh: vertices and faces required')
                for face in item['faces']:
                    if max(face) >= len(item['vertices']) or len(set(face)) != len(face):
                        raise ValueError('mesh: invalid face indices')
                    a = item['vertices'][face[0]]
                    # Any nonzero triangle in the polygon suffices to reject collapsed faces.
                    nonzero = False
                    for i in range(1, len(face)-1):
                        b, c = item['vertices'][face[i]], item['vertices'][face[i+1]]
                        u, v = [b[j]-a[j] for j in range(3)], [c[j]-a[j] for j in range(3)]
                        area2 = sum((u[(j+1)%3]*v[(j+2)%3] - u[(j+2)%3]*v[(j+1)%3])**2 for j in range(3))
                        nonzero = nonzero or area2 > 1e-18
                    if not nonzero:
                        raise ValueError('mesh: degenerate face')
            elif kind not in ('mesh', 'curtain') and (item['vertices'] or item['faces']):
                raise ValueError('non-mesh: vertices and faces must be empty')
            if effective_material and material_map[effective_material]['texture_id'] and kind not in ('box', 'mesh', 'image_plane', 'curtain'):
                raise ValueError('photo textures require box, mesh or image_plane UV mapping')
            vertex_count += len(item['vertices'])
            index_count += sum(len(face) for face in item['faces'])
            if kind == 'curtain':
                # Include both shell sides and the boundary rim created by
                # solidify, rather than counting only the control surface.
                vertex_count += (cloth['segments_x'] + 1) * (cloth['segments_z'] + 1) * 2
                index_count += (cloth['segments_x'] * cloth['segments_z'] * 8
                                + (cloth['segments_x'] + cloth['segments_z']) * 8)
            point_count += len(item['points'])
    if part_count > MAX_PARTS or vertex_count > MAX_VERTICES or index_count > MAX_FACE_INDICES or point_count > MAX_CURVE_POINTS:
        raise ValueError('total custom geometry exceeds scene budget')
    for node in support_map:
        visited = set()
        current = node
        while current in support_map and support_map[current]:
            if current in visited:
                raise ValueError('cyclic or self support relationship')
            visited.add(current)
            current = support_map[current]
    for light in spec['lights']:
        if sum((a-b)**2 for a,b in zip(light['position'], light['target'])) <= 1e-8:
            raise ValueError('light target coincides with light position')
    return spec


if __name__ == '__main__':
    Path(__file__).with_name('scene_schema_v2.json').write_text(json.dumps(V2_SCHEMA, indent=2) + '\n', encoding='utf-8')
