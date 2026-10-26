"""Data contract regression tests; no Blender or online service required."""
import copy
import json
from pathlib import Path
import unittest
from spec_v2 import SCHEMA, material, part, upgrade_v1, validate

ROOT = Path(__file__).parent


def fixture():
    base = json.loads((ROOT / 'examples/living_room.json').read_text())
    spec = upgrade_v1(base)
    spec['materials'] = [material('wood', [.5, .3, .15]), material('photo', [1, 1, 1], texture_id='sign')]
    spec['textures'] = [{'id': 'sign', 'quad': [[.1,.1], [.3,.1], [.3,.2], [.1,.2]]}]
    spec['custom_objects'] = [{'id': 'custom_shelf', 'position': [0, 2, 0],
        'rotation_degrees': [0, 0, 0], 'material_id': 'wood', 'confidence': .8,
        'observed': True, 'support_id': '', 'parts': [part(size=[1,.3,.04], position=[0,0,1])]}]
    spec['lights'] = [{'id': 'spot', 'type': 'AREA', 'position': [0,2,2.5], 'target': [0,2,0],
        'color': [1,.9,.8], 'energy': 100, 'size': .5, 'size_y': .3,
        'shape': 'RECTANGLE', 'spread_degrees': 100}]
    return spec


class ContractTests(unittest.TestCase):
    def test_v1_and_upgrade_preserve_original(self):
        old = json.loads((ROOT / 'examples/living_room.json').read_text())
        before = copy.deepcopy(old)
        self.assertIs(validate(old), old)
        new = upgrade_v1(old)
        new['base_scene']['seed'] = 24
        self.assertEqual(old, before)
        self.assertEqual(validate(fixture())['schema_version'], 2)

    def test_shared_schema_is_strict_and_all_fields_required(self):
        def inspect(rule):
            if rule.get('type') == 'object':
                self.assertEqual(set(rule['required']), set(rule['properties']))
                self.assertIs(rule['additionalProperties'], False)
                for child in rule['properties'].values(): inspect(child)
            elif rule.get('type') == 'array': inspect(rule['items'])
        inspect(SCHEMA)
        for field in ('python', 'path', 'url'):
            bad = fixture(); bad['custom_objects'][0][field] = 'untrusted'
            with self.assertRaises(ValueError): validate(bad)

    def test_numeric_finite_bounds_and_types(self):
        for number in (float('nan'), float('inf'), True, '0', 500):
            bad = fixture(); bad['custom_objects'][0]['position'][0] = number
            with self.subTest(value=number), self.assertRaises(ValueError): validate(bad)
        bad = fixture(); bad['custom_objects'][0]['observed'] = 1
        with self.assertRaises(ValueError): validate(bad)

    def test_duplicate_and_empty_identifiers(self):
        for array in ('materials','textures','lights','custom_objects'):
            bad = fixture(); bad[array].append(copy.deepcopy(bad[array][0]))
            with self.subTest(array=array), self.assertRaises(ValueError): validate(bad)
            bad = fixture(); bad[array][0]['id'] = ' '
            with self.assertRaises(ValueError): validate(bad)
        bad = fixture(); bad['custom_objects'][0]['id'] = bad['base_scene']['objects'][0]['id']
        with self.assertRaises(ValueError): validate(bad)

    def test_unknown_refs(self):
        for location, field in [('materials', 'texture_id'), ('custom_objects', 'material_id'), ('custom_objects', 'support_id')]:
            bad = fixture(); bad[location][0][field] = 'missing'
            with self.subTest(location=location, field=field), self.assertRaises(ValueError): validate(bad)
        bad = fixture(); bad['surface_overrides']['floor'] = 'missing'
        with self.assertRaises(ValueError): validate(bad)
        bad = fixture(); bad['custom_objects'][0]['parts'][0]['material_id'] = 'missing'
        with self.assertRaises(ValueError): validate(bad)

    def test_no_cyclic_supports(self):
        bad = fixture(); group = bad['custom_objects'][0]; group['support_id'] = group['id']
        with self.assertRaises(ValueError): validate(bad)
        group['support_id'] = 'second'; other = copy.deepcopy(group)
        other['id'] = 'second'; other['support_id'] = group['id']; bad['custom_objects'].append(other)
        with self.assertRaises(ValueError): validate(bad)
        other['support_id'] = bad['base_scene']['objects'][0]['id']
        self.assertIs(validate(bad), bad)

    def test_expanded_base_support_ids_and_floor(self):
        spec = fixture(); first = spec['base_scene']['objects'][0]
        first['repeat'] = [2,1]; first['spacing'] = [1,0]
        spec['custom_objects'][0]['support_id'] = first['id'] + '__1_0'
        validate(spec)
        spec['custom_objects'][0]['support_id'] = 'floor'
        validate(spec)
        spec['custom_objects'][0]['support_id'] = first['id']
        with self.assertRaises(ValueError): validate(spec)
        spec['custom_objects'][0]['support_id'] = ''
        collision = copy.deepcopy(first); collision['id'] += '__1_0'; collision['repeat'] = [1,1]
        spec['base_scene']['objects'].append(collision)
        with self.assertRaises(ValueError): validate(spec)

    def test_degenerate_lights_and_photo_quads(self):
        bad = fixture(); bad['lights'][0]['target'] = bad['lights'][0]['position'][:]
        with self.assertRaises(ValueError): validate(bad)
        for quad in ([[.1,.1]]*4, [[0,0],[1,1],[1,0],[0,1]], [[0,0],[0,1],[1,1],[1,0]]):
            bad = fixture(); bad['textures'][0]['quad'] = quad
            with self.assertRaises(ValueError): validate(bad)

    def test_geometry_types(self):
        valid = [part('box', size=[1,.1,1]), part('cylinder', radius=.1, size=[0,0,1]),
                 part('sphere', radius=.1), part('sphere', size=[.2,.3,.4]), part('curve', radius=.01, points=[[0,0,0],[1,0,0]]),
                 part('mesh', vertices=[[0,0,0],[1,0,0],[0,0,1]], faces=[[0,1,2]]),
                 part('image_plane', size=[1,0,1], material_id='photo')]
        spec = fixture(); spec['custom_objects'][0]['parts'] = valid
        self.assertIs(validate(spec), spec)
        bad_parts = [part(), part('cylinder', radius=.1), part('sphere'),
                     part('curve', radius=.1, points=[[0,0,0],[0,0,0]]),
                     part('image_plane', size=[1,0,1]), part('mesh', vertices=[[0,0,0],[1,0,0],[0,0,1]], faces=[[0,1,3]]),
                     part('mesh', vertices=[[0,0,0],[1,0,0],[2,0,0]], faces=[[0,1,2]]),
                     part('box', size=[1,1,1], points=[[0,0,0]])]
        for item in bad_parts:
            bad = fixture(); bad['custom_objects'][0]['parts'] = [item]
            with self.subTest(kind=item['type']), self.assertRaises(ValueError): validate(bad)

    def test_aggregate_geometry_budget(self):
        bad = fixture(); group = bad['custom_objects'][0]
        group['parts'] *= 100
        bad['custom_objects'] = [{**copy.deepcopy(group), 'id': f'group_{i}'} for i in range(9)]
        with self.assertRaisesRegex(ValueError, 'budget'): validate(bad)
        bad = fixture(); bad['custom_objects'][0]['parts'] = [part('mesh', vertices=[[0,0,0]]*2049, faces=[[0,1,2]])]
        with self.assertRaises(ValueError): validate(bad)
        # v1 expansion ceiling still applies to the nested base scene.
        bad = fixture(); first = bad['base_scene']['objects'][0]
        bad['base_scene']['objects'] = [{**copy.deepcopy(first), 'id': str(i), 'repeat': [12,12]} for i in range(5)]
        with self.assertRaises(ValueError): validate(bad)

    def test_generated_examples_are_distinct_and_valid(self):
        paths = list((ROOT / 'examples_v2').glob('*.json'))
        self.assertEqual(len(paths), 3)
        for path in paths:
            spec = validate(json.loads(path.read_text()))
            self.assertTrue(spec['custom_objects'])
            self.assertIn(spec['scene_type'], ('bedroom','living_room','shop'))


if __name__ == '__main__':
    unittest.main()
