"""Material binding references and bounded textile data, including old scenes."""
import copy
import json
import unittest
from pathlib import Path

from model_api import _schema_contract
from procedural_cloth import curtain_mesh
from spec_v2 import SCHEMA, part, validate
from test_spec_v2 import fixture


class MaterialClothContractTests(unittest.TestCase):
    def test_wire_schema_remains_strict(self):
        _schema_contract(SCHEMA)

    def test_legacy_validation_preserves_values_and_new_override_checks(self):
        spec = fixture()
        for group in spec['custom_objects']:
            for item in group['parts']:
                item.pop('cloth', None)
        before = copy.deepcopy(spec)
        self.assertIs(validate(spec), spec)
        self.assertEqual(spec, before)
        identifier = spec['base_scene']['objects'][0]['id']
        spec['base_material_overrides'] = [{'object_id': identifier, 'material_id': 'wood'}]
        validate(spec)
        self.assertNotIn('selector', spec['base_material_overrides'][0])
        for field in ('object_id', 'material_id'):
            bad = copy.deepcopy(spec)
            bad['base_material_overrides'][0][field] = 'does_not_exist'
            with self.assertRaisesRegex(ValueError, 'unknown reference'):
                validate(bad)

    def test_repeat_binding_requires_exact_expanded_id(self):
        spec = fixture()
        entry = spec['base_scene']['objects'][0]
        entry['repeat'] = [2, 1]
        entry['spacing'] = [1, 0]
        spec['base_material_overrides'] = [{'object_id': entry['id'], 'material_id': 'wood'}]
        with self.assertRaisesRegex(ValueError, 'unknown reference'):
            validate(spec)
        spec['base_material_overrides'][0]['object_id'] += '__1_0'
        validate(spec)
        spec['base_material_overrides'][0]['selector'] = {'role': 'body', 'material_name': '', 'slot_index': -1}
        validate(spec)
        spec['base_material_overrides'][0]['selector']['slot_index'] = 64
        with self.assertRaises(ValueError):
            validate(spec)

    def test_invalid_additive_field_types_raise_validation_error(self):
        for value in (None, {}, 'binding'):
            spec = fixture(); spec['base_material_overrides'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(spec)
        for value in (None, {}, 'cloth'):
            spec = fixture(); spec['custom_objects'][0]['parts'][0]['cloth'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(spec)

    def test_curtain_is_deterministic_folded_continuous_and_bounded(self):
        item = part('curtain', size=[1.2, .18, 2])
        spec = fixture()
        spec['custom_objects'][0]['parts'] = [item]
        validate(spec)
        vertices, faces, uv = curtain_mesh(item['size'], item['cloth'][0])
        self.assertEqual((vertices, faces, uv), curtain_mesh(item['size'], item['cloth'][0]))
        self.assertEqual(len(vertices), 65 * 25)
        self.assertEqual(len(faces), 64 * 24)
        self.assertGreater(max(p[1] for p in vertices) - min(p[1] for p in vertices), .09)
        for axis, dimension in enumerate(item['size']):
            self.assertLessEqual(max(abs(p[axis]) for p in vertices), dimension / 2)
        self.assertEqual(uv[0], [0, 1])
        self.assertEqual(uv[-1], [1, 0])
        other = copy.deepcopy(item['cloth'][0]); other['seed'] += 1
        self.assertNotEqual(vertices, curtain_mesh(item['size'], other)[0])

    def test_cloth_rejects_hidden_mesh_data_and_oversized_parameters(self):
        for change in ({'fold_depth': .5}, {'segments_x': 8}, {'sag': 2}, {'thickness': .1}):
            spec = fixture(); item = part('curtain', size=[1.2, .18, 2])
            item['cloth'][0].update(change); spec['custom_objects'][0]['parts'] = [item]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(spec)
        for fields in ({'cloth': []}, {'vertices': [[0, 0, 0]]}, {'radius': .01}):
            spec = fixture(); spec['custom_objects'][0]['parts'] = [part('curtain', size=[1.2, .18, 2], **fields)]
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate(spec)
        spec = fixture(); spec['custom_objects'][0]['parts'][0]['cloth'] = part('curtain')['cloth']
        with self.assertRaisesRegex(ValueError, 'non-curtain'):
            validate(spec)

    def test_generated_geometry_includes_thickness_in_scene_budget(self):
        spec = fixture(); item = part('curtain', size=[1.2, .18, 2])
        item['cloth'][0].update(segments_x=128, segments_z=64)
        spec['custom_objects'][0]['parts'] = [copy.deepcopy(item) for _ in range(7)]
        with self.assertRaisesRegex(ValueError, 'budget'):
            validate(spec)


if __name__ == '__main__':
    unittest.main()
