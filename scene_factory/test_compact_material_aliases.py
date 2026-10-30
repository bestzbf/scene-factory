"""Repeated base-material aliases survive both wire codecs for provenance.

Validation uses exact expanded bindings privately, while the decoded bundle
retains group aliases until the response normalizer records their expansion.
"""
import copy
import unittest

import anchored_scene
import compact_scene
from test_anchored_scene import base_object, wire as anchored_wire
from test_compact_scene_api import complete_bundle
from vision_multiview import normalize_response


class CompactMaterialAliasTests(unittest.TestCase):
    def fixture(self):
        bundle = complete_bundle()
        bundle['shared_scene']['base_scene']['objects'] = [
            base_object('desk_left', position=[-2, 1, 0], repeat=[1, 3], spacing=[0, 1.2])]
        bundle['shared_scene']['base_material_overrides'] = [{
            'object_id': 'desk_left', 'material_id': 'wood',
            'selector': {'role': 'body', 'material_name': '', 'slot_index': -1}}]
        return bundle

    def codecs(self, bundle):
        return [('typed', compact_scene, compact_scene.compact_response(bundle)),
                ('anchored', anchored_scene, anchored_wire(bundle, {
                    'desk_left': {'mode': 'floor', 'xy': [-2, 1]}}))]

    def views(self, bundle):
        return [{**{key: view[key] for key in ('id', 'width', 'height')},
                 'source_image': '/trusted/' + view['id'] + '.png'} for view in bundle['views']]

    def test_group_alias_is_private_for_validation_and_normalizer_records_exact_expansion(self):
        original = self.fixture()
        original_before = copy.deepcopy(original)
        for name, codec, raw in self.codecs(original):
            with self.subTest(codec=name):
                raw_before = copy.deepcopy(raw)
                decoded = codec.expand_response(raw)
                self.assertEqual(raw, raw_before)
                # Codec validation must not consume the normalization work.
                self.assertEqual(decoded['shared_scene']['base_material_overrides'],
                                 original['shared_scene']['base_material_overrides'])
                before_normalizing = copy.deepcopy(decoded)
                normalized, evidence = normalize_response(decoded, self.views(original), 'room')
                expected_ids = ['desk_left__0_0', 'desk_left__0_1', 'desk_left__0_2']
                actual = normalized['shared_scene']['base_material_overrides']
                self.assertEqual([item['object_id'] for item in actual], expected_ids)
                self.assertEqual([item['material_id'] for item in actual], ['wood'] * 3)
                self.assertEqual([item['selector'] for item in actual], [
                    original['shared_scene']['base_material_overrides'][0]['selector']] * 3)
                self.assertEqual(evidence['base_material_overrides'], [{
                    'index': 0, 'from': 'desk_left', 'to': expected_ids,
                    'method': 'exact repeated base group alias expanded in repeat X/Y order'}])
                self.assertEqual(decoded, before_normalizing)
                self.assertEqual(raw, raw_before)
        self.assertEqual(original, original_before)

    def test_unknown_object_or_material_reference_still_rejected_without_mutation(self):
        original = self.fixture()
        for name, codec, raw in self.codecs(original):
            for field, value in (('object_id', 'unknown_desk'), ('material_id', 'unknown_material')):
                invalid = copy.deepcopy(raw)
                invalid['shared_scene']['base_material_overrides'][0][field] = value
                before = copy.deepcopy(invalid)
                with self.subTest(codec=name, field=field), self.assertRaisesRegex(ValueError, 'unknown reference'):
                    codec.expand_response(invalid)
                self.assertEqual(invalid, before)

    def test_alias_and_exact_duplicate_binding_is_still_rejected_after_expansion(self):
        original = self.fixture()
        for name, codec, raw in self.codecs(original):
            duplicate = copy.deepcopy(raw['shared_scene']['base_material_overrides'][0])
            duplicate['object_id'] = 'desk_left__0_1'
            raw['shared_scene']['base_material_overrides'].append(duplicate)
            before = copy.deepcopy(raw)
            with self.subTest(codec=name), self.assertRaisesRegex(ValueError, 'Duplicate base material override'):
                codec.expand_response(raw)
            self.assertEqual(raw, before)


if __name__ == '__main__':
    unittest.main()
