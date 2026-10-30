"""Pure local tests for optional compact staged-detail transport."""
import copy
import json
import unittest

import compact_staged_detail as compact
import staged_analysis as staged
from test_staged_analysis import fixture_plan


def detail_for(kind):
    part = {'type': kind, 'material_id': 'wood', 'data': [0.1, 0.2, 0.3, 0, 0, 0, 0, 0, 0, 0],
            'points': [], 'vertices': [], 'faces': [], 'cloth': []}
    if kind in ('curve', 'polyline'):
        part['data'][9] = .02
        part['points'] = [[0, 0, 0], [1, 0, 0]]
    elif kind == 'mesh':
        part['vertices'] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        part['faces'] = [[0, 1, 2]]
    elif kind == 'curtain':
        part['data'][3:6] = [1, .2, 2]
        part['cloth'] = [{'fold_count': 4, 'fold_depth': .1, 'sag': .01, 'thickness': .01,
                          'segments_x': 20, 'segments_z': 8, 'seed': 1}]
    elif kind == 'box':
        part['data'][3:6] = [1, 1, 1]
    elif kind == 'image_plane':
        part['data'][3:6] = [1, 0, 2]
    elif kind == 'cylinder':
        part['data'][3:6] = [1, 1, 2]
        part['data'][9] = .5
    elif kind == 'sphere':
        part['data'][9] = .5
    return {'group_id': 'fixture', 'parts': [part]}


class CompactStagedDetailTests(unittest.TestCase):
    def test_every_part_type_roundtrips_without_geometry_or_material_loss(self):
        for kind in ('box', 'sphere', 'cylinder', 'curve', 'polyline', 'mesh', 'image_plane', 'curtain'):
            with self.subTest(kind=kind):
                native = detail_for(kind)
                wire = compact.compact_detail(native)
                self.assertEqual(compact.expand_detail(wire, 'fixture'), native)
                self.assertEqual(wire['wire_format'], compact.FORMAT)

    def test_sparse_geometry_preserves_part_order_and_indexes(self):
        native = {'group_id': 'fixture', 'parts': [
            detail_for('box')['parts'][0], detail_for('mesh')['parts'][0],
            detail_for('curve')['parts'][0], detail_for('curtain')['parts'][0]]}
        wire = compact.compact_detail(native)
        self.assertEqual([record['part_index'] for record in wire['geometry']], [1, 2, 3])
        self.assertEqual(compact.expand_detail(wire, 'fixture'), native)

    def test_missing_duplicate_or_wrong_geometry_is_rejected(self):
        wire = compact.compact_detail(detail_for('mesh'))
        with self.assertRaises(ValueError):
            compact.expand_detail({**wire, 'geometry': []}, 'fixture')
        duplicate = {**wire, 'geometry': wire['geometry'] * 2}
        with self.assertRaises(ValueError):
            compact.expand_detail(duplicate, 'fixture')
        wrong = copy.deepcopy(wire)
        wrong['geometry'][0]['points'] = [[0, 0, 0], [1, 0, 0]]
        with self.assertRaises(ValueError):
            compact.expand_detail(wrong, 'fixture')
        outside = copy.deepcopy(wire)
        outside['geometry'][0]['part_index'] = 1
        with self.assertRaises(ValueError):
            compact.expand_detail(outside, 'fixture')
        unrelated = detail_for('box')
        unrelated['parts'][0]['points'] = [[0, 0, 0], [1, 0, 0]]
        with self.assertRaises(ValueError):
            compact.compact_detail(unrelated)

    def test_schema_obeys_provider_subset_and_savings_measure_prompt_schema_response(self):
        group_id = 'fixture'
        schema = compact.detail_schema(group_id, 60)
        staged._schema_contract(schema)
        native = {'group_id': group_id, 'parts': [detail_for('box')['parts'][0] for _ in range(30)]}
        wire = compact.compact_detail(native)
        canonical_schema = staged.detail_schema(group_id, 60)
        size = lambda value: len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        plan = fixture_plan([{'id': 'view_0', 'source_image': 'photo.jpg', 'width': 64, 'height': 48}], 1)
        identity = '\nTRUSTED INPUT IDENTITY (images in exactly this order):' + json.dumps(
            {'scene_id': 'fixture_scene', 'views': [{'id': 'view_0', 'source_image': 'input:view_0',
                                                       'width': 64, 'height': 48}]}, ensure_ascii=False)
        canonical_prompt = staged.make_detail_prompt(plan, 'group_0', identity)
        sparse_prompt = compact.compact_detail_prompt(canonical_prompt)
        canonical_total = len(canonical_prompt.encode('utf-8')) + size(canonical_schema) + size(native)
        compact_total = len(sparse_prompt.encode('utf-8')) + size(schema) + size(wire)
        self.assertLess(compact_total, canonical_total)
        # A one-part output can still lose to schema overhead, so the codec is
        # opt-in and the local byte report must not claim universal savings.
        one = {'group_id': group_id, 'parts': native['parts'][:1]}
        self.assertGreater(len(sparse_prompt.encode()) + size(schema) + size(compact.compact_detail(one)),
                           len(canonical_prompt.encode()) + size(canonical_schema) + size(one))
        self.assertEqual(staged.DEFAULT_OPTIONS['detail_wire_format'], 'canonical')
        self.assertEqual(staged.normalize_options({'detail_wire_format': compact.FORMAT})['detail_wire_format'], compact.FORMAT)
        with self.assertRaises(ValueError):
            staged.normalize_options({'detail_wire_format': 'unknown_codec'})

    def test_scoped_texture_material_enum_survives_sparse_codec(self):
        allowed = ['wood']
        group_id = 'fixture'
        schema = compact.detail_schema(group_id, 60, allowed)
        material_schema = schema['properties']['parts']['items']['properties']['material_id']
        self.assertEqual(material_schema['enum'], ['', 'wood'])
        staged._schema_contract(schema)
        native = {'group_id': group_id, 'parts': [detail_for('box')['parts'][0]]}
        native['parts'][0]['material_id'] = 'wood'
        wire = compact.compact_detail(native, allowed_material_ids=allowed)
        self.assertEqual(compact.expand_detail(wire, group_id, allowed_material_ids=allowed), native)
        wrong = copy.deepcopy(wire)
        wrong['parts'][0]['material_id'] = 'unseen_photo_texture'
        with self.assertRaises(ValueError):
            compact.expand_detail(wrong, group_id, allowed_material_ids=allowed)

    def test_sparse_prompt_retains_spatial_contract_and_removes_old_array_clause(self):
        plan = {'scene_type': 'office', 'room': {'d': [5, 6, 3], 'c': [[.5] * 3] * 3},
                'cameras': [], 'materials': [], 'assets': [], 'lights': [], 'render': [0, 0, 0, 0, 0],
                'surfaces': [], 'omit': [], 'front_wall': False, 'textures': [], 'groups': [
                    {'id': 'fixture', 'p': [0, 0, 0], 'r': [0, 0, 0], 'm': '', 'support': 'floor',
                     'd': [1, 1, 1], 'confidence': .9, 'observed': True, 'instructions': 'fixture'}],
                'assumptions': [], 'base_material_overrides': [], 'exterior_contexts': []}
        prompt = staged.make_detail_prompt(plan, 'fixture', '\nTRUSTED INPUT IDENTITY: fixture')
        sparse = compact.compact_detail_prompt(prompt)
        self.assertIn('group.p + group.rotation * local point', sparse)
        self.assertIn('within room X[-width/2,+width/2]', sparse)
        self.assertIn('Positions+-50m', sparse)
        self.assertNotIn('Explicit unused zeros and arrays[]', sparse)


if __name__ == '__main__':
    unittest.main()
