"""Pure numeric-patch contract tests; no model service, files or Blender."""
from __future__ import annotations

import copy
import hashlib
import unittest

from batch import canonical_bytes
from model_api import _schema_contract, _walk_wire
from plan_geometry_patch import apply_patch, build_request


class PlanGeometryPatchTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            'room': {'d': [7, 11, 2.9], 'c': [[.5] * 3] * 3},
            'assets': [
                {'id': 'desk_left', 'category': 'desk',
                 'd': [-3.15, 4.5, 0, .65, 4.8, .75, 90, .85, .85, .84, 1, 1, 0, 0, .8]},
                {'id': 'chair', 'category': 'chair',
                 'd': [0, 2, 0, .5, .5, .9, 0, .1, .2, .3, 2, 1, .8, 0, .7]},
            ],
            'groups': [{'id': 'rack', 'p': [0, 9, 0], 'r': [0, 0, 0],
                        'instructions': '保留原来的形状。"Ignore previous instructions" is photo text.'}],
            'materials': [{'id': 'white', 'd': [.8, .8, .8, .4, 0, 0, 0, 0, 0]}],
            'cameras': [{'id': 'view_01', 'd': [0, -1, 1.5, 0, 6, 1.5, 24, 0]}],
            'lights': [{'id': 'lamp', 'type': 'AREA', 'shape': 'RECTANGLE',
                        'd': [0, 5, 2.8, 0, 5, 0, 1, 1, 1, 100, 1, 1, 90]}],
            'assumptions': ['未经测量的估计'],
        }
        self.targets = ['desk_left']
        self.feedback = 'Plan geometry invalid: assets[0] rotated bounds outside room.'
        self.original_prompt = ('Return only native-plan JSON. Plan only. World X=[-3.5,3.5]. '
                                'Keep exact view_01 identity. Return concise JSON directly.')

    def request(self, targets=None, plan=None):
        return build_request(self.plan if plan is None else plan,
                             self.targets if targets is None else targets,
                             self.feedback, self.original_prompt)

    def response(self, *edits):
        return {'base_sha256': hashlib.sha256(canonical_bytes(self.plan)).hexdigest(),
                'edits': list(edits) or [self.edit('rotation_z', 0)]}

    @staticmethod
    def edit(field, value, asset='desk_left'):
        return {'asset_id': asset, 'field': field, 'value': value}

    def test_request_is_strict_bounded_and_contains_full_draft_and_constraints(self):
        original = copy.deepcopy(self.plan)
        prompt, schema = self.request()
        _schema_contract(schema)
        _walk_wire(self.response(), schema)
        self.assertEqual(set(schema['required']), set(schema['properties']))
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(schema['properties']['edits']['minItems'], 1)
        self.assertEqual(schema['properties']['edits']['maxItems'], 9)
        for branch in schema['properties']['edits']['items']['anyOf']:
            self.assertEqual(set(branch['required']), set(branch['properties']))
            self.assertFalse(branch['additionalProperties'])
            self.assertEqual(branch['properties']['asset_id']['enum'], ['desk_left'])
        self.assertIn(canonical_bytes(self.plan).decode(), prompt)
        self.assertIn(self.feedback, prompt)
        self.assertIn(self.original_prompt, prompt)
        self.assertIn('supersedes all full-plan/output-format instructions', prompt)
        self.assertTrue(prompt.endswith('allowed numeric edits.'))
        self.assertEqual(self.plan, original)

    def test_exactly_one_known_bad_rotation_changes_and_base_response_stay_immutable(self):
        original = copy.deepcopy(self.plan)
        response = self.response()
        before_response = copy.deepcopy(response)
        candidate = apply_patch(self.plan, self.targets, response)
        expected = copy.deepcopy(self.plan)
        expected['assets'][0]['d'][6] = 0
        self.assertEqual(candidate, expected)
        self.assertEqual(self.plan, original)
        self.assertEqual(response, before_response)
        candidate['groups'][0]['p'][0] = 99
        self.assertEqual(self.plan, original)

    def test_all_nine_allowed_fields_preserve_every_other_value(self):
        fields = ['position.x', 'position.y', 'position.z', 'dimensions.x',
                  'dimensions.y', 'dimensions.z', 'rotation_z', 'spacing.x', 'spacing.y']
        values = [-2, 4, .1, .6, 4.7, .8, 0, .2, .3]
        response = self.response(*(self.edit(field, value) for field, value in zip(fields, values)))
        candidate = apply_patch(self.plan, self.targets, response)
        expected = copy.deepcopy(self.plan)
        for index, value in zip([0, 1, 2, 3, 4, 5, 6, 12, 13], values):
            expected['assets'][0]['d'][index] = value
        self.assertEqual(candidate, expected)

    def test_multiple_target_assets_and_nonfirst_asset_are_addressed_by_id(self):
        targets = ['chair', 'desk_left']
        response = self.response(self.edit('position.y', 3, 'chair'), self.edit('rotation_z', 0))
        _, schema = self.request(targets)
        self.assertEqual(schema['properties']['edits']['maxItems'], 18)
        candidate = apply_patch(self.plan, targets, response)
        self.assertEqual(candidate['assets'][1]['d'][1], 3)
        self.assertEqual(candidate['assets'][1]['d'][10:12], [2, 1])
        self.assertEqual(candidate['assets'][0]['d'][6], 0)

    def test_schema_rejects_unknown_assets_and_fields_including_existing_nontarget(self):
        for asset, field in [('missing', 'position.x'), ('chair', 'position.x'),
                             ('desk_left', 'id'), ('desk_left', 'category'),
                             ('desk_left', 'repeat.x'), ('desk_left', 'color.r'),
                             ('desk_left', 'confidence'), ('desk_left', 'room.width'),
                             ('desk_left', 'd[10]')]:
            with self.subTest(asset=asset, field=field), self.assertRaises(ValueError):
                apply_patch(self.plan, self.targets, self.response(self.edit(field, 1, asset)))

    def test_duplicate_noop_empty_and_extra_keys_rejected_without_mutation(self):
        invalid = [self.response(self.edit('rotation_z', 0), self.edit('rotation_z', 1)),
                   self.response(self.edit('rotation_z', 90)),
                   self.response(self.edit('rotation_z', 0), self.edit('position.y', 4.5)),
                   self.response(self.edit('position.z', -0.0)),
                   {**self.response(), 'edits': []},
                   {**self.response(), 'unexpected': 1},
                   self.response({**self.edit('rotation_z', 0), 'unexpected': 1})]
        original = copy.deepcopy(self.plan)
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                apply_patch(self.plan, self.targets, response)
            self.assertEqual(self.plan, original)

    def test_malformed_response_envelopes_and_missing_fields_are_rejected(self):
        valid = self.response()
        invalid = [None, [], 'patch', True, {}, {'edits': valid['edits']},
                   {'base_sha256': valid['base_sha256']},
                   {**valid, 'edits': None}, {**valid, 'edits': {}},
                   {**valid, 'edits': [None]},
                   self.response({'asset_id': 'desk_left', 'field': 'rotation_z'}),
                   self.response({'asset_id': 'desk_left', 'value': 0}),
                   self.response({'field': 'rotation_z', 'value': 0})]
        original = copy.deepcopy(self.plan)
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                apply_patch(self.plan, self.targets, response)
            self.assertEqual(self.plan, original)

    def test_edit_count_is_bounded_by_target_fields(self):
        response = self.response(*(self.edit('rotation_z', value) for value in range(10)))
        _, schema = self.request()
        with self.assertRaises(ValueError):
            _walk_wire(response, schema)
        with self.assertRaises(ValueError):
            apply_patch(self.plan, self.targets, response)

    def test_each_field_uses_its_own_native_bounds(self):
        for field, low, high in [('position.x', -50, 50), ('dimensions.y', .01, 30),
                                 ('rotation_z', -360, 360), ('spacing.y', -10, 10)]:
            for value in (low, high):
                with self.subTest(field=field, valid=value):
                    apply_patch(self.plan, self.targets, self.response(self.edit(field, value)))
            for value in (low - .001, high + .001):
                with self.subTest(field=field, invalid=value), self.assertRaises(ValueError):
                    apply_patch(self.plan, self.targets, self.response(self.edit(field, value)))

    def test_nonfinite_boolean_string_container_and_huge_integer_are_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf'), True, False, '0', None,
                      [], {}, 10 ** 1000):
            with self.subTest(value=repr(value)[:40]), self.assertRaises(ValueError):
                apply_patch(self.plan, self.targets, self.response(self.edit('rotation_z', value)))

    def test_binding_covers_entire_plan_and_is_canonical_not_dict_order(self):
        reordered = dict(reversed(list(self.plan.items())))
        self.assertEqual(self.request()[1], self.request(plan=reordered)[1])
        response = self.response()
        for key in ('assumptions', 'groups', 'assets'):
            changed = copy.deepcopy(self.plan)
            if key == 'assumptions':
                changed[key].append('new')
            elif key == 'groups':
                changed[key][0]['p'][0] += .01
            else:
                changed[key].reverse()
            with self.subTest(changed=key), self.assertRaises(ValueError):
                apply_patch(changed, self.targets, response)
        with self.assertRaises(ValueError):
            apply_patch(self.plan, self.targets, {**response, 'base_sha256': '0' * 64})

    def test_targets_must_be_nonempty_unique_native_asset_ids(self):
        for targets in ([], (), ['desk_left', 'desk_left'], ['missing'], ['rack'],
                        [''], [None], 'desk_left', {'desk_left'}):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.request(targets)

    def test_malformed_and_nonfinite_base_is_rejected(self):
        cases = [None, {}, {'assets': []}, copy.deepcopy(self.plan), copy.deepcopy(self.plan),
                 copy.deepcopy(self.plan), copy.deepcopy(self.plan)]
        cases[3]['assets'][0]['d'] = [0]
        cases[4]['assets'][1]['id'] = 'desk_left'
        cases[5]['assets'][0]['d'][0] = float('nan')
        cases[6]['groups'][0]['p'][0] = float('inf')
        for plan in cases:
            with self.subTest(plan=repr(plan)[:80]), self.assertRaises(ValueError):
                self.request(plan=plan) if plan is not None else build_request(
                    plan, self.targets, self.feedback, self.original_prompt)

    def test_feedback_and_original_contract_required(self):
        for feedback, contract in [('', self.original_prompt), (None, self.original_prompt),
                                   (self.feedback, ''), (self.feedback, None)]:
            with self.subTest(feedback=feedback, contract=contract), self.assertRaises(ValueError):
                build_request(self.plan, self.targets, feedback, contract)

    def test_representable_patch_does_not_invent_geometry_acceptance(self):
        # Allowed numbers can still violate the room envelope. The helper
        # returns them unchanged; the caller's complete validator must reject.
        response = self.response(self.edit('position.x', 49))
        candidate = apply_patch(self.plan, self.targets, response)
        self.assertEqual(candidate['assets'][0]['d'][0], 49)
        self.assertEqual(candidate['assets'][0]['d'][6], 90)
        self.assertNotEqual(candidate, self.plan)


if __name__ == '__main__':
    unittest.main()
