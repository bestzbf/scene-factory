"""Observed-slot prechecks independent of Blender and external model services."""
import copy
import unittest

from repair_material_validation import validate_candidate_material_bindings as check


class RepairMaterialValidationTests(unittest.TestCase):
    def setUp(self):
        self.before = {'scene_id': 'room', 'shared_scene': {'scene_type': 'other',
            'base_scene': {'seed': 7, 'scene_type': 'other', 'objects': [
                {'id': 'window', 'category': 'window', 'position': [0, 5, 1],
                 'rotation_z_deg': 0, 'dimensions': [2, .2, 1.5], 'color': [.8, .8, .8],
                 'repeat': [1, 1], 'spacing': [0, 0], 'confidence': .9}]},
            'base_material_overrides': []}}
        self.inventory = {'schema_version': 1, 'status': 'passed', 'available_slots': [
            {'object_id': 'window', 'object_name': 'FrameHorizontal', 'slot_index': 0,
             'source_material': 'WarmWhite', 'source_material_key': 'WarmWhite', 'source_material_id': '', 'role': 'body'},
            {'object_id': 'window', 'object_name': 'Mullion', 'slot_index': 0,
             'source_material': 'Steel.001', 'source_material_key': 'Steel', 'source_material_id': 'steel_id', 'role': 'metal'},
            {'object_id': 'window', 'object_name': 'Transom', 'slot_index': 1,
             'source_material': 'Steel.001', 'source_material_key': 'Steel', 'source_material_id': 'steel_id', 'role': 'metal'},
            {'object_id': 'window', 'object_name': 'Panel', 'slot_index': 0,
             'source_material': 'WindowGlass', 'source_material_key': 'WindowGlass', 'source_material_id': '', 'role': 'glass'}]}

    def candidate(self, name='', role='all', slot=-1, target='new_material', object_id='window'):
        result = copy.deepcopy(self.before)
        result['shared_scene']['base_material_overrides'] = [{'object_id': object_id, 'material_id': target,
            'selector': {'role': role, 'material_name': name, 'slot_index': slot}}]
        return result

    def test_real_window_component_names_are_not_material_names(self):
        for name in ('Mullion', 'Transom'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'object_name is a mesh/component label'):
                check(self.before, self.candidate(name), self.inventory)

    def test_source_exact_name_stable_key_id_and_role_only_all_match(self):
        for name, role, slot in [('Steel.001', 'metal', -1), ('Steel', 'metal', -1),
                                 ('steel_id', 'all', -1), ('', 'glass', -1), ('', '', 1)]:
            with self.subTest(name=name, role=role, slot=slot):
                self.assertIsNone(check(self.before, self.candidate(name, role, slot), self.inventory))
        for name, role, slot in [('Steel', 'body', -1), ('Steel', 'metal', 2), ('steel', 'all', -1)]:
            with self.subTest(name=name, role=role, slot=slot), self.assertRaisesRegex(ValueError, 'matches no observed'):
                check(self.before, self.candidate(name, role, slot), self.inventory)

    def test_conflicting_overlap_rejected_but_identical_target_is_consistent(self):
        candidate = self.candidate(role='metal')
        candidate['shared_scene']['base_material_overrides'].append(
            {'object_id': 'window', 'material_id': 'other_material',
             'selector': {'role': 'all', 'material_name': 'Steel', 'slot_index': 0}})
        with self.assertRaisesRegex(ValueError, 'different target materials'):
            check(self.before, candidate, self.inventory)
        candidate['shared_scene']['base_material_overrides'][1]['material_id'] = 'new_material'
        self.assertIsNone(check(self.before, candidate, self.inventory))
        candidate['shared_scene']['base_material_overrides'][1]['material_id'] = 'other_material'
        candidate['shared_scene']['base_material_overrides'][1]['selector']['material_name'] = 'WarmWhite'
        self.assertIsNone(check(self.before, candidate, self.inventory))

    def test_repeated_instance_ids_are_expanded_exactly_without_prefix_guessing(self):
        self.before['shared_scene']['base_scene']['objects'][0]['repeat'] = [2, 1]
        self.inventory['available_slots'] = [{**slot, 'object_id': 'window__1_0'} for slot in self.inventory['available_slots']]
        self.assertIsNone(check(self.before, self.candidate('Steel', object_id='window__1_0'), self.inventory))
        with self.assertRaises(ValueError):
            check(self.before, self.candidate('Mullion', object_id='window__1_0'), self.inventory)
        # Another actual expanded instance has no visible slots in this report.
        with self.assertRaises(ValueError):
            check(self.before, self.candidate('Steel', object_id='window__0_0'), self.inventory)

    def test_geometry_material_source_changes_skip_stale_inventory_but_pose_is_checked(self):
        changes = {'category': 'desk', 'dimensions': [2.1, .2, 1.5], 'color': [.2, .2, .2],
                   'repeat': [2, 1], 'spacing': [1, 0], 'asset_id': 'new_variant',
                   'rotation_z_deg': 20, 'position': [1, 4, 0]}
        for key, value in changes.items():
            candidate = self.candidate('Mullion')
            candidate['shared_scene']['base_scene']['objects'][0][key] = value
            with self.subTest(key=key):
                self.assertIsNone(check(self.before, candidate, self.inventory))
        for key, value in [('position', [1, 4, 1]), ('confidence', .5)]:
            candidate = self.candidate('Mullion')
            candidate['shared_scene']['base_scene']['objects'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check(self.before, candidate, self.inventory)
        candidate = self.candidate('Mullion')
        candidate['shared_scene']['base_scene']['seed'] += 1
        self.assertIsNone(check(self.before, candidate, self.inventory))

    def test_library_diversity_reselection_and_floor_eligibility_skip_stale_inventory(self):
        external = [{'object_id': 'window', 'asset_id': 'library_window', 'material_variant_id': 'white'}]
        candidate = self.candidate('Mullion')
        candidate['shared_scene']['base_scene']['objects'][0]['position'][0] = .5
        with self.assertRaises(ValueError):
            check(self.before, candidate, self.inventory, external)
        candidate['shared_scene']['base_scene']['objects'][0]['position'][2] = 0
        self.assertIsNone(check(self.before, candidate, self.inventory, external))
        candidate = self.candidate('Mullion')
        candidate['shared_scene']['base_scene']['objects'].append(
            {**copy.deepcopy(self.before['shared_scene']['base_scene']['objects'][0]), 'id': 'another'})
        self.assertIsNone(check(self.before, candidate, self.inventory, external))
        # Changes elsewhere must not disable checks for a procedural instance.
        with self.assertRaises(ValueError):
            check(self.before, candidate, self.inventory)

    def test_feedback_is_bounded_fixed_and_never_echoes_untrusted_names(self):
        candidate = self.candidate('credential-shaped-secret-' + 'x'*5000)
        inventory = copy.deepcopy(self.inventory)
        inventory['available_slots'][0]['source_material_key'] = 'private-inventory-value'
        old = copy.deepcopy((self.before, candidate, inventory))
        with self.assertRaises(ValueError) as raised:
            check(self.before, candidate, inventory)
        message = str(raised.exception)
        self.assertLess(len(message), 650)
        self.assertNotIn('credential-shaped-secret', message)
        self.assertNotIn('private-inventory-value', message)
        self.assertEqual((self.before, candidate, inventory), old)

    def test_optional_missing_or_malformed_inventory_is_not_treated_as_proof(self):
        candidate = self.candidate('Mullion')
        self.assertIsNone(check(self.before, candidate, {}))
        malformed = copy.deepcopy(self.inventory)
        del malformed['available_slots'][0]['slot_index']
        self.assertIsNone(check(self.before, candidate, malformed))
        self.assertIsNone(check(self.before, candidate, self.inventory, external_assets={'invalid': True}))


if __name__ == '__main__':
    unittest.main()
