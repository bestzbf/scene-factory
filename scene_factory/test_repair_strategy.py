"""Focused repair must preserve global context and reject unseen-part edits."""
import copy
import unittest

from automatic_repair import empty_edits
from repair_strategy import repair_context, validate_edit_scope
from test_multiview import fixture


class RepairStrategyTests(unittest.TestCase):
    def setUp(self):
        self.bundle = fixture()
        group = self.bundle['shared_scene']['custom_objects'][0]
        group['support_id'] = ''
        extra = copy.deepcopy(group)
        extra['id'] = 'unrelated'
        self.bundle['shared_scene']['custom_objects'] = [group, extra]
        self.group = group
        self.report = {'issues': [
            {'severity': 'major', 'dimension': 'geometry', 'object_ids': [group['id']]},
            {'severity': 'minor', 'dimension': 'geometry', 'object_ids': [group['id']]}]}

    def test_prioritizes_geometry_and_preserves_global_identity_without_mutation(self):
        before = copy.deepcopy(self.bundle)
        context, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['issue_indices'], [0, 1])
        self.assertEqual(focus['inventory_only_group_ids'], ['unrelated'])
        self.assertNotIn('parts', context['shared_scene']['custom_objects'][1])
        self.assertEqual(context['shared_scene']['custom_objects'][0]['parts'], self.group['parts'])
        self.assertEqual(context['shared_scene']['materials'], self.bundle['shared_scene']['materials'])
        self.assertEqual(context['shared_scene']['base_scene'], self.bundle['shared_scene']['base_scene'])
        self.assertEqual(self.bundle, before)

    def test_blocking_issue_first_and_unlocalized_issue_keeps_full_geometry(self):
        self.report['issues'] = [
            {'severity': 'blocking', 'dimension': 'geometry', 'object_ids': []},
            {'severity': 'major', 'dimension': 'geometry', 'object_ids': [self.group['id']]},
        ]
        context, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['issue_indices'], [0])
        self.assertEqual(focus['inventory_only_group_ids'], [])
        self.assertTrue(all('parts' in g for g in context['shared_scene']['custom_objects']))

    def test_support_dependencies_retain_details(self):
        self.bundle['shared_scene']['custom_objects'][1]['support_id'] = self.group['id']
        _, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['inventory_only_group_ids'], [])

    def _add_supported_group(self, identifier, support_id):
        group = copy.deepcopy(self.group)
        group.update(id=identifier, support_id=support_id, material_id='cream_fabric')
        self.bundle['shared_scene']['custom_objects'].append(group)
        return group

    def _repeat_base_bed(self):
        bed = self.bundle['shared_scene']['base_scene']['objects'][0]
        self.assertEqual(bed['id'], 'bed')
        bed.update(repeat=[2, 1], spacing=[2.2, 0], position=[-1.1, 3.8, 0],
                   dimensions=[1, 2.1, .68])

    def test_base_repeat_family_reference_retains_supported_group_chain(self):
        self._repeat_base_bed()
        self._add_supported_group('on_second_bed', 'bed__1_0')
        self._add_supported_group('on_bed_group', 'on_second_bed')
        self.report['issues'] = [{'severity': 'major', 'dimension': 'layout',
                                 'object_ids': [self.group['id'], 'bed']}]
        before = copy.deepcopy(self.bundle)
        context, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['detailed_group_ids'],
                         sorted([self.group['id'], 'on_second_bed', 'on_bed_group']))
        self.assertEqual(focus['inventory_only_group_ids'], ['unrelated'])
        self.assertTrue(all('parts' in group for group in
                            context['shared_scene']['custom_objects']
                            if group['id'] != 'unrelated'))
        self.assertEqual(self.bundle, before)

    def test_base_instance_reference_retains_other_affected_repeat_instances(self):
        # Base edits upsert the complete repeat family, even if the issue names
        # one expanded instance. Its siblings' support geometry must be visible.
        self._repeat_base_bed()
        self._add_supported_group('on_first_bed', 'bed__0_0')
        self._add_supported_group('on_second_bed', 'bed__1_0')
        self.report['issues'] = [{'severity': 'major', 'dimension': 'layout',
                                 'object_ids': [self.group['id'], 'bed__1_0']}]
        _, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['detailed_group_ids'],
                         sorted([self.group['id'], 'on_first_bed', 'on_second_bed']))
        self.assertEqual(focus['inventory_only_group_ids'], ['unrelated'])

    def test_custom_support_chain_traverses_base_before_other_dependents(self):
        self._repeat_base_bed()
        self.group['support_id'] = 'bed__0_0'
        self._add_supported_group('on_second_bed', 'bed__1_0')
        self._add_supported_group('on_bed_group', 'on_second_bed')
        _, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['detailed_group_ids'],
                         sorted([self.group['id'], 'on_second_bed', 'on_bed_group']))
        self.assertEqual(focus['inventory_only_group_ids'], ['unrelated'])

    def test_shared_material_includes_base_override_support_dependencies(self):
        self._repeat_base_bed()
        # Keep this a local object-material case; oak is the fixture's floor
        # material by default and architectural material issues need full context.
        scene = self.bundle['shared_scene']
        scene['surface_overrides'] = dict.fromkeys(scene['surface_overrides'], '')
        self.group['material_id'] = 'oak'
        self._add_supported_group('on_second_bed', 'bed__1_0')
        self._add_supported_group('on_bed_group', 'on_second_bed')
        self.bundle['shared_scene']['base_material_overrides'] = [{
            'object_id': 'bed__1_0', 'material_id': 'oak',
            'selector': {'role': 'body', 'material_name': '', 'slot_index': -1}}]
        self.report['issues'] = [{'severity': 'major', 'dimension': 'materials',
                                 'object_ids': ['oak']}]
        _, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['detailed_group_ids'],
                         sorted([self.group['id'], 'on_second_bed', 'on_bed_group']))
        self.assertEqual(focus['inventory_only_group_ids'], ['unrelated'])

    def test_floor_and_unreferenced_shared_material_do_not_join_groups(self):
        self.group['support_id'] = 'floor'
        self.bundle['shared_scene']['custom_objects'][1]['support_id'] = 'floor'
        _, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['detailed_group_ids'], [self.group['id']])
        self.assertEqual(focus['inventory_only_group_ids'], ['unrelated'])

    def _add_texture(self, material_id=None):
        scene = self.bundle['shared_scene']
        scene['textures'].append({'id': 'photo_crop',
                                  'quad': [[0, 0], [.5, 0], [.5, .5], [0, .5]]})
        if material_id:
            next(material for material in scene['materials']
                 if material['id'] == material_id)['texture_id'] = 'photo_crop'

    def test_mixed_group_and_texture_references_retain_material_users_and_supports(self):
        self._add_texture('cream_fabric')
        textured = self._add_supported_group('textured', '')
        self._add_supported_group('on_textured', textured['id'])
        part_user = self._add_supported_group('textured_part', '')
        part_user['material_id'] = 'metal'
        part_user['parts'][0]['material_id'] = 'cream_fabric'
        self.report['issues'] = [{'severity': 'major', 'dimension': 'materials',
                                 'object_ids': [self.group['id'], 'photo_crop']}]
        before = copy.deepcopy(self.bundle)
        context, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['detailed_group_ids'],
                         sorted([self.group['id'], 'textured', 'on_textured', 'textured_part']))
        self.assertEqual(focus['inventory_only_group_ids'], ['unrelated'])
        self.assertEqual(context['shared_scene']['textures'],
                         self.bundle['shared_scene']['textures'])
        self.assertEqual(self.bundle, before)

    def test_texture_reference_retains_base_binding_users_and_repeat_supports(self):
        self._repeat_base_bed()
        self._add_texture('oak')
        scene = self.bundle['shared_scene']
        scene['surface_overrides'] = dict.fromkeys(scene['surface_overrides'], '')
        self._add_supported_group('on_first_bed', 'bed__0_0')
        self._add_supported_group('on_second_bed', 'bed__1_0')
        self.bundle['shared_scene']['base_material_overrides'] = [{
            'object_id': 'bed__1_0', 'material_id': 'oak',
            'selector': {'role': 'body', 'material_name': '', 'slot_index': -1}}]
        self.report['issues'] = [{'severity': 'major', 'dimension': 'materials',
                                 'object_ids': [self.group['id'], 'photo_crop']}]
        _, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['detailed_group_ids'],
                         sorted([self.group['id'], 'on_first_bed', 'on_second_bed']))
        self.assertEqual(focus['inventory_only_group_ids'], ['unrelated'])

    def test_surface_texture_or_material_reference_keeps_full_geometry(self):
        self._add_texture('oak')
        scene = self.bundle['shared_scene']
        for surface in scene['surface_overrides']:
            for reference in ('photo_crop', 'oak'):
                with self.subTest(surface=surface, reference=reference):
                    scene['surface_overrides'] = dict.fromkeys(scene['surface_overrides'], '')
                    scene['surface_overrides'][surface] = 'oak'
                    self.report['issues'] = [{'severity': 'major', 'dimension': 'materials',
                                             'object_ids': [self.group['id'], reference]}]
                    _, focus = repair_context(self.bundle, self.report)
                    self.assertEqual(focus['inventory_only_group_ids'], [])

    def test_global_texture_reference_outside_selected_family_keeps_full_geometry(self):
        self._add_texture('oak')
        scene = self.bundle['shared_scene']
        for usage in ('surface', 'exterior'):
            with self.subTest(usage=usage):
                scene['surface_overrides'] = dict.fromkeys(scene['surface_overrides'], '')
                scene['exterior_contexts'] = []
                if usage == 'surface':
                    scene['surface_overrides']['floor'] = 'oak'
                else:
                    scene['exterior_contexts'] = [{'window_id': 'window',
                        'texture_id': 'photo_crop', 'depth_m': 1, 'brightness_ev': 0}]
                self.report['issues'] = [
                    {'severity': 'blocking', 'dimension': 'geometry',
                     'object_ids': [self.group['id']]},
                    {'severity': 'minor', 'dimension': 'materials',
                     'object_ids': ['photo_crop']}]
                _, focus = repair_context(self.bundle, self.report)
                self.assertEqual(focus['issue_indices'], [0])
                self.assertEqual(focus['inventory_only_group_ids'], [])

    def test_unreferenced_global_texture_does_not_expand_local_context(self):
        self._add_texture('oak')
        self.bundle['shared_scene']['surface_overrides']['floor'] = 'oak'
        self.bundle['shared_scene']['exterior_contexts'] = [{'window_id': 'window',
            'texture_id': 'photo_crop', 'depth_m': 1, 'brightness_ev': 0}]
        _, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['detailed_group_ids'], [self.group['id']])
        self.assertEqual(focus['inventory_only_group_ids'], ['unrelated'])

    def test_global_issue_outside_selected_family_retains_all_geometry(self):
        for dimension in ('framing', 'lighting', 'realism'):
            with self.subTest(dimension=dimension):
                self.report['issues'] = [
                    {'severity': 'blocking', 'dimension': 'geometry',
                     'object_ids': [self.group['id']]},
                    {'severity': 'minor', 'dimension': dimension,
                     'object_ids': [self.group['id']]},
                ]
                _, focus = repair_context(self.bundle, self.report)
                self.assertEqual(focus['issue_indices'], [0])
                self.assertEqual(focus['inventory_only_group_ids'], [])

    def test_global_issue_with_known_reference_retains_all_geometry(self):
        self.report['issues'] = [{'severity': 'major', 'dimension': 'framing',
                                 'object_ids': [self.group['id']]}]
        _, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['inventory_only_group_ids'], [])

    def test_mixed_known_and_unknown_references_retain_all_geometry(self):
        self.report['issues'] = [{'severity': 'major', 'dimension': 'geometry',
                                 'object_ids': [self.group['id'], 'unknown_part']}]
        _, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['inventory_only_group_ids'], [])

    def test_unresolved_mixed_reference_outside_selected_family_retains_all_geometry(self):
        self.report['issues'] = [
            {'severity': 'blocking', 'dimension': 'geometry',
             'object_ids': [self.group['id']]},
            {'severity': 'minor', 'dimension': 'materials',
             'object_ids': [self.group['id'], 'unknown_part']},
        ]
        _, focus = repair_context(self.bundle, self.report)
        self.assertEqual(focus['issue_indices'], [0])
        self.assertEqual(focus['inventory_only_group_ids'], [])

    def test_hidden_geometry_edits_rejected_but_existing_noop_allowed(self):
        local_report = {'issues': [
            {'severity': 'major', 'dimension': 'geometry', 'object_ids': [self.group['id']]}]}
        _, focus = repair_context(self.bundle, local_report)
        validate_edit_scope(empty_edits(), focus)
        for field, item in [('part_updates', {'group_id': 'unrelated', 'index': 0, 'part': self.group['parts'][0]}),
                            ('replace_groups', {'id': 'unrelated'}),
                            ('remove_groups', 'unrelated')]:
            with self.subTest(field=field):
                edits = empty_edits()
                edits[field] = [item]
                with self.assertRaisesRegex(ValueError, 'omitted'):
                    validate_edit_scope(edits, focus)

    def test_order_independent_context(self):
        import json
        a = repair_context(self.bundle, self.report)
        b = repair_context(json.loads(json.dumps(self.bundle, sort_keys=True)), self.report)
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
