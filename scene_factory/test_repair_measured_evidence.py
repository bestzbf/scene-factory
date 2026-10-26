"""API repair integration for measured slots and immutable view annotations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import automatic_repair as repair
from batch import atomic_json, file_sha256, read_json
from test_automatic_repair import CONFIG, prepare


class RepairMeasuredEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name).resolve()
        self.bundle, self.report = prepare(self.directory)
        self.object_id = self.bundle['shared_scene']['base_scene']['objects'][0]['id']
        self.material_ids = [item['id'] for item in self.bundle['shared_scene']['materials']]
        self.assertGreaterEqual(len(self.material_ids), 2)
        for view in self.bundle['views']:
            Image.new('RGB', (64, 48), (70, 90, 100)).save(view['source_image'])
            Image.new('RGB', (64, 48), (90, 100, 110)).save(self.directory/f'rgb_{view["id"]}.png')
            atomic_json(self.directory/f'annotations_{view["id"]}.json', {
                'custom_part_projection': {'coordinate_origin': 'top_left', 'occlusion_tested': False,
                    'parts': [{'group_id': 'curtain', 'part_index': 0, 'in_frame': False,
                               'normalized_bbox_xyxy': [1.1, .1, 1.5, .8]}]},
                'warnings': ['Synthetic support penetration.'],
                'objects': [{'id': self.object_id, 'world_aabb_m': [[0, 0, -.1], [1, 1, 1]],
                             'support_check': {'status': 'needs_review', 'distance_m': -.1}}],
                'unrelated_annotation_data': {'audit': 1}})
        atomic_json(self.directory/'metadata.json', {'material_binding': {'available_slots': [
            {'object_id': self.object_id, 'object_name': 'Mullion', 'slot_index': 0,
             'source_material': 'Steel.001', 'source_material_key': 'Steel',
             'source_material_id': 'steel_id', 'role': 'metal'}]}})

    def edits(self, name='Steel', role='metal', index=-1):
        edits = repair.empty_edits('Synthetic measured binding.')
        edits['base_material_overrides'] = [[{'object_id': self.object_id,
            'material_id': self.material_ids[0],
            'selector': {'role': role, 'material_name': name, 'slot_index': index}}]]
        return edits

    def test_annotations_sent_as_non_occlusion_evidence_and_bound_in_cache(self):
        with patch.object(repair, 'call_model', return_value=repair.empty_edits()) as call:
            repair.repair(self.bundle, self.directory, self.report, CONFIG)
        prompt = call.call_args.args[0]
        payload = json.JSONDecoder().raw_decode(prompt.split('UNTRUSTED SCENE AND ASSESSMENT DATA:\n')[1])[0]
        self.assertEqual([item['view_id'] for item in payload['per_view_geometry']], ['a', 'b'])
        self.assertFalse(payload['per_view_geometry'][0]['custom_part_projection']['occlusion_tested'])
        self.assertEqual(payload['per_view_geometry'][0]['support_checks'][0]['object_id'], self.object_id)
        self.assertIn('NOT proof of visibility', prompt)
        provenance = read_json(self.directory/'repair_provenance.json')
        self.assertEqual(provenance['annotation_sha256'], {
            f'annotations_{view["id"]}.json': file_sha256(self.directory/f'annotations_{view["id"]}.json')
            for view in self.bundle['views']})
        with patch.object(repair, 'call_model', side_effect=AssertionError('Unchanged cache expected')):
            repair.repair(self.bundle, self.directory, self.report, CONFIG)
        annotation_path = self.directory/'annotations_a.json'
        annotation = read_json(annotation_path)
        annotation['unrelated_annotation_data']['audit'] = 2
        atomic_json(annotation_path, annotation)
        with patch.object(repair, 'call_model', side_effect=AssertionError('No API on changed cache')):
            with self.assertRaisesRegex(ValueError, 'Cached repair inputs'):
                repair.repair(self.bundle, self.directory, self.report, CONFIG)

    def test_annotation_mutation_during_call_rejects_success_commit(self):
        original = copy.deepcopy(self.bundle)
        def mutate(*args, **kwargs):
            path = self.directory/'annotations_b.json'
            annotation = read_json(path)
            annotation['custom_part_projection']['parts'][0]['in_frame'] = True
            atomic_json(path, annotation)
            return self.edits()
        with patch.object(repair, 'call_model', side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, 'input changed'):
                repair.repair(self.bundle, self.directory, self.report, CONFIG)
        self.assertFalse((self.directory/'repair_provenance.json').exists())
        self.assertFalse((self.directory/'repaired_bundle.json').exists())
        self.assertEqual(self.bundle, original)

    def test_previously_absent_annotation_appearing_invalidates_cache(self):
        (self.directory/'annotations_b.json').unlink()
        with patch.object(repair, 'call_model', return_value=repair.empty_edits()):
            repair.repair(self.bundle, self.directory, self.report, CONFIG)
        self.assertIsNone(read_json(self.directory/'repair_provenance.json')['annotation_sha256']['annotations_b.json'])
        atomic_json(self.directory/'annotations_b.json', {'custom_part_projection': {'occlusion_tested': False}})
        with patch.object(repair, 'call_model', side_effect=AssertionError('No API on new evidence')):
            with self.assertRaisesRegex(ValueError, 'Cached repair inputs'):
                repair.repair(self.bundle, self.directory, self.report, CONFIG)

    def test_component_name_rejected_then_corrected_in_same_api_validation_budget(self):
        original = copy.deepcopy(self.bundle)
        with patch.object(repair, 'call_model', side_effect=[self.edits('Mullion'), self.edits('Steel')]) as call:
            candidate = repair.repair(self.bundle, self.directory, self.report, CONFIG, validation_retries=1)
        self.assertEqual(call.call_count, 2)
        self.assertIn('PRIOR ATTEMPT REJECTED', call.call_args.args[0])
        self.assertIn('object_name is a mesh/component label', call.call_args.args[0])
        self.assertEqual(candidate['shared_scene']['base_material_overrides'][0]['selector']['material_name'], 'Steel')
        attempts = read_json(self.directory/'repair_provenance.json')['attempts']
        self.assertEqual([entry['status'] for entry in attempts], ['rejected', 'validated'])
        self.assertEqual(self.bundle, original)

    def test_conflicting_target_bindings_rejected_before_render_candidate_commit(self):
        edits = self.edits()
        edits['base_material_overrides'][0].append({'object_id': self.object_id,
            'material_id': self.material_ids[1], 'selector': {'role': 'all', 'material_name': 'steel_id', 'slot_index': 0}})
        with patch.object(repair, 'call_model', return_value=edits) as call:
            with self.assertRaisesRegex(repair.RepairValidationError, 'exhausted'):
                repair.repair(self.bundle, self.directory, self.report, CONFIG, validation_retries=0)
        self.assertEqual(call.call_count, 1)
        self.assertFalse((self.directory/'repair_provenance.json').exists())
        self.assertFalse((self.directory/'repaired_bundle.json').exists())

    def test_changed_base_geometry_defers_inventory_check_without_mutating_original(self):
        edits = self.edits('new_source_after_rebuild')
        base = copy.deepcopy(self.bundle['shared_scene']['base_scene']['objects'][0])
        base['dimensions'][0] += .1
        edits['base_objects'] = [base]
        original = copy.deepcopy(self.bundle)
        with patch.object(repair, 'call_model', return_value=edits) as call:
            candidate = repair.repair(self.bundle, self.directory, self.report, CONFIG, validation_retries=0)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(candidate['shared_scene']['base_scene']['objects'][0]['dimensions'], base['dimensions'])
        self.assertEqual(self.bundle, original)


if __name__ == '__main__':
    unittest.main()
