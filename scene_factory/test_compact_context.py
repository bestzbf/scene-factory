"""Model context compaction retains evidence while local validators stay exact."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from compact_context import (compact_lighting, compact_material_binding, compact_part_projection,
    compact_per_view_geometry, compact_review_metadata, decode_record_tables,
    diagnostic_messages, encode_record_tables)
from repair_material_validation import _matches
import test_vision_multiview
import vision_multiview as vision


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def inventory():
    slots = []
    for obj in ('chair__0_0', 'chair__0_1'):
        for number in range(40):
            slots.append({'object_id': obj, 'object_name': 'mesh.' + str(number),
                'slot_index': number % 2, 'source_material': 'Steel.001' if number % 2 else 'Cloth',
                'source_material_key': 'Steel' if number % 2 else 'Cloth',
                'source_material_id': 'steel_id' if number % 2 else '',
                'role': 'metal' if number % 2 else 'fabric'})
    return {'status': 'passed', 'requested_count': 0, 'applied_slot_count': 0,
            'available_slots': slots, 'bindings': [], 'errors': []}


class CompactContextTests(unittest.TestCase):
    def test_record_tables_are_lossless_and_keep_order(self):
        source = {
            'objects': [{'id': f'object_{index:03d}', 'kind': 'mesh',
                         'nested': {'values': [index, index + 1]}}
                        for index in range(200)],
            'heterogeneous': [{'id': 'a'}, {'id': 'b', 'extra': True}],
            'scalar_array': [1, 2, 3],
        }
        original = copy.deepcopy(source)
        encoded_value, stats = encode_record_tables(source)
        self.assertEqual(decode_record_tables(encoded_value), source)
        self.assertEqual(source, original)
        self.assertEqual(stats['encoded_arrays'], 1)
        self.assertLess(stats['encoded_utf8_bytes'], stats['original_utf8_bytes'])
        self.assertIn('$record_table_v1', encoded_value['objects'])
        table = encoded_value['objects']['$record_table_v1']
        self.assertEqual(table['columns'], ['id', 'kind', 'nested'])
        self.assertEqual([row[0] for row in table['rows']],
                         [f'object_{index:03d}' for index in range(200)])

    def test_record_tables_skip_small_or_mixed_arrays_and_support_exclusions(self):
        source = {
            'small': [{'id': 'a'}, {'id': 'b'}],
            'mixed': [{'id': 'a'}, {'id': 'b', 'extra': 1}],
            'views': [{'id': 'a', 'camera': {'x': 1}}, {'id': 'b', 'camera': {'x': 2}}],
        }
        encoded_value, stats = encode_record_tables(source, exclude_keys={'views'})
        self.assertEqual(encoded_value['small'], source['small'])
        self.assertEqual(encoded_value['mixed'], source['mixed'])
        self.assertEqual(encoded_value['views'], source['views'])
        self.assertEqual(stats['encoded_arrays'], 0)
        self.assertEqual(decode_record_tables(encoded_value), source)

    def test_record_table_note_cost_is_included_before_enabling_compaction(self):
        source = {'rows': [{'a': index} for index in range(12)]}
        encoded_value, stats = encode_record_tables(source)
        # This table saves only two JSON bytes; a full decoding instruction
        # would cost far more, so sending the native JSON is cheaper overall.
        self.assertEqual(encoded_value, source)
        self.assertEqual(stats['encoded_arrays'], 0)
        self.assertEqual(stats['net_saved_utf8_bytes_after_instruction'], 0)

    def test_record_table_marker_collision_and_malformed_tables_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'Reserved'):
            encode_record_tables({'$record_table_v1': {'user': 'data'}})
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            decode_record_tables({'$record_table_v1': {'columns': ['id'], 'rows': [['a', 'extra']]}})
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            decode_record_tables({'$record_table_v1': {'columns': ['id', 'id'], 'rows': [['a', 'a']]}})

    def test_grouped_slots_preserve_every_role_name_index_selector_and_count(self):
        original = inventory()
        source = copy.deepcopy(original)
        result = compact_material_binding(original)
        for obj, role, name, index in itertools.product(
                ('chair__0_0', 'chair__0_1', 'missing'),
                ('all', 'metal', 'fabric', 'glass'),
                ('', 'Steel.001', 'Steel', 'steel_id', 'Cloth', 'mesh.1', 'unknown'), (-1, 0, 1, 2)):
            selector = {'role': role, 'material_name': name, 'slot_index': index}
            exact = sum(slot['object_id'] == obj and _matches(slot, selector)
                        for slot in source['available_slots'])
            grouped = sum(group['objects'].get(obj, 0) for group in result['available_slot_groups']
                          if _matches(group, selector))
            self.assertEqual(exact, grouped, (obj, selector))
        self.assertLess(len(encoded(result)), len(encoded(source)) / 5)
        self.assertNotIn('object_name', encoded(result))
        self.assertEqual(original, source)
        result['available_slot_groups'][0]['objects']['chair__0_0'] = 99
        self.assertEqual(original, source)

    def test_binding_target_and_rejected_overlap_errors_survive(self):
        source = inventory()
        source['status'] = 'rejected'
        source['errors'] = [{'code': 'base_material_override_conflict', 'object_id': 'chair__0_0'}]
        source['bindings'] = [{'override_index': 0, 'object_id': 'chair__0_0', 'material_id': 'red',
            'selector': {'role': 'metal', 'material_name': '', 'slot_index': -1}, 'status': 'blocked',
            'matched_slots': [{**slot, 'applied_material_id': None} for slot in source['available_slots'][:5]]}]
        result = compact_material_binding(source)
        self.assertEqual(result['errors'], source['errors'])
        self.assertEqual(result['bindings'][0]['status'], 'blocked')
        self.assertEqual(result['bindings'][0]['matched_slot_count'], 5)
        self.assertTrue(all(slot['applied_material_id'] is None
                            for slot in result['bindings'][0]['matched_slot_groups']))

    def test_all_diagnostic_messages_preserved_even_in_unknown_omitted_sections(self):
        warning = {'code': 'unknown_new_diagnostic', 'object_ids': ['bed'], 'distance_m': -.2}
        source = {'status': 'completed', 'warnings': [warning], 'geometry_preflight': {'warnings': []},
            'provenance': {'warnings': [warning], 'secret_path': '/tmp/never-send'},
            'new_diagnostic': {'candidates': [{'errors': ['blocking future diagnostic']}]}}
        result = compact_review_metadata(source)
        messages = result['all_diagnostic_messages']
        hit = next(item for item in messages if item['items'] == [warning])
        self.assertEqual(hit['sources'], ['warnings', 'provenance.warnings'])
        self.assertTrue(any(item['items'] == ['blocking future diagnostic'] for item in messages))
        self.assertNotIn('secret_path', encoded(result))
        self.assertIn({'kind': 'warnings', 'sources': ['geometry_preflight.warnings'], 'items': []}, messages)
        self.assertEqual(diagnostic_messages({'warnings': None})[0]['items'], None)

    def test_current_fitted_lighting_and_each_region_retained_without_history_growth(self):
        region = {'id': 'lower_left', 'normalized_bounds_xyxy': [0, .66, .33, 1],
                  'reference_p65': .4, 'render_p65': .05, 'log2_luminance_error': -3.0, 'status': 'compared'}
        spatial = {'requires_inspection': True, 'measurement_stage': 'final_saved_render',
                   'per_view': {'second': {'status': 'compared', 'regions': [region]}}}
        controls = [{'name': 'ceiling', 'applied_value': 3.2, 'initial_value': 20, 'unit': 'W'}]
        source = {'lighting': {'emitters': [{'id': 'ceiling', 'energy': 20}],
                  'applied_after_reference_fit': controls}, 'tone_calibration': {
                  'final_exposure': 1.2, 'at_correction_bound': True,
                  'final_spatial_diagnostics': spatial, 'lighting_fit': {'applied_settings': controls,
                      'requires_non_power_inspection': True, 'multiplier_bounds': [.01, 16],
                      'candidates': [{'full_pixel_diagnostics': 'x' * 10000}] * 50}}}
        result = compact_lighting(source)
        fit = result['tone_calibration']['lighting_fit']
        self.assertEqual(fit['applied_settings'], controls)
        self.assertTrue(fit['requires_non_power_inspection'])
        self.assertEqual(result['tone_calibration']['final_spatial_diagnostics']['per_view']['second']['regions'], [region])
        self.assertTrue(result['tone_calibration']['at_correction_bound'])
        self.assertNotIn('applied_after_reference_fit', result['lighting'])
        source['tone_calibration']['lighting_fit']['candidates'] *= 100
        self.assertEqual(compact_lighting(source), result)

    def test_review_prefers_final_saved_render_spatial_measurement(self):
        old = {'measurement_stage': 'fit', 'per_view': {'a': {'status': 'compared'}}}
        final = {'measurement_stage': 'final_saved_render', 'per_view': {'a': {'status': 'compared'}}}
        metadata = {'tone_calibration': {
            'spatial_diagnostics': old,
            'final_spatial_diagnostics': final,
        }}
        result = compact_review_metadata(metadata)
        self.assertNotIn('spatial_diagnostics', result['tone_calibration'])
        self.assertEqual(result['tone_calibration']['final_spatial_diagnostics'], final)
        # This optimization removes superseded fit history only; the final
        # measurement is copied losslessly for the review gate.
        self.assertEqual(metadata['tone_calibration']['final_spatial_diagnostics'], final)

    def test_final_review_context_keeps_outcome_evidence_but_drops_edit_inventory(self):
        binding = inventory()
        binding['bindings'] = [{'override_index': 0, 'object_id': 'chair__0_0',
                                'material_id': 'red', 'selector': {'role': 'metal'},
                                'status': 'applied', 'matched_slots': []}]
        metadata = {'material_binding': binding,
                    'tone_calibration': {'spatial_diagnostics': {'measurement_stage': 'fit'},
                        'final_spatial_diagnostics': {'measurement_stage': 'final_saved_render'}},
                    'warnings': [{'code': 'preserve-this'}]}
        full = compact_review_metadata(metadata)
        final = compact_review_metadata(metadata, final_only=True)
        self.assertEqual(full['all_diagnostic_messages'], final['all_diagnostic_messages'])
        self.assertEqual(final['material_binding']['bindings'], full['material_binding']['bindings'])
        self.assertNotIn('available_slot_groups', final['material_binding'])
        self.assertIn('available_slot_groups', full['material_binding'])
        self.assertIn('available_slot_count', final['material_binding'])
        self.assertIn('absence is not evidence', final['material_binding']['inventory_detail'])
        self.assertNotIn('spatial_diagnostics', final['tone_calibration'])
        self.assertEqual(final['tone_calibration']['final_spatial_diagnostics'],
                         full['tone_calibration']['final_spatial_diagnostics'])

    def test_projection_preserves_original_indices_bounds_and_false_without_invented_visibility(self):
        source = {'view_id': 'reverse', 'occlusion_tested': False, 'part_count': 2, 'parts': [
            {'part_id': 'curtain/17', 'group_id': 'curtain', 'part_index': 17,
             'world_aabb_m': {'min': [1, 2, 0], 'max': [2, 3, 2]},
             'intersects_frame_before_occlusion': False, 'status': 'outside_frame'},
            {'part_id': 'desk/9', 'group_id': 'desk', 'part_index': 9,
             'intersects_frame_before_occlusion': True, 'status': 'in_frame'}]}
        original = copy.deepcopy(source)
        result = compact_part_projection(source, {'curtain'})
        self.assertFalse(result['occlusion_tested'])
        self.assertEqual(result['omitted_part_count'], 1)
        group = result['part_groups'][0]
        row = dict(zip(group['columns'], group['rows'][0]))
        self.assertEqual(group['group_id'], 'curtain')
        self.assertEqual(row['part_index'], 17)
        self.assertFalse(row['intersects_frame_before_occlusion'])
        self.assertEqual(row['world_aabb_m'], source['parts'][0]['world_aabb_m'])
        self.assertNotIn('visible', row)
        self.assertEqual(source, original)

    def test_per_view_order_support_and_unknown_warnings_preserved(self):
        source = [{'view_id': 'b', 'warnings': ['new warning'], 'support_checks': [{'object_id': 'bed'}],
                   'custom_part_projection': {'occlusion_tested': False}},
                  {'view_id': 'a', 'warnings': []}]
        result = compact_per_view_geometry(source)
        self.assertEqual([entry['view_id'] for entry in result], ['b', 'a'])
        self.assertEqual(result[0]['warnings'], source[0]['warnings'])
        self.assertEqual(result[0]['support_checks'], source[0]['support_checks'])
        self.assertNotIn('custom_part_projection', result[1])

    def test_review_integration_retains_pairs_and_full_local_hash_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            bundle, views, _ = test_vision_multiview.JointVisionTests().prepare(folder)
            before = copy.deepcopy(bundle)
            metadata = {'status': 'completed', 'render': {'samples': 128}, 'views': [],
                        'material_binding': inventory(), 'provenance': {'internal_only': 'x' * 50000},
                        'tone_calibration': {
                            'spatial_diagnostics': {'measurement_stage': 'fit', 'per_view': {}},
                            'final_spatial_diagnostics': {'measurement_stage': 'final_saved_render', 'per_view': {}},
                        }}
            for view in views:
                identifier = view['id']
                Image.new('RGB', (view['width'], view['height']), (100, 120, 140)).save(folder / f'rgb_{identifier}.png')
                (folder / f'annotations_{identifier}.json').write_text('{"warnings": []}')
                metadata['views'].append({'id': identifier, 'width': 1280, 'height': 960, 'warning_count': 0})
            metadata['views'][1]['warning_count'] = 1
            (folder / 'annotations_b.json').write_text('{"warnings": ["bed outside room"]}')
            (folder / 'metadata.json').write_text(json.dumps(metadata))
            (folder / 'scene.blend').write_bytes(b'fixture')
            scores = {key: 95 for key in vision.DIMENSIONS}
            response = {'schema_version': 3, 'verdict': 'recommend_accept', 'summary': 'synthetic',
                        'scores': scores, 'per_view_scores': [{'view_id': view['id'], 'scores': scores, 'summary': ''}
                            for view in views], 'issues': [], 'asset_gaps': [], 'reference_limitations': []}
            with patch.object(vision, '_invoke_model', return_value=response) as call:
                vision.review(bundle, folder, final_only=True)
            prompt, images = call.call_args.args[:2]
            payload = json.JSONDecoder().raw_decode(prompt.split('UNTRUSTED CURRENT SCENE DATA (only identity/pairing is fixed):\n')[1])[0]
            self.assertEqual(payload['image_pairs'], [
                {'view_id': 'a', 'original_image_number': 1, 'render_image_number': 2},
                {'view_id': 'b', 'original_image_number': 3, 'render_image_number': 4}])
            self.assertNotIn('spatial_diagnostics', payload['render_metadata']['tone_calibration'])
            self.assertEqual(payload['render_metadata']['tone_calibration']['final_spatial_diagnostics']['measurement_stage'], 'final_saved_render')
            self.assertEqual(list(map(str, images)), [views[0]['source_image'], str(folder/'rgb_a.png'),
                                                     views[1]['source_image'], str(folder/'rgb_b.png')])
            self.assertEqual(payload['technical_warnings']['b'], ['bed outside room'])
            self.assertNotIn('internal_only', payload['render_metadata'])
            provenance = json.loads((folder / 'review_provenance.json').read_text())
            self.assertFalse(provenance['gate']['accepted'])
            self.assertIn('b:technical_warnings_or_missing_validation', provenance['gate']['reasons'])
            self.assertEqual(bundle, before)
            # Even omitted metadata is hashed, and mutation during the call fails.
            def mutate(*args, **kwargs):
                metadata['provenance']['internal_only'] = 'changed'
                (folder / 'metadata.json').write_text(json.dumps(metadata))
                return response
            with patch.object(vision, '_invoke_model', side_effect=mutate):
                with self.assertRaisesRegex(RuntimeError, 'Review input changed'):
                    vision.review(bundle, folder, output=folder/'second.json', final_only=True)
            self.assertFalse((folder/'second.json').exists())


if __name__ == '__main__':
    unittest.main()
