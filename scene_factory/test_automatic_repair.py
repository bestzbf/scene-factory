"""No-network tests for automatic typed repair and fail-closed cache integrity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

import automatic_repair as repair
from model_api import ModelAPIError
from spec_v2 import part
from test_multiview import fixture
from vision_multiview import DIMENSIONS


CONFIG = {'provider': 'openai_chat', 'model': 'synthetic-vision',
          'base_url': 'http://127.0.0.1:19000/v1', 'api_key_env': '',
          'max_retries': 0, 'timeout_seconds': 300}


def prepare(directory):
    bundle = fixture()
    bundle['shared_scene']['image'] = {'width': 1280, 'height': 960, 'source_id': 'a'}
    for view in bundle['views']:
        source = directory / (view['id'] + '.jpg')
        Image.new('RGB', (view['width'], view['height']), (70, 90, 110)).save(source)
        view['source_image'] = str(source)
        Image.new('RGB', (view['width'] * 2, view['height'] * 2), (100, 120, 140)).save(directory / ('rgb_' + view['id'] + '.png'))
    scores = {key: 50 for key in DIMENSIONS}
    report = {'schema_version': 3, 'verdict': 'repair', 'summary': 'Camera mismatch',
              'scores': scores, 'per_view_scores': [{'view_id': view['id'], 'scores': scores.copy(), 'summary': 'Mismatch'} for view in bundle['views']],
              'issues': [{'dimension': 'camera', 'severity': 'major', 'view_ids': ['a', 'b'],
                          'evidence': 'Framing differs', 'action': 'Correct perspective', 'object_ids': []}],
              'asset_gaps': [], 'reference_limitations': [], 'revised_bundle': copy.deepcopy(bundle)}
    # Use an actual supported score dimension even if the shared score catalog changes.
    report['issues'][0]['dimension'] = next(iter(DIMENSIONS))
    return bundle, report


class TypedEditTests(unittest.TestCase):
    def test_noop_preserves_source_identity_and_every_field(self):
        bundle = fixture()
        original = copy.deepcopy(bundle)
        result = repair.apply_edits(bundle, repair.empty_edits())
        self.assertEqual(result, original)
        self.assertEqual(bundle, original)
        self.assertIsNot(result, bundle)

    def test_all_cameras_and_shared_transforms_apply_transactionally(self):
        bundle = fixture()
        old = copy.deepcopy(bundle)
        edits = repair.empty_edits()
        edits['cameras'] = [{'view_id': view['id'], 'camera': {**view['camera'], 'lens_mm': 35}} for view in bundle['views']]
        edits['render'] = [{**bundle['shared_scene']['render'], 'exposure': -1.0}]
        edits['lighting'] = [{**bundle['shared_scene']['base_scene']['lighting'], 'world_strength': .08}]
        result = repair.apply_edits(bundle, edits)
        self.assertEqual([view['camera']['lens_mm'] for view in result['views']], [35, 35])
        self.assertEqual(result['shared_scene']['base_scene']['camera']['lens_mm'], 35)
        self.assertEqual(result['shared_scene']['camera']['roll_degrees'], -2)
        self.assertEqual(result['shared_scene']['render']['exposure'], -1)
        self.assertEqual(bundle, old)
        for before, after in zip(bundle['views'], result['views']):
            self.assertEqual({key: before[key] for key in ('id', 'source_image', 'width', 'height')},
                             {key: after[key] for key in ('id', 'source_image', 'width', 'height')})

    def test_updates_use_original_indices_then_remove_descending_and_append(self):
        bundle = fixture()
        group = copy.deepcopy(bundle['shared_scene']['custom_objects'][0])
        group['parts'] = [part('box', size=[1, 1, 1], position=[index, 0, 0]) for index in range(4)]
        bundle['shared_scene']['custom_objects'] = [group]
        edits = repair.empty_edits()
        edits['part_updates'] = [{'group_id': group['id'], 'index': 2, 'part': part('curve', radius=.003, points=[[0, 0, 0], [1, 0, 0]])}]
        edits['remove_parts'] = [{'group_id': group['id'], 'indices': [0, 3]}]
        edits['add_parts'] = [{'group_id': group['id'], 'parts': [part('box', size=[2, 2, 2])]}]
        result = repair.apply_edits(bundle, edits)['shared_scene']['custom_objects'][0]['parts']
        self.assertEqual([item['type'] for item in result], ['box', 'curve', 'box'])
        self.assertEqual(result[0]['position'], [1, 0, 0])
        self.assertEqual(result[1]['radius'], .003)
        self.assertEqual(result[2]['size'], [2, 2, 2])

    def test_common_edits_and_group_insertion_need_no_full_scene_rewrite(self):
        bundle = fixture()
        edits = repair.empty_edits()
        group = copy.deepcopy(bundle['shared_scene']['custom_objects'][0])
        group['id'] = 'new_visible_structure'
        group['support_id'] = ''
        edits['replace_groups'] = [group]
        room = copy.deepcopy(bundle['shared_scene']['base_scene']['room'])
        room['width'] += .2
        edits['room'] = [room]
        edits['architecture'] = [{'omit_surfaces': ['back', 'ceiling'], 'front_wall': False}]
        texture = {'id': 'local_screen', 'quad': [[.1, .1], [.5, .1], [.5, .5], [.1, .5]], 'view_id': 'b'}
        edits['textures'] = [texture]
        result = repair.apply_edits(bundle, edits)
        self.assertEqual(result['shared_scene']['custom_objects'][-1], group)
        self.assertEqual(result['shared_scene']['base_scene']['room'], room)
        self.assertIn({'texture_id': 'local_screen', 'view_id': 'b'}, result['texture_sources'])
        self.assertNotIn('view_id', result['shared_scene']['textures'][-1])

    def test_rejects_unknown_duplicate_conflicting_indices_and_ids(self):
        bundle = fixture()
        group = bundle['shared_scene']['custom_objects'][0]
        cases = []
        edit = repair.empty_edits(); edit['cameras'] = [{'view_id': 'unknown', 'camera': bundle['views'][0]['camera']}]; cases.append(edit)
        edit = repair.empty_edits(); edit['cameras'] = [{'view_id': 'a', 'camera': bundle['views'][0]['camera']}] * 2; cases.append(edit)
        edit = repair.empty_edits(); edit['remove_parts'] = [{'group_id': group['id'], 'indices': [99]}]; cases.append(edit)
        edit = repair.empty_edits(); edit['remove_parts'] = [{'group_id': group['id'], 'indices': [0, 0]}]; cases.append(edit)
        edit = repair.empty_edits(); edit['part_updates'] = [{'group_id': group['id'], 'index': 0, 'part': group['parts'][0]}]; edit['remove_parts'] = [{'group_id': group['id'], 'indices': [0]}]; cases.append(edit)
        edit = repair.empty_edits(); edit['replace_groups'] = [group]; edit['add_parts'] = [{'group_id': group['id'], 'parts': []}]; cases.append(edit)
        edit = repair.empty_edits(); edit['remove_materials'] = ['unknown']; cases.append(edit)
        for index, edits in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(repair.RepairValidationError):
                repair.apply_edits(bundle, edits)

    def test_semantic_failures_are_sanitized_and_unsupported_fields_rejected(self):
        bundle = fixture()
        group = bundle['shared_scene']['custom_objects'][0]
        edit = repair.empty_edits()
        badpart = copy.deepcopy(group['parts'][0]); badpart['material_id'] = 'secret-shaped-untrusted-text'
        edit['part_updates'] = [{'group_id': group['id'], 'index': 0, 'part': badpart}]
        with self.assertRaises(repair.RepairValidationError) as raised:
            repair.apply_edits(bundle, edit)
        self.assertNotIn('secret-shaped', str(raised.exception))
        self.assertIn('unknown', str(raised.exception))
        edit = repair.empty_edits(); edit['source_image'] = '/tmp/changed.png'
        with self.assertRaisesRegex(repair.RepairValidationError, 'typed schema'):
            repair.apply_edits(bundle, edit)


class APIRepairTests(unittest.TestCase):
    def test_token_exhaustion_retries_bounded_without_accepting_partial_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            error = ModelAPIError('output_truncated', 'budget exhausted')
            with patch.object(repair, 'call_model', side_effect=[error, repair.empty_edits()]) as call:
                result = repair.repair(bundle, directory, report, CONFIG, validation_retries=1)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(result, bundle)
            self.assertIn('small number of edits', call.call_args.args[0])
            self.assertEqual(call.call_args.kwargs['config'], repair.public_config(CONFIG))

    def test_spatial_lighting_evidence_is_sent_and_bound_to_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            metadata = {'tone_calibration': {'status': 'selected', 'at_correction_bound': True,
                        'spatial_diagnostics': {'requires_inspection': True, 'reasons': ['regional_mismatch']}}}
            (directory / 'metadata.json').write_text(json.dumps(metadata))
            with patch.object(repair, 'call_model', return_value=repair.empty_edits()) as call:
                repair.repair(bundle, directory, report, CONFIG)
            prompt = call.call_args.args[0]
            self.assertIn('lighting_diagnostics', prompt)
            self.assertIn('regional_mismatch', prompt)
            metadata['tone_calibration']['at_correction_bound'] = False
            (directory / 'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'Cached repair inputs'):
                repair.repair(bundle, directory, report, CONFIG)

    def test_cache_stable_after_sorted_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            with patch.object(repair, 'call_model', return_value=repair.empty_edits()):
                expected = repair.repair(bundle, directory, report, CONFIG)
            bundle = json.loads(json.dumps(bundle, sort_keys=True))
            report = json.loads(json.dumps(report, sort_keys=True))
            with patch.object(repair, 'call_model', side_effect=AssertionError('Cache must survive key order')):
                self.assertEqual(repair.repair(bundle, directory, report, CONFIG), expected)

    def test_failed_usage_records_preserved_on_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            def failed(prompt, images, schema, output, *args, **kwargs):
                output.with_suffix('.api.json').write_text(json.dumps({'status':'failed','usage':{'total_tokens':99}}))
                raise ModelAPIError('timeout', 'sanitized')
            with patch.object(repair, 'call_model', side_effect=failed):
                with self.assertRaises(ModelAPIError):
                    repair.repair(bundle, directory, report, CONFIG)
            old = (directory/'repair_attempt_00.api.json').read_bytes()
            def success(prompt, images, schema, output, *args, **kwargs):
                self.assertEqual(output.name, 'repair_attempt_01.json')
                output.with_suffix('.api.json').write_text(json.dumps({'status':'completed','usage':{'total_tokens':5}}))
                return repair.empty_edits()
            with patch.object(repair, 'call_model', side_effect=success):
                repair.repair(bundle, directory, report, CONFIG)
            self.assertEqual((directory/'repair_attempt_00.api.json').read_bytes(), old)
            self.assertEqual(sum(json.loads(p.read_text())['usage']['total_tokens'] for p in directory.glob('*.api.json')), 104)

    def test_api_receives_all_pairs_exact_config_and_report_no_codex(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            edits = repair.empty_edits(); edits['render'] = [{**bundle['shared_scene']['render'], 'exposure': -.3}]
            with patch.object(repair, 'call_model', return_value=edits) as call:
                candidate = repair.repair(bundle, directory, report, CONFIG)
            args = call.call_args
            self.assertEqual([str(path) for path in args.args[1]], [bundle['views'][0]['source_image'], str(directory/'model_input_images/rgb_a.png'), bundle['views'][1]['source_image'], str(directory/'model_input_images/rgb_b.png')])
            self.assertIn('Framing differs', args.args[0])
            self.assertNotIn(str(directory), args.args[0])
            self.assertEqual(args.kwargs['config']['model'], CONFIG['model'])
            self.assertEqual(candidate['shared_scene']['render']['exposure'], -.3)
            provenance = json.loads((directory/'repair_provenance.json').read_text())
            self.assertFalse(provenance['manual_visual_feedback'])
            self.assertTrue(provenance['candidate_requires_new_render_and_review'])
            self.assertEqual(len(provenance['image_sha256']), 2)
            self.assertEqual(provenance['model_image_input_policy'], 'matched_render_at_reference_resolution_v1')
            self.assertEqual([item['model_input_dimensions_px'] for item in provenance['model_image_inputs']],
                             [[1280, 960], [960, 1280]])
            self.assertTrue(all(item['transform'] == 'render_only_lanczos_downsample_to_reference_dimensions'
                                for item in provenance['model_image_inputs']))
            self.assertEqual(provenance['model_image_input_implementation_sha256'],
                             repair.file_sha256(Path(repair.__file__).with_name('model_image_inputs.py')))
            with patch.object(repair, 'call_model', side_effect=AssertionError('Verified cache must be used')):
                self.assertEqual(repair.repair(bundle, directory, report, CONFIG), candidate)

    def test_bad_semantics_retry_with_safe_feedback_and_no_partial_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            invalid = repair.empty_edits(); invalid['remove_groups'] = ['untrusted-secret']
            with patch.object(repair, 'call_model', side_effect=[invalid, repair.empty_edits()]) as call:
                result = repair.repair(bundle, directory, report, CONFIG, validation_retries=1)
            self.assertEqual(result, bundle)
            self.assertEqual(call.call_count, 2)
            retry_prompt = call.call_args.args[0]
            self.assertIn('PRIOR ATTEMPT REJECTED', retry_prompt)
            self.assertNotIn('untrusted-secret', retry_prompt)
            self.assertLess(call.call_args.args[4], call.call_args_list[0].args[4])
            provenance = json.loads((directory/'repair_provenance.json').read_text())
            self.assertEqual([attempt['status'] for attempt in provenance['attempts']], ['rejected', 'validated'])

    def test_bad_json_retry_but_transport_failure_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            with patch.object(repair, 'call_model', side_effect=[ModelAPIError('schema_mismatch', 'sanitized'), repair.empty_edits()]) as call:
                self.assertEqual(repair.repair(bundle, directory, report, CONFIG), bundle)
                self.assertEqual(call.call_count, 2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            with patch.object(repair, 'call_model', side_effect=ModelAPIError('http_error', 'sanitized')) as call:
                with self.assertRaises(ModelAPIError):
                    repair.repair(bundle, directory, report, CONFIG)
                self.assertEqual(call.call_count, 1)
            self.assertFalse((directory/'repaired_bundle.json').exists())

    def test_exhausted_validation_saves_no_success_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            invalid = repair.empty_edits(); invalid['remove_groups'] = ['missing']
            with patch.object(repair, 'call_model', return_value=invalid) as call:
                with self.assertRaisesRegex(repair.RepairValidationError, 'exhausted'):
                    repair.repair(bundle, directory, report, CONFIG, validation_retries=1)
                self.assertEqual(call.call_count, 2)
            self.assertFalse((directory/'repair_provenance.json').exists())
            self.assertFalse((directory/'repaired_bundle.json').exists())

    def test_changed_each_bound_input_or_output_rejects_cache(self):
        for target in ('source', 'render', 'report', 'bundle', 'config', 'repair', 'candidate'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()
                bundle, report = prepare(directory)
                with patch.object(repair, 'call_model', return_value=repair.empty_edits()):
                    repair.repair(bundle, directory, report, CONFIG)
                config = copy.deepcopy(CONFIG)
                if target == 'source': Path(bundle['views'][1]['source_image']).write_bytes(b'changed')
                elif target == 'render':
                    (directory/'rgb_b.png').unlink()
                    Image.new('RGB', (1920, 2560), (10, 20, 30)).save(directory/'rgb_b.png')
                elif target == 'report': report['summary'] = 'different report'
                elif target == 'bundle':
                    bundle['shared_scene']['render']['exposure'] = -2
                    report['revised_bundle'] = copy.deepcopy(bundle)
                elif target == 'config': config['model'] = 'other-explicit-model'
                elif target == 'repair': (directory/'repair.json').write_text('{}')
                elif target == 'candidate': (directory/'repaired_bundle.json').write_text('{}')
                with patch.object(repair, 'call_model', side_effect=AssertionError('Invalid cache must fail before request')):
                    expected = 'model-input render proxy|Cached repair' if target == 'render' else 'Cached repair'
                    with self.assertRaisesRegex(ValueError, expected):
                        repair.repair(bundle, directory, report, config)

    def test_inputs_cannot_change_during_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            def mutate(*args, **kwargs):
                (directory/'rgb_b.png').unlink()
                Image.new('RGB', (1920, 2560), (10, 20, 30)).save(directory/'rgb_b.png')
                return repair.empty_edits()
            with patch.object(repair, 'call_model', side_effect=mutate):
                with self.assertRaisesRegex(RuntimeError, 'changed during'):
                    repair.repair(bundle, directory, report, CONFIG)
            self.assertFalse((directory/'repair_provenance.json').exists())

    def test_assessment_exact_bundle_and_explicit_model_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            with self.assertRaisesRegex(ValueError, 'config'):
                repair.repair(bundle, directory, report, None)
            report['revised_bundle']['shared_scene']['render']['exposure'] = -1.5
            with patch.object(repair, 'call_model', side_effect=AssertionError('Mismatched report must fail')):
                with self.assertRaisesRegex(ValueError, 'exact current bundle'):
                    repair.repair(bundle, directory, report, CONFIG)


if __name__ == '__main__':
    unittest.main()
