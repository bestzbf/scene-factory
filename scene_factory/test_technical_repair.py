"""Bounded API edits for pre-render failures; no fabricated render or review."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import automatic_repair as repair
from batch import atomic_json, file_sha256, read_json
from compact_context import RECORD_TABLE_INSTRUCTION, decode_record_tables
from model_api import ModelAPIError
from spec_v2 import part
from test_automatic_repair import CONFIG, prepare


def preflight_fixture(directory):
    bundle, _ = prepare(directory)
    # Technical recovery must not depend on an existing render.
    for path in directory.glob('rgb_*.png'):
        path.unlink()
    atomic_json(directory / 'scene_bundle.json', bundle)
    report = {'schema_version': 1, 'status': 'rejected',
              'spec_sha256': file_sha256(directory / 'scene_bundle.json'),
              'errors': [{'code': 'camera_shell_collision', 'view_id': bundle['views'][0]['id']}],
              'warnings': []}
    atomic_json(directory / 'geometry_preflight.json', report)
    edits = repair.empty_edits('Move the measured colliding camera inward.')
    camera = copy.deepcopy(bundle['views'][0]['camera'])
    camera['position'][1] += .2
    edits['cameras'] = [{'view_id': bundle['views'][0]['id'], 'camera': camera}]
    return bundle, {'geometry_preflight.json': report}, edits


class TechnicalRepairTests(unittest.TestCase):
    def test_lossless_scene_context_keeps_all_geometry_diagnostics_and_photos(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, reports, edits = preflight_fixture(directory)
            bundle['shared_scene']['custom_objects'][0]['parts'] = [
                part('box', size=[.2, .3, .4], position=[i * .01, 0, 0])
                for i in range(80)]
            atomic_json(directory / 'scene_bundle.json', bundle)
            report = reports['geometry_preflight.json']
            report['spec_sha256'] = file_sha256(directory / 'scene_bundle.json')
            report['errors'].extend([
                {'code': 'part_outside_room', 'group_id': 'fixture',
                 'part_index': i, 'measured_bounds': [[-1, -1, 0], [1, 1, 2]]}
                for i in range(8)])
            report['warnings'] = [{'code': 'keep_unknown_diagnostic', 'details': ['a', None, False]}]
            atomic_json(directory / 'geometry_preflight.json', report)
            original_bundle = copy.deepcopy(bundle)
            original_reports = copy.deepcopy(reports)
            source_bytes = [Path(v['source_image']).read_bytes() for v in bundle['views']]
            with patch.object(repair, 'call_model', return_value=edits) as call:
                candidate = repair.technical_repair(bundle, directory, reports, CONFIG)
            prompt, images, schema = call.call_args.args[:3]
            payload = json.loads(prompt.split('UNTRUSTED TECHNICAL REPAIR DATA:\n', 1)[1])
            expected_context = copy.deepcopy(bundle)
            for view in expected_context['views']:
                view['source_image'] = 'input:' + view['id']
            self.assertEqual(decode_record_tables(payload['scene_bundle']), expected_context)
            self.assertEqual(payload['scene_bundle']['views'], expected_context['views'])
            self.assertEqual(payload['technical_preflight'], original_reports)
            self.assertEqual(payload['original_images'], [
                {'view_id': v['id'], 'image_number': i + 1} for i, v in enumerate(bundle['views'])])
            self.assertEqual(images, [Path(v['source_image']) for v in bundle['views']])
            self.assertEqual([p.read_bytes() for p in images], source_bytes)
            self.assertEqual(schema, repair.REPAIR_SCHEMA)
            self.assertIn(RECORD_TABLE_INSTRUCTION, prompt)
            self.assertEqual(bundle, original_bundle)
            self.assertEqual(reports, original_reports)
            self.assertEqual(candidate, repair.apply_edits(bundle, edits))
            provenance = read_json(directory / 'technical_repair_provenance.json')
            stats = provenance['scene_bundle_compaction']
            self.assertGreater(stats['encoded_arrays'], 0)
            self.assertGreater(stats['net_saved_utf8_bytes_after_instruction'], 0)
            self.assertEqual(provenance['context_implementation_sha256'],
                             file_sha256(Path(repair.__file__).with_name('compact_context.py')))

    def test_small_context_keeps_native_records_without_decoder_overhead(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, reports, edits = preflight_fixture(directory)
            bundle['shared_scene']['custom_objects'] = []
            bundle['shared_scene']['base_scene']['objects'] = bundle['shared_scene']['base_scene']['objects'][:1]
            atomic_json(directory / 'scene_bundle.json', bundle)
            reports['geometry_preflight.json']['spec_sha256'] = file_sha256(directory / 'scene_bundle.json')
            atomic_json(directory / 'geometry_preflight.json', reports['geometry_preflight.json'])
            with patch.object(repair, 'call_model', return_value=edits) as call:
                repair.technical_repair(bundle, directory, reports, CONFIG)
            self.assertNotIn(RECORD_TABLE_INSTRUCTION, call.call_args.args[0])
            provenance = read_json(directory / 'technical_repair_provenance.json')
            self.assertEqual(provenance['scene_bundle_compaction']['encoded_arrays'], 0)
            self.assertEqual(provenance['scene_bundle_compaction']['saved_utf8_bytes'], 0)

    def test_changed_context_implementation_cannot_reuse_committed_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, reports, edits = preflight_fixture(directory)
            with patch.object(repair, 'call_model', return_value=edits):
                repair.technical_repair(bundle, directory, reports, CONFIG)
            real_hash = repair.file_sha256
            with patch.object(repair, 'file_sha256', side_effect=lambda path:
                              'f' * 64 if Path(path).name == 'compact_context.py' else real_hash(path)), \
                 patch.object(repair, 'call_model', side_effect=AssertionError('Changed context must not call API')):
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    repair.technical_repair(bundle, directory, reports, CONFIG)

    def test_originals_only_exact_hash_binding_and_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, reports, edits = preflight_fixture(directory)
            before = copy.deepcopy(bundle)
            with patch.object(repair, 'call_model', return_value=edits) as call:
                candidate = repair.technical_repair(bundle, directory, reports, CONFIG)
            self.assertEqual(bundle, before)
            self.assertNotEqual(candidate, bundle)
            self.assertEqual(call.call_args.args[1], [Path(v['source_image']) for v in bundle['views']])
            self.assertIn('There is no render and no visual score', call.call_args.args[0])
            self.assertIn('camera_shell_collision', call.call_args.args[0])
            self.assertNotIn(str(directory), call.call_args.args[0])
            self.assertEqual(call.call_args.kwargs['config'], repair.public_config(CONFIG))
            provenance = read_json(directory / 'technical_repair_provenance.json')
            self.assertEqual(provenance['input_spec_sha256'], file_sha256(directory / 'scene_bundle.json'))
            self.assertEqual(provenance['diagnostic_sha256']['geometry_preflight.json'], file_sha256(directory / 'geometry_preflight.json'))
            self.assertEqual(provenance['input_renders'], 0)
            self.assertFalse(provenance['manual_geometry_edits'])
            self.assertFalse(provenance['manual_visual_feedback'])
            self.assertTrue(provenance['candidate_requires_new_render_and_review'])
            self.assertFalse((directory / 'review.json').exists())
            with patch.object(repair, 'call_model', side_effect=AssertionError('Verified cache expected')):
                self.assertEqual(repair.technical_repair(bundle, directory, reports, CONFIG), candidate)

    def test_requires_current_saved_report_bound_to_exact_spec(self):
        for failure in ('missing_report', 'mismatch_report', 'missing_hash', 'wrong_hash', 'passed', 'no_errors', 'wrong_spec'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()
                bundle, reports, _ = preflight_fixture(directory)
                report = reports['geometry_preflight.json']
                if failure == 'missing_report':
                    (directory / 'geometry_preflight.json').unlink()
                elif failure == 'mismatch_report':
                    atomic_json(directory / 'geometry_preflight.json', {})
                elif failure == 'wrong_spec':
                    atomic_json(directory / 'scene_bundle.json', {})
                else:
                    if failure == 'missing_hash': report.pop('spec_sha256')
                    if failure == 'wrong_hash': report['spec_sha256'] = '0' * 64
                    if failure == 'passed': report['status'] = 'passed'
                    if failure == 'no_errors': report['errors'] = []
                    atomic_json(directory / 'geometry_preflight.json', report)
                with patch.object(repair, 'call_model', side_effect=AssertionError('Invalid evidence must not call API')):
                    with self.assertRaises(ValueError):
                        repair.technical_repair(bundle, directory, reports, CONFIG)

    def test_tampered_inputs_outputs_and_configuration_invalidate_commit(self):
        for target in ('source', 'spec_bytes', 'diagnostic', 'candidate', 'edits', 'config', 'new_asset_context'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()
                bundle, reports, edits = preflight_fixture(directory)
                with patch.object(repair, 'call_model', return_value=edits):
                    repair.technical_repair(bundle, directory, reports, CONFIG)
                config = copy.deepcopy(CONFIG)
                if target == 'source': Path(bundle['views'][0]['source_image']).write_bytes(b'changed')
                elif target == 'spec_bytes':
                    with (directory / 'scene_bundle.json').open('a') as stream: stream.write('\n')
                elif target == 'diagnostic':
                    reports['geometry_preflight.json']['warnings'] = ['new evidence']
                    atomic_json(directory / 'geometry_preflight.json', reports['geometry_preflight.json'])
                elif target == 'candidate': atomic_json(directory / 'technical_repaired_bundle.json', {})
                elif target == 'edits': atomic_json(directory / 'technical_repair.json', repair.empty_edits())
                elif target == 'config': config['model'] = 'other-explicit-model'
                elif target == 'new_asset_context': atomic_json(directory / 'asset_bindings.json', {})
                with patch.object(repair, 'call_model', side_effect=AssertionError('Changed cache must fail before API')):
                    with self.assertRaises(ValueError):
                        repair.technical_repair(bundle, directory, reports, config)

    def test_mutation_during_call_cannot_commit(self):
        for target in ('source', 'spec', 'report', 'assets'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()
                bundle, reports, edits = preflight_fixture(directory)
                def mutate(*args, **kwargs):
                    path = {'source': Path(bundle['views'][0]['source_image']), 'spec': directory / 'scene_bundle.json',
                            'report': directory / 'geometry_preflight.json', 'assets': directory / 'asset_selection.json'}[target]
                    path.write_bytes(b'changed')
                    return edits
                with patch.object(repair, 'call_model', side_effect=mutate):
                    with self.assertRaisesRegex(RuntimeError, 'input changed'):
                        repair.technical_repair(bundle, directory, reports, CONFIG)
                self.assertFalse((directory / 'technical_repair_provenance.json').exists())

    def test_token_truncation_retries_within_same_budget_and_keeps_usage(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, reports, edits = preflight_fixture(directory)
            seen = []
            def model(prompt, images, schema, output, timeout, *args, **kwargs):
                seen.append((prompt, timeout))
                atomic_json(output.with_suffix('.api.json'), {'usage': {'total_tokens': 5}})
                if len(seen) == 1:
                    raise ModelAPIError('output_truncated', 'token limit')
                return edits
            with patch.object(repair, 'call_model', side_effect=model):
                repair.technical_repair(bundle, directory, reports, CONFIG, validation_retries=1)
            self.assertEqual(len(seen), 2)
            self.assertLess(seen[1][1], seen[0][1])
            self.assertIn('token budget', seen[1][0])
            self.assertEqual(len(list(directory.glob('*.api.json'))), 2)
            provenance = read_json(directory / 'technical_repair_provenance.json')
            self.assertEqual([a['status'] for a in provenance['attempts']], ['rejected', 'validated'])

    def test_exhausted_semantic_retries_never_write_candidate_or_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, reports, _ = preflight_fixture(directory)
            edits = repair.empty_edits(); edits['remove_groups'] = ['unknown-untrusted-id']
            with patch.object(repair, 'call_model', return_value=edits) as call:
                with self.assertRaisesRegex(repair.RepairValidationError, 'exhausted'):
                    repair.technical_repair(bundle, directory, reports, CONFIG, validation_retries=1)
            self.assertEqual(call.call_count, 2)
            self.assertNotIn('unknown-untrusted-id', call.call_args.args[0])
            for name in ('technical_repair_provenance.json', 'technical_repaired_bundle.json', 'review.json'):
                self.assertFalse((directory / name).exists())

    def test_transport_error_is_not_retried_and_attempt_number_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, reports, edits = preflight_fixture(directory)
            with patch.object(repair, 'call_model', side_effect=ModelAPIError('http_error', 'unavailable')) as call:
                with self.assertRaises(ModelAPIError):
                    repair.technical_repair(bundle, directory, reports, CONFIG)
            self.assertEqual(call.call_count, 1)
            first_prompt = (directory / 'technical_repair_attempt_00.prompt.txt').read_bytes()
            with patch.object(repair, 'call_model', return_value=edits) as call:
                repair.technical_repair(bundle, directory, reports, CONFIG)
            self.assertEqual(call.call_args.args[3].name, 'technical_repair_attempt_01.json')
            self.assertEqual((directory / 'technical_repair_attempt_00.prompt.txt').read_bytes(), first_prompt)


if __name__ == '__main__':
    unittest.main()
