"""No-network integration tests for asset-aware paired review and repair."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import asset_context
import automatic_repair as repair
import vision_multiview as vision
from batch import file_sha256
from test_automatic_repair import CONFIG, prepare


def write_asset_evidence(directory):
    bindings = {'schema_version': 1, 'instances': [{
        'object_id': 'desk', 'asset_id': 'desk_geometry_01',
        'material_variant_id': 'desk_material_02', 'uniform_scale': 1.05,
        'rotation_xyz_deg': [0, 0, 0]}]}
    selection = {'schema_version': 'auto_asset_binding_report_v1',
                 'total_objects': 2, 'bound_objects': 1, 'fallback_objects': 1,
                 'preview': False, 'objects': {
        'desk': {'status': 'bound', 'category': 'desk',
                 'target_dimensions_m': [1.2, .6, .75],
                 'fit': {'dimensions_m': [1.21, .61, .75]},
                 'support': 'floor', 'preview_only': False, 'reasons': [],
                 'candidates': [{'private_model_path': '/private/candidate.glb'}]},
        'chair': {'status': 'procedural_fallback', 'category': 'chair',
                  'reasons': ['no_eligible_candidates']}}}
    for name, value in [('asset_bindings.json', bindings), ('asset_selection.json', selection)]:
        (directory / name).write_text(json.dumps(value))
    return bindings, selection


def prepare_review(directory):
    bundle, assessment = prepare(directory)
    metadata = {'status': 'completed', 'render': {'samples': 128},
                'views': [], 'provenance': {'external_assets': [{
                    'object_id': 'desk', 'asset_id': 'desk_geometry_01',
                    'material_variant_id': 'desk_material_02'}]}}
    for view in bundle['views']:
        (directory / ('annotations_' + view['id'] + '.json')).write_text('{"warnings": []}')
        metadata['views'].append({'id': view['id'], 'width': view['width'],
                                 'height': view['height'], 'warning_count': 0})
    (directory / 'metadata.json').write_text(json.dumps(metadata))
    (directory / 'scene.blend').write_bytes(b'synthetic scene artifact')
    raw_review = copy.deepcopy(assessment)
    del raw_review['revised_bundle']
    return bundle, assessment, raw_review, metadata


class AssetContextTests(unittest.TestCase):
    def test_optional_evidence_requires_complete_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertEqual(asset_context.load_context(directory), ({}, {}))
            self.assertEqual(asset_context.snapshot_inputs(directory, include_metadata=True), {
                'asset_bindings.json': None, 'asset_selection.json': None, 'metadata.json': None})
            (directory / 'asset_bindings.json').write_text('{"instances": []}')
            with self.assertRaisesRegex(ValueError, 'both bindings and selection'):
                asset_context.load_context(directory)

    def test_context_exposes_bindings_and_fit_without_candidate_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bindings, _ = write_asset_evidence(directory)
            context, hashes = asset_context.load_context(directory)
            self.assertEqual(context['instances'], bindings['instances'])
            self.assertEqual(context['objects']['desk']['fit']['dimensions_m'], [1.21, .61, .75])
            self.assertEqual(context['objects']['chair']['reasons'], ['no_eligible_candidates'])
            self.assertEqual(context['coverage']['bound_objects'], 1)
            self.assertFalse(context['appearance_match_verified'])
            self.assertNotIn('candidates', context['objects']['desk'])
            self.assertNotIn('/private/', json.dumps(context))
            self.assertEqual(hashes, {name: file_sha256(directory / name) for name in
                                     ('asset_bindings.json', 'asset_selection.json')})

    def test_review_receives_selection_and_binds_both_sidecars_to_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, _, raw, _ = prepare_review(directory)
            bindings, _ = write_asset_evidence(directory)
            with patch.object(vision, '_invoke_model', return_value=raw) as invoke:
                result = vision.review(bundle, directory, final_only=True, model_config=CONFIG)
            prompt = invoke.call_args.args[0]
            payload = json.JSONDecoder().raw_decode(prompt.split(
                'UNTRUSTED CURRENT SCENE DATA (only identity/pairing is fixed):\n', 1)[1])[0]
            self.assertEqual(payload['asset_selection']['instances'], bindings['instances'])
            self.assertEqual(payload['asset_selection']['objects']['desk']['support'], 'floor')
            self.assertIn('does not recolor a bound library material', prompt)
            self.assertNotIn('private_model_path', prompt)
            provenance = json.loads((directory / 'review_provenance.json').read_text())
            for name in ('asset_bindings.json', 'asset_selection.json'):
                self.assertEqual(provenance['input_hashes'][name], file_sha256(directory / name))
            self.assertEqual(vision.load_review(directory, bundle), result)
            (directory / 'asset_selection.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Cached review input changed: asset_selection.json'):
                vision.load_review(directory, bundle)

    def test_review_rejects_asset_or_metadata_changes_during_api_call(self):
        for target in ('asset_bindings.json', 'asset_selection.json', 'metadata.json'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()
                bundle, _, raw, _ = prepare_review(directory)
                write_asset_evidence(directory)
                def mutate(*args, **kwargs):
                    (directory / target).write_text('{}')
                    return raw
                with patch.object(vision, '_invoke_model', side_effect=mutate):
                    with self.assertRaisesRegex(RuntimeError, 'Review input changed'):
                        vision.review(bundle, directory, final_only=True, model_config=CONFIG)
                self.assertFalse((directory / 'review_provenance.json').exists())

    def test_review_rejects_sidecars_created_during_api_call(self):
        for created in ('both', 'asset_bindings.json', 'asset_selection.json'):
            with self.subTest(created=created), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()
                bundle, _, raw, _ = prepare_review(directory)
                def mutate(*args, **kwargs):
                    if created == 'both':
                        write_asset_evidence(directory)
                    else:
                        (directory / created).write_text('{}')
                    return raw
                with patch.object(vision, '_invoke_model', side_effect=mutate):
                    with self.assertRaisesRegex(RuntimeError, 'Review input changed'):
                        vision.review(bundle, directory, final_only=True, model_config=CONFIG)
                self.assertFalse((directory / 'review_provenance.json').exists())

    def test_review_cache_freezes_absence_and_keeps_legacy_no_asset_cache(self):
        for legacy in (False, True):
            for created in ('both', 'asset_bindings.json', 'asset_selection.json'):
                with self.subTest(legacy=legacy, created=created), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary).resolve()
                    bundle, _, raw, _ = prepare_review(directory)
                    with patch.object(vision, '_invoke_model', return_value=raw):
                        expected = vision.review(bundle, directory, final_only=True, model_config=CONFIG)
                    provenance_path = directory / 'review_provenance.json'
                    provenance = json.loads(provenance_path.read_text())
                    self.assertEqual(provenance['asset_input_sha256'], {
                        'asset_bindings.json': None, 'asset_selection.json': None})
                    if legacy:
                        del provenance['asset_input_sha256']
                        provenance_path.write_text(json.dumps(provenance))
                    self.assertEqual(vision.load_review(directory, bundle), expected)
                    if created == 'both':
                        write_asset_evidence(directory)
                    else:
                        (directory / created).write_text('{}')
                    with self.assertRaisesRegex(ValueError, 'Cached review input changed: asset_'):
                        vision.load_review(directory, bundle)

    def test_repair_receives_selection_and_actual_import_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report, _, metadata = prepare_review(directory)
            bindings, _ = write_asset_evidence(directory)
            with patch.object(repair, 'call_model', return_value=repair.empty_edits()) as invoke:
                candidate = repair.repair(bundle, directory, report, CONFIG)
            prompt = invoke.call_args.args[0]
            payload = json.loads(prompt.split('UNTRUSTED SCENE AND ASSESSMENT DATA:\n', 1)[1])
            self.assertEqual(payload['asset_selection']['instances'], bindings['instances'])
            self.assertEqual(payload['external_assets'], metadata['provenance']['external_assets'])
            self.assertIn('cannot edit a library mesh, invent asset IDs', prompt)
            self.assertNotIn('private_model_path', prompt)
            provenance = json.loads((directory / 'repair_provenance.json').read_text())
            for name in ('asset_bindings.json', 'asset_selection.json', 'metadata.json'):
                self.assertEqual(provenance['asset_input_sha256'][name], file_sha256(directory / name))
            with patch.object(repair, 'call_model', side_effect=AssertionError('Use verified cache')):
                self.assertEqual(repair.repair(bundle, directory, report, CONFIG), candidate)

    def test_repair_rejects_cache_after_asset_or_metadata_changes(self):
        for target in ('asset_bindings.json', 'asset_selection.json', 'metadata.json'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()
                bundle, report, _, _ = prepare_review(directory)
                write_asset_evidence(directory)
                with patch.object(repair, 'call_model', return_value=repair.empty_edits()):
                    repair.repair(bundle, directory, report, CONFIG)
                # Preserve valid JSON so rejection exercises input binding, not decoding.
                original = json.loads((directory / target).read_text())
                original['test_change'] = True
                (directory / target).write_text(json.dumps(original))
                with patch.object(repair, 'call_model', side_effect=AssertionError('Reject before API')):
                    with self.assertRaisesRegex(ValueError, 'Cached repair inputs'):
                        repair.repair(bundle, directory, report, CONFIG)

    def test_repair_rejects_asset_or_metadata_changes_during_api_call(self):
        for target in ('asset_bindings.json', 'asset_selection.json', 'metadata.json'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()
                bundle, report, _, _ = prepare_review(directory)
                write_asset_evidence(directory)
                def mutate(*args, **kwargs):
                    (directory / target).write_text('{}')
                    return repair.empty_edits()
                with patch.object(repair, 'call_model', side_effect=mutate):
                    with self.assertRaisesRegex(RuntimeError, 'asset input changed during'):
                        repair.repair(bundle, directory, report, CONFIG)
                self.assertFalse((directory / 'repair_provenance.json').exists())
                self.assertFalse((directory / 'repaired_bundle.json').exists())

    def test_repair_rejects_optional_inputs_created_during_api_call(self):
        for created in ('both', 'asset_bindings.json', 'asset_selection.json', 'metadata.json'):
            with self.subTest(created=created), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()
                bundle, report = prepare(directory)
                def mutate(*args, **kwargs):
                    if created == 'both':
                        write_asset_evidence(directory)
                    else:
                        (directory / created).write_text('{}')
                    return repair.empty_edits()
                with patch.object(repair, 'call_model', side_effect=mutate):
                    with self.assertRaisesRegex(RuntimeError, 'asset input changed during'):
                        repair.repair(bundle, directory, report, CONFIG)
                self.assertFalse((directory / 'repair_provenance.json').exists())
                self.assertFalse((directory / 'repaired_bundle.json').exists())

    def test_repair_cache_rejects_new_optional_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bundle, report = prepare(directory)
            with patch.object(repair, 'call_model', return_value=repair.empty_edits()):
                repair.repair(bundle, directory, report, CONFIG)
            provenance = json.loads((directory / 'repair_provenance.json').read_text())
            self.assertEqual(provenance['asset_input_sha256'], {
                'asset_bindings.json': None, 'asset_selection.json': None, 'metadata.json': None})
            (directory / 'metadata.json').write_text('{}')
            with patch.object(repair, 'call_model', side_effect=AssertionError('Reject before API')):
                with self.assertRaisesRegex(ValueError, 'Cached repair inputs'):
                    repair.repair(bundle, directory, report, CONFIG)


if __name__ == '__main__':
    unittest.main()
