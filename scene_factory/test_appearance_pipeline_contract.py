"""Cross-stage appearance contracts; no Blender or external model requests.

Synthetic native plan/detail responses exercise the complete staged assembly,
typed-edit transactions and comparison-evidence cache integrity. They do not
claim reconstruction fidelity.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import automatic_repair as repair
from batch import atomic_json, canonical_bytes, file_sha256, read_json
from multiview_spec import validate
import staged_analysis as staged
from test_automatic_repair import CONFIG, prepare
from test_staged_analysis import fixture_detail, fixture_plan


class AppearancePipelineContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.views = []
        for index in range(2):
            source = self.directory / f'original_{index}.png'
            Image.new('RGB', (80, 60), (70 + index * 30, 90, 100)).save(source)
            self.views.append({'id': f'view_{index}', 'source_image': str(source), 'width': 80, 'height': 60})
        self.plan = fixture_plan(self.views, group_count=1)
        self.plan['textures'][0]['quad'] = [[.2, .2], [.6, .2], [.6, .6], [.2, .6]]
        self.plan['assets'].append({'id': 'window', 'category': 'window',
                                   'd': [0, 7.9, 1, 2, .2, 1.2, 0, .6, .6, .6, 1, 1, 0, 0, .9]})
        self.plan['base_material_overrides'] = [{'object_id': 'desk', 'material_id': 'wood',
            'selector': {'role': 'body', 'material_name': '', 'slot_index': -1}}]
        self.plan['exterior_contexts'] = [{'window_id': 'window', 'texture_id': 'crop',
                                           'depth_m': 2, 'brightness_ev': 1.25}]
        self.detail = fixture_detail('group_0', kind='curtain')
        self.detail['parts'][0]['data'][3:6] = [1.2, .2, 1.8]
        self.detail['parts'][0]['cloth'] = [{'fold_count': 4, 'fold_depth': .12, 'sag': .025,
            'thickness': .003, 'segments_x': 24, 'segments_z': 12, 'seed': 17}]

    def bundle(self):
        return staged.to_bundle(self.plan, self.views, 'appearance_fixture', {'group_0': self.detail})[0]

    def test_native_staged_analysis_preserves_bindings_exterior_and_cloth_with_strict_validation(self):
        requests = []
        def model(prompt, images, schema, output, timeout, effort, config):
            is_detail = 'group_id' in schema['properties']
            response = copy.deepcopy(self.detail if is_detail else self.plan)
            if not is_detail and 'groups' in schema['properties']:
                group_properties = schema['properties']['groups']['items']['properties']
                if 'evidence_view_ids' not in group_properties:
                    for group in response.get('groups', []):
                        for key in ('evidence_view_ids', 'evidence_confidence',
                                    'texture_material_ids', 'texture_material_confidence'):
                            group.pop(key, None)
            requests.append((prompt, list(images), copy.deepcopy(schema)))
            atomic_json(output.with_suffix('.raw.json'), response)
            atomic_json(output.with_suffix('.api.json'), {
                'status': 'completed', 'config': staged.public_config(config),
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'schema_sha256': hashlib.sha256(canonical_bytes(schema)).hexdigest(),
                'image_sha256': [file_sha256(path) for path in images]})
            return response
        output = self.directory/'analysis'/'initial_bundle.json'
        with patch.object(staged, 'call_model', side_effect=model), \
             patch('vision_multiview.call_model', side_effect=AssertionError('No CLI fallback')):
            bundle = staged.analyze('appearance_fixture', self.views, output, model_config=CONFIG,
                options={'detail_workers': 1, 'validation_retries': 0})
        scene = validate(bundle)['shared_scene']
        self.assertEqual(scene['base_material_overrides'], self.plan['base_material_overrides'])
        self.assertEqual(scene['exterior_contexts'], self.plan['exterior_contexts'])
        part = scene['custom_objects'][0]['parts'][0]
        self.assertEqual(part['type'], 'curtain')
        self.assertEqual(part['cloth'], self.detail['parts'][0]['cloth'])
        self.assertEqual(part['size'], [1.2, .2, 1.8])
        self.assertEqual(part['position'], [.12, -.03, .9])
        self.assertEqual(bundle['texture_sources'], [{'texture_id': 'crop', 'view_id': 'view_1'}])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0][1], requests[1][1])
        self.assertEqual(read_json(output), bundle)
        self.assertFalse(read_json(output.parent/'analysis_provenance.json')['manual_scene_seed'])
        # Unsafe references, malformed nesting and excessive cloth folds must
        # fail at the same native/expanded validators used by real API output.
        for name in ('unknown_binding', 'unknown_window', 'whole_photo', 'missing_selector', 'cloth_envelope'):
            plan, detail = copy.deepcopy(self.plan), copy.deepcopy(self.detail)
            if name == 'unknown_binding':
                plan['base_material_overrides'][0]['object_id'] = 'missing'
            elif name == 'unknown_window':
                plan['exterior_contexts'][0]['window_id'] = 'desk'
            elif name == 'whole_photo':
                plan['textures'][0]['quad'] = [[0, 0], [1, 0], [1, 1], [0, 1]]
            elif name == 'missing_selector':
                del plan['base_material_overrides'][0]['selector']['role']
            else:
                detail['parts'][0]['cloth'][0]['fold_depth'] = .5
            with self.subTest(name=name), self.assertRaises(ValueError):
                staged.to_bundle(plan, self.views, 'appearance_fixture', {'group_0': detail})

    def test_typed_singleton_arrays_preserve_replace_and_clear_without_mutating_source(self):
        bundle = self.bundle()
        original = copy.deepcopy(bundle)
        unchanged = repair.apply_edits(bundle, repair.empty_edits())
        self.assertEqual(unchanged, original)
        edits = repair.empty_edits('Replace explicitly bound appearance.')
        bindings = copy.deepcopy(self.plan['base_material_overrides'])
        bindings[0]['selector']['role'] = 'all'
        contexts = copy.deepcopy(self.plan['exterior_contexts'])
        contexts[0]['brightness_ev'] = .5
        edits['base_material_overrides'], edits['exterior_contexts'] = [bindings], [contexts]
        revised = repair.apply_edits(bundle, edits)
        self.assertEqual(revised['shared_scene']['base_material_overrides'], bindings)
        self.assertEqual(revised['shared_scene']['exterior_contexts'], contexts)
        clear = repair.empty_edits('Explicitly clear both collections.')
        clear['base_material_overrides'] = [[]]
        clear['exterior_contexts'] = [[]]
        cleared = repair.apply_edits(revised, clear)
        self.assertEqual(cleared['shared_scene']['base_material_overrides'], [])
        self.assertEqual(cleared['shared_scene']['exterior_contexts'], [])
        self.assertEqual(bundle, original)
        self.assertEqual(cleared['shared_scene']['custom_objects'], original['shared_scene']['custom_objects'])
        for key in ('base_material_overrides', 'exterior_contexts'):
            malformed = repair.empty_edits()
            malformed[key] = [[], []]
            with self.subTest(key=key), self.assertRaises(repair.RepairValidationError):
                repair.apply_edits(bundle, malformed)

    def test_typed_removal_rejects_dangling_binding_until_same_transaction_clears_it(self):
        for object_id, reference_key in (('desk', 'base_material_overrides'), ('window', 'exterior_contexts')):
            bundle = self.bundle()
            original = copy.deepcopy(bundle)
            edits = repair.empty_edits()
            edits['remove_base_objects'] = [object_id]
            with self.subTest(object_id=object_id), self.assertRaises(repair.RepairValidationError):
                repair.apply_edits(bundle, edits)
            self.assertEqual(bundle, original)
            edits[reference_key] = [[]]
            revised = repair.apply_edits(bundle, edits)
            self.assertNotIn(object_id, {item['id'] for item in revised['shared_scene']['base_scene']['objects']})
            self.assertEqual(revised['shared_scene'][reference_key], [])
            validate(revised)

    def prepare_api_inputs(self):
        bundle, report = prepare(self.directory)
        # The shared repair fixture normally uses opaque hash inputs. Supply
        # real images here so paired-source evidence is valid image content.
        for index, view in enumerate(bundle['views']):
            Image.new('RGB', (64, 48), (40 + index * 40, 100, 120)).save(view['source_image'])
            Image.new('RGB', (64, 48), (70 + index * 30, 90, 110)).save(self.directory/f'rgb_{view["id"]}.png')
        feedback = {'selected_iteration': 'iteration_00', 'candidates': [
            {'challenger_iteration': 'iteration_01', 'selected': False, 'reason': 'photometric_regression'}]}
        atomic_json(self.directory/'candidate_feedback.json', feedback)
        return bundle, report, feedback

    def test_candidate_feedback_is_sent_hash_bound_and_changed_cache_is_rejected(self):
        bundle, report, feedback = self.prepare_api_inputs()
        with patch.object(repair, 'call_model', return_value=repair.empty_edits()) as call:
            self.assertEqual(repair.repair(bundle, self.directory, report, CONFIG), bundle)
        payload = json.JSONDecoder().raw_decode(call.call_args.args[0].split('UNTRUSTED SCENE AND ASSESSMENT DATA:\n')[1])[0]
        self.assertEqual(payload['candidate_feedback'], feedback)
        provenance = read_json(self.directory/'repair_provenance.json')
        self.assertEqual(provenance['candidate_feedback_sha256'], file_sha256(self.directory/'candidate_feedback.json'))
        with patch.object(repair, 'call_model', side_effect=AssertionError('Unchanged evidence must reuse cache')):
            self.assertEqual(repair.repair(bundle, self.directory, report, CONFIG), bundle)
        original_edits = (self.directory/'repair.json').read_bytes()
        feedback['candidates'][0]['reason'] = 'different_evidence'
        atomic_json(self.directory/'candidate_feedback.json', feedback)
        with patch.object(repair, 'call_model', side_effect=AssertionError('Changed cache must fail before API')):
            with self.assertRaisesRegex(ValueError, 'Cached repair inputs'):
                repair.repair(bundle, self.directory, report, CONFIG)
        self.assertEqual((self.directory/'repair.json').read_bytes(), original_edits)

    def test_candidate_feedback_mutated_during_model_call_cannot_commit_repair(self):
        bundle, report, feedback = self.prepare_api_inputs()
        def changed(*args, **kwargs):
            feedback['selected_iteration'] = 'iteration_99'
            atomic_json(self.directory/'candidate_feedback.json', feedback)
            return repair.empty_edits()
        with patch.object(repair, 'call_model', side_effect=changed) as call:
            with self.assertRaisesRegex(RuntimeError, 'input changed'):
                repair.repair(bundle, self.directory, report, CONFIG)
        self.assertEqual(call.call_count, 1)
        self.assertFalse((self.directory/'repair_provenance.json').exists())
        self.assertFalse((self.directory/'repaired_bundle.json').exists())
        self.assertFalse((self.directory/'repair.json').exists())


if __name__ == '__main__':
    unittest.main()
