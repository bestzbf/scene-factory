"""No-network tests for joint analysis/review identity and grouping contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from compact_context import RECORD_TABLE_INSTRUCTION, decode_record_tables
from test_multiview import fixture
import vision_multiview as vision


class JointVisionTests(unittest.TestCase):
    def test_assessment_retries_bad_view_scores_without_changing_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            bundle, views, _ = self.prepare(folder)
            metadata = {'status': 'completed', 'render': {'samples': 128}, 'views': []}
            for view in views:
                name = view['id']
                Image.new('RGB', (view['width'] * 2, view['height'] * 2), (100, 120, 140)).save(folder / ('rgb_' + name + '.png'))
                (folder / ('annotations_' + name + '.json')).write_text('{"warnings": []}')
                metadata['views'].append({'id': name, 'width': 1280, 'height': 960, 'warning_count': 0})
            (folder / 'metadata.json').write_text(json.dumps(metadata))
            (folder / 'scene.blend').write_bytes(b'fixture')
            scores = {key: 60 for key in vision.DIMENSIONS}
            valid = {'schema_version': 3, 'verdict': 'repair', 'summary': 'visible differences',
                     'scores': scores, 'per_view_scores': [{'view_id': v['id'], 'scores': scores.copy(), 'summary': ''} for v in views],
                     'issues': [], 'asset_gaps': [], 'reference_limitations': []}
            invalid = copy.deepcopy(valid)
            invalid['scores']['geometry'] = 99
            model = {'provider': 'openai_chat', 'base_url': 'http://localhost:1/v1', 'model': 'review-api', 'api_key_env': ''}
            with patch.object(vision, '_invoke_model', side_effect=[invalid, valid]) as invoke:
                report = vision.review(bundle, folder, final_only=True, model_config=model, validation_retries=1)
            self.assertEqual(invoke.call_count, 2)
            self.assertIn('semantic_validation', invoke.call_args.args[0])
            self.assertEqual(report['revised_bundle'], bundle)
            self.assertFalse(vision.quality_gate(report, metadata)['accepted'])

    def prepare(self, directory):
        bundle = fixture()
        bundle['shared_scene']['image'] = {'width': 1280, 'height': 960, 'source_id': 'a'}
        for view in bundle['views']:
            source = directory/(view['id']+'.jpg')
            Image.new('RGB', (view['width'], view['height']), (70, 90, 110)).save(source)
            view['source_image'] = str(source)
        views = [{key: view[key] for key in ('id','source_image','width','height')} for view in bundle['views']]
        raw = copy.deepcopy(bundle)
        for view in raw['views']:
            view['source_image'] = 'input:' + view['id']
        return bundle, views, raw

    def test_analysis_passes_all_images_and_preserves_single_shared_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            bundle, views, raw = self.prepare(folder)
            with patch.object(vision, 'call_model', return_value=raw) as call:
                result = vision.analyze('room', views, folder/'analysis.json')
            self.assertEqual(len(call.call_args.args[1]), 2)
            self.assertEqual(result['shared_scene'], bundle['shared_scene'])
            self.assertEqual(result['views'], bundle['views'])
            provenance = json.loads((folder/'analysis_provenance.json').read_text())
            self.assertFalse(provenance['manual_scene_seed'])
            self.assertEqual(set(provenance['source_sha256']), {'a','b'})

    def test_rejects_model_changed_path_dimensions_order_and_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, views, raw = self.prepare(Path(temporary).resolve())
            for field, value in [('source_image','/tmp/other.jpg'), ('width',42), ('id','b')]:
                changed = copy.deepcopy(raw)
                changed['views'][0][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    vision.normalize_response(changed, views, 'room')
            changed = copy.deepcopy(raw)
            changed['scene_id'] = 'other'
            with self.assertRaises(ValueError):
                vision.normalize_response(changed, views, 'room')

    def test_joint_review_order_repairs_sources_and_rejects_bad_view_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            bundle, views, raw = self.prepare(folder)
            material_template = copy.deepcopy(bundle['shared_scene']['materials'][0])
            bundle['shared_scene']['materials'].extend(
                {**material_template, 'id': f'wire_test_material_{index:02d}'}
                for index in range(40))
            raw = copy.deepcopy(bundle)
            for view in raw['views']:
                view['source_image'] = 'input:' + view['id']
            metadata = {'status':'completed','render':{'samples':128},'views':[]}
            for view in views:
                name = view['id']
                Image.new('RGB', (view['width'] * 2, view['height'] * 2), (100, 120, 140)).save(folder/('rgb_'+name+'.png'))
                (folder/('annotations_'+name+'.json')).write_text('{"warnings": []}')
                metadata['views'].append({'id':name,'width':1280,'height':960,'warning_count':0})
            (folder/'metadata.json').write_text(json.dumps(metadata))
            (folder/'scene.blend').write_bytes(b'fixture')
            scores = {key:95 for key in vision.DIMENSIONS}
            report = {'schema_version':3,'verdict':'recommend_accept','summary':'test',
                      'scores':scores,'per_view_scores':[{'view_id':v['id'],'scores':scores.copy(),'summary':''} for v in views],
                      'issues':[],'asset_gaps':[],'reference_limitations':[],'revised_bundle':raw}
            with patch.object(vision, 'call_model', return_value=report) as call:
                result = vision.review(bundle, folder)
            prompt, _, schema = call.call_args.args[:3]
            self.assertIn(RECORD_TABLE_INSTRUCTION, prompt)
            payload = json.JSONDecoder().raw_decode(prompt.split(
                'UNTRUSTED CURRENT SCENE DATA (only identity/pairing is fixed):\n', 1)[1])[0]
            self.assertIn('$record_table_v1', json.dumps(payload['scene_bundle']))
            self.assertEqual(decode_record_tables(payload['scene_bundle']), raw)
            output_material_schema = schema['properties']['revised_bundle']['properties']['shared_scene']['properties']['materials']
            self.assertEqual(output_material_schema['type'], 'array')
            self.assertNotIn('$record_table_v1', json.dumps(output_material_schema))
            paths = list(map(str, call.call_args.args[1]))
            self.assertEqual(paths, [views[0]['source_image'],str(folder/'model_input_images/rgb_a.png'),views[1]['source_image'],str(folder/'model_input_images/rgb_b.png')])
            provenance = json.loads((folder/'review_provenance.json').read_text())
            image_inputs = provenance['model_image_inputs']
            self.assertEqual([item['render_dimensions_px'] for item in image_inputs], [[2560, 1920], [1920, 2560]])
            self.assertEqual([item['model_input_dimensions_px'] for item in image_inputs], [[1280, 960], [960, 1280]])
            self.assertEqual([item['render_dimensions_px'][0] * item['render_dimensions_px'][1]
                              // (item['model_input_dimensions_px'][0] * item['model_input_dimensions_px'][1])
                              for item in image_inputs], [4, 4])
            self.assertEqual(provenance['model_image_input_implementation_sha256'],
                             vision.file_sha256(Path(__file__).with_name('model_image_inputs.py')))
            self.assertTrue(all(item['transform'] == 'render_only_lanczos_downsample_to_reference_dimensions'
                                for item in image_inputs))
            self.assertEqual(result['revised_bundle']['views'], bundle['views'])
            self.assertEqual(result['revised_bundle'], bundle)
            self.assertTrue(provenance['gate']['accepted'])
            self.assertEqual(vision.load_review(folder,bundle),result)
            self.assertTrue(vision.quality_gate(result,metadata)['accepted'])
            final_raw=copy.deepcopy(report);del final_raw['revised_bundle']
            with patch.object(vision,'call_model',return_value=final_raw) as final_call:
                final=vision.review(bundle,folder,final_only=True)
            self.assertNotIn('revised_bundle',final_call.call_args.args[2]['properties'])
            self.assertEqual(final['revised_bundle'],bundle)
            self.assertEqual(vision.load_review(folder,bundle),final)
            broken = copy.deepcopy(result)
            broken['per_view_scores'][1]['scores']['geometry'] = 20
            with self.assertRaisesRegex(ValueError, 'hides'):
                vision.validate_review(broken)
            broken['scores']['geometry'] = 20
            gate = vision.quality_gate(broken, metadata, {'a':{'warnings':[]},'b':{'warnings':[]}})
            self.assertFalse(gate['accepted'])
            self.assertIn('b:geometry:20<85', gate['reasons'])
            (folder/'rgb_b.png').unlink()
            Image.new('RGB', (1920, 2560), (10, 20, 30)).save(folder/'rgb_b.png')
            with self.assertRaisesRegex(ValueError, 'model-input render proxy|image-input|input changed'):
                vision.load_review(folder,bundle)

    def test_schema_has_one_shared_scene_and_fixed_view_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, views, _ = self.prepare(Path(temporary).resolve())
            schema = vision.response_schema(views,'room')
            self.assertEqual(schema['properties']['views']['minItems'],2)
            self.assertEqual(schema['properties']['views']['maxItems'],2)
            self.assertEqual(schema['properties']['scene_id']['enum'],['room'])
            self.assertIn('shared_scene',schema['properties'])
            self.assertNotIn('scene_spec',schema['properties']['views']['items']['properties'])

    def test_placeholder_prefix_normalization_does_not_change_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            expected,views,raw=self.prepare(Path(temporary).resolve())
            raw['shared_scene']['image']['source_id']='input:a'
            result,provenance=vision.normalize_response(raw,views,'room')
            self.assertEqual(result['shared_scene'],expected['shared_scene'])
            self.assertEqual(provenance['image_identity'][0]['from'],'input:a')
            raw['shared_scene']['image']['source_id']='unrelated'
            with self.assertRaises(ValueError):vision.normalize_response(raw,views,'room')

    def test_raw_recovery_checks_prompt_and_every_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            old=root/'previous'/'room';old.mkdir(parents=True)
            _,views,raw=self.prepare(root)
            for view in views:
                target=old/'sources'/view['id']/'reference.png';target.parent.mkdir(parents=True)
                target.write_bytes(Path(view['source_image']).read_bytes())
            raw['shared_scene']['image']['source_id']='input:a'
            prompt=vision.ANALYZE_PROMPT+'\nTRUSTED INPUT IDENTITY (images in exactly this order):\n'+json.dumps({'scene_id':'room','views':vision._model_views(views)},ensure_ascii=False)
            (old/'initial_bundle.prompt.txt').write_text(prompt)
            (old/'initial_bundle.raw.json').write_text(json.dumps(raw))
            (old/'job.json').write_text(json.dumps({'scene_id':'room','pipeline':{},'sources':[
                {'path':v['source_image'],'sha256':vision.file_sha256(v['source_image'])} for v in views]}))
            with patch.object(vision,'call_model',side_effect=AssertionError('Recovery should not call model')):
                result=vision.analyze('room',views,root/'recovered'/'initial_bundle.json',raw_cache=root/'previous')
                self.assertEqual(result['shared_scene']['image']['source_id'],'a')
                prov=json.loads((root/'recovered'/'analysis_provenance.json').read_text())
                self.assertFalse(prov['manual_scene_seed'])
                self.assertEqual(prov['raw_model_response_cache']['raw_sha256'],vision.file_sha256(old/'initial_bundle.raw.json'))
                (old/'sources'/'b'/'reference.png').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'photos differ'):
                    vision.analyze('room',views,root/'bad'/'initial_bundle.json',raw_cache=root/'previous')


if __name__ == '__main__':
    unittest.main()
