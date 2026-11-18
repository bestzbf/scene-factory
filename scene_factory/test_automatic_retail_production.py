"""Unattended retail contracts: semantic repair, receipts and shared profiles."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image
from batch import atomic_json, read_json
import library_layout
import library_repair
import local_codex
from production_retail import production_inputs, verify_inputs
from factory_config import normalize_config, load_config
from scene_budget import scene_budget_session, BudgetError
from test_library_layout import fixture, catalog_fixture
import vision_multiview as vision


class SharedProfileTests(unittest.TestCase):
    def test_accepts_only_shared_rules_and_detects_texture_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); Image.new('RGB', (8, 8)).save(root/'atlas.png')
            config = {'schema_version': 1, 'atlas': 'atlas.png',
                'defaults': {'fixture_defaults': {'top_stock': True}}, 'scenes': {}}
            atomic_json(root/'profile.json', config)
            record = production_inputs(root/'profile.json')
            verify_inputs(record)
            Image.new('RGB', (8, 8), 'red').save(root/'atlas.png')
            with self.assertRaisesRegex(ValueError, 'changed'): verify_inputs(record)
            for override in ({'scenes': {'one_store': {}}},
                             {'defaults': {'fixtures': {'shelf_01': {'kind': 'cup'}}}},
                             {'defaults': {'moves': []}},
                             {'defaults': {'fixture_defaults': {'kind': 'cup'}}}):
                atomic_json(root/'profile.json', {**config, **override})
                with self.assertRaises(ValueError): production_inputs(root/'profile.json')

    def test_local_config_is_idempotent_and_request_budget_covers_review(self):
        path = Path(__file__).resolve().parents[1]/'configs/retail/codex_production_v1.json'
        config = load_config(path)
        self.assertEqual(normalize_config(config), config)
        config['budget']['max_requests'] = 5
        with self.assertRaisesRegex(ValueError, '>= 8'): normalize_config(config)


class PhotoDetailTests(unittest.TestCase):
    def test_central_backing_is_only_used_for_freestanding_wide_shelves(self):
        from retail_photo_details import is_double_sided
        layout={'room':{'width':8,'depth':12},'architecture':{'omit_surfaces':[],'front_wall':True}}
        spec={'mode':'shelf','back_panel':True}
        frame={'dimensions_m':[5,.7,1.7],'rotation_z_deg':90}
        self.assertTrue(is_double_sided({'position':[0,6,0]},frame,spec,layout))
        self.assertFalse(is_double_sided({'position':[3.5,6,0]},frame,spec,layout))
        layout['retail_details']={'structures':[{'role':'wall_segment','center_m':[.5,6,1.5],
            'size_m':[.1,7,3],'rotation_z_deg':0}]}
        self.assertFalse(is_double_sided({'position':[0,6,0]},frame,spec,layout))

    def test_transposed_fixture_keeps_its_world_footprint_and_faces_inside(self):
        from retail_photo_details import fixture_build_frame
        from library_selection import _corners
        room={'depth':10}
        spec={'mode':'shelf','back_panel':True,'top_height_m':.15}
        for x,expected_turn in [(-3,90),(3,-90)]:
            entry={'target_dimensions_m':[.55,7.8,2.1], 'xy':[x,5], 'rotation_z_deg':0}
            frame=fixture_build_frame(entry,spec,room)
            self.assertEqual(frame['local_axis_turn_degrees'],expected_turn)
            self.assertAlmostEqual(frame['dimensions_m'][2],1.915)
            before=_corners(entry['xy'],entry['target_dimensions_m'],0)
            after=_corners(entry['xy'],frame['dimensions_m'],frame['rotation_z_deg'])
            self.assertEqual(sorted(tuple(round(v,6) for v in p) for p in before),
                             sorted(tuple(round(v,6) for v in p) for p in after))
        self.assertIsNone(fixture_build_frame({**entry,'target_dimensions_m':[2,8,2]},spec,room))

    def test_floor_contact_is_resolved_without_changing_supported_clearance(self):
        raw=fixture();raw['objects'][1]['clearance_m']=.0202
        result,changes=library_layout.canonicalize_model_layout(raw)
        library_layout.validate_layout(result)
        self.assertEqual(result['objects'][1]['clearance_m'],0)
        self.assertEqual(result['objects'][0]['clearance_m'],.001)
        self.assertEqual(changes[0]['from'],.0202)
        self.assertEqual(raw['objects'][1]['clearance_m'],.0202)

    def detail_layout(self):
        raw=fixture()
        raw['objects'][1]['category']='shelf'
        raw['retail_details']={
            'structures':[{'id':'lintel','role':'beam','center_m':[0,4,2.8],
                           'size_m':[3,.2,.3],'rotation_z_deg':0,'color':[.6,.6,.6]}],
            'fixtures':[{'object_id':'table','mode':'shelf','kind':'snack','levels':5,
                         'frame_color':[.65,.65,.65],'rail_color':[.03,.2,.03],
                         'back_panel':False,'endcaps':True,'stock_cell_m':.11,'top_height_m':.15}],
            'floor_tiles':[{'size_m':.6,'color':[.4,.4,.4],'grout_color':[.15,.15,.15],'roughness':.3}]}
        return raw

    def test_model_detail_expands_to_shared_geometry_without_changing_input(self):
        raw=self.detail_layout();before=copy.deepcopy(raw)
        schema=library_layout.schema_for(raw['views'],'room',retail_details=True)
        self.assertIn('retail_details',schema['required'])
        bundle=library_layout.expand_response(raw)
        beam=bundle['shared_scene']['custom_objects'][0]
        self.assertEqual(beam['position'],[0,4,2.8])
        self.assertEqual(beam['parts'][0]['size'],[3,.2,.3])
        self.assertEqual(raw,before)
        from retail_photo_details import inferred_settings
        settings=inferred_settings(raw,{'fixtures':{}})
        self.assertEqual(settings['fixtures']['table']['kind'],'snack')
        self.assertEqual(settings['floor_tiles']['size_m'],.6)

    def test_detail_rejects_unknown_fixtures_and_outside_architecture(self):
        raw=self.detail_layout();raw['retail_details']['fixtures'][0]['object_id']='absent'
        with self.assertRaisesRegex(ValueError,'Unknown or duplicate'):library_layout.validate_layout(raw)
        raw=self.detail_layout();raw['retail_details']['structures'][0]['center_m'][2]=4
        with self.assertRaisesRegex(ValueError,'exceeds'):library_layout.validate_layout(raw)
        raw=self.detail_layout();raw['objects'][1]['category']='structural_column'
        with self.assertRaisesRegex(ValueError,'shelf category'):library_layout.validate_layout(raw)


class LocalReceiptTests(unittest.TestCase):
    def test_cancelled_local_job_does_not_reserve_or_invoke(self):
        from runtime_controls import request_cancellation
        def cancelled():raise RuntimeError('stopped')
        with tempfile.TemporaryDirectory() as tmp, patch.object(local_codex,'identity',return_value={'provider':'fixture'}):
            root=Path(tmp);photo=root/'image';photo.write_bytes(b'image')
            with request_cancellation(cancelled), patch('vision_v2.call_model') as invoke:
                with self.assertRaisesRegex(RuntimeError,'stopped'):
                    local_codex.call_model('inspect',[photo],{},root/'response.json')
                invoke.assert_not_called()
                self.assertFalse((root/'response.codex.json').exists())

    def test_cross_run_import_reuses_only_matching_original_inference(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(local_codex, 'identity', return_value={'provider':'fixture'}):
            root=Path(tmp); image=root/'photo.png';Image.new('RGB',(32,24)).save(image)
            raw=fixture();raw['views'][0].update(width=32,height=24)
            views=[{**raw['views'][0],'source_image':str(image)}]
            source=root/'source'/'room'/'initial_bundle.json'
            target=root/'target'/'initial_bundle.json'
            with local_codex.session(True), patch.object(vision,'call_model',return_value=raw) as invoke:
                first=library_layout.analyze('room',views,source,catalog_fixture(),effort='high')
                replay=library_layout.analyze('room',views,target,catalog_fixture(),effort='high',raw_cache=root/'source')
                self.assertEqual(first,replay);self.assertEqual(invoke.call_count,1)
                provenance=read_json(target.parent/'analysis_provenance.json')
                self.assertFalse(provenance['raw_model_response_cache']['verified_import']['manual_scene_seed'])
                image.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    library_layout.analyze('room',views,root/'changed'/'initial_bundle.json',
                        catalog_fixture(),effort='high',raw_cache=root/'source')
                self.assertEqual(invoke.call_count,1)

    def test_completed_response_reused_and_mutated_images_blocked(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(local_codex, 'identity', return_value={'provider':'fixture'}):
            root = Path(tmp); image = root/'photo.png'; image.write_bytes(b'original')
            schema = {'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}
            with scene_budget_session(root, {'max_requests': 1}), patch('vision_v2.call_model', return_value={'ok':True}) as invoke:
                a = local_codex.call_model('inspect', [image], schema, root/'response.json')
                b = local_codex.call_model('inspect', [image], schema, root/'response.json')
                self.assertEqual(a, b); self.assertEqual(invoke.call_count, 1)
                with self.assertRaises(BudgetError):
                    local_codex.call_model('inspect', [image], schema, root/'second.json')
            image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                local_codex.call_model('inspect', [image], schema, root/'response.json')

    def test_interrupted_response_does_not_automatically_call_again(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(local_codex, 'identity', return_value={'provider':'fixture'}):
            root = Path(tmp); image=root/'image';image.write_bytes(b'image')
            schema={'type':'object','properties':{},'required':[],'additionalProperties':False}
            with patch('vision_v2.call_model', side_effect=RuntimeError('interrupted')) as invoke:
                with self.assertRaises(RuntimeError):local_codex.call_model('inspect',[image],schema,root/'response.json')
                with self.assertRaisesRegex(RuntimeError,'incomplete'):local_codex.call_model('inspect',[image],schema,root/'response.json')
                self.assertEqual(invoke.call_count,1)


class SemanticRepairTests(unittest.TestCase):
    def test_selection_patch_preserves_valid_objects_room_and_all_cameras(self):
        raw=fixture();extra=copy.deepcopy(raw['objects'][1]);extra['id']='unrelated';extra['xy']=[2,4];raw['objects'].append(extra)
        schema,editable,structures=library_repair.selection_patch_contract(raw,
            {'asset_selection_failure':{'errors':[{'code':'no_verified_support_surface','object_id':'remote','support_id':'table'}]}})
        updated=copy.deepcopy(raw['objects'][0]);updated['xy']=[.1,3]
        fixed=library_repair.apply_selection_patch(raw,{'object_updates':[updated],'structure_updates':[],'fixture_updates':[],'reason':'contact'},editable,structures)
        self.assertEqual(fixed['objects'][2],extra);self.assertEqual(fixed['views'],raw['views']);self.assertEqual(fixed['room'],raw['room'])
        with self.assertRaisesRegex(ValueError,'unrelated'):
            library_repair.apply_selection_patch(raw,{'object_updates':[extra],'structure_updates':[],'fixture_updates':[],'reason':''},editable,structures)

    def test_selection_patch_can_add_shared_builder_detail_only_to_failed_fixtures(self):
        raw=PhotoDetailTests().detail_layout()
        detail=copy.deepcopy(raw['retail_details']['fixtures'][0]);raw['retail_details']['fixtures']=[]
        schema,editable,structures=library_repair.selection_patch_contract(raw,
            {'asset_selection_failure':{'errors':[{'object_id':'table'}]}})
        payload={'object_updates':[], 'structure_updates':[], 'fixture_updates':[detail], 'reason':'long physical shelf'}
        fixed=library_repair.apply_selection_patch(raw,payload,editable,structures)
        self.assertEqual(fixed['retail_details']['fixtures'],[detail])
        self.assertEqual(fixed['objects'],raw['objects'])
        self.assertEqual(fixed['views'],raw['views'])
        detail['object_id']='unrelated'
        with self.assertRaisesRegex(ValueError,'unrelated'):
            library_repair.apply_selection_patch(raw,payload,editable,structures)

    def test_selection_patch_rejects_malformed_payloads_without_mutating_layout(self):
        raw=fixture();before=copy.deepcopy(raw)
        schema,editable,structures=library_repair.selection_patch_contract(raw,
            {'asset_selection_failure':{'errors':[{'object_id':'remote'}]}})
        for malformed in (None, raw, {'object_updates':[], 'structure_updates':[], 'fixture_updates':[], 'reason':'', 'views':[]},
                          {'object_updates':[{'id':'remote'}], 'structure_updates':[], 'fixture_updates':[], 'reason':''}):
            with self.subTest(payload=malformed), self.assertRaises(ValueError):
                library_repair.apply_selection_patch(raw,malformed,editable,structures)
        self.assertEqual(raw,before)

    def test_camera_and_layout_repair_preserves_sources_and_reuses_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/'photo.png';Image.new('RGB',(32,24)).save(image)
            raw=fixture();raw['views'][0].update(width=32,height=24)
            views=[{**raw['views'][0], 'source_image':str(image)}]
            bundle=vision.normalize_response(library_layout.expand_response(raw),views,'room')[0]
            revised=copy.deepcopy(raw);revised['views'][0]['camera']['position'][0]=.2
            revised['objects'][0]['xy'][1]=3.1;revised['objects'][1]['xy'][1]=3.1
            config={'provider':'openai_chat','model':'fixture','base_url':'http://127.0.0.1:1/v1','api_key_env':''}
            with patch.object(vision,'_analysis_request',return_value=(revised,{})) as invoke:
                a,layout=library_repair.repair(raw,bundle,root,{'failure':'bounds'},catalog_fixture(),config,phase='technical')
                self.assertEqual(a['views'][0]['source_image'],str(image.resolve()))
                self.assertEqual(a['views'][0]['camera']['position'][0],.2)
                self.assertEqual(layout['objects'][0]['support_id'],'table')
                b,_=library_repair.repair(raw,bundle,root,{'failure':'bounds'},catalog_fixture(),config,phase='technical')
                self.assertEqual(a,b);self.assertEqual(invoke.call_count,1)
                changed=copy.deepcopy(raw);changed['objects'][0]['xy'][0]=.1
                with self.assertRaisesRegex(ValueError,'inputs changed'):
                    library_repair.repair(changed,bundle,root,{'failure':'bounds'},catalog_fixture(),config,phase='technical')

    def test_invalid_source_identity_is_never_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'photo.png';Image.new('RGB',(1,1)).save(image)
            raw=fixture();views=[{**raw['views'][0],'source_image':str(image)}]
            bundle=vision.normalize_response(library_layout.expand_response(raw),views,'room')[0]
            wrong=copy.deepcopy(raw);wrong['views'][0]['source_image']='input:other'
            config={'provider':'openai_chat','model':'fixture','base_url':'http://127.0.0.1:1/v1','api_key_env':''}
            with patch.object(vision,'_analysis_request',return_value=(wrong,{})):
                with self.assertRaises(ValueError):library_repair.repair(raw,bundle,root,{},catalog_fixture(),config,phase='technical')
            self.assertFalse((root/'technical_repair_provenance.json').exists())


class SemanticBatchTests(unittest.TestCase):
    def setUp(self):
        from test_auto_asset_batch import AutomaticAssetBatchTests
        self.save_catalog = lambda: atomic_json(self.catalog_path, self.library.data)
        AutomaticAssetBatchTests.setUp(self)

    def test_batch_reselects_rebuilds_reviews_and_resumes_without_model_calls(self):
        import batch_multiview as runner
        from test_batch_multiview import GroupedBatchTests
        from test_auto_asset_batch import AutomaticAssetBatchTests
        from batch import file_sha256
        self.args.max_repairs=1;self.args.no_review=False
        self.args.pipeline_options={'scene_wire_format':'library_layout_v1','repair_mode':'library_layout','validation_retries':0}
        model={'provider':'openai_chat','model':'fixture','base_url':'http://127.0.0.1:1/v1','api_key_env':''}
        self.args.model_configs={role:model for role in ('analysis','review','repair')}
        raw_holder={}
        def analyze(scene_id,views,output,**kwargs):
            raw=fixture();raw['scene_id']=scene_id
            raw['views']=[{**v,'source_image':'input:'+v['id'], 'camera':copy.deepcopy(raw['views'][0]['camera'])} for v in views]
            raw['objects']=[{'id':'chair','category':'chair','xy':[0,2],
                'target_dimensions_m':[.5,.6,.9],'rotation_z_deg':0,'support_id':'floor','clearance_m':0}]
            result=vision.normalize_response(library_layout.expand_response(raw),views,scene_id)[0]
            raw_holder['raw']=raw
            atomic_json(output,result);atomic_json(output.parent/'initial_layout.json',raw)
            atomic_json(output.parent/'analysis_provenance.json',{'bundle_sha256':file_sha256(output),
                'layout_sha256':file_sha256(output.parent/'initial_layout.json')})
            return result
        def request(*args,**kwargs):
            raw=copy.deepcopy(raw_holder['raw']);raw['objects'][0]['xy'][0]=.25
            return raw,{}
        def review(bundle,directory,**kwargs):
            self.assertTrue(kwargs['final_only'])
            report=AutomaticAssetBatchTests.review_report(self,bundle,score=65 if directory.name=='iteration_00' else 80)
            atomic_json(directory/'review.json',report)
            return report
        group=runner.load_groups(self.groups_path)[0]
        with patch.object(library_layout,'analyze',side_effect=analyze) as analysis, \
             patch.object(runner,'run_stage',side_effect=lambda *a:GroupedBatchTests.fake_render(self,*a)) as render, \
             patch.object(vision,'review',side_effect=review) as reviewer, \
             patch.object(vision,'_analysis_request',side_effect=request) as repair:
            result=runner.run_job(group,self.args,self.pipeline)
            self.assertEqual(result['status'],'needs_review',result)
            self.assertEqual((analysis.call_count,render.call_count,reviewer.call_count,repair.call_count),(1,2,2,1))
            revised=read_json(self.args.output_dir/'lab/iteration_01/library_layout.json')
            self.assertEqual(revised['objects'][0]['xy'][0],.25)
            replay=runner.run_job(group,self.args,self.pipeline)
            self.assertTrue(replay['resumed'])
            self.assertEqual((analysis.call_count,render.call_count,reviewer.call_count,repair.call_count),(1,2,2,1))


if __name__ == '__main__': unittest.main()
