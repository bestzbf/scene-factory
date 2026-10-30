"""Focused grouped-batch contracts; no photo inference or Blender is invoked.

Synthetic tiny images and mocked model/renderer establish grouping, source
association and fail-closed resume semantics independently of visual quality.
"""
from __future__ import annotations
import copy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import batch_multiview as runner
from batch import atomic_json, file_sha256, read_json
from multiview_spec import validate
from spec_v2 import upgrade_v1
import vision_multiview as vision
try:
    from PIL import Image
except ImportError:
    Image = None


def bundle_for(scene_id, views):
    base = read_json(Path(__file__).parent/'examples/living_room.json')
    first = views[0]
    scene = upgrade_v1(base, first['width'], first['height'], first['id'])
    result = {'schema_version':3,'scene_id':scene_id,'shared_scene':scene,
        'architecture':{'omit_surfaces':[],'front_wall':False},
        'views':[{**view,'camera':{**copy.deepcopy(base['camera']),
            'roll_degrees':float(index)}} for index,view in enumerate(views)],
        'texture_sources':[]}
    return validate(result)


class LargeBatchBudgetTests(unittest.TestCase):
    def setUp(self):
        self.policy = {'max_requests': 500, 'max_output_tokens': 2000000,
                       'max_reported_tokens': 5000000}

    def test_budget_required_at_large_run_boundary_but_not_for_dry_run(self):
        runner.require_large_batch_budget(999, None)
        runner.require_large_batch_budget(1000, None, dry_run=True)
        with self.assertRaisesRegex(ValueError, '1,000 or more scenes'):
            runner.require_large_batch_budget(1000, None)
        runner.require_large_batch_budget(10000, self.policy)

    def test_all_three_aggregate_limits_must_be_positive_integers(self):
        for field in ('max_requests', 'max_output_tokens', 'max_reported_tokens'):
            policy = dict(self.policy)
            policy[field] = 0
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                runner.require_large_batch_budget(1000, policy)
        invalid = dict(self.policy, max_requests=True)
        with self.assertRaisesRegex(ValueError, 'max_requests'):
            runner.require_large_batch_budget(1000, invalid)

    def test_coordinator_checks_before_creating_output_or_starting_workers(self):
        args = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, 'batch_budget limits'):
            runner.coordinate([{}] * 1000, args, {})
        self.assertFalse(hasattr(args, 'stop_event'))

    def test_cli_rejects_unbudgeted_large_run_before_provider_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.png'
            source.write_bytes(b'synthetic placeholder; group loader checks file existence')
            groups = root / 'groups.json'
            atomic_json(groups, {'schema_version': 1, 'scenes': [
                {'scene_id': f'scene_{index}', 'images': [str(source)]}
                for index in range(1000)]})
            config = runner.ROOT / 'config_examples' / 'bigmodel-production.json'
            stderr = StringIO()
            with patch('model_api.preflight') as preflight, redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as failure:
                    runner.main(['--config', str(config), '--groups', str(groups),
                                 '--output-dir', str(root / 'output')])
            self.assertEqual(failure.exception.code, 2)
            self.assertIn('positive batch_budget limits', stderr.getvalue())
            preflight.assert_not_called()
            self.assertFalse((root / 'output').exists())


class SharedLedgerPathTests(unittest.TestCase):
    def test_output_must_be_inside_configured_shared_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(output_dir=root/'shards'/'one',
                                   shared_ledger_dir=root/'shards')
            self.assertEqual(runner._resolve_batch_ledger_root(args), (root/'shards').resolve())
            args.output_dir = root/'outside'
            with self.assertRaisesRegex(ValueError, 'must contain every batch output'):
                runner._resolve_batch_ledger_root(args)

    def test_default_ledger_root_is_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(output_dir=root/'output', shared_ledger_dir=None)
            self.assertEqual(runner._resolve_batch_ledger_root(args), (root/'output').resolve())

    def test_cli_resolves_shared_root_relative_to_config_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root/'factory.json'
            config.write_text(json.dumps({
                'schema_version': 1,
                'models': {'analysis': {'provider': 'openai_chat',
                    'base_url': 'https://models.example.test/v1', 'model': 'fixture',
                    'api_key_env': 'SCENE_TEST_KEY', 'response_format': 'json_object'}},
                'batch': {'shared_ledger_dir': 'shared-ledger'}
            }))
            source = root/'source.jpg'
            source.write_bytes(b'not decoded in dry-run')
            groups = root/'groups.json'
            atomic_json(groups, {'schema_version': 1, 'scenes': [
                {'scene_id': 'scene-a', 'images': [str(source)]}]})
            output = root/'shared-ledger'/'shard-a'
            with redirect_stdout(StringIO()):
                self.assertEqual(runner.main(['--config', str(config), '--groups', str(groups),
                                              '--output-dir', str(output), '--dry-run']), 0)
            self.assertFalse(output.exists())
            outside = root/'outside'
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as failure:
                runner.main(['--config', str(config), '--groups', str(groups),
                             '--output-dir', str(outside), '--dry-run'])
            self.assertEqual(failure.exception.code, 2)
            self.assertFalse(outside.exists())

    def test_large_dry_run_prints_bounded_summary_and_budget_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.png'
            source.write_bytes(b'synthetic placeholder; group loader checks file existence')
            groups = root / 'groups.json'
            atomic_json(groups, {'schema_version': 1, 'scenes': [
                {'scene_id': f'scene_{index}', 'images': [str(source)]}
                for index in range(1000)]})
            config = runner.ROOT / 'config_examples' / 'bigmodel-production.json'
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = runner.main(['--config', str(config), '--groups', str(groups),
                                      '--output-dir', str(root / 'output'), '--dry-run'])
            report = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(report['scenes'], 1000)
            self.assertEqual(report['images'], 1000)
            self.assertEqual(len(report['preview_scenes']), 20)
            self.assertEqual(report['omitted_scenes'], 980)
            self.assertEqual(report['live_run_budget_missing'], [
                'max_requests', 'max_output_tokens', 'max_reported_tokens'])
            self.assertFalse((root / 'output').exists())


@unittest.skipIf(Image is None,'Pillow required')
class GroupedBatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sources = []
        for i,color in enumerate([(220,15,5),(5,30,230),(20,180,20),(140,20,130)]):
            source = self.root/f'input_{i}.png'
            Image.new('RGB',(40,20),color).save(source)
            self.sources.append(source)
        self.groups_path = self.root/'groups.json'
        atomic_json(self.groups_path,{'schema_version':1,'scenes':[
            {'scene_id':'lab','images':['input_0.png','input_1.png']},
            {'scene_id':'corridor','images':['input_2.png']},
            {'scene_id':'dorm','images':['input_3.png']}]})
        self.args=SimpleNamespace(output_dir=self.root/'out',max_edge=128,samples=16,
            device='CPU',max_repairs=0,effort='low',vision_timeout=20,
            render_timeout=20,no_review=True,blender='mock-blender')
        self.pipeline={'test_runner':'sha-test'}

    def fake_analyze(self,scene_id,views,output,**kwargs):
        bundle=bundle_for(scene_id,views)
        atomic_json(output,bundle)
        atomic_json(output.parent/'analysis_provenance.json',
            {'bundle_sha256':file_sha256(output),'manual_scene_seed':False})
        return bundle

    def fake_render(self,command,log,timeout):
        out=Path(command[command.index('--output-dir')+1])
        bundle=read_json(Path(command[command.index('--spec')+1]))
        (out/'scene.blend').write_bytes(b'BLENDER-test-shared-scene'+b'\0'*1400)
        outputs=['scene.blend'];views=[]
        for view in bundle['views']:
            name='rgb_'+view['id']+'.png'
            Image.new('RGB',(128,64),(91,103,120)).save(out/name)
            annotation='annotations_'+view['id']+'.json'
            atomic_json(out/annotation,{'warnings':[]})
            outputs.extend([name,annotation])
            views.append({'id':view['id'],'render':name,'annotation':annotation,
                'width':128,'height':64,'warning_count':0})
        atomic_json(out/'metadata.json',{'status':'completed','geometry_build_count':1,
            'camera_count':len(views),'views':views,'outputs':outputs,
            'output_sha256':runner.artifact_hashes(out,outputs),
            'render':{'samples':16}})

    def run_mocked(self, group=None):
        group=group or runner.load_groups(self.groups_path)[0]
        with patch.object(vision,'analyze',side_effect=self.fake_analyze) as analyze,\
             patch.object(runner,'run_stage',side_effect=self.fake_render) as render:
            result=runner.run_job(group,self.args,self.pipeline)
        return result,analyze.call_count,render.call_count

    def test_pipeline_hashes_include_model_image_input_implementation(self):
        hashes = runner.pipeline_hashes()
        for name in ('model_image_inputs.py', 'compact_staged_detail.py', 'plan_geometry_patch.py', 'assets_lib/material_roles.py'):
            self.assertIn(name, hashes)
            self.assertEqual(hashes[name], file_sha256(Path(runner.__file__).parent / name))

    def test_2_plus_1_plus_1_runs_three_shared_scenes(self):
        groups=runner.load_groups(self.groups_path)
        self.assertEqual([len(g['images']) for g in groups],[2,1,1])
        self.assertEqual([g['scene_id'] for g in groups],['lab','corridor','dorm'])
        with patch.object(vision,'analyze',side_effect=self.fake_analyze) as analyze,\
             patch.object(runner,'run_stage',side_effect=self.fake_render) as render:
            results=[runner.run_job(g,self.args,self.pipeline) for g in groups]
        self.assertEqual([r['status'] for r in results],['generated_unreviewed']*3)
        self.assertEqual(analyze.call_count,3)
        self.assertEqual(render.call_count,3)
        for call in render.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index('--appearance-policy')+1], 'photo_reconstruction')
            self.assertEqual(command[command.index('--tone-match')+1], 'reference')
        lab_bundle=read_json(self.args.output_dir/'lab/initial_bundle.json')
        self.assertEqual(len(lab_bundle['views']),2)
        self.assertEqual(len(results[0]['views']),2)
        self.assertEqual([v['id'] for v in analyze.call_args_list[0].args[1]],['view_01','view_02'])
        self.assertEqual(len(list(self.args.output_dir.rglob('scene.blend'))),3)

    def test_crop_pixels_come_from_explicit_source_view(self):
        views=[{'id':f'view_{i+1:02d}','source_image':str(p),'width':40,'height':20}
            for i,p in enumerate(self.sources[:2])]
        bundle=bundle_for('lab',views)
        bundle['shared_scene']['textures']=[{'id':t,'quad':[[0,0],[1,0],[1,1],[0,1]]}
            for t in ('first','second')]
        bundle['texture_sources']=[{'texture_id':'first','view_id':'view_01'},
            {'texture_id':'second','view_id':'view_02'}]
        validate(bundle)
        manifest=read_json(runner.prepare_textures(bundle,self.root/'bundle.json',self.root/'crops'))
        records={item['id']:item for item in manifest['textures']}
        self.assertEqual(records['first']['source_view_id'],'view_01')
        self.assertEqual(records['second']['source_view_id'],'view_02')
        self.assertEqual(records['first']['source_sha256'],file_sha256(self.sources[0]))
        self.assertEqual(records['second']['source_sha256'],file_sha256(self.sources[1]))
        with Image.open(records['first']['path']) as crop:
            self.assertEqual(crop.getpixel((10,10)),(220,15,5))
        with Image.open(records['second']['path']) as crop:
            self.assertEqual(crop.getpixel((10,10)),(5,30,230))

    def test_resume_does_not_analyze_or_render_again(self):
        first,analyses,renders=self.run_mocked()
        self.assertEqual((analyses,renders),(1,1))
        self.assertEqual(first['status'],'generated_unreviewed')
        second,analyses,renders=self.run_mocked()
        self.assertTrue(second['resumed'])
        self.assertEqual((analyses,renders),(0,0))
        self.assertFalse((Path(first['directory'])/'.lock').exists())

    def test_tampered_blend_prevents_successful_resume(self):
        first,_,_=self.run_mocked()
        scene=Path(first['directory'])/first['iteration']/'scene.blend'
        scene.write_bytes(b'tampered')
        second,analyses,renders=self.run_mocked()
        self.assertIn(second['status'],('failed','conflict'))
        self.assertIn('changed or missing',second['error'])
        self.assertEqual((analyses,renders),(0,0))
        self.assertFalse((Path(first['directory'])/'.lock').exists())

    def test_changed_source_or_options_cannot_reuse_previous_run(self):
        first,_,_=self.run_mocked()
        self.args.samples=32
        changed,analyses,renders=self.run_mocked()
        self.assertIn(changed['status'],('failed','conflict'))
        self.assertIn('Input, code or options changed',changed['error'])
        self.assertEqual((analyses,renders),(0,0))
        self.args.samples=16
        Image.new('RGB',(40,20),(1,2,3)).save(self.sources[0])
        changed,_,_=self.run_mocked()
        self.assertIn(changed['status'],('failed','conflict'))
        self.assertIn('Input, code or options changed',changed['error'])

    def test_existing_lock_is_preserved(self):
        folder=self.args.output_dir/'lab';folder.mkdir(parents=True)
        (folder/'.lock').write_text('existing owner')
        result,analyses,renders=self.run_mocked()
        self.assertEqual(result['status'],'locked')
        self.assertEqual((analyses,renders),(0,0))
        self.assertEqual((folder/'.lock').read_text(),'existing owner')

    def test_failed_renderer_is_never_marked_complete(self):
        group=runner.load_groups(self.groups_path)[0]
        with patch.object(vision,'analyze',side_effect=self.fake_analyze),\
             patch.object(runner,'run_stage',side_effect=RuntimeError('test render failure')):
            result=runner.run_job(group,self.args,self.pipeline)
        self.assertEqual(result['status'],'failed')
        self.assertFalse((self.args.output_dir/'lab/result.json').exists())
        self.assertFalse((self.args.output_dir/'lab/.lock').exists())

    def test_cancel_during_asset_verification_prevents_blender_launch(self):
        group=runner.load_groups(self.groups_path)[0]
        stop=threading.Event()
        self.args.stop_event=stop

        def verify_then_stop(catalog, bindings_path):
            stop.set()

        with patch.object(vision,'analyze',side_effect=self.fake_analyze),\
             patch.object(runner,'verify_selected_asset_inputs',side_effect=verify_then_stop),\
             patch.object(runner,'run_stage',side_effect=self.fake_render) as render:
            result=runner.run_job(group,self.args,self.pipeline)
        self.assertEqual(result['status'],'interrupted')
        render.assert_not_called()

    def test_cancel_during_blender_command_setup_prevents_launch(self):
        group=runner.load_groups(self.groups_path)[0]
        stop=threading.Event()
        base_args=self.args
        base_args.stop_event=stop

        class ArgsProxy:
            def __getattr__(self, name):
                if name == 'blender':
                    stop.set()
                    return 'mock-blender'
                return getattr(base_args,name)

        with patch.object(vision,'analyze',side_effect=self.fake_analyze),\
             patch.object(runner,'run_stage',side_effect=self.fake_render) as render:
            result=runner.run_job(group,ArgsProxy(),self.pipeline)
        self.assertEqual(result['status'],'interrupted')
        render.assert_not_called()

    def test_hash_verification_rejects_path_escape(self):
        outside=self.root/'outside.bin';outside.write_bytes(b'same bytes')
        folder=self.root/'inside';folder.mkdir()
        with self.assertRaisesRegex(ValueError,'changed or missing'):
            runner.verify_hashes(folder,{'../outside.bin':file_sha256(outside)})

    def test_metadata_must_keep_every_camera_and_one_geometry_build(self):
        first,_,_=self.run_mocked()
        folder=Path(first['directory'])/first['iteration']
        bundle=read_json(folder/'scene_bundle.json');original=read_json(folder/'metadata.json')
        for key,value in [('camera_count',1),('geometry_build_count',2)]:
            metadata=copy.deepcopy(original);metadata[key]=value
            atomic_json(folder/'metadata.json',metadata)
            with self.assertRaises(ValueError):runner.validate_render(folder,bundle)

    def test_invalid_manifests(self):
        valid=read_json(self.groups_path)
        cases=[]
        bad=copy.deepcopy(valid);bad['schema_version']=2;cases.append(bad)
        bad=copy.deepcopy(valid);bad['scenes']=[];cases.append(bad)
        bad=copy.deepcopy(valid);bad['scenes'][0]['scene_id']='../escape';cases.append(bad)
        bad=copy.deepcopy(valid);bad['scenes'][1]['scene_id']='lab';cases.append(bad)
        bad=copy.deepcopy(valid);bad['scenes'][0]['images']=['input_0.png','input_0.png'];cases.append(bad)
        bad=copy.deepcopy(valid);bad['scenes'][0]['images']=[];cases.append(bad)
        bad=copy.deepcopy(valid);bad['scenes'][0]['images']=[None];cases.append(bad)
        bad=copy.deepcopy(valid);bad['scenes'][0]['seed_spec']='manual.json';cases.append(bad)
        bad=copy.deepcopy(valid);bad['scenes'][0]['images']=['missing.jpg'];cases.append(bad)
        for index,data in enumerate(cases):
            with self.subTest(case=index):
                atomic_json(self.groups_path,data)
                with self.assertRaises((ValueError,FileNotFoundError)):
                    runner.load_groups(self.groups_path)

    def test_bad_photo_is_per_scene_failure(self):
        self.sources[0].write_bytes(b'not an image')
        result,analyses,renders=self.run_mocked()
        self.assertEqual(result['status'],'failed')
        self.assertEqual((analyses,renders),(0,0))

    def test_one_bad_photo_does_not_abort_other_scene_jobs(self):
        self.sources[0].write_bytes(b'not an image')
        with patch.object(vision,'analyze',side_effect=self.fake_analyze), \
             patch.object(runner,'run_stage',side_effect=self.fake_render), \
             patch.object(runner,'pipeline_hashes',return_value=self.pipeline), \
             patch.object(runner,'discover_blender',return_value='mock-blender'):
            code=runner.main(['--groups',str(self.groups_path),'--output-dir',str(self.args.output_dir),
                '--no-review','--workers','3','--max-edge','128','--samples','16'])
        self.assertEqual(code,1)
        summary=read_json(self.args.output_dir/'summary.json')
        self.assertEqual(summary['finished'],3)
        statuses={r['scene_id']:r['status'] for r in summary['scenes']}
        self.assertEqual(statuses,{'lab':'failed','corridor':'generated_unreviewed','dorm':'generated_unreviewed'})

    def test_nonobject_manifest_is_validation_error(self):
        for bad in (None,3,{'schema_version':1,'scenes':[None]},
                    {'schema_version':1,'scenes':[3]}):
            with self.subTest(manifest=bad):
                atomic_json(self.groups_path,bad)
                with self.assertRaises(ValueError):runner.load_groups(self.groups_path)


if __name__=='__main__':unittest.main()
