"""V2 data provenance, per-photo geometry, texture mapping and resume contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import batch_v2 as runner
from batch import atomic_json
from spec_v2 import upgrade_v1, validate
import vision_v2 as vision

try:
    from PIL import Image
except ImportError:
    Image=None


def spec():
    base=json.loads((Path(__file__).parent/'examples/living_room.json').read_text())
    return upgrade_v1(base,width=40,height=20)


def report():
    return {'schema_version':1,'verdict':'recommend_accept','summary':'Test only',
            'scores':{k:95 for k in ('framing','layout','geometry','materials','lighting','realism')},
            'issues':[],'asset_gaps':[],'reference_limitations':[],'revised_spec':spec()}


class V2ReviewTests(unittest.TestCase):
    def test_repeated_support_normalization_keeps_geometry(self):
        from spec_v2 import part
        s=spec();base=copy.deepcopy(s['base_scene']['objects'][0])
        base.update(id='repeated',position=[0,2,0],repeat=[3,1],spacing=[2,0])
        s['base_scene']['objects']=[base]
        s['custom_objects']=[{'id':'item','position':[2.1,2,1],'rotation_degrees':[0,0,0],
            'material_id':'','confidence':.8,'observed':True,'support_id':'repeated','parts':[part(size=[.1,.1,.1])]}]
        fixed,changes=vision.resolve_support_groups(s)
        self.assertEqual(fixed['custom_objects'][0]['support_id'],'repeated__1_0')
        self.assertEqual(fixed['custom_objects'][0]['position'],s['custom_objects'][0]['position'])
        self.assertEqual(len(changes),1);validate(fixed)

    def test_v2_geometry_survives_review_and_cannot_bypass_gate(self):
        r=report();self.assertEqual(vision.validate_review(r)['revised_spec']['schema_version'],2)
        metadata={'status':'completed','warning_count':0,'render':{'width':1280,'height':960,'samples':64}}
        self.assertTrue(vision.quality_gate(r,metadata,{'warnings':[]})['accepted'])
        r['asset_gaps']=['Unreconstructed curtains']
        self.assertFalse(vision.quality_gate(r,metadata,{'warnings':[]})['accepted'])
        r['asset_gaps']=[];metadata['render']['samples']=16
        self.assertFalse(vision.quality_gate(r,metadata,{'warnings':[]})['accepted'])

    def test_review_rejects_nonfinite_geometry_and_instructions(self):
        for key,value in [('render',{'exposure':float('nan'),'world_color':[1,1,1],'replace_base_lights':False}),('python','import os')]:
            r=report();r['revised_spec'][key]=value
            with self.assertRaises(ValueError):vision.validate_review(r)


@unittest.skipIf(Image is None,'Pillow required for photo runner')
class V2BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'photo.png'
        Image.new('RGB',(40,20),(90,120,160)).save(self.source)
        self.seed=self.root/'seed.json';atomic_json(self.seed,spec())
        self.args=runner.parse_arguments(['--images',str(self.source),'--output-dir',str(self.root/'out'),
            '--seed-spec',str(self.seed),'--no-review','--samples','16','--max-edge','128'])
        self.args.blender='mock-blender';self.job=runner.job_for(self.source,self.args,{})

    def fake_render(self,cmd,log,timeout):
        out=Path(cmd[cmd.index('--output-dir')+1])
        (out/'scene.blend').write_bytes(b'BLENDER-test')
        Image.new('RGB',(128,64)).save(out/'rgb.png')
        atomic_json(out/'annotations.json',{'warnings':[]})
        atomic_json(out/'metadata.json',{'status':'completed','mesh_count':2,'asset_count':2,
            'outputs':['scene.blend','rgb.png','annotations.json'],'render':{'width':128,'height':64,'samples':16},'warning_count':0})

    def test_unique_scene_then_resume_and_detect_tampering(self):
        with patch.object(runner,'run_stage',side_effect=self.fake_render) as render:
            first=runner.run_job(self.job,self.args)
            self.assertEqual(first['status'],'generated_unreviewed',first)
            second=runner.run_job(self.job,self.args)
            self.assertTrue(second['resumed']);self.assertEqual(render.call_count,1)
            p=Path(first['directory'])/first['iteration']/'scene.blend';p.write_bytes(b'changed')
            third=runner.run_job(self.job,self.args)
            self.assertEqual(third['status'],'conflict')

    def test_failed_render_is_not_completed_and_lock_released(self):
        with patch.object(runner,'run_stage',side_effect=RuntimeError('render failed')):
            result=runner.run_job(self.job,self.args)
        self.assertEqual(result['status'],'failed')
        self.assertFalse((Path(result['directory'])/'.lock').exists())
        self.assertFalse((Path(result['directory'])/'result.json').exists())

    def test_source_change_after_planning_cannot_resume(self):
        with patch.object(runner,'run_stage',side_effect=self.fake_render):
            self.assertEqual(runner.run_job(self.job,self.args)['status'],'generated_unreviewed')
        Image.new('RGB',(40,20),(1,2,3)).save(self.source)
        self.assertEqual(runner.run_job(self.job,self.args)['status'],'conflict')

    def test_automatic_repair_changes_spec_and_renders_again(self):
        self.args.no_review=False;self.args.max_repairs=1
        self.job=runner.job_for(self.source,self.args,{})
        seen=[]
        def reviewer(source,directory,**kwargs):
            current=json.loads((directory/'scene_spec.json').read_text())
            seen.append(current['camera']['roll_degrees'])
            r=report();r['verdict']='repair';r['revised_spec']=copy.deepcopy(current)
            r['revised_spec']['camera']['roll_degrees']+=2
            atomic_json(directory/'review.json',r)
            return r
        with patch.object(runner,'run_stage',side_effect=self.fake_render) as render, patch.object(vision,'review',side_effect=reviewer):
            result=runner.run_job(self.job,self.args)
        self.assertEqual(result['status'],'needs_review',result)
        self.assertEqual(seen,[0,2]);self.assertEqual(render.call_count,2)
        self.assertEqual(result['iteration'],'iteration_01')

    def test_bad_photo_does_not_block_valid_photo(self):
        bad=self.root/'bad.jpg';bad.write_bytes(b'not an image')
        with patch.object(runner,'run_job',return_value={'job_id':'good','status':'analyzed'}) as work, patch.object(runner,'discover_blender',return_value='mock'):
            code=runner.main(['--images',str(self.source),str(bad),'--output-dir',str(self.root/'batch'),'--specs-only'])
        self.assertEqual(code,1);self.assertEqual(work.call_count,1)
        summary=json.loads((self.root/'batch/summary.json').read_text())
        self.assertEqual(summary['counts'],{'analyzed':1,'failed':1})

    def test_aspect_ratio_and_identity(self):
        self.assertEqual(self.job['render']['width'],128)
        self.assertEqual(self.job['render']['height'],64)
        Image.new('RGB',(20,40),(10,30,50)).save(self.source)
        changed=runner.job_for(self.source,self.args,{})
        self.assertNotEqual(self.job['job_id'],changed['job_id'])
        self.assertEqual((changed['render']['width'],changed['render']['height']),(64,128))

    def test_texture_rectification_preserves_corner_orientation(self):
        from photo_assets_v2 import extract
        im=Image.new('RGB',(40,20))
        for x in range(40):
            for y in range(20):im.putpixel((x,y),(x*6,y*12,0))
        im.save(self.source)
        manifest=extract(self.source,[{'id':'whole','quad':[[0,0],[1,0],[1,1],[0,1]]}],self.root/'textures')
        data=json.loads(manifest.read_text());result=Image.open(data['textures'][0]['path'])
        self.assertLess(result.getpixel((0,0))[0],result.getpixel((result.width-1,0))[0])
        self.assertLess(result.getpixel((0,0))[1],result.getpixel((0,result.height-1))[1])
        self.assertEqual(data['dimensions'],[40,20])

    def test_api_reference_keeps_source_resolution_when_render_is_bounded(self):
        from batch import file_sha256
        from photo_assets_v2 import extract, render_size
        large = self.root / 'high_resolution.jpg'
        Image.new('RGB', (1800, 1200), (80, 90, 100)).save(large, quality=90)
        digest = file_sha256(large)
        self.assertEqual(render_size(large, max_edge=512), (512, 341))
        manifest_path = extract(large, [], self.root / 'large_source')
        manifest = json.loads(manifest_path.read_text())
        reference = Path(manifest['reference_image'])
        with Image.open(reference) as image:
            self.assertEqual(image.size, (1800, 1200))
        self.assertEqual(manifest['dimensions'], [1800, 1200])
        self.assertEqual(manifest['source_sha256'], digest)


if __name__=='__main__':unittest.main()
