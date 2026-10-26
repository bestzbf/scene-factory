"""Real loopback API -> grouped runner contracts; Blender is mocked by default.

Set SCENE_FACTORY_TEST_BLENDER to an installed Blender executable to exercise
real .blend generation. All model responses in these tests are synthetic.
"""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

import batch_multiview as runner
import vision_multiview as vision
from test_batch_multiview import bundle_for
import test_batch_multiview as batch_fixtures


class APIBatchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for index in range(4):
            Image.new('RGB',(64,48),(90+index*20,110,130)).save(self.root/f'{index}.png')
        self.groups = self.root/'groups.json'
        runner.atomic_json(self.groups, {'schema_version':1,'scenes':[
            {'scene_id':'two_views','images':['0.png','1.png']},
            {'scene_id':'single_a','images':['2.png']},
            {'scene_id':'single_b','images':['3.png']}]})
        self.requests = []
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append(body)
                text=body['messages'][0]['content'][0]['text']
                marker='TRUSTED INPUT IDENTITY (images in exactly this order):\n'
                if marker in text:
                    identity=json.JSONDecoder().raw_decode(text.split(marker,1)[1])[0]
                    result=bundle_for(identity['scene_id'],identity['views'])
                    # The shared mock fixture's decorative window has no actual
                    # wall opening. This API transport test needs a physically
                    # valid synthetic room when opting into real Blender; omit
                    # that window here without relaxing production preflight or
                    # changing the shared fixture used by other contract tests.
                    base=result['shared_scene']['base_scene']
                    base['objects']=[item for item in base['objects'] if item['category'] != 'window']
                else:
                    data=json.JSONDecoder().raw_decode(text.split('UNTRUSTED CURRENT SCENE DATA (only identity/pairing is fixed):\n',1)[1])[0]
                    scores={key:50 for key in vision.DIMENSIONS}
                    result={'schema_version':3,'verdict':'repair','summary':'Synthetic integration fixture, not model reconstruction.',
                        'scores':scores,'per_view_scores':[{'view_id':v['id'],'scores':scores,'summary':'fixture'} for v in data['scene_bundle']['views']],
                        'issues':[],'asset_gaps':[],'reference_limitations':['Synthetic API response.']}
                    if 'revised_bundle' in body['response_format']['json_schema']['schema']['properties']:
                        result['revised_bundle']=data['scene_bundle']
                response=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(result)}}],
                                     'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(response)));self.end_headers();self.wfile.write(response)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.stop_server)
        model={'provider':'openai_chat','model':'fixture-vision','base_url':f'http://127.0.0.1:{self.server.server_port}/v1','api_key_env':'','response_format':'json_schema','max_retries':0}
        self.config={'schema_version':1,'models':{'analysis':model},
                     'batch':{'workers':2,'max_repairs':0,'no_review':True},
                     'render':{'max_edge':128,'samples':8,'tone_match':'off','device':'CPU'}}
        self.config_path=self.root/'config.json';self.out=self.root/'output'

    def stop_server(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def run_batch(self):
        runner.atomic_json(self.config_path,self.config)
        command=['--config',str(self.config_path),'--groups',str(self.groups),'--output-dir',str(self.out)]
        blender=os.environ.get('SCENE_FACTORY_TEST_BLENDER')
        with patch.object(vision,'call_model',side_effect=AssertionError('Legacy CLI must not be called')):
            if blender:
                return runner.main(command+['--blender',blender])
            with patch.object(runner,'discover_blender',return_value='fixture-blender'), \
                 patch.object(runner,'run_stage',side_effect=lambda *args: batch_fixtures.GroupedBatchTests.fake_render(self,*args)):
                return runner.main(command)

    def test_grouped_api_resume_and_model_identity(self):
        self.assertEqual(self.run_batch(),0)
        self.assertEqual(len(self.requests),3)
        image_counts=sorted(sum(item['type']=='image_url' for item in req['messages'][0]['content']) for req in self.requests)
        self.assertEqual(image_counts,[1,1,2])
        summary=runner.read_json(self.out/'summary.json')
        self.assertEqual([r['status'] for r in summary['scenes']],['generated_unreviewed']*3)
        self.assertEqual(len(list(self.out.rglob('scene.blend'))),3)
        provenance=runner.read_json(self.out/'two_views/analysis_provenance.json')
        self.assertEqual(provenance['model']['model'],'fixture-vision')
        self.assertFalse(provenance['manual_scene_seed'])
        self.assertEqual(self.run_batch(),0)
        self.assertEqual(len(self.requests),3)
        self.config['models']['analysis']['model']='another-model'
        self.assertEqual(self.run_batch(),1)
        self.assertEqual(len(self.requests),3)
        self.assertTrue(all('changed' in r['error'] for r in runner.read_json(self.out/'summary.json')['scenes']))

    def test_review_can_use_an_independent_model(self):
        self.config['models']['review']={**self.config['models']['analysis'],'model':'fixture-review'}
        self.config['batch']['no_review']=False
        runner.atomic_json(self.groups,{'schema_version':1,'scenes':[{'scene_id':'two_views','images':['0.png','1.png']}]})
        self.assertEqual(self.run_batch(),0)
        self.assertEqual([r['model'] for r in self.requests],['fixture-vision','fixture-review'])
        self.assertEqual(sum(c['type']=='image_url' for c in self.requests[1]['messages'][0]['content']),4)
        result=runner.read_json(self.out/'summary.json')['scenes'][0]
        self.assertEqual(result['status'],'needs_review')
        prov=runner.read_json(self.out/'two_views/iteration_00/review_provenance.json')
        self.assertEqual(prov['model']['model'],'fixture-review')
        self.assertEqual(self.run_batch(),0)
        self.assertEqual(len(self.requests),2)


if __name__=='__main__': unittest.main()
