"""Resume-safe saved-scene refinement batch, using the production refinement stage.

Example: python tools/batch/optimize_demo_batch.py --source-root OLD --out NEW --config CONFIG
Source-root contains shop/scene.blend and shop/scene_bundle.json. Never edits sources.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scene_factory'))
from retail_refinement import read_config


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def fingerprint(source,layout,config,edge,samples,views,runtime=None):
    files=[source/'scene.blend',source/'scene_bundle.json',REPO/'tools/batch/optimize_retail_scene.py',
           REPO/'scene_factory/retail_refinement.py',REPO/'scene_factory/retail_stock_geometry.py',REPO/'scene_factory/retail_visual_detail.py']
    if layout:files.append(layout)
    if config.get('atlas'):files.append(Path(config['atlas']))
    return hashlib.sha256(json.dumps({'files':{str(p):sha(p) for p in files},'config':config,
                                     'edge':edge,'samples':samples,'views':views,'runtime':runtime},sort_keys=True).encode()).hexdigest()


def complete(folder,key,views):
    try:
        state=json.loads((folder/'batch_state.json').read_text())
        check=json.loads((folder/'verification.json').read_text())
        from PIL import Image
        for v in views:
            with Image.open(folder/f'rgb_{v}.png') as im:im.verify()
        return state['fingerprint']==key and state['status']=='complete' and check['passed'] and (folder/'scene.blend').stat().st_size>0
    except (OSError,ValueError,KeyError):return False


def review(source_root,out,summary):
    from PIL import Image,ImageDraw,ImageOps
    overview=[]
    for sid,item in summary.items():
        if item['status']!='complete':continue
        original=source_root/sid
        bundle=json.loads((original/'scene_bundle.json').read_text())
        rows=[]
        for v in bundle['views']:
            vid=v['id'];new=out/sid/f'rgb_{vid}.png'
            if not new.exists():continue
            row=Image.new('RGB',(1200,250),'#202329');d=ImageDraw.Draw(row)
            for col,(label,path) in enumerate([('Reference',Path(v['source_image'])),('Before',original/f'rgb_{vid}.png'),('Refined 3D',new)]):
                d.text((col*400+8,6),f'{sid} / {vid} / {label}',fill='white')
                if path.exists():
                    with Image.open(path) as im:row.paste(ImageOps.contain(im.convert('RGB'),(394,220)),(col*400+3,27))
            rows.append(row)
        if not rows:continue
        sheet=Image.new('RGB',(1200,250*len(rows)),'#202329')
        for i,row in enumerate(rows):sheet.paste(row,(0,i*250))
        sheet.save(out/sid/'all_views_review.jpg',quality=90)
        overview.extend([rows[0],rows[-1]])
    if overview:
        sheet=Image.new('RGB',(1200,250*len(overview)),'#202329')
        for i,row in enumerate(overview):sheet.paste(row,(0,i*250))
        sheet.save(out/'all_shops_review.jpg',quality=92)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source-root','out','config'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--shops',nargs='+');p.add_argument('--views',nargs='+')
    p.add_argument('--source-override',action='append',default=[],metavar='SHOP=PATH',help='Use an existing reviewed editable scene for a shop')
    p.add_argument('--edge',type=int,default=1920);p.add_argument('--samples',type=int,default=256)
    p.add_argument('--blender',default=os.environ.get('BLENDER','/Applications/Blender.app/Contents/MacOS/Blender'))
    p.add_argument('--timeout',type=int,default=7200)
    a=p.parse_args();src=a.source_root.resolve();out=a.out.resolve()
    if src==out or src in out.parents or out in src.parents:p.error('Output must be independent of source')
    if a.edge<64 or not 1<=a.samples<=4096:p.error('Invalid render quality')
    runtime=subprocess.check_output([a.blender,'--version'],text=True).splitlines()[0]
    config=read_config(a.config);out.mkdir(parents=True,exist_ok=True)
    if not config.get('atlas'):
        from make_packaging_atlas import make_atlas
        atlas=out/'packaging_print.png'
        if not atlas.exists():make_atlas(atlas,config.get('seed',17))
        config['atlas']=str(atlas)
    effective=out/'effective_config.json';write(effective,config)
    overrides=dict(v.split('=',1) for v in a.source_override)
    shops=a.shops or sorted(d.name for d in src.iterdir() if (d/'scene.blend').is_file() and (d/'scene_bundle.json').is_file())
    if not shops:p.error('No saved scenes found')
    if any(Path(s).name!=s or s in ('.','..') for s in shops):p.error('Invalid scene ID')
    summary={}
    for sid in shops:
        folder=out/sid;source=Path(overrides.get(sid,src/sid)).resolve();start=time.monotonic()
        try:
            if source==folder or source in folder.parents or folder in source.parents:raise ValueError('Source/output overlap')
            bundle=json.loads((source/'scene_bundle.json').read_text())
            if str(bundle['scene_id'])!=sid:raise ValueError('Override scene ID mismatch')
            views=a.views or [v['id'] for v in bundle['views']]
            if set(views)-{v['id'] for v in bundle['views']}:raise ValueError('Unknown views')
            # Reviewed source preparation may change the support layout. Prefer
            # its colocated layout over the original photograph batch layout.
            layout=source/'initial_layout.json'
            if not layout.exists():layout=Path(bundle['views'][0]['source_image']).parents[2]/'initial_layout.json'
            if not layout.exists():layout=None
            key=fingerprint(source,layout,config,a.edge,a.samples,views,runtime)
            if complete(folder,key,views):
                summary[sid]={'status':'complete','resumed':True,'views':len(views),'source':str(source)}
            else:
                if folder.exists() and any(folder.iterdir()):
                    state=folder/'batch_state.json'
                    if not state.exists() or json.loads(state.read_text()).get('fingerprint')!=key:
                        raise ValueError('Existing output has different inputs/configuration. Choose a new output directory.')
                folder.mkdir(parents=True,exist_ok=True)
                state={'fingerprint':key,'status':'running','source':str(source),'views':views}
                write(folder/'batch_state.json',state)
                cmd=[a.blender,'-b','--factory-startup','--python-exit-code','1','--python',str(REPO/'tools/batch/optimize_retail_scene.py'),'--',
                     '--source',str(source),'--out',str(folder),'--config',str(effective),'--edge',str(a.edge),'--samples',str(a.samples),'--views',*views]
                if layout:cmd+=['--layout',str(layout)]
                with (folder/'blender.log').open('w') as log:
                    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
                if fingerprint(source,layout,config,a.edge,a.samples,views,runtime)!=key:
                    raise RuntimeError('Inputs or refinement code changed during rendering; use a fresh output directory')
                if result.returncode:raise RuntimeError(f'Blender exited {result.returncode}; see {folder}/blender.log')
                check=json.loads((folder/'verification.json').read_text())
                if not check['passed']:raise RuntimeError('Verification failed')
                state['status']='complete';write(folder/'batch_state.json',state)
                if not complete(folder,key,views):raise RuntimeError('Missing or corrupt output')
                summary[sid]={'status':'complete','resumed':False,'views':len(views),'source':str(source),'seconds':round(time.monotonic()-start,1)}
        except Exception as exc:
            summary[sid]={'status':'failed','error':str(exc)}
        write(out/'summary.json',summary);print(sid,json.dumps(summary[sid],ensure_ascii=False),flush=True)
    review(src,out,summary)
    if any(s['status']!='complete' for s in summary.values()):sys.exit(1)


if __name__=='__main__':main()
