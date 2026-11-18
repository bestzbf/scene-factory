"""Rebuild a reviewed shop from a saved demo source, then use the batch renderer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production-root',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--preview',action='store_true')
    p.add_argument('--shop',default='1660108397')
    p.add_argument('--source-root',type=Path,help='Override saved-scene root; contains SHOP/scene.blend and scene_bundle.json')
    p.add_argument('--review-profile',type=Path,default=REPO/'configs/retail/shop_1660108397_review_v11.json')
    p.add_argument('--refine-profile',type=Path)
    p.add_argument('--views',nargs='+')
    p.add_argument('--edge',type=int)
    p.add_argument('--samples',type=int)
    p.add_argument('--blender',default=os.environ.get('BLENDER','/Applications/Blender.app/Contents/MacOS/Blender'))
    a=p.parse_args();prod=a.production_root.resolve();out=a.out.resolve()
    if Path(a.shop).name!=a.shop or a.shop in ('.','..'):p.error('Invalid shop ID')
    source=((a.source_root or prod/'weidian_full_runs/demo_runs/demo_v3')/a.shop).resolve()
    profile=a.review_profile.resolve()
    refine=(a.refine_profile or profile.with_name(profile.name.replace('_review_','_refine_'))).resolve()
    spec=json.loads(profile.read_text())
    builder=REPO/'tools/batch'/('build_reviewed_retail.py' if spec.get('schema_version')=='reviewed_retail_rebuild_v1' else 'prepare_reviewed_scene.py')
    edge=a.edge or (800 if a.preview else 1920)
    samples=a.samples or (32 if a.preview else 256)
    bundle=json.loads((source/'scene_bundle.json').read_text())
    layout=source/'initial_layout.json'
    if not layout.exists():layout=Path(bundle['views'][0]['source_image']).parents[2]/'initial_layout.json'
    if out==source or source in out.parents or out in source.parents:p.error('Source/output overlap')
    files=[source/'scene.blend',source/'scene_bundle.json',profile,refine,Path(__file__).resolve(),builder,
           REPO/'tools/batch/prepare_reviewed_scene.py',REPO/'tools/batch/optimize_demo_batch.py',
           REPO/'tools/batch/optimize_retail_scene.py',REPO/'scene_factory/reviewed_retail_fixtures.py',
           REPO/'scene_factory/retail_stock_geometry.py',REPO/'scene_factory/retail_refinement.py',REPO/'scene_factory/retail_visual_detail.py',
           REPO/'scene_factory/generate_scene.py',(profile.parent/spec['atlas']).resolve()]
    if builder.name=='prepare_reviewed_scene.py':files.append(layout)
    files.extend(Path(v['source_image']) for v in bundle['views'])
    receipt={'files':{str(f):sha(f) for f in files},'preview':a.preview,'edge':edge,'samples':samples,'views':a.views,
             'blender':subprocess.check_output([a.blender,'--version'],text=True).splitlines()[0]}
    out.mkdir(parents=True,exist_ok=True);receipt_path=out/'replay_receipt.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text())!=receipt:
        raise ValueError('Inputs/settings changed; use a new output directory')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    prepared=out/'prepared_source'
    if prepared.exists():
        report=json.loads((prepared/'preparation_report.json').read_text())
        if not all(sha(prepared/name)==digest for name,digest in report['prepared_files_sha256'].items()):
            raise ValueError('Prepared source changed; use a new output directory')
    else:
        command=[a.blender,'-b','--factory-startup','--python-exit-code','1','--python',str(builder),'--',
                 '--source',str(source),'--out',str(prepared),'--profile',str(profile)]
        if builder.name=='prepare_reviewed_scene.py':command+=['--layout',str(layout)]
        subprocess.run(command,check=True)
    command=[sys.executable,str(REPO/'tools/batch/optimize_demo_batch.py'),'--source-root',str(source.parent),
                    '--shops',a.shop,'--source-override',a.shop+'='+str(prepared),
                    '--out',str(out/'renders'),'--config',str(refine),'--blender',a.blender,
                    '--edge',str(edge),'--samples',str(samples)]
    if a.views:command+=['--views',*a.views]
    subprocess.run(command,check=True)


if __name__=='__main__':main()
