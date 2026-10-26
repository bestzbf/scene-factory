#!/usr/bin/env python3
"""Each photo -> custom scene data -> Blender -> original-photo review -> bounded repair."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from batch import atomic_json, canonical_bytes, collect_images, completed_job, discover_blender, file_sha256, number, read_json, run_stage, utc_now
from photo_assets_v2 import extract, image_size, render_size
from spec_v2 import validate, upgrade_v1
import vision_v2

ROOT=Path(__file__).resolve().parent
TERMINAL={'accepted','needs_review','insufficient_information','generated_unreviewed','analyzed'}


class Conflict(ValueError):
    pass


def render_fingerprint(spec,job):
    return hashlib.sha256(canonical_bytes({'spec':spec,'render':job['render'],
        'pipeline':job['pipeline'],'source_sha256':job['source_sha256']})).hexdigest()


def verified_render(iteration,fingerprint):
    state=iteration/'render_state.json'
    if not state.exists():return False
    saved=read_json(state)
    if saved.get('fingerprint')!=fingerprint:raise Conflict('Rendered inputs changed')
    expected={'scene.blend','rgb.png','metadata.json','annotations.json'}
    if set(saved.get('artifacts',{}))!=expected:raise Conflict('Incomplete render manifest')
    for name,digest in saved['artifacts'].items():
        if not (iteration/name).is_file() or file_sha256(iteration/name)!=digest:
            raise Conflict('Intermediate rendered artifact changed: '+name)
    return completed_job(iteration)[0]


def pipeline_hashes():
    paths = list(ROOT.glob('*.py')) + list((ROOT/'assets_lib').rglob('*'))
    return {str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(paths)
            if p.is_file() and '__pycache__' not in str(p) and not p.name.startswith('test_')}


def job_for(source,args,hashes):
    digest=file_sha256(source)
    name=''.join(c if c.isalnum() or c in '-_' else '_' for c in source.stem)[:48]
    width,height=render_size(source,args.max_edge)
    return {'version':2,'job_id':digest[:12]+'_'+name,'source':str(source), 'source_sha256':digest,
            'source_dimensions':list(image_size(source)), 'pipeline':hashes,
            'render':{'width':width,'height':height,'samples':args.samples,'device':args.device},
            'max_repairs':args.max_repairs,'vision_timeout':args.vision_timeout,'render_timeout':args.render_timeout,
            'effort':args.effort,'no_review':args.no_review,'specs_only':args.specs_only,
            'seed_spec_sha256':file_sha256(args.seed_spec) if args.seed_spec else None}


def save_result(folder,result):
    files=[]
    for p in folder.rglob('*'):
        if p.is_file() and p.name not in {'result.json','progress.json','.lock'} and not p.name.endswith('.log'):
            files.append(p)
    result['artifacts']={str(p.relative_to(folder)):file_sha256(p) for p in sorted(files)}
    atomic_json(folder/'result.json',result)


def resume(folder):
    path=folder/'result.json'
    if not path.exists():return None
    result=read_json(path)
    if result['status'] not in TERMINAL:return None
    if not result.get('artifacts'):raise Conflict('Missing artifact integrity manifest')
    for relative,digest in result['artifacts'].items():
        p=(folder/relative).resolve()
        if not p.is_relative_to(folder.resolve()) or not p.is_file() or file_sha256(p)!=digest:
            raise Conflict('Artifact changed: '+relative+'; choose a new output directory')
    if result['status']=='accepted':
        iteration=folder/result['iteration']
        if not vision_v2.quality_gate(read_json(iteration/'review.json'),read_json(iteration/'metadata.json'),read_json(iteration/'annotations.json'))['accepted']:
            raise Conflict('Saved acceptance no longer passes quality gate')
    return {**result,'resumed':True}


def run_job(job,args):
    started=time.monotonic();folder=args.output_dir/job['job_id'];source=Path(job['source'])
    result={'job_id':job['job_id'],'source':str(source),'directory':str(folder),'status':'failed','started_at':utc_now()}
    fd=None
    try:
        folder.mkdir(parents=True,exist_ok=True)
        try:fd=os.open(folder/'.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError:return {**result,'status':'locked','error':'Task already locked; inspect running process before removing .lock'}
        os.write(fd,canonical_bytes({'pid':os.getpid(),'started_at':utc_now()}))
        if (folder/'job.json').exists():
            if read_json(folder/'job.json')!=job:raise Conflict('Input/options/code changed; choose a new output directory')
        elif any(p.name!='.lock' for p in folder.iterdir()):
            raise Conflict('Existing task files without job identity')
        else:atomic_json(folder/'job.json',job)
        if file_sha256(source)!=job['source_sha256']:raise Conflict('Source changed')
        if args.seed_spec and file_sha256(args.seed_spec)!=job['seed_spec_sha256']:raise Conflict('Seed description changed')
        existing=resume(folder)
        if existing:return existing
        # Normalize EXIF once. All model geometry/crops refer to this orientation.
        reference_manifest=extract(source,[],folder/'source')
        normalized=Path(read_json(reference_manifest)['reference_image'])
        initial=folder/'initial_spec.json'
        if not initial.exists():
            atomic_json(folder/'progress.json',{'stage':'analyze','at':utc_now()})
            if args.seed_spec:
                spec=validate(read_json(args.seed_spec))
                if spec['schema_version']==1:spec=upgrade_v1(spec,*job['source_dimensions'],source_id=job['source_sha256'][:16])
                if [spec['image']['width'],spec['image']['height']]!=job['source_dimensions']:
                    raise ValueError('Seed spec dimensions differ from source photo')
                atomic_json(initial,spec)
                atomic_json(folder/'analysis_provenance.json',{'method':'explicit_seed_spec','seed_sha256':job['seed_spec_sha256'],'source_sha256':job['source_sha256']})
            else:
                spec=vision_v2.analyze(normalized,initial,*job['source_dimensions'],timeout=args.vision_timeout,effort=args.effort)
        else:spec=validate(read_json(initial))
        if args.specs_only:
            if file_sha256(source)!=job['source_sha256']:raise Conflict('Source changed before analysis completion')
            result['status']='analyzed';save_result(folder,result);return result
        history=[]
        for index in range(args.max_repairs+1):
            if file_sha256(source)!=job['source_sha256']:raise Conflict('Source changed during processing')
            iteration=folder/f'iteration_{index:02d}';iteration.mkdir(exist_ok=True)
            spec_path=iteration/'scene_spec.json'
            if spec_path.exists() and read_json(spec_path)!=spec:
                raise Conflict('Iteration spec differs from repair history')
            atomic_json(spec_path,spec)
            manifest=extract(normalized,spec['textures'],iteration/'textures')
            atomic_json(folder/'progress.json',{'stage':'render','iteration':index,'at':utc_now()})
            fingerprint=render_fingerprint(spec,job)
            complete=verified_render(iteration,fingerprint)
            if not complete:
                for stale in ['metadata.json','review.json','review_provenance.json']:(iteration/stale).unlink(missing_ok=True)
                r=job['render']
                command=[args.blender,'-b','--factory-startup','--python-exit-code','1','--python',str(ROOT/'generate_v2.py'),'--',
                         '--spec',str(spec_path),'--source-image',str(normalized),'--texture-manifest',str(manifest),
                         '--output-dir',str(iteration),'--width',str(r['width']),'--height',str(r['height']),
                         '--samples',str(r['samples']),'--device',args.device]
                run_stage(command,iteration/'blender.log',args.render_timeout)
                complete,reason=completed_job(iteration)
                if not complete:raise RuntimeError('Blender output invalid: '+reason)
                atomic_json(iteration/'render_state.json',{'fingerprint':fingerprint,'artifacts':{
                    name:file_sha256(iteration/name) for name in ['scene.blend','rgb.png','metadata.json','annotations.json']}})
            result['iteration']=iteration.name
            if args.no_review:
                result['status']='generated_unreviewed';break
            atomic_json(folder/'progress.json',{'stage':'review','iteration':index,'at':utc_now()})
            report_path=iteration/'review.json'
            if report_path.exists() and (iteration/'review_provenance.json').exists():
                prov=read_json(iteration/'review_provenance.json')
                if prov.get('source_sha256')!=file_sha256(normalized) or prov.get('review_sha256')!=file_sha256(report_path):
                    raise Conflict('Cached review identity mismatch')
                expected={'scene_spec.json','rgb.png','scene.blend','metadata.json','annotations.json'}
                if set(prov.get('input_hashes',{}))!=expected or any(file_sha256(iteration/n)!=h for n,h in prov['input_hashes'].items()):
                    raise Conflict('Cached review inputs changed')
                report=vision_v2.validate_review(read_json(report_path))
            else:report=vision_v2.review(normalized,iteration,timeout=args.vision_timeout,effort=args.effort)
            gate=vision_v2.quality_gate(report,read_json(iteration/'metadata.json'),read_json(iteration/'annotations.json'))
            history.append({'iteration':iteration.name,'scores':report['scores'],'summary':report['summary'],'gate':gate,'verdict':report['verdict']})
            result.update(history=history,scores=report['scores'],gate=gate,summary=report['summary'])
            atomic_json(folder/'history.json',history)
            if gate['accepted']:result['status']='accepted';break
            if report['verdict']=='insufficient_information':result['status']='insufficient_information';break
            candidate=validate(report['revised_spec'])
            if index==args.max_repairs or report['verdict']!='repair' or candidate==spec:
                result['status']='needs_review';break
            spec=candidate
        if file_sha256(source)!=job['source_sha256']:raise Conflict('Source changed before completion')
        result['elapsed_seconds']=round(time.monotonic()-started,2)
        result['scope']='Original-photo view; inferred hidden geometry; heuristic quality review'
        save_result(folder,result)
    except Conflict as exc:
        result.update(status='conflict',error=str(exc))
    except Exception as exc:
        result.update(status='failed',error=str(exc))
        if fd is not None:atomic_json(folder/'last_error.json',result)
    finally:
        if fd is not None:os.close(fd);(folder/'.lock').unlink(missing_ok=True)
    return result


def parse_arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-dir',type=Path);source.add_argument('--images',nargs='+',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed-spec',type=Path)
    p.add_argument('--max-edge',type=int,default=1600);p.add_argument('--samples',type=int,default=96)
    p.add_argument('--max-repairs',type=int,default=2);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--vision-timeout',type=int,default=900);p.add_argument('--render-timeout',type=int,default=1800)
    p.add_argument('--effort',choices=['low','medium','high'],default='medium')
    p.add_argument('--device',choices=['AUTO','CPU'],default='AUTO');p.add_argument('--blender')
    p.add_argument('--specs-only',action='store_true');p.add_argument('--no-review',action='store_true');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    for key,low,high in [('max_edge',64,8192),('samples',1,4096),('workers',1,8),('max_repairs',0,5),('vision_timeout',1,7200),('render_timeout',1,14400)]:
        number(getattr(args,key),key,low,high,True)
    return args


def main(argv=None):
    try:
        args=parse_arguments(argv);args.output_dir=args.output_dir.expanduser().resolve()
        images=collect_images(args.input_dir,args.images)
        if args.input_dir and args.output_dir.is_relative_to(args.input_dir.expanduser().resolve()):raise ValueError('Output must be outside input directory')
        if args.seed_spec:
            args.seed_spec=args.seed_spec.expanduser().resolve();validate(read_json(args.seed_spec))
            if len(images)!=1:raise ValueError('--seed-spec is only for one photo')
        hashes=pipeline_hashes();jobs=[];planning_errors=[]
        for source in images:
            try:jobs.append(job_for(source,args,hashes))
            except (ValueError,OSError,RuntimeError) as exc:
                planning_errors.append({'job_id':'invalid_'+hashlib.sha256(str(source).encode()).hexdigest()[:12],
                    'source':str(source),'status':'failed','stage':'image_decode','error':str(exc)})
        if len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate photo contents/names')
        if args.dry_run:
            print(json.dumps({'jobs':[{k:v for k,v in j.items() if k!='pipeline'} for j in jobs],'errors':planning_errors},ensure_ascii=False,indent=2));return 1 if planning_errors else 0
        args.blender=discover_blender(args.blender);args.output_dir.mkdir(parents=True,exist_ok=True)
    except (ValueError,OSError,RuntimeError) as exc:print('ERROR '+str(exc));return 2
    results=list(planning_errors)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_job,j,args) for j in jobs]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps({k:result[k] for k in ['job_id','status','error'] if k in result},ensure_ascii=False),flush=True)
    counts={s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})}
    accepted=[str(Path(r['directory'])/r['iteration']/'scene.blend') for r in results if r['status']=='accepted']
    atomic_json(args.output_dir/'accepted.json',accepted)
    atomic_json(args.output_dir/'summary.json',{'counts':counts,'jobs':results,'accepted_count':len(accepted)})
    if any(r['status'] in {'failed','conflict','locked'} for r in results):return 1
    if any(r['status'] in {'needs_review','insufficient_information'} for r in results):return 3
    return 0


if __name__=='__main__':raise SystemExit(main())
