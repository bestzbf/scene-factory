#!/usr/bin/env python3
"""Re-render a verified automatic batch after a generic renderer correction.

Reuses only hash-verified automatic specifications. Geometry/cameras/materials
are not edited. A fresh joint review assesses the new pixels.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from batch import atomic_json,canonical_bytes,discover_blender,file_sha256,read_json,run_stage
from batch_multiview import prepare_textures,validate_render,verify_hashes,pipeline_hashes,artifact_hashes
import vision_multiview as vision
from multiview_spec import validate
from model_image_inputs import IMAGE_INPUT_POLICY, prepare_render_input

ROOT=Path(__file__).resolve().parent


def assess(bundle,directory,timeout):
    """Final assessment omits an unused full geometry rewrite from the response."""
    schema=copy.deepcopy(vision.REVIEW_SCHEMA)
    del schema['properties']['revised_bundle']
    schema['required'].remove('revised_bundle')
    images=[];pairs=[];annotations={};model_image_inputs=[]
    for index,view in enumerate(bundle['views']):
        render = directory/f'rgb_{view["id"]}.png'
        model_render, image_input = prepare_render_input(view, render, directory)
        images.extend([view['source_image'],model_render])
        model_image_inputs.append(image_input)
        pairs.append({'view_id':view['id'],'original_image_number':index*2+1,'render_image_number':index*2+2})
        annotations[view['id']]=read_json(directory/f'annotations_{view["id"]}.json')
    metadata=read_json(directory/'metadata.json')
    hashes={str(Path(path).resolve()):file_sha256(path) for path in images}
    # The authoritative render hash is retained separately from any model-only
    # proxy so the acceptance result remains bound to the saved final PNG.
    for image_input in model_image_inputs:
        render_path = directory / ('rgb_' + image_input['view_id'] + '.png')
        hashes[str(render_path.resolve())] = file_sha256(render_path)
        model_path = Path(image_input['model_input_path'])
        hashes[str(model_path.resolve())] = image_input['model_input_sha256']
    prompt=vision.REVIEW_PROMPT+'\nFINAL ASSESSMENT ONLY: do not return revised_bundle or rewrite geometry; this output schema intentionally omits it. Evaluate actual new renders against all original images. Return scores, issues and limitations; repair means further work would help but is not performed in this assessment.\n'+json.dumps({'pairs':pairs,'bundle':bundle,'technical_warnings':{k:v['warnings'] for k,v in annotations.items()}},ensure_ascii=False)
    (directory/'review.prompt.txt').write_text(prompt,encoding='utf-8')
    report=vision.call_model(prompt,images,schema,directory/'review.json',timeout,'medium')
    report['revised_bundle']=bundle
    vision.validate_review(report)
    if any(file_sha256(path)!=digest for path,digest in hashes.items()):raise RuntimeError('Assessment images changed')
    atomic_json(directory/'review.json',report)
    gate=vision.quality_gate(report,metadata,annotations)
    atomic_json(directory/'review_provenance.json',{
        'method':'fresh_joint_final_assessment_after_generic_renderer_fix',
        'input_sha256':hashes,
        'model_image_input_policy': IMAGE_INPUT_POLICY,
        'model_image_input_implementation_sha256': file_sha256(
            Path(__file__).with_name('model_image_inputs.py')),
        'model_image_inputs': model_image_inputs,
        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
        'review_sha256':file_sha256(directory/'review.json'),'gate':gate})
    return report,gate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--vision-timeout',type=int,default=900)
    args=p.parse_args();source=args.batch.resolve();out=args.output_dir.resolve()
    if out.exists():raise ValueError('Use a new output directory for verified re-rendering')
    summary=read_json(source/'summary.json')
    if summary['finished']!=summary['total'] or any(not r.get('iteration') for r in summary['scenes']):
        raise ValueError('Source batch must have a rendered iteration for every scene')
    blender=discover_blender();out.mkdir(parents=True)
    code_hashes=pipeline_hashes()
    code_hashes[Path(__file__).name]=file_sha256(__file__)
    atomic_json(out/'renderer_pipeline_hashes.json',code_hashes)
    atomic_json(out/'groups.json',read_json(source/'groups.json'))
    results=[];futures=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for old in summary['scenes']:
            oldfolder=source/old['scene_id']
            olditeration=oldfolder/old['iteration']
            if old.get('artifacts'):verify_hashes(oldfolder,old['artifacts'])
            else:
                # A final review may time out/be interrupted after a verified
                # render. Reuse that render's exact description; never count
                # the interrupted assessment as passed.
                verify_hashes(olditeration,read_json(olditeration/'render_state.json')['artifacts'])
            bundle=validate(read_json(olditeration/'scene_bundle.json'))
            validate_render(olditeration,bundle)
            analysis=read_json(oldfolder/'analysis_provenance.json')
            if analysis['manual_scene_seed'] or analysis['bundle_sha256']!=file_sha256(oldfolder/'initial_bundle.json'):
                raise ValueError('Source analysis provenance mismatch')
            for view in bundle['views']:
                if file_sha256(view['source_image'])!=analysis['source_sha256'][view['id']]:
                    raise ValueError('Source photograph changed')
            oldbundlehash=file_sha256(olditeration/'scene_bundle.json')
            folder=out/old['scene_id'];iteration=folder/'iteration_02_renderer_fix';iteration.mkdir(parents=True)
            for view in bundle['views']:
                original=Path(view['source_image']);target=folder/'sources'/view['id']/'reference.png'
                target.parent.mkdir(parents=True);shutil.copy2(original,target);view['source_image']=str(target)
                first=folder/'iteration_00';first.mkdir(exist_ok=True)
                shutil.copy2(oldfolder/'iteration_00'/f'rgb_{view["id"]}.png',first/f'rgb_{view["id"]}.png')
            # Copy raw/provenance without altering their original source records.
            for name in ['analysis_provenance.json','initial_bundle.raw.json','initial_bundle.prompt.txt']:
                shutil.copy2(oldfolder/name,folder/name)
            atomic_json(folder/'generation_provenance.json',{
                'source_batch':str(source),'source_iteration':old['iteration'],'source_bundle_sha256':oldbundlehash,
                'scene_parameters_modified':False,'source_path_relocation_only':True,
                'renderer_change':'Exterior AREA light proxies hidden from transmission rays; estimated geometry, cameras, materials, light positions/powers unchanged',
                'source_batch_status':old['status'],
                'source_result_sha256':file_sha256(oldfolder/('result.json' if (oldfolder/'result.json').exists() else 'last_error.json')),'pipeline':code_hashes})
            spec=iteration/'scene_bundle.json';atomic_json(spec,bundle)
            manifest=prepare_textures(bundle,spec,iteration/'textures')
            settings=read_json(olditeration/'metadata.json')['render']
            print(json.dumps({'scene_id':old['scene_id'],'stage':'rerender_after_generic_fix'}),flush=True)
            run_stage([blender,'-b','--factory-startup','--python-exit-code','1','--python',str(ROOT/'generate_multiview.py'),'--','--spec',str(spec),'--output-dir',str(iteration),'--texture-manifest',str(manifest),'--max-edge',str(settings['max_edge']),'--samples',str(settings['samples']),'--device','AUTO'],iteration/'blender.log',1800)
            metadata=validate_render(iteration,bundle)
            futures.append((old,folder,iteration,metadata,pool.submit(assess,bundle,iteration,args.vision_timeout)))
        for old,folder,iteration,metadata,future in futures:
            report,gate=future.result()
            result={'scene_id':old['scene_id'],'status':'accepted' if gate['accepted'] else 'needs_review','directory':str(folder),'iteration':iteration.name,'views':metadata['views'],'summary':report['summary'],'gate':gate,'method':'automatic batch model interpretation and repair, then generic renderer correction and fresh joint final assessment','previous_batch':str(source)}
            result['artifacts']=artifact_hashes(folder,[str(x.relative_to(folder)) for x in folder.rglob('*') if x.is_file() and x.name!='result.json' and x.suffix!='.log'])
            atomic_json(folder/'result.json',result);results.append(result)
            atomic_json(out/'summary.json',{'scenes':results,'finished':len(results),'total':len(futures)})
            print(json.dumps({'scene_id':result['scene_id'],'status':result['status']}),flush=True)
    atomic_json(out/'accepted.json',[r for r in results if r['status']=='accepted'])
    from report_multiview import report
    print(report(out),flush=True)


if __name__=='__main__':main()
