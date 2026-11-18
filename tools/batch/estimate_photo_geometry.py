"""Optional batch photo-pose estimation before reviewed 3D layout construction.

Outputs sparse components, camera poses and a review report, never an accepted
metric floor plan. Feature extraction and COLMAP run in separate processes to
avoid loading two OpenMP runtimes together on macOS.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(folder, max_keypoints):
    import itertools
    import numpy as np
    import torch
    # LightGlue imports an optional SIFT backend at module import. We only use
    # SuperPoint here; COLMAP is deliberately confined to the parent process.
    sys.modules['pycolmap']=None
    from lightglue import SuperPoint, LightGlue
    from lightglue.utils import load_image, rbd
    torch.set_grad_enabled(False);torch.set_num_threads(6)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    extractor=SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher=LightGlue(features='superpoint',flash=False,depth_confidence=-1,width_confidence=-1).eval().to(device)
    cached={};arrays={}
    for path in sorted((folder/'images').glob('*.png')):
        cached[path.name]=extractor.extract(load_image(path).to(device))
        arrays[path.name]=cached[path.name]['keypoints'][0].cpu().numpy()
        print('EXTRACT',path.name,len(arrays[path.name]),flush=True)
    for left,right in itertools.combinations(cached,2):
        result=rbd(matcher({'image0':cached[left],'image1':cached[right]}))
        arrays[left+'__'+right]=result['matches'].cpu().numpy().astype(np.uint32)
        print('MATCH',left,right,len(arrays[left+'__'+right]),flush=True)
    temporary=folder/'features.partial.npz'
    np.savez_compressed(temporary,**arrays);temporary.replace(folder/'features.npz')


def estimate(source, folder, keypoints, focal):
    from PIL import Image
    bundle=json.loads((source/'scene_bundle.json').read_text())
    photos={v['id']:Path(v['source_image']) for v in bundle['views']}
    if len(photos)<2:raise ValueError('At least two reference photographs are required')
    if any(Path(k).name!=k or '__' in k for k in photos):raise ValueError('Unsafe view ID')
    sizes={Image.open(path).size for path in photos.values()}
    if len(sizes)!=1:raise ValueError('Shared-intrinsics estimator requires equal image dimensions')
    width,height=next(iter(sizes))
    receipt={'images':{k:sha(v) for k,v in photos.items()},'max_keypoints':keypoints,
             'initial_focal_px':focal or width*.55,'code_sha256':sha(__file__),
             'libraries':{name:importlib.metadata.version(name) for name in ('pycolmap','torch','lightglue')},
             'lightglue_source':importlib.metadata.distribution('lightglue').read_text('direct_url.json')}
    folder.mkdir(parents=True,exist_ok=True);record=folder/'inputs.json'
    if record.exists() and json.loads(record.read_text())!=receipt:
        raise ValueError('Different inputs or estimator settings: use a new output directory')
    record.write_text(json.dumps(receipt,indent=2)+'\n')
    report_path=folder/'photo_geometry_report.json'
    if report_path.exists():return json.loads(report_path.read_text())
    images=folder/'images';images.mkdir(exist_ok=True)
    for view,path in photos.items():
        # Decode instead of renaming arbitrary source file formats to PNG.
        Image.open(path).convert('RGB').save(images/(view+'.png'))
    if not (folder/'features.npz').exists():
        with (folder/'feature_extraction.log').open('w') as log:
            result=subprocess.run([sys.executable,__file__,'--feature-stage',str(folder),'--max-keypoints',str(keypoints)],
                                  stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Feature extraction failed; see '+str(folder/'feature_extraction.log'))
    import numpy as np
    import pycolmap
    dbpath=folder/'database.db'
    # This is a disposable derived database; the saved feature cache remains.
    if dbpath.exists():dbpath.unlink()
    data=np.load(folder/'features.npz');pairs=[]
    with pycolmap.Database.open(dbpath) as db:
        camera=pycolmap.Camera(model='SIMPLE_RADIAL',width=width,height=height,
                              params=[receipt['initial_focal_px'],width/2,height/2,0])
        cid=db.write_camera(camera);ids={}
        for name in photos:
            filename=name+'.png';ids[filename]=db.write_image(pycolmap.Image(name=filename,camera_id=cid))
            db.write_keypoints(ids[filename],data[filename].astype(np.float32))
        for key in data.files:
            if '__' not in key:continue
            a,b=key.split('__');db.write_matches(ids[a],ids[b],data[key]);pairs.append(a+' '+b)
    pair_path=folder/'pairs.txt';pair_path.write_text('\n'.join(pairs)+'\n')
    pycolmap.verify_matches(dbpath,pair_path)
    options=pycolmap.IncrementalPipelineOptions(num_threads=6,min_model_size=3,max_num_models=4,
               random_seed=166,structure_less_registration_fallback=False)
    options.mapper.init_min_tri_angle=2;options.mapper.init_min_num_inliers=40
    options.mapper.abs_pose_min_num_inliers=25;options.mapper.abs_pose_min_inlier_ratio=.15
    options.ba_global_max_num_iterations=100
    sparse=folder/'sparse';sparse.mkdir(exist_ok=True)
    models=pycolmap.incremental_mapping(dbpath,images,sparse,options=options)
    components=[];registered=set()
    for index,model in models.items():
        destination=sparse/str(index);destination.mkdir(exist_ok=True);model.write_text(destination)
        names=sorted(Path(im.name).stem for im in model.images.values());registered.update(names)
        components.append({'id':index,'views':names,'points':model.num_points3D(),
                           'mean_reprojection_error_px':model.compute_mean_reprojection_error(),
                           'mean_track_length':model.compute_mean_track_length()})
    report={'scene_id':str(bundle['scene_id']),'status':'needs_review','components':components,
            'unregistered_views':sorted(set(photos)-registered),'all_views_in_one_component':any(len(x['views'])==len(photos) for x in components),
            'metric_scale_known':False,'limits':['Reprojection residuals measure sparse fitting, not reference layout fidelity.',
            'Disconnected components must not be treated as a shared scene without reviewed alignment.',
            'Room dimensions, fixtures and scale require reviewed evidence.'], 'pycolmap_version':pycolmap.__version__}
    report_path.write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path);p.add_argument('--shops',nargs='+');p.add_argument('--out',type=Path)
    p.add_argument('--max-keypoints',type=int,default=4096);p.add_argument('--focal-px',type=float)
    p.add_argument('--feature-stage',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if not 256<=a.max_keypoints<=16000:p.error('max-keypoints must be between 256 and 16000')
    if a.feature_stage:features(a.feature_stage,a.max_keypoints);return 0
    if a.source_root is None or a.out is None:p.error('--source-root and --out are required')
    source=a.source_root.resolve();out=a.out.resolve()
    if source==out or source in out.parents or out in source.parents:p.error('Independent output directory required')
    shops=a.shops or sorted(x.name for x in source.iterdir() if (x/'scene_bundle.json').is_file())
    if not shops:p.error('No source scenes')
    if any(Path(x).name!=x or x in ('.','..') for x in shops):p.error('Unsafe shop ID')
    out.mkdir(parents=True,exist_ok=True);reports={}
    for shop in shops:
        try:reports[shop]=estimate(source/shop,out/shop,a.max_keypoints,a.focal_px)
        except Exception as e:reports[shop]={'status':'failed','error':str(e)}
        print(shop,json.dumps(reports[shop]),flush=True)
    (out/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    return int(any(x['status']=='failed' for x in reports.values()))


if __name__=='__main__':raise SystemExit(main())
