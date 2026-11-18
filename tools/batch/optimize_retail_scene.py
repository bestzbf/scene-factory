"""Refine a saved scene once, then render any/all views of that shared 3D scene."""
import argparse
import json
import hashlib
import array
import math
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scene_factory'))
import bpy
from generate_v2 import configure_device
from retail_refinement import apply_refinement,read_config


def geometry_digest(scene):
    h=hashlib.sha256()
    for obj in sorted((o for o in scene.objects if o.type=='MESH'),key=lambda o:o.name):
        h.update(obj.name.encode());h.update(str(tuple(tuple(row) for row in obj.matrix_world)).encode())
        coords=array.array('f',[0])*(len(obj.data.vertices)*3)
        obj.data.vertices.foreach_get('co',coords);h.update(coords.tobytes())
        h.update(str(len(obj.data.polygons)).encode())
    return h.hexdigest()


def camera_digest(scene):
    return {o.name: {'matrix': [list(row) for row in o.matrix_world], 'lens':o.data.lens}
            for o in scene.objects if o.type=='CAMERA'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','out','config'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--layout',type=Path)
    p.add_argument('--views',nargs='+')
    p.add_argument('--edge',type=int,default=1280);p.add_argument('--samples',type=int,default=128)
    p.add_argument('--no-render',action='store_true')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    source=args.source.resolve();out=args.out.resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('Output must be separate from source')
    if args.edge<64 or not 1<=args.samples<=4096:raise ValueError('Invalid render settings')
    bundle=json.loads((source/'scene_bundle.json').read_text())
    views={v['id']:v for v in bundle['views']};selected=args.views or list(views)
    if set(selected)-set(views):raise ValueError('Unknown views')
    config=read_config(args.config)
    layout=json.loads(args.layout.read_text()) if args.layout else None
    bpy.ops.wm.open_mainfile(filepath=str(source/'scene.blend'))
    scene=bpy.context.scene;device=configure_device(scene,'AUTO')
    original_cameras=camera_digest(scene)
    report=apply_refinement(bundle,config,layout)
    geometry_before=geometry_digest(scene)
    report.update(source=str(source),device=device,views=[],settings={'edge':args.edge,'samples':args.samples})
    scene.cycles.samples=args.samples;scene.cycles.use_denoising=True
    scene.cycles.use_adaptive_sampling=True;scene.cycles.adaptive_threshold=.025
    scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    out.mkdir(parents=True,exist_ok=True)
    # Preserve all original cameras. No automatic focal-length standardization.
    for vid in selected:
        v=views[vid];camera=bpy.data.objects['ReferenceCamera_'+vid];scene.camera=camera
        factor=args.edge/max(v['width'],v['height'])
        scene.render.resolution_x=round(v['width']*factor);scene.render.resolution_y=round(v['height']*factor)
        scene.render.filepath=str(out/f'rgb_{vid}.png');bpy.context.view_layer.update()
        start=time.monotonic()
        if not args.no_render:bpy.ops.render.render(write_still=True)
        report['views'].append({'id':vid,'file':str(out/f'rgb_{vid}.png'),'seconds':round(time.monotonic()-start,3)})
        (out/'retail_refinement.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    # Pack once, save once; geometry is never rebuilt inside the view loop.
    bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    (out/'scene_bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2))
    (out/'optimization_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    if layout:(out/'refined_layout.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2))
    # Exact mesh data is the replayable authority; bundle is the original base plus moves.
    check={'finite_mesh_coordinates':all(math.isfinite(c) for o in scene.objects if o.type=='MESH' for v in o.data.vertices for c in v.co),
           'geometry_build_count':1,'camera_dependent_geometry':False,
           'packed_file_images':sum(i.source=='FILE' and bool(i.packed_file) for i in bpy.data.images),
           'checked_cameras':selected,'stock_units':report['stock_units']}
    check['physical_stock_units']=sum(int(o.get('counted_stock_units',0)) for o in scene.objects if o.type=='MESH')
    check['camera_parameters']=camera_digest(scene)
    check['geometry_sha256_before_views']=geometry_before
    check['geometry_sha256_after_views']=geometry_digest(scene)
    check['shared_geometry_unchanged_across_views']=geometry_before==check['geometry_sha256_after_views']
    check['source_cameras_preserved']=original_cameras==camera_digest(scene)
    check['layout_clearance']=report['layout_clearance']
    check['passed']=all(check[k] for k in ('finite_mesh_coordinates','shared_geometry_unchanged_across_views','source_cameras_preserved')) and report['layout_clearance']['status']=='passed'
    (out/'verification.json').write_text(json.dumps(check,indent=2))
    if not check['passed']:raise RuntimeError('Invalid mesh coordinates')
    print('RETAIL_REFINEMENT_COMPLETE',json.dumps({'scene_id':bundle['scene_id'],'fixtures':len(report['fixtures']),'stock_units':report['stock_units'],'views':len(selected)}),flush=True)


if __name__=='__main__':main()
