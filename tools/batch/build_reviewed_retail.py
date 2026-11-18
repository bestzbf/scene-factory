"""Build one editable, shared retail scene from a reviewed metric profile.

Blender entry point. Profiles contain fixtures, real props, shell segments and
cameras. Reference photographs are only used as ordinary package UV textures;
there are no camera-facing background planes or per-view geometry changes.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scene_factory'))
from generate_scene import material, box
from reviewed_retail_fixtures import replace_shelf, replace_fridge
from retail_stock_geometry import Mesh
from retail_refinement import _materials


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def footprint_audit(fixtures, room):
    """Separating-axis audit for fixture footprints, excluding supported stock."""
    corners={};issues=[]
    for item in fixtures:
        w,d,_=item['dimensions_m'];x,y,_=item['position'];angle=math.radians(item.get('rotation_z_deg',0))
        c,s=math.cos(angle),math.sin(angle)
        corners[item['id']]=[(x+c*a-s*b,y+s*a+c*b) for a,b in [(-w/2,-d/2),(w/2,-d/2),(w/2,d/2),(-w/2,d/2)]]
        if any(not (-room['width']/2<=px<=room['width']/2 and 0<=py<=room['depth']) for px,py in corners[item['id']]):
            issues.append({'type':'outside_envelope','id':item['id']})
    keys=list(corners)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            depths=[]
            for poly in (corners[a],corners[b]):
                for j in (0,1):
                    dx=poly[j+1][0]-poly[j][0];dy=poly[j+1][1]-poly[j][1];norm=math.hypot(dx,dy)
                    axis=(-dy/norm,dx/norm)
                    aa=[px*axis[0]+py*axis[1] for px,py in corners[a]]
                    bb=[px*axis[0]+py*axis[1] for px,py in corners[b]]
                    depths.append(min(max(aa),max(bb))-max(min(aa),min(bb)))
            if min(depths)>.005:issues.append({'type':'fixture_overlap','ids':[a,b],'minimum_penetration_m':min(depths)})
    return {'passed':not issues,'issues':issues,'fixture_count':len(fixtures),
            'scope':'Declared floor fixture rectangles in XY; excludes merchandise, props and architectural segments.'}


def vessel(mesh, radius, height, z, color, upside_down=False):
    """Closed solid wall around an open, tapered household vessel."""
    segments=32
    # outer base -> outer rim -> inner rim -> inner base
    profile=[(radius*.73,0),(radius,height),(radius-.012,height),
             (radius*.73-.012,.018)]
    if upside_down: profile=[(r,height-h) for r,h in profile]
    for a,b in zip(profile,profile[1:]+profile[:1]):
        for j in range(segments):
            t=j*math.tau/segments;u=(j+1)*math.tau/segments
            mesh.face([(a[0]*math.cos(t),a[0]*math.sin(t),z+a[1]),
                       (a[0]*math.cos(u),a[0]*math.sin(u),z+a[1]),
                       (b[0]*math.cos(u),b[0]*math.sin(u),z+b[1]),
                       (b[0]*math.cos(t),b[0]*math.sin(t),z+b[1])],color)
    bottom=z+height-.012 if upside_down else z+.012
    mesh.face([(radius*.70*math.cos(j*math.tau/segments),radius*.70*math.sin(j*math.tau/segments),bottom) for j in range(segments)],color,
              [(0,0)]*segments)


def prop(spec, stock_materials):
    root=bpy.data.objects.new(spec['id'],None);bpy.context.collection.objects.link(root)
    root.location=spec['position'];root.rotation_euler.z=math.radians(spec.get('rotation_z_deg',0))
    dims=spec['dimensions_m'];w,d,h=dims
    col=material('Prop_'+spec['id'],spec.get('color',[.42,.27,.12]),.55)
    mesh=Mesh('Geometry_'+spec['id']);kind=spec.get('type','box')
    if kind in ('bucket_stack','basin_stack'):
        count=spec.get('count',1);step=spec.get('stack_step',.035)
        if not isinstance(count,int) or count<1 or not math.isfinite(step) or step<0:
            raise ValueError('Invalid vessel stack count or step')
        unit_h=h-step*(count-1)
        if unit_h<=.02:raise ValueError('Invalid nested vessel stack')
        for i in range(count):vessel(mesh,min(w,d)/2,unit_h,i*step,col,spec.get('upside_down',False))
    elif kind=='stool_stack':
        count=spec.get('count',2);step=.055;unit_h=h-step*(count-1)
        if not isinstance(count,int) or count<1 or unit_h<=.05:
            raise ValueError('Invalid stool stack count or height')
        for i in range(count):
            z=i*step
            mesh.box((0,0,z+.025),(w,d,.05),col)
            for x in (-w*.39,w*.39):
                for y in (-d*.39,d*.39):mesh.box((x,y,z+unit_h/2),(.045,.045,unit_h),col)
    elif kind=='counter':
        if spec.get('open_back'):
            for x in (-w/2+.014,w/2-.014):mesh.box((x,0,h/2),(.028,d,h),col)
            mesh.box((0,d/2-.014,h/2),(w,.028,h),col)
            for z in (.04,h*.47):mesh.box((0,0,z),(w,d,.028),col)
            mesh.box((0,0,h*.27),(.024,d,h*.46),col)
        else:
            mesh.box((0,0,h/2),(w,d,h),col)
        top=material('CounterTop',(.69,.72,.68),.33)
        mesh.box((0,0,h+.018),(w+.025,d+.025,.035),top)
    elif kind=='monitor':
        black=material('MonitorBlack',(.012,.015,.019),.3)
        mesh.box((0,0,.02),(w*.7,d,.04),black)
        mesh.box((0,0,h*.4),(.035,.04,h*.75),black)
        mesh.box((0,0,h*.75),(w,.04,h*.5),black)
        if spec.get('screen'):
            screen=material('MonitorScreen',(.025,.035,.075),.16)
            mesh.box((0,-.022,h*.75),(w*.91,.004,h*.43),screen)
    elif kind=='keyboard':
        black=material('KeyboardBlack',(.018,.022,.026),.48)
        keys=material('KeyboardKeys',(.055,.06,.064),.55)
        mesh.box((0,0,h*.25),(w,d,h*.5),black)
        for i in range(14):
            for j in range(5):mesh.box((-w*.46+(i+.5)*w*.92/14,-d*.44+(j+.5)*d*.88/5,h*.7),(w*.056,d*.14,h*.4),keys)
    elif kind=='chair':
        metal=material('ChairFrame',(.075,.08,.085),.3,.5)
        mesh.box((0,0,h*.27),(.045,.045,h*.5),metal)
        for x in (-w*.3,w*.3):
            mesh.box((x,0,.055),(.04,d*.85,.04),metal)
            mesh.box((x,d*.36,h*.70),(.025,.025,h*.6),metal)
        mesh.box((0,0,h*.53),(w,d,.07),col)
        mesh.box((0,d*.36,h*.83),(w,.05,h*.30),col)
    elif kind=='paper_rolls':
        r=spec.get('radius_m',.045);nx=max(1,int(w/(r*2.1)));nz=max(1,int(h/(r*2.1)))
        white=material('PaperRollWhite',(.8,.84,.78),.75)
        for i in range(nx):
            for j in range(nz):
                first=len(mesh.vertices)
                mesh.bottle(-w/2+(i+.5)*w/nx,0,0,r,d,white,white,white,jar=True)
                for k in range(first,len(mesh.vertices)):
                    vx,vy,vz=mesh.vertices[k];mesh.vertices[k]=(vx,vz-d/2,vy+(j+.5)*h/nz)
    else:
        printed=stock_materials[0] if spec.get('texture_tile') is not None else None
        mesh.box((0,0,h/2),dims,col,printed,('-Y','+Y','-X','+X'))
        if printed:
            tile=spec['texture_tile'];tx,ty=tile%8,7-tile//8
            for j in range(len(mesh.faces)):
                if mesh.materials[mesh.indices[j]]==printed:mesh.uvs[j]=[((tx+.02+u*.96)/8,(ty+.02+v*.96)/8) for u,v in mesh.uvs[j]]
        if kind=='carton':
            tape=material('CartonTape',(.58,.43,.25),.35)
            mesh.box((0,0,h+.001),(w*.12,d,.003),tape)
            if spec.get('shipping_label',True):
                mesh.box((0,-d/2-.002,h*.55),(w*.6,.003,h*.22),material('CartonLabel',(.72,.66,.51),.8))
    mesh.finish(root,bevel=0)
    return root


def build(source,out,profile_path):
    source=source.resolve();out=out.resolve();profile_path=profile_path.resolve()
    if out.exists() or source==out or source in out.parents or out in source.parents:
        raise ValueError('Use a new independent output directory')
    profile=json.loads(profile_path.read_text())
    if profile.get('schema_version')!='reviewed_retail_rebuild_v1':raise ValueError('Unsupported reviewed profile')
    if any(not math.isfinite(profile['room'][k]) or profile['room'][k]<=0 for k in ('width','depth','height')):
        raise ValueError('Invalid scene envelope')
    for light in profile['lights']:
        if len(light['position'])!=3 or not all(math.isfinite(v) for v in light['position']) or not 0<=light.get('energy',180)<10000:
            raise ValueError('Invalid light')
    bundle=json.loads((source/'scene_bundle.json').read_text())
    if str(bundle['scene_id'])!=str(profile['scene_id']):raise ValueError('Scene ID mismatch')
    expected={v['id'] for v in bundle['views']}
    if set(profile['cameras'])!=expected:raise ValueError('Every reference camera must be declared')
    ids=[o['id'] for o in profile['fixtures']+profile.get('props',[])+profile['shell']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate object IDs')
    # Validate the whole numeric profile before starting any mesh construction.
    for item in profile['fixtures']+profile.get('props',[])+profile['shell']:
        vals=item['position']+item['dimensions_m']
        if len(vals)!=6 or not all(math.isfinite(v) for v in vals) or min(item['dimensions_m'])<=0:
            raise ValueError('Invalid object bounds: '+item['id'])
    atlas=(profile_path.parent/profile['atlas']).resolve()
    if not atlas.is_file():raise ValueError('Missing packaging atlas')
    clearance=footprint_audit(profile['fixtures'],profile['room'])
    if not clearance['passed']:raise ValueError('Fixture layout rejected: '+json.dumps(clearance['issues']))
    fixed_furniture=profile['fixtures']+[p for p in profile.get('props',[]) if p.get('type')=='counter' and abs(p['position'][2])<.005]
    furniture_clearance=footprint_audit(fixed_furniture,profile['room'])
    furniture_clearance['scope']='Declared shelf/fridge and floor counter/desk rectangles in XY; excludes chairs, supported merchandise and architectural segments.'
    if not furniture_clearance['passed']:raise ValueError('Fixed furniture layout rejected: '+json.dumps(furniture_clearance['issues']))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=32
    scene.cycles.use_denoising=True;scene.view_settings.view_transform='AgX'
    scene.view_settings.exposure=profile.get('exposure',.6)
    scene.world=bpy.data.worlds.new('RetailWorld');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.8,.85,1,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.18
    objects=[];semantic=[];counts=[]
    for item in profile['shell']:
        mat=material('Shell_'+item['id'],item.get('color',[.72,.74,.70]),item.get('roughness',.7))
        obj=box(None,item['id'],item['position'],item['dimensions_m'],mat,0)
        obj.rotation_euler.z=math.radians(item.get('rotation_z_deg',0))
    for item in profile['fixtures']:
        root=bpy.data.objects.new(item['id'],None);bpy.context.collection.objects.link(root)
        fn=replace_fridge if item.get('type')=='fridge' else replace_shelf
        counts.append(fn(root,item,atlas,profile['scene_id']))
    stock_materials=_materials(atlas)
    for item in profile.get('props',[]):prop(item,stock_materials)
    for item in profile['fixtures']+profile.get('props',[]):
        category='shelf' if item in profile['fixtures'] and item.get('type')!='fridge' else item.get('category',item.get('type','box'))
        objects.append({'id':item['id'],'type':'reviewed_mesh','category':category,'position':item['position'],
                        'dimensions':item['dimensions_m'],'rotation_z_deg':item.get('rotation_z_deg',0)})
        semantic.append({'id':item['id'],'category':category,'xy':item['position'][:2],
                         'target_dimensions_m':item['dimensions_m'],'rotation_z_deg':item.get('rotation_z_deg',0),
                         'support_id':item.get('support_id','floor'),'clearance_m':0})
    for i,item in enumerate(profile['lights']):
        light=bpy.data.lights.new('CeilingLight_'+str(i),'AREA');light.energy=item.get('energy',180)
        light.shape='RECTANGLE';light.size=item.get('size',1.5);light.size_y=item.get('size_y',.6)
        light.color=(.90,.96,1);obj=bpy.data.objects.new(light.name,light);scene.collection.objects.link(obj);obj.location=item['position']
    for view in bundle['views']:
        c=profile['cameras'][view['id']];vals=c['position']+c['target']+[c['lens_mm'],c.get('roll_degrees',0)]
        if not all(math.isfinite(v) for v in vals) or not 8<c['lens_mm']<100:raise ValueError('Invalid camera')
        x,y,z=c['position'];room=profile['room']
        if not (-room['width']/2<x<room['width']/2 and 0<y<room['depth'] and .1<z<room['height']):
            raise ValueError('Camera outside scene envelope: '+view['id'])
        cam=bpy.data.cameras.new('ReferenceCamera_'+view['id']);obj=bpy.data.objects.new(cam.name,cam);scene.collection.objects.link(obj)
        obj.location=c['position'];direction=Vector(c['target'])-obj.location
        if direction.length<.01:raise ValueError('Degenerate camera')
        obj.rotation_euler=direction.to_track_quat('-Z','Y').to_euler();obj.rotation_euler.rotate_axis('Z',math.radians(c.get('roll_degrees',0)))
        cam.lens=c['lens_mm'];cam.sensor_width=36;cam.clip_end=100;scene.camera=obj;view['camera']=copy.deepcopy(c)
    scene.camera=bpy.data.objects['ReferenceCamera_'+bundle['views'][0]['id']]
    base=bundle['shared_scene']['base_scene'];base['objects']=objects;base['room']=profile['room']
    bundle['shared_scene']['camera']=copy.deepcopy(bundle['views'][0]['camera'])
    layout={'schema_version':'library_layout_v1','scene_id':str(profile['scene_id']),'room':profile['room'],
            'objects':semantic,'views':copy.deepcopy(bundle['views'])}
    bpy.context.view_layer.update()
    if any(not math.isfinite(c) for obj in scene.objects if obj.type=='MESH' for v in obj.data.vertices for c in v.co):
        raise ValueError('Non-finite mesh')
    out.mkdir(parents=True)
    bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    for name,value in [('scene_bundle.json',bundle),('initial_layout.json',layout),('preparation_profile.json',profile)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    report={'source':str(source),'profile_sha256':sha(profile_path),'atlas_sha256':sha(atlas),'replacements':counts,
            'prepared_files_sha256':{name:sha(out/name) for name in ('scene.blend','scene_bundle.json','initial_layout.json')},
            'geometry_build_count':1,'camera_dependent_geometry':False,'photo_background_planes':0,'fixture_clearance':clearance,
            'fixed_furniture_clearance':furniture_clearance,
            'limitations':profile.get('limitations',[]),'calibration':profile.get('calibration',{})}
    (out/'preparation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('REVIEWED_RETAIL_BUILT',json.dumps({'fixtures':len(counts),'props':len(profile.get('props',[])),'views':len(bundle['views'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','out','profile'):p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);build(a.source,a.out,a.profile)
