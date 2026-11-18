"""Reviewed, shop-specific structural correction for demo shop 1646301141.

Creates actual stocked shelving, endcaps and reference-derived product categories.
All source textures are small object-face crops, never a full-frame camera overlay.
Run in Blender. This is a documented manual correction, not automatic reconstruction.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(REPO/'scene_factory'))
from generate_v2 import configure_device
import generate_scene as g


from retail_stock_geometry import Mesh


def material(name,color,rough=.55,metal=0):
    return g.material('V5_'+name,color,rough,metal)


def image_material(path):
    path=path.resolve()
    name='V5_Reference_'+path.stem
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    image=bpy.data.images.load(str(path),check_existing=True);image.pack()
    mat['aspect']=image.size[0]/max(image.size[1],1)
    tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=image
    mat.node_tree.links.new(tex.outputs['Color'],shader.inputs['Base Color'])
    shader.inputs['Roughness'].default_value=.48
    mat['source_kind']='reviewed_sign_crop' if path.stem=='calligraphy' else 'reviewed_product_face_crop'
    mat['source_file']=str(path)
    return mat


def remove_root(name):
    root=bpy.data.objects.get(name)
    if root:
        for child in list(root.children_recursive):bpy.data.objects.remove(child,do_unlink=True)
        bpy.data.objects.remove(root,do_unlink=True)


def stock_gifts(builder,start,end,xfront,depth,height,mats,frame,rng):
    # Large gift cartons with four display rows, replacing arbitrary tiny packets.
    levels=[.10+(height-.10)*i/4 for i in range(5)]
    builder.box((xfront-depth,(start+end)/2,height/2),(.025,end-start,height),frame)
    for y in (start,end):
        builder.box((xfront-depth/2,y,height/2),(depth,.035,height),frame)
    for z in levels:
        builder.box((xfront-depth/2,(start+end)/2,z),(depth,end-start,.027),frame)
    for row,z in enumerate(levels[:-1]):
        y=start+.035
        while y<end-.08:
            w=min(rng.uniform(.44,.78) if row==3 else rng.uniform(.27,.53),end-y-.03)
            if w<.08:break
            h=min(levels[row+1]-z-.045,rng.uniform(.38,.51))
            d=depth*rng.uniform(.73,.95)
            themes=[[0,1,3,4],[1,3,4,6],[0,1,4,6],[0,2,3,4]]
            face=mats[rng.choice(themes[row])]
            if row==3 and rng.random()<.13:y+=w;continue
            builder.box((xfront-d/2-.008,y+w/2,z+.018+h/2),(d,w-.018,h),frame,face,face_axes=('+X',))
            builder.products+=1;y+=w
    return builder


def stocks(builder,front,x_or_y,z,length,depth,kind,mats,rng):
    white,red,gold,dark,green=mats['white'],mats['red'],mats['gold'],mats['dark'],mats['green']
    step=.07 if kind=='bottles' else .10
    for i in range(max(1,int(length/step))):
        for j in range(max(1,int(depth/.09))):
            if rng.random()<.06:continue
            along=-length/2+step*(i+.55)
            inset=.05+j*.09
            if front=='-Y':x,y=x_or_y[0]+along,x_or_y[1]+inset
            elif front=='+X':x,y=x_or_y[0]-inset,x_or_y[1]+along
            else:x,y=x_or_y[0]+inset,x_or_y[1]+along
            if kind in ('bottles','jars'):
                label=rng.choice(mats['labels'][:4]);h=rng.uniform(.18,.27) if kind=='bottles' else rng.uniform(.11,.16)
                builder.bottle(x,y,z,.027 if kind=='bottles' else .034,h,dark,label,gold if rng.random()<.28 else red,kind=='jars')
            elif kind=='packets':
                h=rng.uniform(.12,.20);w=.08
                face=rng.choice(mats.get('stock_faces',mats['labels'][4:]))
                builder.pouch(x,y,z,.08,.045,h,rng.choice([red,gold,green]),face,front)
            else:
                h=rng.uniform(.12,.22)
                builder.box((x,y,z+h/2),(.078,.085,h),rng.choice([red,gold,green]),rng.choice(mats.get('stock_faces',mats['gifts'])),face_axes=(front,));builder.products+=1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--textures',required=True,type=Path);parser.add_argument('--camera-fit',type=Path,help='Legacy camera 01 fit; reviewed scene configuration takes precedence')
    parser.add_argument('--scene-config',type=Path,help='Reviewed fixture dimensions and optional layered photo surfaces')
    parser.add_argument('--atlas',type=Path,help='Optional printed atlas for remaining source-scene products')
    parser.add_argument('--views',nargs='+',default=['view_01','view_08'])
    parser.add_argument('--edge',type=int,default=960);parser.add_argument('--samples',type=int,default=64)
    parser.add_argument('--save',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.edge<64 or args.samples<1:raise ValueError('edge >= 64 and samples >= 1 are required')
    source=args.source.resolve();out=args.out.resolve()
    if out==source:raise ValueError('Use a separate output directory')
    out.mkdir(parents=True,exist_ok=True)
    bundle=json.loads((source/'scene_bundle.json').read_text())
    if bundle['scene_id']!='1646301141':raise ValueError('This reviewed correction is shop-specific')
    if set(args.views)-{v['id'] for v in bundle['views']}:raise ValueError('Unknown view ID')
    bpy.ops.wm.open_mainfile(filepath=str(source/'scene.blend'));scene=bpy.context.scene
    scene_config=json.loads(args.scene_config.read_text()) if args.scene_config else None
    if scene_config and scene_config.get('detail_texture_dir'):
        detail_dir=Path(scene_config['detail_texture_dir'])
        if not detail_dir.is_absolute():scene_config['detail_texture_dir']=str((args.scene_config.parent/detail_dir).resolve())
    geometry=scene_config.get('geometry',{}) if scene_config else {}
    g.MATERIALS.clear();g.MESHES.clear()
    device=configure_device(scene,'AUTO');rng=random.Random(1646301141)
    mats={name:material(name,color,rough,metal) for name,color,rough,metal in [
        ('frame',[.34,.27,.18],.67,0),('white',[.57,.59,.54],.50,.15),
        ('red',[.44,.025,.022],.42,0),('gold',[.60,.34,.055],.45,.12),
        ('dark',[.055,.04,.025],.25,0),('green',[.07,.30,.065],.48,0)]}
    mats['gifts']=[image_material(args.textures/(name+'.png')) for name in
                   ['red_gift','red_pair','gold_gift','brown_gift','cream_gift','white_gift','red_slim']]
    mats['labels']=[image_material(args.textures/(name+'.png')) for name in ['jar_red','bottle_red','bottle_dark','jar_gold','packet_green','packet_red']]
    stock_dir=Path(scene_config['detail_texture_dir']) if scene_config and scene_config.get('detail_texture_dir') else args.textures
    mats['stock_faces']=[image_material(p) for p in sorted(stock_dir.glob('stock_face_*.png'))] or mats['labels'][4:]
    replacement=['gift_shelf_01','gift_shelf_02','gift_shelf_03','gift_shelf_04',
                 'main_display_01','main_display_02','divider_grocery_shelf','rear_storage_tree']
    # Remove independently imported items whose estimated supports are replaced.
    layout_path=Path(next(v['source_image'] for v in bundle['views'])).parents[2]/'initial_layout.json'
    original_layout=json.loads(layout_path.read_text())
    remove=set(replacement);changed=True
    while changed:
        changed=False
        for o in original_layout['objects']:
            if o.get('support_id') in remove and o['id'] not in remove:remove.add(o['id']);changed=True
    for name in remove:remove_root(name)
    builders=[]
    # A single sided continuous, timber-coloured gift cabinet.
    for i,(a,b) in enumerate([(2.10,4.30),(4.33,6.53),(6.56,8.76),(8.79,geometry.get('gift_end',10.13))]):
        builder=Mesh('V5_GiftCabinet_'+str(i))
        stock_gifts(builder,a,b,geometry.get('gift_front',-2.66),.40,geometry.get('gift_height',2.45),mats['gifts'],mats['frame'],rng)
        builder.finish();builders.append(builder)
    # Dense low island, with shelving on both long sides AND the end facing camera 01.
    builder=Mesh('V5_MainIsland');cx=geometry.get('island_center',-1.28);near=geometry.get('island_near',5.99);far=geometry.get('island_end',9.62);w=.62;h=1.05
    for z in [.08,.32,.57,.81,1.05]:
        builder.box((cx,(near+far)/2,z),(w,far-near,.023),mats['white'])
        builder.box((cx,near+.012,z),(w,.026,.028),mats['green'])
        for x in [cx-w/2,cx+w/2]:builder.box((x,(near+far)/2,z),(.018,far-near,.028),mats['green'])
    for x in [cx-w/2,cx+w/2]:
        for y in [near,far]:
            builder.box((x,y,h/2),(.025,.025,h),mats['white'])
    for k,z in enumerate([.10,.34,.59,.83]):
        stocks(builder,'-Y',(cx,near+.02),z,w-.07,.15,'boxes' if k<3 else 'jars',mats,rng)
        for face,x in [('+X',cx+w/2-.02),('-X',cx-w/2+.02)]:
            stocks(builder,face,(x,(near+far)/2+.12),z,far-near-.36,.15,'packets' if k<3 else 'bottles',mats,rng)
    # Gold-lidded jars and red sauce bottles dominate the near end, not generic cartons.
    for i in range(5):
        for j in range(4):
            x=cx-w*.39+i*.115;y=near+.06+j*.10
            builder.bottle(x,y,h+.013,.034,.14 if i<2 else .24,mats['dark'],mats['labels'][0 if i<2 else 1],mats['gold'],i<2)
    for y in [6.65,6.9,7.16,7.45,7.72]:
        for x in [cx-.21,cx-.03]:
            builder.box((x,y,1.21),(.15,.20,.28),mats['red'],mats['gifts'][1],face_axes=('-Y','+Y'));builder.products+=1
    for y in [8.08,8.42,8.8,9.12]:
        for x in [cx-.2,cx+.04]:
            builder.box((x,y,1.14),(.18,.22,.15),mats['gold'],rng.choice(mats['gifts']),face_axes=('-Y','+Y'));builder.products+=1
    builder.finish();builders.append(builder)
    # The tall divider is a stocked shop fixture, not a wall of random book-like boxes.
    builder=Mesh('V5_GroceryDivider');front=geometry.get('divider_front',-.185);ys=geometry.get('divider_near',5.95);ye=geometry.get('divider_end',9.70);depth=.45;hh=geometry.get('divider_height',2.18)
    builder.box((front+depth,(ys+ye)/2,hh/2),(.022,ye-ys,hh),mats['white'])
    for z in [.08,.43,.78,1.13,1.48,1.83,hh]:
        builder.box((front+depth/2,(ys+ye)/2,z),(depth,ye-ys,.026),mats['white'])
        builder.box((front,(ys+ye)/2,z+.012),(.025,ye-ys,.035),mats['green'])
    for y in [ye]:
        builder.box((front+depth/2,y,hh/2),(depth,.023,hh),mats['white'])
    for k,z in enumerate([.11,.46,.81,1.16,1.51,1.86]):
        stocks(builder,'-X',(front+.025,(ys+ye)/2),z,ye-ys-.09,.35,['packets','packets','packets','jars','bottles','packets'][k],mats,rng)
    for z in [.08,.43,.78,1.13,1.48,1.83]:
        stocks(builder,'-Y',(front+depth/2,ys+.015),z+.04,depth-.045,.16,'packets' if z<1 else 'bottles',mats,rng)
    builder.finish();builders.append(builder)
    # Wheat arrangement: individual stalk geometry in a gold cylindrical vase.
    wheat=Mesh('V5_WheatArrangement');wx,wy=cx+.19,6.80
    wheatmat=material('orange_wheat',[.61,.23,.025],.78)
    wheat.bottle(wx,wy,1.06,.065,.26,mats['gold'],mats['gold'],mats['gold'],True)
    for _ in range(95):
        t=rng.random()*math.tau;rr=rng.uniform(.03,.18);top=1.06+rng.uniform(.51,.71)
        tx=wx+rr*math.cos(t);ty=wy+rr*math.sin(t)
        a=Vector((wx,wy,1.20));b=Vector((tx,ty,top))
        v=b-a
        # Narrow triangular stalks are real mesh; seed heads are small tapered wedges.
        for frac in (0,.03):
            wheat.face([a+Vector((-.001,0,0)),a+Vector((.001,0,0)),b+Vector((.001,0,0)),b-Vector((.001,0,0))],wheatmat)
        for k in range(5):
            z=top-k*.017
            wheat.box((tx+rng.uniform(-.008,.008),ty,z),(.009,.009,.026),wheatmat)
    wheat.finish(bevel=0);builders.append(wheat)
    # The model's storage_tree asset was a solid green item. Replace it with a plant.
    leaves=Mesh('V5_RearPlant');px,py=geometry.get('plant_position',[-.48,10.48])
    leaves.bottle(px,py,0,.17,.31,mats['frame'],mats['frame'],mats['frame'],True)
    for i in range(90):
        az=rng.random()*math.tau;rad=rng.uniform(.06,.48);z=rng.uniform(1.18,1.87)
        point=Vector((px+rad*math.cos(az),py+rad*math.sin(az),z))
        direction=Vector((math.cos(az),math.sin(az),-.3))*rng.uniform(.10,.20)
        side=Vector((-math.sin(az),math.cos(az),0))*.065
        leaves.face([point-direction,point+side,point+direction,point-side],mats['green'])
    leaves.finish(bevel=0);builders.append(leaves)
    # Move loose cartons to the observed foreground wall-side pile.
    for o in list(scene.objects):
        if o.parent is None and o.name.startswith('wall_stock_carton'):
            o.location.y-=2.6
    chair=bpy.data.objects.get('carved_wooden_chair')
    if chair:chair.location.x=-2.90;chair.location.y=5.35
    # An interior rear door, missing entirely in the previous reconstruction.
    door=Mesh('V5_BackDoor')
    door.box((.36,11.40,1.1),(.82,.055,2.2),mats['white'])
    door.box((.36,11.36,1.14),(.68,.018,1.94),material('door',[.20,.22,.22],.65))
    door.box((.36,11.345,.26),(.68,.012,.37),mats['red'])
    door.box((.66,11.32,1.06),(.025,.045,.13),mats['gold'])
    door.finish();builders.append(door)
    # Four separate header panels retain plausible image aspect ratios.
    poster=image_material(args.textures/'calligraphy.png')
    for i,y in enumerate([3.2,5.43,7.66,9.46]):
        header=Mesh('V5_HeaderPanel_'+str(i))
        header.box((-2.675,y,2.67),(.035,1.8 if i<3 else 1.1,.43),mats['frame'],poster)
        header.finish(bevel=0)
    # A short wall pier hides the near cabinet return.
    pier=Mesh('V5_LeftPier')
    pier.box((-2.91,4.93,1.56),(.26,.20,3.12),mats['white'])
    pier.finish();builders.append(pier)
    # Deliberate floor-stock zones preserve the walkable center of both aisles.
    cartons=Mesh('V5_FloorStock')
    kraft=material('kraft',[.40,.27,.14],.86)
    for side in ('left','right'):
        for i in range(11):
            y=6.0+i*.44
            x=(-2.35+rng.uniform(-.18,.12)) if side=='left' else (.25+rng.uniform(-.12,.17))
            height=0
            for layer in range(rng.randint(1,3)):
                w=rng.uniform(.30,.49);d=rng.uniform(.33,.47);h=rng.uniform(.20,.37)
                face=rng.choice([kraft,kraft,kraft,mats['gifts'][0],mats['gifts'][2]])
                cartons.box((x,y,height+h/2),(w,d,h),face);height+=h
    cartons.finish();builders.append(cartons)
    for obj in list(scene.objects):
        if obj.parent is None and obj.name.startswith(('rear_carton','rear_blue_jug','rear_floor_jug','rear_open_carton','rear_loose_gift','rear_low_crate','rear_cleaning')):
            if obj.name.startswith(('rear_carton','rear_open_carton')):
                for child in obj.children:
                    if child.type=='MESH':
                        for slot in child.material_slots:slot.material=kraft
    table=bpy.data.objects.get('tea_table')
    if table:table.location.y-=.7
    camera_fit=json.loads(args.camera_fit.read_text()) if args.camera_fit else {}
    corrected=(scene_config or {}).get('cameras',{}).get('view_01') or camera_fit.get('camera')
    if not corrected:raise ValueError('Provide a scene-config with camera 01 or --camera-fit')
    cam=bpy.data.objects['ReferenceCamera_view_01'];cam.location=corrected['position']
    cam.rotation_euler=(Vector(corrected['target'])-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.rotation_euler.rotate_axis('Z',math.radians(corrected['roll_degrees']));cam.data.lens=corrected['lens_mm']
    surface_report=None
    if scene_config and scene_config.get('surface_detail',False):
        from demo_surface_detail import apply_reviewed_scene
        surface_report=apply_reviewed_scene(scene,bundle,scene_config,Mesh,material,remove_root,args.textures)
    elif scene_config:
        from demo_geometry_detail import apply_geometry_detail
        surface_report=apply_geometry_detail(scene,bundle,scene_config,Mesh,material,remove_root,original_layout)
    # In source 01 the floor is a neutral warm grey, not a smooth white slab.
    floor=bpy.data.materials.get('Floor')
    if floor:
        shader=next(n for n in floor.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        shader.inputs['Base Color'].default_value=(.43,.43,.37,1);shader.inputs['Roughness'].default_value=.46
    from refine_demo_blender import refine_materials
    refine_materials(scene,args.atlas,False,True)
    scene.view_settings.exposure+=scene_config.get('exposure_offset',.50) if scene_config else .50
    if scene_config and scene_config.get('view_transform'):
        scene.view_settings.view_transform=scene_config['view_transform']
        scene.view_settings.look='None'
        scene.view_settings.exposure=scene_config.get('exposure',0)
    if scene_config:
        light_factor=scene_config.get('illumination_factor',1.0)
        for light in scene.objects:
            if light.type=='LIGHT':
                factor=next((value for key,value in scene_config.get('light_energy_factors',{}).items()
                             if key in light.name),light_factor)
                light.data.energy*=factor
        if scene.world and scene.world.use_nodes:
            for node in scene.world.node_tree.nodes:
                if node.type=='BACKGROUND':
                    node.inputs['Strength'].default_value=scene_config.get('world_strength',node.inputs['Strength'].default_value*light_factor)
        for i,spec in enumerate(scene_config.get('fill_lights',[])):
            data=bpy.data.lights.new('V5_ReviewFill_'+str(i),'AREA');data.energy=spec['energy'];data.shape='DISK';data.size=spec['size']
            data.color=spec.get('color',[1,1,1])
            lamp=bpy.data.objects.new(data.name,data);bpy.context.collection.objects.link(lamp)
            lamp.location=spec['position'];lamp.rotation_euler=(Vector(spec['target'])-lamp.location).to_track_quat('-Z','Y').to_euler()
    scene.cycles.samples=args.samples;scene.cycles.use_denoising=True
    scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    report={'shop':'1646301141','source':str(source),'manual_review':True,'device':device,
            'changes':{'replaced_roots':sorted(remove),'fixture_mesh_count':sum(o.type=='MESH' and o.name.startswith('V5_') for o in scene.objects),
                       'counted_stock_units':sum(o.get('counted_stock_units',0) for o in scene.objects),
                       'cameras':{v['id']:v['camera'] for v in bundle['views'] if v['id'] in args.views}},
            'surface_refinement':surface_report,
            'representation':'layered_photo_surfaces' if scene_config and scene_config.get('surface_detail',False) else 'physical_meshes_with_fixed_object_textures',
            'limitations':['Manual scene-specific correction, not an automatic reconstruction result.','Object texture crops retain original illumination; product identity and metric dimensions are estimates.','Camera poses, scale and hidden stock geometry are estimated; counted stock units exclude uncounted props and cartons.'],'views':[]}
    for view_id in args.views:
        view=next(v for v in bundle['views'] if v['id']==view_id);scene.camera=bpy.data.objects['ReferenceCamera_'+view_id]
        f=args.edge/max(view['width'],view['height']);scene.render.resolution_x=round(view['width']*f);scene.render.resolution_y=round(view['height']*f)
        scene.render.filepath=str(out/('rgb_'+view_id+'.png'));bpy.context.view_layer.update();start=time.monotonic()
        bpy.ops.render.render(write_still=True)
        report['views'].append({'id':view_id,'seconds':time.monotonic()-start,'file':scene.render.filepath})
        (out/'layout_refinement.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if args.save:
        bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    (out/'scene_bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2))
    if scene_config:
        (out/'reviewed_scene_config.json').write_text(json.dumps(scene_config,ensure_ascii=False,indent=2))
    print('LAYOUT_REFINEMENT_COMPLETE',json.dumps({'out':str(out),'views':len(report['views']),
          'surface_objects':surface_report['fixture_objects'] if surface_report else []},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
