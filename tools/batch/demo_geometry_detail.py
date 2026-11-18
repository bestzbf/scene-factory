"""Physical, camera-independent detail for the reviewed single-shop demo.

All stock, leaves and fixtures are actual meshes. Product and carton textures are
fixed UV object crops; no camera projection or view-dependent appearance is used.
"""
import math
import random
from pathlib import Path

import bpy
from mathutils import Vector


def support_family(objects,roots):
    result=set(roots)
    changed=True
    while changed:
        changed=False
        for entry in objects:
            if entry.get('support_id') in result and entry['id'] not in result:
                result.add(entry['id']);changed=True
    return result


def textured_material(path,name):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    image=bpy.data.images.load(str(path),check_existing=True);image.pack()
    node=mat.node_tree.nodes.new('ShaderNodeTexImage');node.image=image;node.extension='EXTEND'
    mat.node_tree.links.new(node.outputs['Color'],shader.inputs['Base Color'])
    shader.inputs['Roughness'].default_value=.83
    mat['aspect']=image.size[0]/max(1,image.size[1])
    mat['source_kind']='fixed_product_face_crop';mat['source_file']=str(path)
    return mat


def apply_geometry_detail(scene,bundle,config,Mesh,material,remove_root,layout):
    rng=random.Random(8646301141);geo=config['geometry']
    for vid,values in config['cameras'].items():
        camera=bpy.data.objects['ReferenceCamera_'+vid];camera.location=values['position']
        camera.rotation_euler=(Vector(values['target'])-camera.location).to_track_quat('-Z','Y').to_euler()
        camera.rotation_euler.rotate_axis('Z',math.radians(values.get('roll_degrees',0)))
        camera.data.lens=values['lens_mm']
        next(v for v in bundle['views'] if v['id']==vid)['camera']=values
    # Source support relationships are semantic, not Blender object parenting.
    # Move the tea service with its table so cups no longer float at the old position.
    table=bpy.data.objects.get('tea_table')
    if table:
        original=next(e for e in layout['objects'] if e['id']=='tea_table')
        target_y=config.get('tea_table_y',3.8);delta=target_y-original['xy'][1]
        for name in support_family(layout['objects'],{'tea_table'}):
            obj=bpy.data.objects.get(name)
            if not obj:continue
            if name=='tea_table':obj.location.y=target_y
            else:obj.location.y+=delta
    # Replace the rear pile as a group; supports cannot survive the removed cartons.
    roots={e['id'] for e in layout['objects'] if e['id'].startswith(
        ('rear_carton','rear_blue_jug','rear_floor_jug','rear_open_carton','rear_loose_gift','rear_low_crate','rear_cleaning'))}
    removed=support_family(layout['objects'],roots)
    for name in removed:remove_root(name)
    for name in ('V5_RearPlant','V5_FloorStock','V5_BackDoor'):remove_root(name)
    kraft=material('carton_body',[.30,.21,.13],.86)
    ink=material('carton_ink',[.16,.055,.03],.86)
    tape=material('carton_tape',[.43,.34,.20],.65)
    rim=material('carton_edge',[.35,.25,.15],.88)
    texture_dir=Path(config['detail_texture_dir']) if config.get('detail_texture_dir') else None
    carton_mats=[]
    if texture_dir:
        carton_mats=[textured_material(p,'V5_CartonFace_'+p.stem) for p in sorted(texture_dir.glob('carton_*.png'))]
    case_mats=[textured_material(p,'V5_CaseFace_'+p.stem) for p in sorted(texture_dir.glob('case_*.png'))] if texture_dir else []
    if not carton_mats:carton_mats=[kraft]
    blue=material('water_jug',[.025,.11,.32],.19)
    label=material('water_label',[.63,.68,.68],.42)
    cap=material('water_cap',[.015,.04,.13],.38)
    green=material('stock_carton_green',[.07,.27,.055],.69)
    red=material('stock_carton_red',[.49,.024,.019],.57)
    cream=material('stock_carton_cream',[.58,.49,.27],.71)
    dark=material('box_contents',[.05,.043,.028],.85)
    white=material('tissue_wrap',[.70,.73,.73],.66)
    turquoise=material('tissue_cyan',[.10,.38,.39],.54)
    builders=[]
    # Boxes have separate top faces, tape and imperfectly aligned stacks.
    def carton(builder,x,y,z,w,d,h,printed=True,open_box=False,colour=None,case=False):
        face=rng.choice(case_mats) if case and case_mats else rng.choice(carton_mats) if printed else None
        base=colour or kraft
        if open_box:
            builder.box((x,y,z+.012),(w,d,.024),base)
            for dx,dy,ww,dd in [(-w/2+.01,0,.02,d),(w/2-.01,0,.02,d),(0,-d/2+.01,w,.02),(0,d/2-.01,w,.02)]:
                builder.box((x+dx,y+dy,z+h/2),(ww,dd,h),base,face)
            builder.box((x,y,z+h*.57),(w*.90,d*.9,.02),dark)
            # Angled flaps make the open top silhouette visible from both cameras.
            for side in (-1,1):
                builder.face([(x+side*w/2,y-d/2,z+h),(x+side*w/2,y+d/2,z+h),
                              (x+side*(w/2+.12),y+d/2,z+h-.045),(x+side*(w/2+.12),y-d/2,z+h-.045)],base)
        else:
            builder.box((x,y,z+h/2),(w,d,h),base,face)
            builder.box((x,y,z+h+.001),(.032,d,.002),tape)
            builder.box((x,y-d/2-.001,z+h*.78),(.032,.002,h*.42),tape)
        # Small physical red rules suggest shipping print on distant boxes.
        if not printed:
            for dz in [.16,.20]:builder.box((x,y-d/2-.001,z+h*dz),(w*.62,.002,.006),ink)
    pile=Mesh('V5_PhysicalRearStock')
    jug_positions=[]
    for col,x in enumerate([-2.72,-2.14,-1.56]):
        for depth_index,y in enumerate([11.45,12.02]):
            z=0
            for layer in range((4 if col<2 else 2)+depth_index):
                w,d,h=rng.uniform(.45,.57),rng.uniform(.43,.54),rng.uniform(.37,.48)
                xc=x+rng.uniform(-.035,.035);yc=y+rng.uniform(-.04,.04)
                carton(pile,xc,yc,z,w,d,h,open_box=(col==2 and layer==1 and depth_index==0))
                z+=h
            if col==1:jug_positions.append((x,y,z))
    for x,y,z in jug_positions+[(-1.05,11.45,0),(-.98,11.82,0),(-1.13,12.25,0)]:
        pile.bottle(x,y,z,.135,.48,blue,label,cap)
    for x,y in [(-1.18,11.0),(-.72,11.40),(-.71,11.93)]:
        carton(pile,x,y,0,.38,.42,.32,open_box=True)
        pile.box((x,y,.32),(.26,.30,.13),turquoise)
    pile.finish(bevel=.0018);builders.append(pile)
    floor=Mesh('V5_PhysicalFloorStock')
    for side in (-1,1):
        for index in range(9):
            x=(-2.28 if side<0 else geo['divider_front']-.31)+rng.uniform(-.07,.07)
            y=6.20+index*.45;z=0
            for layer in range(rng.randint(1,3)):
                w,d,h=rng.uniform(.31,.43),rng.uniform(.31,.41),rng.uniform(.22,.33)
                colour=kraft if side<0 else rng.choice([green,red,cream])
                carton(floor,x,y,z,w,d,h,printed=side<0,colour=colour,open_box=(layer==0 and index%4==0),case=side>0)
                z+=h
            if side>0 and index%3==0:
                for i in range(3):floor.bottle(x-.1+i*.10,y,z,.034,.24,green,label,rng.choice([red,cap,cream]))
    floor.finish(bevel=.0016);builders.append(floor)
    # A compact stock rack across the rear wall, replacing the generic long side rack.
    remove_root('rear_household_shelf')
    rack=Mesh('V5_PhysicalRearRack');frame=material('metal_shelf',[.30,.32,.28],.55,.15)
    for z in [.10,.47,.84,1.21,1.58]:rack.box((-1.76,10.22,z),(1.48,.36,.025),frame)
    for x in [-2.49,-1.03]:rack.box((x,10.22,.84),(.024,.36,1.68),frame)
    for row,z in enumerate([.13,.50,.87,1.24]):
        for i in range(9):
            x=-2.39+i*.15;y=10.20
            rack.box((x,y,z+.09),(.12,.20,rng.uniform(.11,.22)),rng.choice([white,turquoise,cream,red]))
    rack.finish();builders.append(rack)
    # Palmate broadleaf clusters replace triangular tree silhouettes.
    tree=Mesh('V5_PhysicalPlant');px,py=geo['plant_position']
    pot=material('plant_pot',[.26,.27,.23],.78)
    bark=material('plant_bark',[.18,.12,.065],.87)
    tree.box((px,py,.16),(.30,.30,.32),pot)
    for offset in [-.02,0,.02]:tree.bottle(px+offset,py,.29,.011,1.0,bark,bark,bark,True)
    leaves=[material('broadleaf_'+str(i),[.016+.007*i,.075+.023*i,.014+.007*i],.4) for i in range(5)]
    for cluster in range(50):
        az=rng.random()*math.tau;radius=rng.uniform(.12,.40);height=rng.uniform(1.42,2.05)
        center=Vector((px+radius*math.cos(az),py+radius*math.sin(az),height))
        base=Vector((px,py,height-.29));side=Vector((.0015,0,0))
        tree.face([base-side,base+side,center+side,center-side],bark)
        for leaf in range(5):
            angle=az+leaf*math.tau/5
            axis=Vector((math.cos(angle),math.sin(angle),rng.uniform(-.35,.15)))*rng.uniform(.13,.20)
            cross=Vector((-math.sin(angle),math.cos(angle),0))*rng.uniform(.035,.05)
            middle=center+axis*.45
            points=[]
            for j in range(12):
                a=j*math.tau/12;p=middle+axis*.55*math.cos(a)+cross*math.sin(a)
                p.z-=.022*math.sin(a)**2;points.append(p)
            tree.face(points,rng.choice(leaves),[(.5+.5*math.cos(j*math.tau/12),.5+.5*math.sin(j*math.tau/12)) for j in range(12)])
    tree.finish(bevel=0);builders.append(tree)
    # Rear door belongs to the perimeter, not the aisle in front of camera 08.
    door=Mesh('V5_PhysicalDoor');door.box((.32,12.54,1.1),(.78,.05,2.20),frame)
    door.box((.32,12.509,1.14),(.65,.016,1.92),material('door_grey',[.09,.105,.10],.7))
    door.box((.32,12.497,.26),(.65,.01,.38),red)
    door.finish();builders.append(door)
    chair=bpy.data.objects.get('carved_wooden_chair')
    if chair:chair.location.x=-2.27;chair.location.y=5.30
    bpy.context.view_layer.update()
    return {'method':'manually reviewed actual mesh geometry; fixed object textures',
            'fixture_objects':[b.name for b in builders],'removed_source_roots':sorted(removed),
            'support_group_moves':['tea_table and all transitive supports'],
            'cameras':config['cameras'],'geometry':geo,
            'limitations':['Object positions, scale and camera parameters are manual estimates.',
                           'Stock types and density approximate the photographs; exact SKUs are not recovered.']}
