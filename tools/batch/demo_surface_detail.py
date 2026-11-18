"""Manual image-based fixture detail for the 1646301141 demo experiment.

Photo pixels are assigned to individual, subdivided object faces. There is no camera
background or full-frame overlay. Stock clusters remain a layered approximation,
with baked source illumination and estimated hidden sides, rather than recovered SKUs.
"""
import json
import math
import random
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def set_camera(camera, values):
    camera.location=values['position']
    camera.rotation_euler=(Vector(values['target'])-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.rotation_euler.rotate_axis('Z',math.radians(values.get('roll_degrees',0)))
    camera.data.lens=values['lens_mm']


class PhotoFaces:
    """Perspective-correct UV samples on real object surfaces, with explicit provenance."""
    def __init__(self,scene,bundle,Mesh):
        self.scene=scene;self.Mesh=Mesh;self.images={};self.materials={};self.faces=[]
        for view in bundle['views']:
            if view['id'] in ('view_01','view_08'):
                self.images[view['id']]=bpy.data.images.load(view['source_image'],check_existing=True)
                self.images[view['id']].name='V5_Source_'+view['id']
                self.images[view['id']].pack()

    def mat(self,view):
        if view not in self.materials:
            mat=bpy.data.materials.new('V5_PhotoStock_'+view);mat.use_nodes=True
            nodes,links=mat.node_tree.nodes,mat.node_tree.links
            output=next(n for n in nodes if n.type=='OUTPUT_MATERIAL')
            for node in list(nodes):
                if node!=output:nodes.remove(node)
            tex=nodes.new('ShaderNodeTexImage');tex.image=self.images[view]
            tex.extension='EXTEND';tex.interpolation='Linear'
            emission=nodes.new('ShaderNodeEmission');emission.inputs['Strength'].default_value=1
            links.new(tex.outputs['Color'],emission.inputs['Color'])
            links.new(emission.outputs[0],output.inputs['Surface'])
            mat['source_kind']='fixed_uv_object_surface'
            mat['limitations']='Fixed UVs on layered photo stock clusters. Baked source illumination.'
            self.materials[view]=mat
        return self.materials[view]

    def add(self,builder,corners,view='view_01',quad=None,steps=(4,4),record=True):
        camera=bpy.data.objects['ReferenceCamera_'+view]
        if record:
            self.faces.append({'object':builder.name,'source_view':view,'world_corners':[list(p) for p in corners],
                               'source_quad_px':quad,'representation':'subdivided stock-cluster face'})
        # Subdivision is needed because source perspective UVs are rational, not affine.
        def point(u,v):
            return (Vector(corners[0])*(1-u)*(1-v)+Vector(corners[1])*u*(1-v)+
                    Vector(corners[2])*u*v+Vector(corners[3])*(1-u)*v)
        nx,ny=steps
        for iy in range(ny):
            for ix in range(nx):
                local=[(ix/nx,iy/ny),((ix+1)/nx,iy/ny),((ix+1)/nx,(iy+1)/ny),(ix/nx,(iy+1)/ny)]
                points=[point(u,v) for u,v in local];uv=[]
                for xyz,(u,v) in zip(points,local):
                    if quad:
                        q=[sum(quad[j][k]*w for j,w in enumerate([(1-u)*(1-v),u*(1-v),u*v,(1-u)*v])) for k in range(2)]
                        im=self.images[view];uv.append((q[0]/im.size[0],1-q[1]/im.size[1]))
                    else:
                        p=world_to_camera_view(self.scene,camera,xyz);uv.append((p.x,p.y))
                builder.face(points,self.mat(view),uv)


def apply_reviewed_scene(scene,bundle,config,Mesh,material,remove_root,textures):
    if config.get('shop')!='1646301141':
        raise ValueError('Reviewed geometry only applies to shop 1646301141')
    rng=random.Random(5646301141);geo=config['geometry']
    for vid,values in config['cameras'].items():
        set_camera(bpy.data.objects['ReferenceCamera_'+vid],values)
        next(v for v in bundle['views'] if v['id']==vid)['camera']=values
    bpy.context.view_layer.update()
    faces=PhotoFaces(scene,bundle,Mesh)
    wood=material('cabinet_oak',[.27,.20,.12],.78)
    dark=material('shelf_shadow',[.085,.075,.05],.82)
    frame=material('shelf_grey',[.33,.35,.31],.56,.15)
    red=material('stock_red',[.36,.018,.015],.48)
    gold=material('stock_lid',[.46,.30,.07],.36,.13)
    kraft=material('carton_kraft',[.31,.22,.135],.89)
    white=material('white_packet',[.70,.73,.71],.65)
    cyan=material('cyan_packet',[.12,.37,.39],.48)
    green=material('stock_green',[.05,.22,.035],.54)
    blue=material('jug_blue',[.055,.14,.29],.22)
    blue_shader=next(n for n in blue.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    transmission=blue_shader.inputs.get('Transmission Weight')
    if transmission is not None:transmission.default_value=.42
    blue_shader.inputs['Alpha'].default_value=.58
    blue.surface_render_method='DITHERED'
    bark=material('tree_bark',[.22,.14,.07],.9)
    # Remove only experiment meshes and explicitly reviewed source objects.
    for obj in list(scene.objects):
        if obj.name.startswith(('V5_GiftCabinet','V5_MainIsland','V5_GroceryDivider','V5_WheatArrangement',
                                'V5_RearPlant','V5_HeaderPanel','V5_FloorStock','V5_BackDoor','V5_LeftPier')):
            bpy.data.objects.remove(obj,do_unlink=True)
    source=Path(next(v['source_image'] for v in bundle['views'])).parents[2]
    layout=json.loads((source/'initial_layout.json').read_text())
    remove={o['id'] for o in layout['objects'] if o['id'].startswith(
        ('rear_carton','rear_blue_jug','rear_floor_jug','rear_open_carton','rear_loose_gift','rear_low_crate','rear_cleaning'))}
    remove.update(['rear_household_shelf','wall_stock_carton_01','wall_stock_carton_02','wall_stock_carton_03','wall_stock_carton_04'])
    changed=True
    while changed:
        changed=False
        for obj in layout['objects']:
            if obj.get('support_id') in remove and obj['id'] not in remove:
                remove.add(obj['id']);changed=True
    for name in remove:remove_root(name)
    objects=[];cx=geo['island_center'];near=geo['island_near'];far=geo['island_end']
    xf=geo['gift_front'];ye=geo['gift_end'];gh=geo['gift_height']
    # The cabinet uses the reference's uneven coloured parcels on separate shallow boxes.
    cabinet=Mesh('V5_ReviewedGiftCabinet');levels=[.09+(.0+gh-.09)*i/4 for i in range(5)]
    cabinet.box((xf-.37,7.45,gh/2),(.027,ye-4.45,gh),dark)
    for z in levels:cabinet.box((xf-.18,(4.45+ye)/2,z),(.39,ye-4.45,.026),wood)
    for y in [4.45,6.12,geo['gift_mid'],8.88,ye]:
        cabinet.box((xf-.18,y,gh/2),(.39,.028,gh),wood)
    for zi,(low,high) in enumerate(zip(levels[:-1],levels[1:])):
        y=4.48
        while y<ye-.07:
            length=min(rng.uniform(.32,.62),ye-y-.025)
            if length<.04:break
            front=xf+rng.uniform(-.012,.012)
            cabinet.box((front-.145,y+length/2,(low+high)/2),(.28,length-.002,high-low-.008),dark)
            faces.add(cabinet,[(front,y,low+.004),(front,y+length-.002,low+.004),
                               (front,y+length-.002,high-.004),(front,y,high-.004)],steps=(4,4))
            y+=length
    # The header is a physical wall panel, sampled only over its object surface.
    cabinet.box((xf-.03,(4.45+ye)/2,gh+.23),(.05,ye-4.45,.46),wood)
    faces.add(cabinet,[(xf+.002,4.45,gh+.02),(xf+.002,ye,gh+.02),
                       (xf+.002,ye,gh+.46),(xf+.002,4.45,gh+.46)],steps=(48,4))
    objects.append(cabinet.finish())
    # Stocked endcap and two long sides retain separate levels and real shelf depth.
    island=Mesh('V5_ReviewedIsland');w=.62;island_levels=[.08,.32,.57,.81,1.05]
    for z in island_levels:
        island.box((cx,(near+far)/2,z),(w,far-near,.023),frame)
        island.box((cx,near,z+.015),(w,.024,.035),wood)
    for x in [cx-w/2,cx+w/2]:
        for y in [near,far]:island.box((x,y,.55),(.019,.019,1.1),frame)
    for lo,hi in zip(island_levels[:-1],island_levels[1:]):
        # The near end is projected from its own photo, avoiding repeated generic cartons.
        for j in range(5):
            a=cx-w/2+.012+j*(w-.024)/5;b=a+(w-.024)/5-.006
            yy=near+.006+rng.uniform(0,.008)
            island.box(((a+b)/2,yy+.040,(lo+hi)/2),(b-a,.07,hi-lo-.026),dark)
            faces.add(island,[(a,yy,lo+.018),(b,yy,lo+.018),(b,yy,hi-.014),(a,yy,hi-.014)],steps=(2,3))
        # Long-side reference is a reviewed stock strip; unknown rear stock remains approximate.
        for side in (-1,1):
            xx=cx+side*(w/2-.003)
            corners=[(xx,near+.02,lo+.018),(xx,far-.02,lo+.018),
                     (xx,far-.02,hi-.014),(xx,near+.02,hi-.014)]
            # BL, BR, TR, TL in source pixels; visible island crop excludes plant and aisle.
            z0=(lo+.018)/1.05;z1=(hi-.014)/1.05
            def side_pixel(t,z):
                low=[211+232*t,193+207*t];high=[190+185*t,87+110*t]
                return [low[k]*(1-z)+high[k]*z for k in range(2)]
            quad=[side_pixel(0,z0),side_pixel(1,z0),side_pixel(1,z1),side_pixel(0,z1)]
            faces.add(island,corners,'view_08',quad,steps=(24,3))
    objects.append(island.finish())
    # Individual sauce jars and bottles reproduce the near-end product grouping.
    products=Mesh('V5_ReviewedIslandProducts')
    label_red=bpy.data.materials['V5_Reference_bottle_red'];label_jar=bpy.data.materials['V5_Reference_jar_red']
    label_dark=bpy.data.materials['V5_Reference_bottle_dark']
    for i in range(6):
        for j in range(4):
            jar=i<3;x=cx-.255+i*.095;y=near+.07+j*.095
            products.bottle(x,y,1.063,.028,.14 if jar else .27,dark,label_jar if jar else label_red,gold,jar)
    giftmat=bpy.data.materials['V5_Reference_red_pair']
    for i,y in enumerate([6.63,6.87,7.12,7.38,7.66]):
        for x in [cx-.20,cx-.025]:
            z=1.19+rng.uniform(-.01,.02)
            products.box((x,y,z),(.15,.20,.25),red,giftmat,face_axes=('-Y','+Y'))
            faces.add(products,[(x-.075,y-.101,z-.125),(x+.075,y-.101,z-.125),
                                (x+.075,y-.101,z+.125),(x-.075,y-.101,z+.125)],steps=(3,3))
            faces.add(products,[(x-.075,y-.10,z+.126),(x+.075,y-.10,z+.126),
                                (x+.075,y+.10,z+.126),(x-.075,y+.10,z+.126)],steps=(3,3))
    for i,y in enumerate([7.92,8.18,8.42,8.70,8.94,9.22,9.45]):
        for j,x in enumerate([cx-.18,cx+.06]):
            z=1.08;hh=rng.uniform(.08,.17)
            mat=white if (i+j)%3 else cyan
            products.box((x,y,z+hh/2),(.20,.22,hh),mat)
            faces.add(products,[(x-.10,y-.11,z+hh+.001),(x+.10,y-.11,z+hh+.001),
                                (x+.10,y+.11,z+hh+.001),(x-.10,y+.11,z+hh+.001)],steps=(3,3))
            if i%3==0:products.box((x+.014,y-.016,z+hh+.028),(.17,.20,.056),white)
            if i%3==1:products.bottle(x+.06,y,1.06,.036,.24,dark,label_dark,gold)
    objects.append(products.finish())
    # The orange stems sit beside the red gift stacks, not at the far end.
    wheat=Mesh('V5_ReviewedWheat');wx,wy=cx+.20,6.99
    orange=material('orange_wheat',[.67,.24,.025],.78)
    wheat.bottle(wx,wy,1.06,.055,.26,gold,gold,gold,True)
    for i in range(160):
        angle=rng.random()*math.tau;r=rng.uniform(.03,.18);height=rng.uniform(1.60,1.80)
        tip=Vector((wx+r*math.cos(angle),wy+r*math.sin(angle),height));base=Vector((wx,wy,1.20))
        side=Vector((.0012,0,0));wheat.face([base-side,base+side,tip+side,tip-side],orange)
        for k in range(5):wheat.box((tip.x,tip.y,tip.z-.012*k),(.007,.007,.018),orange)
    objects.append(wheat.finish(bevel=0))
    # Right divider: dense heterogeneous photo stock only on actual shelf fronts.
    divider=Mesh('V5_ReviewedDivider');front=geo['divider_front'];ys=geo['divider_near'];de=geo['divider_end'];dh=geo['divider_height']
    divider.box((front+.43,(ys+de)/2,dh/2),(.025,de-ys,dh),dark)
    div_levels=[.08,.40,.74,1.08,1.42,1.76,dh]
    for z in div_levels:divider.box((front+.21,(ys+de)/2,z),(.44,de-ys,.024),frame)
    for lo,hi in zip(div_levels[:-1],div_levels[1:]):
        y=ys+.02
        while y<de-.06:
            length=min(rng.uniform(.20,.33),de-y-.02)
            if length<.03:break
            xx=front+rng.uniform(.002,.022)
            divider.box((xx+.155,y+length/2,(lo+hi)/2),(.30,length-.01,hi-lo-.03),dark)
            faces.add(divider,[(xx,y,lo+.015),(xx,y+length-.012,lo+.015),
                               (xx,y+length-.012,hi-.015),(xx,y,hi-.015)],steps=(3,3))
            y+=length
    objects.append(divider.finish())
    # Rear household rack is perpendicular to the island and hides part of the stock pile.
    rack=Mesh('V5_ReviewedRearRack')
    for z in [.10,.48,.86,1.24,1.62]:
        rack.box((-1.78,10.82,z),(1.53,.36,.028),frame)
    for x in [-2.54,-1.02]:rack.box((x,10.82,.84),(.027,.36,1.68),frame)
    for lo,hi in zip([.10,.48,.86,1.24],[.48,.86,1.24,1.62]):
        rack.box((-1.78,10.83,(lo+hi)/2),(1.46,.25,hi-lo-.04),dark)
        faces.add(rack,[(-2.51,10.63,lo+.02),(-1.05,10.63,lo+.02),
                        (-1.05,10.63,hi-.02),(-2.51,10.63,hi-.02)],steps=(16,3))
    objects.append(rack.finish())
    # Stock is visible from the reverse side too; fixed UVs do not change with the render camera.
    reverse=Mesh('V5_ReviewedRearRackReverse')
    for lo,hi in zip([.10,.48,.86,1.24],[.48,.86,1.24,1.62]):
        faces.add(reverse,[(-2.51,11.015,lo+.02),(-1.05,11.015,lo+.02),
                           (-1.05,11.015,hi-.02),(-2.51,11.015,hi-.02)],'view_08',
                  [[270,310],[475,370],[427,257],[251,225]],steps=(16,3))
    objects.append(reverse.finish(bevel=0))
    # Floor stock remains separated into left and right storage strips.
    floorstock=Mesh('V5_ReviewedFloorStock')
    for side in (-1,1):
        for i in range(9):
            yy=6.18+i*.44;xx=-2.19 if side<0 else front-.25
            height=0
            for layer in range(rng.randint(1,3)):
                ww=rng.uniform(.33,.45);dd=rng.uniform(.31,.41);hh=rng.uniform(.22,.33)
                xx2=xx+rng.uniform(-.08,.08);yy2=yy+rng.uniform(-.05,.05)
                floorstock.box((xx2,yy2,height+hh/2),(ww,dd,hh),kraft if side<0 else rng.choice([green,gold,red]))
                if side<0:
                    faces.add(floorstock,[(xx2+ww/2+.001,yy2-dd/2,height),(xx2+ww/2+.001,yy2+dd/2,height),
                                          (xx2+ww/2+.001,yy2+dd/2,height+hh),(xx2+ww/2+.001,yy2-dd/2,height+hh)],steps=(3,3))
                else:
                    faces.add(floorstock,[(xx2-ww/2-.001,yy2-dd/2,height),(xx2-ww/2-.001,yy2+dd/2,height),
                                          (xx2-ww/2-.001,yy2+dd/2,height+hh),(xx2-ww/2-.001,yy2-dd/2,height+hh)],steps=(3,3))
                    faces.add(floorstock,[(xx2-ww/2,yy2-dd/2,height+hh+.001),(xx2+ww/2,yy2-dd/2,height+hh+.001),
                                          (xx2+ww/2,yy2+dd/2,height+hh+.001),(xx2-ww/2,yy2+dd/2,height+hh+.001)],steps=(3,3))
                faces.add(floorstock,[(xx2-ww/2,yy2-dd/2-.001,height),(xx2+ww/2,yy2-dd/2-.001,height),
                                      (xx2+ww/2,yy2-dd/2-.001,height+hh),(xx2-ww/2,yy2-dd/2-.001,height+hh)],steps=(3,3))
                height+=hh
    objects.append(floorstock.finish())
    # Rear cartons: staggered geometry, printed individual faces, open boxes and blue jugs.
    cartons=Mesh('V5_ReviewedRearCartons')
    carton_quads=[[[578,281],[699,315],[723,190],[591,159]],
                  [[589,410],[677,445],[707,343],[606,309]],
                  [[983,462],[1095,512],[1141,380],[1018,344]]]
    for col,xx in enumerate([-2.72,-2.14,-1.56]):
        for depth_index,yy in enumerate([11.47,12.04]):
            height=0
            for layer in range((4 if col<2 else 2)+depth_index):
                ww=rng.uniform(.46,.55);dd=rng.uniform(.42,.53);hh=rng.uniform(.40,.53)
                xc=xx+rng.uniform(-.04,.04);yc=yy+rng.uniform(-.03,.03)
                cartons.box((xc,yc,height+hh/2),(ww,dd,hh),kraft)
                # Original rear cartons are viewed from camera 08; both visible sides have individual UVs.
                faces.add(cartons,[(xc+ww/2+.001,yc-dd/2,height),(xc+ww/2+.001,yc+dd/2,height),
                                  (xc+ww/2+.001,yc+dd/2,height+hh),(xc+ww/2+.001,yc-dd/2,height+hh)],'view_08',carton_quads[(col+layer)%3],steps=(3,3))
                faces.add(cartons,[(xc-ww/2,yc-dd/2-.001,height),(xc+ww/2,yc-dd/2-.001,height),
                                  (xc+ww/2,yc-dd/2-.001,height+hh),(xc-ww/2,yc-dd/2-.001,height+hh)],'view_08',carton_quads[(col+layer+1)%3],steps=(3,3))
                faces.add(cartons,[(xc-ww/2,yc+dd/2+.001,height),(xc+ww/2,yc+dd/2+.001,height),
                                  (xc+ww/2,yc+dd/2+.001,height+hh),(xc-ww/2,yc+dd/2+.001,height+hh)],'view_08',carton_quads[(col+layer+2)%3],steps=(3,3))
                height+=hh
            if col==1:
                cartons.bottle(xx+.02,yy,height,.14,.49,blue,blue,blue)
    for xx,yy in [(-1.12,11.50),(-.92,11.88),(-1.20,12.25)]:cartons.bottle(xx,yy,0,.13,.46,blue,blue,blue)
    for i in range(6):
        xx=-1.05+rng.uniform(-.10,.12);yy=10.88+i*.22
        cartons.box((xx,yy,.10),(.30,.30,.20),kraft)
    objects.append(cartons.finish())
    # Leaf geometry has oval silhouettes and curved midribs instead of triangular fans.
    tree=Mesh('V5_ReviewedTree');px,py=geo['plant_position']
    tree.box((px,py,.18),(.32,.32,.36),frame)
    for i in range(3):
        tree.bottle(px+(i-1)*.027,py,0.30,.016,1.03,bark,bark,bark,True)
    leafmats=[material('leaf_'+str(i),[.016+i*.007,.078+i*.019,.012+i*.005],.53) for i in range(5)]
    for i in range(360):
        angle=rng.random()*math.tau;rad=rng.uniform(.08,.34);zz=rng.uniform(1.10,1.72)
        center=Vector((px+rad*math.cos(angle),py+rad*math.sin(angle),zz))
        axis=Vector((math.cos(angle),math.sin(angle),rng.uniform(-.7,-.15)))*rng.uniform(.08,.145)
        cross=Vector((-math.sin(angle),math.cos(angle),0))*rng.uniform(.035,.058)
        corners=[]
        for j in range(12):
            a=j*math.tau/12;point=center+axis*math.cos(a)+cross*math.sin(a)
            point.z-=.022*math.sin(a)**2;corners.append(point)
        tree.face(corners,rng.choice(leafmats),[(.5+.5*math.cos(j*math.tau/12),.5+.5*math.sin(j*math.tau/12)) for j in range(12)])
    objects.append(tree.finish(bevel=0))
    # Rear door is behind the plant; small geometry changes improve the visible silhouette.
    door=Mesh('V5_ReviewedDoor')
    door.box((.33,12.54,1.1),(.77,.055,2.2),frame)
    door.box((.33,12.505,1.14),(.66,.015,1.94),dark)
    door.box((.33,12.492,.29),(.66,.012,.42),red)
    objects.append(door.finish())
    # Source camera 01 looks past the chair; preserve it near the visible wall pile.
    chair=bpy.data.objects.get('carved_wooden_chair')
    if chair:chair.location.x=-2.65;chair.location.y=5.55
    table=bpy.data.objects.get('tea_table')
    if table:table.location.y=3.8
    scene.camera=bpy.data.objects['ReferenceCamera_view_01']
    bpy.context.view_layer.update()
    return {'method':'manually reviewed layered object surfaces and estimated stock geometry',
            'fixture_objects':[o.name for o in objects],'object_surface_count':len(faces.faces),
            'sources':{k:str(v.filepath) for k,v in faces.images.items()},'faces':faces.faces,
            'cameras':config['cameras'],'geometry':geo,
            'limitations':['Photo textures include source lighting and perspective-dependent occlusion.',
                           'Stock strips are not individual recovered merchandise.',
                           'Camera fitting and physical dimensions use manual estimates; image similarity is not geometric validation.']}
