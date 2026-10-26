"""Dimension-aware bedroom, living-room and retail furniture, built in metres."""
import math
import generate_scene as g


def build(entry, scene_type='other', seed=23):
    kind = entry.get('kind', entry.get('category'))
    if kind not in {'bed','sofa','cabinet','shelf','chair'} or (kind=='chair' and scene_type in {'office','classroom'}):
        return None
    w,d,h=entry['dimensions']
    root=g.root_object(entry['id'])
    root.location=entry['position']
    root.rotation_euler.z=math.radians(entry.get('rotation_z_deg',0))
    root['geometry_source']='dimension_aware_interior_v2'
    rgb=entry['color']
    main=g.material('InteriorMain',rgb,.7 if kind in {'bed','sofa','chair'} else .45)
    fabric=g.material('InteriorFabric',[min(.95,c*1.12+.05) for c in rgb],.87)
    dark=g.material('InteriorFeet',[c*.25 for c in rgb],.43)
    pale=g.material('InteriorLinen',[.76,.74,.68],.88)
    metal=g.material('InteriorHardware',[.24,.26,.27],.28,.8)
    def box(name,xyz,size,mat=main,r=.012):
        return g.box(root,entry['id']+'_'+name,xyz,size,mat,min(r,min(size)*.2))
    def legs(top,footprint=(.4,.4)):
        for x in [-w*footprint[0],w*footprint[0]]:
            for y in [-d*footprint[1],d*footprint[1]]:
                box('Foot',(x,y,top/2),(min(.055,w*.05),min(.055,d*.05),top),dark)
    if kind=='bed':
        legs(h*.10)
        box('Base',(0,0,h*.20),(w,d,h*.20),main,.025)
        box('Mattress',(0,-d*.025,h*.40),(w*.97,d*.94,h*.20),pale,.045)
        box('Headboard',(0,d*.46,h*.65),(w,d*.08,h*.70),main,.04)
        for x in [-w*.24,w*.24]:
            pillow=box('Pillow',(x,d*.29,h*.56),(w*.42,d*.18,h*.12),pale,.06)
            pillow.rotation_euler.z=math.radians(-3 if x<0 else 3)
        # A low amplitude quilt surface in local metres, draped over the sides.
        vertices=[];faces=[];nx,ny=18,18
        for j in range(ny+1):
            y=-d*.48+j/ny*d*.64
            for i in range(nx+1):
                u=i/nx*2-1;x=u*w*.5
                z=h*.515-min(1,max(0,(abs(u)-.86)/.14))*h*.18
                z+=h*.006*math.sin(i*1.7+j*.8)
                vertices.append((x,y,z))
        for j in range(ny):
            for i in range(nx):
                a=j*(nx+1)+i;faces.append((a,a+1,a+nx+2,a+nx+1))
        mesh=g.bpy.data.meshes.new(entry['id']+'_QuiltMesh');mesh.from_pydata(vertices,[],faces);mesh.update()
        quilt=g.mesh_object(entry['id']+'_Quilt',mesh,root,(0,0,0),fabric)
        solid=quilt.modifiers.new('TextileThickness','SOLIDIFY');solid.thickness=.003
        for polygon in mesh.polygons:polygon.use_smooth=True
    elif kind=='sofa':
        legs(h*.13)
        box('LowerFrame',(0,0,h*.27),(w,d*.97,h*.28),main,.04)
        arm=min(w*.09,.20)
        for x in [-w/2+arm/2,w/2-arm/2]:
            box('Arm',(x,0,h*.51),(arm,d,h*.58),main,.055)
        box('Back',(0,d*.36,h*.68),(w-arm*2,d*.24,h*.64),main,.06)
        count=max(1,min(4,round(w/.85)));cw=(w-2*arm-.04)/count
        for i in range(count):
            x=-(count-1)*cw/2+i*cw
            box('Seat',(x,-d*.105,h*.47),(cw-.012,d*.70,h*.16),fabric,.045)
            back=box('BackCushion',(x,d*.22,h*.74),(cw-.012,d*.19,h*.43),fabric,.055)
            back.rotation_euler.x=math.radians(-5)
    elif kind in {'cabinet','shelf'}:
        thick=min(.035,w*.035,d*.08,h*.035)
        for x in [-w/2+thick/2,w/2-thick/2]:
            box('Side',(x,0,h/2),(thick,d,h),main)
        box('Back',(0,d/2-thick/2,h/2),(w-2*thick,thick,h),main)
        levels=max(2,min(7,round(h/.40)+1))
        for j in range(levels):
            z=thick/2+j/(levels-1)*(h-thick)
            box('Shelf',(0,0,z),(w-2*thick,d,thick),main)
        if kind=='cabinet':
            for side in [-1,1]:
                box('Door',(side*w*.249,-d/2+thick/2,h/2),(w*.494,thick,h*.99),main)
                box('Pull',(side*w*.055,-d/2-.014,h*.53),(.012,.025,min(.18,h*.18)),metal,.004)
    elif kind=='chair':
        # Ordinary dining/accent chair for non-office scenes.
        seat=h*.46;legs(seat,(.38,.35))
        box('Seat',(0,-d*.04,seat),(w,d*.86,h*.10),fabric,.025)
        for x in [-w*.39,w*.39]:
            box('BackPost',(x,d*.36,h*.73),(w*.07,d*.075,h*.54),main)
        box('Back',(0,d*.36,h*.81),(w*.9,d*.11,h*.38),fabric,.025)
    return root
