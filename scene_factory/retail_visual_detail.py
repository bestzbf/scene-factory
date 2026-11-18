"""Optional camera-independent retail surface detail, in physical scene units."""
import math


def validate_visual_settings(settings):
    def number(value,lo,hi,name):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('Invalid '+name)
    def color(value,name):
        if not isinstance(value,list) or len(value)!=3:raise ValueError('Invalid '+name)
        for v in value:number(v,0,1,name)
    floor=settings.get('floor_tiles')
    if floor is not None:
        if not isinstance(floor,dict):raise ValueError('floor_tiles must be an object')
        color(floor.get('color'),'floor color');color(floor.get('grout_color',[.12,.12,.12]),'grout color')
        number(floor.get('size_m',.6),.1,2,'tile size');number(floor.get('grout_m',.003),.0005,.02,'grout width')
        number(floor.get('roughness',.32),.05,1,'floor roughness')
    for opts in settings.get('fixtures',{}).values():
        detail=opts.get('shelf_edges')
        if detail is not None:
            if not isinstance(detail,dict):raise ValueError('shelf_edges must be an object')
            color(detail.get('color'),'shelf rail color');number(detail.get('height_m',.028),.005,.035,'rail height')
    entrance=settings.get('entrance')
    if entrance is not None:
        if not isinstance(entrance,dict):raise ValueError('entrance must be an object')
        number(entrance.get('width_m'),.6,4,'door width');number(entrance.get('height_m',2.3),1.8,3,'door height')
        number(entrance.get('center_x',0),-10,10,'door center')


def apply_visual_detail(bundle, settings):
    import bpy
    from mathutils import Vector
    from retail_stock_geometry import Mesh
    from retail_refinement import stock_material
    from generate_scene import material
    validate_visual_settings(settings)
    report={'floor':None,'shelf_edges':[],'entrance':None}
    floor=settings.get('floor_tiles')
    if floor:
        obj=bpy.data.objects.get('Floor')
        if obj is None:raise ValueError('Configured floor tiles require a Floor mesh')
        mat=material('RetailDetailFloor',tuple(floor['color']),floor.get('roughness',.32),0)
        nodes=mat.node_tree.nodes;links=mat.node_tree.links
        shader=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
        geo=nodes.new('ShaderNodeNewGeometry');sep=nodes.new('ShaderNodeSeparateXYZ');links.new(geo.outputs['Position'],sep.inputs[0])
        def op(name,a,b):
            node=nodes.new('ShaderNodeMath');node.operation=name
            for i,v in enumerate((a,b)):
                if isinstance(v,(float,int)):node.inputs[i].default_value=v
                else:links.new(v,node.inputs[i])
            return node.outputs[0]
        size=floor.get('size_m',.6);width=floor.get('grout_m',.003)
        lines=[]
        for axis in ('X','Y'):
            phase=op('MULTIPLY',sep.outputs[axis],1/size)
            fract=op('FRACT',phase,0)
            lines.append(op('LESS_THAN',fract,width/size))
        joint=op('MAXIMUM',*lines)
        mix=nodes.new('ShaderNodeMixRGB');links.new(joint,mix.inputs[0]);mix.inputs[1].default_value=(*floor['color'],1);mix.inputs[2].default_value=(*floor.get('grout_color',[.12,.12,.12]),1)
        if floor.get('terrazzo',False):
            noise=nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=220;noise.inputs['Detail'].default_value=2
            links.new(geo.outputs['Position'],noise.inputs['Vector'])
            ramp=nodes.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.36;ramp.color_ramp.elements[1].position=.62
            ramp.color_ramp.elements[0].color=(*[c*.45 for c in floor['color']],1)
            ramp.color_ramp.elements[1].color=(*[min(1,c*1.35) for c in floor['color']],1)
            links.new(noise.outputs['Fac'],ramp.inputs[0]);links.new(ramp.outputs[0],mix.inputs[1])
        links.new(mix.outputs[0],shader.inputs['Base Color'])
        bump=nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.25;bump.inputs['Distance'].default_value=.001;bump.invert=True
        links.new(joint,bump.inputs['Height']);links.new(bump.outputs['Normal'],shader.inputs['Normal'])
        obj.data=obj.data.copy();obj.data.materials.clear();obj.data.materials.append(mat)
        for poly in obj.data.polygons:poly.material_index=0
        report['floor']={'object':obj.name,**floor,'mapping':'world-space metric procedural tile joints'}
    entrance=settings.get('entrance')
    if entrance:
        if bundle['architecture'].get('front_wall',True):
            raise ValueError('Entrance detail requires an open front in the source architecture')
        room=bundle['shared_scene']['base_scene']['room'];width=room['width'];height=room['height']
        x=entrance.get('center_x',0);w=entrance['width_m'];h=entrance.get('height_m',2.3)
        if abs(x)+w/2>width/2 or h>=height:raise ValueError('Entrance dimensions exceed room')
        frame=material('RetailEntranceFrame',(.045,.05,.052),.3,.65)
        wall=material('RetailEntranceWall',tuple(room.get('wall_color',[.65,.65,.65])),.8,0)
        glass=material('RetailEntranceGlass',(.105,.15,.18),.16,.1)
        shader=next(n for n in glass.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        shader.inputs['Transmission Weight'].default_value=.28
        shader.inputs['IOR'].default_value=1.45
        metal=material('RetailDoorHandle',(.48,.5,.52),.2,.8)
        mesh=Mesh('RetailPhysicalEntrance')
        left=x-w/2;right=x+w/2
        if left>-width/2:mesh.box(((-width/2+left)/2,-.04,height/2),(left+width/2,.08,height),wall)
        if right<width/2:mesh.box(((width/2+right)/2,-.04,height/2),(width/2-right,.08,height),wall)
        mesh.box((x,-.04,(h+height)/2),(w,.08,height-h),wall)
        for px in (left,x,right):mesh.box((px,.012,h/2),(.045,.07,h),frame)
        for z in (.035,h,h*.42):mesh.box((x,.012,z),(w,.07,.04),frame)
        for side in (-1,1):
            px=x+side*w/4
            mesh.box((px,.006,h/2),(w/2-.05,.012,h-.06),glass)
            mesh.box((x+side*.10,.08,1.05),(.016,.03,.38),metal)
            mesh.box((px,.028,h*.58),(w/2-.09,.018,.045),frame)
        obj=mesh.finish(bevel=.001);obj['representation']='editable_storefront_with_glass_and_handles'
        report['entrance']={'object':obj.name,**entrance,'dimensions_status':'estimated from reference; not surveyed'}
    for oid,opts in settings.get('fixtures',{}).items():
        detail=opts.get('shelf_edges')
        if not detail:continue
        root=bpy.data.objects.get(oid)
        if root is None:raise ValueError('Missing shelf for edge details: '+oid)
        inv=root.matrix_world.inverted_safe();levels={};frame_parts=[];frame_points=[]
        for obj in root.children_recursive:
            if obj.type!='MESH' or obj.name.startswith('RetailRefined_'):continue
            transform=inv @ obj.matrix_world
            removable=[]
            for poly in obj.data.polygons:
                if poly.material_index<len(obj.material_slots) and stock_material(obj.material_slots[poly.material_index].material):continue
                pts=[transform @ obj.data.vertices[i].co for i in poly.vertices]
                removable.append(poly.index);frame_points.extend(pts)
                if len(pts)<3 or max(p.z for p in pts)-min(p.z for p in pts)>.001:continue
                if (pts[1]-pts[0]).cross(pts[2]-pts[0]).z<=0:continue
                lo=[min(p[j] for p in pts) for j in range(3)];hi=[max(p[j] for p in pts) for j in range(3)]
                if hi[0]-lo[0]<.35 or hi[1]-lo[1]<.08:continue
                z=round(hi[2],3)
                levels.setdefault(z,[]).append((lo,hi))
            if removable:frame_parts.append((obj,removable))
        rail=material('RetailRail_'+oid,tuple(detail['color']),.35,0)
        white=material('RetailPricePaper',(.87,.86,.72),.8,0)
        ink=material('RetailPriceInk',(.018,.018,.015),.7,0)
        mesh=Mesh('RetailShelfEdges_'+oid);segments=0
        wire=detail.get('wire_frame',False)
        if wire and frame_points and levels:
            from retail_refinement import _delete_faces
            xmin,xmax=min(p.x for p in frame_points),max(p.x for p in frame_points)
            ymin,ymax=min(p.y for p in frame_points),max(p.y for p in frame_points)
            zmin,zmax=min(p.z for p in frame_points),max(p.z for p in frame_points)
            metal=material('RetailWireWhite',(.72,.75,.74),.3,.45)
            for obj,indices in frame_parts:_delete_faces(obj,indices)
            # Real 3D rod grids and thin trays replace the incorrect solid cabinet sides.
            for x in (xmin+.012,xmax-.012):
                for y in (ymin+.012,ymax-.012):mesh.box((x,y,(zmin+zmax)/2),(.023,.023,zmax-zmin),metal)
                for zi in range(1,max(2,int((zmax-zmin)/.12))):
                    z=zmin+zi*.12
                    mesh.box((x,(ymin+ymax)/2,z),(.006,ymax-ymin,.006),metal)
                for yi in range(1,max(2,int((ymax-ymin)/.10))):
                    mesh.box((x,ymin+yi*.10,(zmin+zmax)/2),(.005,.005,zmax-zmin),metal)
            for z in sorted(levels):
                mesh.box(((xmin+xmax)/2,(ymin+ymax)/2,z-.008),(xmax-xmin,ymax-ymin,.016),metal)
            for y in (ymin,ymax):
                mesh.box(((xmin+xmax)/2,y,zmin+.065),(xmax-xmin,.014,.13),rail)

        for z,rects in levels.items():
            lo=[min(a[j] for a,b in rects) for j in range(2)];hi=[max(b[j] for a,b in rects) for j in range(2)]
            for side in (-1,1):
                y=lo[1]-.001 if side<0 else hi[1]+.001
                length=hi[0]-lo[0];height=min(.035,detail.get('height_m',.028))
                mesh.box(((hi[0]+lo[0])/2,y,z-height/2),(length,.008,height),rail)
                # Small editable price labels with separate ink strokes; no camera-facing planes.
                for i in range(max(1,int(length/.23))):
                    x=lo[0]+.12+i*.23
                    if x+.028>hi[0]:continue
                    mesh.box((x,y+side*.005,z-height*.5),(.052,.0015,height*.72),white)
                    for k in range(4):
                        mesh.box((x-.016+k*.009,y+side*.006,z-height*.45),(.004,.001,.009 if k%2 else .006),ink)
                segments+=1
        if mesh.faces:
            obj=mesh.finish(root,bevel=0);obj['representation']='physical_shelf_rails_and_price_labels'
        report['shelf_edges'].append({'id':oid,'levels':len(levels),'segments':segments,'wire_frame':wire})
    return report
