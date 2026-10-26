"""Reusable office chairs and indoor trees for the photo reconstruction.

Public functions return an Empty at the requested world ground position.  Their
children use local coordinates, so moving / rotating each Empty edits the asset.
Chair faces local -Y (backrest at +Y); dimensions are in metres.
"""
import bpy
import bmesh
import math
import random
from mathutils import Vector

_PREFIX = 'OfficeFurniture_'
_MESH_CACHE = {}


def _material(key, color, roughness=0.5, metallic=0.0):
    name = _PREFIX + key
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bs = next(node for node in mat.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
        bs.inputs['Base Color'].default_value = (*color, 1)
        bs.inputs['Roughness'].default_value = roughness
        bs.inputs['Metallic'].default_value = metallic
    return mat


def _mats():
    mats = {
        'fabric': _material('BlackWovenFabric', (0.022, 0.027, 0.029), .83),
        'plastic': _material('BlackNylon', (.018, .022, .024), .40),
        'chrome': _material('Chrome', (.22, .25, .28), .24, .82),
        'rubber': _material('CasterRubber', (.008, .009, .010), .76),
        'wood': _material('OakPlanter', (.48, .285, .105), .52),
        'soil': _material('DarkPottingSoil', (.032, .022, .015), 1),
        'bark': _material('TreeBark', (.16, .115, .065), .9),
    }
    mat = mats['fabric']
    nodes = mat.node_tree.nodes
    if 'FabricWeaveNoise' not in nodes:
        noise = nodes.new('ShaderNodeTexNoise')
        noise.name = 'FabricWeaveNoise'
        noise.inputs['Scale'].default_value = 220
        noise.inputs['Detail'].default_value = 2
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .13
        bump.inputs['Distance'].default_value = .0012
        mat.node_tree.links.new(noise.outputs['Fac'], bump.inputs['Height'])
        mat.node_tree.links.new(bump.outputs['Normal'], next(node for node in nodes if node.type == 'BSDF_PRINCIPLED').inputs['Normal'])
    for i, col in enumerate(((.045,.115,.023), (.075,.19,.033), (.13,.255,.052), (.10,.155,.031))):
        mats['leaf' + str(i)] = _material('Leaf' + str(i), col, .63)
    return mats


def _root(name, location, yaw=0):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    obj.empty_display_type = 'PLAIN_AXES'
    obj.empty_display_size = .25
    obj.location = location
    obj.rotation_euler.z = yaw
    return obj


def _mesh_object(name, mesh, parent, loc=(0,0,0), material=None):
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    obj.location = loc
    if material is not None:
        # Each reusable geometric primitive also shares its material.
        if len(mesh.materials) == 0:
            mesh.materials.append(material)
        elif mesh.materials[0] != material:
            obj.material_slots[0].link = 'OBJECT'
            obj.material_slots[0].material = material
    return obj


def _rounded_mesh(size, bevel):
    key = ('box', tuple(size), bevel)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1)
    for v in bm.verts:
        v.co.x *= size[0]
        v.co.y *= size[1]
        v.co.z *= size[2]
    if bevel > 0:
        result = bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=3,
                                 affect='EDGES', profile=.5)
        for face in result['faces']:
            face.smooth = True
    bm.normal_update()
    mesh = bpy.data.meshes.new(_PREFIX + 'RoundedBox')
    bm.to_mesh(mesh)
    bm.free()
    _MESH_CACHE[key] = mesh
    return mesh


def _box(name, root, loc, size, mat, bevel=.01):
    return _mesh_object(name, _rounded_mesh(size, bevel), root, loc, mat)


def _cylinder_mesh(vertices=12):
    key = ('cylinder', vertices)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    verts = [(math.cos(i*math.tau/vertices), math.sin(i*math.tau/vertices), z)
             for z in (-.5,.5) for i in range(vertices)]
    faces = [tuple(reversed(range(vertices))), tuple(range(vertices, vertices*2))]
    faces += [(i,(i+1)%vertices,(i+1)%vertices+vertices,i+vertices) for i in range(vertices)]
    mesh = bpy.data.meshes.new(_PREFIX + 'Cylinder')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for poly in list(mesh.polygons)[2:]:
        poly.use_smooth = True
    _MESH_CACHE[key] = mesh
    return mesh


def _rod(name, root, a, b, radius, mat, vertices=12):
    a, b = Vector(a), Vector(b)
    delta = b-a
    obj = _mesh_object(name, _cylinder_mesh(vertices), root, (a+b)/2, mat)
    obj.scale = (radius, radius, delta.length)
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = delta.to_track_quat('Z','Y')
    return obj


def _back_mesh():
    key = ('back',)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    verts, faces = [], []
    nx, nz = 10, 10
    for side in (-1, 1):
        for j in range(nz+1):
            t = j/nz
            z = .61 + .48*t
            width = .231 + .045*math.sin(t*math.pi*.85)
            # Gentle lumbar bulge near the bottom and curved side wings.
            cy = .223 + .075*t - .041*math.sin(t*math.pi)
            for i in range(nx+1):
                u = -1 + 2*i/nx
                x = width*u
                y = cy - .052*u*u + side*.024
                # Rounded upper shoulders.
                zz = z - .022*(abs(u)**4)*(t**6)
                verts.append((x,y,zz))
    n = (nx+1)*(nz+1)
    for side in range(2):
        base = side*n
        for j in range(nz):
            for i in range(nx):
                a = base+j*(nx+1)+i
                face=(a,a+1,a+nx+2,a+nx+1)
                faces.append(tuple(reversed(face)) if side == 0 else face)
    for j in range(nz):
        a=j*(nx+1); b=(j+1)*(nx+1)
        faces.append((a,b,b+n,a+n))
        a+=nx; b+=nx
        faces.append((b,a,a+n,b+n))
    for i in range(nx):
        faces.append((i,i+n,i+1+n,i+1))
        a=n-(nx+1)+i
        faces.append((a+1,a+1+n,a+n,a))
    mesh = bpy.data.meshes.new(_PREFIX+'ErgonomicBack')
    mesh.from_pydata(verts,[],faces)
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth=True
    _MESH_CACHE[key]=mesh
    return mesh


def build_chair(name, location, yaw=0):
    """Build a 1.10 m black task chair. yaw is in radians; front is -Y."""
    m = _mats()
    root = _root(name, location, yaw)
    root['asset_type'] = 'office_task_chair'
    root['front_axis'] = '-Y'
    # Softly rounded seat and sculpted padded back.
    _box(name+'_Seat', root, (0,-.015,.48), (.565,.51,.095), m['fabric'], .043)
    _box(name+'_SeatPan', root, (0,.005,.426), (.46,.40,.035), m['plastic'], .016)
    _mesh_object(name+'_BackUpholstery', _back_mesh(), root, material=m['fabric'])
    _rod(name+'_BackSupport',root,(0,.17,.40),(0,.26,.79),.028,m['plastic'])
    _box(name+'_BackSpine',root,(0,.301,.79),(.075,.038,.33),m['plastic'],.016)
    # Adjustable armrests, a single support on each side.
    for sign, label in ((-1,'L'),(1,'R')):
        _rod(name+'_ArmPost'+label,root,(sign*.235,.10,.43),(sign*.318,.075,.66),.021,m['plastic'])
        _box(name+'_ArmPad'+label,root,(sign*.315,-.005,.685),(.077,.30,.048),m['plastic'],.022)
    _rod(name+'_GasLift',root,(0,0,.145),(0,0,.40),.022,m['chrome'],16)
    _rod(name+'_LiftBoot',root,(0,0,.13),(0,0,.275),.039,m['plastic'],16)
    _box(name+'_Mechanism',root,(0,0,.39),(.18,.18,.068),m['plastic'],.014)
    _rod(name+'_HeightLever',root,(.06,-.04,.39),(.255,-.04,.39),.009,m['chrome'],8)
    _box(name+'_LeverGrip',root,(.257,-.04,.39),(.065,.044,.027),m['plastic'],.012)
    for i in range(5):
        angle=i*math.tau/5 + .12
        d=Vector((math.cos(angle),math.sin(angle),0))
        a=d*.035+Vector((0,0,.16))
        b=d*.30+Vector((0,0,.085))
        _rod(name+'_BaseSpoke'+str(i),root,a,b,.023,m['plastic'],10)
        center=d*.305+Vector((0,0,.053))
        tangent=Vector((-d.y,d.x,0))
        _rod(name+'_CasterAxle'+str(i),root,center-tangent*.038,center+tangent*.038,.016,m['plastic'],8)
        for wheel in (-1,1):
            pos=center+tangent*(wheel*.029)
            _rod(name+'_Wheel'+str(i)+str(wheel),root,pos-tangent*.014,pos+tangent*.014,.043,m['rubber'],12)
    return root


def _append_branch(verts,faces,a,b,r1,r2,sides=7):
    a,b=Vector(a),Vector(b)
    axis=(b-a).normalized()
    u=axis.cross(Vector((1,0,0)))
    if u.length < .05:
        u=axis.cross(Vector((0,1,0)))
    u.normalize(); v=axis.cross(u)
    start=len(verts)
    for c,r in ((a,r1),(b,r2)):
        for i in range(sides):
            theta=math.tau*i/sides
            verts.append(tuple(c+r*(math.cos(theta)*u+math.sin(theta)*v)))
    for i in range(sides):
        j=(i+1)%sides
        faces.append((start+i,start+j,start+sides+j,start+sides+i))
    faces.append(tuple(start+i for i in reversed(range(sides))))
    faces.append(tuple(start+sides+i for i in range(sides)))


def _append_leaf(verts,faces,base,direction,length,width,roll):
    direction=Vector(direction).normalized()
    side=direction.cross(Vector((0,0,1)))
    if side.length < .1:
        side=direction.cross(Vector((0,1,0)))
    side.normalize()
    up=side.cross(direction).normalized()
    lateral=side*math.cos(roll)+up*math.sin(roll)
    normal=direction.cross(lateral).normalized()
    start=len(verts)
    segments=6
    for j in range(segments+1):
        t=j/segments
        shape=math.sin(math.pi*t)**.75
        center=Vector(base)+direction*(length*t)+normal*(.018*math.sin(math.pi*t))
        for cross in (-1,0,1):
            p=center+lateral*(cross*width*.5*shape)-normal*(abs(cross)*.007*shape)
            verts.append(tuple(p))
    for j in range(segments):
        a=start+j*3
        faces.extend(((a,a+3,a+4,a+1),(a+1,a+4,a+5,a+2)))


def build_plant(name, location, height=1.7, seed=0):
    """Build a natural indoor tree with a square oak planter and 63 leaf blades.

    All branches are one mesh and all leaves one mesh. height includes the pot;
    the outermost leaves can extend a few centimetres above that nominal value.
    """
    rng=random.Random(seed)
    m=_mats()
    root=_root(name,location)
    root['asset_type']='potted_indoor_tree'
    root['seed']=str(seed)
    scale=height/1.7
    pot_width=.42*min(1.35,max(.75,scale))
    pot_h=.37*min(1.3,max(.8,scale))
    wall=.022
    # Four individual boards leave a real inset soil surface at the top.
    for side in (-1,1):
        _box(name+'_PotX'+str(side),root,(side*(pot_width-wall)/2,0,pot_h/2),
             (wall,pot_width,pot_h),m['wood'],.006)
        _box(name+'_PotY'+str(side),root,(0,side*(pot_width-wall)/2,pot_h/2),
             (pot_width-2*wall,wall,pot_h),m['wood'],.006)
    _box(name+'_PotBottom',root,(0,0,.025),(pot_width,pot_width,.045),m['wood'],.006)
    _box(name+'_Soil',root,(0,0,pot_h-.045),(pot_width-.041,pot_width-.041,.035),m['soil'],.002)
    # Subtle horizontal timber grain is provided as a material bump.
    mat=m['wood']
    if 'PlanterWoodNoise' not in mat.node_tree.nodes:
        nodes=mat.node_tree.nodes; links=mat.node_tree.links
        tex=nodes.new('ShaderNodeTexNoise'); tex.name='PlanterWoodNoise'
        tex.inputs['Scale'].default_value=5; tex.inputs['Detail'].default_value=2
        mapping=nodes.new('ShaderNodeVectorMath'); mapping.operation='MULTIPLY'
        mapping.inputs[1].default_value=(2,2,38)
        coord=nodes.new('ShaderNodeTexCoord')
        links.new(coord.outputs['Generated'],mapping.inputs[0]); links.new(mapping.outputs['Vector'],tex.inputs['Vector'])
        bump=nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value=.18
        bump.inputs['Distance'].default_value=.0014
        links.new(tex.outputs['Fac'],bump.inputs['Height'])
        links.new(bump.outputs['Normal'],next(node for node in nodes if node.type == 'BSDF_PRINCIPLED').inputs['Normal'])
    branch_verts,branch_faces=[],[]
    trunk_points=[Vector((0,0,pot_h-.06)),Vector((.025*scale,-.015*scale,pot_h+.28*scale)),
                  Vector((-.023*scale,.018*scale,pot_h+.64*scale)),Vector((.02*scale,.01*scale,height-.20*scale))]
    for i in range(3):
        _append_branch(branch_verts,branch_faces,trunk_points[i],trunk_points[i+1],(.022-.005*i)*scale,(.017-.004*i)*scale)
    leaf_verts,leaf_faces,leaf_materials=[],[],[]
    for cluster in range(7):
        angle=cluster*2.399963+rng.uniform(-.25,.25)
        layer=cluster/6
        reach=(.22+.15*math.sin(layer*math.pi))*scale
        tip=Vector((math.cos(angle)*reach,math.sin(angle)*reach,pot_h+(.57+.59*layer)*scale))
        origin=Vector((0,0,tip.z-.25*scale))
        elbow=origin.lerp(tip,.57)+Vector((0,0,-.055*scale))
        _append_branch(branch_verts,branch_faces,origin,elbow,.010*scale,.006*scale)
        _append_branch(branch_verts,branch_faces,elbow,tip,.006*scale,.002*scale)
        for j in range(9):
            leaf_angle=angle+j*2.399963+rng.uniform(-.35,.35)
            spread=rng.uniform(.02,.15)*scale
            base=tip+Vector((math.cos(leaf_angle)*spread,math.sin(leaf_angle)*spread,rng.uniform(-.085,.10)*scale))
            _append_branch(branch_verts,branch_faces,tip,base,.0016*scale,.0007*scale,5)
            direction=Vector((math.cos(leaf_angle),math.sin(leaf_angle),rng.uniform(-.45,.65)))
            nfaces=len(leaf_faces)
            _append_leaf(leaf_verts,leaf_faces,base,direction,rng.uniform(.105,.18)*scale,rng.uniform(.045,.075)*scale,rng.uniform(-.75,.75))
            leaf_materials.extend([rng.randrange(4)]*(len(leaf_faces)-nfaces))
    branch_mesh=bpy.data.meshes.new(name+'_BranchesMesh')
    branch_mesh.from_pydata(branch_verts,[],branch_faces); branch_mesh.update()
    for p in branch_mesh.polygons: p.use_smooth=True
    _mesh_object(name+'_TrunkAndTwigs',branch_mesh,root,material=m['bark'])
    leaf_mesh=bpy.data.meshes.new(name+'_FoliageMesh')
    leaf_mesh.from_pydata(leaf_verts,[],leaf_faces); leaf_mesh.update()
    for i in range(4): leaf_mesh.materials.append(m['leaf'+str(i)])
    for poly,index in zip(leaf_mesh.polygons,leaf_materials):
        poly.material_index=index; poly.use_smooth=True
    _mesh_object(name+'_Leaves',leaf_mesh,root)
    return root
