"""Explicit, deterministic shelf replacements for reviewed source preparation.

Physical decks, small uprights and merchandise on both sides and the endcaps.
All dimensions are local metres; no view-dependent geometry or photo overlays.
"""
import math
import random
import bpy
from retail_stock_geometry import Mesh
from retail_refinement import _materials, _unit, stable_seed
from generate_scene import material


def varied_stock(mesh, lo, hi, kind, rng, mats, facing=-1, top=False):
    """A fixed physical assortment; never conditioned on the active camera."""
    choices = {'snack': ['snack','snack','cup','box'],
               'household': ['box','box','cup'],
               'beverage': ['can','can','beverage']}
    selected = rng.choice(choices.get(kind, [kind]))
    available = hi[2]-lo[2]
    hi = list(hi)
    hi[2] = lo[2]+available*rng.uniform(.65,1)
    if top and selected == 'cup':
        layers = rng.choice((1,2,3))
        h = (hi[2]-lo[2])/layers
        for layer in range(layers):
            _unit(mesh,[lo[0],lo[1],lo[2]+layer*h],
                  [hi[0],hi[1],lo[2]+(layer+1)*h],selected,rng,mats,facing)
    else:
        _unit(mesh,lo,hi,selected,rng,mats,facing,
              tile_indices=[16,17,18,19,20] if selected in ('can','beverage') else None,
              atlas_top_origin=selected in ('can','beverage'))


def replace_shelf(root, spec, atlas, scene_id):
    length, depth, height = spec['dimensions_m']
    if not (.5 <= length <= 12 and .18 <= depth <= 1.4 and .7 <= height <= 2.5):
        raise ValueError('Invalid reviewed shelf dimensions')
    for obj in list(root.children_recursive):
        bpy.data.objects.remove(obj, do_unlink=True)
    # The shelf frame is rebuilt in physical local coordinates.
    root.scale = (1,1,1)
    root.rotation_euler = (0,0,math.radians(spec['rotation_z_deg']))
    root.location = spec['position']
    white = material('Reviewed_ShelfFrame', spec.get('frame_color',(.70,.71,.66)), .6)
    green = material('Reviewed_ShelfRail', spec.get('rail_color',(.035,.33,.095)), .5)
    metal = material('Reviewed_ShelfUpright', (.48,.5,.46), .38, .35)
    mesh = Mesh('ReviewedShelf_'+root.name)
    endcaps = spec.get('endcaps',True)
    margin = .20 if endcaps else .035
    double_sided=spec.get('double_sided',False)
    if spec.get('back_panel'):
        panel_y=0 if double_sided else depth/2-.014
        panel_length=length-2*margin if double_sided else length
        mesh.box((0,panel_y,height/2),(panel_length,.025,height),white)
    if spec.get('header_color'):
        header=material('Reviewed_Header_'+root.name,spec['header_color'],.45)
        mesh.box((0,0,height-.08),(length,depth,.16),header)
    mats = _materials(atlas)
    rng = random.Random(stable_seed(scene_id, root.name, 41))
    kind = spec.get('kind','snack')
    levels = spec.get('levels',5)
    if not isinstance(levels,int) or levels < 2 or levels > 12:
        raise ValueError('Shelf levels must be an integer from 2 to 12')
    varied = spec.get('stock_style') == 'reference_mixed'
    if varied:
        mats=(mats[0],mats[1],material('Reviewed_SilverCanTop',(.62,.66,.67),.24,.65))
    bays=max(1,math.ceil(length/1.4))
    for bay in range(bays+1):
        x=-length/2+.018+bay*(length-.036)/bays
        for y in (-depth/2+.018,depth/2-.018):
            mesh.box((x,y,height/2),(.025,.025,height),metal)
    for tier in range(levels):
        z = .07 + (height-.07)*tier/(levels-1)
        mesh.box((0,0,z),(length,depth,.025),white)
        for y in (-depth/2,depth/2):
            mesh.box((0,y,z+.005),(length,.014,.035),green)
        if endcaps:
            for x in (-length/2,length/2):mesh.box((x,0,z+.005),(.014,depth,.035),green)
        cell = spec.get('stock_cell_m', .075 if kind=='beverage' else .10)
        if tier==levels-1:cell=spec.get('top_cell_m',cell)
        h = min(.20, (height-.07)/(levels-1)-.04)
        if tier==levels-1:h=spec.get('top_height',.17)
        if tier==levels-1 and spec.get('top_groups'):
            for group in spec['top_groups']:
                start,end=group['interval_m'];group_kind=group['kind']
                n=max(1,int((end-start)/group.get('cell_m',.15)))
                rows=group.get('rows',3);layers=group.get('layers',1)
                unit_h=group.get('height_m',.22)
                for i in range(n):
                    for j in range(rows):
                        x=-length/2+start+(i+.5)*(end-start)/n
                        y=-depth/2+(j+.5)*depth/rows
                        w=(end-start)/n*.89;d=depth/rows*.80
                        for layer in range(layers):
                            bottom=z+.014+layer*unit_h
                            if group_kind=='blue_box':
                                blue=material('Reviewed_LunchboxBlue',(.025,.40,.62),.32)
                                mesh.box((x,y,bottom+unit_h/2),(w,d,unit_h*.92),blue)
                                mesh.box((x,y,bottom+unit_h*.94),(w*1.015,d*1.015,.009),blue)
                            else:
                                first=len(mesh.vertices)
                                package_depth=min(d,.055) if group_kind=='snack' else d
                                _unit(mesh,[x-w/2,y-package_depth/2,bottom],[x+w/2,y+package_depth/2,bottom+unit_h],group_kind,rng,mats)
                                if group_kind=='snack':
                                    theta=math.pi/2+rng.uniform(-.28,.28)
                                    for k in range(first,len(mesh.vertices)):
                                        vx,vy,vz=mesh.vertices[k];dx,dy=vx-x,vy-y
                                        mesh.vertices[k]=(x+dx*math.cos(theta)-dy*math.sin(theta),y+dx*math.sin(theta)+dy*math.cos(theta),vz)
            continue
        # Multiple rows across the depth, with printed products on both sides.
        nx=max(1,int((length-2*margin)/cell)); ny=max(2,int(depth/.11))
        if double_sided:ny=2*max(1,int((depth/2-.025)/.11))
        for i in range(nx):
            for j in range(ny):
                if tier==levels-1 and kind=='beverage' and spec.get('top_gaps') and (i//4+j//2)%6==0:
                    continue
                x=-length/2+margin+(i+.5)*(length-2*margin)/nx
                y=-depth/2+(j+.5)*depth/ny
                w=(length-2*margin)/nx*.95; d=depth/ny*.90
                if double_sided:
                    rows=ny//2;zone=depth/2-.025
                    y=(-1 if j<rows else 1)*(.025+(j%rows+.5)*zone/rows)
                    d=zone/rows*.90
                if varied:
                    # Adjacent facings form product groups instead of a random
                    # colour chessboard. Stable groups repeat across all views.
                    group_rng=random.Random(stable_seed(scene_id,root.name,1000+tier*100+i//3))
                    stock_kind=kind if tier==levels-1 else spec.get('lower_stock_kind',kind)
                    varied_stock(mesh,[x-w/2,y-d/2,z+.014],[x+w/2,y+d/2,z+.014+h],stock_kind,group_rng,mats,1 if y>0 else -1,tier==levels-1)
                else:
                    _unit(mesh,[x-w/2,y-d/2,z+.014],[x+w/2,y+d/2,z+.014+h],kind,rng,mats,1 if y>0 else -1)
        if endcaps:
            for sign in (-1,1):
                for j in range(max(2,int(depth/.095))):
                    n=max(2,int(depth/.095)); y=-depth/2+(j+.5)*depth/n
                    # Rotate each endcap package around local Z so labels face aisle.
                    start=len(mesh.vertices)
                    fn=varied_stock if varied else _unit
                    end_kind=kind if tier==levels-1 else spec.get('lower_stock_kind',kind)
                    fn(mesh,[-depth/n*.44,-.08,0],[depth/n*.44,.08,h],end_kind,rng,mats)
                    for k in range(start,len(mesh.vertices)):
                        vx,vy,vz=mesh.vertices[k]
                        mesh.vertices[k]=(sign*(length/2-.10)+sign*vy,y-sign*vx,z+.014+vz)
    obj=mesh.finish(root,bevel=0)
    obj['representation']='reviewed_physical_stock_shelf'
    return {'id':root.name,'stock_units':mesh.products,'dimensions_m':spec['dimensions_m']}


def replace_fridge(root, spec, atlas, scene_id):
    width,depth,height=spec['dimensions_m']
    for obj in list(root.children_recursive):bpy.data.objects.remove(obj,do_unlink=True)
    root.scale=(1,1,1);root.location=spec['position'];root.rotation_euler=(0,0,math.radians(spec['rotation_z_deg']))
    mesh=Mesh('ReviewedFridge_'+root.name)
    frame=material('ReviewedFridgeFrame_'+root.name,spec.get('frame_color',(.42,.02,.025)),.4)
    inner=material('ReviewedFridgeInside',(.64,.67,.64),.55)
    mats=_materials(atlas);rng=random.Random(stable_seed(scene_id,root.name,41))
    mesh.box((0,depth/2-.025,height/2),(width,.05,height),inner)
    for x in (-width/2+.025,width/2-.025):mesh.box((x,0,height/2),(.05,depth,height),frame)
    for z in (.04,height-.07):mesh.box((0,0,z),(width,depth,.08 if z<.1 else .14),frame)
    if spec.get('roof_color'):
        roof=material('ReviewedFridgeRoof_'+root.name,spec['roof_color'],.55)
        mesh.box((0,0,height-.006),(width-.02,depth-.02,.012),roof)
    for tier in range(5):
        z=.15+tier*(height-.45)/5
        mesh.box((0,0,z),(width-.08,depth-.05,.025),inner)
        n=max(2,int((width-.1)/.075))
        for j in range(2):
            for i in range(n):
                x=-width/2+.05+(i+.5)*(width-.1)/n;y=-depth/2+.08+j*.13
                _unit(mesh,[x-.031,y-.045,z+.014],[x+.031,y+.045,z+.24],'beverage',rng,mats)
    # Thin real glass front; products remain explicit meshes behind it.
    glass=material('ReviewedFridgeGlass',(.86,.93,.92),.08)
    shader=next(n for n in glass.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    shader.inputs['Transmission Weight'].default_value=1;shader.inputs['IOR'].default_value=1.45
    mesh.box((0,-depth/2-.006,height/2),(width-.09,.006,height-.23),glass)
    mesh.box((width/2-.09,-depth/2-.025,height*.5),(.025,.035,.4),frame)
    obj=mesh.finish(root,bevel=0);obj['representation']='reviewed_physical_glass_fridge'
    return {'id':root.name,'stock_units':mesh.products,'dimensions_m':spec['dimensions_m']}
