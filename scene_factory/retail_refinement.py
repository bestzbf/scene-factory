"""Deterministic retail refinement shared by production and saved-scene batches.

Only recognized stock materials are replaced. Fixture frames, camera poses and
unrecognized imported assets are preserved unless a reviewed profile says otherwise.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random
from pathlib import Path

VERSION = 1
KINDS = {'mixed', 'snack', 'beverage', 'cup', 'household', 'gift', 'tobacco', 'preserve'}
STOCK_FIXTURE_CATEGORIES = frozenset({
    'shelf', 'display_rack', 'wire_display_rack', 'bakery_display', 'open_chiller',
    'glass_display_cabinet', 'glass_door_refrigerator', 'open_display_refrigerator',
    'refrigerator', 'tobacco_display', 'countertop_display_rack', 'cardboard_retail_display',
})


def is_stock_fixture(object_id, category=''):
    # A semantic category is authoritative. A column or box may mention its
    # neighbouring refrigerator in its ID; that must never add stock on top.
    if category:
        return category in STOCK_FIXTURE_CATEGORIES
    return any(k in object_id.lower() for k in
               ('shelf','gondola','island','rack','cabinet','chiller','refrigerator','fridge','bakery'))


def stable_seed(scene_id, object_id, seed=17):
    return int.from_bytes(hashlib.sha256(f'{seed}:{scene_id}:{object_id}'.encode()).digest()[:8], 'big')


def classify(object_id, category=''):
    text = (object_id + ' ' + category).lower()
    if any(k in text for k in ('cup_noodle', 'instant_noodle', 'ramen', 'cupnoodle')): return 'cup'
    if any(k in text for k in ('tobacco', 'cigarette')): return 'tobacco'
    if any(k in text for k in ('gift', 'wine_box')): return 'gift'
    if any(k in text for k in ('drink', 'beverage', 'sauce', 'condiment', 'liquor', 'oil')): return 'beverage'
    if any(k in text for k in ('household', 'cleaning', 'tissue', 'shampoo')): return 'household'
    if any(k in text for k in ('snack', 'packet', 'bakery', 'biscuit', 'grocery')): return 'snack'
    return 'mixed'


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    validate_config(config)
    for key in ('atlas',):
        if config.get(key):
            value = Path(config[key])
            config[key] = str((path.parent / value).resolve() if not value.is_absolute() else value)
    if config.get('atlas') and not Path(config['atlas']).is_file():
        raise ValueError('Packaging atlas is missing: '+config['atlas'])
    config['_config_path'] = str(path)
    return config


def validate_config(config):
    if config.get('schema_version') != VERSION: raise ValueError('Unsupported retail optimization schema')
    if not isinstance(config.get('seed',17),int): raise ValueError('seed must be an integer')
    for key in ('defaults', 'scenes'):
        if not isinstance(config.get(key, {}), dict): raise ValueError(key + ' must be an object')
    for settings in [config.get('defaults', {}), *config.get('scenes', {}).values()]:
        if not isinstance(settings, dict): raise ValueError('Scene settings must be an object')
        if not isinstance(settings.get('fixtures', {}), dict): raise ValueError('fixtures must be an object')
        if not isinstance(settings.get('fixture_defaults', {}), dict): raise ValueError('fixture_defaults must be an object')
        for fixture in [settings.get('fixture_defaults', {}), *settings.get('fixtures', {}).values()]:
            if not isinstance(fixture, dict): raise ValueError('fixture settings must be an object')
            if fixture.get('kind', 'mixed') not in KINDS: raise ValueError('Unknown stock kind')
            for flag in ('enabled','top_stock'):
                if flag in fixture and not isinstance(fixture[flag],bool): raise ValueError(flag+' must be boolean')
        from retail_visual_detail import validate_visual_settings
        validate_visual_settings(settings)
        for item in settings.get('fixture_dimensions', []):
            if not isinstance(item,dict) or not isinstance(item.get('id'),str): raise ValueError('fixture_dimensions requires id')
            dims=item.get('dimensions_m')
            if not isinstance(dims,list) or len(dims)!=3 or not all(isinstance(v,(int,float)) and math.isfinite(v) and .05<=v<=10 for v in dims): raise ValueError('Invalid fixture dimensions')
        for move in settings.get('moves', []):
            if not isinstance(move.get('id'), str): raise ValueError('Move requires an object id')
            vec = move.get('translation_m')
            if not isinstance(vec, list) or len(vec) != 3 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in vec):
                raise ValueError('Move translation must be a finite 3-vector')
            if vec[2]!=0: raise ValueError('Semantic XY layouts support horizontal fixture moves only')
            if max(abs(v) for v in vec) > 2: raise ValueError('Reviewed translations are limited to 2 metres per axis')
    return config


def settings_for(config, scene_id):
    defaults = copy.deepcopy(config.get('defaults', {}))
    scene = config.get('scenes', {}).get(str(scene_id), {})
    defaults.update({k: v for k, v in scene.items() if k != 'fixtures'})
    defaults['fixtures'] = {**config.get('defaults', {}).get('fixtures', {}), **scene.get('fixtures', {})}
    return defaults


def support_groups(objects):
    """Include transitive supported objects; reject ambiguous/cyclic graphs."""
    items = {o['id']: o for o in objects}
    if len(items) != len(objects): raise ValueError('Duplicate object IDs')
    groups = {}
    for oid in items:
        seen = {oid}; parent = items[oid].get('support_id', 'floor')
        while parent in items:
            if parent in seen: raise ValueError('Cyclic support graph: ' + oid)
            seen.add(parent); parent = items[parent].get('support_id', 'floor')
        groups[oid] = [child for child in items if child == oid or _supported_by(child, oid, items)]
    return groups


def _supported_by(child, ancestor, items):
    seen = set()
    while child in items and child not in seen:
        seen.add(child); child = items[child].get('support_id', 'floor')
        if child == ancestor: return True
    return False


def stock_material(mat):
    if not mat or not mat.use_nodes: return False
    return any(n.type == 'TEX_IMAGE' and n.image and n.image.name.lower().startswith(('product_atlas', 'demorefinedpackagingatlas'))
               for n in mat.node_tree.nodes)


def stock_components(obj, root):
    """Group imported stock by welded coordinate connectivity despite UV seams.

    Return local-root bounds and the exact original face IDs. Frames and other
    materials are untouched; unsupported stock materials simply produce no edits.
    """
    from mathutils import Vector
    mesh = obj.data
    slots = {i for i, slot in enumerate(obj.material_slots) if stock_material(slot.material)}
    polys = [p for p in mesh.polygons if p.material_index in slots]
    if not polys: return [], []
    parents = {}; points = {}; faces = []
    transform = root.matrix_world.inverted_safe() @ obj.matrix_world
    def find(k):
        parents.setdefault(k, k)
        while parents[k] != k:
            parents[k] = parents[parents[k]]; k = parents[k]
        return k
    for poly in polys:
        keys = []
        for vi in poly.vertices:
            p = transform @ mesh.vertices[vi].co
            k = tuple(round(v, 6) for v in p); points[k] = p.copy(); keys.append(k); find(k)
        for k in keys[1:]: parents[find(k)] = find(keys[0])
        faces.append((poly.index, keys[0]))
    groups = {}
    for k, point in points.items(): groups.setdefault(find(k), []).append(point)
    result = []
    for pts in groups.values():
        lo = [min(p[i] for p in pts) for i in range(3)]; hi = [max(p[i] for p in pts) for i in range(3)]
        result.append((lo, hi))
    return result, [p.index for p in polys]


def _delete_faces(obj, indices):
    import bmesh
    # A catalog can share a mesh datablock between instances: isolate before editing.
    obj.data = obj.data.copy()
    bm = bmesh.new(); bm.from_mesh(obj.data); bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.faces[i] for i in indices], context='FACES')
    bm.to_mesh(obj.data); bm.free(); obj.data.update()


def _materials(atlas_path):
    import bpy
    from generate_scene import material
    atlas = bpy.data.images.load(str(Path(atlas_path).resolve()), check_existing=True) if atlas_path else None
    if atlas: atlas.pack()
    print_mat = material('RetailRefine_Print', (.65,.3,.08), .47, 0)
    if atlas:
        tex = print_mat.node_tree.nodes.new('ShaderNodeTexImage'); tex.image = atlas
        tex.interpolation = 'Linear'; tex.extension = 'EXTEND'
        shader = next(n for n in print_mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        print_mat.node_tree.links.new(tex.outputs['Color'], shader.inputs['Base Color'])
    colors = [(.58,.025,.02),(.035,.22,.075),(.8,.52,.035),(.72,.67,.48),(.025,.095,.22),(.23,.065,.02)]
    return print_mat, [material('RetailRefine_Body_'+str(i), c, .4, 0) for i,c in enumerate(colors)], material('RetailRefine_Cap',(.85,.81,.69),.3,.25)


def _unit(builder, lo, hi, kind, rng, mats, facing=-1, tile_indices=None, atlas_top_origin=False):
    """Every replacement stays in the original unit's bounding box."""
    printed, bodies, cap = mats
    w,d,h = [hi[i]-lo[i] for i in range(3)]
    if min(w,d,h) < .012 or h > 1.2 or w > 1.5: return False
    x,y,z = (lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]
    # Narrow facings remain separate products; wider catalog blocks become
    # multiple facings, with bounded, deterministic variation in height.
    count = max(1, min(4, round(w / (.075 if kind == 'beverage' else .11))))
    for i in range(count):
        a = lo[0]+(i+.5)*w/count; uw=w/count*.90; uh=h*rng.uniform(.90,1)
        body = rng.choice(bodies); face_start=len(builder.faces); vertex_start=len(builder.vertices)
        if kind == 'cup':
            builder.cup(a,y,z,min(uw,d*.85)/2,uh,body,printed)
        elif kind in ('beverage','can') or (kind in ('mixed','household') and rng.random()<.35):
            r=min(uw,d*.85)/2
            builder.bottle(a,y,z,r,uh,body,printed,cap,jar=kind=='can' or uh<r*3.5)
        elif kind == 'snack' or (kind=='mixed' and rng.random()<.55):
            builder.pouch(a,y-d*.0798,z,uw,d*.84,uh,body,printed,'-Y')
            # Both package faces carry print, including reverse aisle views.
            builder.indices[face_start+1]=builder.materials.index(printed)
            builder.uvs[face_start+1]=[(0,.55),(.06,1),(.94,1),(1,.55),(.94,0),(.06,0)]
        else:
            builder.box((a,y,z+uh/2),(uw,d*.88,uh),body,printed,('-Y','+Y','-X','+X'))
            builder.products+=1
        if facing>0:
            for j in range(vertex_start,len(builder.vertices)):
                vx,vy,vz=builder.vertices[j];builder.vertices[j]=(2*a-vx,2*y-vy,vz)
        tile = rng.choice(tile_indices) if tile_indices else rng.randrange(64)
        tx,ty=tile%8,tile//8
        if atlas_top_origin:ty=7-ty
        for j in range(face_start,len(builder.faces)):
            if builder.materials[builder.indices[j]] == printed:
                builder.uvs[j]=[((tx+.025+u*.95)/8,(ty+.025+v*.95)/8) for u,v in builder.uvs[j]]
    return True


def _top_stock(root, builder, kind, rng, mats, occupied):
    """Fill reviewed exposed top boards, rejecting existing merchandise bounds."""
    from mathutils import Vector
    surfaces=[]
    inv=root.matrix_world.inverted_safe()
    for obj in root.children_recursive:
        if obj.type!='MESH' or obj.name.startswith('RetailRefined_'):continue
        transform=inv @ obj.matrix_world
        for poly in obj.data.polygons:
            if poly.material_index<len(obj.material_slots) and stock_material(obj.material_slots[poly.material_index].material):continue
            pts=[transform @ obj.data.vertices[i].co for i in poly.vertices]
            if len(pts)<3 or max(p.z for p in pts)-min(p.z for p in pts)>.002:continue
            normal=(pts[1]-pts[0]).cross(pts[2]-pts[0])
            if normal.z<=0:continue
            lo=[min(p[i] for p in pts) for i in range(3)];hi=[max(p[i] for p in pts) for i in range(3)]
            if (hi[0]-lo[0])*(hi[1]-lo[1])>.035:surfaces.append((lo,hi))
    if not surfaces:return 0
    z=max(hi[2] for lo,hi in surfaces)
    top=[(lo,hi) for lo,hi in surfaces if abs(hi[2]-z)<.005]
    xmin=min(lo[0] for lo,hi in top);xmax=max(hi[0] for lo,hi in top)
    ymin=min(lo[1] for lo,hi in top);ymax=max(hi[1] for lo,hi in top)
    nx=max(1,int((xmax-xmin)/.13));ny=max(1,min(4,int((ymax-ymin)/.16)))
    before=builder.products
    for ix in range(nx):
        for iy in range(ny):
            x=xmin+(ix+.5)*(xmax-xmin)/nx;y=ymin+(iy+.5)*(ymax-ymin)/ny
            w=min(.12,(xmax-xmin)/nx*.9);d=min(.14,(ymax-ymin)/ny*.85)
            h=rng.uniform(.18,.29)
            lo=[x-w/2,y-d/2,z+.003];hi=[x+w/2,y+d/2,z+.003+h]
            # Include support-graph children and the imported stock, not frames.
            if any(all(min(hi[j],b[j])-max(lo[j],a[j])>.001 for j in range(3)) for a,b in occupied):continue
            if not any(a[0]-.001<=x<=b[0]+.001 and a[1]-.001<=y<=b[1]+.001 for a,b in top):continue
            _unit(builder,lo,hi,kind,rng,mats,1 if y>0 else -1)
            occupied.append((lo,hi))
    return builder.products-before


def _fixture_bounds(semantic):
    import bpy
    from mathutils import Vector
    bounds={}
    for oid,item in semantic.items():
        if item.get('support_id','floor')!='floor':continue
        if item.get('category') not in ('shelf','cabinet','checkout_counter','glass_display_counter','chest_freezer','glass_door_refrigerator','open_display_refrigerator','bakery_display'):continue
        root=bpy.data.objects.get(oid)
        if root is None:continue
        pts=[o.matrix_world @ Vector(c) for o in [root,*root.children_recursive] if o.type=='MESH' for c in o.bound_box]
        if pts:bounds[oid]=([min(p[j] for p in pts) for j in range(3)],[max(p[j] for p in pts) for j in range(3)])
    return bounds


def _move_audit(before,after,moved,room):
    """Conservative AABB guard: reject newly introduced fixture overlaps/boundary excess."""
    problems=[];tested=[]
    for oid in sorted(set(moved)&set(after)):
        a,b=after[oid];old_a,old_b=before[oid]
        low=[-room['width']/2,0,0];high=[room['width']/2,room['depth'],room['height']]
        for j in range(3):
            old=max(0,low[j]-old_a[j],old_b[j]-high[j]);new=max(0,low[j]-a[j],b[j]-high[j])
            if new>old+.03:problems.append({'id':oid,'reason':'new room boundary excess','axis':j,'metres':new})
        for other,(c,d) in after.items():
            if other==oid:continue
            def volume(a,b,c,d):return math.prod(max(0,min(b[j],d[j])-max(a[j],c[j])) for j in range(3))
            old=volume(old_a,old_b,*before[other]);new=volume(a,b,c,d)
            if new>old+.01:problems.append({'id':oid,'other':other,'reason':'increased fixture AABB overlap','old_m3':old,'new_m3':new})
        tested.append(oid)
    return {'status':'rejected' if problems else 'passed','tested_fixtures':tested,'issues':problems,
            'scope':'New AABB fixture overlaps and increased room-boundary violations only. Existing overlaps, support contacts and full mesh collisions are not certified.'}


def apply_refinement(bundle, config, layout=None):
    """Apply once before all views. Also callable by generate_multiview.py."""
    import bpy
    from mathutils import Vector
    from retail_stock_geometry import Mesh
    validate_config(config)
    if not config.get('atlas') or not Path(config['atlas']).is_file():
        raise ValueError('Retail refinement requires a packaging atlas; the batch runner can create it automatically')
    scene=bpy.context.scene; sid=str(bundle['scene_id']); settings=settings_for(config,sid)
    from retail_photo_details import inferred_settings, build_fixtures
    settings=inferred_settings(layout,settings)
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if scene.get('retail_refinement_applied'):
        raise ValueError('Scene already refined; use the unmodified source to prevent duplicate geometry')
    objects=(layout or {}).get('objects',[]); semantic={o['id']:o for o in objects}
    entries={o['id']:o for o in bundle['shared_scene']['base_scene']['objects']}
    unknown=set(settings['fixtures'])-set(entries)
    if unknown: raise ValueError('Unknown fixture IDs: '+', '.join(sorted(unknown)))
    groups=support_groups(objects) if objects else {}
    bpy.context.view_layer.update()
    before_bounds=_fixture_bounds(semantic)
    resized=[]; resized_ids=set()
    for item in settings.get('fixture_dimensions',[]):
        ids=groups.get(item['id'],[item['id']])
        if resized_ids.intersection(ids):raise ValueError('Overlapping dimension support groups')
        resized_ids.update(ids)
        root=bpy.data.objects.get(item['id'])
        if root is None: raise ValueError('Fixture dimension root absent: '+item['id'])
        inv=root.matrix_world.inverted_safe()
        points=[inv @ ob.matrix_world @ Vector(corner)
                for ob in [root,*root.children_recursive] if ob.type=='MESH'
                for corner in ob.bound_box]
        if not points: raise ValueError('Fixture has no geometry: '+item['id'])
        local_dims=[max(q[i] for q in points)-min(q[i] for q in points) for i in range(3)]
        dims=[local_dims[i]*abs(root.scale[i]) for i in range(3)]
        target=item['dimensions_m']
        if min(dims)<1e-5: raise ValueError('Fixture has zero evaluated dimension: '+item['id'])
        old_matrix=root.matrix_world.copy()
        old_support={cid:bpy.data.objects[cid].matrix_world.copy() for cid in ids if cid in entries and cid!=item['id'] and cid in bpy.data.objects}
        root.scale=Vector(root.scale) * Vector([target[i]/dims[i] for i in range(3)])
        bpy.context.view_layer.update()
        delta=root.matrix_world @ old_matrix.inverted_safe()
        for child_id in ids:
            if child_id==item['id'] or child_id not in entries:continue
            child=bpy.data.objects.get(child_id)
            if child is None:raise ValueError('Missing supported object: '+child_id)
            parent=child.parent;follows=False
            while parent:
                if parent==root:follows=True;break
                parent=parent.parent
            old_child=old_support[child_id]
            if not follows:child.matrix_world=delta @ old_child
            bpy.context.view_layer.update()
            ratios=[child.matrix_world.col[i].to_3d().length/max(old_child.col[i].to_3d().length,1e-8) for i in range(3)]
            entries[child_id]['position']=list(child.matrix_world.translation)
            entries[child_id]['dimensions']=[entries[child_id]['dimensions'][i]*ratios[i] for i in range(3)]
            if child_id in semantic:
                semantic[child_id]['xy']=list(child.matrix_world.translation)[:2]
                semantic[child_id]['target_dimensions_m']=entries[child_id]['dimensions'][:]
        entries[item['id']]['dimensions']=list(target)
        if item['id'] in semantic:semantic[item['id']]['target_dimensions_m']=list(target)
        resized.append({'id':item['id'],'dimensions_m':target,'evidence':item.get('evidence','reviewed dimension override')})
    bpy.context.view_layer.update()
    moved=[]; moved_ids=set()
    for move in settings.get('moves',[]):
        oid=move['id']
        if oid not in groups: raise ValueError('Layout support graph required for move: '+oid)
        ids=groups[oid]
        if moved_ids.intersection(ids): raise ValueError('Overlapping move groups would move objects twice')
        omitted=[i for i in ids if i not in entries]
        roots=[bpy.data.objects.get(i) for i in ids if i in entries]
        if any(r is None for r in roots): raise ValueError('Supported object absent from Blender scene: '+oid)
        delta=Vector(move['translation_m']); idset=set(ids)
        # Model support relations need not be Blender parent relations. Move
        # only topmost Blender roots; children follow their existing parents.
        for root in roots:
            parent=root.parent; follows=False
            while parent:
                if parent.name in idset:follows=True;break
                parent=parent.parent
            if not follows: root.matrix_world.translation+=delta
        for i in ids:
            if i in entries:entries[i]['position']=[entries[i]['position'][j]+delta[j] for j in range(3)]
            if i in semantic:
                semantic[i]['xy']=[semantic[i]['xy'][j]+delta[j] for j in range(2)]
        moved.append({'id':oid,'support_group':ids,'source_omitted_ids':omitted,'translation_m':list(delta),'evidence':move.get('evidence','reviewed override')});moved_ids.update(ids)
    bpy.context.view_layer.update()
    move_audit=_move_audit(before_bounds,_fixture_bounds(semantic),moved_ids|resized_ids,bundle['shared_scene']['base_scene']['room'])
    if move_audit['status']!='passed':raise ValueError('Reviewed moves failed clearance guard: '+json.dumps(move_audit['issues']))
    rebuilt, skipped = build_fixtures(bundle,layout,config['atlas'],groups)
    rebuilt_ids={r['id'] for r in rebuilt}
    mats=_materials(config.get('atlas')); records=list(rebuilt)
    for oid,entry in entries.items():
        if oid in rebuilt_ids:continue
        root=bpy.data.objects.get(oid)
        if root is None:
            skipped.append({'id':oid,'reason':'No matching Blender root; expanded instances require explicit IDs'})
            continue
        opts={**settings.get('fixture_defaults',{}), **settings['fixtures'].get(oid,{})}
        if opts.get('enabled', True) is False: continue
        category=semantic.get(oid,{}).get('category','')
        if not is_stock_fixture(oid,category): continue
        kind=opts.get('kind',classify(oid,category))
        if kind=='preserve':continue
        rng=random.Random(stable_seed(sid,oid,config.get('seed',17)))
        builder=Mesh('RetailRefined_'+oid); replaced=0; removed=0; unsupported=0; occupied=[]
        for obj in list(root.children_recursive):
            if obj.type!='MESH':continue
            boxes,indices=stock_components(obj,root)
            if not boxes:continue
            occupied.extend(boxes)
            # Avoid destructive edits when the unknown catalog topology is
            # connected across units or too large to represent merchandise.
            valid=[(lo,hi) for lo,hi in boxes if min(hi[i]-lo[i] for i in range(3))>=.012 and hi[2]-lo[2]<=1.2 and hi[0]-lo[0]<=1.5]
            if len(valid)!=len(boxes):
                unsupported+=len(boxes);continue
            for lo,hi in valid:
                _unit(builder,lo,hi,kind,rng,mats,facing=1 if (lo[1]+hi[1])>0 else -1)
            _delete_faces(obj,indices);replaced+=len(boxes);removed+=len(indices)
        added=0
        if opts.get('top_stock',False):
            for child_id in groups.get(oid,[]):
                if child_id==oid:continue
                child=bpy.data.objects.get(child_id)
                if child:
                    pts=[root.matrix_world.inverted_safe() @ ob.matrix_world @ Vector(c) for ob in [child,*child.children_recursive] if ob.type=='MESH' for c in ob.bound_box]
                    if pts:occupied.append(([min(p[j] for p in pts) for j in range(3)],[max(p[j] for p in pts) for j in range(3)]))
            added=_top_stock(root,builder,kind,rng,mats,occupied)
        if replaced or added:
            mesh=builder.finish(root,bevel=0)
            mesh['representation']='physical_stock_with_fixed_uv';mesh['fixture_id']=oid
            records.append({'id':oid,'kind':kind,'source_stock_components':replaced,'added_top_stock':added,'removed_faces':removed,'stock_units':builder.products,'mesh':mesh.name})
        if unsupported:skipped.append({'id':oid,'reason':'unrecognized connected stock geometry','components':unsupported})
    bpy.context.view_layer.update()
    from retail_visual_detail import apply_visual_detail
    visual_detail=apply_visual_detail(bundle,settings)
    bpy.context.view_layer.update()
    scene['retail_refinement_applied']=signature
    return {'schema_version':VERSION,'scene_id':sid,'config_sha256':signature,'geometry_build_count':1,
            'representation':'physical_meshes_with_fixed_object_uv','camera_dependent_geometry':False,
            'visual_detail':visual_detail,'fixtures':records,'skipped':skipped,'fixture_dimensions':resized,'moves':moved,'layout_clearance':move_audit,
            'photo_inferred_detail':bool((layout or {}).get('retail_details')),
            'stock_units':sum(r['stock_units'] for r in records),
            'limitations':['Heuristic product categories and generic packaging; not recovered SKU identity.',
                           'Unreviewed layout and cameras remain estimates. No measured geometry accuracy claim.']}
