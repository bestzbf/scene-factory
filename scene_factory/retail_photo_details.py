"""Bounded photo-inferred architectural and fixture detail for all retail jobs.

The model supplies data, never code or a per-store human profile. The shared
builder turns the same semantic fields into editable geometry for every view.
"""
import copy
import math

from schema import obj, vector
from spec_v2 import _walk, material, part

ID = {'type': 'string', 'minLength': 1, 'maxLength': 70, 'pattern': r'^[A-Za-z0-9][A-Za-z0-9_-]*$'}
COLOR = vector(3, 0, 1)
KINDS = ['mixed', 'snack', 'beverage', 'can', 'cup', 'household', 'gift', 'tobacco', 'box']
GLAZING_CATEGORIES = {'storefront_glazing', 'glazed_entrance', 'glass_entrance',
                      'glazed_entrance_door', 'glass_entrance_door'}
SCHEMA = obj({
    'structures': {'type': 'array', 'maxItems': 40, 'items': obj({
        'id': ID, 'role': {'type': 'string', 'enum': ['wall_segment', 'column', 'beam', 'door_frame']},
        'center_m': vector(3), 'size_m': vector(3, .02, 30),
        'rotation_z_deg': {'type': 'number', 'minimum': -360, 'maximum': 360},
        'color': COLOR})},
    'fixtures': {'type': 'array', 'maxItems': 160, 'items': obj({
        'object_id': ID, 'mode': {'type': 'string', 'enum': ['preserve', 'shelf', 'fridge']},
        'kind': {'type': 'string', 'enum': KINDS},
        'levels': {'type': 'integer', 'minimum': 2, 'maximum': 10},
        'frame_color': COLOR, 'rail_color': COLOR,
        'back_panel': {'type': 'boolean'}, 'endcaps': {'type': 'boolean'},
        'stock_cell_m': {'type': 'number', 'minimum': .07, 'maximum': .3},
        'top_height_m': {'type': 'number', 'minimum': 0, 'maximum': .4}})},
    'floor_tiles': {'type': 'array', 'maxItems': 1, 'items': obj({
        'size_m': {'type': 'number', 'minimum': .1, 'maximum': 2},
        'color': COLOR, 'grout_color': COLOR,
        'roughness': {'type': 'number', 'minimum': .05, 'maximum': 1}})},
})

CONTRACT = '''
PHOTO-INFERRED RETAIL DETAIL: Supply retail_details as part of this same shared reconstruction.
structures are actual editable rectangular wall segments, columns, overhead beams and door-frame members. center_m is the WORLD geometric centre; size_m is before yaw. Model side-zone openings and cross-aisle lintels as separated physical segments, leaving the aperture empty. A retained shell wall remains opaque: if a real entrance is on that boundary, omit that shell surface and reconstruct its surrounding segments. Never cover an opening with a full wall. Do not duplicate a structural member in both objects and structures. Keep all geometry inside the room, with at most 0.10 m shell thickness beyond the boundary. Do not use image planes or full-photo overlays.
fixtures contains observed merchandising detail keyed to existing object_id, independent of camera. Choose mode=shelf only for open retail shelving, mode=fridge for glass-front beverage refrigerators, preserve otherwise. Estimate real tier count, rail/frame colours, dense product spacing, back panel and endcap presence. Distinguish short cans from tall bottles, soft snack packets, cups, household packages and cartons. top_height_m is the height of stock on the highest deck, included WITHIN the object's target_dimensions_m total height. For each shelf use a local X length and local Y depth before yaw; rotating by 90 degrees changes aisle direction. Shared procedural stock uses those fields and a fixed packaging atlas. It cannot recover exact brands. Explicit objects supported by a fixture retain their measured catalog support, so that fixture will preserve its frame and only refine stock.
floor_tiles contains one metric tile pattern only when the photos show tile joints, or [] otherwise. All sizes and colours must be inferred from these photographs. For camera poses, use each visible fixture's TOP surface and vertical front edge as separate constraints; ceiling coverage, foreground occlusion and aisle vanishing direction must agree with the original in every view. A high surveillance camera looking steeply down must not turn into a horizontal room overview.
'''


def validate_details(raw):
    detail = raw.get('retail_details')
    if detail is None:
        return
    _walk(detail, SCHEMA, 'retail_details')
    entries = {o['id']: o for o in raw['objects']}
    seen = set()
    for spec in detail['fixtures']:
        identifier = spec['object_id']
        if identifier not in entries or identifier in seen:
            raise ValueError('Unknown or duplicate photo fixture: '+identifier)
        category = entries[identifier]['category']
        if spec['mode'] == 'shelf' and category not in {'shelf','display_rack','wire_display_rack','countertop_display_rack','tobacco_display'}:
            raise ValueError('Photo shelf replacement requires a shelf category: '+identifier)
        if spec['mode'] == 'fridge' and category not in {'glass_door_refrigerator','open_display_refrigerator','refrigerator'}:
            raise ValueError('Photo fridge replacement requires a refrigerator category: '+identifier)
        seen.add(identifier)
    seen = set(entries)
    room = raw['room']
    for spec in detail['structures']:
        identifier = spec['id']
        if identifier in seen:
            raise ValueError('Duplicate photo structure: '+identifier)
        seen.add(identifier)
        x,y,z = spec['center_m'];w,d,h=spec['size_m']
        yaw=math.radians(spec['rotation_z_deg'])
        ex=(abs(w*math.cos(yaw))+abs(d*math.sin(yaw)))/2
        ey=(abs(w*math.sin(yaw))+abs(d*math.cos(yaw)))/2
        if (x-ex < -room['width']/2-.101 or x+ex > room['width']/2+.101
                or y-ey < -.101 or y+ey > room['depth']+.101
                or z-h/2 < -.101 or z+h/2 > room['height']+.101):
            raise ValueError('Photo structure exceeds the shared room: '+identifier)
    if detail['floor_tiles'] and 'floor' in raw['architecture']['omit_surfaces']:
        raise ValueError('Photo tile material requires a floor')


def expand_structures(raw, shared):
    """Use existing bounded canonical parts; no executable model output."""
    for spec in raw.get('retail_details', {}).get('structures', []):
        identifier='photo_structure_'+spec['id']
        mat_id='photo_material_'+spec['id']
        shared['materials'].append(material(mat_id, list(spec['color']), roughness=.72))
        shared['custom_objects'].append({
            'id': identifier, 'position': list(spec['center_m']),
            'rotation_degrees': [0,0,spec['rotation_z_deg']], 'material_id': mat_id,
            'confidence': .65, 'observed': True, 'support_id': '',
            'parts': [part('box', size=list(spec['size_m']))]})


def inferred_settings(layout, settings):
    result = copy.deepcopy(settings)
    detail=(layout or {}).get('retail_details', {})
    if detail.get('floor_tiles'):
        result['floor_tiles']=copy.deepcopy(detail['floor_tiles'][0])
    for spec in detail.get('fixtures', []):
        result.setdefault('fixtures', {}).setdefault(spec['object_id'], {}).update(kind=spec['kind'])
    return result


def fixture_build_frame(entry, spec, room):
    """Express long-Y fixture footprints in the shared builder's long-X frame.

    The world footprint and allocated height are unchanged. A single-sided
    transposed fixture faces toward the room interior. This resolves a common
    model encoding of wall runs without shrinking a long run to a catalog seed.
    """
    dimensions=list(entry['dimensions'] if 'dimensions' in entry else entry['target_dimensions_m'])
    width,depth,height=dimensions
    yaw=entry['rotation_z_deg']
    min_length=.5 if spec['mode']=='shelf' else .6 if spec['mode']=='glazing' else .4
    max_depth=.5 if spec['mode']=='glazing' else 1.4
    transpose=(depth>max_depth or width<min_length) and width<=max_depth and depth>=min_length
    if transpose:
        dimensions=[depth,width,height]
        turn=90
        if spec['mode']=='fridge' or spec.get('back_panel'):
            xy=entry['position'][:2] if 'position' in entry else entry['xy']
            angle=math.radians(yaw+turn)
            front=(math.sin(angle),-math.cos(angle))
            interior=(-xy[0],room['depth']/2-xy[1])
            if sum(a*b for a,b in zip(front,interior))<0:turn=-90
        yaw+=turn
    else:
        turn=0
    width,depth,height=dimensions
    if spec['mode']=='shelf':
        dimensions[2]=height-spec['top_height_m']-.035
        valid=.5<=width<=12 and .18<=depth<=1.4 and .7<=dimensions[2]<=2.5
    elif spec['mode']=='fridge':
        valid=.4<=width<=3 and .35<=depth<=1.4 and 1.2<=height<=2.8
    elif spec['mode']=='glazing':
        valid=.6<=width<=8 and .02<=depth<=.5 and 1.8<=height<=3.2
    else:
        valid=False
    return {'dimensions_m':dimensions,'rotation_z_deg':yaw,'local_axis_turn_degrees':turn} if valid else None


def parametric_fixture_dimensions(layout, identifier):
    """Return the exact final allocation only if the shared builder can fulfill it.

    Explicit children depend on measured catalog supports and are ineligible.
    This is also called by selection, so geometry checks use the final bounds.
    """
    objects={o['id']:o for o in layout['objects']}
    entry=objects[identifier]
    if entry['support_id']!='floor' or any(o['support_id']==identifier for o in objects.values()):
        return None
    spec=next((s for s in layout.get('retail_details',{}).get('fixtures',[])
               if s['object_id']==identifier),None)
    if entry['category'] in GLAZING_CATEGORIES and 'retail_details' in layout:
        spec={'mode':'glazing'}
    if not spec:return None
    if fixture_build_frame(entry,spec,layout['room']) is not None:
        return list(entry['target_dimensions_m'])
    return None


def replace_glazing(root,spec):
    """A bounded physical frontage: glass panes, mullions, rails and handles."""
    import bpy
    from retail_stock_geometry import Mesh
    from generate_scene import material
    for child in list(root.children_recursive):bpy.data.objects.remove(child,do_unlink=True)
    root.scale=(1,1,1);root.location=spec['position']
    root.rotation_euler=(0,0,math.radians(spec['rotation_z_deg']))
    width,depth,height=spec['dimensions_m'];mesh=Mesh('AutomaticGlazing_'+root.name)
    frame=material('AutomaticGlazingFrame',(.045,.052,.055),.28,.6)
    glass=material('AutomaticFrontageGlass',(.82,.9,.92),.06,0)
    shader=next(n for n in glass.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    shader.inputs['Transmission Weight'].default_value=.92;shader.inputs['IOR'].default_value=1.45
    handle=material('AutomaticGlassHandle',(.4,.44,.47),.22,.8)
    rail=.035;frame_depth=min(depth,.055);panes=max(1,math.ceil(width/1.35))
    for index in range(panes+1):
        x=-width/2+rail/2+index*(width-rail)/panes
        mesh.box((x,0,height/2),(rail,frame_depth,height),frame)
    for z in (rail/2,height-rail/2):mesh.box((0,0,z),(width,frame_depth,rail),frame)
    pane_width=(width-rail)/panes-rail
    for index in range(panes):
        x=-width/2+rail/2+(index+.5)*(width-rail)/panes
        mesh.box((x,0,height/2),(pane_width,min(.008,depth/3),height-2*rail),glass)
    for sign in (-1,1):
        x=sign*min(.11,width/5)
        mesh.box((x,-depth/2+.006,min(1.05,height/2)),(.016,.012,.32),handle)
    obj=mesh.finish(root,bevel=0);obj['representation']='automatic_physical_glazing'
    return {'id':root.name,'stock_units':0,'dimensions_m':list(spec['dimensions_m'])}


def is_double_sided(entry, frame, spec, layout):
    """Infer a freestanding gondola's central spine from shared architecture."""
    if spec['mode']!='shelf' or not spec.get('back_panel') or frame['dimensions_m'][1]<.5:
        return False
    yaw=math.radians(frame['rotation_z_deg'])
    side=( -math.sin(yaw), math.cos(yaw))
    xy=entry['position'][:2];depth=frame['dimensions_m'][1]
    room=layout['room'];omit=layout['architecture']['omit_surfaces']
    walls=[]
    for name,center,normal in [('left',[-room['width']/2,room['depth']/2],[1,0]),
                               ('right',[room['width']/2,room['depth']/2],[1,0]),
                               ('back',[0,room['depth']],[0,1])]:
        if name not in omit:walls.append((center,normal,0,None))
    if layout['architecture']['front_wall']:walls.append(([0,0],[0,1],0,None))
    for structure in layout.get('retail_details',{}).get('structures',[]):
        if structure['role']!='wall_segment':continue
        w,d,h=structure['size_m'];angle=math.radians(structure['rotation_z_deg'])
        if d>w:angle+=math.pi/2
        normal=[-math.sin(angle),math.cos(angle)]
        walls.append((structure['center_m'][:2],normal,min(w,d),max(w,d)))
    for center,normal,thickness,span in walls:
        if abs(sum(a*b for a,b in zip(side,normal)))<.9:continue
        delta=[xy[i]-center[i] for i in range(2)]
        if span is not None:
            tangent=[normal[1],-normal[0]]
            if abs(sum(a*b for a,b in zip(delta,tangent)))>span/2+frame['dimensions_m'][0]/2:continue
        distance=abs(sum(a*b for a,b in zip(delta,normal)))-depth/2-thickness/2
        if distance<.3:return False
    return True


def build_fixtures(bundle, layout, atlas, groups):
    """Rebuild bounded catalog fixture slots from model-inferred shared detail.

    Fixtures with explicit supported objects keep the verified catalog frame;
    this avoids removing supported children or changing their contact surface.
    """
    import bpy
    from reviewed_retail_fixtures import replace_shelf, replace_fridge
    entries={o['id']:o for o in bundle['shared_scene']['base_scene']['objects']}
    rebuilt=[];skipped=[]
    specs=list((layout or {}).get('retail_details', {}).get('fixtures', []))
    glazing={o['id'] for o in (layout or {}).get('objects',[])
             if 'retail_details' in (layout or {}) and o['category'] in GLAZING_CATEGORIES and o['support_id']=='floor'}
    specs=[s for s in specs if s['object_id'] not in glazing]
    specs += [{'object_id':identifier,'mode':'glazing','top_height_m':0} for identifier in sorted(glazing)]
    for spec in specs:
        identifier=spec['object_id']
        if spec['mode']=='preserve':continue
        root=bpy.data.objects.get(identifier);entry=entries[identifier]
        if root is None:raise ValueError('Photo fixture root is missing: '+identifier)
        if any(i != identifier for i in groups.get(identifier, [])):
            skipped.append({'id':identifier,'reason':'measured support retained for explicit child objects'})
            continue
        frame=fixture_build_frame(entry,spec,bundle['shared_scene']['base_scene']['room'])
        if frame is None:
            skipped.append({'id':identifier,'reason':'outside shared fixture-builder dimension range'})
            continue
        total_height=entry['dimensions'][2]
        payload={**spec, 'position':list(entry['position']), 'rotation_z_deg':entry['rotation_z_deg'],
                 **frame, 'stock_style':'reference_mixed',
                 'top_height':spec['top_height_m']}
        payload['double_sided']=is_double_sided(entry,frame,spec,layout)
        if spec['mode']=='shelf':
            record=replace_shelf(root,payload,atlas,bundle['scene_id'])
        elif spec['mode']=='glazing':
            record=replace_glazing(root,payload)
        else:
            record=replace_fridge(root,payload,atlas,bundle['scene_id'])
        for child in root.children_recursive:
            if child.type=='MESH':
                child['representation']='automatic_photo_inferred_'+spec['mode']
                child['fixture_id']=identifier
        root['geometry_source']='automatic_parametric_retail'
        rebuilt.append({**record,'mode':spec['mode'],'geometry_source':'shared_parametric_builder',
                        'local_axis_turn_degrees':frame['local_axis_turn_degrees'],
                        'double_sided':payload['double_sided'],
                        'detail_source':'automatic_photo_inference','allocated_height_m':total_height})
    return rebuilt,skipped
