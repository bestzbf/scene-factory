"""Deterministic asset selection and explicit floor/surface placement.

The model provides semantic intent and world XY. This module never invents
missing assets or treats an AABB top as a surface. By default it never moves
XY to hide a failure; the opt-in selection.settle_max_shift_m lets it move each
object to the nearest free spot within that many metres (and accept a child
whose centre rests on its support with most of its footprint supported), and
records every move. Collision checks are conservative oriented boxes, not
full mesh physics.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()
from auto_asset_binding import normalize_options
from multiview_spec import validate
from library_layout import validate_layout


class LibrarySelectionError(ValueError):
    def __init__(self, message, report):
        self.report = report
        super().__init__(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _rotate(point, yaw):
    angle = math.radians(yaw)
    c, s = math.cos(angle), math.sin(angle)
    return [c*point[0]-s*point[1], s*point[0]+c*point[1]]


def _corners(position, dimensions, yaw):
    return [[position[0]+p[0], position[1]+p[1]]
            for p in [_rotate([x*dimensions[0]/2,y*dimensions[1]/2],yaw)
                      for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]]]


def _inside(point, polygon, tolerance=1e-6):
    if len(polygon) < 3:
        return False
    signs = []
    for a,b in zip(polygon, polygon[1:]+polygon[:1]):
        cross = (b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])
        if abs(cross)>tolerance:
            signs.append(cross>0)
    return bool(signs) and (all(signs) or not any(signs))


def _overlap_xy(left, right, tolerance=.002):
    for polygon in (left,right):
        for a,b in zip(polygon, polygon[1:]+polygon[:1]):
            normal = [-(b[1]-a[1]),b[0]-a[0]]
            length = math.hypot(*normal)
            if length <= 1e-12:
                continue
            axis = [v/length for v in normal]
            l = [sum(x*y for x,y in zip(p,axis)) for p in left]
            r = [sum(x*y for x,y in zip(p,axis)) for p in right]
            if min(max(l),max(r))-max(min(l),min(r)) <= tolerance:
                return False
    return True


def _fit(asset, target, options, footprint=False):
    source = [v*asset['meters_per_unit'] for v in asset['dimensions_import_units']]
    ratios = [a/b for a,b in zip(source,target)]
    fixed = asset.get('placement_rules',{}).get('allow_uniform_scale') is False
    def result(scale):
        dimensions = [x*scale for x in source]
        return scale,dimensions,max(abs(a-b)/b for a,b in zip(dimensions,target))
    if footprint and not fixed:
        # Settle mode: prefer the scale that keeps the asset inside the model's whole
        # target box, so stacks do not grow and floor-to-ceiling items stay under the
        # ceiling; else one that keeps only the footprint inside, so a row of objects
        # laid out edge to edge still fits.
        for limit in (max(ratios), max(ratios[0], ratios[1])):
            snug = min(options['max_uniform_scale'],max(options['min_uniform_scale'],1/limit))
            candidate = result(snug)
            if candidate[2] <= options['dimension_tolerance']+1e-12:
                return candidate
    scale = min(options['max_uniform_scale'],max(options['min_uniform_scale'],2/(min(ratios)+max(ratios))))
    return result(1. if fixed else scale)


def _category_index(catalog):
    # The catalog is immutable inside a batch snapshot.
    index = getattr(catalog, '_library_category_index', None)
    if index is None:
        index = {}
        for asset in catalog.assets.values():
            index.setdefault(asset.get('category'),[]).append(asset)
        index = {key:tuple(sorted(values,key=lambda a:a['asset_id'])) for key,values in index.items()}
        catalog._library_category_index = index
    return index


SUPPORT_COVERAGE = .6      # settle mode: share of a child's footprint that must rest on its support
MIN_OBJECT_DIMENSION_M, MAX_OBJECT_DIMENSION_M = .01, 30.   # schema.py object 'dimensions' bounds


def _area(polygon):
    return sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(polygon, polygon[1:]+polygon[:1]))/2


def _clip(subject, clip):
    """Intersection of two convex polygons (Sutherland-Hodgman)."""
    if _area(clip) < 0:
        clip = clip[::-1]
    output = [list(p) for p in subject]
    for a, b in zip(clip, clip[1:]+clip[:1]):
        if not output:
            break
        points, output = output, []
        def side(p):
            return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
        previous = points[-1]
        for current in points:
            sc, sp = side(current), side(previous)
            if (sc >= 0) != (sp >= 0):
                t = sp/(sp-sc)
                output.append([previous[0]+t*(current[0]-previous[0]), previous[1]+t*(current[1]-previous[1])])
            if sc >= 0:
                output.append(current)
            previous = current
    return output


def _support_fit(local, polygon):
    """'full' if the footprint lies inside the polygon, 'partial' if its centre does
    and at least SUPPORT_COVERAGE of it is supported, else None."""
    if all(_inside(p, polygon) for p in local):
        return 'full', 1.
    centre = [sum(p[0] for p in local)/len(local), sum(p[1] for p in local)/len(local)]
    if not _inside(centre, polygon):
        return None, 0.
    footprint = abs(_area(local))
    covered = abs(_area(_clip(local, polygon))) if footprint > 0 else 0.
    coverage = covered/footprint if footprint > 0 else 0.
    return ('partial' if coverage >= SUPPORT_COVERAGE else None), coverage


def _penetration(left, right):
    """Overlap depth of two convex XY polygons along their best separating axis; 0 if apart."""
    best = float('inf')
    for polygon in (left, right):
        for a, b in zip(polygon, polygon[1:]+polygon[:1]):
            normal = (b[1]-a[1], a[0]-b[0])
            length = math.hypot(*normal)
            if length <= 1e-12:
                continue
            axis = (normal[0]/length, normal[1]/length)
            l = [p[0]*axis[0]+p[1]*axis[1] for p in left]
            r = [p[0]*axis[0]+p[1]*axis[1] for p in right]
            depth = min(max(l), max(r))-max(min(l), min(r))
            if depth <= 0:
                return 0.
            best = min(best, depth)
    return best


WALL_ANCHOR_M = .15        # a floor object this close to a wall is treated as standing against it
# Settle mode never omits large floor objects (checkout counters, coolers, island and
# end-cap shelves): footprint at least LARGE_FOOTPRINT_M2, or at least
# LARGE_TALL_HEIGHT_M tall with a footprint of LARGE_TALL_FOOTPRINT_M2. They claim
# their spot before smaller objects and may move LARGE_SHIFT_FACTOR times the normal
# budget to find one; a large object that still has no free spot blocks the scene.
LARGE_FOOTPRINT_M2 = .5
LARGE_TALL_HEIGHT_M, LARGE_TALL_FOOTPRINT_M2 = 1.2, .2
LARGE_SHIFT_FACTOR = 2.
FOOTPRINT_SLACK = .02      # settle mode: an asset this much wider/deeper than requested counts as overshooting


def _is_large(dims):
    footprint = dims[0]*dims[1]
    return footprint >= LARGE_FOOTPRINT_M2 or (dims[2] >= LARGE_TALL_HEIGHT_M and footprint >= LARGE_TALL_FOOTPRINT_M2)


def _settle(requests, dims, supports, room, max_shift, step=.02, protected=frozenset()):
    """Find each object the nearest free spot within max_shift metres of the model's position.

    supports maps an object id to its measured support polygons in local metres
    and the height of the highest one. Protected (large) floor objects are placed
    first and may move up to LARGE_SHIFT_FACTOR * max_shift, then the rest. Within each group, objects standing against a wall
    (shelves, coolers) go first, tallest first, then free-standing ones, larger
    footprints first; each keeps the model's position when free. Supported
    objects follow, parent before child, protected and larger ones first. A spot
    is free when it is
    inside the room (floor) or rests on a measured support polygon (fully, or
    with its centre on it and SUPPORT_COVERAGE of its footprint supported), and
    clear of every already placed object whose height range it shares, except
    its own ancestors. An object whose top would reach above the ceiling has no
    free spot. Moving a parent carries its descendants. Returns
    ({object_id: [dx, dy]}, {object_id: [x, y]}, [object ids with no free spot]).
    """
    origin = {k: list(r['xy']) for k, r in requests.items()}
    xy = {k: list(r['xy']) for k, r in requests.items()}
    children = {}
    for key, req in requests.items():
        children.setdefault(req['support_id'], []).append(key)
    radius = {k: math.hypot(dims[k][0], dims[k][1])/2 for k in requests}
    def offsets_within(limit):
        n = int(limit/step+1e-9)
        return sorted(((i*step, j*step) for i in range(-n, n+1) for j in range(-n, n+1)
                       if math.hypot(i*step, j*step) <= limit+1e-9),
                      key=lambda o: (round(math.hypot(*o), 9), o))
    offsets = offsets_within(max_shift)
    large_offsets = offsets_within(max_shift*LARGE_SHIFT_FACTOR) if protected else offsets

    def ancestors(key):
        found = []
        while requests[key]['support_id'] != 'floor':
            key = requests[key]['support_id']; found.append(key)
        return found

    def z_range(key):
        chain = ancestors(key)[::-1] + [key]
        bottom = 0.
        for below, above in zip(chain, chain[1:]):
            bottom += supports.get(below, ([], dims[below][2]))[1]
        return bottom, bottom+dims[key][2]

    def corners_at(key, point):
        return _corners(list(point)+[0], dims[key], requests[key]['rotation_z_deg'])

    def carry(key, dx, dy):
        for child in children.get(key, []):
            for store in (xy, origin):
                store[child] = [store[child][0]+dx, store[child][1]+dy]
            carry(child, dx, dy)

    def wall_distance(key):
        c = corners_at(key, origin[key])
        return min(min(p[0] for p in c)+room['width']/2, room['width']/2-max(p[0] for p in c),
                   room['depth']-max(p[1] for p in c))

    def in_room(corners):
        return all(-room['width']/2-1e-6 <= p[0] <= room['width']/2+1e-6 and -1e-6 <= p[1] <= room['depth']+1e-6
                   for p in corners)

    placed = []

    def clear(key, point, corners):
        low, high = z_range(key)
        skip = set(ancestors(key))
        for other, other_corners, other_low, other_high in placed:
            if other in skip or min(high, other_high)-max(low, other_low) <= .002:
                continue
            if math.hypot(point[0]-xy[other][0], point[1]-xy[other][1]) >= radius[key]+radius[other]:
                continue
            if _penetration(corners, other_corners) > 1e-6:
                return False
        return True

    def on_support(key, corners, parent, wanted):
        px, py = xy[parent]; yaw = requests[parent]['rotation_z_deg']
        local = [_rotate([p[0]-px, p[1]-py], -yaw) for p in corners]
        for polygon in supports.get(parent, ([], 0.))[0]:
            kind, _ = _support_fit(local, polygon)
            if kind == 'full' or (kind == 'partial' and wanted == 'partial'):
                return True
        return False

    floor = children.get('floor', [])
    # Large objects claim their spot first; smaller ones move around them. Against a
    # wall, taller units are fixtures and keep the wall; elsewhere the larger footprint
    # wins, so a slightly taller small display cannot push a long counter aside.
    def floor_key(k):
        against_wall = wall_distance(k) <= WALL_ANCHOR_M
        return (k not in protected, not against_wall, -round(dims[k][2], 2) if against_wall else 0.,
                -dims[k][0]*dims[k][1], k)
    order = sorted(floor, key=floor_key)
    depth_of = {k: len(ancestors(k)) for k in requests}
    order += sorted((k for k in requests if requests[k]['support_id'] != 'floor'),
                    key=lambda k: (depth_of[k], requests[k]['support_id'], k not in protected,
                                   -dims[k][0]*dims[k][1], k))
    unplaced = []
    for key in order:
        support = requests[key]['support_id']
        if any(a in unplaced for a in ancestors(key)):
            unplaced.append(key); continue
        if z_range(key)[1] > room['height']+1e-6:
            # Would reach through the ceiling even on its support's highest surface.
            unplaced.append(key); continue
        chosen = None
        for wanted in (('any',) if support == 'floor' else ('full', 'partial')):
            for dx, dy in (large_offsets if key in protected else offsets):
                point = [round(origin[key][0]+dx, 6), round(origin[key][1]+dy, 6)]
                corners = corners_at(key, point)
                # A supported object may overhang its support, but never the room.
                if not in_room(corners):
                    continue
                if support != 'floor' and not on_support(key, corners, support, wanted):
                    continue
                if clear(key, point, corners):
                    chosen = point
                    break
            if chosen:
                break
        if chosen is None:
            unplaced.append(key)
            chosen = [round(origin[key][0], 6), round(origin[key][1], 6)]
        carry(key, chosen[0]-xy[key][0], chosen[1]-xy[key][1])
        xy[key] = chosen
        if key not in unplaced:
            placed.append((key, corners_at(key, chosen), *z_range(key)))
    moves = {}
    for key, req in requests.items():
        dx, dy = xy[key][0]-req['xy'][0], xy[key][1]-req['xy'][1]
        if math.hypot(dx, dy) > 1e-6 and key not in unplaced:
            moves[key] = [round(dx, 6), round(dy, 6)]
    return moves, xy, unplaced

def select_and_place(bundle, layout, catalog, options=None):
    """Return effective canonical bundle, v1 bindings, and an auditable report.

    All declared objects must bind. Failed candidates are quarantined at scene
    level by the batch coordinator; a generic box is never silently rendered.
    """
    validate(bundle)
    validate_layout(layout, scene_id=bundle['scene_id'])
    options = normalize_options(options)
    result = copy.deepcopy(bundle)
    entries = {e['id']:e for e in result['shared_scene']['base_scene']['objects']}
    requests = {e['id']:e for e in layout['objects']}
    if len(requests)!=len(layout['objects']) or set(entries)!=set(requests):
        raise ValueError('Library intent and canonical object IDs differ')
    report = {'schema_version':'library_selection_v1','scene_id':bundle['scene_id'],
              'input_bundle_sha256':digest(bundle),'layout_sha256':digest(layout),
              'catalog_sha256':getattr(catalog,'source_sha256',None) or digest(catalog.data),
              'status':'pending','objects':{},'errors':[],
              'appearance_match_verified':False,'visual_acceptance':False,
              'collision_method':'conservative oriented bounding boxes; parent/descendant pairs exempt'}
    selected, used = {},Counter()
    index = _category_index(catalog)
    # Readiness re-hashes the model and re-validates the intake receipt. Every
    # object of one category walks the same candidate list, so one scene would
    # repeat that work once per object. Memoize for this call only: files cannot
    # change mid-selection, and a later call still re-checks from scratch, which
    # is what makes a modified asset or receipt fail the next scene.
    readiness = {}
    def candidate_readiness(asset):
        key = asset['asset_id']
        if key not in readiness:
            readiness[key] = catalog.production_readiness(asset)
        return readiness[key]
    # Material hashing is the pricier half of the same repeated walk. Both
    # lookups are read-only, so one success per material is enough for this
    # selection; failures are re-raised every time rather than cached, so a
    # broken material still surfaces on every object that reaches it.
    verified_materials = set()
    def verify_material(material_id):
        if material_id not in verified_materials:
            catalog.verify_material_hashes(material_id)
            verified_materials.add(material_id)
    settling = options.get('settle_max_shift_m', 0.) > 0
    for identifier,req in requests.items():
        intended = 'floor' if req['support_id']=='floor' else 'table'
        candidates, rejected = [],Counter()
        for asset in index.get(req['category'],()):
            if any(child['support_id']==identifier for child in requests.values()) and not asset.get('support_surfaces'):
                rejected['no_verified_support_surfaces']+=1;continue
            if asset.get('scope') != 'single_prop' or asset.get('placement') not in ({'floor'} if intended=='floor' else {'table','surface','support'}):
                rejected['placement_or_scope']+=1;continue
            ready,reasons = candidate_readiness(asset)
            if not ready:
                rejected.update(reasons);continue
            if asset.get('front_axis')!='-Y' or asset.get('up_axis')!='+Z':
                rejected['noncanonical_axes']+=1;continue
            if req['rotation_z_deg'] and asset.get('placement_rules',{}).get('can_rotate_z') is False:
                rejected['rotation_forbidden']+=1;continue
            scale,dims,error = _fit(asset,req['target_dimensions_m'],options,footprint=settling)
            if error > options['dimension_tolerance']+1e-12:
                rejected['dimension_mismatch']+=1;continue
            # The scene contract bounds every placed box to 0.01..30 m per axis. A
            # thin asset (mat, cable, phone case) scaled down can fall below that;
            # reject the candidate here instead of failing validation after selection.
            if not all(MIN_OBJECT_DIMENSION_M <= value <= MAX_OBJECT_DIMENSION_M for value in dims):
                rejected['scaled_size_out_of_bounds']+=1;continue
            for material_id in sorted(asset.get('material_variant_ids',[])):
                material = catalog.get_material_variant(material_id)
                if asset['geometry_variant_id'] not in material['compatible_geometry_ids']:
                    rejected['material_geometry_incompatible']+=1;continue
                verify_material(material_id)
                candidates.append((asset,material_id,scale,dims,error))
        obj_report = {'category':req['category'],'support_id':req['support_id'],
                      'candidates_examined':len(index.get(req['category'],())),
                      'rejection_counts':dict(rejected)}
        report['objects'][identifier]=obj_report
        if not candidates:
            report['errors'].append({'code':'missing_compatible_asset','object_id':identifier,
                                     'category':req['category']});continue
        load=[]
        for child in requests.values():
            if child['support_id']==identifier:
                w,d=child['target_dimensions_m'][:2]
                turned=round(((child['rotation_z_deg']-req['rotation_z_deg'])%180)/90)%2==1
                load.append((d,w) if turned else (w,d))
        def capacity_short(asset,scale):
            # Settle mode: prefer furniture whose measured surfaces can hold what the model put
            # on it (e.g. a clear-top shelf variant over a fully stocked one). Each item must be
            # able to rest on one surface with SUPPORT_COVERAGE, and the items together must not
            # exceed the largest surface's area.
            if not settling or not load:
                return False
            units=asset['meters_per_unit']*scale
            boxes=[]
            for surface in asset.get('support_surfaces',[]):
                polygon=[[v*units for v in point] for point in surface['polygon_xy']]
                xs,ys=[p[0] for p in polygon],[p[1] for p in polygon]
                boxes.append((max(xs)-min(xs),max(ys)-min(ys),abs(_area(polygon))))
            if not boxes:
                return True
            def fits(item):
                return any(min(1.,sx/item[0])*min(1.,sy/item[1])>=SUPPORT_COVERAGE for sx,sy,_ in boxes)
            return not all(fits(item) for item in load) or max(b[2] for b in boxes)<sum(w*d for w,d in load)
        def footprint_overshoot(dims):
            # Settle mode: a model slot in a row of coolers or counters is exactly as wide as
            # the gap between neighbours. Prefer an asset that does not overshoot it.
            if not settling:
                return False
            target=req['target_dimensions_m']
            return dims[0]>target[0]*(1+FOOTPRINT_SLACK) or dims[1]>target[1]*(1+FOOTPRINT_SLACK)
        def rank(c):
            asset,material,scale,dims,error=c
            pair=asset['geometry_variant_id'],material
            tie=digest([options['seed'],bundle['scene_id'],identifier,*pair])
            return (capacity_short(asset,scale),footprint_overshoot(dims),
                    bool(used[pair]) if options['diversity'] else False,round(error,12),tie)
        candidate=min(candidates,key=rank)
        asset,material_id,scale,dims,error=candidate
        used[(asset['geometry_variant_id'],material_id)]+=1
        selected[identifier]=candidate
        obj_report.update(asset_id=asset['asset_id'],material_variant_id=material_id,
                          uniform_scale=scale,dimensions_m=dims,relative_size_error=error,
                          scale_source=asset.get('scale_source'),status='selected')
    if report['errors']:
        report['status']='blocked'
        raise LibrarySelectionError('Library has no compatible assets for some requested objects',report)
    shift = options.get('settle_max_shift_m', 0.)
    if settling:
        # Model layouts are estimates; selected assets can be a few cm larger
        # than the estimate. Settle with the actual selected dimensions, move
        # nothing further than the configured budget, and record every move.
        dims = {key: selected[key][3] for key in requests}
        # Large by the model's requested size or by the selected asset's size: an asset
        # chosen a few cm smaller than requested must not turn a counter into an
        # omittable small object.
        protected = frozenset(key for key in requests if requests[key]['support_id'] == 'floor'
                              and (_is_large(dims[key]) or _is_large(requests[key]['target_dimensions_m'])))
        supports = {}
        for key, (asset, _, scale, _, _) in selected.items():
            units = asset['meters_per_unit']*scale
            surfaces = asset.get('support_surfaces', [])
            supports[key] = ([[[v*units for v in point] for point in surface['polygon_xy']] for surface in surfaces],
                             max([surface['z']*units for surface in surfaces] or [dims[key][2]]))
        moves, settled, unplaced = _settle(requests, dims, supports, result['shared_scene']['base_scene']['room'], shift,
                                           protected=protected)
        requests = {key: {**req, 'xy': [round(settled[key][0], 6), round(settled[key][1], 6)]}
                    for key, req in requests.items()}
        omitted = []
        large_unplaced = sorted(key for key in unplaced if key in protected)
        # Only large objects may move past the normal budget; their descendants ride along.
        far = sorted(key for key, value in moves.items() if key in protected and math.hypot(*value) > shift+1e-6)
        for key in large_unplaced:
            # Never silently dropped: the normal checks below report its overlap or
            # bounds problem and block the scene.
            report['objects'][key]['omission_refused'] = 'large_object'
        if (unplaced and not large_unplaced
                and len(unplaced) <= options.get('settle_max_omitted_fraction', 0.)*len(requests)+1e-9):
            # Every descendant of an unplaced object is itself unplaced, so the
            # remaining support graph stays closed.
            omitted = sorted(unplaced)
            base = result['shared_scene']['base_scene']
            base['objects'] = [entry for entry in base['objects'] if entry['id'] not in unplaced]
            for key in omitted:
                requests.pop(key); entries.pop(key); selected.pop(key)
                report['objects'][key].update(status='omitted',
                    omission_reason='no position free of overlaps on a measured support within the settle budget')
            moves = {key: value for key, value in moves.items() if key not in unplaced}
        report['settling'] = {'max_shift_m': shift, 'moved_objects': len(moves), 'moves_m': moves,
                              'unresolved_objects': unplaced, 'omitted_objects': omitted,
                              'large_objects': sorted(protected), 'large_objects_unplaced': large_unplaced,
                              'large_object_max_shift_m': shift*LARGE_SHIFT_FACTOR,
                              'large_objects_moved_beyond_budget': [key for key in far if key in moves],
                              'max_omitted_fraction': options.get('settle_max_omitted_fraction', 0.),
                              'support_coverage_min': SUPPORT_COVERAGE,
                              'method': 'nearest free spot per object within the budget; large floor objects first and '
                                        'never omitted; children rest on measured support surfaces; categories, sizes '
                                        'and yaw unchanged'}
    resolved,visiting={},set()
    def resolve(identifier):
        if identifier in resolved:return resolved[identifier]
        if identifier in visiting:raise ValueError('Library support graph has a cycle')
        if identifier not in requests:raise ValueError('Unknown support instance '+identifier)
        visiting.add(identifier)
        req=requests[identifier]
        asset,material_id,scale,dims,error=selected[identifier]
        z=0.
        if req['support_id']!='floor':
            parent_id=req['support_id']
            parent=resolve(parent_id)
            parent_asset,_,parent_scale,_,_=selected[parent_id]
            parent_req=requests[parent_id]
            child_corners=_corners(req['xy']+[0],dims,req['rotation_z_deg'])
            local=[_rotate([p[0]-parent[0],p[1]-parent[1]],-parent_req['rotation_z_deg']) for p in child_corners]
            surfaces,partial=[],[]
            units=parent_asset['meters_per_unit']*parent_scale
            for surface in parent_asset.get('support_surfaces',[]):
                polygon=[[v*units for v in p] for p in surface['polygon_xy']]
                if all(_inside(p,polygon) for p in local):
                    surfaces.append(surface)
                elif settling:
                    kind,coverage=_support_fit(local,polygon)
                    if kind=='partial':partial.append((coverage,surface))
            if not surfaces and partial:
                # Settle mode only: centre of mass on the surface and most of the
                # footprint supported (a box a little wider than the one below it).
                coverage,surface=max(partial,key=lambda item:(item[0],item[1]['z']))
                surfaces=[surface]
                report['objects'][identifier]['support_coverage']=round(coverage,4)
            if not surfaces:
                report['errors'].append({'code':'no_verified_support_surface','object_id':identifier,
                                         'support_id':parent_id})
                raise ValueError('Object footprint does not fit a verified support surface: '+identifier)
            surface=max(surfaces,key=lambda s:s['z'])
            z=parent[2]+surface['z']*units
            report['objects'][identifier]['surface_id']=surface['id']
        elif req['clearance_m'] > 1e-6:
            raise ValueError('Floor placements must have zero clearance: '+identifier)
        position=req['xy']+[z+req['clearance_m']]
        entry=entries[identifier]
        entry.update(position=position,dimensions=dims,rotation_z_deg=req['rotation_z_deg'])
        resolved[identifier]=position
        report['objects'][identifier]['position_m']=position
        visiting.remove(identifier)
        return position
    try:
        for identifier in requests:resolve(identifier)
    except (ValueError,KeyError,TypeError) as exc:
        report['status']='blocked'
        if not report['errors']:report['errors'].append({'code':'invalid_support_graph','message':str(exc)})
        raise LibrarySelectionError(str(exc),report) from None
    room=result['shared_scene']['base_scene']['room']
    footprints={}
    for identifier,entry in entries.items():
        corners=_corners(entry['position'],entry['dimensions'],entry['rotation_z_deg'])
        footprints[identifier]=corners
        if (any(p[0]<-room['width']/2-1e-6 or p[0]>room['width']/2+1e-6
                or p[1]<-1e-6 or p[1]>room['depth']+1e-6 for p in corners)
                or entry['position'][2]<-1e-6 or entry['position'][2]+entry['dimensions'][2]>room['height']+1e-6):
            report['errors'].append({'code':'asset_outside_room','object_id':identifier})
    def ancestors(identifier):
        values=set()
        while requests[identifier]['support_id']!='floor':
            identifier=requests[identifier]['support_id'];values.add(identifier)
        return values
    keys=sorted(entries)
    for i,left in enumerate(keys):
        for right in keys[i+1:]:
            if left in ancestors(right) or right in ancestors(left):continue
            l,r=entries[left],entries[right]
            overlap=min(l['position'][2]+l['dimensions'][2],r['position'][2]+r['dimensions'][2])-max(l['position'][2],r['position'][2])
            if overlap>.002 and _overlap_xy(footprints[left],footprints[right]):
                report['errors'].append({'code':'asset_bounds_overlap','object_ids':[left,right]})
    if report['errors']:
        report['status']='blocked'
        raise LibrarySelectionError('Selected asset placement violates geometry constraints',report)
    bindings={'schema_version':1,'instances':[]}
    for identifier,(asset,material_id,scale,_,_) in selected.items():
        bindings['instances'].append({'object_id':identifier,'asset_id':asset['asset_id'],
             'material_variant_id':material_id,'uniform_scale':scale,
             'rotation_xyz_deg':[0,0,requests[identifier]['rotation_z_deg']]})
    validate(result)
    report.update(status='passed',bound_objects=len(selected),fallback_objects=0,
                  omitted_objects=len(report.get('settling',{}).get('omitted_objects',[])),
                  effective_bundle_sha256=digest(result),options=options)
    return result,bindings,report

