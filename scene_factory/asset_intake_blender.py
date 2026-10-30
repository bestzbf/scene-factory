"""Isolated Blender compiler for the minimal opaque, static GLB protocol."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from asset_intake import validate_input, sha, atomic_json, RECEIPT_VERSION, POLICY_VERSION


def inspect_glb(path):
    data=Path(path).read_bytes()
    if len(data)<20: raise ValueError('truncated GLB')
    magic,version,length=struct.unpack_from('<4sII',data)
    if magic!=b'glTF' or version!=2 or length!=len(data): raise ValueError('invalid glTF 2.0 binary header')
    offset=12;document=None;bins=0
    while offset<len(data):
        if offset+8>len(data):raise ValueError('truncated GLB chunk')
        size,kind=struct.unpack_from('<II',data,offset);offset+=8
        if size%4 or offset+size>len(data):raise ValueError('invalid GLB chunk size')
        content=data[offset:offset+size];offset+=size
        if kind==0x4e4f534a:
            if document is not None:raise ValueError('duplicate GLB JSON')
            document=json.loads(content.decode('utf-8'))
        elif kind==0x004e4942: bins+=1
    if not document or bins!=1:raise ValueError('GLB needs JSON and one embedded binary chunk')
    if document.get('animations') or document.get('skins') or document.get('cameras'):raise ValueError('only static mesh assets without animation, skin or cameras are supported')
    if document.get('extensionsRequired'):raise ValueError('required glTF extensions are not supported in the minimal profile')
    for buffer in document.get('buffers',[]):
        if 'uri' in buffer:raise ValueError('external/data URI buffers are not supported; use embedded GLB')
    for image in document.get('images',[]):
        if 'uri' in image or 'bufferView' not in image:raise ValueError('all texture images must be embedded in GLB bufferViews')
        if image.get('mimeType') not in ('image/png','image/jpeg'):raise ValueError('minimal profile supports embedded PNG/JPEG')
    for node in document.get('nodes',[]):
        if node.get('extensions'):raise ValueError('node extensions are unsupported')
    for material in document.get('materials',[]):
        if material.get('alphaMode','OPAQUE')!='OPAQUE':raise ValueError('minimal profile requires opaque materials')
        if material.get('extensions'):raise ValueError('material extensions require explicit conversion before intake')
        if any(float(v)!=0 for v in material.get('emissiveFactor',[0,0,0])) or material.get('emissiveTexture'):raise ValueError('emissive materials are outside the minimal profile')
        for owner in (material,material.get('pbrMetallicRoughness',{})):
            for key,value in owner.items():
                if key.endswith('Texture') and isinstance(value,dict):
                    if value.get('texCoord',0)!=0 or value.get('extensions'):raise ValueError('only first-UV textures without transforms/extensions are supported')
    for mesh in document.get('meshes',[]):
        if mesh.get('weights'):raise ValueError('morph weights are unsupported')
        for primitive in mesh.get('primitives',[]):
            if primitive.get('mode',4)!=4 or primitive.get('targets') or primitive.get('extensions'):raise ValueError('only static triangle primitives are supported')
    return document


def hull(points):
    pts=sorted(set(points))
    if len(pts)<3:return []
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[];upper=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]


def area(poly):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1])))/2


def _exact_support_surfaces(meshes):
    # Only coplanar upward triangles whose connected patch exactly fills a
    # convex polygon are admitted. Holes, gaps, tilted surfaces and overlaps
    # are rejected; bounding-box tops are never used as fabricated supports.
    planes={}
    for obj in meshes:
        mesh=obj.data;mesh.calc_loop_triangles()
        for tri in mesh.loop_triangles:
            points=[obj.matrix_world@mesh.vertices[i].co for i in tri.vertices]
            normal=(points[1]-points[0]).cross(points[2]-points[0])
            if normal.length<=1e-12 or normal.normalized().z<.99999:continue
            z=sum(p.z for p in points)/3
            if max(p.z for p in points)-min(p.z for p in points)>1e-5 or z<1e-4:continue
            xy=[(round(p.x,7),round(p.y,7)) for p in points]
            planes.setdefault(round(z,5),[]).append(xy)
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    vertices=[]; faces=[]
    for obj in meshes:
        offset=len(vertices)
        vertices.extend(obj.matrix_world@v.co for v in obj.data.vertices)
        faces.extend([offset+i for i in p.vertices] for p in obj.data.polygons)
    tree=BVHTree.FromPolygons(vertices,faces)
    ray_top=max(v.z for v in vertices)+1
    surfaces=[]
    for z,triangles in sorted(planes.items()):
        if len(triangles)>2000:continue
        unique={tuple(sorted(t)):t for t in triangles}
        if len(unique)!=len(triangles):continue
        # Connected via a shared edge, not just an incidental touching corner.
        edges={};parents=list(range(len(triangles)))
        def find(i):
            while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
            return i
        for i,t in enumerate(triangles):
            for a,b in zip(t,t[1:]+t[:1]):
                edge=tuple(sorted((a,b)))
                if edge in edges:parents[find(i)]=find(edges[edge])
                else:edges[edge]=i
        groups={}
        for i,t in enumerate(triangles):groups.setdefault(find(i),[]).append(t)
        for parts in groups.values():
            polygon=hull([p for t in parts for p in t]);patch=sum(area(t) for t in parts)
            if not polygon or area(polygon)<.0025 or abs(area(polygon)-patch)>max(1e-7,patch*1e-4):continue
            # Exclude hidden surfaces such as leg tops underneath a tabletop.
            # Testing triangle centroids and inset boundary samples is
            # conservative evidence, not a full object-placement simulation.
            center=(sum(p[0] for p in polygon)/len(polygon),sum(p[1] for p in polygon)/len(polygon))
            samples=[center]+[(sum(p[0] for p in t)/3,sum(p[1] for p in t)/3) for t in parts]
            samples.extend((p[0]*.99+center[0]*.01,p[1]*.99+center[1]*.01) for p in polygon)
            minx,maxx=min(p[0] for p in polygon),max(p[0] for p in polygon)
            miny,maxy=min(p[1] for p in polygon),max(p[1] for p in polygon)
            for ix in range(1,11):
                for iy in range(1,11):
                    x=minx+(maxx-minx)*ix/11;y=miny+(maxy-miny)*iy/11
                    if all((b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])>=-1e-9
                           for a,b in zip(polygon,polygon[1:]+polygon[:1])):samples.append((x,y))
            if any((hit:=tree.ray_cast(Vector((x,y,ray_top)),Vector((0,0,-1))))[0] is None
                   or abs(hit[0].z-z)>1e-4 for x,y in samples):continue
            surfaces.append({'id':f'surface_{len(surfaces):03d}','z':z,'polygon_xy':[list(p) for p in polygon],
                'area_m2':area(polygon),'evidence':'coplanar_upward_connected_triangles_convex_area_match_and_visibility_samples',
                'physical_load_verified':False})
    return surfaces[:128]


def _clip_polygon_rect(poly,x0,x1,y0,y1):
    result=list(poly)
    for axis,bound,keep_greater in ((0,x0,True),(0,x1,False),(1,y0,True),(1,y1,False)):
        if not result:break
        source,result=result,[]
        def inside(p):return p[axis]>=bound if keep_greater else p[axis]<=bound
        prev=source[-1];prev_in=inside(prev)
        for cur in source:
            cur_in=inside(cur)
            if cur_in!=prev_in:
                delta=cur[axis]-prev[axis]
                if abs(delta)>1e-16:
                    t=(bound-prev[axis])/delta
                    result.append((prev[0]+t*(cur[0]-prev[0]),prev[1]+t*(cur[1]-prev[1])))
            if cur_in:result.append(cur)
            prev,prev_in=cur,cur_in
    return result


def _polygons_cover_rect(polygons,x0,x1,y0,y1):
    """Check triangle-union coverage, including overlaps, by planar sweep.

    Between polygon vertices and edge intersections, the ordering of all
    edges is fixed. A vertical gap therefore varies linearly and cannot hide
    between midpoint checks. The numerical distance tolerance is 1 nanometre;
    neither summing triangle areas nor sampling rays is used to fill holes.
    """
    eps=1e-9
    if not polygons:return False
    edges=[];xs={x0,x1}
    for poly in polygons:
        xs.update(p[0] for p in poly)
        edges.extend((a,b) for a,b in zip(poly,poly[1:]+poly[:1]) if abs(a[0]-b[0])>1e-15)
    if len(edges)>256:return False  # adversarial tessellation: fail closed
    for i,(a,b) in enumerate(edges):
        rx,ry=b[0]-a[0],b[1]-a[1]
        for c,d in edges[i+1:]:
            if max(min(a[0],b[0]),min(c[0],d[0]))>=min(max(a[0],b[0]),max(c[0],d[0])):continue
            sx,sy=d[0]-c[0],d[1]-c[1];den=rx*sy-ry*sx
            if abs(den)<1e-18:continue
            qx,qy=c[0]-a[0],c[1]-a[1]
            t=(qx*sy-qy*sx)/den;u=(qx*ry-qy*rx)/den
            if 0<t<1 and 0<u<1:xs.add(a[0]+t*rx)
    ordered=sorted(xs)
    for left,right in zip(ordered,ordered[1:]):
        if right-left<1e-13:continue
        x=(left+right)/2;intervals=[]
        for poly in polygons:
            ys=[]
            for a,b in zip(poly,poly[1:]+poly[:1]):
                if min(a[0],b[0])<x<max(a[0],b[0]):
                    ys.append(a[1]+(x-a[0])*(b[1]-a[1])/(b[0]-a[0]))
            if len(ys)>=2:intervals.append((min(ys),max(ys)))
        high=y0
        for low,top in sorted(intervals):
            if low>high+eps:return False
            high=max(high,top)
        if high<y1-eps:return False
    return True


def _largest_mask_rectangle(rows):
    if not rows:return None
    nx=len(rows[0]);heights=[0]*nx;best=None;best_cells=0
    for iy,row in enumerate(rows):
        for ix,value in enumerate(row):heights[ix]=heights[ix]+1 if value else 0
        stack=[]
        for ix in range(nx+1):
            height=heights[ix] if ix<nx else 0;start=ix
            while stack and stack[-1][1]>height:
                start,h=stack.pop();cells=h*(ix-start)
                if cells>best_cells:best_cells=cells;best=(start,ix,iy-h+1,iy+1)
            if not stack or stack[-1][1]<height:stack.append((start,height))
    return best


def _covered_rectangle(parts,all_triangles):
    points=[point for item in parts for point in item['points']]
    minx,maxx=min(p.x for p in points),max(p.x for p in points)
    miny,maxy=min(p.y for p in points),max(p.y for p in points)
    minz,maxz=min(p.z for p in points),max(p.z for p in points)
    width,height=maxx-minx,maxy-miny
    if min(width,height)<.05 or maxz-minz>.001+1e-9:return None
    nx=max(1,round(64*width/max(width,height)))
    ny=max(1,round(64*height/max(width,height)))
    dx,dy=width/nx,height/ny
    cells={}
    for item in parts:
        poly=item['xy']
        ix0=max(0,int((min(p[0] for p in poly)-minx)/dx));ix1=min(nx-1,int((max(p[0] for p in poly)-minx)/dx))
        iy0=max(0,int((min(p[1] for p in poly)-miny)/dy));iy1=min(ny-1,int((max(p[1] for p in poly)-miny)/dy))
        for iy in range(iy0,iy1+1):
            for ix in range(ix0,ix1+1):
                clipped=_clip_polygon_rect(poly,minx+ix*dx,minx+(ix+1)*dx,miny+iy*dy,miny+(iy+1)*dy)
                if len(clipped)>=3 and area(clipped)>1e-16:cells.setdefault((ix,iy),[]).append(clipped)
    rows=[[False]*nx for _ in range(ny)]
    for (ix,iy),polygons in cells.items():
        # Ignore the outermost cell band as a physical boundary margin.
        if ix in (0,nx-1) or iy in (0,ny-1):continue
        rows[iy][ix]=_polygons_cover_rect(polygons,minx+ix*dx,minx+(ix+1)*dx,miny+iy*dy,miny+(iy+1)*dy)
    # Any projected part of a face above the admitted patch makes that entire
    # cell unavailable. Even a tiny overhang is excluded before choosing a
    # rectangle, instead of depending on visibility sample spacing.
    part_ids={item['id'] for item in parts}
    for item in all_triangles:
        if item['id'] in part_ids or item['maxz']<=maxz+1e-5:continue
        poly=item['xy']
        if max(p[0] for p in poly)<minx or min(p[0] for p in poly)>maxx or max(p[1] for p in poly)<miny or min(p[1] for p in poly)>maxy:continue
        ix0=max(0,int((min(p[0] for p in poly)-minx)/dx));ix1=min(nx-1,int((max(p[0] for p in poly)-minx)/dx))
        iy0=max(0,int((min(p[1] for p in poly)-miny)/dy));iy1=min(ny-1,int((max(p[1] for p in poly)-miny)/dy))
        for iy in range(iy0,iy1+1):
            for ix in range(ix0,ix1+1):
                if not rows[iy][ix]:continue
                clipped=_clip_polygon_rect(poly,minx+ix*dx,minx+(ix+1)*dx,miny+iy*dy,miny+(iy+1)*dy)
                if len(clipped)>=3 and area(clipped)>1e-16:rows[iy][ix]=False
    best=_largest_mask_rectangle(rows)
    if best is None:return None
    ix0,ix1,iy0,iy1=best
    rectangle=[[minx+ix0*dx,miny+iy0*dy],[minx+ix1*dx,miny+iy0*dy],
               [minx+ix1*dx,miny+iy1*dy],[minx+ix0*dx,miny+iy1*dy]]
    if min((ix1-ix0)*dx,(iy1-iy0)*dy)<.05 or area(rectangle)<.0025:return None
    return rectangle,minz,maxz


def _flat_slope(parts):
    # Area-weighted least squares prevents tessellation density from deciding
    # whether a broad tilted sheet counts as a horizontal support.
    centers=[(sum(p.x for p in t['points'])/3,sum(p.y for p in t['points'])/3,
              sum(p.z for p in t['points'])/3,area(t['xy'])) for t in parts]
    total=sum(p[3] for p in centers)
    if total<=0:return float('inf')
    means=[sum(p[i]*p[3] for p in centers)/total for i in range(3)]
    xx=yy=xy=xz=yz=0.
    for x,y,z,w in centers:
        x-=means[0];y-=means[1];z-=means[2]
        xx+=w*x*x;yy+=w*y*y;xy+=w*x*y;xz+=w*x*z;yz+=w*y*z
    den=xx*yy-xy*xy
    if abs(den)<1e-20:return float('inf')
    return math.hypot((xz*yy-yz*xy)/den,(yz*xx-xz*xy)/den)


def _fallback_support_surfaces(meshes,existing):
    """Measure a conservative rectangle on approximately horizontal mesh.

    All source triangles in a candidate are upward, edge-connected, contained
    in a 1 mm height band and have fitted slope <=0.003 (about 0.17 degrees).
    Cells require complete projected triangle-union coverage. Holes, overlaps
    and overhangs are handled geometrically rather than using a convex hull or
    bounding-box top. Numerical tolerances are explicit; this is not physics.
    """
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    all_vertices=[];all_faces=[];all_triangles=[];eligible=[]
    for obj_index,obj in enumerate(meshes):
        mesh=obj.data;mesh.calc_loop_triangles();offset=len(all_vertices)
        vertices=[obj.matrix_world@v.co for v in mesh.vertices]
        all_vertices.extend(vertices);all_faces.extend([offset+i for i in p.vertices] for p in mesh.polygons)
        for tri_index,tri in enumerate(mesh.loop_triangles):
            points=[vertices[i] for i in tri.vertices]
            normal=(points[1]-points[0]).cross(points[2]-points[0])
            item={'id':(obj_index,tri_index),'points':points,
                  'keys':[tuple(round(float(v),7) for v in p) for p in points],
                  'xy':[(float(p.x),float(p.y)) for p in points],
                  'minz':min(p.z for p in points),'maxz':max(p.z for p in points)}
            all_triangles.append(item)
            if normal.length<=1e-12 or normal.z/normal.length<=.01:continue
            if item['minz']<1e-4 or item['maxz']-item['minz']>.001:continue
            eligible.append(item)
    if not eligible:return []
    tree=BVHTree.FromPolygons(all_vertices,all_faces);ray_top=max(v.z for v in all_vertices)+1
    # Area-weighted height modes bound work on dense or adversarial assets.
    histogram={}
    for item in eligible:
        key=round(sum(p.z for p in item['points'])/3/.00025)
        histogram[key]=histogram.get(key,0)+area(item['xy'])
    selected_centers=[]
    for key,weight in sorted(histogram.items(),key=lambda pair:(-pair[1],pair[0])):
        center=key*.00025
        if weight<.0025 or any(abs(center-v)<.0005 for v in selected_centers):continue
        selected_centers.append(center)
        if len(selected_centers)>=16:break
    surfaces=[];seen=set()
    for center in selected_centers:
        candidates=[t for t in eligible if t['minz']>=center-.0005 and t['maxz']<=center+.0005]
        by_edge={};parents=list(range(len(candidates)))
        def find(i):
            while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
            return i
        for i,item in enumerate(candidates):
            for a,b in zip(item['keys'],item['keys'][1:]+item['keys'][:1]):
                by_edge.setdefault(tuple(sorted((a,b))),[]).append(i)
        for indices in by_edge.values():
            if len(indices)==2:parents[find(indices[0])]=find(indices[1])
        components={}
        for i,item in enumerate(candidates):components.setdefault(find(i),[]).append(item)
        for parts in components.values():
            identity=frozenset(t['id'] for t in parts)
            if identity in seen or len(parts)>20000:continue
            seen.add(identity)
            if sum(area(t['xy']) for t in parts)<.0025 or _flat_slope(parts)>.003:continue
            result=_covered_rectangle(parts,all_triangles)
            if result is None:continue
            rectangle,minz,maxz=result
            x0,x1=rectangle[0][0],rectangle[1][0];y0,y1=rectangle[0][1],rectangle[2][1]
            visible=True
            for iy in range(17):
                for ix in range(17):
                    x=x0+(x1-x0)*(ix+.5)/17;y=y0+(y1-y0)*(iy+.5)/17
                    hit,normal,index,distance=tree.ray_cast(Vector((x,y,ray_top)),Vector((0,0,-1)))
                    if hit is None or hit.z<minz-1e-5 or hit.z>maxz+1e-5:visible=False;break
                if not visible:break
            if not visible:continue
            if any(abs(surface['z']-maxz)<1e-5 and surface['area_m2']>=area(rectangle) for surface in existing):continue
            surfaces.append({'id':f'surface_{len(existing)+len(surfaces):03d}','z':float(maxz),
                'polygon_xy':rectangle,
                'area_m2':area(rectangle),
                'evidence':'near_planar_1mm_connected_upward_triangles_full_cell_union_coverage_and_visibility',
                'physical_load_verified':False})
            if len(existing)+len(surfaces)>=128:return surfaces
    return surfaces


def support_surfaces(meshes):
    exact=_exact_support_surfaces(meshes)
    return (exact+_fallback_support_surfaces(meshes,exact))[:128]


def compile_asset(model, metadata_path, output_dir, asset_id):
    import bpy
    from mathutils import Matrix,Vector
    metadata=validate_input(json.loads(Path(metadata_path).read_text()))
    source=inspect_glb(model)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(model))
    bpy.context.view_layer.update()
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    if not meshes:raise ValueError('GLB has no mesh')
    if any(o.modifiers for o in meshes):raise ValueError('modifiers must be applied before intake')
    if any(not o.data.uv_layers or not o.data.polygons for o in meshes):raise ValueError('every mesh requires faces and a UV layer')
    for obj in meshes:
        for polygon in obj.data.polygons:
            if polygon.material_index>=len(obj.material_slots) or obj.material_slots[polygon.material_index].material is None:raise ValueError('every visible face requires a material')
        if any(not math.isfinite(v) for vertex in obj.data.vertices for v in vertex.co):raise ValueError('nonfinite mesh coordinate')
    materials={slot.material for obj in meshes for slot in obj.material_slots if slot.material}
    for material in materials:
        if not material.use_nodes:raise ValueError('material nodes missing')
        for node in material.node_tree.nodes:
            if node.type=='TEX_IMAGE':
                image=node.image
                if image is None or min(image.size)<=0 or not len(image.pixels):raise ValueError('texture image has no readable pixels')
                if not image.packed_file:image.pack()
    coordinates=[o.matrix_world@v.co for o in meshes for v in o.data.vertices]
    low=Vector([min(v[i] for v in coordinates) for i in range(3)]);high=Vector([max(v[i] for v in coordinates) for i in range(3)])
    dims=high-low
    if min(dims)<=1e-8:raise ValueError('asset bounds must have positive extent in all three axes')
    factor=metadata['longest_dimension_m']/max(dims)
    anchor=Vector(((low.x+high.x)/2,(low.y+high.y)/2,low.z))
    normalization=Matrix.Scale(factor,4)@Matrix.Translation(-anchor)
    for obj in meshes:
        world=obj.matrix_world.copy();obj.parent=None
        obj.data=obj.data.copy();obj.data.transform(normalization@world)
        if world.determinant()<0:obj.data.flip_normals()
        obj.matrix_world=Matrix.Identity(4)
    for obj in list(bpy.context.scene.objects):
        if obj not in meshes:bpy.data.objects.remove(obj,do_unlink=True)
    bpy.context.view_layer.update()
    measured=[float(v*factor) for v in dims]
    supports=support_surfaces(meshes)
    bounds={'min':[-measured[0]/2,-measured[1]/2,0],'max':[measured[0]/2,measured[1]/2,measured[2]]}
    triangles=sum(len(p.vertices)-2 for o in meshes for p in o.data.polygons)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(output/'model.glb'),export_format='GLB',use_selection=True,export_yup=True,
        export_cameras=False,export_lights=False,export_animations=False)
    normalized_doc=inspect_glb(output/'model.glb')
    # Reimport to catch export loss, and measure the exact geometry the runtime
    # will see instead of trusting pre-export estimates.
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(output/'model.glb'))
    bpy.context.view_layer.update();reloaded=[o for o in bpy.context.scene.objects if o.type=='MESH']
    coords=[o.matrix_world@v.co for o in reloaded for v in o.data.vertices]
    measured=[max(v[i] for v in coords)-min(v[i] for v in coords) for i in range(3)]
    if not math.isclose(max(measured),metadata['longest_dimension_m'],rel_tol=1e-5,abs_tol=1e-6):raise ValueError('normalized GLB scale roundtrip failed')
    if len(source.get('materials',[]))!=len(normalized_doc.get('materials',[])):raise ValueError('material count changed during GLB normalization')
    digest=sha(output/'model.glb');variant_id=asset_id+'__native'
    asset={'asset_id':asset_id,'family_id':metadata['category'],'geometry_variant_id':asset_id+'__geometry',
        'hash':digest,'dimensions_import_units':measured,'meters_per_unit':1.0,'front_axis':'-Y','up_axis':'+Z',
        'status':'technical_ready','model_format':'glb','category':metadata['category'],
        'placement':'floor' if metadata['placement']=='floor' else 'table','scope':'single_prop',
        'anchor':{'type':'bottom_center','point':[0,0,0]},
        'placement_rules':{'can_rotate_z':True,'allow_uniform_scale':True,'allow_non_uniform_scale':False},
        'quality':{'quality_status':'technical_checked','visual_accepted':False,'geometry_completeness':'producer_declared_not_verified'},
        'scale_source':metadata['scale_source'],'support_surfaces':supports,'material_variant_ids':[variant_id]}
    material={'material_variant_id':variant_id,'compatible_geometry_ids':[asset['geometry_variant_id']],
        'mode':'embedded_gltf','model_sha256':digest,'normal_convention':'OpenGL_Y+','maps':{},'slot_materials':{}}
    receipt={'schema_version':RECEIPT_VERSION,'policy_version':POLICY_VERSION,'status':'technical_ready',
        'metadata':metadata,'asset':asset,'material':material,'blender_version':bpy.app.version_string,
        'checks':{'glb_embedded_standard_opaque':True,'mesh_uv_materials':True,'image_decode':True,'scale_roundtrip':True},
        'measurements':{'triangles':triangles,'mesh_count':len(reloaded),'bounds_m':bounds},
        'limitations':['scale and canonical pose are producer declarations','visual completeness not reviewed',
                        'support patches are geometry evidence, not load-bearing or stability certification'],
        'visual_accepted':False,'external_api_requests':0}
    atomic_json(output/'worker_receipt.json',receipt)


def main():
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--metadata',required=True,type=Path);parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--asset-id',required=True)
    options=parser.parse_args(args);compile_asset(options.input,options.metadata,options.output_dir,options.asset_id)

if __name__=='__main__':main()
