"""Real geometry check for automatic photo-derived retail detail."""
from pathlib import Path
import sys


def main():
    import bpy
    from mathutils import Vector
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from retail_refinement import apply_refinement
    bpy.ops.wm.read_factory_settings(use_empty=True)
    entries=[];objects=[]
    fixtures=[]
    for name,category,position,dims in [
        ('open_snack_shelf','shelf',[-2,3,0],[1.8,.7,1.7]),
        ('long_y_shelf','shelf',[2.8,4,0],[.55,6.5,2.1]),
        ('glazed_entrance','storefront_glazing',[0,.1,0],[3,.12,2.6]),
        ('supported_shelf','shelf',[0,3,0],[1.5,.7,1.8]),
        ('carton','cardboard_box',[0,3,1.8],[.3,.3,.2]),
        ('right_refrigerator_end_column','structural_column',[2,3,0],[.3,.3,2.9])]:
        root=bpy.data.objects.new(name,None);bpy.context.collection.objects.link(root)
        root.location=position
        bpy.ops.mesh.primitive_cube_add(size=1)
        cube=bpy.context.object;cube.parent=root;cube.location=(0,0,dims[2]/2);cube.scale=dims
        entries.append({'id':name,'position':position,'dimensions':dims,'rotation_z_deg':90 if name=='open_snack_shelf' else 0})
        objects.append({'id':name,'category':category,'xy':position[:2],
                        'support_id':'supported_shelf' if name=='carton' else 'floor'})
        if category=='shelf':
            fixtures.append({'object_id':name,'mode':'shelf','kind':'snack','levels':5,
                             'frame_color':[.65,.65,.65],'rail_color':[.03,.25,.06],
                             'back_panel':name=='long_y_shelf','endcaps':True,'stock_cell_m':.11,'top_height_m':.15})
    bundle={'scene_id':'automatic_test','shared_scene':{'base_scene':{
        'room':{'width':8,'depth':8,'height':3},'objects':entries}}}
    layout={'objects':objects,'room':bundle['shared_scene']['base_scene']['room'],
            'architecture':{'omit_surfaces':[],'front_wall':False},
            'retail_details':{'fixtures':fixtures,'structures':[],'floor_tiles':[]}}
    atlas=Path(__file__).resolve().parents[1]/'docs/weidian/demo/batch/packaging_print.png'
    config={'schema_version':1,'atlas':str(atlas),'defaults':{'fixture_defaults':{'top_stock':True}},'scenes':{}}
    report=apply_refinement(bundle,config,layout)
    assert report['photo_inferred_detail']
    assert report['stock_units']>100
    assert any(r['id']=='supported_shelf' and 'support retained' in r['reason'] for r in report['skipped'])
    assert bpy.data.objects.get('carton'), 'Explicit supported object was deleted'
    root=bpy.data.objects['open_snack_shelf'];inv=root.matrix_world.inverted()
    pts=[inv@o.matrix_world@Vector(c) for o in root.children_recursive if o.type=='MESH' for c in o.bound_box]
    assert max(p.z for p in pts)<=1.701 and min(p.z for p in pts)>=-.001
    column=bpy.data.objects['right_refrigerator_end_column']
    pts=[o.matrix_world@Vector(c) for o in column.children_recursive if o.type=='MESH' for c in o.bound_box]
    assert max(p.z for p in pts)<=2.901, 'Structural column acquired merchandise'
    assert all(not r['id'].endswith('column') for r in report['fixtures'])
    root=bpy.data.objects['long_y_shelf']
    pts=[o.matrix_world@Vector(c) for o in root.children_recursive if o.type=='MESH' for c in o.bound_box]
    assert min(p.x for p in pts)>=2.8-.55/2-.02 and max(p.x for p in pts)<=2.8+.55/2+.02
    assert min(p.y for p in pts)>=4-6.5/2-.02 and max(p.y for p in pts)<=4+6.5/2+.02
    assert min(p.z for p in pts)>=-.001 and max(p.z for p in pts)<=2.101
    panels=[]
    for obj in root.children_recursive:
        if obj.type!='MESH':continue
        for poly in obj.data.polygons:
            v=[obj.data.vertices[i].co for i in poly.vertices]
            if max(p.z for p in v)-min(p.z for p in v)>1 and max(p.x for p in v)-min(p.x for p in v)>3:
                panels.append(v)
    assert panels and all(abs(p.y)<.02 for face in panels for p in face), 'Gondola backing blocks an aisle face'
    root=bpy.data.objects['glazed_entrance']
    meshes=[o for o in root.children_recursive if o.type=='MESH']
    pts=[o.matrix_world@Vector(c) for o in meshes for c in o.bound_box]
    assert min(p.x for p in pts)>=-1.501 and max(p.x for p in pts)<=1.501
    assert min(p.y for p in pts)>=.039 and max(p.y for p in pts)<=.161
    assert min(p.z for p in pts)>=-.001 and max(p.z for p in pts)<=2.601
    assert len(meshes)==1 and meshes[0]['representation']=='automatic_photo_inferred_glazing'
    assert any(n.inputs['Transmission Weight'].default_value>.9
               for o in meshes for slot in o.material_slots for n in slot.material.node_tree.nodes
               if n.type=='BSDF_PRINCIPLED'), 'Physical glass material was not created'
    print('AUTOMATIC_PHOTO_DETAILS_TEST_PASSED',report['stock_units'])


if __name__=='__main__':main()
