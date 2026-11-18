"""Blender integration check for replacement transform and deterministic mesh."""
import hashlib
import math
from pathlib import Path
import sys


def main():
    import bpy
    from mathutils import Vector
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from reviewed_retail_fixtures import replace_shelf,replace_fridge
    atlas=Path(__file__).resolve().parents[1]/'docs/weidian/demo/batch/packaging_print.png'
    root=bpy.data.objects.new('FixtureUnderTest',None);bpy.context.collection.objects.link(root)
    spec={'position':[.3,4.,0],'dimensions_m':[1.8,.78,1.48],'rotation_z_deg':90,'kind':'snack'}
    def digest():
        return hashlib.sha256(str([(tuple(v.co)) for o in root.children if o.type=='MESH' for v in o.data.vertices]).encode()).hexdigest()
    replace_shelf(root,spec,atlas,'test');a=digest()
    replace_shelf(root,spec,atlas,'test');assert a==digest(),'Replacement must be deterministic'
    # Replacement must erase a reflected object's old X/Y Euler components.
    root.rotation_euler=(math.pi,.4,0);root.scale=(-1,-1,-1)
    replace_fridge(root,{'position':[0,2,0],'dimensions_m':[.75,.7,2.05],'rotation_z_deg':90},atlas,'test')
    bpy.context.view_layer.update()
    assert abs(root.rotation_euler.x)<1e-6 and abs(root.rotation_euler.y)<1e-6
    assert root.matrix_world.determinant()>0
    points=[root.matrix_world.inverted()@o.matrix_world@Vector(c) for o in root.children for c in o.bound_box]
    assert min(v.z for v in points)>=-.001 and max(v.z for v in points)<=2.051
    assert all(math.isfinite(x) for p in points for x in p)
    print('PASS: deterministic shelves, replacement transform reset, upright bounded fridge')


if __name__=='__main__':main()
