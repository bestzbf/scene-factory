"""Compare fitting projections with Blender's independent camera implementation."""
from pathlib import Path
import sys


def main():
    import bpy
    import numpy as np
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from camera_alignment import project,parameters
    from generate_multiview import add_camera
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    points=[[-1,2,.2],[1,2,.2],[-1,2,1.6],[1,2,1.6],[-1.2,4,.1],[1.2,4,1.8]]
    for width,height in [(1280,720),(720,1280)]:
        scene.render.resolution_x=width;scene.render.resolution_y=height
        scene.render.resolution_percentage=100
        for roll in (-8,0,7):
            values={'position':[.2,.6,2.5],'target':[0,4,1.0],'lens_mm':24,'roll_degrees':roll}
            camera=add_camera({'id':f'test_{width}_{roll}','camera':values,'source_image':'test'})
            bpy.context.view_layer.update()
            uv=[world_to_camera_view(scene,camera,Vector(p)) for p in points]
            actual=np.array([[p.x*width,(1-p.y)*height] for p in uv])
            predicted=project(points,parameters(values),width,height)[0]
            error=float(np.max(np.abs(actual-predicted)))
            assert error<.002,(width,height,roll,error)
    print('CAMERA_PROJECTION_BLENDER_TEST_PASSED')


if __name__=='__main__':main()
