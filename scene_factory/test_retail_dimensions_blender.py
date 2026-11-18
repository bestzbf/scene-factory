"""Run with Blender --background --python; checks local axes and support metadata."""
import copy
import math
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retail_refinement import apply_refinement


def main():
    if bpy is None:
        raise RuntimeError('Run this integration check with Blender Python')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    from retail_stock_geometry import Mesh
    cup = Mesh('cup_bounds_test')
    mat = bpy.data.materials.new('test')
    cup.cup(0, 0, 0, .06, .14, mat, mat)
    assert cup.products == 1
    assert all(abs(x) <= .060001 and abs(y) <= .060001 and 0 <= z <= .140001
               for x, y, z in cup.vertices), 'Cup escaped its allocation bounds'
    root = bpy.data.objects.new('rotated_fixture', None)
    bpy.context.collection.objects.link(root)
    root.location = (0, 3, 0)
    root.rotation_euler.z = math.pi / 2
    bpy.ops.mesh.primitive_cube_add(size=1)
    mesh = bpy.context.object
    mesh.parent = root
    mesh.location = (0, 0, .5)
    mesh.scale = (2, .4, 1)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 3.5, 1.1))
    stock = bpy.context.object
    stock.name = 'supported_stock'
    stock.scale = (.1, .1, .2)
    bpy.context.view_layer.update()
    bundle = {'scene_id': 'test', 'shared_scene': {'base_scene': {
        'room': {'width': 8, 'depth': 8, 'height': 4},
        'objects': [
            {'id': root.name, 'position': [0, 3, 0], 'dimensions': [2, .4, 1]},
            {'id': stock.name, 'position': [0, 3.5, 1.1], 'dimensions': [.1, .1, .2]},
        ]}}}
    layout = {'objects': [
        {'id': root.name, 'category': 'shelf', 'xy': [0, 3]},
        {'id': stock.name, 'category': 'stock', 'xy': [0, 3.5], 'support_id': root.name},
    ]}
    config = {'schema_version': 1, 'atlas': str(Path(__file__).resolve().parents[1] /
        'docs/weidian/demo/batch/packaging_print.png'), 'defaults': {
        'fixtures': {root.name: {'enabled': False}},
        'fixture_dimensions': [{'id': root.name, 'dimensions_m': [3, .6, 2]}]}, 'scenes': {}}
    original = copy.deepcopy(config)
    report = apply_refinement(bundle, config, layout)
    assert config == original, 'validation must not mutate configuration'
    world = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
    extent = [max(v[j] for v in world) - min(v[j] for v in world) for j in range(3)]
    assert all(abs(a-b) < 1e-5 for a,b in zip(extent, [.6, 3, 2])), extent
    assert abs(stock.location.y-3.75) < 1e-5 and abs(stock.location.z-2.2) < 1e-5
    entry = bundle['shared_scene']['base_scene']['objects'][1]
    assert all(abs(a-b)<1e-5 for a,b in zip(entry['dimensions'], [.15,.15,.4]))
    assert abs(layout['objects'][1]['xy'][1]-3.75) < 1e-5
    assert report['layout_clearance']['status'] == 'passed'
    print('RETAIL_DIMENSIONS_TEST_PASSED')


if __name__ == '__main__':
    main()
