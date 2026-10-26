"""Blender regression checks for V2 geometry, photo materials and physical metadata.

Run: blender -b --factory-startup --python-exit-code 1 --python test_v2_renderer.py
Normal Python discovery skips these checks when Blender is unavailable.
"""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import bpy
    import bmesh
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class RendererV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate_v2
        import v2_materials
        from spec_v2 import part, material
        cls.renderer = generate_v2
        cls.materials = v2_materials
        cls.part = staticmethod(part)
        cls.material = staticmethod(material)

    def setUp(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        self.base = {'room': {'width': 6, 'depth': 8, 'height': 3,
                             'wall_color': [.6] * 3, 'floor_color': [.2] * 3, 'ceiling_color': [.8] * 3},
                     'lighting': {'energy': 600, 'color': [1, .9, .8]},
                     'camera': {'position': [0, .3, 1.5], 'target': [0, 4, 1], 'lens_mm': 24},
                     'objects': [], 'assumptions': []}

    def group(self, identifier, parts, **values):
        result = {'id': identifier, 'position': [0, 2, 0], 'rotation_degrees': [0, 0, 0],
                  'material_id': '', 'observed': True, 'confidence': .7, 'support_id': '', 'parts': parts}
        result.update(values)
        return result

    def test_photo_crop_material_uv_front_and_packing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = bpy.data.images.new('CropTestSource', width=2, height=2, alpha=True)
            source.pixels[:] = [1,0,0,1, 0,1,0,1, 0,0,1,1, 1,1,1,1]
            source.filepath_raw = str(directory / 'crop.png')
            source.file_format = 'PNG'
            source.save()
            manifest = directory / 'manifest.json'
            manifest.write_text(json.dumps({'textures': [{'id': 'sign', 'path': 'crop.png'}]}))
            images = self.materials.load_texture_manifest(manifest)
            self.assertIsNotNone(images['sign'].packed_file)
            materials = self.materials.build_materials([
                self.material('photo', [1, 1, 1], texture_id='sign', roughness=.33)], images)
            root = self.renderer.build_custom(self.group('photo', [self.part('image_plane', size=[2, 0, 1])], material_id='photo'), materials)
            plane = root.children[0]
            self.assertEqual([tuple(loop.uv) for loop in plane.data.uv_layers.active.data], [(0,0),(1,0),(1,1),(0,1)])
            self.assertLess(plane.data.polygons[0].normal.y, -.999)
            texture = next(node for node in materials['photo'].node_tree.nodes if node.type == 'TEX_IMAGE')
            self.assertEqual(texture.image, images['sign'])
            self.assertTrue(any(link.from_node == texture and link.to_socket.name == 'Base Color'
                                for link in materials['photo'].node_tree.links))
            with self.assertRaises(ValueError):
                self.materials.build_materials([self.material('missing', [1,1,1], texture_id='absent')], images)

    def test_curve_bounds_include_bevel_and_parent_transform(self):
        root = self.renderer.build_custom(self.group('wire', [self.part('curve', radius=.1,
            points=[[-1, 0, 0], [1, 0, 0]])], position=[1,2,3], rotation_degrees=[0,0,90]), {})
        bpy.context.view_layer.update()
        bounds = self.renderer.evaluated_bounds(root)
        # Blender's bevel cross-section is a finite polygon, not an analytic disk.
        for actual, expected in zip(bounds['min'], [.9, 1, 2.9]):
            self.assertAlmostEqual(actual, expected, delta=.006)
        for actual, expected in zip(bounds['max'], [1.1, 3, 3.1]):
            self.assertAlmostEqual(actual, expected, delta=.006)

    def test_light_replace_add_and_empty_modes(self):
        light = {'id': 'local', 'type': 'AREA', 'position': [0,2,2.5], 'target': [0,2,0],
                 'color': [1,.9,.8], 'energy': 25, 'size': .5, 'size_y': .2,
                 'shape': 'RECTANGLE', 'spread_degrees': 80}
        for replace, explicit, count in [(False, [], 4), (False, [light], 5),
                                          (True, [light], 1), (True, [], 0)]:
            with self.subTest(replace=replace, explicit=bool(explicit)):
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete(use_global=False)
                self.renderer.add_lights(bpy.context.scene,
                    {'lights': explicit, 'render': {'replace_base_lights': replace}}, self.base)
                lights = [obj for obj in bpy.context.scene.objects if obj.type == 'LIGHT']
                self.assertEqual(len(lights), count)
                if explicit:
                    created = next(obj for obj in lights if obj.name.startswith('local'))
                    self.assertAlmostEqual(created.data.size_y, .2)
                    self.assertAlmostEqual(created.data.spread, math.radians(80), places=5)

    def test_support_uses_local_shelf_surface_and_detects_floating_curve(self):
        scene = bpy.context.scene
        camera_data = bpy.data.cameras.new('TestCamera')
        camera = bpy.data.objects.new('TestCamera', camera_data)
        bpy.context.collection.objects.link(camera)
        camera.location = (0, .3, 1.5)
        scene.camera = camera
        shelf = self.group('shelf', [
            self.part('box', size=[1,1,.1], position=[0,0,.75]),
            self.part('box', size=[.1,.1,1.7], position=[.45,.45,.85])])
        wire = self.part('curve', radius=.01, points=[[-.3,0,0],[.3,0,0]])
        contact = self.group('contact', [wire], position=[0,2,.81], support_id='shelf')
        floating = self.group('floating', [wire], position=[0,2,1.01], support_id='shelf', observed=False)
        groups = [shelf, contact, floating]
        roots = [self.renderer.build_custom(group, {}) for group in groups]
        bpy.context.view_layer.update()
        spec = {'scene_type': 'other', 'image': {'width':800,'height':600,'source_id':''}, 'custom_objects': groups}
        report = self.renderer.make_annotations(scene, spec, self.base, [], roots)
        result = {item['id']: item for item in report['objects']}
        self.assertEqual(result['contact']['support_check']['status'], 'approximate_contact')
        self.assertAlmostEqual(result['contact']['support_check']['support_top_z_m'], .8, places=4)
        self.assertAlmostEqual(result['contact']['support_check']['support_aabb_top_z_m'], 1.7, places=4)
        self.assertEqual(result['floating']['support_check']['status'], 'needs_review')
        self.assertAlmostEqual(result['floating']['support_check']['bottom_gap_m'], .2, delta=.001)
        self.assertEqual(result['floating']['evidence_status'], 'inferred_completion')
        self.assertTrue(any(warning['type'] == 'support_contact_mismatch' for warning in report['warnings']))

    def test_closed_mesh_normals_repaired_open_winding_preserved(self):
        closed = self.part('mesh', vertices=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                           faces=[[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
        open_plane = self.part('mesh', vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[2,1,0]])
        root = self.renderer.build_custom(self.group('normals', [closed, open_plane]), {})
        tetra, triangle = sorted(root.children, key=lambda obj: obj.name)
        bm = bmesh.new()
        bm.from_mesh(tetra.data)
        self.assertGreater(bm.calc_volume(signed=True), 0)
        bm.free()
        self.assertLess(triangle.data.polygons[0].normal.z, -.999)

    def test_room_overrides_do_not_bleed_through_shared_meshes(self):
        mats = self.materials.build_materials([
            self.material('left_red', [.8,.1,.1]), self.material('right_blue', [.1,.1,.8])], {})
        room = self.renderer.g.build_room(self.base['room'])
        self.renderer.override_room(room, {'floor':'','ceiling':'','left':'left_red','right':'right_blue','back':''}, mats)
        sides = [obj for obj in room.children if obj.name.startswith('SideWall')]
        left, right = sorted(sides, key=lambda obj: obj.location.x)
        self.assertEqual(left.material_slots[0].material, mats['left_red'])
        self.assertEqual(right.material_slots[0].material, mats['right_blue'])
        self.assertIsNotNone(left.data.uv_layers.active)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RendererV2Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)
