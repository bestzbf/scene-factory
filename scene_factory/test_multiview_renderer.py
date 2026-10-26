"""Blender-only integration checks for shared scenes and real wall omissions."""
import sys
import unittest
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class MultiviewRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate_multiview
        from test_multiview import fixture
        cls.renderer = generate_multiview
        cls.fixture = staticmethod(fixture)

    def test_two_cameras_reuse_identical_geometry_and_real_opening(self):
        bundle = self.fixture()
        handles = self.renderer.build_scene(bundle)
        scene = handles['scene']
        cameras = handles['cameras']
        self.assertEqual(set(cameras), {'a', 'b'})
        self.assertEqual(cameras['a'].name, 'ReferenceCamera_a')
        self.assertEqual(cameras['b'].name, 'ReferenceCamera_b')
        self.assertNotEqual(tuple(cameras['a'].location), tuple(cameras['b'].location))
        original = {obj.as_pointer() for obj in scene.objects if obj.type in {'MESH', 'CURVE'}}
        self.renderer.set_view(scene, cameras['a'], bundle['views'][0], 800)
        self.assertEqual((scene.render.resolution_x, scene.render.resolution_y), (800, 600))
        self.renderer.set_view(scene, cameras['b'], bundle['views'][1], 800)
        self.assertEqual((scene.render.resolution_x, scene.render.resolution_y), (600, 800))
        self.assertEqual(original, {obj.as_pointer() for obj in scene.objects if obj.type in {'MESH', 'CURVE'}})
        room_surfaces = [obj.get('surface_id') for obj in handles['room'].children]
        self.assertNotIn('back', room_surfaces)
        self.assertIn('front', room_surfaces)
        self.assertIn('ceiling', room_surfaces)
        self.assertFalse(any(obj.name.startswith('EndWall') for obj in handles['room'].children))

    def test_preview_settings_use_real_lights_world_and_denoise(self):
        bundle = self.fixture()
        handles = self.renderer.build_scene(bundle)
        self.renderer.configure_render(handles['scene'], bundle['shared_scene'], 64, 'CPU')
        self.assertTrue(handles['scene'].cycles.use_preview_denoising)
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    self.assertTrue(area.spaces.active.shading.use_scene_lights)
                    self.assertTrue(area.spaces.active.shading.use_scene_world)

    def test_exterior_daylight_proxy_does_not_occlude_view_through_glass(self):
        from make_v2_examples import area
        bundle=self.fixture()
        depth=bundle['shared_scene']['base_scene']['room']['depth']
        bundle['shared_scene']['lights']=[area('OutsideProxy',[0,depth+.3,2],[0,1,1],200),area('InsideFixture',[0,2,2],[0,2,0],100)]
        bundle['shared_scene']['render']['replace_base_lights']=True
        handles=self.renderer.build_scene(bundle)
        self.assertFalse(bpy.data.objects['OutsideProxy'].visible_transmission)
        self.assertTrue(bpy.data.objects['InsideFixture'].visible_transmission)
        self.assertEqual(bpy.data.objects['OutsideProxy'].data.energy,200)
        self.assertEqual(handles['lighting']['exterior_proxy_transmission_hidden'],['OutsideProxy'])

    def test_fixture_appearance_requires_matching_geometry_and_no_double_light(self):
        from appearance_lighting import add_fixture_faces
        from make_v2_examples import area
        bundle = self.fixture()
        bundle['shared_scene']['lights'] = [area('Backed',[0,2,2],[0,2,0],100),
                                            area('Unbacked',[1.5,2,2],[1.5,2,0],100)]
        bundle['shared_scene']['render']['replace_base_lights'] = True
        handles = self.renderer.build_scene(bundle, appearance_policy='legacy')
        g = self.renderer.v2.g
        root = g.root_object('fixture')
        panel = g.box(root, 'diffuser', (0,2,2.025), (.55,.55,.02), g.material('white',(.8,.8,.8)), 0)
        lamp = bpy.data.objects['Backed']
        lamp.data.size = lamp.data.size_y = .5
        original_energy = lamp.data.energy
        records = add_fixture_faces(handles['scene'])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['backing'], panel.name)
        face = bpy.data.objects[records[0]['face']]
        self.assertTrue(face.visible_camera)
        self.assertFalse(face.visible_diffuse)
        self.assertFalse(face.visible_shadow)
        self.assertEqual(lamp.data.energy, original_energy)
        self.assertFalse(records[0]['additional_diffuse_emission'])

    def test_ceiling_and_exterior_proxy_are_not_visible_fixtures(self):
        from appearance_lighting import add_fixture_faces
        from make_v2_examples import area
        bundle = self.fixture()
        h = bundle['shared_scene']['base_scene']['room']['height']
        bundle['shared_scene']['lights'] = [area('NearCeiling',[0,2,h-.03],[0,2,0],100)]
        bundle['shared_scene']['render']['replace_base_lights'] = True
        handles = self.renderer.build_scene(bundle, appearance_policy='legacy')
        self.assertEqual(add_fixture_faces(handles['scene']), [])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MultiviewRendererTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
