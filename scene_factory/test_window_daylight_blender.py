"""Actual Cycles transport through the procedural window, not just node tests."""
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import bpy
    import numpy as np
    from mathutils import Vector
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class WindowDaylightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate_scene
        cls.g = generate_scene

    def setUp(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        self.g.MATERIALS.clear()
        self.g.MESHES.clear()
        self.scene = bpy.context.scene
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.device = 'CPU'
        self.scene.cycles.samples = 32
        self.scene.cycles.seed = 3
        self.scene.cycles.use_denoising = False
        self.scene.cycles.use_adaptive_sampling = False
        # Isolate direct irradiance: glossy/indirect interreflection is a
        # separate transport effect and would obscure the aperture comparison.
        self.scene.cycles.max_bounces = 0
        self.scene.render.resolution_x = self.scene.render.resolution_y = 32
        self.scene.render.resolution_percentage = 100
        self.scene.render.image_settings.file_format = 'OPEN_EXR'
        self.scene.render.image_settings.color_mode = 'RGB'
        self.scene.render.image_settings.color_depth = '32'
        self.scene.world = bpy.data.worlds.new('BlackTestWorld')
        self.scene.world.use_nodes = True
        background = next(node for node in self.scene.world.node_tree.nodes if node.type == 'BACKGROUND')
        background.inputs['Strength'].default_value = 0
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def window(self):
        return self.g.asset({'id': 'daylight_test_window', 'kind': 'window',
            'position': [0, 0, 0], 'dimensions': [2, .1, 2], 'color': [.8, .8, .8],
            'rotation_degrees': [0, 0, 0]})

    def transport_scene(self):
        root = self.window()
        panel = next(child for child in root.children if child.name.startswith('Panel'))
        target_root = self.g.root_object('Receiver')
        self.g.box(target_root, 'WhiteReceiver', (.45, -1, 1.45), (.4, .01, .4),
                   self.g.material('ReceiverWhite', (.8, .8, .8), 1), 0)
        light = self.g.add_light('OutsideDaylight', [.45, 1, 1.45], [.45, -1, 1.45],
                                 100, [1, 1, 1], .1)
        data = bpy.data.cameras.new('Meter')
        data.type = 'ORTHO'
        data.ortho_scale = .2
        camera = bpy.data.objects.new('Meter', data)
        bpy.context.collection.objects.link(camera)
        camera.location = (.45, -.1, 1.45)
        camera.rotation_euler = Vector((0, -1, 0)).to_track_quat('-Z', 'Y').to_euler()
        self.scene.camera = camera
        bpy.context.view_layer.update()
        return panel, light

    def radiance(self, name):
        path = str(Path(self.directory.name) / (name + '.exr'))
        self.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(path, check_existing=False)
        try:
            pixels = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(pixels)
            return float(pixels.reshape(-1, 4)[:, :3].mean())
        finally:
            bpy.data.images.remove(image)

    def test_real_exterior_light_passes_window_and_opaque_regression_does_not(self):
        panel, light = self.transport_scene()
        glass = panel.material_slots[0].material
        new_value = self.radiance('glazing')
        panel.hide_render = True
        open_value = self.radiance('open')
        panel.hide_render = False
        panel.material_slots[0].material = self.g.material('OldOpaqueWindow', (.36, .47, .55), .21, .15, .25)
        old_value = self.radiance('old_opaque')
        panel.material_slots[0].material = glass
        print('WINDOW_TRANSPORT', {'open': open_value, 'fixed': new_value, 'old': old_value,
                                   'fixed_ratio': new_value / open_value, 'old_ratio': old_value / open_value})
        self.assertGreater(open_value, .1)
        self.assertGreater(new_value / open_value, .75)
        self.assertLess(new_value / open_value, 1.05)
        self.assertLess(old_value / open_value, .1)
        light.data.energy = 0
        self.assertLess(self.radiance('no_source'), 1e-7)

    def test_window_is_clear_dielectric_without_emission_and_door_stays_opaque(self):
        root = self.window()
        panel = next(child for child in root.children if child.name.startswith('Panel'))
        mat = panel.material_slots[0].material
        shader = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        self.assertEqual(shader.inputs['Transmission Weight'].default_value, 1)
        self.assertEqual(shader.inputs['Metallic'].default_value, 0)
        self.assertEqual(shader.inputs['Emission Strength'].default_value, 0)
        self.assertIn('approximation', mat['glazing_transport'])
        door = self.g.asset({'id': 'door', 'kind': 'door', 'position': [3, 0, 0],
                            'dimensions': [1, .1, 2], 'color': [.6, .5, .4], 'rotation_degrees': [0, 0, 0]})
        door_panel = next(child for child in door.children if child.name.startswith('Panel'))
        door_shader = next(n for n in door_panel.material_slots[0].material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        self.assertEqual(door_shader.inputs['Transmission Weight'].default_value, 0)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WindowDaylightTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
