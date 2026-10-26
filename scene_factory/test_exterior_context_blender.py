"""Cycles verifies visible exterior and unchanged incident daylight separately."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import bpy
    import numpy as np
    from mathutils import Vector
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class ExteriorContextBlenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate_scene as g
        import exterior_context
        cls.g, cls.exterior = g, exterior_context

    def setUp(self):
        bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
        self.g.MATERIALS.clear(); self.g.MESHES.clear()
        self.scene = bpy.context.scene
        self.scene.render.engine = 'CYCLES'; self.scene.cycles.device = 'CPU'
        self.scene.cycles.samples = 32; self.scene.cycles.seed = 7
        self.scene.cycles.use_denoising = False; self.scene.cycles.use_adaptive_sampling = False
        self.scene.cycles.max_bounces = 8; self.scene.cycles.diffuse_bounces = 3
        self.scene.render.resolution_x = self.scene.render.resolution_y = 32
        self.scene.render.resolution_percentage = 100
        self.scene.render.image_settings.file_format = 'OPEN_EXR'
        self.scene.render.image_settings.color_mode = 'RGB'; self.scene.render.image_settings.color_depth = '32'
        self.scene.world = bpy.data.worlds.new('BlackExteriorTestWorld'); self.scene.world.use_nodes = True
        background = next(node for node in self.scene.world.node_tree.nodes if node.type == 'BACKGROUND')
        background.inputs['Strength'].default_value = 0
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.entry = {'id': 'window', 'kind': 'window', 'position': [0, 4, 0],
                      'dimensions': [2, .1, 2], 'color': [.8] * 3, 'rotation_z_deg': 0}
        self.base = {'room': {'width': 4, 'depth': 4, 'height': 3}, 'objects': [self.entry]}
        self.g.asset(self.entry)
        data = bpy.data.cameras.new('Meter'); data.type = 'ORTHO'; data.ortho_scale = .2
        self.camera = bpy.data.objects.new('Meter', data); bpy.context.collection.objects.link(self.camera)
        self.camera.location = (.45, 3, 1.45)
        self.camera.rotation_euler = Vector((0, 1, 0)).to_track_quat('-Z', 'Y').to_euler()
        self.scene.camera = self.camera
        image = bpy.data.images.new('ExteriorCrop', width=2, height=2)
        image.pixels[:] = [.5, .15, .05, 1] * 4
        self.images = {'outside': image}

    def build(self, **changes):
        context = {'window_id': 'window', 'texture_id': 'outside', 'depth_m': .3, 'brightness_ev': 1}
        context.update(changes)
        return self.exterior.build_exterior_contexts(self.base, [context], self.images, [[.45, 3, 1.45]])

    def radiance(self, name):
        path = str(Path(self.temp.name) / (name + '.exr')); self.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(path, check_existing=False)
        pixels = np.empty(len(image.pixels), dtype=np.float32); image.pixels.foreach_get(pixels)
        bpy.data.images.remove(image)
        return float(pixels.reshape(-1, 4)[:, :3].mean())

    def test_exterior_is_visible_through_glazing_without_an_illumination_source(self):
        before = self.radiance('dark_window')
        report = self.build()
        self.assertEqual(report['status'], 'passed')
        visible = self.radiance('visible_exterior')
        self.assertLess(before, .0001)
        self.assertGreater(visible, .1)
        obj = bpy.data.objects[report['contexts'][0]['object_name']]
        self.assertFalse(obj.visible_shadow); self.assertFalse(obj.visible_diffuse)
        print('EXTERIOR_VISIBLE', {'before': before, 'with_context': visible})

    def test_context_neither_blocks_daylight_nor_adds_diffuse_radiance(self):
        root = self.g.root_object('Receiver')
        receiver = self.g.material('ReceiverWhite', (.8,) * 3, 1)
        # The context intentionally appears in glossy reflections, so use a
        # purely diffuse radiometer to measure illumination rather than that
        # permitted visible reflection.
        nodes, links = receiver.node_tree.nodes, receiver.node_tree.links
        diffuse = nodes.new('ShaderNodeBsdfDiffuse'); diffuse.inputs['Color'].default_value = (.8, .8, .8, 1)
        output = next(n for n in nodes if n.type == 'OUTPUT_MATERIAL')
        links.new(diffuse.outputs[0], output.inputs['Surface'])
        self.g.box(root, 'WhiteReceiver', (.45, 2.5, 1.45), (.4, .01, .4), receiver, 0)
        light = self.g.add_light('Daylight', [.45, 5, 1.45], [.45, 2.5, 1.45], 100, [1, 1, 1], .1)
        self.camera.rotation_euler = Vector((0, -1, 0)).to_track_quat('-Z', 'Y').to_euler()
        before = self.radiance('receiver_before')
        report = self.build(brightness_ev=6)
        self.assertEqual(report['status'], 'passed')
        after = self.radiance('receiver_context')
        self.assertGreater(before, .1)
        self.assertAlmostEqual(after / before, 1, delta=.01)
        light.data.energy = 0
        self.assertLess(self.radiance('receiver_no_daylight'), 1e-6)
        print('EXTERIOR_TRANSPORT', {'without_context': before, 'with_context': after, 'ratio': after / before})

    def test_interior_or_inward_window_attachment_rejected(self):
        self.entry['position'][1] = 2
        self.assertEqual(self.build()['errors'][0]['code'], 'exterior_context_invalid_window_attachment')
        self.entry['position'][1] = 4; self.entry['rotation_z_deg'] = 180
        self.assertEqual(self.build()['errors'][0]['code'], 'exterior_context_invalid_window_attachment')

    def test_multiview_production_build_shares_one_context_and_records_it(self):
        import generate_multiview
        from test_multiview import fixture
        bundle = fixture(); spec = bundle['shared_scene']
        spec['base_scene']['room'].update(width=4, depth=4, height=3)
        entry = spec['base_scene']['objects'][0]
        entry.update(id='shared_window', category='window', position=[0, 4, 0],
                     dimensions=[2, .1, 2], rotation_z_deg=0, repeat=[1, 1], spacing=[0, 0])
        spec['base_scene']['objects'] = [entry]
        spec['materials'] = []; spec['custom_objects'] = []
        spec['surface_overrides'] = dict.fromkeys(spec['surface_overrides'], '')
        spec['textures'] = [{'id': 'outside', 'quad': [[.1, .1], [.5, .1], [.5, .5], [.1, .5]]}]
        spec['exterior_contexts'] = [{'window_id': 'shared_window', 'texture_id': 'outside', 'depth_m': .3, 'brightness_ev': 1}]
        bundle['texture_sources'] = []
        bundle['architecture']['omit_surfaces'] = ['back']
        for i, view in enumerate(bundle['views']):
            view['camera']['position'] = [i * .5, 1, 1.5]
            view['camera']['target'] = [0, 4, 1]
        image = self.images['outside']
        image.filepath_raw = str(Path(self.temp.name) / 'outside.png'); image.file_format = 'PNG'; image.save()
        manifest = Path(self.temp.name) / 'manifest.json'
        manifest.write_text(json.dumps({'outside': image.filepath_raw}))
        handles = generate_multiview.build_scene(bundle, texture_manifest=manifest)
        self.assertEqual(handles['exterior_context']['status'], 'passed')
        self.assertEqual(len(handles['exterior_context']['contexts']), 1)
        report = generate_multiview.geometry_preflight(bundle, handles)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['exterior_context']['contexts'][0]['window_id'], 'shared_window')
        objects = [obj for obj in handles['scene'].objects if obj.get('exterior_context_window_id')]
        self.assertEqual(len(objects), 1)
        identity = objects[0].as_pointer()
        for view in bundle['views']:
            generate_multiview.set_view(handles['scene'], handles['cameras'][view['id']], view, 64)
            self.assertEqual(objects[0].as_pointer(), identity)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExteriorContextBlenderTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
