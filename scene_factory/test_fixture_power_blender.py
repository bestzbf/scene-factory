"""Real Blender regression for power-bound, appearance-only fixture faces."""
import math
from pathlib import Path
import sys
import unittest

try:
    import bpy
    import numpy as np
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class FixturePowerBlenderTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_reference_calibration_blender import ReferenceCalibrationBlenderTests
        ReferenceCalibrationBlenderTests.setUpClass()
        self.fixture = ReferenceCalibrationBlenderTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.scene = self.fixture.scene
        self.scene.view_settings.exposure = -6
        self.scene.cycles.samples = 16
        self.scene.render.resolution_x = self.scene.render.resolution_y = 48

    def lamp(self, name='FixturePower', x=0, data=None):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0, 2))
        bpy.context.object.scale = (.8, .8, .05)
        if data is None:
            data = bpy.data.lights.new(name, 'AREA')
            data.shape, data.size, data.energy = 'SQUARE', .8, 100.
        lamp = bpy.data.objects.new(name, data)
        self.scene.collection.objects.link(lamp)
        lamp.location = (x, 0, 1.96)
        return lamp

    def faces(self):
        from appearance_lighting import add_fixture_faces
        records = add_fixture_faces(self.scene)
        return [bpy.data.objects[record['face']] for record in records]

    def strength(self, face):
        bpy.context.view_layer.update()
        emission = next(node for node in face.material_slots[0].material.node_tree.nodes
                        if node.type == 'EMISSION')
        return emission.inputs['Strength'].default_value

    def test_nearby_lamps_match_physical_backing_before_creating_faces(self):
        from appearance_lighting import add_fixture_faces
        first = self.lamp('FirstBacked')
        backing = bpy.context.object
        second = bpy.data.objects.new('SecondBacked', first.data.copy())
        self.scene.collection.objects.link(second)
        second.location = (0, 0, 1.92)
        powers = {lamp.name: lamp.data.energy for lamp in (first, second)}
        records = add_fixture_faces(self.scene)
        self.assertEqual(len(records), 2)
        self.assertEqual({record['backing'] for record in records}, {backing.name})
        self.assertEqual(powers, {lamp.name: lamp.data.energy for lamp in (first, second)})
        self.assertEqual(len([obj for obj in self.scene.objects if obj.type == 'LIGHT']), 2)
        for record in records:
            face = self.scene.objects[record['face']]
            self.assertFalse(face.visible_diffuse)
            self.assertFalse(face.visible_shadow)
            self.assertFalse(record['additional_diffuse_emission'])

    def test_repeated_execution_reuses_faces_and_retains_renamed_source_binding(self):
        from appearance_lighting import add_fixture_faces
        lamp = self.lamp()
        face, = self.faces()
        lamp.name = 'RenamedIdempotentFixture'
        original = (face.as_pointer(), face.data.as_pointer(),
                    face.material_slots[0].material.as_pointer())
        for _ in range(3):
            record, = add_fixture_faces(self.scene)
            self.assertEqual(record['status'], 'reused')
            self.assertEqual(record['light'], lamp.name)
            self.assertEqual(record['face'], face.name)
        self.assertEqual(original, (face.as_pointer(), face.data.as_pointer(),
                                   face.material_slots[0].material.as_pointer()))
        self.assertEqual(len([obj for obj in self.scene.objects
                              if obj.get('fixture_appearance_only')]), 1)

    def test_new_lamp_sees_physical_backing_through_existing_generated_face(self):
        from appearance_lighting import add_fixture_faces
        first = self.lamp('ExistingBacked')
        backing = bpy.context.object
        old_face, = self.faces()
        second = bpy.data.objects.new('NewBacked', first.data.copy())
        self.scene.collection.objects.link(second)
        second.location = (0, 0, 1.92)
        records = add_fixture_faces(self.scene)
        self.assertEqual(len(records), 2)
        created = next(record for record in records if record['light'] == second.name)
        self.assertNotEqual(created['face'], old_face.name)
        self.assertEqual(created['backing'], backing.name)
        self.assertFalse(self.scene.objects[created['backing']].get('fixture_appearance_only'))
        self.assertEqual(len([obj for obj in self.scene.objects
                              if obj.get('fixture_appearance_only')]), 2)

    def test_existing_appearance_without_physical_backing_cannot_support_new_face(self):
        from appearance_lighting import add_fixture_faces
        first = self.lamp('RemovedBacking')
        backing = bpy.context.object
        face, = self.faces()
        bpy.data.objects.remove(backing, do_unlink=True)
        second = bpy.data.objects.new('UnbackedNew', first.data.copy())
        self.scene.collection.objects.link(second)
        second.location = (0, 0, 1.92)
        records = add_fixture_faces(self.scene)
        self.assertEqual([record['face'] for record in records], [face.name])
        self.assertEqual(records[0]['status'], 'reused')
        self.assertEqual(len([obj for obj in self.scene.objects
                              if obj.get('fixture_appearance_only')]), 1)

    def test_edited_existing_face_is_not_replaced_or_duplicated(self):
        from appearance_lighting import add_fixture_faces
        self.lamp()
        face, = self.faces()
        # User-edited appearance graphs are deliberately left intact by the
        # power binder. They still claim this lamp's existing appearance.
        tree = face.material_slots[0].material.node_tree
        tree.nodes.remove(next(node for node in tree.nodes if node.type == 'MIX_SHADER'))
        count = len(tree.nodes)
        record, = add_fixture_faces(self.scene)
        self.assertEqual(record['status'], 'reused')
        self.assertEqual(record['face'], face.name)
        self.assertEqual(len(tree.nodes), count)
        self.assertEqual(len([obj for obj in self.scene.objects
                              if obj.get('fixture_appearance_only')]), 1)

    def test_power_changes_are_visible_in_real_camera_render(self):
        from lighting_fit import collect_lighting_controls, apply_lighting_controls
        lamp = self.lamp()
        face, = self.faces()
        self.scene.camera = self.fixture.camera('FixtureView', (0, 0, 1), (0, 0, 2))
        self.scene.camera.data.ortho_scale = 1.2
        controls, _ = collect_lighting_controls(self.scene)
        rendered = {}
        for factor in (1., .25, 4., 0., 1.):
            # A disabled fixture is supported outside the bounded fitter.
            if factor:
                apply_lighting_controls(controls, {key: factor for key in controls})
            else:
                lamp.data.energy = 0
            expected = min(500., lamp.data.energy / (.8 * .8 * math.pi))
            self.assertAlmostEqual(self.strength(face), expected, places=4)
            rgb, _ = self.fixture.render_rgb('factor_' + str(factor))
            rendered[factor] = float(rgb[12:36, 12:36].mean())
        self.assertLess(rendered[.25], rendered[1.] - .2)
        self.assertGreater(rendered[4.], rendered[1.] + .05)
        self.assertLess(rendered[0.], .01)

    def test_shared_light_datablock_updates_both_faces(self):
        from lighting_fit import collect_lighting_controls, apply_lighting_controls
        first = self.lamp('SharedA', -1)
        self.lamp('SharedB', 1, first.data)
        faces = self.faces()
        self.assertEqual(len(faces), 2)
        controls, _ = collect_lighting_controls(self.scene)
        self.assertEqual(len(controls['interior']), 1)
        for factor in (.25, 4., 1.):
            apply_lighting_controls(controls, {'interior': factor})
            expected = 100 * factor / (.8 * .8 * math.pi)
            for face in faces:
                self.assertAlmostEqual(self.strength(face), expected, places=4)

    def test_legacy_shared_materials_are_isolated_and_world_area_is_respected(self):
        from appearance_lighting import bind_fixture_power_drivers
        a = self.lamp('LegacyA', -1)
        b = self.lamp('LegacyB', 1)
        faces = self.faces()
        for face in faces:
            emission = next(node for node in face.material_slots[0].material.node_tree.nodes
                            if node.type == 'EMISSION')
            emission.inputs['Strength'].driver_remove('default_value')
            for key in ('source_light_object', 'fixture_emitting_area_m2', 'fixture_power_binding_version'):
                del face[key]
        faces[1].data.materials[0] = faces[0].data.materials[0]
        faces[0].scale = (2, .75, 1)
        b.data.energy = 25
        records = bind_fixture_power_drivers(self.scene)
        self.assertEqual([r['status'] for r in records], ['bound', 'bound'])
        self.assertNotEqual(faces[0].material_slots[0].material, faces[1].material_slots[0].material)
        self.assertAlmostEqual(faces[0]['fixture_emitting_area_m2'], .8 * .8 * 1.5, places=6)
        for face in faces:
            lamp = self.scene.objects[face['source_light']]
            self.assertAlmostEqual(self.strength(face), lamp.data.energy / (face['fixture_emitting_area_m2'] * math.pi), places=4)
        a.data.energy = 200
        self.assertAlmostEqual(self.strength(faces[1]), 25 / (.8 * .8 * math.pi), places=4)

    def test_animated_and_foreign_driven_appearance_are_not_overwritten(self):
        from appearance_lighting import bind_fixture_power_drivers
        self.lamp()
        face, = self.faces()
        emission = next(node for node in face.material_slots[0].material.node_tree.nodes
                        if node.type == 'EMISSION')
        socket = emission.inputs['Strength']
        socket.driver_remove('default_value')
        del face['fixture_power_binding_version']
        socket.keyframe_insert('default_value', frame=1)
        record, = bind_fixture_power_drivers(self.scene)
        self.assertEqual(record['reason'], 'animated_appearance_graph')
        emission.id_data.animation_data_clear()
        curve = socket.driver_add('default_value')
        curve.driver.expression = '3.25'
        record, = bind_fixture_power_drivers(self.scene)
        self.assertEqual(record['reason'], 'foreign_strength_driver')
        self.assertEqual(curve.driver.expression, '3.25')

    def test_saved_scene_retains_live_binding_after_reload(self):
        lamp = self.lamp()
        face, = self.faces()
        lamp.name = 'RenamedFixturePower'
        filename = self.fixture.directory / 'fixture_driver.blend'
        face_name, lamp_name = face.name, lamp.name
        bpy.ops.wm.save_as_mainfile(filepath=str(filename))
        bpy.ops.wm.open_mainfile(filepath=str(filename))
        face, lamp = bpy.data.objects[face_name], bpy.data.objects[lamp_name]
        for energy in (25., 400., 100.):
            lamp.data.energy = energy
            self.assertAlmostEqual(self.strength(face), energy / (.8 * .8 * math.pi), places=4)
        self.assertEqual(face['source_light_object'], lamp)

    def test_collect_preserves_animated_world_and_light_node_trees(self):
        from lighting_fit import collect_lighting_controls
        lamp = self.lamp()
        lamp.data.use_nodes = True
        emission = next(node for node in lamp.data.node_tree.nodes if node.type == 'EMISSION')
        emission.inputs['Strength'].keyframe_insert('default_value', frame=1)
        background = next(node for node in self.scene.world.node_tree.nodes if node.type == 'BACKGROUND')
        background.inputs['Strength'].default_value = .25
        background.inputs['Strength'].keyframe_insert('default_value', frame=1)
        controls, ignored = collect_lighting_controls(self.scene)
        self.assertEqual(controls, {})
        self.assertEqual({item['reason'] for item in ignored},
                         {'linked_or_animated_light_node_tree', 'linked_or_animated_world_node_tree'})

    def test_skipped_fixture_binding_is_reported_by_production_controls(self):
        from lighting_fit import collect_lighting_controls
        self.lamp()
        face, = self.faces()
        emission = next(node for node in face.material_slots[0].material.node_tree.nodes
                        if node.type == 'EMISSION')
        emission.inputs['Strength'].driver_remove('default_value')
        emission.inputs['Strength'].keyframe_insert('default_value', frame=1)
        _, ignored = collect_lighting_controls(self.scene)
        self.assertTrue(any(item.get('kind') == 'fixture_appearance' and
                            item['reason'] == 'animated_appearance_graph' for item in ignored))

    def test_dynamic_faces_do_not_add_diffuse_floor_energy(self):
        from lighting_fit import collect_lighting_controls, apply_lighting_controls
        lamp = self.lamp()
        face, = self.faces()
        self.fixture.plane(emission=False)
        self.scene.view_settings.exposure = 0
        self.scene.cycles.samples = 64
        self.scene.camera = self.fixture.camera('FloorView', (0, 0, 1), (0, 0, 0))
        controls, _ = collect_lighting_controls(self.scene)
        for factor in (.25, 4., 1.):
            apply_lighting_controls(controls, {key: factor for key in controls})
            face.hide_render = True
            before, _ = self.fixture.render_rgb('floor_without_' + str(factor))
            face.hide_render = False
            after, _ = self.fixture.render_rgb('floor_with_' + str(factor))
            # Adding an invisible mesh can reorder Cycles sampling; compare
            # energy and the image rather than demanding identical samples.
            self.assertLess(abs(float(before.mean() - after.mean())), .0005)
            np.testing.assert_allclose(before, after, atol=2 / 255 + 1e-7, rtol=0)
            self.assertAlmostEqual(self.strength(face), lamp.data.energy / (.8 * .8 * math.pi), places=4)


if __name__ == '__main__':
    unittest.main(argv=[__file__])
