"""Blender checks for conservative photo reconstruction asset appearance.

Run: Blender -b --factory-startup --python-exit-code 1 --python this_file.py
"""
import sys
import unittest
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class AssetAppearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import detailed_assets
        cls.assets = detailed_assets

    def build(self, kind, policy='photo_reconstruction', color=(.82, .83, .80), suffix=''):
        dimensions = {'desk': (1.4, .75, .76), 'chair': (.65, .65, 1.1),
                      'monitor': (.65, .23, .48)}[kind]
        return self.assets.build_asset({'id': f'{self._testMethodName}_{kind}_{policy}_{suffix}',
                                        'category': kind, 'color': color,
                                        'dimensions': dimensions, 'position': [0, 0, 0]},
                                       appearance_policy=policy)

    def material(self, root, suffix):
        obj = next(obj for obj in root.children_recursive if obj.name.endswith(suffix))
        return obj.material_slots[0].material

    def shader(self, mat):
        return next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')

    def test_neutral_table_has_no_invented_wood_grain(self):
        root = self.build('desk')
        mat = self.material(root, '_LaminateTabletop')
        self.assertEqual(mat['appearance_kind'], 'matte_laminate')
        self.assertFalse(self.shader(mat).inputs['Base Color'].is_linked)
        bump = next(n for n in mat.node_tree.nodes if n.type == 'BUMP')
        self.assertLessEqual(bump.inputs['Distance'].default_value, .0001)
        noise = mat.node_tree.nodes['LaminateMicrofinish']
        self.assertGreaterEqual(noise.inputs['Scale'].default_value, 1000)
        # The white laminate keeps the input colour; lighting corrections must
        # not be disguised as an arbitrary darkening of the material.
        for observed, expected in zip(self.shader(mat).inputs['Base Color'].default_value, (.82, .83, .80)):
            self.assertAlmostEqual(observed, expected, places=5)

    def test_screen_has_no_invented_texture_or_emission(self):
        before = set(bpy.data.images)
        root = self.build('monitor', color=(.025, .028, .03))
        mat = self.material(root, '_UnobservedScreen')
        self.assertEqual(before, set(bpy.data.images))
        self.assertFalse(any(n.type == 'TEX_IMAGE' for n in mat.node_tree.nodes))
        self.assertLess(max(self.shader(mat).inputs['Base Color'].default_value[:3]), .02)
        self.assertEqual(self.shader(mat).inputs['Emission Strength'].default_value, 0)
        self.assertEqual(root['appearance_policy'], 'photo_reconstruction')

    def test_dark_chair_frame_does_not_randomly_become_alloy(self):
        # Both deterministic seed branches must keep an evidence-neutral dark
        # base instead of alternating nylon with bright alloy per asset ID.
        for suffix in ('a', 'b', 'c', 'd'):
            root = self.build('chair', color=(.30, .32, .31), suffix=suffix)
            for obj in root.children_recursive:
                if 'BaseSpoke' in obj.name or 'ArmPad' in obj.name or 'SeatPan' in obj.name:
                    shader = self.shader(obj.material_slots[0].material)
                    self.assertLess(max(shader.inputs['Base Color'].default_value[:3]), .03)
                    self.assertEqual(shader.inputs['Metallic'].default_value, 0)
            # Gray upholstery remains gray; only frame selection changes.
            seat = self.shader(self.material(root, '_Seat'))
            self.assertAlmostEqual(seat.inputs['Base Color'].default_value[0], .30, places=5)

    def test_legacy_stays_available_and_shared_mesh_does_not_leak_material(self):
        legacy = self.build('monitor', 'legacy', color=(.025, .028, .03))
        mat_before = self.material(legacy, '_ActiveScreen')
        self.assertTrue(any(n.type == 'TEX_IMAGE' for n in mat_before.node_tree.nodes))
        photo = self.build('monitor', color=(.025, .028, .03))
        self.assertIs(self.material(legacy, '_ActiveScreen'), mat_before)
        self.assertIsNot(self.material(photo, '_UnobservedScreen'), mat_before)
        self.assertEqual(len(legacy.children_recursive), len(photo.children_recursive))
        self.assertEqual(tuple(legacy['requested_dimensions']), tuple(photo['requested_dimensions']))
        table = self.build('desk', 'legacy')
        self.assertTrue(self.shader(self.material(table, '_OakTabletop')).inputs['Base Color'].is_linked)

    def test_non_neutral_wood_colour_retains_wood_material(self):
        root = self.build('desk', color=(.61, .43, .23))
        self.assertTrue(self.shader(self.material(root, '_OakTabletop')).inputs['Base Color'].is_linked)

    def test_unknown_policy_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'appearance policy'):
            self.build('desk', 'misspelled-policy')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AssetAppearanceTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
