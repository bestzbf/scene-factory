"""Blender-only runtime checks for external catalog material-slot contracts."""
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import bpy  # noqa: F401
except ImportError:
    bpy = None


@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class ExternalAssetRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from external_assets import ExternalAssetImporter
        cls.importer_type = ExternalAssetImporter

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='external_asset_runtime_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
            for item in list(collection):
                collection.remove(item)

    class _Catalog:
        def __init__(self, variants):
            self.variants = variants

        def get_material_variant(self, variant_id):
            return self.variants[variant_id]

    class _Slot:
        def __init__(self, name='', material_name=''):
            self.name = name
            self.material = type('Material', (), {'name': material_name})() if material_name else None

    class _Object:
        def __init__(self, *slots):
            self.material_slots = list(slots)

    class _FileCatalog:
        def __init__(self, root, assets, variants):
            self.root, self.assets, self.variants = root, assets, variants

        def get_asset(self, asset_id):
            return self.assets[asset_id]

        def get_material_variant(self, variant_id):
            return self.variants[variant_id]

        def production_readiness(self, asset):
            return True, []

        def resolve_path(self, relative):
            return self.root / relative

        def verify_hash(self, asset_id):
            asset = self.get_asset(asset_id)
            actual = hashlib.sha256(self.resolve_path(asset['model_file']).read_bytes()).hexdigest()
            if actual != asset['hash']:
                raise ValueError('model hash mismatch')

        def verify_material_hashes(self, variant_id):
            variant = self.get_material_variant(variant_id)
            for maps in [variant['maps'], *variant['slot_materials'].values()]:
                for record in maps.values():
                    actual = hashlib.sha256(self.resolve_path(record['path']).read_bytes()).hexdigest()
                    if actual != record['hash']:
                        raise ValueError('texture hash mismatch')

    def real_importer(self, can_rotate_z=True):
        # Export a genuine two-slot GLB. The .001 suffix belongs to the source
        # contract and must never be stripped as if it were an import collision.
        bpy.ops.mesh.primitive_cube_add(size=2)
        cube = bpy.context.object
        for name in ('seat_fabric', 'seat_fabric.001'):
            cube.data.materials.append(bpy.data.materials.new(name))
        for polygon in cube.data.polygons:
            polygon.material_index = polygon.index % 2
        bpy.ops.export_scene.gltf(filepath=str(self.root / 'model.glb'), export_format='GLB', use_selection=True)
        bpy.data.objects.remove(cube, do_unlink=True)
        maps = {}
        for name, color in [('red', (1, 0, 0, 1)), ('blue', (0, 0, 1, 1))]:
            image = bpy.data.images.new(name, width=1, height=1)
            image.pixels[:] = color
            image.filepath_raw = str(self.root / f'{name}.png')
            image.file_format = 'PNG'
            image.save()
            maps[name] = {'base_color': {'path': f'{name}.png', 'color_space': 'sRGB',
                                       'hash': hashlib.sha256((self.root / f'{name}.png').read_bytes()).hexdigest()}}
        digest = hashlib.sha256((self.root / 'model.glb').read_bytes()).hexdigest()
        assets = {
            asset_id: {'asset_id': asset_id, 'family_id': 'chair', 'geometry_variant_id': asset_id,
                       'model_file': 'model.glb', 'model_format': 'glb', 'hash': digest,
                       'material_variant_ids': ['red', 'blue'], 'front_axis': '-Y',
                       'meters_per_unit': 1.0, 'anchor': {'point': [0, 0, -1]},
                       'dimensions_import_units': [2, 2, 2], 'audit_provenance': {'manifest_gate': True},
                       'placement_rules': {'can_rotate_z': can_rotate_z}}
            for asset_id in ('chair_a', 'chair_b')
        }
        variants = {
            variant_id: {'compatible_geometry_ids': list(assets), 'normal_convention': 'OpenGL_Y+',
                         'maps': maps[variant_id],
                         'slot_materials': {'seat_fabric': maps[variant_id],
                                            'seat_fabric.001': maps['blue' if variant_id == 'red' else 'red']}}
            for variant_id in ('red', 'blue')
        }
        return self.importer_type(self._FileCatalog(self.root, assets, variants))

    def instantiate(self, importer, object_id, asset_id='chair_a', variant='red', rotation=None, **binding):
        entry = {'id': object_id, 'position': [2, 3, 4], 'rotation_degrees': rotation or [0, 0, 0],
                 'category': 'chair'}
        return importer.instantiate(entry, {'object_id': object_id, 'asset_id': asset_id,
                                           'material_variant_id': variant, **binding})

    def importer(self, variants):
        instance = self.importer_type.__new__(self.importer_type)
        instance.catalog = self._Catalog(variants)
        return instance

    def test_declared_slot_must_exist_in_imported_meshes(self):
        importer = self.importer({'chair_mat': {'slot_materials': {'seat_fabric': {}}}})
        with self.assertRaisesRegex(ValueError, 'seat_fabric'):
            importer._validate_material_slots(
                {'asset_id': 'chair_geo', 'material_variant_ids': ['chair_mat']},
                [self._Object(self._Slot('metal_frame'))],
            )

    def test_declared_slots_are_collected_across_meshes_and_material_fallback_name(self):
        importer = self.importer({'chair_mat': {'slot_materials': {
            'seat_fabric': {}, 'metal_frame': {},
        }}})
        importer._validate_material_slots(
            {'asset_id': 'chair_geo', 'material_variant_ids': ['chair_mat']},
            [self._Object(self._Slot(material_name='seat_fabric')),
             self._Object(self._Slot('metal_frame'))],
        )

    def test_metric_anchor_is_converted_to_import_units(self):
        importer = self.importer({})
        anchor = importer._metric_anchor_to_import_units([0.5, -1.0, 1.5], 0.5)
        self.assertEqual(tuple(anchor), (1.0, -2.0, 3.0))

    def test_real_glb_repeated_import_preserves_slots_and_instance_materials(self):
        importer = self.real_importer()
        original_materials = {material: material.name for material in bpy.data.materials}
        first = self.instantiate(importer, 'first', variant='red')
        first_mesh = next(child for child in first.children if child.type == 'MESH')
        first_materials = list(first_mesh.data.materials)
        second = self.instantiate(importer, 'second', asset_id='chair_b', variant='blue')
        third = self.instantiate(importer, 'third', variant='blue')
        for asset_id in ('chair_a', 'chair_b'):
            cached_meshes = importer.cache[asset_id][0]
            names = {material[importer._SOURCE_MATERIAL_NAME] for mesh in cached_meshes for material in mesh.materials}
            self.assertEqual(names, {'seat_fabric', 'seat_fabric.001'})
        self.assertEqual(list(first_mesh.data.materials), first_materials)
        self.assertEqual(first_materials, [importer.materials['red']['seat_fabric'],
                                          importer.materials['red']['seat_fabric.001']])
        for root in (second, third):
            mesh = next(child for child in root.children if child.type == 'MESH')
            self.assertIsNot(mesh.data, first_mesh.data)
            self.assertEqual(list(mesh.data.materials), [importer.materials['blue']['seat_fabric'],
                                                        importer.materials['blue']['seat_fabric.001']])
        for material, name in original_materials.items():
            self.assertEqual(material.name, name)

    def test_failed_import_restores_existing_material_names(self):
        importer = self.real_importer()
        names = {material: material.name for material in bpy.data.materials}
        with self.assertRaises(ValueError):
            importer._import_meshes(self.root / 'unsupported.obj')
        self.assertEqual({material: material.name for material in names}, names)

    def test_can_rotate_z_false_blocks_entry_and_binding_yaw_before_import(self):
        importer = self.real_importer(can_rotate_z=False)
        before = set(bpy.data.objects)
        for binding, rotation in [({}, [0, 0, 90]), ({'rotation_xyz_deg': [0, 0, -45]}, [0, 0, 0])]:
            with self.subTest(binding=binding, rotation=rotation):
                with self.assertRaisesRegex(ValueError, 'can_rotate_z=false'):
                    self.instantiate(importer, 'rejected', rotation=rotation, **binding)
        self.assertEqual(set(bpy.data.objects), before)
        self.assertFalse(importer.cache)

    def test_rotation_policy_allows_zero_full_turns_and_permitted_yaw(self):
        importer = self.real_importer(can_rotate_z=False)
        for index, yaw in enumerate((0, 360, -360)):
            root = self.instantiate(importer, f'fixed_{index}', rotation=[0, 0, yaw])
            self.assertAlmostEqual(root.rotation_euler.z, math.radians(yaw), places=5)
        importer.catalog.assets['chair_a']['placement_rules']['can_rotate_z'] = True
        rotated = self.instantiate(importer, 'rotated', rotation=[0, 0, 90])
        self.assertAlmostEqual(rotated.rotation_euler.z, math.pi / 2, places=5)

    def test_real_glb_applies_metric_anchor_and_uniform_scale(self):
        importer = self.real_importer()
        asset = importer.catalog.assets['chair_a']
        asset['meters_per_unit'] = 0.5
        asset['anchor']['point'] = [0.5, -0.5, 0.5]
        root = self.instantiate(importer, 'anchored', uniform_scale=2)
        bpy.context.view_layer.update()
        points = [child.matrix_world @ vertex.co for child in root.children for vertex in child.data.vertices]
        low = [min(point[axis] for point in points) for axis in range(3)]
        high = [max(point[axis] for point in points) for axis in range(3)]
        self.assertEqual(tuple(root.scale), (1, 1, 1))
        for actual, expected in zip(low + high, [0, 3, 2, 2, 5, 4]):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_real_glb_rejects_manifest_dimensions_that_disagree(self):
        importer = self.real_importer()
        asset = importer.catalog.assets['chair_a']
        asset['dimensions_import_units'] = [2, 2, 2.1]
        before = set(bpy.data.objects)
        with self.assertRaisesRegex(ValueError, 'do not match manifest dimensions_import_units'):
            self.instantiate(importer, 'wrong_dimensions')
        self.assertEqual(set(bpy.data.objects), before)
        self.assertFalse(importer.cache)

    def test_real_glb_dimension_gate_rejects_nonfinite_nonpositive_or_missing_values(self):
        importer = self.real_importer()
        for dimensions in ([2, 2, float('nan')], [2, 2, 0], [True, 2, 2], None):
            with self.subTest(dimensions=dimensions):
                importer.catalog.assets['chair_a']['dimensions_import_units'] = dimensions
                with self.assertRaisesRegex(ValueError, 'finite positive'):
                    self.instantiate(importer, 'invalid_dimensions')

    def test_real_glb_dimension_tolerance_and_legacy_preview(self):
        importer = self.real_importer()
        importer.catalog.assets['chair_a']['dimensions_import_units'] = [2.0001, 2, 2]
        self.instantiate(importer, 'within_tolerance')
        legacy = importer.catalog.assets['chair_b']
        legacy.pop('audit_provenance')
        legacy['dimensions_import_units'] = [10, 20, 30]
        importer.preview = True
        self.instantiate(importer, 'legacy_preview', asset_id='chair_b')

    def test_real_glb_slot_only_variant_must_cover_visible_slots(self):
        importer = self.real_importer()
        variant = importer.catalog.variants['red']
        variant['maps'] = {}
        missing = variant['slot_materials'].pop('seat_fabric.001')
        before = set(bpy.data.objects)
        with self.assertRaisesRegex(ValueError, 'does not cover visible material slots: seat_fabric.001'):
            self.instantiate(importer, 'missing_slot')
        self.assertEqual(set(bpy.data.objects), before)
        self.assertFalse(importer.cache)
        variant['slot_materials']['seat_fabric.001'] = missing
        self.instantiate(importer, 'all_slots_covered')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ExternalAssetRuntimeTests)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
