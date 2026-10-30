"""Blender adapter for validated catalog assets.

GLB is the production interchange format. glTF and FBX remain accepted for
legacy preview catalogues, but all files are hash checked by
:mod:`asset_catalog` before import. The adapter never executes code from an
asset package.
"""
from __future__ import annotations

import json
import math
import uuid

import bpy
from mathutils import Matrix, Vector

from asset_catalog import CatalogError


def finite(value, low, high, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise CatalogError('Invalid ' + label)
    return value


class ExternalAssetImporter:
    _SOURCE_MATERIAL_NAME = '_scene_factory_source_material_name'

    def __init__(self, catalog, preview=False):
        self.catalog, self.preview = catalog, preview
        self.cache = {}
        self.materials = {}
        self.records = []

    def _import_meshes(self, path):
        before = set(bpy.data.objects)
        existing_material_names = {material: material.name for material in bpy.data.materials}
        namespace = uuid.uuid4().hex
        # Blender makes material names globally unique, whereas slot names are
        # local to an asset. Reserve the existing datablocks' names during
        # import so the source names (including a legitimate .001 suffix) are
        # captured exactly, without trying to infer or strip numeric suffixes.
        for index, material in enumerate(existing_material_names):
            material.name = f'_sf_existing_{namespace}_{index}'
        suffix = path.suffix.lower()
        try:
            if suffix in {'.glb', '.gltf'}:
                bpy.ops.import_scene.gltf(filepath=str(path))
            elif suffix == '.fbx':
                bpy.ops.import_scene.fbx(filepath=str(path))
            else:
                raise CatalogError('Unsupported catalog model format: ' + suffix)
            bpy.context.view_layer.update()
            return [obj for obj in set(bpy.data.objects) - before if obj.type == 'MESH'], before
        except Exception:
            for obj in set(bpy.data.objects) - before:
                bpy.data.objects.remove(obj, do_unlink=True)
            raise
        finally:
            imported_materials = set(bpy.data.materials) - set(existing_material_names)
            for index, material in enumerate(imported_materials):
                material[self._SOURCE_MATERIAL_NAME] = material.name
                material.name = f'_sf_imported_{namespace}_{index}'
            for material, name in existing_material_names.items():
                material.name = name

    @staticmethod
    def _material_slot_name(slot):
        """Return the stable name Blender imported for one material slot."""
        material = getattr(slot, 'material', None)
        name = (material.get(ExternalAssetImporter._SOURCE_MATERIAL_NAME)
                if material is not None and hasattr(material, 'get') else None)
        if not name:
            name = getattr(slot, 'name', None)
        if not name:
            name = getattr(material, 'name', None) if material else None
        return name if isinstance(name, str) else ''

    def _validate_material_slots(self, asset, meshes):
        """Reject a model when any declared replacement slot is absent.

        Slot-specific materials are keyed by the source GLB material-slot name.
        Falling back to the default material for a typo or an exporter rename
        makes a batch appear successful while silently changing its appearance,
        so every declared slot across the asset's compatible variants is checked
        immediately after import and before the geometry is cached.
        """
        declared = set()
        variants = []
        for material_id in asset.get('material_variant_ids', []):
            variant = self.catalog.get_material_variant(material_id)
            if variant.get('mode') == 'embedded_gltf':
                self.catalog.verify_material_hashes(material_id)
                continue
            variants.append((material_id, variant))
            declared.update((variant.get('slot_materials') or {}).keys())
        actual = {
            name
            for obj in meshes
            for slot in getattr(obj, 'material_slots', ())
            for name in (self._material_slot_name(slot),)
            if name
        }
        missing = sorted(declared - actual)
        if missing:
            raise CatalogError(
                'Catalog material slots missing from imported model for asset '
                + str(asset.get('asset_id')) + ': ' + ', '.join(missing)
            )
        visible_slots = set()
        for obj in meshes:
            slots = list(getattr(obj, 'material_slots', ()))
            polygons = getattr(getattr(obj, 'data', None), 'polygons', None)
            indices = ({polygon.material_index for polygon in polygons} if polygons is not None
                       else range(len(slots)))
            visible_slots.update(self._material_slot_name(slots[index]) if 0 <= index < len(slots) else ''
                                 for index in indices)
        for material_id, variant in variants:
            if variant.get('maps'):
                continue
            uncovered = sorted(visible_slots - set((variant.get('slot_materials') or {}).keys()))
            if uncovered:
                raise CatalogError('Material variant ' + material_id
                                   + ' without default maps does not cover visible material slots: '
                                   + ', '.join(name or '<unassigned>' for name in uncovered))

    @staticmethod
    def _metric_anchor_to_import_units(point, meters_per_unit):
        """Convert a manifest anchor in metres to the source mesh units."""
        if (not isinstance(point, (list, tuple)) or len(point) != 3
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(float(value)) for value in point)):
            raise CatalogError('anchor point must contain three finite numbers')
        if (isinstance(meters_per_unit, bool) or not isinstance(meters_per_unit, (int, float))
                or not math.isfinite(float(meters_per_unit)) or float(meters_per_unit) <= 0):
            raise CatalogError('Explicit metric anchor requires meters_per_unit')
        return Vector(point) / float(meters_per_unit)

    def _geometry(self, asset):
        key = asset['asset_id']
        if key in self.cache:
            return self.cache[key]
        self.catalog.verify_hash(key)
        path = self.catalog.resolve_path(asset['model_file'])
        meshes, before = self._import_meshes(path)
        try:
            if not meshes:
                raise CatalogError('Catalog model contains no mesh objects')
            self._validate_material_slots(asset, meshes)
            if any(obj.modifiers or not obj.data.uv_layers for obj in meshes):
                raise CatalogError('Static meshes with UVs and applied modifiers are required')
            yaw = {'-Y': 0, '+X': -90, '+Y': 180, '-X': 90, None: 0}
            if asset.get('front_axis') not in yaw:
                raise CatalogError('front_axis must be horizontal')
            turn = Matrix.Rotation(math.radians(yaw[asset.get('front_axis')]), 4, 'Z')
            copies = []
            for obj in meshes:
                mesh = obj.data.copy()
                mesh.transform(turn @ obj.matrix_world)
                copies.append(mesh)
            points = [v.co for mesh in copies for v in mesh.vertices]
            if not points:
                raise CatalogError('Empty catalog geometry')
            manifest_gate = bool((asset.get('audit_provenance') or {}).get('manifest_gate')
                                 or (asset.get('audit_provenance') or {}).get('intake_gate'))
            if manifest_gate and any(not math.isfinite(component) for point in points for component in point):
                raise CatalogError('Manifest asset geometry contains non-finite coordinates')
            low = Vector([min(v[i] for v in points) for i in range(3)])
            high = Vector([max(v[i] for v in points) for i in range(3)])
            dimensions = list(high - low)
            if manifest_gate:
                declared_dimensions = asset.get('dimensions_import_units')
                if (not isinstance(declared_dimensions, (list, tuple)) or len(declared_dimensions) != 3
                        or any(isinstance(value, bool) or not isinstance(value, (int, float))
                               or not math.isfinite(value) or value <= 0 for value in declared_dimensions)
                        or any(not math.isfinite(value) or value <= 0 for value in dimensions)):
                    raise CatalogError('Manifest asset dimensions must contain three finite positive values')
                if not all(math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-5)
                           for actual, expected in zip(dimensions, declared_dimensions)):
                    raise CatalogError('Imported dimensions do not match manifest dimensions_import_units: '
                                       + repr(dimensions) + ' != ' + repr(declared_dimensions))
            declared = asset.get('anchor') if isinstance(asset.get('anchor'), dict) else {}
            point = declared.get('point') if isinstance(declared, dict) else None
            if isinstance(point, (list, tuple)) and len(point) == 3:
                # Catalog anchors are declared in metres; imported mesh
                # vertices are still in the source import unit until the
                # instance root receives its metric scale.
                anchor = self._metric_anchor_to_import_units(point, asset.get('meters_per_unit'))
            else:
                anchor = Vector(((low.x + high.x) / 2, (low.y + high.y) / 2, low.z))
            for mesh in copies:
                mesh.transform(Matrix.Translation(-anchor))
            self.cache[key] = copies, dimensions, anchor
            return self.cache[key]
        finally:
            for obj in set(bpy.data.objects) - before:
                bpy.data.objects.remove(obj, do_unlink=True)

    def _new_material(self, name, maps, normal_convention, tint=None):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = next(node for node in nodes if node.type == 'BSDF_PRINCIPLED')
        aliases = {'base_color': 'Base Color', 'baseColor': 'Base Color', 'normal': 'Normal',
                   'roughness': 'Roughness', 'metallic': 'Metallic', 'ao': 'AO'}
        for raw_kind, record in (maps or {}).items():
            if not record.get('path'):
                continue
            kind = aliases.get(raw_kind, raw_kind)
            if kind not in ('Base Color', 'Roughness', 'Metallic', 'Normal', 'AO'):
                raise CatalogError('Unsupported material map: ' + str(raw_kind))
            texture = nodes.new('ShaderNodeTexImage')
            texture.image = bpy.data.images.load(str(self.catalog.resolve_path(record['path'])), check_existing=True)
            texture.image.colorspace_settings.name = 'sRGB' if kind == 'Base Color' else 'Non-Color'
            texture.image.pack()
            if kind == 'Normal':
                if normal_convention not in (None, 'OpenGL_Y+', 'OpenGL', 'opengl', '+Y'):
                    raise CatalogError('Unsupported normal convention: ' + str(normal_convention))
                if normal_convention is None and not self.preview:
                    raise CatalogError('Production normal convention must be verified')
                normal = nodes.new('ShaderNodeNormalMap')
                links.new(texture.outputs['Color'], normal.inputs['Color'])
                links.new(normal.outputs['Normal'], bsdf.inputs['Normal'])
            elif kind == 'AO':
                texture.label = 'AO (not baked into base color)'
            else:
                links.new(texture.outputs['Color'], bsdf.inputs[kind])
        tint = tint or [1, 1, 1]
        if len(tint) != 3:
            raise CatalogError('base_color_multiplier requires RGB')
        for value in tint:
            finite(value, 0, 1, 'base_color_multiplier')
        if tint != [1, 1, 1]:
            socket = bsdf.inputs['Base Color']
            source = socket.links[0].from_socket if socket.links else None
            if source:
                mix = nodes.new('ShaderNodeMixRGB'); mix.blend_type = 'MULTIPLY'
                mix.inputs[0].default_value = 1
                mix.inputs[2].default_value = (*tint, 1)
                links.new(source, mix.inputs[1]); links.new(mix.outputs[0], socket)
            else:
                socket.default_value = (*tint, 1)
        return mat

    def _materials_for(self, asset, variant_id):
        variant = self.catalog.get_material_variant(variant_id)
        self.catalog.verify_material_hashes(variant_id)
        if variant_id not in asset['material_variant_ids'] or asset['geometry_variant_id'] not in variant['compatible_geometry_ids']:
            raise CatalogError('Incompatible material/geometry pair')
        cached = self.materials.setdefault(variant_id, {})
        if '__default__' not in cached:
            cached['__default__'] = self._new_material('Catalog_' + variant_id, variant.get('maps', {}),
                                                        variant.get('normal_convention'), variant.get('base_color_multiplier'))
        for slot_name, maps in (variant.get('slot_materials') or {}).items():
            if slot_name not in cached:
                cached[slot_name] = self._new_material('Catalog_' + variant_id + '_' + str(slot_name), maps,
                                                        variant.get('normal_convention'), variant.get('base_color_multiplier'))
        return cached

    def _material(self, asset, variant_id):
        return self._materials_for(asset, variant_id)['__default__']

    def instantiate(self, entry, binding):
        allowed = {'object_id', 'asset_id', 'material_variant_id', 'uniform_scale', 'rotation_xyz_deg', 'preview_height_m'}
        if set(binding) - allowed or binding.get('object_id') != entry['id']:
            raise CatalogError('Invalid asset binding fields or object ID')
        asset = self.catalog.get_asset(binding['asset_id'])
        ready, reasons = self.catalog.production_readiness(asset)
        if not self.preview and not ready:
            raise CatalogError('Asset is not production ready: ' + ', '.join(reasons))
        rotation = binding.get('rotation_xyz_deg', entry['rotation_degrees'])
        if not isinstance(rotation, list) or len(rotation) != 3:
            raise CatalogError('rotation_xyz_deg must contain three values')
        rotation = [math.radians(finite(value, -360, 360, 'rotation')) for value in rotation]
        if (asset.get('placement_rules', {}).get('can_rotate_z') is False
                and not math.isclose(math.remainder(rotation[2], math.tau), 0, abs_tol=1e-9)):
            raise CatalogError('Asset placement rules forbid Z rotation (can_rotate_z=false)')
        meshes, dimensions, _anchor = self._geometry(asset)
        units = asset.get('meters_per_unit')
        scale_source = 'catalog'
        if units is None:
            if not self.preview or 'preview_height_m' not in binding:
                raise CatalogError('Unknown physical scale requires explicit preview_height_m in preview mode')
            units = finite(binding['preview_height_m'], .01, 30, 'preview_height_m') / dimensions[2]
            scale_source = 'preview_height_estimate_not_measured'
        scale = finite(units, .000001, 10000, 'meters_per_unit') * finite(binding.get('uniform_scale', 1), .01, 100, 'uniform_scale')
        variant = self.catalog.get_material_variant(binding['material_variant_id'])
        native = variant.get('mode') == 'embedded_gltf'
        if native:
            self.catalog.verify_material_hashes(binding['material_variant_id'])
            if (binding['material_variant_id'] not in asset['material_variant_ids']
                    or variant.get('compatible_geometry_ids') != [asset['geometry_variant_id']]):
                raise CatalogError('Incompatible native material/geometry pair')
            materials = None
        else:
            materials = self._materials_for(asset, binding['material_variant_id'])
        root = bpy.data.objects.new(entry['id'], None)
        bpy.context.collection.objects.link(root)
        root.location = entry['position']; root.rotation_euler = rotation; root.scale = (scale, scale, scale)
        root['category'] = entry['category']
        root['geometry_source'] = 'external_catalog_' + str(asset.get('model_format') or 'fbx')
        for index, mesh in enumerate(meshes):
            # Cached geometry is shared by repeated instances.  Copy the mesh
            # datablock before assigning material slots so one material
            # variant cannot mutate another instance.
            instance_mesh = mesh.copy()
            obj = bpy.data.objects.new(entry['id'] + '_mesh_' + str(index), instance_mesh)
            bpy.context.collection.objects.link(obj); obj.parent = root
            if native:
                # Preserve the native glTF graph, packed MR channels and scalar
                # factors. Isolate mutable material state per instance.
                for slot in obj.material_slots:
                    if slot.material is None:
                        raise CatalogError('Native appearance has an empty material slot')
                    slot.material = slot.material.copy()
                continue
            if not obj.material_slots:
                obj.data.materials.append(materials['__default__'])
            for slot in obj.material_slots:
                slot_name = self._material_slot_name(slot)
                slot.material = materials.get(slot_name, materials['__default__'])
        record = {'object_id': entry['id'], 'asset_id': asset['asset_id'], 'family_id': asset['family_id'],
                  'geometry_variant_id': asset['geometry_variant_id'], 'material_variant_id': binding['material_variant_id'],
                  'model_sha256': asset['hash'], 'meters_per_import_unit_applied': scale, 'scale_source': scale_source,
                  'preview_only': self.preview, 'readiness_issues': reasons}
        if native:
            record.update(material_mode='embedded_gltf', quality_status='technical_checked',
                          visual_accepted=False, declared_scale_source=asset.get('scale_source'))
        root['asset_provenance'] = json.dumps(record, ensure_ascii=False)
        self.records.append(record)
        return root
