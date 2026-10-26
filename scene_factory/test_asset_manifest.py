import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from .asset_manifest import ManifestError, build_catalog, validate_manifest, validate_manifest_or_raise
    from .asset_catalog import load_catalog
except ImportError:
    from asset_manifest import ManifestError, build_catalog, validate_manifest, validate_manifest_or_raise
    from asset_catalog import load_catalog


class AssetManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "chair" / "geo_001"
        (self.root / "textures").mkdir(parents=True)
        (self.root / "previews").mkdir()
        (self.root / "reference").mkdir()
        self.files = {}
        for relative, content in {
            "model.glb": b"fake-glb",
            "collision.glb": b"fake-collision",
            "textures/base.png": b"base-color",
            "textures/roughness.png": b"roughness",
            "previews/front.png": b"front",
            "previews/side.png": b"side",
            "previews/back.png": b"back",
            "previews/three_quarter.png": b"three-quarter",
            "previews/placement_test.png": b"placement",
            "reference/front.png": b"reference",
        }.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            self.files[relative] = path
        self.manifest_path = self.root / "asset_manifest.json"
        self.manifest = self._make_manifest()
        self._write()

    def _digest(self, relative):
        return hashlib.sha256(self.files[relative].read_bytes()).hexdigest()

    def _make_manifest(self):
        return {
            "schema_version": "asset_manifest_v1",
            "asset_id": "office_chair_geo_001",
            "family_id": "office_chair",
            "geometry_variant_id": "office_chair_geo_001",
            "status": "asset_accepted",
            "model": {
                "format": "glb", "path": "model.glb", "sha256": self._digest("model.glb"),
                "dimensions_import_units": [0.52, 0.55, 0.86],
                "dimensions_m": [0.52, 0.55, 0.86], "meters_per_import_unit": 1.0,
                "scale_source": "measured", "scale_confidence": 1.0,
                "canonical_up_axis": "+Z", "canonical_front_axis": "-Y",
                "triangle_count": 42000, "object_count": 1,
                "transform": {"location": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
            },
            "placement": {
                "support_type": "floor", "anchor_type": "bottom_center",
                "anchor_point_m": [0, 0, 0],
                "contact_points_m": [[-0.2, -0.2, 0], [0.2, -0.2, 0]],
                "mount_points": [], "clearance_m": 0,
                "can_rotate_z": True, "allow_uniform_scale": True,
                "allow_non_uniform_scale": False,
                "collision_path": "collision.glb", "collision_sha256": self._digest("collision.glb"),
            },
            "semantics": {
                "category": "chair", "description": "office chair", "tags": ["office", "fabric"],
                "scope": "single_prop", "completeness": "complete", "confidence": 0.94,
                "source": "human_verified",
            },
            "uv": {
                "layout_id": "office_chair_uv_001", "channels": 1,
                "normal_convention": "OpenGL_Y+", "texel_density_policy": "family_normalized",
            },
            "material_variant_ids": ["office_chair_mat_dark"],
            "materials": {
                "office_chair_mat_dark": {
                    "material_variant_id": "office_chair_mat_dark",
                    "compatible_geometry_ids": ["office_chair_geo_001"],
                    "uv_layout_id": "office_chair_uv_001", "normal_convention": "OpenGL_Y+",
                    "maps": {
                        "base_color": {"path": "textures/base.png", "color_space": "sRGB", "sha256": self._digest("textures/base.png")},
                        "roughness": {"path": "textures/roughness.png", "color_space": "Non-Color", "sha256": self._digest("textures/roughness.png")},
                        "metallic": {"path": None, "color_space": "Non-Color", "sha256": None},
                    },
                }
            },
            "previews": {
                name: {"path": f"previews/{name}.png", "sha256": self._digest(f"previews/{name}.png")}
                for name in ("front", "side", "back", "three_quarter", "placement_test")
            },
            "quality": {key: "passed" for key in ("file_integrity", "import", "geometry", "uv", "materials", "previews", "placement_test")}
                      | {"production_ready": True},
            "provenance": {
                "source_images": ["reference/front.png"], "generator": "fixture",
                "generator_version": "1", "generation_parameters_sha256": "a" * 64,
                "license": "internal", "created_at": "2026-10-21T00:00:00Z",
            },
        }

    def _write(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_valid_accepted_manifest(self):
        report = validate_manifest(self.root)
        self.assertTrue(report["valid"], report["errors"])
        self.assertTrue(report["production_ready"])
        self.assertEqual(validate_manifest_or_raise(self.manifest_path)["asset_id"], "office_chair_geo_001")

    def test_structure_only_validation_cannot_report_production_ready(self):
        self.files["model.glb"].unlink()
        report = validate_manifest(self.root, verify_hashes=False)
        self.assertTrue(report["valid"], report["errors"])
        self.assertFalse(report["production_ready"])
        self.assertIn("files_unverified", {item["code"] for item in report["warnings"]})
        report = validate_manifest(self.root, verify_hashes=False, require_production=True)
        self.assertFalse(report["valid"])
        self.assertFalse(report["production_ready"])

    def test_hash_mismatch_and_path_escape_are_blocking(self):
        self.manifest["model"]["sha256"] = "0" * 64
        self.manifest["materials"]["office_chair_mat_dark"]["maps"]["base_color"]["path"] = "../outside.png"
        self._write()
        report = validate_manifest(self.root)
        codes = {item["code"] for item in report["errors"]}
        self.assertFalse(report["valid"])
        self.assertIn("hash_mismatch", codes)
        self.assertIn("path_escape", codes)

    def test_unknown_scale_cannot_be_accepted(self):
        self.manifest["model"]["scale_source"] = "unknown"
        self._write()
        report = validate_manifest(self.root)
        self.assertFalse(report["valid"])
        self.assertTrue(any(item["code"] == "production_gate" for item in report["errors"]))

    def test_multi_material_list_form_and_gltf_is_technical_only(self):
        self.manifest["status"] = "technical_checked"
        self.manifest["quality"]["production_ready"] = False
        self.manifest["model"]["format"] = "gltf"
        self.manifest["model"]["path"] = "model.gltf"
        old = self.files["model.glb"]
        new = self.root / "model.gltf"
        new.write_bytes(old.read_bytes())
        self.manifest["model"]["sha256"] = hashlib.sha256(new.read_bytes()).hexdigest()
        self.manifest["materials"] = [self.manifest["materials"].pop("office_chair_mat_dark")]
        self._write()
        report = validate_manifest(self.root)
        self.assertTrue(report["valid"], report["errors"])
        self.assertFalse(report["production_ready"])

    def test_accepted_gltf_is_rejected_by_production_gate(self):
        self.manifest["model"]["format"] = "gltf"
        self.manifest["model"]["path"] = "model.gltf"
        new = self.root / "model.gltf"
        new.write_bytes(self.files["model.glb"].read_bytes())
        self.manifest["model"]["sha256"] = hashlib.sha256(new.read_bytes()).hexdigest()
        self._write()
        report = validate_manifest(self.root)
        self.assertFalse(report["valid"])
        self.assertTrue(any(item["code"] == "production_gate" for item in report["errors"]))

    def test_required_base_color_and_collision_hash_cannot_be_omitted(self):
        self.manifest["materials"]["office_chair_mat_dark"]["maps"]["base_color"] = {"path": None, "sha256": None}
        self.manifest["placement"].pop("collision_sha256")
        self._write()
        report = validate_manifest(self.root)
        codes = {item["code"] for item in report["errors"]}
        self.assertIn("path_missing", codes)
        self.assertIn("hash_missing", codes)

    def test_accepted_previews_require_distinct_image_files(self):
        front = self.manifest["previews"]["front"]
        self.manifest["previews"]["side"] = copy.deepcopy(front)
        self.manifest["previews"]["back"] = {
            "path": "model.glb", "sha256": self._digest("model.glb")
        }
        self._write()
        report = validate_manifest(self.root)
        codes = {item["code"] for item in report["errors"]}
        self.assertIn("preview_duplicate", codes)
        self.assertIn("preview_type_invalid", codes)

    def test_base_color_multiplier_is_validated_and_preserved(self):
        material = self.manifest["materials"]["office_chair_mat_dark"]
        material["base_color_multiplier"] = [0.72, 0.85, 1.0]
        self._write()
        output = Path(self.temp.name) / "multiplier-catalog.json"
        catalog_data = build_catalog(Path(self.temp.name) / "chair", output)
        self.assertEqual(
            catalog_data["material_variants"]["office_chair_mat_dark"]["base_color_multiplier"],
            [0.72, 0.85, 1.0],
        )
        loaded = load_catalog(output)
        self.assertEqual(
            loaded.get_material_variant("office_chair_mat_dark")["base_color_multiplier"],
            [0.72, 0.85, 1.0],
        )
        material["base_color_multiplier"] = [1.1, 0.85, 1.0]
        self._write()
        report = validate_manifest(self.root)
        self.assertIn("multiplier_invalid", {item["code"] for item in report["errors"]})

    def test_accepted_requires_applied_transform_and_preview_hashes(self):
        self.manifest["model"].pop("transform")
        self.manifest["previews"]["front"].pop("sha256")
        self._write()
        report = validate_manifest(self.root)
        codes = {item["code"] for item in report["errors"]}
        self.assertIn("production_gate", codes)
        self.assertIn("preview_hash_missing", codes)

    def test_slot_materials_are_validated_and_can_replace_default_maps(self):
        material = self.manifest["materials"].pop("office_chair_mat_dark")
        material["maps"] = {}
        material["slot_materials"] = {
            "seat_fabric": {
                "base_color": {"path": "textures/base.png", "color_space": "sRGB", "sha256": self._digest("textures/base.png")},
                "roughness": {"path": "textures/roughness.png", "color_space": "Non-Color", "sha256": self._digest("textures/roughness.png")},
            }
        }
        self.manifest["materials"]["office_chair_mat_dark"] = material
        self._write()
        report = validate_manifest(self.root)
        self.assertTrue(report["valid"], report["errors"])

    def test_unsupported_material_maps_are_blocked_before_catalog_build(self):
        maps = self.manifest["materials"]["office_chair_mat_dark"]["maps"]
        output = Path(self.temp.name) / "catalog.json"
        for name in ("emissive", "height", "opacity", "transmission"):
            with self.subTest(map=name):
                maps[name] = copy.deepcopy(maps["base_color"])
                self._write()
                report = validate_manifest(self.root)
                self.assertIn("map_unknown", {item["code"] for item in report["errors"]})
                with self.assertRaises(ManifestError):
                    build_catalog(self.root.parent, output)
                self.assertFalse(output.exists())
                del maps[name]

    def test_map_aliases_are_canonicalized_for_the_importer(self):
        maps = self.manifest["materials"]["office_chair_mat_dark"]["maps"]
        maps["BASE COLOR"] = maps.pop("base_color")
        maps["ROUGHNESS"] = maps.pop("roughness")
        self._write()
        output = Path(self.temp.name) / "catalog.json"
        catalog = build_catalog(self.root.parent, output)
        actual = catalog["material_variants"]["office_chair_mat_dark"]["maps"]
        self.assertEqual(set(actual), {"base_color", "roughness"})
        maps["Base Color"] = copy.deepcopy(maps["BASE COLOR"])
        self._write()
        report = validate_manifest(self.root)
        self.assertIn("map_duplicate", {item["code"] for item in report["errors"]})

    def test_build_catalog_is_fail_closed_and_loads_manifest_asset(self):
        self.manifest["materials"]["office_chair_mat_dark"]["base_color_multiplier"] = [0.8, 0.9, 1.0]
        self._write()
        output = Path(self.temp.name) / "catalog.json"
        catalog_data = build_catalog(Path(self.temp.name) / "chair", output)
        self.assertEqual(catalog_data["schema_version"], "asset_catalog_v1")
        self.assertEqual(catalog_data["source_manifests"], ["geo_001/asset_manifest.json"])
        loaded = load_catalog(output)
        ready, reasons = loaded.production_readiness("office_chair_geo_001")
        self.assertTrue(ready, reasons)
        self.assertTrue(loaded.verify_material_hashes("office_chair_mat_dark"))
        self.assertEqual(loaded.get_material_variant("office_chair_mat_dark")["base_color_multiplier"], [0.8, 0.9, 1.0])
        loaded.assets["office_chair_geo_001"]["quality"]["production_ready"] = False
        ready, reasons = loaded.production_readiness("office_chair_geo_001")
        self.assertFalse(ready)
        self.assertIn("quality_not_production_ready", reasons)

    def test_build_catalog_rejects_one_bad_package_without_output(self):
        bad_root = Path(self.temp.name) / "bad" / "geo_001"
        bad_root.mkdir(parents=True)
        (bad_root / "asset_manifest.json").write_text("{}", encoding="utf-8")
        output = Path(self.temp.name) / "bad-catalog.json"
        with self.assertRaises(ManifestError):
            build_catalog(Path(self.temp.name), output)
        self.assertFalse(output.exists())

    def test_build_catalog_rejects_unknown_geometry_without_replacing_output(self):
        output = Path(self.temp.name) / "catalog.json"
        build_catalog(self.root.parent, output)
        original = output.read_bytes()
        self.manifest["materials"]["office_chair_mat_dark"]["compatible_geometry_ids"].append("missing_geometry")
        self._write()
        # The package's own geometry is compatible, but the assembled catalog
        # must reject references to variants absent from this library.
        self.assertTrue(validate_manifest(self.root)["valid"])
        with self.assertRaisesRegex(ManifestError, "runtime validation.*unknown geometry"):
            build_catalog(self.root.parent, output)
        self.assertEqual(output.read_bytes(), original)

    def test_build_catalog_rejects_linked_and_broken_manifests(self):
        linked_package = self.root.parent / "linked_package"
        linked_package.mkdir()
        linked_manifest = linked_package / "asset_manifest.json"
        output = Path(self.temp.name) / "catalog.json"
        for target in (self.manifest_path, linked_package / "missing.json"):
            with self.subTest(target=target):
                linked_manifest.symlink_to(target)
                try:
                    with self.assertRaisesRegex(ManifestError, "symlink"):
                        build_catalog(self.root.parent, output)
                    self.assertFalse(output.exists())
                finally:
                    linked_manifest.unlink()

    def test_build_catalog_rejects_linked_package_directory(self):
        (self.root.parent / "linked_package").symlink_to(self.root, target_is_directory=True)
        output = Path(self.temp.name) / "catalog.json"
        with self.assertRaisesRegex(ManifestError, "symlink directories"):
            build_catalog(self.root.parent, output)
        self.assertFalse(output.exists())

    def test_cli_returns_nonzero_for_invalid_manifest(self):
        self.manifest["canonical"] = "invalid"
        self.manifest["model"]["canonical_front_axis"] = "+X"
        self._write()
        result = subprocess.run([sys.executable, "-m", "scene_factory.asset_manifest", "validate", str(self.root)],
                                capture_output=True, text=True, cwd=Path(__file__).parents[1])
        self.assertEqual(result.returncode, 1)
        self.assertIn('"valid": false', result.stdout)

    def test_or_raise_exposes_compact_failure(self):
        self.manifest["placement"]["allow_non_uniform_scale"] = True
        self._write()
        with self.assertRaises(ManifestError):
            validate_manifest_or_raise(self.root)


if __name__ == "__main__":
    unittest.main()
