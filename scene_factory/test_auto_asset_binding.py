"""Automatic library binding contracts; normal Python, no provider or Blender."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    from .auto_asset_binding import generate_bindings, normalize_options
    from .asset_catalog import AssetCatalog
    from .spec_v2 import upgrade_v1
except ImportError:
    from auto_asset_binding import generate_bindings, normalize_options
    from asset_catalog import AssetCatalog
    from spec_v2 import upgrade_v1


class AutoAssetBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("model.glb", "collision.glb", "base.png", "roughness.png", "manifest.json"):
            (self.root / name).write_bytes(("fixture-" + name).encode())
        base = json.loads((Path(__file__).parent / "examples/living_room.json").read_text())
        base["objects"] = [{"id": "chair", "category": "chair", "position": [0, 2, 0],
            "dimensions": [.5, .6, .9], "rotation_z_deg": 0, "color": [.3, .3, .3],
            "repeat": [1, 1], "spacing": [1, 1], "confidence": .9}]
        self.bundle = {"schema_version": 3, "scene_id": "test_room",
            "shared_scene": upgrade_v1(base, 640, 480, "view_01"),
            "architecture": {"omit_surfaces": [], "front_wall": False},
            "views": [{"id": "view_01", "source_image": "photo.png", "width": 640, "height": 480,
                "camera": {**copy.deepcopy(base["camera"]), "roll_degrees": 0}}],
            "texture_sources": []}
        self.data = {"schema_version": "asset_catalog_v1", "root": str(self.root),
            "assets": [self.asset("chair_01")], "material_variants": {"mat_01": {
                "material_variant_id": "mat_01", "compatible_geometry_ids": ["chair_01"],
                "normal_convention": "OpenGL_Y+", "maps": {
                    "base_color": self.map_record("base.png", "sRGB"),
                    "roughness": self.map_record("roughness.png", "Non-Color")},
                "audit_provenance": {"manifest_sha256": self.digest("manifest.json")}}}}

    def digest(self, name):
        return hashlib.sha256((self.root / name).read_bytes()).hexdigest()

    def map_record(self, name, color_space):
        return {"path": name, "color_space": color_space, "hash": self.digest(name)}

    def asset(self, name):
        return {"asset_id": name, "family_id": "office_chair", "geometry_variant_id": name,
            "model_file": "model.glb", "model_format": "glb", "hash": self.digest("model.glb"),
            "dimensions_import_units": [.5, .6, .9], "meters_per_unit": 1,
            "up_axis": "+Z", "front_axis": "-Y", "symmetric_front": False,
            "status": "approved", "category": "chair", "scope": "single_prop", "placement": "floor",
            "anchor": {"type": "bottom_center", "point": [0, 0, 0]},
            "placement_rules": {"can_rotate_z": True, "allow_uniform_scale": True,
                                "allow_non_uniform_scale": False},
            "quality": {"production_ready": True, "complete_for_floor_placement": True},
            "collision_file": "collision.glb", "collision_hash": self.digest("collision.glb"),
            "material_variant_ids": ["mat_01"], "manifest_file": "manifest.json",
            "audit_provenance": {"manifest_gate": True, "manifest_sha256": self.digest("manifest.json")}}

    def run_binding(self, options=None, preview=False):
        return generate_bindings(self.bundle, AssetCatalog(self.data), options, allow_preview=preview)

    def object(self):
        return self.bundle["shared_scene"]["base_scene"]["objects"][0]

    def rejected_reasons(self, report, object_id="chair"):
        return report["objects"][object_id]["candidates"][0]["reasons"]

    def test_exact_category_and_dimensions_bind_without_mutating_inputs(self):
        before_bundle, before_data = copy.deepcopy(self.bundle), copy.deepcopy(self.data)
        bindings, report = self.run_binding()
        self.assertEqual(bindings, {"schema_version": 1, "instances": [{"object_id": "chair",
            "asset_id": "chair_01", "material_variant_id": "mat_01", "uniform_scale": 1,
            "rotation_xyz_deg": [0, 0, 0]}]})
        self.assertEqual((report["total_objects"], report["bound_objects"], report["fallback_objects"]), (1, 1, 0))
        record = report["objects"]["chair"]
        self.assertEqual(record["fit"]["relative_dimension_errors"], [0, 0, 0])
        self.assertFalse(record["support"]["collision_or_contact_verified"])
        self.assertFalse(report["appearance_match_verified"])
        self.assertEqual(self.bundle, before_bundle)
        self.assertEqual(self.data, before_data)

    def test_uniform_scale_and_shape_mismatch(self):
        self.data["assets"][0]["dimensions_import_units"] = [.4, .48, .72]
        bindings, _ = self.run_binding()
        self.assertAlmostEqual(bindings["instances"][0]["uniform_scale"], 1.25)
        self.data["assets"][0]["dimensions_import_units"] = [.4, .9, .72]
        bindings, report = self.run_binding()
        self.assertEqual(bindings["instances"], [])
        self.assertIn("dimension_mismatch", self.rejected_reasons(report))

    def test_scaling_can_be_forbidden_by_asset(self):
        asset = self.data["assets"][0]
        asset["placement_rules"]["allow_uniform_scale"] = False
        asset["dimensions_import_units"] = [.4, .48, .72]
        bindings, report = self.run_binding({"dimension_tolerance": .1})
        self.assertEqual(bindings["instances"], [])
        self.assertIn("dimension_mismatch", self.rejected_reasons(report))
        bindings, report = self.run_binding({"min_uniform_scale": 1.1})
        self.assertIn("uniform_scale_forbidden", self.rejected_reasons(report))

    def test_potential_support_height_cannot_shift_children(self):
        self.object().update(category="table", dimensions=[1.2, .8, .75])
        asset = self.data["assets"][0]
        asset.update(category="table", dimensions_import_units=[1.2, .8, .9])
        bindings, report = self.run_binding()
        self.assertEqual(bindings["instances"], [])
        self.assertIn("support_height_mismatch", self.rejected_reasons(report))
        self.assertGreater(report["objects"]["chair"]["candidates"][0]["fit"]["fitted_dimensions_m"][2], .81)

    def test_malformed_placement_rules_reject_candidate_without_crash(self):
        for rules in (["can_rotate_z"], "floor", 8):
            self.data["assets"][0]["placement_rules"] = rules
            bindings, report = self.run_binding()
            self.assertEqual(bindings["instances"], [])
            self.assertIn("placement_rules_unknown", self.rejected_reasons(report))

    def test_unknown_category_and_explicit_category_mapping(self):
        self.data["assets"][0]["category"] = "office_chair"
        bindings, report = self.run_binding()
        self.assertEqual(bindings["instances"], [])
        self.assertEqual(report["objects"]["chair"]["reasons"], ["no_category_match"])
        bindings, _ = self.run_binding({"category_map": {"chair": {"families": ["office_chair"]}}})
        self.assertEqual(len(bindings["instances"]), 1)

    def test_floor_support_and_non_floor_categories_are_conservative(self):
        for z in (.75, .02, -.02):
            self.object()["position"][2] = z
            bindings, report = self.run_binding()
            self.assertEqual(bindings["instances"], [])
            self.assertEqual(report["objects"]["chair"]["reasons"], ["floor_support_unverified"])
        self.object().update(category="monitor", position=[0, 2, 0])
        _, report = self.run_binding()
        self.assertEqual(report["objects"]["chair"]["reasons"], ["automatic_support_not_expressible"])

    def test_production_requires_manifest_approval_and_placement_metadata(self):
        for key, value, reason in (("status", "draft", "asset_not_approved"),
                ("scope", "partial_prop", "complete_single_prop_required"),
                ("up_axis", "+Y", "up_axis_not_canonical"),
                ("anchor", None, "bottom_center_anchor_unverified"),
                ("placement_rules", {}, "placement_rules_unknown"),
                ("audit_provenance", {}, "production_manifest_unverified")):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data["assets"][0][key] = value
                bindings, report = generate_bindings(self.bundle, AssetCatalog(data))
                self.assertEqual(bindings["instances"], [])
                self.assertIn(reason, self.rejected_reasons(report))

    def test_preview_does_not_infer_unknown_scale_anchor_or_front(self):
        asset = self.data["assets"][0]
        asset.update(status="draft", meters_per_unit=None, front_axis=None, anchor=None)
        bindings, report = self.run_binding(preview=True)
        self.assertEqual(bindings["instances"], [])
        self.assertTrue({"unknown_scale", "bottom_center_anchor_unverified", "semantic_front_not_horizontal"}
                        <= set(self.rejected_reasons(report)))

    def test_preview_front_rotation_swaps_xy_dimensions(self):
        asset = self.data["assets"][0]
        asset.update(front_axis="+X", dimensions_import_units=[.6, .5, .9])
        bindings, report = self.run_binding(preview=True)
        self.assertEqual(bindings["instances"][0]["uniform_scale"], 1)
        self.assertEqual(report["objects"]["chair"]["fit"]["source_dimensions_m"], [.5, .6, .9])
        bindings, report = self.run_binding()
        self.assertEqual(bindings["instances"], [])
        self.assertIn("production_front_not_canonical", self.rejected_reasons(report))

    def test_rotation_rule_is_enforced(self):
        self.data["assets"][0]["placement_rules"]["can_rotate_z"] = False
        self.object()["rotation_z_deg"] = 90
        bindings, report = self.run_binding()
        self.assertEqual(bindings["instances"], [])
        self.assertIn("rotation_forbidden", self.rejected_reasons(report))
        self.object()["rotation_z_deg"] = 360
        self.assertEqual(len(self.run_binding()[0]["instances"]), 1)

    def test_model_collision_and_texture_changes_reject_candidates(self):
        for name, expected in (("model.glb", "model_hash_invalid"), ("collision.glb", "collision_invalid"),
                               ("base.png", "material_hash_invalid")):
            with self.subTest(file=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(b"replacement")
                bindings, report = self.run_binding()
                self.assertEqual(bindings["instances"], [])
                reasons = self.rejected_reasons(report)
                for material in report["objects"]["chair"]["candidates"][0]["materials"]:
                    reasons += material["reasons"]
                self.assertIn(expected, reasons)
                path.write_bytes(original)

    def test_incompatible_material_and_wrong_color_spaces_fall_back(self):
        material = self.data["material_variants"]["mat_01"]
        material["compatible_geometry_ids"] = []
        bindings, report = self.run_binding()
        self.assertEqual(bindings["instances"], [])
        self.assertIn("material_geometry_incompatible", report["objects"]["chair"]["candidates"][0]["materials"][0]["reasons"])
        material["compatible_geometry_ids"] = ["chair_01"]
        material["maps"]["base_color"]["color_space"] = "Non-Color"
        _, report = self.run_binding()
        self.assertIn("base_color_space_invalid", report["objects"]["chair"]["candidates"][0]["materials"][0]["reasons"])

    def test_manifest_and_material_provenance_must_match_actual_files(self):
        path = self.root / "manifest.json"
        original = path.read_bytes()
        path.write_bytes(b"changed upstream manifest")
        bindings, report = self.run_binding()
        self.assertEqual(bindings["instances"], [])
        self.assertIn("production_manifest_hash_invalid", self.rejected_reasons(report))
        path.write_bytes(original)
        self.data["material_variants"]["mat_01"]["audit_provenance"]["manifest_sha256"] = "a" * 64
        bindings, report = self.run_binding()
        self.assertEqual(bindings["instances"], [])
        self.assertIn("material_manifest_mismatch", report["objects"]["chair"]["candidates"][0]["materials"][0]["reasons"])

    def test_repeat_expansion_and_pair_diversity_are_reproducible(self):
        self.object()["repeat"] = [2, 2]
        self.data["assets"].append(self.asset("chair_02"))
        self.data["material_variants"]["mat_01"]["compatible_geometry_ids"].append("chair_02")
        first, report = self.run_binding({"diversity": True, "seed": 14})
        second, second_report = self.run_binding({"diversity": True, "seed": 14})
        self.assertEqual(first, second)
        self.assertEqual(report, second_report)
        self.assertEqual([entry["object_id"] for entry in first["instances"]],
                         ["chair__0_0", "chair__0_1", "chair__1_0", "chair__1_1"])
        records = list(report["objects"].values())
        self.assertNotEqual(records[0]["geometry_variant_id"], records[1]["geometry_variant_id"])
        self.assertEqual([record["reused"] for record in records], [False, False, True, True])
        self.assertEqual(records[2]["selection_reason"], "variant_pool_exhausted")
        self.assertEqual(records[3]["position_m"], [1, 3, 0])

    def test_expanded_id_collision_is_rejected(self):
        self.object()["repeat"] = [2, 1]
        collision = copy.deepcopy(self.object())
        collision.update(id="chair__0_0", repeat=[1, 1])
        self.bundle["shared_scene"]["base_scene"]["objects"].append(collision)
        with self.assertRaisesRegex(ValueError, "expanded base object ids collide"):
            self.run_binding()

    def test_catalog_path_input_and_hash_report(self):
        path = self.root / "catalog.json"
        path.write_text(json.dumps(self.data))
        first, report = generate_bindings(self.bundle, path)
        self.assertEqual(len(first["instances"]), 1)
        for field in ("bundle_sha256", "catalog_sha256"):
            self.assertEqual(len(report[field]), 64)
        self.object()["rotation_z_deg"] = 5
        _, changed = generate_bindings(self.bundle, path)
        self.assertNotEqual(changed["bundle_sha256"], report["bundle_sha256"])

    def test_invalid_options_are_rejected(self):
        cases = [{"unknown": True}, {"seed": True}, {"seed": -1}, {"dimension_tolerance": float("nan")},
                 {"min_uniform_scale": 2, "max_uniform_scale": 1}, {"diversity": 1},
                 {"category_map": {"unknown": {"families": ["x"]}}},
                 {"category_map": {"chair": {"families": ["x", "x"]}}},
                 {"category_map": {"chair": {"categories": ["../chair"]}}},
                 {"category_map": {"chair": {}}}, {"floor_tolerance_m": .03}]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                normalize_options(options)


if __name__ == "__main__":
    unittest.main()
