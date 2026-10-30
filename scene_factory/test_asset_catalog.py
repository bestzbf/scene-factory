import hashlib
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from . import asset_catalog as catalog_module
    from .asset_catalog import AssetCatalog, CatalogError, audit_draft_to_catalog, load_catalog, sha256_file
except ImportError:
    import asset_catalog as catalog_module
    from asset_catalog import AssetCatalog, CatalogError, audit_draft_to_catalog, load_catalog, sha256_file


class AssetCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "assets"
        (self.root / "chair" / "textures").mkdir(parents=True)
        self.model = self.root / "chair" / "chair.fbx"
        self.model.write_bytes(b"fixture-fbx")
        (self.root / "chair" / "textures" / "base.png").write_bytes(b"texture")
        self.model_hash = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.draft = Path(self.tmp.name) / "draft.json"
        raw = {
            "asset_root": str(self.root), "assets": [{
                "asset_id": "chair_01", "source_name": "Chair",
                "model_file": "chair/chair.fbx", "model_sha256": self.model_hash,
                "semantic_description": {"category": "chair", "scope": "single_prop", "tags": ["chair"], "issues": []},
                "geometry_measured": {"dimensions_units": [1, 2, 3]},
                "physical_scale": {"metres_per_import_unit": None},
                "orientation": {"semantic_front_axis": None},
                "placement": {"support_type": "floor"},
                "materials": {"maps": {"Base Color": {"path": "chair/textures/base.png", "color_space": "sRGB"},
                                           "Normal": {"path": None, "color_space": "Non-Color"}}, "connection_test": "local"},
                "quality": {"production_placement_ready": False, "complete_for_floor_placement": True, "issues": []},
                "provenance": {"generation_batch": "test"}
            }]
        }
        self.draft.write_text(json.dumps(raw), encoding="utf-8")

    def test_draft_conversion_preserves_unknowns_and_relative_paths(self):
        out = Path(self.tmp.name) / "catalog.json"
        catalog = audit_draft_to_catalog(self.draft, out)
        self.assertEqual(catalog["schema_version"], "asset_catalog_v1")
        self.assertEqual(catalog["assets"][0]["family_id"], "chair")
        self.assertIsNone(catalog["assets"][0]["meters_per_unit"])
        self.assertIsNone(catalog["assets"][0]["front_axis"])
        material = next(iter(catalog["material_variants"].values()))
        self.assertEqual(material["normal_convention_status"], "unknown")
        self.assertIn("normal_convention_unknown", material["audit_flags"])
        loaded = load_catalog(out)
        self.assertEqual(loaded.resolve_path("chair/chair.fbx"), self.model.resolve())
        self.assertEqual(loaded.source_sha256, sha256_file(out))
        self.assertTrue(loaded.verify_hash("chair_01"))

    def test_path_escape_and_hash_mismatch_rejected(self):
        bad = json.loads(self.draft.read_text())
        bad["assets"][0]["model_file"] = "../outside.fbx"
        path = Path(self.tmp.name) / "bad.json"; path.write_text(json.dumps(bad))
        with self.assertRaises(CatalogError): audit_draft_to_catalog(path, Path(self.tmp.name) / "x.json")
        bad["assets"][0]["model_file"] = "chair/chair.fbx"
        bad["assets"][0]["model_sha256"] = "0" * 64
        path.write_text(json.dumps(bad))
        with self.assertRaises(CatalogError): audit_draft_to_catalog(path, Path(self.tmp.name) / "x.json")

    def test_preview_resolver_is_deterministic_and_records_reuse(self):
        out = Path(self.tmp.name) / "catalog.json"
        audit_draft_to_catalog(self.draft, out)
        catalog = load_catalog(out)
        resolver = catalog.resolver(seed=7)
        with self.assertRaises(CatalogError): resolver.resolve("chair", "scene", "obj")
        first = resolver.resolve("chair", "scene", "obj", allow_preview=True)
        second = resolver.resolve("chair", "scene", "obj", allow_preview=True)
        self.assertEqual(first["geometry_variant_id"], second["geometry_variant_id"])
        self.assertEqual(first["material_variant_id"], second["material_variant_id"])
        self.assertFalse(first["reused"])
        self.assertFalse(second["reused"])
        other = resolver.resolve("chair", "scene", "obj2", allow_preview=True)
        self.assertTrue(other["reused"])
        self.assertGreaterEqual(other["reuse_count"], 2)

    def test_production_gate_rejects_unknown_scale_front_and_classification(self):
        out = Path(self.tmp.name) / "catalog.json"
        audit_draft_to_catalog(self.draft, out)
        catalog = load_catalog(out)
        ready, reasons = catalog.production_readiness("chair_01")
        self.assertFalse(ready)
        self.assertIn("asset_not_approved", reasons)
        self.assertIn("unknown_scale", reasons)
        self.assertIn("unknown_semantic_front", reasons)

    def make_ready_catalog(self, variants=2):
        data = audit_draft_to_catalog(self.draft, Path(self.tmp.name) / "ready.json")
        asset = data["assets"][0]
        asset.update(status="approved", meters_per_unit=0.5, front_axis="+Y")
        original_material = next(iter(data["material_variants"].values()))
        for index in range(1, variants):
            extra = copy.deepcopy(asset)
            extra.update(asset_id=f"chair_{index + 1}", geometry_variant_id=f"chair_geometry_{index + 1}")
            material = copy.deepcopy(original_material)
            material_id = f"material_{index + 1}"
            material.update(material_variant_id=material_id, compatible_geometry_ids=[extra["geometry_variant_id"]])
            extra["material_variant_ids"] = [material_id]
            data["assets"].append(extra)
            data["material_variants"][material_id] = material
        return AssetCatalog(data)

    def test_different_geometry_pairs_used_before_pool_repeats(self):
        catalog = self.make_ready_catalog(2)
        one = catalog.resolver(seed=17)
        results = [one.resolve("chair", "scene", f"object_{i}") for i in range(3)]
        self.assertNotEqual(results[0]["geometry_variant_id"], results[1]["geometry_variant_id"])
        self.assertFalse(results[1]["reused"])
        self.assertTrue(results[2]["reused"])
        self.assertEqual(results[2]["selection_reason"], "variant_pool_exhausted")
        two = catalog.resolver(seed=17)
        self.assertEqual(results, [two.resolve("chair", "scene", f"object_{i}") for i in range(3)])
        self.assertEqual(one.resolve("chair", "scene", "object_0"), results[0])
        fresh_scene = one.resolve("chair", "scene_two", "object_0")
        self.assertFalse(fresh_scene["reused"])

    def test_material_variants_cannot_cross_incompatible_uv_geometries(self):
        catalog = self.make_ready_catalog(2)
        assets = list(catalog.assets.values())
        with self.assertRaisesRegex(CatalogError, "incompatible"):
            catalog.resolver().resolve("chair", "scene", "o", geometry_variant_id=assets[0]["geometry_variant_id"],
                material_variant_id=assets[1]["material_variant_ids"][0])
        result = catalog.resolver().resolve("chair", "scene", "o", geometry_variant_id=assets[0]["geometry_variant_id"],
            material_variant_id=assets[0]["material_variant_ids"][0])
        self.assertEqual(result["asset_id"], assets[0]["asset_id"])
        self.assertEqual(result["selection_reason"], "explicit_pair")

    def test_multiple_materials_on_one_geometry_avoid_pair_repetition(self):
        data = self.make_ready_catalog(1).data
        asset = data["assets"][0]
        material = copy.deepcopy(next(iter(data["material_variants"].values())))
        material["material_variant_id"] = "blue_material"
        data["material_variants"]["blue_material"] = material
        asset["material_variant_ids"].append("blue_material")
        resolver = AssetCatalog(data).resolver()
        a = resolver.resolve("chair", "scene", "a")
        b = resolver.resolve("chair", "scene", "b")
        self.assertEqual(a["geometry_variant_id"], b["geometry_variant_id"])
        self.assertNotEqual(a["material_variant_id"], b["material_variant_id"])

    def test_production_gate_rechecks_cached_draft(self):
        data = audit_draft_to_catalog(self.draft, Path(self.tmp.name) / "catalog.json")
        resolver = AssetCatalog(data).resolver()
        resolver.resolve("chair", "scene", "obj", allow_preview=True)
        with self.assertRaises(CatalogError):
            resolver.resolve("chair", "scene", "obj")

    def test_completeness_classification_and_symmetry_gate(self):
        catalog = self.make_ready_catalog(1)
        asset = catalog.get_asset("chair_01")
        asset.update(front_axis=None, symmetric_front=True)
        self.assertTrue(catalog.production_readiness(asset)[0])
        asset["category"] = None
        self.assertIn("classification_unknown", catalog.production_readiness(asset)[1])
        asset["quality"]["complete_for_floor_placement"] = False
        self.assertIn("incomplete_geometry", catalog.production_readiness(asset)[1])

    def test_file_replacement_is_rejected_for_model_and_texture(self):
        catalog = self.make_ready_catalog(1)
        texture = self.root / "chair/textures/base.png"
        texture.write_bytes(b"unexpected replacement")
        with self.assertRaisesRegex(CatalogError, "texture hash"):
            catalog.resolver().resolve("chair", "scene", "o", allow_preview=True)
        self.model.write_bytes(b"unexpected model")
        with self.assertRaisesRegex(CatalogError, "model hash"):
            catalog.verify_hash("chair_01")

    def test_pinned_batch_hashes_are_reused_but_stat_changes_are_rehashed(self):
        catalog = self.make_ready_catalog(1)
        material_id = catalog.assets["chair_01"]["material_variant_ids"][0]
        paths = [self.model, self.root / "chair/textures/base.png"]
        hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        catalog.pin_snapshot_hashes(hashes)
        with patch.object(catalog_module, "sha256_file", wraps=sha256_file) as digest:
            for _ in range(4):
                catalog.verify_hash("chair_01")
                catalog.verify_material_hashes(material_id)
            self.assertEqual(digest.call_count, 0)
            self.model.write_bytes(b"replacement with new stat")
            with self.assertRaisesRegex(CatalogError, "model hash"):
                catalog.verify_hash("chair_01")
            self.assertEqual(digest.call_count, 1)

    def test_duplicate_ids_and_json_keys_rejected(self):
        data = self.make_ready_catalog(1).data
        data["assets"].append(copy.deepcopy(data["assets"][0]))
        with self.assertRaisesRegex(CatalogError, "unique"):
            AssetCatalog(data)
        path = Path(self.tmp.name) / "duplicate.json"
        path.write_text('{"root":"a","root":"b"}')
        with self.assertRaisesRegex(CatalogError, "duplicate JSON"):
            load_catalog(path)

    def test_pinning_rejects_file_changed_before_pin(self):
        catalog = self.make_ready_catalog(1)
        path = self.model.resolve()
        self.model.write_bytes(b"replacement before pin")
        with self.assertRaisesRegex(CatalogError, "changed after the batch snapshot"):
            catalog.pin_snapshot_hashes({str(path): self.model_hash})
        self.assertIsNone(catalog._snapshot_file_hashes)

    def test_pinning_rejects_file_changed_during_hash(self):
        catalog = self.make_ready_catalog(1)
        path = self.model.resolve()
        def mutate_after_read(source):
            digest = sha256_file(source)
            Path(source).write_bytes(b"replacement while hashing")
            return digest
        with patch.object(catalog_module, "sha256_file", side_effect=mutate_after_read):
            with self.assertRaisesRegex(CatalogError, "changed while hashing"):
                catalog.pin_snapshot_hashes({str(path): self.model_hash})
        self.assertIsNone(catalog._snapshot_file_hashes)

    def test_pinning_rejects_stale_caller_stat_signature(self):
        catalog = self.make_ready_catalog(1)
        path = self.model.resolve()
        digest, signature = AssetCatalog.stable_file_digest(path)
        self.model.write_bytes(b"changed after caller hash")
        with self.assertRaisesRegex(CatalogError, "changed after the batch snapshot"):
            catalog.pin_snapshot_hashes({str(path): digest}, {str(path): signature})
        self.assertIsNone(catalog._snapshot_file_hashes)

    def test_force_rehashes_pinned_model_and_material(self):
        catalog = self.make_ready_catalog(1)
        material_id = catalog.assets["chair_01"]["material_variant_ids"][0]
        paths = [self.model.resolve(), (self.root / "chair/textures/base.png").resolve()]
        catalog.pin_snapshot_hashes({str(path): sha256_file(path) for path in paths})
        with patch.object(catalog_module, "sha256_file", wraps=sha256_file) as digest:
            catalog.verify_hash("chair_01", force=True)
            catalog.verify_material_hashes(material_id, force=True)
            catalog.verify_file_hash(paths[0], self.model_hash, force=True)
            self.assertEqual(digest.call_count, 3)
            catalog.verify_hash("chair_01")
            catalog.verify_material_hashes(material_id)
            self.assertEqual(digest.call_count, 3)

    def test_snapshot_rejects_file_and_parent_symlinks(self):
        catalog = self.make_ready_catalog(1)
        root = self.root.resolve()
        alias = root / "alias.fbx"
        alias.symlink_to(self.model.resolve())
        parent_alias = root / "alias_chair"
        parent_alias.symlink_to(root / "chair", target_is_directory=True)
        for path in (alias, parent_alias / "chair.fbx"):
            with self.assertRaisesRegex(CatalogError, "symlink"):
                catalog.pin_snapshot_hashes({str(path): self.model_hash})
            with self.assertRaisesRegex(CatalogError, "symlink"):
                catalog.verify_file_hash(path, self.model_hash)

    def test_pinned_root_directory_substitution_is_rejected(self):
        catalog = self.make_ready_catalog(1)
        root = self.root.resolve()
        model = self.model.resolve()
        catalog.pin_snapshot_hashes({str(model): self.model_hash})
        moved = root.with_name("moved-assets")
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(CatalogError, "symlink"):
            catalog.verify_hash("chair_01")

    def test_symlink_escape_windows_paths_and_source_overwrite_rejected(self):
        catalog = self.make_ready_catalog(1)
        outside = Path(self.tmp.name) / "outside.fbx"
        outside.write_bytes(b"outside")
        (self.root / "escape.fbx").symlink_to(outside)
        for path in ("escape.fbx", "../outside.fbx", "/tmp/outside.fbx", "C:\\file.fbx"):
            with self.assertRaises(CatalogError): catalog.resolve_path(path)
        with self.assertRaises(CatalogError):
            audit_draft_to_catalog(self.draft, self.model)
        with self.assertRaises(CatalogError):
            audit_draft_to_catalog(self.draft, self.draft)

    def test_resolution_does_not_modify_catalog_audit_and_caller_cannot_mutate_result_cache(self):
        catalog = self.make_ready_catalog(1)
        before = copy.deepcopy(catalog.data)
        resolver = catalog.resolver()
        result = resolver.resolve("chair", "scene", "object")
        result["audit_provenance"]["source_model_sha256"] = "changed"
        repeated = resolver.resolve("chair", "scene", "object")
        self.assertEqual(repeated["audit_provenance"]["source_model_sha256"], self.model_hash)
        self.assertEqual(catalog.data, before)


if __name__ == "__main__":
    unittest.main()
