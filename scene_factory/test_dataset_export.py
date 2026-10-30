"""Formal dataset boundaries with real runner/review provenance, mocked inference/render."""
import copy
import hashlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from batch import atomic_json, canonical_bytes, file_sha256, read_json
import batch_multiview as runner
from dataset_export import DatasetExportError, export_dataset
import dataset_export
from multiview_spec import resolve_sources, validate
from test_batch_multiview import bundle_for
import vision_multiview as vision


class DatasetExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "batch"
        self.destination = self.root / "dataset_v1"
        self.inputs = []
        for index in range(2):
            path = self.root / f"source_{index}.png"
            Image.new("RGB", (80, 45), (50 + index * 40, 80, 120)).save(path)
            self.inputs.append(path)
        self.args = SimpleNamespace(output_dir=self.output, max_edge=1280, samples=64,
                                    device="CPU", max_repairs=0, effort="low", vision_timeout=20,
                                    render_timeout=20, no_review=False, blender="mock-blender")

    def analyze(self, scene_id, views, path, **kwargs):
        bundle = bundle_for(scene_id, views)
        bundle["shared_scene"]["textures"] = [{"id": "test_crop", "quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
        bundle["texture_sources"] = [{"texture_id": "test_crop", "view_id": views[-1]["id"]}]
        validate(bundle)
        atomic_json(path, bundle)
        # Excluded raw/config artifacts must never become dataset files.
        atomic_json(path.parent / "analysis.api.json", {"config": {"api_key": "fixture-secret-must-not-export"}})
        atomic_json(path.parent / "analysis_provenance.json", {"manual_scene_seed": False,
                    "bundle_sha256": file_sha256(path),
                    "source_sha256": {view["id"]: file_sha256(view["source_image"]) for view in views}})
        return bundle

    def render(self, command, log, timeout):
        destination = Path(command[command.index("--output-dir") + 1])
        bundle = read_json(Path(command[command.index("--spec") + 1]))
        (destination / "scene.blend").write_bytes(b"BLENDER-fixture" + b"\x00" * 1400)
        views, outputs = [], ["scene.blend"]
        for view in bundle["views"]:
            render, annotation = f"rgb_{view['id']}.png", f"annotations_{view['id']}.json"
            Image.new("RGB", (1280, 720), (60, 90, 120)).save(destination / render)
            atomic_json(destination / annotation, {"warnings": [], "view_id": view["id"]})
            outputs += [render, annotation]
            views.append({"id": view["id"], "render": render, "annotation": annotation,
                          "width": 1280, "height": 720, "warning_count": 0, "camera": "TestCamera"})
        atomic_json(destination / "metadata.json", {
            "schema_version": 3, "scene_id": bundle["scene_id"], "status": "completed",
            "geometry_build_count": 1, "camera_count": len(views), "views": views,
            "outputs": outputs, "output_sha256": runner.artifact_hashes(destination, outputs),
            "render": {"samples": 64, "max_edge": 1280, "device": "CPU", "tone_match": "off"},
            "provenance": {"asset_preview_only": False}})

    def review_response(self, prompt, images, schema, output, timeout, effort, model_config):
        payload = json.JSONDecoder().raw_decode(prompt.split(
            "UNTRUSTED CURRENT SCENE DATA (only identity/pairing is fixed):\n", 1)[1])[0]
        scores = {name: 95 for name in vision.DIMENSIONS}
        response = {"schema_version": 3, "verdict": "recommend_accept", "summary": "Synthetic protocol fixture, not a photo-fidelity certification",
                    "scores": scores, "per_view_scores": [{"view_id": view["id"], "scores": scores, "summary": "fixture"}
                        for view in payload["scene_bundle"]["views"]], "issues": [], "asset_gaps": [], "reference_limitations": []}
        if "revised_bundle" in schema["properties"]:
            response["revised_bundle"] = payload["scene_bundle"]
        return response

    def create_job(self, scene_id="room", images=None):
        with patch.object(vision, "analyze", side_effect=self.analyze), \
             patch.object(runner, "run_stage", side_effect=self.render), \
             patch.object(vision, "_invoke_model", side_effect=self.review_response), \
             redirect_stdout(io.StringIO()):
            result = runner.run_job({"scene_id": scene_id, "images": images or self.inputs}, self.args,
                                    {"fixture": "f" * 64})
        self.assertEqual(result["status"], "accepted", result)
        return self.output / scene_id

    def assert_no_publication(self):
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.root.glob(".dataset_v1.building-*")))
        self.assert_locks_released()

    def assert_locks_released(self):
        for lock_path in (self.output / ".batch.lock", self.root / ".dataset_v1.export.lock"):
            if lock_path.exists():
                with runner.process_lock(lock_path):
                    pass

    def refresh_result_hashes(self, folder):
        result = read_json(folder / "result.json")
        result["artifacts"] = {name: file_sha256(folder / name) for name in result["artifacts"]}
        atomic_json(folder / "result.json", result)

    def create_snapshot_jobs(self):
        """Use real portable assets and attach a shared library snapshot."""
        from asset_manifest import build_catalog
        from test_asset_manifest import AssetManifestTests
        fixture = AssetManifestTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        catalog_path = self.root / "library.json"
        data = build_catalog(fixture.root.parent, catalog_path)
        bindings_path = self.root / "bindings.json"
        atomic_json(bindings_path, {"schema_version": 1, "instances": [{"object_id": "sofa",
            "asset_id": data["assets"][0]["asset_id"],
            "material_variant_id": next(iter(data["material_variants"]))}]})
        self.args.asset_catalog = catalog_path
        self.args.asset_bindings = bindings_path
        self.args.asset_preview = False
        folders = [self.create_job(identifier) for identifier in ("room_a", "room_b")]
        extra = self.root / "unselected_library_asset.bin"
        extra.write_bytes(b"immutable unused asset")
        hashes = read_json(folders[0] / "job.json")["asset_library"]["input_sha256"]
        hashes[str(extra)] = file_sha256(extra)
        snapshot = {"schema_version": 1, "catalog": str(catalog_path),
                    "catalog_sha256": file_sha256(catalog_path), "input_sha256": hashes}
        snapshot["snapshot_sha256"] = hashlib.sha256(canonical_bytes(snapshot)).hexdigest()
        snapshot_path = self.output / runner.ASSET_SNAPSHOT_FILE
        atomic_json(snapshot_path, snapshot)
        reference = {"path": str(snapshot_path), "sha256": file_sha256(snapshot_path)}
        for folder in folders:
            job = read_json(folder / "job.json")
            del job["asset_library"]["input_sha256"]
            job["asset_library"]["input_snapshot"] = reference
            atomic_json(folder / "job.json", job)
            iteration = folder / "iteration_00"
            state = read_json(iteration / "render_state.json")
            state["fingerprint"] = hashlib.sha256(canonical_bytes({
                "bundle": read_json(iteration / "scene_bundle.json"),
                "job": job, "asset_selection": {}})).hexdigest()
            atomic_json(iteration / "render_state.json", state)
            self.refresh_result_hashes(folder)
        return snapshot_path, extra

    def test_snapshot_export_hashes_library_once_per_pass_and_never_reuses_across_exports(self):
        from asset_catalog import AssetCatalog
        snapshot_path, extra = self.create_snapshot_jobs()
        hashed_extra = 0
        original_digest = AssetCatalog.stable_file_digest
        def hashing(path):
            nonlocal hashed_extra
            if Path(path) == extra:
                hashed_extra += 1
            return original_digest(path)
        with patch.object(AssetCatalog, "stable_file_digest", new=staticmethod(hashing)):
            manifest = export_dataset(self.output, self.destination)
        self.assertEqual(manifest["scene_count"], 2)
        self.assertEqual(hashed_extra, 2)  # one complete check in each pass
        extra.write_bytes(b"changed after the previous successful export")
        second = self.root / "dataset_v2"
        with self.assertRaisesRegex(DatasetExportError, "[Aa]sset library input changed"):
            export_dataset(self.output, second)
        self.assertFalse(second.exists())
        self.assertFalse(list(self.root.glob(".dataset_v2.building-*")))

    def test_shared_catalog_is_parsed_once_per_export_not_cached_across_exports(self):
        import asset_catalog
        self.create_snapshot_jobs()
        with patch.object(asset_catalog, "load_catalog", wraps=asset_catalog.load_catalog) as load:
            first = export_dataset(self.output, self.destination)
            self.assertEqual(first["scene_count"], 2)
            self.assertEqual(load.call_count, 1)
            second = export_dataset(self.output, self.root / "dataset_v2")
            self.assertEqual(second["scene_count"], 2)
            self.assertEqual(load.call_count, 2)

    def test_cached_catalog_change_after_scene_verification_prevents_publication(self):
        self.create_snapshot_jobs()
        original_export = dataset_export._export_scene
        exported = []

        def exporting(record, destination, **kwargs):
            if exported:
                # Both scenes passed source_check; mutate just before the
                # second asset copy to exercise the parsed-catalog cache hit.
                catalog = Path(record["job"]["asset_library"]["catalog"])
                catalog.write_bytes(catalog.read_bytes() + b" ")
            entry = original_export(record, destination, **kwargs)
            exported.append(record["folder"].name)
            return entry

        with patch.object(dataset_export, "_export_scene", new=exporting):
            with self.assertRaisesRegex(DatasetExportError, "Asset catalog changed"):
                export_dataset(self.output, self.destination)
        self.assertEqual(exported, ["room_a"])
        self.assert_no_publication()

    def test_fresh_final_snapshot_guard_rejects_unselected_asset_changed_during_copy(self):
        snapshot_path, extra = self.create_snapshot_jobs()
        original_export = dataset_export._export_scene
        def exporting(record, destination, **kwargs):
            entry = original_export(record, destination, **kwargs)
            extra.write_bytes(b"changed while exporting selected assets")
            return entry
        with patch.object(dataset_export, "_export_scene", new=exporting):
            with self.assertRaisesRegex(DatasetExportError, "[Aa]sset library input changed"):
                export_dataset(self.output, self.destination)
        self.assert_no_publication()

    def test_snapshot_reference_changed_during_copy_prevents_publication(self):
        snapshot_path, extra = self.create_snapshot_jobs()
        original_export = dataset_export._export_scene
        def exporting(record, destination, **kwargs):
            entry = original_export(record, destination, **kwargs)
            snapshot_path.write_bytes(snapshot_path.read_bytes() + b" ")
            return entry
        with patch.object(dataset_export, "_export_scene", new=exporting):
            with self.assertRaisesRegex(DatasetExportError, "snapshot changed"):
                export_dataset(self.output, self.destination)
        self.assert_no_publication()

    def test_portable_multiview_package_excludes_raw_api_files_and_uses_all_accepted_jobs(self):
        first = self.create_job("room_a")
        self.create_job("room_b", self.inputs[:1])
        # A subset from the last invocation must not hide older accepted jobs.
        atomic_json(self.output / "accepted.json", [])
        atomic_json(self.output / "summary.json", {"scenes": []})
        manifest = export_dataset(self.output, self.destination)
        self.assertEqual(manifest["scene_count"], 2)
        self.assertEqual(manifest["view_count"], 3)
        self.assertEqual([s["scene_id"] for s in manifest["scenes"]], ["room_a", "room_b"])
        moved = self.root / "moved_dataset"
        self.destination.rename(moved)
        shutil.rmtree(self.output)
        for source in self.inputs:
            source.unlink()
        for entry in manifest["scenes"]:
            scene = moved / entry["directory"]
            bundle = validate(read_json(scene / "scene_bundle.json"))
            self.assertTrue(all(p.is_file() for p in resolve_sources(bundle, scene / "scene_bundle.json").values()))
            for name, digest in entry["files_sha256"].items():
                self.assertEqual(file_sha256(scene / name), digest)
            crops = read_json(scene / "textures/manifest.json")["textures"]
            self.assertEqual(len(crops), 1)
            self.assertTrue((scene / "textures" / crops[0]["path"]).is_file())
            for file in scene.rglob("*.json"):
                self.assertNotIn("fixture-secret-must-not-export", file.read_text())
            self.assertFalse((scene / "job.json").exists())
            self.assertFalse((scene / "analysis.api.json").exists())
            self.assertEqual(read_json(scene / "evidence/visual_review.json")["revised_bundle"], bundle)

    def test_unreviewed_or_failed_scenes_are_not_formal_data(self):
        folder = self.create_job()
        result = read_json(folder / "result.json")
        for status in ("generated_unreviewed", "needs_review", "review_failed", "failed"):
            with self.subTest(status=status):
                result["status"] = status
                atomic_json(folder / "result.json", result)
                with self.assertRaisesRegex(DatasetExportError, "No accepted"):
                    export_dataset(self.output, self.destination)
                self.assert_no_publication()

    def test_changed_scene_or_original_prevents_publication(self):
        folder = self.create_job()
        scene = folder / "iteration_00/scene.blend"
        original = scene.read_bytes()
        scene.write_bytes(original + b"tampered")
        with self.assertRaisesRegex(DatasetExportError, "changed or missing"):
            export_dataset(self.output, self.destination)
        self.assert_no_publication()
        scene.write_bytes(original)
        self.inputs[0].write_bytes(b"source changed")
        with self.assertRaisesRegex(DatasetExportError, "Source photo changed"):
            export_dataset(self.output, self.destination)
        self.assert_no_publication()

    def test_accepted_flag_without_current_review_is_rejected(self):
        folder = self.create_job()
        result = read_json(folder / "result.json")
        result["final_review_iteration"] = "iteration_01"
        atomic_json(folder / "result.json", result)
        with self.assertRaisesRegex(DatasetExportError, "Final review"):
            export_dataset(self.output, self.destination)
        self.assert_no_publication()

    def test_newer_unfinished_iteration_invalidates_earlier_acceptance(self):
        folder = self.create_job()
        (folder / "iteration_01").mkdir()
        with self.assertRaisesRegex(DatasetExportError, "newer iteration"):
            export_dataset(self.output, self.destination)
        self.assert_no_publication()

    def test_preview_marker_is_rejected_even_if_result_claims_accepted(self):
        folder = self.create_job()
        job = read_json(folder / "job.json")
        job["asset_library"] = {"preview": True}
        atomic_json(folder / "job.json", job)
        self.refresh_result_hashes(folder)
        with self.assertRaisesRegex(DatasetExportError, "preview"):
            export_dataset(self.output, self.destination)
        self.assert_no_publication()

    def test_review_tampering_or_omitted_provenance_cannot_be_hidden_by_result_hashes(self):
        folder = self.create_job()
        review_path = folder / "iteration_00/review.json"
        review = read_json(review_path)
        review["summary"] += " modified after review"
        atomic_json(review_path, review)
        self.refresh_result_hashes(folder)
        with self.assertRaisesRegex(DatasetExportError, "Cached review changed"):
            export_dataset(self.output, self.destination)
        self.assert_no_publication()

    def test_fresh_gate_rejects_forged_high_level_acceptance(self):
        folder = self.create_job()
        review_path = folder / "iteration_00/review.json"
        provenance_path = folder / "iteration_00/review_provenance.json"
        review, provenance = read_json(review_path), read_json(provenance_path)
        review["scores"]["realism"] = 20
        review["per_view_scores"][0]["scores"]["realism"] = 20
        atomic_json(review_path, review)
        provenance["review_sha256"] = file_sha256(review_path)
        atomic_json(provenance_path, provenance)
        self.refresh_result_hashes(folder)
        with self.assertRaisesRegex(DatasetExportError, "Fresh quality gate"):
            export_dataset(self.output, self.destination)
        self.assert_no_publication()

    def test_locked_scene_existing_destination_and_path_escape_are_rejected(self):
        folder = self.create_job()
        (folder / ".lock").write_text("running")
        with self.assertRaisesRegex(DatasetExportError, "locked"):
            export_dataset(self.output, self.destination)
        self.assert_no_publication()
        (folder / ".lock").unlink()
        self.destination.mkdir()
        marker = self.destination / "keep.txt"
        marker.write_text("existing")
        with self.assertRaisesRegex(DatasetExportError, "already exists"):
            export_dataset(self.output, self.destination)
        self.assertEqual(marker.read_text(), "existing")
        with self.assertRaisesRegex(DatasetExportError, "valid scene identifiers"):
            export_dataset(self.output, self.root / "other", scene_ids=["../room"])
        result = read_json(folder / "result.json")
        result["artifacts"]["../outside.bin"] = "a" * 64
        atomic_json(folder / "result.json", result)
        with self.assertRaisesRegex(DatasetExportError, "escapes"):
            export_dataset(self.output, self.root / "escaped")

    def test_symlink_jobs_and_artifacts_are_rejected(self):
        folder = self.create_job()
        blend = folder / "iteration_00/scene.blend"
        outside = self.root / "outside.blend"
        blend.rename(outside)
        blend.symlink_to(outside)
        with self.assertRaisesRegex(DatasetExportError, "Symlink"):
            export_dataset(self.output, self.destination)
        self.assert_no_publication()

    def test_inputs_changing_during_copy_leave_no_partial_dataset(self):
        folder = self.create_job()
        original_copy = dataset_export._copy
        changed = False
        def copying(source, target, expected=None):
            nonlocal changed
            digest = original_copy(source, target, expected)
            if not changed:
                changed = True
                self.inputs[0].write_bytes(b"changed during export")
            return digest
        with patch.object(dataset_export, "_copy", side_effect=copying):
            with self.assertRaisesRegex(DatasetExportError, "Source photo changed|input changed"):
                export_dataset(self.output, self.destination)
        self.assert_no_publication()

    def test_large_export_holds_only_one_verified_scene_record_at_a_time(self):
        output = self.root / "many_jobs"
        output.mkdir()
        identifiers = [f"room_{index:05d}" for index in range(10_000)]
        for identifier in identifiers:
            (output / identifier).mkdir()

        class TrackedRecord(dict):
            active = 0
            peak = 0

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                type(self).active += 1
                type(self).peak = max(type(self).peak, type(self).active)

            def __del__(self):
                type(self).active -= 1

        calls, exported = [], set()

        def verify(folder, *, snapshot_guard=None):
            identifier = folder.name
            calls.append(("verify", identifier))
            return TrackedRecord(folder=folder, result_sha256=identifier * 64)

        def export_scene(record, destination, **kwargs):
            identifier = record["folder"].name
            calls.append(("export", identifier))
            exported.add(identifier)
            destination.mkdir()
            return {"scene_id": identifier, "views": [{"id": "view_01"}]}

        destination = self.root / "many_dataset"
        with patch.object(dataset_export, "_verified_job", new=verify), \
                patch.object(dataset_export, "_export_scene", new=export_scene):
            manifest = dataset_export._export_snapshot(output, destination, identifiers,
                                                       include_scene_entries=False)

        self.assertEqual(TrackedRecord.peak, 1)
        self.assertEqual(TrackedRecord.active, 0)
        self.assertEqual(exported, set(identifiers))
        self.assertEqual(manifest["scene_count"], len(identifiers))
        self.assertEqual(manifest["view_count"], len(identifiers))
        self.assertNotIn("scenes", manifest)
        self.assertEqual(manifest["manifest_path"], str(destination / "dataset_manifest.json"))
        disk_manifest = read_json(destination / "dataset_manifest.json")
        self.assertEqual([entry["scene_id"] for entry in disk_manifest["scenes"]], identifiers)
        self.assertEqual(calls[:4], [("verify", identifiers[0]), ("export", identifiers[0]),
                                     ("verify", identifiers[1]), ("export", identifiers[1])])
        self.assertEqual(calls[2 * len(identifiers):], [("verify", identifier) for identifier in identifiers])

    def test_export_revalidation_rejects_a_scene_commit_changed_mid_export(self):
        output = self.root / "changed_job"
        output.mkdir()
        identifiers = ["room"]
        folder = output / "room"
        folder.mkdir()
        result_path = folder / "result.json"
        result_path.write_text("first", encoding="utf-8")

        def verify(scene_folder, *, snapshot_guard=None):
            return {"folder": scene_folder, "result_sha256": file_sha256(scene_folder / "result.json")}

        def export_scene(record, destination, **kwargs):
            destination.mkdir()
            (record["folder"] / "result.json").write_text("changed", encoding="utf-8")
            return {"scene_id": record["folder"].name, "views": []}

        destination = self.root / "changed_dataset"
        with patch.object(dataset_export, "_verified_job", new=verify), \
                patch.object(dataset_export, "_export_scene", new=export_scene):
            with self.assertRaisesRegex(DatasetExportError, "result changed during export"):
                dataset_export._export_snapshot(output, destination, identifiers,
                                                include_scene_entries=False)
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob(".changed_dataset.building-*")))

    def test_active_batch_or_destination_lock_blocks_export(self):
        self.create_job()
        for lock_path in (self.output / ".batch.lock", self.root / ".dataset_v1.export.lock"):
            with self.subTest(lock=lock_path.name):
                with runner.process_lock(lock_path):
                    with self.assertRaisesRegex(DatasetExportError, "locked by a running process"):
                        export_dataset(self.output, self.destination)
                self.assert_no_publication()

    def test_failed_export_retains_reusable_locks_and_can_be_retried(self):
        folder = self.create_job()
        marker = folder / ".lock"
        marker.write_text("running")
        with self.assertRaisesRegex(DatasetExportError, "locked"):
            export_dataset(self.output, self.destination)
        self.assert_no_publication()
        self.assertTrue((self.root / ".dataset_v1.export.lock").is_file())
        self.assertTrue((self.output / ".batch.lock").is_file())
        marker.unlink()
        manifest = export_dataset(self.output, self.destination)
        self.assertEqual(manifest["scene_count"], 1)
        self.assert_locks_released()

    def test_successful_export_releases_but_does_not_unlink_stable_locks(self):
        self.create_job()
        export_dataset(self.output, self.destination)
        self.assertTrue((self.root / ".dataset_v1.export.lock").is_file())
        self.assertTrue((self.output / ".batch.lock").is_file())
        self.assert_locks_released()

    def test_compact_export_return_keeps_full_manifest_on_disk(self):
        self.create_job()
        manifest = export_dataset(self.output, self.destination, include_scene_entries=False)
        self.assertEqual(manifest["scene_count"], 1)
        self.assertNotIn("scenes", manifest)
        self.assertEqual(manifest["manifest_path"], str(self.destination / "dataset_manifest.json"))
        saved = read_json(self.destination / "dataset_manifest.json")
        self.assertEqual([entry["scene_id"] for entry in saved["scenes"]], ["room"])

    def test_destination_appearing_at_publication_cannot_be_overwritten(self):
        self.create_job()
        publish = dataset_export._publish
        def concurrent_creator(staging, destination):
            destination.mkdir()
            # Even an empty concurrently-created destination must survive.
            publish(staging, destination)
        with patch.object(dataset_export, "_publish", side_effect=concurrent_creator):
            with self.assertRaises(DatasetExportError):
                export_dataset(self.output, self.destination)
        self.assertTrue(self.destination.is_dir())
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertFalse(list(self.root.glob(".dataset_v1.building-*")))
        self.assert_locks_released()

    def test_exported_selected_assets_still_validate_after_relocation(self):
        from asset_manifest import build_catalog, validate_manifest
        from asset_catalog import load_catalog
        from test_asset_manifest import AssetManifestTests
        fixture = AssetManifestTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        catalog_path = self.root / "library.json"
        catalog_data = build_catalog(fixture.root.parent, catalog_path)
        bindings_path = self.root / "bindings.json"
        atomic_json(bindings_path, {"schema_version": 1, "instances": [{"object_id": "sofa", "asset_id": catalog_data["assets"][0]["asset_id"],
            "material_variant_id": next(iter(catalog_data["material_variants"]))}]})
        self.args.asset_catalog = catalog_path
        self.args.asset_bindings = bindings_path
        self.args.asset_preview = False
        folder = self.create_job()
        export_dataset(self.output, self.destination)
        moved = self.root / "relocated"
        self.destination.rename(moved)
        shutil.rmtree(fixture.root.parent)
        scene = moved / "scenes/room"
        catalog = load_catalog(scene / "asset_catalog.json")
        for asset in catalog.assets.values():
            self.assertTrue(catalog.production_readiness(asset)[0])
            self.assertTrue(validate_manifest(catalog.resolve_path(asset["manifest_file"]), require_production=True)["production_ready"])
        for material_id in catalog.materials:
            self.assertTrue(catalog.verify_material_hashes(material_id))
        self.assertEqual(read_json(scene / "rebuild.json")["asset_catalog"], "asset_catalog.json")


if __name__ == "__main__":
    unittest.main()
