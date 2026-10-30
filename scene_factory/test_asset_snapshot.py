"""Batch-wide asset snapshot guards; no Blender or provider calls."""
from __future__ import annotations

import hashlib
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import batch_multiview as runner
    from .batch import atomic_json, file_sha256
except ImportError:
    import batch_multiview as runner
    from batch import atomic_json, file_sha256


class AssetSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.photo = self.root / "photo.jpg"
        self.photo.write_bytes(b"photo")
        self.asset = self.root / "asset.glb"
        self.asset.write_bytes(b"asset")
        self.catalog = self.root / "catalog.json"
        self.catalog.write_text("{}", encoding="utf-8")
        self.snapshot = self.root / runner.ASSET_SNAPSHOT_FILE
        self.write_snapshot()
        self.job = {
            "sources": [{"path": str(self.photo),
                         "sha256": hashlib.sha256(self.photo.read_bytes()).hexdigest()}],
            "asset_library": {"input_snapshot": {
                "path": str(self.snapshot.resolve()), "sha256": file_sha256(self.snapshot)}}}

    def write_snapshot(self, missing=()):
        inputs = {str(self.catalog): file_sha256(self.catalog),
                  str(self.asset): file_sha256(self.asset)}
        inputs.update({str(path): None for path in missing})
        payload = {"schema_version": 1, "catalog": str(self.catalog),
                   "catalog_sha256": inputs[str(self.catalog)], "input_sha256": inputs}
        payload["snapshot_sha256"] = hashlib.sha256(runner.canonical_bytes(payload)).hexdigest()
        atomic_json(self.snapshot, payload)
        if hasattr(self, "job"):
            self.job["asset_library"]["input_snapshot"]["sha256"] = file_sha256(self.snapshot)

    def test_large_asset_manifest_is_hashed_once_per_batch_snapshot(self):
        from asset_catalog import AssetCatalog
        guard = runner.AssetSnapshotGuard()
        with patch.object(AssetCatalog, "stable_file_digest", wraps=AssetCatalog.stable_file_digest) as digest:
            runner.source_check(self.job, guard)
            first = digest.call_count
            runner.source_check(self.job, guard)
            self.assertEqual(digest.call_count, first)  # unchanged identity; no repeated asset reads

    def test_asset_change_is_rejected_on_fresh_batch_validation(self):
        self.asset.write_bytes(b"changed asset")
        with self.assertRaisesRegex(ValueError, "Asset library input changed"):
            runner.source_check(self.job, runner.AssetSnapshotGuard())

    def test_snapshot_change_is_rejected(self):
        self.snapshot.write_bytes(self.snapshot.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            runner.source_check(self.job, runner.AssetSnapshotGuard())

    def test_cached_guard_detects_unselected_asset_change(self):
        guard = runner.AssetSnapshotGuard()
        runner.source_check(self.job, guard)
        self.asset.write_bytes(b"changed unselected candidate")
        with self.assertRaisesRegex(ValueError, "input changed"):
            runner.source_check(self.job, guard)

    def test_cached_guard_rejects_present_input_replaced_by_symlink(self):
        guard = runner.AssetSnapshotGuard()
        runner.source_check(self.job, guard)
        saved = self.root / 'saved_asset.glb'
        self.asset.rename(saved)
        self.asset.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            runner.source_check(self.job, guard)

    def test_cached_guard_checks_missing_inputs_and_their_parents(self):
        parent = self.root / 'optional'
        parent.mkdir()
        missing = parent / 'collision.glb'
        for replacement in ('file', 'parent_symlink'):
            with self.subTest(replacement=replacement):
                self.write_snapshot(missing=(missing,))
                guard = runner.AssetSnapshotGuard()
                runner.source_check(self.job, guard)
                if replacement == 'file':
                    missing.write_bytes(b'new collision mesh')
                    with self.assertRaisesRegex(ValueError, 'missing.*appeared'):
                        runner.source_check(self.job, guard)
                    missing.unlink()
                else:
                    parent.rmdir()
                    target = self.root / 'replacement'
                    target.mkdir()
                    parent.symlink_to(target, target_is_directory=True)
                    with self.assertRaisesRegex(ValueError, 'symlink'):
                        runner.source_check(self.job, guard)

    def test_cached_guard_detects_catalog_and_snapshot_changes(self):
        for changed, expected in ((self.catalog, "catalog changed"),
                                  (self.snapshot, "snapshot changed")):
            with self.subTest(path=changed.name):
                self.write_snapshot()
                guard = runner.AssetSnapshotGuard()
                runner.source_check(self.job, guard)
                changed.write_bytes(changed.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, expected):
                    runner.source_check(self.job, guard)

    def test_seed_rejects_catalog_changed_after_snapshot_pin(self):
        from asset_catalog import AssetCatalog

        snapshot = runner.read_json(self.snapshot)
        stats = {str(path): AssetCatalog.stable_file_digest(path)[1]
                 for path in (self.catalog, self.asset)}
        reference = self.job["asset_library"]["input_snapshot"]
        self.catalog.write_text('{"changed":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "catalog changed"):
            runner.AssetSnapshotGuard().seed(reference, snapshot, stats)

    def test_malformed_snapshot_digest_and_catalog_fingerprint_are_rejected(self):
        snapshot = runner.read_json(self.snapshot)
        malformed_digest = copy.deepcopy(snapshot)
        malformed_digest["snapshot_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "snapshot digest"):
            runner.AssetSnapshotGuard._validate_snapshot(malformed_digest)

        malformed_catalog = copy.deepcopy(snapshot)
        malformed_catalog["input_sha256"][str(self.catalog)] = "f" * 64
        payload = {key: value for key, value in malformed_catalog.items()
                   if key != "snapshot_sha256"}
        malformed_catalog["snapshot_sha256"] = hashlib.sha256(
            runner.canonical_bytes(payload)).hexdigest()
        with self.assertRaisesRegex(ValueError, "catalog fingerprint"):
            runner.AssetSnapshotGuard._validate_snapshot(malformed_catalog)

    def test_guards_have_independent_verified_input_state(self):
        first = runner.AssetSnapshotGuard()
        second = runner.AssetSnapshotGuard()
        runner.source_check(self.job, first)
        runner.source_check(self.job, second)
        self.asset.write_bytes(b"changed after both guards were seeded")
        with self.assertRaisesRegex(ValueError, "input changed"):
            runner.source_check(self.job, second)


if __name__ == "__main__":
    unittest.main()
