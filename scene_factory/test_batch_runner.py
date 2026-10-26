"""Meaningful runner contract checks with subprocess stand-ins, no Blender renders."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import batch


class BatchRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.converter = self.root / "photo_to_spec.py"
        self.generator = self.root / "generate_scene.py"
        self.converter.write_text("# converter revision 1\n")
        self.generator.write_text("# generator revision 1\n")
        self.image = self.root / "office.jpg"
        self.image.write_bytes(b"reference-photo-fixture")
        self.output = self.root / "output"
        self.job = batch.make_job(self.image, 640, 480, 16, self.converter, self.generator)

    def run_job(self, **options):
        return batch.run_job(self.job, self.output, self.converter, self.generator, "mock-blender", **options)

    @staticmethod
    def successful_process(command, **kwargs):
        if "--image" in command:
            destination = Path(command[command.index("--output") + 1])
            batch.atomic_json(destination, {"schema_version": 1, "objects": [{"type": "table"}]})
        else:
            directory = Path(command[command.index("--output-dir") + 1])
            (directory / "scene.blend").write_bytes(b"BLENDER-fixture")
            batch.atomic_json(directory / "metadata.json", {"status": "completed", "outputs": ["scene.blend"]})
        return subprocess.CompletedProcess(command, 0)

    def test_dry_run_is_deterministic_and_starts_no_process(self):
        with patch.object(batch.subprocess, "run") as process:
            first = self.run_job(dry_run=True)
            directory = self.output / self.job["job_id"]
            initial = (directory / "job.json").read_bytes()
            second = self.run_job(dry_run=True)
            self.assertEqual(initial, (directory / "job.json").read_bytes())
            self.assertEqual(first["commands"], second["commands"])
            self.assertEqual(first["status"], "planned")
            process.assert_not_called()

    def test_success_resumes_but_changed_config_and_generator_conflict(self):
        with patch.object(batch.subprocess, "run", side_effect=self.successful_process) as process:
            self.assertEqual(self.run_job()["status"], "completed")
            self.assertEqual(process.call_count, 2)
            self.assertEqual(self.run_job()["status"], "skipped")
            self.assertEqual(process.call_count, 2)
        directory = self.output / self.job["job_id"]
        preserved = (directory / "job.json").read_bytes()
        self.job["render"]["samples"] = 32
        self.assertEqual(self.run_job()["status"], "conflict")
        self.assertEqual(preserved, (directory / "job.json").read_bytes())
        self.job["render"]["samples"] = 16
        self.generator.write_text("# generator revision 2\n")
        self.job = batch.make_job(self.image, 640, 480, 16, self.converter, self.generator)
        self.assertEqual(self.run_job()["status"], "conflict")

    def test_exit_zero_without_geometry_is_failed(self):
        def incomplete(command, **kwargs):
            if "--image" in command:
                return self.successful_process(command, **kwargs)
            directory = Path(command[command.index("--output-dir") + 1])
            batch.atomic_json(directory / "metadata.json", {"status": "completed"})
            return subprocess.CompletedProcess(command, 0)
        with patch.object(batch.subprocess, "run", side_effect=incomplete):
            result = self.run_job()
        self.assertEqual(result["status"], "failed")
        self.assertIn("scene.blend is missing", result["error"])

    def test_spec_only_can_continue_with_reuse_spec(self):
        with patch.object(batch.subprocess, "run", side_effect=self.successful_process) as process:
            self.assertEqual(self.run_job(specs_only=True)["status"], "spec_ready")
            self.assertEqual(process.call_count, 1)
            self.assertEqual(self.run_job(reuse_spec=True)["status"], "completed")
            self.assertEqual(process.call_count, 2)
            self.assertIn("--spec", process.call_args.args[0])

    def test_failed_job_does_not_prevent_other_jobs_and_manifest_is_valid(self):
        second_image = self.root / "second.png"
        second_image.write_bytes(b"second-reference")
        first_digest = self.job["image_sha256"][:12]
        def partial(command, **kwargs):
            if "--spec" in command and first_digest in command[command.index("--spec") + 1]:
                return subprocess.CompletedProcess(command, 7)
            return self.successful_process(command, **kwargs)
        arguments = ["--images", str(self.image), str(second_image), "--output-dir", str(self.output), "--workers", "2"]
        with patch.object(batch, "ROOT", self.root), patch.object(batch, "discover_blender", return_value="mock-blender"), patch.object(batch.subprocess, "run", side_effect=partial), contextlib.redirect_stdout(io.StringIO()):
            code = batch.main(arguments)
        summary = batch.read_json(self.output / "summary.json")
        self.assertEqual(code, 1)
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertEqual(summary["counts"]["completed"], 1)
        records = [json.loads(line) for line in (self.output / "manifest.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 2)

    def test_asset_quality_and_dressing_reach_blender(self):
        self.job = batch.make_job(self.image, 640, 480, 16, self.converter, self.generator, 'detailed', True)
        _, command = batch.job_commands(self.job, self.output / self.job['job_id'], self.converter, self.generator, 'blender')
        self.assertEqual(command[command.index('--asset-quality') + 1], 'detailed')
        self.assertIn('--desk-details', command)
        self.run_job(dry_run=True)
        self.job['render']['asset_quality'] = 'basic'
        self.assertEqual(self.run_job(dry_run=True)['status'], 'conflict')

    def test_changed_texture_invalidates_job_version(self):
        with patch.object(batch, 'ROOT', self.root):
            assets = self.root / 'assets_lib' / 'textures'
            assets.mkdir(parents=True)
            texture = assets / 'screen.png'
            texture.write_bytes(b'version1')
            old = batch.make_job(self.image, 640, 480, 16, self.converter, self.generator)
            texture.write_bytes(b'version2')
            new = batch.make_job(self.image, 640, 480, 16, self.converter, self.generator)
            self.assertNotEqual(old['pipeline']['assets'], new['pipeline']['assets'])

    def test_timeout_is_recorded_as_failure(self):
        with patch.object(batch.subprocess, "run", side_effect=subprocess.TimeoutExpired("stand-in", 1)):
            result = self.run_job(timeout=1)
        self.assertEqual(result["status"], "failed")
        self.assertIn("timeout", result["error"])


if __name__ == "__main__":
    unittest.main()
