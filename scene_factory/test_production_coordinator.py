"""Production queue/recovery contracts using synthetic images and mocked workers.

The lock crash test uses a real subprocess. No remote model API or Blender
render is invoked, and synthetic acceptance records test index integrity only.
"""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from batch import atomic_json, read_json
import batch_multiview as runner
import test_batch_multiview as fixtures


class ProductionCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.args = SimpleNamespace(output_dir=self.root/'output', workers=2)
        self.source = self.root/'source.txt'
        self.source.write_text('synthetic queue identity')

    def groups(self, count):
        return [{'scene_id': 'scene_'+str(index), 'images': [self.source]} for index in range(count)]

    @staticmethod
    def outcome(group, args, hashes):
        return {'scene_id': group['scene_id'], 'status': 'generated_unreviewed'}

    def run_queue(self, groups, work=None):
        with patch.object(runner, 'run_job', side_effect=work or self.outcome), \
             patch('report_multiview.report'), redirect_stdout(io.StringIO()):
            return runner._coordinate(groups, self.args, {})

    def accepted_fixture(self, group):
        """Commit a minimal synthetic gate result; photo quality is not tested."""
        folder = self.args.output_dir/group['scene_id']
        folder.mkdir(parents=True)
        (folder/'artifact.bin').write_bytes(b'synthetic immutable artifact')
        atomic_json(folder/'job.json', {'sources': []})
        result = {'scene_id': group['scene_id'], 'directory': str(folder), 'status': 'accepted',
            'iteration': 'iteration_00', 'final_review_iteration': 'iteration_00',
            'gate': {'accepted': True, 'reasons': []},
            'artifacts': runner.artifact_hashes(folder, ['artifact.bin', 'job.json'])}
        atomic_json(folder/'result.json', result)
        return result

    def test_active_output_lock_rejects_second_writer_and_releases(self):
        self.args.output_dir.mkdir()
        lock = self.args.output_dir/'.batch.lock'
        with runner.process_lock(lock):
            with self.assertRaises(runner.OutputInUseError):
                self.run_queue(self.groups(1))
            self.assertFalse((self.args.output_dir/'groups.json').exists())
        self.assertEqual(self.run_queue(self.groups(1)), 0)
        self.assertTrue(lock.is_file())  # Stable inode prevents split ownership.

    def test_queue_only_submits_available_worker_slots(self):
        started = threading.Event()
        release = threading.Event()
        calls = []
        errors = []
        mutex = threading.Lock()
        def work(group, args, hashes):
            with mutex:
                calls.append(group['scene_id'])
                if len(calls) == args.workers:
                    started.set()
            release.wait(10)
            return self.outcome(group, args, hashes)
        from concurrent.futures import ThreadPoolExecutor as RealPool
        submissions = []
        class RecordingPool(RealPool):
            def submit(self, *args, **kwargs):
                submissions.append(args[1]['scene_id'])
                return super().submit(*args, **kwargs)
        def invoke():
            try:
                self.run_queue(self.groups(60), work)
            except BaseException as exc:
                errors.append(exc)
        with patch.object(runner, 'ThreadPoolExecutor', RecordingPool):
            thread = threading.Thread(target=invoke)
            thread.start()
            try:
                self.assertTrue(started.wait(10))
                self.assertEqual(len(submissions), 2)
            finally:
                release.set()
                thread.join(20)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 60)

    def test_worker_exception_isolated_and_queue_continues(self):
        def work(group, args, hashes):
            if group['scene_id'] == 'scene_0':
                raise RuntimeError('unexpected worker exception')
            return self.outcome(group, args, hashes)
        self.assertEqual(self.run_queue(self.groups(7), work), 1)
        summary = read_json(self.args.output_dir/'summary.json')
        self.assertEqual(summary['finished'], 7)
        self.assertEqual(summary['scenes'][0]['status'], 'failed')
        self.assertTrue(all(item['status'] == 'generated_unreviewed' for item in summary['scenes'][1:]))
        self.assertEqual(read_json(self.args.output_dir/'run_state.json')['state'], 'completed')

    def test_incremental_and_subset_runs_preserve_previous_accepted_index(self):
        groups = self.groups(3)
        old = self.accepted_fixture(groups[0])
        atomic_json(self.args.output_dir/'groups.json', {'schema_version': 1, 'scenes': [
            {'scene_id': groups[0]['scene_id'], 'images': [str(self.source)]}]})
        self.assertEqual(self.run_queue(groups[1:]), 0)
        summary = read_json(self.args.output_dir/'summary.json')
        self.assertEqual(len(summary['scenes']), 3)
        self.assertEqual(read_json(self.args.output_dir/'accepted.json'), [old])
        self.assertEqual(self.run_queue(groups[2:]), 0)
        self.assertEqual(len(read_json(self.args.output_dir/'summary.json')['scenes']), 3)
        self.assertEqual(read_json(self.args.output_dir/'accepted.json'), [old])

    def test_previous_accepted_corruption_is_removed_from_dataset_index(self):
        groups = self.groups(2)
        self.accepted_fixture(groups[0])
        (self.args.output_dir/groups[0]['scene_id']/'artifact.bin').write_bytes(b'tampered')
        self.assertEqual(self.run_queue(groups[1:]), 0)
        self.assertEqual(read_json(self.args.output_dir/'accepted.json'), [])
        previous = {item['scene_id']: item for item in read_json(self.args.output_dir/'summary.json')['scenes']}['scene_0']
        self.assertEqual(previous['status'], 'failed')
        self.assertFalse(previous['gate']['accepted'])

    def test_corrupt_summary_rebuilt_from_scene_commits(self):
        group = self.groups(1)[0]
        result = self.accepted_fixture(group)
        (self.args.output_dir/'summary.json').write_text('{interrupted')
        records = runner.recorded_results(self.args.output_dir)
        self.assertEqual(records[group['scene_id']], result)
        self.assertTrue((self.args.output_dir/'index_recovery.json').is_file())

    def test_missing_historical_photo_does_not_block_new_scene(self):
        missing = self.root/'deleted-photo.png'
        self.args.output_dir.mkdir()
        atomic_json(self.args.output_dir/'groups.json', {'schema_version': 1,
            'scenes': [{'scene_id': 'old', 'images': [str(missing)]}]})
        self.assertEqual(self.run_queue(self.groups(1)), 0)
        groups = read_json(self.args.output_dir/'groups.json')['scenes']
        self.assertEqual([item['scene_id'] for item in groups], ['old', 'scene_0'])

    def test_different_photos_for_same_id_rejected_before_work(self):
        groups = self.groups(1)
        self.assertEqual(self.run_queue(groups), 0)
        before = (self.args.output_dir/'summary.json').read_bytes()
        other = self.root/'other.txt'
        other.write_text('other')
        with self.assertRaises(ValueError), patch.object(runner, 'run_job') as work:
            runner._coordinate([{'scene_id': 'scene_0', 'images': [other]}], self.args, {})
        work.assert_not_called()
        self.assertEqual((self.args.output_dir/'summary.json').read_bytes(), before)

    def test_interrupt_stops_scheduling_and_preserves_pending_ids(self):
        real_wait = runner.wait
        calls = []
        waits = [0]
        def work(group, args, hashes):
            calls.append(group['scene_id'])
            args.stop_event.wait(5)
            return {'scene_id': group['scene_id'], 'status': 'interrupted'}
        def interrupt_once(*args, **kwargs):
            waits[0] += 1
            if waits[0] == 1:
                raise KeyboardInterrupt
            return real_wait(*args, **kwargs)
        with patch.object(runner, 'wait', side_effect=interrupt_once):
            self.assertEqual(self.run_queue(self.groups(8), work), 130)
        self.assertEqual(len(calls), 2)
        state = read_json(self.args.output_dir/'run_state.json')
        self.assertEqual(state['state'], 'interrupted')
        self.assertEqual(len(state['pending_scene_ids']), 6)
        self.assertEqual(len(state['resumable_scene_ids']), 8)
        self.assertEqual(read_json(self.args.output_dir/'accepted.json'), [])


@unittest.skipIf(fixtures.Image is None, 'Pillow required')
class SceneCrashRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.GroupedBatchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_crashed_process_kernel_lock_and_modern_marker_are_recovered(self):
        folder = self.fixture.args.output_dir/'lab'
        folder.mkdir(parents=True)
        # This subprocess exits without Python finally blocks or explicit unlock.
        code = '''
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import batch_multiview as runner
folder = Path(sys.argv[2])
with runner.process_lock(folder/'.process.lock'):
    runner.atomic_json(folder/'.lock', {'lock_protocol': runner.LOCK_PROTOCOL, 'pid': os.getpid()})
    os._exit(17)
'''
        process = subprocess.run([sys.executable, '-c', code, str(Path(runner.__file__).parent), str(folder)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        self.assertEqual(process.returncode, 17, process.stderr.decode())
        self.assertTrue((folder/'.lock').is_file())
        with redirect_stdout(io.StringIO()):
            result, analyses, renders = self.fixture.run_mocked()
        self.assertEqual(result['status'], 'generated_unreviewed')
        self.assertEqual((analyses, renders), (1, 1))
        self.assertFalse((folder/'.lock').exists())
        self.assertNotIn('.process.lock', result['artifacts'])

    def test_valid_resume_clears_later_error_marker_without_changing_commit(self):
        with redirect_stdout(io.StringIO()):
            initial, _, _ = self.fixture.run_mocked()
            self.fixture.args.samples += 1
            failed, _, _ = self.fixture.run_mocked()
            self.fixture.args.samples -= 1
            resumed, analyses, renders = self.fixture.run_mocked()
        folder = Path(initial['directory'])
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(resumed['status'], 'generated_unreviewed')
        self.assertTrue(resumed['resumed'])
        self.assertEqual((analyses, renders), (0, 0))
        self.assertFalse((folder/'last_error.json').exists())
        self.assertEqual(read_json(folder/'result.json'), initial)

    def test_interrupted_initialization_without_job_commit_can_retry(self):
        folder = self.fixture.args.output_dir/'lab'
        folder.mkdir(parents=True)
        atomic_json(folder/'last_error.json', {'scene_id': 'lab', 'status': 'failed',
            'error': 'Synthetic pre-job transient failure'})
        with redirect_stdout(io.StringIO()):
            result, analyses, renders = self.fixture.run_mocked()
        self.assertEqual(result['status'], 'generated_unreviewed')
        self.assertEqual((analyses, renders), (1, 1))
        self.assertFalse((folder/'last_error.json').exists())


if __name__ == '__main__':
    unittest.main()
