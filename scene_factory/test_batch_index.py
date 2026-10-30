"""Incremental checkpoint scaling and abrupt coordinator recovery; no API/Blender."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from batch import atomic_json, read_json
import batch_multiview as runner
from batch_index import EVENTS_FILE, append_result, read_results, read_snapshot, reset_events


class BatchIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_torn_event_does_not_hide_later_events(self):
        append_result(self.root, 'run', 1, {'scene_id': 'first', 'status': 'failed'})
        with (self.root/'batch_events.jsonl').open('ab') as handle:
            handle.write(b'{"schema_version":')
        append_result(self.root, 'run', 2, {'scene_id': 'second', 'status': 'failed'})
        self.assertEqual([item['scene_id'] for item in read_results(self.root)], ['first', 'second'])

    def test_artifact_digest_cache_reuses_unchanged_content_within_run(self):
        artifact = self.root / 'scene' / 'artifact.bin'
        artifact.parent.mkdir()
        artifact.write_bytes(b'large synthetic artifact')
        digest = runner.file_sha256(artifact)
        real_file_sha256 = runner.file_sha256
        cache = {}
        with patch.object(runner, 'file_sha256', wraps=runner.file_sha256) as hashed:
            runner.verify_hashes(artifact.parent, {'artifact.bin': digest}, cache=cache)
            first_calls = hashed.call_count
            runner.verify_hashes(artifact.parent, {'artifact.bin': digest}, cache=cache)
            self.assertEqual(hashed.call_count, first_calls)

            artifact.write_bytes(b'changed synthetic artifact')
            changed = real_file_sha256(artifact)
            runner.verify_hashes(artifact.parent, {'artifact.bin': changed}, cache=cache)
            self.assertEqual(hashed.call_count, first_calls + 1)

    def test_scene_commit_overrides_event_and_accepted_evidence_is_verified(self):
        folder = self.root/'scene'
        folder.mkdir()
        committed = {'scene_id': 'scene', 'status': 'generated_unreviewed'}
        atomic_json(folder/'result.json', committed)
        append_result(self.root, 'run', 1, {'scene_id': 'scene', 'status': 'accepted'})
        append_result(self.root, 'run', 2, {'scene_id': 'uncommitted', 'status': 'accepted'})
        records = runner.recorded_results(self.root)
        self.assertEqual(records['scene'], committed)
        self.assertEqual(records['uncommitted']['status'], 'failed')

    def test_requested_recovery_ids_are_skipped_for_coordinator_resume(self):
        # The coordinator's worker performs the authoritative resume check for
        # requested IDs.  Recovery must not spend a second full artifact walk
        # on a record it will replace before the first checkpoint.
        folder = self.root / 'requested'
        folder.mkdir()
        accepted = {'scene_id': 'requested', 'status': 'accepted',
                    'gate': {'accepted': True}, 'iteration': 'iteration_00',
                    'final_review_iteration': 'iteration_00', 'artifacts': {}}
        atomic_json(folder / 'result.json', accepted)
        atomic_json(self.root / 'summary.json', {'run_id': 'old', 'scenes': [accepted]})

        recovered = runner.recorded_results(self.root)
        self.assertEqual(recovered['requested']['status'], 'failed')
        self.assertNotIn('requested', runner.recorded_results(
            self.root, skip_scene_ids={'requested'}))

    def test_coordinator_failure_survives_missing_and_corrupt_summary(self):
        failed = {'scene_id': 'scene', 'status': 'failed', 'failed_stage': 'coordinator',
                  'error': 'worker never committed a result'}
        append_result(self.root, 'run', 1, failed)
        self.assertEqual(runner.recorded_results(self.root)['scene'], failed)
        (self.root/'summary.json').write_text('{interrupted')
        self.assertEqual(runner.recorded_results(self.root)['scene'], failed)

    def test_start_snapshot_filters_previous_run_events_before_rotation(self):
        append_result(self.root, 'previous', 1, {'scene_id': 'scene', 'status': 'failed'})
        atomic_json(self.root/'summary.json', {'run_id': 'current',
            'scenes': [{'scene_id': 'scene', 'status': 'pending'}]})
        self.assertEqual(runner.recorded_results(self.root)['scene']['status'], 'pending')
        reset_events(self.root)
        self.assertEqual(list(read_results(self.root)), [])

    def test_coordinator_releases_requested_recovery_records(self):
        class TrackingRecords(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.popped = []

            def pop(self, key, *args):
                self.popped.append(key)
                return super().pop(key, *args)

        requested = {'scene_id': 'requested', 'images': [self.root/'source']}
        historical = {'scene_id': 'historical', 'images': [self.root/'old-source']}
        recovered = TrackingRecords({
            'requested': {'scene_id': 'requested', 'status': 'accepted'},
            'historical': {'scene_id': 'historical', 'status': 'accepted'},
        })
        output = self.root/'output'
        args = SimpleNamespace(output_dir=output, workers=1)

        def work(group, args, hashes):
            return {'scene_id': group['scene_id'], 'status': 'generated_unreviewed'}

        with patch.object(runner, 'recorded_results', return_value=recovered), \
                patch.object(runner, 'reconcile_groups', return_value=[requested, historical]), \
                patch.object(runner, 'run_job', side_effect=work), \
                patch('report_multiview.report'), redirect_stdout(io.StringIO()):
            self.assertEqual(runner._coordinate([requested], args, {}), 0)

        self.assertIn('requested', recovered.popped)
        self.assertNotIn('historical', recovered.popped)
        summary = read_json(output/'summary.json')
        self.assertEqual({item['scene_id']: item['status'] for item in summary['scenes']},
                         {'requested': 'generated_unreviewed', 'historical': 'accepted'})

    def test_terminal_snapshot_restores_full_event_result_after_hot_status_compaction(self):
        output = self.root / 'output'
        source = self.root / 'source'
        source.write_text('synthetic source')
        groups = [{'scene_id': 'scene', 'images': [source]}]
        result = {'scene_id': 'scene', 'directory': str(output / 'scene'),
                  'status': 'accepted', 'iteration': 'iteration_00',
                  'history': [{'scores': {'realism': 92}}],
                  'artifacts': {'iteration_00/scene.blend': 'a' * 64}}

        def work(group, args, hashes):
            return result

        args = SimpleNamespace(output_dir=output, workers=1)
        with patch.object(runner, 'run_job', side_effect=work), \
             patch('report_multiview.report'), redirect_stdout(io.StringIO()):
            self.assertEqual(runner._coordinate(groups, args, {}), 0)
        summary = read_json(output / 'summary.json')
        self.assertEqual(summary['scenes'], [result])
        self.assertEqual(read_json(output / 'accepted.json'), [result])

    def test_terminal_snapshot_ignores_stale_live_run_state_and_events(self):
        for terminal in ('completed', 'interrupted', 'stopped'):
            with self.subTest(terminal=terminal):
                directory = self.root / terminal
                summary = {'run_id': 'current', 'run_state': terminal,
                    'incremental_events': EVENTS_FILE, 'completed_this_run': 1,
                    'updated_at': 'terminal-checkpoint', 'total': 1,
                    'scenes': [{'scene_id': 'scene', 'status': 'generated_unreviewed'}]}
                atomic_json(directory/'summary.json', summary)
                atomic_json(directory/'run_state.json', {'run_id': 'current', 'state': 'running',
                    'completed_this_run': 0, 'updated_at': 'older-live-checkpoint'})
                append_result(directory, 'current', 1, {'scene_id': 'scene', 'status': 'pending'})
                value = read_snapshot(directory)
                self.assertEqual(value['run_state'], terminal)
                self.assertEqual(value['updated_at'], 'terminal-checkpoint')
                self.assertEqual(value['completed_this_run'], 1)
                self.assertEqual(value['finished'], 1)
                self.assertEqual(value['scenes'], summary['scenes'])

    def test_new_live_run_ignores_old_terminal_state_then_merges_its_progress(self):
        atomic_json(self.root/'summary.json', {'run_id': 'new', 'run_state': 'running',
            'incremental_events': EVENTS_FILE, 'completed_this_run': 0,
            'updated_at': 'new-start', 'total': 2, 'scenes': [
                {'scene_id': 'a', 'status': 'pending'}, {'scene_id': 'b', 'status': 'pending'}]})
        atomic_json(self.root/'run_state.json', {'run_id': 'previous', 'state': 'completed',
            'completed_this_run': 2, 'updated_at': 'previous-end'})
        append_result(self.root, 'previous', 1, {'scene_id': 'a', 'status': 'accepted'})
        value = read_snapshot(self.root)
        self.assertEqual(value['run_state'], 'running')
        self.assertEqual(value['updated_at'], 'new-start')
        self.assertEqual(value['completed_this_run'], 0)
        self.assertEqual(value['finished'], 0)
        append_result(self.root, 'new', 1, {'scene_id': 'a', 'status': 'generated_unreviewed'})
        atomic_json(self.root/'run_state.json', {'run_id': 'new', 'state': 'stopping',
            'completed_this_run': 1, 'updated_at': 'new-progress'})
        value = read_snapshot(self.root)
        self.assertEqual(value['run_state'], 'stopping')
        self.assertEqual(value['updated_at'], 'new-progress')
        self.assertEqual(value['completed_this_run'], 1)
        self.assertEqual(value['finished'], 1)
        self.assertEqual([item['status'] for item in value['scenes']], ['generated_unreviewed', 'pending'])

    def test_final_summary_survives_exit_before_terminal_run_state_commit(self):
        # Reproduce the real coordinator checkpoint order. Terminate directly
        # after its final summary commit, before accepted.json/run_state.json.
        code = '''
import os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import batch_multiview as runner
from batch import atomic_json
output = Path(sys.argv[2])
groups = [{'scene_id': 'scene', 'images': [output.parent/'source.txt']}]
def work(group, args, hashes):
    result = {'scene_id': group['scene_id'], 'status': 'generated_unreviewed'}
    atomic_json(output/group['scene_id']/'result.json', result)
    return result
def commit_then_exit(path, value):
    atomic_json(path, value)
    if path.name == 'summary.json' and value.get('run_state') == 'completed':
        os._exit(17)
runner.run_job = work
runner.atomic_json = commit_then_exit
runner._coordinate(groups, SimpleNamespace(output_dir=output, workers=1), {})
'''
        output = self.root/'output'
        process = subprocess.run([sys.executable, '-c', code,
            str(Path(runner.__file__).parent), str(output)], capture_output=True, timeout=30)
        self.assertEqual(process.returncode, 17, process.stderr.decode())
        summary = read_json(output/'summary.json')
        stale = read_json(output/'run_state.json')
        self.assertEqual(summary['run_state'], 'completed')
        self.assertEqual(stale['state'], 'running')
        self.assertEqual(stale['run_id'], summary['run_id'])
        value = read_snapshot(output)
        self.assertEqual(value['run_state'], 'completed')
        self.assertEqual(value['updated_at'], summary['updated_at'])
        self.assertEqual(value['completed_this_run'], 1)
        self.assertEqual(value['finished'], 1)
        self.assertEqual(value['scenes'][0]['status'], 'generated_unreviewed')

    def test_abrupt_exit_recovers_commits_events_and_resumes(self):
        # Exit after a worker commit that was never returned to the coordinator.
        code = '''
import os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import batch_multiview as runner
from batch import atomic_json
output = Path(sys.argv[2])
groups = [{'scene_id': str(i), 'images': [output.parent/'source.txt']} for i in range(4)]
def work(group, args, hashes):
    folder = output/group['scene_id']
    folder.mkdir()
    result = {'scene_id': group['scene_id'], 'status': 'generated_unreviewed'}
    atomic_json(folder/'result.json', result)
    if group['scene_id'] == '2':
        os._exit(17)
    return result
runner.run_job = work
runner._coordinate(groups, SimpleNamespace(output_dir=output, workers=1), {})
'''
        output = self.root/'output'
        process = subprocess.run([sys.executable, '-c', code,
            str(Path(runner.__file__).parent), str(output)], capture_output=True, timeout=30)
        self.assertEqual(process.returncode, 17, process.stderr.decode())
        self.assertEqual(read_json(output/'summary.json')['completed_this_run'], 0)
        self.assertEqual(read_json(output/'run_state.json')['completed_this_run'], 2)
        self.assertEqual(len(list(read_results(output))), 2)
        self.assertEqual(len([record for record in runner.recorded_results(output).values()
                             if record['status'] == 'generated_unreviewed']), 3)
        groups = [{'scene_id': str(i), 'images': [self.root/'source.txt']} for i in range(4)]
        def resume(group, args, hashes):
            path = output/group['scene_id']/'result.json'
            return read_json(path) if path.exists() else {'scene_id': group['scene_id'], 'status': 'generated_unreviewed'}
        with patch.object(runner, 'run_job', side_effect=resume), patch('report_multiview.report'), redirect_stdout(io.StringIO()):
            self.assertEqual(runner._coordinate(groups, SimpleNamespace(output_dir=output, workers=1), {}), 0)
        self.assertEqual(read_json(output/'summary.json')['finished'], 4)
        self.assertEqual(len(list(read_results(output))), 4)

    def test_ten_thousand_synthetic_scene_commits_have_linear_checkpoint_writes(self):
        count = 10_000
        output = self.root/'output'
        output.mkdir()
        args = SimpleNamespace(output_dir=output, workers=1)
        # Reverse the IDs to ensure terminal snapshots preserve groups.json
        # order rather than sorting event results by identity.
        groups = [{'scene_id': f'scene_{count-i-1:05}', 'images': [self.root/'source']} for i in range(count)]
        writes = {'summary.json': [], 'accepted.json': [], 'run_state.json': []}
        manifest = {f'iteration_00/asset_{i:03}.bin': 'a'*64 for i in range(32)}

        class TrackingRecords(dict):
            event_assignments = 0

            def __setitem__(self, key, value):
                self.event_assignments += 1
                super().__setitem__(key, value)

        # Keep one prior historical job in the same batch. Requested records
        # are skipped by recovery and restored from their durable events.
        historical = {'scene_id': 'historical', 'images': [str(self.root/'old-source')]}
        atomic_json(output/'groups.json', {'schema_version': 1, 'scenes': [historical]})
        atomic_json(output/'historical'/'result.json', {
            'scene_id': 'historical', 'status': 'generated_unreviewed', 'artifacts': {'old.bin': 'b'*64}})
        recover = runner.recorded_results
        recovered_records = []

        def tracked_recovery(*args, **kwargs):
            records = TrackingRecords(recover(*args, **kwargs))
            recovered_records.append(records)
            return records

        def work(group, args, hashes):
            result = {'scene_id': group['scene_id'], 'status': 'generated_unreviewed', 'artifacts': manifest}
            atomic_json(output/group['scene_id']/'result.json', result)
            return result
        def observe(path, value):
            if path.name in writes:
                writes[path.name].append(len(json.dumps(value)))
            atomic_json(path, value)
        started = time.perf_counter()
        with patch.object(runner, 'recorded_results', side_effect=tracked_recovery), \
                patch.object(runner, 'run_job', side_effect=work), patch.object(runner, 'atomic_json', side_effect=observe), \
                patch('report_multiview.report'), redirect_stdout(io.StringIO()):
            self.assertEqual(runner._coordinate(groups, args, {}), 0)
        elapsed = time.perf_counter() - started
        self.assertEqual(len(writes['summary.json']), 2)
        self.assertEqual(len(writes['accepted.json']), 2)
        self.assertEqual(recovered_records[0].event_assignments, count)
        runtime_state_writes = writes['run_state.json'][1:-1]
        self.assertEqual(len(runtime_state_writes), count)
        self.assertLess(max(runtime_state_writes), 512)
        summary = read_json(output/'summary.json')
        self.assertEqual(summary['finished'], count + 1)
        self.assertEqual([item['scene_id'] for item in summary['scenes']],
                         ['historical', *[group['scene_id'] for group in groups]])
        self.assertEqual(summary['scenes'][0]['artifacts'], {'old.bin': 'b'*64})
        self.assertTrue(all(item['status'] == 'generated_unreviewed' and item['artifacts'] == manifest
                            for item in summary['scenes'][1:]))
        self.assertEqual(len(list(read_results(output))), count)
        self.assertEqual(len(runner.recorded_results(output)), count + 1)
        # Compare actual event bytes against copying the growing accumulated
        # artifact manifests on every completion (the previous implementation).
        event_bytes = (output/'batch_events.jsonl').stat().st_size
        previous_minimum = len(json.dumps(manifest)) * count * (count+1)//2
        self.assertLess(event_bytes, previous_minimum//1000)
        print(f'10,000 commits: elapsed={elapsed:.2f}s; events={event_bytes:,} bytes; '
              f'full snapshots={len(writes["summary.json"])}; '
              f'runtime state max={max(runtime_state_writes)} bytes; '
              f'previous manifest-only writes>={previous_minimum:,} bytes')


if __name__ == '__main__':
    unittest.main()
