"""Durable incremental coordinator events for large batch runs.

Callers serialize writes under the batch output lock. Events are a recoverable
cache, never evidence of acceptance: per-scene commits and their hashes remain
authoritative. One run's log is retained until its successor has snapshotted it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from batch import canonical_bytes
from multiview_spec import SAFE_ID


EVENTS_FILE = 'batch_events.jsonl'


def reset_events(directory):
    """Rotate after a durable start snapshot, under the output ownership lock."""
    directory = Path(directory)
    with tempfile.NamedTemporaryFile(dir=directory, prefix='.batch_events.', delete=False) as handle:
        temporary = Path(handle.name)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, directory/EVENTS_FILE)
    finally:
        temporary.unlink(missing_ok=True)


def append_result(directory, run_id, sequence, result):
    """Append one committed scene outcome and fsync it before advancing state."""
    path = Path(directory)/EVENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {'schema_version': 1, 'run_id': run_id, 'sequence': sequence,
             'scene_id': result['scene_id'], 'result': result}
    with path.open('ab') as handle:
        # A crashed append can leave a partial final line. The separator keeps
        # subsequent committed events independently readable after recovery.
        handle.write(b'\n' + canonical_bytes(event) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())


def read_results(directory, run_id=None):
    """Read valid event records; scene result commits are reconciled afterward."""
    path = Path(directory)/EVENTS_FILE
    if not path.is_file():
        return
    with path.open('rb') as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict) or (run_id is not None and event.get('run_id') != run_id):
                continue
            result = event.get('result')
            if (event.get('schema_version') == 1 and isinstance(event.get('scene_id'), str)
                    and isinstance(result, dict) and result.get('scene_id') == event['scene_id']):
                yield result


def read_snapshot(directory):
    """Merge the last snapshot and matching events for read-only diagnostics.

    These records are operational observations, never verified acceptance.
    Boundary snapshots win over their own events after completion/interruption.
    """
    from batch import read_json
    directory = Path(directory)
    summary = read_json(directory / 'summary.json')
    if not isinstance(summary, dict):
        raise ValueError('summary.json must be a JSON object')
    if not isinstance(summary.get('scenes'), list):
        raise ValueError('summary.json scenes must be a list')
    if summary.get('incremental_events') != EVENTS_FILE:
        return summary
    records = {}
    for item in summary['scenes']:
        identifier = item.get('scene_id') if isinstance(item, dict) else None
        if not isinstance(identifier, str) or not SAFE_ID.fullmatch(identifier):
            raise ValueError('Invalid recorded scene identity')
        records[identifier] = item
    run_id = summary.get('run_id')
    if not isinstance(run_id, str) or not run_id:
        raise ValueError('Incremental summary requires run identity')
    live_snapshot = summary.get('run_state') in ('running', 'stopping')
    if live_snapshot:
        for item in read_results(directory, run_id):
            identifier = item.get('scene_id')
            if identifier in records:
                records[identifier] = item
    result = {**summary, 'scenes': list(records.values())}
    state_path = directory / 'run_state.json'
    try:
        state = read_json(state_path)
    except (OSError, ValueError):
        state = {}
    # Boundary checkpoints replace summary before run_state. A crash between
    # those writes leaves an older live run_state with the same run identity;
    # it must not downgrade the terminal summary or its completed count.
    if live_snapshot and isinstance(state, dict) and state.get('run_id') == run_id:
        for source, target in [('state', 'run_state'), ('updated_at', 'updated_at'),
                               ('completed_this_run', 'completed_this_run'),
                               ('batch_control', 'batch_control'),
                               ('provider_rate_control', 'provider_rate_control'),
                               ('disk_space_guard', 'disk_space_guard')]:
            if source in state:
                result[target] = state[source]
    result['finished'] = sum(item.get('status') not in ('pending', 'running') for item in records.values())
    return result
