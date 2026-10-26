"""Offline operational snapshots and reported provider usage; no model calls."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

from batch import read_json
from multiview_spec import SAFE_ID

TOKEN_FIELDS = ('input_tokens', 'output_tokens', 'total_tokens', 'prompt_tokens',
                'completion_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')


def batch_status(directory, *, verify=False, limit=100):
    directory = Path(directory).expanduser().resolve()
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError('status limit must be an integer between 1 and 10000')
    summary = read_json(directory/'summary.json')
    records = summary['scenes']
    if verify:
        from batch_multiview import recorded_results
        records = list(recorded_results(directory).values())
    scenes = []
    for record in records[:limit]:
        scene_id = record.get('scene_id', '')
        if not isinstance(scene_id, str) or not SAFE_ID.fullmatch(scene_id):
            raise ValueError('Invalid recorded scene identity')
        scene = {key: record[key] for key in ('scene_id', 'status', 'iteration',
                 'elapsed_seconds', 'failed_stage', 'error', 'stop_reason') if key in record}
        progress = directory/scene_id/'progress.json'
        if progress.is_file():
            try:
                value = read_json(progress)
                scene['last_progress'] = {key: value[key] for key in ('stage', 'at', 'iteration', 'status') if key in value}
            except (OSError, ValueError, TypeError):
                scene['progress_unavailable'] = True
        from scene_budget import budget_snapshot, BudgetError
        try:
            budget = budget_snapshot(directory / scene_id)
            if budget is not None:
                scene['budget'] = budget
        except BudgetError:
            scene['budget_unavailable'] = True
        scenes.append(scene)
    return {'directory': str(directory), 'run_state': summary.get('run_state', 'unknown'),
            'updated_at': summary.get('updated_at'), 'total': summary.get('total', len(records)),
            'counts': dict(sorted(Counter(item.get('status', 'unknown') for item in records).items())),
            'accepted_artifacts_verified': verify, 'scenes': scenes,
            'omitted_scenes': max(0, len(records)-limit),
            'note': 'Read-only snapshot; running indicates the last coordinator checkpoint. Re-run the same run command to resume.'}


def usage_report(directory):
    """Sum only reported per-transport tokens; never infer a provider invoice."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError('Batch output directory does not exist')
    groups = {}
    unreadable = 0
    for path in directory.rglob('*.api.json'):
        if path.is_symlink():
            continue
        try:
            record = read_json(path)
            config = record.get('config', {})
            identity = (str(config.get('provider', 'unknown')), str(config.get('model', 'unknown')))
            stats = groups.setdefault(identity, {'provider': identity[0], 'model': identity[1],
                     'logical_calls': 0, 'transport_attempts': 0, 'failed_calls': 0,
                     'attempts_with_reported_usage': 0, 'tokens': {}, 'elapsed_seconds': 0})
            attempts = record.get('transport_attempts', [])
            if not isinstance(attempts, list):
                raise ValueError('Invalid transport attempts')
            count = record.get('attempts', 0)
            if type(count) is not int or count < 0:
                raise ValueError('Invalid attempt count')
            reports = [attempt.get('usage') for attempt in attempts if isinstance(attempt, dict) and attempt.get('usage')]
            # Old provenance stored usage only at the logical-call level.
            if not reports and record.get('usage'):
                reports = [record['usage']]
            stats['logical_calls'] += 1
            stats['transport_attempts'] += max(count, len(attempts))
            stats['failed_calls'] += record.get('status') != 'completed'
            elapsed = record.get('elapsed_seconds', 0)
            if type(elapsed) in (int, float) and 0 <= elapsed <= 86400:
                stats['elapsed_seconds'] += elapsed
            for report in reports:
                if not isinstance(report, dict):
                    continue
                numbers = {key: report[key] for key in TOKEN_FIELDS
                           if type(report.get(key)) is int and 0 <= report[key] <= 10**12}
                if numbers:
                    stats['attempts_with_reported_usage'] += 1
                    for key, value in numbers.items():
                        stats['tokens'][key] = stats['tokens'].get(key, 0) + value
        except (OSError, ValueError, AttributeError, TypeError):
            unreadable += 1
    models = [groups[key] for key in sorted(groups)]
    for item in models:
        item['elapsed_seconds'] = round(item['elapsed_seconds'], 3)
    from scene_budget import budget_snapshot, BudgetError
    budgets, invalid_budgets = [], 0
    for path in sorted(directory.rglob('scene_budget.json')):
        try:
            if path.is_symlink():
                raise BudgetError('invalid_scene_budget', 'Budget ledger cannot be a symlink')
            snapshot = budget_snapshot(path.parent)
            budgets.append({'scene': str(path.parent.relative_to(directory)), **snapshot})
        except (BudgetError, TypeError):
            invalid_budgets += 1
    return {'directory': str(directory), 'logical_calls': sum(item['logical_calls'] for item in models),
            'transport_attempts': sum(item['transport_attempts'] for item in models), 'models': models,
            'scene_budgets': budgets, 'invalid_budget_records': invalid_budgets,
            'unreadable_records': unreadable, 'invoice_complete': False,
            'note': 'Provider-reported tokens only, including failed attempts when usage was returned. Missing responses/usage and provider billing adjustments are unknown. No cost estimate.'}
