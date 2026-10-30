"""Offline operational snapshots and reported provider usage; no model calls."""
from __future__ import annotations

from collections import Counter
import fnmatch
import hashlib
import os
from pathlib import Path
import re

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
    from batch_index import read_snapshot
    summary = read_snapshot(directory)
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
    controls = _batch_control_snapshot(directory, summary)
    return {'directory': str(directory), 'run_state': summary.get('run_state', 'unknown'),
            'updated_at': summary.get('updated_at'), 'total': summary.get('total', len(records)),
            'counts': dict(sorted(Counter(item.get('status', 'unknown') for item in records).items())),
            'accepted_artifacts_verified': verify, 'scenes': scenes,
            'omitted_scenes': max(0, len(records)-limit),
            'batch_control': controls.get('batch_control'),
            'provider_rate_control': controls.get('provider_rate_control'),
            'note': 'Read-only snapshot; running indicates the last coordinator checkpoint. Re-run the same run command to resume.'}


def _batch_control_snapshot(directory, summary=None):
    """Prefer live SQLite state, then a run checkpoint for a shared ledger.

    In shared-shard mode the ledger lives above the shard output directory, so
    a standalone status command cannot infer its path from just ``output``.
    The coordinator already checkpoints redacted aggregate counters there.
    """
    from batch_guard import snapshot as batch_snapshot
    direct = batch_snapshot(directory)
    candidates = []
    if isinstance(summary, dict):
        candidates.append(summary)
    state_path = Path(directory)/'run_state.json'
    if state_path.is_file():
        try:
            state = read_json(state_path)
            if isinstance(state, dict):
                candidates.insert(0, state)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    checkpoint = {}
    for value in candidates:
        if value.get('batch_control') is not None:
            checkpoint = {key: value[key] for key in ('batch_control', 'provider_rate_control')
                          if value.get(key) is not None}
            break
    if direct is not None:
        checkpoint['batch_control'] = direct
    return checkpoint


def _usage_stage(path):
    """Recognize factory output contracts, never infer a role from prompt text."""
    name, parent = path.name, path.parent
    if parent.name == 'initial_bundle_stages':
        if re.fullmatch(r'plan_attempt_\d+\.api\.json', name):
            return 'analysis.plan'
        if re.fullmatch(r'detail_.+_attempt_\d+\.api\.json', name):
            return 'analysis.detail'
    if (re.fullmatch(r'initial_bundle(?:_retry_\d+)?\.api\.json', name)
            and (parent / 'job.json').is_file()):
        return 'analysis.monolithic'
    iteration = parent.parent if parent.name == 'repair_from_best' else parent
    if re.fullmatch(r'iteration_\d+', iteration.name):
        if parent == iteration and re.fullmatch(r'review(?:_retry_\d+)?\.api\.json', name):
            return 'review'
        if re.fullmatch(r'repair_attempt_\d+\.api\.json', name):
            return 'repair.visual'
        if parent == iteration and re.fullmatch(r'technical_repair_attempt_\d+\.api\.json', name):
            return 'repair.technical'
    if parent.name == 'final_assessment' and re.fullmatch(r'review(?:_retry_\d+)?\.api\.json', name):
        return 'review'
    return 'unknown'


def _unsent_usage_attempt(attempt):
    # A connecting failure is also unsent. Contradictory provider evidence must
    # never disappear merely because a record contains request_started=False.
    return (isinstance(attempt, dict)
            and attempt.get('phase') in ('not_sent', 'connecting')
            and attempt.get('request_started') is False
            and attempt.get('request_sent') in (None, False)
            and attempt.get('http_status') is None
            and attempt.get('response_bytes') in (None, 0)
            and attempt.get('usage') in (None, {})
            and not attempt.get('usage_present')
            and not attempt.get('automatic_replay_blocked')
            and not attempt.get('usage_invalid')
            and not attempt.get('completion_marker_missing'))


def _local_usage_validation(path, record, stage):
    """Only staged records currently bind individual local verdicts to a call."""
    if stage not in ('analysis.plan', 'analysis.detail'):
        return 'unknown'
    state_path = path.with_name(path.name.removesuffix('.api.json') + '.validation.json')
    try:
        if state_path.is_symlink():
            return 'unknown'
        state = read_json(state_path)
        prompt_hash = record.get('prompt_sha256')
        number = int(re.search(r'_attempt_(\d+)\.api\.json$', path.name).group(1)) + 1
        if (not isinstance(prompt_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', prompt_hash)
                or state.get('actual_prompt_sha256') != prompt_hash
                or type(state.get('attempt')) is not int or state['attempt'] != number
                or record.get('status') != 'completed'):
            return 'unknown'
        if state.get('status') == 'failed':
            return 'rejected'
        if (state.get('status') == 'completed'
                and state.get('artifacts', {}).get('api') == hashlib.sha256(path.read_bytes()).hexdigest()):
            return 'validated'
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return 'unknown'


def _usage_stats(**identity):
    return {**identity, 'logical_calls': 0, 'transport_attempts': 0,
            'proven_unsent_attempts': 0, 'failed_calls': 0,
            'attempts_with_reported_usage': 0, 'attempts_without_reported_usage': 0,
            'tokens': {}, 'failed_call_tokens': {}, 'elapsed_seconds': 0,
            'payload_metrics': {'calls_with_metrics': 0, 'calls_with_serialized_request': 0,
                                'request_body_utf8_bytes': 0,
                                'attempts_with_request_body_metrics': 0,
                                'attempt_request_body_utf8_bytes': 0,
                                'input_image_count': 0, 'input_image_source_bytes': 0,
                                'input_image_base64_bytes': 0, 'response_bytes': 0,
                                'output_text_utf8_bytes': 0},
            'local_validation': {'validated_calls': 0, 'rejected_calls': 0,
                                 'unknown_calls': 0, 'rejected_tokens': {}}}


def _add_tokens(target, numbers):
    for key, value in numbers.items():
        target[key] = target.get(key, 0) + value


def _payload_counter(value):
    return value if type(value) is int and 0 <= value <= 10**15 else 0


def _add_payload_metrics(stats, record, attempts):
    """Aggregate recorded byte/pixel payload evidence without token inference."""
    payload = stats['payload_metrics']
    metrics = record.get('request_metrics')
    if isinstance(metrics, dict):
        payload['calls_with_metrics'] += 1
        body_bytes = metrics.get('serialized_request_body_utf8_bytes')
        if type(body_bytes) is int and 0 <= body_bytes <= 10**15:
            payload['calls_with_serialized_request'] += 1
            payload['request_body_utf8_bytes'] += body_bytes
        images = metrics.get('images')
        if isinstance(images, list):
            for image in images:
                if not isinstance(image, dict):
                    continue
                payload['input_image_count'] += 1
                payload['input_image_source_bytes'] += _payload_counter(image.get('source_bytes'))
                payload['input_image_base64_bytes'] += _payload_counter(image.get('base64_bytes'))
    for attempt in attempts:
        attempt_metrics = attempt.get('request_metrics')
        if isinstance(attempt_metrics, dict):
            attempt_bytes = attempt_metrics.get('serialized_request_body_utf8_bytes')
            if type(attempt_bytes) is int and 0 <= attempt_bytes <= 10**15:
                payload['attempts_with_request_body_metrics'] += 1
                payload['attempt_request_body_utf8_bytes'] += attempt_bytes
        payload['response_bytes'] += _payload_counter(attempt.get('response_bytes'))
    output_bytes = _payload_counter(record.get('output_text_utf8_bytes'))
    stats['payload_metrics']['output_text_utf8_bytes'] += output_bytes


def _usage_paths(directory):
    """Discover both record types in one walk, without following directory links.

    Yield tagged paths as they are discovered. API records are consumed by
    ``usage_report`` immediately; only budget paths need to be retained for
    deterministic sorting after accounting. Include matching directory names
    as well as files so malformed records retain their existing classification.
    """
    for parent, directories, files in os.walk(directory, followlinks=False):
        root = Path(parent)
        for name in directories + files:
            if fnmatch.fnmatch(name, '*.api.json'):
                yield 'api', root / name
            elif fnmatch.fnmatch(name, 'scene_budget.json'):
                yield 'budget', root / name


def usage_report(directory):
    """Sum only reported per-transport tokens; never infer a provider invoice."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError('Batch output directory does not exist')
    groups, stages, model_stages = {}, {}, {}
    unreadable = 0
    budget_paths = []
    for kind, path in _usage_paths(directory):
        if kind == 'budget':
            budget_paths.append(path)
            continue
        if path.is_symlink():
            continue
        try:
            record = read_json(path)
            config = record.get('config', {})
            identity = (str(config.get('provider', 'unknown')), str(config.get('model', 'unknown')))
            attempts = record.get('transport_attempts', [])
            if not isinstance(attempts, list) or any(not isinstance(item, dict) for item in attempts):
                raise ValueError('Invalid transport attempts')
            unsent = sum(_unsent_usage_attempt(item) for item in attempts)
            count = record.get('attempts', 0)
            if type(count) is not int or count < 0:
                raise ValueError('Invalid attempt count')
            count = max(count, len(attempts)) - unsent
            attempts = [item for item in attempts if not _unsent_usage_attempt(item)]
            reports = [attempt.get('usage') for attempt in attempts if isinstance(attempt, dict) and attempt.get('usage')]
            # Old provenance stored usage only at the logical-call level.
            if not reports and record.get('usage') and (count or not unsent):
                reports = [record['usage']]
            numbers = [{key: report[key] for key in TOKEN_FIELDS
                        if type(report.get(key)) is int and 0 <= report[key] <= 10**12}
                       for report in reports if isinstance(report, dict)]
            numbers = [report for report in numbers if report]
            stage = _usage_stage(path)
            validation = _local_usage_validation(path, record, stage)
            targets = [groups.setdefault(identity, _usage_stats(provider=identity[0], model=identity[1])),
                       stages.setdefault(stage, _usage_stats(stage=stage)),
                       model_stages.setdefault((*identity, stage), _usage_stats(stage=stage))]
            elapsed = record.get('elapsed_seconds', 0)
            for stats in targets:
                stats['logical_calls'] += 1
                stats['transport_attempts'] += count
                stats['proven_unsent_attempts'] += unsent
                stats['failed_calls'] += record.get('status') != 'completed'
                stats['attempts_with_reported_usage'] += len(numbers)
                stats['attempts_without_reported_usage'] += max(0, count - len(numbers))
                stats['local_validation'][validation + '_calls'] += 1
                if type(elapsed) in (int, float) and 0 <= elapsed <= 86400:
                    stats['elapsed_seconds'] += elapsed
                _add_payload_metrics(stats, record, attempts)
                for report in numbers:
                    _add_tokens(stats['tokens'], report)
                    if record.get('status') != 'completed':
                        _add_tokens(stats['failed_call_tokens'], report)
                    if validation == 'rejected':
                        _add_tokens(stats['local_validation']['rejected_tokens'], report)
        except (OSError, ValueError, AttributeError, TypeError):
            unreadable += 1
    models = [groups[key] for key in sorted(groups)]
    for item in [*models, *stages.values(), *model_stages.values()]:
        item['elapsed_seconds'] = round(item['elapsed_seconds'], 3)
    for identity, item in groups.items():
        item['stages'] = [model_stages[(*identity, stage)] for stage in sorted(stages)
                          if (*identity, stage) in model_stages]
    from scene_budget import budget_snapshot, BudgetError
    budgets, invalid_budgets = [], 0
    for path in sorted(budget_paths):
        try:
            if path.is_symlink():
                raise BudgetError('invalid_scene_budget', 'Budget ledger cannot be a symlink')
            snapshot = budget_snapshot(path.parent)
            budgets.append({'scene': str(path.parent.relative_to(directory)), **snapshot})
        except (BudgetError, TypeError):
            invalid_budgets += 1
    summary_path = directory/'summary.json'
    if summary_path.is_file():
        try:
            summary = read_json(summary_path)
        except (OSError, ValueError, TypeError, KeyError):
            summary = None
    else:
        summary = None
    controls = _batch_control_snapshot(directory, summary)
    payload_fields = tuple(_usage_stats()['payload_metrics'])
    payload_totals = {key: sum(item['payload_metrics'][key] for item in models)
                      for key in payload_fields}
    return {'directory': str(directory), 'logical_calls': sum(item['logical_calls'] for item in models),
            'transport_attempts': sum(item['transport_attempts'] for item in models), 'models': models,
            'payload_metrics': payload_totals,
            'proven_unsent_attempts': sum(item['proven_unsent_attempts'] for item in models),
            'attempts_without_reported_usage': sum(item['attempts_without_reported_usage'] for item in models),
            'stages': [stages[key] for key in sorted(stages)],
            'scene_budgets': budgets, 'invalid_budget_records': invalid_budgets,
            'batch_control': controls.get('batch_control'),
            'provider_rate_control': controls.get('provider_rate_control'),
            'unreadable_records': unreadable, 'invoice_complete': False,
            'scope': 'API provenance (*.api.json) only; Codex CLI logs are not included.',
            'elapsed_seconds_semantics': 'Sum of recorded logical-call durations, including waits/retries; not batch wall time or throughput.',
            'payload_metrics_semantics': 'payload_metrics are recorded byte counts only; serialized request, image, response and output sizes are not provider token estimates or billing.',
            'stage_classification': 'Recognized factory output paths only; other paths remain unknown.',
            'note': 'Provider-reported token fields only, never add totals to their components. Missing/partial usage and billing adjustments are unknown. Failed-call and locally rejected costs are evidence categories, not an estimate of avoidable waste. Local verdicts require matching staged prompt/attempt evidence; repair/review verdicts remain unknown. Visual rejection is not a local validation failure. No cost estimate.'}
