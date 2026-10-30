"""Read-only status of a running or finished scene-factory batch.

    python3 -I monitor_full_batch.py RUN_DIR [--groups GROUPS_JSON | --total N] [--top 25]

Shows progress, rendered/blocked/failed counts, token use with a projection for the
whole batch, ETA, scenes that spent their single model request without a layout,
and the asset categories that block the most scenes.

Reads batch_events.jsonl, the batch ledger (SQLite, opened read-only) and each failed
scene's asset_selection.json. RUN_DIR.pid (written by the launcher) tells whether the
batch is alive. Makes no network or model calls and writes nothing.
"""
import argparse
import collections
import datetime
import json
import os
import sqlite3
import time
from pathlib import Path



def events(run):
    path = run / 'batch_events.jsonl'
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            out.append(json.loads(line)['result'])
    return out


def ledger(run, alive):
    """Batch ledger totals. A WAL database cannot be opened mode=ro once its owner has
    exited (the -shm file is gone), so a stopped batch is read with immutable=1."""
    path = run / 'batch_guard.sqlite3'
    if not path.exists():
        return {}
    flags = 'mode=ro' if alive else 'mode=ro&immutable=1'
    try:
        db = sqlite3.connect(f'file:{path}?{flags}', uri=True)
        db.row_factory = sqlite3.Row
        try:
            row = db.execute('SELECT requests, output_tokens_accounted, reported_tokens, pending_attempts, '
                             'unknown_usage_attempts, consecutive_failures, stop_reason FROM state').fetchone()
            return dict(row) if row else {}
        finally:
            db.close()
    except sqlite3.Error as exc:
        return {'unreadable': type(exc).__name__}


def kind(result):
    if result.get('status') in ('accepted', 'generated_unreviewed', 'needs_review'):
        return 'rendered'
    stage, error = result.get('failed_stage') or '', result.get('error') or ''
    if stage == 'joint_photo_analysis':
        if 'HTTP 429' in error:
            return 'model_call_http_429'
        if 'stopped further requests' in error:
            return 'model_call_refused_batch_stopped'
        if 'may have executed' in error:
            return 'model_call_lost_after_send'
        if 'network' in error.lower() or 'connection' in error.lower():
            return 'model_call_network_unsent'
        return 'model_call_other'
    if stage == 'automatic_asset_selection':
        return 'asset_selection_blocked'
    return f'failed_at_{stage or "unknown"}'


def missing(run, result):
    path = run / result['scene_id'] / 'iteration_00' / 'asset_selection.json'
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding='utf-8'))
    objects = report.get('objects', {})
    blocked = [e for e in report.get('errors', []) if e.get('code') == 'missing_compatible_asset']
    reasons = collections.Counter()
    for error in blocked:
        obj = objects.get(error.get('object_id'), {})
        rejected = sorted((obj.get('rejection_counts') or {}).keys())
        reasons[(obj.get('category'), 'no_category' if not rejected else '+'.join(rejected))] += 1
    other = collections.Counter(e.get('code') for e in report.get('errors', []) if e.get('code') != 'missing_compatible_asset')
    return {'objects': len(objects), 'blocked': len(blocked), 'reasons': reasons, 'other': other}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--groups', type=Path, help='groups.json the batch was started with (for the total while it runs)')
    parser.add_argument('--total', type=int, default=None, help='scenes requested (default: from run_state.json)')
    parser.add_argument('--top', type=int, default=25)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.total is None:
        state_file = run / 'run_state.json'
        state_data = json.loads(state_file.read_text()) if state_file.exists() else {}
        # run_state.json lists the requested scenes once the batch finishes; while it runs,
        # count the scenes in the groups file the batch was started with.
        requested = state_data.get('requested_scene_ids')
        if requested is None:
            requested = json.loads(args.groups.read_text())['scenes'] if args.groups else []
        args.total = len(requested)
    pid_file = run.with_name(run.name + '.pid')
    pid = int(pid_file.read_text()) if pid_file.exists() else None
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
    results = events(run)
    kinds = collections.Counter(kind(r) for r in results)
    print(f'batch pid {pid} {"RUNNING" if alive else "NOT RUNNING"}; finished {len(results)} / {args.total} scenes')
    for name, count in kinds.most_common():
        print(f'  {name:32s} {count:5d}  ({count / max(len(results), 1):.0%})')
    summary = run / 'summary.json'
    if summary.exists() and not alive:
        info = json.loads(summary.read_text(encoding='utf-8'))
        stop = ((info.get('batch_control') or {}).get('summary') or {}).get('stop_reason')
        print(f'run_state {info.get("run_state")!r}, stop_reason {stop!r}')
    state = ledger(run, alive)
    if state:
        print('ledger:', json.dumps(state, ensure_ascii=False))
    starts = [datetime.datetime.fromisoformat(r['started_at']).timestamp() for r in results if r.get('started_at')]
    if starts:
        elapsed = time.time() - min(starts)
        rate = len(results) / elapsed * 3600
        left = args.total - len(results)
        print(f'throughput {rate:.1f} scenes/h over {elapsed / 3600:.1f} h; ~{left / max(rate, 1e-9):.1f} h left at this rate')
        if state.get('requests'):
            per = state['reported_tokens'] / max(state['requests'], 1)
            print(f'tokens: {state["reported_tokens"]:,} reported over {state["requests"]} requests (~{per:,.0f}/request); '
                  f'projected ~{per * args.total / 1e6:.1f}M for {args.total} scenes')
    stuck = []
    for scene in sorted(p for p in run.iterdir() if p.is_dir()):
        budget = scene / 'scene_budget.json'
        if (budget.exists() and not (scene / 'analysis_provenance.json').exists()
                and (scene / 'last_error.json').exists()          # finished (failed), not in flight
                and json.loads(budget.read_text(encoding='utf-8')).get('requests', 0) >= 1):
            stuck.append(scene.name)
    if stuck:
        print(f'{len(stuck)} scenes used their one model request without a committed layout; a resume skips them, '
              f'they need a separate run: {" ".join(stuck)}')
    blocked = [info for info in (missing(run, r) for r in results if kind(r) == 'asset_selection_blocked') if info]
    if blocked:
        fractions = sorted(info['blocked'] / max(info['objects'], 1) for info in blocked)
        reasons, other, per_scene = collections.Counter(), collections.Counter(), collections.Counter()
        for info in blocked:
            other.update(info['other'])
            per_scene.update({category for category, _ in info['reasons']})   # once per scene
            for (_, reason), count in info['reasons'].items():
                reasons[reason] += count
        single = sum(1 for info in blocked if info['blocked'] == 1)
        print(f'asset-blocked scenes: median {fractions[len(fractions) // 2]:.0%} of objects blocked; '
              f'{single} blocked by a single object; reasons {dict(reasons)}; other errors {dict(other)}')
        print(f'top missing categories (scenes blocked), {len(per_scene)} distinct:')
        for category, count in per_scene.most_common(args.top):
            print(f'  {count:4d}  {category}')


if __name__ == '__main__':
    main()
