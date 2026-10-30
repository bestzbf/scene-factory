"""Persistent batch request/output limits and shared provider backpressure.

SQLite updates one aggregate row and one request row per transport attempt.
Output caps are reserved before transport; unknown usage never releases them.
Reported-token thresholds stop later requests and do not cap unknown input or
image costs. Enter one session around the coordinator to cover all its threads.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import errno
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

from scene_budget import _usage

LEDGER_NAME = 'batch_guard.sqlite3'
OWNER_DIRECTORY = '.batch_request_owners'
POLICY_FIELDS = ('max_requests', 'max_output_tokens', 'max_reported_tokens')
CIRCUIT_DEFAULTS = {'failure_threshold': 5, 'cooldown_seconds': 10}
_registry_lock = threading.RLock()
_active = {}


class BatchBudgetError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


# Public short name for transport adapters that already import BudgetError.
BudgetError = BatchBudgetError


def _invalid(message):
    return BatchBudgetError('invalid_batch_budget', message)


def normalize_policy(policy=None):
    policy = {} if policy is None else policy
    if not isinstance(policy, dict) or set(policy)-set(POLICY_FIELDS)-set(CIRCUIT_DEFAULTS):
        raise _invalid('Batch budget must contain only supported limits')
    result = {key: policy.get(key, 0) for key in POLICY_FIELDS}
    if any(type(value) is not int or not 0 <= value <= 10**15 for value in result.values()):
        raise _invalid('Batch budget limits must be nonnegative integers; zero disables a limit')
    return {**result, **normalize_circuit({key: policy[key] for key in CIRCUIT_DEFAULTS if key in policy})}


def normalize_circuit(circuit=None):
    circuit = {} if circuit is None else circuit
    if not isinstance(circuit, dict) or set(circuit)-set(CIRCUIT_DEFAULTS):
        raise _invalid('Batch circuit contains unknown controls')
    result = {key: circuit.get(key, value) for key, value in CIRCUIT_DEFAULTS.items()}
    if type(result['failure_threshold']) is not int or not 1 <= result['failure_threshold'] <= 1000:
        raise _invalid('Batch circuit failure_threshold must be between 1 and 1000')
    delay = result['cooldown_seconds']
    if type(delay) not in (int, float) or not math.isfinite(delay) or not 0 <= delay <= 86400:
        raise _invalid('Batch circuit cooldown_seconds must be between 0 and 86400')
    return result


def _inside(path, root):
    return path == root or root in path.parents


class BatchBudget:
    def __init__(self, root, policy, resume=False, stop_event=None, *, shared=False):
        self.root, self.policy, self.resume = root, policy, resume
        self.shared = bool(shared)
        self.circuit = {key: policy[key] for key in CIRCUIT_DEFAULTS}
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.path = root/LEDGER_NAME
        self._lock = threading.RLock()
        self._db = None
        # A shard's directory lock may be acquired by its replacement process.
        # Only this invocation's unique, never-reused lock identifies the
        # original owner of a pending provider request.
        self._owner_generation = uuid.uuid4().hex if self.shared else ''
        self._owner_fd = None
        self._owner_path = None

    def _acquire_owner_lock(self):
        if not self.shared:
            return
        directory = self.root / OWNER_DIRECTORY
        if directory.is_symlink():
            raise _invalid('Batch request owner directory must not be a symbolic link')
        directory.mkdir(exist_ok=True)
        path = directory / (self._owner_generation + '.lock')
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
        try:
            fd = os.open(path, flags, 0o600)
            self._owner_fd, self._owner_path = fd, path
            if os.name == 'nt':
                import msvcrt
                os.write(fd, b' ')
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if self._owner_fd is not None:
                os.close(self._owner_fd)
                self._owner_fd = None
                self._owner_path.unlink(missing_ok=True)
                self._owner_path = None
            raise _invalid('Batch request owner lock is unavailable; no new request was sent') from None

    @staticmethod
    def _lock_is_held(path):
        """Probe an existing generation lock without creating or replacing it."""
        path = Path(path)
        flags = os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
        try:
            fd = os.open(path, flags, 0o600)
        except OSError:
            return False
        try:
            if os.name == 'nt':
                import msvcrt
                if not os.fstat(fd).st_size:
                    return False
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    return exc.errno in (errno.EAGAIN, errno.EACCES)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                return False
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EACCES):
                    return True
                return False
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    def _pending_belongs_to_live_shards(self, db):
        """Permit live shared shards while still blocking crash ambiguity.

        Pending reservations are normally fail-closed: a fresh process cannot
        know whether a previous POST reached the provider. Shared shard mode
        is the one exception, and only when every pending attempt is covered
        by its original invocation's unique generation lock. A replacement
        coordinator can acquire the shard lock but cannot acquire that old
        generation's lock, so it cannot make an orphan reservation look live.
        Legacy rows without generation evidence always block recovery.
        """
        columns = {row['name'] for row in db.execute('PRAGMA table_info(attempts)')}
        if 'owner_generation' not in columns:
            return False
        rows = db.execute("SELECT output,owner_generation FROM attempts WHERE state='reserved'").fetchall()
        if not rows:
            return False
        owners = set()
        for row in rows:
            try:
                path = (self.root / Path(row['output'])).resolve()
                if not path.is_relative_to(self.root):
                    return False
                owner = row['owner_generation']
                if not isinstance(owner, str) or not re.fullmatch(r'[0-9a-f]{32}', owner):
                    return False
                owners.add(owner)
            except (OSError, ValueError, TypeError):
                return False
        directory = self.root / OWNER_DIRECTORY
        if directory.is_symlink():
            return False
        return all(owner != self._owner_generation and
                   self._lock_is_held(directory / (owner + '.lock')) for owner in owners)

    @contextmanager
    def _transaction(self):
        with self._lock:
            try:
                self._db.execute('BEGIN IMMEDIATE')
                yield self._db
                self._db.commit()
            except sqlite3.Error:
                self._db.rollback()
                self.stop_event.set()
                raise _invalid('Batch ledger transaction failed; no new requests are permitted') from None
            except BaseException:
                self._db.rollback()
                raise

    def initialize(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if any(Path(str(self.path)+suffix).is_symlink() for suffix in ('', '-wal', '-shm', '-journal')):
            raise _invalid('Batch ledger files must not be symbolic links')
        existed = self.path.exists()
        # Never silently introduce a fresh ledger after already billed work.
        if not existed:
            for path in self.root.rglob('*.api.json'):
                try:
                    record = json.loads(path.read_text())
                except (ValueError, OSError):
                    raise _invalid('Existing API audit is unreadable; cannot initialize batch usage') from None
                if not isinstance(record, dict) or record.get('attempts', 0) or record.get('transport_attempts'):
                    raise _invalid('Existing API attempts require their original batch ledger or a fresh output directory')
        try:
            self._acquire_owner_lock()
            self._db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute('PRAGMA journal_mode=WAL')
            self._db.execute('PRAGMA synchronous=FULL')
            with self._transaction() as db:
                db.execute('''CREATE TABLE IF NOT EXISTS state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), version INTEGER NOT NULL,
                    root TEXT NOT NULL, policy TEXT NOT NULL, circuit TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0, output_tokens_accounted INTEGER NOT NULL DEFAULT 0,
                    reported_tokens INTEGER NOT NULL DEFAULT 0, unknown_usage_attempts INTEGER NOT NULL DEFAULT 0,
                    pending_attempts INTEGER NOT NULL DEFAULT 0, consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    pending_output_tokens INTEGER NOT NULL DEFAULT 0,
                    cooldown_until REAL NOT NULL DEFAULT 0, stop_reason TEXT NOT NULL DEFAULT '')''')
                db.execute('''CREATE TABLE IF NOT EXISTS attempts (
                    request_id TEXT PRIMARY KEY, output TEXT NOT NULL, reserved_output_tokens INTEGER NOT NULL,
                    service TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'reserved', transport TEXT,
                    owner_generation TEXT NOT NULL DEFAULT '')''')
                db.execute('''CREATE TABLE IF NOT EXISTS services (
                    service TEXT PRIMARY KEY, consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    cooldown_until REAL NOT NULL DEFAULT 0)''')
                row = db.execute('SELECT * FROM state WHERE singleton=1').fetchone()
                identity = (str(self.root), json.dumps(self.policy, sort_keys=True), json.dumps(self.circuit, sort_keys=True))
                if row is None:
                    if existed:
                        raise _invalid('Existing batch ledger has no state; do not reset it')
                    db.execute('INSERT INTO state(singleton, version, root, policy, circuit) VALUES(1,3,?,?,?)', identity)
                elif row['version'] not in (1, 2, 3) or (row['root'], row['policy'], row['circuit']) != identity:
                    raise _invalid('Batch root or controls changed; use the original policy or a new output directory')
                elif row['pending_attempts'] and not (self.shared and self._pending_belongs_to_live_shards(db)):
                    self.stop_event.set()
                    raise BatchBudgetError('ambiguous_previous_request',
                        'A previous batch request has a pending reservation and may already have consumed tokens; '
                        'automatic resume is blocked. Inspect provider records; do not reset the ledger.')
                else:
                    # A v1 ledger with no pending requests has no in-flight
                    # output capacity. Migrate once without scanning history.
                    # Pending v1 ledgers remain blocked by the replay guard.
                    if row['version'] == 1:
                        db.execute('ALTER TABLE state ADD COLUMN pending_output_tokens INTEGER NOT NULL DEFAULT 0')
                    if row['version'] in (1, 2):
                        columns = {item['name'] for item in db.execute('PRAGMA table_info(attempts)')}
                        if 'owner_generation' not in columns:
                            db.execute("ALTER TABLE attempts ADD COLUMN owner_generation TEXT NOT NULL DEFAULT ''")
                        db.execute('UPDATE state SET version=3 WHERE singleton=1')
                    if row['stop_reason']:
                        if self.resume and row['stop_reason'] not in POLICY_FIELDS:
                            db.execute("UPDATE state SET stop_reason='' WHERE singleton=1")
                            db.execute('UPDATE services SET consecutive_failures=0')
                        else:
                            self.stop_event.set()
        except sqlite3.Error:
            self.close()
            raise _invalid('Batch ledger is unreadable; do not reset previous reservations') from None
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._owner_fd is not None:
            # Closing releases the kernel lock even on abnormal process exit.
            # Generation names are never reused; retain crash leftovers as
            # unlocked evidence, and remove only this cleanly closed owner.
            os.close(self._owner_fd)
            self._owner_fd = None
            self._owner_path.unlink(missing_ok=True)
            self._owner_path = None

    def snapshot(self):
        with self._lock:
            return _snapshot_row(self._db.execute('SELECT * FROM state WHERE singleton=1').fetchone())

    @property
    def stop_reason(self):
        return self.snapshot()['summary']['stop_reason']

    def reserve(self, output, max_output_tokens, config, deadline):
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 262144:
            raise _invalid('Batch reservation requires a positive request output cap')
        while True:
            if time.monotonic() >= deadline:
                raise BatchBudgetError('timeout', 'Batch provider queue exceeded this request deadline')
            stopped = None
            cooldown = 0
            capacity_wait = False
            with self._transaction() as db:
                row = db.execute('SELECT * FROM state WHERE singleton=1').fetchone()
                stopped = row['stop_reason']
                if not stopped:
                    # Pending attempts may settle below their output caps, or
                    # be proven unsent and refund a request slot. Only fixed
                    # settled/unknown usage can latch permanent exhaustion.
                    fixed_requests = row['requests'] - row['pending_attempts']
                    fixed_output = row['output_tokens_accounted'] - row['pending_output_tokens']
                    checks = (
                        ('max_requests', fixed_requests >= self.policy['max_requests']),
                        ('max_output_tokens', fixed_output + max_output_tokens > self.policy['max_output_tokens']),
                        ('max_reported_tokens', row['reported_tokens'] >= self.policy['max_reported_tokens']))
                    stopped = next((name for name, exceeded in checks if self.policy[name] and exceeded), '')
                    if stopped:
                        db.execute('UPDATE state SET stop_reason=? WHERE singleton=1', (stopped,))
                    capacity_wait = ((self.policy['max_requests'] and row['requests'] >= self.policy['max_requests'])
                        or (self.policy['max_output_tokens'] and
                            row['output_tokens_accounted'] + max_output_tokens > self.policy['max_output_tokens']))
                if not stopped and not capacity_wait:
                    service = json.dumps([config.get('provider'), config.get('base_url')], separators=(',', ':'))
                    db.execute('INSERT OR IGNORE INTO services(service) VALUES(?)', (service,))
                    service_row = db.execute('SELECT * FROM services WHERE service=?', (service,)).fetchone()
                    cooldown = max(0, service_row['cooldown_until']-time.time())
                    if not cooldown:
                        if self.stop_event.is_set():
                            raise BatchBudgetError('batch_stopped', 'Batch is stopping; no new request was sent')
                        request_id = uuid.uuid4().hex
                        db.execute('INSERT INTO attempts(request_id,output,reserved_output_tokens,service,owner_generation) VALUES(?,?,?,?,?)',
                                   (request_id, output.relative_to(self.root).as_posix(), max_output_tokens, service, self._owner_generation))
                        db.execute('''UPDATE state SET requests=requests+1,
                            output_tokens_accounted=output_tokens_accounted+?,
                            pending_output_tokens=pending_output_tokens+?,
                            pending_attempts=pending_attempts+1, unknown_usage_attempts=unknown_usage_attempts+1
                            WHERE singleton=1''', (max_output_tokens, max_output_tokens))
            if stopped:
                self.stop_event.set()
                code = 'batch_budget_exceeded' if stopped in POLICY_FIELDS else 'batch_circuit_open'
                raise BatchBudgetError(code, 'Batch request policy stopped further requests: '+stopped)
            if not cooldown and not capacity_wait:
                return _Ticket(self, request_id)
            remaining = deadline-time.monotonic()
            if remaining <= 0 or cooldown >= remaining:
                raise BatchBudgetError('timeout', 'Shared provider cooldown exceeds this request deadline')
            # Poll only the aggregate row while another request holds capacity;
            # this also observes settlements made by another process.
            delay = .05 if capacity_wait else min(cooldown, 1)
            if self.stop_event.wait(min(delay, remaining)):
                reason = self.stop_reason
                if reason:
                    code = 'batch_budget_exceeded' if reason in POLICY_FIELDS else 'batch_circuit_open'
                    raise BatchBudgetError(code, 'Batch request policy stopped further requests: '+reason)
                raise BatchBudgetError('batch_stopped', 'Batch is stopping; no new request was sent')

    def finish(self, request_id, transport):
        usage = _usage(transport)
        status = transport.get('http_status')
        # DNS/TCP/TLS setup can fail before the POST starts. Match provider
        # admission refunds: a proven-unsent connection attempt consumes no
        # request/output capacity. Transport failures still feed the circuit.
        not_sent = (transport.get('phase') in ('not_sent', 'connecting') and transport.get('request_started') is False
                    and status is None and transport.get('response_bytes') in (None, 0)
                    and transport.get('usage') in (None, {})
                    and transport.get('request_sent') in (None, False) and not transport.get('automatic_replay_blocked')
                    and not transport.get('usage_invalid'))
        if not_sent:
            usage = {'reported_output_tokens': 0, 'reported_tokens': 0, 'usage_complete': True}
        with self._transaction() as db:
            item = db.execute('SELECT * FROM attempts WHERE request_id=?', (request_id,)).fetchone()
            if item is None:
                raise _invalid('Batch reservation is missing; no capacity can be released')
            if item['state'] == 'settled':
                return
            output = usage['reported_output_tokens'] if usage['usage_complete'] else max(
                item['reserved_output_tokens'], usage['reported_output_tokens'] or 0)
            db.execute('UPDATE attempts SET state=\'settled\', transport=? WHERE request_id=?',
                       (json.dumps(transport, sort_keys=True, allow_nan=False), request_id))
            row = db.execute('SELECT * FROM state WHERE singleton=1').fetchone()
            service_row = db.execute('SELECT * FROM services WHERE service=?', (item['service'],)).fetchone()
            failure = (bool(transport.get('network_error_type')) or
                       (not not_sent and (status is None or status == 429 or
                        (type(status) is int and 500 <= status <= 599) or
                        transport.get('automatic_replay_blocked') is True)))
            failures = (service_row['consecutive_failures']+1 if failure else
                        service_row['consecutive_failures'] if not_sent else 0)
            stopped = row['stop_reason']
            cooldown_until = service_row['cooldown_until']
            if status in (401, 402, 403):
                stopped = stopped or 'http_'+str(status)
            elif failure:
                if failures >= self.circuit['failure_threshold']:
                    stopped = stopped or 'consecutive_provider_failures'
                retry = transport.get('retry_after_seconds', 0)
                retry = retry if type(retry) in (float, int) and math.isfinite(retry) else 0
                cooldown_until = max(cooldown_until, time.time()+max(self.circuit['cooldown_seconds'], retry))
            db.execute('UPDATE services SET consecutive_failures=?, cooldown_until=? WHERE service=?',
                       (failures, cooldown_until, item['service']))
            db.execute('''UPDATE state SET requests=requests-?, output_tokens_accounted=output_tokens_accounted+?,
                reported_tokens=reported_tokens+?, pending_attempts=pending_attempts-1,
                pending_output_tokens=pending_output_tokens-?,
                unknown_usage_attempts=unknown_usage_attempts-?, consecutive_failures=?,
                cooldown_until=?, stop_reason=? WHERE singleton=1''',
                (int(not_sent), output-item['reserved_output_tokens'], usage['reported_tokens'],
                 item['reserved_output_tokens'], int(usage['usage_complete']),
                 failures, cooldown_until, stopped))
        if stopped:
            self.stop_event.set()


class _Ticket:
    def __init__(self, budget, request_id):
        self.request_id, self._budget = request_id, budget

    def finish(self, transport):
        self._budget.finish(self.request_id, transport)


@contextmanager
def batch_session(root, policy=None, resume=False, stop_event=None, *, shared=False):
    """Bind the whole resolved output tree across threads to durable controls."""
    root = Path(root).expanduser().resolve()
    if type(resume) is not bool:
        raise _invalid('Batch resume must be an explicit boolean')
    with _registry_lock:
        if any(_inside(root, other) or _inside(other, root) for other in _active):
            raise _invalid('Overlapping batch sessions are not permitted')
        budget = BatchBudget(root, normalize_policy(policy), resume, stop_event, shared=shared)
        budget.initialize()
        _active[root] = budget
    try:
        yield budget
    finally:
        with _registry_lock:
            del _active[root]
        budget.close()


def reserve_request(output, max_output_tokens, config, *, deadline=None):
    output = Path(output).expanduser().resolve()
    with _registry_lock:
        budget = next((value for root, value in _active.items() if _inside(output, root)), None)
    if deadline is None:
        deadline = time.monotonic()+config.get('timeout_seconds', 600)
    return budget.reserve(output, max_output_tokens, config, deadline) if budget is not None else None


def current_stop(output=None):
    with _registry_lock:
        budgets = [budget for root, budget in _active.items()
                   if output is None or _inside(Path(output).expanduser().resolve(), root)]
    for budget in budgets:
        reason = budget.stop_reason
        if reason:
            return reason
    return None


def current_stop_event(output=None):
    """Return the active batch cancellation event for a request output tree."""
    path = Path(output).expanduser().resolve() if output is not None else None
    with _registry_lock:
        budget = next((value for root, value in _active.items()
                       if path is None or _inside(path, root)), None)
    return budget.stop_event if budget is not None else None


def _snapshot_row(row):
    if row is None:
        raise _invalid('Batch ledger has no state')
    return {'policy': json.loads(row['policy']), 'circuit': json.loads(row['circuit']),
            'summary': {key: row[key] for key in ('requests', 'output_tokens_accounted', 'reported_tokens',
                'unknown_usage_attempts', 'pending_attempts', 'consecutive_failures', 'cooldown_until', 'stop_reason')},
            'note': 'Output capacity is conservative; reported-token thresholds cannot cap unknown input/image costs.'}


def snapshot(root):
    """Read counters without a session, locks, writes, or scanning request history."""
    path = Path(root).expanduser().resolve()/LEDGER_NAME
    if path.is_symlink():
        raise _invalid('Batch ledger must not be a symbolic link')
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            return _snapshot_row(db.execute('SELECT * FROM state WHERE singleton=1').fetchone())
    except (sqlite3.Error, ValueError, KeyError):
        raise _invalid('Batch ledger snapshot is unreadable; do not reset it') from None
