"""Optional durable provider RPM/TPM admission, shared across scene threads.

A session groups every model on the same provider/base_url into one quota scope;
credentials and provider payloads are never stored. SQLite retains at most 61
one-second buckets per scope and only current-window outstanding reservations.
The extra second makes the rolling-minute approximation conservative.

TPM admission reserves max_output_tokens + the operator's explicit input token
allowance (including all images/schema). This is shaping, not a tokenizer or a
billing cap: unknown tokenizer/image costs cannot be inferred. Trustworthy full
usage settles the allowance; missing/partial/invalid usage keeps it reserved.
Unknown POST outcomes are never released early. Existing replay guards remain
responsible for whether a previous request can safely be sent again.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import sqlite3
import threading
import time
import uuid

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

from scene_budget import _usage

LEDGER_NAME = 'provider_limits.sqlite3'
FIELDS = ('requests_per_minute', 'tokens_per_minute', 'input_tokens_per_request')
_registry_lock = threading.RLock()
_active = {}


class ProviderLimitError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def normalize_rate_limit(value=None, *, max_output_tokens=32768):
    """Strict, complete opt-in config; zero disables the corresponding quota."""
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - set(FIELDS):
        raise ValueError('rate_limit must contain only requests_per_minute, tokens_per_minute, input_tokens_per_request')
    result = {key: value.get(key, 0) for key in FIELDS}
    if any(type(number) is not int or not 0 <= number <= 10**12 for number in result.values()):
        raise ValueError('rate_limit controls must be nonnegative integers no greater than 1000000000000')
    if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 262144:
        raise ValueError('rate_limit requires a valid max_output_tokens')
    if result['tokens_per_minute']:
        if not result['input_tokens_per_request']:
            raise ValueError('TPM requires positive input_tokens_per_request including prompt, schema and images')
        if max_output_tokens + result['input_tokens_per_request'] > result['tokens_per_minute']:
            raise ValueError('One max_output_tokens + input_tokens_per_request reservation exceeds tokens_per_minute')
    return result


def _scope(config):
    # model and API key do not partition a quota shared by an endpoint.
    return hashlib.sha256((config['provider'] + '\n' + config['base_url']).encode()).hexdigest()


def _inside(path, root):
    return path == root or root in path.parents


def _unsent(transport):
    return (transport.get('phase') in ('not_sent', 'connecting')
            and transport.get('request_started') is False
            and transport.get('request_sent') in (None, False)
            and transport.get('http_status') is None
            and transport.get('response_bytes') in (None, 0)
            and transport.get('usage') in (None, {})
            and not transport.get('automatic_replay_blocked')
            and not transport.get('usage_invalid'))


class ProviderLimiter:
    """One ledger; transactions coordinate independent processes on local disk."""
    def __init__(self, root, configs, stop_event=None, *, clock=None, monotonic=None, waiter=None):
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / LEDGER_NAME
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.clock = clock or time.time
        self.monotonic = monotonic or time.monotonic
        self.waiter = waiter or self.stop_event.wait
        self._lock = threading.RLock()
        self._db = None
        # Configs arriving here must already have canonical provider/base_url.
        self.policies = {}
        grouped = {}
        for config in configs:
            control = normalize_rate_limit(config.get('rate_limit'), max_output_tokens=config.get('max_output_tokens', 32768))
            grouped.setdefault(_scope(config), []).append((config, control))
        for scope, entries in grouped.items():
            rpm = [control['requests_per_minute'] for _, control in entries if control['requests_per_minute']]
            tpm = [control['tokens_per_minute'] for _, control in entries if control['tokens_per_minute']]
            policy = {'requests_per_minute': min(rpm) if rpm else 0,
                      'tokens_per_minute': min(tpm) if tpm else 0}
            if not any(policy.values()):
                continue
            if policy['tokens_per_minute']:
                for config, control in entries:
                    required = config.get('max_output_tokens', 32768) + control['input_tokens_per_request']
                    if not control['input_tokens_per_request'] or required > policy['tokens_per_minute']:
                        raise ValueError('Every model sharing a TPM endpoint needs a positive input reservation that fits its strictest quota')
            self.policies[scope] = policy

    @contextmanager
    def _transaction(self):
        with self._lock:
            try:
                self._db.execute('BEGIN IMMEDIATE')
                yield self._db
                self._db.commit()
            except sqlite3.Error:
                self._db.rollback()
                raise ProviderLimitError('invalid_provider_rate_ledger', 'Provider rate ledger transaction failed; no new request was sent') from None
            except BaseException:
                self._db.rollback()
                raise

    @staticmethod
    def _prune(db, now):
        # A bucket containing second t remains active until t+61, never less
        # than 60 seconds after any admission in that bucket.
        cutoff = math.floor(now) - 60
        db.execute('DELETE FROM buckets WHERE second < ?', (cutoff,))
        db.execute('DELETE FROM pending WHERE second < ?', (cutoff,))

    def initialize(self):
        if not self.policies:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        if any(Path(str(self.path)+suffix).is_symlink() for suffix in ('', '-wal', '-shm', '-journal')):
            raise ProviderLimitError('invalid_provider_rate_ledger', 'Provider rate ledger files must not be symbolic links')
        try:
            self._db = sqlite3.connect(self.path, timeout=1, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute('PRAGMA journal_mode=WAL')
            self._db.execute('PRAGMA synchronous=FULL')
            with self._transaction() as db:
                db.execute('CREATE TABLE IF NOT EXISTS metadata(version INTEGER NOT NULL)')
                row = db.execute('SELECT version FROM metadata').fetchall()
                if not row:
                    db.execute('INSERT INTO metadata VALUES(1)')
                elif len(row) != 1 or row[0]['version'] != 1:
                    raise ProviderLimitError('invalid_provider_rate_ledger', 'Provider rate ledger version is unsupported')
                db.execute('''CREATE TABLE IF NOT EXISTS policies (
                    scope TEXT PRIMARY KEY, rpm INTEGER NOT NULL, tpm INTEGER NOT NULL)''')
                db.execute('''CREATE TABLE IF NOT EXISTS buckets (
                    scope TEXT NOT NULL, second INTEGER NOT NULL, requests INTEGER NOT NULL,
                    tokens INTEGER NOT NULL, PRIMARY KEY(scope,second))''')
                db.execute('''CREATE TABLE IF NOT EXISTS pending (
                    request_id TEXT PRIMARY KEY, scope TEXT NOT NULL, second INTEGER NOT NULL,
                    reserved_tokens INTEGER NOT NULL)''')
                db.execute('CREATE INDEX IF NOT EXISTS pending_second ON pending(second)')
                self._prune(db, self.clock())
                for scope, policy in self.policies.items():
                    row = db.execute('SELECT rpm,tpm FROM policies WHERE scope=?', (scope,)).fetchone()
                    values = (policy['requests_per_minute'], policy['tokens_per_minute'])
                    active = db.execute('SELECT 1 FROM buckets WHERE scope=? LIMIT 1', (scope,)).fetchone()
                    if row and (row['rpm'], row['tpm']) != values and active:
                        raise ProviderLimitError('provider_rate_policy_changed', 'Provider quota changed inside an active rolling minute; keep the original quota or wait for expiry')
                    db.execute('INSERT OR REPLACE INTO policies VALUES(?,?,?)', (scope, *values))
        except sqlite3.Error:
            self.close()
            raise ProviderLimitError('invalid_provider_rate_ledger', 'Provider rate ledger is unreadable; no new request was sent') from None
        except BaseException:
            self.close()
            raise

    def close(self):
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    def reserve(self, config, deadline):
        scope = _scope(config)
        policy = self.policies.get(scope)
        control = normalize_rate_limit(config.get('rate_limit'), max_output_tokens=config['max_output_tokens'])
        if policy is None:
            if control['requests_per_minute'] or control['tokens_per_minute']:
                raise ProviderLimitError('provider_rate_scope_missing', 'Configured provider quota is outside this rate session')
            return None
        # Match all model calls on a limited endpoint, even if a caller omitted
        # its own RPM. TPM input allowance remains explicit per model config.
        reserved = config['max_output_tokens'] + control['input_tokens_per_request']
        if policy['tokens_per_minute'] and (not control['input_tokens_per_request'] or reserved > policy['tokens_per_minute']):
            raise ProviderLimitError('provider_rate_reservation_exceeded', 'Request input/output reservation does not fit the shared provider TPM quota')
        started = self.monotonic()
        while True:
            if self.stop_event.is_set():
                raise ProviderLimitError('batch_stopped', 'Batch is stopping; no new request was sent')
            if self.monotonic() >= deadline:
                raise ProviderLimitError('timeout', 'Provider rate queue exceeded this request deadline')
            now = self.clock()
            second = math.floor(now)
            delay = .1
            with self._transaction() as db:
                self._prune(db, now)
                # Another process may have opened this root with new controls
                # after a quiet minute. Never silently use a different policy.
                row = db.execute('SELECT rpm,tpm FROM policies WHERE scope=?', (scope,)).fetchone()
                if row is None or (row['rpm'], row['tpm']) != (policy['requests_per_minute'], policy['tokens_per_minute']):
                    raise ProviderLimitError('provider_rate_policy_changed', 'Provider quota differs from this active session')
                rows = db.execute('SELECT second,requests,tokens FROM buckets WHERE scope=? ORDER BY second', (scope,)).fetchall()
                requests = sum(row['requests'] for row in rows)
                tokens = sum(row['tokens'] for row in rows)
                blocked = ((policy['requests_per_minute'] and requests + 1 > policy['requests_per_minute'])
                           or (policy['tokens_per_minute'] and tokens + reserved > policy['tokens_per_minute']))
                if not blocked:
                    if self.stop_event.is_set():
                        raise ProviderLimitError('batch_stopped', 'Batch is stopping; no new request was sent')
                    if self.monotonic() >= deadline:
                        raise ProviderLimitError('timeout', 'Provider rate queue exceeded this request deadline')
                    request_id = uuid.uuid4().hex
                    db.execute('INSERT INTO buckets VALUES(?,?,1,?) ON CONFLICT(scope,second) DO UPDATE SET requests=requests+1,tokens=tokens+excluded.tokens',
                               (scope, second, reserved))
                    db.execute('INSERT INTO pending VALUES(?,?,?,?)', (request_id, scope, second, reserved))
                    return _Ticket(self, request_id, self.monotonic()-started)
                # Poll settlements briefly. Expiry alone may need up to 61 s;
                # bounded waits keep stop events and deadlines responsive.
                if rows:
                    delay = min(.25, max(.001, rows[0]['second'] + 61 - now))
            remaining = deadline-self.monotonic()
            if remaining <= 0:
                raise ProviderLimitError('timeout', 'Provider rate queue exceeded this request deadline')
            if self.waiter(min(delay, remaining)):
                raise ProviderLimitError('batch_stopped', 'Batch is stopping; no new request was sent')

    def finish(self, request_id, transport):
        usage = _usage(transport)
        unsent = _unsent(transport)
        with self._transaction() as db:
            self._prune(db, self.clock())
            item = db.execute('SELECT * FROM pending WHERE request_id=?', (request_id,)).fetchone()
            if item is None:
                return  # Already settled, or its rolling-minute window expired.
            tokens = 0 if unsent else (usage['reported_tokens'] if usage['usage_complete'] else
                                     max(item['reserved_tokens'], usage['reported_tokens']))
            db.execute('UPDATE buckets SET requests=requests-?,tokens=tokens+? WHERE scope=? AND second=?',
                       (int(unsent), tokens-item['reserved_tokens'], item['scope'], item['second']))
            db.execute('DELETE FROM pending WHERE request_id=?', (request_id,))

    def snapshot(self):
        if self._db is None:
            return {'scopes': [], 'bucket_count': 0, 'pending_count': 0}
        with self._transaction() as db:
            self._prune(db, self.clock())
            rows = db.execute('''SELECT scope,SUM(requests) AS requests,SUM(tokens) AS tokens,
                COUNT(*) AS buckets FROM buckets GROUP BY scope''').fetchall()
            pending = db.execute('SELECT COUNT(*) FROM pending').fetchone()[0]
            return {'scopes': [dict(row) for row in rows],
                    'bucket_count': sum(row['buckets'] for row in rows), 'pending_count': pending}


class _Ticket:
    def __init__(self, limiter, request_id, waited_seconds):
        self._limiter = limiter
        self.request_id = request_id
        self.waited_seconds = waited_seconds

    def finish(self, transport):
        self._limiter.finish(self.request_id, transport)


@contextmanager
def provider_limit_session(root, configs, stop_event=None):
    """Bind a resolved output tree to provider admission across all its threads.

    ``configs`` is the complete iterable of model configurations used by this
    run. Same endpoint shares the strictest positive RPM and TPM of its configs.
    The SQLite file must live on reliable local storage; it coordinates only
    processes using that same ledger, not unrelated output directories/hosts.
    """
    from model_api import normalize_config
    configs = [normalize_config(config) for config in configs if config is not None]
    root = Path(root).expanduser().resolve()
    with _registry_lock:
        if any(_inside(root, other) or _inside(other, root) for other in _active):
            raise ProviderLimitError('invalid_provider_rate_session', 'Overlapping provider rate sessions are not permitted')
        limiter = ProviderLimiter(root, configs, stop_event)
        limiter.initialize()
        _active[root] = limiter
    try:
        yield limiter
    finally:
        with _registry_lock:
            del _active[root]
        limiter.close()


def reserve_request(output, config, *, deadline):
    """Reserve exactly one transport attempt; disabled default is a no-op."""
    path = Path(output).expanduser().resolve()
    with _registry_lock:
        limiter = next((value for root, value in _active.items() if _inside(path, root)), None)
    if limiter is not None:
        return limiter.reserve(config, deadline)
    control = normalize_rate_limit(config.get('rate_limit'), max_output_tokens=config['max_output_tokens'])
    if control['requests_per_minute'] or control['tokens_per_minute']:
        raise ProviderLimitError('provider_rate_session_required', 'Configured provider rate limits require provider_limit_session around this output tree')
    return None


def current_stop_event(output=None):
    """Return the active provider-session cancellation event for an output."""
    path = Path(output).expanduser().resolve() if output is not None else None
    with _registry_lock:
        limiter = next((value for root, value in _active.items()
                        if path is None or _inside(path, root)), None)
    return limiter.stop_event if limiter is not None else None
