"""Persistent, opt-in limits shared by every API attempt in one scene.

Output capacity is reserved before transport and released after consistent full
usage or durable proof that no POST began. Unknown usage is not free.
Reported-token limits stop subsequent calls, not calls already in flight, and
cannot bound unknown input/image costs. No provider or API adapter is imported.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid

try:
    import fcntl
except ImportError:
    fcntl = None


POLICY_FIELDS = ('max_requests', 'max_output_tokens', 'max_reported_tokens')
LEDGER_NAME = 'scene_budget.json'
LOCK_NAME = '.scene_budget.lock'
SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
_USAGE_FIELDS = ('prompt_tokens', 'completion_tokens', 'input_tokens',
                 'output_tokens', 'total_tokens', 'cache_creation_input_tokens',
                 'cache_read_input_tokens')
_registry_lock = threading.RLock()
_active = {}
_file_locks = {}

# Only transport facts needed to prove a request was never sent are persisted
# in the scene ledger.  Request bodies, headers and provider payloads are never
# copied into this file.  ``state`` distinguishes a durable reservation from a
# settled transport proof and from a migrated v1 record with no proof.
_TRANSPORT_FIELDS = ('phase', 'request_started', 'request_sent', 'http_status',
                     'response_bytes', 'usage', 'automatic_replay_blocked',
                     'usage_invalid', 'completion_marker_missing', 'usage_present')
_TRANSPORT_PHASES = {'reserved', 'connecting', 'not_sent', 'sending_request',
                     'waiting_response', 'receiving_response', 'response_complete',
                     'legacy_unknown'}


class BudgetError(RuntimeError):
    """Sanitized failure with a stable API-compatible code."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _invalid(message):
    return BudgetError('invalid_scene_budget', message)


def normalize_policy(policy=None):
    """Return three nonnegative integer limits; zero disables each limit."""
    if policy is None:
        policy = {}
    if not isinstance(policy, dict) or set(policy) - set(POLICY_FIELDS):
        raise _invalid('Scene budget must be an object containing only supported limits')
    result = {}
    for name in POLICY_FIELDS:
        value = policy.get(name, 0)
        if type(value) is not int or value < 0:
            raise _invalid(f'Scene budget {name} must be a nonnegative integer; zero disables it')
        result[name] = value
    return result


def _inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _integer(value):
    return type(value) is int and value >= 0


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite constant')


def _read(path):
    try:
        with path.open(encoding='utf-8') as handle:
            return json.load(handle, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except (OSError, ValueError, UnicodeError, RecursionError):
        raise _invalid('Scene budget ledger or previous API audit is unreadable; do not reset it') from None


def _write(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise _invalid('Scene budget could not be durably recorded; no new request is permitted') from None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _usage(transport):
    """Keep known counters while marking missing, invalid or inconsistent usage."""
    raw = transport.get('usage')
    if not isinstance(raw, dict):
        raw = {}
    counters = {key: raw[key] for key in _USAGE_FIELDS
                if type(raw.get(key)) is int and 0 <= raw[key] <= 10**12}
    valid = bool(counters) and all(key in counters for key in _USAGE_FIELDS if key in raw)
    input_values = [counters['prompt_tokens']] if 'prompt_tokens' in counters else []
    # Anthropic input excludes cache creation/read input. OpenAI prompt_tokens
    # already includes cached input. Use alternate totals, never add both.
    cache = counters.get('cache_creation_input_tokens', 0) + counters.get('cache_read_input_tokens', 0)
    if 'input_tokens' in counters:
        input_values.append(counters['input_tokens'] + cache)
    output_values = [counters[key] for key in ('completion_tokens', 'output_tokens') if key in counters]
    known_input = max(input_values) if input_values else None
    known_output = max(output_values) if output_values else None
    known_total = max(known_input or 0, cache) + (known_output or 0)
    if 'total_tokens' in counters:
        known_total = max(known_total, counters['total_tokens'])
    consistent = (len(set(input_values)) <= 1 and len(set(output_values)) <= 1)
    if known_input is not None and known_output is not None and 'total_tokens' in counters:
        consistent = consistent and counters['total_tokens'] == known_input + known_output
    complete = (valid and consistent and known_input is not None and known_output is not None
                and transport.get('phase') == 'response_complete'
                and type(transport.get('http_status')) is int
                and 200 <= transport['http_status'] < 300
                and transport.get('usage_invalid') is not True
                and transport.get('automatic_replay_blocked') is not True
                and transport.get('completion_marker_missing') is not True)
    return {'usage': counters, 'reported_input_tokens': known_input,
            'reported_output_tokens': known_output, 'reported_tokens': known_total,
            'usage_complete': complete}


def _proven_unsent(transport):
    """Return true only when the transport proves that no POST began.

    This intentionally mirrors the provider limiter's conservative predicate.
    Missing or contradictory send evidence keeps the reservation accounted.
    """
    return (isinstance(transport, dict)
            and transport.get('state') == 'settled'
            and transport.get('phase') in ('not_sent', 'connecting')
            and transport.get('request_started') is False
            and transport.get('request_sent') in (None, False)
            and transport.get('http_status') is None
            and transport.get('response_bytes') in (None, 0)
            and transport.get('usage') in (None, {})
            and not transport.get('usage_present')
            and not transport.get('automatic_replay_blocked')
            and not transport.get('usage_invalid')
            and not transport.get('completion_marker_missing'))


def _transport_record(transport, state='settled'):
    """Persist an allow-listed, JSON-safe transport proof/state."""
    if not isinstance(transport, dict):
        raise _invalid('Scene budget settlement requires transport metadata')
    record = {'state': state}
    invalid_usage = False
    usage_present = False
    for key in _TRANSPORT_FIELDS:
        if key not in transport:
            continue
        value = transport[key]
        if key == 'usage':
            if value is None:
                record[key] = None
            elif not isinstance(value, dict):
                record[key] = {}
                invalid_usage = True
                usage_present = True
            else:
                filtered = {}
                invalid = False
                for name, count in value.items():
                    if name in _USAGE_FIELDS and type(count) is int and 0 <= count <= 10**12:
                        filtered[name] = count
                    elif name in _USAGE_FIELDS:
                        invalid = True
                record[key] = filtered
                if value:
                    usage_present = True
                if invalid:
                    invalid_usage = True
        elif key in ('request_started', 'request_sent', 'automatic_replay_blocked',
                     'usage_invalid', 'completion_marker_missing', 'usage_present'):
            if value is not None and type(value) is not bool:
                raise _invalid('Scene budget transport proof contains a non-boolean flag')
            record[key] = value
        elif key in ('http_status', 'response_bytes'):
            if value is not None and (type(value) is not int or value < 0 or value > 10**12):
                raise _invalid('Scene budget transport proof contains an invalid numeric field')
            record[key] = value
        elif key == 'phase':
            if not isinstance(value, str) or value not in _TRANSPORT_PHASES:
                raise _invalid('Scene budget transport proof contains an invalid phase')
            record[key] = value
    if invalid_usage:
        record['usage_invalid'] = True
    if usage_present:
        record['usage_present'] = True
    return record


def _totals(attempts):
    refunded = lambda item: item.get('state') == 'settled' and _proven_unsent(item.get('transport'))
    accounted = [item for item in attempts if not refunded(item)]
    return {'requests': len(accounted),
            'output_tokens_accounted': sum(
                item['reported_output_tokens'] if item['usage_complete'] else
                max(item['reserved_output_tokens'], item['reported_output_tokens'] or 0)
                for item in accounted),
            'reported_tokens': sum(item['reported_tokens'] for item in accounted),
            'unknown_usage_attempts': sum(not item['usage_complete'] for item in accounted),
            'pending_attempts': sum(item['state'] == 'reserved' for item in attempts)}


class _SceneBudget:
    def __init__(self, root, policy):
        self.root = root
        self.policy = policy
        self.enabled = any(policy.values())
        self.ledger_path = root / LEDGER_NAME
        self.lock_path = root / LOCK_NAME
        self.references = 1
        with _registry_lock:
            self.thread_lock = _file_locks.setdefault(root, threading.RLock())

    @contextmanager
    def _locked(self):
        if fcntl is None:
            raise _invalid('Persistent scene budgeting requires POSIX fcntl.flock on this platform')
        with self.thread_lock:
            if self.lock_path.is_symlink() or self.ledger_path.is_symlink():
                raise _invalid('Scene budget files must not be symbolic links')
            existed = self.lock_path.exists()
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                with self.lock_path.open('a+', encoding='utf-8') as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        yield existed
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                raise _invalid('Scene budget lock is unavailable; no new request is permitted') from None

    def _load(self):
        if not self.ledger_path.is_file():
            raise _invalid('Scene budget ledger is missing; do not reset previous reservations')
        return self._validate_ledger(_read(self.ledger_path))

    def _validate_ledger(self, ledger):
        """Validate one already-read ledger without fetching a newer version."""
        keys = {'schema_version', 'scene_root', 'policy', 'attempts', *_totals([])}
        if (not isinstance(ledger, dict) or set(ledger) != keys
                or type(ledger.get('schema_version')) is not int
                or ledger['schema_version'] not in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION)
                or ledger.get('scene_root') != str(self.root)
                or ledger.get('policy') != self.policy):
            raise _invalid('Scene budget schema, scene directory, or limits changed; use a new output directory')
        # Equality alone allows bools to masquerade as integer limits.
        if normalize_policy(ledger['policy']) != self.policy:
            raise _invalid('Scene budget ledger policy is invalid')
        attempts = ledger['attempts']
        if not isinstance(attempts, list):
            raise _invalid('Scene budget ledger attempts are invalid')
        entry_keys = {'request_id', 'output', 'state', 'reserved_output_tokens', 'usage',
                      'reported_input_tokens', 'reported_output_tokens', 'reported_tokens', 'usage_complete'}
        if ledger['schema_version'] == SCHEMA_VERSION:
            entry_keys.add('transport')
        identifiers = set()
        for item in attempts:
            if (not isinstance(item, dict) or set(item) != entry_keys
                    or not isinstance(item.get('request_id'), str)
                    or not re.fullmatch(r'[0-9a-f]{32}', item['request_id'])
                    or item['request_id'] in identifiers
                    or not isinstance(item.get('output'), str) or not item['output']
                    or Path(item['output']).is_absolute() or '..' in Path(item['output']).parts
                    or item.get('state') not in ('reserved', 'settled')
                    or not _integer(item.get('reserved_output_tokens')) or item['reserved_output_tokens'] < 1
                    or type(item.get('usage_complete')) is not bool
                    or not _integer(item.get('reported_tokens'))
                    or any(value is not None and not _integer(value) for value in
                           (item.get('reported_input_tokens'), item.get('reported_output_tokens')))
                    or not isinstance(item.get('usage'), dict)
                    or set(item['usage']) - set(_USAGE_FIELDS)
                    or any(not _integer(value) or value > 10**12 for value in item['usage'].values())):
                raise _invalid('Scene budget ledger contains an invalid reservation')
            identifiers.add(item['request_id'])
            if ledger['schema_version'] == SCHEMA_VERSION:
                proof = item['transport']
                if not isinstance(proof, dict) or proof.get('state') not in ('reserved', 'settled', 'legacy'):
                    raise _invalid('Scene budget transport proof is invalid')
                if proof['state'] == 'legacy':
                    if proof != {'state': 'legacy', 'phase': 'legacy_unknown'}:
                        raise _invalid('Scene budget legacy transport proof has been modified')
                elif proof['state'] == 'reserved':
                    if item['state'] != 'reserved' or proof != {'state': 'reserved', 'phase': 'reserved'}:
                        raise _invalid('Scene budget pending transport proof has been modified')
                elif item['state'] != 'settled' or set(proof)-{'state', *_TRANSPORT_FIELDS} or _transport_record(proof) != proof:
                    raise _invalid('Scene budget settled transport proof has been modified')
                if item['state'] == 'reserved' and proof['state'] != 'reserved':
                    raise _invalid('Scene budget pending transport proof has been modified')
            possible = _usage({'usage': item['usage'], 'phase': 'response_complete', 'http_status': 200})
            if item['usage_complete'] and (item['state'] != 'settled' or not possible['usage_complete']):
                raise _invalid('Scene budget ledger contains an invalid completed reservation')
            if item['state'] == 'reserved' and (item['usage'] or item['usage_complete']
                    or item['reported_input_tokens'] is not None or item['reported_output_tokens'] is not None
                    or item['reported_tokens'] != 0):
                raise _invalid('Scene budget pending reservation has been modified')
            for field in ('reported_input_tokens', 'reported_output_tokens', 'reported_tokens'):
                if item[field] != possible[field]:
                    raise _invalid('Scene budget usage counters are inconsistent')
            if ledger['schema_version'] == SCHEMA_VERSION and item['transport']['state'] == 'settled':
                proof_usage = _usage(item['transport'])
                if any(item[field] != proof_usage[field] for field in proof_usage):
                    raise _invalid('Scene budget usage differs from its transport proof')
        expected = _totals(attempts)
        if any(not _integer(ledger.get(key)) or ledger[key] != value for key, value in expected.items()):
            raise _invalid('Scene budget totals are inconsistent; do not reset the ledger')
        return ledger

    def initialize(self):
        if not self.enabled:
            if self.ledger_path.exists() or self.lock_path.exists():
                raise _invalid('Existing scene budget cannot be disabled; use the recorded limits or a new output directory')
            return
        with self._locked() as lock_existed:
            if self.ledger_path.exists():
                ledger = self._load()
                if ledger['pending_attempts']:
                    raise BudgetError('ambiguous_previous_request',
                        'A previous scene request has an unfinished reservation and may already have consumed tokens; '
                        'automatic scene resume was blocked. Check provider records before explicitly retrying in a new output directory.')
                if ledger['schema_version'] == LEGACY_SCHEMA_VERSION:
                    # v1 had no send proof. Preserve every historical counter;
                    # never infer a refund from a separately mutable API audit.
                    for item in ledger['attempts']:
                        item['transport'] = {'state': 'legacy', 'phase': 'legacy_unknown'}
                    ledger['schema_version'] = SCHEMA_VERSION
                    _write(self.ledger_path, ledger)
                return
            if lock_existed:
                raise _invalid('Scene budget ledger is missing beside an existing lock; do not reset it')
            for path in self.root.rglob('*.api.json'):
                record = _read(path)
                if not isinstance(record, dict):
                    raise _invalid('Previous API audit is invalid; existing usage cannot be reset')
                attempts = record.get('attempts', 0)
                transports = record.get('transport_attempts', [])
                if not _integer(attempts) or not isinstance(transports, list):
                    raise _invalid('Previous API attempt counts are invalid; existing usage cannot be reset')
                if attempts or transports:
                    raise _invalid('Previous API attempts exist without a budget ledger; use a fresh output directory')
            _write(self.ledger_path, {'schema_version': SCHEMA_VERSION, 'scene_root': str(self.root),
                                     'policy': self.policy, 'attempts': [], **_totals([])})

    def snapshot(self):
        """Read validated persisted counters, including pending/unknown attempts."""
        if not self.enabled:
            return None
        with self._locked():
            return self._load()

    def reserve(self, output, max_output_tokens):
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise _invalid('Each scene-budget reservation requires a positive integer output cap')
        with self._locked():
            ledger = self._load()
            limits = self.policy
            if limits['max_requests'] and ledger['requests'] >= limits['max_requests']:
                raise BudgetError('scene_budget_exceeded', 'Scene request-attempt limit reached; no new request was sent')
            if (limits['max_output_tokens'] and ledger['output_tokens_accounted'] + max_output_tokens
                    > limits['max_output_tokens']):
                raise BudgetError('scene_budget_exceeded',
                                  'Scene output capacity cannot reserve the full configured request cap; no new request was sent')
            if limits['max_reported_tokens'] and ledger['reported_tokens'] >= limits['max_reported_tokens']:
                raise BudgetError('scene_budget_exceeded',
                                  'Scene reported-token threshold reached; no new request was sent')
            request_id = uuid.uuid4().hex
            ledger['attempts'].append({'request_id': request_id, 'output': output.relative_to(self.root).as_posix(),
                'state': 'reserved', 'reserved_output_tokens': max_output_tokens,
                'usage': {}, 'reported_input_tokens': None, 'reported_output_tokens': None,
                'reported_tokens': 0, 'usage_complete': False,
                'transport': {'state': 'reserved', 'phase': 'reserved'}})
            ledger.update(_totals(ledger['attempts']))
            _write(self.ledger_path, ledger)
        return _Ticket(self, request_id)

    def finish(self, request_id, transport):
        if not isinstance(transport, dict):
            raise _invalid('Scene budget settlement requires transport metadata')
        with self._locked():
            ledger = self._load()
            item = next((item for item in ledger['attempts'] if item['request_id'] == request_id), None)
            if item is None:
                raise _invalid('Scene budget reservation is missing; no usage can be released')
            if item['state'] == 'settled':
                return
            item['transport'] = _transport_record(transport)
            item.update(_usage(item['transport']))
            item['state'] = 'settled'
            ledger.update(_totals(ledger['attempts']))
            _write(self.ledger_path, ledger)


class _Ticket:
    def __init__(self, budget, request_id):
        self.request_id = request_id
        self._budget = budget

    def finish(self, transport):
        """Settle once; incomplete/unknown usage retains conservative output capacity."""
        self._budget.finish(self.request_id, transport)


@contextmanager
def scene_budget_session(root, policy=None):
    """Bind resolved scene descendants across worker threads to one durable ledger.

    Disjoint scenes run concurrently. Identical root/limits may nest; overlapping
    roots are rejected to prevent bypassing a parent with a narrower scope.
    Disabled sessions create no files. New processes must enter this context
    using the same scene root; flock coordinates their ledger transactions.
    """
    policy = normalize_policy(policy)
    root = Path(root).expanduser().resolve()
    with _registry_lock:
        budget = _active.get(root)
        if budget is not None:
            if budget.policy != policy:
                raise _invalid('Concurrent scene budget limits differ for the same directory')
            budget.references += 1
        else:
            if any(_inside(root, other) or _inside(other, root) for other in _active):
                raise _invalid('Overlapping active scene budget directories are not permitted')
            budget = _SceneBudget(root, policy)
            budget.initialize()
            _active[root] = budget
    try:
        yield budget
    finally:
        with _registry_lock:
            budget.references -= 1
            if not budget.references:
                del _active[root]


def reserve_request(output, max_output_tokens):
    """Reserve one network attempt, or return None outside an enabled scene.

    Call immediately before *each* transport attempt, including retries. Once
    returned, the persisted reservation survives process termination.
    """
    output = Path(output).expanduser().resolve()
    with _registry_lock:
        budget = next((value for root, value in _active.items() if _inside(output, root)), None)
        if budget is None or not budget.enabled:
            return None
        return budget.reserve(output, max_output_tokens)


def budget_snapshot(root):
    """Read policy and summary without an active session or any filesystem write.

    Atomic ledger replacement permits a coherent read without creating a lock.
    Unknown output/input usage is represented explicitly in the summary.
    """
    root = Path(root).expanduser().resolve()
    path = root / LEDGER_NAME
    if path.is_symlink():
        raise _invalid('Scene budget ledger must not be a symbolic link')
    if not path.exists():
        return None
    ledger = _read(path)
    if not isinstance(ledger, dict) or 'policy' not in ledger:
        raise _invalid('Scene budget ledger policy is missing')
    budget = _SceneBudget(root, normalize_policy(ledger['policy']))
    ledger = budget._validate_ledger(ledger)
    return {'policy': ledger['policy'], 'summary': {key: ledger[key] for key in _totals([])}}
