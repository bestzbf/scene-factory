"""Receipt-bound local Codex inference for unattended batch stages.

Uses the existing logged-in CLI. It never reads or copies authentication files.
Only request counts are bounded here; CLI subscription/token billing is external.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
from contextlib import contextmanager
from contextvars import ContextVar

from batch import atomic_json, canonical_bytes, file_sha256, read_json

_ACTIVE = ContextVar('scene_factory_local_codex', default=False)


def active():
    return _ACTIVE.get()


@contextmanager
def session(enabled):
    token = _ACTIVE.set(bool(enabled))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def identity():
    cli = os.environ.get('CODEX_BIN') or shutil.which('codex')
    if not cli:
        raise RuntimeError('Codex CLI not found; configure/login locally')
    result = subprocess.run([cli, '--version'], capture_output=True, text=True, timeout=15, check=True)
    return {'provider': 'codex_cli', 'executable': str(Path(cli).resolve()),
            'version': result.stdout.strip(), 'model': 'configured_cli_default'}


def call_model(prompt, images, schema, output, timeout=900, effort='high', invoke=None):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.with_name(output.stem + '.codex.json')
    raw_path = output.with_name(output.stem + '.raw.json')
    binding = {'model': identity(), 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
               'schema_sha256': hashlib.sha256(canonical_bytes(schema)).hexdigest(),
               'image_sha256': [file_sha256(p) for p in images], 'effort': effort,
               'implementation_sha256': file_sha256(__file__)}
    if receipt.exists():
        cached = read_json(receipt)
        if cached.get('binding') != binding:
            raise ValueError('Local Codex stage inputs changed; use a new output directory')
        if cached.get('status') != 'completed':
            raise RuntimeError('Local Codex attempt is incomplete; automatic duplicate invocation is blocked')
        if not raw_path.is_file() or file_sha256(raw_path) != cached.get('output_sha256'):
            raise ValueError('Local Codex output changed or is missing')
        result = read_json(raw_path)
        from spec_v2 import _walk
        _walk(result, schema, 'local_codex')
        return result
    if raw_path.exists():
        raise ValueError('Unbound local Codex response exists; automatic replay is blocked')
    from runtime_controls import api_slot
    from scene_budget import reserve_request
    started = time.monotonic()
    with api_slot(timeout) as waited:
        if waited >= timeout:
            raise RuntimeError('Local Codex concurrency wait exhausted the stage timeout')
        ticket = reserve_request(output, 1)
        atomic_json(receipt, {'binding': binding, 'status': 'started',
                             'token_cap_enforced': False, 'request_budget_only': True})
        try:
            if invoke is None:
                from vision_v2 import call_model as invoke
            result = invoke(prompt, images, schema, output, timeout-waited, effort)
            from spec_v2 import _walk
            _walk(result, schema, 'local_codex')
            if binding['image_sha256'] != [file_sha256(p) for p in images]:
                raise RuntimeError('Local Codex input images changed during inference')
            atomic_json(raw_path, result)
            atomic_json(receipt, {'binding': binding, 'status': 'completed',
                'output_sha256': file_sha256(raw_path), 'request_budget_only': True,
                'token_cap_enforced': False, 'elapsed_seconds': round(time.monotonic()-started, 3)})
            return result
        except Exception:
            atomic_json(receipt, {'binding': binding, 'status': 'incomplete',
                                 'automatic_retry': False, 'request_budget_only': True})
            raise
        finally:
            if ticket:
                ticket.finish({'phase': 'receiving_response', 'request_started': True, 'request_sent': True})
