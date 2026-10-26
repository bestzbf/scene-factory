"""Multimodal JSON transport for explicitly configured model APIs.

This module never executes model output. Provider capability choices are explicit:
unsupported structured-output requests fail instead of silently changing mode.
Secrets are read from exactly ``api_key_env`` at request time and never recorded.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit, urlunsplit

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

from batch import atomic_json, read_json
from spec_v2 import _walk
from runtime_controls import api_session as api_concurrency_limit


PROVIDERS = {'openai_chat', 'openai_responses', 'anthropic'}
ENDPOINTS = {'openai_chat': '/chat/completions',
             'openai_responses': '/responses', 'anthropic': '/messages'}
DEFAULT_URLS = {'openai_chat': 'https://api.openai.com/v1',
                'openai_responses': 'https://api.openai.com/v1',
                'anthropic': 'https://api.anthropic.com'}
FIELDS = {'provider', 'model', 'base_url', 'api_key_env', 'response_format', 'auth_scheme', 'token_parameter',
          'max_output_tokens', 'max_prompt_characters', 'timeout_seconds', 'max_retries',
          'reasoning_effort', 'temperature', 'stream'}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 80 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_SSE_EVENT_BYTES = 2 * 1024 * 1024
SCHEMA_NAME = 'scene_factory_result'


class ModelAPIError(RuntimeError):
    """An intentionally sanitized provider failure, with a stable failure code."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def guard_ambiguous_model_resume(api_records):
    """Do not replay a previous POST whose provider outcome is ambiguous.

    Callers pass only their exact stage's local audit paths. Successful
    committed stages are reused before this check; missing records and explicit
    rejections such as HTTP 402 do not prevent normal recovery. This reads no
    credentials, prompts, image payloads or generated scene data.
    """
    for path in api_records:
        path = Path(path)
        if not path.is_file():
            continue
        try:
            record = read_json(path)
        except (OSError, ValueError, UnicodeError, RecursionError):
            raise ValueError('Previous model API audit is unreadable; inspect the saved stage before retrying') from None
        if not isinstance(record, dict):
            raise ValueError('Previous model API audit must be an object; inspect the saved stage before retrying')
        attempts = record.get('transport_attempts', [])
        if not isinstance(attempts, list) or any(not isinstance(item, dict) for item in attempts):
            raise ValueError('Previous model API audit has invalid transport records; inspect the saved stage before retrying')
        if (record.get('failure_code') in ('response_interrupted', 'request_outcome_unknown')
                or any(item.get('automatic_replay_blocked') is True for item in attempts)):
            raise ModelAPIError('ambiguous_previous_request',
                'A previous model request may already have consumed tokens; automatic resume was blocked. '
                'Check the provider request records, then explicitly retry in a new output directory.')


def _number(value, name, minimum, maximum, integer=False):
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be a finite {"integer" if integer else "number"} between {minimum} and {maximum}')
    return value


def _loopback(host):
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _base_url(value):
    if not isinstance(value, str) or not value or len(value) > 2048 or any(ord(c) < 33 for c in value):
        raise ValueError('base_url must be an absolute HTTP(S) URL without whitespace')
    try:
        parts = urlsplit(value)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise ValueError('base_url has an invalid host or port') from None
    if not host or parts.scheme not in ('https', 'http') or parts.username is not None or parts.password is not None or '?' in value or '#' in value:
        raise ValueError('base_url requires HTTP(S), a host, and no credentials, query or fragment')
    if '@' in parts.netloc or '%' in host or not host.isascii() or (port is not None and not 1 <= port <= 65535):
        raise ValueError('base_url has an invalid host or port')
    if parts.scheme != 'https' and not _loopback(host):
        raise ValueError('HTTPS is required except for loopback local servers')
    if not re.fullmatch(r'[A-Za-z0-9._~/-]*', parts.path) or '..' in parts.path.split('/') or '//' in parts.path:
        raise ValueError('base_url has an unsupported path')
    netloc = f'[{host}]' if ':' in host else host
    if port is not None:
        netloc += ':' + str(port)
    return urlunsplit((parts.scheme, netloc, parts.path.rstrip('/'), '', ''))


def normalize_config(config):
    """Validate and canonicalize a plain config dict without reading credentials.

    provider/model/api_key_env are explicit. An empty api_key_env is permitted
    only for loopback local servers. reasoning_effort/temperature are opt-in.
    """
    if not isinstance(config, dict):
        raise ValueError('model API config must be an object')
    if set(config) - FIELDS:
        raise ValueError('model API config contains unknown fields')
    provider = config.get('provider')
    if not isinstance(provider, str) or provider not in PROVIDERS:
        raise ValueError('provider must be openai_chat, openai_responses, or anthropic')
    model = config.get('model')
    if not isinstance(model, str) or not model.strip() or model != model.strip() or len(model) > 256 or any(ord(c) < 32 for c in model):
        raise ValueError('model must be a nonempty model identifier')
    base_url = _base_url(config.get('base_url', DEFAULT_URLS[provider]))
    host = urlsplit(base_url).hostname
    env = config.get('api_key_env')
    if not isinstance(env, str) or len(env) > 255 or (env and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', env)):
        raise ValueError('api_key_env must explicitly name an environment variable')
    if not env and not _loopback(host):
        raise ValueError('api_key_env may be empty only for loopback local servers')
    if host.rstrip('.') == 'api.apikey.fun' and env != 'APIKEY_FUN_API_KEY':
        raise ValueError('api.apikey.fun requires the dedicated APIKEY_FUN_API_KEY environment variable')
    mode = config.get('response_format', 'prompt' if provider == 'anthropic' else 'json_schema')
    if not isinstance(mode, str) or mode not in ('json_schema', 'json_object', 'prompt'):
        raise ValueError('response_format must be json_schema, json_object, or prompt')
    if provider == 'anthropic' and mode != 'prompt':
        raise ValueError('anthropic currently supports response_format=prompt only in this adapter')
    result = {'provider': provider, 'model': model, 'base_url': base_url,
              'api_key_env': env, 'response_format': mode,
              'max_output_tokens': _number(config.get('max_output_tokens', 32768), 'max_output_tokens', 1, 262144, True),
              'timeout_seconds': _number(config.get('timeout_seconds', 600), 'timeout_seconds', .1, 86400),
              'max_retries': _number(config.get('max_retries', 2), 'max_retries', 0, 10, True)}
    # Keep omitted defaults absent so existing public-config fingerprints and
    # saved stage caches remain compatible. This bounds caller prompt text,
    # not schema/image tokens or the provider's billed token count.
    if 'max_prompt_characters' in config:
        result['max_prompt_characters'] = _number(
            config['max_prompt_characters'], 'max_prompt_characters', 0, 2000000, True)
    auth_scheme = config.get('auth_scheme', 'api_key' if provider == 'anthropic' else 'bearer')
    if auth_scheme not in ('api_key', 'bearer'):
        raise ValueError('auth_scheme must be api_key or bearer')
    if provider != 'anthropic' and auth_scheme != 'bearer':
        raise ValueError('OpenAI-compatible adapters require auth_scheme=bearer')
    result['auth_scheme'] = auth_scheme
    if provider == 'openai_chat':
        token_parameter = config.get('token_parameter', 'max_completion_tokens')
        if token_parameter not in ('max_completion_tokens', 'max_tokens'):
            raise ValueError('token_parameter must be max_completion_tokens or max_tokens')
        result['token_parameter'] = token_parameter
    elif 'token_parameter' in config:
        raise ValueError('token_parameter is only configurable for openai_chat')
    if 'stream' in config:
        if type(config['stream']) is not bool:
            raise ValueError('stream must be a boolean')
        if provider != 'openai_chat':
            raise ValueError('stream is supported only for openai_chat')
        result['stream'] = config['stream']
    if 'reasoning_effort' in config and config['reasoning_effort'] is not None:
        if provider == 'anthropic':
            raise ValueError('reasoning_effort is not supported by the anthropic adapter')
        if not isinstance(config['reasoning_effort'], str) or config['reasoning_effort'] not in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
            raise ValueError('reasoning_effort is unsupported')
        result['reasoning_effort'] = config['reasoning_effort']
    if 'temperature' in config and config['temperature'] is not None:
        result['temperature'] = _number(config['temperature'], 'temperature', 0, 2)
        if provider == 'anthropic' and result['temperature'] > 1:
            raise ValueError('anthropic temperature must be between 0 and 1')
    return result


def public_config(config):
    """Return normalized configuration suitable for fingerprints/provenance."""
    return normalize_config(config)


def _credential(config):
    env = config['api_key_env']
    if not env:
        return ''
    key = os.environ.get(env, '')
    if not key or key != key.strip() or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise ModelAPIError('missing_credential', f'Configure a nonempty valid credential locally in {env}; never paste it into chat')
    return key


def preflight(config):
    """Check config and local credential presence only; make no network request."""
    config = normalize_config(config)
    _credential(config)
    return public_config(config)


def _schema_contract(rule, depth=0):
    """Reject unsupported schema features rather than silently ignoring them."""
    if not isinstance(rule, dict) or depth > 40:
        raise ValueError('schema must use the bounded scene schema subset')
    typ = rule.get('type')
    common = {'type', 'enum', 'description', 'title'}
    allowed = {'object': {'properties', 'required', 'additionalProperties'},
               'array': {'items', 'minItems', 'maxItems'},
               'number': {'minimum', 'maximum'}, 'integer': {'minimum', 'maximum'},
               'string': {'minLength', 'maxLength', 'pattern'}, 'boolean': set()}
    if not isinstance(typ, str) or typ not in allowed or set(rule) - common - allowed[typ]:
        raise ValueError('schema contains unsupported validation features')
    if typ == 'object':
        properties = rule.get('properties')
        if (not isinstance(properties, dict) or rule.get('additionalProperties') is not False
                or not isinstance(rule.get('required'), list)
                or set(rule['required']) != set(properties)):
            raise ValueError('schema objects must require all properties and disallow additional properties')
        for child in properties.values():
            _schema_contract(child, depth + 1)
    elif typ == 'array':
        _schema_contract(rule.get('items'), depth + 1)
    elif typ == 'string':
        if 'pattern' in rule:
            if not isinstance(rule['pattern'], str) or len(rule['pattern']) > 1000:
                raise ValueError('schema string pattern must be a bounded regular expression')
            try:
                re.compile(rule['pattern'])
            except re.error:
                raise ValueError('schema contains an invalid string pattern') from None


def _walk_string_constraints(value, rule):
    """Enforce the V3 string constraints not present in spec_v2._walk."""
    if rule['type'] == 'object':
        for key, child in rule['properties'].items():
            if key in value:
                _walk_string_constraints(value[key], child)
    elif rule['type'] == 'array':
        for item in value:
            _walk_string_constraints(item, rule['items'])
    elif rule['type'] == 'string':
        if len(value) < rule.get('minLength', 0):
            raise ValueError('string shorter than minimum')
        if 'pattern' in rule and re.search(rule['pattern'], value) is None:
            raise ValueError('string does not match pattern')


def _image_payloads(images):
    encoded, hashes = [], []
    total = 0
    if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 100:
        raise ValueError('images must contain between 1 and 100 ordered image paths')
    for path in images:
        path = Path(path)
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError('an image exceeds the 20 MiB upload bound; resize it locally')
        data = path.read_bytes()
        total += len(data)
        if total > MAX_TOTAL_IMAGE_BYTES or len(data) > MAX_IMAGE_BYTES:
            raise ValueError('image payload exceeds the local upload bound')
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            mime = 'image/png'
        elif data.startswith(b'\xff\xd8\xff'):
            mime = 'image/jpeg'
        elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
            mime = 'image/webp'
        else:
            raise ValueError('images must be PNG, JPEG, or WebP; convert other formats locally')
        encoded.append((mime, base64.b64encode(data).decode('ascii')))
        hashes.append(hashlib.sha256(data).hexdigest())
    return encoded, hashes


def _payload(prompt, images, schema, config):
    mode, provider = config['response_format'], config['provider']
    instruction = prompt + '\n\nReturn exactly one JSON object matching the supplied schema. Do not return code, commentary, or markdown.'
    if mode != 'json_schema':
        instruction += '\nJSON SCHEMA:\n' + json.dumps(schema, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    is_responses = provider == 'openai_responses'
    content = [{'type': 'input_text' if is_responses else 'text', 'text': instruction}]
    for index, (mime, data) in enumerate(images, 1):
        content.append({'type': 'input_text' if is_responses else 'text', 'text': f'Image {index} (input order):'})
        if provider == 'anthropic':
            content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': mime, 'data': data}})
        elif is_responses:
            content.append({'type': 'input_image', 'image_url': f'data:{mime};base64,{data}'})
        else:
            content.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{data}'}})
    payload = {'model': config['model']}
    if provider == 'openai_responses':
        payload.update({'input': [{'role': 'user', 'content': content}],
                        'max_output_tokens': config['max_output_tokens'], 'store': False})
        if mode == 'json_schema':
            payload['text'] = {'format': {'type': 'json_schema', 'name': SCHEMA_NAME, 'strict': True, 'schema': schema}}
        elif mode == 'json_object':
            payload['text'] = {'format': {'type': 'json_object'}}
        if 'reasoning_effort' in config:
            payload['reasoning'] = {'effort': config['reasoning_effort']}
    elif provider == 'openai_chat':
        payload.update({'messages': [{'role': 'user', 'content': content}],
                        config['token_parameter']: config['max_output_tokens']})
        if mode == 'json_schema':
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {'name': SCHEMA_NAME, 'strict': True, 'schema': schema}}
        elif mode == 'json_object':
            payload['response_format'] = {'type': 'json_object'}
        if 'reasoning_effort' in config:
            payload['reasoning_effort'] = config['reasoning_effort']
        if 'stream' in config:
            payload['stream'] = config['stream']
    else:
        payload.update({'messages': [{'role': 'user', 'content': content}], 'max_tokens': config['max_output_tokens']})
    if 'temperature' in config:
        payload['temperature'] = config['temperature']
    return payload


def _retry_after(value):
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            delay = (when - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 0
    return max(0, min(30, delay)) if math.isfinite(delay) else 0


def _response_chunks(response, deadline, stats):
    """Read bounded chunks with one deadline even when a server trickles bytes."""
    total = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelAPIError('timeout', 'Model API exceeded the total request timeout')
        # HTTP/1.0 may detach the socket from the connection after headers.
        stream_socket = getattr(getattr(response.fp, 'raw', None), '_sock', None)
        if stream_socket is not None:
            stream_socket.settimeout(remaining)
        chunk = response.read1(min(65536, MAX_RESPONSE_BYTES - total + 1))
        if not chunk:
            return
        total += len(chunk)
        stats['response_bytes'] = total
        if total > MAX_RESPONSE_BYTES:
            raise ModelAPIError('response_too_large', 'Model API response exceeded the 32 MiB bound')
        yield chunk


def _sse_events(chunks):
    """Frame SSE events across arbitrary HTTP chunks and CR/LF boundaries.

    Only data/event fields matter. Comments and id/retry fields are ignored as
    prescribed by SSE; multiple data lines are joined with a newline. A final
    event without its terminating blank line is incomplete and never dispatched.
    """
    pending = bytearray()
    data, event, size = [], b'', 0
    first_line, skip_lf = True, False
    for chunk in chunks:
        if skip_lf:
            if chunk.startswith(b'\n'):
                chunk = chunk[1:]
            skip_lf = False
        pending.extend(chunk)
        while True:
            positions = [i for i in (pending.find(b'\n'), pending.find(b'\r')) if i >= 0]
            if not positions:
                if len(pending) + size > MAX_SSE_EVENT_BYTES:
                    raise ModelAPIError('response_too_large', 'Model API streaming event exceeded the local bound')
                break
            end = min(positions)
            line = bytes(pending[:end])
            delimiter = pending[end]
            del pending[:end + 1]
            if delimiter == 13:
                if pending.startswith(b'\n'):
                    del pending[:1]
                elif not pending:
                    skip_lf = True
            if first_line:
                line = line.removeprefix(b'\xef\xbb\xbf')
                first_line = False
            size += len(line) + 1
            if size > MAX_SSE_EVENT_BYTES:
                raise ModelAPIError('response_too_large', 'Model API streaming event exceeded the local bound')
            if not line:
                if data or event == b'error':
                    yield event, b'\n'.join(data)
                data, event, size = [], b'', 0
            elif not line.startswith(b':'):
                field, _, value = line.partition(b':')
                if value.startswith(b' '):
                    value = value[1:]
                if field == b'data':
                    data.append(value)
                elif field == b'event':
                    event = value


def _safe_finish(value):
    # Never put arbitrary provider text into a diagnostic file.
    allowed = ('stop', 'length', 'content_filter', 'tool_calls', 'function_call',
               'end_turn', 'max_tokens', 'stop_sequence', 'refusal', 'pause_turn')
    return value if isinstance(value, str) and value in allowed else 'unknown'


def _read_chat_stream(chunks, stats):
    parts, finish, refusal = [], None, False
    stats.update(events=0, reasoning_characters=0, content_characters=0, done=False)
    for event, data in _sse_events(chunks):
        if event == b'error':
            raise ModelAPIError('invalid_envelope', 'Streaming API returned an error event')
        if data.strip() == b'[DONE]':
            stats['done'] = True
            break
        try:
            item = json.loads(data, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
            raise ModelAPIError('invalid_envelope', 'Streaming API returned invalid JSON event data') from None
        if not isinstance(item, dict) or item.get('error'):
            raise ModelAPIError('invalid_envelope', 'Streaming API returned an unsuccessful event')
        stats['events'] += 1
        usage = _usage(item)
        if _usage_invalid(item):
            stats['usage_invalid'] = True
        if usage:
            stats['usage'] = usage
        choices = item.get('choices')
        if not isinstance(choices, list) or len(choices) > 1:
            raise ModelAPIError('invalid_envelope', 'Streaming API must return exactly one completion')
        if not choices:
            continue  # OpenAI emits a separate usage event with choices=[].
        choice = choices[0]
        if (not isinstance(choice, dict) or type(choice.get('index', 0)) is not int
                or choice.get('index', 0) != 0):
            raise ModelAPIError('invalid_envelope', 'Streaming API returned an invalid completion index')
        was_finished = finish is not None
        if choice.get('finish_reason') is not None:
            reason = _safe_finish(choice['finish_reason'])
            if finish is not None and finish != reason:
                raise ModelAPIError('invalid_envelope', 'Streaming API returned conflicting finish reasons')
            finish = reason
            stats['finish_reason'] = finish
        delta = choice.get('delta')
        if not isinstance(delta, dict):
            raise ModelAPIError('invalid_envelope', 'Streaming API returned an invalid delta')
        for field in ('reasoning_content', 'reasoning'):
            if isinstance(delta.get(field), str):
                stats['reasoning_characters'] += len(delta[field])
        if delta.get('tool_calls') or delta.get('function_call'):
            raise ModelAPIError('incomplete', 'Streaming API requested unexpected tools; no scene was accepted')
        refusal = refusal or bool(delta.get('refusal'))
        content = delta.get('content')
        if content is not None:
            if not isinstance(content, str):
                raise ModelAPIError('invalid_envelope', 'Streaming API returned non-text content')
            if was_finished and content:
                raise ModelAPIError('invalid_envelope', 'Streaming API returned text after the completion finished')
            parts.append(content)
            stats['content_characters'] += len(content)
    if not stats['done']:
        stats['completion_marker_missing'] = True
        raise ModelAPIError('incomplete', 'Streaming API ended before its completion marker')
    return {'choices': [{'finish_reason': finish, 'message': {
        'content': ''.join(parts), 'refusal': refusal}}], 'usage': stats.get('usage', {})}


def _request(config, body, key, deadline, *, transport=None):
    """Direct HTTP prevents automatic redirect following and proxy key leakage."""
    url = urlsplit(config['base_url'])
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ModelAPIError('timeout', 'Model API exceeded the total request timeout')
    connection_type = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
    conn = connection_type(url.hostname, url.port, timeout=remaining)
    transport = transport if transport is not None else {}
    # A POST can already be executing if its write or response read fails.
    # Record only fixed stage labels/booleans, never headers or request data.
    transport.update(phase='connecting', request_started=False, request_sent=False)
    streaming = config.get('stream', False)
    headers = {'Content-Type': 'application/json',
               'Accept': 'text/event-stream' if streaming else 'application/json', 'User-Agent': 'scene-factory/1'}
    if config['provider'] == 'anthropic':
        headers['anthropic-version'] = '2023-06-01'
        if key:
            if config['auth_scheme'] == 'bearer':
                headers['Authorization'] = 'Bearer ' + key
            else:
                headers['x-api-key'] = key
    elif key:
        headers['Authorization'] = 'Bearer ' + key
    response = None
    try:
        conn.connect()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelAPIError('timeout', 'Model API exceeded the total request timeout')
        conn.sock.settimeout(remaining)
        endpoint = ENDPOINTS[config['provider']]
        if config['provider'] == 'anthropic' and not url.path.endswith('/v1'):
            endpoint = '/v1' + endpoint
        transport.update(phase='sending_request', request_started=True)
        conn.request('POST', url.path + endpoint, body=body, headers=headers)
        transport.update(phase='waiting_response', request_sent=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelAPIError('timeout', 'Model API exceeded the total request timeout')
        conn.sock.settimeout(remaining)
        response = conn.getresponse()
        status, retry = response.status, _retry_after(response.getheader('Retry-After'))
        transport['http_status'] = status
        transport['phase'] = 'receiving_response'
        if not 200 <= status < 300:
            # Provider error bodies can echo credentials or image payloads. Discard.
            return status, retry, None
        chunks = _response_chunks(response, deadline, transport)
        if streaming:
            mime = (response.getheader('Content-Type') or '').split(';', 1)[0].strip().lower()
            if mime != 'text/event-stream':
                raise ModelAPIError('invalid_envelope', 'Streaming API must return Content-Type text/event-stream')
            envelope = _read_chat_stream(chunks, transport)
            transport['phase'] = 'response_complete'
            return status, retry, json.dumps(envelope, ensure_ascii=False).encode('utf-8')
        raw = b''.join(chunks)
        transport['phase'] = 'response_complete'
        return status, retry, raw
    finally:
        if response is not None:
            response.close()
        conn.close()


def _envelope_text(envelope, provider):
    if not isinstance(envelope, dict) or envelope.get('error'):
        raise ModelAPIError('invalid_envelope', 'Model API did not return a successful response envelope')
    if provider == 'openai_chat':
        choices = envelope.get('choices')
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ModelAPIError('invalid_envelope', 'Chat API must return exactly one completion')
        choice = choices[0]
        message = choice.get('message')
        if not isinstance(message, dict):
            raise ModelAPIError('invalid_envelope', 'Chat API returned an invalid message')
        if message.get('refusal') or choice.get('finish_reason') == 'content_filter':
            raise ModelAPIError('refusal', 'Model declined the request; no scene was accepted')
        if choice.get('finish_reason') == 'length' and not message.get('tool_calls'):
            raise ModelAPIError('output_truncated', 'Chat output exhausted the token budget; no partial JSON was accepted')
        if choice.get('finish_reason') != 'stop' or message.get('tool_calls'):
            raise ModelAPIError('incomplete', 'Chat response was incomplete or requested tools; no scene was accepted')
        content = message.get('content')
        if isinstance(content, str):
            return content
        if isinstance(content, list) and all(isinstance(v, dict) and v.get('type') == 'text' and isinstance(v.get('text'), str) for v in content):
            return ''.join(v['text'] for v in content)
        raise ModelAPIError('invalid_envelope', 'Chat response contained no JSON text')
    if provider == 'openai_responses':
        if envelope.get('status') == 'incomplete' and isinstance(envelope.get('incomplete_details'), dict) and envelope['incomplete_details'].get('reason') == 'max_output_tokens':
            raise ModelAPIError('output_truncated', 'Responses output exhausted the token budget; no partial JSON was accepted')
        if envelope.get('status') != 'completed' or envelope.get('incomplete_details'):
            raise ModelAPIError('incomplete', 'Responses API output was incomplete; no scene was accepted')
        output = envelope.get('output')
        if not isinstance(output, list):
            raise ModelAPIError('invalid_envelope', 'Responses API returned invalid output')
        texts = []
        for item in output:
            if not isinstance(item, dict):
                raise ModelAPIError('invalid_envelope', 'Responses API returned invalid output')
            if item.get('type') == 'reasoning':
                continue
            if item.get('type') != 'message' or item.get('status', 'completed') != 'completed':
                raise ModelAPIError('incomplete', 'Responses API returned unexpected or incomplete output')
            content = item.get('content')
            if not isinstance(content, list):
                raise ModelAPIError('invalid_envelope', 'Responses API returned invalid content')
            for part in content:
                if not isinstance(part, dict):
                    raise ModelAPIError('invalid_envelope', 'Responses API returned invalid content')
                if part.get('type') == 'refusal':
                    raise ModelAPIError('refusal', 'Model declined the request; no scene was accepted')
                if part.get('type') != 'output_text' or not isinstance(part.get('text'), str):
                    raise ModelAPIError('invalid_envelope', 'Responses API returned unexpected content')
                texts.append(part['text'])
        if not texts:
            raise ModelAPIError('invalid_envelope', 'Responses API contained no JSON text')
        return ''.join(texts)
    if envelope.get('stop_reason') == 'refusal':
        raise ModelAPIError('refusal', 'Model declined the request; no scene was accepted')
    if envelope.get('stop_reason') == 'max_tokens':
        raise ModelAPIError('output_truncated', 'Anthropic output exhausted the token budget; no partial JSON was accepted')
    if envelope.get('stop_reason') != 'end_turn':
        raise ModelAPIError('incomplete', 'Anthropic response was incomplete or requested tools; no scene was accepted')
    content = envelope.get('content')
    if not isinstance(content, list) or not content:
        raise ModelAPIError('invalid_envelope', 'Anthropic response contained no JSON text')
    texts = []
    for part in content:
        if isinstance(part, dict) and part.get('type') in ('thinking', 'redacted_thinking'):
            continue
        if not isinstance(part, dict) or part.get('type') != 'text' or not isinstance(part.get('text'), str):
            raise ModelAPIError('invalid_envelope', 'Anthropic response contained unsupported content')
        texts.append(part['text'])
    if not texts:
        raise ModelAPIError('invalid_envelope', 'Anthropic response contained no JSON text')
    return ''.join(texts)


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite JSON constant')


def _extract_json(text, schema):
    text = text.strip()
    fenced = re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```', text)
    if fenced:
        text = fenced.group(1)
    try:
        result = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError):
        raise ModelAPIError('invalid_json', 'Model output was not one complete JSON object') from None
    if not isinstance(result, dict):
        raise ModelAPIError('invalid_json', 'Model output must be one JSON object')
    try:
        _walk(result, schema, 'result')
        _walk_string_constraints(result, schema)
    except (ValueError, TypeError, KeyError, RecursionError):
        # Validation paths or unexpected field names may contain returned secrets.
        raise ModelAPIError('schema_mismatch', 'Model JSON failed local schema validation; no scene was accepted') from None
    return result


def _usage(envelope):
    """Whitelist numeric usage fields; never persist arbitrary provider strings."""
    if not isinstance(envelope, dict):
        return {}
    usage = envelope.get('usage', {})
    if not isinstance(usage, dict):
        return {}
    keys = ('input_tokens', 'output_tokens', 'total_tokens', 'prompt_tokens', 'completion_tokens',
            'cache_creation_input_tokens', 'cache_read_input_tokens')
    return {key: usage[key] for key in keys if type(usage.get(key)) is int and 0 <= usage[key] <= 10**12}


def _usage_invalid(envelope):
    """Preserve invalid-counter presence without persisting arbitrary data."""
    if not isinstance(envelope, dict) or 'usage' not in envelope:
        return False
    usage = envelope['usage']
    if not isinstance(usage, dict):
        return True
    fields = ('input_tokens', 'output_tokens', 'total_tokens', 'prompt_tokens', 'completion_tokens',
              'cache_creation_input_tokens', 'cache_read_input_tokens')
    return any(key in usage and (type(usage[key]) is not int or not 0 <= usage[key] <= 10**12)
               for key in fields)


def _envelope_metadata(envelope, provider):
    """Retain billing and finish metadata before validating the generated JSON."""
    result = {'usage': _usage(envelope)}
    if _usage_invalid(envelope):
        result['usage_invalid'] = True
    if not isinstance(envelope, dict):
        return result
    if provider == 'openai_chat':
        choices = envelope.get('choices')
        if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
            if choices[0].get('finish_reason') is not None:
                result['finish_reason'] = _safe_finish(choices[0]['finish_reason'])
    elif provider == 'anthropic':
        if envelope.get('stop_reason') is not None:
            result['finish_reason'] = _safe_finish(envelope['stop_reason'])
    else:
        state = envelope.get('status')
        result['response_status'] = state if state in ('completed', 'incomplete', 'failed', 'cancelled', 'queued', 'in_progress') else 'unknown'
        details = envelope.get('incomplete_details')
        if isinstance(details, dict):
            reason = details.get('reason')
            result['incomplete_reason'] = reason if reason in ('max_output_tokens', 'content_filter') else 'unknown'
    return result


def call_model(prompt, images, schema, output, timeout=600, effort='medium', *, config=None):
    """Apply batch-wide backpressure before executing the configured API call."""
    from runtime_controls import api_slot
    _number(timeout, 'timeout', .1, 86400)
    config = normalize_config(config)
    output_path = Path(output)
    guard_ambiguous_model_resume([output_path.with_name(output_path.stem + '.api.json')])
    budget = min(timeout, config['timeout_seconds'])
    try:
        with api_slot(budget) as waited:
            remaining = budget - waited
            if remaining < .1:
                raise ModelAPIError('timeout', 'Model API concurrency queue exceeded the request timeout')
            return _call_model(prompt, images, schema, output, timeout=remaining, effort=effort, config=config)
    except TimeoutError:
        raise ModelAPIError('timeout', 'Model API concurrency queue exceeded the request timeout') from None


def _call_model(prompt, images, schema, output, timeout=600, effort='medium', *, config=None):
    """Return validated model JSON and save .raw.json and sanitized .api.json.

    ``effort`` is kept for drop-in CLI compatibility; API reasoning settings are
    sent only when explicitly present in config. timeout and timeout_seconds
    cap the entire request/retry period, whichever is smaller.
    """
    config = normalize_config(config)
    _number(timeout, 'timeout', .1, 86400)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('prompt must be a nonempty string')
    _schema_contract(schema)
    if schema.get('type') != 'object':
        raise ValueError('schema root must be an object')
    output = Path(output)
    started = time.monotonic()
    # Report only sizes, never prompt content, image paths, or credentials.
    # Schema size is its compact UTF-8 JSON representation. Neither these
    # measurements nor the character guard are token/cost estimates.
    request_metrics = {
        'prompt_characters': len(prompt),
        'prompt_utf8_bytes': len(prompt.encode('utf-8')),
        'schema_utf8_bytes': len(json.dumps(schema, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')),
        'image_count': len(images) if isinstance(images, (list, tuple)) else 0,
        'max_output_tokens': config['max_output_tokens'],
    }
    record = {'config': public_config(config), 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'schema_sha256': hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
              'image_sha256': [], 'attempts': 0, 'status': 'failed', 'transport_attempts': [],
              'request_metrics': request_metrics}
    budget_ticket = None
    def finish_budget():
        nonlocal budget_ticket
        if budget_ticket is not None:
            from scene_budget import BudgetError
            try:
                budget_ticket.finish(record['transport_attempts'][-1])
            except BudgetError as exc:
                raise ModelAPIError(exc.code, str(exc)) from None
            budget_ticket = None
    try:
        prompt_limit = config.get('max_prompt_characters', 0)
        if prompt_limit and request_metrics['prompt_characters'] > prompt_limit:
            raise ModelAPIError('input_budget_exceeded',
                                'Prompt exceeds configured max_prompt_characters; compact request text or explicitly raise the character limit')
        key = _credential(config)
        encoded, hashes = _image_payloads(images)
        record['image_sha256'] = hashes
        body = json.dumps(_payload(prompt, encoded, schema, config), ensure_ascii=False, allow_nan=False).encode('utf-8')
        deadline = time.monotonic() + min(timeout, config['timeout_seconds'])
        for attempt in range(config['max_retries'] + 1):
            finish_budget()
            from scene_budget import BudgetError, reserve_request
            try:
                budget_ticket = reserve_request(output, config['max_output_tokens'])
            except BudgetError as exc:
                raise ModelAPIError(exc.code, str(exc)) from None
            record['attempts'] = attempt + 1
            transport = {'attempt': attempt + 1, 'stream': config.get('stream', False),
                         'request_metrics': dict(request_metrics)}
            if budget_ticket is not None:
                transport['scene_budget_request_id'] = budget_ticket.request_id
            record['transport_attempts'].append(transport)
            try:
                status, retry, raw = _request(config, body, key, deadline, transport=transport)
                record['http_status'] = status
                transport['http_status'] = status
                if 200 <= status < 300:
                    transport['phase'] = 'response_complete'
            except (OSError, http.client.HTTPException) as exc:
                # Type names explain TLS/DNS/socket failures without retaining
                # arbitrary exception bodies that could contain credentials.
                record['network_error_type'] = type(exc).__name__
                transport['network_error_type'] = type(exc).__name__
                # No idempotency guarantee is available for configured model
                # endpoints. Retrying after a possible POST write can charge
                # for a second generation even when the first one succeeded.
                may_have_executed = (transport.get('request_started', False)
                                     or transport.get('http_status') is not None
                                     or transport.get('response_bytes', 0) > 0)
                if may_have_executed:
                    transport['automatic_replay_blocked'] = True
                if time.monotonic() >= deadline:
                    raise ModelAPIError('timeout', 'Model API exceeded the total request timeout') from None
                if may_have_executed:
                    if transport.get('http_status') is not None or transport.get('response_bytes', 0) > 0:
                        raise ModelAPIError('response_interrupted', 'Model API response was interrupted after the request may have executed; automatic replay was blocked') from None
                    raise ModelAPIError('request_outcome_unknown', 'Model API connection failed after sending began; the request may have executed, so automatic replay was blocked') from None
                if attempt == config['max_retries']:
                    raise ModelAPIError('network_error', f'Model API network request failed ({type(exc).__name__}); check the configured endpoint and connectivity') from None
                status, retry, raw = None, 0, None
            finally:
                for field in ('http_status', 'usage', 'usage_invalid', 'finish_reason'):
                    if field in transport:
                        record[field] = transport[field]
            if status is not None and 200 <= status < 300:
                try:
                    envelope = json.loads(raw, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
                except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
                    raise ModelAPIError('invalid_envelope', 'Model API returned an invalid JSON response envelope') from None
                metadata = _envelope_metadata(envelope, config['provider'])
                record.update(metadata)
                transport.update(metadata)
                result = _extract_json(_envelope_text(envelope, config['provider']), schema)
                atomic_json(output.with_name(output.stem + '.raw.json'), result)
                record['status'] = 'completed'
                return result
            retryable = status is None or status == 429 or (status is not None and 500 <= status <= 599)
            if not retryable or attempt == config['max_retries']:
                if status is not None and 300 <= status < 400:
                    raise ModelAPIError('redirect_blocked', 'Model API redirect was blocked; configure the final endpoint explicitly')
                raise ModelAPIError('http_error', f'Model API returned HTTP {status}; verify model, endpoint, response_format, quota, and credentials')
            delay = max(retry, min(8, .5 * 2**attempt))
            remaining = deadline - time.monotonic()
            if remaining <= delay:
                raise ModelAPIError('timeout', 'Model API retry would exceed the total request timeout')
            time.sleep(delay)
    except ModelAPIError as exc:
        if exc.code == 'incomplete' and record['transport_attempts']:
            last = record['transport_attempts'][-1]
            if last.get('request_started') is True and last.get('completion_marker_missing') is True:
                last['automatic_replay_blocked'] = True
        if exc.code == 'timeout' and record['transport_attempts']:
            last = record['transport_attempts'][-1]
            status = last.get('http_status')
            if (last.get('request_started') is True
                    and last.get('phase') != 'response_complete'
                    and (status is None or 200 <= status < 300)):
                # Deadline checks inside _request/_response_chunks raise this
                # directly rather than entering the network-exception handler.
                # Explicit 429/5xx responses are known rejections and retain
                # their normal resumability after a backoff deadline expires.
                last['automatic_replay_blocked'] = True
        record['failure_code'] = exc.code
        raise
    finally:
        # Commit the audit before releasing the durable reservation. A crash
        # between these writes leaves pending capacity and blocks resume,
        # rather than losing both the replay guard and the reservation.
        record['elapsed_seconds'] = round(time.monotonic() - started, 3)
        atomic_json(output.with_name(output.stem + '.api.json'), record)
        try:
            finish_budget()
        except ModelAPIError as exc:
            record.update(status='failed', failure_code=exc.code)
            atomic_json(output.with_name(output.stem + '.api.json'), record)
            raise
