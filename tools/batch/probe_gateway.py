"""Small, explicit checks of an OpenAI-compatible gateway before a batch uses it.

    with-weidian-key python3 -I probe_gateway.py models
    with-weidian-key python3 -I probe_gateway.py text   [--model gpt-6.1-sol]
    with-weidian-key python3 -I probe_gateway.py schema [--model gpt-6.1-sol]   # strict json_schema + one image
    with-weidian-key python3 -I probe_gateway.py bodysize --mb 15 30 50          # free: deliberately invalid JSON

Base URL from GATEWAY_BASE_URL (default https://ai.vdian.net/openai); key from
WEIDIAN_AI_API_KEY. Connects directly (no proxy: the gateway is on the internal network). Prints only HTTP
status, error type/code/message, model ids, usage and the parsed output, with secrets
redacted. Never prints the key. `text` and `schema` each send one tiny paid request.
"""
import argparse
import base64
import http.client
import json
import os
import re
import struct
import sys
import time
import zlib
from urllib.parse import urlsplit

BASE = os.environ.get('GATEWAY_BASE_URL', 'https://ai.vdian.net/openai')
REDACT = [(re.compile(r'(sk-|cr_)[A-Za-z0-9_\-]{6,}'), r'\1***'), (re.compile(r'org-[A-Za-z0-9]{6,}'), 'org-***'),
          (re.compile(r'proj_[A-Za-z0-9]{6,}'), 'proj_***')]


def clean(text):
    for pattern, replacement in REDACT:
        text = pattern.sub(replacement, text)
    return text


def red_png(size=64):
    raw = b''.join(b'\x00' + b'\xd0\x20\x20' * size for _ in range(size))
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def call(method, path, key, body=None, timeout=300):
    url = urlsplit(BASE)
    conn = http.client.HTTPSConnection(url.hostname, url.port or 443, timeout=timeout)
    headers = {'Authorization': f'Bearer {key}', 'Accept': 'application/json', 'User-Agent': 'scene-factory/1'}
    data = None
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    started = time.monotonic()
    conn.request(method, url.path + path, body=data, headers=headers)
    response = conn.getresponse()
    raw = response.read(4 * 1024 * 1024)
    elapsed = time.monotonic() - started
    print(f'HTTP {response.status} in {elapsed:.1f}s, content-type {response.getheader("Content-Type")}, {len(raw)} bytes')
    for name, value in response.getheaders():
        if name.lower().startswith(('x-ratelimit', 'retry-after')):
            print(f'  {name}: {value}')
    try:
        return response.status, json.loads(raw)
    except ValueError:
        print('  non-JSON body:', clean(raw[:300].decode('utf-8', 'replace')))
        return response.status, None


def report(status, payload):
    if not isinstance(payload, dict):
        return
    error = payload.get('error')
    if error:
        error = error if isinstance(error, dict) else {'message': error}
        print('  error.type:', error.get('type'), '| code:', error.get('code'))
        print('  error.message:', clean(str(error.get('message')))[:600])
        return
    usage = payload.get('usage') or {}
    print('  status:', payload.get('status'), '| model:', payload.get('model'),
          '| usage:', {k: usage.get(k) for k in ('input_tokens', 'output_tokens', 'total_tokens')})
    texts = [part.get('text') for item in payload.get('output', []) if item.get('type') == 'message'
             for part in item.get('content', []) if part.get('type') == 'output_text']
    print('  output_text:', clean(' '.join(t for t in texts if t))[:400])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('check', choices=('models', 'text', 'schema', 'bodysize'))
    parser.add_argument('--model', default='gpt-6.1-sol')
    parser.add_argument('--mb', type=float, nargs='+', default=[15, 30, 50])
    args = parser.parse_args()
    key = os.environ.get('WEIDIAN_AI_API_KEY')
    if not key:
        sys.exit('WEIDIAN_AI_API_KEY is not set (run through the with-weidian-key wrapper)')
    if args.check == 'bodysize':
        bodysize(key, args.mb)
        return
    if args.check == 'models':
        status, payload = call('GET', '/models', key, timeout=60)
        if isinstance(payload, dict) and isinstance(payload.get('data'), list):
            ids = sorted(m.get('id', '') for m in payload['data'])
            related = [i for i in ids if re.search(r'6\.1|sol', i)]
            print(f'  models visible: {len(ids)}; sol/6.1 related: {related}; {args.model} listed: {args.model in ids}')
        else:
            report(status, payload)
        return
    body = {'model': args.model, 'store': False, 'max_output_tokens': 2048, 'reasoning': {'effort': 'low'}}
    if args.check == 'text':
        body['input'] = 'Reply with the single word OK.'
    else:
        image = base64.b64encode(red_png()).decode('ascii')
        body['input'] = [{'role': 'user', 'content': [
            {'type': 'input_text', 'text': 'What is the dominant colour of this image? Answer in the schema.'},
            {'type': 'input_image', 'image_url': f'data:image/png;base64,{image}'}]}]
        body['text'] = {'format': {'type': 'json_schema', 'name': 'probe', 'strict': True, 'schema': {
            'type': 'object', 'additionalProperties': False, 'required': ['colour', 'confidence'],
            'properties': {'colour': {'type': 'string'}, 'confidence': {'type': 'number'}}}}}
    status, payload = call('POST', '/responses', key, body)
    report(status, payload)


def bodysize(key, sizes_mb):
    """Upload bodies of batch size that are deliberately invalid JSON (unterminated string).

    The gateway must read the whole body before it can parse it, so a size limit shows
    up as 413 or a reset connection, and an accepted size as a JSON parse error (4xx).
    Nothing valid reaches the model, so no tokens are used."""
    url = urlsplit(BASE)
    for mb in sizes_mb:
        body = b'{"model":"probe","pad":"' + b'x' * int(mb * 1024 * 1024)
        conn = http.client.HTTPSConnection(url.hostname, url.port or 443, timeout=300)
        started = time.monotonic()
        try:
            conn.request('POST', url.path + '/responses', body=body, headers={
                'Authorization': f'Bearer {key}', 'Content-Type': 'application/json', 'User-Agent': 'scene-factory/1'})
            response = conn.getresponse()
            raw = response.read(2048)
            print(f'{mb:5.1f} MB -> HTTP {response.status} in {time.monotonic() - started:.1f}s: '
                  f'{clean(raw[:200].decode("utf-8", "replace"))}')
        except (OSError, http.client.HTTPException) as exc:
            print(f'{mb:5.1f} MB -> {type(exc).__name__} after {time.monotonic() - started:.1f}s')
        finally:
            conn.close()


if __name__ == '__main__':
    main()
