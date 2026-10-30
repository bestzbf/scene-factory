"""One minimal Responses API call to see why the batch got HTTP 429.

    with-openai-key python3 -I diagnose_openai_429.py [--model gpt-6.1-sol]

Sends a few input tokens with max_output_tokens=16 through the loopback CONNECT proxy
(TLS stays end to end). Prints only the HTTP status, the rate-limit headers and the
error type/code/message, with key/org/project identifiers redacted. Never prints the key.
"""
import argparse
import http.client
import json
import os
import re
import sys

REDACT = [(re.compile(r'sk-[A-Za-z0-9_\-]{6,}'), 'sk-***'), (re.compile(r'org-[A-Za-z0-9]{6,}'), 'org-***'),
          (re.compile(r'proj_[A-Za-z0-9]{6,}'), 'proj_***')]


def clean(text):
    for pattern, replacement in REDACT:
        text = pattern.sub(replacement, text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-6.1-sol')
    parser.add_argument('--proxy', default='127.0.0.1:7897')
    args = parser.parse_args()
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        sys.exit('OPENAI_API_KEY is not set (run through the with-openai-key wrapper)')
    host, port = args.proxy.split(':')
    if host not in ('127.0.0.1', 'localhost', '::1'):
        sys.exit('proxy must be loopback')
    conn = http.client.HTTPSConnection(host, int(port), timeout=120)
    conn.set_tunnel('api.openai.com', 443)
    body = json.dumps({'model': args.model, 'input': 'Reply with OK.', 'max_output_tokens': 16})
    conn.request('POST', '/v1/responses', body=body,
                 headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    response = conn.getresponse()
    raw = response.read(65536)
    print('HTTP', response.status)
    for name, value in response.getheaders():
        if name.lower().startswith('x-ratelimit') or name.lower() in ('retry-after', 'openai-processing-ms'):
            print(f'  {name}: {value}')
    try:
        payload = json.loads(raw)
    except ValueError:
        print('  non-JSON body,', len(raw), 'bytes')
        return
    error = payload.get('error')
    if error:
        print('  error.type:', error.get('type'))
        print('  error.code:', error.get('code'))
        print('  error.message:', clean(str(error.get('message')))[:600])
    else:
        usage = payload.get('usage') or {}
        print('  status:', payload.get('status'), '| usage:', {k: usage.get(k) for k in ('input_tokens', 'output_tokens', 'total_tokens')})


if __name__ == '__main__':
    main()
