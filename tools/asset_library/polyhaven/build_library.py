"""Download, pack and ingest the Poly Haven plan into a new library version directory.

Every stage resumes: finished downloads, packages and intake receipts are reused.
Downloads go through the official API with an identifying User-Agent and a small
number of parallel connections. No model API is called.

    python build_library.py WORK_DIR [PLAN_JSON]

WORK_DIR receives raw/, packages/, library/ and pack_reports/. PLAN_JSON defaults
to plan.json next to this script (246 CC0 Poly Haven models with category,
placement and published size).
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
if len(sys.argv) < 2:
    sys.exit(__doc__)
ROOT = Path(sys.argv[1]).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
RAW, PACKAGES, LIBRARY, REPORTS = ROOT / 'raw', ROOT / 'packages', ROOT / 'library', ROOT / 'pack_reports'
BLENDER = os.environ.get('BLENDER', '/Applications/Blender.app/Contents/MacOS/Blender')
UA = {'User-Agent': 'SceneFactoryIntake/0.3.42 (internal dataset build)'}
REPO = str(HERE.parents[2])   # repository root

plan = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else HERE / 'plan.json').read_text(encoding='utf-8'))
started = time.monotonic()


def log(message):
    print(f"{time.strftime('%H:%M:%S')} [{time.monotonic() - started:6.0f}s] {message}", flush=True)


def fetch(url):
    request = urllib.request.Request(url, headers=UA)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def download(item):
    target = RAW / item['id']
    if (target / '.complete').is_file():
        return item['id'], 'cached'
    target.mkdir(parents=True, exist_ok=True)
    files = json.loads(fetch(f"https://api.polyhaven.com/files/{item['id']}"))
    bundle = files['gltf']['1k']['gltf']
    entries = [(bundle['url'], target / os.path.basename(bundle['url']), bundle)]
    entries += [(spec['url'], target / relative, spec) for relative, spec in bundle.get('include', {}).items()]
    for url, path, spec in entries:
        data = fetch(url)
        if spec.get('size') and len(data) != spec['size']:
            raise ValueError(f"size mismatch for {path.name}")
        if spec.get('md5') and hashlib.md5(data).hexdigest() != spec['md5']:
            raise ValueError(f"md5 mismatch for {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (target / 'info.json').write_text(json.dumps(item['info'], ensure_ascii=False), encoding='utf-8')
    (target / 'files.json').write_text(json.dumps(files), encoding='utf-8')
    (target / '.complete').write_text('ok')
    return item['id'], 'downloaded'


def pack(item):
    destination = PACKAGES / f"polyhaven_{item['id']}"
    report = REPORTS / f"{item['id']}.json"
    if (destination / 'model.glb').is_file() and (destination / 'asset.json').is_file() and report.is_file():
        return item['id'], 'cached'
    result = subprocess.run(
        [BLENDER, '-b', '--factory-startup', '--python-exit-code', '1',
         '--python', str(HERE / 'pack_polyhaven.py'), '--',
         str(RAW / item['id']), str(destination), item['category'], item['placement'], str(report)],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return item['id'], 'pack_failed: ' + (result.stdout + result.stderr).strip()[-240:]
    return item['id'], 'packed'


def run_stage(name, function, workers):
    log(f"=== {name}: {len(plan)} models, {workers} workers ===")
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (identifier, status) in enumerate(pool.map(lambda item: _safe(function, item), plan), 1):
            if status not in ('cached', 'downloaded', 'packed'):
                failures.append((identifier, status))
                log(f"{name} FAILED {identifier}: {status}")
            if count % 25 == 0 or count == len(plan):
                log(f"{name} {count}/{len(plan)} ({len(failures)} failed)")
    return failures


def _safe(function, item):
    try:
        return function(item)
    except Exception as error:
        return item['id'], f"{type(error).__name__}: {error}"


download_failures = run_stage('download', download, 4)
pack_failures = run_stage('pack', pack, 4)

sys.path.insert(0, REPO)
from scene_factory.asset_intake import ingest_library  # noqa: E402

log(f"=== intake into {LIBRARY} ===")
counts = {'technical_ready': 0, 'quarantined': 0}


def progress(record):
    counts[record['status']] = counts.get(record['status'], 0) + 1
    done = sum(counts.values())
    if record['status'] != 'technical_ready':
        log(f"intake QUARANTINED {record.get('asset_id')}")
    if done % 25 == 0:
        log(f"intake {done} ({counts['technical_ready']} ready / {counts['quarantined']} quarantined)")


intake = ingest_library(PACKAGES, LIBRARY, blender=BLENDER, timeout=300, progress=progress)

summary = {
    'planned': len(plan),
    'download_failures': download_failures,
    'pack_failures': pack_failures,
    'technical_ready': intake['technical_ready'],
    'quarantined': intake['quarantined'],
    'api_requests_to_model_providers': intake['api_requests'],
    'catalog': intake['catalog'],
    'elapsed_seconds': round(time.monotonic() - started, 1),
}
(ROOT / 'build_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
log('SUMMARY ' + json.dumps(summary, ensure_ascii=False))
