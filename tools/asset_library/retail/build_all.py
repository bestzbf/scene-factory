"""Build, ingest and sample-render the planned retail assets into a new library version.

Stages resume: an existing package with a matching spec is reused, and intake
reuses committed receipts. Builds run several Blender processes in parallel.
No network and no model API.

    python build_all.py WORK_DIR        # WORK_DIR/specs/*.json written by plan_retail.py --write

Writes WORK_DIR/packages, WORK_DIR/library (the ingested catalog) and build_summary.json.
Environment: BLENDER (default /Applications/Blender.app/Contents/MacOS/Blender), RETAIL_BUILD_WORKERS (6).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
if len(sys.argv) < 2:
    sys.exit(__doc__)
ROOT = Path(sys.argv[1]).resolve()
SPECS, PACKAGES, LIBRARY, LOGS = ROOT / 'specs', ROOT / 'packages', ROOT / 'library', ROOT / 'build_logs'
BLENDER = os.environ.get('BLENDER', '/Applications/Blender.app/Contents/MacOS/Blender')
REPO = HERE.parents[2]   # repository root
WORKERS = int(os.environ.get('RETAIL_BUILD_WORKERS', '6'))
started = time.monotonic()
GENERATOR = HERE / 'build_retail_asset.py'
GENERATOR_SHA256 = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()


def log(message):
    print(f"{time.strftime('%H:%M:%S')} [{time.monotonic() - started:6.0f}s] {message}", flush=True)


def build(spec_path):
    spec = json.loads(spec_path.read_text())
    package = PACKAGES / spec['asset_id']
    report = package / 'build_report.json'
    if (package / 'model.glb').is_file() and report.is_file():
        previous = json.loads(report.read_text())
        if previous['spec'] == spec and previous.get('generator_sha256') == GENERATOR_SHA256:
            return spec['asset_id'], 'cached'
    shutil.rmtree(package, ignore_errors=True)
    staging = PACKAGES / ('.staging-' + spec['asset_id'])
    shutil.rmtree(staging, ignore_errors=True)
    LOGS.mkdir(exist_ok=True)
    with (LOGS / f"{spec['asset_id']}.log").open('w') as stream:
        result = subprocess.run([BLENDER, '-b', '--factory-startup', '--python-exit-code', '1', '--python',
                                 str(GENERATOR), '--', str(spec_path), str(staging)],
                                stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        return spec['asset_id'], 'build_failed'
    built = json.loads((staging / 'build_report.json').read_text())
    built['generator_sha256'] = GENERATOR_SHA256
    (staging / 'build_report.json').write_text(json.dumps(built, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(staging, package)
    return spec['asset_id'], 'built'


def main():
    specs = sorted(SPECS.glob('*.json'))
    PACKAGES.mkdir(exist_ok=True)
    log(f'=== build: {len(specs)} specs, {WORKERS} Blender workers ===')
    failures = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for count, (asset_id, status) in enumerate(pool.map(build, specs), 1):
            if status == 'build_failed':
                failures.append(asset_id); log(f'BUILD FAILED {asset_id} (see build_logs/{asset_id}.log)')
            if count % 100 == 0 or count == len(specs):
                log(f'build {count}/{len(specs)} ({len(failures)} failed)')
    # Intake reads only the two-file protocol: stage model.glb + asset.json per package.
    intake_input = ROOT / 'intake_input'
    shutil.rmtree(intake_input, ignore_errors=True)
    for package in sorted(p for p in PACKAGES.iterdir() if p.is_dir() and not p.name.startswith('.')):
        target = intake_input / package.name
        target.mkdir(parents=True)
        for name in ('model.glb', 'asset.json'):
            os.link(package / name, target / name)
    sys.path.insert(0, str(REPO / 'scene_factory'))
    from asset_intake import ingest_library  # noqa: E402
    counts = {}

    def progress(record):
        counts[record['status']] = counts.get(record['status'], 0) + 1
        if record['status'] != 'technical_ready':
            log(f"intake QUARANTINED {record['asset_id']}: {record.get('error')}")
        done = sum(counts.values())
        if done % 200 == 0:
            log(f'intake {done}: {counts}')

    log(f'=== intake into {LIBRARY} ===')
    report = ingest_library(intake_input, LIBRARY, blender=BLENDER, timeout=300, progress=progress)
    summary = {'specs': len(specs), 'build_failed': failures, 'technical_ready': report['technical_ready'],
               'quarantined': report['quarantined'], 'catalog': report.get('catalog'),
               'elapsed_seconds': round(time.monotonic() - started, 1)}
    (ROOT / 'build_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    log(f'DONE {summary}')


if __name__ == '__main__':
    main()
