#!/usr/bin/env python3
"""Bounded photo -> scene -> visual review -> repair batch workflow.

The existing batch.py remains a geometry/export runner. This runner additionally
requires a source-image review and keeps rejected scenes out of accepted output.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from batch import (ROOT, atomic_json, canonical_bytes, collect_images, completed_job,
                   discover_blender, file_sha256, make_job, number, prepare_job,
                   read_json, run_stage, utc_now)
from schema import validate
from visual_review import PROMPT, REVIEW_SCHEMA, quality_gate, review_scene, validate_review

VERSION = 1
SNAPSHOT_FILES = ('scene_spec.json', 'scene.blend', 'rgb.png', 'metadata.json', 'annotations.json',
                  'review.json', 'review_provenance.json')
TERMINAL = ('accepted', 'needs_review', 'insufficient_information')


def contained(directory, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('invalid relative artifact path')
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('artifact outside job directory')
    return path


def snapshot(directory, iteration):
    result = {}
    for name in SNAPSHOT_FILES:
        path = iteration / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Missing reviewed artifact: ' + str(path))
        result[str(path.relative_to(directory))] = file_sha256(path)
    return result


def resume_result(directory):
    path = directory / 'quality_result.json'
    if not path.exists():
        return None
    result = read_json(path)
    if result.get('status') not in TERMINAL:
        return None
    hashes = result.get('artifact_hashes', {})
    if len(hashes) != len(SNAPSHOT_FILES):
        raise ValueError('Incomplete final artifact manifest; use a new output directory')
    for name, digest in hashes.items():
        artifact = contained(directory, name)
        if not artifact.is_file() or file_sha256(artifact) != digest:
            raise ValueError('Reviewed artifact changed; use a new output directory: ' + name)
    iteration = contained(directory, result['iteration'])
    expected = {str((iteration / name).relative_to(directory)) for name in SNAPSHOT_FILES}
    if set(hashes) != expected:
        raise ValueError('Final manifest does not describe selected iteration')
    if result['status'] == 'accepted':
        report = read_json(iteration / 'review.json')
        gate = quality_gate(report, read_json(iteration / 'metadata.json'), read_json(iteration / 'annotations.json'))
        if not gate['accepted']:
            raise ValueError('Stored acceptance no longer passes gate')
    return result


def cached_review(source, iteration):
    if not (iteration / 'review.json').exists() or not (iteration / 'review_provenance.json').exists():
        return None
    provenance = read_json(iteration / 'review_provenance.json')
    if provenance.get('review_sha256') != file_sha256(iteration / 'review.json'):
        raise ValueError('Cached review changed or lacks integrity binding; use a new output directory')
    paths = {'source': source, 'render': iteration / 'rgb.png', 'spec': iteration / 'scene_spec.json',
             'metadata': iteration / 'metadata.json', 'annotations': iteration / 'annotations.json',
             'blend': iteration / 'scene.blend'}
    if any(not p.is_file() or provenance.get(key + '_sha256') != file_sha256(p) for key, p in paths.items()):
        raise ValueError('Cached review input changed; use a new output directory')
    report = validate_review(read_json(iteration / 'review.json'))
    return report, provenance


def render_scene(spec_path, iteration, job, args):
    render = job['render']
    command = [args.blender, '-b', '--factory-startup', '--python-exit-code', '1', '--python',
               str(ROOT / 'generate_scene.py'), '--', '--spec', str(spec_path), '--output-dir', str(iteration),
               '--width', str(render['width']), '--height', str(render['height']), '--samples', str(render['samples']),
               '--asset-quality', 'detailed']
    if render['desk_details']:
        command.append('--desk-details')
    # Invalidate completion first; a failed attempt must never inherit success.
    for name in ('metadata.json', 'review.json', 'review_provenance.json'):
        (iteration / name).unlink(missing_ok=True)
    run_stage(command, iteration / 'blender.log', args.render_timeout)
    complete, reason = completed_job(iteration)
    if not complete:
        raise RuntimeError('Blender validation failed: ' + reason)
    for name in ('rgb.png', 'annotations.json'):
        if not (iteration / name).is_file():
            raise RuntimeError('Missing ' + name)


def make_quality_job(image, args):
    job = make_job(image, args.width, args.height, args.samples, ROOT / 'photo_to_spec.py',
                   ROOT / 'generate_scene.py', 'detailed', args.desk_details)
    job['quality_pipeline'] = {
        'version': VERSION, 'max_repairs': args.max_repairs,
        'review_timeout': args.review_timeout, 'render_timeout': args.render_timeout,
        'reviewer_sha256': file_sha256(ROOT / 'visual_review.py'),
        'runner_sha256': file_sha256(Path(__file__)),
        'review_prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
        'review_schema_sha256': hashlib.sha256(canonical_bytes(REVIEW_SCHEMA)).hexdigest(),
        'seed_spec_sha256': file_sha256(args.seed_spec) if args.seed_spec else None,
        'extra_generator_modules': {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(ROOT.glob('lighting*.py'))},
        'policy': 'all>=85;realism>=90;no-major-issues;no-asset-gaps;no-technical-warnings;min1280x720,64samples',
    }
    return job


def run_quality_job(job, args):
    start = time.monotonic()
    source = Path(job['image'])
    directory = Path(args.output_dir) / job['job_id']
    result = {'job_id': job['job_id'], 'image': str(source), 'source_sha256': job['image_sha256'],
              'started_at': utc_now(), 'status': 'failed'}
    lock_fd = None
    prepared = False
    try:
        directory = prepare_job(job, args.output_dir)
        prepared = True
        result['directory'] = str(directory)
        if args.dry_run:
            return {**result, 'status': 'planned'}
        lock = directory / '.quality.lock'
        try:
            lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return {**result, 'status': 'locked', 'error': 'Another runner or interrupted job owns .quality.lock; inspect before removing it'}
        os.write(lock_fd, canonical_bytes({'pid': os.getpid(), 'started_at': utc_now()}))
        if file_sha256(source) != job['image_sha256']:
            raise ValueError('Source image changed before resume')
        previous = resume_result(directory)
        if previous:
            result = {**previous, 'resumed': True}
            return result
        atomic_json(directory / 'quality_result.json', {**result, 'status': 'running'})
        initial = directory / 'initial_spec.json'
        if not initial.exists():
            if args.seed_spec:
                atomic_json(initial, validate(read_json(args.seed_spec)))
                atomic_json(directory / 'analysis_provenance.json', {
                    'source_image': str(source), 'source_sha256': job['image_sha256'],
                    'method': 'explicit user-selected existing scene spec', 'seed_spec': str(args.seed_spec),
                    'seed_spec_sha256': file_sha256(args.seed_spec)})
            else:
                run_stage([sys.executable, str(ROOT / 'photo_to_spec.py'), '--image', str(source),
                           '--output', str(initial), '--timeout', str(args.review_timeout)],
                          directory / 'photo_to_spec.log', args.review_timeout + 15)
        spec = validate(read_json(initial))
        history = []
        for index in range(args.max_repairs + 1):
            if file_sha256(source) != job['image_sha256']:
                raise ValueError('Source image changed during processing')
            iteration = directory / f'iteration_{index:02d}'
            iteration.mkdir(exist_ok=True)
            spec_path = iteration / 'scene_spec.json'
            if spec_path.exists() and read_json(spec_path) != spec:
                raise ValueError('Existing iteration spec differs from repair history')
            atomic_json(spec_path, spec)
            atomic_json(directory / 'progress.json', {'stage': 'render', 'iteration': index, 'at': utc_now()})
            complete, _ = completed_job(iteration)
            if not complete:
                render_scene(spec_path, iteration, job, args)
            atomic_json(directory / 'progress.json', {'stage': 'visual_review', 'iteration': index, 'at': utc_now()})
            reviewed = cached_review(source, iteration)
            report, provenance = reviewed or review_scene(source, iteration, timeout=args.review_timeout)
            gate = quality_gate(report, read_json(iteration / 'metadata.json'), read_json(iteration / 'annotations.json'))
            history.append({'iteration': index, 'scores': report['scores'], 'verdict': report['verdict'],
                            'gate': gate, 'summary': report['summary']})
            atomic_json(directory / 'history.json', history)
            result.update({'iteration': iteration.name, 'history': history, 'gate': gate,
                           'scores': report['scores'], 'summary': report['summary'], 'asset_gaps': report['asset_gaps'],
                           'artifact_hashes': snapshot(directory, iteration)})
            if gate['accepted']:
                result['status'] = 'accepted'
                break
            if report['verdict'] == 'insufficient_information':
                result['status'] = 'insufficient_information'
                break
            candidate = validate(report['revised_spec'])
            if index == args.max_repairs or report['verdict'] != 'repair' or candidate == spec:
                result['status'] = 'needs_review'
                result['stop_reason'] = ('repair_budget_exhausted' if index == args.max_repairs else
                                         'no_supported_repair' if candidate == spec else report['verdict'])
                break
            spec = candidate
        result['scope'] = 'Only reference-camera appearance assessed; single-image geometry and hidden surfaces are inferred'
    except ValueError as exc:
        result.update(status='conflict', error=str(exc))
    except Exception as exc:
        result.update(status='failed', error=str(exc))
    finally:
        if lock_fd is not None:
            try:
                if not result.get('resumed'):
                    result.update(finished_at=utc_now(), elapsed_seconds=round(time.monotonic() - start, 3))
                # Publish while still holding the lock; preserve prior terminal results on conflicts.
                if prepared and result['status'] != 'conflict':
                    atomic_json(directory / 'quality_result.json', result)
            finally:
                os.close(lock_fd)
                (directory / '.quality.lock').unlink(missing_ok=True)
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-dir', type=Path)
    inputs.add_argument('--images', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed-spec', type=Path, help='Use existing spec for exactly one source image; still rerenders/reviews')
    parser.add_argument('--blender')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-repairs', type=int, default=2, help='At most this many revised renders per image (0..5)')
    parser.add_argument('--review-timeout', type=float, default=600)
    parser.add_argument('--render-timeout', type=float, default=1800)
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=1200)
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--desk-details', action='store_true', help='Inferred accessories; not recovered photo evidence')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    try:
        for key, low, high in [('workers', 1, 8), ('max_repairs', 0, 5), ('width', 64, 8192),
                               ('height', 64, 8192), ('samples', 1, 4096)]:
            number(getattr(args, key), key, low, high, True)
        for key in ('review_timeout', 'render_timeout'):
            number(getattr(args, key), key, 1, 86400)
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.input_dir and args.output_dir.is_relative_to(args.input_dir.expanduser().resolve()):
            raise ValueError('Output directory must be outside input directory to avoid re-ingesting renders')
        images = collect_images(args.input_dir, args.images)
        if args.seed_spec:
            args.seed_spec = args.seed_spec.expanduser().resolve()
            if len(images) != 1:
                raise ValueError('--seed-spec is only valid for one image')
            validate(read_json(args.seed_spec))
        args.blender = discover_blender(args.blender) if not args.dry_run else args.blender or 'blender'
        jobs = [make_quality_job(image, args) for image in images]
        ids = [job['job_id'] for job in jobs]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate image content and stem; remove duplicates before dispatch')
    except (ValueError, OSError) as exc:
        print('ERROR ' + str(exc), file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_quality_job, job, args) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with (args.output_dir / 'quality_manifest.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            print(json.dumps({key: result[key] for key in ('job_id', 'status', 'scores', 'error') if key in result}, ensure_ascii=False), flush=True)
    counts = {status: sum(r['status'] == status for r in results) for status in sorted({r['status'] for r in results})}
    accepted = [{'job_id': r['job_id'], 'directory': r['directory'], 'iteration': r['iteration'],
                 'blend': str(Path(r['directory']) / r['iteration'] / 'scene.blend')} for r in results if r['status'] == 'accepted']
    atomic_json(args.output_dir / 'accepted.json', accepted)
    atomic_json(args.output_dir / 'quality_summary.json', {'counts': counts, 'total': len(results),
                'accepted_count': len(accepted), 'acceptance_rate': len(accepted) / len(results),
                'calibration': 'Uncalibrated model gate; not a measured photorealism success rate',
                'at': utc_now(), 'jobs': results})
    if any(r['status'] in ('failed', 'conflict', 'locked') for r in results):
        return 1
    return 3 if any(r['status'] in ('needs_review', 'insufficient_information') for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
