#!/usr/bin/env python3
"""Grouped originals -> joint automatic interpretation -> shared Blender -> joint review.

No seed specification or scene-specific code is accepted by this entry point.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager, nullcontext
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

from batch import atomic_json, canonical_bytes, discover_blender, file_sha256, read_json, run_stage, utc_now
from multiview_spec import MAX_VIEWS, SAFE_ID, validate, texture_view_map, resolve_sources
from photo_assets_v2 import extract, image_size
from candidate_selection import candidate_metrics, select_candidate
from batch_index import EVENTS_FILE, append_result, read_results, reset_events
from runtime_controls import render_session, render_slot

ROOT = Path(__file__).resolve().parent
LOCK_PROTOCOL = 'advisory_v1'
LARGE_BATCH_BUDGET_SCENES = 1000
JOB_CONTROL_FILES = {'result.json', 'progress.json', '.lock', '.process.lock', '.scene_budget.lock'}
ASSET_SNAPSHOT_FILE = '.asset-library-snapshot.json'


class OutputInUseError(RuntimeError):
    """Another live process owns the output; PID values are informational only."""


@contextmanager
def process_lock(path):
    """Kernel-backed lock, automatically released on process exit or a crash.

    The file is deliberately retained. Removing a lock inode while another
    process has opened it would allow two processes to lock different inodes.
    Advisory locks are supported on local filesystems; shared/network outputs
    must provide the same operating-system locking guarantees.
    """
    path = Path(path)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    locked = False
    try:
        if os.name == 'nt':
            import msvcrt
            if not os.fstat(fd).st_size:
                os.write(fd, b' ')
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise OutputInUseError('Output is owned by another running process: ' + str(path)) from exc
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise OutputInUseError('Output is owned by another running process: ' + str(path)) from exc
                raise
        locked = True
        payload = canonical_bytes({'lock_protocol': LOCK_PROTOCOL, 'pid': os.getpid(), 'at': utc_now()})
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload)
        os.ftruncate(fd, len(payload))
        os.fsync(fd)
        yield
    finally:
        if locked:
            if os.name == 'nt':
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def _batch_asset_scope(args):
    """Mark workers as sharing the coordinator's locked asset snapshot."""
    sentinel = object()
    previous = getattr(args, '_asset_batch_scope_active', sentinel)
    args._asset_batch_scope_active = True
    try:
        yield
    finally:
        if previous is sentinel:
            try:
                delattr(args, '_asset_batch_scope_active')
            except AttributeError:
                pass
        else:
            args._asset_batch_scope_active = previous


class JobInterrupted(RuntimeError):
    pass


class DiskSpacePause(RuntimeError):
    """A scene was not started because the configured free-space floor was hit."""

    def __init__(self, minimum, available=None, cause=None):
        self.minimum, self.available = minimum, available
        self.cause_type = type(cause).__name__ if cause is not None else None
        if cause is not None:
            message = f'Cannot inspect free space on the output volume ({type(cause).__name__}); scene intake paused'
        else:
            message = f'Free space {available} bytes is below configured minimum {minimum} bytes; scene intake paused'
        super().__init__(message)


DISK_SPACE_PAUSE_EXIT_CODE = 75


def _disk_probe_path(path):
    """Return an existing path on the volume that will contain ``path``.

    ``disk_usage`` cannot stat a not-yet-created output directory.  Walk to
    the nearest existing parent so startup admission works for a fresh output
    directory without creating anything first.
    """
    path = Path(path).expanduser()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _resolve_batch_ledger_root(args):
    """Return the durable budget/rate ledger root for this output shard.

    A configured shared root lets independent output directories (and separate
    processes) consume one SQLite budget and provider quota. Each shard must
    remain inside that root so its request audit can be tied to the ledger.
    """
    output = Path(args.output_dir).expanduser().resolve()
    configured = getattr(args, 'shared_ledger_dir', None)
    root = output if configured is None else Path(configured).expanduser().resolve()
    if not output.is_relative_to(root):
        raise ValueError('batch.shared_ledger_dir must contain every batch output directory')
    return root


def check_minimum_free_disk(path, minimum):
    """Raise when output-volume free space is below the configured admission floor."""
    if type(minimum) is not int or not 0 <= minimum <= 10**15:
        raise ValueError('min_free_disk_bytes must be an integer between 0 and 1000000000000000')
    if not minimum:
        return None
    try:
        available = shutil.disk_usage(path).free
    except OSError as exc:
        raise DiskSpacePause(minimum, cause=exc) from None
    if available < minimum:
        raise DiskSpacePause(minimum, available)
    return available


def _record_disk_space_pause(args, pause):
    """Publish a fail-closed disk pause and stop later admissions.

    Disk admission is deliberately separate from ``stop_event``.  The latter
    is also observed inside API/render work and therefore would cancel scenes
    that were already admitted when free space fell below the floor.
    """
    state = getattr(args, '_disk_space_guard_state', None)
    lock = getattr(args, '_disk_space_guard_lock', None)
    if state is not None:
        if lock is None:
            lock = threading.Lock()
        with lock:
            state['available_bytes'] = pause.available
            state['stop_reason'] = ('disk_usage_unavailable' if pause.available is None
                                    else 'minimum_free_disk_bytes')
            if pause.available is None:
                state['error_type'] = pause.cause_type or 'OSError'
    event = getattr(args, 'disk_space_stop_event', None)
    if event is not None:
        event.set()


def check_scene_start_disk_space(args):
    """Check immediately before a non-resumed scene begins expensive work."""
    gate = getattr(args, 'disk_space_stop_event', None)
    if gate is not None and gate.is_set():
        state = getattr(args, '_disk_space_guard_state', {}) or {}
        minimum = getattr(args, 'min_free_disk_bytes', 0)
        raise DiskSpacePause(minimum, state.get('available_bytes'))
    try:
        available = check_minimum_free_disk(_disk_probe_path(args.output_dir),
                                            getattr(args, 'min_free_disk_bytes', 0))
        ledger_root = getattr(args, '_batch_ledger_root', None)
        if ledger_root is not None and Path(ledger_root).resolve() != Path(args.output_dir).resolve():
            ledger_available = check_minimum_free_disk(_disk_probe_path(ledger_root),
                                                       getattr(args, 'min_free_disk_bytes', 0))
            if ledger_available is not None:
                available = min(available, ledger_available) if available is not None else ledger_available
        state = getattr(args, '_disk_space_guard_state', None)
        lock = getattr(args, '_disk_space_guard_lock', None)
        if state is not None:
            with lock or threading.Lock():
                state['available_bytes'] = available
        return available
    except DiskSpacePause as exc:
        _record_disk_space_pause(args, exc)
        raise


def publish_disk_space_pause(pause, *, stage='disk_space_preflight'):
    """Emit a machine-readable pause without touching batch indexes."""
    payload = {'stage': stage, 'state': 'disk_paused', 'stop_reason': 'disk_space_guard',
               'minimum_free_bytes': pause.minimum, 'available_bytes': pause.available,
               'error': str(pause)}
    if pause.available is None and pause.cause_type is not None:
        payload['error_type'] = pause.cause_type
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return DISK_SPACE_PAUSE_EXIT_CODE


def check_interrupted(args):
    event = getattr(args, 'stop_event', None)
    if event is not None and event.is_set():
        raise JobInterrupted('Batch interrupted; resume with the same command to reuse committed stages')


def load_groups(path):
    path = Path(path).resolve()
    data = read_json(path)
    if not isinstance(data,dict) or set(data) != {'schema_version', 'scenes'} or data['schema_version'] != 1:
        raise ValueError('Group manifest requires schema_version=1 and scenes')
    if not isinstance(data['scenes'], list) or not data['scenes']:
        raise ValueError('scenes must be a nonempty list')
    seen, groups = set(), []
    for group in data['scenes']:
        if not isinstance(group,dict) or set(group) != {'scene_id', 'images'}:
            raise ValueError('Each group contains only scene_id and images')
        identifier = group['scene_id']
        if not isinstance(identifier, str) or not SAFE_ID.fullmatch(identifier) or identifier in seen:
            raise ValueError('Duplicate or invalid scene_id')
        seen.add(identifier)
        if not isinstance(group['images'], list) or not 1 <= len(group['images']) <= MAX_VIEWS:
            raise ValueError(f'A scene requires 1..{MAX_VIEWS} image paths')
        sources = []
        for source in group['images']:
            if not isinstance(source, str) or not source:
                raise ValueError('Image paths must be nonempty strings')
            source = Path(source).expanduser()
            source = source.resolve() if source.is_absolute() else (path.parent / source).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            if source in sources:
                raise ValueError('Duplicate image within a scene')
            sources.append(source)
        groups.append({'scene_id': identifier, 'images': sources})
    return groups


def require_large_batch_budget(scene_count, policy, *, dry_run=False):
    """Require finite aggregate limits before a large run can spend tokens."""
    if type(scene_count) is not int or scene_count < 0:
        raise ValueError('scene_count must be a nonnegative integer')
    if dry_run or scene_count < LARGE_BATCH_BUDGET_SCENES:
        return
    missing = large_batch_budget_missing(policy)
    if missing:
        raise ValueError(
            'Live batches with 1,000 or more scenes require positive batch_budget limits for '
            + ', '.join(missing) + '; set them from your provider quota. '
            'Dry-run remains available without these limits.')


def large_batch_budget_missing(policy):
    required = ('max_requests', 'max_output_tokens', 'max_reported_tokens')
    if not isinstance(policy, dict):
        return list(required)
    return [key for key in required if type(policy.get(key)) is not int or policy[key] <= 0]


def pipeline_hashes():
    names = ['batch_multiview.py', 'vision_multiview.py', 'multiview_spec.py', 'generate_multiview.py',
             'vision_v2.py', 'generate_v2.py', 'generate_scene.py', 'spec_v2.py', 'schema.py',
             'photo_assets_v2.py', 'v2_materials.py', 'interior_assets.py', 'detailed_assets.py',
             'batch.py', 'visual_review.py', 'appearance_lighting.py', 'calibrate_reference.py', 'reference_tone.py',
             'model_api.py', 'factory_config.py', 'staged_analysis.py', 'automatic_repair.py',
             'cli.py', '__init__.py', '__main__.py', 'report_multiview.py',
             'asset_catalog.py', 'asset_manifest.py', 'external_assets.py', 'auto_asset_binding.py', 'asset_context.py',
             'library_layout.py', 'library_selection.py', 'asset_intake.py', 'asset_intake_blender.py',
             'diagnostics.py', 'dataset_export.py', 'runtime_controls.py', 'candidate_selection.py',
             'lighting_fit.py', 'base_material_binding.py', 'procedural_cloth.py', 'exterior_context.py', 'part_projection.py', 'repair_material_validation.py',
             'compact_context.py', 'compact_scene.py', 'anchored_scene.py', 'pattern_scene.py', 'pattern_prompt.py', 'compact_staged_detail.py', 'plan_geometry_patch.py', 'repair_strategy.py', 'scene_budget.py', 'model_image_inputs.py', 'batch_index.py', 'batch_guard.py', 'provider_limits.py']
    paths = [ROOT / name for name in names if (ROOT/name).is_file()] + [p for p in (ROOT/'assets_lib').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(paths)}


def prepare_textures(bundle, spec_path, output):
    sources, mapping = resolve_sources(bundle, spec_path), texture_view_map(bundle)
    records, inputs = [], []
    output = Path(output)
    for view in bundle['views']:
        view_id = view['id']
        crops = [t for t in bundle['shared_scene']['textures'] if mapping[t['id']] == view_id]
        manifest = read_json(extract(sources[view_id], crops, output/view_id))
        for record in manifest['textures']:
            records.append({**record, 'source_view_id': view_id, 'source_sha256': manifest['source_sha256']})
        inputs.append({'view_id': view_id, 'source_sha256': manifest['source_sha256']})
    path = output/'manifest.json'
    atomic_json(path, {'textures': records, 'sources': inputs})
    return path


def artifact_hashes(folder, relative_paths):
    return {name: file_sha256(folder/name) for name in sorted(set(relative_paths))}


def _artifact_signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def verify_hashes(folder, hashes, cache=None):
    if not hashes:
        raise ValueError('Missing artifact hashes')
    folder = Path(folder).resolve()
    for name, digest in hashes.items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError('Artifact changed or missing: '+name)
        signature = _artifact_signature(path)
        key = str(path)
        cached = cache.get(key) if cache is not None else None
        if cached is not None and cached[0] == signature:
            actual = cached[1]
        else:
            actual = file_sha256(path)
            if cache is not None:
                cache[key] = (signature, actual)
        if actual != digest:
            raise ValueError('Artifact changed or missing: '+name)


def validate_render(folder, bundle):
    meta = read_json(folder/'metadata.json')
    if meta['status'] != 'completed' or meta['geometry_build_count'] != 1:
        raise ValueError('Shared scene render did not complete')
    if meta['camera_count'] != len(bundle['views']) or [v['id'] for v in meta['views']] != [v['id'] for v in bundle['views']]:
        raise ValueError('Missing/mismatched reference view outputs')
    verify_hashes(folder, meta['output_sha256'])
    for view in meta['views']:
        if image_size(folder/view['render']) != (view['width'], view['height']):
            raise ValueError('Rendered image dimensions mismatch')
    if (folder/'scene.blend').stat().st_size < 1000:
        raise ValueError('Blender scene file is empty')
    return meta


def rejected_preflight(directory, spec_path):
    """Read only current, hash-bound blocking preflight reports.

    A stale file or unrelated Blender crash must not trigger a model call.
    Callers remove reports before launching a new Blender attempt; committed
    technical repairs may resume from immutable reports without relaunching.
    """
    reports = {}
    expected = file_sha256(spec_path)
    for name in ('geometry_preflight.json', 'lighting_preflight.json'):
        path = directory / name
        if not path.is_file():
            continue
        report = read_json(path)
        if (not isinstance(report, dict) or report.get('spec_sha256') != expected
                or report.get('status') not in ('passed', 'rejected')):
            raise ValueError('Preflight report is stale or not bound to the current candidate')
        reports[name] = report
    return reports if any(r['status'] == 'rejected' and r.get('errors') for r in reports.values()) else None


def final_spatial_diagnostics(bundle, directory, metadata):
    """Meter saved final PNGs, never tone-fitting preview images.

    These grid diagnostics protect candidate selection from large photometric
    regressions. They are not a reconstruction accuracy or acceptance score.
    """
    import numpy as np
    from PIL import Image
    from reference_tone import compute_spatial_stats, compare_spatial_stats
    sources, renders, inputs = {}, {}, {}
    viewmeta = {item['id']: item for item in metadata['views']}
    def meter(path):
        with Image.open(path) as picture:
            # Bounded working memory for large user photos. This still meters
            # the saved final render, not a separately fitted preview.
            picture = picture.convert('RGB')
            picture.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            return compute_spatial_stats(np.asarray(picture))
    for view in bundle['views']:
        source = Path(view['source_image'])
        render = directory / viewmeta[view['id']]['render']
        sources[view['id']] = meter(source)
        renders[view['id']] = meter(render)
        inputs[view['id']] = {'source_sha256': file_sha256(source), 'render_sha256': file_sha256(render)}
    return {'measurement': 'saved_final_render_png', 'meter_max_edge': 1024, 'image_sha256': inputs,
            'spatial_diagnostics': compare_spatial_stats(sources, renders)}


def prepare_best_repair_source(best, current_iteration, feedback):
    """Make a distinct immutable repair input when rolling back a candidate.

    Earlier repair commits must never be overwritten or replayed as the next
    proposed fix. The current iteration owns a separate snapshot of the best
    render/review, and automatic_repair binds its own outputs to those inputs.
    """
    source = best['directory']
    destination = current_iteration if source == current_iteration else current_iteration/'repair_from_best'
    destination.mkdir(exist_ok=True)
    if destination != source:
        names = set(best['metadata']['outputs']) | {'metadata.json', 'scene_bundle.json'}
        names.update(name for name in ('asset_bindings.json', 'asset_selection.json', 'review.json',
                                      'review_provenance.json') if (source/name).is_file())
        hashes = artifact_hashes(source, names)
        manifest = {'selected_iteration': source.name, 'input_sha256': hashes}
        manifest_path = destination/'repair_source.json'
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise ValueError('Rollback repair source changed')
        for name, digest in hashes.items():
            src, dst = source/name, destination/name
            if not dst.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Repair snapshot path escapes destination')
            if dst.exists():
                if not dst.is_file() or file_sha256(dst) != digest:
                    raise ValueError('Rollback repair snapshot changed: '+name)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
        atomic_json(manifest_path, manifest)
    feedback_path = destination/'candidate_feedback.json'
    if feedback_path.exists() and read_json(feedback_path) != feedback:
        raise ValueError('Candidate selection feedback changed on resume')
    atomic_json(feedback_path, feedback)
    return destination


class AssetSnapshotGuard:
    """Operation-scoped verification of an immutable candidate library.

    Every input's identity is checked on every call; unchanged files reuse
    their verified digest, while changed identities are freshly hashed.
    Selected files are also freshly hashed at render/commit boundaries.
    Resume, index verification and each export pass use new guards.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._verified = {}
        self._signatures = {}

    @staticmethod
    def _reference(reference):
        if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
            raise ValueError('Invalid asset library snapshot reference')
        path, expected = reference['path'], reference['sha256']
        if (not isinstance(path, str) or not Path(path).is_absolute()
                or not isinstance(expected, str) or len(expected) != 64
                or set(expected) - set('0123456789abcdef')):
            raise ValueError('Invalid asset library snapshot reference')
        return path, expected

    @staticmethod
    def _validate_snapshot(snapshot):
        if (not isinstance(snapshot, dict) or snapshot.get('schema_version') != 1
                or not isinstance(snapshot.get('catalog'), str)
                or not Path(snapshot['catalog']).is_absolute()
                or not isinstance(snapshot.get('catalog_sha256'), str)
                or len(snapshot['catalog_sha256']) != 64
                or set(snapshot['catalog_sha256']) - set('0123456789abcdef')
                or not isinstance(snapshot.get('input_sha256'), dict)):
            raise ValueError('Invalid asset library snapshot')
        payload = {name: value for name, value in snapshot.items() if name != 'snapshot_sha256'}
        if snapshot.get('snapshot_sha256') != hashlib.sha256(canonical_bytes(payload)).hexdigest():
            raise ValueError('Invalid asset library snapshot digest')
        catalog = snapshot['catalog']
        digest = snapshot['catalog_sha256']
        inputs = snapshot['input_sha256']
        if inputs.get(catalog) != digest:
            raise ValueError('Invalid asset library catalog fingerprint')
        for source, expected in inputs.items():
            if not isinstance(source, str) or not Path(source).is_absolute():
                raise ValueError('Invalid asset library input path')
            if expected is None:
                continue
            if (not isinstance(expected, str) or len(expected) != 64
                    or set(expected) - set('0123456789abcdef')):
                raise ValueError('Invalid asset library input hash: ' + source)
        return snapshot

    @staticmethod
    def _read_snapshot(snapshot_path, expected_digest):
        from asset_catalog import AssetCatalog
        snapshot_path = AssetCatalog._checked_hash_path(snapshot_path)
        digest, signature = AssetCatalog.stable_file_digest(snapshot_path)
        if digest != expected_digest:
            raise ValueError('Asset library snapshot changed or is missing')
        snapshot = read_json(snapshot_path)
        AssetSnapshotGuard._validate_snapshot(snapshot)
        # Ensure the parsed value came from the bytes covered by the digest.
        if AssetCatalog.stable_file_digest(snapshot_path)[0] != digest:
            raise ValueError('Asset library snapshot changed during verification')
        return snapshot, signature

    @staticmethod
    def _verify_inputs(snapshot, previous=None, force=()):
        """Return identities only after checking all present and missing inputs."""
        from asset_catalog import AssetCatalog
        previous = {} if previous is None else previous
        if not isinstance(previous, dict):
            raise ValueError('Invalid asset library input signatures')
        signatures = {}
        for source, expected in snapshot['input_sha256'].items():
            source_path = Path(source)
            if expected is None:
                # Missing paths have no regular-file stat below, so their
                # entire path still needs its own symlink check.
                source_path = AssetCatalog._checked_hash_path(source_path)
                if source_path.exists():
                    raise ValueError('Previously missing asset library input appeared: ' + source)
                continue
            label = 'catalog' if source == snapshot['catalog'] else 'input'
            try:
                # _stat_signature checks every component for symlinks itself.
                # Repeating that same walk here doubles large-library stats.
                signature = AssetCatalog._stat_signature(source_path)
                if source in force or signature != previous.get(source):
                    digest, signature = AssetCatalog.stable_file_digest(source_path)
                    if digest != expected:
                        raise ValueError('Asset library ' + label + ' changed: ' + source)
            except OSError as exc:
                raise ValueError('Asset library ' + label + ' is unavailable: ' + source) from exc
            signatures[source] = signature
        return signatures

    def seed(self, reference, snapshot, input_stats=None):
        """Seed using freshly pinned identities, still checking current inputs."""
        from asset_catalog import AssetCatalog
        path, expected = self._reference(reference)
        with self._lock:
            snapshot_path = AssetCatalog._checked_hash_path(path)
            loaded, snapshot_signature = self._read_snapshot(snapshot_path, expected)
            if loaded != snapshot:
                raise ValueError('Asset library snapshot changed during preparation')
            # Pinning already hashed large model/texture files. Check all of
            # their identities again and rehash changes; always freshly hash
            # the catalog so stale parsed catalog contents cannot seed a guard.
            input_signatures = self._verify_inputs(loaded, input_stats, force=(loaded['catalog'],))
            key = (path, expected)
            self._verified[key] = copy.deepcopy(loaded)
            self._signatures[key] = (snapshot_signature, input_signatures)

    def verify(self, reference):
        path, expected = self._reference(reference)
        key = (path, expected)
        with self._lock:
            from asset_catalog import AssetCatalog
            snapshot_path = AssetCatalog._checked_hash_path(path)
            snapshot = self._verified.get(key)
            if snapshot is not None:
                snapshot_signature, input_signatures = self._signatures[key]
                try:
                    if AssetCatalog._stat_signature(snapshot_path) != snapshot_signature:
                        snapshot, snapshot_signature = self._read_snapshot(snapshot_path, expected)
                except (OSError, ValueError) as exc:
                    raise ValueError('Asset library snapshot changed or is missing') from exc
            else:
                try:
                    snapshot, snapshot_signature = self._read_snapshot(snapshot_path, expected)
                except (OSError, ValueError) as exc:
                    raise ValueError('Asset library snapshot changed or is missing') from exc
                input_signatures = None
            input_signatures = self._verify_inputs(snapshot, input_signatures)
            self._verified[key] = snapshot
            self._signatures[key] = (snapshot_signature, input_signatures)


def source_check(job, snapshot_guard=None):
    for source in job['sources']:
        if file_sha256(source['path']) != source['sha256']:
            raise ValueError('Source photo changed during processing')
    library = job.get('asset_library', {})
    reference = library.get('input_snapshot')
    if reference is not None:
        (snapshot_guard or AssetSnapshotGuard()).verify(reference)
        return
    # Legacy/manual job records keep their small, scene-local input map.
    for path, digest in library.get('input_sha256', {}).items():
        if digest is None:
            if Path(path).exists() or Path(path).is_symlink():
                raise ValueError('Previously missing asset library input appeared: ' + path)
            continue
        if file_sha256(path) != digest:
            raise ValueError('Asset library input changed during processing')


def catalog_snapshot(catalog, catalog_path):
    """Freeze every candidate file, including missing files, before analysis.

    Repairs may select another asset. Hashing only the first selected pair
    would let later iterations consume a silently changed candidate.
    """
    catalog_path = Path(catalog_path).expanduser().resolve()
    paths = {catalog_path}
    for asset in catalog.assets.values():
        for key in ('model_file', 'collision_file', 'manifest_file'):
            if asset.get(key):
                paths.add(catalog.resolve_path(asset[key], must_exist=False))
    for material in catalog.materials.values():
        for maps in [material.get('maps', {}), *(material.get('slot_materials') or {}).values()]:
            for record in maps.values():
                if record.get('path'):
                    paths.add(catalog.resolve_path(record['path'], must_exist=False))
    hashes, stats = {}, {}
    for path in sorted(paths):
        key = str(path)
        if path == catalog_path:
            digest, signature = catalog.source_sha256, catalog.source_stat_signature
            if catalog._stat_signature(path) != signature:
                raise ValueError('Asset library catalog changed after it was loaded')
        elif path.is_file():
            digest, signature = catalog.stable_file_digest(path)
        else:
            digest, signature = None, None
        hashes[key] = digest
        if signature is not None:
            stats[key] = signature
    return hashes, stats


def prepare_batch_asset_snapshot(args):
    """Pin and verify the full auto-binding library once for this batch run."""
    args._asset_snapshot_guard = AssetSnapshotGuard()
    # Args namespaces are sometimes reused by direct run_job callers. Never
    # let a catalog or snapshot from an earlier invocation survive preparation.
    args._asset_catalog_instance = None
    args._asset_catalog_sha256 = None
    args._asset_snapshot_ref = None
    if not getattr(args, 'asset_auto_bind', False):
        if getattr(args, 'asset_catalog', None):
            from asset_catalog import load_catalog
            args._asset_catalog_instance = load_catalog(args.asset_catalog)
        return
    from asset_catalog import load_catalog
    catalog_path = Path(args.asset_catalog).expanduser().resolve()
    # An args object may outlive a catalog version. Always reload its contents.
    catalog = load_catalog(catalog_path)
    catalog_digest = catalog.source_sha256
    input_hashes, input_stats = catalog_snapshot(catalog, catalog_path)
    if input_hashes[str(catalog_path)] != catalog_digest:
        raise ValueError('Asset library catalog changed while taking snapshot')
    catalog.pin_snapshot_hashes(input_hashes, input_stats)
    args._asset_catalog_sha256 = catalog_digest
    from auto_asset_binding import _hash as canonical_catalog_hash
    catalog._batch_catalog_sha256 = canonical_catalog_hash(catalog.data)
    payload = {'schema_version': 1, 'catalog': str(catalog_path),
               'catalog_sha256': args._asset_catalog_sha256, 'input_sha256': input_hashes}
    payload['snapshot_sha256'] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.output_dir / ASSET_SNAPSHOT_FILE
    if snapshot_path.is_symlink():
        raise ValueError('Asset library snapshot cannot be a symlink')
    if snapshot_path.exists():
        if read_json(snapshot_path) != payload:
            raise ValueError('Asset library changed since this batch was started; use a new output directory')
    else:
        # Do not start a potentially large immutable-library write after the
        # admission floor has been reached.  Existing snapshots remain intact.
        check_scene_start_disk_space(args)
        atomic_json(snapshot_path, payload)
    reference = {'path': str(snapshot_path.resolve()), 'sha256': file_sha256(snapshot_path)}
    args._asset_catalog_instance = catalog
    args._asset_snapshot_ref = reference
    args._asset_snapshot_guard.seed(reference, payload, input_stats)


def verify_selected_asset_inputs(catalog, bindings_path):
    """Recheck only bound files immediately around Blender use."""
    if catalog is None or bindings_path is None:
        return
    bindings = read_json(bindings_path)
    if not isinstance(bindings, dict) or not isinstance(bindings.get('instances'), list):
        raise ValueError('Invalid asset bindings')
    for binding in bindings['instances']:
        asset = catalog.get_asset(binding['asset_id'])
        material = catalog.get_material_variant(binding['material_variant_id'])
        catalog.verify_hash(asset['asset_id'], force=True)
        catalog.verify_material_hashes(binding['material_variant_id'], force=True)
        collision_file = asset.get('collision_file')
        if collision_file:
            collision = catalog.resolve_path(collision_file)
            if not asset.get('collision_hash') or not catalog.verify_file_hash(collision, asset['collision_hash'], force=True):
                raise ValueError('Selected asset collision file changed: ' + asset['asset_id'])
        manifest_file = asset.get('manifest_file')
        if manifest_file:
            manifest = catalog.resolve_path(manifest_file)
            expected = (asset.get('audit_provenance') or {}).get('manifest_sha256')
            if not expected or not catalog.verify_file_hash(manifest, expected, force=True):
                raise ValueError('Selected asset manifest changed: ' + asset['asset_id'])


def prepare_asset_selection(bundle, catalog, iteration, options, catalog_digest, preview=False):
    """Commit deterministic iteration-local bindings and their selection evidence."""
    from auto_asset_binding import generate_bindings
    bindings, report = generate_bindings(bundle, catalog, options=options, allow_preview=preview)
    report['catalog_sha256'] = catalog_digest
    payloads = {'asset_bindings.json': bindings, 'asset_selection.json': report}
    for name, payload in payloads.items():
        path = iteration / name
        if path.exists() and read_json(path) != payload:
            raise ValueError('Automatic asset selection differs from recorded inputs: ' + name)
        atomic_json(path, payload)
    return iteration / 'asset_bindings.json', artifact_hashes(iteration, payloads)


def run_job(group, args, hashes):
    """Run a job in the active batch scope or create an isolated one.

    The coordinator owns ``.batch.lock`` for its full lifetime and marks that
    scope before submitting workers. A direct caller has no such guarantee:
    acquire the same output lock, reload the asset catalog/snapshot, and use a
    shallow args copy so a later invocation cannot inherit stale cache fields.
    """
    if getattr(args, '_asset_batch_scope_active', False) or not getattr(args, 'asset_catalog', None):
        return _run_job_scoped(group, args, hashes)

    import copy
    scoped_args = copy.copy(args)
    scoped_args.output_dir = Path(args.output_dir).expanduser().resolve()
    scoped_args.output_dir.mkdir(parents=True, exist_ok=True)
    scoped_args._asset_scope_preparation_error = None
    try:
        with process_lock(scoped_args.output_dir / '.batch.lock'):
            try:
                prepare_batch_asset_snapshot(scoped_args)
            except Exception as exc:
                # Let the normal per-scene failure path write last_error.json,
                # which takes precedence over a stale accepted result on resume.
                scoped_args._asset_scope_preparation_error = exc
            scoped_args._asset_standalone_scope = True
            return _run_job_scoped(group, scoped_args, hashes)
    except OutputInUseError as exc:
        folder = scoped_args.output_dir / group['scene_id']
        return {'scene_id': group['scene_id'], 'status': 'locked', 'directory': str(folder),
                'error': str(exc), 'finished_at': utc_now()}


def _run_job_scoped(group, args, hashes):
    import vision_multiview as vision
    started = time.monotonic()
    folder = args.output_dir/group['scene_id']
    result = {'scene_id': group['scene_id'], 'status': 'failed', 'directory': str(folder), 'started_at': utc_now()}
    best_candidate = None
    attempted_iteration = None
    fd = None
    advisory = None
    budget_session = None
    pipeline_options = getattr(args, 'pipeline_options', {})
    current_stage = 'initializing'
    def progress(event, **extra):
        nonlocal current_stage
        current_stage = event
        if 'stage' in extra:
            extra['analysis_stage'] = extra.pop('stage')
        atomic_json(folder/'progress.json', {'stage': event, 'at': utc_now(), **extra})
        print(json.dumps({'scene_id': group['scene_id'], 'stage': event, **extra}, ensure_ascii=False), flush=True)
    try:
        check_interrupted(args)
        # A committed result is reusable even when no new bytes can be
        # admitted.  For every other scene, reject before creating its folder
        # or writing a job/lock record; the coordinator performs the same
        # check before enqueueing.
        reusable = False
        if (folder/'result.json').is_file():
            try:
                reusable = read_json(folder/'result.json').get('status') in (
                    'accepted', 'needs_review', 'generated_unreviewed')
            except (OSError, ValueError, AttributeError):
                reusable = False
        if not reusable:
            try:
                check_scene_start_disk_space(args)
            except DiskSpacePause as exc:
                result.update(status='paused', failed_stage='disk_space_preflight',
                              stop_reason='disk_space_guard', error=str(exc), finished_at=utc_now())
                return result
        stagnation_patience = pipeline_options.get('repair_stagnation_patience', 2)
        if type(stagnation_patience) is not int or not 0 <= stagnation_patience <= 5:
            raise ValueError('pipeline.repair_stagnation_patience must be an integer between 0 and 5')
        if folder.is_symlink():
            raise ValueError('Scene output directory must not be a symlink')
        folder.mkdir(parents=True, exist_ok=True)
        advisory = process_lock(folder/'.process.lock')
        try:
            advisory.__enter__()
        except OutputInUseError:
            advisory = None
            return {**result, 'status': 'locked', 'error': 'Another live process owns this scene'}
        try:
            fd = os.open(folder/'.lock', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                stale = read_json(folder/'.lock')
            except (OSError, ValueError):
                stale = None
            # Only this protocol guarantees that every writer first owns the
            # advisory lock. Legacy markers cannot be reclaimed by PID checks.
            if not isinstance(stale, dict) or stale.get('lock_protocol') != LOCK_PROTOCOL:
                return {**result, 'status': 'locked', 'error': 'Legacy or unknown lock; inspect its owner before removing .lock'}
            (folder/'.lock').unlink()
            fd = os.open(folder/'.lock', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        os.write(fd, canonical_bytes({'lock_protocol': LOCK_PROTOCOL, 'pid': os.getpid(), 'at': utc_now()}))
        scope_error = getattr(args, '_asset_scope_preparation_error', None)
        if scope_error is not None:
            raise ValueError(str(scope_error)) from scope_error
        job = {'version': 4, 'scene_id': group['scene_id'],
               'sources': [{'path': str(p), 'sha256': file_sha256(p), 'dimensions': list(image_size(p))} for p in group['images']],
               'pipeline': hashes, 'options': {k: getattr(args,k) for k in ('max_edge','samples','device','max_repairs','effort','vision_timeout','render_timeout','no_review')}}
        asset_catalog_path = getattr(args, 'asset_catalog', None)
        asset_bindings_path = getattr(args, 'asset_bindings', None)
        auto_bind = bool(getattr(args, 'asset_auto_bind', False))
        catalog = None
        if auto_bind and (not asset_catalog_path or asset_bindings_path):
            raise ValueError('Automatic asset binding requires a catalog and no manual bindings')
        if not auto_bind and bool(asset_catalog_path) != bool(asset_bindings_path):
            raise ValueError('asset_catalog and asset_bindings must be supplied together')
        if auto_bind:
            from auto_asset_binding import normalize_options
            catalog = getattr(args, '_asset_catalog_instance', None)
            if catalog is None:
                with process_lock(args.output_dir/'.batch.lock'):
                    prepare_batch_asset_snapshot(args)
                catalog = args._asset_catalog_instance
            selection_options = normalize_options(getattr(args, 'asset_selection_options', {}))
            snapshot_reference = getattr(args, '_asset_snapshot_ref', None)
            if snapshot_reference is None:
                raise ValueError('Automatic asset binding requires a batch asset snapshot')
            job['asset_library'] = {'catalog': str(Path(asset_catalog_path).resolve()),
                'catalog_sha256': args._asset_catalog_sha256, 'mode': 'automatic',
                'selection': selection_options, 'preview': bool(getattr(args, 'asset_preview', False)),
                'input_snapshot': snapshot_reference}
        elif asset_catalog_path:
            from asset_catalog import load_catalog
            catalog = getattr(args, '_asset_catalog_instance', None) or load_catalog(asset_catalog_path)
            bindings = read_json(asset_bindings_path)
            asset_paths = {Path(asset_catalog_path).resolve(), Path(asset_bindings_path).resolve()}
            for binding in bindings['instances']:
                asset = catalog.get_asset(binding['asset_id'])
                material = catalog.get_material_variant(binding['material_variant_id'])
                catalog.verify_hash(asset['asset_id'])
                catalog.verify_material_hashes(binding['material_variant_id'])
                if not getattr(args, 'asset_preview', False):
                    ready, reasons = catalog.production_readiness(asset)
                    if not ready:
                        raise ValueError('Asset not production ready: ' + ', '.join(reasons))
                asset_paths.add(catalog.resolve_path(asset['model_file']))
                asset_paths.update(catalog.resolve_path(m['path']) for m in material['maps'].values() if m.get('path'))
                for slot_maps in (material.get('slot_materials') or {}).values():
                    asset_paths.update(catalog.resolve_path(m['path']) for m in slot_maps.values() if m.get('path'))
                if asset.get('collision_file'):
                    asset_paths.add(catalog.resolve_path(asset['collision_file']))
            job['asset_library'] = {'catalog': str(Path(asset_catalog_path).resolve()),
                                    'catalog_sha256': file_sha256(asset_catalog_path),
                                    'bindings': str(Path(asset_bindings_path).resolve()),
                                    'bindings_sha256': file_sha256(asset_bindings_path),
                                    'preview': bool(getattr(args, 'asset_preview', False)),
                                    'input_sha256': {str(p): file_sha256(p) for p in sorted(asset_paths)}}
        job['options'].update(appearance_policy=getattr(args,'appearance_policy','photo_reconstruction'),
                              tone_match=getattr(args,'tone_match','reference'))
        job['options']['pipeline'] = pipeline_options
        model_configs = getattr(args, 'model_configs', None)
        job['models'] = model_configs or {}
        budget_policy = getattr(args, 'scene_budget_policy', None)
        if budget_policy is not None:
            from scene_budget import normalize_policy
            job['budget'] = normalize_policy(budget_policy)
            if any(job['budget'].values()) and not model_configs:
                raise ValueError('Scene budgets require configured model APIs')
        if (folder/'job.json').exists():
            if read_json(folder/'job.json') != job:
                raise ValueError('Input, code or options changed; use a new output directory')
        elif any(p.name not in {'.lock', '.process.lock', 'last_error.json'} for p in folder.iterdir()):
            raise ValueError('Output contains unrelated files; use a new output directory')
        else:
            atomic_json(folder/'job.json', job)
        source_check(job, getattr(args, '_asset_snapshot_guard', None))
        if (folder/'result.json').exists():
            previous = read_json(folder/'result.json')
            if previous['status'] in ('accepted','needs_review','generated_unreviewed'):
                verify_hashes(folder, previous['artifacts'], cache=getattr(args, '_artifact_digest_cache', None))
                if catalog is not None:
                    # Snapshot-backed jobs were fully checked by source_check
                    # immediately above, including every selected and
                    # unselected model, texture, collision and manifest input.
                    # Their committed render artifacts are verified separately
                    # here. Rehashing the same selected assets for every
                    # accepted scene makes an asset shared by thousands of
                    # scenes an avoidable repeated disk scan. Legacy/manual
                    # jobs lack that complete inventory (notably the manifest
                    # hash), so preserve their explicit selected-file check.
                    if not job.get('asset_library', {}).get('input_snapshot'):
                        previous_bindings = (folder/previous['iteration']/'asset_bindings.json') if auto_bind else asset_bindings_path
                        verify_selected_asset_inputs(catalog, previous_bindings)
                # A later failed invocation may have checked incompatible
                # options without touching this valid committed result.
                if 'last_error.json' not in previous['artifacts']:
                    (folder/'last_error.json').unlink(missing_ok=True)
                return {**previous, 'resumed': True}
        # A committed scene can be reused without consuming any new disk
        # space.  Perform the admission check only after that resume path so a
        # batch can still reconcile completed scenes when the volume is tight.
        try:
            check_scene_start_disk_space(args)
        except DiskSpacePause as exc:
            result.update(status='paused', failed_stage='disk_space_preflight',
                          stop_reason='disk_space_guard', error=str(exc), finished_at=utc_now())
            return result
        from scene_budget import scene_budget_session
        session = scene_budget_session(folder, budget_policy)
        session.__enter__()
        budget_session = session
        views = []
        for index, source in enumerate(job['sources']):
            check_interrupted(args)
            identifier = f'view_{index+1:02d}'
            manifest = read_json(extract(source['path'], [], folder/'sources'/identifier))
            views.append({'id': identifier, 'source_image': manifest['reference_image'], 'width': manifest['dimensions'][0], 'height': manifest['dimensions'][1]})
        initial = folder/'initial_bundle.json'
        # Provenance is written last and commits the payload. A payload without
        # provenance can be left by interruption and must be rebuilt/reassembled.
        if (folder/'analysis_provenance.json').exists():
            bundle = validate(read_json(initial))
            prov = read_json(folder/'analysis_provenance.json')
            if prov['bundle_sha256'] != file_sha256(initial):
                raise ValueError('Initial analysis artifact changed')
        else:
            check_interrupted(args)
            progress('joint_photo_analysis', images=len(views))
            extra = {'raw_cache':args.reuse_analysis_from} if getattr(args,'reuse_analysis_from',None) else {}
            if model_configs:
                extra['model_config'] = model_configs['analysis']
            if pipeline_options.get('scene_wire_format') == 'library_layout_v1':
                if extra.get('raw_cache'):
                    raise ValueError('library_layout_v1 resumes its own committed analysis; cross-directory raw_cache is unsupported')
                import library_layout
                bundle = library_layout.analyze(group['scene_id'], views, initial, catalog=catalog,
                    timeout=args.vision_timeout, effort=args.effort, model_config=model_configs['analysis'])
            elif pipeline_options.get('analysis_mode') == 'staged':
                if not model_configs:
                    raise ValueError('Staged analysis requires configured model APIs')
                if extra.get('raw_cache'):
                    raise ValueError('Staged analysis resumes its own stage cache; raw_cache is monolithic only')
                import staged_analysis
                bundle = staged_analysis.analyze(group['scene_id'], views, initial,
                    timeout=args.vision_timeout, effort=args.effort, model_config=model_configs['analysis'],
                    options=pipeline_options.get('staged'), progress=progress)
            else:
                if pipeline_options.get('scene_wire_format', 'canonical') != 'canonical':
                    extra['wire_format'] = pipeline_options['scene_wire_format']
                bundle = vision.analyze(group['scene_id'], views, initial, timeout=args.vision_timeout, effort=args.effort, **extra)
        library_layout_data = None
        if pipeline_options.get('scene_wire_format') == 'library_layout_v1':
            layout_path = folder/'initial_layout.json'
            provenance = read_json(folder/'analysis_provenance.json')
            if not layout_path.is_file() or provenance.get('layout_sha256') != file_sha256(layout_path):
                raise ValueError('Library layout artifact changed or is incomplete')
            library_layout_data = read_json(layout_path)
        # A standalone job is its own immutable-library interval. Rehash the
        # complete inventory once after analysis so an unselected candidate
        # modified while the model was working cannot be silently accepted.
        # Coordinator workers share the batch-start scan and only force-check
        # selected dependencies at their Blender boundaries.
        if auto_bind and getattr(args, '_asset_standalone_scope', False):
            source_check(job, AssetSnapshotGuard())
        history, technical_history, attempted_bundles = [], [], []
        candidate_history = []
        consecutive_nonimproving = 0
        for index in range(args.max_repairs+1):
            check_interrupted(args)
            source_check(job, getattr(args, '_asset_snapshot_guard', None))
            attempted_bundles.append(bundle)
            iteration = folder/f'iteration_{index:02d}'
            attempted_iteration = iteration.name
            iteration.mkdir(exist_ok=True)
            library_selection_payloads = None
            if library_layout_data is not None:
                from library_selection import select_and_place, LibrarySelectionError
                progress('automatic_asset_selection', iteration=index)
                try:
                    bundle, library_bindings, library_report = select_and_place(
                        bundle, library_layout_data, catalog, options=selection_options)
                except LibrarySelectionError as error:
                    atomic_json(iteration/'asset_selection.json', error.report)
                    raise
                library_selection_payloads = {'asset_bindings.json': library_bindings,
                                              'asset_selection.json': library_report}
            spec_path = iteration/'scene_bundle.json'
            if spec_path.exists():
                if read_json(spec_path) != bundle:
                    raise ValueError('Iteration differs from recorded analysis/review')
            else:
                atomic_json(spec_path, bundle)
            selection_hashes = {}
            iteration_bindings_path = asset_bindings_path
            if library_selection_payloads is not None:
                for name, payload in library_selection_payloads.items():
                    path = iteration/name
                    if path.exists() and read_json(path) != payload:
                        raise ValueError('Library selection differs from recorded inputs: '+name)
                    atomic_json(path,payload)
                iteration_bindings_path = iteration/'asset_bindings.json'
                selection_hashes = artifact_hashes(iteration,library_selection_payloads)
            elif auto_bind:
                progress('automatic_asset_selection', iteration=index)
                iteration_bindings_path, selection_hashes = prepare_asset_selection(
                    bundle, catalog, iteration, selection_options, job['asset_library']['catalog_sha256'],
                    preview=job['asset_library']['preview'])
            fingerprint = hashlib.sha256(canonical_bytes({'bundle':bundle,'job':job,
                'asset_selection': selection_hashes})).hexdigest()
            state_path = iteration/'render_state.json'
            def preflight_candidate(diagnostics):
                """Consume this iteration's one shared repair opportunity."""
                if (index == args.max_repairs or not model_configs
                        or pipeline_options.get('repair_mode', 'typed') != 'typed'):
                    raise RuntimeError('Blocking preflight failed; no configured automatic repair budget remains')
                check_interrupted(args)
                progress('automatic_technical_repair', iteration=index)
                import automatic_repair
                candidate = automatic_repair.technical_repair(
                    bundle, iteration, diagnostics, model_configs['repair'],
                    timeout=args.vision_timeout, effort=args.effort,
                    validation_retries=pipeline_options.get('validation_retries', 1))
                if candidate == bundle:
                    raise RuntimeError('Technical repair produced no change for a blocking preflight error')
                technical_history.append({'iteration': index, 'status': 'repaired_candidate',
                    'diagnostics': {name: file_sha256(iteration/name) for name in diagnostics},
                    'repair_provenance_sha256': file_sha256(iteration/'technical_repair_provenance.json'),
                    'candidate_requires_new_render_and_review': True})
                atomic_json(folder/'technical_history.json', technical_history)
                result['technical_history'] = technical_history
                return validate(candidate)
            if (iteration/'technical_repair_provenance.json').is_file():
                if state_path.exists():
                    raise ValueError('Iteration contains conflicting render and preflight-repair commits')
                diagnostics = rejected_preflight(iteration, spec_path)
                if not diagnostics:
                    raise ValueError('Committed technical repair lacks current blocking preflight evidence')
                bundle = preflight_candidate(diagnostics)
                continue
            if state_path.exists():
                state = read_json(state_path)
                if state['fingerprint'] != fingerprint:
                    raise ValueError('Render inputs changed')
                verify_hashes(iteration, state['artifacts'])
                verify_selected_asset_inputs(catalog, iteration_bindings_path)
                metadata = validate_render(iteration,bundle)
            else:
                manifest = prepare_textures(bundle, spec_path, iteration/'textures')
                progress('queued_for_blender', iteration=index)
                try:
                    with render_slot(lambda: check_interrupted(args)) as queue_seconds:
                        check_interrupted(args)
                        progress('blender_render', iteration=index, render_queue_seconds=round(queue_seconds, 3))
                        verify_selected_asset_inputs(catalog, iteration_bindings_path)
                        # Remove evidence from prior uncommitted attempts before
                        # launching: a new unrelated crash cannot inherit it.
                        for name in ('geometry_preflight.json', 'lighting_preflight.json'):
                            (iteration/name).unlink(missing_ok=True)
                        check_interrupted(args)
                        command = [args.blender,'-b','--factory-startup','--python-exit-code','1','--python',str(ROOT/'generate_multiview.py'),'--',
                                   '--spec',str(spec_path),'--output-dir',str(iteration),'--texture-manifest',str(manifest),
                                   '--max-edge',str(args.max_edge),'--samples',str(args.samples),'--device',args.device,
                                   '--appearance-policy',job['options']['appearance_policy'],'--tone-match',job['options']['tone_match']]
                        if asset_catalog_path:
                            command.extend(['--asset-catalog',str(Path(asset_catalog_path).resolve()),
                                            '--asset-bindings',str(Path(iteration_bindings_path).resolve())])
                            if getattr(args, 'asset_preview', False):
                                command.append('--asset-preview')
                        check_interrupted(args)
                        run_stage(command,iteration/'blender.log',args.render_timeout)
                        verify_selected_asset_inputs(catalog, iteration_bindings_path)
                except JobInterrupted:
                    raise
                except Exception:
                    diagnostics = rejected_preflight(iteration, spec_path)
                    if not diagnostics:
                        raise
                    bundle = preflight_candidate(diagnostics)
                    continue
                metadata = validate_render(iteration,bundle)
                atomic_json(state_path, {'fingerprint':fingerprint, 'artifacts':artifact_hashes(iteration,
                    metadata['outputs']+['metadata.json','scene_bundle.json']+list(selection_hashes))})
            result['iteration'] = iteration.name
            result['views'] = metadata['views']
            # A previous iteration's scores never certify newly rendered geometry.
            for key in ('gate', 'summary', 'final_review_iteration'):
                result.pop(key, None)
            if args.no_review:
                result['status'] = 'generated_unreviewed'
                break
            check_interrupted(args)
            progress('joint_reference_review', iteration=index, images=len(views)*2)
            # The reviewer receives all originals paired with all corresponding renders.
            if (iteration/'review_provenance.json').exists():
                report = vision.load_review(iteration,bundle)
            else:
                extra = {'model_config': model_configs['review']} if model_configs else {}
                typed_repair = bool(model_configs and pipeline_options.get('repair_mode', 'typed') == 'typed')
                if typed_repair:
                    extra['validation_retries'] = pipeline_options.get('validation_retries', 1)
                report = vision.review(bundle,iteration,timeout=args.vision_timeout,effort=args.effort,
                    final_only=(typed_repair or index==args.max_repairs),**extra)
            gate = vision.quality_gate(report,metadata)
            if job.get('asset_library', {}).get('preview'):
                gate['accepted'] = False
                gate.setdefault('reasons', []).append('draft_asset_preview_only')
            photometry = final_spatial_diagnostics(bundle, iteration, metadata)
            atomic_json(iteration/'final_spatial_diagnostics.json', photometry)
            measured_metadata = {**metadata, 'final_spatial_diagnostics': photometry}
            challenger = {'directory': iteration, 'bundle': bundle, 'metadata': metadata,
                          'report': report, 'gate': gate,
                          'metrics': candidate_metrics(report, gate, measured_metadata)}
            best_candidate, decision = select_candidate(best_candidate, challenger, iteration=iteration.name)
            # Rebuild this count from verified reviews on every run, including
            # resume. Technical preflight attempts are unreviewed and do not
            # count as a visual regression or reset an existing streak.
            consecutive_nonimproving = 0 if decision['selected'] else consecutive_nonimproving + 1
            decision['consecutive_nonimproving'] = consecutive_nonimproving
            decision['selected_iteration'] = best_candidate['directory'].name
            decision['challenger_metrics'] = challenger['metrics']
            decision['selected_metrics'] = best_candidate['metrics']
            decision['evidence_sha256'] = artifact_hashes(iteration,
                ['scene_bundle.json', 'metadata.json', 'final_spatial_diagnostics.json']
                + [name for name in ('review.json', 'review_provenance.json') if (iteration/name).is_file()])
            candidate_history.append(decision)
            selection_record = {'schema_version': 1,
                'method': 'independent_review_with_final_png_photometric_regression_guards',
                'selected_iteration': best_candidate['directory'].name, 'candidates': candidate_history,
                'repair_stagnation': {'patience': stagnation_patience,
                                      'consecutive_nonimproving': consecutive_nonimproving},
                'limitations': 'Heuristic candidate ranking; acceptance still requires an independent final review gate.'}
            atomic_json(folder/'candidate_selection.json', selection_record)
            history.append({'iteration':index,'summary':report['summary'],'scores':report['scores'],'gate':gate,'verdict':report['verdict'],
                            'selected_as_best': decision['selected'], 'selection_reason': decision['reason']})
            atomic_json(folder/'history.json',history)
            best_report = best_candidate['report']
            result.update({'history':history,'summary':best_report['summary'],'gate':best_candidate['gate'],
                           'iteration':best_candidate['directory'].name, 'views':best_candidate['metadata']['views'],
                           'final_review_iteration': best_candidate['directory'].name,
                           'candidate_selection': {'selected_iteration': best_candidate['directory'].name,
                                                   'candidate_count': len(candidate_history),
                                                   'consecutive_nonimproving': consecutive_nonimproving}})
            if gate['accepted']:
                result['status']='accepted'
                break
            if index == args.max_repairs or best_report['verdict'] != 'repair':
                result['status']='needs_review'
                break
            if stagnation_patience and consecutive_nonimproving >= stagnation_patience:
                # Stop before buying another repair/render/review cycle. The
                # independent gate stays unchanged and the best actual render
                # remains selected; a budget stop can never imply acceptance.
                result['status'] = 'needs_review'
                result['stop_reason'] = 'repair_stagnation'
                break
            if model_configs and pipeline_options.get('repair_mode', 'typed') == 'typed':
                check_interrupted(args)
                progress('automatic_visual_repair', iteration=index)
                import automatic_repair
                repair_directory = prepare_best_repair_source(best_candidate, iteration, selection_record)
                revised = automatic_repair.repair(best_candidate['bundle'], repair_directory, best_report,
                    model_config=model_configs['repair'], timeout=args.vision_timeout, effort=args.effort,
                    validation_retries=pipeline_options.get('validation_retries', 1))
            else:
                revised = best_report['revised_bundle']
            if revised == best_candidate['bundle']:
                result['status'] = 'needs_review'
                result['stop_reason'] = 'repair_produced_no_change'
                break
            # A legacy full-bundle review contains a fixed proposal. If it
            # loses selection, rolling back also restores that old proposal.
            # Stop before repeating any attempted scene; keep the best actual
            # render and its review without spending the remaining budget.
            if any(revised == old for old in attempted_bundles):
                result['status'] = 'needs_review'
                result['stop_reason'] = 'repeated_repair'
                break
            bundle=validate(revised)
        source_check(job, getattr(args, '_asset_snapshot_guard', None))
        if catalog is not None and result.get('iteration'):
            final_bindings = (folder/result['iteration']/'asset_bindings.json') if auto_bind else asset_bindings_path
            verify_selected_asset_inputs(catalog, final_bindings)
        result['elapsed_seconds']=round(time.monotonic()-started,2)
        result['method']='automatic grouped photo analysis; shared geometry; matched multiview review'
        (folder/'last_error.json').unlink(missing_ok=True)
        result['artifacts']=artifact_hashes(folder,[str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name not in JOB_CONTROL_FILES and p.suffix != '.log'])
        atomic_json(folder/'result.json',result)
        progress('complete',status=result['status'])
    except Exception as exc:
        if attempted_iteration is not None:
            result['failed_iteration'] = attempted_iteration
        if best_candidate is not None:
            # A later failed review/render cannot displace the last reviewed
            # output. The attempt is still failed, never accepted by rollback.
            result.update(iteration=best_candidate['directory'].name,
                          views=best_candidate['metadata']['views'], summary=best_candidate['report']['summary'],
                          final_review_iteration=best_candidate['directory'].name)
        result['error']=str(exc)
        if getattr(exc, 'code', None) in ('scene_budget_exceeded', 'invalid_scene_budget',
                'batch_budget_exceeded', 'invalid_batch_budget', 'batch_circuit_open', 'batch_stopped'):
            result['stop_reason'] = exc.code
        result['failed_stage']=current_stage
        result['elapsed_seconds']=round(time.monotonic()-started,2)
        if isinstance(exc, JobInterrupted):
            result['status'] = 'interrupted'
            result['gate'] = {'accepted': False, 'reasons': ['batch_interrupted']}
        if result.get('iteration'):
            # Preserve the last complete .blend/render, but never certify a failed cycle.
            if not isinstance(exc, JobInterrupted):
                result['status']='review_failed' if current_stage in ('joint_reference_review','automatic_visual_repair') else 'failed'
            result['gate']={'accepted':False,'reasons':['pipeline_error:'+current_stage]}
            if result.get('stop_reason') == 'scene_budget_exceeded':
                result['status'] = 'needs_review'
                result['gate'] = {'accepted': False, 'reasons': ['scene_budget_exceeded']}
            result['artifacts']=artifact_hashes(folder,[str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name not in JOB_CONTROL_FILES | {'last_error.json'} and p.suffix != '.log'])
            atomic_json(folder/'result.json',result)
        if fd is not None:
            atomic_json(folder/'last_error.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        if budget_session is not None:
            budget_session.__exit__(None, None, None)
        if fd is not None:
            os.close(fd)
            (folder/'.lock').unlink(missing_ok=True)
        if advisory is not None:
            advisory.__exit__(None, None, None)
    return result


def _record_failure(group, args, exc, stage='coordinator'):
    return {'scene_id': group['scene_id'], 'directory': str(args.output_dir/group['scene_id']),
            'status': 'failed', 'failed_stage': stage, 'error': str(exc), 'finished_at': utc_now()}


def reconcile_groups(directory, groups):
    """Append independent scenes without losing earlier groups or changing IDs."""
    path = Path(directory)/'groups.json'
    records = {}
    if path.exists():
        data = read_json(path)
        if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('scenes'), list):
            raise ValueError('Existing groups.json is invalid; preserve it and inspect before continuing')
        for record in data['scenes']:
            if (not isinstance(record, dict) or set(record) != {'scene_id', 'images'}
                    or not isinstance(record.get('scene_id'), str) or not SAFE_ID.fullmatch(record['scene_id'])
                    or record['scene_id'] in records or not isinstance(record.get('images'), list)
                    or not 1 <= len(record['images']) <= MAX_VIEWS
                    or not all(isinstance(value, str) and value and Path(value).is_absolute() for value in record['images'])):
                raise ValueError('Existing groups.json contains invalid scene identities')
            # Historical photos can be offline while another scene is added.
            # They remain indexed; their accepted gate is validated separately.
            records[record['scene_id']] = {'scene_id': record['scene_id'], 'images': [Path(value).resolve() for value in record['images']]}
    for group in groups:
        previous = records.get(group['scene_id'])
        if previous and previous['images'] != group['images']:
            raise ValueError('Existing scene_id has different input photos; use a new scene_id or output directory: ' + group['scene_id'])
        records[group['scene_id']] = group
    return list(records.values())


def recorded_results(directory, snapshot_guard=None, artifact_cache=None, skip_scene_ids=()):
    """Recover outcomes after a coordinator crash without trusting accepted.json.

    A newer last_error marks the latest attempt. Equal timestamps cannot prove
    which attempt committed last, so the failure marker wins that ambiguity.
    Terminal artifacts must still match before an accepted record can survive
    index reconciliation.
    """
    snapshot_guard = snapshot_guard or AssetSnapshotGuard()
    # A coordinator invocation revalidates requested scene commits in
    # ``run_job`` immediately before returning them.  Do not decode and hash
    # the same full result here only to discard it before the first checkpoint.
    # Unrequested historical scenes still go through the complete recovery
    # path below and remain in the compatibility indexes.
    skip_scene_ids = frozenset(skip_scene_ids or ())
    directory = Path(directory)
    records = {}
    snapshot_run_id = None
    summary = directory/'summary.json'
    if summary.exists():
        try:
            data = read_json(summary)
            snapshot_run_id = data.get('run_id')
            for result in data.get('scenes', []):
                if (isinstance(result, dict) and isinstance(result.get('scene_id'), str)
                        and SAFE_ID.fullmatch(result['scene_id']) and result['scene_id'] not in skip_scene_ids):
                    records[result['scene_id']] = result
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            atomic_json(directory/'index_recovery.json', {'at': utc_now(), 'error': str(exc),
                'action': 'Rebuilding summary and accepted index from per-scene commits'})
    # Events cover a coordinator crash between scene commit and the next
    # snapshot. Per-scene result.json remains authoritative below.
    for result in read_results(directory, snapshot_run_id):
        if (isinstance(result.get('scene_id'), str) and SAFE_ID.fullmatch(result['scene_id'])
                and result['scene_id'] not in skip_scene_ids):
            records[result['scene_id']] = result
    for folder in sorted(directory.iterdir()):
        if not folder.is_dir() or folder.is_symlink() or not SAFE_ID.fullmatch(folder.name):
            continue
        if folder.name in skip_scene_ids:
            continue
        candidates = [folder/name for name in ('result.json', 'last_error.json') if (folder/name).is_file()]
        if not candidates:
            continue
        # Old records have no shared monotonic attempt identifier. Coarse or
        # preserved timestamps must not restore an accepted result over an
        # equally recent failure; successful resume explicitly clears it.
        latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns,
                                                   path.name == 'last_error.json'))
        try:
            result = read_json(latest)
            if result.get('scene_id') != folder.name:
                raise ValueError('Recorded scene identity mismatch')
            records[folder.name] = result
        except (OSError, ValueError, AttributeError) as exc:
            records[folder.name] = {'scene_id': folder.name, 'directory': str(folder), 'status': 'failed',
                                    'failed_stage': 'index_validation', 'error': str(exc)}
    for scene_id, result in list(records.items()):
        if result.get('status') != 'accepted':
            continue
        try:
            folder = directory/scene_id
            if folder.is_symlink() or not folder.is_dir():
                raise ValueError('Accepted scene directory is unavailable')
            if (folder/'.lock').exists():
                raise ValueError('Scene has a running or interrupted attempt; resume it before indexing acceptance')
            committed = read_json(folder/'result.json')
            if committed != result:
                raise ValueError('Accepted scene index differs from its per-scene result commit')
            if result.get('gate', {}).get('accepted') is not True or result.get('final_review_iteration') != result.get('iteration'):
                raise ValueError('Accepted scene is missing final review evidence')
            verify_hashes(folder, result.get('artifacts'), cache=artifact_cache)
            job = read_json(folder/'job.json')
            if job.get('asset_library', {}).get('preview'):
                raise ValueError('Draft preview cannot enter the accepted dataset')
            source_check(job, snapshot_guard)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            records[scene_id] = {**result, 'status': 'failed', 'failed_stage': 'index_validation', 'error': str(exc),
                                 'gate': {'accepted': False, 'reasons': ['artifact_validation_failed']}}
    return records


def _coordinate(groups, args, hashes):
    """Run at most workers futures with incremental completion checkpoints.

    The kernel lock guards groups/index/checkpoints against concurrent runs.
    A first interrupt stops scheduling and lets in-flight work finish its
    current API/render call; committed stages are available on the next run.
    Full summary/accepted snapshots are written at start and termination.
    During execution, batch_events.jsonl holds outcomes and run_state.json
    holds counts; neither rewrites the growing result set or scene ID lists.
    """
    # Resolve before touching the output tree. Shards may opt into one shared
    # ledger root for cross-process budget and provider quota accounting.
    args._batch_ledger_root = _resolve_batch_ledger_root(args)
    min_free_disk_bytes = getattr(args, 'min_free_disk_bytes', 0)
    if (type(min_free_disk_bytes) is not int or not 0 <= min_free_disk_bytes <= 10**15):
        raise ValueError('min_free_disk_bytes must be an integer between 0 and 1000000000000000')
    # Reject before creating the output directory or touching any existing
    # group/index snapshots.  Resolve a missing output path through its
    # nearest existing parent to inspect the destination volume.
    try:
        check_minimum_free_disk(_disk_probe_path(args.output_dir), min_free_disk_bytes)
        if args._batch_ledger_root != args.output_dir:
            check_minimum_free_disk(_disk_probe_path(args._batch_ledger_root), min_free_disk_bytes)
    except DiskSpacePause as exc:
        return publish_disk_space_pause(exc)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if min_free_disk_bytes:
            return publish_disk_space_pause(DiskSpacePause(min_free_disk_bytes, cause=exc))
        raise
    from batch_guard import batch_session, current_stop
    args.stop_event = threading.Event()
    guard = (batch_session(args._batch_ledger_root, getattr(args, 'batch_budget_policy', None),
                          resume=getattr(args, 'resume_batch', False), stop_event=args.stop_event,
                          shared=args._batch_ledger_root != args.output_dir)
             if getattr(args, 'model_configs', None) else nullcontext())
    from provider_limits import provider_limit_session
    models = getattr(args, 'model_configs', None)
    active_models = [models['analysis']] if models else []
    if models:
        if not getattr(args, 'no_review', True):
            active_models.append(models['review'])
        if (getattr(args, 'max_repairs', 0)
                and getattr(args, 'pipeline_options', {}).get('repair_mode', 'typed') == 'typed'):
            active_models.append(models['repair'])
    provider_guard = (provider_limit_session(args._batch_ledger_root, active_models, stop_event=args.stop_event)
                      if active_models else nullcontext())
    disk_space_guard = ({'minimum_free_bytes': min_free_disk_bytes,
                         'available_bytes': None, 'stop_reason': None}
                        if min_free_disk_bytes else None)
    args.disk_space_stop_event = threading.Event()
    args._disk_space_guard_state = disk_space_guard
    args._disk_space_guard_lock = threading.Lock()
    with process_lock(args.output_dir/'.batch.lock'), ExitStack() as stack:
        # Recheck under the output lock, before entering budget/limit sessions
        # or writing the pinned asset snapshot, groups, or index files.
        try:
            check_minimum_free_disk(_disk_probe_path(args.output_dir), min_free_disk_bytes)
            if args._batch_ledger_root != args.output_dir:
                check_minimum_free_disk(_disk_probe_path(args._batch_ledger_root), min_free_disk_bytes)
        except DiskSpacePause as exc:
            return publish_disk_space_pause(exc)
        stack.enter_context(_batch_asset_scope(args))
        batch_control = stack.enter_context(guard)
        provider_control = stack.enter_context(provider_guard)
        # Asset/catalog snapshot metadata belongs to this output directory and
        # may only be initialized after acquiring its coordinator lock.
        try:
            prepare_batch_asset_snapshot(args)
        except DiskSpacePause as exc:
            return publish_disk_space_pause(exc, stage='disk_space_asset_snapshot')
        all_groups = reconcile_groups(args.output_dir, groups)
        requested = [group['scene_id'] for group in groups]
        requested_set = frozenset(requested)
        try:
            check_scene_start_disk_space(args)
        except DiskSpacePause as exc:
            return publish_disk_space_pause(exc, stage='disk_space_groups_index')
        atomic_json(args.output_dir/'groups.json', {'schema_version': 1, 'scenes': [
            {'scene_id': group['scene_id'], 'images': [str(path) for path in group['images']]} for group in all_groups]})
        # Share the already-pinned asset inventory with recovery. A fresh
        # guard here would re-hash every library model/texture once more before
        # workers resume; the batch guard already checked identities under the
        # coordinator lock.
        args._artifact_digest_cache = {}
        try:
            check_scene_start_disk_space(args)
            records = recorded_results(args.output_dir, getattr(args, '_asset_snapshot_guard', None),
                                       args._artifact_digest_cache, skip_scene_ids=requested_set)
        except DiskSpacePause as exc:
            return publish_disk_space_pause(exc, stage='disk_space_recovery_index')
        for group in all_groups:
            if group['scene_id'] in requested_set:
                continue
            records.setdefault(group['scene_id'], {'scene_id': group['scene_id'],
                'directory': str(args.output_dir/group['scene_id']), 'status': 'pending'})
        run_id = uuid.uuid4().hex
        current, completed = {}, set()
        for scene_id in requested:
            # An old accepted outcome is unavailable while its inputs are
            # being checked for this invocation; interruption cannot retain it.
            # Drop the recovered payload now rather than retaining a second
            # full result (including artifact manifests) beside ``current``.
            # ``run_job`` performs the authoritative per-scene resume checks
            # before returning an accepted result, so this does not bypass
            # validation and keeps checkpoint memory proportional to one copy.
            records.pop(scene_id, None)
            current[scene_id] = {'scene_id': scene_id, 'directory': str(args.output_dir/scene_id), 'status': 'pending'}
        started_at = utc_now()
        interrupted = False
        event_sequence = 0

        def snapshot_records():
            """Merge historical records with full results from this run's events.

            ``current`` intentionally contains only lightweight live status
            entries after an event is durably appended. Reading the event log
            at the two terminal snapshot boundaries restores the complete
            result payload without retaining every artifact manifest in the
            coordinator's hot loop. Event results are merged in place so the
            terminal snapshot does not briefly hold separate event and
            combined dictionaries of the same payloads.
            """
            # Preserve the previous merge precedence: this invocation's
            # current/pending status supersedes any recovered historical
            # record, then a durable event supersedes that live status.
            records.update(current)
            for item in read_results(args.output_dir, run_id):
                records[item['scene_id']] = item
            ordered = []
            known = set()
            for group in all_groups:
                scene_id = group['scene_id']
                item = records.get(scene_id)
                if item is not None:
                    ordered.append(item)
                    known.add(scene_id)
            for scene_id in sorted(set(records) - known):
                ordered.append(records[scene_id])
            return ordered

        def checkpoint(state, *, snapshot=False):
            """Persist cheap run state; materialize compatibility indexes at boundaries."""
            control_state = {'batch_control': batch_control.snapshot()} if batch_control is not None else {}
            if provider_control is not None and provider_control.policies:
                control_state['provider_rate_control'] = provider_control.snapshot()
            if disk_space_guard is not None:
                with args._disk_space_guard_lock:
                    control_state['disk_space_guard'] = dict(disk_space_guard)
            if not snapshot:
                atomic_json(args.output_dir/'run_state.json', {'schema_version': 1, 'run_id': run_id,
                    'state': state, 'started_at': started_at, 'updated_at': utc_now(),
                    'requested': len(requested), 'completed_this_run': len(completed),
                    'incremental_events': EVENTS_FILE, **control_state})
                return
            ordered = snapshot_records()
            atomic_json(args.output_dir/'summary.json', {'schema_version': 2, 'scenes': ordered,
                'finished': sum(item.get('status') not in ('pending', 'running') for item in ordered),
                'total': len(ordered), 'run_id': run_id, 'run_state': state,
                'requested': len(groups), 'completed_this_run': len(completed), 'updated_at': utc_now(),
                'incremental_events': EVENTS_FILE, **control_state})
            atomic_json(args.output_dir/'accepted.json', [item for item in ordered if item.get('status') == 'accepted'])
            atomic_json(args.output_dir/'run_state.json', {'schema_version': 1, 'run_id': run_id,
                'state': state, 'started_at': started_at, 'updated_at': utc_now(), **control_state,
                'requested_scene_ids': requested, 'completed_scene_ids': sorted(completed),
                'pending_scene_ids': [identifier for identifier in requested if identifier not in completed],
                'resumable_scene_ids': [identifier for identifier in requested if current[identifier]['status']
                    in ('failed', 'review_failed', 'interrupted', 'locked', 'pending', 'running')]})

        checkpoint('running', snapshot=True)
        reset_events(args.output_dir)
        pending = {}
        iterator = iter(groups)
        pool = ThreadPoolExecutor(max_workers=args.workers)

        def disk_space_allows_enqueue():
            """Fail closed before taking another scene from the bounded queue."""
            if disk_space_guard is None:
                return True
            try:
                check_minimum_free_disk(_disk_probe_path(args.output_dir), min_free_disk_bytes)
            except DiskSpacePause as exc:
                first = not args.disk_space_stop_event.is_set()
                _record_disk_space_pause(args, exc)
                if first:
                    with args._disk_space_guard_lock:
                        details = dict(disk_space_guard)
                    print(json.dumps({'stage': 'disk_space_stop', **details}, ensure_ascii=False), flush=True)
                return False
            return True

        def fill():
            while (len(pending) < args.workers and not args.stop_event.is_set()
                   and not args.disk_space_stop_event.is_set()):
                group = next(iterator, None)
                if group is None:
                    break
                if not disk_space_allows_enqueue():
                    break
                pending[pool.submit(run_job, group, args, hashes)] = group
                current[group['scene_id']]['status'] = 'running'

        try:
            fill()
            while pending:
                try:
                    done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                    for future in done:
                        group = pending.pop(future)
                        try:
                            result = future.result()
                            if not isinstance(result, dict) or result.get('scene_id') != group['scene_id']:
                                raise ValueError('Worker returned an invalid scene result')
                        except Exception as exc:
                            result = _record_failure(group, args, exc)
                        if result.get('status') == 'paused' and result.get('stop_reason') == 'disk_space_guard':
                            # The worker was admitted but found that free space
                            # fell before expensive work began. It did not
                            # commit an attempt, so keep it pending and do not
                            # append a terminal outcome.
                            current[group['scene_id']]['status'] = 'pending'
                        else:
                            current[group['scene_id']] = result
                            completed.add(group['scene_id'])
                            event_sequence += 1
                            append_result(args.output_dir, run_id, event_sequence, result)
                            # The durable event now owns the full result. Keeping
                            # only this live status avoids accumulating large
                            # artifact/history payloads across a long batch.
                            current[group['scene_id']] = {
                                'scene_id': result['scene_id'],
                                'directory': result.get('directory', str(args.output_dir/group['scene_id'])),
                                'status': result.get('status', 'failed'),
                            }
                    if done:
                        checkpoint('stopping' if args.stop_event.is_set() else
                                   ('paused_disk_space' if args.disk_space_stop_event.is_set() else 'running'))
                        fill()
                except KeyboardInterrupt:
                    interrupted = True
                    args.stop_event.set()
                    checkpoint('stopping')
                    print(json.dumps({'stage': 'stopping', 'message': 'No new jobs will start; current API/render calls will finish. Run the same command to resume.'}), flush=True)
            for scene_id in requested:
                if scene_id not in completed:
                    current[scene_id]['status'] = ('pending' if args.disk_space_stop_event.is_set()
                                                   else 'interrupted')
            checkpoint('paused_disk_space' if args.disk_space_stop_event.is_set() else
                       ('stopped' if current_stop(args.output_dir) else
                        ('interrupted' if interrupted else 'completed')), snapshot=True)
        except BaseException:
            args.stop_event.set()
            # Wait before releasing output ownership: a worker may still be
            # committing artifacts even when the coordinator failed.
            pool.shutdown(wait=True, cancel_futures=True)
            checkpoint('interrupted', snapshot=True)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if not (disk_space_guard and disk_space_guard['stop_reason']):
            try:
                from report_multiview import report as make_report
                make_report(args.output_dir)
            except Exception as exc:
                # Comparisons are diagnostic artifacts; a missing/corrupt original
                # must not discard committed scene/index checkpoints.
                atomic_json(args.output_dir/'report_error.json', {'at': utc_now(), 'error': str(exc)})
                print(json.dumps({'stage': 'report_failed', 'error': str(exc)}, ensure_ascii=False), flush=True)
        results = [current[scene_id] for scene_id in requested]
        print(json.dumps({'summary': str(args.output_dir/'summary.json'),
                          'statuses': {item['scene_id']: item['status'] for item in results}}, ensure_ascii=False), flush=True)
        return (130 if interrupted else DISK_SPACE_PAUSE_EXIT_CODE if args.disk_space_stop_event.is_set()
            else int(bool(current_stop(args.output_dir)) or any(
            item['status'] in ('failed', 'locked', 'review_failed', 'interrupted') for item in results)))


def coordinate(groups, args, hashes):
    require_large_batch_budget(len(groups), getattr(args, 'batch_budget_policy', None))
    from model_api import api_concurrency_limit
    with api_concurrency_limit(getattr(args, 'api_workers', 2)), \
            render_session(getattr(args, 'render_workers', 1)):
        return _coordinate(groups, args, hashes)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,help='Model API and batch/render settings; omit to use the legacy Codex CLI')
    p.add_argument('--groups',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-edge',type=int,default=1280)
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--api-workers',type=int,default=2,help='Maximum in-flight model HTTP requests, including staged detail calls')
    p.add_argument('--render-workers',type=int,default=1,help='Maximum concurrent Blender processes in this batch; raise only after measuring memory/GPU capacity')
    p.add_argument('--min-free-disk-bytes',type=int,default=0,
                   help='Stop enqueueing scenes below this free-space threshold; 0 disables the guard')
    p.add_argument('--shared-ledger-dir',type=Path,
                   help='Shared local SQLite budget/rate ledger root; every output shard must be inside it')
    p.add_argument('--max-repairs',type=int,default=1)
    p.add_argument('--vision-timeout',type=int,default=900)
    p.add_argument('--render-timeout',type=int,default=1800)
    p.add_argument('--effort',choices=['low','medium','high'],default='medium')
    p.add_argument('--device',choices=['AUTO','CPU'],default='AUTO')
    p.add_argument('--blender')
    p.add_argument('--reuse-analysis-from',type=Path,help='Recover unchanged raw model responses from a previous batch with identical prompts/photos; never accepts a scene seed')
    p.add_argument('--no-review',action='store_true')
    p.add_argument('--review',dest='no_review',action='store_false')
    p.add_argument('--appearance-policy',choices=['photo_reconstruction','legacy'],default='photo_reconstruction')
    p.add_argument('--tone-match',choices=['reference','off'],default='reference')
    p.add_argument('--asset-catalog',type=Path, help='Validated asset_catalog_v1 JSON')
    p.add_argument('--asset-bindings',type=Path, help='Sidecar bindings: object_id -> approved asset/material variant')
    p.add_argument('--asset-auto-bind',action='store_true', help='Automatically select catalog assets per scene; omit manual bindings')
    p.add_argument('--asset-preview',action='store_true', help='Allow draft assets for preview; never use for accepted datasets')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume-batch', action='store_true',
                   help='Explicitly reopen a stopped provider circuit; persisted usage/limits are retained')
    peek=argparse.ArgumentParser(add_help=False)
    peek.add_argument('--config',type=Path)
    peek.add_argument('--shared-ledger-dir',type=Path)
    known,_=peek.parse_known_args(argv)
    runtime=None
    if known.config:
        from factory_config import load_config
        try:
            runtime=load_config(known.config)
        except (OSError,ValueError) as exc:
            p.error(str(exc))
        p.set_defaults(**runtime['batch'],**runtime['render'])
    else:
        p.set_defaults(no_review=False)
    args=p.parse_args(argv)
    args.model_configs=runtime['models'] if runtime else None
    args.pipeline_options=runtime['pipeline'] if runtime else {}
    args.scene_budget_policy=runtime.get('budget') if runtime else None
    args.batch_budget_policy=runtime.get('batch_budget') if runtime else None
    args.shared_ledger_dir = None
    if known.shared_ledger_dir is not None:
        # An explicit CLI override follows the caller's working directory.
        args.shared_ledger_dir = known.shared_ledger_dir.expanduser().resolve()
    elif runtime and runtime['batch'].get('shared_ledger_dir') is not None:
        configured_ledger = Path(runtime['batch']['shared_ledger_dir']).expanduser()
        if not configured_ledger.is_absolute():
            configured_ledger = known.config.expanduser().resolve().parent / configured_ledger
        args.shared_ledger_dir = configured_ledger.resolve()
    args.asset_selection_options = {}
    if runtime and runtime.get('asset_library', {}).get('catalog'):
        config_directory = args.config.expanduser().resolve().parent
        def configured_asset_path(value):
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (config_directory / path).resolve()
        args.asset_catalog = configured_asset_path(runtime['asset_library']['catalog'])
        binding_value = runtime['asset_library']['bindings']
        args.asset_bindings = configured_asset_path(binding_value) if binding_value else None
        args.asset_auto_bind = runtime['asset_library']['auto_bind']
        args.asset_selection_options = runtime['asset_library']['selection']
        args.asset_preview = runtime['asset_library']['preview']
    if args.asset_auto_bind and (not args.asset_catalog or args.asset_bindings):
        p.error('--asset-auto-bind requires --asset-catalog and forbids --asset-bindings')
    if not args.asset_auto_bind and bool(args.asset_catalog) != bool(args.asset_bindings):
        p.error('--asset-catalog and --asset-bindings must be supplied together unless --asset-auto-bind is used')
    for key,low,high in [('max_edge',64,8192),('samples',1,4096),('workers',1,8),('api_workers',1,32),('render_workers',1,8),('min_free_disk_bytes',0,10**15),('max_repairs',0,5),('vision_timeout',1,7200),('render_timeout',1,14400)]:
        if not low <= getattr(args,key) <= high:
            p.error(f'{key} must be {low}..{high}')
    groups=load_groups(args.groups)
    args.output_dir=args.output_dir.expanduser().resolve()
    try:
        _resolve_batch_ledger_root(args)
    except ValueError as exc:
        p.error(str(exc))
    if any(source == args.output_dir or source.is_relative_to(args.output_dir) for group in groups for source in group['images']):
        p.error('Keep output directory separate from input photos')
    if args.dry_run:
        preview = [{'scene_id': group['scene_id'],
                    'images': [str(path) for path in group['images']]}
                   for group in groups[:20]]
        if len(groups) >= LARGE_BATCH_BUDGET_SCENES:
            print(json.dumps({
                'dry_run': True,
                'scenes': len(groups),
                'images': sum(len(group['images']) for group in groups),
                'preview_scenes': preview,
                'omitted_scenes': len(groups) - len(preview),
                'live_run_budget_missing': large_batch_budget_missing(args.batch_budget_policy),
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps([{'scene_id':group['scene_id'],'images':[str(path) for path in group['images']]}
                              for group in groups],indent=2))
        return 0
    try:
        require_large_batch_budget(len(groups), args.batch_budget_policy)
    except ValueError as exc:
        p.error(str(exc))
    if args.model_configs:
        from model_api import preflight
        try:
            preflight(args.model_configs['analysis'])
            if not args.no_review:
                preflight(args.model_configs['review'])
            if args.max_repairs and args.pipeline_options.get('repair_mode') == 'typed':
                preflight(args.model_configs['repair'])
        except (ValueError,RuntimeError) as exc:
            p.error(str(exc))
    args.blender=discover_blender(args.blender)
    hashes=pipeline_hashes()
    try:
        return coordinate(groups, args, hashes)
    except OutputInUseError as exc:
        p.error(str(exc))


if __name__=='__main__':
    raise SystemExit(main())
