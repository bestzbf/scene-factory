#!/usr/bin/env python3
"""Batch reconstruct individual reference photos through a scene spec and Blender."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
RUNNER_VERSION = 2


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def atomic_json(path, value):
    """Replace a complete JSON file atomically, including on interrupted runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def number(value, name, minimum, maximum, integer=False):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or not minimum <= value <= maximum:
        kind = "integer" if integer else "number"
        raise ValueError(f"{name} must be a {kind} between {minimum} and {maximum}")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_blender(explicit=None):
    candidate = explicit or os.environ.get("BLENDER_BIN")
    if candidate:
        resolved = shutil.which(str(candidate)) or str(Path(candidate).expanduser())
        if Path(resolved).is_file() and os.access(resolved, os.X_OK):
            return str(Path(resolved).resolve())
        raise ValueError(f"Blender executable is unavailable: {candidate}")
    on_path = shutil.which("blender")
    if on_path:
        return str(Path(on_path).resolve())
    for app in (Path("/Applications/Blender.app/Contents/MacOS/Blender"), Path.home() / "Applications/Blender.app/Contents/MacOS/Blender"):
        if app.is_file() and os.access(app, os.X_OK):
            return str(app)
    raise ValueError("Blender was not found. Pass --blender /path/to/blender or set BLENDER_BIN.")


def collect_images(input_dir=None, images=None):
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    if input_dir:
        directory = Path(input_dir).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"input directory does not exist: {directory}")
        candidates = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    else:
        candidates = [Path(path).expanduser().resolve() for path in (images or [])]
    result = []
    seen = set()
    for path in candidates:
        path = path.resolve()
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"input image is missing or empty: {path}")
        if path.suffix.lower() not in extensions:
            raise ValueError(f"unsupported image extension: {path}")
        if path not in seen:
            result.append(path)
            seen.add(path)
    if not result:
        raise ValueError("no input images found")
    return result


def make_job(image, width, height, samples, converter, generator, asset_quality="detailed", desk_details=False):
    digest = file_sha256(image)
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in Path(image).stem).strip("._")[:60] or "image"
    return {
        "schema_version": SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "job_id": f"{digest[:12]}_{safe_stem}",
        "image": str(Path(image).resolve()),
        "image_sha256": digest,
        "render": {"width": width, "height": height, "samples": samples, "asset_quality": asset_quality, "desk_details": desk_details},
        "pipeline": {"photo_to_spec_sha256": file_sha256(converter), "generate_scene_sha256": file_sha256(generator),
                     "schema_sha256": file_sha256(ROOT / "schema.py") if (ROOT / "schema.py").is_file() else None,
                     "assets": {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(list(ROOT.glob("detailed_assets.py"))+list(ROOT.glob("desk_details.py"))+list((ROOT / "assets_lib").rglob("*"))) if p.is_file() and "__pycache__" not in str(p)}},
    }


def prepare_job(job, output_dir):
    directory = Path(output_dir) / job["job_id"]
    job_path = directory / "job.json"
    directory.mkdir(parents=True, exist_ok=True)
    if job_path.exists():
        try:
            if read_json(job_path) != job:
                raise ValueError("input, parameters, or pipeline differs from job.json; choose another --output-dir")
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot safely reuse existing job.json: {exc}") from exc
    elif any(directory.iterdir()):
        raise ValueError("job directory contains files without job.json; choose another --output-dir")
    else:
        atomic_json(job_path, job)
    return directory


def valid_spec(directory):
    try:
        spec = read_json(Path(directory) / "scene_spec.json")
        return isinstance(spec, dict) and bool(spec)
    except (OSError, ValueError):
        return False


def completed_job(directory):
    """Zero exit status is insufficient: verify metadata and nonempty geometry."""
    directory = Path(directory)
    try:
        metadata = read_json(directory / "metadata.json")
        if not isinstance(metadata, dict) or metadata.get("status") != "completed":
            return False, "metadata.json does not report status=completed"
        blend = directory / "scene.blend"
        if not blend.is_file() or not blend.stat().st_size:
            return False, "scene.blend is missing or empty"
        outputs = metadata.get("outputs", [])
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        if not isinstance(outputs, list):
            return False, "metadata.outputs must be a list or dictionary of relative file paths"
        for name in outputs:
            if not isinstance(name, str) or not name or Path(name).is_absolute():
                return False, "metadata.outputs contains an invalid relative file path"
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()):
                return False, "metadata.outputs contains a path outside the job directory"
            if not path.is_file() or not path.stat().st_size:
                return False, f"declared output is missing or empty: {name}"
        return True, "completed"
    except (OSError, ValueError, TypeError) as exc:
        return False, str(exc)


def job_commands(job, directory, converter, generator, blender):
    directory = Path(directory).resolve()
    render = job["render"]
    spec = directory / "scene_spec.json"
    return [sys.executable, str(Path(converter).resolve()), "--image", job["image"], "--output", str(spec)], [
        str(blender or "blender"), "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(Path(generator).resolve()), "--",
        "--spec", str(spec), "--output-dir", str(directory), "--width", str(render["width"]), "--height", str(render["height"]), "--samples", str(render["samples"]), "--asset-quality", render.get("asset_quality","basic"),
    ] + (["--desk-details"] if render.get("desk_details") else [])


def run_stage(command, logfile, timeout):
    with Path(logfile).open("a", encoding="utf-8") as log:
        log.write(f"\nBatch attempt: {utc_now()}\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"process exited with code {result.returncode}; see {Path(logfile).name}")


def run_job(job, output_dir, converter, generator, blender, timeout=1800, specs_only=False, reuse_spec=False, dry_run=False):
    start = time.monotonic()
    record = {"job_id": job["job_id"], "image": job["image"], "image_sha256": job["image_sha256"], "started_at": utc_now()}
    directory = Path(output_dir) / job["job_id"]
    prepared = False
    try:
        directory = prepare_job(job, output_dir)
        prepared = True
        record["directory"] = str(directory.resolve())
        convert_command, render_command = job_commands(job, directory, converter, generator, blender)
        record["commands"] = [convert_command] if specs_only else [convert_command, render_command]
        complete, _ = completed_job(directory)
        if complete:
            record["status"] = "skipped"
        elif dry_run:
            record["status"] = "planned"
        else:
            # A stale success marker must not validate a later failed attempt.
            if not specs_only and (directory / "metadata.json").exists():
                (directory / "metadata.json").unlink()
            atomic_json(directory / "runner_status.json", {"status": "running", "started_at": record["started_at"]})
            if not (reuse_spec and valid_spec(directory)):
                record["stage"] = "photo_to_spec"
                run_stage(convert_command, directory / "photo_to_spec.log", timeout)
            if not valid_spec(directory):
                raise RuntimeError("photo_to_spec did not produce a nonempty JSON object in scene_spec.json")
            if specs_only:
                record["status"] = "spec_ready"
            else:
                record["stage"] = "blender"
                run_stage(render_command, directory / "blender.log", timeout)
                complete, reason = completed_job(directory)
                if not complete:
                    raise RuntimeError(f"Blender output validation failed: {reason}")
                record["status"] = "completed"
            record.pop("stage", None)
    except ValueError as exc:
        record.update({"status": "conflict", "error": str(exc)})
    except Exception as exc:
        error = f"stage exceeded timeout of {timeout}s; see stage log" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)
        record.update({"status": "failed", "error": error})
    record.update({"finished_at": utc_now(), "elapsed_seconds": round(time.monotonic() - start, 3)})
    if prepared and not dry_run:
        atomic_json(directory / "runner_status.json", record)
    return record


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", type=Path, help="Recursively discover supported reference images")
    inputs.add_argument("--images", nargs="+", type=Path, help="Reference images to reconstruct independently")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "photos", help="Each image gets its own job directory")
    parser.add_argument("--blender", help="Blender executable; otherwise BLENDER_BIN, PATH, or macOS app")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent jobs, 1..8 (default: 1)")
    parser.add_argument("--timeout", type=float, default=1800, help="Timeout per pipeline stage in seconds (default: 1800)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--asset-quality", choices=["basic","detailed"], default="detailed")
    parser.add_argument("--desk-details", action="store_true", help="Add inferred supported desktop details, separately annotated")
    parser.add_argument("--specs-only", action="store_true", help="Read photos into scene specs without starting Blender")
    parser.add_argument("--reuse-spec", action="store_true", help="Reuse a valid existing scene_spec.json in unfinished jobs")
    parser.add_argument("--dry-run", action="store_true", help="Write job plans and manifest without starting either pipeline stage")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    try:
        number(args.workers, "workers", 1, 8, True)
        number(args.timeout, "timeout", 1, 86400)
        number(args.width, "width", 64, 8192, True)
        number(args.height, "height", 64, 8192, True)
        number(args.samples, "samples", 1, 4096, True)
        images = collect_images(args.input_dir, args.images)
        converter = ROOT / "photo_to_spec.py"
        generator = ROOT / "generate_scene.py"
        for path in (converter, generator):
            if not path.is_file():
                raise ValueError(f"pipeline script is missing: {path}")
        jobs = [make_job(image, args.width, args.height, args.samples, converter, generator, args.asset_quality, args.desk_details) for image in images]
        # Identical content and filename can recur in different input folders.
        seen = set()
        jobs = [job for job in jobs if not (job["job_id"] in seen or seen.add(job["job_id"]))]
        if args.dry_run or args.specs_only:
            blender = args.blender or os.environ.get("BLENDER_BIN") or "blender"
        else:
            blender = discover_blender(args.blender)
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    started_at = utc_now()
    records = []
    # Workers have independent scene folders; only the main thread appends manifest.
    with (output / "manifest.jsonl").open("a", encoding="utf-8") as manifest, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job, output, converter, generator, blender, args.timeout, args.specs_only, args.reuse_spec, args.dry_run) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            manifest.flush()
            os.fsync(manifest.fileno())
            detail = f": {record['error']}" if "error" in record else ""
            print(f"{record['job_id']}: {record['status']}{detail}", flush=True)
    counts = {key: sum(item["status"] == key for item in records) for key in ("completed", "skipped", "spec_ready", "planned", "failed", "conflict")}
    atomic_json(output / "summary.json", {"schema_version": 1, "started_at": started_at, "finished_at": utc_now(), "output": str(output), "counts": counts, "jobs": sorted(records, key=lambda item: item["job_id"])})
    print(f"Summary: {output / 'summary.json'}")
    return 1 if counts["failed"] or counts["conflict"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
