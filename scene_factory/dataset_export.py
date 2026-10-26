"""Verify and publish a portable snapshot of accepted grouped scene jobs.

The batch indexes are operational summaries, not an approval authority.  This
exporter reads each job's final result, verifies its committed artifacts and
matched-image review, and recomputes the quality gate before publishing.  Model
prompts, raw responses, request logs and provider configuration are deliberately
excluded.  A successful export certifies the pipeline's heuristic review only;
it does not establish metric accuracy or unseen geometry correctness.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

from batch import atomic_json, canonical_bytes, file_sha256, utc_now
from multiview_spec import SAFE_ID, resolve_sources, texture_view_map, validate


SCHEMA_VERSION = "scene_dataset_v1"
_ITERATION = re.compile(r"iteration_[0-9]{2,}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_PHOTO_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


class DatasetExportError(ValueError):
    """An export cannot safely certify or publish its requested inputs."""


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise DatasetExportError("Duplicate JSON key in export input")
            result[key] = value
        return result
    def reject_constant(value):
        raise DatasetExportError("Nonfinite JSON in export input")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=reject_constant)


def _local(root, value, *, directory=False):
    """Resolve only regular, nonsymlink paths strictly within a trusted root."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise DatasetExportError("Invalid artifact path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise DatasetExportError("Artifact path escapes its job")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise DatasetExportError("Symlink artifacts cannot enter a formal dataset")
    candidate = current.resolve()
    if not candidate.is_relative_to(root.resolve()) or not (candidate.is_dir() if directory else candidate.is_file()):
        raise DatasetExportError("Artifact missing or outside its job: " + value)
    return candidate


def _hashes(root, records):
    if not isinstance(records, dict) or not records:
        raise DatasetExportError("Missing committed artifact hashes")
    for name, digest in records.items():
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise DatasetExportError("Invalid artifact SHA256")
        if file_sha256(_local(root, name)) != digest:
            raise DatasetExportError("Artifact changed or missing: " + name)


def _require_committed(folder, result, names):
    missing = set(names) - set(result["artifacts"])
    if missing:
        raise DatasetExportError("Required evidence absent from committed result: " + ", ".join(sorted(missing)))
    return [_local(folder, name) for name in names]


def _false_preview(value):
    """Reject any explicit preview marker, including nested importer records."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"preview", "preview_only", "asset_preview_only"} and item is not False:
                raise DatasetExportError("Draft/preview assets cannot enter a formal dataset")
            _false_preview(item)
    elif isinstance(value, list):
        for item in value:
            _false_preview(item)


def _verified_job(folder):
    """Validate the exact final reviewed iteration, never a batch summary flag."""
    from batch_multiview import source_check, validate_render
    import vision_multiview as vision

    if (folder / ".lock").exists():
        raise DatasetExportError("Scene job is locked/running: " + folder.name)
    result = _json(_local(folder, "result.json"))
    if result.get("status") != "accepted" or result.get("gate", {}).get("accepted") is not True:
        raise DatasetExportError("Scene is not accepted: " + folder.name)
    if result.get("scene_id") != folder.name:
        raise DatasetExportError("Scene identity differs from its directory")
    iteration_name = result.get("iteration")
    if (not isinstance(iteration_name, str) or not _ITERATION.fullmatch(iteration_name)
            or result.get("final_review_iteration") != iteration_name):
        raise DatasetExportError("Final review does not certify the final iteration")
    if any(p.is_dir() and _ITERATION.fullmatch(p.name) and int(p.name[10:]) > int(iteration_name[10:])
           for p in folder.iterdir()):
        raise DatasetExportError("A newer iteration exists after the accepted review")
    _hashes(folder, result.get("artifacts"))
    required = ["job.json", "initial_bundle.json", "analysis_provenance.json"]
    required += [iteration_name + "/" + name for name in
                 ("scene_bundle.json", "render_state.json", "metadata.json", "scene.blend",
                  "review.json", "review_provenance.json", "textures/manifest.json")]
    _require_committed(folder, result, required)
    job = _json(folder / "job.json")
    if job.get("scene_id") != folder.name or job.get("options", {}).get("no_review") is not False:
        raise DatasetExportError("A reviewed matching job is required")
    _false_preview(job.get("asset_library", {}))
    source_check(job)
    analysis = _json(folder / "analysis_provenance.json")
    if (analysis.get("manual_scene_seed") is not False
            or analysis.get("bundle_sha256") != file_sha256(folder / "initial_bundle.json")):
        raise DatasetExportError("Missing automatic photo-analysis provenance")
    initial = validate(_json(folder / "initial_bundle.json"))
    job_sources = job.get("sources", [])
    expected_ids = [f"view_{index + 1:02d}" for index in range(len(job_sources))]
    initial_sources = resolve_sources(initial, folder / "initial_bundle.json")
    expected_sources = {view_id: file_sha256(path) for view_id, path in initial_sources.items()}
    if (initial["scene_id"] != folder.name
            or [view["id"] for view in initial["views"]] != expected_ids
            or analysis.get("source_sha256") != expected_sources):
        raise DatasetExportError("Initial photo-analysis provenance does not match job source order")
    iteration = _local(folder, iteration_name, directory=True)
    bundle = validate(_json(iteration / "scene_bundle.json"))
    if bundle["scene_id"] != folder.name:
        raise DatasetExportError("Final bundle has a different scene identity")
    if ([{key: view[key] for key in ("id", "source_image", "width", "height")} for view in initial["views"]]
            != [{key: view[key] for key in ("id", "source_image", "width", "height")} for view in bundle["views"]]):
        raise DatasetExportError("Final iteration changed the original source identities")
    state = _json(iteration / "render_state.json")
    _hashes(iteration, state.get("artifacts"))
    selection_hashes = {name: file_sha256(iteration / name) for name in
                        ("asset_bindings.json", "asset_selection.json") if (iteration / name).is_file()}
    if not job.get("asset_library", {}).get("mode") == "automatic":
        selection_hashes = {}
    expected = hashlib.sha256(canonical_bytes({"bundle": bundle, "job": job,
                                               "asset_selection": selection_hashes})).hexdigest()
    if state.get("fingerprint") != expected:
        raise DatasetExportError("Final render fingerprint differs from its job inputs")
    metadata = validate_render(iteration, bundle)
    _false_preview(metadata)
    report = vision.load_review(iteration, bundle)
    provenance = _json(iteration / "review_provenance.json")
    annotations = {}
    for view in bundle["views"]:
        names = [f"{iteration_name}/rgb_{view['id']}.png", f"{iteration_name}/annotations_{view['id']}.json"]
        _require_committed(folder, result, names)
        annotations[view["id"]] = _json(iteration / f"annotations_{view['id']}.json")
    gate = vision.quality_gate(report, metadata, annotations)
    if not gate["accepted"] or provenance.get("gate", {}).get("accepted") is not True:
        raise DatasetExportError("Fresh quality gate rejected scene: " + ", ".join(gate["reasons"]))
    if len(job_sources) != len(bundle["views"]):
        raise DatasetExportError("Source count differs from the reviewed bundle")
    sources = resolve_sources(bundle, iteration / "scene_bundle.json")
    source_records = []
    for index, view in enumerate(bundle["views"]):
        manifest_name = f"sources/{view['id']}/manifest.json"
        _require_committed(folder, result, [manifest_name])
        manifest = _json(folder / manifest_name)
        reference = sources[view["id"]]
        try:
            relative = reference.relative_to(folder.resolve()).as_posix()
        except ValueError:
            raise DatasetExportError("Reviewed reference is outside the scene job") from None
        _require_committed(folder, result, [relative])
        original = job_sources[index]
        if (Path(manifest["reference_image"]).resolve() != reference
                or Path(manifest["source_image"]).resolve() != Path(original["path"]).resolve()
                or manifest["source_sha256"] != original["sha256"]):
            raise DatasetExportError("Original/reference association changed")
        source_records.append({"view_id": view["id"], "reference": reference, "original": Path(original["path"]),
                               "original_sha256": original["sha256"]})
    return {"folder": folder, "result": result, "job": job, "bundle": bundle,
            "metadata": metadata, "report": report, "provenance": provenance, "gate": gate,
            "iteration": iteration, "sources": source_records,
            "result_sha256": file_sha256(folder / "result.json")}


def _copy(source, target, expected=None):
    if source.is_symlink() or not source.is_file():
        raise DatasetExportError("Only regular files may be exported")
    digest = expected or file_sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if file_sha256(source) != digest or file_sha256(target) != digest:
        raise DatasetExportError("Export input changed while copying")
    return digest


def _publish(staging, destination):
    """Atomic directory rename with an OS-enforced no-overwrite condition."""
    if os.name == "nt":
        # Windows rename refuses an existing destination.
        staging.rename(destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    old, new = os.fsencode(staging), os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        status = rename(old, new, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        status = rename(-100, old, -100, new, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise DatasetExportError("This platform lacks atomic no-overwrite dataset publication")
    if status != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _export_assets(record, destination):
    """Include only selected catalog records and their explicit dependencies."""
    from asset_catalog import AssetCatalog, load_catalog
    from asset_manifest import validate_manifest

    library = record["job"].get("asset_library")
    if not library:
        return {}
    catalog = load_catalog(library["catalog"])
    binding_path = (record["iteration"] / "asset_bindings.json" if library.get("mode") == "automatic"
                    else Path(library["bindings"]))
    bindings = _json(binding_path)
    if set(bindings) != {"schema_version", "instances"} or bindings["schema_version"] != 1:
        raise DatasetExportError("Invalid asset bindings")
    asset_ids = sorted({item["asset_id"] for item in bindings["instances"]})
    material_ids = sorted({item["material_variant_id"] for item in bindings["instances"]})
    paths = set()
    selected_assets = []
    manifest_paths = []
    for asset_id in asset_ids:
        asset = catalog.get_asset(asset_id)
        ready, reasons = catalog.production_readiness(asset)
        if not ready:
            raise DatasetExportError("Selected asset is no longer production-ready: " + ", ".join(reasons))
        paths.add(asset["model_file"])
        if asset.get("collision_file"):
            paths.add(asset["collision_file"])
        if asset.get("manifest_file"):
            manifest = catalog.resolve_path(asset["manifest_file"])
            if file_sha256(manifest) != (asset.get("audit_provenance") or {}).get("manifest_sha256"):
                raise DatasetExportError("Selected asset manifest changed")
            checked = validate_manifest(manifest, require_production=True)
            if not checked["production_ready"]:
                raise DatasetExportError("Selected asset manifest no longer passes production validation")
            data = _json(manifest)
            # A manifest is preserved byte-for-byte. Include every declared
            # dependency so its validation remains possible after relocation.
            dependencies = [data["model"]["path"], data["placement"]["collision_path"]]
            dependencies += [p["path"] if isinstance(p, dict) else p for p in data["previews"].values()]
            dependencies += data["provenance"]["source_images"]
            for material in data["materials"].values():
                for maps in [material.get("maps", {}), *(material.get("slot_materials") or {}).values()]:
                    dependencies += [p["path"] for p in maps.values() if p.get("path")]
            for name in dependencies:
                path = _local(manifest.parent, name)
                paths.add(path.relative_to(catalog.root).as_posix())
            paths.add(asset["manifest_file"])
            manifest_paths.append(asset["manifest_file"])
        asset["material_variant_ids"] = [mid for mid in asset["material_variant_ids"] if mid in material_ids]
        selected_assets.append(asset)
    materials = {mid: catalog.get_material_variant(mid) for mid in material_ids}
    for mid, material in materials.items():
        catalog.verify_material_hashes(mid)
        for maps in [material.get("maps", {}), *(material.get("slot_materials") or {}).values()]:
            paths.update(p["path"] for p in maps.values() if p.get("path"))
    for name in sorted(paths):
        _copy(catalog.resolve_path(name), destination / "assets" / name)
    portable_catalog = {"schema_version": "asset_catalog_v1", "root": "assets",
                        "assets": selected_assets, "material_variants": materials,
                        "source_manifests": manifest_paths}
    atomic_json(destination / "asset_catalog.json", portable_catalog)
    atomic_json(destination / "asset_bindings.json", bindings)
    # Validate using the destination root rather than the old absolute catalog.
    AssetCatalog(portable_catalog, catalog_path=destination / "asset_catalog.json")
    return {"asset_catalog": "asset_catalog.json", "asset_bindings": "asset_bindings.json"}


def _model_identity(value):
    return {key: value[key] for key in ("provider", "model") if key in value}


def _export_scene(record, destination):
    folder, iteration = record["folder"], record["iteration"]
    result, bundle = record["result"], copy.deepcopy(record["bundle"])
    destination.mkdir()
    source_map, source_index = {}, []
    for source in record["sources"]:
        view_id = source["view_id"]
        reference_name = f"references/{view_id}.png"
        suffix = source["original"].suffix.lower()
        if suffix not in _PHOTO_SUFFIXES:
            suffix = ".image"
        original_name = f"originals/{view_id}{suffix}"
        _copy(source["reference"], destination / reference_name)
        _copy(source["original"], destination / original_name, source["original_sha256"])
        source_map[view_id] = reference_name
        source_index.append({"view_id": view_id, "reference": reference_name, "original": original_name,
                             "original_sha256": source["original_sha256"],
                             "reference_sha256": file_sha256(destination / reference_name)})
    for view in bundle["views"]:
        view["source_image"] = source_map[view["id"]]
    validate(bundle)
    atomic_json(destination / "scene_bundle.json", bundle)
    _copy(iteration / "scene.blend", destination / "scene.blend")
    views = []
    for view, metadata in zip(bundle["views"], record["metadata"]["views"]):
        rgb, annotation = f"rgb_{view['id']}.png", f"annotations_{view['id']}.json"
        _copy(iteration / rgb, destination / rgb)
        _copy(iteration / annotation, destination / annotation)
        views.append({key: metadata[key] for key in ("id", "camera", "width", "height", "warning_count") if key in metadata}
                     | {"render": rgb, "annotation": annotation, "source_image": source_map[view["id"]]})
    manifest = _json(iteration / "textures/manifest.json")
    mapping = texture_view_map(bundle)
    crops = []
    for crop in manifest.get("textures", []):
        if crop["id"] not in mapping or crop["id"] in {p["id"] for p in crops}:
            raise DatasetExportError("Texture/source identity differs from its bundle")
        if crop.get("source_view_id") != mapping[crop["id"]]:
            raise DatasetExportError("Texture belongs to a different source view")
        path = Path(crop["path"])
        path = path.resolve() if path.is_absolute() else (iteration / "textures" / path).resolve()
        try:
            relative = path.relative_to(folder).as_posix()
        except ValueError:
            raise DatasetExportError("Texture path is outside the job") from None
        _require_committed(folder, result, [relative])
        name = crop["sha256"] + ".png"
        if not _DIGEST.fullmatch(crop["sha256"]):
            raise DatasetExportError("Invalid texture digest")
        _copy(path, destination / "textures" / name, crop["sha256"])
        crops.append({key: crop[key] for key in ("id", "sha256", "quad", "size", "source_view_id", "source_sha256") if key in crop}
                     | {"path": name})
    if {p["id"] for p in crops} != set(mapping):
        raise DatasetExportError("A scene texture has no committed crop")
    atomic_json(destination / "textures/manifest.json", {"textures": crops,
                "sources": [{"view_id": p["view_id"], "source_sha256": p["reference_sha256"]} for p in source_index]})
    asset_arguments = _export_assets(record, destination)
    metadata = {key: record["metadata"][key] for key in
                ("schema_version", "status", "scene_id", "scene_type", "blender_version", "geometry_build_count",
                 "camera_count", "asset_count", "mesh_count", "render", "appearance_policy") if key in record["metadata"]}
    metadata.update(views=views, dataset_status="accepted", source_records=source_index)
    atomic_json(destination / "metadata.json", metadata)
    review = copy.deepcopy(record["report"])
    review["revised_bundle"] = copy.deepcopy(bundle)
    atomic_json(destination / "evidence/visual_review.json", review)
    evidence = {"validated_at": utc_now(), "source_result_sha256": record["result_sha256"],
                "source_job_sha256": result["artifacts"]["job.json"],
                "source_bundle_sha256": file_sha256(iteration / "scene_bundle.json"),
                "source_review_sha256": file_sha256(iteration / "review.json"),
                "source_review_provenance_sha256": file_sha256(iteration / "review_provenance.json"),
                "source_render_state_sha256": file_sha256(iteration / "render_state.json"),
                "review_iteration": iteration.name, "review_model": _model_identity(record["provenance"].get("model", {})),
                "gate": record["gate"], "pipeline_sha256": record["job"].get("pipeline", {}),
                "portable_evidence_rewrites_source_paths": True,
                "packed_blend_policy": "Renderer calls pack_all before saving; the exporter verifies committed bytes, not Blender internals",
                "scope": "Accepted by matched-view heuristic visual review; dimensions and hidden geometry remain estimates"}
    atomic_json(destination / "evidence/validation.json", evidence)
    rebuild = {"renderer": "scene_factory/generate_multiview.py", "paths_relative_to": ".",
               "spec": "scene_bundle.json", "texture_manifest": "textures/manifest.json", **asset_arguments,
               "render": {key: value for key, value in record["metadata"].get("render", {}).items()
                          if key in {"samples", "max_edge", "device", "tone_match"}},
               "appearance_policy": record["job"]["options"].get("appearance_policy", "photo_reconstruction")}
    atomic_json(destination / "rebuild.json", rebuild)
    files = {p.relative_to(destination).as_posix(): file_sha256(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    entry = {"scene_id": bundle["scene_id"], "status": "accepted", "directory": "scenes/" + bundle["scene_id"],
             "blend": "scene.blend", "bundle": "scene_bundle.json", "views": views, "files_sha256": files,
             "review_iteration": iteration.name, "gate": record["gate"]}
    atomic_json(destination / "scene_manifest.json", entry)
    return entry


def export_dataset(output_dir, destination, *, scene_ids=None):
    """Atomically publish accepted scene jobs to a new portable directory.

    ``scene_ids=None`` scans all job directories, not just the latest invocation's
    summary. Explicit IDs must all be accepted. Any invalid accepted scene fails
    the entire export. Existing destinations are never merged or overwritten;
    export a new version to publish an updated dataset. Returns the manifest.
    """
    from batch_multiview import OutputInUseError, process_lock

    output = Path(output_dir).expanduser().resolve()
    destination = Path(destination).expanduser().absolute()
    if not output.is_dir():
        raise DatasetExportError("Batch output directory does not exist")
    if destination.exists() or destination.is_symlink():
        raise DatasetExportError("Dataset destination already exists; choose a new version")
    destination = destination.resolve()
    if destination.is_relative_to(output) or output.is_relative_to(destination):
        raise DatasetExportError("Dataset destination and batch jobs must be separate")
    if scene_ids is not None:
        if (not isinstance(scene_ids, (list, tuple)) or not scene_ids
                or any(not isinstance(s, str) or not SAFE_ID.fullmatch(s) for s in scene_ids)
                or len(set(scene_ids)) != len(scene_ids)):
            raise DatasetExportError("scene_ids must contain unique valid scene identifiers")
        identifiers = sorted(scene_ids)
    else:
        identifiers = None
    lock_path = destination.parent / ("." + destination.name + ".export.lock")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Hold the runner's stable kernel lock before discovering jobs and
        # throughout publication, so cooperating batch runs cannot change this
        # snapshot. Retained lock files are reusable after exit or a crash.
        with process_lock(output / ".batch.lock"), process_lock(lock_path):
            return _export_snapshot(output, destination, identifiers)
    except OutputInUseError as exc:
        raise DatasetExportError("Dataset export is locked by a running process: " + str(exc)) from exc
    except (KeyError, TypeError, OSError, ValueError) as exc:
        if isinstance(exc, DatasetExportError):
            raise
        raise DatasetExportError("Dataset export refused: " + str(exc)) from exc


def _export_snapshot(output, destination, identifiers):
    """Discover, verify and publish while source/destination locks are held."""
    if identifiers is None:
        identifiers = []
        for folder in sorted(output.iterdir()):
            if folder.is_symlink():
                if (folder / "result.json").is_file():
                    raise DatasetExportError("Symlink jobs cannot enter a formal dataset")
                continue
            if folder.is_dir() and SAFE_ID.fullmatch(folder.name) and (folder / "result.json").is_file():
                if _json(folder / "result.json").get("status") == "accepted":
                    identifiers.append(folder.name)
    if not identifiers:
        raise DatasetExportError("No accepted scenes are available for formal export")
    staging = None
    try:
        if destination.exists() or destination.is_symlink():
            raise DatasetExportError("Dataset destination already exists")
        records = [_verified_job(_local(output, identifier, directory=True)) for identifier in identifiers]
        staging = Path(tempfile.mkdtemp(prefix="." + destination.name + ".building-", dir=destination.parent))
        (staging / "scenes").mkdir()
        entries = [_export_scene(record, staging / "scenes" / record["folder"].name) for record in records]
        # Validate again before publication. A job/source/asset changing during
        # copying must not create a certified mixed-time snapshot.
        for record in records:
            current = _verified_job(record["folder"])
            if current["result_sha256"] != record["result_sha256"]:
                raise DatasetExportError("Scene result changed during export")
        manifest = {"schema_version": SCHEMA_VERSION, "created_at": utc_now(), "status": "published",
                    "scene_count": len(entries), "view_count": sum(len(e["views"]) for e in entries),
                    "accepted_only": True, "scenes": entries,
                    "quality_scope": "Uncalibrated matched-view model review; not a metric reconstruction certificate"}
        atomic_json(staging / "dataset_manifest.json", manifest)
        (staging / "README.txt").write_text(
            "Accepted scene dataset, scene_dataset_v1\n\n"
            "Open scenes/<scene_id>/scene.blend in Blender. Images are packed by the generating renderer.\n"
            "scene_bundle.json uses local references; rebuild.json lists the renderer inputs.\n"
            "Cropped textures and selected external asset dependencies are included.\n"
            "The renderer is supplied by the scene-factory package, not this dataset.\n"
            "Check dataset_manifest.json and scene_manifest.json for SHA256 inventories.\n"
            "Provider configurations, API logs, prompts and raw model responses are excluded.\n"
            "Acceptance is a heuristic visual review. Scale, hidden surfaces and fidelity are not guaranteed.\n",
            encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise DatasetExportError("Dataset destination appeared during export")
        _publish(staging, destination)
        staging = None
        return manifest
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
