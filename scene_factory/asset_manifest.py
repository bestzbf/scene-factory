"""Strict validation for ``asset_manifest_v1`` asset packages.

The manifest is deliberately validated independently of Blender.  Blender
import/render checks are represented by the quality fields and are required
before an asset can be marked ``asset_accepted``; this module verifies the
machine-readable parts of that contract, file containment and file hashes.

Use the command line entry point with::

    python -m scene_factory.asset_manifest validate /path/to/asset_package

The validator accepts a manifest file or its immediate containing directory
(not a recursive library root) and returns a JSON-serialisable report. It never
changes an asset package. Use build-catalog to validate a whole library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = "asset_manifest_v1"
MANIFEST_FILENAME = "asset_manifest.json"
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
AXES = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
STATUSES = {"draft", "technical_checked", "asset_accepted", "needs_repair", "deprecated"}
SCALE_SOURCES = {"measured", "known_reference", "reference_inferred", "unknown"}
SUPPORT_TYPES = {"floor", "table", "wall", "ceiling", "support", "holder", "free"}
SCOPES = {"single_prop", "assembly", "partial"}
COMPLETENESS = {"complete", "partial", "unknown"}
FORMATS = {"glb", "gltf"}
PRODUCTION_FORMAT = "glb"
# Keep this contract aligned with ExternalAssetImporter._new_material. Future
# material profiles need end-to-end importer support before being admitted.
MAP_NAMES = {"base_color", "normal", "roughness", "metallic", "ao"}
REQUIRED_PREVIEWS = ("front", "side", "back", "three_quarter", "placement_test")
QUALITY_CHECKS = ("file_integrity", "import", "geometry", "uv", "materials", "previews", "placement_test")
PASS_VALUES = {"passed", "pass", True}
PREVIEW_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ManifestError(ValueError):
    """Raised by :func:`validate_manifest_or_raise` for an invalid package."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r}")


def _unique_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=_reject_constant,
                          object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ManifestError(f"invalid manifest JSON at {path}: {exc}") from None


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest(path: Path) -> tuple[Path, Path]:
    path = path.expanduser()
    if path.is_symlink():
        raise ManifestError(f"manifest path cannot be a symlink: {path}")
    path = path.resolve()
    if path.is_dir():
        path = path / MANIFEST_FILENAME
    if path.is_symlink():
        raise ManifestError(f"manifest file cannot be a symlink: {path}")
    if not path.is_file():
        raise ManifestError(f"manifest does not exist: {path}")
    return path, path.parent.resolve()


def _safe_relative(root: Path, value: Any, label: str, *, must_exist: bool,
                   errors: list[dict[str, str]], verify_files: bool = True) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        errors.append({"code": "path_invalid", "path": label, "message": "path must be a non-empty string"})
        return None
    path = Path(value)
    # Backslashes and drive prefixes are rejected even on POSIX so packages
    # cannot be interpreted differently on another host.
    if path.is_absolute() or "\\" in value or ":" in value or any(part in ("", ".", "..") for part in path.parts):
        errors.append({"code": "path_escape", "path": label, "message": "path must be relative and stay below the asset root"})
        return None
    lexical = root / path
    # Reject symlinks even when they resolve to another file inside the root;
    # otherwise the package can change its meaning after validation.
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            errors.append({"code": "path_symlink", "path": label, "message": "symlink files are not allowed in an asset package"})
            return None
    candidate = lexical.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        errors.append({"code": "path_escape", "path": label, "message": "path resolves outside the asset root"})
        return None
    if candidate == root.resolve() or candidate == root.resolve() / MANIFEST_FILENAME:
        errors.append({"code": "path_invalid", "path": label, "message": "asset file path cannot refer to the manifest or root"})
        return None
    if must_exist and not candidate.is_file():
        errors.append({"code": "file_missing", "path": label, "message": f"file does not exist: {path.as_posix()}"})
        return None
    return candidate


def _id(value: Any, label: str, errors: list[dict[str, str]], *, required: bool = True) -> bool:
    if value is None and not required:
        return True
    if not isinstance(value, str) or not value.strip() or not IDENTIFIER_RE.fullmatch(value):
        errors.append({"code": "identifier_invalid", "path": label,
                       "message": "identifier must match [A-Za-z0-9][A-Za-z0-9_.-]*"})
        return False
    return True


def _string(value: Any, label: str, errors: list[dict[str, str]], *, required: bool = True) -> bool:
    if value is None and not required:
        return True
    if not isinstance(value, str) or not value.strip():
        errors.append({"code": "field_invalid", "path": label, "message": "must be a non-empty string"})
        return False
    return True


def _vec(value: Any, label: str, errors: list[dict[str, str]], *, positive: bool = False,
         nonnegative: bool = False) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_finite(v) for v in value):
        errors.append({"code": "vector_invalid", "path": label, "message": "must contain three finite numbers"})
        return False
    if positive and any(float(v) <= 0 for v in value):
        errors.append({"code": "vector_invalid", "path": label, "message": "all dimensions must be positive"})
        return False
    if nonnegative and any(float(v) < 0 for v in value):
        errors.append({"code": "vector_invalid", "path": label, "message": "values must be non-negative"})
        return False
    return True


def _sha(value: Any, label: str, errors: list[dict[str, str]], *, required: bool = True) -> bool:
    if value is None and not required:
        return True
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        errors.append({"code": "hash_invalid", "path": label, "message": "must be a SHA-256 hexadecimal digest"})
        return False
    return True


def _record_error(errors: list[dict[str, str]], code: str, path: str, message: str) -> None:
    errors.append({"code": code, "path": path, "message": message})


def _map_records(raw: Any, label: str, errors: list[dict[str, str]]) -> dict[str, Mapping[str, Any]]:
    """Normalize ``materials``/``material_variants`` list or object forms."""
    if raw is None:
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw, Mapping):
        items = list(raw.items())
        for key, record in items:
            if not isinstance(key, str) or not isinstance(record, Mapping):
                _record_error(errors, "material_invalid", label, "material records must be objects")
                continue
            value = dict(record)
            value.setdefault("material_variant_id", key)
            result[key] = value
    elif isinstance(raw, list):
        for index, record in enumerate(raw):
            if not isinstance(record, Mapping):
                _record_error(errors, "material_invalid", f"{label}[{index}]", "material record must be an object")
                continue
            material_id = record.get("material_variant_id")
            if not isinstance(material_id, str):
                _record_error(errors, "material_invalid", f"{label}[{index}].material_variant_id", "material ID is required")
                continue
            if material_id in result:
                _record_error(errors, "duplicate_id", f"{label}[{index}].material_variant_id", "duplicate material variant ID")
                continue
            result[material_id] = record
    else:
        _record_error(errors, "material_invalid", label, "must be an object or array")
    return result


def _preview_records(raw: Any, label: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        result = dict(raw)
        return result
    if isinstance(raw, list):
        result = {}
        for index, item in enumerate(raw):
            if isinstance(item, str):
                # A list of paths is accepted only when names are conventional.
                name = Path(item).stem
                result[name] = item
            elif isinstance(item, Mapping) and isinstance(item.get("view"), str):
                result[item["view"]] = item
            else:
                _record_error(errors, "preview_invalid", f"{label}[{index}]", "preview must be a path or {view,path} object")
        return result
    _record_error(errors, "preview_invalid", label, "must be an object or array")
    return {}


def _path_hash(root: Path, value: Any, label: str, errors: list[dict[str, str]],
               *, verify_files: bool, required: bool = True) -> None:
    if value is None and not required:
        return
    if not isinstance(value, Mapping):
        _record_error(errors, "map_invalid", label, "texture record must be an object")
        return
    path_value = value.get("path")
    digest = value.get("sha256", value.get("hash"))
    if path_value is None:
        if required:
            _record_error(errors, "path_missing", f"{label}.path", "a texture path is required")
            _record_error(errors, "hash_missing", f"{label}.sha256", "a texture path requires a SHA-256 hash")
        elif digest is not None:
            _record_error(errors, "hash_invalid", f"{label}.sha256", "a missing texture path must have a null hash")
        return
    path = _safe_relative(root, path_value, f"{label}.path", must_exist=verify_files, errors=errors)
    if not _sha(digest, f"{label}.sha256", errors):
        return
    if verify_files and path is not None:
        actual = _sha256(path)
        if actual.lower() != str(digest).lower():
            _record_error(errors, "hash_mismatch", f"{label}.sha256", "texture hash does not match file")


def _validate_materials(data: Mapping[str, Any], root: Path, errors: list[dict[str, str]],
                        *, verify_files: bool) -> dict[str, Mapping[str, Any]]:
    raw = data.get("materials")
    if raw is None:
        raw = data.get("material_variants")
    records = _map_records(raw, "materials", errors)
    ids = data.get("material_variant_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
        _record_error(errors, "field_invalid", "material_variant_ids", "must be a non-empty array of IDs")
        ids = []
    if len(set(ids)) != len(ids):
        _record_error(errors, "duplicate_id", "material_variant_ids", "duplicate material variant IDs")
    for material_id in ids:
        _id(material_id, "material_variant_ids", errors)
        if material_id not in records:
            _record_error(errors, "material_missing", f"materials.{material_id}", "material variant record is missing")
    for material_id in records:
        if material_id not in ids:
            _record_error(errors, "material_orphan", f"materials.{material_id}", "material record is not referenced by material_variant_ids")
    geometry_id = data.get("geometry_variant_id")
    uv = data.get("uv")
    uv_layout_id = uv.get("layout_id") if isinstance(uv, Mapping) else None
    uv_normal = uv.get("normal_convention") if isinstance(uv, Mapping) else None
    for material_id, material in records.items():
        label = f"materials.{material_id}"
        _id(material.get("material_variant_id"), f"{label}.material_variant_id", errors)
        if material.get("material_variant_id") != material_id:
            _record_error(errors, "identifier_mismatch", f"{label}.material_variant_id", "record ID does not match material key")
        multiplier = material.get("base_color_multiplier")
        if multiplier is not None:
            if not isinstance(multiplier, list) or len(multiplier) != 3:
                _record_error(errors, "multiplier_invalid", f"{label}.base_color_multiplier",
                              "must contain three finite values in the range [0, 1]")
            elif any(not _finite(value) or float(value) < 0 or float(value) > 1 for value in multiplier):
                _record_error(errors, "multiplier_invalid", f"{label}.base_color_multiplier",
                              "must contain three finite values in the range [0, 1]")
        compatible = material.get("compatible_geometry_ids", material.get("geometry_compatibility"))
        if not isinstance(compatible, list) or not compatible:
            _record_error(errors, "field_invalid", f"{label}.compatible_geometry_ids", "must be a non-empty array")
        else:
            for item in compatible:
                _id(item, f"{label}.compatible_geometry_ids", errors)
            if geometry_id not in compatible:
                _record_error(errors, "material_incompatible", label, "material does not declare this geometry variant")
        material_uv = material.get("uv_layout_id")
        if material_uv is not None and material_uv != uv_layout_id:
            _record_error(errors, "uv_incompatible", f"{label}.uv_layout_id", "material UV layout differs from asset UV layout")
        normal = material.get("normal_convention")
        if normal != "OpenGL_Y+":
            _record_error(errors, "normal_convention_invalid", f"{label}.normal_convention", "must be OpenGL_Y+")
        elif uv_normal is not None and uv_normal != normal:
            _record_error(errors, "normal_convention_mismatch", f"{label}.normal_convention", "does not match uv.normal_convention")
        maps = material.get("maps", {})
        if not isinstance(maps, Mapping):
            _record_error(errors, "map_invalid", f"{label}.maps", "maps object is required")
            maps = {}
        slot_materials = material.get("slot_materials", {})
        if not isinstance(slot_materials, Mapping):
            _record_error(errors, "map_invalid", f"{label}.slot_materials", "slot_materials must be an object")
            slot_materials = {}
        if not maps and not slot_materials:
            _record_error(errors, "map_missing", f"{label}.maps", "maps or slot_materials with base_color is required")

        def validate_map_set(map_set: Any, map_label: str) -> None:
            if not isinstance(map_set, Mapping):
                _record_error(errors, "map_invalid", map_label, "maps must be an object")
                return
            if not any(str(name).lower().replace(" ", "_") == "base_color" for name in map_set):
                _record_error(errors, "map_missing", map_label, "base_color map is required")
            seen_names: set[str] = set()
            for map_name, record in map_set.items():
                canonical = str(map_name).lower().replace(" ", "_")
                if canonical not in MAP_NAMES:
                    _record_error(errors, "map_unknown", f"{map_label}.{map_name}", "unsupported texture map name")
                    continue
                if canonical in seen_names:
                    _record_error(errors, "map_duplicate", f"{map_label}.{map_name}", "texture map is declared more than once")
                seen_names.add(canonical)
                _path_hash(root, record, f"{map_label}.{map_name}", errors, verify_files=verify_files,
                           required=(canonical == "base_color"))
                if isinstance(record, Mapping):
                    expected_space = "sRGB" if canonical == "base_color" else "Non-Color"
                    if record.get("color_space") != expected_space:
                        _record_error(errors, "color_space_invalid", f"{map_label}.{map_name}.color_space",
                                      f"{canonical} maps must use {expected_space}")

        validate_map_set(maps, f"{label}.maps") if maps else None
        for slot_name, slot_maps in slot_materials.items():
            if not isinstance(slot_name, str) or not slot_name.strip():
                _record_error(errors, "slot_invalid", f"{label}.slot_materials", "slot names must be non-empty strings")
                continue
            validate_map_set(slot_maps, f"{label}.slot_materials.{slot_name}")
    return records


def _validate_placement(data: Mapping[str, Any], root: Path, errors: list[dict[str, str]],
                        *, verify_files: bool) -> None:
    placement = data.get("placement")
    if not isinstance(placement, Mapping):
        _record_error(errors, "field_missing", "placement", "placement object is required")
        return
    support = placement.get("support_type")
    if support not in SUPPORT_TYPES:
        _record_error(errors, "support_invalid", "placement.support_type", f"must be one of {sorted(SUPPORT_TYPES)}")
    _string(placement.get("anchor_type"), "placement.anchor_type", errors)
    _vec(placement.get("anchor_point_m"), "placement.anchor_point_m", errors)
    contacts = placement.get("contact_points_m", [])
    if contacts is None:
        contacts = []
    if not isinstance(contacts, list):
        _record_error(errors, "field_invalid", "placement.contact_points_m", "must be an array")
    else:
        for index, point in enumerate(contacts):
            _vec(point, f"placement.contact_points_m[{index}]", errors)
    mounts = placement.get("mount_points", [])
    if mounts is None:
        mounts = []
    if not isinstance(mounts, list):
        _record_error(errors, "field_invalid", "placement.mount_points", "must be an array")
    else:
        for index, point in enumerate(mounts):
            _vec(point, f"placement.mount_points[{index}]", errors)
    # A floor asset may use its bottom-center anchor as the contact contract;
    # tabletop placement needs an explicit contact surface because the support
    # height is inferred from that surface rather than from the asset origin.
    if support == "table" and not contacts:
        _record_error(errors, "contact_missing", "placement.contact_points_m", f"{support} assets require contact points")
    if support in {"wall", "ceiling", "holder"} and not mounts:
        _record_error(errors, "mount_missing", "placement.mount_points", f"{support} assets require mount points")
    clearance = placement.get("clearance_m", 0.0)
    if not _finite(clearance) or float(clearance) < 0:
        _record_error(errors, "field_invalid", "placement.clearance_m", "must be a non-negative finite number")
    for key in ("can_rotate_z", "allow_uniform_scale", "allow_non_uniform_scale"):
        if type(placement.get(key)) is not bool:
            _record_error(errors, "field_invalid", f"placement.{key}", "must be boolean")
    if placement.get("allow_non_uniform_scale") is True:
        _record_error(errors, "non_uniform_scale_forbidden", "placement.allow_non_uniform_scale", "must be false")
    collision = placement.get("collision_path")
    if collision is not None:
        collision_path = _safe_relative(root, collision, "placement.collision_path", must_exist=verify_files, errors=errors)
        collision_hash = placement.get("collision_sha256")
        if collision_hash is None:
            _record_error(errors, "hash_missing", "placement.collision_sha256", "collision mesh requires a SHA-256 hash")
        elif _sha(collision_hash, "placement.collision_sha256", errors) and verify_files and collision_path is not None:
            if _sha256(collision_path).lower() != str(collision_hash).lower():
                _record_error(errors, "hash_mismatch", "placement.collision_sha256", "collision hash does not match file")


def _validate_parts(data: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    semantics = data.get("semantics")
    scope = semantics.get("scope") if isinstance(semantics, Mapping) else None
    parts = data.get("parts", [])
    if parts is None:
        parts = []
    if not isinstance(parts, list):
        _record_error(errors, "parts_invalid", "parts", "must be an array")
        return
    seen: set[str] = set()
    for index, part in enumerate(parts):
        label = f"parts[{index}]"
        if not isinstance(part, Mapping):
            _record_error(errors, "parts_invalid", label, "part must be an object")
            continue
        part_id = part.get("part_id", part.get("id"))
        _id(part_id, f"{label}.part_id", errors)
        if isinstance(part_id, str):
            if part_id in seen:
                _record_error(errors, "duplicate_id", f"{label}.part_id", "duplicate part ID")
            seen.add(part_id)
        _string(part.get("name", part_id), f"{label}.name", errors)
        parent = part.get("parent_id")
        if parent is not None and not isinstance(parent, str):
            _record_error(errors, "parts_invalid", f"{label}.parent_id", "parent ID must be a string")
        if isinstance(parent, str) and parent == part_id:
            _record_error(errors, "parts_cycle", f"{label}.parent_id", "part cannot parent itself")
    for index, part in enumerate(parts):
        if isinstance(part, Mapping) and isinstance(part.get("parent_id"), str) and part["parent_id"] not in seen:
            _record_error(errors, "parts_parent_missing", f"parts[{index}].parent_id", "parent part does not exist")
    parents = {part.get("part_id", part.get("id")): part.get("parent_id")
               for part in parts if isinstance(part, Mapping) and isinstance(part.get("part_id", part.get("id")), str)}
    visited: set[str] = set()
    active: set[str] = set()

    def visit(part_id: str) -> None:
        if part_id in active:
            _record_error(errors, "parts_cycle", "parts", "part parent graph contains a cycle")
            return
        if part_id in visited:
            return
        active.add(part_id)
        parent = parents.get(part_id)
        if isinstance(parent, str) and parent in parents:
            visit(parent)
        active.remove(part_id)
        visited.add(part_id)

    for part_id in parents:
        visit(part_id)
    if scope == "assembly" and not parts:
        _record_error(errors, "parts_missing", "parts", "assembly assets require an explicit part tree")


def _validate_previews(data: Mapping[str, Any], root: Path, errors: list[dict[str, str]], *, verify_files: bool,
                       require_hashes: bool = False) -> None:
    previews = _preview_records(data.get("previews"), "previews", errors)
    if not previews:
        _record_error(errors, "previews_missing", "previews", "previews object is required")
        return
    seen_paths: set[str] = set()
    for name in REQUIRED_PREVIEWS:
        record = previews.get(name)
        value = record.get("path") if isinstance(record, Mapping) else record
        if value is None:
            _record_error(errors, "preview_missing", f"previews.{name}", "required preview is missing")
            continue
        path = _safe_relative(root, value, f"previews.{name}", must_exist=verify_files, errors=errors)
        if path is not None and require_hashes:
            path_key = path.as_posix()
            if path_key in seen_paths:
                _record_error(errors, "preview_duplicate", f"previews.{name}",
                              "asset_accepted previews must use distinct files")
            seen_paths.add(path_key)
            if path.suffix.lower() not in PREVIEW_SUFFIXES:
                _record_error(errors, "preview_type_invalid", f"previews.{name}",
                              "asset_accepted previews must use a supported image file extension")
        digest = record.get("sha256") if isinstance(record, Mapping) else None
        if require_hashes and not isinstance(record, Mapping):
            _record_error(errors, "preview_hash_missing", f"previews.{name}.sha256",
                          "asset_accepted requires each preview to include a SHA-256 hash")
        elif require_hashes and digest is None:
            _record_error(errors, "preview_hash_missing", f"previews.{name}.sha256",
                          "asset_accepted requires each preview to include a SHA-256 hash")
        if digest is not None:
            if _sha(digest, f"previews.{name}.sha256", errors) and verify_files and path is not None:
                if _sha256(path).lower() != str(digest).lower():
                    _record_error(errors, "hash_mismatch", f"previews.{name}.sha256", "preview hash does not match file")


def _validate_provenance(data: Mapping[str, Any], root: Path, errors: list[dict[str, str]], *, verify_files: bool,
                         required: bool) -> None:
    provenance = data.get("provenance")
    if not isinstance(provenance, Mapping):
        if required:
            _record_error(errors, "provenance_missing", "provenance", "provenance object is required")
        return
    for key in ("generator", "generator_version", "license", "created_at"):
        _string(provenance.get(key), f"provenance.{key}", errors, required=required)
    images = provenance.get("source_images")
    if required and (not isinstance(images, list) or not images):
        _record_error(errors, "provenance_invalid", "provenance.source_images", "must be a non-empty array")
    if isinstance(images, list):
        for index, image in enumerate(images):
            _safe_relative(root, image, f"provenance.source_images[{index}]", must_exist=verify_files, errors=errors)
    parameters_hash = provenance.get("generation_parameters_sha256")
    if parameters_hash is not None or required:
        _sha(parameters_hash, "provenance.generation_parameters_sha256", errors, required=required)


def _production_gate(data: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    status = data.get("status")
    quality = data.get("quality") if isinstance(data.get("quality"), Mapping) else {}
    if status == "asset_accepted":
        if quality.get("production_ready") is not True:
            _record_error(errors, "production_gate", "quality.production_ready", "asset_accepted requires production_ready=true")
        for key in QUALITY_CHECKS:
            value = quality.get(key)
            if not (value is True or (isinstance(value, str) and value in {"passed", "pass"})):
                _record_error(errors, "production_gate", f"quality.{key}", "must be passed for asset_accepted")
        model = data.get("model") if isinstance(data.get("model"), Mapping) else {}
        if str(model.get("format", "")).lower() != PRODUCTION_FORMAT:
            _record_error(errors, "production_gate", "model.format", "asset_accepted requires model.format=glb")
        if Path(str(model.get("path", ""))).suffix.lower() != ".glb":
            _record_error(errors, "production_gate", "model.path", "asset_accepted requires a .glb model file")
        if model.get("scale_source") == "unknown":
            _record_error(errors, "production_gate", "model.scale_source", "unknown scale cannot be asset_accepted")
        semantics = data.get("semantics") if isinstance(data.get("semantics"), Mapping) else {}
        if semantics.get("scope") == "partial" or semantics.get("completeness") != "complete":
            _record_error(errors, "production_gate", "semantics.completeness", "asset_accepted requires complete geometry")
        placement = data.get("placement") if isinstance(data.get("placement"), Mapping) else {}
        if placement.get("allow_uniform_scale") is not True:
            _record_error(errors, "production_gate", "placement.allow_uniform_scale",
                          "asset_accepted requires allow_uniform_scale=true because the runtime only applies uniform scale")
        if placement.get("allow_non_uniform_scale") is not False:
            _record_error(errors, "production_gate", "placement.allow_non_uniform_scale", "must be false")
        if not isinstance(placement.get("collision_path"), str):
            _record_error(errors, "production_gate", "placement.collision_path", "asset_accepted requires collision mesh")
        if not isinstance(placement.get("collision_sha256"), str) or not SHA256_RE.fullmatch(placement["collision_sha256"]):
            _record_error(errors, "production_gate", "placement.collision_sha256", "asset_accepted requires collision_sha256")
        model = data.get("model") if isinstance(data.get("model"), Mapping) else {}
        transform = model.get("transform")
        if not isinstance(transform, Mapping):
            _record_error(errors, "production_gate", "model.transform",
                          "asset_accepted requires applied-transform evidence")
        else:
            for key in ("location", "rotation", "scale"):
                if key not in transform:
                    _record_error(errors, "production_gate", f"model.transform.{key}",
                                  "asset_accepted requires location, rotation and scale")
    elif quality.get("production_ready") is True:
        _record_error(errors, "status_invalid", "quality.production_ready", "only asset_accepted may be production_ready")


def validate_manifest(path: str | Path, *, verify_hashes: bool = True,
                      require_production: bool = False) -> dict[str, Any]:
    """Validate a manifest file or asset directory.

    The returned report has ``valid``, ``production_ready``, ``errors`` and
    ``warnings`` keys.  ``valid`` means the full manifest contract is valid;
    ``production_ready`` is only true for an ``asset_accepted`` manifest with
    no errors.  ``require_production`` additionally rejects any non-accepted
    status, which is useful for importers and dataset assembly.
    """
    try:
        manifest_path, root = _resolve_manifest(Path(path))
    except ManifestError as exc:
        return {"valid": False, "production_ready": False, "manifest_path": str(Path(path)),
                "asset_root": str(Path(path)), "errors": [{"code": "manifest_missing", "path": str(path), "message": str(exc)}],
                "warnings": []}
    try:
        data = _json_load(manifest_path)
    except ManifestError as exc:
        return {"valid": False, "production_ready": False, "manifest_path": str(manifest_path),
                "asset_root": str(root), "errors": [{"code": "json_invalid", "path": str(manifest_path), "message": str(exc)}],
                "warnings": []}
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not verify_hashes:
        warnings.append({"code": "files_unverified", "path": "$",
                         "message": "file existence and hashes were not checked; this is structural validation only"})
        if require_production:
            _record_error(errors, "production_gate", "$", "production validation requires file and hash checks")
    if not isinstance(data, Mapping):
        _record_error(errors, "root_invalid", "$", "manifest root must be an object")
        return {"valid": False, "production_ready": False, "manifest_path": str(manifest_path),
                "asset_root": str(root), "errors": errors, "warnings": warnings}
    if data.get("schema_version") != SCHEMA_VERSION:
        _record_error(errors, "schema_invalid", "schema_version", f"must equal {SCHEMA_VERSION}")
    for key in ("asset_id", "family_id", "geometry_variant_id"):
        _id(data.get(key), key, errors)
    status = data.get("status")
    if status not in STATUSES:
        _record_error(errors, "status_invalid", "status", f"must be one of {sorted(STATUSES)}")
    model = data.get("model")
    if not isinstance(model, Mapping):
        _record_error(errors, "field_missing", "model", "model object is required")
        model = {}
    model_format = model.get("format")
    if not isinstance(model_format, str) or model_format.lower() not in FORMATS:
        _record_error(errors, "format_invalid", "model.format", "must be glb or gltf")
    model_path = _safe_relative(root, model.get("path"), "model.path", must_exist=verify_hashes, errors=errors)
    if isinstance(model.get("path"), str):
        suffix = Path(model["path"]).suffix.lower().lstrip(".")
        if suffix and suffix not in FORMATS:
            _record_error(errors, "format_invalid", "model.path", "model path extension must be .glb or .gltf")
        if isinstance(model_format, str) and suffix and suffix != model_format.lower():
            _record_error(errors, "format_mismatch", "model.path", "model format must match the file extension")
    model_hash_ok = _sha(model.get("sha256"), "model.sha256", errors)
    if verify_hashes and model_hash_ok and model_path is not None and _sha256(model_path).lower() != str(model["sha256"]).lower():
        _record_error(errors, "hash_mismatch", "model.sha256", "model hash does not match file")
    dimensions_units_ok = _vec(model.get("dimensions_import_units"), "model.dimensions_import_units", errors, positive=True)
    dimensions_m_ok = _vec(model.get("dimensions_m"), "model.dimensions_m", errors, positive=True)
    meters = model.get("meters_per_import_unit")
    if not _finite(meters) or float(meters) <= 0:
        _record_error(errors, "scale_invalid", "model.meters_per_import_unit", "must be a positive finite number")
    elif dimensions_units_ok and dimensions_m_ok:
        expected = [float(value) * float(meters) for value in model["dimensions_import_units"]]
        if any(not math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-8)
               for a, b in zip(expected, model["dimensions_m"])):
            _record_error(errors, "dimensions_mismatch", "model.dimensions_m", "dimensions_m must equal import dimensions multiplied by meters_per_import_unit")
    scale_source = model.get("scale_source")
    if scale_source not in SCALE_SOURCES:
        _record_error(errors, "scale_source_invalid", "model.scale_source", f"must be one of {sorted(SCALE_SOURCES)}")
    confidence = model.get("scale_confidence")
    if not _finite(confidence) or not 0 <= float(confidence) <= 1:
        _record_error(errors, "confidence_invalid", "model.scale_confidence", "must be between 0 and 1")
    if model.get("canonical_up_axis") != "+Z":
        _record_error(errors, "axis_invalid", "model.canonical_up_axis", "must be +Z")
    if model.get("canonical_front_axis") != "-Y":
        _record_error(errors, "axis_invalid", "model.canonical_front_axis", "must be -Y")
    for key in ("source_up_axis", "source_front_axis"):
        if model.get(key) is not None and model.get(key) not in AXES:
            _record_error(errors, "axis_invalid", f"model.{key}", "must be a signed coordinate axis")
    for key in ("triangle_count", "object_count"):
        value = model.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            _record_error(errors, "count_invalid", f"model.{key}", "must be a positive integer")
    transform = model.get("transform")
    if transform is not None:
        if not isinstance(transform, Mapping):
            _record_error(errors, "transform_invalid", "model.transform", "must be an object")
        else:
            for key, expected in (("location", [0, 0, 0]), ("rotation", [0, 0, 0]), ("scale", [1, 1, 1])):
                if key not in transform or not _vec(transform.get(key), f"model.transform.{key}", errors):
                    continue
                if any(not math.isclose(float(a), float(b), abs_tol=1e-6) for a, b in zip(transform[key], expected)):
                    _record_error(errors, "transform_invalid", f"model.transform.{key}", f"must be {expected}")
    semantics = data.get("semantics")
    if not isinstance(semantics, Mapping):
        _record_error(errors, "field_missing", "semantics", "semantics object is required")
        semantics = {}
    for key in ("category", "description", "source"):
        _string(semantics.get(key), f"semantics.{key}", errors)
    tags = semantics.get("tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        _record_error(errors, "field_invalid", "semantics.tags", "must be an array of non-empty strings")
    if semantics.get("scope") not in SCOPES:
        _record_error(errors, "field_invalid", "semantics.scope", f"must be one of {sorted(SCOPES)}")
    if semantics.get("completeness") not in COMPLETENESS:
        _record_error(errors, "field_invalid", "semantics.completeness", f"must be one of {sorted(COMPLETENESS)}")
    semantic_confidence = semantics.get("confidence")
    if not _finite(semantic_confidence) or not 0 <= float(semantic_confidence) <= 1:
        _record_error(errors, "confidence_invalid", "semantics.confidence", "must be between 0 and 1")
    uv = data.get("uv")
    if not isinstance(uv, Mapping):
        _record_error(errors, "field_missing", "uv", "uv object is required")
        uv = {}
    _string(uv.get("layout_id"), "uv.layout_id", errors)
    channels = uv.get("channels")
    if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1 or channels > 8:
        _record_error(errors, "uv_invalid", "uv.channels", "must be an integer from 1 to 8")
    if uv.get("normal_convention") != "OpenGL_Y+":
        _record_error(errors, "normal_convention_invalid", "uv.normal_convention", "must be OpenGL_Y+")
    _string(uv.get("texel_density_policy"), "uv.texel_density_policy", errors)
    _validate_placement(data, root, errors, verify_files=verify_hashes)
    _validate_parts(data, errors)
    _validate_materials(data, root, errors, verify_files=verify_hashes)
    _validate_previews(data, root, errors, verify_files=verify_hashes,
                       require_hashes=(status == "asset_accepted"))
    _validate_provenance(data, root, errors, verify_files=verify_hashes,
                         required=(status == "asset_accepted"))
    quality = data.get("quality")
    if not isinstance(quality, Mapping):
        _record_error(errors, "field_missing", "quality", "quality object is required")
    elif not isinstance(quality.get("production_ready"), bool):
        _record_error(errors, "field_invalid", "quality.production_ready", "must be boolean")
    _production_gate(data, errors)
    if require_production and status != "asset_accepted":
        _record_error(errors, "production_gate", "status", "production validation requires status=asset_accepted")
    if status in {"draft", "needs_repair"}:
        warnings.append({"code": "not_production_ready", "path": "status", "message": f"status={status} is not callable in production"})
    production_ready = status == "asset_accepted" and not errors and verify_hashes
    return {"valid": not errors, "production_ready": production_ready,
            "manifest_path": str(manifest_path), "asset_root": str(root),
            "asset_id": data.get("asset_id"), "status": status,
            "errors": errors, "warnings": warnings}


def validate_manifest_or_raise(path: str | Path, *, verify_hashes: bool = True,
                               require_production: bool = False) -> dict[str, Any]:
    """Validate and raise :class:`ManifestError` when the package is invalid."""
    report = validate_manifest(path, verify_hashes=verify_hashes, require_production=require_production)
    if not report["valid"]:
        summary = "; ".join(f"{item['path']}: {item['message']}" for item in report["errors"][:5])
        if len(report["errors"]) > 5:
            summary += f"; and {len(report['errors']) - 5} more errors"
        raise ManifestError(summary or "manifest validation failed")
    return report


def _relative_package_file(asset_root: Path, package_root: Path, value: str, label: str) -> str:
    """Return a package file path relative to the catalog root."""
    candidate = (package_root / value).resolve()
    try:
        return candidate.relative_to(asset_root.resolve()).as_posix()
    except ValueError as exc:
        raise ManifestError(f"{label} resolves outside asset root") from exc


def _catalog_map_record(asset_root: Path, package_root: Path, record: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = record.get("path")
    if not isinstance(path, str):
        raise ManifestError(f"{label}.path is required")
    return {"path": _relative_package_file(asset_root, package_root, path, label),
            "color_space": record.get("color_space"), "hash": record.get("sha256", record.get("hash"))}


def _discover_manifests(root: Path) -> list[Path]:
    """Discover packages without silently omitting linked manifests or folders."""
    manifests: list[Path] = []

    def traversal_error(error: OSError) -> None:
        raise ManifestError(f"cannot inspect asset root: {error}") from error

    for directory, subdirectories, filenames in os.walk(root, followlinks=False, onerror=traversal_error):
        current = Path(directory)
        for name in subdirectories:
            candidate = current / name
            if candidate.is_symlink():
                raise ManifestError(f"asset root cannot contain symlink directories: {candidate}")
            if name == MANIFEST_FILENAME:
                raise ManifestError(f"manifest must be a file: {candidate}")
        if MANIFEST_FILENAME in filenames:
            # Include symlinks, including broken links: validate_manifest must
            # reject them rather than building an apparently complete catalog.
            manifests.append(current / MANIFEST_FILENAME)
    return sorted(manifests)


def build_catalog(asset_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Validate all packages below ``asset_root`` and write ``asset_catalog_v1``.

    Only manifests that pass the production gate are included.  The function is
    intentionally fail-closed: one invalid package aborts the build instead of
    producing a partial catalog that looks complete to downstream workers.
    """
    root = Path(asset_root).expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"asset root is not a directory: {root}")
    manifests = _discover_manifests(root)
    if not manifests:
        raise ManifestError(f"no {MANIFEST_FILENAME} found below {root}")
    reports = [validate_manifest(path, require_production=True) for path in manifests]
    failures = [report for report in reports if not report["valid"]]
    if failures:
        details = []
        for report in failures:
            details.append(f"{report.get('manifest_path')}: " + "; ".join(item["message"] for item in report["errors"][:4]))
        raise ManifestError("catalog build rejected invalid packages: " + " | ".join(details))
    assets: list[dict[str, Any]] = []
    materials: dict[str, dict[str, Any]] = {}
    asset_ids: set[str] = set()
    geometry_ids: set[str] = set()
    for manifest_path in manifests:
        package_root = manifest_path.parent.resolve()
        data = _json_load(manifest_path)
        asset_id = data["asset_id"]
        geometry_id = data["geometry_variant_id"]
        if asset_id in asset_ids or geometry_id in geometry_ids:
            raise ManifestError(f"duplicate asset or geometry ID: {asset_id}")
        asset_ids.add(asset_id); geometry_ids.add(geometry_id)
        model = data["model"]
        placement = data["placement"]
        semantics = data["semantics"]
        quality = data["quality"]
        model_file = _relative_package_file(root, package_root, model["path"], f"{asset_id}.model")
        collision_file = _relative_package_file(root, package_root, placement["collision_path"], f"{asset_id}.collision")
        manifest_relative = _relative_package_file(root, package_root, MANIFEST_FILENAME, f"{asset_id}.manifest")
        manifest_digest = _sha256(manifest_path)
        material_ids = list(data["material_variant_ids"])
        asset = {
            "asset_id": asset_id, "family_id": data["family_id"], "geometry_variant_id": geometry_id,
            "model_file": model_file, "model_format": model["format"].lower(), "hash": model["sha256"],
            "dimensions_import_units": list(model["dimensions_import_units"]), "dimensions_m": list(model["dimensions_m"]),
            "meters_per_unit": model["meters_per_import_unit"], "front_axis": "-Y", "up_axis": "+Z",
            "symmetric_front": False, "status": "approved", "source_name": data.get("source_name"),
            "placement": placement["support_type"],
            "anchor": {"type": placement["anchor_type"], "point": list(placement["anchor_point_m"])},
            "placement_rules": {
                "anchor_type": placement["anchor_type"],
                "anchor_point_m": list(placement["anchor_point_m"]),
                "contact_points_m": [list(point) for point in placement.get("contact_points_m", [])],
                "mount_points": [list(point) for point in placement.get("mount_points", [])],
                "clearance_m": placement.get("clearance_m", 0.0),
                "can_rotate_z": placement["can_rotate_z"],
                "allow_uniform_scale": placement["allow_uniform_scale"],
                "allow_non_uniform_scale": placement["allow_non_uniform_scale"],
            },
            "collision_file": collision_file, "collision_hash": placement.get("collision_sha256"),
            "scope": semantics["scope"], "category": semantics["category"], "tags": list(semantics.get("tags", [])),
            "issues": [], "quality": {"complete_for_floor_placement": semantics["completeness"] == "complete",
                                        "production_ready": bool(quality.get("production_ready"))},
            "material_variant_ids": material_ids, "manifest_file": manifest_relative,
            "audit_provenance": {"manifest_sha256": manifest_digest, "manifest_status": data["status"],
                                  "manifest_gate": True,
                                  "provenance": data.get("provenance", {})},
        }
        assets.append(asset)
        raw_materials = data.get("materials", data.get("material_variants", {}))
        if isinstance(raw_materials, list):
            raw_materials = {item["material_variant_id"]: item for item in raw_materials}
        for material_id in material_ids:
            source = raw_materials[material_id]
            material = {
                "material_variant_id": material_id,
                "compatible_geometry_ids": list(source.get("compatible_geometry_ids", source.get("geometry_compatibility", []))),
                "uv_layout_id": source.get("uv_layout_id", data["uv"]["layout_id"]),
                "normal_convention": source.get("normal_convention", "OpenGL_Y+"),
                "maps": {}, "slot_materials": {}, "appearance_tags": list(source.get("appearance_tags", [])),
                "audit_provenance": {"manifest_sha256": manifest_digest},
            }
            if source.get("base_color_multiplier") is not None:
                material["base_color_multiplier"] = list(source["base_color_multiplier"])
            for map_name, record in source.get("maps", {}).items():
                if record.get("path"):
                    canonical_name = str(map_name).lower().replace(" ", "_")
                    material["maps"][canonical_name] = _catalog_map_record(root, package_root, record,
                                                                        f"{asset_id}.{material_id}.{map_name}")
            for slot_name, maps in source.get("slot_materials", {}).items():
                material["slot_materials"][slot_name] = {
                    str(map_name).lower().replace(" ", "_"): _catalog_map_record(root, package_root, record,
                                                   f"{asset_id}.{material_id}.{slot_name}.{map_name}")
                    for map_name, record in maps.items() if record.get("path")
                }
            if material_id in materials and materials[material_id] != material:
                raise ManifestError(f"material variant is defined differently by multiple assets: {material_id}")
            materials[material_id] = material
    catalog = {
        "schema_version": "asset_catalog_v1", "root": str(root),
        "source_manifests": [
            _relative_package_file(root, path.parent.resolve(), MANIFEST_FILENAME, "source_manifest")
            for path in manifests
        ], "assets": assets,
        "material_variants": materials,
        "selection_policy": {"approved_only_by_default": True, "avoid_duplicate_variants": True,
                              "manifest_gate": True},
    }
    destination = Path(output_path).expanduser().resolve()
    if destination == root or root in destination.parents:
        raise ManifestError("catalog output must be outside asset root")
    # Validate the assembled graph with the actual consumer, before creating
    # or replacing any file. Individual manifests cannot prove that referenced
    # geometry variants exist elsewhere in this batch.
    try:
        from .asset_catalog import AssetCatalog, CatalogError
    except ImportError:
        from asset_catalog import AssetCatalog, CatalogError
    try:
        AssetCatalog(catalog, catalog_path=destination)
    except CatalogError as exc:
        raise ManifestError(f"assembled catalog failed runtime validation: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return catalog


def validate_asset_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible descriptive alias for :func:`validate_manifest`."""
    return validate_manifest(*args, **kwargs)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a strict asset_manifest_v1 package")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate one asset package or asset_manifest.json (not recursive)")
    validate.add_argument("path", type=Path)
    validate.add_argument("--no-hash", action="store_true", help="skip file existence and SHA-256 checks")
    validate.add_argument("--require-production", action="store_true", help="require status=asset_accepted")
    build = commands.add_parser("build-catalog", help="validate all packages and write asset_catalog_v1")
    build.add_argument("asset_root", type=Path)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        report = validate_manifest(args.path, verify_hashes=not args.no_hash,
                                   require_production=args.require_production)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["valid"] else 1
    if args.command == "build-catalog":
        try:
            catalog = build_catalog(args.asset_root, args.output)
        except (ManifestError, OSError) as exc:
            print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False, indent=2))
            return 1
        print(json.dumps({"valid": True, "catalog": str(args.output.expanduser().resolve()),
                          "assets": len(catalog["assets"])}, ensure_ascii=False, indent=2))
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
