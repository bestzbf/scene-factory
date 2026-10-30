"""Small, strict asset catalog for bulk scene generation.

The catalog deliberately separates measured import-space geometry from physical
scale and semantic metadata.  An asset can therefore be useful for preview
renders while still being refused by the production resolver.

Catalog files are JSON and use paths relative to ``root``.  The source audit
file is never modified by :func:`audit_draft_to_catalog`.

Minimal v1 structure::

    {"schema_version": "asset_catalog_v1", "root": "/assets",
     "assets": [{"asset_id": "chair_01", "family_id": "chair",
                 "geometry_variant_id": "chair_shape_01", "model_file": "chair.fbx",
                 "hash": "<SHA-256>", "dimensions_import_units": [1, 1, 2],
                 "meters_per_unit": null, "front_axis": null, "symmetric_front": false,
                 "status": "draft", "category": "chair", "placement": "floor",
                 "scope": "single_prop", "issues": [],
                 "quality": {"complete_for_floor_placement": null},
                 "material_variant_ids": ["chair_material_01"], "audit_provenance": {}}],
     "material_variants": {"chair_material_01": {
         "material_variant_id": "chair_material_01",
         "compatible_geometry_ids": ["chair_shape_01"],
         "normal_convention": null, "normal_convention_status": "unknown",
         "maps": {"Base Color": {"path": "color.png", "color_space": "sRGB", "hash": "<SHA-256>"}},
         "audit_provenance": {}}}}

Load with ``load_catalog(path)``; use one ``catalog.resolver(seed=42)`` per batch
to resolve families into compatible geometry/material pairs. Models/textures
are hash-verified, but the catalog is not a cryptographic approval authority:
its approved status must come from a trusted upstream asset review workflow.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "asset_catalog_v1"
_STATUSES = {"draft", "approved", "technical_ready"}
_HASH_CHARS = set("0123456789abcdef")
_AXES = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CatalogError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - _HASH_CHARS


class CatalogError(ValueError):
    """Raised when a catalog is malformed or unsafe to use."""


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
                          object_pairs_hook=_unique_keys)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"invalid JSON at {path}:{exc.lineno}:{exc.colno}") from None


def _reject_constant(value: str) -> None:
    raise CatalogError(f"non-finite JSON value {value!r}")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CatalogError(f"{label} must be a finite number")
    return float(value)


def _safe_relative(value: str, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise CatalogError(f"{label} must be a non-empty path")
    path = Path(value)
    if "\\" in value or ":" in value or path.is_absolute() or any(part == ".." for part in path.parts):
        raise CatalogError(f"{label} must stay below catalog root")
    return path


def _relative_path(root: Path, value: str, label: str, must_exist: bool = False) -> str:
    relative = _safe_relative(value, label)
    current = root.resolve()
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise CatalogError(f"{label} cannot use symlink paths")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CatalogError(f"{label} resolves outside catalog root") from exc
    if must_exist and not candidate.is_file():
        raise CatalogError(f"{label} does not exist: {relative}")
    return relative.as_posix()


def _normalise_maps(root: Path, maps: Mapping[str, Any], label: str, verify_files: bool) -> Dict[str, Dict[str, Any]]:
    if not isinstance(maps, Mapping):
        raise CatalogError(f"{label} must be an object")
    result: Dict[str, Dict[str, Any]] = {}
    for map_name, record in maps.items():
        if not isinstance(map_name, str) or not isinstance(record, Mapping):
            raise CatalogError(f"{label} contains an invalid map record")
        path = record.get("path")
        if path is not None:
            path = _relative_path(root, path, f"{label}.{map_name}.path", verify_files)
        color_space = record.get("color_space")
        if color_space not in ("sRGB", "Non-Color"):
            raise CatalogError(f"{label}.{map_name} has unsupported color_space")
        digest = sha256_file(root / path) if verify_files and path else record.get("hash", record.get("sha256"))
        result[map_name] = {"path": path, "color_space": color_space, "hash": digest}
    return result


def _family_id(asset: Mapping[str, Any]) -> str:
    semantic = asset.get("semantic_description") or {}
    category = semantic.get("category")
    if isinstance(category, str) and category.strip():
        return category.strip()
    asset_id = str(asset.get("asset_id", "unknown"))
    return f"unclassified_{asset_id}"


def _asset_from_draft(raw: Mapping[str, Any], root: Path, source_digest: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    asset_id = raw.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise CatalogError("draft asset requires asset_id")
    model_file = _relative_path(root, raw.get("model_file"), f"assets[{asset_id}].model_file", True)
    semantic = raw.get("semantic_description") or {}
    geometry = raw.get("geometry_measured") or {}
    physical = raw.get("physical_scale") or {}
    orientation = raw.get("orientation") or {}
    placement = raw.get("placement") or {}
    quality = raw.get("quality") or {}
    dimensions = geometry.get("dimensions_units")
    if not isinstance(dimensions, list) or len(dimensions) != 3:
        raise CatalogError(f"assets[{asset_id}].geometry_measured.dimensions_units must have 3 values")
    dimensions = [_finite_number(v, f"assets[{asset_id}].dimensions_import_units") for v in dimensions]
    model_hash = raw.get("model_sha256")
    if not isinstance(model_hash, str) or len(model_hash) != 64 or set(model_hash.lower()) - _HASH_CHARS:
        raise CatalogError(f"assets[{asset_id}].model_sha256 is not a SHA-256 digest")
    actual_hash = sha256_file(root / model_file)
    if actual_hash.lower() != model_hash.lower():
        raise CatalogError(f"assets[{asset_id}] model hash does not match {model_file}")
    family_id = _family_id(raw)
    geometry_id = f"{family_id}__geometry__{asset_id}"
    material_id = f"{family_id}__material__{asset_id}"
    maps = _normalise_maps(root, (raw.get("materials") or {}).get("maps", {}), f"assets[{asset_id}].materials.maps", True)
    material = {
        "material_variant_id": material_id,
        "maps": maps,
        "compatible_geometry_ids": [geometry_id],
        "normal_convention": None,
        "normal_convention_status": "unknown",
        "audit_flags": ["normal_convention_unknown"] if maps.get("Normal") else [],
        "audit_provenance": {
            "source_draft_sha256": source_digest,
            "source_asset_sha256": _hash_json(raw),
            "source_connection_test": (raw.get("materials") or {}).get("connection_test"),
        },
    }
    issues = list(semantic.get("issues") or []) + list(quality.get("issues") or [])
    # The legacy draft format cannot prove the strict manifest contract
    # (applied transforms, preview hashes, collision evidence, and complete
    # material metadata).  Keep converted records preview-only until they are
    # re-exported through asset_manifest_v1 and pass its production gate.
    status = "draft"
    asset = {
        "asset_id": asset_id,
        "family_id": family_id,
        "geometry_variant_id": geometry_id,
        "model_file": model_file,
        "hash": model_hash.lower(),
        "dimensions_import_units": dimensions,
        "meters_per_unit": physical.get("metres_per_import_unit"),
        "front_axis": orientation.get("semantic_front_axis"),
        "up_axis": orientation.get("post_import_up"),
        "symmetric_front": False,
        "status": status,
        "source_name": raw.get("source_name"),
        "placement": placement.get("support_type"),
        "scope": semantic.get("scope"),
        "category": semantic.get("category"),
        "tags": list(semantic.get("tags") or []),
        "issues": issues,
        "quality": {
            "file_integrity": quality.get("file_integrity"),
            "local_import": quality.get("local_import"),
            "two_view_render": quality.get("two_view_render"),
            "complete_for_floor_placement": quality.get("complete_for_floor_placement"),
        },
        "material_variant_ids": [material_id],
        "audit_provenance": {
            "source_draft_sha256": source_digest,
            "source_asset_sha256": _hash_json(raw),
            "source_model_sha256": model_hash.lower(),
            "generation_batch": (raw.get("provenance") or {}).get("generation_batch"),
        },
    }
    return asset, material


def audit_draft_to_catalog(draft_path: os.PathLike[str] | str, output_path: os.PathLike[str] | str,
                           root: Optional[os.PathLike[str] | str] = None) -> Dict[str, Any]:
    """Convert an audited draft into a machine-readable catalog.

    Model and texture paths are checked against the asset root and model hashes
    are verified. Output must be outside the source asset tree.
    """
    draft_file = Path(draft_path).expanduser().resolve()
    raw = _json_load(draft_file)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("assets"), list):
        raise CatalogError("audit draft must contain an assets array")
    root_value = root if root is not None else raw.get("asset_root")
    if not root_value:
        raise CatalogError("audit draft requires asset_root or --root")
    root_path = Path(root_value).expanduser()
    root_path = root_path.resolve()
    if not root_path.is_dir():
        raise CatalogError(f"asset root is not a directory: {root_path}")
    source_digest = _hash_json(raw)
    assets: List[Dict[str, Any]] = []
    materials: Dict[str, Dict[str, Any]] = {}
    ids = set()
    geometry_ids = set()
    for item in raw["assets"]:
        if not isinstance(item, Mapping):
            raise CatalogError("each draft asset must be an object")
        asset, material = _asset_from_draft(item, root_path, source_digest)
        if asset["asset_id"] in ids or asset["geometry_variant_id"] in geometry_ids:
            raise CatalogError(f"duplicate asset or geometry id: {asset['asset_id']}")
        ids.add(asset["asset_id"])
        geometry_ids.add(asset["geometry_variant_id"])
        assets.append(asset)
        if material["material_variant_id"] in materials:
            raise CatalogError(f"duplicate material variant id: {material['material_variant_id']}")
        materials[material["material_variant_id"]] = material
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root_path),
        "source_audit_draft": str(draft_file),
        "source_audit_sha256": source_digest,
        "assets": assets,
        "material_variants": materials,
        "selection_policy": {
            "approved_only_by_default": True,
            "avoid_duplicate_variants": True,
            "allow_preview_drafts": True,
        },
    }
    destination = Path(output_path).expanduser().resolve()
    if destination == draft_file or root_path == destination or root_path in destination.parents:
        raise CatalogError("catalog output must not overwrite the audit or source asset tree")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return catalog


def _validate_catalog(data: Mapping[str, Any], catalog_path: Path) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise CatalogError(f"catalog requires schema_version={SCHEMA_VERSION}")
    if not isinstance(data.get("root"), str) or not data["root"]:
        raise CatalogError("catalog.root is required")
    if not isinstance(data.get("assets"), list) or not isinstance(data.get("material_variants"), Mapping):
        raise CatalogError("catalog requires assets[] and material_variants{}")
    ids, geometry_ids = set(), set()
    for asset in data["assets"]:
        if not isinstance(asset, Mapping):
            raise CatalogError("catalog assets must be objects")
        for key in ("asset_id", "family_id", "geometry_variant_id", "model_file", "hash", "status"):
            if not isinstance(asset.get(key), str) or not asset[key]:
                raise CatalogError(f"asset missing {key}")
        if asset["status"] not in _STATUSES:
            raise CatalogError(f"invalid asset status: {asset['status']}")
        if asset["status"] == "technical_ready" and (asset.get("audit_provenance") or {}).get("intake_gate") != "asset_intake_v1":
            raise CatalogError('technical_ready requires the asset_intake_v1 receipt gate')
        for id_key in ("asset_id", "family_id", "geometry_variant_id"):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", asset[id_key]):
                raise CatalogError(f"invalid identifier: {id_key}")
        if not _valid_hash(asset["hash"]):
            raise CatalogError(f"asset {asset['asset_id']} hash must be SHA-256")
        if asset.get("front_axis") not in _AXES | {None}:
            raise CatalogError("front_axis must be a signed coordinate axis or null")
        if asset.get("up_axis") not in _AXES | {None}:
            raise CatalogError("up_axis must be a signed coordinate axis or null")
        if type(asset.get("symmetric_front", False)) is not bool:
            raise CatalogError("symmetric_front must be boolean")
        for key in ("category", "placement", "scope"):
            if asset.get(key) is not None and not isinstance(asset[key], str):
                raise CatalogError(f"asset {key} must be a string or null")
        if asset.get("meters_per_unit") is not None and _finite_number(asset["meters_per_unit"], "meters_per_unit") <= 0:
            raise CatalogError("meters_per_unit must be positive")
        if asset["asset_id"] in ids or asset["geometry_variant_id"] in geometry_ids:
            raise CatalogError("asset and geometry IDs must be unique")
        ids.add(asset["asset_id"]); geometry_ids.add(asset["geometry_variant_id"])
        dims = asset.get("dimensions_import_units")
        if not isinstance(dims, list) or len(dims) != 3:
            raise CatalogError(f"asset {asset['asset_id']} dimensions_import_units must have 3 values")
        for value in dims:
            if _finite_number(value, f"asset {asset['asset_id']} dimension") <= 0:
                raise CatalogError("dimensions must be positive")
        _safe_relative(asset["model_file"], f"asset {asset['asset_id']}.model_file")
        if not isinstance(asset.get("material_variant_ids", []), list):
            raise CatalogError(f"asset {asset['asset_id']}.material_variant_ids must be an array")
        if len(set(asset.get("material_variant_ids", []))) != len(asset.get("material_variant_ids", [])):
            raise CatalogError("duplicate material IDs on asset")
        if asset.get("collision_file") is not None:
            _safe_relative(asset["collision_file"], f"asset {asset['asset_id']}.collision_file")
            if asset.get("collision_hash") is not None and not _valid_hash(asset["collision_hash"]):
                raise CatalogError(f"asset {asset['asset_id']} collision_hash must be SHA-256")
        manifest_gate = bool((asset.get("audit_provenance") or {}).get("manifest_gate"))
        if manifest_gate:
            if not isinstance(asset.get("collision_file"), str) or not asset.get("collision_file"):
                raise CatalogError(f"asset {asset['asset_id']} requires collision_file")
            if not _valid_hash(asset.get("collision_hash")):
                raise CatalogError(f"asset {asset['asset_id']} requires collision_hash")
    for material_id, material in data["material_variants"].items():
        if not isinstance(material_id, str) or not isinstance(material, Mapping):
            raise CatalogError("material_variants records must be objects")
        if material.get("material_variant_id") != material_id:
            raise CatalogError(f"material variant key mismatch: {material_id}")
        if not isinstance(material.get("compatible_geometry_ids"), list):
            raise CatalogError(f"material variant {material_id} requires compatible_geometry_ids")
        if len(set(material["compatible_geometry_ids"])) != len(material["compatible_geometry_ids"]):
            raise CatalogError("duplicate geometry compatibility IDs")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", material_id):
            raise CatalogError("invalid material variant identifier")
        if any(item not in geometry_ids for item in material["compatible_geometry_ids"]):
            raise CatalogError("material compatibility references unknown geometry")
        multiplier = material.get("base_color_multiplier")
        if multiplier is not None:
            if not isinstance(multiplier, list) or len(multiplier) != 3:
                raise CatalogError(f"material variant {material_id}.base_color_multiplier must contain three values")
            for value in multiplier:
                if _finite_number(value, f"material variant {material_id}.base_color_multiplier") < 0 or value > 1:
                    raise CatalogError(f"material variant {material_id}.base_color_multiplier must be in [0, 1]")
        _normalise_maps(Path(data["root"]).expanduser().resolve(), material.get("maps", {}), f"material_variants.{material_id}.maps", False)
        for record in material.get("maps", {}).values():
            if record.get("path") and not _valid_hash(record.get("hash")):
                raise CatalogError("texture file requires SHA-256 hash")
        slot_materials = material.get("slot_materials", {})
        if not isinstance(slot_materials, Mapping):
            raise CatalogError(f"material variant {material_id}.slot_materials must be an object")
        for slot_name, slot_maps in slot_materials.items():
            if not isinstance(slot_name, str) or not slot_name or not isinstance(slot_maps, Mapping):
                raise CatalogError(f"material variant {material_id} has invalid slot material")
            normalized_slot = _normalise_maps(Path(data["root"]).expanduser().resolve(), slot_maps,
                                               f"material_variants.{material_id}.slot_materials.{slot_name}", False)
            for record in normalized_slot.values():
                if record.get("path") and not _valid_hash(record.get("hash")):
                    raise CatalogError("slot texture file requires SHA-256 hash")
    for asset in data["assets"]:
        for material_id in asset.get("material_variant_ids", []):
            if material_id not in data["material_variants"]:
                raise CatalogError("asset references unknown material variant")


class AssetCatalog:
    """Loaded catalog with safe lookup, audit and deterministic resolution."""

    def __init__(self, data: Mapping[str, Any], catalog_path: Optional[os.PathLike[str] | str] = None):
        if not isinstance(data, Mapping) or not isinstance(data.get("root"), str):
            raise CatalogError("catalog.root is required")
        self.data = copy.deepcopy(dict(data))
        self.catalog_path = Path(catalog_path).expanduser().resolve() if catalog_path else None
        root = Path(self.data["root"]).expanduser()
        if not root.is_absolute() and self.catalog_path:
            root = self.catalog_path.parent / root
        if root.is_symlink():
            raise CatalogError("catalog root cannot use symlink paths")
        self.root = root.resolve()
        self._checked_hash_path(self.root)
        self.data["root"] = str(self.root)
        _validate_catalog(self.data, self.catalog_path or Path("<memory>"))
        self.assets = {asset["asset_id"]: asset for asset in self.data["assets"]}
        self.materials = dict(self.data["material_variants"])
        self._geometry_to_asset = {asset["geometry_variant_id"]: asset for asset in self.assets.values()}
        self._audit_provenance = {key: copy.deepcopy(asset.get("audit_provenance", {}))
                                  for key, asset in self.assets.items()}
        self._material_audit_provenance = {key: copy.deepcopy(material.get("audit_provenance", {}))
                                           for key, material in self.materials.items()}
        self._snapshot_file_hashes = None
        self._snapshot_file_stats = {}
        self._snapshot_lock = threading.RLock()
        self._intake_receipt_cache = {}

    @staticmethod
    def _checked_hash_path(path: os.PathLike[str] | str) -> Path:
        # Callers pass canonical roots so platform aliases are already resolved.
        # Never resolve here: it would hide a substituted parent symlink.
        candidate = Path(os.path.abspath(Path(path).expanduser()))
        components = [*reversed(candidate.parents), candidate]
        for component in components:
            try:
                mode = component.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise CatalogError("catalog input cannot use symlink paths: " + str(candidate))
        return candidate

    @staticmethod
    def _stat_signature(path: Path):
        path = AssetCatalog._checked_hash_path(path)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise CatalogError("catalog input is not a regular file: " + str(path))
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    @classmethod
    def stable_file_digest(cls, path: os.PathLike[str] | str):
        path = cls._checked_hash_path(path)
        try:
            before = cls._stat_signature(path)
            digest = sha256_file(path)
            after = cls._stat_signature(path)
        except OSError as exc:
            raise CatalogError("catalog input is unavailable: " + str(path)) from exc
        if before != after:
            raise CatalogError("catalog input changed while hashing: " + str(path))
        return digest, after

    def pin_snapshot_hashes(self, hashes: Mapping[str, Optional[str]], stats=None) -> None:
        """Reverify snapshot hashes with stable file identities before caching."""
        with self._snapshot_lock:
            if stats is not None and not isinstance(stats, Mapping):
                raise CatalogError("catalog snapshot stats must be a mapping")
            normalized, verified_stats = {}, {}
            for name, digest in hashes.items():
                path = self._checked_hash_path(name)
                key = str(path)
                if digest is None:
                    if path.exists():
                        raise CatalogError("previously missing catalog input appeared: " + key)
                else:
                    if not _valid_hash(digest):
                        raise CatalogError("invalid catalog snapshot hash: " + key)
                    if stats is None:
                        actual, signature = self.stable_file_digest(path)
                        if actual.lower() != digest.lower():
                            raise CatalogError("catalog input changed after the batch snapshot: " + key)
                    else:
                        try:
                            signature = self._stat_signature(path)
                        except OSError as exc:
                            raise CatalogError("catalog input is unavailable: " + key) from exc
                        if signature != stats.get(key):
                            raise CatalogError("catalog input changed after the batch snapshot: " + key)
                    verified_stats[key] = signature
                normalized[key] = digest
            self._snapshot_file_hashes = normalized
            self._snapshot_file_stats = verified_stats

    def verify_file_hash(self, path: os.PathLike[str] | str, expected: str, *, force: bool = False) -> bool:
        """Verify a file, optionally forcing a fresh stable hash."""
        with self._snapshot_lock:
            resolved = self._checked_hash_path(path)
            key = str(resolved)
            if self._snapshot_file_hashes is not None and key in self._snapshot_file_hashes:
                snapshot_digest = self._snapshot_file_hashes[key]
                if snapshot_digest is None:
                    raise CatalogError("catalog input was missing in the batch snapshot: " + key)
                if snapshot_digest.lower() != str(expected).lower():
                    raise CatalogError("catalog-declared hash differs from the batch snapshot: " + key)
                try:
                    signature = self._stat_signature(resolved)
                except OSError as exc:
                    raise CatalogError("catalog input is unavailable: " + key) from exc
                if not force and signature == self._snapshot_file_stats.get(key):
                    return True
                actual, signature = self.stable_file_digest(resolved)
                if actual.lower() != snapshot_digest.lower():
                    raise CatalogError("catalog input changed after the batch snapshot: " + key)
                self._snapshot_file_stats[key] = signature
                return True
            actual, _ = self.stable_file_digest(resolved)
            if actual.lower() != str(expected).lower():
                raise CatalogError("catalog file hash mismatch: " + key)
            return True

    def get_asset(self, asset_id: str) -> Dict[str, Any]:
        try:
            return copy.deepcopy(self.assets[asset_id])
        except KeyError as exc:
            raise CatalogError(f"unknown asset_id: {asset_id}") from exc

    def get_material_variant(self, material_variant_id: str) -> Dict[str, Any]:
        try:
            return copy.deepcopy(self.materials[material_variant_id])
        except KeyError as exc:
            raise CatalogError(f"unknown material_variant_id: {material_variant_id}") from exc

    def resolve_path(self, relative_path: str, must_exist: bool = True) -> Path:
        self._checked_hash_path(self.root)
        relative = _relative_path(self.root, relative_path, "asset path", must_exist)
        return (self.root / relative).resolve()

    def verify_hash(self, asset_id: str, *, force: bool = False) -> bool:
        asset = self.get_asset(asset_id)
        model = self.resolve_path(asset["model_file"], must_exist=True)
        try:
            self.verify_file_hash(model, asset["hash"], force=force)
        except CatalogError:
            raise CatalogError(f"asset {asset_id} model hash mismatch")
        return True

    def verify_material_hashes(self, material_variant_id: str, *, force: bool = False) -> bool:
        material = self.get_material_variant(material_variant_id)
        if material.get('mode') == 'embedded_gltf':
            compatible = material.get('compatible_geometry_ids', [])
            if len(compatible) != 1 or compatible[0] not in self._geometry_to_asset:
                raise CatalogError('embedded material requires one exact geometry')
            asset = self._geometry_to_asset[compatible[0]]
            if ((asset.get('audit_provenance') or {}).get('intake_gate') != 'asset_intake_v1'
                    or material.get('model_sha256') != asset['hash']):
                raise CatalogError('embedded material model identity differs')
            self.verify_hash(asset['asset_id'], force=force)
            self.verify_intake_receipt(asset)
            return True
        records = list(material.get("maps", {}).values())
        for slot_maps in (material.get("slot_materials") or {}).values():
            records.extend(slot_maps.values())
        for record in records:
            if record.get("path"):
                try:
                    valid = self.verify_file_hash(self.resolve_path(record["path"]), str(record.get("hash", "")), force=force)
                except CatalogError:
                    raise CatalogError(f"material {material_variant_id} texture hash mismatch") from None
                if not valid:
                    raise CatalogError(f"material {material_variant_id} texture hash mismatch")
        return True

    def verify_intake_receipt(self, asset):
        """Validate technical admission separately from visual approval.

        The content-addressed receipt pins all normalized geometry, material,
        scale and support metadata consumed by the layout/runtime. It is not a
        certificate of semantic completeness or photographic fidelity.
        """
        provenance = asset.get('audit_provenance') or {}
        if provenance.get('intake_gate') != 'asset_intake_v1' or provenance.get('manifest_gate'):
            raise CatalogError('invalid intake gate')
        if not _valid_hash(provenance.get('manifest_sha256')):
            raise CatalogError('intake receipt hash missing')
        path = self.resolve_path(asset.get('manifest_file'))
        self.verify_file_hash(path, provenance['manifest_sha256'])
        cache_key = (str(path), provenance['manifest_sha256'])
        with self._snapshot_lock:
            receipt = self._intake_receipt_cache.get(cache_key)
            if receipt is None:
                receipt = _json_load(path)
                self._intake_receipt_cache[cache_key] = receipt
        checks = receipt.get('checks', {})
        if (receipt.get('schema_version') != 'asset_intake_receipt_v1'
                or receipt.get('policy_version') != 'asset_intake_v1'
                or receipt.get('status') != 'technical_ready'
                or any(checks.get(k) is not True for k in
                    ('glb_embedded_standard_opaque', 'mesh_uv_materials', 'image_decode', 'scale_roundtrip'))):
            raise CatalogError('intake technical checks missing')
        comparable = {k: v for k, v in asset.items()
                      if k not in ('model_file', 'manifest_file', 'audit_provenance')}
        if receipt.get('asset') != comparable:
            raise CatalogError('catalog asset differs from intake receipt')
        metadata = receipt.get('metadata', {})
        if (asset.get('status') != 'technical_ready' or asset.get('meters_per_unit') != 1
                or asset.get('category') != metadata.get('category')
                or asset.get('scale_source') != metadata.get('scale_source')
                or asset.get('placement') != {'floor': 'floor', 'surface': 'table'}.get(metadata.get('placement'))
                or not math.isclose(max(asset['dimensions_import_units']), metadata.get('longest_dimension_m', -1),
                                    rel_tol=1e-5, abs_tol=1e-6)):
            raise CatalogError('intake semantic/scale identity differs')
        identifiers = asset.get('material_variant_ids', [])
        if len(identifiers) != 1 or identifiers[0] not in self.materials:
            raise CatalogError('intake requires one native appearance')
        material = {k: v for k, v in self.materials[identifiers[0]].items() if k != 'audit_provenance'}
        if receipt.get('material') != material or material.get('model_sha256') != asset['hash']:
            raise CatalogError('embedded material differs from intake receipt')
        self.verify_hash(asset['asset_id'])
        return receipt

    def production_readiness(self, asset_or_id: Mapping[str, Any] | str) -> Tuple[bool, List[str]]:
        asset = self.get_asset(asset_or_id) if isinstance(asset_or_id, str) else asset_or_id
        if (asset.get('audit_provenance') or {}).get('intake_gate'):
            try:
                self.verify_intake_receipt(asset)
                return True, []
            except (CatalogError, OSError, ValueError, TypeError, KeyError):
                return False, ['intake_receipt_invalid']
        reasons: List[str] = []
        if asset.get("status") != "approved": reasons.append("asset_not_approved")
        scale = asset.get("meters_per_unit")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) <= 0:
            reasons.append("unknown_scale")
        if not asset.get("front_axis") and asset.get("symmetric_front") is not True:
            reasons.append("unknown_semantic_front")
        if asset.get("category") in (None, "", "unknown"):
            reasons.append("classification_unknown")
        if asset.get("placement") not in ("floor", "table", "wall", "ceiling", "support", "holder", "free"):
            reasons.append("placement_unverified")
        quality = asset.get("quality") or {}
        # Catalogs built from asset_manifest_v1 carry a manifest gate and must
        # retain the manifest's explicit production-ready decision.  Legacy
        # draft catalogs do not have this marker and remain preview/compatibility
        # handled by their existing status and quality checks.
        manifest_gate = bool((asset.get("audit_provenance") or {}).get("manifest_gate"))
        if manifest_gate and quality.get("production_ready") is not True:
            reasons.append("quality_not_production_ready")
        if quality.get("complete_for_floor_placement") is False or "partial" in str(asset.get("scope", "")):
            reasons.append("incomplete_geometry")
        elif quality.get("complete_for_floor_placement") is not True:
            reasons.append("geometry_completeness_unverified")
        try: self.verify_hash(asset["asset_id"])
        except CatalogError: reasons.append("model_hash_invalid")
        collision_file = asset.get("collision_file")
        if manifest_gate and (not collision_file or not _valid_hash(asset.get("collision_hash"))):
            reasons.append("collision_invalid")
        if collision_file:
            try:
                collision = self.resolve_path(collision_file)
                if not asset.get("collision_hash"):
                    raise CatalogError("collision hash mismatch")
                self.verify_file_hash(collision, str(asset["collision_hash"]))
            except CatalogError:
                if "collision_invalid" not in reasons:
                    reasons.append("collision_invalid")
        return not reasons, reasons

    def resolver(self, seed: int = 0) -> "AssetResolver":
        return AssetResolver(self, seed=seed)

    def resolve(self, **kwargs: Any) -> Dict[str, Any]:
        return self.resolver(seed=int(kwargs.pop("seed", 0))).resolve(**kwargs)


def load_catalog(path: os.PathLike[str] | str) -> AssetCatalog:
    catalog_path = Path(path).expanduser()
    if catalog_path.is_symlink():
        raise CatalogError("catalog input cannot use symlink paths: " + str(catalog_path))
    catalog_path = catalog_path.resolve()
    before = AssetCatalog._stat_signature(catalog_path)
    content = catalog_path.read_bytes()
    after = AssetCatalog._stat_signature(catalog_path)
    if before != after:
        raise CatalogError("catalog input changed while loading: " + str(catalog_path))
    try:
        data = json.loads(content.decode("utf-8"), parse_constant=_reject_constant,
                          object_pairs_hook=_unique_keys)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"invalid JSON at {catalog_path}:{exc.lineno}:{exc.colno}") from None
    if not isinstance(data, Mapping):
        raise CatalogError("catalog JSON must be an object")
    catalog = AssetCatalog(data, catalog_path=catalog_path)
    catalog.source_sha256 = hashlib.sha256(content).hexdigest()
    catalog.source_stat_signature = after
    return catalog


class AssetResolver:
    """Deterministic scene-local resolver that records pair reuse.

    Reuse one resolver for successive objects and views of the same scene.
    A fresh resolver produces identical results for identical request order;
    diversity allocation is intentionally dependent on prior object requests.
    No global cross-worker allocation or persistent state is implied.
    """

    def __init__(self, catalog: AssetCatalog, seed: int = 0):
        self.catalog = catalog
        self.seed = int(seed)
        self._scene_pairs: Dict[str, List[Tuple[str, str]]] = {}
        self._object_pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._results: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _candidates(self, family_id: str, allow_preview: bool) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        assets = [a for a in self.catalog.assets.values() if a.get("family_id") == family_id]
        if not assets:
            raise CatalogError(f"no assets for family_id: {family_id}")
        result = []
        for asset in assets:
            ready, reasons = self.catalog.production_readiness(asset)
            if not allow_preview and not ready:
                continue
            self.catalog.verify_hash(asset["asset_id"])
            for material_id in asset.get("material_variant_ids", []):
                material = self.catalog.materials.get(material_id)
                if not material or asset["geometry_variant_id"] not in material.get("compatible_geometry_ids", []):
                    continue
                self.catalog.verify_material_hashes(material_id)
                result.append((asset, material))
        if not result:
            mode = "preview" if allow_preview else "approved production"
            raise CatalogError(f"no {mode} variants available for family_id: {family_id}")
        return sorted(result, key=lambda pair: (pair[0]["geometry_variant_id"], pair[1]["material_variant_id"]))

    def resolve(self, family_id: str, scene_id: str, object_id: str, *, seed: Optional[int] = None,
                geometry_variant_id: Optional[str] = None, material_variant_id: Optional[str] = None,
                allow_preview: bool = False, diversity: bool = True) -> Dict[str, Any]:
        if not isinstance(scene_id, str) or not isinstance(object_id, str) or not scene_id or not object_id:
            raise CatalogError("scene_id and object_id are required for deterministic resolution")
        key = (scene_id, object_id)
        candidates = self._candidates(family_id, allow_preview)
        if key in self._object_pairs:
            pair = self._object_pairs[key]
            if (geometry_variant_id and geometry_variant_id != pair[0]) or (material_variant_id and material_variant_id != pair[1]):
                raise CatalogError("same object_id cannot change an already selected variant")
            if pair not in [(a["geometry_variant_id"], m["material_variant_id"]) for a, m in candidates]:
                raise CatalogError("cached object selection no longer matches the family or readiness policy")
            return copy.deepcopy(self._results[key])
        if geometry_variant_id:
            candidates = [p for p in candidates if p[0]["geometry_variant_id"] == geometry_variant_id]
        if material_variant_id:
            candidates = [p for p in candidates if p[1]["material_variant_id"] == material_variant_id]
        if not candidates:
            raise CatalogError("requested geometry/material pair is unavailable or incompatible")
        used = set(self._scene_pairs.get(scene_id, []))
        salt = self.seed if seed is None else int(seed)
        ranked = sorted(candidates, key=lambda p: _hash_json([salt, scene_id, object_id, p[0]["geometry_variant_id"], p[1]["material_variant_id"]]))
        choice = next((p for p in ranked if (p[0]["geometry_variant_id"], p[1]["material_variant_id"]) not in used), ranked[0]) if diversity else ranked[0]
        pair = (choice[0]["geometry_variant_id"], choice[1]["material_variant_id"])
        self._object_pairs[key] = pair
        self._scene_pairs.setdefault(scene_id, []).append(pair)
        count = self._scene_pairs[scene_id].count(pair)
        result = self._result(choice[0], choice[1], count)
        result["selection_reason"] = "explicit_pair" if geometry_variant_id and material_variant_id else (
            "variant_pool_exhausted" if diversity and pair in used else "deterministic_selection")
        result["preview_only"] = not self.catalog.production_readiness(choice[0])[0]
        self._results[key] = copy.deepcopy(result)
        return result

    def _result(self, asset: Mapping[str, Any], material: Mapping[str, Any], count: int) -> Dict[str, Any]:
        return {
            "asset_id": asset["asset_id"],
            "family_id": asset["family_id"],
            "geometry_variant_id": asset["geometry_variant_id"],
            "material_variant_id": material["material_variant_id"],
            "model_file": asset["model_file"],
            "material_maps": copy.deepcopy(material.get("maps", {})),
            "slot_materials": copy.deepcopy(material.get("slot_materials", {})),
            "placement": asset.get("placement"),
            "anchor": copy.deepcopy(asset.get("anchor")),
            "collision_file": asset.get("collision_file"),
            "collision_hash": asset.get("collision_hash"),
            "reused": count > 1,
            "reuse_count": count,
            "audit_provenance": copy.deepcopy(self.catalog._audit_provenance[asset["asset_id"]]),
            "material_audit_provenance": copy.deepcopy(self.catalog._material_audit_provenance[material["material_variant_id"]]),
        }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build or inspect a strict asset catalog")
    parser.add_argument("--audit-draft", metavar="PATH", help="audited draft JSON to convert")
    parser.add_argument("--output", metavar="PATH", help="catalog JSON output path")
    parser.add_argument("--root", metavar="PATH", help="override draft asset_root")
    args = parser.parse_args(argv)
    if bool(args.audit_draft) != bool(args.output):
        parser.error("--audit-draft and --output must be supplied together")
    if args.audit_draft:
        audit_draft_to_catalog(args.audit_draft, args.output, root=args.root)
        return 0
    parser.error("provide --audit-draft and --output")


if __name__ == "__main__":
    raise SystemExit(_main())
