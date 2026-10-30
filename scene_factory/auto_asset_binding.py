"""Conservative, deterministic library selection for a validated shared scene.

This module has no Blender dependency.  It only binds existing base objects;
custom constructions remain unchanged.  Category and measured dimension checks
are retrieval filters, not a claim of visual or photometric similarity.  Floor
placement is the only automatic placement currently expressible by the base
object schema.  All other supports require a future explicit support contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()
    from .asset_catalog import AssetCatalog, CatalogError, load_catalog, sha256_file
    from .multiview_spec import validate
    from .schema import CATEGORIES
else:
    from asset_catalog import AssetCatalog, CatalogError, load_catalog, sha256_file
    from multiview_spec import validate
    from schema import CATEGORIES


POLICY_VERSION = "auto_asset_binding_v1"
DEFAULT_OPTIONS = {
    "seed": 0,
    "dimension_tolerance": 0.15,
    "min_uniform_scale": 0.8,
    "max_uniform_scale": 1.25,
    "floor_tolerance_m": 0.0001,
    "diversity": False,
    "category_map": {},
}
# Opt-in settings stay absent unless configured, so existing normalized options
# and job fingerprints are unchanged. settle_max_shift_m (library_layout_v1
# only): 0/absent keeps the strict behaviour where any overlap or a child
# overhanging its support blocks the scene; a positive value lets the selector
# move objects by at most this many metres each and records every move.
# settle_max_omitted_fraction (requires settle_max_shift_m): objects that settling
# cannot place may be left out, with their descendants, when they make up at most
# this share of the scene; each omission is listed in the selection report.
OPTIONAL_OPTIONS = {"settle_max_shift_m": (0, 1.0), "settle_max_omitted_fraction": (0, 0.25)}
_FLOOR_CATEGORIES = {"desk", "table", "chair", "sofa", "bed", "cabinet", "shelf", "plant", "rug", "box", "column"}
_OBJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_CATALOG_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_MAP_NAMES = {"base_color": "base_color", "baseColor": "base_color", "Base Color": "base_color",
              "normal": "normal", "Normal": "normal", "roughness": "roughness", "Roughness": "roughness",
              "metallic": "metallic", "Metallic": "metallic", "ao": "ao", "AO": "ao"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _number(value, low, high, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(label + " must be a finite number in the allowed range")
    return float(value)


def normalize_options(options=None):
    """Validate selector settings; no provider credentials or model calls exist."""
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options) - set(DEFAULT_OPTIONS) - set(OPTIONAL_OPTIONS):
        raise ValueError("Unknown automatic asset selection options")
    result = {**copy.deepcopy(DEFAULT_OPTIONS), **copy.deepcopy(options)}
    for name, (low, high) in OPTIONAL_OPTIONS.items():
        if name in result:
            result[name] = _number(result[name], low, high, "selection." + name)
    if result.get("settle_max_omitted_fraction", 0) > 0 and not result.get("settle_max_shift_m", 0) > 0:
        raise ValueError("selection.settle_max_omitted_fraction requires selection.settle_max_shift_m")
    if type(result["seed"]) is not int or not 0 <= result["seed"] <= 2147483647:
        raise ValueError("selection.seed must be an integer in 0..2147483647")
    for name, low, high in (("dimension_tolerance", 0, 0.5), ("min_uniform_scale", 0.01, 100),
                            ("max_uniform_scale", 0.01, 100), ("floor_tolerance_m", 0, 0.0001)):
        result[name] = _number(result[name], low, high, "selection." + name)
    if result["min_uniform_scale"] > result["max_uniform_scale"]:
        raise ValueError("selection.min_uniform_scale exceeds max_uniform_scale")
    if type(result["diversity"]) is not bool:
        raise ValueError("selection.diversity must be true or false")
    mapping = result["category_map"]
    if not isinstance(mapping, dict):
        raise ValueError("selection.category_map must be an object")
    normalized = {}
    for category, rule in mapping.items():
        if category not in CATEGORIES or not isinstance(rule, dict) or set(rule) - {"families", "categories"}:
            raise ValueError("category_map requires canonical scene categories and families/categories rules")
        normalized[category] = {}
        for name in ("families", "categories"):
            values = rule.get(name, [])
            if (not isinstance(values, list) or any(not isinstance(v, str) or not _CATALOG_ID.fullmatch(v) for v in values)
                    or len(values) != len(set(values))):
                raise ValueError("category_map " + name + " must contain unique canonical identifiers")
            normalized[category][name] = sorted(values)
        if not any(normalized[category].values()):
            raise ValueError("category_map rule cannot be empty")
    result["category_map"] = normalized
    return result


def _expanded_objects(bundle):
    """Mirror generate_v2.expand_objects without importing Blender."""
    expanded = []
    for entry in bundle["shared_scene"]["base_scene"]["objects"]:
        for i in range(entry["repeat"][0]):
            for j in range(entry["repeat"][1]):
                expanded.append({**copy.deepcopy(entry),
                    "id": entry["id"] + ("__%d_%d" % (i, j) if entry["repeat"] != [1, 1] else ""),
                    "position": [entry["position"][0] + i * entry["spacing"][0],
                                 entry["position"][1] + j * entry["spacing"][1], entry["position"][2]]})
    return sorted(expanded, key=lambda item: item["id"])


def _category_matches(asset, category, options):
    rule = options["category_map"].get(category)
    if rule is None:
        return asset.get("category") == category
    return (asset.get("family_id") in rule["families"] or asset.get("category") in rule["categories"])


def _geometry_reasons(catalog, asset, entry, options, preview):
    reasons = []
    try:
        ready, readiness = catalog.production_readiness(asset)
    except OSError:
        ready, readiness = False, ["asset_files_unreadable"]
    if not ready and not preview:
        reasons.extend(readiness)
    if not preview:
        if not (asset.get("audit_provenance") or {}).get("manifest_gate"):
            reasons.append("production_manifest_unverified")
        provenance = asset.get("audit_provenance") or {}
        manifest_digest = provenance.get("manifest_sha256")
        try:
            manifest_path = catalog.resolve_path(asset.get("manifest_file"))
            if (not isinstance(manifest_digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", manifest_digest)
                    or not catalog.verify_file_hash(manifest_path, manifest_digest)):
                reasons.append("production_manifest_hash_invalid")
        except (CatalogError, OSError, TypeError):
            reasons.append("production_manifest_hash_invalid")
        if asset.get("scope") != "single_prop":
            reasons.append("complete_single_prop_required")
    # Preview relaxes approval only. Unknown physical units and anchors never
    # become inferred measurements merely because a preview was requested.
    units = asset.get("meters_per_unit")
    if type(units) not in (int, float) or not math.isfinite(units) or not .000001 <= units <= 10000:
        reasons.append("unknown_scale")
    if asset.get("up_axis") != "+Z":
        reasons.append("up_axis_not_canonical")
    front = asset.get("front_axis")
    if front not in {"-X", "+X", "-Y", "+Y"} and not (front is None and asset.get("symmetric_front") is True):
        reasons.append("semantic_front_not_horizontal")
    if not preview and front != "-Y":
        reasons.append("production_front_not_canonical")
    if asset.get("placement") != "floor":
        reasons.append("placement_mismatch")
    anchor = asset.get("anchor")
    if (not isinstance(anchor, dict) or anchor.get("type") != "bottom_center"
            or not isinstance(anchor.get("point"), list) or len(anchor["point"]) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in anchor["point"])):
        reasons.append("bottom_center_anchor_unverified")
    rules = asset.get("placement_rules")
    if not isinstance(rules, dict):
        rules = {}
    if any(type(rules.get(key)) is not bool for key in ("can_rotate_z", "allow_uniform_scale")):
        reasons.append("placement_rules_unknown")
    if rules.get("can_rotate_z") is False and not math.isclose(math.remainder(entry["rotation_z_deg"], 360), 0, abs_tol=1e-9):
        reasons.append("rotation_forbidden")
    if Path(asset["model_file"]).suffix.lower() != ".glb" and not preview:
        reasons.append("production_requires_glb")
    try:
        catalog.verify_hash(asset["asset_id"])
    except (CatalogError, OSError):
        reasons.append("model_hash_invalid")
    if reasons:
        return sorted(set(reasons)), None
    dimensions = [float(d) * units for d in asset["dimensions_import_units"]]
    # The importer turns semantic +X/-X fronts to -Y before instantiation.
    # Legacy catalogs measure dimensions before that rotation, so swap XY.
    if front in {"-X", "+X"}:
        dimensions[0], dimensions[1] = dimensions[1], dimensions[0]
    target = entry["dimensions"]
    ratio = [a / b for a, b in zip(dimensions, target)]
    # Minimize the maximum per-axis relative error with a single scale.
    scale = 2 / (min(ratio) + max(ratio))
    scale = max(options["min_uniform_scale"], min(options["max_uniform_scale"], scale))
    if rules["allow_uniform_scale"] is False:
        scale = 1.0
        if not options["min_uniform_scale"] <= scale <= options["max_uniform_scale"]:
            return ["uniform_scale_forbidden"], None
    fitted = [v * scale for v in dimensions]
    errors = [abs(a - b) / b for a, b in zip(fitted, target)]
    fit = {"uniform_scale": scale, "source_dimensions_m": dimensions,
           "fitted_dimensions_m": fitted, "relative_dimension_errors": errors,
           "max_relative_dimension_error": max(errors)}
    if max(errors) > options["dimension_tolerance"] + 1e-12:
        return ["dimension_mismatch"], fit
    # Base objects lack a support graph. Keep potential support heights fixed
    # rather than leaving monitors/decor at an old tabletop/shelf height.
    # A matching bounding height is still not proof of a usable contact surface.
    if entry["category"] in {"desk", "table", "shelf", "cabinet", "box"} and abs(fitted[2] - target[2]) > .005:
        return ["support_height_mismatch"], fit
    return [], fit


def _material_reasons(catalog, asset, material_id, preview):
    material = catalog.materials.get(material_id)
    if material is None or asset["geometry_variant_id"] not in material.get("compatible_geometry_ids", []):
        return ["material_geometry_incompatible"]
    reasons = []
    if not preview and material.get("normal_convention") != "OpenGL_Y+":
        reasons.append("normal_convention_unverified")
    if (not preview and (material.get("audit_provenance") or {}).get("manifest_sha256")
            != (asset.get("audit_provenance") or {}).get("manifest_sha256")):
        reasons.append("material_manifest_mismatch")
    try:
        catalog.verify_material_hashes(material_id)
    except (CatalogError, OSError):
        reasons.append("material_hash_invalid")
    map_groups = [material.get("maps", {})] + list((material.get("slot_materials") or {}).values())
    has_base = False
    for maps in map_groups:
        for name, record in maps.items():
            if not record.get("path"):
                continue
            canonical = _MAP_NAMES.get(name)
            if canonical is None:
                reasons.append("unsupported_material_map")
            elif canonical == "base_color":
                has_base = True
                if record.get("color_space") != "sRGB":
                    reasons.append("base_color_space_invalid")
            elif record.get("color_space") != "Non-Color":
                reasons.append("data_map_color_space_invalid")
            if canonical == "normal" and material.get("normal_convention") not in {"OpenGL_Y+", "OpenGL", "opengl", "+Y"}:
                # The production importer also refuses unknown normal maps.
                if not preview or material.get("normal_convention") is not None:
                    reasons.append("normal_convention_unverified")
    if not has_base:
        reasons.append("base_color_missing")
    return sorted(set(reasons))


def generate_bindings(bundle, catalog, options=None, *, allow_preview=False):
    """Return ``(bindings_v1, report)`` without modifying bundle or catalog.

    ``category_map`` rules explicitly allow catalog ``families`` or
    ``categories`` for each canonical scene category.  With no rule, exact
    category equality is required.  ``diversity=False`` prioritizes dimensional
    fit; enabling it uses unused compatible pairs before repeating a pair.
    Failed candidates are recorded and left unbound for the caller's existing
    procedural fallback.  This never calls an image/mesh generation service.
    """
    validate(bundle)
    options = normalize_options(options)
    if type(allow_preview) is not bool:
        raise ValueError("allow_preview must be true or false")
    if not isinstance(catalog, AssetCatalog):
        catalog = load_catalog(catalog)
    bindings, records, used = [], {}, Counter()
    for entry in _expanded_objects(bundle):
        identifier, category = entry["id"], entry["category"]
        record = {"status": "fallback", "category": category, "position_m": list(entry["position"]),
                  "target_dimensions_m": list(entry["dimensions"]), "reasons": [], "candidates": []}
        records[identifier] = record
        if not _OBJECT_ID.fullmatch(identifier):
            record["reasons"] = ["unsafe_object_id"]
            continue
        if category not in _FLOOR_CATEGORIES:
            record["reasons"] = ["automatic_support_not_expressible"]
            continue
        if abs(entry["position"][2]) > options["floor_tolerance_m"]:
            record["reasons"] = ["floor_support_unverified"]
            continue
        record["support"] = {"type": "floor", "source": "base_category_and_bottom_z",
                             "collision_or_contact_verified": False}
        assets = sorted((a for a in catalog.assets.values() if _category_matches(a, category, options)),
                        key=lambda a: a["asset_id"])
        candidates = []
        for asset in assets:
            rejected, fit = _geometry_reasons(catalog, asset, entry, options, allow_preview)
            candidate_record = {"asset_id": asset["asset_id"], "reasons": rejected, "materials": []}
            if fit is not None:
                candidate_record["fit"] = fit
            record["candidates"].append(candidate_record)
            if rejected:
                continue
            for material_id in sorted(asset.get("material_variant_ids", [])):
                material_reasons = _material_reasons(catalog, asset, material_id, allow_preview)
                candidate_record["materials"].append({"material_variant_id": material_id, "reasons": material_reasons})
                if not material_reasons:
                    candidates.append((asset, material_id, fit))
            if not asset.get("material_variant_ids"):
                candidate_record["reasons"] = ["no_material_variants"]
        if not candidates:
            record["reasons"] = ["no_category_match" if not assets else "no_compatible_candidate"]
            continue
        def rank(candidate):
            asset, material_id, fit = candidate
            pair = asset["geometry_variant_id"], material_id
            tie = _hash([options["seed"], bundle["scene_id"], identifier, *pair])
            return ((bool(used[pair]) if options["diversity"] else False),
                    round(fit["max_relative_dimension_error"], 12), tie)
        asset, material_id, fit = min(candidates, key=rank)
        pair = asset["geometry_variant_id"], material_id
        reused = bool(used[pair])
        used[pair] += 1
        binding = {"object_id": identifier, "asset_id": asset["asset_id"], "material_variant_id": material_id,
                   "uniform_scale": fit["uniform_scale"], "rotation_xyz_deg": [0, 0, entry["rotation_z_deg"]]}
        bindings.append(binding)
        material = catalog.get_material_variant(material_id)
        record.update(status="bound", reasons=[], binding=copy.deepcopy(binding), fit=copy.deepcopy(fit),
                      geometry_variant_id=asset["geometry_variant_id"], family_id=asset["family_id"],
                      reused=reused, reuse_count=used[pair],
                      selection_reason=("variant_pool_exhausted" if options["diversity"] and reused
                                        else "deterministic_dimension_selection"),
                      preview_only=allow_preview,
                      provenance={"model_sha256": asset["hash"],
                                  "asset_audit": copy.deepcopy(asset.get("audit_provenance", {})),
                                  "material_audit": copy.deepcopy(material.get("audit_provenance", {}))})
    catalog_digest = getattr(catalog, "_batch_catalog_sha256", None) or _hash(catalog.data)
    report = {"schema_version": "auto_asset_binding_report_v1", "policy_version": POLICY_VERSION,
              "scene_id": bundle["scene_id"], "bundle_sha256": _hash(bundle), "catalog_sha256": catalog_digest,
              "options": options, "preview": allow_preview, "total_objects": len(records),
              "bound_objects": len(bindings), "fallback_objects": len(records) - len(bindings),
              "appearance_match_verified": False, "objects": records}
    return {"schema_version": 1, "instances": bindings}, report
