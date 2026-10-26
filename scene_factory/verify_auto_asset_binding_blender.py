"""Real Blender smoke check for selector -> manifest catalog -> shared renderer.

Run with Blender, not normal Python::

    Blender -b --factory-startup --python-exit-code 1 --python \
      verify_auto_asset_binding_blender.py -- --output-dir /absolute/new/folder

Everything is a synthetic fixture. Passing proves technical integration only;
it does not certify photograph reconstruction, realism, or dataset acceptance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from asset_catalog import load_catalog
from asset_manifest import build_catalog, validate_manifest
from auto_asset_binding import generate_bindings
from multiview_spec import validate
from spec_v2 import upgrade_v1
import generate_multiview as renderer


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_near(actual, expected, label, tolerance=2e-5):
    if len(actual) != len(expected) or any(abs(a - b) > tolerance for a, b in zip(actual, expected)):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def world_bounds(root):
    meshes = [obj for obj in root.children_recursive if obj.type == "MESH"]
    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    if not points:
        raise AssertionError("Imported object has no mesh")
    low = [min(point[axis] for point in points) for axis in range(3)]
    high = [max(point[axis] for point in points) for axis in range(3)]
    return low, high, meshes


def save_texture(path, name, color):
    image = bpy.data.images.new(name, width=8, height=8)
    image.pixels[:] = list(color) * 64
    image.file_format = "PNG"
    image.filepath_raw = str(path)
    image.save()
    return image


def create_package(package):
    (package / "textures").mkdir(parents=True)
    (package / "previews").mkdir()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    parts = []
    boxes = [("Top", [0, 0, .72], [1.2, .65, .06])]
    boxes += [(f"Leg_{i}_{j}", [i * .52, j * .245, .345], [.08, .08, .69])
              for i in (-1, 1) for j in (-1, 1)]
    for name, position, dimensions in boxes:
        bpy.ops.mesh.primitive_cube_add(size=1, location=position)
        obj = bpy.context.object
        obj.name = name
        obj.dimensions = dimensions
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        parts.append(obj)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    model = bpy.context.object
    model.name = "SyntheticTable"
    bpy.context.scene.cursor.location = [0, 0, 0]
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    bpy.context.view_layer.update()
    assert_near(list(model.location), [0, 0, 0], "export location")
    assert_near(list(model.scale), [1, 1, 1], "export scale")
    assert_near(list(model.dimensions), [1.2, .65, .75], "source dimensions")
    if not model.data.uv_layers or any(polygon.area < 1e-9 for polygon in model.data.polygons):
        raise AssertionError("Synthetic source must have UVs and nondegenerate faces")
    model.data.calc_loop_triangles()
    triangles = len(model.data.loop_triangles)
    base = save_texture(package / "textures" / "brown.png", "SyntheticBrown", (.24, .08, .025, 1))
    save_texture(package / "textures" / "gray.png", "SyntheticGray", (.12, .14, .16, 1))
    save_texture(package / "textures" / "roughness.png", "SyntheticRoughness", (.6, .6, .6, 1))
    material = bpy.data.materials.new("SyntheticTableMaterial")
    material.use_nodes = True
    texture = material.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = base
    principled = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    principled.inputs["Roughness"].default_value = .6
    material.node_tree.links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    model.data.materials.append(material)
    bpy.ops.export_scene.gltf(filepath=str(package / "model.glb"), export_format="GLB", use_selection=True)
    bpy.ops.export_scene.gltf(filepath=str(package / "collision.glb"), export_format="GLB", use_selection=True,
                              export_materials="NONE")

    # Five actual camera renders, including a separate contact-on-floor view.
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x = scene.render.resolution_y = 96
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    camera_data = bpy.data.cameras.new("AssetPreviewCamera")
    camera = bpy.data.objects.new("AssetPreviewCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.75
    for name, location in (("front", [0, -3, .5]), ("side", [3, 0, .5]),
                           ("back", [0, 3, .5]), ("three_quarter", [2, -3, 2]),
                           ("placement_test", [2, -3, 1.3])):
        if name == "placement_test":
            bpy.ops.mesh.primitive_plane_add(size=3)
        camera.location = location
        camera.rotation_euler = (Vector([0, 0, .375]) - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(package / "previews" / (name + ".png"))
        bpy.ops.render.render(write_still=True)

    def record(relative, color_space=None):
        result = {"path": relative, "sha256": digest(package / relative)}
        if color_space:
            result["color_space"] = color_space
        return result

    materials = {}
    for name, file in (("synthetic_table_brown", "brown.png"), ("synthetic_table_gray", "gray.png")):
        materials[name] = {"material_variant_id": name, "compatible_geometry_ids": ["synthetic_table_geo"],
            "uv_layout_id": "synthetic_table_uv", "normal_convention": "OpenGL_Y+",
            "maps": {"base_color": record("textures/" + file, "sRGB"),
                     "roughness": record("textures/roughness.png", "Non-Color")}}
    manifest = {"schema_version": "asset_manifest_v1", "asset_id": "synthetic_table_geo",
        "family_id": "synthetic_table", "geometry_variant_id": "synthetic_table_geo", "status": "asset_accepted",
        "model": {"format": "glb", "path": "model.glb", "sha256": digest(package / "model.glb"),
            "dimensions_import_units": [1.2, .65, .75], "dimensions_m": [1.2, .65, .75],
            "meters_per_import_unit": 1, "scale_source": "measured", "scale_confidence": 1,
            "canonical_up_axis": "+Z", "canonical_front_axis": "-Y", "triangle_count": triangles,
            "object_count": 1, "transform": {"location": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}},
        "placement": {"support_type": "floor", "anchor_type": "bottom_center", "anchor_point_m": [0, 0, 0],
            "contact_points_m": [[x * .52, y * .245, 0] for x in (-1, 1) for y in (-1, 1)],
            "mount_points": [], "clearance_m": 0, "can_rotate_z": True, "allow_uniform_scale": True,
            "allow_non_uniform_scale": False, "collision_path": "collision.glb",
            "collision_sha256": digest(package / "collision.glb")},
        "semantics": {"category": "table", "description": "Synthetic technical fixture table; not photo-derived",
            "tags": ["synthetic_fixture"], "scope": "single_prop", "completeness": "complete",
            "confidence": 1, "source": "deterministic_script"},
        "uv": {"layout_id": "synthetic_table_uv", "channels": 1, "normal_convention": "OpenGL_Y+",
               "texel_density_policy": "fixture_only"},
        "material_variant_ids": list(materials), "materials": materials,
        "previews": {name: record("previews/" + name + ".png") for name in
                     ("front", "side", "back", "three_quarter", "placement_test")},
        "quality": {key: "passed" for key in ("file_integrity", "import", "geometry", "uv", "materials", "previews", "placement_test")}
                   | {"production_ready": True},
        "provenance": {"source_images": ["previews/three_quarter.png"], "generator": "synthetic integration fixture",
            "generator_version": "1", "generation_parameters_sha256": hashlib.sha256(b"synthetic-table-v1").hexdigest(),
            "license": "synthetic_fixture", "created_at": "2026-10-21T00:00:00Z"}}
    write_json(package / "asset_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    package = out / "assets" / "synthetic_table_geo"
    create_package(package)
    manifest_report = validate_manifest(package, require_production=True)
    write_json(out / "manifest_validation.json", manifest_report)
    if not manifest_report["valid"]:
        raise AssertionError(manifest_report["errors"])
    catalog_path = out / "asset_catalog.json"
    build_catalog(out / "assets", catalog_path)
    catalog = load_catalog(catalog_path)

    base = json.loads((ROOT / "examples" / "living_room.json").read_text())
    base["room"].update(width=6, depth=5, height=3)
    base["objects"] = [{"id": "table", "category": "table", "position": [-1, 2, 0],
        "dimensions": [1.44, .78, .9], "rotation_z_deg": 90, "color": [.2, .1, .05],
        "repeat": [2, 1], "spacing": [2, 0], "confidence": 1}]
    views = []
    for name, position, target in (("front", [0, .2, 1.6], [0, 2.2, .65]),
                                    ("back", [0, 4.8, 1.8], [0, 2, .65])):
        views.append({"id": name, "source_image": str(package / "previews" / "three_quarter.png"),
            "width": 96, "height": 96,
            "camera": {"position": position, "target": target, "lens_mm": 24, "roll_degrees": 0}})
    bundle = validate({"schema_version": 3, "scene_id": "synthetic_asset_binding",
        "shared_scene": upgrade_v1(base, 96, 96, "front"),
        "architecture": {"omit_surfaces": [], "front_wall": False}, "views": views, "texture_sources": []})
    bindings, report = generate_bindings(bundle, catalog, {"diversity": True})
    if report["bound_objects"] != 2 or len({item["material_variant_id"] for item in bindings["instances"]}) != 2:
        raise AssertionError("Both repeated objects must bind to distinct compatible material variants")
    for entry in bindings["instances"]:
        assert_near([entry["uniform_scale"]], [1.2], "selector scale")
    write_json(out / "scene_bundle.json", bundle)
    write_json(out / "asset_bindings.json", bindings)
    write_json(out / "asset_selection.json", report)
    scene_out = out / "scene"
    previous_argv = sys.argv
    try:
        sys.argv = [str(ROOT / "generate_multiview.py"), "--", "--spec", str(out / "scene_bundle.json"),
            "--output-dir", str(scene_out), "--asset-catalog", str(catalog_path),
            "--asset-bindings", str(out / "asset_bindings.json"), "--max-edge", "128", "--samples", "8",
            "--device", "CPU", "--tone-match", "off"]
        renderer.main()
    finally:
        sys.argv = previous_argv

    scene = bpy.context.scene
    roots = [obj for obj in scene.objects if obj.get("geometry_source") == "external_catalog_glb"]
    if {obj.name for obj in roots} != {"table__0_0", "table__1_0"}:
        raise AssertionError("Expected exactly the two external objects")
    object_records = []
    for root in sorted(roots, key=lambda obj: obj.name):
        expected_x = -1 if root.name == "table__0_0" else 1
        low, high, meshes = world_bounds(root)
        dimensions = [b - a for a, b in zip(low, high)]
        assert_near(list(root.location), [expected_x, 2, 0], "instance position")
        assert_near(dimensions, [.78, 1.44, .9], "rotated world dimensions")
        assert_near([(low[i] + high[i]) / 2 for i in range(2)] + [low[2]], [expected_x, 2, 0], "bottom-center anchor")
        if not all(mesh.data.uv_layers and mesh.data.materials for mesh in meshes):
            raise AssertionError("Imported UV and material slots are missing")
        object_records.append({"object_id": root.name, "position_m": list(root.location),
            "world_dimensions_m": dimensions, "world_bottom_m": low[2],
            "mesh_count": len(meshes), "provenance": json.loads(root["asset_provenance"])})
    mesh_identity = {obj.as_pointer() for obj in scene.objects if obj.type == "MESH"}
    camera_records = []
    for view in views:
        camera = bpy.data.objects["ReferenceCamera_" + view["id"]]
        renderer.set_view(scene, camera, view, 128)
        if mesh_identity != {obj.as_pointer() for obj in scene.objects if obj.type == "MESH"}:
            raise AssertionError("Switching camera rebuilt/duplicated geometry")
        camera_records.append({"view_id": view["id"], "camera": camera.name,
            "render_exists": (scene_out / ("rgb_" + view["id"] + ".png")).is_file(),
            "same_geometry_objects": True})
    metadata = json.loads((scene_out / "metadata.json").read_text())
    if metadata["geometry_build_count"] != 1 or metadata["camera_count"] != 2 or metadata["asset_builder_counts"]["external"] != 2:
        raise AssertionError("Shared renderer metadata differs from actual scene")
    verification = {"status": "passed", "synthetic_fixture": True,
        "photo_reconstruction_validated": False, "dataset_accepted": False,
        "scope": "Real GLB, manifest/catalog, selection, material variants, import, scale, floor anchor, placement and shared cameras",
        "blender_version": bpy.app.version_string, "manifest_gate_passed": manifest_report["production_ready"],
        "geometry_build_count": metadata["geometry_build_count"], "objects": object_records,
        "views": camera_records, "elapsed_seconds": round(time.monotonic() - started, 3)}
    write_json(out / "verification.json", verification)
    print("AUTO_ASSET_BINDING_BLENDER_VERIFIED " + str(out), flush=True)


if __name__ == "__main__":
    main()
