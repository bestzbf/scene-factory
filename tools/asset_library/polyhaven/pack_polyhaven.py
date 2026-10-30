"""Pack one Poly Haven glTF bundle into model.glb + asset.json.

Run inside Blender. Geometry and materials are re-exported as-is (only the
external textures get embedded). The declared longest edge comes from Poly
Haven's published dimensions; the imported mesh's own bounding box is written
to a separate report so any disagreement can be reviewed later.
"""
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

src, dst, category, placement, report = sys.argv[sys.argv.index('--') + 1:][:5]
src, dst, report = Path(src), Path(dst), Path(report)
info = json.loads((src / 'info.json').read_text(encoding='utf-8'))
gltf = next(src.glob('*.gltf'))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(gltf))
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
points = [o.matrix_world @ Vector(corner) for o in meshes for corner in o.bound_box]
lo = [min(p[i] for p in points) for i in range(3)]
hi = [max(p[i] for p in points) for i in range(3)]
mesh_dims = [hi[i] - lo[i] for i in range(3)]

dst.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(filepath=str(dst / 'model.glb'), export_format='GLB', export_image_format='AUTO')

declared = round(max(info['dimensions']) / 1000.0, 4)
metadata = {'schema_version': 'scene_asset_input_v1', 'category': category, 'placement': placement,
            'longest_dimension_m': declared, 'scale_source': 'known_reference'}
(dst / 'asset.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps({
    'declared_longest_m': declared,
    'mesh_dims_m': [round(v, 4) for v in mesh_dims],
    'mesh_longest_m': round(max(mesh_dims), 4),
    'mesh_to_declared_ratio': round(max(mesh_dims) / declared, 4),
    'glb_bytes': (dst / 'model.glb').stat().st_size,
}) + '\n', encoding='utf-8')
print('PACKED', dst.name)
