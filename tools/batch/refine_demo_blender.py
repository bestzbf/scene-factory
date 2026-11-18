"""Opt-in, non-destructive look refinement of an existing editable demo scene.

Run with Blender --background --python refine_demo_blender.py -- --help.
The saved input scene is never overwritten. All changes are recorded in refinement.json.
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--views', nargs='+', default=['view_01'])
    p.add_argument('--atlas', type=Path)
    p.add_argument('--detail', action='store_true')
    p.add_argument('--floor-tiles', action='store_true')
    p.add_argument('--edge', type=int, default=960)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--exposure', type=float, default=0.0, help='EV offset, applied only once')
    p.add_argument('--camera-config', type=Path)
    p.add_argument('--trials', type=Path, help='JSON map of trial names to camera overrides; one view only')
    p.add_argument('--save', action='store_true')
    return p.parse_args(sys.argv[sys.argv.index('--') + 1:])


def refine_materials(scene, atlas_path, detail, floor_tiles):
    report = {'atlas_nodes': 0, 'beveled_meshes': 0, 'surface_materials': []}
    atlas = bpy.data.images.load(str(atlas_path.resolve()), check_existing=True) if atlas_path else None
    if atlas:
        atlas.name = 'DemoRefinedPackagingAtlas'
        atlas.colorspace_settings.name = 'sRGB'
        atlas.pack()
    used_materials = {s.material for o in scene.objects if o.type == 'MESH'
                      for s in o.material_slots if s.material}
    stock_meshes = set()
    for mat in used_materials:
        if not mat.use_nodes:
            continue
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        product = False
        for node in nodes:
            if node.type == 'TEX_IMAGE' and node.image and node.image.name.startswith('product_atlas'):
                product = True
                if atlas:
                    node.image = atlas
                    node.interpolation = 'Linear'
                    node.extension = 'EXTEND'
                    report['atlas_nodes'] += 1
        if product and detail:
            for shader in [n for n in nodes if n.type == 'BSDF_PRINCIPLED']:
                shader.inputs['Roughness'].default_value = .36
                coat = shader.inputs.get('Coat Weight')
                if coat is not None:
                    coat.default_value = .12
                noise = nodes.new('ShaderNodeTexNoise')
                noise.inputs['Scale'].default_value = 180
                noise.inputs['Detail'].default_value = 2
                bump = nodes.new('ShaderNodeBump')
                bump.inputs['Strength'].default_value = .10
                bump.inputs['Distance'].default_value = .00035
                links.new(noise.outputs['Fac'], bump.inputs['Height'])
                links.new(bump.outputs['Normal'], shader.inputs['Normal'])
            stock_meshes.update(o for o in scene.objects if o.type == 'MESH'
                                and any(s.material == mat for s in o.material_slots))
    if detail:
        for obj in stock_meshes:
            if 'DemoEdgeHighlights' not in obj.modifiers:
                bevel = obj.modifiers.new('DemoEdgeHighlights', 'BEVEL')
                bevel.width = .0012
                bevel.segments = 2
                bevel.limit_method = 'ANGLE'
                bevel.angle_limit = math.radians(35)
                bevel.use_clamp_overlap = True
                report['beveled_meshes'] += 1
        for name in ('WallPaint', 'Ceiling'):
            mat = bpy.data.materials.get(name)
            if not mat or not mat.use_nodes:
                continue
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            shader = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if not shader or shader.inputs['Normal'].is_linked:
                continue
            tex = nodes.new('ShaderNodeTexNoise'); tex.inputs['Scale'].default_value = 90
            bump = nodes.new('ShaderNodeBump')
            bump.inputs['Strength'].default_value = .12
            bump.inputs['Distance'].default_value = .0006
            links.new(tex.outputs['Fac'], bump.inputs['Height'])
            links.new(bump.outputs['Normal'], shader.inputs['Normal'])
            report['surface_materials'].append(name)
    if floor_tiles:
        floor = bpy.data.objects.get('Floor')
        if floor and floor.active_material:
            floor.active_material = floor.active_material.copy()
            mat = floor.active_material
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            shader = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
            # World-space joints retain a physical 0.6 m tile size regardless of room size.
            coords = nodes.new('ShaderNodeTexCoord')
            separate = nodes.new('ShaderNodeSeparateXYZ')
            links.new(coords.outputs['Object'], separate.inputs[0])
            masks = []
            for axis in ('X', 'Y'):
                scale = nodes.new('ShaderNodeMath'); scale.operation = 'MULTIPLY'
                scale.inputs[1].default_value = 1 / .60
                links.new(separate.outputs[axis], scale.inputs[0])
                fract = nodes.new('ShaderNodeMath'); fract.operation = 'FRACT'
                links.new(scale.outputs[0], fract.inputs[0])
                line = nodes.new('ShaderNodeMath'); line.operation = 'LESS_THAN'
                line.inputs[1].default_value = .004
                links.new(fract.outputs[0], line.inputs[0]); masks.append(line)
            mask = nodes.new('ShaderNodeMath'); mask.operation = 'MAXIMUM'
            for i, line in enumerate(masks): links.new(line.outputs[0], mask.inputs[i])
            mix = nodes.new('ShaderNodeMixRGB'); mix.blend_type = 'MIX'
            base = tuple(shader.inputs['Base Color'].default_value)
            mix.inputs[1].default_value = base
            mix.inputs[2].default_value = tuple(c * .60 for c in base[:3]) + (1,)
            links.new(mask.outputs[0], mix.inputs[0])
            links.new(mix.outputs[0], shader.inputs['Base Color'])
            shader.inputs['Roughness'].default_value = .44
            report['surface_materials'].append('Floor: estimated 0.6m tile joints')
    return report


def main():
    args = arguments()
    if args.edge < 64 or args.samples < 1:
        raise ValueError('edge >= 64 and samples >= 1 required')
    source = args.source.resolve(); out = args.out.resolve()
    if out == source:
        raise ValueError('Output must differ from source')
    out.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((source / 'scene_bundle.json').read_text())
    views = {v['id']: v for v in bundle['views']}
    if set(args.views) - set(views):
        raise ValueError('Unknown view IDs')
    if args.trials and len(args.views) != 1:
        raise ValueError('Camera trials require exactly one view')
    trials = json.loads(args.trials.read_text()) if args.trials else {'final': None}
    if not isinstance(trials, dict) or not trials:
        raise ValueError('Trials must be a nonempty JSON object')
    for name in trials:
        if not name or name in ('.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Unsafe trial name')
    camera_config = json.loads(args.camera_config.read_text()) if args.camera_config else {}
    bpy.ops.wm.open_mainfile(filepath=str(source / 'scene.blend'))
    scene = bpy.context.scene
    # Device preferences are process-local, even when scene.cycles.device is saved.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scene_factory'))
    from generate_v2 import configure_device
    device = configure_device(scene, 'AUTO')
    report = {'source': str(source), 'settings': {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}, 'changes': refine_materials(
                  scene, args.atlas, args.detail, args.floor_tiles), 'views': [],
              'limitations': 'Estimated geometry and camera poses. Packaging atlas is appearance detail, not SKU identification.'}
    report['device'] = device
    refined_bundle = copy.deepcopy(bundle)
    scene.view_settings.exposure += args.exposure
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.resolution_percentage = 100
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = .025
    for trial_name, trial_overrides in trials.items():
        for view_id in args.views:
            view = views[view_id]
            camera = bpy.data.objects['ReferenceCamera_' + view_id]
            overrides = trial_overrides if args.trials else camera_config.get(view_id, {})
            if not isinstance(overrides, dict):
                raise ValueError('Camera overrides must be an object')
            values = {**view['camera'], **overrides}
            camera.location = values['position']
            camera.rotation_euler = (Vector(values['target']) - camera.location).to_track_quat('-Z', 'Y').to_euler()
            camera.rotation_euler.rotate_axis('Z', math.radians(values.get('roll_degrees', 0)))
            camera.data.lens = values['lens_mm']
            for rv in refined_bundle['views']:
                if rv['id'] == view_id:
                    rv['camera'] = values
            scene.camera = camera
            factor = args.edge / max(view['width'], view['height'])
            scene.render.resolution_x = round(view['width'] * factor)
            scene.render.resolution_y = round(view['height'] * factor)
            trial_out = out / trial_name if args.trials else out
            trial_out.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(trial_out / ('rgb_' + view_id + '.png'))
            bpy.context.view_layer.update()
            started = time.monotonic()
            bpy.ops.render.render(write_still=True)
            report['views'].append({'id': view_id, 'trial': trial_name, 'seconds': round(time.monotonic() - started, 2),
                                    'camera_overrides': overrides, 'render': scene.render.filepath})
            (out / 'refinement.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save:
        # Preserve source originals; pack replacement textures for portable editing.
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(out / 'scene.blend'))
    # This bundle records camera overrides only; the refinement manifest and blend
    # additionally record the material graph changes and exposure actually used.
    (out / 'scene_bundle.json').write_text(json.dumps(refined_bundle, ensure_ascii=False, indent=2))
    print('REFINEMENT_COMPLETE', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
