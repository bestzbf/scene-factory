"""Apply explicit reviewed cameras and architecture to a saved scene, once.

Run in Blender. Outputs a new source for optimize_demo_batch --source-override.
Never changes the input. This is a manual profile, not automatic calibration.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector, Matrix

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scene_factory'))
from generate_scene import material, box
from retail_refinement import support_groups, _fixture_bounds, _move_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('source', 'out', 'profile', 'layout'):
        p.add_argument('--'+k, type=Path, required=True)
    a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    source, out = a.source.resolve(), a.out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Source/output overlap')
    if out.exists():
        raise ValueError('Use a new preparation output directory')
    profile = json.loads(a.profile.read_text())
    bundle = json.loads((source/'scene_bundle.json').read_text())
    layout = json.loads(a.layout.read_text())
    if str(profile['scene_id']) != str(bundle['scene_id']):
        raise ValueError('Scene ID mismatch')
    views = {v['id']: v for v in bundle['views']}
    if set(profile.get('cameras', {})) - set(views):
        raise ValueError('Unknown camera')
    bpy.ops.wm.open_mainfile(filepath=str(source/'scene.blend'))
    if bpy.context.scene.get('reviewed_preparation'):
        raise ValueError('Already prepared')
    entries = {o['id']:o for o in bundle['shared_scene']['base_scene']['objects']}
    semantic = {o['id']:o for o in layout['objects']}
    bpy.context.view_layer.update()
    before_bounds=_fixture_bounds(semantic)
    groups = support_groups(layout['objects'])
    mirrored = set()
    for oid in profile.get('reflect_x_groups', []):
        if oid not in groups:
            raise ValueError('Unknown support root: '+oid)
        mirrored.update(groups[oid])
    transform = Matrix.Diagonal((-1, 1, 1, 1))
    bpy.context.view_layer.update()
    old = {i:bpy.data.objects[i].matrix_world.copy() for i in mirrored if i in entries}
    def depth(oid):
        obj=bpy.data.objects[oid]; count=0
        while obj.parent: count+=1; obj=obj.parent
        return count
    for oid in sorted(old, key=depth):
        matrix=old[oid]
        bpy.data.objects[oid].matrix_world = transform @ matrix
        bpy.context.view_layer.update()
    bpy.context.view_layer.update()
    for oid in mirrored:
        if oid in entries:
            entries[oid]['position'][0] *= -1
            entries[oid]['rotation_z_deg'] = -entries[oid].get('rotation_z_deg', 0)
        semantic[oid]['xy'][0] *= -1
        semantic[oid]['rotation_z_deg'] = -semantic[oid].get('rotation_z_deg', 0)
    replacements=[]
    omitted=set()
    for oid in profile.get('remove_roots', []):
        if oid not in groups: raise ValueError('Unknown root to remove: '+oid)
        for child_id in groups[oid]:
            omitted.add(child_id)
            obj=bpy.data.objects.get(child_id)
            if obj:
                for descendant in list(obj.children_recursive): bpy.data.objects.remove(descendant,do_unlink=True)
                bpy.data.objects.remove(obj,do_unlink=True)
    if profile.get('replace_fixtures'):
        from reviewed_retail_fixtures import replace_shelf, replace_fridge
        atlas=(a.profile.resolve().parent/profile['atlas']).resolve()
        for item in profile['replace_fixtures']:
            oid=item['id']
            for child in groups[oid]:
                if child==oid:continue
                omitted.add(child)
                obj=bpy.data.objects.get(child)
                if obj:
                    for descendant in list(obj.children_recursive):bpy.data.objects.remove(descendant,do_unlink=True)
                    bpy.data.objects.remove(obj,do_unlink=True)
            root=bpy.data.objects[oid]
            fn=replace_fridge if item.get('type')=='fridge' else replace_shelf
            replacements.append(fn(root,item,atlas,bundle['scene_id']))
            entries[oid].update(position=item['position'],dimensions=item['dimensions_m'],rotation_z_deg=item['rotation_z_deg'])
            semantic[oid].update(xy=item['position'][:2],target_dimensions_m=item['dimensions_m'],rotation_z_deg=item['rotation_z_deg'])
        bundle['shared_scene']['base_scene']['objects']=[o for o in entries.values() if o['id'] not in omitted]
        layout['objects']=[o for o in semantic.values() if o['id'] not in omitted]
    originals = {}
    for vid, camera in profile.get('cameras', {}).items():
        vals = [*camera['position'], *camera['target'], camera['lens_mm'], camera.get('roll_degrees', 0)]
        if len(vals) != 8 or not all(math.isfinite(x) for x in vals) or not 8 <= camera['lens_mm'] <= 100:
            raise ValueError('Invalid camera')
        originals[vid] = copy.deepcopy(views[vid]['camera'])
        obj = bpy.data.objects['ReferenceCamera_'+vid]
        obj.location = camera['position']
        direction = Vector(camera['target'])-obj.location
        room=bundle['shared_scene']['base_scene']['room']
        if not (-room['width']/2 < obj.location.x < room['width']/2 and 0 < obj.location.y < room['depth'] and .1 < obj.location.z < room['height']-.02):
            raise ValueError('Reviewed camera outside room: '+vid)
        if direction.length < .01:
            raise ValueError('Degenerate camera')
        obj.rotation_euler = direction.to_track_quat('-Z','Y').to_euler()
        obj.rotation_euler.rotate_axis('Z', math.radians(camera.get('roll_degrees',0)))
        obj.data.lens = camera['lens_mm']
        views[vid]['camera'] = copy.deepcopy(camera)
        for v in layout.get('views', []):
            if v['id'] == vid: v['camera'] = copy.deepcopy(camera)
    bundle['shared_scene']['camera'] = copy.deepcopy(bundle['views'][0]['camera'])
    for item in profile.get('architecture_boxes', []):
        if not all(math.isfinite(v) for v in item['position']+item['dimensions']) or min(item['dimensions']) <= 0:
            raise ValueError('Invalid architectural box')
        mat = material('Reviewed_'+item['id'], item.get('color',[.72,.74,.70]), .8)
        obj = box(None, 'Reviewed_'+item['id'], item['position'], item['dimensions'], mat, 0)
        obj['reviewed_architecture'] = True
    bpy.context.view_layer.update()
    clearance=_move_audit(before_bounds,_fixture_bounds({o['id']:o for o in layout['objects']}),set(entries),bundle['shared_scene']['base_scene']['room'])
    if clearance['status']!='passed':
        raise ValueError('Preparation clearance rejected: '+json.dumps(clearance['issues']))
    if any(not math.isfinite(c) for obj in bpy.context.scene.objects if obj.type=='MESH' for v in obj.data.vertices for c in v.co):
        raise ValueError('Non-finite mesh coordinate')
    out.mkdir(parents=True)
    bpy.context.scene['reviewed_preparation'] = str(a.profile.resolve())
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    for name, value in [('scene_bundle.json',bundle),('initial_layout.json',layout),('preparation_profile.json',profile),
                        ('preparation_report.json',{'source':str(source),'original_cameras':originals,
                         'reflected_object_ids':sorted(mirrored),'replacements':replacements,'removed_supported_object_ids':sorted(omitted),'clearance':clearance,
                         'profile_sha256':hashlib.sha256(a.profile.read_bytes()).hexdigest(),
                         'source_blend_sha256':hashlib.sha256((source/'scene.blend').read_bytes()).hexdigest(),
                         'code_sha256':{str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                             [Path(__file__).resolve(),REPO/'scene_factory/reviewed_retail_fixtures.py',REPO/'scene_factory/retail_stock_geometry.py',REPO/'scene_factory/retail_refinement.py']},
                         'limitations':'Reviewed estimates; reflected imported meshes are authoritative in scene.blend.'})]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    report=json.loads((out/'preparation_report.json').read_text())
    report['prepared_files_sha256']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['scene.blend','scene_bundle.json','initial_layout.json']}
    (out/'preparation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
