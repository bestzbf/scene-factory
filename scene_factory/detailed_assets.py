"""Detailed, deterministic procedural assets for the photo scene factory.

All dimensions are metres, the anchor is bottom-centre, and local front is -Y.
Assets derive from the locally authored office reconstruction, not a downloaded
model library. Textures are the original locally drawn demonstration screens.

Public integration: ``build(entry, api, seed=23, appearance_policy='legacy')`` returns an Empty, or None for
unsupported kinds. ``api`` exposes box, rod, material and root_object helpers;
``build_asset(entry, seed=23)`` also works standalone inside Blender.

``photo_reconstruction`` is the conservative appearance policy for photograph
matching: neutral desk colours produce laminate rather than invented oak grain,
office chair frames stay dark, and screens have no invented demonstration image.
Explicitly observed screen images can still be supplied as custom image planes
by the scene description. The legacy policy preserves existing example assets.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import bpy
from mathutils import Vector

try:
    from .assets_lib import office_furniture as furniture
except ImportError:
    from assets_lib import office_furniture as furniture

SUPPORTED_KINDS = frozenset({'desk', 'table', 'chair', 'plant', 'monitor', 'computer'})
TEXTURE_DIR = Path(__file__).resolve().parent / 'assets_lib' / 'textures'
SCREEN_FILES = ('code_screen.png', 'desktop_screen.png', 'gallery_screen.png', 'model_screen.png')
SOURCE = 'local_procedural_office_v1'
_TEXTURE_MATERIALS = {}
_SCREEN_MESHES = {}
_WOOD_MATERIALS = {}
_FABRIC_MATERIALS = {}
_LAMINATE_MATERIALS = {}
APPEARANCE_POLICIES = frozenset({'legacy', 'photo_reconstruction'})


def _principled(mat):
    return next(node for node in mat.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')


def _choice(entry, seed):
    # Python's hash changes across processes; sha256 does not.
    data = f"{seed}:{entry['id']}:{entry.get('kind', entry.get('category', ''))}"
    return int.from_bytes(hashlib.sha256(data.encode('utf-8')).digest()[:8], 'big')


def _standalone_api():
    return SimpleNamespace(
        box=lambda root, name, pos, size, mat, bevel=.012: furniture._box(name, root, pos, size, mat, bevel),
        rod=lambda root, name, a, b, radius, mat, vertices=12: furniture._rod(name, root, a, b, radius, mat, vertices),
        material=lambda name, color, roughness=.55, metallic=0, emission=0: _plain_material(name, color, roughness, metallic, emission),
        root_object=lambda name: furniture._root(name, (0, 0, 0)),
    )


def _plain_material(name, color, roughness=.55, metallic=0, emission=0):
    key = hashlib.sha256(repr((tuple(color), roughness, metallic, emission)).encode()).hexdigest()[:16]
    mat = furniture._material('Factory_' + key, tuple(color[:3]), roughness, metallic)
    shader = _principled(mat)
    if emission:
        shader.inputs['Emission Color'].default_value = (*color[:3], 1)
        shader.inputs['Emission Strength'].default_value = emission
    return mat


def _wood(api, color):
    key = tuple(color)
    if key in _WOOD_MATERIALS:
        return _WOOD_MATERIALS[key]
    mat = api.material('Detailed_OakBase', color, .48).copy()
    mat.name = 'Detailed_OakLaminate_' + hashlib.sha256(repr(key).encode()).hexdigest()[:8]
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = _principled(mat)
    coord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeVectorMath'); mapping.operation = 'MULTIPLY'
    mapping.inputs[1].default_value = (1.7, 68, 4)
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 4
    noise.inputs['Detail'].default_value = 3
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = .15
    ramp.color_ramp.elements[0].color = (*(v * .69 for v in color), 1)
    ramp.color_ramp.elements[1].position = .85
    ramp.color_ramp.elements[1].color = (*(min(v * 1.08, 1) for v in color), 1)
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = .10
    bump.inputs['Distance'].default_value = .0012
    links.new(coord.outputs['Generated'], mapping.inputs[0])
    links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], ramp.inputs[0])
    links.new(ramp.outputs['Color'], shader.inputs['Base Color'])
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], shader.inputs['Normal'])
    _WOOD_MATERIALS[key] = mat
    return mat


def _fabric(api, color):
    key = tuple(color)
    if key not in _FABRIC_MATERIALS:
        mat = api.material('Detailed_FabricBase', color, .83).copy()
        mat.name = 'Detailed_WovenUpholstery_' + hashlib.sha256(repr(key).encode()).hexdigest()[:8]
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        noise = nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 220
        noise.inputs['Detail'].default_value = 2
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .13
        bump.inputs['Distance'].default_value = .0012
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], _principled(mat).inputs['Normal'])
        _FABRIC_MATERIALS[key] = mat
    return _FABRIC_MATERIALS[key]


def _laminate(api, color):
    """Fine isotropic surface finish, without colour noise or synthetic grain.

    Object-space coordinates keep the finish microscopic on a metre-sized
    table. A photo's lighting/shadows must not be baked into its base colour.
    """
    key = tuple(color)
    if key not in _LAMINATE_MATERIALS:
        mat = api.material('Detailed_LaminateBase', color, .51).copy()
        mat.name = 'Detailed_MatteLaminate_' + hashlib.sha256(repr(key).encode()).hexdigest()[:8]
        mat['appearance_kind'] = 'matte_laminate'
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        shader = _principled(mat)
        coord = nodes.new('ShaderNodeTexCoord')
        noise = nodes.new('ShaderNodeTexNoise')
        noise.name = 'LaminateMicrofinish'
        noise.inputs['Scale'].default_value = 1800
        noise.inputs['Detail'].default_value = 2
        finish = nodes.new('ShaderNodeMapRange')
        finish.inputs['From Min'].default_value = 0
        finish.inputs['From Max'].default_value = 1
        finish.inputs['To Min'].default_value = .48
        finish.inputs['To Max'].default_value = .54
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .06
        bump.inputs['Distance'].default_value = .00004
        links.new(coord.outputs['Object'], noise.inputs['Vector'])
        links.new(noise.outputs['Fac'], finish.inputs['Value'])
        links.new(finish.outputs['Result'], shader.inputs['Roughness'])
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], shader.inputs['Normal'])
        _LAMINATE_MATERIALS[key] = mat
    return _LAMINATE_MATERIALS[key]


def _is_neutral(color):
    # Colour alone cannot establish a wood species or grain. Near-neutral
    # tables, including dark laminate, therefore receive a plain finish.
    return max(color) - min(color) <= .12


def _screen_mesh(w, h):
    key = (w, h)
    if key not in _SCREEN_MESHES:
        mesh = bpy.data.meshes.new('DetailedMonitorScreenMesh')
        mesh.from_pydata([(-w/2, 0, -h/2), (w/2, 0, -h/2), (w/2, 0, h/2), (-w/2, 0, h/2)], [], [(0, 1, 2, 3)])
        mesh.uv_layers.new()
        for loop, uv in zip(mesh.uv_layers.active.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
            loop.uv = uv
        mesh.update()
        _SCREEN_MESHES[key] = mesh
    return _SCREEN_MESHES[key]


def _unobserved_screen(root, api, w=.617, h=.344, y=-.026, z=.288):
    # Isolate from the color-keyed material cache: an unrelated dark object
    # must never acquire the semantic screen selector merely by matching RGB.
    mat = api.material('Detailed_UnobservedDarkLCD', (.008, .011, .014), .28).copy()
    mat.name = 'Detailed_MonitorScreen'
    mat['scene_material_id'] = 'monitor_screen'
    mat['appearance_kind'] = 'unobserved_dark_screen'
    obj = furniture._mesh_object(root.name + '_UnobservedScreen', _screen_mesh(w, h), root, (0, y, z), mat)
    obj['screen_content'] = 'unobserved; neutral non-emissive display'
    return obj


def _screen(root, filename, w=.617, h=.344, y=-.026, z=.288):
    path = TEXTURE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f'Required detailed monitor texture is missing: {path}')
    if filename not in _TEXTURE_MATERIALS:
        mat = bpy.data.materials.new('DetailedScreen_' + Path(filename).stem)
        mat['scene_material_id'] = 'monitor_screen'
        mat.use_nodes = True
        shader = _principled(mat)
        shader.inputs['Roughness'].default_value = .38
        shader.inputs['Emission Strength'].default_value = .25
        texture = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texture.image = bpy.data.images.load(str(path), check_existing=True)
        texture.image.pack()
        mat.node_tree.links.new(texture.outputs['Color'], shader.inputs['Base Color'])
        mat.node_tree.links.new(texture.outputs['Color'], shader.inputs['Emission Color'])
        _TEXTURE_MATERIALS[filename] = mat
    mat = _TEXTURE_MATERIALS[filename]
    return furniture._mesh_object(root.name + '_ActiveScreen', _screen_mesh(w, h), root, (0, y, z), mat)


def _fit_and_place(root, entry):
    """Fit actual child bounds to requested dimensions, with no anchor drift."""
    bpy.context.view_layer.update()
    inv = root.matrix_world.inverted()
    points = [inv @ obj.matrix_world @ Vector(corner)
              for obj in root.children_recursive if obj.type == 'MESH'
              for corner in obj.bound_box]
    if not points:
        raise ValueError(f'Detailed asset has no geometry: {entry["id"]}')
    mins = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    maxs = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    offset = Vector(((mins.x + maxs.x) / 2, (mins.y + maxs.y) / 2, mins.z))
    for obj in root.children:
        obj.location -= offset
    dimensions = tuple(float(v) for v in entry['dimensions'])
    if len(dimensions) != 3 or any(v <= 0 or not math.isfinite(v) for v in dimensions):
        raise ValueError('Detailed asset dimensions must be three finite positive metres')
    root.scale = tuple(dimensions[i] / (maxs[i] - mins[i]) for i in range(3))
    root.location = entry.get('position', (0, 0, 0))
    angles = entry.get('rotation_degrees', (0, 0, entry.get('rotation_z_deg', entry.get('yaw', 0))))
    root.rotation_euler = tuple(math.radians(float(v)) for v in angles)
    root['requested_dimensions'] = dimensions
    root['anchor'] = 'bottom_center'
    root['front_axis'] = '-Y'
    return root


def build(entry, api=None, seed=23, appearance_policy='legacy'):
    """Return an editable asset; use photo_reconstruction for reference matching.

    The policy changes material choices, not dimensions, placement or topology.
    It deliberately avoids inventing wood grain, screen content or shiny chair
    variants when those properties are not represented in the input contract.
    """
    if appearance_policy not in APPEARANCE_POLICIES:
        raise ValueError('Unknown detailed asset appearance policy: ' + str(appearance_policy))
    reconstruction = appearance_policy == 'photo_reconstruction'
    kind = entry.get('kind', entry.get('category'))
    if kind not in SUPPORTED_KINDS:
        return None
    api = api or _standalone_api()
    if isinstance(api, dict):
        api = SimpleNamespace(**api)
    choice = _choice(entry, seed)
    name = entry['id']
    default = {'desk': (.61, .43, .23), 'table': (.61, .43, .23),
               'plant': (.075, .19, .033), 'chair': (.022, .027, .029)}
    color = tuple(entry.get('color', default.get(kind, (.022, .028, .033)))[:3])
    black = api.material('Detailed_GraphiteABS', (.017, .021, .024), .44 if reconstruction else .38)
    rubber = api.material('Detailed_BlackRubber', (.008, .011, .012), .78)
    metal = api.material('Detailed_BrushedAluminum', (.39, .43, .44), .29, .72)
    pale = api.material('Detailed_WarmWhite', (.80, .80, .76), .58)
    base = api.material('Detailed_' + kind, color, entry.get('roughness', .55))
    root = None

    if kind == 'chair':
        root = furniture.build_chair(name, (0, 0, 0))
        upholstery = _fabric(api, color)
        for obj in root.children_recursive:
            if ('Seat' in obj.name and 'SeatPan' not in obj.name) or 'BackUpholstery' in obj.name:
                obj.material_slots[0].link = 'OBJECT'
                obj.material_slots[0].material = upholstery
            if reconstruction and obj.material_slots and (
                    'BlackNylon' in obj.material_slots[0].material.name or 'BaseSpoke' in obj.name):
                obj.material_slots[0].link = 'OBJECT'
                obj.material_slots[0].material = black
            elif 'BaseSpoke' in obj.name and choice % 2:
                obj.material_slots[0].link = 'OBJECT'
                obj.material_slots[0].material = metal
        variant = 'ergonomic_' + ('alloy_base' if choice % 2 and not reconstruction else 'nylon_base')
    elif kind == 'plant':
        root = furniture.build_plant(name, (0, 0, 0), height=1.7, seed=choice)
        leaf_materials = [api.material('Detailed_Foliage', tuple(min(v * factor, 1) for v in color), .63)
                          for factor in (.72, .94, 1.22, 1.08)]
        for obj in root.children_recursive:
            if obj.name.endswith('_Leaves'):
                for index, mat in enumerate(leaf_materials):
                    obj.data.materials[index] = mat
        variant = 'indoor_tree_' + str(choice % 10000)
    else:
        root = api.root_object(name)
        B = lambda label, pos, size, mat=base, bevel=.006: api.box(root, name + '_' + label, pos, size, mat, bevel)
        R = lambda label, a, b, radius, mat=metal: api.rod(root, name + '_' + label, a, b, radius, mat)
        if kind in ('desk', 'table'):
            laminate = reconstruction and _is_neutral(color)
            tabletop = _laminate(api, color) if laminate else _wood(api, color)
            frame = pale if reconstruction or choice % 2 else metal
            B('LaminateTabletop' if laminate else 'OakTabletop', (0, 0, .762), (1.4, .76, .042), tabletop, .009)
            for x in (-.61, .61):
                for y in (-.30, .30):
                    B('SquareLeg', (x, y, .37), (.043, .043, .74), frame, .004)
                    B('LegFoot', (x, y, .012), (.045, .045, .024), rubber, .003)
                B('SideFrame', (x, 0, .681), (.043, .64, .08), frame, .004)
            B('RearRail', (0, .30, .699), (1.22, .038, .055), frame, .004)
            if kind == 'desk':
                B('CableTrayFloor', (0, .205, .635), (1.12, .19, .012), metal, .003)
                for y in (.116, .294):
                    B('CableTraySide', (0, y, .66), (1.12, .012, .05), metal, .003)
                for x in (-.43, .43):
                    B('CableTrayBracket', (x, .25, .695), (.025, .025, .11), frame, .003)
            variant = ('laminate_' if laminate else 'oak_') + ('white_frame' if reconstruction or choice % 2 else 'metal_frame')
        elif kind == 'monitor':
            B('Bezel', (0, 0, .285), (.65, .044, .39), base, .008)
            B('StandStem', (0, .025, .11), (.045, .045, .19), black, .006)
            B('StandFoot', (0, -.015, .009), (.26, .19, .018), black, .006)
            for i in range(9):
                B('RearVent', (-.205 + i * .026, .024, .24), (.011, .006, .065), rubber, .001)
            B('InventorySticker', (-.24, .024, .386), (.058, .0015, .026), pale, 0)
            for i in range(6):
                B('InventoryBarcode', (-.259 + i * .006, .025, .384), (.002, .001, .012), black, 0)
            B('FrontBadge', (0, -.023, .099), (.048, .001, .003), metal, 0)
            led = api.material('Detailed_MonitorPower', (.12, .55, .24), .35, emission=0 if reconstruction else .6)
            B('PowerIndicator', (.285, -.023, .099), (.006, .001, .002), led, 0)
            if reconstruction:
                _unobserved_screen(root, api)
                variant = 'thin_bezel_unobserved_screen'
            else:
                filename = SCREEN_FILES[choice % len(SCREEN_FILES)]
                _screen(root, filename)
                variant = 'thin_bezel_' + Path(filename).stem
        elif kind == 'computer':
            B('TowerCase', (0, 0, .295), (.225, .43, .59), base, .01)
            B('FrontPanel', (0, -.220, .295), (.201, .013, .558), black, .006)
            for z in (.435, .475):
                B('DriveBay', (0, -.230, z), (.17, .006, .028), rubber, .001)
            for x in (-.05, 0):
                B('USBSurround', (x, -.233, .365), (.028, .003, .010), metal, .001)
                B('USBPort', (x, -.235, .365), (.020, .002, .005), rubber, 0)
            led = api.material('Detailed_TowerPower', (.12, .7, .27), .35, emission=.8)
            R('PowerRing', (.065, -.235, .365), (.065, -.238, .365), .008, metal)
            R('PowerLED', (.065, -.238, .365), (.065, -.240, .365), .005, led)
            for i in range(12):
                B('FrontVent', (-.08 + i * .014, -.23, .17), (.004, .004, .17), rubber, .001)
            for i in range(9):
                B('SideVent', (.113, -.08 + i * .025, .225), (.002, .013, .16), rubber, .001)
            for x in (-.078, .078):
                for y in (-.16, .16):
                    B('TowerFoot', (x, y, -.008), (.038, .055, .016), rubber, .003)
            variant = 'vented_tower_' + str(choice % 2)
            if choice % 2:
                B('FrontTrim', (-.094, -.228, .29), (.006, .003, .50), metal, .001)

    root['asset_variant'] = variant
    root['asset_source'] = SOURCE
    root['asset_quality'] = 'detailed'
    root['appearance_policy'] = appearance_policy
    root['variation_seed'] = str(choice)
    return _fit_and_place(root, entry)


def build_asset(entry, seed=23, appearance_policy='legacy'):
    """Standalone convenience entry point with the same coordinate contract."""
    return build(entry, None, seed, appearance_policy=appearance_policy)
