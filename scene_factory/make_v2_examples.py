"""Generate three labelled synthetic fixtures; no photo analysis or network use.

These fixtures test distinct space types and expressive geometry. They are not
claimed to reconstruct any photograph or to pass a visual realism gate.
"""
import json
import math
from pathlib import Path
from spec_v2 import material, part, upgrade_v1, validate

ROOT = Path(__file__).resolve().parent


def entry(identifier, kind, position, size, color, yaw=0, repeat=(1,1), spacing=(0,0)):
    return {'id': identifier, 'category': kind, 'position': position, 'dimensions': size,
            'color': color, 'rotation_z_deg': yaw, 'repeat': list(repeat),
            'spacing': list(spacing), 'confidence': .6}


def base_scene(kind, width, depth, height, objects):
    return {'schema_version': 1, 'scene_type': kind,
            'description': '人工编写的多类型场景测试样例，非自动读图结果', 'seed': 23,
            'room': {'width': width, 'depth': depth, 'height': height,
                     'wall_color': [.68,.66,.61], 'floor_color': [.30,.27,.23],
                     'ceiling_color': [.8,.8,.76]},
            'camera': {'position': [0,.4,1.65], 'target': [0,depth*.62,1.15], 'lens_mm': 24},
            'lighting': {'energy': 600, 'color': [1,.93,.82], 'world_strength': .12},
            'objects': objects,
            'assumptions': ['所有尺寸、材质和光照均为人工测试设定，不能用于评价照片重建准确性。']}


def group(identifier, position, parts, material_id='', support_id='', rotation=(0,0,0)):
    return {'id': identifier, 'position': position, 'rotation_degrees': list(rotation),
            'material_id': material_id, 'confidence': .6, 'observed': False,
            'support_id': support_id, 'parts': parts}


def area(identifier, position, target, energy, color=(1,.93,.82), size=1, size_y=.5):
    return {'id': identifier, 'type': 'AREA', 'position': position, 'target': target,
            'color': list(color), 'energy': energy, 'size': size, 'size_y': size_y,
            'shape': 'RECTANGLE', 'spread_degrees': 120}


def grid(nx, ny, point):
    vertices = [point(i/nx, j/ny) for j in range(ny+1) for i in range(nx+1)]
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i
            faces.append([a,a+1,a+nx+2,a+nx+1])
    return vertices, faces


def blanket(width, depth):
    vertices, faces = grid(20, 24, lambda u,v: [(u-.5)*width,(v-.5)*depth,
        .045*math.sin(u*math.pi*8)*math.sin(v*math.pi)+.018*math.cos(v*math.pi*11)])
    return part('mesh', vertices=vertices, faces=faces)


def pillow(width, depth, height):
    # Closed curved mesh with rounded sides, parameterized as a deformed sphere.
    rings, sectors = 10, 20
    vertices = [[0,0,-height/2]]
    for j in range(1,rings):
        theta = -math.pi/2 + j*math.pi/rings
        for i in range(sectors):
            phi = 2*math.pi*i/sectors
            def signed_power(x): return math.copysign(abs(x)**.55,x)
            vertices.append([width/2*math.cos(theta)*signed_power(math.cos(phi)),
                             depth/2*math.cos(theta)*signed_power(math.sin(phi)),
                             height/2*math.sin(theta)])
    top = len(vertices); vertices.append([0,0,height/2])
    faces = [[0,1+(i+1)%sectors,1+i] for i in range(sectors)]
    for j in range(rings-2):
        for i in range(sectors):
            a = 1+j*sectors+i; b = 1+j*sectors+(i+1)%sectors
            faces.append([a,b,b+sectors,a+sectors])
    faces += [[top,1+(rings-2)*sectors+i,1+(rings-2)*sectors+(i+1)%sectors] for i in range(sectors)]
    return part('mesh', vertices=vertices, faces=faces)


def curtain(width, height):
    vertices, faces = grid(30, 12, lambda u,v: [(u-.5)*width,
        .055*math.sin(u*math.pi*14)+.012*math.sin(v*math.pi*3+u*7),v*height])
    return part('mesh', vertices=vertices, faces=faces)


def enrich(base):
    scene = upgrade_v1(base)
    scene['materials'] = [
        material('oak', [.39,.24,.12], roughness=.48, noise_scale=5, bump_strength=.14, bump_distance=.001),
        material('fabric', [.42,.48,.44], roughness=.87, noise_scale=300, bump_strength=.22, bump_distance=.0005),
        material('cream_fabric', [.72,.68,.57], roughness=.9, noise_scale=350, bump_strength=.18, bump_distance=.0005),
        material('metal', [.035,.04,.045], roughness=.31, metallic=.8),
        material('warm_white', [.77,.74,.67], roughness=.67, noise_scale=35, bump_strength=.12, bump_distance=.0004),
    ]
    scene['surface_overrides']['floor'] = 'oak'
    scene['surface_overrides']['back'] = 'warm_white'
    scene['render']['replace_base_lights'] = True
    return scene


def bedroom():
    base = base_scene('bedroom',5,6,2.8,[
        entry('bed','bed',[0,3.8,0],[1.8,2.1,.68],[.42,.32,.23]),
        entry('left_nightstand','cabinet',[-1.3,4.3,0],[.48,.48,.55],[.36,.24,.15]),
        entry('right_nightstand','cabinet',[1.3,4.3,0],[.48,.48,.55],[.36,.24,.15]),
        entry('wardrobe','cabinet',[-1.9,1.75,0],[.7,1.8,2.35],[.58,.55,.49]),
        entry('window','window',[2.45,3.7,.65],[1.8,.08,1.8],[.73,.76,.77],90),
        entry('rug','rug',[0,3.15,0],[2.6,3.0,.018],[.38,.34,.29]),
    ])
    scene = enrich(base)
    scene['custom_objects'] = [
        group('duvet',[0,3.55,.38],[blanket(1.82,1.5)],'fabric'),
        group('pillows',[0,4.5,.445],[{**pillow(.7,.43,.18),'position':[-.45,0,0]},
                                    {**pillow(.7,.43,.18),'position':[.45,0,0]}],'cream_fabric'),
        group('curtain_left',[2.37,2.7,.35],[curtain(.55,2.15)],'cream_fabric',rotation=(0,0,90)),
        group('curtain_right',[2.37,4.7,.35],[curtain(.55,2.15)],'cream_fabric',rotation=(0,0,90)),
        group('bedside_lamps',[-1.3,4.3,.55],[
            part('cylinder',position=[0,0,.015],radius=.10,size=[0,0,.03],material_id='metal'),
            part('cylinder',position=[0,0,.20],radius=.012,size=[0,0,.36],material_id='metal'),
            part('cylinder',position=[0,0,.39],radius=.13,size=[0,0,.19],material_id='cream_fabric')],support_id='left_nightstand'),
    ]
    scene['lights'] = [area('window_light',[2.3,3.7,1.6],[-.8,3.3,.7],220,(.78,.88,1),1.4,1.7),
                       area('ceiling_light',[0,3.2,2.65],[0,3.2,0],90,size=.65,size_y=.65)]
    return validate(scene)


def living_room():
    base = json.loads((ROOT/'examples/living_room.json').read_text())
    base['description'] = '人工编写的客厅 V2 测试样例，非自动读图结果'
    scene = enrich(base)
    scene['custom_objects'] = [
        group('sofa_cushions',[0,4.7,.72],[{**pillow(.5,.21,.46),'position':[-.8,0,0], 'rotation_degrees':[0,12,-9]},
                                            {**pillow(.5,.21,.46),'position':[.8,0,0], 'rotation_degrees':[0,-12,8]}], 'cream_fabric'),
        group('coffee_books',[.2,3.15,.44],[part('box',size=[.32,.23,.025],position=[0,0,.013],material_id='fabric'),
                                             part('box',size=[.29,.21,.021],position=[.02,0,.037],rotation_degrees=[0,0,11],material_id='cream_fabric')],support_id='coffee'),
        group('window_drapes',[2.84,3.4,.25],[curtain(.65,2.4)],'cream_fabric',rotation=(0,0,90)),
        group('floor_lamp',[-1.9,3.9,0],[
            part('cylinder',radius=.18,size=[0,0,.035],position=[0,0,.018]),
            part('curve',radius=.014,points=[[0,0,.03],[0,0,1.5],[.08,0,1.75],[.4,0,1.8]]),
            part('sphere',radius=.11,position=[.4,0,1.72],material_id='cream_fabric')], 'metal'),
    ]
    scene['lights'] = [area('daylight',[2.75,3.5,1.7],[-1,4.5,1],300,(.82,.90,1),1.5,1.5),
                       area('pendant_emitter',[0,3.3,2.49],[0,3.3,0],100,size=.35,size_y=.35)]
    return validate(scene)


def shop():
    base = base_scene('other',7,8,3.2,[
        entry('rear_shelves','shelf',[-2.0,7.35,0],[1.8,.45,2.35],[.45,.31,.18],repeat=(3,1),spacing=(2,0)),
        entry('island_table','table',[0,4.0,0],[2.0,1.0,.85],[.43,.30,.18]),
        entry('counter','cabinet',[-2.2,2.0,0],[1.4,.65,1.0],[.55,.50,.42]),
        entry('display_rug','rug',[0,4.0,0],[3.0,2.0,.012],[.40,.38,.32]),
    ])
    scene = enrich(base); scene['scene_type'] = 'shop'
    scene['materials'] += [material('bottle_green',[.07,.23,.14],roughness=.22,metallic=.05),
                           material('packaging',[.65,.35,.14],roughness=.58),
                           material('glass',[.85,.91,.92],roughness=.08,transmission=.88)]
    groups = []
    for shelf in range(3):
        pieces=[]
        for row,z in enumerate(.035+j*(2.35-.035)/6 for j in range(6)):
            for i in range(7):
                x=(i-3)*.23
                if (i+row)%2:
                    pieces += [part('cylinder',radius=.055,size=[0,0,.20],position=[x,0,z+.10],material_id='bottle_green'),
                               part('cylinder',radius=.027,size=[0,0,.05],position=[x,0,z+.225],material_id='metal')]
                else:
                    pieces.append(part('box',size=[.13,.15,.23],position=[x,0,z+.115],material_id='packaging'))
        groups.append(group(f'shelf_products_{shelf}',[-2+shelf*2,7.30,0],pieces))
    groups.append(group('island_display',[0,4,.85],[
        part('box',size=[1.35,.55,.06],position=[0,0,.03],material_id='oak'),
        part('box',size=[1.35,.012,.38],position=[0,-.28,.25],material_id='glass'),
        part('box',size=[1.35,.012,.38],position=[0,.28,.25],material_id='glass'),
        *[part('box',size=[.16,.2,.2],position=[(i-2)*.23,0,.16],material_id='packaging') for i in range(5)]],support_id='island_table'))
    groups.append(group('ceiling_track',[0,4.8,3.02],[
        part('box',size=[4.8,.05,.06]),
        *[part('cylinder',radius=.065,size=[0,0,.18],position=[x,0,-.1],rotation_degrees=[20,0,0]) for x in (-2,-1,0,1,2)]],'metal'))
    scene['custom_objects'] = groups
    scene['lights'] = [area(f'track_emitter_{i}',[x,4.8,2.80],[x,7.0,1.1],100,(1,.92,.8),.12,.12)
                       for i,x in enumerate((-2,-1,0,1,2))]
    scene['lights'].append(area('island_light',[0,3.5,2.9],[0,4,.9],180,size=1.0,size_y=.5))
    return validate(scene)


def main():
    directory = ROOT/'examples_v2'; directory.mkdir(exist_ok=True)
    for name, factory in [('bedroom',bedroom),('living_room',living_room),('shop',shop)]:
        spec = factory()
        (directory/(name+'.json')).write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'{name}: {len(spec["custom_objects"])} custom groups, {sum(len(o["parts"]) for o in spec["custom_objects"])} parts')


if __name__ == '__main__': main()
