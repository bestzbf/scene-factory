"""Photo-specific corridor trial; human-assisted image interpretation, not measured recovery.
Builds a bounded V2 scene and the shared V3 multi-view input wrapper.
"""
from pathlib import Path
import json, math, random
from spec_v2 import upgrade_v1, validate, part, material

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'deliveries/2026-10-20/corridor/manual_exploration'
SOURCE=ROOT/'inputs/2026-10-20/corridor/view_01.jpg'
OUT.mkdir(parents=True,exist_ok=True)
W,D,H=2.8,8.0,2.85
camera={'position':[.74,.38,1.63],'target':[-1.43,7.8,1.30],'lens_mm':18.5}
base={'schema_version':1,'scene_type':'other','description':'Photo 3: narrow institutional corridor with room 323, rear glazed wall and polished cream tiles. Dimensions and hidden geometry are inferred.','seed':20261020,'room':{'width':W,'depth':D,'height':H,'wall_color':[.78,.79,.77],'floor_color':[.60,.56,.46],'ceiling_color':[.75,.76,.74]},'camera':camera,'lighting':{'energy':500,'color':[1,.96,.90],'world_strength':.20},'objects':[{'id':'ceiling_detector','category':'box','position':[.3,5.25,2.77],'dimensions':[.07,.07,.045],'rotation_z_deg':0,'color':[.82,.82,.78],'repeat':[1,1],'spacing':[0,0],'confidence':.65}], 'assumptions':[
'Interpretation provenance: assistant visually inspected photo 3 and authored this declarative scene; no fresh autonomous vision-model batch inference is claimed.',
'Corridor estimated width 2.8 m, visible depth 8.0 m, ceiling 2.85 m using an approximately 0.9 m door, 0.6 m ceiling tiles and familiar architectural proportions; these are not measurements.',
'Camera estimated 1.63 m high with 18.5 mm lens and -13 degree roll; exposure and white balance approximate the source.',
'Left room 323 is a closed paneled door; further left doorway is genuinely open into a simplified dark side room.',
'Back wall is replaced with a real window opening and editable frames/guardrail; glass is simplified to avoid falsely opaque daylight.',
'Only local sign/number/extinguisher markings and a small external foliage patch use reference pixels. No complete scene image is used as a background substitute.',
'Unseen doorway interior and outside scenery are inferred; trees are represented by local cropped foliage on a distant outdoor plane, with approximate spatial depth.',
'Floor roughness, luminaire power and wall reflectance are visually fitted, not recovered physical measurements.'
]}
s=upgrade_v1(base,1280,960,SOURCE.name)
s['camera']['roll_degrees']=-13
s['render']={'exposure':.20,'world_color':[.72,.80,.90],'replace_base_lights':True}
materials=[
material('paint',[.80,.81,.79],roughness=.88,noise_scale=95,bump_strength=.08,bump_distance=.0009),
material('ceiling',[.75,.77,.75],roughness=.9,noise_scale=170,bump_strength=.08,bump_distance=.0008),
material('tile',[.65,.61,.53],roughness=.075,noise_scale=30,bump_strength=.035,bump_distance=.0005),
material('tile_alt',[.63,.59,.51],roughness=.10),
material('grout',[.21,.20,.17],roughness=.8),
material('black_stone',[.025,.033,.031],roughness=.16,noise_scale=90,bump_strength=.08,bump_distance=.0006),
material('door',[.49,.51,.48],roughness=.47,noise_scale=70,bump_strength=.045,bump_distance=.0007),
material('door_edge',[.39,.42,.39],roughness=.53),
material('aluminium',[.50,.54,.54],roughness=.23,metallic=.72),
material('ceiling_grid',[.36,.39,.38],roughness=.65,metallic=.13),
material('brass',[.27,.15,.07],roughness=.35,metallic=.68),
material('black',[.008,.011,.012],roughness=.45),
material('red',[.29,.012,.015],roughness=.5),
material('bin_green',[.018,.21,.095],roughness=.45),
material('bin_lid',[.03,.16,.09],roughness=.23),
material('paper',[.69,.70,.59],roughness=.85),
material('side_room',[.24,.26,.25],roughness=.8),
material('exterior', [1,1,1],roughness=1,texture_id='foliage'),
material('sign', [1,1,1],roughness=.7,texture_id='wall_sign'),
material('number', [1,1,1],roughness=.55,texture_id='number_323'),
material('fire_label', [1,1,1],roughness=.65,texture_id='fire_box_label'),
]
s['materials']=materials
s['surface_overrides']={'floor':'tile','ceiling':'ceiling','left':'paint','right':'paint','back':'paint'}
def quad(id,p): return {'id':id,'quad':[[x/1280,y/960] for x,y in p]}
s['textures']=[quad('foliage',[(998,306),(1050,299),(1036,387),(989,386)]),quad('wall_sign',[(485,272),(538,282),(543,361),(490,359)]),quad('number_323',[(318,170),(356,181),(354,201),(321,190)]),quad('fire_box_label',[(150,708),(260,667),(278,784),(182,825)])]
def B(p,size,mat='',rot=None):return part('box',position=list(p),size=list(size),material_id=mat,rotation_degrees=rot or [0,0,0])
def C(p,r,h,mat='',rot=None):return part('cylinder',position=list(p),radius=r,size=[0,0,h],material_id=mat,rotation_degrees=rot or [0,0,0])
def P(p,w,h,mat='',rot=None):return part('image_plane',position=list(p),size=[w,0,h],material_id=mat,rotation_degrees=rot or [0,0,0])
def G(id,parts,mat='paint',pos=(0,0,0),rot=(0,0,0),confidence=.8,observed=True,support=''):
 s['custom_objects'].append({'id':id,'position':list(pos),'rotation_degrees':list(rot),'material_id':mat,'confidence':confidence,'observed':observed,'support_id':support,'parts':parts})
# Left wall contains a real open doorway from y=5.10 to 6.15. Near room door has its own shallow frame.
G('left_wall_segments',[B((-W/2-.075,2.55,H/2),(.15,5.1,H)),B((-W/2-.075,7.075,H/2),(.15,1.85,H)),B((-W/2-.075,5.625,2.58),(.15,1.05,.54))])
G('side_room_inferred',[B((-2.5,5.625,-.045),(2.2,1.1,.09),'black_stone'),B((-3.55,5.625,1.4),(.10,1.1,2.8),'side_room'),B((-2.5,6.23,1.4),(2.2,.1,2.8),'side_room'),B((-2.5,5.04,1.4),(2.2,.1,2.8),'side_room')],observed=False,confidence=.28)
# Glossy tile grid. Thin grout slightly above the continuous reflective floor.
p=[]
for x in [-1.2,-.6,0,.6,1.2]:p.append(B((x,D/2,.001),(.0025,D,.002),'grout'))
for i in range(1,14):p.append(B((0,.6*i,.001),(W,.0025,.002),'grout'))
# Black stone wall skirting and floor border, independently editable.
for x in [-1.355,1.355]:p.append(B((x,D/2,.006),(.09,D,.012),'black_stone'))
for y,len_ in [(2.55,5.10),(7.075,1.85)]:p.append(B((-1.383,y,.095),(.032,len_,.19),'black_stone'))
p.extend([B((1.383,4,.095),(.032,8,.19),'black_stone'),B((0,7.95,.095),(2.8,.035,.19),'black_stone')])
G('polished_floor_grout_and_border',p,confidence=.94)
# Fine ceiling suspension grid, interrupted visually by wall boundaries.
p=[]
for x in [-1.2,-.6,0,.6,1.2]:p.append(B((x,D/2,H-.008),(.011,D,.016),'ceiling_grid'))
for i in range(14):p.append(B((0,.6*i,H-.009),(W,.011,.017),'ceiling_grid'))
G('suspended_ceiling_grid',p,confidence=.93)
# Closed room 323 paneled metal door, local X is horizontal along corridor; front=-Y rotated onto left wall.
p=[B((0,0,1.12),(.91,.055,2.24),'door')]
for x in [-.49,.49]:p.extend([B((x,.005,1.16),(.07,.11,2.32),'door_edge'),B((x,-.06,1.16),(.045,.035,2.32),'door')])
p.extend([B((0,.005,2.30),(1.05,.11,.08),'door_edge'),B((0,-.06,2.30),(1.03,.035,.05),'door')])
# Raised panel mouldings, border rails rather than textured flat panels.
for x in [-.225,.225]:
 for z,ph,pw in [(1.94,.24,.27),(1.40,.20,.27),(.91,.65,.28),(.30,.37,.28)]:
  p.append(B((x,-.039,z),(pw,.014,ph),'door_edge'))
  p.append(B((x,-.050,z),(pw-.035,.016,ph-.035),'door'))
  for xx in [x-pw/2,x+pw/2]:p.append(B((xx,-.056,z),(.016,.016,ph),'door'))
  for zz in [z-ph/2,z+ph/2]:p.append(B((x,-.056,zz),(pw,.016,.016),'door'))
p.extend([B((0,-.045,1.67),(.88,.02,.028),'door_edge'),B((.39,-.07,.99),(.055,.035,.25),'brass'),C((.39,-.105,1.04),.019,.12,'brass',[90,0,0]),B((.33,-.16,1.04),(.14,.025,.024),'brass'),P((0,-.064,1.91),.145,.07,'number')])
G('room_323_paneled_door',p,pos=(-1.375,1.55,0),rot=(0,0,90),confidence=.94)
# Signs faithfully sampled as local rectangular surface details.
G('room_323_wall_sign',[B((0,0,0),(.24,.023,.39),'brass'),P((0,-.015,0),.225,.375,'sign')],pos=(-1.37,2.24,1.63),rot=(0,0,90),confidence=.93)
G('open_room_frame',[B((-1.36,5.10,1.15),(.08,.07,2.3),'door'),B((-1.36,6.15,1.15),(.08,.07,2.3),'door'),B((-1.36,5.625,2.29),(.08,1.12,.065),'door'),B((-1.39,6.07,1.13),(.05,.08,2.23),'aluminium')])
G('exit_sign',[B((-1.36,5.625,2.59),(.035,.30,.12),'paper'),B((-1.337,5.625,2.59),(.007,.23,.07),'bin_green')])
G('fire_call_buttons',[B((-1.365,4.48,1.40),(.04,.095,.12),'red'),B((-1.36,4.48,1.59),(.04,.07,.09),'red')])
# Front fire-extinguisher cabinet and tube details.
G('fire_extinguisher_cabinet',[B((0,0,.31),(.43,.24,.62),'red'),P((0,-.123,.31),.41,.56,'fire_label'),B((0,0,.63),(.45,.25,.025),'door'),B((-.1,.01,.66),(.22,.19,.05),'paper')],pos=(-1.10,.38,0),rot=(0,0,19),support='floor',confidence=.88)
G('near_black_umbrella',[C((-1.28,.18,.43),.015,.86,'black'),part('curve',points=[[-1.28,.18,.82],[-1.25,.15,.90],[-1.18,.15,.91],[-1.16,.15,.85]],radius=.013,material_id='black')],confidence=.65)
# Back facade: actual aperture; thin side returns and lintel only, with silver window mullions.
G('rear_window_wall',[B((-1.33,D,1.425),(.14,.16,H)),B((1.33,D,1.425),(.14,.16,H)),B((0,D,2.79),(2.52,.16,.12)),B((0,D,.11),(2.52,.16,.22))])
p=[]
for x in [-1.25,-.625,0,.625,1.25]:p.append(B((x,7.975,1.49),(.037,.065,2.54),'aluminium'))
for z in [.23,.93,2.25,2.76]:p.append(B((0,7.975,z),(2.54,.065,.037),'aluminium'))
for z in [.27,.59,.97]:p.append(C((0,7.80,z),.018,2.48,'aluminium',[0,90,0]))
for x in [-1.2,-.6,0,.6,1.2]:p.append(C((x,7.80,.61),.014,.73,'aluminium'))
G('rear_aluminium_window_and_guardrail',p,confidence=.94)
G('outdoor_foliage_reference_patch',[P((0,8.8,1.75),4.2,3.4,'exterior')],observed=True,confidence=.60)
# Right high side window and window reveals, simplified closed high glazing.
G('right_high_window',[B((1.375,6.65,2.27),(.08,1.04,.65),'door_edge'),B((1.33,6.65,2.27),(.025,.96,.57),'black'),B((1.30,6.65,2.27),(.026,.035,.60),'aluminium'),B((1.30,6.65,2.24),(.026,1.0,.03),'aluminium')],confidence=.80)
G('right_wall_notice',[B((1.372,6.05,1.56),(.018,.17,.33),'paper'),B((1.357,6.05,1.70),(.006,.17,.028),'bin_green')],confidence=.77)
# Green recycling bin, open interior, hinged thin lid.
p=[B((0,0,.025),(.36,.36,.05),'bin_green')]
for x in [-.18,.18]:p.append(B((x,0,.23),(.025,.37,.45),'bin_green'))
for y in [-.18,.18]:p.append(B((0,y,.23),(.35,.025,.45),'bin_green'))
p.extend([B((0,0,.45),(.42,.40,.035),'bin_lid'),B((0,-.01,.47),(.28,.23,.009),'black'),B((0,.15,.51),(.39,.035,.12),'bin_green')])
G('green_bin',p,pos=(1.03,7.49,0),support='floor',confidence=.93)
G('water_bottle_stand',[B((1.08,7.83,.74),(.24,.15,.035),'aluminium'),C((1.08,7.83,.91),.063,.30,'paper'),B((1.08,7.83,1.07),(.12,.12,.035),'bin_green')],confidence=.60)
G('corridor_information_stand',[B((-1.08,6.99,.03),(.29,.35,.06),'aluminium'),C((-1.08,6.99,.72),.016,1.4,'aluminium'),B((-1.08,6.99,1.2),(.25,.045,.76),'door'),B((-1.08,6.96,1.2),(.19,.008,.69),'paper')],confidence=.75)
def L(id,p,t,power,size,sy,color):return {'id':id,'type':'AREA','position':p,'target':t,'color':color,'energy':power,'size':size,'size_y':sy,'shape':'RECTANGLE','spread_degrees':160}
s['lights']=[L('window_daylight',[0,8.24,1.83],[-.2,3.0,.75],1050,2.35,2.3,[.89,.95,1]),L('exterior_patch_illumination',[0,8.3,1.85],[0,8.85,1.85],950,3.0,2.6,[1,1,1]),L('ceiling_bounce_fill',[0,4.5,2.82],[0,4.5,.5],25,1.1,2.5,[1,.97,.91])]
validate(s)
(OUT/'scene_spec.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
wrapper={'schema_version':3,'scene_id':'corridor','scene':s,'views':[{'id':'view_01','image':{'width':1280,'height':960,'source_id':SOURCE.name,'path':str(SOURCE)},'camera':{**camera,'roll_degrees':s['camera']['roll_degrees']}}],'texture_sources':[{'texture_id':t['id'],'view_id':'view_01'} for t in s['textures']],'architecture':{'omit_surfaces':['back','left'],'front_wall':True}}
(OUT/'scene_multiview.json').write_text(json.dumps(wrapper,ensure_ascii=False,indent=2)+'\n')
(OUT/'provenance.json').write_text(json.dumps({'mode':'human_assisted_photo_interpretation','source':str(SOURCE),'estimated_room_m':[W,D,H],'custom_groups':len(s['custom_objects']),'parts':sum(len(g['parts']) for g in s['custom_objects']),'assumptions':base['assumptions'],'full_scene_billboard':False},ensure_ascii=False,indent=2)+'\n')
print(OUT, len(s['custom_objects']),sum(len(g['parts']) for g in s['custom_objects']))
