"""Offline examples for renderer tests; not claimed to be automatic photo analyses."""
import json
from pathlib import Path
from schema import validate
root=Path(__file__).resolve().parent/'examples';root.mkdir(exist_ok=True)
def entry(id,kind,pos,size,col,yaw=0,repeat=(1,1),spacing=(0,0)):
 return dict(id=id,category=kind,position=pos,dimensions=size,color=col,rotation_z_deg=yaw,repeat=list(repeat),spacing=list(spacing),confidence=.6)
def scene(typ,w,d,h):
 return {'schema_version':1,'scene_type':typ,'description':'手写合成示例，仅用于验证通用建模器，非自动读图结果','seed':23,'room':{'width':w,'depth':d,'height':h,'wall_color':[.66,.65,.61],'floor_color':[.30,.27,.23],'ceiling_color':[.8,.8,.76]},'camera':{'position':[0,.35,1.6],'target':[0,d*.6,1.25],'lens_mm':20},'lighting':{'energy':650,'color':[1,.93,.82],'world_strength':.2},'objects':[],'assumptions':['此为人工编写的合成布局，不能用于评价照片重建准确性。']}
s=scene('living_room',6,7,3)
s['objects']=[entry('sofa','sofa',[0,4.8,0],[2.5,1.0,1.0],[.18,.27,.32]),entry('coffee','table',[0,3.1,0],[1.5,.7,.44],[.48,.3,.15]),entry('rug','rug',[0,3.5,0],[3.3,2.9,.015],[.45,.42,.34]),entry('bookcase','shelf',[-2.35,5.4,0],[1.,.4,2.1],[.48,.3,.15]),entry('plant','plant',[2.25,5.8,0],[.8,.8,1.8],[.08,.23,.06]),entry('window','window',[2.96,3.5,.8],[2.5,.1,1.8],[.8,.8,.76],90),entry('pendant','pendant_light',[0,3.3,2.5],[.45,.45,.4],[.8,.8,.76])]
validate(s);(root/'living_room.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
s=scene('office',10,13,3.6);s['lighting']['energy']=1500
s['camera']={'position':[.5,.4,1.7],'target':[0,8,1.9],'lens_mm':20}
s['objects']=[entry('desks','desk',[-3.5,3.,0],[1.4,.75,.76],[.59,.43,.24],repeat=(5,4),spacing=(1.45,2.5)),entry('monitors','monitor',[-3.5,3.14,.76],[.61,.2,.46],[.025,.03,.035],repeat=(5,4),spacing=(1.45,2.5)),entry('chairs','chair',[-3.5,2.2,0],[.56,.57,1.05],[.026,.03,.032],180,repeat=(5,4),spacing=(1.45,2.5)),entry('column','column',[3.9,4.8,0],[.7,.7,3.6],[.7,.7,.67]),entry('windows','window',[4.96,2.4,.8],[2.,.08,2.4],[.8,.8,.75],90,repeat=(1,5),spacing=(0,2.1)),entry('tree','plant',[-4.5,6.,0],[.8,.8,2.0],[.07,.23,.06]),entry('battens','linear_light',[-2.5,2.2,3.1],[1.7,.12,.08],[.8,.8,.75],repeat=(3,4),spacing=(2.8,2.7))]
validate(s);(root/'office.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
