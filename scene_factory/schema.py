"""Shared, bounded scene-description contract; units are metres, Z is up."""
import json, math
from pathlib import Path
CATEGORIES=['desk','table','chair','sofa','bed','cabinet','shelf','monitor','computer','plant','window','door','column','beam','linear_light','pendant_light','rug','box']
def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
number={'type':'number'}
def vector(n,lo=-50,hi=50):
    return {'type':'array','items':{'type':'number','minimum':lo,'maximum':hi},'minItems':n,'maxItems':n}
color=vector(3,0,1)
SCHEMA=obj({
 'schema_version':{'type':'integer','enum':[1]},
 'scene_type':{'type':'string','enum':['office','living_room','bedroom','classroom','kitchen','other']},
 'description':{'type':'string'},
 'seed':{'type':'integer','minimum':0,'maximum':2147483647},
 'room':obj({'width':{'type':'number','minimum':2,'maximum':30},'depth':{'type':'number','minimum':2,'maximum':40},'height':{'type':'number','minimum':2,'maximum':8},'wall_color':color,'floor_color':color,'ceiling_color':color}),
 'camera':obj({'position':vector(3),'target':vector(3),'lens_mm':{'type':'number','minimum':12,'maximum':85}}),
 'lighting':obj({'energy':{'type':'number','minimum':50,'maximum':4000},'color':color,'world_strength':{'type':'number','minimum':0,'maximum':2}}),
 'objects':{'type':'array','minItems':1,'maxItems':160,'items':obj({
     'id':{'type':'string'},'category':{'type':'string','enum':CATEGORIES},'position':vector(3),
     'dimensions':vector(3,.01,30),'rotation_z_deg':{'type':'number','minimum':-360,'maximum':360},
     'color':color,'repeat':{'type':'array','items':{'type':'integer','minimum':1,'maximum':12},'minItems':2,'maxItems':2},
     'spacing':vector(2,-10,10),'confidence':{'type':'number','minimum':0,'maximum':1}
 })},
 'assumptions':{'type':'array','items':{'type':'string'},'maxItems':30}
})
def validate(spec):
    # A dependency-free strict recursive validator for the subset above.
    def walk(value,rule,path):
        typ=rule.get('type')
        valid={'object':isinstance(value,dict),'array':isinstance(value,list),'string':isinstance(value,str),'number':type(value) in (int,float),'integer':type(value) is int}.get(typ,False)
        if not valid: raise ValueError(f'{path}: expected {typ}')
        if 'enum' in rule and value not in rule['enum']: raise ValueError(f'{path}: unsupported value {value}')
        if typ=='object':
            if set(value)!=set(rule['properties']): raise ValueError(f'{path}: missing or extra keys: {set(value)^set(rule["properties"])}')
            for k,v in value.items(): walk(v,rule['properties'][k],path+'.'+k)
        elif typ=='array':
            if not rule.get('minItems',0)<=len(value)<=rule.get('maxItems',100000): raise ValueError(f'{path}: invalid length')
            for i,v in enumerate(value): walk(v,rule['items'],f'{path}[{i}]')
        elif typ in ('number','integer'):
            if not math.isfinite(value) or not rule.get('minimum',-math.inf)<=value<=rule.get('maximum',math.inf): raise ValueError(f'{path}: out of bounds')
    walk(spec,SCHEMA,'scene')
    ids=[o['id'] for o in spec['objects']]
    if len(ids)!=len(set(ids)): raise ValueError('scene.objects: duplicate ids')
    count=sum(o['repeat'][0]*o['repeat'][1] for o in spec['objects'])
    if count>600: raise ValueError('expanded object count exceeds 600')
    if sum((a-b)**2 for a,b in zip(spec['camera']['position'],spec['camera']['target']))<.01: raise ValueError('camera target coincides with camera position')
    return spec
if __name__=='__main__':
    Path(__file__).with_name('scene_schema.json').write_text(json.dumps(SCHEMA,indent=2),encoding='utf-8')
