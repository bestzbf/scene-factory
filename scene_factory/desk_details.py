"""Deterministic inferred desk accessories, with explicit instance annotations.

Details are placed relative to monitors and must fit on an existing desktop.
They are decorative hypotheses, not detections from the source photograph.
"""
import hashlib, math, random
from mathutils import Vector


def populate(entries, roots, api, seed):
    added, added_roots = [], []
    desks = [(e,r) for e,r in zip(entries,roots) if e['kind'] in ('desk','table')]
    monitors = [(e,r) for e,r in zip(entries,roots) if e['kind']=='monitor']
    solid = [api['aabb'](r) for e,r in zip(entries,roots) if e['kind'] in ('monitor','computer','box','plant')]
    black=api['material']('Accessory graphite',(.023,.028,.032),.48)
    keymat=api['material']('Keycaps',(.055,.066,.072),.64)
    pale=api['material']('Notebook paper',(.8,.78,.66),.85)
    green=api['material']('Thermos sage',(.34,.44,.29),.34)
    red=api['material']('Keyboard accent',(.64,.06,.025),.45)
    silver=api['material']('Accessory aluminum',(.38,.42,.44),.28,.65)
    def support(x,y,z,sx,sy,yaw):
        # All footprint corners must fit one supporting tabletop (in desk-local XY).
        for de,dr in desks:
            top=de['position'][2]+de['dimensions'][2]
            if abs(z-top)>.035: continue
            da=math.radians(de['rotation_z_deg']); ca,sa=math.cos(da),math.sin(da)
            valid=True
            for dx,dy in [(-sx/2,-sy/2),(-sx/2,sy/2),(sx/2,-sy/2),(sx/2,sy/2)]:
                px=x+dx*math.cos(yaw)-dy*math.sin(yaw)-de['position'][0]
                py=y+dx*math.sin(yaw)+dy*math.cos(yaw)-de['position'][1]
                lx=px*ca+py*sa;ly=-px*sa+py*ca
                if abs(lx)>de['dimensions'][0]/2-.025 or abs(ly)>de['dimensions'][1]/2-.025:valid=False;break
            if valid:return de['id']
        return None
    def build(kind,src,pos,size,angle):
        support_id=support(*pos,size[0],size[1],angle)
        if not support_id:return None
        halfx=(abs(math.cos(angle))*size[0]+abs(math.sin(angle))*size[1])/2
        halfy=(abs(math.sin(angle))*size[0]+abs(math.cos(angle))*size[1])/2
        bounds={'min':[pos[0]-halfx,pos[1]-halfy,pos[2]],'max':[pos[0]+halfx,pos[1]+halfy,pos[2]+size[2]]}
        for b in solid:
            if all(min(bounds['max'][k],b['max'][k])-max(bounds['min'][k],b['min'][k])>.003 for k in range(3)):return None
        identifier=src['id']+'__detail_'+kind
        root=api['root_object'](identifier)
        B=lambda name,p,s,m=black,bev=.004:api['box'](root,name,p,s,m,bev)
        if kind=='keyboard':
            B('Keyboard case',(0,0,.008),(size[0],size[1],.016),black,.006)
            for row in range(5):
                for col in range(14):
                    B('Keycap',(-.195+col*.03,-.056+row*.028,.021),(.024,.022,.01),red if row==4 and col==0 else keymat,.002)
            B('Space bar',(-.005,-.055,.027),(.15,.022,.008),keymat,.002)
        elif kind=='mouse':
            api['sphere'](root,'Mouse shell',(0,0,size[2]/2),size,black)
        elif kind=='notebook':
            B('Notebook cover',(0,0,.004),(size[0],size[1],.008),green,.003)
            B('Paper block',(0,0,.012),(size[0]-.006,size[1]-.005,.008),pale,.002)
        elif kind=='thermos':
            api['rod'](root,'Thermos body',(0,0,.005),(0,0,size[2]-.015),size[0]/2,green,32)
            api['rod'](root,'Thermos cap',(0,0,size[2]-.015),(0,0,size[2]),size[0]/2,silver,32)
        elif kind=='power_strip':
            B('Power strip',(0,0,.018),size,pale,.006)
            for y in [-.065,0,.065]:
                for x in [-.012,.012]:B('Socket',(x,y,.037),(.006,.016,.002),black,.001)
        root.location=pos;root.rotation_euler.z=angle
        root['category']=kind;root['geometry_source']='inferred_desktop_detail';root['support_id']=support_id
        entry={'id':identifier,'kind':kind,'category':kind,'position':list(pos),'dimensions':list(size),'rotation_z_deg':math.degrees(angle),'rotation_degrees':[0,0,math.degrees(angle)],'confidence':0.0,'observed':False,'source':'procedural_dressing','support_id':support_id,'parent_monitor_id':src['id'],'notes':'Added detail, not detected in reference photo.'}
        added.append(entry);added_roots.append(root);solid.append(bounds)
        return root
    for m,mroot in monitors:
        z=m['position'][2]; angle=math.radians(m['rotation_z_deg'])
        rng=random.Random(int(hashlib.sha256((str(seed)+m['id']).encode()).hexdigest()[:16],16))
        def point(dx,dy):return (m['position'][0]+dx*math.cos(angle)-dy*math.sin(angle),m['position'][1]+dx*math.sin(angle)+dy*math.cos(angle),z)
        build('keyboard',m,point(0,-.30),(.43,.15,.031),angle)
        build('mouse',m,point(.30,-.30),(.061,.10,.033),angle)
        if rng.random()<.5:build('thermos',m,point(-.43,-.16),(.073,.073,.22),angle)
        else:build('notebook',m,point(-.43,-.20),(.16,.21,.02),angle)
        if rng.random()<.3:build('power_strip',m,point(.39,.03),(.07,.23,.038),angle)
    return added,added_roots
