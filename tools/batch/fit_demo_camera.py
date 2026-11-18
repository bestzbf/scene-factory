"""Fit an estimated perspective camera to explicit, reviewed 3D/2D landmarks.

No image warping: output is a Blender camera. Landmarks and residuals are saved.
Requires scipy in addition to the project's image dependencies.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def project(points, parameters, width, height):
    x,y,z,yaw,pitch,roll,lens = parameters
    yaw,pitch,roll = np.radians([yaw,pitch,roll])
    forward=np.array([math.sin(yaw)*math.cos(pitch),math.cos(yaw)*math.cos(pitch),-math.sin(pitch)])
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    rr=right*math.cos(roll)+up*math.sin(roll)
    uu=up*math.cos(roll)-right*math.sin(roll)
    delta=np.asarray(points)-[x,y,z]
    depth=delta@forward
    focal=width*lens/36
    return np.column_stack([width/2+focal*(delta@rr)/depth,height/2-focal*(delta@uu)/depth])


def parameters(camera):
    pos=np.array(camera['position'])
    direction=np.array(camera['target'])-pos
    yaw=math.degrees(math.atan2(direction[0],direction[1]))
    pitch=math.degrees(math.atan2(-direction[2],np.linalg.norm(direction[:2])))
    return np.r_[pos,yaw,pitch,camera.get('roll_degrees',0),camera['lens_mm']]


def fit(spec):
    width,height=spec.get('image_size',[1280,720])
    points=np.array([l['world'] for l in spec['landmarks']])
    targets=np.array([l['pixel'] for l in spec['landmarks']])
    p0=parameters(spec['initial_camera'])
    weights=np.array([l.get('weight',1) for l in spec['landmarks']])
    lower=spec.get('lower',[-3.1,0,2.3,-20,5,-12,12])
    upper=spec.get('upper',[1,6,3.6,20,55,12,40])
    prior=np.array([.6,.8,.25,6,6,3,6])
    def residual(p):
        reproj=((project(points,p,width,height)-targets)*np.sqrt(weights[:,None])).ravel()
        return np.r_[reproj, (p-p0)/prior*spec.get('prior_weight',2)]
    fits=[]
    for lens in (16,20,24,28,32):
        start=np.clip(p0,np.array(lower)+1e-5,np.array(upper)-1e-5)
        start[-1]=np.clip(lens,lower[-1]+1e-5,upper[-1]-1e-5)
        result=least_squares(residual,start,bounds=(lower,upper),loss='soft_l1',f_scale=15,max_nfev=1200)
        fits.append(result)
    best=min(fits,key=lambda r:r.cost)
    p=best.x; yaw,pitch=np.radians(p[3:5])
    target=(p[:3]+np.array([math.sin(yaw)*math.cos(pitch),math.cos(yaw)*math.cos(pitch),-math.sin(pitch)])*5).tolist()
    before=project(points,p0,width,height);after=project(points,p,width,height)
    rows=[]
    for i,l in enumerate(spec['landmarks']):
        rows.append({**l,'before':before[i].tolist(),'after':after[i].tolist(),
                     'before_error_px':float(np.linalg.norm(before[i]-targets[i])),
                     'after_error_px':float(np.linalg.norm(after[i]-targets[i]))})
    names=['x','y','z','yaw','pitch','roll','lens_mm']
    active=[names[i] for i,value in enumerate(best.active_mask) if value]
    return {'camera':{'position':p[:3].tolist(),'target':target,'lens_mm':float(p[6]),'roll_degrees':float(p[5])},
            'solver_success':bool(best.success),'parameters_at_bounds':active,
            'warnings':['Parameters reached allowed bounds; do not assume the fit is a valid calibration.'] if active else [],
            'before_mean_error_px':float(np.linalg.norm(before-targets,axis=1).mean()),
            'after_mean_error_px':float(np.linalg.norm(after-targets,axis=1).mean()),
            'landmarks':rows,'parameters':p.tolist(),
            'limitations':'Manually estimated 3D landmarks and metric scale; fitting residual is not independent geometric accuracy.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('constraints',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();spec=json.loads(args.constraints.read_text());result=fit(spec)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='landmarks'},ensure_ascii=False))
