"""Bounded camera fitting to photo landmarks identified by the vision model.

Only cameras change. All observations and reprojection diagnostics stay in the
selection report. Sparse, inconsistent, or boundary-hitting fits are rejected.
"""
import copy
import math

from schema import obj, vector

CORNERS = {
    'bottom_center': (0,0,0), 'top_center': (0,0,1),
    'bottom_front_left': (-.5,-.5,0), 'bottom_front_right': (.5,-.5,0),
    'bottom_back_left': (-.5,.5,0), 'bottom_back_right': (.5,.5,0),
    'top_front_left': (-.5,-.5,1), 'top_front_right': (.5,-.5,1),
    'top_back_left': (-.5,.5,1), 'top_back_right': (.5,.5,1),
}
SCHEMA = {'type':'array','maxItems':240,'items':obj({
    'view_id':{'type':'string','minLength':1,'maxLength':80},
    'object_id':{'type':'string','minLength':1,'maxLength':80},
    'corner':{'type':'string','enum':list(CORNERS)},
    'image_xy':vector(2,0,1),
    'confidence':{'type':'number','minimum':0,'maximum':1},
})}
CONTRACT = '''
CAMERA LANDMARKS: Return camera_anchors with 6–12 reliable correspondences for each view when identifiable, or no anchors for an ambiguous view. Each anchor is a specific named object's bottom/top bounding corner in its local axes, with the corresponding visible point in the ORIGINAL PHOTO (image_xy normalized x/width,y/height from upper left). Front is local -Y before rotation_z_deg. Top is the allocated total object height, including merchandise, not an arbitrary lower shelf. Use at least 3 distinct objects spread in depth and across the photo, and both vertical/horizontal edges. Never label an occluded or guessed corner as high confidence. Do not copy coordinates from the rendered image; the local numerical solver projects the declared 3D points and fits the camera to these original-photo observations. These are approximate visual correspondences, not surveyed measurements. They will be rejected if inconsistent or underconstrained. The repaired layout must remain one scene shared by all views.
'''


def project(points, parameters, width, height):
    import numpy as np
    x,y,z,yaw,pitch,roll,lens=parameters
    yaw,pitch,roll=np.radians([yaw,pitch,roll])
    forward=np.array([math.sin(yaw)*math.cos(pitch),math.cos(yaw)*math.cos(pitch),-math.sin(pitch)])
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    rr=right*math.cos(roll)+up*math.sin(roll)
    uu=up*math.cos(roll)-right*math.sin(roll)
    delta=np.asarray(points)-[x,y,z];depth=delta@forward
    safe=np.where(depth>.05,depth,.05)
    # Blender AUTO sensor fit uses the longer image edge for square pixels.
    focal=max(width,height)*lens/36
    return np.column_stack([width/2+focal*(delta@rr)/safe,height/2-focal*(delta@uu)/safe]),depth


def parameters(camera):
    import numpy as np
    pos=np.array(camera['position']);direction=np.array(camera['target'])-pos
    return np.r_[pos, math.degrees(math.atan2(direction[0],direction[1])),
                 math.degrees(math.atan2(-direction[2],np.linalg.norm(direction[:2]))),
                 camera.get('roll_degrees',0),camera['lens_mm']]


def fit_camera(camera, world, image_xy, confidence, room, width, height):
    import numpy as np
    from scipy.optimize import least_squares
    p0=parameters(camera);points=np.array(world);target=np.array(image_xy)*[width,height]
    weights=np.sqrt(confidence)
    lower=np.maximum(p0-[1.25,1.25,.6,40,35,8,18],[-room['width']/2+.03,.03,.4,p0[3]-40,-10,-45,12])
    upper=np.minimum(p0+[1.25,1.25,.6,40,35,8,18],[room['width']/2-.03,room['depth']-.03,room['height']-.03,p0[3]+40,85,45,85])
    if np.any(lower>=upper) or np.any(p0<lower) or np.any(p0>upper):
        return None,{'status':'rejected','reason':'initial_camera_outside_supported_fit_bounds'}
    prior=np.array([.65,.65,.3,15,15,4,9])
    def residual(p):
        pixels,depth=project(points,p,width,height)
        return np.r_[((pixels-target)*weights[:,None]).ravel(),
                     (p-p0)/prior*3,np.minimum(depth-.1,0)*1000]
    result=least_squares(residual,np.clip(p0,lower+1e-6,upper-1e-6),bounds=(lower,upper),
                         loss='soft_l1',f_scale=12,max_nfev=400)
    before=np.linalg.norm(project(points,p0,width,height)[0]-target,axis=1)
    after,depth=project(points,result.x,width,height);errors=np.linalg.norm(after-target,axis=1)
    record={'status':'rejected','before_median_px':float(np.median(before)),
            'after_median_px':float(np.median(errors)),'after_p90_px':float(np.percentile(errors,90)),
            'solver_success':bool(result.success),'parameters_at_bounds':bool(any(result.active_mask)),
            'anchor_count':len(world),'scope':'photo-estimated landmarks; independent render review still required'}
    if (not result.success or any(result.active_mask) or min(depth)<=.1
            or record['after_median_px']>=record['before_median_px']*.8
            or record['after_median_px']>max(width,height)*.035
            or record['after_p90_px']>max(width,height)*.09):
        return None,record
    p=result.x;yaw,pitch=np.radians(p[3:5])
    direction=np.array([math.sin(yaw)*math.cos(pitch),math.cos(yaw)*math.cos(pitch),-math.sin(pitch)])
    record['status']='applied'
    return {'position':p[:3].tolist(),'target':(p[:3]+direction*5).tolist(),
            'lens_mm':float(p[6]),'roll_degrees':float(p[5])},record


def align(bundle, layout):
    result=copy.deepcopy(bundle);records={}
    anchors=layout.get('camera_anchors',[])
    if not anchors:return result,records
    entries={o['id']:o for o in result['shared_scene']['base_scene']['objects']}
    for view in result['views']:
        observations=[a for a in anchors if a['view_id']==view['id'] and a['confidence']>=.8]
        keys={(a['object_id'],a['corner']) for a in observations}
        if len(keys)!=len(observations):raise ValueError('Duplicate camera anchor')
        if len(observations)<6 or len({a['object_id'] for a in observations})<3:
            records[view['id']]={'status':'skipped','reason':'at_least_6_anchors_on_3_objects_required'};continue
        pixels=[a['image_xy'] for a in observations]
        if (max(p[0] for p in pixels)-min(p[0] for p in pixels)<.3
                or max(p[1] for p in pixels)-min(p[1] for p in pixels)<.2):
            records[view['id']]={'status':'skipped','reason':'anchors_not_spread_across_image'};continue
        world=[]
        for anchor in observations:
            entry=entries[anchor['object_id']];x,y,z=CORNERS[anchor['corner']]
            w,d,h=entry['dimensions'];yaw=math.radians(entry['rotation_z_deg'])
            x,y=x*w,y*d
            world.append([entry['position'][0]+x*math.cos(yaw)-y*math.sin(yaw),
                          entry['position'][1]+x*math.sin(yaw)+y*math.cos(yaw),entry['position'][2]+z*h])
        camera,record=fit_camera(view['camera'],world,pixels,[a['confidence'] for a in observations],
                                result['shared_scene']['base_scene']['room'],view['width'],view['height'])
        records[view['id']]=record
        if camera:view['camera']=camera
    first=result['views'][0]['camera']
    result['shared_scene']['base_scene']['camera']={k:copy.deepcopy(first[k]) for k in ('position','target','lens_mm')}
    result['shared_scene']['camera']['roll_degrees']=first['roll_degrees']
    return result,records
