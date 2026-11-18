"""Blender checks for reviewed layout rejection and bounded prop construction."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/batch'))
from build_reviewed_retail import footprint_audit,prop


def main():
    import bpy
    room={'width':10,'depth':10,'height':3}
    a={'id':'a','position':[0,3,0],'dimensions_m':[2,.6,1.4],'rotation_z_deg':0}
    b={**a,'id':'b','position':[0,3.3,0]}
    assert not footprint_audit([a,b],room)['passed']
    b.update(position=[2,3,0],rotation_z_deg=90)
    assert footprint_audit([a,b],room)['passed']
    b['position']=[6,3,0]
    assert not footprint_audit([b],room)['passed']
    counter={'id':'counter','type':'counter','position':[-1,5,0],'dimensions_m':[1.18,.63,.86]}
    desk={'id':'desk','type':'counter','position':[-1.7,4.6,0],'dimensions_m':[.56,.55,.73]}
    assert not footprint_audit([counter,desk],room)['passed']
    desk.update(position=[-1.79,4.6,0],dimensions_m=[.36,.55,.73])
    assert footprint_audit([counter,desk],room)['passed']
    bad={'id':'bad_stack','position':[0,2,0],'dimensions_m':[.4,.4,.1],
         'type':'stool_stack','count':5}
    try:prop(bad,None)
    except ValueError:pass
    else:raise AssertionError('A negative-height stool stack must be rejected')
    assert hasattr(bpy.data.objects['bad_stack'].rotation_euler,'rotate_axis')
    print('PASS: fixture and desk/counter collision rejection, envelope bounds, invalid stool rejection, Euler roll API')


if __name__=='__main__':main()
