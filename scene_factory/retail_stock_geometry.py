"""Shared, editable retail stock meshes with fixed object UVs (Blender runtime)."""
import math
import bpy
from mathutils import Vector


class Mesh:
    def __init__(self,name):
        self.name=name;self.vertices=[];self.faces=[];self.indices=[];self.uvs=[];self.materials=[];self.products=0
    def face(self,points,material,uv=None):
        start=len(self.vertices);self.vertices.extend(points);self.faces.append(tuple(range(start,start+len(points))))
        if material not in self.materials:self.materials.append(material)
        self.indices.append(self.materials.index(material))
        self.uvs.append(uv or [(0,0),(1,0),(1,1),(0,1)])
    def box(self,center,size,mat,face_mat=None,face_axes=()):
        x,y,z=center;w,d,h=[v/2 for v in size]
        corners=[(x-w,y-d,z-h),(x+w,y-d,z-h),(x+w,y+d,z-h),(x-w,y+d,z-h),
                 (x-w,y-d,z+h),(x+w,y-d,z+h),(x+w,y+d,z+h),(x-w,y+d,z+h)]
        orders=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(2,3,7,6),(3,0,4,7),(1,2,6,5)]
        axes=[None,None,'-Y','+Y','-X','+X']
        for i,order in enumerate(orders):
            points=[corners[k] for k in order]
            textured=face_mat and i>1 and (not face_axes or axes[i] in face_axes)
            # Cover the physical face without distorting the texture's aspect ratio.
            uv=None
            if textured and face_mat.get('aspect'):
                aspect=(Vector(points[1])-Vector(points[0])).length/max((Vector(points[2])-Vector(points[1])).length,1e-6)
                tex_aspect=face_mat['aspect']
                uw=min(1,aspect/tex_aspect);vh=min(1,tex_aspect/aspect)
                u,v=(1-uw)/2,(1-vh)/2
                uv=[(u,v),(u+uw,v),(u+uw,v+vh),(u,v+vh)]
            self.face(points,face_mat if textured else mat,uv)
    def bottle(self,x,y,z,r,h,body,label,cap,jar=False):
        jug=r>.08
        n=32 if jug else 16
        profile=[(0,.84),(.04,1),(.65,1),(.75,.85),(.80,.44),(.96,.38),(1,.38)]
        if jug:profile=[(0,.74),(.05,1),(.25,1),(.45,1),(.65,1),(.75,.86),(.83,.38),(.95,.34),(1,.34)]
        if jar:profile=[(0,.91),(.06,1),(.88,1),(.92,1.02),(1,1.02)]
        for k,((a,ra),(b,rb)) in enumerate(zip(profile[:-1],profile[1:])):
            mat=cap if a>.89 else label if (.25<=a<.45 if jug else .04<=a<.75) else body
            for j in range(n):
                t=j*2*math.pi/n;t2=(j+1)*2*math.pi/n
                pts=[(x+r*ra*math.cos(t),y+r*ra*math.sin(t),z+a*h),
                     (x+r*ra*math.cos(t2),y+r*ra*math.sin(t2),z+a*h),
                     (x+r*rb*math.cos(t2),y+r*rb*math.sin(t2),z+b*h),
                     (x+r*rb*math.cos(t),y+r*rb*math.sin(t),z+b*h)]
                # Repeat one label twice around a small cylindrical bottle.
                self.face(pts,mat,[(j/(n/2)%1,0),((j+1)/(n/2)%1 or 1,0),((j+1)/(n/2)%1 or 1,1),(j/(n/2)%1,1)])
        radius=profile[-1][1]*r
        self.face([(x+radius*math.cos(j*2*math.pi/n),y+radius*math.sin(j*2*math.pi/n),z+h) for j in range(n)],cap,[(.5+.5*math.cos(j*2*math.pi/n),.5+.5*math.sin(j*2*math.pi/n)) for j in range(n)])
        self.products+=1
    def cup(self,x,y,z,r,h,body,printed):
        """Tapered food cup with a sealed lid, within radius r and height h."""
        n=20
        for j in range(n):
            a=j*2*math.pi/n;b=(j+1)*2*math.pi/n
            self.face([(x+r*.72*math.cos(a),y+r*.72*math.sin(a),z),
                       (x+r*.72*math.cos(b),y+r*.72*math.sin(b),z),
                       (x+r*.97*math.cos(b),y+r*.97*math.sin(b),z+h*.95),
                       (x+r*.97*math.cos(a),y+r*.97*math.sin(a),z+h*.95)],
                      printed,[(j/n,0),((j+1)/n,0),((j+1)/n,1),(j/n,1)])
            self.face([(x+r*.97*math.cos(a),y+r*.97*math.sin(a),z+h*.95),
                       (x+r*.97*math.cos(b),y+r*.97*math.sin(b),z+h*.95),
                       (x+r*math.cos(b),y+r*math.sin(b),z+h),
                       (x+r*math.cos(a),y+r*math.sin(a),z+h)],body)
        ring=[(x+r*math.cos(j*2*math.pi/n),y+r*math.sin(j*2*math.pi/n),z+h) for j in range(n)]
        uv=[(.5+.5*math.cos(j*2*math.pi/n),.5+.5*math.sin(j*2*math.pi/n)) for j in range(n)]
        self.face(ring,printed,uv)
        self.face([(x+r*.72*math.cos(j*2*math.pi/n),y+r*.72*math.sin(j*2*math.pi/n),z) for j in reversed(range(n))],body,uv)
        self.products+=1

    def pouch(self,x,y,z,w,d,h,body,printed,front):
        # A bulging foil packet with a narrow sealed top, rather than a rigid book shape.
        points=[(-w*.44,-d*.35,0),(w*.44,-d*.35,0),(w*.50,-d*.50,h*.55),
                (w*.44,-d*.31,h),( -w*.44,-d*.31,h),(-w*.50,-d*.50,h*.55)]
        if front in ('-X','+X'):
            sign=1 if front=='-X' else -1
            points=[(sign*py,px,pz) for px,py,pz in points]
        pts=[(x+px,y+py,z+pz) for px,py,pz in points]
        # Use all six UV corners; the printed panel follows the pinched bag outline.
        self.face(pts,printed,[(.06,0),(.94,0),(1,.55),(.94,1),(.06,1),(0,.55)])
        back=[(p[0],p[1]+d,p[2]) if front=='-Y' else (p[0]+(d if front=='-X' else -d),p[1],p[2]) for p in pts]
        self.face(back[::-1],body,[(0,0)]*6)
        for i in range(6):self.face([pts[i],pts[(i+1)%6],back[(i+1)%6],back[i]],body)
        self.products+=1
    def finish(self,parent=None,bevel=.001):
        mesh=bpy.data.meshes.new(self.name);mesh.from_pydata(self.vertices,[],self.faces);mesh.update()
        for m in self.materials:mesh.materials.append(m)
        uv=mesh.uv_layers.new(name='UVMap')
        for polygon,mat,coords in zip(mesh.polygons,self.indices,self.uvs):
            polygon.material_index=mat
            for loop,co in zip(polygon.loop_indices,coords):uv.data[loop].uv=co
        obj=bpy.data.objects.new(self.name,mesh);bpy.context.collection.objects.link(obj)
        obj['counted_stock_units']=self.products
        if parent:obj.parent=parent
        if bevel:
            mod=obj.modifiers.new('PhysicalEdgeRadius','BEVEL');mod.width=bevel;mod.segments=2;mod.limit_method='ANGLE'
        return obj

