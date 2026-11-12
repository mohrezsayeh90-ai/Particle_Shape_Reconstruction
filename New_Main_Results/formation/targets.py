"""Deterministic buffered masks, hexagonal lattices, layouts, and covariance."""
import numpy as np
from functools import lru_cache
from shapely.geometry import Point,Polygon,LineString
from shapely.affinity import scale,translate
from scipy.spatial.distance import pdist

PITCH=.027;RADIUS=.0125;BUFFER=.0004


def mask_shape(name,center=(1.52,.15),width=.88,height=.23):
    cx,cy=center
    segment=(width-height)/2
    stadium=LineString([(-segment,0),(segment,0)]).buffer(height/2,quad_segs=128)
    vertices={
      'arrow':[(-1,-.72),(-.22,-.72),(-.22,-1),(1,0),(-.22,1),(-.22,.72),(-1,.72),(-.64,0)],
      'fish':[(-1,.85),(-.58,.37),(-.28,.78),(.25,.95),(.70,.72),(1,0),(.70,-.72),(.25,-.95),(-.28,-.78),(-.58,-.37),(-1,-.85),(-.82,0)]}
    if name in ('stadium','perforated'):
        shape=stadium
        if name=='perforated':
            for u in (-.38,.38):
                hole=scale(Point(u*width/2,0).buffer(1,quad_segs=96),.18*width/2,.33*height/2)
                shape=shape.difference(hole)
    else:shape=Polygon([(u*width/2,v*height/2) for u,v in vertices[name]])
    return translate(shape,cx,cy)


@lru_cache(maxsize=32)
def ordered_lattice(name,center=(1.52,.15),width=.88,height=.23):
    shape=mask_shape(name,center,width,height);safe=shape.buffer(-RADIUS-BUFFER)
    candidates=[]
    for j in range(-20,21):
        for i in range(-50,51):
            x=center[0]+PITCH*(i+.5*(j%2));y=center[1]+j*PITCH*np.sqrt(3)/2
            if safe.covers(Point(x,y)):candidates.append((x,y))
    points=np.array(sorted(candidates));order=[]
    if not len(points):return points
    distance=np.sum((points-center)**2,axis=1);chosen=int(np.argmin(distance))
    nearest=np.full(len(points),np.inf)
    for _ in range(len(points)):
        order.append(chosen);nearest=np.minimum(nearest,np.sum((points-points[chosen])**2,axis=1))
        nearest[order]=-1;chosen=int(np.argmax(nearest))
    return points[order]


def target_points(case):
    center=tuple(case.get('center',[1.52,.15]));n=case['N'];name=case['shape']
    if name=='ellipse':
        a=np.arange(6)*np.pi/3
        base=np.vstack([np.column_stack([.080*np.cos(a),.080*np.sin(a)]),
                        np.column_stack([.0465*np.cos(a+np.pi/6),.0465*np.sin(a+np.pi/6)])])
        ratio=case.get('q',1.);theta=np.deg2rad(case.get('theta',0.))
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        return base@np.diag([np.sqrt(ratio),1/np.sqrt(ratio)])@rotation.T+center
    if name=='small_stadium':
        # Translate an ordering generated only once around the origin.
        return ordered_lattice('stadium',(0.,0.),.32,.11)[:n]+center
    if name=='smoke':return np.array([[1.25,.105],[1.38,.15],[1.51,.195],[1.64,.15],[1.77,.105],[1.9,.195]])[:n]
    return ordered_lattice(name,center)[:n]


def make_ports(case):
    if case.get('mode')=='free':return []
    m=case.get('m',3);layout=case.get('layout','equispaced')
    if layout=='wall':
        sequence=[3,4,2,5,1,6];top=set(sequence[:case['top_count']])
        if case.get('reflected'):top=set(range(1,7))-top
        return [dict(id=f'P{j}',side=int(j in top),x=j*2.2/7) for j in range(1,7)]
    fractions=[.50,.25,.75,.125,.875,.375][:m] if layout=='nested' else np.arange(1,m+1)/(m+1)
    return [dict(id=f'{"B" if side==0 else "T"}{j+1}',side=side,x=float(x*2.2)) for side in (0,1) for j,x in enumerate(fractions)]


def preflight(points,requested,ports,cap):
    if len(points)!=requested:return 'target_geometric_infeasibility','insufficient buffered hexagonal centers'
    if np.any(points[:,0]<RADIUS) or np.any(points[:,0]>2.2-RADIUS) or np.any(points[:,1]<RADIUS) or np.any(points[:,1]>.30-RADIUS):
        return 'target_geometric_infeasibility','physical target disks cross channel walls'
    if len(points)>1 and np.min(pdist(points))<.025-1e-10:return 'target_geometric_infeasibility','target disks overlap'
    if ports and cap is not None and len(ports)*cap<requested:return 'structural_capacity_infeasibility','sum of available port capacities below N'
    return None,None


def covariance_metrics(target,achieved):
    def describe(p):
        mean=p.mean(0);cov=(p-mean).T@(p-mean)/len(p);values,vectors=np.linalg.eigh(cov)
        iso=values[-1]<=1e-20 or (values[-1]-values[0])/values[-1]<.05
        angle=None if iso else float(np.rad2deg(np.arctan2(vectors[1,-1],vectors[0,-1]))%180)
        return mean,cov,values,angle
    mt,ct,vt,at=describe(target);ma,ca,va,aa=describe(achieved)
    return dict(target_centroid=mt.tolist(),achieved_centroid=ma.tolist(),centroid_shift=float(np.linalg.norm(ma-mt)),
                target_covariance=ct.tolist(),achieved_covariance=ca.tolist(),
                covariance_relative_frobenius=float(np.linalg.norm(ca-ct)/np.linalg.norm(ct)),
                achieved_axis_ratio=float(np.sqrt(va[-1]/max(va[0],1e-30))),target_axis_angle_deg=at,
                achieved_axis_angle_deg=aa,axis_angle_error_deg=None if at is None or aa is None else float(abs((aa-at+90)%180-90)),
                isotropy_rule='omit angle if eigenvalue gap / largest < 0.05 or largest <=1e-20')
