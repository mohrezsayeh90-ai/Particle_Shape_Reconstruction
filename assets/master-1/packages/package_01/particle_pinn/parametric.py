"""Parametric S-channel PINN with exact velocity boundary conditions.

Positive inputs are sampled uniformly in log10 space, then invalid geometries
are rejected. Coordinates are physical nondimensional Cartesian coordinates.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from torch import nn

BASE=np.array([.2,.5,1.,.025,1000.])
NAMES=('rin','rout','re','diameter','density_ratio')


@dataclass(frozen=True)
class Case:
    rin: float=.2
    rout: float=.5
    re: float=1.
    diameter: float=.025
    density_ratio: float=1000.

    @property
    def radius(self): return (self.rin+self.rout)/2
    @property
    def h(self): return (self.rout-self.rin)/2
    @property
    def length(self): return 2*math.pi*self.radius
    def array(self): return np.array([getattr(self,n) for n in NAMES])
    def valid(self):
        return (np.isfinite(self.array()).all() and min(self.array())>0
                and self.rin<self.rout and self.h/self.radius<=.85
                and self.diameter < 2*self.h)
    def json(self): return asdict(self)


def anchors():
    return {name:Case(*(BASE*factor)) for name,factor in [('low',.1),('base',1.),('high',10.)]}


def sample_cases(rng,n,span=1.):
    """Conditional log-uniform distribution, not a full Cartesian product."""
    accepted=[];draws=0
    while len(accepted)<n:
        values=BASE*10**rng.uniform(-span,span,(max(32,2*n),5))
        for row in values:
            draws+=1
            if Case(*row).valid(): accepted.append(row)
            if len(accepted)==n: break
    return np.array(accepted),draws-n


def map_points(q,eta,params):
    """Torch analytic tube coordinates; q=s/L and eta=n/h."""
    r=(params[:,0]+params[:,1])/2;h=(params[:,1]-params[:,0])/2
    second=q>.5;angle=torch.where(second,2*math.pi*q-math.pi,2*math.pi*q)
    sin,cos=torch.sin(angle),torch.cos(angle)
    sin=torch.where((q==0)|(q==.5)|(q==1),torch.zeros_like(sin),sin)
    x=torch.where(second,3*r-r*cos,r-r*cos)
    y=torch.where(second,-r*sin,r*sin)
    nx=torch.where(second,cos,-cos);ny=sin
    return torch.stack((x+h*eta*nx,y+h*eta*ny),dim=1)


def coordinates(xy,params):
    r=(params[:,0]+params[:,1])/2;h=(params[:,1]-params[:,0])/2
    # The legacy spline has tiny cap-normal roundoff. Select the first/last
    # cap by x as well as y so an outlet point never maps into the first bend.
    second=(xy[:,0]>3*r)|((xy[:,0]>=r)&(xy[:,1]<0))
    dx=xy[:,0]-torch.where(second,3*r,r)
    dy=torch.where(second,-xy[:,1],xy[:,1]).clamp_min(0)
    alpha=torch.atan2(dy,dx)
    angle=math.pi-alpha
    q=torch.where(second,angle+math.pi,angle)/(2*math.pi)
    radius=torch.sqrt(dx.square()+dy.square())
    eta=torch.where(second,(r-radius)/h,(radius-r)/h)
    tangent=torch.stack((torch.sin(angle),torch.where(second,-torch.cos(angle),torch.cos(angle))),dim=1)
    return q,eta,tangent


class ParametricFlow(nn.Module):
    """5 -> 128 x 6 -> 3; log geometry/Re conditioning; hard wall/cap velocity."""
    def __init__(self,width=128,depth=6):
        super().__init__();self.width=width;self.depth=depth
        layers=[nn.Linear(5,width),nn.Tanh()]
        for _ in range(depth-1): layers.extend([nn.Linear(width,width),nn.Tanh()])
        layers.append(nn.Linear(width,3));self.net=nn.Sequential(*layers)
        self.register_buffer('base',torch.tensor(BASE[:3],dtype=torch.float32))
        for layer in self.net:
            if isinstance(layer,nn.Linear): nn.init.xavier_normal_(layer.weight);nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.net[-1].weight)

    def features(self,xy,p):
        r=(p[:,0]+p[:,1])/2
        return torch.cat((torch.stack((xy[:,0]/r-2,xy[:,1]/r),dim=1),torch.log10(p[:,:3]/self.base)),dim=1)

    def forward(self,xy,p):
        q,eta,t=coordinates(xy,p)
        raw=self.net(self.features(xy,p))
        r=(p[:,0]+p[:,1])/2;h=(p[:,1]-p[:,0])/2
        outlet=torch.stack((4*r,torch.zeros_like(r)),dim=1)
        pg=self.net(self.features(outlet,p))[:,2]
        mask=torch.sin(math.pi*q).square()*(1-eta.square())
        # A raw circular tangent has an |y| cusp at the S-bend interface.
        # Smooth its horizontal extension so velocity has no prescribed
        # first-derivative jump there. Wall/cap velocities remain exact.
        smooth=xy[:,1].abs()/torch.sqrt(xy[:,1].square()+(.04*r).square())
        base_t=torch.stack((t[:,0]*smooth,t[:,1]),dim=1)
        uv=(1-eta.square())[:,None]*base_t+mask[:,None]*raw[:,:2]
        pressure=4*math.pi*r/h.square()*(1-q+raw[:,2]-pg)
        return torch.cat((uv,pressure[:,None]),dim=1)

    def evaluate(self,xy,case):
        device=self.base.device
        a=torch.as_tensor(np.atleast_2d(xy),dtype=torch.float32,device=device)
        p=torch.as_tensor(case.array()[None,:3],dtype=torch.float32,device=device).expand(len(a),-1)
        with torch.no_grad(): return self(a,p).cpu().numpy()


def grad(y,x): return torch.autograd.grad(y.sum(),x,create_graph=True)[0]


def pde_residual(model,xy,p):
    out=model(xy,p);u,v,pressure=out.unbind(1)
    du,dv,dp=grad(u,xy),grad(v,xy),grad(pressure,xy)
    lu=grad(du[:,0],xy)[:,0]+grad(du[:,1],xy)[:,1]
    lv=grad(dv[:,0],xy)[:,0]+grad(dv[:,1],xy)[:,1]
    h=(p[:,1]-p[:,0])/2;re=p[:,2]
    return torch.stack((h*(du[:,0]+dv[:,1]),
        h.square()*(re*(u*du[:,0]+v*du[:,1])+dp[:,0]-lu),
        h.square()*(re*(u*dv[:,0]+v*dv[:,1])+dp[:,1]-lv)),dim=1)


def drag_acceleration(fluid,velocity,diameter,re,density_ratio):
    """Schiller-Naumann for Rep<1000; Cd=.44 beyond; isolated rigid sphere."""
    slip=np.asarray(fluid)-np.asarray(velocity)
    rep=re*diameter*np.linalg.norm(slip,axis=-1)
    correction=np.where(rep<1000,1+.15*rep**.687,.44*rep/24)
    return 18*correction[...,None]*slip/(density_ratio*re*diameter**2)
