"""Sensor representation and exact integrals of its quadratic field products."""
import numpy as np
import torch
from .physics import NSENSORS, AMAX

def check_samples(values):
    a=np.asarray(values,dtype=float)
    if a.shape!=(NSENSORS,2) or not np.isfinite(a).all():
        raise ValueError(f'Expected finite field samples with shape ({NSENSORS},2)')
    if np.max(np.abs(a).sum(1))>AMAX+1e-10:
        raise ValueError('Require |a_left(t)|+|a_right(t)| <= sqrt(0.5)')
    return a

def evaluate_and_integrate(a,s):
    """a: [batch,sensors,2], s: [batch,1]. Return a(s), int_0^s [L²,R²,LR].

    Field is piecewise linear; products are piecewise quadratic. Integrals
    are exact for this representation and remain differentiable in s and a.
    """
    n=a.shape[1]-1
    x=s[:,0]*n
    idx=torch.floor(x).long().clamp(0,n-1)
    u=(x-idx)[:,None]
    start=a[:,:-1,:]
    delta=a[:,1:,:]-start
    pairs=((0,0),(1,1),(0,1))
    c0=torch.stack([start[:,:,i]*start[:,:,j] for i,j in pairs],2)
    c1=torch.stack([start[:,:,i]*delta[:,:,j]+delta[:,:,i]*start[:,:,j] for i,j in pairs],2)
    c2=torch.stack([delta[:,:,i]*delta[:,:,j] for i,j in pairs],2)
    full=(c0+c1/2+c2/3)/n
    cumulative=torch.cat((torch.zeros_like(full[:,:1]),full.cumsum(1)),1)
    row=torch.arange(len(a),device=a.device)
    integ=cumulative[row,idx]+(c0[row,idx]*u+c1[row,idx]*u*u/2+c2[row,idx]*u**3/3)/n
    return a[row,idx]+u*delta[row,idx],integ

def random_samples(n,device):
    grid=torch.linspace(0,1,NSENSORS,device=device)
    modes=torch.arange(1,6,device=device)
    phase=2*torch.pi*torch.rand(n,2,5,device=device)
    coeff=torch.randn(n,2,5,device=device)/modes**1.5
    a=(coeff[:,:,:,None]*torch.sin(2*torch.pi*modes[None,None,:,None]*grid+phase[:,:,:,None])).sum(2).transpose(1,2)
    a=a+torch.randn(n,1,2,device=device)
    # Also train on arbitrary sensor values, ramps, constants and no field.
    m=n//4
    a[:m]=2*torch.rand(m,NSENSORS,2,device=device)-1
    a[m:2*m]=a[m:2*m,:1]
    a[2*m:3*m]=a[2*m:3*m,:1]*(1-grid[None,:,None])+a[2*m:3*m,-1:]*grid[None,:,None]
    norm=a.abs().sum(2).amax(1)[:,None,None].clamp_min(1e-8)
    a=a/norm*AMAX*torch.rand(n,1,1,device=device)**.3
    block=max(1,n//32)
    a[:block]=0
    # Explicit domain faces and one-source functions prevent the training
    # distribution from almost always activating both sources below the limit.
    for k,pair in enumerate(((AMAX,0.),(0.,AMAX),(AMAX/2,AMAX/2),(AMAX/2,-AMAX/2))):
        a[(k+1)*block:(k+2)*block]=a.new_tensor(pair)
    a[-(n//8):,:,1]=0
    a[-(n//4):-(n//8),:,0]=0
    return a
