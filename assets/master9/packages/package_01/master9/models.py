"""Residual-trained fluid and local inertial flow-map PINNs. No labelled solutions."""
import torch
from torch import nn
from .physics import *
class MLP(nn.Module):
    def __init__(self,n,out,width=96,depth=3):
        super().__init__();layers=[]
        for _ in range(depth):layers.extend((nn.Linear(n,width),nn.Tanh()));n=width
        layers.append(nn.Linear(n,out));self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x)
class FlowPINN(nn.Module):
    def __init__(self):super().__init__();self.net=MLP(1,1,32,2)
    def forward(self,q,p):return (1-q)*self.net(2*q-1)
class ParticlePINN(nn.Module):
    def __init__(self,flow):
        super().__init__();self.flow=flow;self.net=MLP(9,2)
        for p in self.flow.parameters():p.requires_grad_(False)
    def correction(self,s,q0,v0,u,p):
        vscale=torch.tensor([.001*TS/RV,.06*TS/D],device=s.device,dtype=s.dtype)
        inp=torch.cat((2*s-1,q0/torch.tensor([.7,2.],device=s.device),v0/vscale,u/.5,encode_params(p)),1)
        return self.net(inp)
    def terms(self,s,q0,v0,u,p,second=False):
        n=self.correction(s,q0,v0,u,p)
        ns=torch.cat([torch.autograd.grad(n[:,i].sum(),s,create_graph=True,retain_graph=True)[0] for i in range(2)],1)
        nss=torch.zeros_like(n)
        if second:
            nss=torch.cat([torch.autograd.grad(ns[:,i].sum(),s,create_graph=True,retain_graph=True)[0] for i in range(2)],1)
        t=H*s;eps=eps_of(p);f0=rhs(q0,u,p,self.flow)
        e=torch.exp(-t/eps)
        q=q0+t*f0+eps*(-torch.expm1(-t/eps))*(v0-f0)+t*t*n
        v=f0+e*(v0-f0)+2*t*n+t*t*ns/H
        # Algebraically cancel stiff exponential terms before computing the residual.
        balance=f0+2*(t+eps)*n+(t*t+4*eps*t)*ns/H+eps*t*t*nss/H**2
        return q,v,balance
    def forward(self,s,q0,v0,u,p):
        t=H*s;eps=eps_of(p);f0=rhs(q0,u,p,self.flow)
        return q0+t*f0+eps*(-torch.expm1(-t/eps))*(v0-f0)+t*t*self.correction(s,q0,v0,u,p)

def sample(n,device,seed_generator=None):
    a=torch.rand((n,9),device=device,generator=seed_generator)
    eta=.001*6**a[:,0:1];mean=.005+.025*a[:,1:2];rho=950+150*a[:,2:3]
    p=torch.cat((rho*mean*2*RV/eta,mean,rho),1)
    q=torch.cat((-.7+1.4*a[:,3:4],-2+3.7*a[:,4:5]),1)
    v=torch.cat(((-.001+.002*a[:,5:6])*TS/RV,.06*a[:,6:7]*TS/D),1)
    u=-.5+a[:,7:8];s=a[:,8:9].requires_grad_(True)
    return s,q,v,u,p
