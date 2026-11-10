"""Physics-lifted branch/trunk DeepONet; no solved-trajectory training labels."""
import torch
from torch import nn
from .physics import *
from .waveforms import evaluate_and_integrate,random_samples

def mlp(n,out,width=96):
    return nn.Sequential(nn.Linear(n,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),
                         nn.Linear(width,width),nn.Tanh(),nn.Linear(width,out))

class MagneticDeepONet(nn.Module):
    def __init__(self,rank=48):
        super().__init__()
        self.rank=rank
        self.branch=mlp(2*NSENSORS+6,2*rank)
        self.trunk=mlp(1,2*rank)
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self,s,a,q0,p,h):
        _,integ=evaluate_and_integrate(a,s)
        flow=torch.cat((torch.zeros_like(s),p[:,1:2]/UREF*(1-q0[:,0:1]**2)),1)
        baseline=q0+h*(s*flow+magnetic_velocity(q0,integ,p))
        params=torch.cat((torch.log(p[:,0:1]/.00245),
                          (p[:,1:2]-.0175)/.0125,(p[:,2:3]-1025)/75),1)
        features=torch.cat((a.flatten(1)/AMAX,q0/q0.new_tensor([.7,2.]),params,h/HMAX),1)
        b=self.branch(features).reshape(-1,2,self.rank)
        t=self.trunk(2*s-1).reshape(-1,2,self.rank)
        correction=(b*t).sum(2)/self.rank**.5
        energy=(a*a).sum(2).mean(1,keepdim=True)/AMAX**2
        return baseline+h*h*s*s*energy*correction

def residual(model,s,a,q0,p,h,create_graph=True):
    q=model(s,a,q0,p,h)
    dq=torch.cat([torch.autograd.grad(q[:,i].sum(),s,create_graph=create_graph,
                                    retain_graph=True)[0] for i in range(2)],1)/h
    av,_=evaluate_and_integrate(a,s)
    return (dq-rhs(q,av,p))/h

def sample(n,device):
    v=torch.rand(n,7,device=device)
    p=torch.cat((.001*6**v[:,0:1],.005+.025*v[:,1:2],950+150*v[:,2:3]),1)
    q=torch.cat((-.7+1.4*v[:,3:4],-2+3.7*v[:,4:5]),1)
    h=HMIN*(HMAX/HMIN)**v[:,5:6]
    return v[:,6:7].detach().requires_grad_(True),random_samples(n,device),q,p,h
