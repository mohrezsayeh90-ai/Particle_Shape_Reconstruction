"""Master3 physics, extracted from c17460d with no change to launch bounds.

Planning uses a checked NumPy/Numba evaluator of the same frozen MLP. Acceptance
uses the original float32 PyTorch forward and a separately written analytical
carrier/DOP853 replay. Clipping is an RHS extension for event bracketing only.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from numba import njit
from scipy.integrate import solve_ivp

L,H,RE,DIAMETER,RHO,GAP=2.2,.30,10.,.025,1000.,1e-5
ACCEPT,SEARCH=.026,.0263
WEIGHT_SHA='af2f21b195e2d25b877482cd527110054f42bf92200f64e39572375219e5e581'
BOUNDS=np.array([[0,L],[-.4,1.5],[.15,1.2],[0,L],[-.4,1.5],[-1.2,-.15]])
MID=BOUNDS.mean(1)
torch.set_num_threads(1)


class FrozenFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('lower',torch.tensor([0,0,*BOUNDS[:,0]],dtype=torch.float32))
        self.register_buffer('span',torch.tensor([L,H,*(BOUNDS[:,1]-BOUNDS[:,0])],dtype=torch.float32))
        layers=[nn.Linear(8,64),nn.Tanh()]
        for _ in range(3):layers += [nn.Linear(64,64),nn.Tanh()]
        layers += [nn.Linear(64,3)];self.net=nn.Sequential(*layers)

    def forward(self,z):
        a=self.net(2*(z-self.lower)/self.span-1)
        x,y=z[:,0]/L,z[:,1]/H;p=4*y*(1-y);cap=x*(1-x)
        return torch.stack((p*(1+cap*a[:,0]),p*cap*a[:,1],8*L/(RE*H**2)*(1-x)*a[:,2]),dim=1)


@njit(cache=True)
def compiled_field(x,y,c,weights,biases):
    z=np.empty(8,dtype=np.float32)
    z[0]=2*x/L-1;z[1]=2*y/H-1
    for j in range(6):z[j+2]=2*(c[j]-BOUNDS[j,0])/(BOUNDS[j,1]-BOUNDS[j,0])-1
    for k in range(5):
        z=weights[k]@z+biases[k]
        if k<4:z=np.tanh(z)
    p=np.float32(4*y/H*(1-y/H));cap=np.float32(x/L*(1-x/L))
    return np.array([p*(1+cap*z[0]),p*cap*z[1]])


@njit(cache=True)
def fast_rhs(state,c,weights,biases,neural):
    x=min(L,max(0.,state[0]));y=min(H,max(0.,state[1]))
    uv=compiled_field(x,y,c,weights,biases) if neural else np.array([4*y/H*(1-y/H),0.])
    du=uv[0]-state[2];dv=uv[1]-state[3];rep=RE*DIAMETER*np.sqrt(du*du+dv*dv)
    factor=1+.15*rep**.687 if rep<1000 else .44*rep/24
    k=18*factor/(RHO*RE*DIAMETER**2)
    return np.array([state[2],state[3],k*du,k*dv])


class Physics:
    def __init__(self,checkpoint):
        checkpoint=Path(checkpoint)
        self.sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if self.sha!=WEIGHT_SHA:raise ValueError('Frozen checkpoint checksum mismatch')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cfg=saved['config']
        assert (cfg['L'],cfg['H'],cfg['diameter'],cfg['density_ratio'])==(L,H,DIAMETER,RHO)
        assert cfg['drag']=='schiller-naumann'
        self.model=FrozenFlow();self.model.load_state_dict(saved['model'],strict=True)
        self.model.eval().requires_grad_(False)
        self.weights=tuple(saved['model'][f'net.{j}.weight'].numpy().copy() for j in range(0,9,2))
        self.biases=tuple(saved['model'][f'net.{j}.bias'].numpy().copy() for j in range(0,9,2))

    def field(self,xy,c):
        xy=np.atleast_2d(xy)
        packed=np.column_stack((xy,np.broadcast_to(c,(len(xy),6))))
        with torch.inference_mode():return self.model(torch.tensor(packed,dtype=torch.float32)).numpy()[:,:2]

    def trace(self,c,side,age,mode='coarse',max_step=None):
        c=np.asarray(c,float)
        if side not in (0,1) or np.any(c<BOUNDS[:,0]-1e-12) or np.any(c>BOUNDS[:,1]+1e-12):
            raise ValueError('Invalid side or launch bounds')
        j=3*side;y=DIAMETER/2+GAP if side==0 else H-DIAMETER/2-GAP
        initial=[c[j],y,c[j+1],c[j+2]]
        def reference_rhs(t,z):
            # Independent planar Poiseuille/Schiller--Naumann implementation.
            eta=np.clip(z[1],0,H)/H
            slip=np.array([4*eta*(1-eta),0.])-z[2:]
            rp=RE*DIAMETER*np.linalg.norm(slip)
            f=1+.15*rp**.687 if rp<1000 else .44*rp/24
            return np.r_[z[2:],18*f*slip/(RHO*RE*DIAMETER**2)]
        def torch_rhs(t,z):
            uv=self.field(np.clip(z[:2],[0,0],[L,H])[None],c)[0]
            slip=uv-z[2:];rp=RE*DIAMETER*np.linalg.norm(slip)
            f=1+.15*rp**.687 if rp<1000 else .44*rp/24
            return np.r_[z[2:],18*f*slip/(RHO*RE*DIAMETER**2)]
        if mode=='reference':rhs=reference_rhs;method='DOP853';rtol,atol=1e-10,1e-12;step=.0125
        elif mode=='pinn':rhs=torch_rhs;method='RK45';rtol,atol=2e-8,2e-10;step=.0175
        else:
            rhs=lambda t,z:fast_rhs(z,c,self.weights,self.biases,mode=='direct')
            method='RK45';rtol,atol=(2e-7,2e-9) if mode=='direct' else (2e-6,2e-8);step=.035 if mode=='direct' else .06
        events=[lambda t,z:z[1]-DIAMETER/2,lambda t,z:H-DIAMETER/2-z[1],lambda t,z:z[0],lambda t,z:L-z[0]]
        for e in events:e.terminal=True;e.direction=-1
        sol=solve_ivp(rhs,[0,max(float(age),1e-8)],initial,method=method,rtol=rtol,atol=atol,
                      max_step=max_step or step,events=events,dense_output=True)
        sol.end=float(sol.t[-1]);sol.valid=bool(sol.success and sol.end>=age-1e-8)
        sol.event=next((name for name,ts in zip(['lower_wall','upper_wall','inlet_exit','outlet_exit'],sol.t_events) if len(ts)),None)
        return sol

    def evaluator_check(self):
        rng=np.random.default_rng(20261112);err=0.
        for _ in range(128):
            c=rng.uniform(BOUNDS[:,0],BOUNDS[:,1]);xy=rng.uniform([0,0],[L,H])
            a=compiled_field(*xy,c,self.weights,self.biases);b=self.field(xy,c)[0]
            err=max(err,float(np.max(abs(a-b))))
        if err>3e-6:raise AssertionError(f'Compiled/PyTorch evaluator disagreement: {err}')
        return {'points':128,'max_velocity_abs_difference':err,'threshold':3e-6,'passed':True}
