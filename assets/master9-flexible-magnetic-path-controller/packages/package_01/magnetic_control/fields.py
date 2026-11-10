"""Parameter-conditioned, residual-only PINN and bounded magnetic trajectory control.

The validated source notebook supplies the reduced one-way particle equations.
This file adds a bounded *effective signed force* input; it is not Maxwell/coil CFD.
No ODE trajectory or target path is used to train the forward PINN.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import time
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    # Fixed physical parameters: only the magnetic waveform varies in this stage.
    mu0: float = 4 * math.pi * 1e-7
    Ms: float = 1e6
    Msp: float = 4.78e5
    Rmag: float = 0.02
    Rv: float = 75e-6
    rp: float = 500e-9
    mean: float = 0.015
    eta: float = 0.003276913272013514
    beta: float = 1.0
    X0: float = 0.0
    Z0: float = -2.0
    T: float = 4.0  # dimensionless horizon tau; physical time is tau*d/(2*mean)
    umax: float = 0.5
    K: int = 5      # degree-four Bernstein waveform, five bounded coefficients
    width: int = 64
    depth: int = 3
    seed: int = 42

    @property
    def d(self):
        return (1 + self.beta) * self.Rmag

    @property
    def time_scale(self):
        return self.d / (2 * self.mean)

    @property
    def C(self):
        return self.mu0 * self.rp**2 * 3 * self.Ms**2 / (
            18 * self.eta * self.Rv * self.mean * (1 + self.beta)**4)

    @property
    def drag(self):
        return 6 * math.pi * self.eta * self.rp


def bernstein(s, K):
    """s: [N,1], result [N,K], partition of unity on [0,1]."""
    return torch.cat([math.comb(K-1,k) * s**k * (1-s)**(K-1-k)
                      for k in range(K)], dim=1)


def waveform(s, a):
    """a are signed force coefficients, not signed magnetic field strengths."""
    return (bernstein(s, a.shape[1]) * a).sum(dim=1, keepdim=True)


def waveform_numpy(s, a):
    a = np.asarray(a, dtype=float)
    s = np.atleast_1d(s).astype(float)
    b = np.column_stack([math.comb(len(a)-1,k) * s**k * (1-s)**(len(a)-1-k)
                         for k in range(len(a))])
    return b @ a


class ParametricPINN(nn.Module):
    """A single shared network: (s, a0,...,a4) -> (X,Z)."""
    def __init__(self, cfg=Config()):
        super().__init__()
        self.cfg = cfg
        layers = []
        n = cfg.K + 1
        for _ in range(cfg.depth):
            layers += [nn.Linear(n, cfg.width), nn.Tanh()]
            n = cfg.width
        layers += [nn.Linear(n, 2)]
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, s, a):
        c = self.cfg
        h = self.net(torch.cat((2*s-1, a/c.umax), dim=1))
        # Exact fixed release; the baseline is unforced prescribed Poiseuille flow.
        return torch.cat((c.X0+s*h[:,0:1], c.Z0+c.T*s+s*h[:,1:2]), dim=1)


def residual(model, s, a, create_graph=True):
    c = model.cfg
    q = model(s,a)
    dx = torch.autograd.grad(q[:,0].sum(),s,create_graph=create_graph,retain_graph=True)[0]/c.T
    dz = torch.autograd.grad(q[:,1].sum(),s,create_graph=create_graph,retain_graph=True)[0]/c.T
    rx = dx - c.C*waveform(s,a)/(1+q[:,1:2]**2)**3
    rz = dz - (1-q[:,0:1]**2)
    return torch.cat((rx,rz), dim=1)


def train(cfg=Config(), adam_steps=6000, lbfgs_steps=600, batch=768,
          device='cpu', verbose=True, boundary_steps=500):
    """Pure physics training. Independent references are in reference_solve only."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    model = ParametricPINN(cfg).to(device=device,dtype=torch.float64)
    sampler = torch.quasirandom.SobolEngine(cfg.K+1,scramble=True,seed=cfg.seed)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,adam_steps,eta_min=0.0001)
    history=[]
    tic=time.perf_counter()
    for step in range(adam_steps):
        points=sampler.draw(batch).to(device=device,dtype=torch.float64)
        s=points[:,0:1].detach().requires_grad_(True)
        a=cfg.umax*(2*points[:,1:]-1)
        # Include coefficient-domain faces, constant controls and the no-field case.
        a[:batch//8]=torch.sign(a[:batch//8])*cfg.umax
        a[batch//8:batch//4]=a[batch//8:batch//4,0:1]
        a[batch//4:batch//4+8]=0
        optimizer.zero_grad(set_to_none=True)
        r=residual(model,s,a)
        loss=(r*r).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step%1000==0 or step==adam_steps-1:
            value=float(loss.detach())
            history.append(dict(stage='adam',step=step+1,loss=value))
            if verbose: print(f'Adam {step+1}/{adam_steps}: residual MSE={value:.3e}',flush=True)
    # Fixed collocation set for deterministic L-BFGS. No solution labels.
    points=sampler.draw(6144).to(device=device,dtype=torch.float64)
    s=points[:,0:1].detach().requires_grad_(True)
    a=cfg.umax*(2*points[:,1:]-1)
    a[:768]=torch.sign(a[:768])*cfg.umax
    a[768:1280]=a[768:1280,0:1]
    a[1280:1312]=0
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1.0,max_iter=lbfgs_steps,
        tolerance_grad=1e-10,tolerance_change=1e-13,history_size=40,line_search_fn='strong_wolfe')
    calls=[0]
    def closure():
        optimizer.zero_grad(set_to_none=True)
        if s.grad is not None: s.grad=None
        r=residual(model,s,a)
        loss=(r*r).mean()
        loss.backward()
        calls[0]+=1
        if verbose and calls[0]%200==0:
            print(f'L-BFGS closure {calls[0]}: residual MSE={float(loss.detach()):.3e}',flush=True)
        return loss
    optimizer.step(closure)
    final=float((residual(model,s,a)**2).mean().detach())
    history.append(dict(stage='lbfgs',step=calls[0],loss=final))
    metadata=dict(config=asdict(cfg),seconds=time.perf_counter()-tic,history=history,
        training='ODE residuals + hard fixed initial conditions only; no solve_ivp or target trajectories',
        settings=dict(adam_steps=adam_steps,lbfgs_steps=lbfgs_steps,batch=batch,lbfgs_points=6144),
        device=device,dtype='float64')
    if boundary_steps:
        refinement=boundary_refine(model,steps=boundary_steps,verbose=verbose)
        metadata['history'].append(refinement)
        metadata['settings']['boundary_steps']=boundary_steps
        metadata['seconds']=time.perf_counter()-tic
    model.eval()
    return model,metadata


def boundary_refine(model,steps=500,verbose=True):
    """Residual-only refinement on systematic control-domain boundaries.

    These are collocation constraints, never reference-solution observations.
    """
    from itertools import product
    c=model.cfg
    device=next(model.parameters()).device
    dtype=next(model.parameters()).dtype
    base=torch.quasirandom.SobolEngine(c.K+1,scramble=True,seed=c.seed+190).draw(3072).to(device=device,dtype=dtype)
    controls=np.array(list(product([-c.umax,c.umax],repeat=c.K)))
    controls=np.vstack((controls,np.repeat(np.linspace(-c.umax,c.umax,9)[:,None],c.K,axis=1)))
    grid=(np.arange(97)+.319)/97
    ss=torch.tensor(np.tile(grid,len(controls))[:,None],dtype=dtype,device=device)
    aa=torch.tensor(np.repeat(controls,len(grid),axis=0),dtype=dtype,device=device)
    s=torch.cat((base[:,0:1],ss)).detach().requires_grad_(True)
    a=torch.cat((c.umax*(2*base[:,1:]-1),aa))
    opt=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=steps,line_search_fn='strong_wolfe',
        tolerance_grad=2e-11,tolerance_change=1e-14,history_size=50)
    calls=[0]
    def closure():
        opt.zero_grad(set_to_none=True)
        if s.grad is not None: s.grad=None
        r=residual(model,s,a)
        loss=(r*r).mean()
        loss.backward()
        calls[0]+=1
        if verbose and calls[0]%200==0:
            print(f'Boundary refinement {calls[0]}: residual MSE={float(loss.detach()):.3e}',flush=True)
        return loss
    opt.step(closure)
    return dict(stage='boundary_lbfgs',calls=calls[0],loss=float((residual(model,s,a)**2).mean().detach()),
                points=len(s),control_data='PDE/ODE collocation only')


def check_coefficients(a,cfg):
    a=np.asarray(a,dtype=float)
    if a.shape != (cfg.K,) or not np.all(np.isfinite(a)):
        raise ValueError(f'Expected {cfg.K} finite coefficients.')
    if np.max(np.abs(a))>cfg.umax+1e-12:
        raise ValueError('Magnetic coefficients outside the trained domain; retraining is required.')
    return a


def predict(model,a,t_seconds,physical=True):
    c=model.cfg
    a=check_coefficients(a,c)
    tt=np.atleast_1d(t_seconds).astype(float)
    if not np.all(np.isfinite(tt)) or np.any(tt<0) or np.any(tt>c.T*c.time_scale+1e-12):
        raise ValueError('Time outside the trained pre-event horizon.')
    device=next(model.parameters()).device
    s=torch.tensor(tt[:,None]/(c.T*c.time_scale),dtype=torch.float64,device=device)
    aa=torch.tensor(a,dtype=torch.float64,device=device)[None].expand(len(tt),-1)
    with torch.no_grad(): q=model(s,aa).cpu().numpy()
    # The exact axial solution obeys Z <= Z0+tau <= 2 on this fixed horizon.
    # A small surrogate error near the endpoint is assessed by validation, not
    # mistaken for an actual outlet event in the exactly unforced case.
    if np.any(np.abs(q[:,0])>=1):
        raise RuntimeError('Predicted wall event; post-event motion is outside this model.')
    return q*np.array([c.Rv,c.d]) if physical else q




"""Solved fully-developed cylindrical flow, solved exterior magnetic potential,
and a parameter-conditioned particle PINN. One-way field-to-particle coupling.
Requires the definitions in parametric_magnetic_pinn.py in the same namespace.
"""
from numpy.polynomial import chebyshev as cheb
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, lsq_linear

class SmallNet(nn.Module):
    def __init__(self,inputs,width=40,depth=3):
        super().__init__()
        layers=[]
        for _ in range(depth):
            layers += [nn.Linear(inputs,width),nn.Tanh()];inputs=width
        layers += [nn.Linear(width,1)]
        self.net=nn.Sequential(*layers)
        for l in self.net:
            if isinstance(l,nn.Linear):nn.init.xavier_normal_(l.weight);nn.init.zeros_(l.bias)
    def forward(self,x):return self.net(x)

def grad(y,x):
    return torch.autograd.grad(y.sum(),x,create_graph=True,retain_graph=True)[0]

class FluidPINN(nn.Module):
    def __init__(self):
        super().__init__();self.velocity=SmallNet(1,16,2);self.pressure=SmallNet(1,16,2)
    def w(self,q):return (1-q)*self.velocity(2*q-1)
    def p(self,xi):return 1-xi+xi*(1-xi)*self.pressure(2*xi-1)
    def equation(self,points):
        q=points[:,0:1].detach().requires_grad_(True)
        xi=points[:,1:2].detach().requires_grad_(True)
        w=self.w(q);wq=grad(w,q)
        return q*grad(wq,q)+wq-grad(self.p(xi),xi)

def boundary_phi(x,z):
    # Exterior cylindrical-source potential, used on the computational boundary ONLY.
    return (x+1)/((x+1)**2+z*z)

class MagneticPINN(nn.Module):
    ax=.4;az=2.4
    def __init__(self):super().__init__();self.correction=SmallNet(2,40,3)
    def forward(self,points):
        x=points[:,0:1];z=points[:,1:2];a=x/self.ax;b=z/self.az
        l=(1-a)/2;r=(1+a)/2;bot=(1-b)/2;top=(1+b)/2
        # Coons lifting: boundary data only; this is not the interior analytic solution.
        lift=l*boundary_phi(-self.ax,z)+r*boundary_phi(self.ax,z)
        lift+=bot*boundary_phi(x,-self.az)+top*boundary_phi(x,self.az)
        corners=l*(bot*boundary_phi(-self.ax,-self.az)+top*boundary_phi(-self.ax,self.az))
        corners+=r*(bot*boundary_phi(self.ax,-self.az)+top*boundary_phi(self.ax,self.az))
        return lift-corners+(1-a*a)*(1-b*b)*self.correction(torch.cat((a,b),1))
    def equation(self,points):
        p=points.detach().requires_grad_(True)
        g=grad(self(p),p)
        return grad(g[:,0:1],p)[:,0:1]+grad(g[:,1:2],p)[:,1:2]
    def field_force(self,points):
        # B/Bscale=-grad(phi); F/(alpha*Bscale^2/d)=grad(|b|^2)/2.
        p=points.detach().requires_grad_(True)
        b=-grad(self(p),p)
        f=.5*grad((b*b).sum(1,keepdim=True),p)
        return b,f

def train_field(model,points,adam=1500,lbfgs=500,label='field'):
    model=model.double();tic=time.perf_counter()
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    for j in range(adam):
        opt.zero_grad();loss=(model.equation(points)**2).mean();loss.backward();opt.step()
        if j%500==0:print(label,'Adam',j,float(loss.detach()),flush=True)
    opt=torch.optim.LBFGS(model.parameters(),max_iter=lbfgs,history_size=40,
        tolerance_grad=1e-11,tolerance_change=1e-14,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=(model.equation(points)**2).mean();loss.backward();return loss
    opt.step(closure)
    loss=float((model.equation(points)**2).mean().detach())
    for p in model.parameters():p.requires_grad_(False)
    print(label,'finished',loss,'seconds',time.perf_counter()-tic,flush=True)
    return model,dict(loss=loss,seconds=time.perf_counter()-tic)

def analytic_magnetic(points):
    # Independent post-training reference for the prescribed exterior-source BVP.
    x,z=np.asarray(points).T;y=x+1;r2=y*y+z*z
    b=np.column_stack(((y*y-z*z)/r2**2,2*y*z/r2**2))
    f=-2*np.column_stack((y,z))/r2[:,None]**3
    return b,f

def check_fields(fluid,mag,cfg):
    rng=np.random.default_rng(72341)
    p=np.column_stack((rng.uniform(-.39,.39,2000),rng.uniform(-2.39,2.39,2000)))
    b,f=mag.field_force(torch.tensor(p,dtype=torch.float64))
    br,fr=analytic_magnetic(p)
    channel=np.column_stack((rng.uniform(-.8,.8,1200)*cfg.Rv/cfg.d,rng.uniform(-2,2,1200)))
    bc,fc=mag.field_force(torch.tensor(channel,dtype=torch.float64));bcr,fcr=analytic_magnetic(channel)
    rel=lambda x,y:float(np.linalg.norm(x-y)/np.linalg.norm(y))
    q=torch.linspace(0,1,401,dtype=torch.float64)[:,None]
    with torch.no_grad():w=fluid.w(q).numpy();pr=fluid.p(q).numpy()
    report=dict(fluid_velocity_max_error=float(abs(w-(1-q.numpy())).max()),
        fluid_pressure_max_error=float(abs(pr-(1-q.numpy())).max()),
        magnetic_B_relative_L2=rel(b.detach().numpy(),br),
        magnetic_force_relative_L2=rel(f.detach().numpy(),fr),
        channel_B_relative_L2=rel(bc.detach().numpy(),bcr),
        channel_force_relative_L2=rel(fc.detach().numpy(),fcr),
        channel_force_max_absolute=float(abs(fc.detach().numpy()-fcr).max()),
        heldout_laplace_rms=float(torch.sqrt((mag.equation(torch.tensor(p,dtype=torch.float64))**2).mean()).detach()))
    report['checks']=dict(flow_velocity=report['fluid_velocity_max_error']<.002,
        flow_pressure=report['fluid_pressure_max_error']<.002,
        B_field=report['channel_B_relative_L2']<.01,
        force_gradient=report['channel_force_relative_L2']<.02,
        laplace_residual=report['heldout_laplace_rms']<.03)
    report['all_passed']=all(report['checks'].values())
    return report

class LearnedForces:
    """Differentiable Chebyshev compression of forces from the TRAINED magnetic PINN.
    Interior analytical fields are never used to populate this approximation.
    """
    def __init__(self,mag,cfg,coeff=None):
        self.cfg=cfg
        if coeff is None:
            xx=np.cos(np.pi*(np.arange(9)+.5)/9)
            zz=np.cos(np.pi*(np.arange(129)+.5)/129)
            a,b=np.meshgrid(xx,zz,indexing='ij')
            points=np.column_stack((a.ravel()*.8*cfg.Rv/cfg.d,b.ravel()*2))
            _,f=mag.field_force(torch.tensor(points,dtype=torch.float64))
            V=cheb.chebvander2d(a.ravel(),b.ravel(),[4,48]).reshape(len(points),-1)
            self.coeff=np.linalg.lstsq(V,f.detach().numpy(),rcond=None)[0].reshape(5,49,2)
        else:self.coeff=np.asarray(coeff)
    @staticmethod
    def basis(v,n):
        terms=[torch.ones_like(v),v]
        for _ in range(2,n):terms.append(2*v*terms[-1]-terms[-2])
        return torch.cat(terms[:n],dim=1)
    def torch(self,X,Z):
        cx=self.basis(X/.8,5);cz=self.basis(Z/2,49)
        coef=torch.tensor(self.coeff,dtype=X.dtype,device=X.device)
        return torch.einsum('ni,nj,ijk->nk',cx,cz,coef)
    def numpy(self,X,Z):
        return np.array([cheb.chebval2d(X/.8,Z/2,self.coeff[:,:,i]) for i in range(2)])
    def audit(self,mag):
        rng=np.random.default_rng(93220);X=rng.uniform(-.8,.8,1000);Z=rng.uniform(-2,2,1000)
        points=np.column_stack((X*self.cfg.Rv/self.cfg.d,Z))
        _,f=mag.field_force(torch.tensor(points,dtype=torch.float64))
        fitted=self.numpy(X,Z).T;err=float(abs(fitted-f.detach().numpy()).max())
        return dict(max_force_interpolation_error=err,all_passed=err<.002)

FLUID=None;FORCES=None

def full_rhs_torch(X,Z,u,c):
    # Both magnetic-force components are retained, including axial magnetic drift.
    fl=FORCES.torch(X,Z);fr=FORCES.torch(-X,Z)*torch.tensor([-1.,1.],dtype=X.dtype,device=X.device)
    f=torch.where((u>=0).expand(-1,2),fr,fl)*abs(u)
    Bscale=c.mu0*c.Ms*c.Rmag**2/(2*c.d**2)
    alpha=3*(4*math.pi*c.rp**3/3)/c.mu0
    k=alpha*Bscale**2/(c.drag*2*c.mean*c.Rv)
    return torch.cat((k*f[:,0:1],FLUID.w(X*X)+k*c.Rv/c.d*f[:,1:2]),1)

def residual(model,s,a,create_graph=True):
    q=model(s,a)
    dx=torch.autograd.grad(q[:,0].sum(),s,create_graph=create_graph,retain_graph=True)[0]/model.cfg.T
    dz=torch.autograd.grad(q[:,1].sum(),s,create_graph=create_graph,retain_graph=True)[0]/model.cfg.T
    return torch.cat((dx,dz),1)-full_rhs_torch(q[:,0:1],q[:,1:2],waveform(s,a),model.cfg)

def reference_solve(a,s,cfg=Config(),learned=False):
    """learned=True isolates particle-PINN error; False audits the entire field chain."""
    a=check_coefficients(a,cfg)
    Bscale=cfg.mu0*cfg.Ms*cfg.Rmag**2/(2*cfg.d**2)
    alpha=3*(4*math.pi*cfg.rp**3/3)/cfg.mu0
    k=alpha*Bscale**2/(cfg.drag*2*cfg.mean*cfg.Rv)
    def rhs(tau,q):
        X,Z=q;u=float(waveform_numpy(tau/cfg.T,a)[0]);side=1 if u<0 else -1
        if learned:
            f=FORCES.numpy(side*X,Z)*np.array([side,1])
            with torch.no_grad():w=float(FLUID.w(torch.tensor([[X*X]],dtype=torch.float64))[0,0])
        else:
            _,fa=analytic_magnetic([[side*X*cfg.Rv/cfg.d,Z]])
            f=fa[0]*[side,1];w=1-X*X
        return [k*abs(u)*f[0],w+k*cfg.Rv/cfg.d*abs(u)*f[1]]
    def wall(t,q):return .8-abs(q[0])
    wall.terminal=True;wall.direction=-1
    sol=solve_ivp(rhs,[0,cfg.T],[cfg.X0,cfg.Z0],rtol=1e-10,atol=1e-12,
        method='DOP853',dense_output=True,max_step=.04,events=wall)
    if not sol.success or sol.t[-1]<cfg.T-1e-8:raise RuntimeError('Trajectory outside the declared field/particle domain.')
    return sol.sol(np.asarray(s)*cfg.T).T

def validate_full(model,n=32,seed=92731):
    c=model.cfg;rng=np.random.default_rng(seed);s=np.linspace(0,1,201)
    controls=np.vstack((rng.uniform(-c.umax,c.umax,(n,c.K)),
        np.full((1,c.K),-.5),np.full((1,c.K),.5),np.zeros((1,c.K))))
    rows=[]
    for a in controls:
        pred=predict(model,a,s*c.T*c.time_scale,False)
        oracle=reference_solve(a,s,c)
        learned=reference_solve(a,s,c,learned=True)
        rows.append(dict(coefficients=a.tolist(),
            end_to_end_max_X_Rv=float(abs(pred[:,0]-oracle[:,0]).max()),
            end_to_end_max_Z_d=float(abs(pred[:,1]-oracle[:,1]).max()),
            particle_only_max_X_Rv=float(abs(pred[:,0]-learned[:,0]).max()),
            particle_only_max_Z_d=float(abs(pred[:,1]-learned[:,1]).max())))
    p=torch.quasirandom.SobolEngine(c.K+1,scramble=True,seed=seed+1).draw(4096).double()
    st=p[:,0:1].detach().requires_grad_(True);a=(2*p[:,1:]-1)*c.umax
    rr=residual(model,st,a,False).detach().numpy()
    r=dict(n_cases=len(rows),cases=rows,residual_rms=float(np.sqrt(np.mean(rr**2))))
    r['max_X_Rv']=max(v['end_to_end_max_X_Rv'] for v in rows)
    r['max_Z_d']=max(v['end_to_end_max_Z_d'] for v in rows)
    r['checks']=dict(end_to_end_X=r['max_X_Rv']<.01,end_to_end_Z=r['max_Z_d']<.01,
        residual=r['residual_rms']<.003)
    r['all_passed']=all(r['checks'].values())
    return r

def track_x(model,s,desired_X):
    """Optimize transverse target; axial timing is an output of solved dynamics."""
    c=model.cfg;s=np.asarray(s);target=np.asarray(desired_X)
    if target.shape!=s.shape or abs(target[0]-c.X0)>1e-9 or max(abs(target))>=.7:
        raise ValueError('Bad target shape, release point or wall margin.')
    st=torch.tensor(s[:,None],dtype=torch.float64);tar=torch.tensor(target[:,None],dtype=torch.float64)
    nominal=c.Z0+c.T*s
    req=np.gradient(target,s)/c.T*(1+nominal**2)**3/c.C
    B=bernstein(st,c.K).numpy()
    initial=lsq_linear(B,req,bounds=(-c.umax,c.umax)).x
    states=[p.requires_grad for p in model.parameters()]
    for p in model.parameters():p.requires_grad_(False)
    def obj(a):
        at=torch.tensor(a,dtype=torch.float64,requires_grad=True)
        q=model(st,at[None].expand(len(s),-1))
        loss=((q[:,0:1]-tar)**2).mean()+1e-8*((at[1:]-at[:-1])**2).mean()
        g=torch.autograd.grad(loss,at)[0]
        return float(loss.detach()),g.detach().numpy()
    try:opt=minimize(obj,initial,jac=True,bounds=[(-c.umax,c.umax)]*c.K,
        method='L-BFGS-B',options=dict(maxiter=400,ftol=1e-15,gtol=1e-10))
    finally:
        for p,state in zip(model.parameters(),states):p.requires_grad_(state)
    pred=predict(model,opt.x,s*c.T*c.time_scale,False);replay=reference_solve(opt.x,s,c)
    error=float(abs(replay[:,0]-target).max())
    return dict(coefficients=opt.x.tolist(),target_X=target,prediction=pred,replay=replay,
        u=waveform_numpy(s,opt.x),max_tracking_error_Rv=error,
        accepted=bool(opt.success and error<.01),message=str(opt.message))

def force_table(cfg,a,s,q):
    u=waveform_numpy(s,a);side=np.where(u<0,1.,-1.)
    f=FORCES.numpy(side*q[:,0],q[:,1]).T*np.column_stack((side,np.ones(len(s))))*abs(u[:,None])
    bs=cfg.mu0*cfg.Ms*cfg.Rmag**2/(2*cfg.d**2)
    alpha=3*(4*math.pi*cfg.rp**3/3)/cfg.mu0
    F=alpha*bs**2/cfg.d*f
    # Overdamped model: drag balances magnetic force in both components.
    return np.column_stack((s*cfg.T*cfg.time_scale,q[:,0]*cfg.Rv,q[:,1]*cfg.d,u,F,-F))
