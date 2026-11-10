"""SI parameters, nondimensional inertial Stokes model and independent oracle."""
from dataclasses import dataclass,asdict
import numpy as np
import torch
RV=75e-6; D=.04; UREF=.015; TS=D/(2*UREF); RP=500e-9; RHOP=5200.; MU0=4*np.pi*1e-7
BS=MU0*1e6*.02**2/(2*D**2)
ALPHA=3*(4*np.pi*RP**3/3)/MU0
H=.025
RANGES={'eta_Pas':[.001,.006],'mean_m_s':[.005,.03],'rho_kg_m3':[950.,1100.],
        'release_X':[-.7,.7],'release_Z':[-2.,1.7],'initial_vx_m_s':[-.001,.001],
        'initial_vz_m_s':[0.,.06],'u':[-.5,.5]}
@dataclass
class Parameters:
    mean:float=.015
    eta:float=.003276913272013514
    rho:float=1000.
    x0:float=0.
    z0:float=-.08
    vx0:float=0.
    vz0:float=.03
    @property
    def Re(self):return self.rho*self.mean*2*RV/self.eta
    @property
    def relaxation_s(self):return 2*(RHOP+.5*self.rho)*RP**2/(9*self.eta)
    @classmethod
    def from_re(cls,Re,mean=.015,rho=1000.,**kw):
        if not np.isfinite(Re) or Re<=0:raise ValueError('Re must be positive')
        return cls(mean=mean,rho=rho,eta=rho*mean*2*RV/Re,**kw).validate()
    def validate(self):
        entries={'eta_Pas':self.eta,'mean_m_s':self.mean,'rho_kg_m3':self.rho,
            'release_X':self.x0/RV,'release_Z':self.z0/D,'initial_vx_m_s':self.vx0,'initial_vz_m_s':self.vz0}
        for name,x in entries.items():
            lo,hi=RANGES[name]
            if not np.isfinite(x) or x<lo-1e-12 or x>hi+1e-12:raise ValueError(f'{name} outside trained domain {lo,hi}')
        return self
    def tensor(self,device='cpu',dtype=torch.float32):
        return torch.tensor([[self.Re,self.mean,self.rho]],device=device,dtype=dtype)
    def initial(self):return np.array([self.x0/RV,self.z0/D]),np.array([self.vx0*TS/RV,self.vz0*TS/D])

def eta_of(p):return p[:,2:3]*p[:,1:2]*2*RV/p[:,0:1]
def eps_of(p):return 2*(RHOP+.5*p[:,2:3])*RP**2/(9*eta_of(p)*TS)
def gain_of(p):return ALPHA*BS**2/(6*np.pi*eta_of(p)*RP*2*UREF*RV)
def encode_params(p):
    eta=eta_of(p)
    # Re is supplied explicitly; dependent viscosity is never sampled independently of it.
    return torch.cat((torch.log(p[:,0:1]/.75)/2,(p[:,1:2]-.0175)/.0125,(p[:,2:3]-1025)/75),1)
def force(q,u):
    side=torch.where(u<0,1.,-1.)
    y=1+side*q[:,0:1]*RV/D;z=q[:,1:2];den=(y*y+z*z)**3
    return torch.cat((-2*y*side/den,-2*z/den),1)*abs(u)
def rhs(q,u,p,flow=None):
    ff=force(q,u);k=gain_of(p)
    w=1-q[:,0:1]**2 if flow is None else flow(q[:,0:1]**2,p)
    return torch.cat((k*ff[:,0:1],p[:,1:2]/UREF*w+k*RV/D*ff[:,1:2]),1)
def oracle_rhs(q,u,par):
    x,z=q;side=1. if u<0 else -1.;y=1+side*x*RV/D
    ff=-2*np.array([side*y,z])/(y*y+z*z)**3*abs(u)
    k=ALPHA*BS**2/(6*np.pi*par.eta*RP*2*UREF*RV)
    return np.array([k*ff[0],par.mean/UREF*(1-x*x)+k*RV/D*ff[1]])
def oracle_step(q,v,u,par,h=H):
    from scipy.integrate import solve_ivp
    eps=par.relaxation_s/TS
    def fun(t,y):return np.r_[y[2:],(oracle_rhs(y[:2],u,par)-y[2:])/eps]
    sol=solve_ivp(fun,[0,h],np.r_[q,v],method='Radau',rtol=2e-8,atol=1e-10)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y[:2,-1],sol.y[2:,-1]
