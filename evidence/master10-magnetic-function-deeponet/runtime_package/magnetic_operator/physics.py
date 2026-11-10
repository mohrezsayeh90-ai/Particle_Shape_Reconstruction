"""Same source geometry/material law as Master9, now with field superposition.

q=(x/R,z/d), time tau=t/TS. Quasistatic, unsaturated induced dipole,
unbounded Stokes drag, prescribed Poiseuille flow, overdamped particle.
"""
from dataclasses import dataclass
import numpy as np
import torch

R, D, RP, UREF = 75e-6, .04, 500e-9, .015
TS = D/(2*UREF)
MU0 = 4*np.pi*1e-7
BS = MU0*1e6*.02**2/(2*D**2)
ALPHA = 3*(4*np.pi*RP**3/3)/MU0
AMAX = np.sqrt(.5)
HMAX, HMIN, NSENSORS = .1, .00625, 17

@dataclass(frozen=True)
class Parameters:
    eta: float = .003276913272013514
    mean: float = .015
    rho: float = 1000.

    def validate(self):
        for v, lo, hi in ((self.eta,.001,.006),(self.mean,.005,.03),(self.rho,950,1100)):
            if not np.isfinite(v) or not lo <= v <= hi:
                raise ValueError('Physical parameters outside the declared domain')
        return self

    def tensor(self, device='cpu', dtype=torch.float32):
        return torch.tensor([[self.eta,self.mean,self.rho]],device=device,dtype=dtype)

    @property
    def relaxation_s(self):
        return 2*(5200+.5*self.rho)*RP**2/(9*self.eta)

def source(q, side):
    """B/Bs and its Jacobian in (x/d,z/d), differentiable closed form."""
    y = 1+side*q[:,0]*R/D
    z = q[:,1]
    r2 = y*y+z*z
    b = torch.stack((side*(y*y-z*z)/r2**2, 2*y*z/r2**2),1)
    c = -2*y*(y*y-3*z*z)/r2**3
    e = -2*side*z*(3*y*y-z*z)/r2**3
    j = torch.stack((c,e,e,-c),1).reshape(-1,2,2)
    return b,j

def force_basis(q):
    bl,jl = source(q,1)
    br,jr = source(q,-1)
    contract = lambda j,b: torch.einsum('nki,nk->ni',j,b)
    return torch.stack((contract(jl,bl),contract(jr,br),
                        contract(jl,br)+contract(jr,bl)),1)

def gain(p):
    return ALPHA*BS**2/(6*np.pi*p[:,0:1]*RP*2*UREF*R)

def magnetic_velocity(q, products, p):
    f = (force_basis(q)*products[:,:,None]).sum(1)
    return gain(p)*f*q.new_tensor([1.,R/D])

def rhs(q,a,p):
    products=torch.stack((a[:,0]**2,a[:,1]**2,a[:,0]*a[:,1]),1)
    v=magnetic_velocity(q,products,p)
    return v+torch.cat((torch.zeros_like(q[:,0:1]),p[:,1:2]/UREF*(1-q[:,0:1]**2)),1)

def numpy_field_force(q,a):
    """Independent complex-analytic potential implementation; no torch formula reuse.

    b_x-i*b_z = a_left/(1+w)^2 - a_right/(1-w)^2,
    w=x/d+i*z/d. grad(|B|^2)/2 from the complex derivative.
    """
    w=q[0]*R/D+1j*q[1]
    b=a[0]/(1+w)**2-a[1]/(1-w)**2
    db=-2*a[0]/(1+w)**3-2*a[1]/(1-w)**3
    f=np.array([(np.conj(b)*db).real,(np.conj(b)*1j*db).real])
    return np.array([b.real,-b.imag]),f

def numpy_rhs(q,a,par):
    _,f=numpy_field_force(q,a)
    k=ALPHA*BS**2/(6*np.pi*par.eta*RP*2*UREF*R)
    return k*f*np.array([1.,R/D])+[0.,par.mean/UREF*(1-q[0]**2)]

def observables(field,times_s,path_m):
    """Field and magnetic force along a predicted/measured path, in tesla/newtons.

    Uses the original callable rather than the sensor-interpolated field.
    Columns: time_s, Bx_T, Bz_T, Fmx_N, Fmz_N.
    """
    times=np.asarray(times_s,float);path=np.asarray(path_m,float)
    if times.ndim!=1 or path.shape!=(len(times),2) or not np.isfinite(path).all() or not np.isfinite(times).all():
        raise ValueError('Invalid path/time arrays')
    rows=[]
    for t,pos in zip(times,path):
        a=np.asarray(field(float(t)),float)
        if a.shape!=(2,) or not np.isfinite(a).all() or np.abs(a).sum()>AMAX+1e-10:
            raise ValueError('Invalid source amplitudes')
        b,f=numpy_field_force(pos/[R,D],a)
        rows.append(np.r_[t,BS*b,ALPHA*BS**2/D*f])
    return np.asarray(rows)
