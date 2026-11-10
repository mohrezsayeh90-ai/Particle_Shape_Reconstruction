"""Held-out physical tests; independent Radau trajectories never enter training."""
from pathlib import Path
from dataclasses import asdict,replace
import json,time
import numpy as np,torch
from scipy.integrate import trapezoid
from .models import *
from .train import load

def transition(model,q,v,u,par,h=H,velocity=False):
    if not np.isfinite(u) or abs(u)>.5+1e-12:raise ValueError('u outside trained domain')
    if h<0 or h>H+1e-12:raise ValueError('local horizon outside trained domain')
    dtype=next(model.parameters()).dtype
    s=torch.tensor([[h/H]],dtype=dtype,requires_grad=velocity)
    qt=torch.tensor(np.asarray(q)[None],dtype=dtype);vt=torch.tensor(np.asarray(v)[None],dtype=dtype)
    ut=torch.tensor([[u]],dtype=dtype);p=par.tensor(dtype=dtype)
    if velocity:
        with torch.enable_grad():qn,vn,_=model.terms(s,qt,vt,ut,p)
        return qn.detach().numpy()[0],vn.detach().numpy()[0]
    with torch.no_grad():return model(s,qt,vt,ut,p).numpy()[0]

def run(output,require_gpu=True):
    output=Path(output);model,runtime=load(output);model=model.double();rng=np.random.default_rng(919)
    rows={}
    def check(name,passed,**metrics):
        rows[name]={'passed':bool(passed),**metrics};print(name,rows[name],flush=True)
    check('gpu_used_for_training',runtime['device']=='cuda' if require_gpu else True,device=runtime['device'],gpu=runtime['gpu'])
    # Parametric fluid profile and physical boundary/flux constraints, independent oracle 1-q.
    a=np.linspace(0,1,161);profile_errors=[];flux=[];wall=[]
    corner_params=[Parameters(mean=u,eta=e,rho=r) for u in (.005,.03) for e in (.001,.006) for r in (950.,1100.)]
    with torch.no_grad():
        for par in corner_params:
            p=par.tensor(dtype=torch.float64).repeat(len(a),1)
            w=model.flow(torch.tensor(a[:,None],dtype=torch.float64),p).numpy()[:,0]
            profile_errors.append(float(abs(w-(1-a)).max()));flux.append(float(abs(trapezoid(w,a)-.5)));wall.append(float(abs(w[-1])))
    check('fluid_Re_and_U_profile',max(profile_errors)<.003,max_normalized_error=max(profile_errors),Re_range=[.11875,4.95])
    check('fluid_no_slip_and_mass_flux',max(wall)<1e-12 and max(flux)<.0015,max_wall_velocity=max(wall),max_normalized_flux_error=max(flux))
    # Re is a dependent dimensionless group: reject inconsistent out-of-domain combinations.
    pars=Parameters.from_re(.8,mean=.015,rho=1000.)
    err=abs(pars.Re-.8)
    try:Parameters.from_re(10000.);reject=False
    except ValueError:reject=True
    check('Re_viscosity_consistency',err<1e-12 and reject,Re_roundtrip_error=err,invalid_combination_rejected=reject)
    # IC position and velocity at every release-domain corner, and both initial-speed bounds.
    pos=[];vel=[]
    for x in (-.7,.7):
      for z in (-2,1.7):
       for vx in (-.001,.001):
        for vz in (0,.06):
         par=Parameters(x0=x*RV,z0=z*D,vx0=vx,vz0=vz).validate();q,v=par.initial()
         qp,vp=transition(model,q,v,.25,par,h=0,velocity=True)
         pos.append(float(abs(qp-q).max()));vel.append(float(abs(vp-v).max()))
    check('release_position_hard_IC',max(pos)<1e-12,max_dimensionless_error=max(pos))
    check('initial_particle_velocity_hard_IC',max(vel)<1e-12,max_dimensionless_error=max(vel))
    # Initial velocity is a real sub-microsecond transient, not a persistent free parameter.
    relerr=[];micro_errors=[]
    for eta in (.001,.006):
      par=Parameters(eta=eta,x0=0,z0=-.04,vx0=.001,vz0=.06)
      q,v=par.initial();eps=par.relaxation_s/TS
      for n in (.01,1.,10.,20.):
       qp,vp=transition(model,q,v,0,par,h=n*eps,velocity=True)
       qo,vo=oracle_step(q,v,0,par,h=n*eps)
       micro_errors.append(float(np.linalg.norm((qp-qo)*np.array([RV,D]))))
       relerr.append(float(np.linalg.norm((vp-vo)*np.array([RV,D])/TS)))
    check('initial_speed_Stokes_relaxation',max(relerr)<1e-5 and max(micro_errors)<1e-9,max_velocity_error_m_s=max(relerr),max_position_error_m=max(micro_errors),relaxation_time_s=[2*(RHOP+500)*RP**2/(9*.006),2*(RHOP+500)*RP**2/(9*.001)])
    # A held-out PINN residual test contains no oracle/trajectory labels.
    torch.manual_seed(1109);s,q,v,u,p=sample(4096,'cpu')
    s=s.double().detach().requires_grad_(True);q=q.double();v=v.double();u=u.double();p=p.double()
    pred,_,bal=model.terms(s,q,v,u,p,second=True);res=(bal-rhs(pred,u,p,model.flow)).detach().numpy()
    rms=float(np.sqrt(np.mean(res**2)));check('heldout_inertial_equation',rms<.002,residual_rms=rms)
    # Independent full-inertia Radau, random interior + physical corners.
    cases=corner_params+[Parameters(mean=rng.uniform(.005,.03),eta=np.exp(rng.uniform(np.log(.001),np.log(.006))),rho=rng.uniform(950,1100),x0=rng.uniform(-.6,.6)*RV,z0=rng.uniform(-2,1.6)*D,vx0=rng.uniform(-.001,.001),vz0=rng.uniform(0,.06)) for _ in range(40)]
    ex=[];ez=[]
    for par in cases:
        q,v=par.initial();u=float(rng.uniform(-.5,.5));pred=transition(model,q,v,u,par)
        oracle,_=oracle_step(q,v,u,par)
        ex.append(float(abs(pred[0]-oracle[0])*RV*1e6));ez.append(float(abs(pred[1]-oracle[1])*D*1e6))
    check('independent_parametric_Radau',max(ex)<.15 and max(ez)<3.,cases=len(cases),max_x_error_um=max(ex),max_z_error_um=max(ez))
    # Inverse-viscosity scaling of terminal mobility, while unforced axial flow is fixed.
    p1=Parameters(eta=.001);p2=Parameters(eta=.006);q=np.array([.2,0.])
    def neural_velocity(par,u):
        qt=torch.tensor(q[None],dtype=torch.float64);ut=torch.tensor([[u]],dtype=torch.float64)
        with torch.no_grad():return rhs(qt,ut,par.tensor(dtype=torch.float64),model.flow).numpy()[0]
    ratio=neural_velocity(p1,.2)[0]/neural_velocity(p2,.2)[0]
    check('viscosity_magnetic_mobility',abs(ratio-6)<1e-8,mobility_ratio=ratio,expected=6.)
    # Fluid density alone changes Re but not the fully developed steady profile (eta,U fixed).
    qa,va=Parameters(rho=950).initial();a1=transition(model,qa,va,.2,Parameters(rho=950));a2=transition(model,qa,va,.2,Parameters(rho=1100))
    delta=(a1-a2)*np.array([RV,D])*1e6
    check('density_Re_invariance',abs(delta[0])<.1 and abs(delta[1])<3,max_difference_um=abs(delta).tolist())
    # Particle slip Reynolds number: Stokes drag is used within a low-Re_p regime.
    max_rep=1100*2*RP*.061/.001
    check('Stokes_drag_validity',max_rep<.1,max_conservative_particle_Re=max_rep)
    rows['all_passed']=all(r['passed'] for r in rows.values())
    (output/'physics_tests.json').write_text(json.dumps(rows,indent=2))
    if not rows['all_passed']:raise RuntimeError('Physical acceptance failed; inspect physics_tests.json')
    return rows
