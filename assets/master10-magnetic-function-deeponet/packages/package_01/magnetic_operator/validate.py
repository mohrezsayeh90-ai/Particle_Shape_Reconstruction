"""Predefined held-out acceptance criteria; no validation data enters training."""
from pathlib import Path
import json,time
import numpy as np
import torch
from scipy.integrate import solve_ivp
from .physics import *
from .model import sample,residual
from .waveforms import random_samples,evaluate_and_integrate
from .inference import local_predict,reference_local,predict,reference
from .train import load

def run(output,require_gpu=True):
    output=Path(output);torch.set_num_threads(2)
    model=load(output);rng=np.random.default_rng(1067);torch.manual_seed(1067)
    report={};cases=[]
    def check(name,passed,**data):
        report[name]={'passed':bool(passed),**data};print(name,report[name],flush=True)
    info=json.loads((output/'training.json').read_text())
    check('gpu_training',info['device']=='cuda' or not require_gpu,device=info['device'],gpu=info['gpu'])
    # Independent complex-potential vs differentiable field formula, including cross terms.
    q=np.column_stack((rng.uniform(-.7,.7,512),rng.uniform(-2,1.7,512)))
    a=rng.uniform(-.3,.3,(512,2));products=np.column_stack((a[:,0]**2,a[:,1]**2,a[:,0]*a[:,1]))
    ft=(force_basis(torch.tensor(q))*torch.tensor(products)[:,:,None]).sum(1).numpy()
    fn=np.array([numpy_field_force(qi,ai)[1] for qi,ai in zip(q,a)])
    ferr=float(abs(ft-fn).max())
    check('independent_field_and_cross_terms',ferr<1e-12,max_force_error=ferr)
    # Recover old Master9 force for one source; reversing its polarity preserves force.
    old=[]
    for side,ai in ((1,[.5,0]),(-1,[0,.5])):
        y=1+side*q[:,0]*R/D;z=q[:,1]
        legacy=-2*np.column_stack((side*y,z))/(y*y+z*z)[:,None]**3*.25
        new=np.array([numpy_field_force(qi,ai)[1] for qi in q])
        old.append(float(abs(legacy-new).max()))
    check('Master9_single_source_reduction',max(old)<1e-12,max_error=max(old))
    f1=numpy_field_force([.2,.3],[.25,-.2])[1]
    f2=numpy_field_force([.2,.3],[-.25,.2])[1]
    check('global_polarity_invariance',np.max(abs(f1-f2))<1e-12)
    # Validate exact antiderivative by automatic differentiation, independently of network.
    ar=random_samples(256,'cpu').double();s=torch.rand(256,1,dtype=torch.float64,requires_grad=True)
    av,integ=evaluate_and_integrate(ar,s)
    derivative=torch.cat([torch.autograd.grad(integ[:,i].sum(),s,retain_graph=True)[0] for i in range(3)],1)
    expected=torch.stack((av[:,0]**2,av[:,1]**2,av[:,0]*av[:,1]),1)
    ierr=float((derivative-expected).abs().max().detach())
    check('quadratic_waveform_integrals',ierr<1e-12,max_error=ierr)
    args=[v.double().detach() for v in sample(1024,'cpu')];args[0].requires_grad_(True)
    res=residual(model,*args,create_graph=False).detach().numpy()
    rms=float(np.sqrt(np.mean(res**2)))
    check('heldout_physics_residual',rms<.03,scaled_residual_rms=rms)
    # Hard IC, independently varying viscosity/U/rho, arbitrary sensor functions.
    ar=random_samples(64,'cpu').double().numpy();maxic=0
    for i in range(len(ar)):
        p=Parameters(eta=float(np.exp(rng.uniform(np.log(.001),np.log(.006)))),mean=float(rng.uniform(.005,.03)),rho=float(rng.uniform(950,1100)))
        q0=np.array([rng.uniform(-.5,.5),rng.uniform(-1.8,1.3)])
        h=float(np.exp(rng.uniform(np.log(HMIN),np.log(HMAX))))
        ss=np.linspace(0,1,33)
        pred=local_predict(model,ar[i],q0,p,h,ss)
        truth=reference_local(ar[i],q0,p,h,ss)
        maxic=max(maxic,float(abs(pred[0]-q0).max()))
        err=abs(pred-truth)*[R*1e6,D*1e6]
        cases.append({'x_um':float(err[:,0].max()),'z_um':float(err[:,1].max())})
    ex=max(c['x_um'] for c in cases);ez=max(c['z_um'] for c in cases)
    check('exact_initial_position',maxic<1e-12,max_error=maxic)
    check('independent_local_DOP853',ex<.15 and ez<3.,cases=len(cases),max_x_um=ex,max_z_um=ez)
    # Whole unseen functions, not merely unseen collocation points.
    functions={
        'zero':lambda t:(0.,0.),
        'constant_left':lambda t:(.32,0.),
        'constant_right':lambda t:(0.,.32),
        'sinusoidal_two_sources':lambda t:(.23*np.sin(2*np.pi*.8*t),.2*np.cos(2*np.pi*.37*t)),
        'chirp':lambda t:(.26*np.sin(2*np.pi*(.3*t+.35*t*t)),.1),
        'gaussian_pulse':lambda t:(.42*np.exp(-((t-1.4)/.24)**2),-.12),
        'triangular':lambda t:(.3*(2*abs(2*((t*.7)%1)-1)-1),.08),
    }
    full=[];times=np.linspace(0,3.,301);curves={}
    for name,field in functions.items():
        p=Parameters();pred,meta=predict(model,field,times,p)
        start=time.perf_counter();truth=reference(field,times,p);reference_seconds=time.perf_counter()-start
        err=abs(pred-truth)*1e6
        row={'name':name,'x_um':float(err[:,0].max()),'z_um':float(err[:,1].max()),
             'inference_seconds':meta['seconds'],'reference_seconds':reference_seconds,'windows':len(meta['windows'])}
        full.append(row);curves[name]=(pred,truth)
        np.savetxt(output/(name+'.csv'),np.column_stack((times,pred,truth)),delimiter=',',
                   header='time_s,x_operator_m,z_operator_m,x_reference_m,z_reference_m',comments='')
    check('unseen_callable_rollouts',all(r['x_um']<.75 and r['z_um']<15 for r in full),cases=full)
    # Full-inertia model comparison quantifies the approximation; not used as labels.
    par=Parameters(eta=.001);eps=par.relaxation_s/TS;q0=np.array([.1,-.8]);v0=np.array([.001*TS/R,.06*TS/D])
    field=lambda t: np.array([.3*np.cos(2*np.pi*t),.12])
    def inertial(t,y):return np.r_[y[2:],(numpy_rhs(y[:2],field(t*TS),par)-y[2:])/eps]
    sol=solve_ivp(inertial,[0,.1],np.r_[q0,v0],method='Radau',rtol=1e-9,atol=1e-11,dense_output=True)
    overdamped=solve_ivp(lambda t,q:numpy_rhs(q,field(t*TS),par),[0,.1],q0,method='DOP853',rtol=1e-11,atol=1e-13,dense_output=True)
    ts=np.r_[np.logspace(-10,-3,50),np.linspace(.001,.1,50)]
    err=abs(sol.sol(ts)[:2]-overdamped.sol(ts))*np.array([R,D])[:,None]*1e6
    check('overdamped_vs_inertial_position',sol.success and overdamped.success and float(err.max())<.03,
          max_x_um=float(err[0].max()),max_z_um=float(err[1].max()),relaxation_s=par.relaxation_s,
          note='Position only. Initial velocity and submicrosecond velocity transient are not predicted.')
    rejected={}
    for name,field in {'excess_amplitude':lambda t:(1.,0.),'nan':lambda t:(np.nan,0.),
                       'underresolved':lambda t:(.4*np.sin(2*np.pi*400*t),0.)}.items():
        try:predict(model,field,np.linspace(0,.5,20));rejected[name]=False
        except ValueError:rejected[name]=True
    check('invalid_or_underresolved_inputs_rejected',all(rejected.values()),cases=rejected)
    # A changed future must have only small influence on predicted past.
    a=np.full((NSENSORS,2),[.2,.1]);b=a.copy();b[9:]=[-.1,.3]
    qa=local_predict(model,a,[0.,-.5],Parameters(),.05,[.125,.25,.5])
    qb=local_predict(model,b,[0.,-.5],Parameters(),.05,[.125,.25,.5])
    leak=abs(qa-qb)*[R*1e6,D*1e6]
    check('future_prefix_sensitivity',float(leak[:,0].max())<.05 and float(leak[:,1].max())<1.,
          max_x_um=float(leak[:,0].max()),max_z_um=float(leak[:,1].max()),
          note='Approximate check, architecture is not strictly causal within each supplied window.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,(name,(pred,truth)) in zip(axes.flat,[(n,v) for n,v in curves.items() if n!='zero']):
        ax.plot(truth[:,1]*1000,truth[:,0]*1e6,label='DOP853')
        ax.plot(pred[:,1]*1000,pred[:,0]*1e6,'--',label='DeepONet')
        ax.set(title=name,xlabel='z (mm)',ylabel='x (um)');ax.legend()
    fig.tight_layout();fig.savefig(output/'unseen_trajectories.png',dpi=170);plt.close(fig)
    report['all_passed']=all(v['passed'] for v in report.values())
    (output/'validation.json').write_text(json.dumps(report,indent=2))
    if not report['all_passed']:raise RuntimeError('Acceptance failed; inspect validation.json')
    return report
