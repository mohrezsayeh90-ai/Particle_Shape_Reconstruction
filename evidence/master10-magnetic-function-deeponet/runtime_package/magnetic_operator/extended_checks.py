"""Additional domain-face and parametric rollout checks, held out from training."""
from pathlib import Path
import json
import numpy as np
import torch
from .physics import *
from .inference import predict,reference,local_predict,reference_local
from .train import load

def run(output):
    output=Path(output);model=load(output);report={}
    q=torch.tensor([[.23,-.8]],dtype=torch.float64,requires_grad=True)
    b,j=source(q,1)
    ja=torch.stack([torch.autograd.grad(b[:,i].sum(),q,retain_graph=True)[0] for i in range(2)],1)
    ja[:,:,0]*=D/R
    error=float((ja-j).abs().max().detach())
    report['source_Maxwell_and_Jacobian']={'passed':error<1e-12 and float((j[:,0,0]+j[:,1,1]).abs().max().detach())<1e-12,
                                         'max_Jacobian_error':error}
    rows=[]
    for eta,U,rho,x0 in ((.001,.005,950,-.3),(.006,.03,1100,.3),(.001,.03,1100,0.),(.006,.005,950,.4)):
        par=Parameters(eta=eta,mean=U,rho=rho)
        field=lambda t:(.15*np.sin(2*np.pi*.9*t),.12*np.cos(2*np.pi*.6*t))
        times=np.linspace(0,1.25,201);initial=(x0*R,-.04)
        pred,meta=predict(model,field,times,par,initial)
        truth=reference(field,times,par,initial)
        err=np.max(abs(pred-truth)*1e6,axis=0)
        rows.append({'eta':eta,'mean':U,'rho':rho,'X0':x0,'x_um':float(err[0]),'z_um':float(err[1]),
                     'passed':bool(err[0]<.75 and err[1]<15)})
    report['parametric_rollouts']={'passed':all(r['passed'] for r in rows),'cases':rows}
    rows=[]
    for ai in ((AMAX,0),(0,AMAX),(AMAX/2,AMAX/2),(AMAX/2,-AMAX/2)):
        a=np.tile(ai,(NSENSORS,1));par=Parameters(eta=.001,mean=.03);s=np.linspace(0,1,33)
        pred=local_predict(model,a,np.array([.2,-.5]),par,.05,s)
        truth=reference_local(a,[.2,-.5],par,.05,s)
        err=np.max(abs(pred-truth)*[R*1e6,D*1e6],axis=0)
        rows.append({'amplitudes':list(ai),'x_um':float(err[0]),'z_um':float(err[1]),
                     'passed':bool(err[0]<.15 and err[1]<3.)})
    report['maximum_amplitude_faces']={'passed':all(r['passed'] for r in rows),'h_tau':.05,'cases':rows}
    try:predict(model,lambda t:(0.,0.),np.linspace(0,20,100));rejected=False
    except ValueError:rejected=True
    report['domain_exit_rejected']={'passed':rejected}
    # Physical density invariance is imposed by removing rho from the ODE, but
    # rho remains in the branch for compatibility with the parameter structure.
    # Measure any spurious learned dependence explicitly.
    a=np.tile([.25,.1],(NSENSORS,1));s=np.linspace(0,1,33)
    qa=local_predict(model,a,[.2,-.5],Parameters(rho=950),.05,s)
    qb=local_predict(model,a,[.2,-.5],Parameters(rho=1100),.05,s)
    err=np.max(abs(qa-qb)*[R*1e6,D*1e6],axis=0)
    report['density_invariance']={'passed':bool(err[0]<.02 and err[1]<1.),'x_um':float(err[0]),'z_um':float(err[1])}
    report['all_passed']=all(r['passed'] for r in report.values())
    (output/'extended_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not report['all_passed']:raise RuntimeError('Extended acceptance failed')
    return report
