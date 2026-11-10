from pathlib import Path
import json,time,os
import numpy as np,torch
from .models import *

def train(output,flow_steps=3000,particle_steps=12000,batch=2048,require_gpu=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if require_gpu and not torch.cuda.is_available():raise RuntimeError('A GPU run was requested but CUDA is unavailable.')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cpu':torch.set_num_threads(2)
    torch.manual_seed(903);np.random.seed(903)
    runtime={'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'torch':str(torch.__version__),'seed':903,'batch':batch,'labelled_trajectories':0,'labelled_controls':0,'loss':'fluid momentum residual; inertial Stokes residual + hard position/velocity IC'}
    print(json.dumps(runtime),flush=True);start=time.time()
    flow=FlowPINN().to(device);opt=torch.optim.Adam(flow.parameters(),lr=.001)
    log=[]
    for step in range(flow_steps):
        _,_,_,_,p=sample(batch,device);q=torch.rand((batch,1),device=device,requires_grad=True)
        w=flow(q,p);wq=torch.autograd.grad(w.sum(),q,create_graph=True)[0]
        wqq=torch.autograd.grad(wq.sum(),q,create_graph=True)[0]
        res=q*wqq+wq+1
        loss=(res*res).mean()
        grid=torch.linspace(0,1,65,device=device)[:,None]
        flux=torch.trapezoid(flow(grid,p[:1].expand(65,-1))[:,0],grid[:,0])
        loss=loss+10*(flux-.5)**2
        opt.zero_grad();loss.backward();opt.step()
        if step%500==0:print('flow',step,float(loss.detach()),flush=True);log.append(['flow',step,float(loss.detach())])
        if step in (1500,2500):
            for g in opt.param_groups:g['lr']*=.3
    flow.eval()
    for p in flow.parameters():p.requires_grad_(False)
    model=ParticlePINN(flow).to(device);opt=torch.optim.Adam(model.net.parameters(),lr=.001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,particle_steps,eta_min=2e-5)
    for step in range(particle_steps):
        s,q,v,u,p=sample(batch,device)
        # Include initial relaxation layer using logarithmically spaced physical times.
        if step%4==0:
            eps=eps_of(p).detach();s=(eps/H*torch.exp(torch.empty_like(s).uniform_(-5,4))).requires_grad_(True)
        pred,_,balance=model.terms(s,q,v,u,p,second=True)
        res=(balance-rhs(pred,u,p,flow))/H
        loss=(res*res).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.net.parameters(),10);opt.step();schedule.step()
        if step%500==0:
            val=float(loss.detach());print('particle',step,val,flush=True);log.append(['particle',step,val])
            torch.save({'model':model.cpu().state_dict(),'runtime':runtime},output/'checkpoint.pt');model.to(device)
            (output/'training_progress.json').write_text(json.dumps(log,indent=2))
    if device=='cuda':torch.cuda.synchronize()
    runtime.update(flow_steps=flow_steps,particle_steps=particle_steps,training_seconds=time.time()-start,ranges=RANGES)
    torch.save({'model':model.cpu().state_dict(),'runtime':runtime},output/'checkpoint.pt')
    (output/'training.json').write_text(json.dumps(runtime,indent=2));(output/'training_progress.json').write_text(json.dumps(log,indent=2))
    return model,runtime


def load(output):
    torch.set_num_threads(2)
    ck=torch.load(Path(output)/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=ParticlePINN(FlowPINN());model.load_state_dict(ck['model']);model.eval()
    return model,ck['runtime']
