from pathlib import Path
import json,time,hashlib
import torch
from .model import MagneticDeepONet,residual,sample
from .physics import *

def train(output,steps=18000,batch=1024,require_gpu=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError('GPU training requested, CUDA unavailable')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2);torch.manual_seed(1041)
    model=MagneticDeepONet().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=1e-5)
    start=time.perf_counter();history=[]
    for step in range(steps):
        args=sample(batch,device)
        loss=residual(model,*args).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10)
        opt.step();sched.step()
        if step%500==0 or step==steps-1:
            row={'step':step+1,'loss':float(loss.detach()),'seconds':time.perf_counter()-start}
            history.append(row);print(row,flush=True)
            torch.save({'model':model.state_dict(),'rank':model.rank},output/'checkpoint.pt')
            (output/'progress.json').write_text(json.dumps(history,indent=2))
    if device=='cuda':torch.cuda.synchronize()
    info={'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
          'steps':steps,'batch':batch,'seconds':time.perf_counter()-start,'seed':1041,
          'torch':str(torch.__version__),'labelled_trajectories':0,
          'method':'physics-lifted branch/trunk DeepONet; overdamped ODE residual only',
          'sensors_per_source':NSENSORS,'h_tau':[HMIN,HMAX],
          'amplitude_constraint':'abs(a_left)+abs(a_right)<=sqrt(0.5)',
          'checkpoint_sha256':hashlib.sha256((output/'checkpoint.pt').read_bytes()).hexdigest()}
    (output/'training.json').write_text(json.dumps(info,indent=2))
    return model,info

def load(output,device='cpu',dtype=torch.float64):
    data=torch.load(Path(output)/'checkpoint.pt',map_location=device,weights_only=True)
    model=MagneticDeepONet(data['rank']).to(device=device,dtype=dtype)
    model.load_state_dict(data['model']);model.eval()
    return model
