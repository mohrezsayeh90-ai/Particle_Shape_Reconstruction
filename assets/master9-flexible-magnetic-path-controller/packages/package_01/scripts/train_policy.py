"""Train once over the state/action domain, independent of any requested path."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, json, time
from dataclasses import asdict
import numpy as np
import torch
from magnetic_control.runtime import load_fields, dynamics, ROOT
from magnetic_control.policy import InversePolicy, features

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=2500);a=ap.parse_args()
    cfg=load_fields();torch.manual_seed(317);rng=np.random.default_rng(317)
    # These labels come from the loaded, trained field PINNs, not reference trajectories.
    q=np.column_stack((rng.uniform(-.7,.7,60000),rng.uniform(-2,2,60000)))
    u=rng.uniform(-cfg.umax,cfg.umax,len(q))
    vel=dynamics(q,u,cfg)
    x=torch.tensor(features(q,vel[:,0],cfg),dtype=torch.float32)
    y=torch.tensor(u[:,None]/cfg.umax,dtype=torch.float32)
    model=InversePolicy();opt=torch.optim.Adam(model.parameters(),lr=.002)
    start=time.time()
    for step in range(a.steps):
        idx=torch.randint(len(x),(1024,));loss=((model(x[idx])-y[idx])**2).mean()
        opt.zero_grad();loss.backward();opt.step()
        if step in (1500,2100):
            for g in opt.param_groups:g['lr']*=.25
        if step%500==0: print(step,float(loss.detach()),flush=True)
    # Independent held-out states/actions, with a different RNG and no training reuse.
    rng=np.random.default_rng(9801)
    qt=np.column_stack((rng.uniform(-.7,.7,12000),rng.uniform(-2,2,12000)))
    ut=rng.uniform(-cfg.umax,cfg.umax,len(qt));vt=dynamics(qt,ut,cfg)
    with torch.no_grad():pred=model(torch.tensor(features(qt,vt[:,0],cfg),dtype=torch.float32)).numpy()[:,0]*cfg.umax
    report=dict(seed=317,steps=a.steps,training_samples=len(q),heldout_samples=len(qt),
        command_mae=float(np.abs(pred-ut).mean()),command_max_error=float(np.abs(pred-ut).max()),
        training_seconds=time.time()-start,label_source='Master8 trained fluid/magnetic PINN fields; no target paths',
        state_domain={'X':[-.7,.7],'Z':[-2,2],'u':[-cfg.umax,cfg.umax]})
    report['passed']=report['command_max_error']<.015
    torch.save(dict(state_dict=model.state_dict(),config=asdict(cfg),training=report),ROOT/'checkpoints/inverse_policy.pt')
    (ROOT/'results/training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));assert report['passed']
if __name__=='__main__':main()
