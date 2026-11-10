"""Trained inverse of the learned field dynamics; closed-loop tracking is separate."""
import numpy as np
import torch
from torch import nn
from .runtime import ROOT, gain

class InversePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
    def forward(self,x): return self.net(x)

def features(q, vx, cfg):
    q=np.asarray(q)
    scale=2*gain(cfg)*cfg.umax/(1+q[...,1]**2)**3
    return np.stack((q[...,0]/.8,q[...,1]/2,np.asarray(vx)/scale),axis=-1)

class Controller:
    def __init__(self,cfg):
        ck=torch.load(ROOT/'checkpoints/inverse_policy.pt',map_location='cpu',weights_only=True)
        from dataclasses import asdict
        if ck['config'] != asdict(cfg): raise ValueError('Policy/config mismatch; retrain.')
        self.model=InversePolicy().eval()
        self.model.load_state_dict(ck['state_dict'])
        self.cfg=cfg
    def command(self,q,vx):
        feat=features(q,vx,self.cfg)
        if not np.isfinite(feat).all(): raise ValueError('Nonfinite controller input')
        # Extrapolation is rejected by clipping to the trained, bounded actuation domain.
        feat[...,2]=np.clip(feat[...,2],-1.01,1.01)
        with torch.no_grad(): u=self.model(torch.as_tensor(feat,dtype=torch.float32)).numpy()[...,0]
        return np.clip(u,-1,1)*self.cfg.umax
