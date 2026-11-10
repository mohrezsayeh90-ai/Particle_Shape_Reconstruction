"""Load audited Master8 fields; do not silently change physical parameters."""
from pathlib import Path
import numpy as np
import torch
from . import fields as f
ROOT = Path(__file__).resolve().parents[1]

def load_fields():
    torch.set_num_threads(1)
    ck = torch.load(ROOT/'checkpoints/fields.pt', map_location='cpu', weights_only=True)
    cfg = f.Config(**ck['config'])
    f.FLUID = f.FluidPINN().double()
    f.MAG = f.MagneticPINN().double()
    f.FLUID.load_state_dict(ck['fluid'])
    f.MAG.load_state_dict(ck['magnetic'])
    for model in (f.FLUID, f.MAG):
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
    f.FORCES = f.LearnedForces(f.MAG, cfg, ck['coefficients'])
    return cfg

def gain(cfg):
    bs = cfg.mu0*cfg.Ms*cfg.Rmag**2/(2*cfg.d**2)
    alpha = 3*(4*np.pi*cfg.rp**3/3)/cfg.mu0
    return alpha*bs**2/(cfg.drag*2*cfg.mean*cfg.Rv)

def dynamics(q, u, cfg, oracle=False):
    """d(X,Z)/d(tau). Oracle uses independent analytical fields, not the PINN."""
    X, Z = np.asarray(q).T
    u = np.asarray(u)
    side = np.where(u < 0, 1., -1.)
    if oracle:
        _, force = f.analytic_magnetic(np.column_stack((np.ravel(side*X*cfg.Rv/cfg.d), np.ravel(Z))))
        force = force.reshape(np.shape(X)+(2,))
        velocity = 1-X*X
    else:
        force = np.moveaxis(f.FORCES.numpy(side*X,Z), 0, -1)
        with torch.no_grad():
            velocity = f.FLUID.w(torch.tensor(np.reshape(X*X,(-1,1)), dtype=torch.float64)).numpy().reshape(np.shape(X))
    fx = force[...,0]*side*abs(u)
    fz = force[...,1]*abs(u)
    return np.stack((gain(cfg)*fx, velocity+gain(cfg)*cfg.Rv/cfg.d*fz), axis=-1)
