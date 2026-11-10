"""Export simulated trajectory, bounded actuation and effective magnetic quantities."""
from pathlib import Path
import json,numpy as np,torch
from .runtime import dynamics
from . import fields as f

def save_result(result,cfg,destination):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    q=result['q'];u=result['u'];side=np.where(u<0,1.,-1.)
    points=np.column_stack((side*q[:,0]*cfg.Rv/cfg.d,q[:,1]))
    b,_=f.MAG.field_force(torch.tensor(points,dtype=torch.float64))
    bs=cfg.mu0*cfg.Ms*cfg.Rmag**2/(2*cfg.d**2)
    bmag=np.linalg.norm(b.detach().numpy(),axis=1)*bs*np.sqrt(abs(u))
    # Stokes drag balances the effective magnetic force in this overdamped model.
    velocity=dynamics(q,u,cfg)
    fx=cfg.drag*2*cfg.mean*cfg.Rv/cfg.d*velocity[:,0]
    with torch.no_grad():flow=f.FLUID.w(torch.tensor(q[:,0:1]**2,dtype=torch.float64)).numpy()[:,0]
    fz=cfg.drag*2*cfg.mean*(velocity[:,1]-flow)
    data=np.column_stack((result['tau']*cfg.time_scale,result['target']*cfg.Rv,q[:,0]*cfg.Rv,q[:,1]*cfg.d,u,side,np.sqrt(abs(u)),bmag,fx,fz))
    np.savetxt(dest/'control.csv',data,delimiter=',',comments='',header='time_s,target_x_m,x_m,z_m,u,source_side_minus1_is_right,source_amplitude_scale,B_magnitude_T,Fmag_x_N,Fmag_z_N')
    (dest/'report.json').write_text(json.dumps(result['report'],indent=2))
    return dest
