"""Online constrained inversion of the residual-trained PINN; no control labels."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from scipy.optimize import brentq
from .physics import *
from .validate import transition

def follow(model,par,amplitude=.02,z_end=1.4,max_steps=700,target_csv=None):
    par.validate();q,v=par.initial();qr=q.copy();vr=v.copy();z0=q[1];x0=q[0]
    offset=np.tanh((z0+.3)*2)
    def target(z):return x0+amplitude*(np.tanh((z+.3)*2)-offset)
    if target_csv is not None:
        from scipy.interpolate import PchipInterpolator
        table=np.loadtxt(target_csv,delimiter=',',skiprows=1)
        if table.ndim!=2 or table.shape[1]!=2 or len(table)<4 or not np.isfinite(table).all():raise ValueError('Expected >=4 finite z_m,x_m rows')
        zz=table[:,0]/D;xx=table[:,1]/RV
        if np.any(np.diff(zz)<=0) or abs(zz[0]-z0)>1e-9 or abs(xx[0]-x0)>1e-9 or max(abs(xx))>=.7 or zz[-1]>1.4:
            raise ValueError('Path must start at release, increase in z, stay within |X|<.7 and end at Z<=1.4')
        z_end=float(zz[-1]);curve=PchipInterpolator(zz,xx,extrapolate=False)
        def target(z):return float(curve(np.clip(z,zz[0],zz[-1])))
    rows=[]
    for step in range(max_steps):
        if abs(q[0])>.7 or q[1]>1.7 or qr[1]>1.8 or abs(qr[0])>.7:break
        def error(u):
            nxt=transition(model,q,v,u,par)
            return nxt[0]-target(nxt[1])
        lo,hi=error(-.5),error(.5)
        if lo*hi<=0:u=float(brentq(error,-.5,.5,xtol=1e-9))
        else:
            candidates=[-.5,0.,.5];u=min(candidates,key=lambda u:abs(error(u)))
        rows.append([step*H*TS,q[0]*RV,q[1]*D,target(q[1])*RV,u,qr[0]*RV,qr[1]*D,target(qr[1])*RV])
        q,v=transition(model,q,v,u,par,velocity=True)
        qr,vr=oracle_step(qr,vr,u,par)
        if q[1]>=z_end:
            rows.append([(step+1)*H*TS,q[0]*RV,q[1]*D,target(q[1])*RV,u,qr[0]*RV,qr[1]*D,target(qr[1])*RV]);break
    a=np.array(rows)
    err=float(np.max(abs(a[:,5]-a[:,7]))*1e6)
    drift_x=float(np.max(abs(a[:,1]-a[:,5]))*1e6);drift_z=float(np.max(abs(a[:,2]-a[:,6]))*1e6)
    report={'parameters':asdict(par),'Re':par.Re,'max_independent_tracking_error_um':err,
        'max_model_oracle_x_difference_um':drift_x,'max_model_oracle_z_difference_um':drift_z,
        'completed':bool(q[1]>=z_end and qr[1]>=z_end-.1),'steps':len(a),'max_abs_u':float(max(abs(a[:,4])))}
    report['passed']=bool(report['completed'] and err<.75 and drift_x<.5 and drift_z<150)
    return a,report

def demos(model,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);fig,axes=plt.subplots(2,2,figsize=(11,7));reports={}
    cases={'nominal':Parameters(),
        'low_speed_high_viscosity':Parameters(mean=.005,eta=.006,x0=-.15*RV,z0=-1.8*D,vx0=.001,vz0=0),
        'high_Re_fast_release':Parameters(mean=.03,eta=.001,rho=1100,x0=.15*RV,z0=-1.7*D,vx0=-.001,vz0=.06),
        'shifted_release':Parameters(mean=.02,eta=.004,rho=950,x0=-.25*RV,z0=-1.6*D,vx0=.0005,vz0=.005)}
    for ax,(name,par) in zip(axes.flat,cases.items()):
        a,r=follow(model,par);reports[name]=r;print('control',name,r,flush=True)
        np.savetxt(output/f'control_{name}.csv',a,delimiter=',',comments='',header='time_s,x_PINN_m,z_PINN_m,target_x_at_PINN_z_m,u,x_Radau_m,z_Radau_m,target_x_at_Radau_z_m')
        ax.plot(a[:,2]*1000,a[:,3]*1e6,'k--',label='Target');ax.plot(a[:,2]*1000,a[:,1]*1e6,label='Parametric PINN');ax.plot(a[:,6]*1000,a[:,5]*1e6,':',label='Independent inertial Radau')
        ax.set(title=name,xlabel='z (mm)',ylabel='x (micrometres)');ax.legend(fontsize=7)
    reports['all_passed']=all(r['passed'] for r in reports.values());(output/'control_tests.json').write_text(json.dumps(reports,indent=2))
    fig.tight_layout();fig.savefig(output/'control_results.png',dpi=160)
    if not reports['all_passed']:raise RuntimeError('Path control acceptance failed')
    return reports
