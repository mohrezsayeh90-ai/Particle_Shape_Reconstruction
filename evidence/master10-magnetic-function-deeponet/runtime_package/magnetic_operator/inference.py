"""Fast windowed inference, adaptive field sampling, and independent ODE oracle."""
import time
import numpy as np
import torch
from scipy.integrate import solve_ivp
from .physics import *
from .waveforms import check_samples

def check_state(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(2,) or not np.isfinite(q).all() or abs(q[0])>.7+1e-10 or not -2-1e-10<=q[1]<=1.7+1e-10:
        raise ValueError('Particle outside validated domain |x/R|<=0.7, -2<=z/d<=1.7')
    return q

def local_predict(model,a,q,p,h,s):
    a=check_samples(a);p.validate();q=check_state(q)
    s=np.atleast_1d(s).astype(float)
    if not HMIN-1e-12<=h<=HMAX+1e-12 or not np.isfinite(s).all() or np.any((s<0)|(s>1)):
        raise ValueError('Invalid local horizon/query time')
    # A raw neural step above .05 accumulated excessive axial error in the
    # independent GPU acceptance. Compose two supported shorter maps instead.
    # Halving a uniform piecewise-linear sensor grid preserves its knots exactly.
    if h>.05+1e-12:
        result=np.empty((len(s),2));state=q.copy()
        grid=np.linspace(0,1,NSENSORS)
        for j in range(2):
            subgrid=(j+grid)/2
            aa=np.column_stack([np.interp(subgrid,grid,a[:,k]) for k in range(2)])
            indices=np.flatnonzero((s>=j/2)&(s<=(j+1)/2))
            queries=np.r_[2*s[indices]-j,1.]
            step=local_predict(model,aa,state,p,h/2,np.clip(queries,0,1))
            result[indices]=step[:-1];state=step[-1]
        return result
    parameter=next(model.parameters());device=parameter.device;dtype=parameter.dtype
    n=len(s)
    with torch.no_grad():
        out=model(torch.as_tensor(s[:,None],device=device,dtype=dtype),
            torch.as_tensor(a[None],device=device,dtype=dtype).expand(n,-1,-1),
            torch.as_tensor(q[None],device=device,dtype=dtype).expand(n,-1),
            p.tensor(device,dtype).expand(n,-1),
            torch.full((n,1),h,device=device,dtype=dtype))
    return out.cpu().numpy()

def reference_local(a,q,p,h,s):
    """DOP853 on the sensor-interpolated input, never used by training."""
    a=check_samples(a);grid=np.linspace(0,h,NSENSORS)
    def fun(t,q):
        av=np.array([np.interp(t,grid,a[:,i]) for i in range(2)])
        return numpy_rhs(q,av,p)
    sol=solve_ivp(fun,[0,h],q,method='DOP853',rtol=2e-10,atol=2e-12,
                  max_step=h/(4*(NSENSORS-1)),dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.sol(np.atleast_1d(s)*h).T

def predict(model,field,times_s,parameters=Parameters(),initial_m=(0.,-.08),
            sampling_tolerance=.005,max_window_tau=.05):
    """field(t_seconds) -> (a_left,a_right), signed relative SOURCE amplitudes.

    No retraining and no ODE solver in this path. Known time waveform required
    per window. Sampling checks are a heuristic, not a proof against aliasing.
    Under-resolved input or domain exit raises; neither is silently extrapolated.
    """
    par=parameters.validate();times=np.asarray(times_s,dtype=float)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or times[0]!=0 or np.any(np.diff(times)<=0):
        raise ValueError('Query times must start at zero and increase strictly')
    if not np.isfinite(sampling_tolerance) or sampling_tolerance<=0:
        raise ValueError('sampling_tolerance must be positive')
    if not np.isfinite(max_window_tau) or not HMIN<=max_window_tau<=HMAX:
        raise ValueError('max_window_tau outside trained horizon domain')
    q=check_state(np.asarray(initial_m,dtype=float)/[R,D]);out=np.empty((len(times),2));out[0]=q
    t=0.;stop=times[-1];windows=[];start=time.perf_counter()
    while t<stop-1e-12:
        h=max_window_tau
        while True:
            grid=np.linspace(0,1,NSENSORS)
            a=check_samples([field(min(stop,t+v*h*TS)) for v in grid])
            # Off-grid probes, including non-midpoint locations, detect most
            # under-resolved inputs. Finite probes cannot certify any callable.
            probes=np.concatenate(((np.arange(NSENSORS-1)+.5)/(NSENSORS-1),
                                   (np.arange(NSENSORS-1)+.271)/(NSENSORS-1)))
            actual=np.asarray([field(min(stop,t+v*h*TS)) for v in probes],float)
            if actual.shape!=(len(probes),2) or not np.isfinite(actual).all() or np.any(np.abs(actual).sum(1)>AMAX+1e-10):
                raise ValueError('Field callable invalid between sensors')
            interp=np.column_stack([np.interp(probes,grid,a[:,i]) for i in range(2)])
            discrepancy=float(np.max(np.abs(actual-interp)))
            if discrepancy<=sampling_tolerance:break
            if h/2<HMIN-1e-12:
                raise ValueError('Waveform under-resolved: use reference solver or finer/retrained operator')
            h/=2
        end=min(stop,t+h*TS);fraction=(end-t)/(h*TS)
        indices=np.flatnonzero((times>t+1e-12)&(times<=end+1e-12))
        queries=np.r_[(times[indices]-t)/(h*TS),np.linspace(0,fraction,33),fraction]
        pred=local_predict(model,a,q,par,h,np.clip(queries,0,1))
        for state in pred:check_state(state)
        out[indices]=pred[:len(indices)]
        q=pred[-1];windows.append({'start_s':t,'end_s':end,'h_tau':h,'sampling_error':discrepancy})
        t=end
    return out*np.array([R,D]),{'seconds':time.perf_counter()-start,'windows':windows,
                                'ode_solves':0,'retraining_steps':0}

def reference(field,times_s,parameters=Parameters(),initial_m=(0.,-.08),max_step_s=.002):
    """Numerical solution of the ORIGINAL supplied callable, not its sensor samples.

    For known discontinuities, split the solve at each discontinuity. This method
    accepts more waveforms than the trained fast operator but is not instant.
    """
    par=parameters.validate();times=np.asarray(times_s,float)
    if times.ndim!=1 or len(times)<2 or times[0]!=0 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid times')
    q0=check_state(np.asarray(initial_m,float)/[R,D])
    def fun(t,q):
        a=np.asarray(field(t*TS),float)
        if a.shape!=(2,) or not np.isfinite(a).all() or np.abs(a).sum()>AMAX+1e-10:
            raise ValueError('Invalid field callable')
        return numpy_rhs(q,a,par)
    def boundary(t,q):return min(.7-abs(q[0]),q[1]+2,1.7-q[1])
    boundary.terminal=True;boundary.direction=-1
    sol=solve_ivp(fun,[0,times[-1]/TS],q0,method='DOP853',rtol=2e-10,atol=2e-12,
                  max_step=max_step_s/TS,events=boundary,dense_output=True)
    if not sol.success or sol.t[-1]<times[-1]/TS-1e-10:raise ValueError('Reference trajectory exited domain')
    return sol.sol(times/TS).T*np.array([R,D])
