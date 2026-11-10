"""Closed-loop following of spatial waypoints or a transverse timed path, in SI units."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from .runtime import dynamics
from .policy import Controller

class Target:
    def __init__(self, table, cfg, mode='spatial'):
        table=np.asarray(table,dtype=float)
        if table.ndim!=2 or table.shape[1]!=2 or len(table)<4 or not np.isfinite(table).all():
            raise ValueError('Supply >=4 finite rows of [z_m,x_m] or [time_s,x_m].')
        if mode not in ('spatial','time'):raise ValueError('mode must be spatial or time')
        axis=table[:,0]/(cfg.d if mode=='spatial' else cfg.time_scale)
        x=table[:,1]/cfg.Rv
        if np.any(np.diff(axis)<=0):raise ValueError('Path axis must strictly increase.')
        start=cfg.Z0 if mode=='spatial' else 0.
        if abs(axis[0]-start)>1e-8 or abs(x[0]-cfg.X0)>1e-8:
            raise ValueError('Path must start at the trained release position/time.')
        if max(abs(x))>=.7:raise ValueError('Path outside the trained transverse domain.')
        if mode=='spatial' and (axis[-1]>1.8 or axis[-1]<=axis[0]):
            raise ValueError('Spatial endpoint must be <= Z=1.8, leaving validation margin.')
        if mode=='time' and (axis[-1]>cfg.T+1e-10 or axis[-1]<=0):
            raise ValueError('Time exceeds the inherited horizon.')
        self.mode=mode;self.axis=axis;self.curve=PchipInterpolator(axis,x,extrapolate=False)
        self.slope=self.curve.derivative();self.cfg=cfg
    def evaluate(self,tau,q):
        v=np.clip(q[1] if self.mode=='spatial' else tau,self.axis[0],self.axis[-1])
        return float(self.curve(v)),float(self.slope(v))

def track(target,cfg,oracle=False,feedback=4.,max_step=.015,flow_factor=1.):
    policy=Controller(cfg)
    def command(tau,q):
        desired,slope=target.evaluate(tau,q)
        u=0.
        for _ in range(3):
            # The inverse controller sees learned fields only, never oracle fields.
            axial=float(dynamics(q,u,cfg)[1])
            vx=slope*(axial if target.mode=='spatial' else 1.)+feedback*(desired-q[0])
            u=float(policy.command(q,vx))
        return u
    def rhs(tau,q):
        d=dynamics(q,command(tau,q),cfg,oracle=oracle)
        # Optional unmodelled flow perturbation for closed-loop robustness validation.
        d[1]+=(flow_factor-1)*(1-q[0]**2)
        return d
    def finish(tau,q):return q[1]-target.axis[-1]
    finish.terminal=True;finish.direction=1
    # Initial Z=-2 is a valid boundary. Separate upper Z / lower Z events avoid initial root.
    def wall(tau,q):return .7-abs(q[0])
    wall.terminal=True;wall.direction=-1
    def axial_limit(tau,q):return 2.-q[1]
    axial_limit.terminal=True;axial_limit.direction=-1
    horizon=cfg.T if target.mode=='spatial' else target.axis[-1]
    events=[wall,axial_limit]+([finish] if target.mode=='spatial' else [])
    sol=solve_ivp(rhs,[0,horizon],[cfg.X0,cfg.Z0],rtol=2e-7,atol=1e-9,
        max_step=max_step,dense_output=True,events=events)
    if not sol.success:raise RuntimeError(sol.message)
    tau=np.linspace(0,sol.t[-1],501);q=sol.sol(tau).T
    u=np.array([command(t,qi) for t,qi in zip(tau,q)])
    desired=np.array([target.evaluate(t,qi)[0] for t,qi in zip(tau,q)])
    err=q[:,0]-desired
    reached=(bool(len(sol.t_events[2])) if target.mode=='spatial' else abs(sol.t[-1]-horizon)<1e-8)
    report=dict(max_error_Rv=float(abs(err).max()),rmse_Rv=float(np.sqrt(np.mean(err**2))),
        max_error_um=float(abs(err).max()*cfg.Rv*1e6),completed=bool(reached),
        saturated_fraction=float(np.mean(abs(u)>=cfg.umax*.999)),
        accepted=bool(reached and abs(err).max()<.01),oracle=oracle,flow_factor=flow_factor,
        duration_s=float(tau[-1]*cfg.time_scale),feedback_gain=feedback)
    # The independent analytic model also replays the learned-plant open-loop schedule.
    def replay_rhs(t,q):return dynamics(q,np.interp(t,tau,u),cfg,oracle=True)
    replay=solve_ivp(replay_rhs,[0,tau[-1]],[cfg.X0,cfg.Z0],t_eval=tau,rtol=1e-9,atol=1e-11,max_step=max_step)
    if not replay.success:raise RuntimeError('Independent replay failed')
    replay_target=np.array([target.evaluate(t,qi)[0] for t,qi in zip(tau,replay.y.T)])
    replay_error=float(abs(replay.y[0]-replay_target).max())
    report['independent_open_loop_replay_max_error_Rv']=replay_error
    if not oracle:
        report['accepted']=bool(report['accepted'] and replay_error<.01 and np.max(abs(replay.y[0]))<.7 and np.max(abs(replay.y[1]))<=2.)
    return dict(report=report,tau=tau,q=q,u=u,target=desired,replay=replay.y.T)
