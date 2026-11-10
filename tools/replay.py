"""Independent numerical replay of saved releases; no training or optimization.

Reads notebook literals as data, loads tensors with weights_only=True, and
reimplements the documented plane-Poiseuille / Schiller--Naumann equations.
Never executes notebook cells. All lengths/times are nondimensional.
"""
from pathlib import Path
import ast, base64, csv, hashlib, io, itertools, json, time, zlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'verification';OUT.mkdir(exist_ok=True)
L,H,UC,RE,D,RHO,GAP=2.2,.30,1.,10.,.025,1000.,1e-5
BOUNDS=np.array([[0,L],[-.4,1.5],[.15,1.2],[0,L],[-.4,1.5],[-1.2,-.15]])
PS=8*UC*L/(RE*H**2)
torch.set_num_threads(1)

class FrozenFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('lower',torch.tensor([0,0,*BOUNDS[:,0]],dtype=torch.float32))
        self.register_buffer('span',torch.tensor([L,H,*(BOUNDS[:,1]-BOUNDS[:,0])],dtype=torch.float32))
        layers=[nn.Linear(8,64),nn.Tanh()]
        for _ in range(3):layers += [nn.Linear(64,64),nn.Tanh()]
        layers += [nn.Linear(64,3)];self.net=nn.Sequential(*layers)
    def forward(self,z):
        a=self.net(2*(z-self.lower)/self.span-1)
        x,y=z[:,0]/L,z[:,1]/H;p=4*UC*y*(1-y);cap=x*(1-x)
        return torch.stack((p*(1+cap*a[:,0]),p*cap*a[:,1],PS*(1-x)*a[:,2]),dim=1)

def load_data():
    cases=[];checkpoint=None
    for name in ['master3-clean-revised-version','dense-duck-formation-20261104-211d7de258']:
        folder=ROOT/'evidence'/name; nb=json.loads(next(folder.glob('*.ipynb')).read_text())
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            tree=ast.parse(''.join(cell['source']))
            for node in ast.walk(tree):
                if not isinstance(node,ast.Constant) or not isinstance(node.value,str):continue
                s=node.value
                if s.startswith('eN') and len(s)>1000:
                    try:data=json.loads(zlib.decompress(base64.b64decode(s)))
                    except Exception:continue
                    if isinstance(data,dict):
                        for key,value in data.items():
                            if isinstance(value,dict) and 'schedules' in value:
                                cases.append((name,key,value))
                if name.startswith('dense') and s.startswith('UEsDBAAACAg'):
                    checkpoint=base64.b64decode(s)
    assert checkpoint is not None
    (OUT/'original_frozen_checkpoint.pt').write_bytes(checkpoint)
    state=torch.load(io.BytesIO(checkpoint),map_location='cpu',weights_only=True)
    assert state['config']['L']==L and state['config']['H']==H
    assert state['config']['diameter']==D and state['config']['density_ratio']==RHO
    assert state['config']['drag']=='schiller-naumann'
    model=FrozenFlow();model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
    return cases,model,hashlib.sha256(checkpoint).hexdigest()

def field(model,xy,c):
    packed=np.column_stack((np.atleast_2d(xy),np.broadcast_to(c,(len(np.atleast_2d(xy)),6))))
    with torch.inference_mode():return model(torch.tensor(packed,dtype=torch.float32)).numpy()

def integrate(schedule,model=None,max_step=.0175):
    x,y,u,v=[schedule[k] for k in ['x_launch','y_launch','u_launch','v_launch']]
    a=schedule['age_at_target'];side=0 if schedule['launcher']=='bottom' else 1
    controls=BOUNDS.mean(axis=1);controls[3*side:3*side+3]=[x,u,v]
    def rhs(t,q):
        if model is None:
            eta=np.clip(q[1],0,H)/H;uf=np.array([4*UC*eta*(1-eta),0.])
        else:uf=field(model,np.clip(q[:2],[0,0],[L,H]),controls)[0,:2]
        slip=uf-q[2:];rep=RE*D*np.linalg.norm(slip)
        factor=1+.15*rep**.687 if rep<1000 else .44*rep/24
        acceleration=18*factor/(RHO*RE*D**2)*slip
        return [q[2],q[3],*acceleration]
    events=[lambda t,q:q[1]-D/2,lambda t,q:H-D/2-q[1],lambda t,q:q[0],lambda t,q:L-q[0]]
    for e in events:e.terminal=True;e.direction=-1
    sol=solve_ivp(rhs,[0,a],[x,y,u,v],method='DOP853' if model is None else 'RK45',
                  rtol=1e-10 if model is None else 2e-8,atol=1e-12 if model is None else 2e-10,
                  max_step=max_step,dense_output=True,events=events)
    assert sol.success and sol.t[-1]>=a-1e-8,'A particle exits/hits a wall before target time'
    return sol

def pair_minimum(paths,ages,samples):
    result=float('inf');who=None
    for i in range(len(paths)):
        for j in range(i):
            duration=min(ages[i],ages[j]);grid=np.linspace(0,duration,samples)
            def dist(b):
                b=np.atleast_1d(b)
                return np.linalg.norm(paths[i].sol(ages[i]-b)[:2]-paths[j].sol(ages[j]-b)[:2],axis=0)
            values=dist(grid);minimum=float(values.min())
            local=np.flatnonzero((values[1:-1]<=values[:-2])&(values[1:-1]<=values[2:]))+1
            for k in local:
                opt=minimize_scalar(lambda t:float(dist(t)[0]),bounds=(grid[k-1],grid[k+1]),method='bounded',options={'xatol':1e-12})
                minimum=min(minimum,float(opt.fun))
            if minimum<result:result=minimum;who=[j+1,i+1]
    return result,who

def verify_case(name,key,r,model):
    begin=time.monotonic();schedules=r['schedules'];n=len(schedules);ages=np.array([s['age_at_target'] for s in schedules])
    targets=np.array(r['targets']);actual=np.array(r['actual']);T=r['target_time']
    assert len(targets)==n==len(actual)
    for s in schedules:
        assert s['release_time']>=0 and abs(s['release_time']+s['age_at_target']-T)<1e-10
        side=0 if s['launcher']=='bottom' else 1; expected=D/2+GAP if side==0 else H-D/2-GAP
        assert abs(s['y_launch']-expected)<1e-10
        c=np.array([s['x_launch'],s['u_launch'],s['v_launch']]);b=BOUNDS[3*side:3*side+3]
        assert np.all(c>=b[:,0]-1e-10) and np.all(c<=b[:,1]+1e-10)
    if 'six_fixed' in key:
        assert len(set(s['station_id'] for s in schedules))==6
        assert all(min(abs(s['x_launch']-v) for v in [.25*L,.5*L,.75*L])<1e-12 for s in schedules)
    references=[integrate(s,max_step=.0125) for s in schedules]
    neural=[integrate(s,model=model) for s in schedules]
    ref_end=np.array([p.y[:2,-1] for p in references]);net_end=np.array([p.y[:2,-1] for p in neural])
    ref_error=float(np.linalg.norm(ref_end-targets,axis=1).max());net_error=float(np.linalg.norm(net_end-targets,axis=1).max())
    ref_delta=float(np.linalg.norm(ref_end-np.array(r['analytical_reference']),axis=1).max())
    net_delta=float(np.linalg.norm(net_end-actual,axis=1).max())
    reference_separation,pair=pair_minimum(references,ages,1601)
    coarse_separation,_=pair_minimum(references,ages,401)
    neural_separation,netpair=pair_minimum(neural,ages,801)
    # Re-run the nearest reference pair at a smaller step size to probe numerical sensitivity.
    fine=[integrate(schedules[i-1],max_step=.00625) for i in pair]
    step_delta=max(float(np.linalg.norm(f.y[:2,-1]-references[i-1].y[:2,-1])) for f,i in zip(fine,pair))
    accepted=bool(ref_error<=.004 and net_error<=.003 and np.max(np.linalg.norm(ref_end-net_end,axis=1))<=.004
                  and reference_separation>=D+.001 and neural_separation>=D+.001)
    # A deliberately shifted launch clock must not accidentally still describe a simultaneous target snapshot.
    shifted_consistency=abs((schedules[0]['release_time']+.1)+ages[0]-T)<1e-10
    assert not shifted_consistency
    result=dict(source=name,case=key,particles=n,accepted=accepted,
                max_reference_target_error=ref_error,max_replayed_PINN_target_error=net_error,
                max_reference_saved_discrepancy=ref_delta,max_PINN_saved_discrepancy=net_delta,
                reference_minimum_separation=reference_separation,reference_nearest_pair=pair,
                PINN_minimum_separation=neural_separation,PINN_nearest_pair=netpair,
                required_separation=D+.001,separation_401_to_1601_delta=abs(reference_separation-coarse_separation),
                nearest_pair_half_step_endpoint_delta=step_delta,clock_shift_negative_control_rejected=True,
                elapsed_seconds=time.monotonic()-begin)
    folder=OUT/key;folder.mkdir(exist_ok=True)
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    (folder/'saved_result.json').write_text(json.dumps(r))
    with (folder/'release_schedule.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(schedules[0]));writer.writeheader();writer.writerows(schedules)
    # Trajectories are freshly computed, not copied from cached rendering arrays.
    grid=np.linspace(0,T,501);xy=np.full((n,len(grid),2),np.nan)
    for i,(p,s) in enumerate(zip(references,schedules)):
        present=grid>=s['release_time'];xy[i,present]=p.sol(grid[present]-s['release_time'])[:2].T
    np.savez_compressed(folder/'reference_trajectories.npz',time=grid,positions=xy,targets=targets,actual_PINN=net_end,reference=ref_end)
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    cases,model,sha=load_data();reports=[]
    for name,key,r in cases:reports.append(verify_case(name,key,r,model))
    summary=dict(checkpoint_sha256=sha,scope='Independent frozen-weight / analytical ODE replay; no training, no new inverse optimization, no experimental validation',cases=reports,all_accepted=all(r['accepted'] for r in reports))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    assert summary['all_accepted'],'At least one release replay fails its original acceptance thresholds'
