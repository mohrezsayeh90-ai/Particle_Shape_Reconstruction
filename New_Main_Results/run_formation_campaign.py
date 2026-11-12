#!/usr/bin/env python3
"""Resumable CPU campaign. All writes are inside this experiment bundle.

Examples:
  python run_formation_campaign.py --stage prepare
  python run_formation_campaign.py --stage history --resume
  python run_formation_campaign.py --stage smoke --workers 3 --resume
  python run_formation_campaign.py --stage core --workers 4 --resume
  python run_formation_campaign.py --stage extended --workers 4 --resume
  python plot_formation_campaign.py
An optional --limit selects how many pending trials to execute, without changing
the common 24/50/35 optimizer caps. Unexecuted trials remain explicitly pending.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-formation-cache')
import argparse,csv,hashlib,json,sys,time,traceback,platform
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from formation.physics import Physics,BOUNDS,WEIGHT_SHA,L,H
from formation.targets import target_points,make_ports,preflight,ordered_lattice
from formation.planner import Planner,digest
from formation.verification import write_json,write_csv,verify,schedule_rows,controls_from_schedule
from formation.experiments import matrix,priority

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs'/'formation_20261112_cpu'
CHECKPOINT=ROOT/'inputs'/'frozen_checkpoint.pt'


def save_case_inputs(case):
    folder=RUN/'cases'/case['id'];folder.mkdir(parents=True,exist_ok=True)
    points=target_points(case)
    write_csv(folder/'target_points.csv',[dict(particle=i+1,x=x,y=y) for i,(x,y) in enumerate(points)])
    write_csv(folder/'ports.csv',case['ports'],['id','side','x'])
    write_json(folder/'config.json',case)
    return folder,points


def prepare():
    RUN.mkdir(parents=True,exist_ok=True);cases=matrix()
    manifest=RUN/'cases.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if digest(old)!=digest(cases):raise RuntimeError('Existing campaign configuration differs; use a new run directory')
    else:write_json(manifest,cases)
    cfg=dict(run_id=RUN.name,source_revision='c17460d',seeds=[11,29,47],physics=dict(L=L,H=H,Re=10,diameter=.025,density_ratio=1000,launch_gap=1e-5),
             age_bounds=[.03,5.],historical_age_bounds='preserved in archived schedules',duration_lower_bound_harmonized=True,
             acceptance=dict(pinn_endpoint=.003,reference_endpoint=.004,endpoint_difference=.004,min_distance=.026,search_distance=.0263),
             budget=dict(endpoint_nfev=24,coarse_nfev=50,direct_nfev=35),numba_planning_evaluator='same frozen MLP, checked against PyTorch',
             acceptance_evaluators=['PyTorch float32 / RK45','independent analytical Poiseuille / DOP853'],weight_sha256=WEIGHT_SHA,
             worker_cpu_threads=1,optional_layout_G5='not scheduled; optional after core matrix',
             handoff_sha256=hashlib.sha256((ROOT/'EXPERIMENT_HANDOFF.txt').read_bytes()).hexdigest())
    write_json(RUN/'config.json',cfg)
    import scipy,torch,numba,shapely,matplotlib
    write_json(RUN/'runtime.json',dict(python=sys.version,platform=platform.platform(),cpus=os.cpu_count(),gpu=torch.cuda.is_available(),
               numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,numba=numba.__version__,shapely=shapely.__version__,matplotlib=matplotlib.__version__))
    geometry=[]
    for shape in ['stadium','arrow','fish','perforated']:
        points=ordered_lattice(shape)
        write_csv(RUN/'targets'/(shape+'_complete_farthest_order.csv'),[dict(order=i+1,x=x,y=y) for i,(x,y) in enumerate(points)])
        geometry.append(dict(shape=shape,buffered_capacity=len(points),pitch=.027,particle_radius=.0125,buffer=.0004))
    write_csv(RUN/'geometric_capacity.csv',geometry)
    pending=[dict(id='A1_'+s,status='pending',reason='original Kaggle output payload unavailable; current public Kaggle URL denied HTTP 403 by proxy; no Kaggle credential binding',required_action='recover original output package or run separately labelled historical-setting rerun') for s in ['triangle','S','C','T']]
    pending += [dict(id=s,status='pending_prerequisite',reason=r) for s,r in [
        ('G1_schedule_reflection','requires an accepted new schedule for paired numerical symmetry replay'),
        ('G3_largest_accepted_fish_fallback','depends on completed fish density sweep'),
        ('G4_low_margin','select after nominal accepted schedules; 64 draws per declared perturbation scenario'),
        ('G4_high_margin','select after nominal accepted schedules; 64 draws per declared perturbation scenario')]]
    if not (RUN/'conditional_cases.json').exists():write_json(RUN/'conditional_cases.json',pending)
    update_index()
    print('planned_trials',len(cases),'geometric_capacities',geometry,flush=True)


def update_index():
    cases=json.loads((RUN/'cases.json').read_text());rows=[];byhash={}
    for c in cases:
        p=RUN/'cases'/c['id']/'metrics.json'
        if p.exists():byhash[c['configuration_sha256']]=(c['id'],json.loads(p.read_text()))
    for c in cases:
        p=RUN/'cases'/c['id']/'metrics.json';m=json.loads(p.read_text()) if p.exists() else None
        reuse=None
        if m is None and c['configuration_sha256'] in byhash:reuse,m=byhash[c['configuration_sha256']]
        if m is None:
            points=target_points(c);reason,detail=preflight(points,c['N'],c['ports'],c.get('cap'))
            status='excluded_before_solve' if reason else 'pending'
            running=RUN/'cases'/c['id']/'running.json'
            if not reason and running.exists():
                try:
                    os.kill(int(json.loads(running.read_text())['pid']),0);status='running'
                except (ProcessLookupError,KeyError,ValueError):detail='interrupted attempt; resumable'
        else:status=m['status'];reason=m.get('failure_type');detail=m.get('detail')
        ref=(m or {}).get('replays',{}).get('reference',{});pin=(m or {}).get('replays',{}).get('pinn',{})
        minimum=ref.get('minimum',{}).get('distance')
        rows.append(dict(id=c['id'],group=c['group'],stage=c['stage'],shape=c['shape'],N=c['N'],mode=c['mode'],m=c['m'],layout=c['layout'],
                         cap=c.get('cap'),center_x=c['center'][0],center_y=c['center'][1],q=c.get('q'),theta=c.get('theta'),seed=c['seed'],
                         status=status,accepted=(m or {}).get('accepted',False),failure_type=reason,detail=detail,
                         reused_case=reuse,configured_ports=None if c['mode']=='free' else len(c['ports']),used_ports=(m or {}).get('used_ports'),
                         pinn_max_error=pin.get('max_error'),reference_max_error=ref.get('max_error'),reference_min_distance=minimum,
                         clearance_margin=None if minimum is None else minimum-.026,max_endpoint_difference=(m or {}).get('max_endpoint_difference'),
                         target_time=(m or {}).get('target_time'),elapsed_seconds=(m or {}).get('elapsed_seconds'),budget_tier=c['budget_tier'],
                         min_age=c['min_age'],max_age=c['max_age'],minimum_emission_gap=c['min_gap'],collision_aware=c['collision_aware'],
                         L_star=L,H_star=H,Re=10,diameter_star=.025,density_ratio=1000,
                         weight_sha256=WEIGHT_SHA,target_sha256=c['target_sha256'],layout_sha256=c['layout_sha256'],configuration_sha256=c['configuration_sha256']))
    families={}
    keys=['group','shape','N','mode','m','layout','cap','center_x','center_y','q','theta','budget_tier','min_age','max_age','minimum_emission_gap','collision_aware']
    for row in rows:families.setdefault(tuple(row[k] for k in keys),[]).append(row)
    for family in families.values():
        terminal=[r for r in family if r['status'] in ['accepted','rejected']]
        complete=len({r['seed'] for r in terminal})==3
        for row in family:
            row['completed_starts_in_comparison']=len(terminal)
            row['planned_starts_in_comparison']=len(family)
            row['matched_three_seed_complete']=complete
            row['comparison_status']='smoke_validation' if row['group']=='smoke' else 'matched_three_seed_complete' if complete else 'screening_or_incomplete' if terminal else 'unexecuted'
    for source in sorted((ROOT/'inputs'/'historical').iterdir()):
        if not source.is_dir():continue
        file=RUN/'historical_replays'/source.name/'metrics.json';m=json.loads(file.read_text()) if file.exists() else {}
        row={k:None for k in rows[0]};ref=m.get('replays',{}).get('reference',{});pin=m.get('replays',{}).get('pinn',{})
        row.update(id='A2_'+source.name,group='A2',stage='history',shape='duck',N=m.get('N'),mode='free' if 'independent' in source.name else 'fixed',
                   status=m.get('status','pending'),accepted=m.get('accepted',False),failure_type=m.get('failure_type'),
                   pinn_max_error=pin.get('max_error'),reference_max_error=ref.get('max_error'),reference_min_distance=ref.get('minimum',{}).get('distance'),
                   target_time=m.get('target_time'),weight_sha256=WEIGHT_SHA,L_star=L,H_star=H,Re=10,diameter_star=.025,density_ratio=1000,budget_tier='archived_schedule_replay')
        rows.append(row)
    for c in json.loads((RUN/'conditional_cases.json').read_text()):
        row={k:None for k in rows[0]};row.update(id=c['id'],group=c['id'].split('_')[0],stage='conditional',status=c['status'],detail=c.get('reason'))
        rows.append(row)
    write_csv(RUN/'cases_manifest.csv',rows);write_csv(RUN/'summary.csv',rows)
    return rows


_PHYSICS=None
def physics():
    global _PHYSICS
    if _PHYSICS is None:_PHYSICS=Physics(CHECKPOINT)
    return _PHYSICS


def worker(case):
    # The first dense screening batch started before cost vectorization. Keep
    # its scalar backend reproducible; the objective and solve caps are equal.
    case['allocation_backend']='scalar' if case['N']==100 and case['seed']==11 and case['group']=='D1' else 'batched'
    start=time.monotonic();folder,targets=save_case_inputs(case)
    reason,detail=preflight(targets,case['N'],case['ports'],case.get('cap'))
    if reason:
        m=dict(status='excluded_before_solve',accepted=False,failure_type=reason,detail=detail,N=case['N'])
        write_json(folder/'metrics.json',m);return case['id'],m
    print('START',case['id'],case['shape'],case['N'],case['mode'],case['m'],'seed',case['seed'],flush=True)
    write_json(folder/'running.json',dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat()))
    case['progress_path']=str(folder)
    try:
        p=physics();coarse=Planner(p,case,'coarse');direct=Planner(p,case,'direct')
        options=coarse.candidate_options(targets,RUN/'cache')
        z,ports=coarse.choose(options,case['seed'])
        print('ALLOCATED',case['id'],'seconds',round(time.monotonic()-start,2),flush=True)
        z,stage1=coarse.refine(z,ports,targets,case['coarse_nfev'])
        print('COARSE',case['id'],stage1['nfev'],'seconds',round(time.monotonic()-start,2),flush=True)
        z,stage2=direct.refine(z,ports,targets,case['direct_nfev'])
        physical=[direct.physical(row,port) for row,port in zip(z.reshape(-1,direct.dim),ports)]
        controls=np.array([r[0] for r in physical]);sides=[r[1] for r in physical];ages=np.array([r[2] for r in physical])
        ids=[f'free_{side}_{c[3*side]:.14g}' for c,side in zip(controls,sides)] if direct.free else [case['ports'][i]['id'] for i in ports]
        rows=schedule_rows(p,controls,sides,ages,targets,ids)
        np.savez_compressed(folder/'planner_state.npz',z=z,assignments=ports)
        write_json(folder/'optimization.json',[stage1,stage2])
        m=verify(p,rows,targets,folder,case)
        m.update(elapsed_seconds=time.monotonic()-start,allocation_backend=case['allocation_backend'],stages=[{k:v for k,v in x.items() if k!='history'} for x in [stage1,stage2]],
                 optimizer_budget_exhausted=bool(stage1['nfev']>=case['coarse_nfev'] or stage2['nfev']>=case['direct_nfev']),
                 seed=case['seed'],cold_start=True,budget_tier=case['budget_tier'],target_sha256=case['target_sha256'],layout_sha256=case['layout_sha256'])
        write_json(folder/'metrics.json',m)
        print('DONE',case['id'],m['status'],m.get('failure_type'),'seconds',round(m['elapsed_seconds'],2),flush=True)
        return case['id'],m
    except Exception as exc:
        reason='candidate_shortage' if 'candidate_shortage' in str(exc) else 'solver_error'
        m=dict(status='rejected',accepted=False,failure_type=reason,detail=str(exc),elapsed_seconds=time.monotonic()-start,N=case['N'])
        write_json(folder/'metrics.json',m);(folder/'error.log').write_text(traceback.format_exc())
        print('ERROR',case['id'],type(exc).__name__,str(exc),flush=True);return case['id'],m


def history():
    p=physics();records=[]
    for source in sorted((ROOT/'inputs'/'historical').iterdir()):
        if not source.is_dir():continue
        folder=RUN/'historical_replays'/source.name
        if (folder/'metrics.json').exists():continue
        saved=json.loads((source/'saved_result.json').read_text());targets=np.array(saved['targets'])
        rows=[];free='independent' in source.name
        for i,r in enumerate(saved['schedules']):
            r=dict(r);r['side']=0 if r['launcher']=='bottom' else 1
            r['port_id']=f'free_{i+1}' if free else r['station_id'];rows.append(r)
        case=dict(provenance='new_replay_of_archived_schedule',mode='free' if free else 'fixed',ports=make_ports({'m':3}))
        m=verify(p,rows,targets,folder,case);records.append(dict(case=source.name,accepted=m['accepted'],N=m['N'],failure=m['failure_type']))
        write_json(folder/'archived_metrics.json',json.loads((source/'result.json').read_text()))
        print('HISTORICAL',source.name,m['accepted'],m['failure_type'],flush=True)
    write_json(RUN/'historical_replay_index.json',records)


def selfcheck():
    p=physics();checks={'compiled_evaluator':p.evaluator_check()}
    c=dict(mode='free',ports=[],min_age=.03,max_age=5.,collision_aware=True)
    plan=Planner(p,c,'coarse');z=np.array([[.30,.4,.1,.04],[.309,.39,.11,.045],[.6,.5,.15,.055]]).ravel();ports=np.zeros(3,int)
    _,_,target=plan.endpoints(z,ports);jac=plan.jacobian(z,ports,target).toarray()
    numerical=np.empty_like(jac);step=1e-6
    for k in range(len(z)):
        d=np.zeros_like(z);d[k]=step
        numerical[:,k]=(plan.residual(z+d,ports,target)-plan.residual(z-d,ports,target))/(2*step)
    relative=float(np.linalg.norm(jac-numerical)/max(np.linalg.norm(numerical),1e-12))
    checks['sparse_derivative']={'relative_l2':relative,'threshold':.005,'passed':relative<.005,'dense_matrix_used_only_for_small_test':list(jac.shape)}
    mapping=make_ports({'m':3})
    checks['historical_side_mapping']={'passed':all(port['side']==i//3 and abs(port['x']-[.55,1.1,1.65][i%3])<1e-12 for i,port in enumerate(mapping))}
    checks['capacity_preflight']={'passed':preflight(target_points({'shape':'stadium','N':24}),24,mapping,1)[0]=='structural_capacity_infeasibility'}
    c['check_allocation_vectorization']=True
    options=[[dict(row=row.copy(),port=0,error=0.) for row in [r,np.clip(r+np.array([.001,0,0,.002]),0,1)]] for r in z.reshape(-1,4)]
    plan.choose(options,11)
    checks['allocation_vectorization']={'passed':True,'comparison':'batched versus scalar pair costs, tolerance 1e-10'}
    write_json(RUN/'selfcheck.json',checks);print(json.dumps(checks),flush=True)
    if not all(v['passed'] for v in checks.values()):raise RuntimeError('Campaign self-check failed')


def reload_check():
    for folder in sorted((RUN/'cases').glob('*')):
        path=folder/'metrics.json'
        if path.exists() and json.loads(path.read_text()).get('accepted'):
            rows=list(csv.DictReader((folder/'release_schedule.csv').open()))
            for r in rows:
                for k in ['particle','side']:r[k]=int(r[k])
                for k in ['x_launch','y_launch','u_launch','v_launch','age_at_target','release_time','target_x','target_y']:r[k]=float(r[k])
            targets=np.loadtxt(folder/'target_points.csv',delimiter=',',skiprows=1)[:,1:]
            case=json.loads((folder/'config.json').read_text());m=verify(physics(),rows,targets,RUN/'reload_check',case)
            write_json(RUN/'reload_check'/'source.json',dict(case=folder.name,schedule_sha256=hashlib.sha256((folder/'release_schedule.csv').read_bytes()).hexdigest()))
            if not m['accepted']:raise RuntimeError('Saved accepted schedule did not survive reload')
            print('RELOAD_VERIFIED',folder.name,flush=True);return
    raise RuntimeError('No accepted new case available for reload check')


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--stage',choices=['prepare','check','history','smoke','core','extended','all','reload'],default='prepare')
    a.add_argument('--resume',action='store_true');a.add_argument('--workers',type=int,default=1);a.add_argument('--limit',type=int)
    a.add_argument('--group');a.add_argument('--seed',type=int);a.add_argument('--N',type=int)
    args=a.parse_args();prepare()
    if args.stage=='prepare':return
    if args.stage=='check':selfcheck();return
    if args.stage=='history':history();return
    if args.stage=='reload':reload_check();return
    cases=json.loads((RUN/'cases.json').read_text());done=update_index();done={c['id']:c for c in done}
    selected=[];seen=set()
    for c in sorted(cases,key=priority):
        if args.stage!='all' and c['stage']!=args.stage:continue
        if args.group and c['group']!=args.group:continue
        if args.seed is not None and c['seed']!=args.seed:continue
        if args.N is not None and c['N']!=args.N:continue
        if done[c['id']]['status']!='pending':continue
        if c['configuration_sha256'] in seen:continue
        seen.add(c['configuration_sha256']);selected.append(c)
    if args.limit is not None:selected=selected[:args.limit]
    if not args.resume and any((RUN/'cases'/c['id']/'metrics.json').exists() for c in selected):raise RuntimeError('Use --resume or a fresh run')
    print('EXECUTING',len(selected),'trials, workers',args.workers,flush=True)
    if args.workers==1:
        for c in selected:worker(c);update_index()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(worker,c):c['id'] for c in selected}
            for f in as_completed(futures):
                f.result();update_index()
    update_index()


if __name__=='__main__':main()
