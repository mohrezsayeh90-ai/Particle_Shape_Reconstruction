#!/usr/bin/env python3
"""Save critical-pair events and a paired reflection replay; never optimize."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,json
import numpy as np
from formation.physics import H
from formation.verification import controls_from_schedule,verify,write_json
from run_formation_campaign import ROOT,RUN,physics


def read(p):return json.loads(p.read_text())


def critical_event(folder):
    m=read(folder/'metrics.json');rows=list(csv.DictReader((folder/'release_schedule.csv').open()))
    controls=controls_from_schedule(rows);ages=np.array([float(r['age_at_target']) for r in rows])
    releases=np.array([float(r['release_time']) for r in rows]);sides=[int(r['side']) for r in rows]
    arrays={};events={}
    for mode in ['reference','pinn']:
        near=m['replays'][mode]['minimum']
        if near['pair'] is None:continue
        time=near['time'];pair=np.array(near['pair'])-1
        shared_start=max(releases[pair]);delta=min(.01,max(1e-4,(m['target_time']-shared_start)/20))
        times=np.array([max(shared_start,time-delta),time,min(m['target_time'],time+delta)])
        paths=[physics().trace(c,s,a,mode) for c,s,a in zip(controls,sides,ages)]
        points=np.full((3,len(rows),2),np.nan)
        for j,t in enumerate(times):
            for i,p in enumerate(paths):
                local=t-releases[i]
                if 0<=local<=p.end+1e-10:points[j,i]=p.sol(min(local,p.end))[:2]
        arrays[mode+'_times']=times;arrays[mode+'_positions']=points;arrays[mode+'_pair']=pair+1
        events[mode]=dict(refined_time=time,distance=near['distance'],pair=near['pair'],
                          frame_times=times.tolist(),minimum_at_window_end=abs(time-m['target_time'])<1e-8,
                          minimum_at_shared_start=abs(time-shared_start)<1e-8,
                          note='Frames are restricted to the shared physical lifetime; boundary minima have no extrapolated before/after frame.')
    np.savez_compressed(folder/'critical_event_snapshots.npz',**arrays)
    write_json(folder/'critical_event.json',events)


def reflection():
    folder=RUN/'reflection'
    if (folder/'comparison.json').exists():return
    candidates=[]
    for file in (RUN/'cases').glob('*/metrics.json'):
        m=read(file)
        if m.get('accepted'):candidates.append((m['N'],file.parent))
    if not candidates:return
    source=max(candidates,key=lambda v:(v[0],v[1].name))[1]
    case=read(source/'config.json');nominal=read(source/'metrics.json')
    rows=list(csv.DictReader((source/'release_schedule.csv').open()))
    for r in rows:
        r['particle']=int(r['particle']);r['side']=1-int(r['side']);r['launcher']='bottom' if r['side']==0 else 'top'
        for key in ['x_launch','y_launch','u_launch','v_launch','age_at_target','release_time','target_x','target_y']:r[key]=float(r[key])
        r['y_launch']=H-r['y_launch'];r['v_launch']=-r['v_launch'];r['target_y']=H-r['target_y']
        r['port_id']='mirror_'+str(r['port_id'])
    case['ports']=[dict(id='mirror_'+str(p['id']),side=1-p['side'],x=p['x']) for p in case['ports']]
    case['provenance']='reflected_saved_accepted_schedule';case['center'][1]=H-case['center'][1]
    targets=np.array([[r['target_x'],r['target_y']] for r in rows])
    result=verify(physics(),rows,targets,folder,case)
    data=np.load(folder/'trajectories.npz');base=np.load(source/'trajectories.npz')
    comparisons={}
    for mode in ['reference','pinn']:
        expected=base[mode+'_endpoints'].copy();expected[:,1]=H-expected[:,1]
        comparisons[mode]=dict(max_endpoint_reflection_residual=float(np.linalg.norm(data[mode+'_endpoints']-expected,axis=1).max()),
                               signed_minimum_distance_change=result['replays'][mode]['minimum']['distance']-nominal['replays'][mode]['minimum']['distance'])
    write_json(folder/'config.json',case)
    write_json(folder/'comparison.json',dict(source_case=source.name,N=nominal['N'],accepted=result['accepted'],
               paired_reference_symmetry=comparisons['reference'],frozen_PINN_asymmetry=comparisons['pinn'],
               interpretation='Same saved schedule reflected, with no optimizer rerun; PINN asymmetry is reported separately.'))


def main():
    folders=[RUN/'historical_replays'/name for name in ['dense_duck_six_fixed','dense_duck_independent_wall_ports']]
    for shape in ['stadium','arrow','fish','perforated']:
        candidates=[]
        for file in (RUN/'cases').glob('*/metrics.json'):
            c=read(file.parent/'config.json');m=read(file)
            if c['shape']==shape and m.get('accepted'):candidates.append((c['N'],file.parent))
        if candidates:folders.append(max(candidates,key=lambda v:(v[0],v[1].name))[1])
    for file in (RUN/'cases').glob('*/metrics.json'):
        case=read(file.parent/'config.json')
        if case['group']=='G2' and 'replays' in read(file):folders.append(file.parent)
    for folder in folders:
        if (folder/'metrics.json').exists() and not (folder/'critical_event.json').exists():
            critical_event(folder);print('CRITICAL_EVENT',folder.name,flush=True)
    reflection()


if __name__=='__main__':main()
