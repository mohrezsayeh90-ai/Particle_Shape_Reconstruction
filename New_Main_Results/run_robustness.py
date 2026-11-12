#!/usr/bin/env python3
"""Seeded numerical release sensitivity with fixed nominal schedule and T*.

Uses independent reference-only replays with the reference endpoint threshold
0.004; reports unconditional and bounds-conditional fractions separately.
No invalid draw is clipped. These are sensitivity scenarios, not reliability
measurements. --limit limits scenarios, never the declared 64 draws per scenario.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from formation.physics import BOUNDS
from formation.verification import controls_from_schedule,pair_minimum,write_csv,write_json
from run_formation_campaign import ROOT,RUN,physics,update_index


def selection():
    path=RUN/'robustness'/'selection.json'
    if path.exists():return json.loads(path.read_text())
    candidates=[]
    for file in (RUN/'cases').glob('*/metrics.json'):
        m=json.loads(file.read_text())
        if m.get('accepted') and m.get('N',0)<=24 and m.get('replays',{}).get('reference',{}).get('minimum',{}).get('distance') is not None:
            candidates.append((m['replays']['reference']['minimum']['distance']-.026,file.parent.name))
    if len(candidates)<2:raise RuntimeError('Need at least two accepted nominal schedules')
    candidates.sort();chosen={'low_margin':candidates[0][1],'high_margin':candidates[-1][1],
                              'rule':'minimum and maximum refined reference margin among completed accepted new cases with N<=24 at selection time; representative CPU subset, not an N=100 robustness claim'}
    write_json(path,chosen);return chosen


def run_scenario(label,caseid,kind,level):
    output=RUN/'robustness'/f'{label}_{kind}_level{level}'
    if (output/'metrics.json').exists():return
    source=RUN/'cases'/caseid;case=json.loads((source/'config.json').read_text());nominal=json.loads((source/'metrics.json').read_text())
    assert nominal['accepted'] and nominal['max_endpoint_difference']<=.004
    rows=list(csv.DictReader((source/'release_schedule.csv').open()));controls=controls_from_schedule(rows)
    sides=np.array([int(r['side']) for r in rows]);releases=np.array([float(r['release_time']) for r in rows]);T=nominal['target_time']
    targets=np.loadtxt(source/'target_points.csv',delimiter=',',skiprows=1)[:,1:]
    ports=[r['port_id'] for r in rows];rng=np.random.default_rng(39011+level*100+['nominal','position','axial_velocity','transverse_velocity','timing','joint','port_misplacement'].index(kind)*7+(0 if label=='low_margin' else 1000))
    stds={'x':[0,.025*.025,.05*.025][level],'u':[0,.01,.025][level],
          'v':[0,.01,.025][level],'time':[0,.005,.015][level]}
    trials=[];start=time.monotonic()
    for draw in range(64):
        c=controls.copy();release=releases.copy();n=len(c)
        if kind in ['position','joint','port_misplacement']:
            if kind=='port_misplacement':
                offsets={p:rng.normal(0,stds['x']) for p in sorted(set(ports))};noise=np.array([offsets[p] for p in ports])
            else:noise=rng.normal(0,stds['x'],n)
            c[np.arange(n),3*sides]+=noise
        if kind in ['axial_velocity','joint']:c[np.arange(n),3*sides+1]+=rng.normal(0,stds['u'],n)
        if kind in ['transverse_velocity','joint']:c[np.arange(n),3*sides+2]+=rng.normal(0,stds['v'],n)
        if kind in ['timing','joint']:release+=rng.normal(0,stds['time'],n)
        ages=T-release
        in_bounds=bool(np.all(c>=BOUNDS[:,0]) and np.all(c<=BOUNDS[:,1]) and np.all(release>=0) and np.all(ages>=case.get('min_age',.03)) and np.all(ages<=case.get('max_age',5.)))
        row=dict(draw=draw,within_bounds=in_bounds,accepted=False,reason='invalid_draw',max_endpoint_error=None,minimum_distance=None)
        if in_bounds:
            paths=[physics().trace(cc,int(side),a,'reference') for cc,side,a in zip(c,sides,ages)]
            complete=all(p.valid for p in paths);end=np.array([p.y[:2,-1] for p in paths]);error=float(np.linalg.norm(end-targets,axis=1).max())
            near=pair_minimum(paths,ages,target_time=T) if complete else {'distance':None}
            passed=complete and error<=.004 and near['distance']>=.026
            row.update(accepted=bool(passed),reason=None if passed else 'invalid_boundary_event' if not complete else 'endpoint_error' if error>.004 else 'clearance_failure',
                       max_endpoint_error=error,minimum_distance=near['distance'])
        trials.append(row)
        if (draw+1)%16==0:print('ROBUSTNESS',label,kind,level,draw+1,'/64',flush=True)
    valid=sum(r['within_bounds'] for r in trials);passed=sum(r['accepted'] for r in trials)
    from collections import Counter
    metrics=dict(caseid=caseid,selection_label=label,scenario=kind,level=level,standard_deviations=stds,draws=64,
                 solver='independent analytical Poiseuille / DOP853',threshold=.004,fixed_nominal_target_time=T,within_bounds=valid,
                 accepted=passed,unconditional_pass_fraction=passed/64,bounds_conditional_pass_fraction=passed/valid if valid else None,
                 failure_counts=dict(Counter(r['reason'] for r in trials if r['reason'])),elapsed_seconds=time.monotonic()-start,
                 interpretation='numerical sensitivity only; not measured hardware reliability')
    write_csv(output/'trials.csv',trials);write_json(output/'metrics.json',metrics)
    print('ROBUSTNESS_DONE',label,kind,level,passed,'/64',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--limit',type=int);args=p.parse_args()
    selected=selection();scenarios=[]
    for label in ['low_margin','high_margin']:
        scenarios.append((label,selected[label],'nominal',0))
        for kind in ['position','axial_velocity','transverse_velocity','timing','joint','port_misplacement']:
            for level in [1,2]:scenarios.append((label,selected[label],kind,level))
    write_json(RUN/'robustness'/'scenarios_manifest.json',[dict(label=a,caseid=b,scenario=c,level=d,
                  status='completed' if (RUN/'robustness'/f'{a}_{c}_level{d}'/'metrics.json').exists() else 'pending',draws=64) for a,b,c,d in scenarios])
    pending=[s for s in scenarios if not (RUN/'robustness'/f'{s[0]}_{s[2]}_level{s[3]}'/'metrics.json').exists()]
    if args.limit is not None:pending=pending[:args.limit]
    for s in pending:run_scenario(*s)
    write_json(RUN/'robustness'/'scenarios_manifest.json',[dict(label=a,caseid=b,scenario=c,level=d,
                  status='completed' if (RUN/'robustness'/f'{a}_{c}_level{d}'/'metrics.json').exists() else 'pending',draws=64) for a,b,c,d in scenarios])
    results=[json.loads(p.read_text()) for p in (RUN/'robustness').glob('*/metrics.json')]
    write_csv(RUN/'robustness'/'summary.csv',[{k:v for k,v in r.items() if k not in ['standard_deviations','failure_counts']} for r in results])


if __name__=='__main__':main()
