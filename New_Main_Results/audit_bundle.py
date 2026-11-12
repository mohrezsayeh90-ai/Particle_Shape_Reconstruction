#!/usr/bin/env python3
"""Audit saved numerical artifacts and figure provenance, without planning."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,json,hashlib
from pathlib import Path
import numpy as np
from formation.targets import target_points,make_ports
from formation.verification import write_json
from formation.physics import BOUNDS,WEIGHT_SHA
from run_formation_campaign import ROOT,RUN


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    failures=[];accepted=0;checked=0
    cases=read(RUN/'cases.json');index={c['id']:c for c in cases}
    for file in sorted((RUN/'cases').glob('*/metrics.json')):
        case=index[file.parent.name];m=read(file);checked+=1
        if m.get('failure_type')=='solver_error':failures.append(f'{file.parent.name}: programming/solver error must be resolved or explicitly pending')
        if not m.get('accepted'):continue
        accepted+=1
        for name in ['release_schedule.csv','target_points.csv','ports.csv','optimization.json','endpoint_errors.csv','trajectories.npz']:
            if not (file.parent/name).exists():failures.append(f'{file.parent.name}: missing {name}')
        rows=list(csv.DictReader((file.parent/'release_schedule.csv').open()));data=np.load(file.parent/'trajectories.npz')
        controls=np.array([[float(r['x_launch']),float(r['u_launch']),float(r['v_launch'])] for r in rows])
        sides=np.array([int(r['side']) for r in rows]);ages=np.array([float(r['age_at_target']) for r in rows]);release=np.array([float(r['release_time']) for r in rows])
        valid=len(rows)==case['N'] and np.allclose(data['targets'],target_points(case),rtol=0,atol=1e-14)
        for side in (0,1):valid=valid and np.all(controls[sides==side]>=BOUNDS[3*side:3*side+3,0]) and np.all(controls[sides==side]<=BOUNDS[3*side:3*side+3,1])
        valid=valid and np.all(ages>=case['min_age']-1e-10) and np.all(ages<=case['max_age']+1e-10) and np.all(release>=0) and np.allclose(ages+release,m['target_time'],rtol=0,atol=1e-10)
        if case['mode']=='fixed':
            ports={p['id']:p for p in case['ports']}
            valid=valid and all(r['port_id'] in ports and int(r['side'])==ports[r['port_id']]['side'] and abs(float(r['x_launch'])-ports[r['port_id']]['x'])<1e-12 for r in rows)
        if case['cap'] is not None:valid=valid and max(sum(r['port_id']==p for r in rows) for p in {r['port_id'] for r in rows})<=case['cap']
        for mode,tolerance in [('pinn',.003),('reference',.004)]:
            errors=np.linalg.norm(data[mode+'_endpoints']-data['targets'],axis=1)
            valid=valid and errors.max()<=tolerance and abs(errors.max()-m['replays'][mode]['max_error'])<1e-12 and m['replays'][mode]['completed'] and m['replays'][mode]['minimum']['distance']>=.026
            valid=valid and m['replays'][mode]['half_step_check'] is not None
        valid=valid and np.linalg.norm(data['pinn_endpoints']-data['reference_endpoints'],axis=1).max()<=.004 and m['weight_sha256']==WEIGHT_SHA
        if not valid:failures.append(f'{file.parent.name}: saved schedule/endpoint acceptance inconsistency')
    summary=list(csv.DictReader((RUN/'summary.csv').open()))
    if not set(index).issubset({r['id'] for r in summary}):failures.append('Some predeclared matrix IDs are missing from merged summary')
    plotted=list(csv.DictReader((RUN/'figure_inputs/summary.csv').open()))
    terminal=lambda rows:{r['id'] for r in rows if r['id'] in index and r['status'] in ['accepted','rejected']}
    if terminal(summary)!=terminal(plotted):failures.append('Summary changed during plotting; refresh figures after completed case writes settle')
    aggregate=list(csv.DictReader((RUN/'tables/aggregate_comparisons.csv').open()))
    if sum(int(r['executed_starts']) for r in aggregate)!=len(terminal(plotted)):failures.append('Aggregate table executed totals differ from plotted summary')
    if any(int(r['executed_starts'])>3 for r in aggregate):failures.append('Different scientific settings were pooled into a three-seed comparison')
    figures=read(RUN/'figures_manifest.json');checked_figures=0
    for f in figures:
        if f['status']!='rendered':continue
        checked_figures+=1
        for row in f['inputs']:
            path=ROOT/row['path']
            if not path.exists() or sha(path)!=row['sha256']:failures.append(f"{f['id']}: changed/missing figure input {row['path']}")
        for extension in ['pdf','png']:
            if sha(ROOT/f[extension])!=f[extension+'_sha256']:failures.append(f"{f['id']}: invalid {extension} checksum")
    if sha(ROOT/'inputs/frozen_checkpoint.pt')!=WEIGHT_SHA:failures.append('Frozen checkpoint checksum changed')
    checks=read(RUN/'selfcheck.json')
    if not all(c['passed'] for c in checks.values()):failures.append('Numerical selfcheck failed')
    report=dict(passed=not failures,actual_new_attempts_checked=checked,accepted_schedules_checked=accepted,
                rendered_figures_checked=checked_figures,predeclared_matrix_rows=len(cases),failures=failures,
                scope='Saved-artifact consistency, not an additional optimizer run or a mathematical proof of collision freedom.')
    write_json(RUN/'audit_report.json',report);print(json.dumps(report),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
