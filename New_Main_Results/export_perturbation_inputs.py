#!/usr/bin/env python3
"""Reconstruct seeded perturbation inputs and spot-replay saved scenarios.

Outputs explicitly identify reconstructed parameters; stored measured trial
metrics are never synthesized or replaced. All 64 bounds flags are checked;
the first valid draw in each scenario is independently replayed again.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,json
import numpy as np
from formation.physics import BOUNDS
from formation.verification import controls_from_schedule,pair_minimum,write_json
from run_formation_campaign import ROOT,RUN,physics


def read(path):return json.loads(path.read_text())


def main():
    audit=[]
    kinds=['nominal','position','axial_velocity','transverse_velocity','timing','joint','port_misplacement']
    for file in sorted((RUN/'robustness').glob('*/metrics.json')):
        m=read(file);source=RUN/'cases'/m['caseid'];case=read(source/'config.json')
        rows=list(csv.DictReader((source/'release_schedule.csv').open()));trials=list(csv.DictReader((file.parent/'trials.csv').open()))
        controls=controls_from_schedule(rows);releases=np.array([float(r['release_time']) for r in rows]);sides=np.array([int(r['side']) for r in rows]);ports=[r['port_id'] for r in rows]
        kind=m['scenario'];level=m['level'];T=m['fixed_nominal_target_time'];std=m['standard_deviations']
        seed=39011+level*100+kinds.index(kind)*7+(0 if m['selection_label']=='low_margin' else 1000)
        rng=np.random.default_rng(seed);all_controls=[];all_release=[];all_ages=[];valid=[]
        for i in range(64):
            c=controls.copy();release=releases.copy();n=len(c)
            if kind in ['position','joint','port_misplacement']:
                if kind=='port_misplacement':
                    offsets={p:rng.normal(0,std['x']) for p in sorted(set(ports))};noise=np.array([offsets[p] for p in ports])
                else:noise=rng.normal(0,std['x'],n)
                c[np.arange(n),3*sides]+=noise
            if kind in ['axial_velocity','joint']:c[np.arange(n),3*sides+1]+=rng.normal(0,std['u'],n)
            if kind in ['transverse_velocity','joint']:c[np.arange(n),3*sides+2]+=rng.normal(0,std['v'],n)
            if kind in ['timing','joint']:release+=rng.normal(0,std['time'],n)
            ages=T-release
            ok=bool(np.all(c>=BOUNDS[:,0]) and np.all(c<=BOUNDS[:,1]) and np.all(release>=0) and np.all(ages>=case['min_age']) and np.all(ages<=case['max_age']))
            all_controls.append(c);all_release.append(release);all_ages.append(ages);valid.append(ok)
        assert valid==[r['within_bounds']=='True' for r in trials],f'Bounds flags differ in {file.parent.name}'
        assert sum(r['accepted']=='True' for r in trials)==m['accepted']
        np.savez_compressed(file.parent/'perturbation_inputs.npz',controls=np.array(all_controls),release_times=np.array(all_release),ages=np.array(all_ages),sides=sides,within_bounds=valid,target_time=T)
        write_json(file.parent/'perturbation_provenance.json',dict(seed=seed,generator='numpy.random.default_rng / PCG64',draws=64,
                   provenance='parameters deterministically reconstructed after the campaign from the exact saved runner logic; trial result metrics are original computed outputs',
                   bounds_flags_match_saved_trials=True,source_case=m['caseid'],generator_order=kinds,shared_port_order='sorted port IDs',numpy=np.__version__))
        row=dict(scenario=file.parent.name,bounds_flags_checked=64,passed=True,draw_replayed=None)
        if any(valid):
            i=valid.index(True);paths=[physics().trace(c,int(s),a,'reference') for c,s,a in zip(all_controls[i],sides,all_ages[i])]
            targets=np.loadtxt(source/'target_points.csv',delimiter=',',skiprows=1)[:,1:]
            error=float(np.linalg.norm(np.array([p.y[:2,-1] for p in paths])-targets,axis=1).max())
            complete=all(p.valid for p in paths);nearest=pair_minimum(paths,all_ages[i],target_time=T) if complete else {'distance':None}
            stored=trials[i];error_difference=abs(error-float(stored['max_endpoint_error']))
            distance_difference=abs(nearest['distance']-float(stored['minimum_distance'])) if nearest['distance'] is not None else None
            passed=error_difference<1e-10 and (distance_difference is None or distance_difference<1e-10)
            row.update(draw_replayed=i,max_endpoint_error_difference=error_difference,minimum_distance_difference=distance_difference,passed=passed)
        audit.append(row)
    report=dict(passed=all(r['passed'] for r in audit),completed_scenarios=len(audit),checks=audit,
                interpretation='Parameter reconstruction and one repeated reference replay per scenario where at least one input draw was within bounds.')
    write_json(RUN/'robustness/perturbation_replay_audit.json',report)
    if not report['passed']:raise RuntimeError('Seeded perturbation replay mismatch')
    print('PERTURBATION_REPLAY_AUDIT',report['passed'],len(audit),'scenarios',flush=True)


if __name__=='__main__':main()
