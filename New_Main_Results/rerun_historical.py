#!/usr/bin/env python3
"""Execute selected original planner definitions, never notebook training cells.

Fresh reruns are labelled separately from unavailable original Kaggle payloads.
Original 3-start triangle and six-station S/C/T settings are preserved. The
original source may produce failed finite-search cases; these are retained.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-formation-cache')
from pathlib import Path
import argparse,ast,hashlib,json,time,itertools,csv,shutil,base64,zlib,math,copy
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares,brentq,minimize_scalar,linear_sum_assignment
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import HTML,FileLink,display
from formation.physics import Physics,BOUNDS,MID,L,H
from formation.verification import write_json,verify
from run_formation_campaign import ROOT,RUN


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['triangle','letters'],default='triangle');args=parser.parse_args()
    folder=RUN/'historical_reruns'/args.case
    if folder.exists():
        if (folder/'original_result.json').exists() or (folder/'rerun_index.json').exists():raise RuntimeError('Completed rerun exists; refusing overwrite')
        folder=folder.with_name(folder.name+'_'+str(time.time_ns()))
    folder.mkdir(parents=True);p=Physics(ROOT/'inputs/frozen_checkpoint.pt')
    constants=json.loads((ROOT/'inputs/master3_literal_constants.json').read_text())
    env=globals().copy();env.update(constants)
    frozen_config=torch.load(ROOT/'inputs/frozen_checkpoint.pt',map_location='cpu',weights_only=True)['config']
    env.update(dict(RUN=folder,model=p.model,CONFIG=frozen_config,
                    U_C=1.,RE=10.,DIAMETER=.025,DENSITY_RATIO=1000.,LAUNCH_GAP=1e-5,DRAG_LAW='schiller-naumann',
                    CONTROL_BOUNDS=BOUNDS,CONTROL_MID=MID,CONTROL_SPAN=BOUNDS[:,1]-BOUNDS[:,0],
                    DTYPE=torch.float32,DEVICE='cpu',PRESSURE_SCALE=8*L/(10*H**2),
                    TARGET_CENTER=np.array([.86*L,.5*H]),AGE_MAX=5.,COARSE_MAX_NFEV=50,DIRECT_MAX_NFEV=35,
                    ODE_RTOL=2e-7,ODE_ATOL=2e-9,TRAJECTORY_MAX_STEP=.035,
                    VALIDATION=json.loads((ROOT/'inputs/archived_flow_validation.json').read_text()),
                    FORMATION_RESULTS=[],SEED_U_SPEEDS=(.25,.70,1.20),SEED_V_SPEEDS=np.linspace(.15,1.20,15)))
    assert env['VALIDATION']['passed']
    definitions=ROOT/'inputs/master3_selected_definitions.py'
    exec(compile(definitions.read_text(),str(definitions),'exec'),env)
    env['direct_field']=lambda xy,c:env['evaluate_flow'](p.model,xy,c)[:,:2]
    write_json(folder/'config.json',env['CONFIG'])
    write_json(folder/'rerun_provenance.json',dict(provenance='new_historical_settings_rerun',original_output_payload='unavailable_proxy_403',
                source_revision='c17460d',definitions_sha256=hashlib.sha256(definitions.read_bytes()).hexdigest(),
                checkpoint_sha256=p.sha,original_literal_constants=constants,requested_case=args.case,training_steps=0))
    if args.case=='triangle':
        lookup=env['FlowLookup'](p.model)
        coarse=env['IndependentLauncherPlanner'](lookup,'original lookup')
        direct=env['IndependentLauncherPlanner'](env['direct_field'],'original direct PINN',rtol=env['ODE_RTOL'],atol=env['ODE_ATOL'],max_step=.035)
        bank=env['independent_seed_bank'](coarse)
        result=env['plan_formation']('triangle',coarse,direct,bank)
        write_json(folder/'original_result.json',result)
        if 'schedules' in result:
            rows=result['schedules']
            for i,r in enumerate(rows):r.update(port_id=f'free_{i+1}',side=0 if r['launcher']=='bottom' else 1)
            verify(p,rows,np.array(result['targets']),folder/'fresh_verification',dict(mode='free',provenance='new_historical_settings_rerun'))
    else:
        settings=ROOT/'inputs/master3_six_station_settings.py'
        exec(compile(settings.read_text(),str(settings),'exec'),env)
        result=env['_run_six_fixed_station_task'](env['SIX_STATION_SETTINGS'])
        write_json(folder/'rerun_index.json',dict(run=str(result['run'].relative_to(ROOT)),settings=env['SIX_STATION_SETTINGS']))
        for item in result['results']:
            if not item.get('schedules'):continue
            rows=item['schedules']
            for r in rows:r.update(port_id=r['station_id'],side=0 if r['launcher']=='bottom' else 1)
            from formation.targets import make_ports
            verify(p,rows,np.array(item['targets']),folder/'fresh_verification'/item['kind'],
                   dict(mode='fixed',ports=make_ports({'m':3}),provenance='new_historical_settings_rerun',max_age=8.))


if __name__=='__main__':main()
