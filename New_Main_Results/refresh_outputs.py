#!/usr/bin/env python3
"""Refresh indexes and replay historical rerun outputs saved before a UI error."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json
from collections import Counter
import numpy as np
from formation.targets import make_ports
from formation.verification import verify,write_json,write_csv
from run_formation_campaign import ROOT,RUN,physics,update_index


def read(path):return json.loads(path.read_text())


def refresh():
    # Some original notebook helpers wrote result.json before their dashboard
    # wrapper failed. Those real schedules are still usable for fresh replay.
    for file in sorted((RUN/'historical_reruns').glob('letters*/six_fixed_stations_*/formations/*/result.json')):
        result=read(file);output=file.parents[3]/'fresh_verification'/result['kind']
        if (output/'metrics.json').exists() or not result.get('schedules'):continue
        rows=result['schedules']
        for r in rows:r.update(port_id=r['station_id'],side=0 if r['launcher']=='bottom' else 1)
        verify(physics(),rows,np.array(result['targets']),output,
               dict(mode='fixed',ports=make_ports({'m':3}),max_age=8.,provenance='new_historical_settings_rerun'))
        write_json(output/'source.json',dict(original_result=str(file.relative_to(ROOT)),
                   note='New numerical historical-settings rerun. Original output payload unavailable; not recovered historical evidence.'))
    conditional=read(RUN/'conditional_cases.json')
    for item in conditional:
        if item['id'].startswith('A1_'):
            kind=item['id'][3:]
            pattern='triangle*/fresh_verification/metrics.json' if kind=='triangle' else f'letters*/fresh_verification/{kind}*/metrics.json'
            files=sorted((RUN/'historical_reruns').glob(pattern))
            if files:
                m=read(files[-1]);item.update(status='rerun_accepted' if m['accepted'] else 'rerun_rejected',
                    reason='Fresh original-settings rerun independently verified; original Kaggle payload remains unavailable.',
                    metrics=str(files[-1].relative_to(ROOT)),accepted=m['accepted'])
        elif item['id']=='G1_schedule_reflection' and (RUN/'reflection/comparison.json').exists():
            item.update(status='completed',reason='Saved accepted schedule reflected and independently replayed; see reflection/comparison.json')
        elif item['id'].startswith('G4_'):
            label=item['id'][3:];files=list((RUN/'robustness').glob(label+'_*/metrics.json'))
            item.update(status='completed' if len(files)==13 else 'partial' if files else 'pending',
                        reason=f'{len(files)}/13 declared perturbation scenarios completed, 64 draws each; see robustness/scenarios_manifest.json')
    write_json(RUN/'conditional_cases.json',conditional)
    rows=update_index()
    write_csv(RUN/'pending_cases.csv',[r for r in rows if r['status'] in ['pending','running','partial','pending_prerequisite']])
    write_csv(RUN/'excluded_cases.csv',[r for r in rows if r['status']=='excluded_before_solve'])
    actual=[]
    for path in sorted((RUN/'cases').glob('*/metrics.json')):
        c=read(path.parent/'config.json');m=read(path)
        backend='scalar' if c['N']==100 and c['seed']==11 and c['group']=='D1' else 'batched'
        actual.append(dict(id=c['id'],group=c['group'],seed=c['seed'],N=c['N'],status=m['status'],
                           allocation_backend=m.get('allocation_backend',backend),
                           strict_verification='both solvers; all-pair shared lifetime; half-step critical-pair check',
                           role='individual seeded start; final matched comparison requires all three seeds',
                           elapsed_seconds=m.get('elapsed_seconds')))
    write_csv(RUN/'execution_provenance.csv',actual)
    write_json(RUN/'inventory.json',dict(planned_matrix_rows=len(read(RUN/'cases.json')),
              distinct_planned_configurations=len({c['configuration_sha256'] for c in read(RUN/'cases.json')}),
              actual_new_attempts=len(actual),actual_new_status_counts=dict(Counter(r['status'] for r in actual)),
              matrix_status_counts=dict(Counter(r['status'] for r in rows if r['id'] in {c['id'] for c in read(RUN/'cases.json')})),
              reused_matrix_rows=sum(bool(r.get('reused_case')) for r in rows),
              historical_duck_replays=4,historical_original_payload='unavailable; original logs retained',
              historical_rerun_statuses={c['id']:c['status'] for c in conditional if c['id'].startswith('A1_')},
              compute_constraint='CPU-only environment, 5 allocated CPUs. Full 1014-row matrix is resumable and exceeds this interactive first-pass budget. All pending IDs remain explicit.',
              scope='Only this portable results bundle is mirrored; proposal text and existing thesis PDFs are unchanged.'))


if __name__=='__main__':refresh()
