#!/usr/bin/env python3
"""Refresh, render, audit and checksum a quiescent publication snapshot.

This command writes only inside this bundle. It does not copy repositories,
edit thesis prose, train a network, run a new search, commit, or push.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/formation_20261112_cpu'


def portable(path):
    parts=path.relative_to(ROOT).parts
    return '__pycache__' not in parts and 'cache' not in parts and path.suffix not in ['.pyc','.tmp'] and path.name!='SHA256SUMS'


def main():
    for command in ['refresh_outputs.py','save_diagnostics.py','export_perturbation_inputs.py','plot_formation_campaign.py','audit_bundle.py','write_report.py']:
        subprocess.run([sys.executable,str(ROOT/command)],cwd=ROOT,check=True)
    logs=RUN/'logs';logs.mkdir(exist_ok=True)
    for source in sorted(Path('/tmp').glob('formation-*.log')):
        shutil.copyfile(source,logs/source.name)
    groups=defaultdict(list)
    for file in (RUN/'cases').glob('*/metrics.json'):
        m=json.loads(file.read_text());c=json.loads((file.parent/'config.json').read_text())
        if 'elapsed_seconds' in m:groups[c['group']].append(m['elapsed_seconds'])
    report=dict(allocated_cpus=5,gpu=False,completed_attempt_wall_seconds_by_group={k:dict(trials=len(v),sum=sum(v),maximum=max(v)) for k,v in sorted(groups.items())},
                caution='Elapsed times overlap across processes and are not CPU billing measurements. Pending trials are listed individually; no full-matrix completion is claimed.')
    (RUN/'compute_report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and portable(path):lines.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(ROOT)))
    (ROOT/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
    print('PORTABLE_SNAPSHOT',len(lines),'files',flush=True)


if __name__=='__main__':main()
