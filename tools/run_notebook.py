"""Execute an archived notebook in a new folder using this Python interpreter.

Original source snapshots stay unchanged. Flag overrides are recorded in the
run directory. Training and inverse optimization are opt-in for cached models.
"""
from pathlib import Path
import argparse,ast,datetime,hashlib,json,os,tempfile,time

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('slug',choices=[r['ref'].split('/')[-1] for r in json.loads((ROOT/'catalog.json').read_text())])
    parser.add_argument('--output',type=Path,help='A new or empty output folder')
    parser.add_argument('--fresh-training',action='store_true',help='Set existing TRAIN_FROM_SCRATCH flags to True')
    parser.add_argument('--reoptimize',action='store_true',help='Discard embedded duck result caches, preserving the physical model')
    parser.add_argument('--prepare-only',action='store_true',help='Save and inspect the execution copy without starting a kernel')
    args=parser.parse_args()
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    import sys
    source=next((ROOT/'evidence'/args.slug).glob('*.ipynb'))
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out=(args.output or ROOT/'experiments'/stamp/args.slug).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a new or empty output folder')
    out.mkdir(parents=True,exist_ok=True)
    nb=nbformat.read(source,as_version=4);changed=[]
    for ci,cell in enumerate(nb.cells):
        if cell.cell_type!='code':continue
        try:tree=ast.parse(cell.source)
        except SyntaxError:continue
        lines=cell.source.splitlines(keepends=True);replacements=[]
        for node in tree.body:
            if not isinstance(node,ast.Assign):continue
            targets=[getattr(x,'id','') for x in node.targets]
            if args.fresh_training and targets in [['TRAIN_FROM_SCRATCH'],['TRAIN_AGAIN']]:
                replacements.append((node.lineno-1,node.end_lineno,targets[0]+' = True\n'))
            if args.reoptimize and any(x in ['_DENSE_PRECOMPUTED','_DUCK_PRECOMPUTED'] for x in targets):
                replacements.append((node.lineno-1,node.end_lineno,targets[0]+' = {}\n'))
        for start,end,value in sorted(replacements,reverse=True):
            lines[start:end]=[value];changed.append({'cell':ci,'assignment':value.strip()})
        cell.source=''.join(lines)
    always_trains={'master-1','master-2','master3-2dsimplechannel-general-up-down-particle','master3-clean-revised-version','master4','master6-validation-2d-magnetic-pinn'}
    if args.fresh_training and not any(x['assignment'].startswith(('TRAIN_FROM_SCRATCH','TRAIN_AGAIN')) for x in changed) and args.slug not in always_trains:
        raise ValueError('This notebook has no fresh-training mode; see its source recipe')
    if args.reoptimize and not any('PRECOMPUTED' in x['assignment'] for x in changed):raise ValueError('No supported duck cache was found')
    nbformat.write(nb,out/'input.ipynb')
    record={'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'started_utc':stamp,'fresh_training_requested':args.fresh_training,'reoptimize_requested':args.reoptimize,
            'overrides':changed,'status':'running','note':'Notebook defaults still apply; some original notebooks train from scratch by default.'}
    info=out/'execution.json';info.write_text(json.dumps(record,indent=2))
    if args.prepare_only:
        record['status']='prepared';info.write_text(json.dumps(record,indent=2))
        print('Prepared execution copy:',out)
        return
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.local/matplotlib'))
    (ROOT/'.local').mkdir(exist_ok=True)
    os.environ.setdefault('IPYTHONDIR',str(ROOT/'.local/ipython'))
    os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.local/cache'))
    begin=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='shape-kernel-') as tmp:
        kroot=Path(tmp)/'kernels';spec=kroot/'shape-runtime';spec.mkdir(parents=True)
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Particle Shape current Python','language':'python'}))
        manager=KernelManager(kernel_name='shape-runtime',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kroot)]),connection_file=str(Path(tmp)/'connection.json'))
        client=NotebookClient(nb,km=manager,timeout=-1,resources={'metadata':{'path':str(out)}})
        try:
            client.execute();record['status']='complete'
        except Exception as e:
            record['status']='failed';record['error_type']=type(e).__name__;raise
        finally:
            nbformat.write(nb,out/'executed.ipynb');record['elapsed_seconds']=time.monotonic()-begin
            info.write_text(json.dumps(record,indent=2))
            if manager.has_kernel:manager.shutdown_kernel(now=True)
    print('Execution:',out,'status:',record['status'])

if __name__=='__main__':main()
