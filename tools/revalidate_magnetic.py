"""Re-run original acceptance suites on CPU without retraining or changing thresholds."""
from pathlib import Path
import argparse,shutil,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('case',choices=['master9','master10-magnetic-function-deeponet'])
args=parser.parse_args()
package=ROOT/'evidence'/args.case/'runtime_package'
out=ROOT/'verification'/args.case;out.mkdir(exist_ok=True)
for p in (package/'pretrained').iterdir():
    if not (out/p.name).exists():shutil.copyfile(p,out/p.name)
sys.path.insert(0,str(package));torch.set_num_threads(1)
if args.case=='master9':
    from master9.validate import run
    from master9.train import load
    from master9.control import demos
    physics=run(out);model,_=load(out);controls=demos(model.double(),out)
    assert physics['all_passed'] and controls['all_passed']
else:
    from magnetic_operator.validate import run
    from magnetic_operator.extended_checks import run as extra
    tests=run(out);extended=extra(out)
    assert tests['all_passed'] and extended['all_passed']
print('CPU inference / validation complete; original training provenance is retained.')
