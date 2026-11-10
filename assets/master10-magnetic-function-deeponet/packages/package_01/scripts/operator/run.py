"""Train/validate the function-conditioned operator."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from magnetic_operator.train import train
from magnetic_operator.validate import run
from magnetic_operator.extended_checks import run as extended_checks

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/operator')
    p.add_argument('--train',action='store_true');p.add_argument('--steps',type=int,default=18000)
    p.add_argument('--allow-cpu',action='store_true');a=p.parse_args()
    if a.train:train(a.output,steps=a.steps,require_gpu=not a.allow_cpu)
    run(a.output,require_gpu=not a.allow_cpu)
    extended_checks(a.output)
