"""Predict a new temporal waveform from a CSV without retraining."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from magnetic_operator.train import load
from magnetic_operator.physics import Parameters,AMAX,observables
from magnetic_operator.inference import predict,reference

def main():
    p=argparse.ArgumentParser()
    p.add_argument('field_csv');p.add_argument('--checkpoint',default='results/operator')
    p.add_argument('--output',default='results/operator/custom')
    p.add_argument('--eta',type=float,default=.003276913272013514)
    p.add_argument('--mean',type=float,default=.015);p.add_argument('--rho',type=float,default=1000.)
    p.add_argument('--x0',type=float,default=0.);p.add_argument('--z0',type=float,default=-.08)
    p.add_argument('--points',type=int,default=301);p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    data=np.genfromtxt(args.field_csv,delimiter=',',names=True)
    if data.dtype.names!=('time_s','a_left','a_right') or data.ndim!=1 or len(data)<2:
        raise ValueError('Expected CSV columns time_s,a_left,a_right and at least two rows')
    table=np.column_stack([data[n] for n in data.dtype.names])
    if not np.isfinite(table).all() or table[0,0]!=0 or np.any(np.diff(table[:,0])<=0):
        raise ValueError('Finite, strictly increasing time starting at zero required')
    if np.max(np.abs(table[:,1:]).sum(1))>AMAX+1e-10:raise ValueError('Field outside amplitude domain')
    if args.points<2:raise ValueError('At least two output points required')
    def field(t):return np.array([np.interp(t,table[:,0],table[:,i]) for i in (1,2)])
    times=np.linspace(0,table[-1,0],args.points)
    parameters=Parameters(eta=args.eta,mean=args.mean,rho=args.rho).validate()
    path,meta=predict(load(args.checkpoint),field,times,parameters,(args.x0,args.z0))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    np.savetxt(output/'trajectory.csv',np.column_stack((times,path)),delimiter=',',
               header='time_s,x_m,z_m',comments='')
    np.savetxt(output/'magnetic_observables.csv',observables(field,times,path),delimiter=',',
               header='time_s,Bx_T,Bz_T,Fmx_N,Fmz_N',comments='')
    if args.verify:
        truth=reference(field,times,parameters,(args.x0,args.z0))
        error=abs(path-truth)*1e6
        meta['max_x_error_um']=float(error[:,0].max());meta['max_z_error_um']=float(error[:,1].max())
        np.savetxt(output/'reference.csv',np.column_stack((times,truth)),delimiter=',',header='time_s,x_m,z_m',comments='')
    (output/'report.json').write_text(json.dumps(meta,indent=2))
    summary={k:v for k,v in meta.items() if k!='windows'}
    summary['window_count']=len(meta['windows']);summary['output']=str(output)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
