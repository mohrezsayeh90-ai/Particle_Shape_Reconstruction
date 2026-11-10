import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from magnetic_control.runtime import load_fields,ROOT
from magnetic_control import fields as f
from magnetic_control.tracking import Target,track

def main():
 c=load_fields()
 fields=f.check_fields(f.FLUID,f.MAG,c);fields['compression']=f.FORCES.audit(f.MAG)
 (ROOT/'results/field_validation.json').write_text(json.dumps(fields,indent=2))
 assert fields['all_passed'] and fields['compression']['all_passed']
 z=np.linspace(-2,1.7,181);s=(z+2)/3.7
 curves={
  'bump': .045*(np.exp(-((z+.1)/.65)**2)-np.exp(-((-2+.1)/.65)**2)),
  'reverse': -.04*np.sin(2*np.pi*s)*np.sin(np.pi*s)**2,
  'deflection': .035*(np.tanh((z+.4)*2)-np.tanh((-2+.4)*2)),
  'unreachable': .6*np.sin(10*np.pi*s),
 }
 reports={};fig,axs=plt.subplots(2,2,figsize=(11,7))
 for ax,(name,x) in zip(axs.flat,curves.items()):
  table=np.column_stack((z*c.d,x*c.Rv));np.savetxt(ROOT/f'examples/{name}.csv',table,delimiter=',',header='z_m,x_m',comments='')
  target=Target(table,c)
  result=track(target,c);oracle=track(target,c,oracle=True,flow_factor=1.1 if name=='reverse' else 1.)
  reports[name]={'learned_plant':result['report'],'independent_closed_loop':oracle['report']}
  np.savetxt(ROOT/f'results/{name}.csv',np.column_stack((result['tau']*c.time_scale,result['target']*c.Rv,result['q'][:,0]*c.Rv,result['q'][:,1]*c.d,result['u'],result['replay'][:,0]*c.Rv,result['replay'][:,1]*c.d)),delimiter=',',header='time_s,target_x_m,x_m,z_m,u,replay_x_m,replay_z_m',comments='')
  ax.plot(table[:,0]*1000,table[:,1]*1e6,'k--',label='Target')
  ax.plot(result['q'][:,1]*c.d*1000,result['q'][:,0]*c.Rv*1e6,label='Trained fields + feedback')
  ax.plot(oracle['q'][:,1]*c.d*1000,oracle['q'][:,0]*c.Rv*1e6,':',label='Independent physics')
  ax.set(title=name,xlabel='z (mm)',ylabel='x (micrometres)');ax.legend(fontsize=7)
  print(name,json.dumps(reports[name]),flush=True)
 # Timed path input and explicit rejection checks.
 time=np.linspace(0,c.T*c.time_scale,161);s=time/time[-1]
 x=.02*np.sin(np.pi*s)**3
 reports['timed_path']=track(Target(np.column_stack((time,x*c.Rv)),c,'time'),c)['report']
 invalid=[]
 for table in (np.array([[0,0]]),np.column_stack((z[::-1]*c.d,np.zeros(len(z)))),np.column_stack((z*c.d,np.ones(len(z))*c.Rv))):
  try:Target(table,c)
  except ValueError:invalid.append(True)
 reports['invalid_inputs_rejected']=len(invalid)==3
 reports['passed']=all(reports[n]['learned_plant']['accepted'] and reports[n]['independent_closed_loop']['accepted'] for n in ('bump','reverse','deflection')) and not reports['unreachable']['learned_plant']['accepted'] and reports['timed_path']['accepted'] and reports['invalid_inputs_rejected']
 (ROOT/'results/validation.json').write_text(json.dumps(reports,indent=2));fig.tight_layout();fig.savefig(ROOT/'results/tracking.png',dpi=160)
 assert reports['passed'], 'Tracking acceptance failed; inspect results/validation.json'
if __name__=='__main__': main()
