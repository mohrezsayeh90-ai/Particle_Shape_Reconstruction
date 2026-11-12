"""Strict all-pair shared-clock verification, independent of plotting grids."""
import csv,json,time,os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from shapely.geometry import Point
from shapely.ops import unary_union
from .physics import ACCEPT,BOUNDS,DIAMETER,H,GAP
from .targets import covariance_metrics,mask_shape


def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    if fields is None:fields=list(rows[0]) if rows else []
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    os.replace(temporary,path)


def write_json(path,data):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n')
    os.replace(temporary,path)


def pair_minimum(paths,ages,indices=None,samples=801,target_time=None):
    T=float(max(ages)+.25 if target_time is None else target_time);best={'distance':1e30,'time':None,'pair':None,'sampled_distance':1e30}
    pairs=indices if indices is not None else ((i,j) for i in range(len(paths)) for j in range(i))
    for i,j in pairs:
        duration=min(ages[i],ages[j]);back=np.unique(np.r_[np.linspace(0,duration,samples),ages[ages<=duration]])
        def distance(t):
            t=np.atleast_1d(t)
            return np.linalg.norm(paths[i].sol(ages[i]-t)[:2]-paths[j].sol(ages[j]-t)[:2],axis=0)
        values=distance(back);k=int(values.argmin());value=float(values[k]);when=float(back[k])
        raw=value
        local=np.flatnonzero((values[1:-1]<=values[:-2])&(values[1:-1]<=values[2:]))+1
        for k in local:
            opt=minimize_scalar(lambda t:float(distance(t)[0]),bounds=(back[k-1],back[k+1]),method='bounded',options={'xatol':1e-12})
            if opt.fun<value:value=float(opt.fun);when=float(opt.x)
        if value<best['distance']:best=dict(distance=value,time=T-when,pair=[int(j+1),int(i+1)],sampled_distance=raw)
    if len(paths)<2:return {'distance':None,'time':None,'pair':None,'sampled_distance':None}
    return best


def schedule_rows(physics,controls,sides,ages,targets,port_ids=None):
    T=float(max(ages)+.25);rows=[]
    for i,(c,side,a,g) in enumerate(zip(controls,sides,ages,targets)):
        j=3*int(side)
        rows.append(dict(particle=i+1,port_id=port_ids[i] if port_ids else f'free_{i+1}',side=int(side),
                         launcher='bottom' if side==0 else 'top',x_launch=float(c[j]),
                         y_launch=DIAMETER/2+GAP if side==0 else H-DIAMETER/2-GAP,
                         u_launch=float(c[j+1]),v_launch=float(c[j+2]),age_at_target=float(a),
                         release_time=T-float(a),target_x=float(g[0]),target_y=float(g[1])))
    return rows


def controls_from_schedule(rows):
    controls=[]
    for r in rows:
        side=int(r.get('side',0 if r['launcher']=='bottom' else 1));j=side*3;c=BOUNDS.mean(1)
        c[j:j+3]=[float(r['x_launch']),float(r['u_launch']),float(r['v_launch'])];controls.append(c)
    return np.array(controls)


def verify(physics,rows,targets,folder,case,save_arrays=True):
    start=time.monotonic();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    controls=controls_from_schedule(rows);sides=np.array([int(r.get('side',0 if r['launcher']=='bottom' else 1)) for r in rows])
    ages=np.array([float(r['age_at_target']) for r in rows]);releases=np.array([float(r['release_time']) for r in rows])
    T=float(max(releases+ages));clock_error=float(np.max(abs(releases+ages-T)))
    bounds_ok=bool(np.all(controls>=BOUNDS[:,0]) and np.all(controls<=BOUNDS[:,1]) and np.all(releases>=0) and clock_error<1e-8
                   and np.all(ages>=case.get('min_age',0)-1e-10) and np.all(ages<=case.get('max_age',1e30)+1e-10))
    minimum_gap=None
    for port in sorted(set(str(r.get('port_id',r.get('station_id','free'))) for r in rows)):
        times=sorted(float(r['release_time']) for r in rows if str(r.get('port_id',r.get('station_id','free')))==port)
        if len(times)>1:minimum_gap=min(np.diff(times).min(),minimum_gap if minimum_gap is not None else 1e30)
    timing_ok=case.get('min_gap',0)<=0 or (minimum_gap is None or minimum_gap>=case['min_gap']-1e-9)
    replays={};paths={};endpoints={};end_rows=[]
    for mode in ('pinn','reference'):
        pp=[physics.trace(c,int(s),a,mode) for c,s,a in zip(controls,sides,ages)]
        paths[mode]=pp;points=np.array([p.y[:2,-1] for p in pp]);endpoints[mode]=points
        complete=bool(all(p.valid for p in pp));error=np.linalg.norm(points-targets,axis=1)
        nearest=pair_minimum(pp,ages,target_time=T) if complete else dict(distance=None,time=None,pair=None,sampled_distance=None)
        sensitivity=None
        if complete and nearest['pair']:
            ij=[i-1 for i in nearest['pair']]
            finer=[physics.trace(controls[i],int(sides[i]),ages[i],mode,max_step=.00625 if mode=='reference' else .00875) for i in ij]
            fine_min=pair_minimum(finer,ages[ij],samples=1601,target_time=T)
            sensitivity=dict(max_endpoint_difference=max(float(np.linalg.norm(p.y[:2,-1]-pp[i].y[:2,-1])) for p,i in zip(finer,ij)),
                             minimum_distance_difference=abs(fine_min['distance']-nearest['distance']),finer_minimum=fine_min['distance'])
            if fine_min['distance']<nearest['distance']:
                nearest['distance']=fine_min['distance']
                nearest['time']=fine_min['time']
        replays[mode]=dict(completed=complete,completion_fraction=float(np.mean([p.valid for p in pp])),
                           max_error=float(error.max()),rms_error=float(np.sqrt(np.mean(error**2))),
                           minimum=nearest,half_step_check=sensitivity,
                           termination_events=[p.event for p in pp],solver='DOP853' if mode=='reference' else 'RK45')
        for i,(g,pt,e) in enumerate(zip(targets,points,error)):
            end_rows.append(dict(particle=i+1,solver=mode,target_x=g[0],target_y=g[1],actual_x=pt[0],actual_y=pt[1],euclidean_error=e,completed=pp[i].valid))
    delta=float(np.max(np.linalg.norm(endpoints['pinn']-endpoints['reference'],axis=1)))
    min_values=[r['minimum']['distance'] for r in replays.values()]
    clearance=all(x is not None and x>=ACCEPT for x in min_values)
    complete=all(r['completed'] for r in replays.values())
    endpoint_ok=replays['pinn']['max_error']<=.003 and replays['reference']['max_error']<=.004 and delta<=.004
    ids=[str(r.get('port_id',r.get('station_id',f'free_{i}'))) for i,r in enumerate(rows)]
    loads={port:ids.count(port) for port in sorted(set(ids))}
    capacity_ok=case.get('cap') is None or max(loads.values(),default=0)<=case['cap']
    table={str(p['id']):p for p in case.get('ports',[])}
    fixed_coordinates_ok=case.get('mode')=='free' or all(ids[i] in table and int(table[ids[i]]['side'])==int(sides[i]) and abs(table[ids[i]]['x']-float(r['x_launch']))<1e-12 for i,r in enumerate(rows))
    accepted=bool(bounds_ok and timing_ok and capacity_ok and fixed_coordinates_ok and complete and endpoint_ok and clearance)
    reason=None if accepted else 'invalid_boundary_event' if not complete else 'invalid_launch_bounds' if not bounds_ok or not fixed_coordinates_ok else 'capacity_violation' if not capacity_ok else 'emission_gap_failure' if not timing_ok else 'endpoint_error' if not endpoint_ok else 'clearance_failure'
    metrics=dict(accepted=accepted,status='accepted' if accepted else 'rejected',failure_type=reason,
                 provenance=case.get('provenance','new_experiment'),N=len(rows),N_achieved=int(sum(p.valid for p in paths['reference'])),
                 configured_ports=None if case.get('mode')=='free' else len(case.get('ports',[])),used_ports=len(loads),port_loads=loads,
                 top_emissions=int(sum(sides)),bottom_emissions=int(len(sides)-sum(sides)),
                 target_time=T,longest_duration=float(max(ages)),median_duration=float(np.median(ages)),
                 release_span=float(np.ptp(releases)),minimum_same_port_gap=None if minimum_gap is None else float(minimum_gap),
                 clock_error=clock_error,launch_bounds_ok=bounds_ok,min_gap_ok=timing_ok,capacity_ok=capacity_ok,fixed_coordinates_ok=fixed_coordinates_ok,
                 max_endpoint_difference=delta,replays=replays,weight_sha256=physics.sha,
                 verification_seconds=time.monotonic()-start,numerical_verification_only=True)
    if complete:metrics['covariance']=covariance_metrics(targets,endpoints['reference'])
    if complete and case.get('shape') in ('stadium','arrow','fish','perforated'):
        shape=mask_shape(case['shape'],case.get('center',[1.52,.15]));occupancies=[]
        for resolution in (64,128):
            disks=unary_union([Point(*xy).buffer(DIAMETER/2,quad_segs=resolution) for xy in endpoints['reference']])
            inside=shape.intersection(disks).area;union_area=disks.area
            holes=unary_union([Point(0,0).buffer(0)])
            if case['shape']=='perforated':holes=mask_shape('stadium',case.get('center',[1.52,.15])).difference(shape)
            occupancies.append(dict(resolution=resolution,inside_coverage=inside/shape.area,
                                    overspill=(union_area-inside)/union_area,hole_intrusion_area=holes.intersection(disks).area))
        metrics['occupancy']=occupancies[-1];metrics['occupancy_convergence']=occupancies
    write_csv(folder/'release_schedule.csv',rows);write_csv(folder/'endpoint_errors.csv',end_rows)
    if save_arrays:
        grid=np.unique(np.r_[np.linspace(0,T,601),releases,T]);arrays={'time':grid,'targets':targets,'release_times':releases,'ages':ages}
        for mode,pp in paths.items():
            positions=np.full((len(rows),len(grid),2),np.nan)
            for i,(p,release) in enumerate(zip(pp,releases)):
                present=(grid>=release)&(grid<=release+p.end+1e-10)
                positions[i,present]=p.sol(np.clip(grid[present]-release,0,p.end))[:2].T
            dmin=np.full(len(grid),np.nan);active=np.isfinite(positions[:,:,0]).sum(0)
            for k in range(len(grid)):
                points=positions[np.isfinite(positions[:,k,0]),k]
                if len(points)>1:
                    from scipy.spatial.distance import pdist
                    dmin[k]=pdist(points).min()
            arrays[mode+'_positions']=positions;arrays[mode+'_endpoints']=endpoints[mode];arrays[mode+'_sampled_dmin']=dmin;arrays[mode+'_active']=active
        np.savez_compressed(folder/'trajectories.npz',**arrays)
    write_json(folder/'metrics.json',metrics)
    return metrics
