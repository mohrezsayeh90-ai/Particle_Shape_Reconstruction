"""Generalized fixed-port / free-location Master3 planner.

Reuses the c17460d seed-bank, bounded endpoint fit, collision residual and
structured sparse-Jacobian strategy. Port sides come from an explicit table.
Only search iterates may clamp terminated paths; verification never does.
"""
from collections import OrderedDict,Counter
import hashlib,pickle,time,os
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,linear_sum_assignment,brentq
from scipy.sparse import coo_matrix
from .physics import BOUNDS,MID,L,H,SEARCH,DIAMETER,GAP


def digest(value):
    import json
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


class Planner:
    def __init__(self,physics,case,mode='coarse'):
        self.physics=physics;self.case=case;self.mode=mode;self.ports=case['ports'];self.free=case['mode']=='free'
        self.dim=4 if self.free else 3;self.max_age=case.get('max_age',5.);self.min_age=case.get('min_age',.03)
        self.cache=OrderedDict();self.calls=0;self.samples=61;self.pair_indices=None
        self.collision=case.get('collision_aware',True)

    def physical(self,row,port):
        side=int(port) if self.free else int(self.ports[int(port)]['side']);j=3*side
        c=MID.copy();q=np.clip(row,0,1)
        if self.free:c[j:j+3]=BOUNDS[j:j+3,0]+q[:3]*np.diff(BOUNDS[j:j+3],axis=1)[:,0]
        else:
            c[j]=self.ports[int(port)]['x'];c[j+1:j+3]=BOUNDS[j+1:j+3,0]+q[:2]*np.diff(BOUNDS[j+1:j+3],axis=1)[:,0]
        age=self.min_age+q[-1]*(self.max_age-self.min_age)
        return c,side,float(age)

    def pack(self,c,side,age):
        j=3*side;idx=slice(j,j+3) if self.free else slice(j+1,j+3)
        return np.r_[(c[idx]-BOUNDS[idx,0])/np.diff(BOUNDS[idx],axis=1)[:,0],(age-self.min_age)/(self.max_age-self.min_age)]

    def one_path(self,row,port):
        key=(int(port),*np.round(row,12))
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        c,side,age=self.physical(row,port);p=self.physics.trace(c,side,age,self.mode)
        self.cache[key]=p;self.calls+=1
        if len(self.cache)>2048:self.cache.popitem(last=False)
        return p

    def endpoints(self,z,ports):
        rows=np.asarray(z).reshape(-1,self.dim);paths=[self.one_path(row,p) for row,p in zip(rows,ports)]
        ages=np.array([self.physical(row,p)[2] for row,p in zip(rows,ports)])
        actual=np.array([p.sol(min(a,p.end))[:2] for p,a in zip(paths,ages)])
        return paths,ages,actual

    def seed_bank(self,cache):
        key=digest(dict(ports=self.ports,free=self.free,max_age=self.max_age,min_age=self.min_age))
        path=Path(cache)/('bank_'+key+'.pkl')
        if path.exists():return pickle.loads(path.read_bytes())
        bank={}
        for port in (range(2) if self.free else range(len(self.ports))):
            rows=[];side=port if self.free else self.ports[port]['side'];j=3*side
            x=.01 if self.free else self.ports[port]['x']
            for ux in np.linspace(-.4,1.5,4):
                for vy in np.linspace(*BOUNDS[j+2],15):
                    c=MID.copy();c[j:j+3]=[x,ux,vy]
                    p=self.physics.trace(c,side,self.max_age,'coarse')
                    if p.end<=self.min_age:continue
                    aa=np.linspace(self.min_age,p.end,251);xy=p.sol(aa)[:2].T
                    rows.append(dict(c=c,path=p,ages=aa,xy=xy,side=side))
            bank[port]=rows
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_bytes(pickle.dumps(bank));os.replace(tmp,path)
        return bank

    def candidate_options(self,targets,cache):
        key=digest(dict(targets=targets.tolist(),ports=self.ports,free=self.free,min_age=self.min_age,max_age=self.max_age,endpoint_cap=24))
        cached=Path(cache)/('options_'+key+'.pkl')
        if cached.exists():return pickle.loads(cached.read_bytes())
        bank=self.seed_bank(cache);options=[]
        for i,target in enumerate(targets):
            group=[]
            for port,seeds in bank.items():
                if self.free:
                    for seed in seeds:
                        yy=seed['xy'][:,1]-target[1];cross=np.flatnonzero(yy[:-1]*yy[1:]<=0)
                        if not len(cross):continue
                        k=int(cross[0]);p=seed['path']
                        age=brentq(lambda a:float(p.sol(a)[1]-target[1]),seed['ages'][k],seed['ages'][k+1])
                        c=seed['c'].copy();c[3*port]+=target[0]-p.sol(age)[0]
                        if not 0<=c[3*port]<=L:continue
                        row=self.pack(c,port,age);actual=self.one_path(row,port)
                        if actual.end>=age-1e-8:group.append(dict(row=row,port=port,error=float(np.linalg.norm(actual.sol(age)[:2]-target))))
                    continue
                starts=[]
                for seed in seeds:
                    dd=np.linalg.norm(seed['xy']-target,axis=1);k=int(dd.argmin())
                    starts.append((float(dd[k]),self.pack(seed['c'],seed['side'],seed['ages'][k])))
                starts.sort(key=lambda x:x[0])
                for _,row in starts[:2]:
                    def residual(v):
                        p=self.one_path(v,port);a=self.physical(v,port)[2]
                        return np.r_[(p.sol(min(a,p.end))[:2]-target)/H,3*max(0,a-p.end)/self.max_age]
                    opt=least_squares(residual,np.clip(row,1e-8,1-1e-8),bounds=(0,1),max_nfev=24,diff_step=2e-4,ftol=1e-8,xtol=1e-8,gtol=1e-8)
                    p=self.one_path(opt.x,port);a=self.physical(opt.x,port)[2]
                    if p.end>=a-1e-8:group.append(dict(row=opt.x,port=port,error=float(np.linalg.norm(p.sol(a)[:2]-target))))
            if self.free:
                selected=[]
                for side in (0,1):
                    subset=sorted([o for o in group if o['port']==side],key=lambda o:o['error'])
                    if not subset:continue
                    chosen=[subset.pop(0)]
                    while subset and len(chosen)<8:
                        k=max(range(len(subset)),key=lambda k:min(np.linalg.norm(subset[k]['row']-o['row']) for o in chosen)-20*subset[k]['error']/H)
                        chosen.append(subset.pop(k))
                    selected.extend(chosen)
                group=selected
            options.append(group)
            if (i+1)%10==0:print('candidate_targets',i+1,'/',len(targets),flush=True)
        cached.parent.mkdir(parents=True,exist_ok=True)
        tmp=cached.with_suffix(f'.{os.getpid()}.tmp');tmp.write_bytes(pickle.dumps(options));os.replace(tmp,cached)
        return options

    def choose(self,options,seed):
        rng=np.random.default_rng(seed);n=len(options);cap=self.case.get('cap')
        if any(not o for o in options):raise RuntimeError('candidate_shortage')
        if self.free:ids=[min(range(len(g)),key=lambda k:g[k]['error']) for g in options]
        else:
            slots=np.repeat(np.arange(len(self.ports)),min(n,cap) if cap is not None else n)
            costs=np.array([[min((o['error'] for o in group if o['port']==p),default=1e6) for p in slots] for group in options])
            costs+=rng.uniform(0,1e-10,costs.shape)
            ii,jj=linear_sum_assignment(costs)
            if len(ii)!=n or np.any(costs[ii,jj]>1e5):raise RuntimeError('candidate_shortage')
            ids=[None]*n
            for i,k in zip(ii,jj):ids[i]=min((j for j,o in enumerate(options[i]) if o['port']==slots[k]),key=lambda j:options[i][j]['error'])
        def pair_cost(a,b):
            if not self.collision:return 0.
            p=self.one_path(a['row'],a['port']);q=self.one_path(b['row'],b['port'])
            ai=self.physical(a['row'],a['port'])[2];aj=self.physical(b['row'],b['port'])[2]
            back=np.linspace(0,min(ai,aj),31)
            distance=np.linalg.norm(p.sol(np.minimum(ai-back,p.end))[:2]-q.sol(np.minimum(aj-back,q.end))[:2],axis=0)
            cost=20*(max(0.,SEARCH-float(distance.min()))/H)**2
            if a['port']==b['port'] and not self.free and self.case.get('min_gap',0):cost+=20*(max(0,self.case['min_gap']-abs(ai-aj))/self.max_age)**2
            return cost
        def score(i,o,exclude=None):
            return (o['error']/H)**2+sum(pair_cost(o,options[j][ids[j]]) for j in range(n) if j!=i and j!=exclude)
        def batched_scores(i):
            # Same shared-clock pair cost as above. Evaluate one particle's
            # alternatives in chunks against the CURRENT selected trajectories.
            # No all-alternative pair cache is built or retained.
            candidates=options[i];cost=np.array([(o['error']/H)**2 for o in candidates])
            if not self.collision and not self.case.get('min_gap',0):return cost
            selected=[options[j][ids[j]] for j in range(n)]
            ai=np.array([self.physical(o['row'],o['port'])[2] for o in candidates])
            aj=np.array([self.physical(o['row'],o['port'])[2] for o in selected])
            f=np.linspace(0,1,31)
            others=np.delete(np.arange(n),i)
            for chunk in np.array_split(others,max(1,int(np.ceil(len(others)/32)))):
                if not len(chunk):continue
                back=np.minimum(ai[:,None],aj[chunk][None,:])[:,:,None]*f
                pi=np.array([self.one_path(o['row'],o['port']).sol(np.minimum(ai[k]-back[k],self.one_path(o['row'],o['port']).end).ravel())[:2].T.reshape(len(chunk),31,2)
                             for k,o in enumerate(candidates)])
                pj=np.array([self.one_path(selected[j]['row'],selected[j]['port']).sol(np.minimum(aj[j]-back[:,k],self.one_path(selected[j]['row'],selected[j]['port']).end).ravel())[:2].T.reshape(len(candidates),31,2)
                             for k,j in enumerate(chunk)]).swapaxes(0,1)
                distance=np.linalg.norm(pi-pj,axis=-1).min(axis=-1)
                if self.collision:cost+=np.sum(20*(np.maximum(0,SEARCH-distance)/H)**2,axis=1)
                if not self.free and self.case.get('min_gap',0):
                    same=np.array([[o['port']==selected[j]['port'] for j in chunk] for o in candidates])
                    cost+=np.sum(20*(np.maximum(0,self.case['min_gap']-abs(ai[:,None]-aj[chunk]))/self.max_age)**2*same,axis=1)
            return cost
        for sweep in range(4):
            changed=False
            for i in rng.permutation(n):
                current=options[i][ids[i]];loads=Counter(options[j][ids[j]]['port'] for j in range(n))
                costs=np.array([score(i,o) for o in options[i]]) if self.case.get('allocation_backend')=='scalar' else batched_scores(i)
                old=costs[ids[i]]
                if self.case.get('check_allocation_vectorization'):
                    expected=np.array([score(i,o) for o in options[i]])
                    assert np.max(abs(expected-costs))<1e-10,'Vectorized allocation changed pair costs'
                for k in rng.permutation(len(options[i])):
                    candidate=options[i][k];port=candidate['port']
                    if not self.free and cap is not None and port!=current['port'] and loads[port]>=cap:continue
                    value=costs[k]
                    if value+1e-14<old:ids[i]=k;current=candidate;old=value;changed=True
            # Capacity-preserving swaps can move assignments even when all slots are full.
            if not self.free and cap is not None:
                for i,j in zip(rng.permutation(n)[::2],rng.permutation(n)[1::2]):
                    if i==j:continue
                    a=options[i][ids[i]];b=options[j][ids[j]]
                    oi=[(k,o) for k,o in enumerate(options[i]) if o['port']==b['port']]
                    oj=[(k,o) for k,o in enumerate(options[j]) if o['port']==a['port']]
                    if not oi or not oj:continue
                    ki,ai=min(oi,key=lambda v:score(i,v[1],j));kj,bj=min(oj,key=lambda v:score(j,v[1],i))
                    if score(i,ai,j)+score(j,bj,i)+pair_cost(ai,bj)<score(i,a,j)+score(j,b,i)+pair_cost(a,b):ids[i]=ki;ids[j]=kj;changed=True
            if not changed:break
        selected=[options[i][k] for i,k in enumerate(ids)]
        return np.array([o['row'] for o in selected]).ravel(),np.array([o['port'] for o in selected])

    def evaluate(self,z,ports):
        paths,ages,actual=self.endpoints(z,ports);n=len(paths);S=self.samples
        pairs=np.array([(i,j) for i in range(n) for j in range(i)],int).reshape(-1,2)
        f=np.linspace(0,1,S);positions=np.zeros((len(pairs),2,S,4));times=np.zeros((len(pairs),2,S))
        # Chunk per-particle queries rather than allocating an all-alternative cache.
        for i,p in enumerate(paths):
            for axis in (0,1):
                ids=np.flatnonzero(pairs[:,axis]==i)
                for chunk in np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))):
                    if not len(chunk):continue
                    other=pairs[chunk,1-axis];tt=ages[i]-np.minimum(ages[i],ages[other,None])*f
                    times[chunk,axis]=tt
                    positions[chunk,axis]=p.sol(np.clip(tt,0,p.end).ravel()).T.reshape(len(chunk),S,4)
                    positions[chunk,axis,:,2:]*=(tt<=p.end+1e-9)[...,None]
        difference=positions[:,0,:,:2]-positions[:,1,:,:2];distance=np.linalg.norm(difference,axis=-1)
        return paths,ages,actual,pairs,times,positions,difference,distance

    def residual(self,z,ports,targets):
        paths,ages,actual,pairs,times,states,difference,distance=self.evaluate(z,ports)
        out=np.r_[((actual-targets)/H).ravel(),[3*max(0,a-p.end)/self.max_age for p,a in zip(paths,ages)],
                  (4*np.maximum(0,SEARCH-distance)/H).ravel() if self.collision else np.zeros(distance.size)]
        if not self.free and self.case.get('min_gap',0):
            for port in np.unique(ports):
                ii=np.flatnonzero(ports==port);ordered=ii[np.argsort(ages[ii])]
                out=np.r_[out,4*np.maximum(0,self.case['min_gap']-np.diff(ages[ordered]))/self.max_age]
        return out

    def jacobian(self,z,ports,targets):
        rows=np.asarray(z).reshape(-1,self.dim);n=len(rows);S=self.samples;span=self.max_age-self.min_age
        paths,ages,actual,pairs,times,states,difference,distance=self.evaluate(z,ports)
        unit=difference/np.maximum(distance[...,None],1e-12);active=(distance<SEARCH)&self.collision
        rr=[];cc=[];vv=[]
        def add(r,col,value):
            r=np.atleast_1d(r).ravel();value=np.atleast_1d(value).ravel();keep=abs(value)>1e-14
            rr.extend(r[keep].tolist());cc.extend([col]*int(keep.sum()));vv.extend(value[keep].tolist())
        for i in range(n):
            ids=np.flatnonzero(np.any(pairs==i,axis=1));axis=(pairs[ids,1]==i).astype(int)
            local,kk=np.nonzero(active[ids]);pi=ids[local];ax=axis[local];other=pairs[pi,1-ax]
            query=times[pi,ax,kk];old=states[pi,ax,kk,:2]
            sign=np.where(ax==0,1.,-1.)
            for control in range(self.dim-1):
                delta=2e-4 if self.mode=='direct' else 2e-5
                if rows[i,control]+delta>1:delta=-delta
                trial=rows[i].copy();trial[control]+=delta;p=self.one_path(trial,ports[i])
                ep=p.sol(min(ages[i],p.end))[:2]
                col=i*self.dim+control
                add([2*i,2*i+1],col,(ep-actual[i])/(delta*H))
                add([2*n+i],col,[3*(max(0,ages[i]-p.end)-max(0,ages[i]-paths[i].end))/(delta*self.max_age)])
                if len(pi):
                    changed=p.sol(np.clip(query,0,p.end))[:2].T
                    derivative=(changed-old)/delta
                    add(3*n+pi*S+kk,col,-4*sign*np.sum(unit[pi,kk]*derivative,axis=1)/H)
            col=i*self.dim+self.dim-1
            velocity=paths[i].sol(ages[i])[2:] if ages[i]<=paths[i].end+1e-9 else np.zeros(2)
            add([2*i,2*i+1],col,span*velocity/H)
            if ages[i]>paths[i].end+1e-9:add([2*n+i],col,[3*span/self.max_age])
            if len(pi):
                f=kk/(S-1);young=ages[i]<ages[other];vi=states[pi,ax,kk,2:];vj=states[pi,1-ax,kk,2:]
                deriv=np.where(young[:,None],(1-f[:,None])*vi+f[:,None]*vj,vi)*span
                add(3*n+pi*S+kk,col,-4*sign*np.sum(unit[pi,kk]*deriv,axis=1)/H)
        extra=0
        if not self.free and self.case.get('min_gap',0):
            for port in np.unique(ports):
                ii=np.flatnonzero(ports==port);ordered=ii[np.argsort(ages[ii])]
                for a,b in zip(ordered[:-1],ordered[1:]):
                    if ages[b]-ages[a]<self.case['min_gap']:
                        add([3*n+len(pairs)*S+extra],a*self.dim+self.dim-1,[4*span/self.max_age])
                        add([3*n+len(pairs)*S+extra],b*self.dim+self.dim-1,[-4*span/self.max_age])
                    extra+=1
        return coo_matrix((vv,(rr,cc)),shape=(3*n+len(pairs)*S+extra,n*self.dim)).tocsr()

    def refine(self,z,ports,targets,cap):
        history=[];begin=time.monotonic()
        def fun(point):
            value=self.residual(point,ports,targets)
            history.append(dict(evaluation=len(history)+1,cost=float(value@value/2),seconds=time.monotonic()-begin))
            if self.case.get('progress_path'):
                from .verification import write_json
                write_json(Path(self.case['progress_path'])/f'progress_{self.mode}.json',history[-1])
            return value
        opt=least_squares(fun,np.clip(z,1e-9,1-1e-9),jac=lambda x:self.jacobian(x,ports,targets),
                          bounds=(0,1),tr_solver='lsmr',max_nfev=cap,ftol=5e-9,xtol=5e-8,gtol=5e-8)
        return opt.x,dict(stage=self.mode,nfev=int(opt.nfev),njev=int(opt.njev or 0),cost=float(opt.cost),
                          message=str(opt.message),solver_success=bool(opt.success),elapsed_seconds=time.monotonic()-begin,
                          path_integrations=self.calls,history=history)
