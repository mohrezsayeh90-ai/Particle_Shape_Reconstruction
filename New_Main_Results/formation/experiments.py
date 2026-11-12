"""Predeclared core and extended case inventory, including unexecuted cases."""
import itertools,math
from .planner import digest
from .targets import target_points,make_ports

SEEDS=(11,29,47)


def matrix():
    cases=[]
    def add(group,stage='core',**kw):
        for seed in kw.pop('seeds',SEEDS):
            c=dict(group=group,stage=stage,shape='stadium',N=24,mode='fixed',m=3,
                   layout='equispaced',cap=None,min_age=.03,max_age=5.,min_gap=0.,
                   center=[1.52,.15],seed=seed,collision_aware=True,budget_tier='baseline_50_35',
                   coarse_nfev=50,direct_nfev=35,endpoint_nfev=24,provenance='new_experiment',cold_start=True)
            c.update(kw);c['ports']=make_ports(c)
            c['id']=f'{group}_{digest(c)[:12]}'
            cases.append(c)
    for mode,m in [('fixed',1),('fixed',3),('free',0)]:add('smoke',stage='smoke',shape='smoke',N=6,mode=mode,m=m,seeds=(11,))
    for n,m in itertools.product([24,48,72,100],range(1,7)):add('B1',N=n,m=m)
    for n in [24,48,72,100]:add('B1',N=n,mode='free',m=0)
    for n,m in itertools.product([48,100],range(1,7)):add('B2',N=n,m=m,layout='nested')
    for n in [24,48,100]:
        base=math.ceil(n/6)
        for cap in [base-1,base,base+2,2*base,None]:add('C1',N=n,cap=cap)
    for n in [6,24]:add('C1_one_shot',N=n,cap=1)
    for n,m,gap in itertools.product([24,100],[3], [.02,.04,.08]):add('C2',stage='extended',N=n,m=m,min_gap=gap)
    for shape,n,mode in itertools.product(['stadium','arrow','fish','perforated'],[24,48,72,100],['fixed3','fixed6','free']):
        add('D1',shape=shape,N=n,mode='free' if mode=='free' else 'fixed',m=0 if mode=='free' else int(mode[-1]))
    xs=[.52,.75,1.,1.25,1.50,1.75,1.95];ys=[.075,.10,.125,.15,.175,.20,.225]
    for x,y in itertools.product(xs,ys):add('E1',shape='small_stadium',N=18,center=[x,y])
    for x,y in itertools.product(xs,ys):
        if y==.15 or x in [.75,1.25,1.75]:add('E1_free',shape='small_stadium',N=18,mode='free',m=0,center=[x,y])
    for q in [1.,1.5,2.,2.75]:
        for theta in ([0] if q==1 else [-60,-30,0,30,60,90]):
            add('F1',shape='ellipse',N=12,center=[1.40,.15],q=q,theta=theta)
    for x,y in itertools.product([.85,1.40,1.85],[.115,.15,.185]):
        add('F2',stage='extended',shape='ellipse',N=12,center=[x,y],q=1.,theta=0)
        for q,theta in itertools.product([1.5,2.75],[-45,0,45,90]):
            add('F2',stage='extended',shape='ellipse',N=12,center=[x,y],q=q,theta=theta)
    for top,y,reflect in itertools.product(range(7),[.105,.15,.195],[False,True]):
        add('G1',stage='extended',shape='small_stadium',N=18,center=[1.40,y],layout='wall',top_count=top,reflected=reflect)
    for n,shape in [(6,'small_stadium'),(48,'stadium')]:
        for collision in [False,True]:add('G2',stage='extended',N=n,shape=shape,collision_aware=collision)
    for n,age in itertools.product([48,100],[2.,3.5,5.]):add('G3',stage='extended',N=n,shape='fish',m=6,max_age=age)
    # Freeze all coordinates now, before any optimizer outcome can be observed.
    for c in cases:
        points=target_points(c)
        c['target_sha256']=digest(points.tolist());c['layout_sha256']=digest(c['ports'])
        # Equal B/D requests share computations but remain visible in the index.
        scientific={k:v for k,v in c.items() if k not in ['id','group','stage','m','layout','top_count','reflected']}
        c['configuration_sha256']=digest(scientific)
    return cases


def priority(c):
    if c['stage']=='smoke':return (0,c['mode'],c['m'])
    if c['group'] in ['B1','D1'] and c['N']==100 and c['m'] in [0,3,6]:return (1,c['seed'],c['shape'],c['m'])
    if c['group']=='F1':return (2,c['seed'],c['q'],c['theta'])
    if c['group'].startswith('E1'):return (3,c['seed'],abs(c['center'][1]-.15),c['center'][0])
    return (4,c['seed'],c['group'],c['N'],c['m'])
