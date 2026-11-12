#!/usr/bin/env python3
"""Render saved campaign data without running a planner or training a network."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-formation-cache')
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Ellipse
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from PIL import Image,ImageOps,ImageDraw
import arabic_reshaper
from bidi.algorithm import get_display
from formation.verification import write_json,write_csv
from formation.targets import mask_shape,target_points,make_ports

ROOT=Path(__file__).resolve().parent;RUN=ROOT/'runs'/'formation_20261112_cpu';OUT=RUN/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'mathtext.fontset':'dejavusans','pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
COLORS={'reference':'#147d92','pinn':'#c26a1b','target':'#252525','pending':'#e6e6e6','failed':'#b45a45'}


def read_json(p):return json.loads(Path(p).read_text())
def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def caption(text):return get_display(arabic_reshaper.reshape(text))


class Figures:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=True)
        self.cases=read_json(RUN/'cases.json');self.index={c['id']:c for c in self.cases}
        self.summary_path=RUN/'figure_inputs'/'summary.csv';self.summary_path.parent.mkdir(exist_ok=True)
        self.summary_path.write_bytes((RUN/'summary.csv').read_bytes())
        self.summary=list(csv.DictReader(self.summary_path.open()))
        self.completed=[];self.manifest=[]
        for row in self.summary:
            if row['id'] not in self.index:continue
            path=RUN/'cases'/(row.get('reused_case') or row['id'])/'metrics.json'
            if path.exists():self.completed.append((self.index[row['id']],read_json(path),path.parent))

    def save(self,fig,code,title,inputs,description):
        fig.text(.5,.01,caption(title),ha='center',va='bottom',fontsize=10)
        pdf=OUT/(code+'.pdf');png=OUT/(code+'.png')
        fig.savefig(pdf,bbox_inches='tight',metadata={'Title':title,'CreationDate':None,'ModDate':None})
        fig.savefig(png,dpi=300,bbox_inches='tight');plt.close(fig)
        self.manifest.append(dict(id=code,caption_fa=title,status='rendered',description=description,
                                  plotting_script='plot_formation_campaign.py',
                                  inputs=[dict(path=str(Path(p).relative_to(ROOT)),sha256=hashfile(p)) for p in inputs],
                                  pdf=str(pdf.relative_to(ROOT)),png=str(png.relative_to(ROOT)),pdf_sha256=hashfile(pdf),png_sha256=hashfile(png)))

    def pending(self,code,reason):self.manifest.append(dict(id=code,status='pending',reason=reason))

    def f01(self):
        triangle=sorted((RUN/'historical_reruns').glob('triangle*/fresh_verification/metrics.json'))
        letters=sorted((RUN/'historical_reruns').glob('letters*/fresh_verification/*/metrics.json'))
        files=([triangle[-1]] if triangle else [])+letters
        if not files:return self.pending('F01','historical payload unavailable and no fresh historical-setting rerun completed')
        fig,axes=plt.subplots(4,2,figsize=(12,9));inputs=[]
        for i,kind in enumerate(['triangle','S','C','T']):
            candidates=[f for f in files if ('triangle' in str(f) if kind=='triangle' else f.parent.name.startswith(kind))]
            if not candidates:
                for ax in axes[i]:ax.text(.5,.5,kind+' | historical-setting rerun pending',transform=ax.transAxes,ha='center');ax.axis('off')
                continue
            file=candidates[-1];d=np.load(file.parent/'trajectories.npz');m=read_json(file);inputs.extend([file,file.parent/'trajectories.npz'])
            rows=list(csv.DictReader((file.parent/'release_schedule.csv').open()));targets=d['targets']
            for j,row in enumerate(rows):
                xy=d['reference_positions'][j];axes[i,0].plot(*xy.T,lw=.65,alpha=.7)
                axes[i,0].plot(float(row['x_launch']),float(row['y_launch']),'^' if int(row['side'])==0 else 'v',ms=4,color='black')
                if 'station_id' in row:axes[i,0].annotate(row['station_id'],(float(row['x_launch']),float(row['y_launch'])),fontsize=6)
            axes[i,0].set(xlim=(0,2.2),ylim=(0,.3),aspect='equal',xlabel=r'$x^*$',ylabel=r'$y^*$',title=kind+' | new historical-setting rerun')
            for pt in d['reference_endpoints']:axes[i,1].add_patch(Circle(pt,.0125,fc=COLORS['reference'],ec='black',lw=.4,alpha=.65))
            axes[i,1].plot(*targets.T,'+',color='black',ms=5);axes[i,1].plot(*d['pinn_endpoints'].T,'x',color=COLORS['pinn'],ms=4)
            # Actual original edge IDs are saved in the original result file.
            originals=list(file.parent.parent.glob('original_result.json'))
            if kind!='triangle':originals=list(file.parents[2].glob('six_fixed_stations_*/formations/'+file.parent.name+'/result.json'))
            for original in originals:
                record=read_json(original)
                if kind!='triangle' and not record.get('kind','').startswith(kind):continue
                inputs.append(original)
                for a,b in record.get('target_edges',[]):axes[i,1].plot(*targets[[a,b]].T,color='black',lw=.5,alpha=.5)
            axes[i,1].set(xlim=(1.6,2.18),ylim=(0,.30),aspect='equal',xlabel=r'$x^*$',ylabel=r'$y^*$',title=f"N={m['N']} | {m['status']}")
        legend=[Line2D([],[],marker='o',ls='none',mfc=COLORS['reference'],mec='black',label='Reference disks'),Line2D([],[],marker='+',color='black',ls='none',label='Targets'),Line2D([],[],marker='x',color=COLORS['pinn'],ls='none',label='Frozen PINN')]
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,frameon=False,fontsize=8)
        fig.tight_layout(rect=(0,.09,1,1))
        self.save(fig,'F01','بازاجرای تازهٔ مثلث و حروف با تنظیمات تاریخی.',inputs,'Fresh historical-setting reruns, not recovered Kaggle payloads. Missing rows are explicitly pending.')

    def f02(self):
        folders=[RUN/'historical_replays'/s for s in ['duck_six_fixed','duck_independent_wall_ports','dense_duck_six_fixed','dense_duck_independent_wall_ports']]
        folders=[p for p in folders if (p/'metrics.json').exists()]
        if len(folders)!=4:return self.pending('F02','archived schedule replay not yet complete')
        fig,axes=plt.subplots(2,2,figsize=(10,5.8));inputs=[];numbers=[]
        for ax,folder in zip(axes.flat,folders):
            data=np.load(folder/'trajectories.npz');metrics=read_json(folder/'metrics.json');inputs.extend([folder/'metrics.json',folder/'trajectories.npz'])
            target=data['targets'];ref=data['reference_endpoints'];pin=data['pinn_endpoints']
            for xy in ref:ax.add_patch(Circle(xy,.0125,fc=COLORS['reference'],ec='black',lw=.45,alpha=.65))
            ax.plot(*target.T,'+',color=COLORS['target'],ms=5,mew=.8)
            ax.plot(*pin.T,'x',color=COLORS['pinn'],ms=3,mew=.65)
            ax.set(xlim=(1.35,2.18),ylim=(0,.30),aspect='equal',xlabel=r'$x^*$',ylabel=r'$y^*$',
                   title=f"N={metrics['N']} | {'free locations' if 'independent' in folder.name else 'six fixed ports'}")
            ax.grid(alpha=.16)
            numbers.append(dict(case=folder.name,N=metrics['N'],accepted=metrics['accepted'],
                         pinn_max_error=metrics['replays']['pinn']['max_error'],reference_max_error=metrics['replays']['reference']['max_error'],
                         pinn_min_distance=metrics['replays']['pinn']['minimum']['distance'],reference_min_distance=metrics['replays']['reference']['minimum']['distance']))
        legend=[Line2D([],[],marker='o',ls='none',mfc=COLORS['reference'],mec='black',label='Reference disks'),
                Line2D([],[],marker='+',color='black',ls='none',label='Targets'),Line2D([],[],marker='x',color=COLORS['pinn'],ls='none',label='Frozen PINN')]
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.06),ncol=3,frameon=False)
        fig.subplots_adjust(bottom=.20,hspace=.45,wspace=.25)
        write_csv(RUN/'tables'/'F02_archived_ducks.csv',numbers)
        self.save(fig,'F02','مقایسهٔ آرایش اردک در زمان هدف مشترک.',inputs,
                  'Fresh numerical replays of four archived schedules; fixed/free dense panels have different N.')

    def f03(self):
        fig,axes=plt.subplots(1,3,figsize=(12,4.1),gridspec_kw={'width_ratios':[1.1,1.1,1]});inputs=[]
        groups=[c for c in self.cases if c['group']=='B1'];statuses={r['id']:r for r in self.summary}
        image=np.full((4,7),np.nan);margin=np.full_like(image,np.nan);error=np.full_like(image,np.nan)
        for i,n in enumerate([24,48,72,100]):
            for j,m in enumerate([1,2,3,4,5,6,0]):
                rr=[statuses[c['id']] for c in groups if c['N']==n and c['m']==m]
                attempted=[r for r in rr if r['status'] in ('accepted','rejected')]
                if attempted:
                    image[i,j]=sum(r['accepted']=='True' for r in attempted)/len(attempted)
                    vals=[float(r['clearance_margin']) for r in attempted if r['clearance_margin']]
                    if vals:margin[i,j]=max(vals)
                    vals=[float(r['reference_max_error']) for r in attempted if r['reference_max_error']]
                    if vals:error[i,j]=min(vals)
                for ax in axes:ax.text(j,i,f'{len(attempted)}/3' if attempted else 'pending',ha='center',va='center',fontsize=7,bbox=dict(fc='white',ec='none',alpha=.65,pad=.5))
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
        im=axes[0].imshow(image,vmin=0,vmax=1,cmap=cmap,aspect='auto');fig.colorbar(im,ax=axes[0],label='Pass fraction among executed starts')
        if np.isfinite(margin).any():
            im=axes[1].imshow(margin,cmap='coolwarm',aspect='auto');fig.colorbar(im,ax=axes[1],label=r'$d_{\min}^*-0.026$')
        else:axes[1].imshow(np.full_like(image,np.nan),cmap=cmap,aspect='auto')
        im=axes[2].imshow(error,cmap=cmap,aspect='auto');fig.colorbar(im,ax=axes[2],label='Best reference max endpoint error')
        for ax in axes:ax.set(xticks=range(7),xticklabels=['2','4','6','8','10','12','free'],yticks=range(4),yticklabels=[24,48,72,100],xlabel='Available ports',ylabel='N')
        axes[0].set_title('Acceptance | executed / planned');axes[1].set_title('Best refined reference margin');axes[2].set_title('Best max endpoint error')
        fig.tight_layout(rect=(0,.12,1,1))
        inputs=[self.summary_path,RUN/'cases.json']
        self.save(fig,'F03','اثر تعداد پرتابگرها و ذرات بر کیفیت آرایش.',inputs,'Pending cells are explicitly labelled; counts are executed starts out of three, not completed matched comparisons. Best errors and margins may come from different starts.')
        fig,ax=plt.subplots(figsize=(7,4))
        for layout,color,offset in [('equispaced','black',0),('nested',COLORS['reference'],7)]:
            for m in range(1,7):
                ports=make_ports(dict(m=m,layout=layout));xs=[p['x'] for p in ports if p['side']==0]
                ax.plot(xs,np.full(m,m+offset),'o',ms=3.5,color=color)
        ax.set(xlabel=r'$x^*$',ylabel='m per wall (lower: equispaced; upper: nested)',xlim=(0,2.2),yticks=[1,3,6,8,10,13],yticklabels=['1','3','6','1','3','6'],title='Port coordinates | mirrored walls')
        fig.tight_layout(rect=(0,.14,1,1));self.save(fig,'F03_ports','مختصات ثابت پرتابگرها در چیدمان هم‌فاصله و تودرتو.',[RUN/'cases.json'],'Configured port coordinates, independent of optimizer outcomes. Equispaced layouts are not nested.')

    def f05(self):
        fig,axes=plt.subplots(4,3,figsize=(10,9));inputs=[]
        for i,shape in enumerate(['stadium','arrow','fish','perforated']):
            for j,n in enumerate([24,72,100]):
                ax=axes[i,j];subset=[v for v in self.completed if v[0]['group']=='D1' and v[0]['shape']==shape and v[0]['N']==n]
                best=min(subset,key=lambda v:(not v[1].get('accepted',False),v[1].get('replays',{}).get('reference',{}).get('max_error',1e30))) if subset else None
                outline=mask_shape(shape)
                for geom in ([outline] if outline.geom_type=='Polygon' else outline.geoms):
                    ax.plot(*geom.exterior.xy,color='0.65',lw=.8)
                    for hole in geom.interiors:ax.plot(*hole.xy,color='0.65',lw=.8)
                ax.set(xlim=(1.03,2.01),ylim=(0,.30),aspect='equal',xlabel=r'$x^*$',ylabel=r'$y^*$')
                if best and (best[2]/'trajectories.npz').exists():
                    c,m,p=best;data=np.load(p/'trajectories.npz');points=data['reference_endpoints']
                    for xy in points:ax.add_patch(Circle(xy,.0125,fc=COLORS['reference'] if m['accepted'] else '#dba793',ec='black',lw=.35))
                    ax.plot(*data['targets'].T,'+',ms=2,color='black')
                    ax.set_title(f"{shape} N={n} | {c['mode']} m={c['m']}\n{m['status']}",fontsize=8)
                    inputs.extend([p/'metrics.json',p/'trajectories.npz'])
                else:
                    ax.set_title(f'{shape} N={n} | pending',fontsize=8)
                    pts=target_points(dict(shape=shape,N=n));ax.plot(*pts.T,'o',mfc='none',mec='0.75',ms=2)
        handles=[Line2D([],[],marker='o',ls='none',mfc=COLORS['reference'],mec='black',label='Reference disks'),Line2D([],[],marker='+',ls='none',color='black',label='Targets'),Line2D([],[],marker='o',ls='none',mfc='none',mec='0.7',label='Pending targets only')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,frameon=False,fontsize=8)
        fig.tight_layout(rect=(0,.09,1,1));self.save(fig,'F05','آرایش‌های هدف پُر و وضعیت نتایج محاسبه‌شده.',inputs+[self.summary_path],'Actual reference disks for executed cases; hollow target-only panels are pending, not simulated formations.')

        measured=[v for v in self.completed if v[0]['group']=='D1' and 'occupancy' in v[1]]
        if measured:
            fig,axes=plt.subplots(1,3,figsize=(12,3.8))
            for shape,color in zip(['stadium','arrow','fish','perforated'],['#147d92','#c26a1b','#8b4789','#448a35']):
                for c,m,p in measured:
                    if c['shape']!=shape:continue
                    for ax,key in zip(axes,['inside_coverage','overspill','hole_intrusion_area']):
                        ax.plot(c['N'],m['occupancy'][key],'o' if m['accepted'] else 'x',color=color,ms=4)
                axes[0].plot([],[],'o',color=color,label=shape)
            for ax,key in zip(axes,['Inside area / mask area','Outside disk area / disk union','Hole intrusion area']):ax.set(xlabel='N',ylabel=key);ax.grid(alpha=.15)
            axes[0].legend(frameon=False,fontsize=7);fig.tight_layout(rect=(0,.13,1,1))
            self.save(fig,'F05_occupancy','پوشش، بیرون‌زدگی و نفوذ به حفره‌های ماسک.',[p/'metrics.json' for c,m,p in measured],'Actual polygon union of physical-radius disks; circles are accepted and crosses rejected. Two polygon resolutions are saved to assess convergence.')

    def f04(self):
        subset=[v for v in self.completed if v[0]['group'] in ['C1','C1_one_shot']]
        if not subset:return self.pending('F04','no emission-cap case has a computed schedule')
        fig,axes=plt.subplots(1,3,figsize=(13,4));data=np.full((3,5),np.nan)
        statuses={r['id']:r for r in self.summary}
        for i,n in enumerate([24,48,100]):
            base=math.ceil(n/6)
            for j,cap in enumerate([base-1,base,base+2,2*base,None]):
                rows=[statuses[c['id']] for c in self.cases if c['group']=='C1' and c['N']==n and c['cap']==cap]
                done=[r for r in rows if r['status'] in ['accepted','rejected']]
                if rows and all(r['status']=='excluded_before_solve' for r in rows):axes[0].text(j,i,'excluded\ncapacity',ha='center',va='center',fontsize=6)
                elif done:data[i,j]=sum(r['accepted']=='True' for r in done)/len(done);axes[0].text(j,i,f'{len(done)}/3',ha='center',va='center',fontsize=7)
                else:axes[0].text(j,i,'pending',ha='center',va='center',fontsize=6)
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee');im=axes[0].imshow(data,aspect='auto',vmin=0,vmax=1,cmap=cmap);fig.colorbar(im,ax=axes[0],label='Pass fraction, executed starts')
        axes[0].set(xticks=range(5),xticklabels=['b−1','b','b+2','2b','unlimited'],yticks=range(3),yticklabels=[24,48,100],xlabel='Per-port cap (b=ceil(N/6))',ylabel='N',title='Configured capacity')
        candidates=[v for v in subset if v[1].get('accepted')]
        selected=max(candidates or subset,key=lambda v:(v[0]['cap'] is not None,v[0]['N']))
        c,m,p=selected;rows=list(csv.DictReader((p/'release_schedule.csv').open()));ids=sorted(set(r['port_id'] for r in rows))
        axes[1].bar(ids,[sum(r['port_id']==port for r in rows) for port in ids],color=COLORS['reference']);axes[1].tick_params(axis='x',rotation=45)
        axes[1].set(ylabel='Actual emissions',title=f"N={c['N']}, cap={c['cap']} | {m['status']}")
        for i,port in enumerate(ids):axes[2].scatter([float(r['release_time']) for r in rows if r['port_id']==port],np.full(sum(r['port_id']==port for r in rows),i),s=14,marker='|',color=COLORS['reference'])
        axes[2].set(yticks=range(len(ids)),yticklabels=ids,xlabel=r'$t^*$',ylabel='Port ID',title='Saved release raster')
        fig.tight_layout(rect=(0,.13,1,1));self.save(fig,'F04','سقف انتشار، بار واقعی پرتابگرها و زمان‌بندی رهاسازی.',[self.summary_path,p/'release_schedule.csv',p/'metrics.json'],'Structural exclusions are annotated outside measured colors. Actual loads and release times are read from the selected saved schedule.')

    def f06(self):
        data=[v for v in self.completed if v[0]['group'] in ['E1','E1_free']]
        if not data:return self.pending('F06','center-grid solves have not completed')
        xs=[.52,.75,1.,1.25,1.5,1.75,1.95];ys=[.075,.10,.125,.15,.175,.20,.225]
        fig,axes=plt.subplots(2,3,figsize=(14,7))
        for col,metric in enumerate(['pass','error','margin']):
            measured=[float(m['accepted']) if metric=='pass' else m['replays']['reference']['max_error'] if metric=='error' else m['replays']['reference']['minimum']['distance']-.026 for c,m,p in data if m.get('replays',{}).get('reference',{}).get('minimum',{}).get('distance') is not None]
            low,high=(0,1) if metric=='pass' else (min(measured),max(measured))
            if low==high:high=low+1e-8
            for ax,mode in zip(axes[:,col],['fixed','free']):
                a=np.full((7,7),np.nan)
                for i,y in enumerate(ys):
                    for j,x in enumerate(xs):
                        subset=[v for v in data if v[0]['center']==[x,y] and v[0]['mode']==mode and 'replays' in v[1]]
                        if subset:
                            values=[float(v[1]['accepted']) if metric=='pass' else v[1]['replays']['reference']['max_error'] if metric=='error' else v[1]['replays']['reference']['minimum']['distance']-.026 for v in subset]
                            a[i,j]=np.mean(values) if metric=='pass' else np.median(values)
                        ax.text(x,y,f'{len(subset)}/3' if subset else '—',ha='center',va='center',fontsize=6,bbox=dict(fc='white',ec='none',alpha=.7,pad=.5))
                xe=np.r_[xs[0]-(xs[1]-xs[0])/2,(np.array(xs[:-1])+xs[1:])/2,xs[-1]+(xs[-1]-xs[-2])/2]
                ye=np.r_[ys[0]-.0125,(np.array(ys[:-1])+ys[1:])/2,ys[-1]+.0125]
                ax.set_facecolor('#eeeeee');im=ax.pcolormesh(xe,ye,a,cmap='coolwarm' if metric=='margin' else 'viridis',vmin=low,vmax=high,shading='flat')
                ax.set(xlabel=r'$x_c^*$',ylabel=r'$y_c^*$',title=mode+' | '+{'pass':'pass fraction','error':'median endpoint error','margin':'median clearance margin'}[metric])
                fig.colorbar(im,ax=ax,shrink=.8)
        fig.text(.5,.055,'Labels show executed starts / 3; blank cells are pending.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.09,1,1))
        self.save(fig,'F06','اثر مرکز هدف بر پذیرش، خطا و فاصلهٔ کمینه.',[self.summary_path],'Counts are executed/planned starts. Missing cells are blank; errors/margins use medians among executed starts, with shared fixed/free color scales and true grid coordinates.')
        fig,axes=plt.subplots(2,4,figsize=(14,6));cuts=[('axial',None),('transverse',.75),('transverse',1.25),('transverse',1.75)]
        for col,(direction,fixed) in enumerate(cuts):
            for mode in ['fixed','free']:
                for seed,color in zip([11,29,47],['#147d92','#c26a1b','#8b4789']):
                    subset=[v for v in data if v[0]['mode']==mode and v[0]['seed']==seed and (v[0]['center'][1]==.15 if direction=='axial' else v[0]['center'][0]==fixed) and 'replays' in v[1]]
                    subset.sort(key=lambda v:v[0]['center'][0 if direction=='axial' else 1])
                    if not subset:continue
                    coords=[v[0]['center'][0 if direction=='axial' else 1] for v in subset]
                    for row,metric in enumerate(['error','margin']):
                        values=[v[1]['replays']['reference']['max_error'] if metric=='error' else v[1]['replays']['reference']['minimum']['distance']-.026 for v in subset]
                        axes[row,col].plot(coords,values,'o-' if mode=='fixed' else 'x--',color=color,ms=3,lw=.7,label=f'{mode} seed {seed}')
            for row in range(2):axes[row,col].set(xlabel=r'$x_c^*$' if direction=='axial' else r'$y_c^*$',ylabel='Reference endpoint error' if row==0 else 'Reference clearance margin',title='y*=0.15' if direction=='axial' else f'x*={fixed}');axes[row,col].grid(alpha=.15)
        handles,labels=axes[0,0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=7)
        fig.tight_layout(rect=(0,.13,1,1));self.save(fig,'F06_slices','برش‌های طولی و عرضی مرکز هدف با نمایش جداگانهٔ seedها.',[self.summary_path],'Only executed starts are plotted; lines connect sampled centers and do not interpolate feasibility guarantees.')

    def f07(self):
        data=[v for v in self.completed if v[0]['group']=='F1' and 'covariance' in v[1]]
        if not data:return self.pending('F07','central ellipse solves have not completed')
        examples=[(1.,0),(1.5,0),(2.,30),(2.75,-60),(2.75,0),(2.75,90)]
        selected=[next((v for v in data if (v[0]['q'],v[0]['theta'],v[0]['seed'])==(q,t,11)),None) for q,t in examples]
        fig,axes=plt.subplots(3,3,figsize=(12,10));inputs=[]
        for ax,item in zip(axes[:2].flat,selected):
            if item is None:ax.text(.5,.5,'pending',ha='center',transform=ax.transAxes);ax.axis('off');continue
            c,m,p=item;d=np.load(p/'trajectories.npz');t=d['targets'];a=d['reference_endpoints']
            ax.plot(*t.T,'o',mfc='none',mec='black',ms=4,label='Target');ax.plot(*a.T,'x',color=COLORS['reference'],ms=4,label='Reference')
            for points,style,color in [(t,'--','black'),(a,'-',COLORS['reference'])]:
                mean=points.mean(0);cov=(points-mean).T@(points-mean)/len(points);val,vec=np.linalg.eigh(cov)
                angle=np.rad2deg(np.arctan2(vec[1,-1],vec[0,-1]))
                ax.add_patch(Ellipse(mean,2*np.sqrt(val[-1]),2*np.sqrt(max(val[0],0)),angle=angle,fill=False,ls=style,ec=color,lw=1))
            orientation='angle undefined' if c['q']==1 else f"θ={c['theta']}°"
            ax.set(xlim=(1.20,1.60),ylim=(0,.30),aspect='equal',xlabel=r'$x^*$',ylabel=r'$y^*$',title=f"q={c['q']} {orientation} | {m['status']}")
            delta=a.mean(0)-t.mean(0);length=float(np.linalg.norm(delta))
            ax.arrow(*t.mean(0),*delta,width=0,head_width=length/3,head_length=length/2,length_includes_head=True,color='#a2345b',lw=.6)
            ax.text(.02,.96,f'centroid shift={length:.2e}',transform=ax.transAxes,va='top',fontsize=6)
            inputs.extend([p/'metrics.json',p/'trajectories.npz'])
        for ax,metric,title in zip(axes[2],['covariance','endpoint','margin'],['Median covariance relative error','Median reference endpoint error','Median reference clearance margin']):
            values=np.full((4,6),np.nan)
            for i,q in enumerate([1.,1.5,2.,2.75]):
                for j,theta in enumerate([-60,-30,0,30,60,90]):
                    subset=[v for v in data if v[0]['q']==q and v[0]['theta']==theta]
                    if not subset:continue
                    result=[v[1]['covariance']['covariance_relative_frobenius'] if metric=='covariance' else v[1]['replays']['reference']['max_error'] if metric=='endpoint' else v[1]['replays']['reference']['minimum']['distance']-.026 for v in subset]
                    values[i,j]=np.median(result);ax.text(j,i,f'{len(subset)}/3',ha='center',va='center',fontsize=6,color='black',bbox=dict(fc='white',alpha=.7,ec='none',pad=1))
            im=ax.imshow(values,aspect='auto',cmap='viridis' if metric!='margin' else 'coolwarm');fig.colorbar(im,ax=ax,shrink=.8)
            ax.set(xticks=range(6),xticklabels=[-60,-30,0,30,60,90],yticks=range(4),yticklabels=[1,1.5,2,2.75],xlabel=r'$\theta$ (degrees)',ylabel='q',title=title)
        inputs.extend(p/'metrics.json' for c,m,p in data)
        handles=[Line2D([],[],marker='o',mfc='none',mec='black',ls='none',label='Target centers'),Line2D([],[],marker='x',color=COLORS['reference'],ls='none',label='Reference centers'),Line2D([],[],ls='--',color='black',label='Target covariance'),Line2D([],[],color=COLORS['reference'],label='Achieved covariance')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=4,frameon=False,fontsize=8)
        fig.text(.5,.065,'Clouds: seed 11. Map labels: executed starts / 3; maps include rejected starts.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.10,1,1));self.save(fig,'F07','مقایسهٔ کوواریانس هدف و آرایش حاصل و نقشهٔ هندسهٔ بیضی.',list(dict.fromkeys(inputs)),'Clouds show seed 11; maps include all executed starts, with executed/planned counts. One-sigma covariance ellipses, actual centroid arrows without amplification; isotropic q=1 has no orientation error.')

    def f09(self):
        folders=[RUN/'historical_replays'/n for n in ['dense_duck_six_fixed','dense_duck_independent_wall_ports']];inputs=[]
        for shape in ['arrow','fish']:
            candidates=[v for v in self.completed if v[0]['group']=='D1' and v[0]['shape']==shape and v[1].get('accepted')]
            if candidates:folders.append(max(candidates,key=lambda v:(v[0]['N'],v[2].name))[2])
        folders=[p for p in folders if (p/'trajectories.npz').exists()]
        if not folders:return self.pending('F09','no completed dense replays')
        fig,axes=plt.subplots(len(folders),4,figsize=(14,2.7*len(folders)),squeeze=False,gridspec_kw={'width_ratios':[2.5,1,1,1]})
        for row,p in zip(axes,folders):
            ax=row[0];d=np.load(p/'trajectories.npz');m=read_json(p/'metrics.json')
            for mode,style in [('reference','-'),('pinn','--')]:
                ax.plot(d['time'],d[mode+'_sampled_dmin'],style,color=COLORS[mode],label=mode,lw=1)
                mm=m['replays'][mode]['minimum'];ax.plot(mm['time'],mm['distance'],'o',mfc='none',mec=COLORS[mode],ms=5)
            ax.axhline(.026,color='black',ls=':',lw=1,label='acceptance 0.026')
            ax.set(xlabel=r'$t^*$',ylabel=r'$d_{\min}^*(t^*)$',title=f"N={m['N']} | refined reference pair {m['replays']['reference']['minimum']['pair']}",ylim=(.023,.09))
            count=ax.twinx();count.step(d['time'],d['reference_active'],where='post',color='0.6',alpha=.35,lw=.7);count.set_ylabel('Active particles',color='0.5',fontsize=7);count.tick_params(axis='y',labelsize=6,colors='0.5')
            ax.grid(alpha=.15);inputs.extend([p/'metrics.json',p/'trajectories.npz'])
            if (p/'critical_event_snapshots.npz').exists():
                event=np.load(p/'critical_event_snapshots.npz');pair=event['reference_pair'].astype(int)-1
                middle=event['reference_positions'][1,pair].mean(0)
                for k,snapshot in enumerate(row[1:]):
                    pts=event['reference_positions'][k]
                    for i,xy in enumerate(pts):
                        if np.isfinite(xy).all():snapshot.add_patch(Circle(xy,.0125,fc=COLORS['reference'] if i in pair else '0.8',ec='black' if i in pair else 'none',alpha=.65,lw=.5))
                    snapshot.plot(*pts[pair].T,'o-',color='#ac3458',ms=2,lw=.8)
                    for i in pair:snapshot.annotate(str(i+1),pts[i],fontsize=6)
                    title=['before','refined minimum','after'][k]
                    if k==2 and abs(event['reference_times'][2]-event['reference_times'][1])<1e-10:title='window ends at minimum'
                    if k==0 and abs(event['reference_times'][0]-event['reference_times'][1])<1e-10:title='shared lifetime starts'
                    snapshot.set(xlim=(middle[0]-.055,middle[0]+.055),ylim=(max(0,middle[1]-.055),min(.3,middle[1]+.055)),aspect='equal',xlabel=r'$x^*$',ylabel=r'$y^*$',title=title+f"\nt*={event['reference_times'][k]:.5f}")
                    snapshot.tick_params(labelsize=6)
                inputs.extend([p/'critical_event_snapshots.npz',p/'critical_event.json'])
            else:
                for snapshot in row[1:]:snapshot.text(.5,.5,'Event replay pending',ha='center',transform=snapshot.transAxes);snapshot.axis('off')
        axes[0,0].legend(ncol=3,frameon=False,loc='upper right',fontsize=6)
        fig.tight_layout(rect=(0,.04,1,1));self.save(fig,'F09','فاصلهٔ کمینه، تعداد ذرات فعال و سازوکار نزدیک‌ترین جفت.',inputs,'Lines use saved samples; open circles show refined all-pair minima. Snapshots are exact independent replays at the critical event. No trajectories are extrapolated beyond shared lifetimes.')

    def f12(self):
        files=sorted((RUN/'robustness').glob('*/metrics.json'))
        if not files:return self.pending('F12','perturbed release scenarios have not completed')
        data=[read_json(p) for p in files];fig,axes=plt.subplots(2,1,figsize=(12,7));labels=[]
        kinds=[('nominal',0)]+[(k,l) for k in ['position','axial_velocity','transverse_velocity','timing','joint','port_misplacement'] for l in [1,2]]
        for ax,label in zip(axes,['low_margin','high_margin']):
            for i,(kind,level) in enumerate(kinds):
                subset=[m for m in data if m['selection_label']==label and m['scenario']==kind and m['level']==level]
                if subset:
                    m=subset[0];ax.bar(i-.16,m['unconditional_pass_fraction'],width=.30,color=COLORS['reference'])
                    if m['bounds_conditional_pass_fraction'] is not None:ax.bar(i+.16,m['bounds_conditional_pass_fraction'],width=.30,fc='none',ec=COLORS['pinn'],lw=1)
                    ax.text(i,1.025,f"{m['accepted']}/64\nvalid={m['within_bounds']}",ha='center',fontsize=6)
                else:ax.text(i,.15,'pending',rotation=90,ha='center',fontsize=7,color='0.45')
            ax.set(ylim=(0,1.16),ylabel='Numerical pass fraction',title=label,xticks=range(len(kinds)),xticklabels=[k.replace('_',' ')+f' {l}' for k,l in kinds])
            ax.tick_params(axis='x',rotation=28,labelsize=7);ax.grid(axis='y',alpha=.15)
        axes[0].plot([],[],color=COLORS['reference'],lw=6,label='All 64 draws');axes[0].plot([],[],color=COLORS['pinn'],label='Conditional on bounds');axes[0].legend(frameon=False,loc='lower left')
        fig.tight_layout(rect=(0,.06,1,1));self.save(fig,'F12','حساسیت عددی برنامهٔ ثابت به خطاهای رهاسازی.',files+[RUN/'robustness'/'selection.json'],'64 seeded draws for each completed scenario; invalid draws retained. Reference-only threshold 0.004; fixed nominal T. This is numerical sensitivity, not hardware reliability.')

    def f10(self):
        file=RUN/'reflection'/'comparison.json'
        if not file.exists():return self.pending('F10','paired schedule reflection and wall-availability runs pending')
        comparison=read_json(file);source=RUN/'cases'/comparison['source_case'];base=np.load(source/'trajectories.npz');mirrored=np.load(file.parent/'trajectories.npz')
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,mode in zip(axes,['reference','pinn']):
            expected=base[mode+'_endpoints'].copy();expected[:,1]=.3-expected[:,1]
            residual=mirrored[mode+'_endpoints']-expected
            ax.scatter(expected[:,1],mirrored[mode+'_endpoints'][:,1],s=8,color=COLORS[mode])
            ax.plot([0,.3],[0,.3],':',color='black',lw=.8)
            ax.set(xlim=(0,.3),ylim=(0,.3),aspect='equal',xlabel='Reflected nominal y*',ylabel='Replayed mirrored y*',title=mode+f" | max residual {np.linalg.norm(residual,axis=1).max():.2e}")
        axes[2].text(.5,.5,'Wall-availability sweep\npending\n\nReflection uses a saved schedule;\nno optimizer rerun.',ha='center',va='center',transform=axes[2].transAxes);axes[2].axis('off')
        fig.tight_layout(rect=(0,.13,1,1));self.save(fig,'F10','بررسی عددی تقارن با بازتاب یک برنامهٔ پذیرفته‌شده.',[file,file.parent/'trajectories.npz',source/'trajectories.npz'],'Paired saved-schedule reflection separates exact reference symmetry from frozen-PINN asymmetry. Wall-availability optimizer comparisons remain pending.')

    def f11(self):
        data=[v for v in self.completed if v[0]['group'] in ['G2','G3'] and 'replays' in v[1]]
        if not data:return self.pending('F11','deadline and ablation runs pending')
        fig,axes=plt.subplots(1,3,figsize=(13,4));inputs=[]
        for c,m,p in data:
            ref=m['replays']['reference'];distance=ref['minimum']['distance']
            if distance is None:continue
            if c['group']=='G2':
                axes[0].scatter(ref['max_error'],distance-.026,marker='o' if c['collision_aware'] else 'x',color=COLORS['reference'] if c['collision_aware'] else COLORS['pinn'],s=35)
                axes[0].annotate(f"N={c['N']} s{c['seed']}",(ref['max_error'],distance-.026),fontsize=6,xytext=(3,3),textcoords='offset points')
            else:axes[2].plot(m['target_time'],distance-.026,'o' if m['accepted'] else 'x',label=f"N={c['N']} max age={c['max_age']}")
            inputs.append(p/'metrics.json')
        axes[0].axhline(0,color='black',ls=':',lw=.8);axes[0].axvline(.004,color='black',ls=':',lw=.8)
        axes[0].set(xlabel='Reference max endpoint error',ylabel='Reference clearance margin',title='Ablation | x: endpoint only; o: clearance')
        axes[0].set_xscale('log')
        chosen=[v for v in data if v[0]['group']=='G2' and not v[0]['collision_aware'] and (v[2]/'critical_event_snapshots.npz').exists()]
        if chosen:
            c,m,p=min(chosen,key=lambda v:v[1]['replays']['reference']['minimum']['distance']);event=np.load(p/'critical_event_snapshots.npz');pair=event['reference_pair'].astype(int)-1;points=event['reference_positions'][1];middle=points[pair].mean(0)
            for i,xy in enumerate(points):
                if np.isfinite(xy).all():axes[1].add_patch(Circle(xy,.0125,fc=COLORS['pinn'] if i in pair else '0.8',ec='black',alpha=.6,lw=.4))
            axes[1].plot(*points[pair].T,'o-',color='black',ms=2,lw=.7)
            axes[1].set(xlim=(middle[0]-.065,middle[0]+.065),ylim=(max(0,middle[1]-.065),min(.3,middle[1]+.065)),aspect='equal',xlabel=r'$x^*$',ylabel=r'$y^*$',title=f"Endpoint only | critical pair {[int(i+1) for i in pair]}")
            inputs.extend([p/'critical_event_snapshots.npz',p/'critical_event.json'])
        else:axes[1].text(.5,.5,'Critical event replay pending',ha='center',transform=axes[1].transAxes);axes[1].axis('off')
        if any(v[0]['group']=='G3' for v in data):
            axes[2].set(xlabel='Achieved T*',ylabel='Reference clearance margin',title='Deadline data | partial sweep');axes[2].legend(fontsize=6)
            axes[2].text(.02,.98,'Only executed/reused starts are shown.\nOther bounds/seeds remain pending.',transform=axes[2].transAxes,va='top',fontsize=6)
        else:axes[2].text(.5,.5,'Fish deadline sweep\npending',ha='center',transform=axes[2].transAxes);axes[2].axis('off')
        fig.tight_layout(rect=(0,.13,1,1));self.save(fig,'F11','اثر هزینهٔ فاصله بر برنامه‌ریزی و وضعیت پیمایش زمان هدف.',inputs,'Endpoint-only removes clearance in allocation and refinement, but is verified by the same strict full acceptance test. Missing deadline cases remain pending.')

    def tables(self):
        groups={}
        for c,m,p in self.completed:
            key=(c['group'],c['shape'],c['N'],c['mode'],c['m'],str(c.get('cap')),tuple(c['center']),c.get('q'),c.get('theta'),c['budget_tier'],c['max_age'],c['min_gap'],c['collision_aware'],c['layout'])
            groups.setdefault(key,[]).append((c,m,p))
        aggregate=[]
        for key,values in groups.items():
            good=[v for v in values if v[1]['accepted']];errors=[v[1].get('replays',{}).get('reference',{}).get('max_error') for v in values]
            errors=[x for x in errors if x is not None]
            aggregate.append(dict(group=key[0],shape=key[1],N=key[2],mode=key[3],m=key[4],cap=key[5],center=str(key[6]),q=key[7],theta=key[8],budget_tier=key[9],
                                  max_age=key[10],minimum_emission_gap=key[11],collision_aware=key[12],layout=key[13],
                                  executed_starts=len(values),reused_starts=sum(p.name!=c['id'] for c,m,p in values),planned_starts=1 if key[0]=='smoke' else 3,accepted_starts=len(good),matched_three_seed_complete=len(values)==3 and len({v[0]['seed'] for v in values})==3,
                                  pass_fraction_executed=len(good)/len(values),best_reference_error=min(errors) if errors else None))
        write_csv(RUN/'tables'/'aggregate_comparisons.csv',aggregate)
        covariance=[]
        for c,m,p in self.completed:
            if c['group'] not in ['F1','F2'] or 'covariance' not in m:continue
            covariance.append(dict(id=c['id'],group=c['group'],q=c['q'],theta=c['theta'],center_x=c['center'][0],center_y=c['center'][1],seed=c['seed'],accepted=m['accepted'],**m['covariance']))
        write_csv(RUN/'tables'/'ellipse_covariance.csv',covariance)
        highest={}
        for c,m,p in self.completed:
            if m['accepted']:
                key=(c['shape'],c['mode']);highest[key]=max(highest.get(key,0),c['N'])
        write_csv(RUN/'tables'/'highest_verified_counts.csv',[dict(shape=s,mode=mode,N=n) for (s,mode),n in sorted(highest.items())])
        # Compact publication table: rows never convert a missing result into zero error.
        tex=['\\begin{tabular}{p{11cm}rr}','Case family & Accepted & Executed/planned \\\\','\\hline']
        for r in aggregate:
            label=', '.join(f'{k}={r[k]}' for k in ['group','shape','N','mode','m','layout','cap','center','q','theta','max_age','minimum_emission_gap','collision_aware','budget_tier'])
            label=label.replace('_',r'\_')
            tex.append(f"{label} & {r['accepted_starts']} & {r['executed_starts']}/{r['planned_starts']} \\\\")
        tex.append('\\end{tabular}');(RUN/'tables'/'aggregate_comparisons.tex').write_text('\n'.join(tex)+'\n')

    def inventory(self):
        main=[r for r in self.summary if r['id'] in self.index]
        counts={s:sum(r['status']==s for r in main) for s in ['accepted','rejected','pending','running','excluded_before_solve']}
        actual={str(p):(c,m,p) for c,m,p in self.completed}
        actual_counts={s:sum(m['status']==s for c,m,p in actual.values()) for s in ['accepted','rejected']}
        lines=['# موجودی نتایج جدید تشکیل آرایش','',
               'این بسته شامل نتایج محاسبه‌شده و فهرست صریح اجراهای باقی‌مانده است.',
               'پذیرش هر حالت با دو بازپخش مستقل و بررسی همهٔ جفت‌ها تعیین می‌شود.',
               'متن پروپوزال و PDF قبلی در این مرحله تغییر نکرده‌اند؛ فقط همین پوشه بین دو مخزن کپی می‌شود.','',
               f"تعداد اجراهای جدید مستقل: **{len(actual)}**؛ پذیرفته‌شده: **{actual_counts['accepted']}**؛ نپذیرفته‌شده: **{actual_counts['rejected']}**.",
               'این شمارش، بازپخش چهار اردک تاریخی و بازاجراهای مثلث و حروف را شامل نمی‌شود.',
               'شمارش جدول زیر مربوط به همهٔ درخواست‌های ماتریس است و درخواست‌های هم‌ارزِ بازاستفاده‌شده را نیز دربرمی‌گیرد.','',
               '| وضعیت | تعداد |','| --- | ---: |']
        for s,n in counts.items():lines.append(f'| {s} | {n} |')
        lines+=['','[گزارش کوتاه](EXPERIMENT_REPORT_FA.md) · [فهرست همهٔ حالت‌ها و وضعیت آن‌ها](runs/formation_20261112_cpu/cases_manifest.csv) · [خلاصهٔ ادغام‌شده](runs/formation_20261112_cpu/summary.csv) · [روش بازتولید و ادامهٔ اجرا](REPRODUCE.md)','',
                '## بالاترین تعداد تأییدشده در اجراهای جدید','',
                '| ماسک | نوع رهاسازی | تعداد ذره |','| --- | --- | ---: |']
        highest=list(csv.DictReader((RUN/'tables/highest_verified_counts.csv').open()))
        for r in highest:lines.append(f"| {r['shape']} | {r['mode']} | {r['N']} |")
        lines+=['','این‌ها بالاترین تعداد **تأییدشده در دادهٔ موجود** هستند؛ حد ظرفیت فیزیکی یا تعداد بهینه نیستند.','',
                '## مقایسه‌های کامل سه-seed','',
                '| گروه | مقایسه‌های کامل | پذیرفته / انجام‌شده | شروع‌های بازاستفاده‌شده |','| --- | ---: | ---: | ---: |']
        aggregate=list(csv.DictReader((RUN/'tables/aggregate_comparisons.csv').open()))
        for group in sorted({r['group'] for r in aggregate}):
            complete=[r for r in aggregate if r['group']==group and r['matched_three_seed_complete']=='True']
            if complete:lines.append(f"| {group} | {len(complete)} | {sum(int(r['accepted_starts']) for r in complete)} / {sum(int(r['executed_starts']) for r in complete)} | {sum(int(r['reused_starts']) for r in complete)} |")
        lines+=['','بقیهٔ مقایسه‌ها غربالگری یا اجراهای ناقص‌اند؛ هر برنامهٔ پذیرفته‌شده، حتی در غربالگری، آزمون سخت‌گیرانهٔ مستقل را گذرانده است.',
                'بودجهٔ پایهٔ جدید: سقف ۲۴ ارزیابی برازش اولیه و ۵۰/۳۵ ارزیابی مراحل مشترک؛ سه seed برابر 11، 29 و 47.',
                'پراکندگی seedها حساسیت جست‌وجو است و عدم‌قطعیت اندازه‌گیری تجربی محسوب نمی‌شود.','',
                '## شواهد تاریخی و بازاجراها','',
                'چهار برنامهٔ اردک 20/28/42/64 از آرشیو، با هر دو حل‌گر دوباره بازپخش شده‌اند؛ مقادیر قدیمی در کنار محاسبات تازه جدا نگه داشته شده‌اند.',
                'بستهٔ خروجی اصلی مثلث و S/C/T در آرشیو موجود نبود و مسیر عمومی Kaggle با HTTP 403 بسته بود؛ کد اصلی و وزن ذخیره‌شده برای بازاجرای جداگانه استفاده شده است.',
                'تنظیمات تاریخی حروف، از جمله سقف مدت 8، حفظ شده‌اند و با سقف مدت 5 در مقایسه‌های جدید مخلوط نمی‌شوند.','']
        for c in read_json(RUN/'conditional_cases.json'):
            if c['id'].startswith('A1_'):lines.append(f"- {c['id']}: {c['status']}"+(f" · [معیارهای بازاجرای تازه]({c['metrics']})" if c.get('metrics') else ''))
        lines+=['','## شکل‌ها','',
                '[نمای یک‌جای شکل‌ها](runs/formation_20261112_cpu/figures/contact_sheet.png) · [ورودی‌ها و هش هر شکل](runs/formation_20261112_cpu/figures_manifest.json)','']
        for f in self.manifest:
            if f['status']=='rendered':lines.append(f"- [{f['id']}: {f['caption_fa']}]({f['pdf']}) · [PNG]({f['png']})")
            else:lines.append(f"- {f['id']}: pending — {f['reason']}")
        lines+=['','## دامنهٔ نتیجه','',
                'خطاها و فاصله‌ها، بررسی عددی در مدل ثابت هستند.',
                'این خروجی‌ها تأیید سخت‌افزار یا اثبات برخوردناپذیری پیوسته نیستند.',
                'شکست جست‌وجو نیز اثبات ناممکن‌بودن فیزیکی آرایش نیست.',
                'تغییر مدت پرواز و بودجه، در تنظیمات هر اجرا مشخص شده است.',
                'مرجع چهار اردک، برنامه‌های قدیمی است؛ محاسبات بازپخش تازه جدا ذخیره شده‌اند.',
                'نبود خروجی اصلی مثلث و حروف، به‌عنوان نتیجهٔ علمی شکست‌خورده ثبت نشده است.',
                'جدول بالاترین تعداد پذیرفته‌شده در `runs/formation_20261112_cpu/tables/highest_verified_counts.csv` موجود است.',
                'جدول مقایسه‌ها، تعداد شروع‌های کامل‌شده را صریحاً نگه می‌دارد.',
                'فقط ردیف‌های `matched_three_seed_complete=True` مقایسهٔ کامل سه‌بذری هستند.','',
                '## محاسبات باقی‌مانده','',
                'محیط CPU-only با ۵ پردازندهٔ تخصیص‌یافته است. ماتریس 1014 درخواستی از بودجهٔ این مرحله بزرگ‌تر است؛ اجرای کامل همهٔ حالت‌ها ادعا نمی‌شود.',
                'وضعیت دقیق تک‌تک حالت‌های اجرا‌نشده در CSV ثبت شده و دستور `--resume` آن‌ها را ادامه می‌دهد.',
                'پیمایش تودرتو، چگالی‌های میانی ماسک‌ها، seedهای باقی‌ماندهٔ مقایسه‌های متراکم و مرکز، شبکهٔ توسعه‌یافتهٔ بیضی، دیواره‌ها و زمان هدف فقط به اندازهٔ وضعیت ثبت‌شده کامل‌اند.',
                'طرح اختیاری G5 برنامه‌ریزی نشده است؛ F08 و هر مؤلفهٔ فاقد داده در شکل‌های دیگر به‌صراحت pending است.',
                'کد رسم از فایل‌های ذخیره‌شده استفاده می‌کند و بهینه‌سازی یا آموزش دوباره انجام نمی‌دهد.']
        (ROOT/'README_FA.md').write_text('\n'.join(lines)+'\n')
        (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
        rendered=[m for m in self.manifest if m['status']=='rendered']
        if rendered:
            sheet=Image.new('RGB',(1200,440*math.ceil(len(rendered)/2)),'white');draw=ImageDraw.Draw(sheet)
            for i,f in enumerate(rendered):
                im=Image.open(ROOT/f['png']).convert('RGB');im.thumbnail((570,395))
                x=(i%2)*600+(600-im.width)//2;y=(i//2)*440+25
                sheet.paste(im,(x,y));draw.text(((i%2)*600+15,(i//2)*440+5),f['id'],fill='black')
            sheet.save(OUT/'contact_sheet.png')

    def run(self):
        self.f01();self.f02();self.f03();self.f04();self.f05();self.f06();self.f07();self.f09();self.f10();self.f11();self.f12()
        for code,reason in [('F08','central controls reused from F1; remaining locations and extended ellipse geometries pending'),('F13','optional held-out layout design not scheduled')]:self.pending(code,reason)
        self.tables();self.inventory();write_json(RUN/'figures_manifest.json',sorted(self.manifest,key=lambda x:x['id']))
        write_json(RUN/'main_text_candidates.json',{'candidate_ids':['F01','F02','F03','F06','F07','F09'],
                   'status':{m['id']:m['status'] for m in self.manifest if m['id'] in ['F01','F02','F03','F06','F07','F09']},
                   'selection_note':'Results inventory only; no thesis prose or Results chapter has been edited.'})
        print('rendered',[m['id'] for m in self.manifest if m['status']=='rendered'])


if __name__=='__main__':Figures().run()
