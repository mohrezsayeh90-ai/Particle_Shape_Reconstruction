"""Print-size panels for the Stokes particle PINN article."""
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from ..thesis_results import plots as legacy
from ..thesis_results.evidence import ARCHIVE, read_csv, write_csv
from .data import (ABLATION, CLEARANCE, DENSITY, DUCK, GALLERY, LETTERS, RADIUS,
                   SOURCE, TRIANGLE, document)

PINN, DOP = "#b94926", "#233b61"
PALETTE = ("#005f73","#b34825","#6e4381","#30794b")
WIDTH = 7.0
EXAMPLES = ("lower_wall_slow","upper_wall_fast","lower_contact","inlet_exit")


def panels(rows,cols,*,height,**kwargs):
    return plt.subplots(rows,cols,figsize=(WIDTH,height),layout="constrained",**kwargs)


def tag(ax,key):
    ax.set_title(key,loc="left",fontweight="bold")


def standard_legend(fig,handles,columns=3):
    fig.legend(handles=handles,loc="outside lower center",ncol=columns,frameon=False)


def formation_legend(fig, *, guide=False):
    handles = [Line2D([],[],marker="+",ls="none",color=DOP,label="Assigned center"),
               Patch(facecolor="#c0dcd6",edgecolor="#36756a",label=r"DOP853 disk at $T^*$")]
    if guide:
        handles.append(Line2D([],[],ls=":",color="#718293",label="Target guide"))
    standard_legend(fig,handles)


def snapshot(ax,case,*,extent=None,guide=False):
    cfg,metrics,arrays,_ = case
    target = arrays["target"]
    outlines = cfg["outline"]
    geometry = np.vstack((target,np.asarray(outlines))) if outlines else target
    if outlines:
        ax.plot(*np.asarray(outlines).T,color="#718293",linestyle=":",lw=.8)
    if guide:
        for i,j in cfg["target_edges"]:
            ax.plot(*target[[i,j]].T,color="#718293",linestyle=":",lw=.9)
    for center in arrays["dop853_endpoint"]:
        if np.isfinite(center).all():
            ax.add_patch(Circle(center,RADIUS,facecolor="#c0dcd6",edgecolor="#36756a",lw=.65))
    ax.scatter(*target.T,marker="+",color=DOP,s=15,zorder=5,lw=.8)
    lo,hi = (geometry.min(axis=0),geometry.max(axis=0)) if extent is None else extent
    ax.set(xlim=(lo[0]-.024,hi[0]+.024),ylim=(lo[1]-.024,hi[1]+.024),xlabel=r"$x^*$",ylabel=r"$y^*$")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    return metrics["methods"]["dop853"]


def duck(ev,out):
    fig,axes = panels(2,2,height=5.2,gridspec_kw={"height_ratios":[2,1]})
    groups = ("duck_six_fixed","duck_independent_wall_ports",
              "dense_duck_six_fixed","dense_duck_independent_wall_ports")
    records = [r for r in ev.rows if r["study"] in groups]
    for i,(ax,key) in enumerate(zip(axes[0],DUCK)):
        case = ev.case(key)
        snapshot(ax,case)
        cfg,m,_,schedule = case
        mode = ("Six fixed ports","Free wall")[i]
        total = sum(r["status"]=="accepted" for r in records if r["study"]==groups[i])
        start = min(float(r["release_time"]) for r in schedule)
        ax.set_title(f"{chr(97+i)}  {mode}, N={cfg['N']}, seed 11\n"
                     f"Accepted ({total}/3); global interval [{start:.2f}, {m['target_time']:.3f}]")
    grid = axes[1,0].get_gridspec()
    for ax in axes[1]:
        ax.remove()
    panel = fig.add_subplot(grid[1,:])
    table_rows = []
    for group in groups:
        items = [r for r in records if r["study"]==group]
        margins = [float(r["ref_clearance_margin"]) for r in items]
        table_rows.append([f"{items[0]['N']} {items[0]['mode']}",f"{sum(r['status']=='accepted' for r in items)}/3",
                           f"[{min(margins):+.3e}, {max(margins):+.3e}]",
                           "Overlap" if max(margins)<-.001 else "One shortfall" if min(margins)<0 else "Accepted"])
    panel.axis("off")
    tab = panel.table(cellText=table_rows,colLabels=["Historical target","Accepted","DOP853 margin range","Outcome"],loc="center",cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1,1.25)
    for (r,_),cell in tab.get_celld().items():
        cell.set_edgecolor("#ced8e2")
        if r==0:
            cell.set_facecolor("#e8eff6")
    formation_legend(fig)
    ev.summary(records,out/"data/duck_trials.csv")
    return fig,[r["id"] for r in records]


def ablation(ev,out):
    fig,axes = panels(2,2,height=5.4)
    labels = ("Endpoint-only design","Spacing-aware design")
    records = [r for r in ev.rows if r["study"]=="clearance_objective"]
    export = []
    series = {}
    cfgs = [ev.matrix[key] for key in ABLATION]
    for key in ("target_sha256","budgets","ports","min_gap","min_age","max_age","cap"):
        assert cfgs[0][key]==cfgs[1][key]
    assert [c["collision_aware"] for c in cfgs]==[False,True]
    for i,key in enumerate(ABLATION):
        case = ev.case(key)
        ax = axes[0,i]
        stats = snapshot(ax,case)
        tag(ax,chr(97+i))
        ax.set_title(f"{labels[i]} · 18/18 at $T^*$\nMax endpoint error {stats['endpoint_max']:.2e}",loc="center")
        times,distance,minimum,T = ev.trace(key)
        series[f"time_{i}"],series[f"distance_{i}"] = times,distance
        export.extend(dict(case=key,pair_i=minimum["i"],pair_j=minimum["j"],global_time=t,
                           distance=d,refined_minimum=t==minimum["time"],target_time=T)
                      for t,d in zip(times,distance))
        ax = axes[1,0]
        color = PALETTE[i]
        ax.plot(times,distance,color=color,label=f"{minimum['i']}/{minimum['j']} · {('endpoint only','spacing aware')[i]}")
        ax.axvline(T,color=color,ls=":",lw=.85)
        ax.plot(minimum["time"],minimum["distance"],marker="o",ms=3,color=color)
    ax = axes[1,0]
    tag(ax,"c")
    ax.set(xlabel=r"Global time $s^*$",ylabel=r"Critical-pair distance $d^*$")
    ax.axhline(CLEARANCE,color="#353535",ls="--",lw=.8)
    ax.axhline(2*RADIUS,color="#666666",ls="-.",lw=.8)
    ax.legend(loc="upper left",frameon=False,fontsize=7)
    ax = axes[1,1]
    tag(ax,"d")
    for i,key in enumerate(ABLATION):
        block = sorted((r for r in records if r["id"].rsplit("_s",1)[0]==key.rsplit("_s",1)[0]),key=lambda r:int(r["seed"]))
        vals = [float(r["ref_clearance_margin"])+CLEARANCE for r in block]
        for j,(r,value) in enumerate(zip(block,vals)):
            ax.plot(i+(j-1)*.10,value,marker=("o","s","^")[j],color=PALETTE[i],ms=4,
                    label=f"Seed {r['seed']}" if i==0 else None)
    ax.axhline(CLEARANCE,color="#353535",ls="--",lw=.8,label="Required distance")
    ax.axhline(2*RADIUS,color="#666666",ls="-.",lw=.8,label="Disk contact")
    ax.set(xlim=(-.45,1.45),xticks=[0,1],xticklabels=["Endpoint only\n0/3 accepted","Spacing aware\n3/3 accepted"],
           ylabel=r"DOP853 $d_{\min}^*$")
    ax.legend(loc="center",frameon=False,fontsize=7,ncol=2)
    formation_legend(fig,guide=True)
    write_csv(out/"data/clearance_pair_traces.csv",export)
    np.savez_compressed(out/"data/clearance_pair_traces.npz",**series)
    ev.summary(records,out/"data/clearance_trials.csv")
    return fig,[r["id"] for r in records]


def method():
    fig,ax = plt.subplots(figsize=(WIDTH,2.55),layout="constrained")
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis("off")
    cells = (
        (.12,.76,"Inputs\n$x_0^*,y_0^*,v_{x0}^*,v_{y0}^*,t^*$"),
        (.38,.76,"Release features\n$x$: $1-z^2, zw, w^2, v_{x0}^*/U_c^*$\n$y$: $w$"),
        (.65,.76,"Two linear readouts\n$q_0+t^*v_0+\\dfrac{t^{*2}}{t^*+s}N$"),
        (.89,.76,"Particle position $q$\n$\\dot q,\\ddot q$ by autodiff"),
        (.38,.25,"Time features\nChebyshev degree 24\nQR conditioned"),
        (.65,.25,"Prescribed Poiseuille profile\n$U(y^*)=[4U_c^*\\eta(1-\\eta),0]$"),
        (.89,.25,"Stokes residual\n$\\tau_p\\ddot q+\\dot q-U(y^*)$"),
    )
    for x,y,label in cells:
        ax.text(x,y,label,ha="center",va="center",fontsize=8,
                bbox=dict(boxstyle="round,pad=.4",fc="#eff4f8",ec="#597489",lw=.6))
    for a,b in [((.224,.76),(.268,.76)),((.501,.76),(.555,.76)),((.747,.76),(.789,.76)),
                ((.14,.66),(.28,.34)),((.469,.34),(.56,.63)),((.773,.25),(.797,.25)),
                ((.89,.67),(.89,.37))]:
        ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=.8,color="#334458"))
    ax.text(.035,.03,r"$\eta=y^*/H^*,\quad z=2y_0^*/H^*-1,\quad w=v_{y0}^*/1.20$",fontsize=8)
    return fig


def paths(data,rows,names=EXAMPLES):
    fig = plt.figure(figsize=(WIDTH,5.15 if len(names)==4 else 6.6),layout="constrained")
    n = len(names)//2
    grid = fig.add_gridspec(n+1,2,height_ratios=[1]*n+[1.45])
    ids = list(data["case_id"])
    by_id = {r["case"]:r for r in rows}
    display = ("Lower-wall launch","Upper-wall launch","Interior launch","Interior launch")
    colors = plt.get_cmap("tab10")
    for i,key in enumerate(names):
        idx = ids.index(key)
        ax = fig.add_subplot(grid[i//2,i%2])
        a,b = data["dop853_display_position"][idx],data["pinn_display_position"][idx]
        lo,hi = max(-.025,a[:,0].min()-.06),min(2.225,a[:,0].max()+.06)
        if hi-lo<.38:
            mid = np.clip((lo+hi)/2,.165,2.035)
            lo,hi = mid-.19,mid+.19
        legacy.channel(ax,(lo,hi))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.plot(*a.T,color=DOP,lw=1.25)
        ax.plot(*b.T,color=PINN,ls="--",lw=.85,marker="+",markevery=[-1],ms=4)
        ax.scatter(*a[0],s=11,color="#212121",zorder=5)
        ax.scatter(*a[-1],s=14,facecolor="none",edgecolor=DOP,marker="D",zorder=4)
        title = display[i] if names==EXAMPLES else key.replace("_"," ")
        event = by_id[key]["reference_event"].replace("_"," ")
        ax.set_title(f"{chr(65+i)}  {title} · {event} (zoom)")
    ax = fig.add_subplot(grid[-1,:])
    for i,key in enumerate(names):
        idx=ids.index(key)
        err=np.linalg.norm(data["signed_error"][idx,:,:2],axis=-1)
        shown=np.where(err>0,err,np.nan)
        ax.plot(data["common_time"][idx],shown,color=colors(i),lw=1,label=chr(65+i))
    ax.set(yscale="log",xlabel=r"Elapsed time $t^*$",ylabel=r"Position error $||e_q^*||_2$")
    ax.legend(ncol=len(names),frameon=False,loc="lower right",bbox_to_anchor=(1,1.0),borderaxespad=.1)
    standard_legend(fig,[Line2D([],[],color=DOP,label="DOP853"),
                         Line2D([],[],color=PINN,ls="--",marker="+",label="PINN"),
                         Line2D([],[],ls="none",color="#222",marker="o",label="Release"),
                         Line2D([],[],ls="none",color=DOP,marker="D",mfc="none",label="DOP853 event")],4)
    return fig


def accuracy(rows,summary,out,data,*,single=False):
    err = abs(data["signed_error"][...,:4]).max(axis=(0,1))
    np.testing.assert_allclose(err,np.r_[summary["worst_max_abs_position"],summary["worst_max_abs_velocity"]],
                               rtol=1e-8,atol=1e-15)
    if single:
        fig,axes=plt.subplots(3,1,figsize=(85/25.4,6.4),layout="constrained")
    else:
        fig,axes=panels(1,3,height=2.85,gridspec_kw={"width_ratios":[1.1,1.05,1.3]})
    ax,bx,cx = axes
    stats = []
    for i,key in enumerate(("x","y","vx","vy")):
        values = np.array([float(r[f"max_{key}_error"]) for r in rows])
        j = np.linspace(-.17,.17,len(values))
        ax.scatter(i+j[values>0],values[values>0],s=7,color=DOP,alpha=.64)
        ax.scatter(i+j[values==0],np.full(sum(values==0),1e-16),s=8,marker="x",color=PINN)
        stats.append(dict(quantity=key,n=len(values),zero_count=sum(values==0),median=np.median(values),
                          p95=np.quantile(values,.95),maximum=values.max()))
    ax.set(yscale="log",xticks=range(4),xticklabels=["x*","y*","vx*","vy*"],ylabel="Peak absolute error")
    ax.plot([],[],marker="x",ls="none",color=PINN,label="Exact zero")
    ax.legend(frameon=False,loc="upper left",fontsize=6.9)
    categories = legacy.CATEGORIES
    names = ["Lower","Upper","Inlet","Outlet","Horizon"]
    matrix = np.zeros((5,5),int)
    events = np.array([float(r["signed_event_error"]) for r in rows])
    for i,r in enumerate(rows):
        a,b = (categories.index(r[k]) for k in ("reference_event","neural_event"))
        matrix[a,b]+=1
        bx.plot(events[i]*1e7,a+.15*np.sin(i),marker=".",ms=3.5,color=DOP)
    bx.axvline(0,color="#768290",lw=.8)
    bx.set(yticks=range(5),yticklabels=names,xlabel=r"$(t^*_{PINN}-t^*_{DOP853})\times10^7$")
    cx.pcolormesh(np.arange(6)-.5,np.arange(6)-.5,matrix,cmap="Blues")
    for i,j in np.ndindex(matrix.shape):
        cx.text(j,i,str(matrix[i,j]),ha="center",va="center",color="white" if matrix[i,j]>22 else "#213649")
    cx.set(xticks=range(5),xticklabels=names,yticks=range(5),yticklabels=names,
           xlabel="PINN",ylabel="DOP853")
    cx.tick_params(axis="x",rotation=41)
    cx.invert_yaxis()
    for axis,label in zip(axes,"abc"):
        tag(axis,label)
    sep = "\n" if single else "   ·   "
    fig.suptitle(f"Event type agreement {summary['matching_events']}/70{sep}Composite pass {summary['passed_count']}/70",fontsize=9)
    a = abs(events)
    stats.append(dict(quantity="event_time",n=len(a),zero_count=sum(a==0),median=np.median(a),p95=np.quantile(a,.95),maximum=a.max()))
    write_csv(out/"data/validation_statistics.csv",stats)
    return fig


def training():
    fig = legacy.training()
    fig.set_size_inches(WIDTH,4.35)
    ax = fig.axes[-1]
    ax.texts[-1].remove()
    ax.plot([],[],ls="--",color="#24282c",label="Selected step 4808")
    ax.legend(frameon=False,loc="upper right")
    return fig


def slices(rows):
    fig=legacy.slices_figure(rows)
    fig.set_size_inches(WIDTH,7.5)
    fig.legends[0].texts[-1].set_text("State unavailable at t*=1")
    fig.axes[2].set_title("Common-interval error")
    fig.axes[-1].set_ylabel(r"$\log_{10}$ peak component position error")
    return fig


def triangle(ev):
    cfg,metric,arrays,schedule=ev.case(TRIANGLE)
    fig, axes=panels(2,1,height=3.8,gridspec_kw={"height_ratios":[1,1.28]})
    ax,bx=axes
    legacy.channel(ax)
    tag(ax,"a")
    for pin,ref in zip(arrays["pinn"],arrays["dop853"]):
        ax.plot(*ref[:,:2].T,color=DOP,lw=1.3)
        ax.plot(*pin[:,:2].T,color=PINN,lw=.8,ls="--")
    ax.scatter(*arrays["release"][:,:2].T,marker="^",s=17,color="#222")
    ax.scatter(*arrays["target"].T,marker="+",s=24,color=PINN)
    ax.scatter(*arrays["dop853_endpoint"].T,marker="o",s=25,facecolor="none",edgecolor=DOP)
    critical=metric["methods"]["dop853"]["minimum"]
    T=metric["target_time"]
    for i,r in enumerate(schedule,1):
        start=float(r["release_time"])
        color=PINN if i in (critical["i"],critical["j"]) else DOP
        bx.hlines(i,start,T,color=color,lw=1.6)
        bx.plot(start,i,"o",ms=3,color=color)
    bx.axvline(T,color=DOP,ls="-.",lw=.8,label=f"$T^*$ = {T:.4f}")
    bx.axvline(critical["time"],color=PINN,ls="--",lw=.8,
               label=f"Pair {critical['i']}/{critical['j']} minimum at $s^*$ = {critical['time']:.4f}")
    bx.set(xlabel=r"Global time $s^*$",ylabel="Target ID",yticks=[1,2,3],ylim=(.55,4.10))
    bx.legend(loc="upper left",frameon=False)
    tag(bx,"b")
    standard_legend(fig,[Line2D([],[],color=DOP,label="DOP853 path"),
      Line2D([],[],color=PINN,ls="--",label="PINN path"),Line2D([],[],color="#222",ls="none",marker="^",label="Release"),
      Line2D([],[],color=PINN,ls="none",marker="+",label="Assigned center"),
      Line2D([],[],color=DOP,ls="none",marker="o",mfc="none",label=r"DOP853 center at $T^*$")],5)
    return fig


def density(ev,out):
    fig, axes=panels(2,2,height=4.1,gridspec_kw={"height_ratios":[1.2,1]})
    first,second = [ev.case(r) for r in DENSITY]
    assert first[0]["target_sha256"]==second[0]["target_sha256"]
    evidence = [first,second]
    data_rows = []
    for i,(case,title) in enumerate(zip(evidence,("Six fixed ports","Free wall"))):
        cfg,metric,arrays,schedule = case
        ms=snapshot(axes[0,i],case)
        axes[0,i].set_title(f"{chr(97+i)}  {title} · seed 11 · 3/3 accepted")
        ax=axes[1,i]
        for side,color,mark in ((0,DOP,"^"),(1,PINN,"v")):
            entries=[r for r in schedule if int(r["side"])==side]
            ax.scatter([float(r["x0"]) for r in entries],[float(r["release_time"]) for r in entries],
                       marker=mark,color=color,s=10,label=("Lower","Upper")[side],alpha=.8)
        T=metric["target_time"]
        ax.set(xlabel=r"Launch $x_0^*$",ylabel=r"Release time $s^*$",xlim=(0,2.2),ylim=(0,max(r[1]["target_time"] for r in evidence)*1.07))
        ax.axhline(T,color="#777",ls=":",lw=.75)
        ax.set_title(f"{chr(99+i)}  $T^*$={T:.3f} · margin {ms['clearance_margin']:+.5e}")
        if i==0:
            ax.legend(loc="upper left",frameon=False,ncol=2)
        data_rows.append(dict(case=DENSITY[i],mode=cfg["mode"],seed=11,completed=ms["completed"],
            target_hash=cfg["target_sha256"],target_time=T,endpoint_rms=ms["endpoint_rms"],
            endpoint_max=ms["endpoint_max"],d_min=ms["minimum"]["distance"],margin=ms["clearance_margin"]))
    write_csv(out/"data/matched_density.csv",data_rows)
    formation_legend(fig,guide=True)
    return fig


def sweep(ev,out):
    fig,axes=panels(2,3,height=4.55)
    records=[r for r in ev.rows if r["study"]=="density" or r["stage"]=="C"]
    modes=[
        ([("Fixed",lambda r:r["study"]=="density" and r["mode"]=="fixed"),
          ("Free wall",lambda r:r["study"]=="density" and r["mode"]=="free")],
         ["24","48","72","100"],lambda r:r["N"],"Density","Target N"),
        ([("Equispaced",lambda r:r["study"]=="equispaced"),
          ("Nested",lambda r:r["study"]=="nested"),
          ("Free wall",lambda r:r["study"]=="free")],
         ["2","4","6","8","10","12","Free"],lambda r:r["port_count"] or "Free","Locations","Ports (N=24)"),
        ([("Fixed",lambda r:r["study"]=="capacity")],
         ["3","4","6","∞"],lambda r:r["cap"] or "∞","Capacity","Cap per port (N=24)")]
    for col,(groups,keys,get,title,xlabel) in enumerate(modes):
        ax,bx=axes[:,col]
        for i,(label,belongs) in enumerate(groups):
            marker=("o","s","^")[i if label!="Free wall" else 2]
            color=PALETTE[i if label!="Free wall" else 2]
            offset=(i-(len(groups)-1)/2)*.26 if label!="Free wall" or col==0 else 0
            for j,k in enumerate(keys):
                planned=[r for r in records if belongs(r) and get(r)==k]
                attempted=[r for r in planned if r["status"] in ("accepted","rejected","search_failed","error")]
                accepted=sum(r["status"]=="accepted" for r in attempted)
                if attempted:
                    ratio=accepted/len(attempted)
                    ax.plot(j+offset,ratio,marker=marker,color=color,ls="none",ms=4)
                    ax.text(j+offset,ratio+.075+i*.090,f"{accepted}/{len(attempted)}",
                            fontsize=6.7,color=color,ha="center")
                rejected=sum(r["status"]=="preflight_rejected" for r in planned)
                if rejected:
                    ax.text(j,.50,f"{rejected} preflight\n0 attempted",ha="center",fontsize=6.5,color="#575f66")
                for k,r in enumerate(sorted(attempted,key=lambda r:int(r["seed"]))):
                    if r["ref_clearance_margin"]:
                        bx.plot(j+offset+(k-1)*.045,float(r["ref_clearance_margin"]),marker=marker,ls="none",
                                ms=4,mec=color,mfc=color if r["status"]=="accepted" else "none")
            ax.plot([],[],marker=marker,ls="none",color=color,label=label)
        ax.set(title=title,ylim=(-.13,1.29),yticks=(0,.5,1))
        ax.legend(frameon=False,loc="lower left",fontsize=6.6)
        bx.set_xticks(range(len(keys)),keys)
        bx.set(xlabel=xlabel)
        ax.set_xticks(range(len(keys)),[""]*len(keys))
        for a in (ax,bx):
            a.set_xlim(-.55,len(keys)-.45)
            a.grid(axis="y",alpha=.14)
        bx.axhline(0,color="#333",ls="--",lw=.65)
        bx.axhline(-.001,color="#777",ls="-.",lw=.75)
    axes[0,0].set_ylabel("Accepted / attempted")
    axes[1,0].set_ylabel("DOP853 clearance margin")
    standard_legend(fig,[Line2D([],[],ls="--",color="#333",label="Acceptance"),
                        Line2D([],[],ls="-.",color="#777",label="Disk contact"),
                        Line2D([],[],ls="none",marker="o",mfc="none",color="#333",label="Rejected")])
    ev.summary(records,out/"data/formation_scope.csv")
    return fig,[r["id"] for r in records]


def geometry(ev,out):
    fig,axes=panels(1,2,height=3.05)
    ids=[]
    records=[r for r in ev.rows if r["study"] in ("center","ellipse")]
    for ax,study,title in zip(axes,("center","ellipse"),("Stadium center · N=18","Fixed-area ellipse · N=12")):
        items=[(ev.matrix[r["id"]],r) for r in records if r["study"]==study]
        ids.extend(r["id"] for c,r in items)
        def coord(c):
            return c["center"] if study=="center" else (c["theta"],c["q"])
        xs=sorted({coord(c)[0] for c,_ in items})
        ys=sorted({coord(c)[1] for c,_ in items})
        cells=np.full((len(ys),len(xs)),np.nan)
        for y,x in np.ndindex(cells.shape):
            match=[r for c,r in items if list(coord(c))==[xs[x],ys[y]]]
            if match:
                total=sum(r["status"]=="accepted" for r in match)
                cells[y,x]=total
                ax.text(x,y,f"{total}/{len(match)}",ha="center",va="center",color="white" if total==3 else "#202a34")
            elif study=="ellipse" and ys[y]==1:
                ax.text(x,y,"—",ha="center",va="center",color="#696969")
        cmap=ListedColormap(("#dca789","#cbdbe7","#81abc6","#2c678b"))
        cmap.set_bad("#ddd")
        ax.pcolormesh(np.arange(len(xs)+1)-.5,np.arange(len(ys)+1)-.5,cells,cmap=cmap,vmin=-.5,vmax=3.5)
        ax.set(title=title,xticks=range(len(xs)),xticklabels=[f"{v:g}" for v in xs],
               yticks=range(len(ys)),yticklabels=[f"{v:g}" for v in ys],
               xlabel=r"Center $x^*$" if study=="center" else "Rotation (deg)",
               ylabel=r"Center $y^*$" if study=="center" else "Aspect ratio")
    standard_legend(fig,[Patch(facecolor="#ddd",label="Unscheduled · circle angle immaterial"),
                        Patch(facecolor="#2c678b",label="All seeds accepted")],2)
    ev.summary(records,out/"data/geometry_trials.csv")
    return fig,ids


def gallery(ev,out,keys,labels,filename):
    fig,axes=panels(1,len(keys),height=3.05 if keys==LETTERS else 2.3)
    cases=[ev.case(key) for key in keys]
    common=None
    if keys==LETTERS:
        points=np.concatenate([item[2]["target"] for item in cases])
        common=(points.min(axis=0),points.max(axis=0))
    records=[]
    for ax,key,label,case in zip(axes,keys,labels,cases):
        cfg,metrics,_,_=case
        stats=snapshot(ax,case,extent=common,guide=keys==LETTERS)
        ax.set_title(f"{label}\n{stats['completed']}/{cfg['N']} at $T^*$")
        records.append(dict(case=key,seed=cfg["seed"],mode=cfg["mode"],N=cfg["N"],status=metrics["status"],
                            target_time=metrics["target_time"],d_min=stats["minimum"]["distance"],
                            margin=stats["clearance_margin"],endpoint_rms=stats["endpoint_rms"],endpoint_max=stats["endpoint_max"]))
    formation_legend(fig,guide=True)
    write_csv(out/"data"/filename,records)
    return fig


def reflection(ev,out):
    fig,axes=panels(2,2,height=3.55,sharey="row",gridspec_kw={"height_ratios":[1,1.5]})
    rows=[r for r in ev.rows if r["study"]=="reflection"]
    for i,layout in enumerate(("wall","wall_reflected")):
        chosen=[r for r in rows if r["layout"]==layout]
        key=next(r["id"] for r in chosen if r["seed"]=="11")
        case=ev.case(key)
        ax=axes[0,i]
        legacy.channel(ax)
        ax.scatter(*np.asarray(case[0]["targets"]).T,marker="+",color="#717b84",s=9)
        for p in case[0]["ports"]:
            ax.plot(p["x"],.30*p["side"],marker="v" if p["side"] else "^",color=DOP,ms=4)
        count=sum(r["status"]=="accepted" for r in chosen)
        ax.set_title(f"{('Original layout','Reflected layout')[i]} · {count}/3 accepted")
        ax=axes[1,i]
        for r in chosen:
            ax.plot(int(r["seed"]),float(r["ref_clearance_margin"]),"o",color=DOP,
                    mfc=DOP if r["status"]=="accepted" else "none")
        ax.axhline(0,color="#333",ls="--",lw=.8)
        ax.axhline(-.001,color="#777",ls="-.",lw=.8)
        ax.set(xticks=(11,29,47),xlabel="Seed",ylabel="DOP853 clearance margin")
    ev.summary(rows,out/"data/reflection_trials.csv")
    return fig,[r["id"] for r in rows]
