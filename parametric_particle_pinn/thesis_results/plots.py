"""Render publication figures from the recorded arrays and tables."""
from collections import Counter
import json

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

from .evidence import ARCHIVE, read_csv, write_csv

CATEGORIES = ("lower_wall","upper_wall","inlet_exit","outlet","time_limit")
COLORS = ("#187c72","#2465a2","#873783","#c25120","#62666c")
LABELS = ("Lower wall","Upper wall","Inlet exit","Outlet","Horizon")
PINN, DOP = "#b94926", "#233b61"
H, L, RADIUS = .30, 2.2, .0125
FLOOR = 1e-13
REPRESENTATIVES = ("lower_wall_slow","upper_wall_fast","interior_rising",
                   "lower_contact","inlet_exit","slow_time_limit")


def channel(ax, limits=None):
    ax.hlines([0,H],0,L,color="#334458",linewidth=.8)
    ax.hlines([RADIUS,H-RADIUS],0,L,color="#8491a1",linestyle=":",linewidth=.7)
    ax.vlines([0,L],0,H,color="#8491a1",linestyle="--",linewidth=.7)
    ax.set_xlim(*(limits or (-.025,2.225)))
    ax.set_ylim(-.012,.312)
    ax.set_aspect("equal",adjustable="box")
    ax.set(xlabel=r"$x^*$",ylabel=r"$y^*$")


def methods_diagram():
    fig,ax = plt.subplots(figsize=(9,3.55),layout="constrained")
    ax.set(xlim=(0,10),ylim=(0,4))
    ax.axis("off")
    boxes = [
        (.15,2.55,2.05,.85,"Release inputs\n$(x_0^*,y_0^*,v_{x0}^*,v_{y0}^*)$"),
        (.15,.95,2.05,.85,"Elapsed time $t^*$"),
        (2.65,2.55,1.98,.85,"Release features\n$1-z^2, zw,w^2,v_{x0}^*/U_c$"),
        (2.65,.95,1.98,.85,"Chebyshev features\nDegree 24, QR conditioned"),
        (5.05,2.55,2.5,.85,"Two learned readouts\n$q_0+t v_0+t^2 N/(t+s)$"),
        (8.02,2.55,1.82,.85,"Position $q$\n$\\dot q,\\ddot q$ by autodiff"),
        (8.02,.95,1.82,.85,"Stokes residual\n$\\tau_p\\ddot q+\\dot q-U(y)$"),
        (5.05,.95,2.5,.85,"Analytical carrier\n$U=[4U_c(y/H)(1-y/H),0]$"),
    ]
    for x,y,w,h,label in boxes:
        ax.add_patch(Rectangle((x,y),w,h,facecolor="#eff4f8",edgecolor="#58728b"))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=9)
    for start,end in [((2.2,2.98),(2.65,2.98)),((2.2,1.38),(2.65,1.38)),
                      ((4.63,2.98),(5.05,2.98)),((4.63,1.55),(5.05,2.63)),
                      ((7.55,2.98),(8.02,2.98)),((8.93,2.55),(8.93,1.80)),
                      ((7.55,1.38),(8.02,1.38))]:
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color="#334458"))
    ax.hlines([.19,.74],.2,3.1,color="#334458")
    ax.hlines([.24,.69],.2,3.1,color="#8491a1",linestyles=":")
    ax.annotate("",xy=(1.25,.48),xytext=(1.10,.23),arrowprops=dict(arrowstyle="->",color=PINN))
    ax.annotate("",xy=(2.45,.46),xytext=(2.30,.70),arrowprops=dict(arrowstyle="->",color=PINN))
    ax.text(3.25,.46,"Walls and particle-center contact lines",va="center",fontsize=9)
    return fig


def trajectories(data, rows):
    fig = plt.figure(figsize=(9.5,8.9),layout="constrained")
    grid = fig.add_gridspec(5,2,height_ratios=[1,1,1,1.25,1.45])
    per_id = {r["case"]:r for r in rows}
    ids = list(data["case_id"])
    for i,name in enumerate(REPRESENTATIVES):
        ax = fig.add_subplot(grid[i//2,i%2])
        k = ids.index(name)
        ref,pin = data["dop853_display_position"][k],data["pinn_display_position"][k]
        extent = (max(-.02,min(ref[:,0].min(),pin[:,0].min())-.07),
                  min(2.22,max(ref[:,0].max(),pin[:,0].max())+.07))
        if extent[1]-extent[0]<.36:
            middle = np.clip(np.mean(extent),.16,2.04)
            extent = (middle-.18,middle+.18)
        channel(ax,extent)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.plot(*ref.T,color=DOP,lw=2,label="DOP853")
        ax.plot(*pin.T,color=PINN,lw=.9,ls="--",label="PINN")
        ax.scatter(*ref[0],s=18,color="#222222",zorder=4,label="Release")
        ax.scatter(*ref[-1],s=27,marker="D",edgecolor=DOP,facecolor="white",zorder=4,label="DOP853 event")
        ax.scatter(*pin[-1],s=14,marker="+",color=PINN,zorder=5,label="PINN event")
        label = per_id[name]["reference_event"]
        event = LABELS[CATEGORIES.index(label)]
        title = name.replace("_"," ").replace("inlet exit","Inlet release").replace("slow time limit","Slow drift")
        ax.set_title(f"{title} | {event}",fontsize=9)
        if i==0:
            fig.legend(*ax.get_legend_handles_labels(),ncol=5,loc="outside upper center",
                       fontsize=8,frameon=False)
    for component, label in enumerate(("x", "y")):
        axis = fig.add_subplot(grid[3,component])
        for j in (0,1,2):
            k = ids.index(REPRESENTATIVES[j])
            color = plt.get_cmap("tab10")(j)
            axis.plot(data["dop853_display_time"][k],data["dop853_display_position"][k,:,component],
                      color=color,lw=1.5)
            axis.plot(data["pinn_display_time"][k],data["pinn_display_position"][k,:,component],
                      color=color,lw=.7,ls="--")
        axis.set(xlabel=r"$t^*$",ylabel=rf"${label}^*$")
        axis.grid(alpha=.15)
    ax = fig.add_subplot(grid[4,0])
    ay = fig.add_subplot(grid[4,1],sharex=ax)
    for j,name in enumerate(REPRESENTATIVES):
        k = ids.index(name)
        for comp,axis in enumerate((ax,ay)):
            values = abs(data["signed_error"][k,:,comp])
            axis.plot(data["common_time"][k],np.maximum(values,FLOOR),
                      label=name.replace("_"," "),color=plt.get_cmap("tab10")(j),lw=1)
    for label,axis in zip(("x","y"),(ax,ay)):
        axis.set(xlabel=r"$t^*$",ylabel=rf"$|e_{{{label}}}^*|$",yscale="log")
        axis.grid(alpha=.18)
    ay.legend(fontsize=7,frameon=False,loc="upper left")
    ax.text(.02,.04,r"Display floor $10^{-13}$",transform=ax.transAxes,fontsize=7)
    return fig


def validation_overview(rows, summary):
    fig, axes = plt.subplots(2,2,figsize=(9.1,6.6),layout="constrained")
    ax,bx,cx,dx = axes.flat
    event = np.array([r["reference_event"] for r in rows])
    whiskers = np.linspace(-.20,.20,len(rows))
    for i,e in enumerate(CATEGORIES):
        included = event==e
        for j,k in enumerate(("x","y","vx","vy")):
            values = np.array([float(r["max_"+k+"_error"]) for r in rows])
            mask = included & (values>0)
            ax.scatter(j+whiskers[mask],np.maximum(values[mask],FLOOR),
                       color=COLORS[i],s=12,alpha=.8)
            mask = included & (values==0)
            ax.scatter(j+whiskers[mask],np.full(mask.sum(),FLOOR),
                       color=COLORS[i],s=11,marker="x")
        signed = np.array([float(r["signed_event_error"]) for r in rows])
        bx.scatter(signed[included], i+whiskers[included]/2, s=13, color=COLORS[i])
    ax.set(yscale="log",xticks=range(4),xticklabels=["x*","y*","vx*","vy*"],ylabel="Per-case maximum absolute error")
    ax.text(.02,.04,r"$\times$ = exact zero (at $10^{-13}$)",transform=ax.transAxes,fontsize=7)
    ax.grid(axis="y",alpha=.15)
    bx.axvline(0,color="#98a3ae",lw=.8)
    bx.set(yticks=range(5),yticklabels=LABELS,xlabel=r"Signed event error $t_{PINN}^*-t_{DOP853}^*$")
    bx.ticklabel_format(axis="x",style="sci",scilimits=(0,0))
    confusion = np.zeros((5,5),int)
    for row in rows:
        confusion[CATEGORIES.index(row["reference_event"]),CATEGORIES.index(row["neural_event"])] += 1
    cx.imshow(confusion,cmap="Blues",vmin=0,vmax=confusion.max())
    for i,j in np.ndindex(confusion.shape):
        cx.text(j,i,str(confusion[i,j]),ha="center",va="center",
                color="white" if confusion[i,j]>.65*confusion.max() else "#27394c")
    short = ["Lower","Upper","Inlet","Outlet","Horizon"]
    cx.set(xticks=range(5),yticks=range(5),xticklabels=short,yticklabels=short,
           xlabel="PINN event",ylabel="DOP853 event")
    dx.axis("off")
    dx.set_title(f"Configured checks passed: {summary['passed_count']}/{summary['case_count']}",loc="left",fontsize=10)
    values = []
    for key in ("x","y","vx","vy","event_time"):
        col = "event_time_error" if key=="event_time" else f"max_{key}_error"
        z = np.array([float(r[col]) for r in rows])
        values.append([key,len(z),*(f"{v:.2e}" for v in (np.median(z),np.quantile(z,.95),z.max()))])
    table = dx.table(cellText=values,colLabels=["Quantity","n","Median","95th pct.","Max"],
                     colWidths=[.25,.11,.21,.21,.21],cellLoc="center",loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1,1.45)
    for (i,_),cell in table.get_celld().items():
        cell.set_edgecolor("#d5dfe8")
        cell.set_facecolor("#eaf1f7" if i==0 else "white")
    return fig


def training():
    rows = [json.loads(line) for line in (ARCHIVE/"training.jsonl").read_text().splitlines()]
    summary = json.loads((ARCHIVE/"training_summary.json").read_text())
    fig,axes = plt.subplots(2,1,sharex=True,figsize=(8.4,5.2),layout="constrained")
    stages = [(0,1600,"Transverse Adam (x frozen)"),(1600,4600,"Horizontal Adam (y frozen)"),
              (4600,4808,"L-BFGS")]
    for i,ax in enumerate(axes):
        for j,(lo,hi,label) in enumerate(stages):
            ax.axvspan(lo,hi,color=("#ecf3f5","#f5f2ea","#e8eef7")[j])
            if i==0:
                ax.text((lo+hi)/2,.92 if j<2 else .80,label,ha="center",
                        transform=ax.get_xaxis_transform(),fontsize=8)
        steps = [r["step"] for r in rows]
        ax.plot(steps,[r["dev_scaled_mse"][i] for r in rows],".-",ms=3.8,color=DOP,label="Development")
        valid = [r for r in rows if r["train_scaled_mse"] is not None]
        ax.scatter([r["step"] for r in valid],[r["train_scaled_mse"][i] for r in valid],
                   facecolors="none",edgecolors=PINN,s=15,label="Training samples")
        ax.axvline(summary["best_step"],color="#24282c",ls="--",lw=.8)
        ax.set(yscale="log",ylabel=f"Scaled {'x' if i==0 else 'y'} residual MSE")
        ax.grid(axis="y",alpha=.15)
    axes[0].legend(loc="lower left",frameon=False)
    axes[1].set(xlabel="Optimizer iteration (Adam + L-BFGS)",xlim=(-80,5060))
    axes[1].text(.02,.10,"Selected 4808  |  208 L-BFGS iterations, 353 closure evaluations",
                 transform=axes[1].transAxes,fontsize=8)
    return fig


def slices_figure(rows):
    fig, axes = plt.subplots(3,3,figsize=(10,8),layout="constrained")
    for i,key in enumerate(("lower","upper","interior")):
        group = [r for r in rows if r["slice"]==key]
        nx = len({r["axis_x"] for r in group})
        x = np.array([float(r["axis_x"]) for r in group]).reshape(-1,nx)
        y = np.array([float(r["axis_y"]) for r in group]).reshape(-1,nx)
        event = np.array([CATEGORIES.index(r["pinn_event"]) for r in group]).reshape(x.shape)
        stop = np.array([float(r["pinn_event_time"]) for r in group]).reshape(x.shape)
        err = np.array([max(float(r["max_x_error"]),float(r["max_y_error"])) for r in group]).reshape(x.shape)
        invalid = np.array([r["valid_pinn"]!="True" or r["valid_dop853"]!="True" for r in group]).reshape(x.shape)
        axes[i,0].pcolormesh(x,y,event,cmap=ListedColormap(COLORS),vmin=-.5,vmax=4.5,shading="nearest")
        axes[i,0].scatter(x[invalid],y[invalid],s=4,marker="x",color="white",lw=.55)
        tt = axes[i,1].pcolormesh(x,y,stop,cmap="viridis",vmin=0,vmax=5,shading="nearest")
        er = axes[i,2].pcolormesh(x,y,np.log10(np.maximum(err,FLOOR)),cmap="magma",vmin=-13,vmax=-5,shading="nearest")
        axes[i,0].set_ylabel(key.capitalize()+("\n$y$-velocity at release" if key=="interior" else "\nInward speed"))
        for j,ax in enumerate(axes[i]):
            ax.set_xlabel(r"$y_0^*$" if key=="interior" else r"$v_{x0}^*$")
            if i==0:
                ax.set_title(("PINN event","PINN event time","Max component position error")[j])
    fig.colorbar(tt,ax=axes[:,1],label=r"$t^*$",fraction=.06,pad=.025)
    fig.colorbar(er,ax=axes[:,2],label=r"$\log_{10}$ absolute error",fraction=.06,pad=.025)
    handles = [Line2D([],[],color=c,label=l,lw=4) for c,l in zip(COLORS,LABELS)]
    handles.append(Line2D([],[],marker="x",color="#62666c",ls="none",label="Terminated by t*=1"))
    fig.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False,fontsize=8)
    return fig


def load_case(output, row):
    folder = output/"formation"/row["id"]
    with np.load(folder/"trajectories.npz") as archive:
        data = {name:archive[name] for name in archive.files}
    return data, json.loads((folder/"metrics.json").read_text())


def formation_overview(output, row):
    data, metric = load_case(output,row)
    fig,axes = plt.subplots(2,1,figsize=(9.3,5.3),height_ratios=[1,1.3],layout="constrained")
    ax,bx = axes
    channel(ax)
    for color,ref,pin in zip(plt.get_cmap("Dark2").colors,data["dop853"],data["pinn"]):
        ax.plot(*ref[:,:2].T,color=color,lw=1.65)
        ax.plot(*pin[:,:2].T,color=color,ls="--",lw=.8)
    ax.scatter(*data["release"][:,:2].T,marker="^",s=26,color="#24282c",label="Release")
    ax.scatter(*data["target"].T,s=52,facecolors="none",edgecolors=DOP,label="Assigned target")
    ax.scatter(*data["dop853_endpoint"].T,s=20,marker="+",color=PINN,label="DOP853 at T")
    ax.legend(loc="upper left",bbox_to_anchor=(0,1.42),frameon=False,ncol=3,fontsize=8)
    ax.set_title("Solid DOP853  |  dashed PINN",loc="right",fontsize=8)
    T = metric["target_time"]
    critical = metric["methods"]["dop853"]["minimum"]
    for i,start in enumerate(data["release_time"],1):
        color = "#a23d21" if i in (critical["i"],critical["j"]) else "#435d7b"
        bx.hlines(i,start,T,color=color,lw=2)
        bx.scatter([start,T],[i,i],color=color,s=16)
    bx.axvline(T,color="#2b353e",lw=.9,ls="--")
    bx.axvline(critical["time"],color=PINN,lw=.9,ls=":")
    bx.set(xlabel=r"Global time $s^*$",ylabel="Assigned target ID",
           yticks=np.arange(1,len(data["target"])+1),ylim=(.55,len(data["target"])+.5))
    bx.set_title(f"T={T:.4f}  |  Critical pair {critical['i']}–{critical['j']}  |  Minimum spacing {critical['distance']:.5f}",
                 loc="left",fontsize=9)
    return fig


def formation_zoom(ax, data, case, metrics, *, title, compact=False):
    xy = data["target"]
    geom = np.asarray(case["outline"]) if case["outline"] else xy
    pad = .031
    ax.set(xlim=(geom[:,0].min()-pad,geom[:,0].max()+pad),
           ylim=(geom[:,1].min()-pad,geom[:,1].max()+pad),xlabel=r"$x^*$",ylabel=r"$y^*$")
    ax.set_aspect("equal",adjustable="box")
    if case["outline"]:
        ax.plot(*geom.T,lw=.8,color="#738397")
    for i,j in case["target_edges"] or []:
        ax.plot(*xy[[i,j]].T,color="#9fa8b3",lw=.9)
    for endpoint in data["dop853_endpoint"]:
        if np.isfinite(endpoint).all():
            ax.add_patch(Circle(endpoint,RADIUS,facecolor="#bed8d1",edgecolor="#35756a",linewidth=.55,alpha=.85))
    ax.scatter(*xy.T,color=DOP,s=8,marker="+",zorder=4)
    stats = metrics["methods"]["dop853"]
    line_break = "\n" if compact else "  |  "
    ax.set_title(f"{title}  |  {stats['completed']}/{len(xy)} at T\n"
                 f"RMS {stats['endpoint_rms']:.1e}, max {stats['endpoint_max']:.1e}"
                 f"{line_break}Clearance margin {stats['clearance_margin']:+.4f}",fontsize=8)


def dense_comparison(output,matrix,pair):
    fig,axes = plt.subplots(2,1,figsize=(8.7,5.8),layout="constrained")
    for ax,row in zip(axes,pair):
        data, metrics = load_case(output,row)
        formation_zoom(ax,data,matrix[row["id"]],metrics,
                       title="Six fixed ports" if row["mode"]=="fixed" else "Free wall locations")
    fig.legend(handles=[Line2D([],[],marker="+",ls="none",color=DOP,label="Assigned target"),
                        Patch(facecolor="#bed8d1",edgecolor="#35756a",label="DOP853 disk at T")],
               frameon=False,loc="outside lower center",ncol=2,fontsize=8)
    return fig


def formation_sweeps(rows):
    fig,axes = plt.subplots(3,3,figsize=(11,8.4),layout="constrained",height_ratios=[1,1.1,1.1])
    for column,key in enumerate(("density","launchers","capacity")):
        top,middle,bottom = axes[:,column]
        if key=="density":
            categories = [str(n) for n in (24,48,72,100)]
            groups = [(mode,[r for r in rows if r["study"]=="density" and r["mode"]==mode])
                      for mode in ("fixed","free")]
            value = lambda r: r["N"]
            label = "Target N"
        elif key=="launchers":
            categories = [str(n) for n in (2,4,6,8,10,12)]+["Free"]
            groups = [(study,[r for r in rows if r["stage"]=="C" and
                              (r["study"]==study or study=="equispaced" and r["study"]=="free")])
                      for study in ("equispaced","nested")]
            value = lambda r: r["port_count"] or "Free"
            label = "Total ports (N=24)"
        else:
            categories = ["3","4","6","Unlimited"]
            groups = [("six fixed ports",[r for r in rows if r["study"]=="capacity"])]
            value = lambda r: r["cap"] or "Unlimited"
            label = "Per-port cap (N=24)"
        for i,(mode,group) in enumerate(groups):
            color = (DOP,PINN)[i]
            offset = 0 if len(groups)==1 else (-.16 if i==0 else .16)
            for j,category in enumerate(categories):
                chosen = [r for r in group if value(r)==category]
                if not chosen:
                    continue
                attempted = [r for r in chosen if r["status"] not in ("unattempted","preflight_rejected")]
                accepted = sum(r["status"]=="accepted" for r in attempted)
                rejected = sum(r["status"]=="preflight_rejected" for r in chosen)
                if attempted:
                    rate = accepted/len(attempted)
                    top.scatter(j+offset,rate,color=color,marker=("o","s")[i],s=18)
                    top.text(j+offset,rate+.064+.075*i,f"{accepted}/{len(attempted)}",
                             color=color,ha="center",fontsize=7)
                if rejected:
                    top.text(j+offset,.47,f"{rejected} preflight",color="#62666c",
                             ha="center",fontsize=7,rotation=45)
                for k,row in enumerate(chosen):
                    margin = row["ref_clearance_margin"]
                    if margin and row["ref_interval"]=="full_to_T":
                        style = dict(edgecolors=color,facecolors=color if row["status"]=="accepted" else "none",
                                     s=24,marker=("o","s")[i])
                        x = j+offset+(k-1)*.046
                        middle.scatter(x,float(margin),**style)
                        bottom.scatter(x,max(float(row["ref_endpoint_max"]),FLOOR),**style)
            top.plot([],[],marker=("o","s")[i],color=color,linestyle="none",label=mode)
        top.set(title=key.capitalize(),ylim=(-.12,1.26),yticks=[0,.5,1])
        top.legend(frameon=False,loc="lower left",fontsize=7)
        bottom.set_xticks(range(len(categories)),categories)
        bottom.set(xlabel=label,yscale="log")
        middle.axhline(0,color="#788591",lw=.8,ls="--")
        for a in (top,middle,bottom):
            a.set_xlim(-.52,len(categories)-.48)
            a.grid(axis="y",alpha=.15)
        for ax in (top,middle):
            ax.set_xticks(range(len(categories)),[""]*len(categories))
    axes[0,0].set_ylabel("Accepted / attempted")
    axes[1,0].set_ylabel("DOP853 clearance margin")
    axes[2,0].set_ylabel("DOP853 max endpoint error")
    fig.legend(handles=[Line2D([],[],marker="o",ls="none",markerfacecolor="none",color=DOP,
                              label="Rejected replay"),
                        Line2D([],[],ls="none",label=r"Endpoint error floor $10^{-13}$")],
               loc="outside lower center",ncol=2,frameon=False,fontsize=8)
    return fig


def extension_maps(rows):
    fig, axes = plt.subplots(1,2,figsize=(8.5,3.35),layout="constrained")
    center = [r for r in rows if r["study"]=="center"]
    ellipse = [r for r in rows if r["study"]=="ellipse"]
    for ax,group,key,title in ((axes[0],center,"center","Target-center slice"),
                               (axes[1],ellipse,"ellipse","Ellipse rotation")):
        xs = sorted({float(r["factor_x"]) for r in group})
        ys = sorted({float(r["factor_y"]) for r in group})
        counts = np.full((len(ys),len(xs)),np.nan)
        for j,y in enumerate(ys):
            for i,x in enumerate(xs):
                chosen = [r for r in group if float(r["factor_x"])==x and float(r["factor_y"])==y]
                actual = [r for r in chosen if r["status"]!="unattempted"]
                if actual:
                    counts[j,i] = sum(r["status"]=="accepted" for r in actual)
                    ax.text(i,j,f"{int(counts[j,i])}/{len(actual)}",ha="center",va="center",
                            color="white" if counts[j,i]>=2 else "black",fontsize=9)
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#d7d7d7")
        ax.imshow(counts,vmin=0,vmax=3,cmap=cmap,origin="lower",aspect="auto")
        ax.set(xticks=range(len(xs)),xticklabels=[str(x) for x in xs],
               yticks=range(len(ys)),yticklabels=[str(y) for y in ys],title=title,
               xlabel=r"$x_c^*$" if key=="center" else "Rotation (degrees)",
               ylabel=r"$y_c^*$" if key=="center" else "Aspect ratio")
    fig.legend(handles=[Patch(facecolor="#d7d7d7",label="No scheduled angle at this aspect ratio")],
               loc="outside lower center",frameon=False,fontsize=8)
    return fig


def letter_figure(output,matrix,rows):
    fig,axes = plt.subplots(1,3,figsize=(9.7,3.65),layout="constrained")
    for ax,row in zip(axes,rows):
        data, metrics = load_case(output,row)
        formation_zoom(ax,data,matrix[row["id"]],metrics,title=row["shape"][0],compact=True)
    return fig


def generate(output):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,
                         "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
                         "savefig.dpi":320})
    destination = output/"figures"
    destination.mkdir(exist_ok=True)
    catalog = []

    def save(identity,title,fig,ids,source,methods,quantity,use,caption):
        pdf,png = (f"{identity}_{title}.{extension}" for extension in ("pdf","png"))
        fig.savefig(destination/pdf)
        fig.savefig(destination/png,dpi=320)
        plt.close(fig)
        catalog.append(dict(id=identity,pdf=f"figures/{pdf}",png=f"figures/{png}",
                            case_ids=";".join(ids),source=source,methods=methods,
                            quantity=quantity,use=use,caption_fa=caption))

    validation = output/"validation"
    if (validation/"time_series.npz").exists():
        with np.load(validation/"time_series.npz") as archive:
            data = {name:archive[name] for name in archive.files}
        rows = read_csv(validation/"case_diagnostics.csv")
        summary = json.loads((validation/"summary.json").read_text())
        save("P01","method",methods_diagram(),[],"checkpoint config and model.py",
             "ParticlePINN; analytical carrier","inputs, output, derivatives (dimensionless)","methods",
             "ساختار پارامتری PINN ذره و محاسبهٔ باقیماندهٔ معادلات استوکس.")
        save("P02","trajectories",trajectories(data,rows),REPRESENTATIVES,
             "validation/time_series.npz; validation/events.csv","PINN; DOP853",
             "channel coordinates and absolute position errors (dimensionless)","validation",
             "مقایسهٔ مسیر و نقطهٔ توقف در شش شرط رهاسازی، همراه با خطای مؤلفه‌ای در زمان مشترک.")
        save("P03","validation",validation_overview(rows,summary),data["case_id"],
             "validation/case_diagnostics.csv; validation/error_percentiles.csv","PINN; DOP853",
             "per-case maximum absolute errors; event agreement","validation",
             "توزیع خطاهای موقعیت و سرعت و انطباق نوع رویداد برای ۷۰ شرط آزمون.")
        save("P04","training",training(),[],"../../cpu/training.jsonl; ../../cpu/training_summary.json",
             "recorded training and development residuals","scaled residual MSE","appendix",
             "روند ثبت‌شدهٔ آموزش و توسعه و گام انتخاب‌شدهٔ وزن مدل.")
    if (output/"slices/grid.csv").exists():
        save("P05","release_slices",slices_figure(read_csv(output/"slices/grid.csv")),
             ["lower","upper","interior"],"slices/grid.csv; slices/matched.npz","PINN; DOP853",
             "event category, event time and matched-interval position errors","validation",
             "برش‌های فضای شرایط رهاسازی و خطای مسیر در بازهٔ فیزیکی مشترک.")
    form = output/"formation"
    if (form/"summary.csv").exists():
        rows = read_csv(form/"summary.csv")
        matrix = {r["id"]:r for r in json.loads((form/"matrix.json").read_text())}
        accepted = [r for r in rows if r["status"]=="accepted"]
        representative = next((r for r in accepted if r["study"]=="triangle"),None)
        if representative:
            save("F01","verified_triangle",formation_overview(output,representative),[representative["id"]],
                 "formation/<case>/schedule.csv; trajectories.npz; metrics.json","direct PINN; Stokes DOP853",
                 "channel, release clock, and critical clearance (dimensionless)","formation",
                 "شکل‌دهی مثلث، مسیر ذرات و زمان‌های رهاسازی روی ساعت مشترک.")
        pairs = {(r["N"],r["seed"],r["mode"]):r for r in accepted if r["study"]=="density"}
        for n,seed in sorted({(n,seed) for n,seed,_ in pairs},key=lambda pair:(-int(pair[0]),int(pair[1]))):
            if all((n,seed,mode) in pairs for mode in ("fixed","free")):
                comparison = [pairs[(n,seed,mode)] for mode in ("fixed","free")]
                save("F02","matched_density",dense_comparison(output,matrix,comparison),[r["id"] for r in comparison],
                     "formation/matrix.json; <case>/endpoints.csv; trajectories.npz; metrics.json",
                     "direct PINN schedule; Stokes DOP853 endpoint disks","endpoint position, RMS, and clearance",
                     "formation","مقایسهٔ تطبیق‌یافتهٔ استادیوم با پورت ثابت و مکان آزاد رهاسازی.")
                break
        comparative = [r for r in rows if r["study"]=="density" or r["stage"]=="C"]
        if any(r["status"]=="accepted" for r in comparative):
            save("F03","formation_sweeps",formation_sweeps(rows),[r["id"] for r in comparative],
                 "formation/summary.csv","direct PINN search; Stokes DOP853",
                 "attempt counts, clearance margins and max endpoint errors (dimensionless)","formation",
                 "تعداد موفقیت‌های تأییدشده و حاشیهٔ فاصله در تغییر تعداد ذرات، پورت و ظرفیت.")
        mapped = [r for r in rows if r["study"] in ("center","ellipse")]
        if any(r["status"]!="unattempted" for r in mapped):
            for r in mapped:
                c = matrix[r["id"]]
                r["factor_x"],r["factor_y"] = c["center"] if r["study"]=="center" else (c["theta"],c["q"])
            save("F04","geometry_slices",extension_maps(mapped),[r["id"] for r in mapped],
                 "formation/matrix.json; formation/summary.csv","direct PINN search; Stokes DOP853",
                 "number of verified trials at each target configuration","appendix",
                 "تعداد آزمایش‌های تأییدشده در برش مکان مرکز استادیوم و هندسهٔ بیضی.")
        letters = [next((r for r in accepted if r["shape"]==name),None) for name in ("S_6","C_6","T_6")]
        if all(letters):
            save("F05","letters",letter_figure(output,matrix,letters),[r["id"] for r in letters],
                 "formation/matrix.json; <case>/endpoints.csv; trajectories.npz; metrics.json",
                 "direct PINN schedule; Stokes DOP853 endpoint disks","assigned centers and replay endpoint disks",
                 "formation","حروف S، C و T با شش پورت ثابت و افق پرواز پنج واحد بی‌بُعد.")
    if not catalog:
        raise ValueError("No evidence found to render.")
    write_csv(output/"figure_catalog.csv",catalog)
    return {"count":len(catalog),"ids":[r["id"] for r in catalog]}
