"""Deterministic release-condition slices with matched independent references."""
from time import perf_counter

import numpy as np

from ..inference import predict_trajectory, wall_release
from ..validate import reference
from .evidence import write_csv, write_json


def sweep(model, output):
    started = perf_counter()
    config = model.config
    dest = output/"slices"
    dest.mkdir(exist_ok=True)
    x0, vx0, query = .66, .35, 1.
    vx = np.linspace(*config["domain"]["vx0"],19)
    inward = np.linspace(*config["domain"]["inward_speed"],17)
    radius = config["physics"]["diameter"]/2
    ys = np.linspace(radius,config["physics"]["height"]-radius,19)
    vy = np.linspace(*config["domain"]["interior_vy0"],19)
    rows = []
    networks, refs, masks = {}, {}, {}
    for name in ("lower","upper","interior"):
        axis_x,axis_y = (ys,vy) if name=="interior" else (vx,inward)
        u,v = np.meshgrid(axis_x,axis_y)
        if name=="interior":
            releases = np.stack((np.full_like(u,x0),u,np.full_like(u,vx0),v),axis=-1)
        else:
            releases = wall_release(config,name,x0,u,v)
        releases = releases.reshape(-1,4)
        recorded = predict_trajectory(model,*releases.T,[query])
        references = [reference(config,r) for r in releases]
        stop = np.array([p[1] for p in references])
        end = recorded["event"]["time"]
        common = np.minimum(stop,end)[:,None]*np.linspace(0,1,121)
        predicted = predict_trajectory(model,*releases.T,common)
        assert predicted["valid"].all()
        truth = np.stack([p[2](t) for p,t in zip(references,common)])
        actual = np.concatenate((predicted["position"],predicted["velocity"]),axis=-1)
        networks[name],refs[name],masks[name] = actual,truth,common
        error = abs(actual-truth).max(axis=1)
        for i,release in enumerate(releases):
            p = references[i]
            valid = query <= p[1]
            ref = p[2](np.array([query]))[0,:2] if valid else [np.nan,np.nan]
            xy = recorded["position"][i,0]
            rows.append(dict(slice=name,row=i//len(axis_x),column=i%len(axis_x),
                             axis_x=u.ravel()[i],axis_y=v.ravel()[i],
                             x0=release[0],y0=release[1],vx0=release[2],vy0=release[3],
                             query_time=query,pinn_event=recorded["event"]["type"][i],
                             pinn_event_time=end[i],dop853_event=p[0],dop853_event_time=p[1],
                             valid_pinn=bool(recorded["valid"][i,0]),valid_dop853=valid,
                             pinn_x=xy[0],pinn_y=xy[1],dop853_x=ref[0],dop853_y=ref[1],
                             max_x_error=error[i,0],max_y_error=error[i,1],
                             max_vx_error=error[i,2],max_vy_error=error[i,3]))
    write_csv(dest/"grid.csv",rows)
    np.savez_compressed(dest/"matched.npz",**{f"{name}_{key}":val for name,items in
                       (("pinn",networks),("dop853",refs),("common_time",masks)) for key,val in items.items()})
    lower,upper = networks["lower"],networks["upper"]
    reflection = upper.copy()
    reflection[...,1] = config["physics"]["height"]-reflection[...,1]
    reflection[...,3] *= -1
    result = dict(count=len(rows), grids=dict(lower=[17,19],upper=[17,19],interior=[19,19]),
                  x0=x0, interior_vx0=vx0, query_time=query,
                  horizontal_velocity=vx.tolist(), inward_speed=inward.tolist(),
                  interior_y0=ys.tolist(), interior_vy0=vy.tolist(),
                  matching_events=sum(r["pinn_event"]==r["dop853_event"] for r in rows),
                  maximum_mirror_error=abs(lower-reflection).max(axis=(0,1)).tolist(),
                  maximum_mirror_time_grid_delta=abs(masks["lower"]-masks["upper"]).max(),
                  query_invalid=sum(not r["valid_pinn"] or not r["valid_dop853"] for r in rows))
    result["elapsed_seconds"] = perf_counter()-started
    write_json(dest/"summary.json",result)
    return result
