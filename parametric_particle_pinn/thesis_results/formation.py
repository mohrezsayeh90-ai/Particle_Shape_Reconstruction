"""Inverse release search using only the direct PINN; strict same-clock replay."""
from collections import Counter
from time import perf_counter

import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment, minimize_scalar

from New_Main_Results.formation.planner import Planner
from New_Main_Results.formation.targets import covariance_metrics, preflight
from ..inference import _events
from ..validate import reference
from .evidence import exact_state, write_csv, write_json

SEARCH, CLEARANCE, DIAMETER = .0263, .026, .025
EVENT_TOL = 1e-10


class Path:
    def __init__(self, kind, end, solve):
        self.kind, self.end, self.solve = kind, float(end), solve

    def sol(self, time):
        return self.solve(np.asarray(time)).T


class SearchPhysics:
    """Adapter used only by the legacy normalized sparse refinement routine."""
    def __init__(self, engine):
        self.engine = engine

    def trace(self, controls, side, _age, _mode):
        x, vx, vy = controls[3*side:3*side+3]
        release = self.engine.wall(side, x, vx, abs(vy))
        kind, end = self.engine.event(release)
        return Path(kind, end, lambda t: self.engine.position(release, t, velocity=True))


def candidates(engine, case, targets):
    """Invert the learned endpoint map at a predeclared discrete age bank."""
    lo, hi = case["min_age"], case["max_age"]
    budget = case["budgets"]
    ages = np.linspace(lo, hi, budget["age_nodes"])
    pinned = case["mode"] == "fixed"
    ports = case["ports"] if pinned else [{"side": 0, "x": 0.}, {"side": 1, "x": 0.}]
    bases = engine.basis(ages)
    vx_low, vx_high = engine.domain["vx0"]
    vy_low, vy_high = engine.domain["inward_speed"]
    options = []
    for target in targets:
        group = []
        for index, port in enumerate(ports):
            side = port["side"]
            y0 = engine.wall(side, 0., 0., vy_low)[1]
            vy = (target[1]-y0)/(ages+bases[:,4]/engine.domain["interior_vy0"][1])
            base_release = np.column_stack((np.zeros_like(ages), np.full_like(ages,y0),
                                            np.zeros_like(ages), vy))
            intercept = engine.position(base_release, ages, bases)[:,0]
            response = ages+bases[:,3]/engine.physics["centerline_speed"]
            if pinned:
                px = np.full((len(ages),1),port["x"])
                pv = ((target[0]-port["x"]-intercept)/response)[:,None]
            else:
                pv = np.broadcast_to(np.linspace(vx_low,vx_high,budget["free_velocity_nodes"]), (len(ages),budget["free_velocity_nodes"]))
                px = target[0]-intercept[:,None]-response[:,None]*pv
            sign = 1 if side == 0 else -1
            good = ((sign*vy[:,None]>=vy_low) & (sign*vy[:,None]<=vy_high) &
                    (pv>=vx_low) & (pv<=vx_high) & (px>=0) & (px<=engine.physics["length"]))
            aa, vv = np.nonzero(good)
            if not len(aa):
                continue
            # Spread trials across the feasible interval and velocity grid.
            trial = np.unique(np.r_[np.linspace(0,len(aa)-1,min(2*budget["candidates_per_slot"],len(aa))).round().astype(int)])
            valid = []
            for k in trial:
                i,j = aa[k],vv[k]
                release = engine.wall(side, px[i,j], pv[i,j], sign*vy[i])
                _, end = engine.event(release)
                if end >= ages[i]:
                    valid.append((release, ages[i], index))
            if len(valid) > budget["candidates_per_slot"]:
                keep = np.linspace(0,len(valid)-1,budget["candidates_per_slot"]).round().astype(int)
                valid = [valid[k] for k in keep]
            group.extend(valid)
        options.append(group)
    return options


@njit(cache=True)
def allocation_score(index, other, ages, curves):
    back = min(ages[index], ages[other])
    scale = curves.shape[1]-1
    minimum = 10.
    for k in range(41):
        t = back*k/40
        a = min(scale, t*scale/ages[index])
        b = min(scale, t*scale/ages[other])
        i,j = min(int(a),scale-1),min(int(b),scale-1)
        dx = curves[index,i,0]*(1-a+i)+curves[index,i+1,0]*(a-i)-curves[other,j,0]*(1-b+j)-curves[other,j+1,0]*(b-j)
        dy = curves[index,i,1]*(1-a+i)+curves[index,i+1,1]*(a-i)-curves[other,j,1]*(1-b+j)-curves[other,j+1,1]*(b-j)
        distance = (dx*dx+dy*dy)**.5
        minimum = min(minimum, distance)
    return max(0., SEARCH-minimum)**2


def allocate(engine, case, options):
    rng = np.random.default_rng(case["seed"])
    boundaries = np.r_[0,np.cumsum([len(g) for g in options])]
    groups = [np.arange(a,b) for a,b in zip(boundaries[:-1],boundaries[1:])]
    if any(not len(g) for g in groups):
        return None
    releases, ages, ports = zip(*(entry for group in options for entry in group))
    releases, ages, ports = np.array(releases), np.array(ages), np.array(ports)
    pinned = case["mode"] == "fixed"
    cap = case["cap"]
    if pinned and cap is not None:
        slots = np.repeat(np.arange(len(case["ports"])),cap)
        cost = np.array([np.where(np.isin(slots, ports[g]),0.,1e6) for g in groups])
        cost += rng.uniform(0,1e-8,cost.shape)
        rows, cols = linear_sum_assignment(cost)
        if len(rows)!=len(groups) or (cost[rows,cols]>1).any():
            return None
        selected = np.array([rng.choice(groups[i][ports[groups[i]]==slots[col]]) for i,col in zip(rows,cols)])
    else:
        selected = np.array([rng.choice(g) for g in groups])
    times = ages[:,None]*np.linspace(1,0,129)
    curves = engine.position(releases[:,None,:], times)
    sweeps, evaluations = 0, 0
    for _ in range(case["budgets"]["allocation_sweeps"]):
        sweeps += 1
        changed = False
        counts = Counter(ports[selected])
        for i in rng.permutation(len(groups)):
            current = selected[i]
            counts[ports[current]] -= 1
            competitors = np.delete(selected,i)
            feasible = groups[i] if not pinned or cap is None else [k for k in groups[i] if counts[ports[k]]<cap]
            def score(k):
                value = 0.
                for j in competitors:
                    if case["collision_aware"]:
                        value += allocation_score(k,j,ages,curves)
                    if pinned and ports[k] == ports[j]:
                        value += max(0.,case["min_gap"]-abs(ages[k]-ages[j]))**2
                return value
            best = score(current)
            evaluations += 1
            for k in feasible:
                value = score(k)
                evaluations += 1
                if value < best-1e-16:
                    selected[i],best = k,value
            counts[ports[selected[i]]] += 1
            changed |= current != selected[i]
        if not changed:
            break
    return (releases[selected],ages[selected],ports[selected],
            dict(allocation_sweeps=sweeps,candidate_scores=evaluations,candidate_count=len(ages)))


def refine_small(engine, case, releases, ages, ports, targets):
    """Keep the legacy bounded sparse objective for small formations."""
    planner = Planner(SearchPhysics(engine),case,"direct")
    rows = []
    for release, age, slot in zip(releases,ages,ports):
        side = int(slot) if case["mode"] == "free" else case["ports"][slot]["side"]
        controls = np.zeros(6)
        controls[3*side:3*side+3] = release[[0,2,3]]
        rows.append(planner.pack(controls,side,age))
    z = np.asarray(rows).ravel()
    step, diagnostic = planner.refine(z,ports,targets,case["budgets"]["nfev"])
    changed = [planner.physical(row,port) for row,port in zip(step.reshape(-1,planner.dim),ports)]
    states, times = [], []
    for c,side,age in changed:
        x,vx,vy = c[3*side:3*side+3]
        states.append(engine.wall(side,x,vx,abs(vy)))
        times.append(age)
    return np.array(states), np.array(times), diagnostic


def synchronized_minima(paths, releases, stop, samples):
    """Evaluate all pairs with their individual overlap endpoints and local refinement."""
    mesh = np.linspace(float(min(releases)),float(max(stop)),samples)
    starts = np.asarray(releases)
    ends = np.asarray(stop)
    states = np.full((len(paths),len(mesh),2),np.nan)
    for i,path in enumerate(paths):
        valid = (mesh>=starts[i]) & (mesh<=ends[i])
        if valid.any():
            states[i,valid] = path.solve(np.clip(mesh[valid]-starts[i],0,path.end))[:,:2]
    result = []
    for i,j in zip(*np.triu_indices(len(paths),1)):
        lo,hi = max(starts[i],starts[j]),min(ends[i],ends[j])
        if hi < lo:
            continue
        def distance(t):
            a,b = (paths[k].solve(np.clip(t-starts[k],0,paths[k].end))[:2] for k in (i,j))
            return float(np.linalg.norm(a-b))
        first,last = np.searchsorted(mesh,[lo,hi],side="left")
        mids = mesh[first:last]
        dv = np.linalg.norm(states[i,first:last]-states[j,first:last],axis=-1)
        ts = np.r_[lo,mids,hi]
        ds = np.r_[distance(lo),dv,distance(hi)]
        sampled = int(ds.argmin())
        winner = (float(ds[sampled]),float(ts[sampled]))
        for k in np.flatnonzero((ds[1:-1]<=ds[:-2]) & (ds[1:-1]<ds[2:]))+1:
            if ts[k+1]-ts[k-1] < 1e-13:
                continue
            opt = minimize_scalar(distance,bounds=(ts[k-1],ts[k+1]),method="bounded",
                                  options={"xatol":1e-11})
            if opt.fun < winner[0]:
                winner = (float(opt.fun),float(opt.x))
        # A minimum just before/after an overlap endpoint may be missed by
        # interior grid samples. Refine that edge bracket as well.
        if sampled in (0,len(ts)-1) and hi-lo>1e-12:
            time = ts[sampled]
            a,b = (paths[k].solve(np.clip(time-starts[k],0,paths[k].end)) for k in (i,j))
            slope = (a[:2]-b[:2]) @ (a[2:]-b[2:])
            towards_interior = slope < 0 if sampled==0 else slope > 0
            edge = (ts[0],ts[1]) if sampled==0 else (ts[-2],ts[-1])
            if towards_interior and edge[1]-edge[0]>1e-12:
                opt = minimize_scalar(distance,bounds=edge,method="bounded",options={"xatol":1e-11})
                if opt.fun < winner[0]:
                    winner = (float(opt.fun),float(opt.x))
        result.append(dict(i=int(i+1),j=int(j+1),distance=winner[0],time=winner[1],
                           sampled_distance=float(ds[sampled]),overlap_start=float(lo),overlap_end=float(hi)))
    return result


def replay(engine, model, case, folder, releases, ages, slots, targets):
    T = .25+float(max(ages))
    clock = T-ages
    pinned = case["mode"] == "fixed"
    ids = [case["ports"][k]["id"] if pinned else f"free_{i+1}" for i,k in enumerate(slots)]
    sides = [case["ports"][k]["side"] if pinned else int(k) for k in slots]
    rows = [dict(target_id=i+1,port_id=ids[i],side=sides[i],x0=r[0],y0=r[1],vx0=r[2],vy0=r[3],
                 release_time=clock[i],age_at_target=ages[i],target_x=targets[i,0],target_y=targets[i,1])
            for i,r in enumerate(releases)]
    # Freeze intended inputs before the independent solver reads a schedule.
    write_csv(folder/"schedule.csv",rows)
    event_kind,event_end = _events(model,releases)
    neural = [Path(k,e,lambda t,r=row: engine.position(r,t,velocity=True))
              for row,k,e in zip(releases,event_kind,event_end)]
    refs = []
    for row in releases:
        k,e,sol = reference(model.config,row)
        # The initial-event reference returns a row array even for scalar calls.
        refs.append(Path(k,e,lambda t,s=sol: s(t).reshape(np.shape(t)+(4,))))
    methods = {}
    paths_out = {}
    shared = np.linspace(float(clock.min()),T,401)
    all_endpoints, all_completion = [], []
    for label,paths in (("pinn",neural),("dop853",refs)):
        ends = np.array([p.end for p in paths])
        complete = ages-ends <= EVENT_TOL
        stop = np.minimum(T,clock+ends)
        positions = np.full((len(paths),len(shared),4),np.nan)
        valid = np.zeros(positions.shape[:2],bool)
        endpoints = np.full_like(targets,np.nan)
        terminal = np.array([p.solve(min(a,p.end)) for a,p in zip(ages,paths)])
        endpoints[complete] = terminal[complete,:2]
        for i,p in enumerate(paths):
            valid[i] = (shared>=clock[i]) & (shared<=stop[i])
            if valid[i].any():
                positions[i,valid[i]] = p.solve(np.clip(shared[valid[i]]-clock[i],0,p.end))
        all_endpoints.append(endpoints)
        all_completion.append(complete)
        pairs = synchronized_minima(paths,clock,stop,801)
        expected_pairs = len(paths)*(len(paths)-1)//2
        finest = min((row["distance"] for row in pairs),default=np.inf)
        near = {(row["i"],row["j"]) for row in pairs if row["distance"] <= finest+.0005}
        finer = synchronized_minima(paths,clock,stop,1601)
        previous = {(row["i"],row["j"]):row for row in pairs}
        for row in finer:
            key = (row["i"],row["j"])
            coarse = previous[key]
            row.update(coarse_distance=coarse["distance"], near_minimum=key in near,
                       refined_delta=abs(coarse["distance"]-row["distance"]),
                       i_terminated_before_T=bool(not complete[row["i"]-1]),
                       j_terminated_before_T=bool(not complete[row["j"]-1]))
        if finer:
            write_csv(folder/f"pair_minima_{label}.csv",finer)
        else:
            (folder/f"pair_minima_{label}.csv").write_text("i,j,distance,time,overlap_start,overlap_end\n")
        critical = min(finer,key=lambda r:r["distance"]) if finer else None
        err = np.linalg.norm(endpoints[complete]-targets[complete],axis=-1)
        methods[label] = dict(completed=int(complete.sum()), compared_pairs=len(finer), expected_pairs=expected_pairs,
                             minimum=critical, interval="full_to_T" if complete.all() else "valid_partial_overlaps",
                             endpoint_max=float(err.max()) if err.size else None,
                             endpoint_rms=float(np.sqrt(np.mean(err**2))) if err.size else None,
                             clearance_margin=critical["distance"]-CLEARANCE if critical else None,
                             maximum_refined_resolution_delta=max((r["refined_delta"] for r in finer),default=None),
                             covariance=covariance_metrics(targets,endpoints) if complete.all() else None)
        for i,p in enumerate(paths):
            rows[i].update({f"{label}_event":p.kind,f"{label}_event_age":p.end,
                            f"{label}_stop_global":stop[i],f"{label}_complete":bool(complete[i]),
                            f"{label}_last_x":terminal[i,0],f"{label}_last_y":terminal[i,1]})
        paths_out.update({label:positions,f"{label}_valid":valid,
                          f"{label}_terminal":terminal,f"{label}_stop_global":stop})
    pin,dop = all_endpoints
    write_csv(folder/"endpoints.csv", (dict(**r,pinn_x=pin[i,0],pinn_y=pin[i,1],dop853_x=dop[i,0],
                   dop853_y=dop[i,1],pinn_error=np.linalg.norm(pin[i]-targets[i]),
                   dop853_error=np.linalg.norm(dop[i]-targets[i])) for i,r in enumerate(rows)))
    np.savez_compressed(folder/"trajectories.npz",shared_time=shared,release_time=clock,
                        release=releases,age_at_target=ages,target=targets,pinn_endpoint=pin,
                        dop853_endpoint=dop,**paths_out)
    overlap = all_completion[0] & all_completion[1]
    exact = np.array([exact_state(model.config,r,np.asarray(a)).reshape(4)[:2]
                      for r,a in zip(releases[all_completion[1]],ages[all_completion[1]])])
    transfer = float(np.linalg.norm(pin[overlap]-dop[overlap],axis=-1).max()) if overlap.any() else None
    layout_ok = np.allclose(clock+ages,T,rtol=0,atol=1e-12) and all(
        case["min_age"]-1e-12<=a<=case["max_age"]+1e-12 for a in ages)
    if pinned:
        layout_ok &= all(releases[i,0]==case["ports"][k]["x"] for i,k in enumerate(slots))
        layout_ok &= case["cap"] is None or max(Counter(ids).values())<=case["cap"]
        layout_ok &= all(np.all(np.diff(np.sort(clock[np.array(ids)==port]))>=case["min_gap"]-1e-12)
                         for port in set(ids))
    failure = []
    if not layout_ok:
        failure.append("schedule_constraints")
    if not overlap.all():
        failure.append("early_termination")
    for label, tolerance in (("pinn",.003),("dop853",.004)):
        entry = methods[label]
        if entry["endpoint_max"] is not None and entry["endpoint_max"]>tolerance:
            failure.append(label+"_endpoint_miss")
        if entry["minimum"] is None or entry["minimum"]["distance"]<CLEARANCE:
            distance = entry["minimum"]["distance"] if entry["minimum"] else np.inf
            failure.append(label+("_disk_overlap" if distance<DIAMETER else "_clearance_shortfall"))
    if transfer is not None and transfer>.004:
        failure.append("transfer_error")
    return dict(status="accepted" if not failure else "rejected",reasons=failure,target_time=T,
                event_tolerance=EVENT_TOL,methods=methods,transfer_max=transfer,
                exact_vs_dop853_max=float(np.linalg.norm(exact-dop[all_completion[1]],axis=-1).max()) if exact.size else None)


def run_case(engine,model,case,folder,options=None):
    folder.mkdir(parents=True,exist_ok=True)
    start = perf_counter()
    targets = np.asarray(case["targets"])
    reason, detail = preflight(targets,case["N"],case["ports"],case["cap"])
    if reason:
        result = dict(status="preflight_rejected",reasons=[reason],detail=detail)
    else:
        options = candidates(engine,case,targets) if options is None else options
        missing = [i+1 for i,o in enumerate(options) if not o]
        selection = allocate(engine,case,options)
        if selection is None:
            result = dict(status="search_failed",reasons=["candidate_shortage" if missing else "capacity_assignment_shortage"],
                          targets_without_candidates=missing, per_target_candidates=[len(x) for x in options])
        else:
            release,ages,slots,search = selection
            if case["N"]<=6:
                release,ages,diag = refine_small(engine,case,release,ages,slots,targets)
                search["joint_refinement"] = diag
            result = replay(engine,model,case,folder,release,ages,slots,targets)
            result["search"] = search
    result.update(case_id=case["id"],engine="direct_particle_pinn",replay="analytical_stokes_dop853",
                  elapsed_seconds=perf_counter()-start)
    write_json(folder/"metrics.json",result)
    return result
