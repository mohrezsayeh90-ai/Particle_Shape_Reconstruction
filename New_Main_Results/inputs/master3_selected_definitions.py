def exact_flow(xy, controls=None):
    """Independent plane Poiseuille reference; not an interior training dataset."""
    xy = np.atleast_2d(np.asarray(xy, float))
    eta = xy[:, 1]/H
    return np.column_stack((4*U_C*eta*(1-eta), np.zeros(len(xy)),
                            8*U_C*(L-xy[:, 0])/(RE*H**2)))

def flow_inputs(xy, controls):
    xy = np.atleast_2d(np.asarray(xy, float))
    c = np.asarray(controls, float)
    if c.ndim == 1: c = np.broadcast_to(c, (len(xy), 6))
    if xy.shape[1] != 2 or c.shape != (len(xy), 6):
        raise ValueError('Expected Nx2 coordinates and six shared or Nx6 controls.')
    return np.column_stack((xy, c))

def evaluate_flow(net, xy, controls=CONTROL_MID, batch_size=8192):
    packed = flow_inputs(xy, controls)
    parameter = next(net.parameters())
    chunks = []
    with torch.no_grad():
        for start in range(0, len(packed), batch_size):
            z = torch.as_tensor(packed[start:start+batch_size],
                                dtype=parameter.dtype, device=parameter.device)
            chunks.append(net(z).cpu().numpy())
    return np.vstack(chunks)

def check_controls(controls):
    c = np.asarray(controls, float)
    if c.shape != (6,) or not np.isfinite(c).all():
        raise ValueError('Each trajectory needs six finite flow-conditioning slots; active launch triplet is particle-specific.')
    if np.any(c < CONTROL_BOUNDS[:, 0]-1e-12) or np.any(c > CONTROL_BOUNDS[:, 1]+1e-12):
        raise ValueError('Launcher controls are outside the trained box.')
    return np.clip(c, CONTROL_BOUNDS[:, 0], CONTROL_BOUNDS[:, 1])

def particle_acceleration(fluid, particle_velocity):
    slip = np.asarray(fluid)-np.asarray(particle_velocity)
    rep = RE*DIAMETER*np.linalg.norm(slip)
    factor = 1.0
    if DRAG_LAW == 'schiller-naumann':
        factor = 1+.15*rep**.687 if rep < 1000 else .44*rep/24
    return 18*factor*slip/(DENSITY_RATIO*RE*DIAMETER**2)

class ParticlePath:
    def __init__(self, solution, side, controls, status):
        self.sol, self.side = solution, side
        self.controls, self.status = controls.copy(), status
        self.end = float(solution.t[-1])
        self.ok = bool(solution.success and np.isfinite(solution.y).all())

    def state(self, ages, strict=True):
        a = np.atleast_1d(np.asarray(ages, float))
        if strict and (np.any(a < -1e-9) or np.any(a > self.end+1e-8)):
            raise ValueError('Requested a particle state outside its physical trajectory.')
        return self.sol.sol(np.clip(a, 0, self.end)).T

def trace_particle(field, controls, side, tmax=TRAJECTORY_TMAX,
                   rtol=ODE_RTOL, atol=ODE_ATOL, max_step=TRAJECTORY_MAX_STEP):
    """field(xy, particle_controls) -> Nx2 or Nx3; no particle feedback to fluid."""
    c = check_controls(controls)
    if side not in (0, 1): raise ValueError('side=0 is bottom, side=1 is top.')
    j = 0 if side == 0 else 3
    y0 = DIAMETER/2+LAUNCH_GAP if side == 0 else H-DIAMETER/2-LAUNCH_GAP
    initial = [c[j], y0, c[j+1], c[j+2]]
    calls = 0
    def rhs(t, state):
        nonlocal calls
        calls += 1
        if calls > 100000: raise RuntimeError('ODE RHS budget exhausted.')
        # Event-bracketing stages can leave the domain. Query the nearest boundary;
        # never treat a neural extrapolation outside the rectangle as a solution.
        xy = np.clip(state[:2], [0, 0], [L, H])[None, :]
        fluid = np.asarray(field(xy, c))[0, :2]
        if not np.isfinite(fluid).all(): raise FloatingPointError('Non-finite fluid velocity.')
        acc = particle_acceleration(fluid, state[2:])
        return [state[2], state[3], acc[0], acc[1]]
    def lower(t, z): return z[1]-DIAMETER/2
    def upper(t, z): return H-DIAMETER/2-z[1]
    def inlet(t, z): return z[0]
    def outlet(t, z): return L-z[0]
    events = [lower, upper, inlet, outlet]
    for event in events: event.terminal, event.direction = True, -1
    tau = DENSITY_RATIO*RE*DIAMETER**2/18
    # Implicit integration avoids stiffness when users lower density or diameter.
    method = 'Radau' if tau < max_step/3 else 'RK45'
    solution = solve_ivp(rhs, (0, tmax), initial, events=events, dense_output=True,
        method=method, rtol=rtol, atol=atol, max_step=max_step)
    hit = next((i for i, values in enumerate(solution.t_events) if len(values)), None)
    status = (['lower_wall', 'upper_wall', 'inlet_exit', 'outlet'][hit] if hit is not None
              else 'time_limit' if solution.success else 'solver_failed')
    return ParticlePath(solution, side, c, status)

class FlowLookup:
    """2D speed table at midpoint controls; final verification uses all 8 inputs.

    This reduction is valid for the specified one-way fully developed flow only.
    Conditional drift is measured before optimization; the table is not a new MLP.
    """
    def __init__(self, net, nx=241, ny=121):
        self.x, self.y = np.linspace(0, L, nx), np.linspace(0, H, ny)
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        xy = np.column_stack((X.ravel(), Y.ravel()))
        values = evaluate_flow(net, xy, CONTROL_MID)[:, :2].reshape(nx, ny, 2)
        self.interpolate = RegularGridInterpolator((self.x, self.y), values,
                                                   bounds_error=True)
        self.values = values
    def __call__(self, xy, controls=None):
        return self.interpolate(np.clip(np.atleast_2d(xy), [0, 0], [L, H]))

def target_points(kind):
    if kind in ['triangle', 'square'] or kind.startswith('circle_'):
        n = 3 if kind == 'triangle' else 4 if kind == 'square' else int(kind.split('_')[1])
        if not 3 <= n <= 40: raise ValueError('Polygon count must be 3..40.')
        phase = np.pi/4 if kind == 'square' else np.pi/2
        a = phase+np.arange(n)*2*np.pi/n
        pts = TARGET_CENTER+TARGET_RADIUS*np.column_stack((np.cos(a), np.sin(a)))
        closed = True
    elif kind.startswith('s_'):
        n = int(kind.split('_')[1]); t = np.linspace(0, 1, 2001)
        curve = np.column_stack((S_WIDTH/2*np.sin(2*np.pi*t), S_HEIGHT*(.5-t)))
        # Equal arclength spacing avoids visually uneven point density.
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        knots = np.linspace(0, arc[-1], n)
        pts = TARGET_CENTER+np.column_stack([np.interp(knots, arc, curve[:, j]) for j in (0, 1)])
        closed = False
    elif kind.startswith('chevron_'):
        n = int(kind.split('_')[1]); t = np.linspace(-1, 1, n)
        pts = TARGET_CENTER+np.column_stack((TARGET_RADIUS*(1-2*np.abs(t)), TARGET_RADIUS*t))
        closed = False
    else: raise ValueError('Use triangle, square, circle_N, s_N or chevron_N.')
    if not 3 <= len(pts) <= 40: raise ValueError('Use 3..40 particles.')
    radius = DIAMETER/2
    if np.any(pts <= [radius, radius]) or np.any(pts >= [L-radius, H-radius]):
        raise ValueError(f'{kind}: target does not fit inside the rectangle with radius clearance.')
    distances = np.linalg.norm(pts[:, None]-pts[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    if distances.min() < DIAMETER+COLLISION_MARGIN:
        raise ValueError(f'{kind}: target points overlap at the configured diameter.')
    return pts, closed

def independent_optimization_snapshot(paths, controls, assignment, ages, targets, stage, step):
    """Physical ODE data for EVERY particle at an actual recorded solver state."""
    ages = np.asarray(ages, float)
    T = float(ages.max()+LEAD_TIME)
    actual = np.array([p.state([a], strict=False)[0, :2] for p, a in zip(paths, ages)])
    errors = np.linalg.norm(actual-targets, axis=1)
    launches, samples = [], []
    for i, (path, c, side, age) in enumerate(zip(paths, controls, assignment, ages)):
        side = int(side); j = 3*side
        launches.append(dict(particle=i+1, side=side, launcher=['bottom', 'top'][side],
            x_launch=float(c[j]), y_launch=float(path.state([0])[0, 1]),
            u_launch=float(c[j+1]), v_launch=float(c[j+2]),
            release_time=float(T-age), age_at_target=float(age)))
        tt = np.linspace(0, min(age, path.end), 81)
        samples.append(dict(particle=i+1, side=side, release=float(T-age),
            flow_controls=np.asarray(c).tolist(), ages=tt.tolist(), states=path.state(tt).tolist()))
    return dict(stage=stage, step=int(step), target_time=T,
        continuous_decisions=4*len(paths), launches=launches, paths=samples,
        actual=actual.tolist(), max_error=float(errors.max()),
        rms_error=float(np.sqrt(np.mean(errors**2))))

class IndependentLauncherPlanner:
    """One launch triplet and one physical flight age per particle, never shared.

    The six conditioning slots of the existing flow PINN remain unchanged.
    For each trajectory the active wall's three slots carry THAT particle's
    launch settings and the unused wall's three slots remain at their midpoints.
    This is valid for the declared one-way, control-independent carrier flow.
    """
    def __init__(self, field, label, rtol=2e-6, atol=2e-8, max_step=.06):
        self.field, self.label = field, label
        self.rtol, self.atol, self.max_step = rtol, atol, max_step
        self.cache, self.calls = OrderedDict(), 0

    def pack(self, controls, side, age):
        j = 3*int(side)
        return np.r_[(np.asarray(controls)[j:j+3]-CONTROL_BOUNDS[j:j+3, 0]) /
                     CONTROL_SPAN[j:j+3], age/AGE_MAX]

    def physical(self, row, side):
        row = np.clip(np.asarray(row, float), 0, 1)
        j = 3*int(side)
        controls = CONTROL_MID.copy()
        controls[j:j+3] = CONTROL_BOUNDS[j:j+3, 0]+row[:3]*CONTROL_SPAN[j:j+3]
        return controls, float(row[3]*AGE_MAX)

    def one_path(self, row, side):
        key = (int(side), *np.round(np.asarray(row), 12))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        controls, age = self.physical(row, side)
        path = trace_particle(self.field, controls, int(side), tmax=max(age, 1e-8),
            rtol=self.rtol, atol=self.atol, max_step=self.max_step)
        if not path.ok: raise RuntimeError('Particle trajectory integration failed.')
        self.cache[key] = path
        self.calls += 1
        if len(self.cache) > 1024: self.cache.popitem(last=False)
        return path

    def endpoints(self, z, assignment):
        rows = np.asarray(z).reshape(len(assignment), 4)
        paths = [self.one_path(row, side) for row, side in zip(rows, assignment)]
        ages = rows[:, 3]*AGE_MAX
        actual = np.array([p.state([min(a, p.end)])[0, :2] for p, a in zip(paths, ages)])
        return paths, ages, actual

    def residual(self, z, assignment, targets):
        paths, ages, actual = self.endpoints(z, assignment)
        residual = list(((actual-targets)/H).ravel())
        # A terminated path cannot be silently held at its terminal position.
        residual.extend(3*max(0., a-p.end)/AGE_MAX for p, a in zip(paths, ages))
        required = DIAMETER+COLLISION_MARGIN+CLEARANCE_RESERVE
        for i in range(len(paths)):
            for j in range(i):
                backwards = np.linspace(0, min(ages[i], ages[j]), COLLISION_SAMPLES)
                # Clipping here keeps invalid iterates finite; the lifespan
                # residual penalizes them and strict final checks reject them.
                pi = paths[i].state(ages[i]-backwards, strict=False)[:, :2]
                pj = paths[j].state(ages[j]-backwards, strict=False)[:, :2]
                distance = np.linalg.norm(pi-pj, axis=1)
                residual.extend(4*np.maximum(0, required-distance)/H)
        return np.asarray(residual)

    def refine(self, z, assignment, targets, max_nfev):
        z = np.clip(np.asarray(z, float).ravel(), 0, 1)
        history, replay = [], []
        def record(point):
            paths, ages, _ = self.endpoints(point, assignment)
            controls = [self.physical(row, side)[0]
                        for row, side in zip(np.asarray(point).reshape(-1, 4), assignment)]
            replay.append(independent_optimization_snapshot(paths, controls, assignment, ages,
                targets, self.label, len(replay)))
        def objective(point):
            residual = self.residual(point, assignment, targets)
            _, _, actual = self.endpoints(point, assignment)
            history.append(dict(evaluation=len(history), cost=float(np.dot(residual, residual)/2),
                max_error=float(np.linalg.norm(actual-targets, axis=1).max()),
                actual=actual.tolist()))
            return residual
        def jacobian(point):
            base = self.residual(point, assignment, targets)
            record(point)
            jac = np.empty((len(base), len(point)))
            for k in range(len(point)):
                # Absolute steps remain useful near zero; each perturbation
                # only re-integrates its own particle because of row caching.
                delta = 2e-4 if point[k]+2e-4 <= 1 else -2e-4
                trial = point.copy(); trial[k] += delta
                jac[:, k] = (self.residual(trial, assignment, targets)-base)/delta
            return jac
        result = least_squares(objective, z, jac=jacobian,
            bounds=(np.zeros(len(z)), np.ones(len(z))), max_nfev=max_nfev,
            ftol=5e-9, xtol=5e-8, gtol=5e-8)
        record(result.x)
        return result.x, dict(message=result.message, nfev=int(result.nfev),
            cost=float(result.cost), optimizer_terminated=bool(result.success), history=history,
            continuous_decisions=int(len(z)), per_particle_decisions=4, replay=replay)

def independent_seed_bank(planner):
    """Reusable seed trajectories; a shape can draw each particle from ANY row.

    Because the physical developed flow is independent of x, a seed trajectory
    can first be translated in x. It is then integrated again at its actual
    position with the lookup, and later with all eight direct PINN inputs.
    Translation is an initializer only; it is never the accepted trajectory.
    """
    bank = []
    for side, ux, speed in itertools.product((0, 1), SEED_U_SPEEDS, SEED_V_SPEEDS):
        j = 3*side
        c = CONTROL_MID.copy()
        c[j:j+3] = [min(.01, L/100), ux, speed if side == 0 else -speed]
        if np.any(c < CONTROL_BOUNDS[:, 0]) or np.any(c > CONTROL_BOUNDS[:, 1]):
            continue
        path = trace_particle(planner.field, c, side, tmax=AGE_MAX,
            rtol=planner.rtol, atol=planner.atol, max_step=planner.max_step)
        if path.ok and path.end > .02:
            ages = np.linspace(0, path.end, 401)
            bank.append(dict(controls=c, side=side, path=path,
                             ages=ages, points=path.state(ages)[:, :2]))
    if not bank: raise RuntimeError('No usable independent launch seeds.')
    print(f'Independent seed bank: {len(bank)} reusable paths; controls can differ for every particle.', flush=True)
    return bank

def options_for_particle(planner, bank, target):
    options = []
    for seed in bank:
        path, side, c = seed['path'], seed['side'], seed['controls'].copy()
        ydiff = seed['points'][:, 1]-target[1]
        crossings = np.flatnonzero(ydiff[:-1]*ydiff[1:] <= 0)
        if not len(crossings): continue
        k = int(crossings[0])
        age = brentq(lambda a: float(path.state([a])[0, 1]-target[1]),
                     seed['ages'][k], seed['ages'][k+1], xtol=1e-11)
        if age < .02: continue
        state = path.state([age])[0]
        j = 3*side
        c[j] += target[0]-state[0]
        if not 0 <= c[j] <= L: continue
        row = planner.pack(c, side, age)
        translated = planner.one_path(row, side)
        if translated.end < age-1e-8: continue
        actual = translated.state([age])[0, :2]
        options.append(dict(row=row, side=side, path=translated, age=age,
                            error=float(np.linalg.norm(actual-target))))
    if not options:
        raise RuntimeError(f'No seed reaches target {target.tolist()} within current bounds. '
                           'Expand the seed grid or flight-age budget; this is not an impossibility proof.')
    # Retain diverse launch locations, speeds and ages on each available wall.
    chosen = []
    for side in (0, 1):
        group = sorted([o for o in options if o['side'] == side], key=lambda o: o['error'])
        if not group: continue
        selected = [group.pop(0)]
        while group and len(selected) < CANDIDATES_PER_SIDE:
            k = max(range(len(group)), key=lambda k:
                min(np.linalg.norm(group[k]['row']-s['row']) for s in selected) -
                20*group[k]['error']/H)
            selected.append(group.pop(k))
        chosen.extend(selected)
    return chosen

def choose_independent_seeds(options, start):
    """Discrete search for a compatible set of individual launch candidates."""
    generator = np.random.default_rng(SEED+301+start)
    ids = [int(generator.integers(len(o))) for o in options]
    if start == 0:
        ids = [min(range(len(o)), key=lambda k: o[k]['error']) for o in options]
    cache = {}
    required = DIAMETER+COLLISION_MARGIN+CLEARANCE_RESERVE
    def pair_cost(i, ki, j, kj):
        if j > i: i, ki, j, kj = j, kj, i, ki
        key = (i, ki, j, kj)
        if key not in cache:
            a, b = options[i][ki], options[j][kj]
            backwards = np.linspace(0, min(a['age'], b['age']), COLLISION_SAMPLES)
            p = a['path'].state(a['age']-backwards)[:, :2]
            q = b['path'].state(b['age']-backwards)[:, :2]
            deficit = max(0., required-float(np.linalg.norm(p-q, axis=1).min()))/H
            cache[key] = 20*deficit**2
        return cache[key]
    for sweep in range(8):
        changed = False
        for i in generator.permutation(len(options)):
            costs = [(o['error']/H)**2 + sum(pair_cost(i, k, j, ids[j])
                      for j in range(len(options)) if i != j)
                     for k, o in enumerate(options[i])]
            k = int(np.argmin(costs))
            if costs[k]+1e-14 < costs[ids[i]]:
                ids[i], changed = k, True
        if not changed: break
    chosen = [options[i][k] for i, k in enumerate(ids)]
    return np.array([o['row'] for o in chosen]).ravel(), np.array([o['side'] for o in chosen], int)

def minimum_separation(paths, ages, samples=PAIR_VERIFY_SAMPLES, refine=True):
    """Distance only while both particles exist, using the SAME global clock."""
    minimum = float('inf')
    for i in range(len(paths)):
        for j in range(i):
            end = min(ages[i], ages[j])
            def distance(backwards):
                backwards = np.atleast_1d(backwards)
                a = paths[i].state(ages[i]-backwards)[:, :2]
                b = paths[j].state(ages[j]-backwards)[:, :2]
                return np.linalg.norm(a-b, axis=1)
            times = np.linspace(0, end, samples)
            d = distance(times); minimum = min(minimum, float(d.min()))
            if refine and end > 0:
                local = np.flatnonzero((d[1:-1] <= d[:-2]) & (d[1:-1] <= d[2:]))+1
                for k in local:
                    result = minimize_scalar(lambda t: float(distance(t)[0]),
                        bounds=(times[k-1], times[k+1]), method='bounded', options={'xatol': 1e-10})
                    minimum = min(minimum, float(result.fun))
    return minimum

def independent_candidate(planner, z, assignment, targets, diagnostics):
    paths, ages, actual = planner.endpoints(z, assignment)
    valid = all(p.ok and a <= p.end+1e-8 for p, a in zip(paths, ages))
    errors = np.linalg.norm(actual-targets, axis=1)
    sep = minimum_separation(paths, ages) if valid else 0.
    score = (max(0, float(errors.max())-POSITION_TOLERANCE)/H +
             2*max(0, DIAMETER+COLLISION_MARGIN+CLEARANCE_RESERVE-sep)/H + (0 if valid else 10))
    return dict(z=np.asarray(z).copy(), assignment=np.asarray(assignment).copy(), paths=paths,
        ages=ages, actual=actual, errors=errors, separation=sep, score=score, diagnostics=diagnostics)

def plan_formation(kind, coarse, direct, bank):
    targets, closed = target_points(kind)
    candidates, logs = [], []
    options = [options_for_particle(coarse, bank, point) for point in targets]
    print(kind, 'candidate counts per particle:', [len(o) for o in options], flush=True)
    for start in range(OPT_STARTS):
        try:
            initial, assignment = choose_independent_seeds(options, start)
            candidates.append(independent_candidate(coarse, initial, assignment, targets,
                                                      dict(stage='independent_seed_selection')))
            z, diag = coarse.refine(initial, assignment, targets, COARSE_MAX_NFEV)
            candidate = independent_candidate(coarse, z, assignment, targets, diag)
            candidates.append(candidate)
            print(f'{kind} start {start+1}: max error={candidate["errors"].max():.4g}; '
                  f'min separation={candidate["separation"]:.5g}; '
                  f'{4*len(targets)} independent continuous decisions', flush=True)
        except (RuntimeError, FloatingPointError, ValueError) as exc:
            logs.append(dict(stage='lookup', start=start+1, error=str(exc)))
    if not candidates:
        return dict(kind=kind, success=False, reason='All independent searches failed.', failures=logs)
    candidates.sort(key=lambda c: (c['score'], float(c['errors'].max())))
    verified = []
    for candidate in candidates[:min(2, len(candidates))]:
        coarse_replay = candidate['diagnostics'].get('replay', [])
        if not coarse_replay:
            seed_controls = [coarse.physical(row, side)[0] for row, side in
                zip(candidate['z'].reshape(-1, 4), candidate['assignment'])]
            coarse_replay = [independent_optimization_snapshot(candidate['paths'], seed_controls,
                candidate['assignment'], candidate['ages'], targets, 'independent seed selection', 0)]
        try:
            verified.append(independent_candidate(direct, candidate['z'], candidate['assignment'], targets,
                                                  dict(stage='direct_before_refinement', coarse_replay=coarse_replay)))
            z, diag = direct.refine(candidate['z'], candidate['assignment'], targets, DIRECT_MAX_NFEV)
            diag['coarse_history'] = candidate['diagnostics'].get('history', [])
            diag['coarse_replay'] = coarse_replay
            verified.append(independent_candidate(direct, z, candidate['assignment'], targets, diag))
        except (RuntimeError, FloatingPointError, ValueError) as exc:
            logs.append(dict(stage='direct', error=str(exc)))
    if not verified:
        return dict(kind=kind, success=False, reason='Direct PINN search failed.', failures=logs)
    verified.sort(key=lambda c: (c['score'], float(c['errors'].max())))
    best = verified[0]
    assignment, ages = best['assignment'], best['ages']
    controls = [direct.physical(row, side)[0]
                for row, side in zip(best['z'].reshape(-1, 4), assignment)]
    strict_paths, reference_paths = [], []
    for c, side, age in zip(controls, assignment, ages):
        strict_paths.append(trace_particle(direct.field, c, int(side), tmax=age,
            rtol=2e-9, atol=2e-11, max_step=TRAJECTORY_MAX_STEP/2))
        reference_paths.append(trace_particle(exact_flow, c, int(side), tmax=age,
            rtol=2e-10, atol=2e-12, max_step=TRAJECTORY_MAX_STEP/2))
    strict_valid = all(p.ok and a <= p.end+1e-8 for p, a in zip(strict_paths, ages))
    reference_valid = all(p.ok and a <= p.end+1e-8 for p, a in zip(reference_paths, ages))
    actual = np.array([p.state([a], strict=False)[0, :2] for p, a in zip(strict_paths, ages)])
    reference = np.array([p.state([a], strict=False)[0, :2] for p, a in zip(reference_paths, ages)])
    errors = np.linalg.norm(actual-targets, axis=1)
    ref_errors = np.linalg.norm(reference-targets, axis=1)
    delta = np.linalg.norm(reference-actual, axis=1)
    sep = minimum_separation(strict_paths, ages) if strict_valid else 0.
    ref_sep = minimum_separation(reference_paths, ages) if reference_valid else 0.
    T = float(ages.max()+LEAD_TIME)
    release = T-ages
    position_ok = bool(strict_valid and errors.max() <= POSITION_TOLERANCE)
    clearance_ok = bool(strict_valid and sep >= DIAMETER+COLLISION_MARGIN)
    reference_ok = bool(reference_valid and ref_errors.max() <= REFERENCE_TOLERANCE and
                        delta.max() <= REFERENCE_TOLERANCE and ref_sep >= DIAMETER+COLLISION_MARGIN)
    quality_ok = bool(VALIDATION['passed'])
    success = bool(position_ok and clearance_ok and reference_ok and quality_ok)
    schedule, trajectories = [], []
    for i, (c, side, path) in enumerate(zip(controls, assignment, strict_paths)):
        j = 3*side
        schedule.append(dict(particle=i+1, launcher=['bottom', 'top'][side],
            x_launch=float(c[j]), y_launch=float(path.state([0])[0, 1]),
            u_launch=float(c[j+1]), v_launch=float(c[j+2]),
            release_time=float(release[i]), age_at_target=float(ages[i]),
            target_x=float(targets[i, 0]), target_y=float(targets[i, 1]),
            actual_x=float(actual[i, 0]), actual_y=float(actual[i, 1]),
            position_error=float(errors[i]), reference_error=float(ref_errors[i])))
        sample_ages = np.linspace(0, min(ages[i], path.end), 501)
        trajectories.append(dict(particle=i+1, side=int(side), release=float(release[i]),
            flow_controls=c.tolist(), ages=sample_ages.tolist(), states=path.state(sample_ages).tolist()))
    reasons = []
    if not position_ok: reasons.append('target tolerance missed or premature trajectory termination')
    if not clearance_ok: reasons.append('particle clearance failed')
    if not reference_ok: reasons.append('analytical-flow cross-check failed')
    if not quality_ok: reasons.append('flow model validation failed')
    diagnostics = best['diagnostics']
    replay = diagnostics.get('coarse_replay', [])+diagnostics.get('replay', [])
    replay.append(independent_optimization_snapshot(strict_paths, controls, assignment,
        ages, targets, 'strict verification', len(replay)))
    return dict(kind=kind, closed=closed, success=success,
        reason='all numerical checks passed' if success else '; '.join(reasons),
        flow_validated=quality_ok, position_success=position_ok, clearance_success=clearance_ok,
        reference_success=reference_ok, target_time=T, control_mode='per-particle-independent',
        continuous_decisions=4*len(targets), targets=targets.tolist(), actual=actual.tolist(),
        analytical_reference=reference.tolist(), max_error=float(errors.max()),
        rms_error=float(np.sqrt(np.mean(errors**2))), max_reference_error=float(ref_errors.max()),
        max_PINN_reference_delta=float(delta.max()), min_pair_distance=float(sep),
        reference_min_pair_distance=float(ref_sep), required_pair_distance=DIAMETER+COLLISION_MARGIN,
        schedules=schedule, paths=trajectories, failures=logs,
        optimization={k: v for k, v in diagnostics.items() if k not in ['replay', 'coarse_replay']},
        optimization_replay=replay,
        interpretation='Finite independent per-particle multistart search; simultaneous snapshot only',
        collision_check='Shared-time sampling and local scalar refinement; no contact forces or continuous certificate')

def write_formation_dashboard(results, filename, run_directory):
    directory = Path(run_directory)
    cfg = json.loads((directory/'config.json').read_text())
    opt = json.loads((directory/'optimization_config.json').read_text())
    validation_path = directory/'validation.json'
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    usable = [r for r in results if r.get('paths') and r.get('targets')]
    if not usable: raise ValueError('No saved trajectories. Run the formation experiments first.')
    if any(r.get('control_mode') != 'per-particle-independent' for r in usable):
        raise ValueError('This app needs independent per-particle results, not an old shared-launcher run.')
    payload = dict(results=usable, failures=[r for r in results if not r.get('paths')],
        L=cfg['L'], H=cfg['H'], U=cfg['U_C'], diameter=cfg['diameter'],
        position_tolerance=opt['position_tolerance'], validation=bool(validation.get('passed')),
        bounds=cfg['bounds'], names=cfg['controls'])
    # Protect the script boundary if future custom experiment names contain markup.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    template = r'''<!DOCTYPE html>
<html lang="fa" dir="rtl"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>آزمایشگاه تشکیل شکل با ذرات</title>
<style>
:root{--bg:#080f1c;--panel:#111e31;--line:#26354d;--muted:#91a4bd;--text:#e9f1fc;--teal:#51e0c5;--blue:#4ebded;--rose:#ff7297;--amber:#ffca78}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 12% 0,#153148 0,transparent 47%),var(--bg);color:var(--text);font:14px Tahoma,Arial,sans-serif}button,input,select{font:inherit}button,select{cursor:pointer}main{max-width:1450px;margin:auto;padding:28px}header{display:flex;align-items:flex-start;justify-content:space-between;gap:20px}.eyebrow{font:11px Arial;letter-spacing:3px;color:var(--teal);margin:0 0 12px;direction:ltr;text-align:right}h1{font-size:29px;line-height:1.5;margin:0 0 8px}p{color:var(--muted);line-height:1.9;margin:6px 0}header .badge{margin-top:8px;white-space:nowrap}.badge{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:30px;padding:8px 12px;font-size:11px;background:#ffffff04}.dot{height:7px;width:7px;border-radius:50%;background:var(--teal);box-shadow:0 0 15px currentColor}.experiments{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:23px 0 17px}.experiment{background:#101d2f;border:1px solid var(--line);border-radius:13px;text-align:right;padding:13px 15px;color:var(--text);display:flex;gap:12px;align-items:center;transition:.2s}.experiment:hover{background:#192d43}.experiment.active{border-color:var(--teal);background:#123239;box-shadow:0 0 0 1px #51e0c51a}.experiment svg{width:39px;height:42px;flex-shrink:0}.experiment b{font-size:13px;display:block;margin-bottom:6px}.experiment small{font-size:10px;color:var(--muted)}.tag{display:block;margin-top:5px;color:var(--teal);font-size:10px}.tag.warn{color:var(--amber)}.toolbar,.panel,.metrics>div,.launcher{border:1px solid var(--line);background:linear-gradient(130deg,#152338,#101b2d);border-radius:14px}.toolbar{padding:15px;margin-bottom:15px}.row{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.row.spread{justify-content:space-between}.modes{background:#080f1c;border:1px solid var(--line);border-radius:10px;padding:4px;display:flex;gap:3px}.mode{border:0;border-radius:7px;background:transparent;color:var(--muted);padding:9px 13px}.mode.active{background:#254357;color:var(--text)}button.action,select{border:1px solid #334965;border-radius:8px;background:#1b2c43;color:var(--text);padding:8px 13px}button.primary{background:var(--teal);color:#09202a;border-color:var(--teal);font-weight:bold}button:disabled{opacity:.4;cursor:not-allowed}.scrubber{display:flex;align-items:center;gap:15px;margin-top:16px;direction:ltr}input[type=range]{accent-color:var(--teal);flex:1;min-width:90px}.clock{font:12px ui-monospace,Consolas,monospace;min-width:160px;text-align:center;direction:ltr}.stage{margin-top:10px;min-height:21px;color:var(--muted);font-size:11px}.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:15px}.metrics>div{padding:13px 16px}.metrics small{font-size:10px;color:var(--muted);display:block;margin-bottom:9px}.metrics b{font:19px ui-monospace,Consolas,monospace;display:block;direction:ltr;text-align:right}.metrics b.good{color:var(--teal)}.metrics b.warn{color:var(--amber)}.visuals{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(0,1fr);gap:15px}.panel{overflow:hidden;min-width:0}.panelhead{display:flex;justify-content:space-between;align-items:center;padding:15px 17px;border-bottom:1px solid var(--line);gap:10px}.panelhead h2{font-size:12px;margin:0}.panelhead small{font-size:10px;color:var(--muted)}canvas{display:block;width:100%}.scene{height:360px}.zoom{height:360px}.panelnote{padding:10px 17px;min-height:42px;color:var(--muted);font-size:10px;border-top:1px solid var(--line);line-height:1.8}.legend{display:flex;gap:14px;flex-wrap:wrap;padding:12px 17px;color:var(--muted);font-size:10px}.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-left:5px;background:var(--blue)}.options{margin:14px 1px;color:var(--muted);display:flex;gap:20px;flex-wrap:wrap;font-size:11px}.options label{cursor:pointer;display:flex;align-items:center;gap:6px}input[type=checkbox]{accent-color:var(--teal)}.lower{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:15px;margin-top:15px}.chart{height:170px}.launchers{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:14px}.launcher{padding:12px;background:#091424}.launcher h3{margin:0 0 12px;font-size:11px}.launcher.bottom h3{color:var(--blue)}.launcher.top h3{color:var(--rose)}.valuegrid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;direction:ltr}.valuegrid span{color:var(--muted);font:10px Arial}.valuegrid b{display:block;margin-top:7px;font:14px ui-monospace,Consolas,monospace;color:var(--text)}.releases{padding:4px 17px 16px}.release{display:grid;grid-template-columns:19px 1fr 45px;gap:8px;align-items:center;margin:9px 0;font-size:10px}.releasebar{height:5px;background:#263950;border-radius:4px;position:relative;direction:ltr}.releasebar i{height:5px;width:2px;position:absolute;top:0;background:var(--blue);border-radius:2px}.release.top .releasebar i{background:var(--rose)}.release span{color:var(--muted)}.release.active span{color:var(--text)}.statusbox{padding:13px 17px;border-top:1px solid var(--line);line-height:1.9;color:var(--muted);font-size:11px}.statusbox strong{color:var(--teal)}.statusbox.warn strong{color:var(--amber)}.foot{display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap;align-items:center;margin-top:18px}.foot p{font-size:10px;max-width:760px}.empty{padding:20px;color:var(--amber)}
@media(max-width:1000px){main{padding:18px}.experiments{grid-template-columns:repeat(3,1fr)}.visuals,.lower{grid-template-columns:1fr 1fr}.scene,.zoom{height:310px}.metrics{grid-template-columns:repeat(3,1fr)}h1{font-size:24px}}
@media(max-width:680px){main{padding:12px}header{display:block}.experiments{grid-template-columns:repeat(2,1fr)}.visuals,.lower{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.metrics>div:last-child{grid-column:span 2}.clock{min-width:110px;font-size:10px}.scene{height:250px}.zoom{height:320px}.row{gap:6px}.row.spread{gap:12px}.mode{padding:9px}.experiment{padding:11px}.launchers{padding:12px}header .badge{margin:8px 0 0}}
.launch-summary{padding:12px 17px;color:var(--muted);font-size:11px;line-height:1.8}.launch-table-wrap{direction:ltr;overflow:auto;max-height:300px;margin:0 14px 15px;border:1px solid var(--line);border-radius:8px}.launch-table{border-collapse:collapse;width:100%;min-width:480px;font:11px ui-monospace,Consolas,monospace;direction:ltr}.launch-table th,.launch-table td{padding:10px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}.launch-table th{position:sticky;top:0;background:#172b41;color:var(--muted);z-index:1;font:10px Tahoma}.launch-table tr td:first-child{color:var(--blue)}.launch-table tr.top td:first-child{color:var(--rose)}.launch-table tr:not(.active) td{opacity:.5}.launch-table tr.active{background:#51e0c505}
</style></head><body><main>
<header><div><div class="eyebrow">PARTICLE FORMATION LAB / INDEPENDENT WALL PORTS</div><h1>از رهاسازی تا شکل هدف</h1><p>مکان، سرعت و زمان رهاسازی مستقل هر ذره؛ یک شکل مشترک نزدیک خروجی.</p></div><span class="badge" id="validation"></span></header>
<nav class="experiments" id="experiments" aria-label="انتخاب آزمایش"></nav>
<section class="toolbar"><div class="row spread"><div class="modes"><button class="mode active" id="mode-release">رهاسازی و تشکیل شکل</button><button class="mode" id="mode-opt">بازپخش بهینه‌سازی</button></div><div class="row"><button class="action primary" id="play">▶ پخش</button><button class="action" id="restart">از ابتدا</button><button class="action" id="finish">نمای نهایی</button><select id="speed" aria-label="سرعت پخش"><option value="0.25">×۰٫۲۵</option><option value="0.5">×۰٫۵</option><option value="1" selected>×۱</option><option value="2">×۲</option><option value="4">×۴</option></select></div></div>
<div class="scrubber"><output class="clock" id="clock"></output><input id="scrub" type="range" min="0" max="1" step="0.0001" value="0" aria-label="زمان یا گام بهینه‌سازی"></div><div class="stage" id="stage"></div></section>
<section class="metrics"><div><small>ذرات رهاشده</small><b id="released"></b></div><div><small>ذرات در تلرانس هدف، اکنون</small><b id="arrived"></b></div><div><small id="error-label">بیشترین خطای نهایی</small><b id="error"></b></div><div><small>کمترین فاصله در کل مسیر نهایی</small><b id="clearance"></b></div><div><small>نتیجهٔ آزمون نهایی</small><b id="verdict"></b></div></section>
<section class="visuals"><div class="panel"><div class="panelhead"><h2>جریان و رهاسازی از دو دیواره</h2><small id="geometry"></small></div><canvas class="scene" id="scene" aria-label="پخش ذرات داخل کانال"></canvas><div class="legend"><span><i></i>دیوارهٔ پایین</span><span><i style="background:var(--rose)"></i>دیوارهٔ بالا</span><span><i style="background:#e3ecfa"></i>شکل هدف</span><span><i style="background:var(--amber)"></i>مرجع تحلیلی</span></div><div class="panelnote">هر علامت شماره‌دار روی دیواره، محل رهاسازی همان ذره است. رنگ‌ها فقط دیوارهٔ بالا یا پایین را مشخص می‌کنند.</div></div>
<div class="panel"><div class="panelhead"><h2>بزرگ‌نمایی ناحیهٔ تشکیل شکل</h2><small id="zoom-label"></small></div><canvas class="zoom" id="zoom" aria-label="بزرگ‌نمایی شکل هدف و خطا"></canvas><div class="legend"><span>خط‌چین: هدف</span><span>دایرهٔ رنگی: موقعیت واقعی</span><span>خط باریک: خطای موقعیت</span></div><div class="panelnote" id="dynamic-note"></div></div></section>
<div class="options"><label><input type="checkbox" id="paths" checked>مسیرهای ذرات</label><label><input type="checkbox" id="labels" checked>شمارهٔ ذرات</label><label><input type="checkbox" id="fluid" checked>نمای جریان پوازوی</label><label><input type="checkbox" id="reference">مرجع تحلیلی در زمان نهایی</label></div>
<section class="lower"><div class="panel"><div class="panelhead"><h2>کنترل‌های مستقل هر ذره و برنامهٔ رهاسازی</h2><small id="time-T"></small></div><div class="launch-summary" id="launch-summary"></div><div class="launch-table-wrap"><table class="launch-table"><thead><tr><th>#</th><th>دیواره</th><th>x₀</th><th>u₀</th><th>v₀</th><th>رهاسازی</th><th>مدت پرواز</th></tr></thead><tbody id="launchers"></tbody></table></div><div class="releases" id="releases"></div></div>
<div class="panel"><div class="panelhead"><h2>خطای موقعیت در گام‌های ثبت‌شده</h2><small id="trace-count"></small></div><canvas id="convergence" class="chart" aria-label="روند واقعی بهینه‌سازی"></canvas><div class="statusbox" id="result-status"></div><div class="statusbox">شکل در یک لحظه نزدیک خروجی سنجیده می‌شود. نتیجهٔ ناموفق نیز با موقعیت و خطای واقعی نمایش داده می‌شود.</div></div></section>
<div class="foot"><div class="row"><button class="action" id="csv">دریافت برنامهٔ رهاسازی CSV</button><button class="action" id="json">دریافت نتیجهٔ کامل JSON</button></div><p>بازپخش بهینه‌سازی از حالت‌های محاسبه‌شدهٔ حل‌گر ساخته شده است. کنترل زمان، سرعت نمایش را تغییر می‌دهد؛ داده‌های فیزیکی ذخیره‌شده همان نتایج محاسبه‌شده‌اند.</p></div>
</main><script>
'use strict';
const DATA=__PAYLOAD__, $=id=>document.getElementById(id),COLORS=['#4ebded','#ff7297'];
const TITLES={triangle:'مثلث',square:'مربع',circle_10:'دایرهٔ ۱۰ ذره‌ای',s_8:'حرف S',chevron_7:'پیکان'};
let selected=0, mode='release', fraction=0, playing=false, lastTS=null, renderWallTime=0,lastPaint=0,launcherKey='',releaseKey='';
const current=()=>DATA.results[selected],trace=()=>current().optimization_replay||[];
const number=(v,n=4)=>Number.isFinite(v)?v.toFixed(n):'—';
const errorText=v=>Number.isFinite(v)?(Math.abs(v)>0&&Math.abs(v)<1e-4?v.toExponential(2):v.toFixed(5)):'—';
const stageName=s=>s==='lookup'?'جست‌وجو با جدول میدان':s==='direct PINN'?'اصلاح با PINN کامل':s==='strict verification'?'بررسی دقیق نهایی':s==='independent seed selection'?'انتخاب اولیهٔ مستقل هر ذره':s;

function icon(r){let p=r.targets,x=p.map(q=>q[0]),y=p.map(q=>q[1]),cx=(Math.max(...x)+Math.min(...x))/2,cy=(Math.max(...y)+Math.min(...y))/2,s=28/Math.max(Math.max(...x)-Math.min(...x),Math.max(...y)-Math.min(...y),.001);let xy=p.map(q=>[20+(q[0]-cx)*s,21-(q[1]-cy)*s]);let coords=xy.concat(r.closed?[xy[0]]:[]).map(a=>a.join(',')).join(' ');return `<svg viewBox="0 0 40 44" aria-hidden="true"><polyline points="${coords}" fill="none" stroke="${r.success?'#51e0c5':'#ffca78'}" stroke-width="1" opacity=".5"/>${xy.map(q=>`<circle cx="${q[0]}" cy="${q[1]}" r="1.8" fill="${r.success?'#51e0c5':'#ffca78'}"/>`).join('')}</svg>`}
function buildCards(){const box=$('experiments');box.replaceChildren();DATA.results.forEach((r,i)=>{let b=document.createElement('button');b.className='experiment'+(i===selected?' active':'');b.dataset.index=i;b.innerHTML=icon(r)+`<span><b>${TITLES[r.kind]||r.kind}</b><small>${r.targets.length} ذره · ${r.continuous_decisions} متغیر مستقل</small><span class="tag ${r.success?'':'warn'}">${r.success?'در تلرانس آزمون':'بهترین نتیجهٔ یافت‌شده'}</span></span>`;b.onclick=()=>selectExperiment(i);box.appendChild(b)})}
function selectExperiment(i){selected=i;fraction=0;playing=false;if(mode==='opt'&&!trace().length)mode='release';buildCards();updateMode();updateStatic();render()}
function setMode(value){if(value==='opt'&&!trace().length)return;mode=value;fraction=0;playing=false;updateMode();render()}
function updateMode(){$('mode-release').classList.toggle('active',mode==='release');$('mode-opt').classList.toggle('active',mode==='opt');$('mode-opt').disabled=!trace().length;$('error-label').textContent=mode==='opt'?'بیشترین خطا در گام فعلی':'بیشترین خطای نهایی'}
function view(){let r=current();if(mode==='release')return {T:r.target_time,t:fraction*r.target_time,launches:r.schedules.map(s=>({...s,side:Number(s.launcher==='top')})),actual:r.actual,paths:r.paths,step:null};let history=trace(),index=Math.min(history.length-1,Math.floor(fraction*history.length)),s=history[index];return {T:s.target_time,t:s.target_time,launches:s.launches,actual:s.actual,paths:s.paths,step:s,index}}

function sample(path,age){let times=path.ages,z=path.states;if(!times.length||age<-1e-8||age>times[times.length-1]+1e-7)return null;if(times[times.length-1]===0)return z[0].slice(0,2);let t=Math.max(0,Math.min(age,times[times.length-1])),lo=0,hi=times.length-1;while(hi-lo>1){let m=(lo+hi)>>1;if(times[m]<t)lo=m;else hi=m}let a=(t-times[lo])/(times[hi]-times[lo]||1);return [z[lo][0]*(1-a)+z[hi][0]*a,z[lo][1]*(1-a)+z[hi][1]*a]}
function screen(canvas,bounds){let rect=canvas.getBoundingClientRect(),w=Math.max(100,rect.width),h=Math.max(100,rect.height),dpr=Math.min(2,window.devicePixelRatio||1);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}let ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);let [xmin,xmax,ymin,ymax]=bounds,scale=Math.min((w-52)/(xmax-xmin),(h-64)/(ymax-ymin)),cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;return {ctx,w,h,scale,map:(x,y)=>[w/2+(x-cx)*scale,h/2-(y-cy)*scale]}}
function bounds(zoom){let r=current();if(!zoom)return [-.07,DATA.L+.07,-.09,DATA.H+.09];let xx=r.targets.map(p=>p[0]),yy=r.targets.map(p=>p[1]),cx=(Math.min(...xx)+Math.max(...xx))/2,cy=(Math.min(...yy)+Math.max(...yy))/2,span=Math.max(Math.max(...xx)-Math.min(...xx),Math.max(...yy)-Math.min(...yy),.10)+.085;return [cx-span/2,cx+span/2,cy-span/2,cy+span/2]}
function line(ctx,points,color,width=1,dash=[]){if(!points.length)return;ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();ctx.setLineDash([])}
function dot(ctx,p,r,fill,glow=false){ctx.fillStyle=fill;ctx.shadowColor=fill;ctx.shadowBlur=glow?12:0;ctx.beginPath();ctx.arc(...p,r,0,2*Math.PI);ctx.fill();ctx.shadowBlur=0}
function arrow(ctx,from,to,color){line(ctx,[from,to],color,2);let a=Math.atan2(to[1]-from[1],to[0]-from[0]),d=6;ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(...to);ctx.lineTo(to[0]-d*Math.cos(a-.5),to[1]-d*Math.sin(a-.5));ctx.lineTo(to[0]-d*Math.cos(a+.5),to[1]-d*Math.sin(a+.5));ctx.closePath();ctx.fill()}
function drawScene(id,zoom,V){let S=screen($(id),bounds(zoom)),{ctx,map,scale,w,h}=S,r=current(),p0=map(0,DATA.H),p1=map(DATA.L,0);ctx.save();ctx.beginPath();ctx.rect(p0[0],p0[1],p1[0]-p0[0],p1[1]-p0[1]);ctx.clip();let gradient=ctx.createLinearGradient(p0[0],0,p1[0],0);gradient.addColorStop(0,'#183b50');gradient.addColorStop(1,'#122b3d');ctx.fillStyle=gradient;ctx.fillRect(p0[0],p0[1],p1[0]-p0[0],p1[1]-p0[1]);
if($('fluid').checked){for(let j=1;j<9;j++){let y=DATA.H*j/9,speed=4*DATA.U*(y/DATA.H)*(1-y/DATA.H);line(ctx,[map(0,y),map(DATA.L,y)],'#83b9d811',1);for(let k=0;k<11;k++){let x=((k/11*DATA.L+renderWallTime*speed*.11)%(DATA.L+.06))-.03;let a=map(x,y),b=map(x+.018,y);line(ctx,[a,b],'#91c4df45',1);line(ctx,[b,[b[0]-3,b[1]-2]],'#91c4df45',1)}}}ctx.restore();
line(ctx,[map(0,0),map(DATA.L,0)],'#4e718b',3);line(ctx,[map(0,DATA.H),map(DATA.L,DATA.H)],'#4e718b',3);
if(!zoom){ctx.font='11px Tahoma';ctx.fillStyle='#a9bed2';ctx.textAlign='center';ctx.fillText('ورودی پوازوی',...map(.1,DATA.H+.045));ctx.fillText('خروجی',...map(DATA.L-.08,DATA.H+.045));ctx.fillStyle='#7390ac';ctx.font='10px Arial';ctx.fillText('x = 0',...map(0,-.043));ctx.fillText('x = '+DATA.L,...map(DATA.L,-.043));let b=bounds(true),tl=map(b[0],b[3]),br=map(b[1],b[2]);ctx.setLineDash([5,5]);ctx.strokeStyle='#a6badb50';ctx.strokeRect(tl[0],tl[1],br[0]-tl[0],br[1]-tl[1]);ctx.setLineDash([])}
let target=r.targets.map(p=>map(...p));line(ctx,target.concat(r.closed?[target[0]]:[]),'#e3ecfa88',1.3,[5,5]);target.forEach(p=>{line(ctx,[[p[0]-4,p[1]-4],[p[0]+4,p[1]+4]],'#e7f0ff',1);line(ctx,[[p[0]+4,p[1]-4],[p[0]-4,p[1]+4]],'#e7f0ff',1)});
if($('paths').checked){let seen=new Set();V.paths.forEach(p=>{let key=p.side+':'+p.release;if(seen.has(key))return;seen.add(key);let maxAge=p.max_age??p.ages[p.ages.length-1];line(ctx,p.states.filter((_,i)=>p.ages[i]<=maxAge+1e-7).map(z=>map(z[0],z[1])),p.side?'#ff72972a':'#4ebded2a',1.2)})}
let active=V.paths.map(p=>sample(p,V.t-p.release));let near=new Set();for(let i=0;i<active.length;i++)for(let j=0;j<i;j++)if(active[i]&&active[j]&&Math.hypot(active[i][0]-active[j][0],active[i][1]-active[j][1])<r.required_pair_distance){near.add(i);near.add(j);line(ctx,[map(...active[i]),map(...active[j])],'#ff626f',1.8)}
V.paths.forEach((p,i)=>{let q=active[i];if(!q)return;let point=map(...q),rad=Math.max(zoom?5:3.5,DATA.diameter*scale/2),color=near.has(i)?'#ff6578':COLORS[p.side];let age=V.t-p.release;
let tail=p.states.map((z,k)=>({z,t:p.ages[k]})).filter(a=>a.t<=age&&a.t>=age-.2).map(a=>map(a.z[0],a.z[1]));line(ctx,tail.concat([point]),color,zoom?2.2:2);
dot(ctx,point,rad,color,true);let shine=[point[0]-rad*.25,point[1]-rad*.25];dot(ctx,shine,Math.max(.7,rad*.16),'#ffffffa8');if(zoom){line(ctx,[target[i],point],color+'a8',.8)}if($('labels').checked){ctx.font=(zoom?'11px':'9px')+' Arial';ctx.textAlign='center';ctx.fillStyle='#eaf5ff';ctx.fillText(String(i+1),point[0],point[1]-rad-5)}});
if($('reference').checked&&(mode==='opt'&&V.step.stage==='strict verification'||mode==='release'&&fraction>.995)){r.analytical_reference.forEach(q=>{let p=map(...q);ctx.beginPath();ctx.strokeStyle='#ffca78';ctx.lineWidth=1.4;ctx.arc(...p,Math.max(2,DATA.diameter*scale*.25),0,2*Math.PI);ctx.stroke()})}
for(let i=0;i<V.launches.length;i++){let launch=V.launches[i],side=launch.side,x=launch.x_launch,y=side?DATA.H:0,port=map(x,y),color=COLORS[side],live=V.t>=launch.release_time;ctx.globalAlpha=live?1:.3;ctx.fillStyle=color;ctx.shadowColor=color;ctx.shadowBlur=live?8:0;ctx.fillRect(port[0]-4,port[1]-4,8,8);ctx.shadowBlur=0;let velocityScale=zoom?.025:.06;arrow(ctx,port,map(x+launch.u_launch*velocityScale,y+launch.v_launch*velocityScale),color);if($('labels').checked&&!zoom){ctx.fillStyle=color;ctx.textAlign='center';ctx.font='9px Arial';ctx.fillText(String(launch.particle),port[0],port[1]+(side?-1:1)*(13+10*(i%2)))}ctx.globalAlpha=1;if(V.t>=launch.release_time&&V.t-launch.release_time<.08){ctx.strokeStyle=color+'88';ctx.beginPath();ctx.arc(...port,12,0,2*Math.PI);ctx.stroke()}}

return active}
function drawChart(V){let c=$('convergence'),box=c.getBoundingClientRect(),w=box.width,h=box.height,dpr=Math.min(2,window.devicePixelRatio||1);let history=trace(),chartKey=[selected,mode,mode==='opt'?V.index:-1,w,h].join(':');if(c.dataset.chartKey===chartKey)return;c.dataset.chartKey=chartKey;let ctx=c.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.font='10px Tahoma';ctx.fillStyle='#91a4bd';ctx.textAlign='center';if(history.length<2){ctx.fillText('برای این نتیجه، تاریخچهٔ گام‌های بهینه‌سازی ثبت نشده است.',w/2,h/2);return}let values=history.map(s=>Math.log10(Math.max(s.max_error,1e-8))),tol=Math.log10(DATA.position_tolerance),low=Math.min(...values,tol)-.3,high=Math.max(...values,tol)+.3,fx=i=>40+i*(w-65)/(history.length-1),fy=v=>20+(high-v)*(h-48)/(high-low);for(let i=0;i<4;i++){let v=low+(high-low)*i/3,y=fy(v);line(ctx,[[40,y],[w-18,y]],'#27384d',.7);ctx.fillStyle='#91a4bd';ctx.textAlign='right';ctx.font='9px Arial';ctx.fillText((10**v).toExponential(0),34,y+3)}line(ctx,[[40,fy(tol)],[w-18,fy(tol)]],'#51e0c584',1,[4,4]);line(ctx,values.map((v,i)=>[fx(i),fy(v)]),'#51e0c5',1.7);let index=mode==='opt'?V.index:history.length-1;line(ctx,[[fx(index),14],[fx(index),h-23]],'#ffca7855',1);dot(ctx,[fx(index),fy(values[index])],4,'#ffca78');ctx.fillStyle='#91a4bd';ctx.font='9px Tahoma';ctx.textAlign='left';ctx.fillText('تلرانس هدف',45,Math.max(13,fy(tol)-5));ctx.textAlign='center';ctx.fillText('گام‌های ثبت‌شدهٔ واقعی',w/2,h-7)}
function updateStatic(){let r=current(),hist=trace();$('geometry').textContent=`L = ${DATA.L} / H = ${DATA.H}`;$('trace-count').textContent=hist.length?hist.length+' حالت ثبت‌شده':'بدون تاریخچه';$('verdict').textContent=r.success?'PASS':'BEST FOUND';$('verdict').className=r.success?'good':'warn';$('clearance').textContent=number(r.min_pair_distance);$('clearance').className=r.clearance_success?'good':'warn';$('result-status').className='statusbox'+(r.success?'':' warn');$('result-status').innerHTML=`<strong>${r.success?'هدف، فاصلهٔ ذرات و مرجع مستقل تأیید شدند.':'تمام معیارهای نهایی برآورده نشده‌اند.'}</strong><br>تلرانس موقعیت: <span dir="ltr">${number(DATA.position_tolerance)}</span> · فاصلهٔ لازم: <span dir="ltr">${number(r.required_pair_distance)}</span><br>بیشترین خطا در مرجع تحلیلی: <span dir="ltr">${errorText(r.max_reference_error)}</span>`;$('mode-opt').title=hist.length?'بازپخش حالت‌های ثبت‌شدهٔ حل‌گر':'برای داده‌های قدیمی، سلول‌های بهینه‌سازی نسخهٔ جدید را دوباره اجرا کنید.'}
function render(){let r=current(),V=view();$('scrub').value=fraction;$('play').textContent=playing?'❚❚ توقف':'▶ پخش';let active=drawScene('scene',false,V);drawScene('zoom',true,V);drawChart(V);
let count=active.filter(Boolean).length,inTol=active.filter((p,i)=>p&&Math.hypot(p[0]-r.targets[i][0],p[1]-r.targets[i][1])<=DATA.position_tolerance).length;$('released').textContent=count+' / '+r.targets.length;$('arrived').textContent=inTol+' / '+r.targets.length;let err=mode==='opt'?V.step.max_error:r.max_error;$('error').textContent=errorText(err);$('error').className=err<=DATA.position_tolerance?'good':'warn';$('time-T').textContent='T = '+number(V.T,3);$('zoom-label').textContent=mode==='opt'?'شکل در پایان برنامهٔ گام فعلی':'موقعیت ذرات در زمان فعلی';$('clock').textContent=mode==='opt'?`${V.index+1} / ${trace().length}`:`t = ${number(V.t,3)} / ${number(V.T,3)}`;$('stage').textContent=mode==='opt'?`گام ${V.index+1} از ${trace().length} · ${stageName(V.step.stage)} · نمایش حالت محاسبه‌شدهٔ واقعی`:`بازپخش رهاسازی بهینه · ${count} ذره از ${r.targets.length} ذره وارد کانال شده‌اند`;
$('dynamic-note').textContent=mode==='opt'?'در هر گام، موقعیت‌ها، سرعت‌های مستقل هر ذره و زمان رهاسازی واقعاً با حل‌گر محاسبه شده‌اند.':'ذرات در زمان‌های متفاوت رها می‌شوند و در زمان مشترک T با شکل هدف مقایسه می‌شوند.';
let nextLauncher=V.launches.map(s=>[s.side,s.x_launch,s.u_launch,s.v_launch,s.release_time,s.age_at_target].join(',')).join(':');if(launcherKey!==nextLauncher){launcherKey=nextLauncher;$('launch-summary').textContent=V.launches.length+' ذره · '+(4*V.launches.length)+' متغیر پیوستهٔ مستقل · پورت رهاسازی مستقل برای هر ذره';$('launchers').innerHTML=V.launches.map(s=>`<tr class="${s.side?'top':''}"><td>${s.particle}</td><td>${s.side?'بالا':'پایین'}</td>${[s.x_launch,s.u_launch,s.v_launch,s.release_time,s.age_at_target].map(v=>`<td>${number(v,4)}</td>`).join('')}</tr>`).join('');}Array.from($('launchers').children).forEach((row,i)=>row.classList.toggle('active',V.t>=V.launches[i].release_time));

let nextRelease=V.T+':'+V.paths.map(p=>p.side+','+p.release).join(':');if(releaseKey!==nextRelease){releaseKey=nextRelease;$('releases').innerHTML=V.paths.map((p,i)=>{let span=V.T-p.release,live=V.t>=p.release;return `<div class="release ${p.side?'top':''} ${live?'active':''}"><span dir="ltr">${i+1}</span><div class="releasebar"><i style="left:${p.release/V.T*100}%;width:${span/V.T*100}%;opacity:${live?1:.25}"></i></div><span dir="ltr">${number(p.release,3)}</span></div>`}).join('');}Array.from($('releases').children).forEach((row,i)=>{row.classList.toggle('active',V.t>=V.paths[i].release);row.querySelector('i').style.opacity=V.t>=V.paths[i].release?1:.25})}
function download(name,text,type){let blob=new Blob([type.startsWith('text/csv')?'\ufeff':'',text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('mode-release').onclick=()=>setMode('release');$('mode-opt').onclick=()=>setMode('opt');$('play').onclick=()=>{if(fraction>=1)fraction=0;playing=!playing;lastTS=null;render()};$('restart').onclick=()=>{fraction=0;playing=false;render()};$('finish').onclick=()=>{fraction=1;playing=false;render()};$('scrub').oninput=e=>{fraction=+e.target.value;playing=false;render()};for(let id of ['paths','labels','fluid','reference'])$(id).onchange=()=>render();
$('csv').onclick=()=>{let r=current(),rows=r.schedules,keys=Object.keys(rows[0]),quote=v=>'"'+String(v).replaceAll('"','""')+'"';download(r.kind+'_release_schedule.csv',[keys.map(quote).join(','),...rows.map(row=>keys.map(k=>quote(row[k])).join(','))].join('\r\n'),'text/csv;charset=utf-8')};$('json').onclick=()=>download(current().kind+'_complete_result.json',JSON.stringify(current(),null,2),'application/json');
function tick(ts){if(!$('fluid'))return;let elapsed=lastTS===null?0:Math.min((ts-lastTS)/1000,.05);lastTS=ts;renderWallTime=ts/1000;if(playing){fraction=Math.min(1,fraction+elapsed*(+$('speed').value)/(mode==='opt'?9:12));if(fraction>=1)playing=false}if((playing||$('fluid').checked)&&ts-lastPaint>=40){lastPaint=ts;render()}requestAnimationFrame(tick)}
$('validation').innerHTML='<i class="dot" style="background:'+(DATA.validation?'#51e0c5':'#ffca78')+'"></i>'+(DATA.validation?'میدان سیال اعتبارسنجی شده':'میدان سیال تأیید نشده');
window.addEventListener('resize',()=>render());buildCards();updateMode();updateStatic();render();requestAnimationFrame(tick);
// Read-only diagnostics for automated UI checks and scientific replay audits.
window.formationLab={state:()=>({shape:current().kind,mode,fraction,playing,trace:trace().length,view:view()}),data:DATA};
</script></body></html>'''
    filename.write_text(template.replace('__PAYLOAD__', encoded), encoding='utf-8')
    return filename

def write_schedule(path, rows):
    if not rows: return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def plot_formation(result, directory):
    if 'actual' not in result: return
    target, actual = np.array(result['targets']), np.array(result['actual'])
    reference = np.array(result['analytical_reference'])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), layout='constrained',
                             gridspec_kw={'height_ratios': [1, 2]})
    ax = axes[0]
    for trajectory in result['paths']:
        xy = np.array(trajectory['states'])[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=['#0786a5', '#da4e70'][trajectory['side']], alpha=.65)
    for row in result['schedules']:
        side = int(row['launcher'] == 'top')
        ax.scatter(row['x_launch'], H if side else 0, marker='s', s=40, color=['#0786a5', '#da4e70'][side])
        ax.annotate(str(row['particle']), (row['x_launch'], H if side else 0),
                    xytext=(3, 5 if side else -12), textcoords='offset points', fontsize=8)
    for a in axes: a.axhline(0, color='k'); a.axhline(H, color='k'); a.set(xlabel='x', ylabel='y', aspect='equal')
    ax.set(xlim=(0, L), ylim=(-.015, H+.015), title='Independent launch position and emission time per particle; colour denotes wall')
    ax = axes[1]
    order = np.r_[np.arange(len(target)), 0] if result['closed'] else np.arange(len(target))
    ax.plot(target[order, 0], target[order, 1], '--', color='#8691a2', linewidth=1)
    ax.scatter(target[:, 0], target[:, 1], marker='x', s=80, color='black', label='Target')
    ax.scatter(actual[:, 0], actual[:, 1], s=65, facecolors='none', edgecolors='#007eac', label='Direct PINN')
    ax.scatter(reference[:, 0], reference[:, 1], s=18, color='#f18b28', label='Analytical flow, same schedule')
    for i, (a, b) in enumerate(zip(target, actual)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#d14f66', linewidth=1)
        ax.annotate(str(i+1), b, xytext=(4, 4), textcoords='offset points', fontsize=9)
    ax.set(xlim=(max(0, target[:, 0].min()-.08), min(L, target[:, 0].max()+.08)),
           ylim=(0, H), title=f'{result["kind"]}: {"PASS" if result["success"] else "BEST FOUND / CHECKS FAILED"}'
           f' | maximum position error = {result["max_error"]:.4g}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.savefig(directory/'formation.png', dpi=240)
    plt.show()

def _run_six_fixed_station_task(settings):
    import copy, csv, hashlib, html, itertools, json, math, shutil, time
    from collections import OrderedDict
    from pathlib import Path
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares, linear_sum_assignment, minimize_scalar
    from IPython.display import HTML, FileLink, display

    required = ('model', 'RUN', 'L', 'H', 'DIAMETER', 'CONTROL_BOUNDS', 'CONTROL_MID',
                'trace_particle', 'direct_field', 'exact_flow', 'FlowLookup',
                'VALIDATION', 'write_formation_dashboard')
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError('Run the final notebook through its app-definition cell first. Missing: '+', '.join(missing))
    if not VALIDATION.get('passed', False):
        raise RuntimeError('The frozen flow model must pass validation before this task.')
    cfg = dict(settings)
    center = np.asarray(cfg['target_center'], float)
    positive = ('width', 'height', 'max_flight_age', 'min_flight_age', 'lead_time',
                'position_tolerance', 'reference_tolerance', 'collision_margin')
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('target_center must contain two finite coordinates.')
    if any(not np.isfinite(cfg[k]) or cfg[k] <= 0 for k in positive):
        raise ValueError('Letter sizes, ages and tolerances must be positive and finite.')
    if cfg['min_flight_age'] >= cfg['max_flight_age']:
        raise ValueError('min_flight_age must be less than max_flight_age.')
    counts = ('seed_u_count', 'seed_v_count', 'fits_per_station_target', 'options_per_pair',
              'pair_fit_nfev', 'matchings_to_test', 'matching_restarts', 'joint_starts',
              'joint_nfev', 'direct_nfev', 'collision_samples', 'verify_samples')
    if any(not isinstance(cfg[k], (int, np.integer)) or cfg[k] < 1 for k in counts):
        raise ValueError('Search budgets must be positive integers.')
    if cfg['seed_u_count'] < 2 or cfg['seed_v_count'] < 2 or cfg['collision_samples'] < 3 or cfg['verify_samples'] < 21:
        raise ValueError('Use at least two seed speeds, three collision samples and 21 verification samples.')
    if cfg['clearance_reserve'] < 0 or not np.isfinite(cfg['clearance_reserve']):
        raise ValueError('clearance_reserve must be finite and nonnegative.')
    if not cfg['shapes'] or any(k not in ('S', 'C', 'T') for k in cfg['shapes']):
        raise ValueError('shapes must be a nonempty selection of S, C and T.')

    # Particle order is ALWAYS B1,B2,B3,T1,T2,T3. Station x is NEVER optimized.
    stations = [dict(station_id=('B' if side == 0 else 'T')+str(k+1),
                     particle=3*side+k+1, side=side, x_launch=float(L*fraction))
                for side in (0, 1) for k, fraction in enumerate((.25, .50, .75))]
    sides = np.array([s['side'] for s in stations], int)
    required_gap = DIAMETER+cfg['collision_margin']
    search_gap = required_gap+cfg['clearance_reserve']
    bounds = np.asarray(CONTROL_BOUNDS, float)
    age_span = cfg['max_flight_age']-cfg['min_flight_age']
    generator = np.random.default_rng(cfg['seed'])
    task_run = Path(RUN)/('six_fixed_stations_'+str(time.time_ns()))
    task_run.mkdir(parents=True)
    saved_cfg = json.loads((Path(RUN)/'config.json').read_text())
    (task_run/'config.json').write_text(json.dumps(saved_cfg, indent=2))
    (task_run/'validation.json').write_text(json.dumps(VALIDATION, indent=2))
    task_cfg = dict(cfg, stations=stations, continuous_decisions=18,
                    position_mode='fixed L/4,L/2,3L/4 on each wall; one shot per station',
                    decisions='u0,v0,flight age for each station; target assignment is discrete',
                    target_time='T=max(flight ages)+lead_time; release_i=T-age_i')
    task_cfg['shapes'] = [k+'_6' for k in cfg['shapes']]
    (task_run/'optimization_config.json').write_text(json.dumps(task_cfg, indent=2))
    print('SIX FIXED STATIONS:', [(s['station_id'], s['x_launch']) for s in stations], flush=True)
    print('Target centre:', center.tolist(), '| output:', task_run, flush=True)

    def weights_hash():
        return hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes()
                                      for v in model.state_dict().values())).hexdigest()
    original_hash = weights_hash()
    model.eval()  # No optimizer or neural-weight updates are performed.

    def targets(kind):
        # Exactly SIX points for each letter. C is open to the right.
        if kind == 'S':
            points = np.array([[.5,.5],[-.5,.5],[-.5,.1],[.5,-.1],[.5,-.5],[-.5,-.5]])
            edges = [(i, i+1) for i in range(5)]
        elif kind == 'C':
            theta = np.linspace(np.pi/4, 7*np.pi/4, 6)
            points = .5*np.column_stack((np.cos(theta), np.sin(theta)))
            edges = [(i, i+1) for i in range(5)]
        else:
            points = np.array([[-.5,.5],[0,.5],[.5,.5],[0,1/6],[0,-1/6],[0,-.5]])
            # T is branched: never connect the right bar tip diagonally to its stem.
            edges = [(0,1),(1,2),(1,3),(3,4),(4,5)]
        goal = center+points*np.array([cfg['width'], cfg['height']])
        r = DIAMETER/2
        if np.any(goal <= [r,r]) or np.any(goal >= [L-r,H-r]):
            raise ValueError(f'{kind}: target does not fit inside the channel with particle-radius clearance.')
        d = np.linalg.norm(goal[:,None]-goal[None,:],axis=2)
        np.fill_diagonal(d,np.inf)
        if d.min() < required_gap:
            raise ValueError(f'{kind}: target points overlap. Increase letter size or reduce the fixed particle diameter in a new physical run.')
        return goal, edges

    goals = {kind:targets(kind) for kind in cfg['shapes']}

    class Planner:
        def __init__(self, field, label, rtol=2e-6, atol=2e-8, max_step=.06):
            self.field,self.label = field,label
            self.rtol,self.atol,self.max_step = rtol,atol,max_step
            self.cache,self.calls = OrderedDict(),0

        def physical(self, row, station):
            q = np.clip(np.asarray(row,float),0,1)
            side = stations[station]['side']; j=3*side
            c = np.asarray(CONTROL_MID,float).copy()
            c[j] = stations[station]['x_launch']  # Exact fixed hardware coordinate.
            c[j+1:j+3] = bounds[j+1:j+3,0]+q[:2]*np.ptp(bounds[j+1:j+3],axis=1)
            age = cfg['min_flight_age']+q[2]*age_span
            return c,float(age)

        def row(self, ux,vy,age,station):
            j=3*stations[station]['side']
            uv = (np.array([ux,vy])-bounds[j+1:j+3,0])/np.ptp(bounds[j+1:j+3],axis=1)
            return np.clip(np.r_[uv,(age-cfg['min_flight_age'])/age_span],0,1)

        def path(self, row, station):
            key=(int(station),*np.round(np.asarray(row),12))
            if key in self.cache:
                self.cache.move_to_end(key);return self.cache[key]
            c,age=self.physical(row,station)
            p=trace_particle(self.field,c,int(sides[station]),tmax=age,
                            rtol=self.rtol,atol=self.atol,max_step=self.max_step)
            if not p.ok:raise RuntimeError('Particle ODE failed at '+stations[station]['station_id'])
            self.cache[key]=p;self.calls+=1
            if len(self.cache)>2048:self.cache.popitem(last=False)
            return p

        def endpoints(self, z):
            rows=np.asarray(z,float).reshape(6,3)
            paths=[self.path(row,i) for i,row in enumerate(rows)]
            ages=np.array([self.physical(row,i)[1] for i,row in enumerate(rows)])
            actual=np.array([p.state([a],strict=False)[0,:2] for p,a in zip(paths,ages)])
            return paths,ages,actual

        def residual(self,z,goal):
            paths,ages,actual=self.endpoints(z)
            values=list(((actual-goal)/H).ravel())
            values.extend(3*max(0.,a-p.end)/cfg['max_flight_age'] for p,a in zip(paths,ages))
            for i in range(6):
                for j in range(i):
                    backwards=np.linspace(0,min(ages[i],ages[j]),cfg['collision_samples'])
                    a=paths[i].state(ages[i]-backwards,strict=False)[:,:2]
                    b=paths[j].state(ages[j]-backwards,strict=False)[:,:2]
                    values.extend(4*np.maximum(0,search_gap-np.linalg.norm(a-b,axis=1))/H)
            return np.asarray(values)

        def refine(self,z,goal,budget):
            replay=[]
            def jacobian(point):
                base=self.residual(point,goal)
                paths,ages,_=self.endpoints(point)
                replay.append(snapshot(self,point,paths,ages,goal,self.label,len(replay)))
                jac=np.empty((len(base),18))
                for k in range(18):
                    delta=2e-4 if point[k]+2e-4<=1 else -2e-4
                    trial=point.copy();trial[k]+=delta
                    jac[:,k]=(self.residual(trial,goal)-base)/delta
                return jac
            fit=least_squares(lambda z:self.residual(z,goal),np.clip(np.asarray(z).ravel(),0,1),
                jac=jacobian,bounds=(np.zeros(18),np.ones(18)),max_nfev=budget,
                ftol=5e-9,xtol=5e-8,gtol=5e-8)
            paths,ages,_=self.endpoints(fit.x)
            replay.append(snapshot(self,fit.x,paths,ages,goal,self.label,len(replay)))
            return fit.x,replay,dict(nfev=int(fit.nfev),message=fit.message,cost=float(fit.cost))

    def separation(paths,ages,count,refine=False):
        minimum=float('inf')
        for i in range(len(paths)):
            for j in range(i):
                stop=min(ages[i],ages[j])
                def distance(backwards):
                    t=np.atleast_1d(backwards)
                    a=paths[i].state(ages[i]-t,strict=False)[:,:2]
                    b=paths[j].state(ages[j]-t,strict=False)[:,:2]
                    return np.linalg.norm(a-b,axis=1)
                times=np.linspace(0,stop,count);d=distance(times)
                minimum=min(minimum,float(d.min()))
                if refine and stop>0:
                    minima=np.flatnonzero((d[1:-1]<d[:-2])&(d[1:-1]<=d[2:]))+1
                    for k in minima:
                        fit=minimize_scalar(lambda t:float(distance(t)[0]),
                            bounds=(times[k-1],times[k+1]),method='bounded',options={'xatol':1e-10})
                        minimum=min(minimum,float(fit.fun))
        return minimum

    def snapshot(planner,z,paths,ages,goal,stage,step):
        actual=np.array([p.state([a],strict=False)[0,:2] for p,a in zip(paths,ages)])
        T=float(max(ages)+cfg['lead_time']);launches=[];samples=[]
        for i,(row,p,age) in enumerate(zip(np.asarray(z).reshape(6,3),paths,ages)):
            c,_=planner.physical(row,i);j=3*sides[i]
            launches.append(dict(particle=i+1,station_id=stations[i]['station_id'],side=int(sides[i]),
                launcher=['bottom','top'][sides[i]],x_launch=float(c[j]),y_launch=float(p.state([0])[0,1]),
                u_launch=float(c[j+1]),v_launch=float(c[j+2]),release_time=float(T-age),age_at_target=float(age)))
            tt=np.linspace(0,min(age,p.end),81)
            samples.append(dict(particle=i+1,side=int(sides[i]),release=float(T-age),
                                flow_controls=c.tolist(),ages=tt.tolist(),states=p.state(tt).tolist()))
        errors=np.linalg.norm(actual-goal,axis=1)
        return dict(stage=stage,step=int(step),target_time=T,continuous_decisions=18,
                    launches=launches,paths=samples,actual=actual.tolist(),
                    max_error=float(errors.max()),rms_error=float(np.sqrt(np.mean(errors**2))))

    lookup=FlowLookup(model)
    coarse=Planner(lookup,'lookup')
    direct=Planner(direct_field,'direct PINN',rtol=2e-7,atol=2e-9,max_step=.035)

    # Forward seed bank: all positions are the SIX requested station positions.
    bank=[[] for _ in stations]
    for i,station in enumerate(stations):
        j=3*station['side']
        for ux,vy in itertools.product(np.linspace(*bounds[j+1],cfg['seed_u_count']),
                                       np.linspace(*bounds[j+2],cfg['seed_v_count'])):
            row=coarse.row(ux,vy,cfg['max_flight_age'],i)
            p=coarse.path(row,i)
            if p.end<cfg['min_flight_age']:continue
            tt=np.linspace(cfg['min_flight_age'],p.end,401)
            bank[i].append(dict(row=row,ages=tt,points=p.state(tt)[:,:2]))
        if not bank[i]:raise RuntimeError('No valid seed paths from '+station['station_id'])
        print('Seed bank',station['station_id'],len(bank[i]),'paths; x fixed at',station['x_launch'],flush=True)

    def fit_station_target(station,target):
        choices=[]
        for seed in bank[station]:
            distances=np.linalg.norm(seed['points']-target,axis=1);k=int(distances.argmin())
            row=seed['row'].copy();row[2]=(seed['ages'][k]-cfg['min_flight_age'])/age_span
            choices.append((float(distances[k]),row))
        choices.sort(key=lambda x:x[0]);starts=[]
        for error,row in choices:
            if all(np.linalg.norm(row-old)>.06 for old in starts):starts.append(row)
            if len(starts)>=cfg['fits_per_station_target']:break
        options=[]
        for initial in starts:
            def residual(row):
                p=coarse.path(row,station);_,age=coarse.physical(row,station)
                xy=p.state([age],strict=False)[0,:2]
                return np.r_[(xy-target)/H,3*max(0.,age-p.end)/cfg['max_flight_age']]
            fit=least_squares(residual,initial,bounds=(np.zeros(3),np.ones(3)),
                diff_step=.002,max_nfev=cfg['pair_fit_nfev'],ftol=1e-9,xtol=1e-8,gtol=1e-8)
            p=coarse.path(fit.x,station);_,age=coarse.physical(fit.x,station)
            if age>p.end+1e-8:continue
            error=float(np.linalg.norm(p.state([age])[0,:2]-target))
            options.append(dict(row=fit.x,path=p,age=age,error=error))
        if not options:
            row=choices[0][1];p=coarse.path(row,station);_,age=coarse.physical(row,station)
            if age<=p.end+1e-8:
                options.append(dict(row=row,path=p,age=age,error=choices[0][0]))
        options.sort(key=lambda p:p['error'])
        return options[:cfg['options_per_pair']]

    def build_candidates(options,goal):
        # A permutation assigns each of the six stations to ONE distinct letter point.
        lower=np.array([[min((q['error'] for q in options[i][j]),default=1e3)
                         for j in range(6)] for i in range(6)])
        perms=list(itertools.permutations(range(6)));generator.shuffle(perms)
        perms.sort(key=lambda p:sum(max(0,lower[i,j]-cfg['position_tolerance']*.25)**2
                                    for i,j in enumerate(p)))
        _,hungarian=linear_sum_assignment(lower)
        matching=[tuple(hungarian)]+perms[:min(720,cfg['matchings_to_test'])]
        matching=list(dict.fromkeys(matching));pool=[];pair_cache={}
        def pair_penalty(i,gi,ki,j,gj,kj):
            key=(i,gi,ki,j,gj,kj)
            if key not in pair_cache:
                a,b=options[i][gi][ki],options[j][gj][kj]
                gap=separation([a['path'],b['path']],[a['age'],b['age']],cfg['collision_samples'])
                pair_cache[key]=20*(max(0,search_gap-gap)/H)**2
            return pair_cache[key]
        for perm in matching:
            selected=[options[i][perm[i]] for i in range(6)]
            if any(not row for row in selected):continue
            for restart in range(cfg['matching_restarts']):
                ids=[0 if restart==0 else int(generator.integers(len(row))) for row in selected]
                for sweep in range(5):
                    changed=False
                    for i in generator.permutation(6):
                        costs=[(q['error']/H)**2+sum(pair_penalty(i,perm[i],k,j,perm[j],ids[j])
                            for j in range(6) if i!=j) for k,q in enumerate(selected[i])]
                        k=int(np.argmin(costs))
                        if costs[k]+1e-14<costs[ids[i]]:ids[i]=k;changed=True
                    if not changed:break
                picked=[selected[i][ids[i]] for i in range(6)]
                z=np.array([q['row'] for q in picked]).ravel()
                error=max(q['error'] for q in picked)
                sep=separation([q['path'] for q in picked],[q['age'] for q in picked],cfg['collision_samples'])
                score=max(0,error-cfg['position_tolerance'])/H+2*max(0,search_gap-sep)/H
                pool.append(dict(z=z,perm=np.array(perm,int),score=score,error=error))
        pool.sort(key=lambda q:(q['score'],q['error']))
        unique=[]
        for item in pool:
            if all(np.linalg.norm(item['z']-old['z'])>.02 or not np.array_equal(item['perm'],old['perm']) for old in unique):
                unique.append(item)
            if len(unique)>=cfg['joint_starts']:break
        return unique

    def candidate(planner,z,goal,perm,replay):
        paths,ages,actual=planner.endpoints(z)
        valid=all(p.ok and a<=p.end+1e-8 for p,a in zip(paths,ages))
        error=float(np.linalg.norm(actual-goal,axis=1).max())
        sep=separation(paths,ages,cfg['collision_samples']) if valid else 0.
        score=max(0,error-cfg['position_tolerance'])/H+2*max(0,search_gap-sep)/H+(0 if valid else 10)
        return dict(z=np.asarray(z).copy(),goal=goal,perm=perm.copy(),replay=replay,score=score,error=error)

    def plan_letter(kind):
        goal,edges=goals[kind];options=[]
        for i,station in enumerate(stations):
            options.append([fit_station_target(i,point) for point in goal])
            print(kind,station['station_id'],'individual errors:',
                  [round(min((q['error'] for q in row),default=1e3),5) for row in options[-1]],flush=True)
        seeds=build_candidates(options,goal)
        if not seeds:raise RuntimeError('No complete six-station assignment. Current target may be unreachable within the fixed velocity/age bounds.')
        pool=[]
        for k,seed in enumerate(seeds):
            assigned=goal[seed['perm']]
            paths,ages,_=coarse.endpoints(seed['z'])
            first=[snapshot(coarse,seed['z'],paths,ages,assigned,'fixed station seed selection',0)]
            pool.append(candidate(coarse,seed['z'],assigned,seed['perm'],first))
            z,replay,diag=coarse.refine(seed['z'],assigned,cfg['joint_nfev'])
            pool.append(candidate(coarse,z,assigned,seed['perm'],first+replay))
            print(kind,'joint start',k+1,'lookup error',pool[-1]['error'],flush=True)
        pool.sort(key=lambda q:(q['score'],q['error']));verified=[]
        for item in pool[:2]:
            verified.append(candidate(direct,item['z'],item['goal'],item['perm'],item['replay']))
            z,replay,diag=direct.refine(item['z'],item['goal'],cfg['direct_nfev'])
            verified.append(candidate(direct,z,item['goal'],item['perm'],item['replay']+replay))
        verified.sort(key=lambda q:(q['score'],q['error']));best=verified[0]
        rows=best['z'].reshape(6,3);assigned=best['goal'];controls=[];ages=[]
        strict=[];reference=[]
        for i,row in enumerate(rows):
            c,age=direct.physical(row,i);controls.append(c);ages.append(age)
            strict.append(trace_particle(direct.field,c,int(sides[i]),tmax=age,rtol=2e-9,atol=2e-11,max_step=.0175))
            reference.append(trace_particle(exact_flow,c,int(sides[i]),tmax=age,rtol=2e-10,atol=2e-12,max_step=.0175))
        ages=np.array(ages)
        valid=all(p.ok and a<=p.end+1e-8 for p,a in zip(strict,ages))
        refvalid=all(p.ok and a<=p.end+1e-8 for p,a in zip(reference,ages))
        actual=np.array([p.state([a],strict=False)[0,:2] for p,a in zip(strict,ages)])
        ref=np.array([p.state([a],strict=False)[0,:2] for p,a in zip(reference,ages)])
        errors=np.linalg.norm(actual-assigned,axis=1);referrors=np.linalg.norm(ref-assigned,axis=1)
        delta=np.linalg.norm(actual-ref,axis=1)
        sep=separation(strict,ages,cfg['verify_samples'],True) if valid else 0.
        refsep=separation(reference,ages,cfg['verify_samples'],True) if refvalid else 0.
        T=float(max(ages)+cfg['lead_time']);schedule=[];paths=[]
        for i,(c,p,age) in enumerate(zip(controls,strict,ages)):
            j=3*sides[i]
            schedule.append(dict(particle=i+1,station_id=stations[i]['station_id'],launcher=['bottom','top'][sides[i]],
                x_launch=float(c[j]),y_launch=float(p.state([0])[0,1]),u_launch=float(c[j+1]),v_launch=float(c[j+2]),
                release_time=float(T-age),age_at_target=float(age),target_point=int(best['perm'][i])+1,
                target_x=float(assigned[i,0]),target_y=float(assigned[i,1]),actual_x=float(actual[i,0]),actual_y=float(actual[i,1]),
                position_error=float(errors[i]),reference_error=float(referrors[i])))
            tt=np.linspace(0,min(age,p.end),501)
            paths.append(dict(particle=i+1,side=int(sides[i]),release=float(T-age),flow_controls=c.tolist(),
                              ages=tt.tolist(),states=p.state(tt).tolist()))
        # Assert exact station use, not just approximate position penalties.
        assert len(schedule)==6 and len({r['station_id'] for r in schedule})==6
        assert all(r['x_launch']==s['x_launch'] for r,s in zip(schedule,stations))
        position_ok=bool(valid and errors.max()<=cfg['position_tolerance'])
        clearance_ok=bool(valid and sep>=required_gap)
        reference_ok=bool(refvalid and referrors.max()<=cfg['reference_tolerance'] and
                          delta.max()<=cfg['reference_tolerance'] and refsep>=required_gap)
        success=bool(position_ok and clearance_ok and reference_ok and VALIDATION['passed'])
        replay=best['replay']+[snapshot(direct,best['z'],strict,ages,assigned,'strict verification',len(best['replay']))]
        inverse=np.argsort(best['perm'])
        mapped_edges=[[int(inverse[a]),int(inverse[b])] for a,b in edges]
        reasons=[]
        if not position_ok:reasons.append('target tolerance missed or premature path termination')
        if not clearance_ok:reasons.append('clearance failed')
        if not reference_ok:reasons.append('analytical-flow cross-check failed')
        return dict(kind=kind+'_6',closed=False,success=success,
            reason='all numerical checks passed' if success else '; '.join(reasons),
            control_mode='per-particle-independent',launch_position_mode='six-fixed-stations',
            continuous_decisions=18,per_station_decisions=['u0','v0','flight_age'],stations=stations,
            flow_validated=bool(VALIDATION['passed']),position_success=position_ok,clearance_success=clearance_ok,
            reference_success=reference_ok,target_time=T,target_center=center.tolist(),
            targets=assigned.tolist(),target_edges=mapped_edges,actual=actual.tolist(),analytical_reference=ref.tolist(),
            max_error=float(errors.max()),rms_error=float(np.sqrt(np.mean(errors**2))),max_reference_error=float(referrors.max()),
            max_PINN_reference_delta=float(delta.max()),min_pair_distance=float(sep),reference_min_pair_distance=float(refsep),
            required_pair_distance=required_gap,schedules=schedule,paths=paths,optimization_replay=replay,
            interpretation='Finite multistart search; no guarantee of global feasibility; one simultaneous six-particle snapshot',
            collision_check='Shared-clock samples plus refinement of observed minima; no particle contact forces')

    def save_csv(filename,records):
        if not records:return
        with filename.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

    def plot_result(result,folder):
        goal=np.array(result['targets']);actual=np.array(result['actual'])
        fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained',gridspec_kw={'height_ratios':[1,2]})
        for i,p in enumerate(result['paths']):
            xy=np.array(p['states'])[:,:2];color=['#0786a5','#da4e70'][p['side']]
            axes[0].plot(xy[:,0],xy[:,1],color=color,alpha=.7)
            axes[0].scatter(stations[i]['x_launch'],H if p['side'] else 0,color=color,marker='s')
            axes[0].annotate(stations[i]['station_id'],(stations[i]['x_launch'],H if p['side'] else 0),
                xytext=(3,6 if p['side'] else -12),textcoords='offset points',fontsize=8)
        for ax in axes:
            ax.axhline(0,color='k');ax.axhline(H,color='k');ax.set(xlabel='x',ylabel='y',aspect='equal')
        axes[0].set(xlim=(0,L),ylim=(-.025,H+.025),title='Six fixed stations: one particle per station')
        for a,b in result['target_edges']:axes[1].plot(goal[[a,b],0],goal[[a,b],1],'--',color='#8493a8')
        axes[1].scatter(goal[:,0],goal[:,1],marker='x',color='k',label='Target')
        axes[1].scatter(actual[:,0],actual[:,1],facecolors='none',edgecolors='#0786a5',s=80,label='Direct PINN')
        ref=np.array(result['analytical_reference']);axes[1].scatter(ref[:,0],ref[:,1],s=15,color='#e8a342',label='Analytical flow')
        for i,p in enumerate(actual):axes[1].annotate(stations[i]['station_id'],p,xytext=(4,4),textcoords='offset points',fontsize=8)
        axes[1].set(xlim=(max(0,goal[:,0].min()-.07),min(L,goal[:,0].max()+.07)),ylim=(0,H),
                    title=result['kind']+(' — PASS' if result['success'] else ' — BEST FOUND')+
                    ' | max error = %.3g'%result['max_error'])
        axes[1].legend(loc='upper left',bbox_to_anchor=(1.01,1));fig.savefig(folder/'formation.png',dpi=220)
        plt.show()

    def adapt_app(filename):
        # Reuse the final notebook's app; add explicit graph edges so T is a T.
        page=filename.read_text(encoding='utf-8')
        if "const TITLES={" not in page:
            raise RuntimeError('Use the app-definition cell from Rectangular_Poiseuille_Independent_Final.ipynb.')
        page=page.replace("const TITLES={","const TITLES={S_6:'S — شش ایستگاه',C_6:'C — شش ایستگاه',T_6:'T — شش ایستگاه',",1)
        begin=page.index('function icon(r){');end=page.index('\nfunction buildCards()',begin)
        page=page[:begin]+'''function icon(r){let p=r.targets,x=p.map(q=>q[0]),y=p.map(q=>q[1]),cx=(Math.max(...x)+Math.min(...x))/2,cy=(Math.max(...y)+Math.min(...y))/2,s=28/Math.max(Math.max(...x)-Math.min(...x),Math.max(...y)-Math.min(...y),.001),xy=p.map(q=>[20+(q[0]-cx)*s,21-(q[1]-cy)*s]),col=r.success?'#51e0c5':'#ffca78';return `<svg viewBox="0 0 40 44" aria-hidden="true">${r.target_edges.map(([a,b])=>`<line x1="${xy[a][0]}" y1="${xy[a][1]}" x2="${xy[b][0]}" y2="${xy[b][1]}" stroke="${col}" opacity=".5"/>`).join('')}${xy.map(q=>`<circle cx="${q[0]}" cy="${q[1]}" r="1.8" fill="${col}"/>`).join('')}</svg>`}
''' +page[end:]
        old="line(ctx,target.concat(r.closed?[target[0]]:[]),'#e3ecfa88',1.3,[5,5]);"
        if old not in page:raise RuntimeError('The current notebook app has an unsupported target renderer.')
        page=page.replace(old,"r.target_edges.forEach(([a,b])=>line(ctx,[target[a],target[b]],'#e3ecfa88',1.3,[5,5]));",1)
        page=page.replace('(4*V.launches.length)','(3*V.launches.length)')
        page=page.replace('پورت رهاسازی مستقل برای هر ذره','۶ ایستگاه ثابت؛ یک پرتاب از هر ایستگاه')
        page=page.replace('مکان، سرعت و زمان رهاسازی مستقل هر ذره؛ یک شکل مشترک نزدیک خروجی.',
            'شش ایستگاه ثابت در یک‌چهارم، نیمه و سه‌چهارم کانال؛ سرعت و زمان رهاسازی مستقل.')
        page=page.replace('از رهاسازی تا شکل هدف','S، C و T با شش ایستگاه ثابت')
        page=page.replace('نزدیک خروجی','در ناحیهٔ هدف تعیین‌شده')
        page=page.replace('در هر گام، موقعیت‌ها، سرعت‌های مستقل هر ذره و زمان رهاسازی واقعاً با حل‌گر محاسبه شده‌اند.',
            'ایستگاه‌ها ثابت‌اند؛ در هر گام سرعت‌ها و زمان‌های رهاسازی با حل‌گر محاسبه شده‌اند.')
        filename.write_text(page,encoding='utf-8')

    results=[]
    for kind in cfg['shapes']:
        folder=task_run/'formations'/(kind+'_6');folder.mkdir(parents=True)
        print('\nDESIGN',kind,'— 6 fixed stations / 18 continuous variables',flush=True)
        try:result=plan_letter(kind)
        except (RuntimeError,ValueError,FloatingPointError) as error:
            result=dict(kind=kind+'_6',success=False,reason=str(error))
        results.append(result);(folder/'result.json').write_text(json.dumps(result,indent=2))
        if result.get('schedules'):
            save_csv(folder/'release_schedule.csv',result['schedules']);plot_result(result,folder)
            write_formation_dashboard([result],folder/'replay.html',task_run);adapt_app(folder/'replay.html')
        print(json.dumps({k:result[k] for k in ('kind','success','reason','max_error','min_pair_distance') if k in result}),flush=True)
        (task_run/'summary.json').write_text(json.dumps([{k:v for k,v in r.items()
            if k not in ('paths','optimization_replay')} for r in results],indent=2))
    assert weights_hash()==original_hash,'Unexpected neural-weight change.'
    (task_run/'weight_integrity.json').write_text(json.dumps(dict(unchanged=True,sha256=original_hash),indent=2))
    usable=[r for r in results if r.get('paths')]
    app=None
    if usable:
        app=write_formation_dashboard(usable,task_run/'six_station_SCT_lab.html',task_run);adapt_app(app)
        display(HTML('<iframe title="Six fixed station formation lab" style="width:100%;height:1700px;border:0;border-radius:14px" '
            'sandbox="allow-scripts allow-downloads" srcdoc="'+html.escape(app.read_text(),quote=True)+'"></iframe>'))
        display(FileLink(str(app)))
    save_csv(task_run/'summary.csv',[{k:r.get(k) for k in ('kind','success','max_error','max_reference_error','min_pair_distance','reason')} for r in results])
    archive=shutil.make_archive(str(task_run),'zip',root_dir=task_run)
    display(FileLink(archive))
    print('Finished:',sum(bool(r.get('success')) for r in results),'/',len(results),'passed.',flush=True)
    print('A failed finite search is not a proof of impossibility; station positions and target sizes were never relaxed.')
    return dict(run=task_run,results=results,app=app,archive=archive,stations=stations)
