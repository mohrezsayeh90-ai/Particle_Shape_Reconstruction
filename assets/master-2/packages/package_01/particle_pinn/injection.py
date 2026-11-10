"""Fixed S-channel, a three-input flow PINN, and six-parameter particle queries.

The particle initial conditions condition the Lagrangian IVP, never the fluid.
All quantities use the nondimensionalization of the original repository.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn

from .parametric import ParametricFlow, drag_acceleration, map_points, pde_residual

R_IN = 0.2
R_OUT = 0.5
RADIUS = (R_IN + R_OUT) / 2
HALF_WIDTH = (R_OUT - R_IN) / 2
LENGTH = 2 * math.pi * RADIUS
PARAMETER_NAMES = ("re", "diameter", "density_ratio", "x_p0", "u_p0", "v_p0")
SCOPE = "fixed-s-channel-variable-injection-v1"


@dataclass(frozen=True)
class InjectionCase:
    re: float = 1.0
    diameter: float = 0.025
    density_ratio: float = 1000.0
    x_p0: float = 0.0
    u_p0: float = 0.0
    v_p0: float = 1.0

    def array(self):
        return np.array([getattr(self, key) for key in PARAMETER_NAMES])

    def validate(self):
        if not np.isfinite(self.array()).all():
            raise ValueError("Particle parameters must be finite.")
        if min(self.re, self.diameter, self.density_ratio) <= 0:
            raise ValueError("Re, diameter and density ratio must be positive.")
        if abs(self.x_p0) + self.diameter / 2 >= HALF_WIDTH:
            raise ValueError("Particle center/radius intersects an inlet wall.")
        return self

    def json(self):
        return asdict(self)


@dataclass(frozen=True)
class ParameterRanges:
    re: tuple = (0.1, 10.0)
    diameter: tuple = (0.0025, 0.25)
    density_ratio: tuple = (100.0, 10000.0)
    x_p0: tuple = (-0.14, 0.14)
    u_p0: tuple = (-0.5, 0.5)
    v_p0: tuple = (0.25, 1.5)

    def validate(self):
        for key in PARAMETER_NAMES:
            low, high = getattr(self, key)
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError(f"Invalid range for {key}.")
        if min(self.re[0], self.diameter[0], self.density_ratio[0]) <= 0:
            raise ValueError("Positive ranges are required for Re, diameter and density ratio.")
        if self.diameter[1] >= 2 * HALF_WIDTH:
            raise ValueError("Maximum particle diameter must be below channel width 0.3.")
        if self.x_p0[0] <= -HALF_WIDTH or self.x_p0[1] >= HALF_WIDTH:
            raise ValueError("x_p0 range must lie strictly inside (-0.15, 0.15).")
        # Every diameter must admit at least one center in the configured range.
        nearest = min(max(0.0, self.x_p0[0]), self.x_p0[1])
        if abs(nearest) + self.diameter[1] / 2 >= HALF_WIDTH:
            raise ValueError("Some configured diameters admit no inlet position in x_p0 range.")
        return self

    def json(self):
        return {key: list(getattr(self, key)) for key in PARAMETER_NAMES}


def sample_injections(rng, count, ranges):
    """Log-uniform positive parameters; uniform x conditional on sphere clearance.

    Unlike geometry rejection in the original sampler, all samples have the same
    two radii. Speeds retain their signs and are sampled linearly, never in log.
    """
    ranges.validate()
    if count < 1:
        raise ValueError("count must be positive")
    values = np.empty((count, 6))
    for i, key in enumerate(PARAMETER_NAMES[:3]):
        low, high = getattr(ranges, key)
        values[:, i] = np.exp(rng.uniform(np.log(low), np.log(high), count))
    clearance = HALF_WIDTH - values[:, 1] / 2
    low = np.maximum(ranges.x_p0[0], -clearance + 1e-8)
    high = np.minimum(ranges.x_p0[1], clearance - 1e-8)
    if np.any(low > high):
        raise ValueError("No numerically admissible inlet position for a sampled diameter.")
    values[:, 3] = rng.uniform(low, high)
    for i, key in enumerate(PARAMETER_NAMES[4:], 4):
        values[:, i] = rng.uniform(*getattr(ranges, key), count)
    return [InjectionCase(*row).validate() for row in values]


class FixedGeometryFlow(ParametricFlow):
    """Actual trainable MLP input: [x, y, Re] -> [u, v, p].

    Reuses the original hard wall/cap boundary ansatz and physical derivatives.
    Geometry is constant in both the MLP and all residual/flux calculations.
    """
    def __init__(self, width=128, depth=6):
        super().__init__(width, depth)
        self.net[0] = nn.Linear(3, width)
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)

    def features(self, xy, p):
        return torch.cat((torch.stack((xy[:, 0] / RADIUS - 2,
                                        xy[:, 1] / RADIUS), dim=1),
                          torch.log10(p[:, 2:3])), dim=1)

    @staticmethod
    def geometry_parameters(re):
        re = re.reshape(-1, 1)
        return torch.cat((torch.full_like(re, R_IN), torch.full_like(re, R_OUT), re), dim=1)

    def forward(self, xy, re):
        return super().forward(xy, self.geometry_parameters(re))

    def evaluate(self, xy, case):
        re = case.re if hasattr(case, "re") else float(case)
        if not np.isfinite(re) or re <= 0:
            raise ValueError("Re must be finite and positive.")
        points = torch.as_tensor(np.atleast_2d(xy), dtype=self.base.dtype, device=self.base.device)
        reynolds = points.new_full((len(points), 1), re)
        with torch.no_grad():
            return self(points, reynolds).cpu().numpy()


def sample_flow_batch(rng, size, re_range, device="cpu", anchor_re=True):
    low, high = re_range
    values = np.exp(rng.uniform(np.log(low), np.log(high), size))
    if anchor_re and size >= 3:
        values[:3] = [low, np.sqrt(low * high), high]
    re = torch.tensor(values[:, None], dtype=torch.float32, device=device)
    p = FixedGeometryFlow.geometry_parameters(re)
    q = torch.tensor(rng.uniform(0.002, 0.998, size), dtype=torch.float32, device=device)
    eta = torch.tensor(rng.uniform(-0.995, 0.995, size), dtype=torch.float32, device=device)
    xy = map_points(q, eta, p).detach().requires_grad_(True)
    return xy, re


def flow_objective(model, xy, re, continuity_weight=20.0):
    p = model.geometry_parameters(re)
    # Adapter preserves original PDE scaling without passing radii to the MLP.
    residual = pde_residual(lambda x, pp: model(x, pp[:, 2:3]), xy, p)
    continuity = residual[:, 0].square().mean()
    momentum = residual[:, 1:].square().mean()
    indices = torch.linspace(0, len(re) - 1, min(6, len(re)), device=re.device).long()
    rr = re[indices].repeat_interleave(51, dim=0)
    pp = model.geometry_parameters(rr)
    q = re.new_tensor([0.2, 0.5, 0.8]).repeat_interleave(17).repeat(len(indices))
    eta = torch.linspace(-1, 1, 17, device=re.device).repeat(3 * len(indices))
    from .parametric import coordinates
    xx = map_points(q, eta, pp)
    _, _, tangent = coordinates(xx, pp)
    tangential = (model(xx, rr)[:, :2] * tangent).sum(1).reshape(len(indices), 3, 17)
    flux = (torch.trapezoid(tangential, eta[:17], dim=2) / (4 / 3) - 1).square().mean()
    total = continuity_weight * continuity + momentum + 5 * flux
    return total, continuity, momentum, flux


def channel_coordinates(xy):
    """Exact signed longitudinal/transverse coordinates, extended beyond caps."""
    xy = np.atleast_2d(xy)
    second = (xy[:, 0] > 3 * RADIUS) | ((xy[:, 0] >= RADIUS) & (xy[:, 1] < 0))
    dx = xy[:, 0] - np.where(second, 3 * RADIUS, RADIUS)
    dy = np.where(second, -xy[:, 1], xy[:, 1])
    angle = math.pi - np.arctan2(np.maximum(dy, 0), dx)
    s = np.where(second, angle + math.pi, angle) * RADIUS
    radial = np.sqrt(dx ** 2 + xy[:, 1] ** 2)
    eta = np.where(second, RADIUS - radial, radial - RADIUS) / HALF_WIDTH
    before = (xy[:, 0] < RADIUS) & (xy[:, 1] < 0)
    after = (xy[:, 0] > 3 * RADIUS) & (xy[:, 1] > 0)
    s = np.where(before, xy[:, 1], np.where(after, LENGTH + xy[:, 1], s))
    # Tangent extension gives planar, rather than circular, cap clearances.
    eta = np.where(before, -xy[:, 0] / HALF_WIDTH, eta)
    eta = np.where(after, (xy[:, 0] - 4 * RADIUS) / HALF_WIDTH, eta)
    return s, eta


def trace_injection(field, case, tmax=10.0, samples=260, drag="schiller-naumann",
                    max_evaluations=20000):
    """Explicit initial position and BOTH initial velocity components, with events.

    field(xy) must return u,v,p. No training is required when an IVP initial
    condition changes. Initial velocity is never overwritten by fluid velocity.
    """
    case.validate()
    if not np.isfinite(tmax) or tmax <= 0 or samples < 2:
        raise ValueError("tmax must be positive and samples must be >= 2.")
    if drag not in ("schiller-naumann", "stokes"):
        raise ValueError("Unknown drag law.")
    initial = np.array([case.x_p0, 0.0, case.u_p0, case.v_p0])
    evaluations = 0

    def rhs(time, state):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_evaluations:
            raise RuntimeError("Particle RHS evaluation budget exhausted.")
        fluid = np.asarray(field(state[None, :2]))[0, :2]
        if not np.isfinite(fluid).all():
            raise FloatingPointError("Non-finite flow along particle path.")
        if drag == "stokes":
            acc = 18 * (fluid - state[2:]) / (case.density_ratio * case.re * case.diameter ** 2)
        else:
            acc = drag_acceleration(fluid, state[2:], case.diameter, case.re, case.density_ratio)
        return np.r_[state[2:], acc]

    def wall(time, state):
        _, eta = channel_coordinates(state[None, :2])
        return HALF_WIDTH * (1 - abs(eta[0])) - case.diameter / 2

    def outlet(time, state):
        return LENGTH - channel_coordinates(state[None, :2])[0][0]

    def inlet(time, state):
        value = channel_coordinates(state[None, :2])[0][0]
        # Avoid a spurious t=0 exit for a resting particle accelerated inward.
        return 1e-12 if time == 0 and case.v_p0 >= 0 else value

    for event in (wall, outlet, inlet):
        event.terminal = True
        event.direction = -1
    tau = case.density_ratio * case.re * case.diameter ** 2 / 18
    speed = max(1.0, math.hypot(case.u_p0, case.v_p0))
    max_step = min(0.025, HALF_WIDTH / (10 * speed))
    try:
        sol = solve_ivp(rhs, (0, tmax), initial,
                        method="Radau" if tau < max_step / 3 else "RK45",
                        rtol=2e-6, atol=2e-8, max_step=max_step,
                        dense_output=True, events=(wall, outlet, inlet))
        times = np.linspace(0, sol.t[-1], samples) if sol.t[-1] > 0 else np.array([0.0])
        state = sol.sol(times).T
        outcome = next((name for name, event in zip(("stuck", "exited", "backflow_exit"), sol.t_events)
                        if len(event)), "time_limit" if sol.success else "integration_failed")
        # Keep the pre-impact velocity: terminal contact ends this trajectory.
        fluid = np.asarray(field(state[:, :2]))[:, :2]
        rep = case.re * case.diameter * np.linalg.norm(fluid - state[:, 2:], axis=1)
        return dict(t=times, state=state, outcome=outcome, message=sol.message,
                    max_particle_re=float(rep.max()), rhs_evaluations=evaluations)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return dict(t=np.array([0.0]), state=initial[None, :], outcome="integration_failed",
                    message=str(error), max_particle_re=None, rhs_evaluations=evaluations)


def load_injection_model(checkpoint, device="cpu"):
    """Load new-format selected weights; reject the old five-input checkpoints."""
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    if ck.get("scope") == "fixed-s-channel-smooth-v2":
        from .smooth_flow_v2 import SmoothCartesianFlow
        cfg = ck["config"]
        model = SmoothCartesianFlow(cfg["width"],cfg["depth"]).to(device)
        model.load_state_dict(ck["model"])
        return model.eval(), cfg
    if ck.get("scope") != SCOPE:
        raise ValueError("Incompatible checkpoint: train the fixed-geometry model from scratch.")
    cfg = ck["config"]
    model = FixedGeometryFlow(cfg["width"], cfg["depth"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), cfg
