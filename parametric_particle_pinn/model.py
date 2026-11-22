"""Hard initial conditions and automatic derivatives in nondimensional time."""

import json
from pathlib import Path

import torch
from torch import nn

CONFIG_PATH = Path(__file__).with_name("config.json")
RELEASE_INPUTS = ("x0", "y0", "vx0", "vy0")


def read_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())


def relaxation_time(config):
    p = config["physics"]
    return p["density_ratio"] * p["reynolds"] * p["diameter"] ** 2 / 18


def input_bounds(config):
    p, d = config["physics"], config["domain"]
    radius = p["diameter"] / 2
    return [
        [0, d["horizon"]], [0, p["length"]], [radius, p["height"] - radius],
        d["vx0"], d["interior_vy0"],
    ]


class ParticlePINN(nn.Module):
    """A conditional correction with learned time coefficients and release features."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        p, architecture = config["physics"], config["model"]
        if p["drag_law"] != "stokes":
            raise ValueError("ParticlePINN requires linear Stokes drag.")
        dtype = getattr(torch, architecture["dtype"])
        self.register_buffer("bounds", torch.tensor(input_bounds(config), dtype=dtype))
        degree = architecture["degree"]
        self.register_buffer("frequencies", torch.arange(degree + 1, dtype=dtype))
        self.register_buffer("basis_transform", torch.eye(degree + 1, dtype=dtype))
        self.horizontal = nn.Linear(degree + 1, 4, bias=False, dtype=dtype)
        self.transverse = nn.Linear(degree + 1, 1, bias=False, dtype=dtype)
        nn.init.zeros_(self.horizontal.weight)
        nn.init.zeros_(self.transverse.weight)

    def _time_features(self, time):
        # Logarithmic normalization resolves rapid early transients. The
        # endpoint inset keeps acos differentiable at t=0 and the horizon.
        horizon = self.config["domain"]["horizon"]
        scale = self.config["model"]["time_feature_scale"]
        shaped = 2 * torch.log1p(time / scale) / torch.log1p(time.new_tensor(horizon / scale)) - 1
        basis = torch.cos(torch.acos(0.999 * shaped) * self.frequencies)
        return time.square() / (time + scale) * basis

    def precondition(self):
        """Whiten the time basis in the Stokes differential operator norm.

        This uses the differential operator and time grid only. No release or
        reference trajectory values enter the transform.
        """
        horizon = self.config["domain"]["horizon"]
        grid = torch.linspace(0, horizon, 513, dtype=self.bounds.dtype, device=self.bounds.device)
        nodes = torch.cat((grid, horizon * (grid / horizon).pow(3)))
        features = lambda t: self._time_features(t.view(1))
        first = torch.func.jacfwd(features)
        second = torch.func.jacfwd(first)
        operator = torch.func.vmap(first)(nodes) + relaxation_time(self.config) * torch.func.vmap(second)(nodes)
        _, upper = torch.linalg.qr(operator / len(nodes)**0.5, mode="reduced")
        with torch.no_grad():
            self.basis_transform.copy_(torch.linalg.solve_triangular(
                upper, torch.eye(upper.shape[0], device=upper.device, dtype=upper.dtype), upper=True))

    def forward(self, time, release):
        # Inputs keep their physical scale on the autograd graph.
        time_basis = self._time_features(time) @ self.basis_transform
        a, b, c, d = self.horizontal(time_basis).unbind(dim=-1)
        e = self.transverse(time_basis).squeeze(-1)
        p = self.config["physics"]
        height_coordinate = 2 * release[..., 1] / p["height"] - 1
        vx = release[..., 2] / p["centerline_speed"]
        vy = release[..., 3] / self.bounds[4, 1]
        correction = torch.stack((
            a * (1 - height_coordinate.square()) + b * height_coordinate * vy
            + c * vy.square() + d * vx,
            e * vy,
        ), dim=-1)
        return release[..., :2] + time * release[..., 2:] + correction


def particle_state(model, time, release, *, train=False, acceleration=True):
    """Return positions and first and optionally second elapsed-time derivatives."""
    with torch.enable_grad():
        time = time.detach().requires_grad_(True)
        position = model(time, release)
        velocity = [
            torch.autograd.grad(component.sum(), time, create_graph=train or acceleration,
                                retain_graph=train or acceleration or axis == 0)[0]
            for axis, component in enumerate(position.unbind(dim=-1))
        ]
        if acceleration:
            second = torch.cat([
                torch.autograd.grad(component.sum(), time, create_graph=train,
                                    retain_graph=train or axis == 0)[0]
                for axis, component in enumerate(velocity)
            ], dim=-1)
        else:
            second = None
        return position, torch.cat(velocity, dim=-1), second


def scaled_residual(config, position, velocity, acceleration):
    p = config["physics"]
    eta = position[..., 1] / p["height"]
    carrier = torch.stack((4 * p["centerline_speed"] * eta * (1 - eta),
                           torch.zeros_like(eta)), dim=-1)
    scale = velocity.new_tensor(config["training"]["residual_velocity_scale"])
    return (relaxation_time(config) * acceleration + velocity - carrier) / scale


def load_checkpoint(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    config = data["config"]
    if config["schema"] != read_config()["schema"]:
        raise ValueError("Unsupported particle checkpoint schema.")
    model = ParticlePINN(config).to(device).eval()
    model.load_state_dict(data["model_state"])
    model.requires_grad_(False)
    return model
