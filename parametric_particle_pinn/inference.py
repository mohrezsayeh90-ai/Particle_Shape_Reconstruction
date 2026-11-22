"""Batched trajectory inference and scalar root finding on the neural path."""

import numpy as np
import torch
from scipy.optimize import brentq

from .domain import check_releases, wall_release  # public standalone adapter
from .model import ParticlePINN, load_checkpoint, particle_state

EVENTS = ("lower_wall", "upper_wall", "inlet_exit", "outlet")
AXES = (1, 1, 0, 0)


def _evaluate(model, releases, times, *, acceleration=True):
    batch, count = times.shape
    shapes = (batch, count, 2)
    time = model.bounds.new_tensor(times.reshape(-1, 1))
    release = model.bounds.new_tensor(releases).repeat_interleave(count, dim=0)
    values = [[], [], []] if acceleration else [[], []]
    for start in range(0, len(time), 4096):
        stop = start + 4096
        state = particle_state(model, time[start:stop], release[start:stop],
                               acceleration=acceleration)
        for component, value in zip(values, state):
            component.append(value.detach().cpu().numpy())
    return tuple(np.concatenate(component).reshape(shapes) for component in values)


def _events(model, releases):
    config = model.config
    p, d = config["physics"], config["domain"]
    horizon = d["horizon"]
    radius = p["diameter"] / 2
    offsets = np.array([radius, p["height"] - radius, 0.0, p["length"]])
    signs = np.array([1, -1, 1, -1])
    samples = int(np.ceil(horizon / config["validation"]["event_scan_step"]))
    mesh = np.linspace(0, horizon, samples + 1)
    positions, velocities = _evaluate(model, releases,
                                      np.broadcast_to(mesh, (len(releases), len(mesh))),
                                      acceleration=False)
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise FloatingPointError("Non-finite neural trajectory during event scan.")
    distances = (positions[..., AXES] - offsets) * signs
    rates = velocities[..., AXES] * signs
    event_types, event_times = [], []
    for index, release in enumerate(releases):
        initial_rate = rates[index, 0]
        first_type, first_time = "time_limit", horizon
        for face, event in enumerate(EVENTS):
            values, slopes = distances[index, :, face], rates[index, :, face]
            if values[0] == 0 and (initial_rate[face] < 0 or
                                   (face < 2 and initial_rate[face] == 0)):
                first_type, first_time = event, 0.0
                break

            axis, offset, sign = AXES[face], offsets[face], signs[face]

            def scalar(time, velocity=False):
                t = model.bounds.new_tensor([[time]])
                r = model.bounds.new_tensor(release[None, :])
                if velocity:
                    _, v, _ = particle_state(model, t, r, acceleration=False)
                    return sign * v[0, axis].item()
                with torch.no_grad():
                    return sign * (model(t, r)[0, axis].item() - offset)

            # Inspect sign changes and minima for brief excursions and tangencies.
            crossing = (values[:-1] >= 0) & (values[1:] < 0)
            endpoint = (values[-1] == 0 and slopes[-1] < 0)
            minimum = (slopes[:-1] < 0) & (slopes[1:] >= 0)
            for left in np.flatnonzero(crossing | minimum):
                start, stop = mesh[left:left + 2]
                if start >= first_time:
                    break
                if crossing[left]:
                    hit = brentq(scalar, start, stop, xtol=1e-12)
                else:
                    bottom = brentq(lambda t: scalar(t, velocity=True), start, stop, xtol=1e-12)
                    clearance = scalar(bottom)
                    if face < 2 and abs(clearance) <= 1e-12:
                        hit = bottom
                    elif clearance < 0 and values[left] > 0:
                        hit = brentq(scalar, start, bottom, xtol=1e-12)
                    else:
                        continue
                if hit < first_time:
                    first_type, first_time = event, hit
                break
            if endpoint and first_type == "time_limit":
                first_type = event
        event_types.append(first_type)
        event_times.append(first_time)
    return np.array(event_types), np.array(event_times)


def predict_trajectory(checkpoint, x0, y0, vx0, vy0, times, *, device="cpu"):
    """Predict at shared 1-D times or one array per broadcast release condition.

    Returns nondimensional time, position, velocity, acceleration, an event
    dictionary and validity mask. Trajectory state is NaN beyond the first event.
    Pass an already loaded model to reuse it across calls.
    """
    model = checkpoint if isinstance(checkpoint, ParticlePINN) else load_checkpoint(checkpoint, device)
    releases = np.stack(np.broadcast_arrays(x0, y0, vx0, vy0), axis=-1).astype(float)
    check_releases(model.config, releases)
    batch_shape = releases.shape[:-1]
    releases = releases.reshape(-1, 4)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if not times.size or not np.isfinite(times).all():
        raise ValueError("Provide finite nonempty elapsed times.")
    horizon = model.config["domain"]["horizon"]
    if np.any((times < 0) | (times > horizon)):
        raise ValueError(f"Elapsed times must be in [0, {horizon}].")
    output_shape = (*batch_shape, times.shape[-1])
    if times.ndim != 1 and times.shape != output_shape:
        raise ValueError("Use shared 1-D times or match the release batch axes.")
    elapsed = np.broadcast_to(times, output_shape)
    flat_times = elapsed.reshape(len(releases), -1)
    kind, end = _events(model, releases)
    valid = flat_times <= end[:, None]
    # Evaluate invalid queries at t=0 only, then mask all state fields.
    state = _evaluate(model, releases, np.where(valid, flat_times, 0.0))
    result = {"elapsed_time": elapsed, "valid": valid.reshape(output_shape),
              "event": {"type": kind.reshape(batch_shape), "time": end.reshape(batch_shape)}}
    if not batch_shape:
        result["event"] = {"type": kind[0].item(), "time": end[0].item()}
    for name, values in zip(("position", "velocity", "acceleration"), state):
        values[~valid] = np.nan
        result[name] = values.reshape(*output_shape, 2)
    return result
