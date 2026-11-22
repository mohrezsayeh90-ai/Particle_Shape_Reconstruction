"""Independent Stokes ODE validation and time-matched neural comparisons."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.integrate import solve_ivp

from .domain import sample_releases, wall_release
from .inference import predict_trajectory
from .model import load_checkpoint


def reference(config, release):
    p = config["physics"]
    height, length = p["height"], p["length"]
    radius = p["diameter"] / 2
    tau = p["density_ratio"] * p["reynolds"] * p["diameter"] ** 2 / 18
    horizon = config["domain"]["horizon"]

    def rhs(_time, state):
        y, vx, vy = state[1:]
        eta = y / height
        return vx, vy, (4 * p["centerline_speed"] * eta * (1 - eta) - vx) / tau, -vy / tau

    def lower(_time, state):
        return state[1] - radius

    def upper(_time, state):
        return height - radius - state[1]

    def inlet(_time, state):
        return state[0]

    def outlet(_time, state):
        return length - state[0]

    for name, clearance, rate in (
        ("lower_wall", lower(0, release), release[3]),
        ("upper_wall", upper(0, release), -release[3]),
        ("inlet_exit", inlet(0, release), release[2]),
        ("outlet", outlet(0, release), -release[2]),
    ):
        if clearance == 0 and (rate < 0 or (rate == 0 and name.endswith("wall"))):
            return name, 0.0, lambda t: np.broadcast_to(release, (np.size(t), 4))

    events = [lower, upper, inlet, outlet]
    for event in events:
        event.terminal, event.direction = True, -1
    sol = solve_ivp(rhs, (0, horizon), release, method="DOP853", rtol=2e-12, atol=2e-14,
                    max_step=0.05, dense_output=True, events=events)
    if not sol.success:
        raise RuntimeError(sol.message)
    names = ("lower_wall", "upper_wall", "inlet_exit", "outlet")
    kind = next((name for name, times in zip(names, sol.t_events) if len(times)), "time_limit")
    return kind, float(sol.t[-1]), lambda t: sol.sol(t).T


def conditions(config, split):
    seed = config["training"]["final_seed" if split == "final" else "dev_seed"]
    # A separate, fixed RNG stream also separates diagnostic dev trajectories
    # from the residual-only development bank used to select best.pt.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    if split == "dev":
        return [(f"dev_{i:02d}", row) for i, row in
                enumerate(sample_releases(config, rng, 32, boundaries=True))]
    p, d = config["physics"], config["domain"]
    height, length = p["height"], p["length"]
    radius = p["diameter"] / 2
    vmin, vmax = d["vx0"]
    smin, smax = d["inward_speed"]
    cases = [
        ("lower_wall_slow", wall_release(config, "lower", 0.315, vmin, smin)),
        ("lower_wall_fast", wall_release(config, "lower", 0.55, vmax, smax)),
        ("upper_wall_slow", wall_release(config, "upper", 0.72, 0.02, smin)),
        ("upper_wall_fast", wall_release(config, "upper", 1.10, -0.22, smax)),
        ("lower_contact", (0.9, 0.048, -0.15, -0.60)),
        ("upper_contact", (0.8, 0.25, 0.55, 0.47)),
        ("near_lower_contact", (1.1, radius + 1e-5, 0.7, -1.1)),
        ("near_upper_contact", (1.1, height - radius - 1e-5, 0.7, 1.1)),
        ("inlet_exit", (0.022, 0.13, vmin, 0.0)),
        ("outlet_exit", (length - 0.08, 0.21, vmax, 0.12)),
        ("inlet_edge_inward", (0.0, 0.15, 0.45, 0.0)),
        ("outlet_edge_inward", (length, 0.085, -0.22, -0.08)),
        ("inlet_edge_outward", (0.0, 0.18, vmin, 0.0)),
        ("outlet_edge_outward", (length, 0.23, vmax, 0.0)),
        ("lower_center_edge", (0.22, radius, vmax, smin)),
        ("upper_center_edge", (0.19, height - radius, 0.0, -smin)),
        ("lower_center_contact_at_release", (0.63, radius, 0.57, 0.0)),
        ("upper_center_contact_at_release", (0.91, height - radius, -0.11, 0.0)),
        ("slow_time_limit", (0.36, 0.022, -0.05, 0.0)),
        ("mid_height_zero_vy", (0.21, height / 2, 0.13, 0.0)),
        ("interior_rising", (0.31, 0.11, -0.28, 0.23)),
        ("interior_falling", (0.47, 0.21, 0.96, -0.42)),
    ]
    cases.extend((f"unseen_{i:02d}", row) for i, row in
                 enumerate(sample_releases(config, rng, 48)))
    return cases


def _plots(output, names, times, neural, truth):
    import matplotlib.pyplot as plt

    selected = [0, 2, 4, 5, 8, 9, 18, 20] if len(names) > 20 else list(range(len(names)))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    for ax, index in zip(axes.flat, selected):
        ref, pred = truth[index], neural["position"][index]
        ax.plot(ref[:, 0], ref[:, 1], color="#193856", label="DOP853", linewidth=2)
        ax.plot(pred[:, 0], pred[:, 1], "--", color="#d24b30", label="PINN")
        ax.set(title=names[index].replace("_", " "), xlabel="x", ylabel="y")
        ax.ticklabel_format(useOffset=False)
    axes.flat[0].legend()
    fig.savefig(output / "trajectories.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    values = np.concatenate((neural["position"], neural["velocity"]), axis=-1)
    for component, (ax, label) in enumerate(zip(axes.flat, ("x", "y", "vx", "vy"))):
        for time, errors in zip(times, abs(values[..., component] - truth[..., component])):
            ax.plot(time, errors, color="#1d678a", alpha=0.15)
        ax.set(xlabel="elapsed time", ylabel=f"|{label} error|", ylim=(0, None))
        ax.set_yscale("symlog", linthresh=1e-7)
    fig.savefig(output / "errors.png", dpi=170)
    plt.close(fig)


def validate(checkpoint, output, split="final"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = load_checkpoint(checkpoint)
    config = model.config
    torch.set_num_threads(config["training"]["threads"])
    p, tol = config["physics"], config["validation"]
    names, releases = zip(*conditions(config, split))
    releases = np.array(releases)
    tau = p["density_ratio"] * p["reynolds"] * p["diameter"]**2 / 18
    unit = np.linspace(0, 1, tol["samples_per_case"])
    # One warm-up run, then compare inference and numerical integration on the
    # same releases and same physical query grids, including event detection.
    predict_trajectory(model, *releases[0], [0.0])
    refs = [reference(config, row) for row in releases]
    reference_ends = np.array([item[1] for item in refs])
    reference_times = reference_ends[:, None] * unit
    start = perf_counter()
    benchmark = predict_trajectory(model, *releases.T, reference_times)
    neural_seconds = perf_counter() - start
    start = perf_counter()
    refs = [reference(config, row) for row in releases]
    for item, times in zip(refs, reference_times):
        item[2](times)
    ode_seconds = perf_counter() - start

    ends = np.minimum(reference_ends, benchmark["event"]["time"])
    times = ends[:, None] * unit
    neural = predict_trajectory(model, *releases.T, times)
    assert neural["valid"].all()
    truth = np.stack([item[2](t) for item, t in zip(refs, times)])
    position, velocity = neural["position"], neural["velocity"]
    position_error = abs(position - truth[..., :2])
    velocity_error = abs(velocity - truth[..., 2:])
    decay = np.exp(-times / tau)
    exact_vy = releases[:, 3, None] * decay
    exact_y = releases[:, 1, None] - tau * releases[:, 3, None] * np.expm1(-times / tau)
    transverse_errors = abs(np.stack((position[..., 1] - exact_y, velocity[..., 1] - exact_vy), axis=-1))
    independent_error = abs(np.stack((truth[..., 1] - exact_y, truth[..., 3] - exact_vy), axis=-1))
    eta = position[..., 1] / p["height"]
    carrier = np.stack((4 * p["centerline_speed"] * eta * (1 - eta), np.zeros_like(eta)), axis=-1)
    residual = ((tau * neural["acceleration"] + velocity - carrier) /
                config["training"]["residual_velocity_scale"])
    initial_error = abs(np.concatenate((position[:, 0], velocity[:, 0]), axis=-1) - releases)

    rows = []
    for i, (name, ref) in enumerate(zip(names, refs)):
        error_q, error_v = position_error[i].max(axis=0), velocity_error[i].max(axis=0)
        rms_q = np.sqrt((position_error[i]**2).mean(axis=0))
        rms_v = np.sqrt((velocity_error[i]**2).mean(axis=0))
        rms_r = np.sqrt((residual[i]**2).mean(axis=0))
        peak_r = abs(residual[i]).max(axis=0)
        events_agree = bool(neural["event"]["type"][i] == ref[0])
        error_t = abs(float(neural["event"]["time"][i]) - ref[1])
        passed = bool(
            events_agree and error_t <= tol["event_time_max"] and
            np.all(error_q <= tol["absolute_position_max"]) and
            np.all(error_v <= tol["absolute_velocity_max"]) and
            np.all(rms_r <= tol["scaled_residual_rms"]) and
            np.all(peak_r <= tol["scaled_residual_max"]) and
            initial_error[i].max() <= tol["initial_max"] and
            independent_error[i].max() <= tol["reference_check_max"])
        rows.append(dict(
            case=name, **dict(zip(("x0", "y0", "vx0", "vy0"), releases[i])),
            reference_event=ref[0], neural_event=neural["event"]["type"][i],
            reference_end=ref[1], neural_end=neural["event"]["time"][i],
            common_end=ends[i], event_time_error=error_t, event_agrees=events_agree,
            max_x_error=error_q[0], max_y_error=error_q[1],
            max_vx_error=error_v[0], max_vy_error=error_v[1],
            rms_x_error=rms_q[0], rms_y_error=rms_q[1],
            rms_vx_error=rms_v[0], rms_vy_error=rms_v[1],
            rms_scaled_rx=rms_r[0], rms_scaled_ry=rms_r[1],
            max_scaled_rx=peak_r[0], max_scaled_ry=peak_r[1],
            max_initial_error=initial_error[i].max(),
            max_exact_y_error=transverse_errors[i, :, 0].max(),
            max_exact_vy_error=transverse_errors[i, :, 1].max(),
            max_reference_transverse_error=independent_error[i].max(), passed=passed))
    fields = list(rows[0])
    with (output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    coverage = sorted({item[0] for item in refs})
    summary = dict(
        split=split, checkpoint=str(checkpoint), case_count=len(rows),
        passed_count=sum(row["passed"] for row in rows),
        accepted=all(row["passed"] for row in rows), reference_event_coverage=coverage,
        failed_cases=[row["case"] for row in rows if not row["passed"]],
        worst_max_abs_position=position_error.max(axis=(0, 1)).tolist(),
        worst_max_abs_velocity=velocity_error.max(axis=(0, 1)).tolist(),
        worst_rms_scaled_residual=np.sqrt((residual**2).mean(axis=1)).max(axis=0).tolist(),
        worst_max_scaled_residual=abs(residual).max(axis=(0, 1)).tolist(),
        worst_event_time_error=max(row["event_time_error"] for row in rows),
        matching_events=sum(row["event_agrees"] for row in rows),
        worst_initial_error=float(initial_error.max()),
        worst_independent_transverse_error=float(independent_error.max()),
        neural_batch_seconds=neural_seconds, independent_ode_seconds=ode_seconds,
        ode_over_neural=ode_seconds/neural_seconds, thresholds=tol,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plots(output, names, times, neural, truth)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("dev", "final"), default="final")
    args = parser.parse_args()
    summary = validate(args.checkpoint, args.output, args.split)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["accepted"] else 1)


if __name__ == "__main__":
    main()
