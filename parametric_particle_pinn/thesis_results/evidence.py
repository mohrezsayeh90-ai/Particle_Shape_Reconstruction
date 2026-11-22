"""Plots consume frozen, dimensionless evidence from the validation protocol."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ..inference import predict_trajectory
from ..model import relaxation_time
from ..validate import conditions, reference

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "parametric_particle_pinn/results/cpu"
CHECKPOINT = ARCHIVE / "best.pt"
WEIGHT_HASH = "29d4d0a4784330a2543f83ade8bc0bb1f3f5c6085950b62ed4156bf293d1e51a"
SOURCE = "c5b1223934156ce3a59dc771910d2ab4beb8e2d8"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    tmp = path.with_suffix(".tmp")
    rows = list(rows)
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def acceleration(config, state):
    p = config["physics"]
    height = state[..., 1] / p["height"]
    u = 4 * p["centerline_speed"] * height * (1 - height)
    return (np.stack((u, np.zeros_like(u)), axis=-1) - state[..., 2:]) / relaxation_time(config)


def exact_state(config, release, time):
    x, y, vx, vy = np.moveaxis(np.asarray(release),-1,0)[...,None]
    p = config["physics"]
    tau = relaxation_time(config)
    s = np.asarray(time) / tau
    e = np.exp(-s)
    k = -np.expm1(-s)
    n = tau * vy / p["height"]
    m = y / p["height"] + n
    speed = 4 * p["centerline_speed"]
    a, b, c = speed * m * (1-m), speed * (2*m-1)*n, -speed * n*n
    d = vx-a+c
    return np.stack((x+a*time+d*tau*k+b*tau*(k-s*e)+c*tau/2*np.expm1(-2*s),
                     y+tau*vy*k, a+d*e+b*s*e-c*e*e, vy*e), axis=-1)


def family(config, release):
    p = config["physics"]
    gap = p["diameter"]/2 + p["launch_gap"]
    low, high = config["domain"]["inward_speed"]
    if release[1] == gap and low <= release[3] <= high:
        return "lower_launcher"
    if release[1] == p["height"]-gap and low <= -release[3] <= high:
        return "upper_launcher"
    if release[1] in (p["diameter"]/2, p["height"]-p["diameter"]/2):
        return "contact_line"
    return "interior"


def extract_validation(model, output):
    cfg = model.config
    names, releases = zip(*conditions(cfg, "final"))
    archived, rerun = (read_csv(path / "cases.csv") for path in (ARCHIVE / "validation", output))
    assert [r["case"] for r in archived] == [r["case"] for r in rerun] == list(names)
    archived_summary = json.loads((ARCHIVE / "validation/summary.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    for key in ("passed_count", "matching_events", "reference_event_coverage",
                "worst_max_abs_position", "worst_max_abs_velocity", "worst_event_time_error",
                "worst_initial_error", "worst_max_scaled_residual"):
        if isinstance(summary[key], (list, float, int)) and key != "reference_event_coverage":
            np.testing.assert_allclose(summary[key], archived_summary[key], rtol=1e-5, atol=1e-12)
        else:
            assert summary[key] == archived_summary[key]
    releases = np.array(releases)
    refs = [reference(cfg, row) for row in releases]
    ref_end = np.array([item[1] for item in refs])
    events = predict_trajectory(model, *releases.T, [0.0])["event"]
    pin_end = events["time"]
    unit = np.linspace(0, 1, cfg["validation"]["samples_per_case"])
    common = np.minimum(pin_end, ref_end)[:, None] * unit
    pred = predict_trajectory(model, *releases.T, common)
    assert pred["valid"].all()
    neural = np.concatenate([pred["position"], pred["velocity"], pred["acceleration"]], axis=-1)
    ref = np.stack([item[2](t) for item, t in zip(refs, common)])
    closed = np.stack([exact_state(cfg, row, t) for row, t in zip(releases, common)])
    full_ref = np.stack([item[2](item[1]*unit) for item in refs])
    cross = np.stack([exact_state(cfg, row, end*unit) for row, end in zip(releases, ref_end)])
    exact_error = abs(full_ref-cross).max(axis=(0, 1))
    archived_exact = json.loads((ARCHIVE / "analytical_crosscheck.json").read_text())
    np.testing.assert_allclose(exact_error, archived_exact["max_absolute_error"], rtol=0.05, atol=3e-14)
    ref = np.concatenate((ref, acceleration(cfg, ref)), axis=-1)
    closed = np.concatenate((closed, acceleration(cfg, closed)), axis=-1)
    error = neural-ref
    residual = relaxation_time(cfg)*(neural[..., 4:]-acceleration(cfg, neural[..., :4]))
    residual /= cfg["training"]["residual_velocity_scale"]
    display = np.linspace(0, 1, 401)
    pin_times, dop_times = pin_end[:, None]*display, ref_end[:, None]*display
    display_prediction = predict_trajectory(model, *releases.T, pin_times)
    families = np.array([family(cfg, row) for row in releases])
    for i, row in enumerate(rerun):
        assert row["passed"] == archived[i]["passed"]
        for j, component in enumerate(("x", "y", "vx", "vy")):
            for stat, value in (("max", abs(error[i, :, j]).max()),
                                ("rms", np.sqrt((error[i, :, j]**2).mean()))):
                np.testing.assert_allclose(value, float(row[f"{stat}_{component}_error"]),
                                           rtol=1e-5, atol=1e-12)
    np.savez_compressed(output / "time_series.npz", case_id=names, family=families,
                        release=releases, common_time=common, common_valid=pred["valid"],
                        pinn=neural, dop853=ref, closed_form=closed, signed_error=error,
                        signed_scaled_residual=residual, pinn_display_time=pin_times,
                        dop853_display_time=dop_times, pinn_display_position=display_prediction["position"],
                        pinn_display_valid=display_prediction["valid"],
                        dop853_display_position=np.stack([item[2](t)[:, :2] for item, t in zip(refs, dop_times)]))
    write_csv(output / "events.csv", (
        dict(case=name, family=families[i], method=method, event=kind, elapsed=end,
             x=point[0], y=point[1])
        for i, name in enumerate(names)
        for method, kind, end, point in (
            ("PINN", events["type"][i], pin_end[i], display_prediction["position"][i, -1]),
            ("DOP853", refs[i][0], ref_end[i], full_ref[i, -1]))))
    write_csv(output / "case_diagnostics.csv", (
        dict(**row, family=families[i], bank="specified" if i < 22 else "seeded",
             signed_event_error=pin_end[i]-ref_end[i],
             max_ax_error=abs(error[i, :, 4]).max(), max_ay_error=abs(error[i, :, 5]).max(),
             rms_ax_error=np.sqrt((error[i, :, 4]**2).mean()),
             rms_ay_error=np.sqrt((error[i, :, 5]**2).mean()),
             failure_reason="" if row["passed"] == "True" else "configured acceptance threshold")
        for i, row in enumerate(rerun)))
    write_csv(output / "error_percentiles.csv", (
        dict(component=component, n=len(values), zero_count=np.count_nonzero(values == 0),
             median=np.median(values), p95=np.quantile(values, .95), maximum=max(values))
        for component, values in zip(("x","y","vx","vy","ax","ay","event_time"), [
            *abs(error).max(axis=1).T, abs(pin_end-ref_end)])))
    write_json(output / "array_schema.json", dict(
        units="x,y / L0; time * U0/L0; velocities / U0; acceleration * L0/U0^2",
        release_columns=["x0", "y0", "vx0", "vy0"],
        state_columns=["x", "y", "vx", "vy", "ax", "ay"],
        common_shape=list(neural.shape), per_method_display_shape=list(pin_times.shape),
        residual_columns=["tau*rx/1", "tau*ry/1.2"], exact_vs_dop853_max=exact_error.tolist(),
        event_convention="one row per method per case, including zero-duration cases",
        acceleration_reference="new diagnostic from Stokes RHS applied to reference state",
        archived_case_inventory_match=True))
    return summary
