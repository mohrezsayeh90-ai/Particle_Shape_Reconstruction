#!/usr/bin/env python3
"""Train fixed-geometry one-way flow and export variable-injection trajectories."""
import argparse
import copy
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/injection_matplotlib")
import numpy as np
import torch

from particle_pinn.injection import (
    R_IN, R_OUT, RADIUS, HALF_WIDTH, PARAMETER_NAMES, SCOPE,
    InjectionCase, ParameterRanges, FixedGeometryFlow, sample_injections,
    sample_flow_batch, flow_objective, trace_injection, load_injection_model,
)
from particle_pinn.parametric import map_points


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=convert) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def draw_walls(ax):
    q = torch.linspace(0, 1, 401)
    p = FixedGeometryFlow.geometry_parameters(torch.ones((len(q), 1)))
    for eta in (-1.0, 1.0):
        xy = map_points(q, torch.full_like(q, eta), p).numpy()
        ax.plot(xy[:, 0], xy[:, 1], color="0.3", lw=0.8)
    ax.set(aspect="equal", xlabel="x / L0", ylabel="y / L0")


def scenario_cases(ranges, sweep_count, random_count, seed):
    center = InjectionCase(
        re=math.sqrt(np.prod(ranges.re)), diameter=math.sqrt(np.prod(ranges.diameter)),
        density_ratio=math.sqrt(np.prod(ranges.density_ratio)),
        x_p0=float(np.clip(0, *ranges.x_p0)), u_p0=float(np.clip(0, *ranges.u_p0)),
        v_p0=float(np.clip(1, *ranges.v_p0)))
    clearance = HALF_WIDTH - center.diameter / 2 - 1e-8
    center = replace(center, x_p0=float(np.clip(center.x_p0, -clearance, clearance))).validate()
    scenarios = []
    for key in ("x_p0", "u_p0", "v_p0"):
        low, high = getattr(ranges, key)
        if key == "x_p0":
            low, high = max(low, -clearance), min(high, clearance)
        for i, value in enumerate(np.linspace(low, high, sweep_count)):
            scenarios.append((f"{key}_{i:02d}", key, replace(center, **{key: float(value)}).validate()))
    if random_count:
        scenarios += [(f"random_{i:03d}", "random", case) for i, case in enumerate(
            sample_injections(np.random.default_rng(seed), random_count, ranges))]
    return scenarios


def export_results(model, out, ranges, args, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = out / "data"
    data.mkdir(exist_ok=True)
    errors = []
    summaries = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, re in zip(axes[0], (ranges.re[0], math.sqrt(np.prod(ranges.re)), ranges.re[1])):
        q, eta = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(-1, 1, 31), indexing="ij")
        p = model.geometry_parameters(torch.full((q.numel(), 1), re))
        xy = map_points(q.flatten(), eta.flatten(), p).numpy()
        prediction = model.evaluate(xy, re)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Non-finite field in final export.")
        speed = np.linalg.norm(prediction[:, :2], axis=1)
        artist = ax.pcolormesh(xy[:, 0].reshape(q.shape), xy[:, 1].reshape(q.shape),
                              speed.reshape(q.shape), shading="gouraud", cmap="viridis")
        draw_walls(ax)
        fig.colorbar(artist, ax=ax, shrink=0.7, label="Speed / U0")
        ax.set_title(f"Fixed geometry | Re = {re:g}")
        write_csv(data / f"field_re_{re:g}.csv", [dict(x=float(x[0]), y=float(x[1]),
            u=float(y[0]), v=float(y[1]), p=float(y[2])) for x, y in zip(xy, prediction)])
    scenarios = scenario_cases(ranges, args.sweep_count, args.particles, args.seed + 772)
    for name, group, case in scenarios:
        field = lambda points, c=case: model.evaluate(points, c)
        track = trace_injection(field, case, tmax=args.tmax, samples=args.track_samples)
        summary = dict(case=name, group=group, **case.json(), outcome=track["outcome"],
                       duration=float(track["t"][-1]), max_particle_re=track["max_particle_re"],
                       rhs_evaluations=track["rhs_evaluations"], message=track["message"])
        summaries.append(summary)
        write_csv(data / f"trajectory_{name}.csv", [dict(t=float(t), x=float(z[0]), y=float(z[1]),
            u_p=float(z[2]), v_p=float(z[3])) for t, z in zip(track["t"], track["state"])])
        if track["outcome"] == "integration_failed":
            errors.append(dict(stage="trajectory", case=name, error=track["message"]))
        if group != "random":
            ax = axes[1, ("x_p0", "u_p0", "v_p0").index(group)]
            ax.plot(track["state"][:, 0], track["state"][:, 1], lw=1.2,
                    label=f"{getattr(case, group):.3g} ({track['outcome']})")
            ax.scatter(*track["state"][0, :2], s=10)
    for ax, key in zip(axes[1], ("x_p0", "u_p0", "v_p0")):
        draw_walls(ax)
        ax.set_title(f"Vary only {key}")
        ax.legend(title=key, frameon=False, fontsize=6, loc="best")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"injection_summary.{ext}", dpi=250)
    plt.close(fig)
    write_csv(data / "trajectory_summary.csv", summaries)
    write_json(out / "trajectory_summary.json", summaries)
    write_csv(data / "training_history.csv", history)
    if history:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for key in ("total", "continuity", "momentum", "flux"):
            ax.semilogy([h["step"] for h in history], [max(h[key], 1e-15) for h in history], label=key)
        ax.set(xlabel="Optimizer step", ylabel="Training loss")
        ax.legend(frameon=False)
        fig.savefig(out / "training_loss.png", dpi=220)
        plt.close(fig)
    validation = []
    if not args.skip_reference:
        # Independent FEM data are only used AFTER training and model selection.
        from particle_pinn.geometry import Channel
        from particle_pinn.fem import solve_flow
        channel = Channel("serpentine", R_IN, R_OUT)
        points = channel.sample(300, np.random.default_rng(args.seed + 995), margin=0.02)
        for re in (ranges.re[0], math.sqrt(np.prod(ranges.re)), ranges.re[1]):
            try:
                ref = solve_flow(channel, re, ns=args.reference_ns, nn=args.reference_nn, max_iter=150)
                if not ref.diagnostics["solver_ok"]:
                    raise RuntimeError("FEM did not converge: " + str(ref.diagnostics))
                truth = ref.evaluate(points)
                pred = model.evaluate(points, re)
                velocity_error = float(np.linalg.norm(pred[:, :2] - truth[:, :2]) / np.linalg.norm(truth[:, :2]))
                pp, pr = pred[:, 2] - pred[:, 2].mean(), truth[:, 2] - truth[:, 2].mean()
                pressure_error = float(np.linalg.norm(pp - pr) / np.linalg.norm(pr))
                validation.append(dict(re=re, velocity_relative_l2=velocity_error,
                                       pressure_relative_l2=pressure_error, reference=ref.diagnostics))
            except Exception as error:
                errors.append(dict(stage="FEM_validation", re=re, error=str(error)))
    write_json(out / "validation.json", dict(completed=validation, skipped=args.skip_reference,
        note="Single-resolution independent FEM check, not a mesh-convergence study."))
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--hours", type=float, help="Optional training time cap; exports run afterwards")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--check-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--continuity-weight", type=float, default=20.0)
    parser.add_argument("--particles", type=int, default=12, help="Number of random final IVP evaluations")
    parser.add_argument("--sweep-count", type=int, default=5)
    parser.add_argument("--track-samples", type=int, default=260)
    parser.add_argument("--tmax", type=float, default=10.0)
    parser.add_argument("--reference-ns", type=int, default=64)
    parser.add_argument("--reference-nn", type=int, default=14)
    parser.add_argument("--skip-reference", action="store_true")
    for key in PARAMETER_NAMES:
        parser.add_argument("--" + key.replace("_", "-") + "-range", type=float, nargs=2,
                            default=getattr(ParameterRanges(), key))
    args = parser.parse_args()
    if min(args.steps, args.samples, args.width, args.depth, args.check_every, args.sweep_count) < 1:
        parser.error("Training counts and dimensions must be positive.")
    if args.particles < 0 or args.track_samples < 2 or args.reference_ns < 12 or args.reference_nn < 5:
        parser.error("Require particles >= 0, track-samples >= 2, reference-ns >= 12, reference-nn >= 5.")
    if args.hours is not None and (not np.isfinite(args.hours) or args.hours <= 0):
        parser.error("hours must be positive and finite.")
    if not np.isfinite([args.tmax, args.continuity_weight]).all() or min(args.tmax, args.continuity_weight) <= 0:
        parser.error("tmax and continuity-weight must be positive and finite.")
    ranges = ParameterRanges(**{key: tuple(getattr(args, key + "_range")) for key in PARAMETER_NAMES})
    try:
        ranges.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose an empty output directory; --resume can read a checkpoint from a previous directory.")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable.")
    return args, ranges


def main():
    args, ranges = parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = FixedGeometryFlow(args.width, args.depth).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(scope=SCOPE, fixed_geometry=dict(rin=R_IN, rout=R_OUT), ranges=ranges.json(),
        physical_parameters=list(PARAMETER_NAMES), flow_inputs=["x", "y", "Re"],
        particle_initial_position=["x_p0", 0.0], particle_initial_velocity=["u_p0", "v_p0"],
        coupling="one-way", particle_solver="event-aware numerical IVP with Schiller-Naumann drag",
        training="PDE + integral flux, exact velocity boundary ansatz; no interior CFD labels",
        particle_initial_conditions_trained=False,
        units="Original nondimensional L0, U0, L0/U0, mu*U0/L0; not SI")
    history, validation = [], []
    step, best_step, best_score = 0, 0, float("inf")
    best = copy.deepcopy(model.state_dict())
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=True)
        if ck.get("scope") != SCOPE:
            raise ValueError("Old geometry-parametric checkpoints are incompatible; start fresh.")
        for key in ("steps", "samples", "width", "depth", "seed", "continuity_weight", "ranges"):
            if ck["config"][key] != config[key]:
                raise ValueError("Resume configuration mismatch: " + key)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        rng.bit_generator.state = ck["numpy_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if args.device == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in ck["cuda_rng"]])
        step, history, validation = ck["step"], ck["history"], ck["validation"]
        best, best_step, best_score = ck["best_model"], ck["best_step"], ck["best_score"]
        if step >= args.steps:
            raise ValueError("Checkpoint already reached the configured step target.")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "config.json", config)
    probe_xy, probe_re = sample_flow_batch(np.random.default_rng(args.seed + 809),
                                         min(args.samples, 512), ranges.re, args.device)
    started = time.monotonic()
    failures = []

    def select():
        nonlocal best, best_step, best_score
        model.eval()
        result = flow_objective(model, probe_xy.detach().clone().requires_grad_(True), probe_re,
                                args.continuity_weight)
        score = float(result[0].detach())
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite held-out residual.")
        validation.append(dict(step=step, fixed_loss=score))
        if score < best_score:
            best_score, best_step = score, step
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(dict(event="FIXED_PROBE", step=step, loss=score, best_step=best_step)), flush=True)

    def save():
        atomic_save(dict(scope=SCOPE, step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else None,
            config=config, history=history, validation=validation,
            best_model=best, best_score=best_score, best_step=best_step), args.out / "checkpoint.pt")
        atomic_save(dict(scope=SCOPE, step=best_step, model=best, config=config,
            selection="Lowest weighted residual on fixed held-out probe; no FEM data"), args.out / "best_model.pt")
        write_json(args.out / "fixed_validation.json", validation)

    print(json.dumps(dict(event="TRAINING_STARTED", fixed_radii=[R_IN, R_OUT], ranges=ranges.json(),
                         flow_inputs=3, physical_parameters=6, device=args.device)), flush=True)
    try:
        select()
        while step < args.steps:
            if args.hours is not None and time.monotonic() - started >= args.hours * 3600:
                break
            model.train()
            xy, re = sample_flow_batch(rng, args.samples, ranges.re, args.device)
            optimizer.zero_grad(set_to_none=True)
            result = flow_objective(model, xy, re, args.continuity_weight)
            if not torch.isfinite(result[0]):
                raise FloatingPointError("Non-finite training loss.")
            result[0].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient.")
            optimizer.step()
            scheduler.step()
            step += 1
            if step == 1 or step % 100 == 0 or step % args.check_every == 0 or step == args.steps:
                row = dict(step=step, elapsed_seconds=time.monotonic() - started,
                           **dict(zip(("total", "continuity", "momentum", "flux"),
                                      [float(value.detach()) for value in result])),
                           lr=scheduler.get_last_lr()[0])
                history.append(row)
                print(json.dumps(row), flush=True)
            if step % args.check_every == 0:
                select()
                save()
        select()
    except Exception as error:
        failures.append(dict(stage="training", step=step, error=str(error), traceback=traceback.format_exc()))
    training_seconds = time.monotonic() - started
    save()
    if math.isfinite(best_score):
        model.load_state_dict(best)
        model.cpu().eval()
        try:
            failures.extend(export_results(model, args.out, ranges, args, history))
        except Exception as error:
            failures.append(dict(stage="final_export", error=str(error), traceback=traceback.format_exc()))
    write_json(args.out / "failures.json", failures)
    write_json(args.out / "run_complete.json", dict(step=step, selected_step=best_step,
        fixed_loss=best_score if math.isfinite(best_score) else None,
        training_seconds=training_seconds, total_seconds=time.monotonic() - started,
        stop_reason="execution_failure" if failures else "step_target" if step == args.steps else "time_budget",
        failures=len(failures), device=args.device,
        note="Execution completion is not a convergence or accuracy guarantee."))
    print("RESULTS:", str(args.out.resolve()), flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
