"""Physics-only Adam and fixed-bank L-BFGS training."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .domain import sample_releases
from .model import CONFIG_PATH, ParticlePINN, particle_state, read_config, scaled_residual


def _bank(model, rng, count):
    config = model.config
    releases = sample_releases(config, rng, count, boundaries=True)
    units = rng.random(count)
    units[::4] **= 3
    units[1::24] = 0
    units[2::24] = 1
    return (model.bounds.new_tensor(units[:, None] * config["domain"]["horizon"]),
            model.bounds.new_tensor(releases))


def _loss(model, bank, *, train):
    state = particle_state(model, *bank, train=train)
    return scaled_residual(model.config, *state).square().mean(dim=0)


def train(config, output, device="cpu"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    settings = config["training"]
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"])
    rng = np.random.default_rng(settings["seed"])
    model = ParticlePINN(config).to(device)
    model.precondition()
    dev = _bank(model, np.random.default_rng(settings["dev_seed"]), 1536)
    metadata = {
        "torch": str(torch.__version__), "device": str(device), "optimizer": ["Adam"],
        "normalization": model.bounds.tolist(),
        "initial_ansatz": "q0 + t*v0 + t^2/(t+s)*learned_correction",
    }
    best = float("inf")
    best_step = 0
    clock = perf_counter()

    def save(filename, step, score):
        data = {"config": config, "model_state": model.state_dict(), "step": step,
                "dev_scaled_mse": score, "training": metadata.copy()}
        temp = output / (filename + ".tmp")
        torch.save(data, temp)
        temp.replace(output / filename)

    def check(phase, step, train_components):
        nonlocal best, best_step
        dev_components = _loss(model, dev, train=False).detach().tolist()
        score = sum(dev_components)
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite development loss.")
        row = dict(phase=phase, step=step, train_scaled_mse=train_components,
                   dev_scaled_mse=dev_components, elapsed_seconds=perf_counter() - clock)
        with (output / "training.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if score < best:
            best, best_step = score, step
            save("best.pt", step, dev_components)
        return dev_components

    components = check("initial", 0, None)
    step = 0
    for phase, phase_steps, rate in (
        ("adam_transverse", settings["transverse_adam_steps"], settings["transverse_adam_rate"]),
        ("adam_horizontal", settings["horizontal_adam_steps"], settings["horizontal_adam_rate"]),
    ):
        component = 1 if phase == "adam_transverse" else 0
        head = model.transverse if component == 1 else model.horizontal
        adam = torch.optim.Adam(head.parameters(), lr=rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            adam, max(phase_steps, 1), eta_min=settings["learning_rate_min"])
        for phase_step in range(1, phase_steps + 1):
            model.zero_grad(set_to_none=True)
            losses = _loss(model, _bank(model, rng, settings["batch_size"]), train=True)
            losses[component].backward()
            adam.step()
            scheduler.step()
            step += 1
            if phase_step % settings["check_every"] == 0 or phase_step == phase_steps:
                components = check(phase, step, losses.detach().tolist())

    total_refinement = settings["lbfgs_steps"]
    if total_refinement:
        metadata["optimizer"].append("horizontal L-BFGS (strong Wolfe)")
        fixed_bank = _bank(model, rng, settings["lbfgs_bank_size"])
        block = settings["lbfgs_block_steps"]
        lbfgs = torch.optim.LBFGS(model.horizontal.parameters(), max_iter=block, max_eval=2 * block,
                                  history_size=50, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            model.zero_grad(set_to_none=True)
            loss = _loss(model, fixed_bank, train=True)[0]
            loss.backward()
            return loss

        adam_total = step
        for start in range(0, total_refinement, block):
            lbfgs.param_groups[0]["max_iter"] = min(block, total_refinement - start)
            previous_step = step
            lbfgs.step(closure)
            state = lbfgs.state[model.horizontal.weight]
            refinement = state["n_iter"]
            step = adam_total + refinement
            metadata["lbfgs_iterations"] = refinement
            metadata["lbfgs_closure_evaluations"] = state["func_evals"]
            components = check("lbfgs", step, _loss(model, fixed_bank, train=False).tolist())
            if step == previous_step:
                break
    save("final.pt", step, components)
    summary = {"training_completed": True, "best_step": best_step, "final_step": step,
               "best_dev_total_scaled_mse": best, "elapsed_seconds": perf_counter() - clock,
               **metadata}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(read_config(args.config), args.output, args.device)


if __name__ == "__main__":
    main()
