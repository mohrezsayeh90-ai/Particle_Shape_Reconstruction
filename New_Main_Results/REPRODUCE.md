# Reproduce or resume the numerical campaign

This standalone bundle is mirrored byte-for-byte between the research and
proposal repositories. No proposal prose, bibliography, template or thesis PDF
is edited by these commands. Archived inputs are read-only; fresh replay data
and new experiments are saved under `runs/formation_20261112_cpu/`.

Use Python 3.12 and create an environment outside the bundle:

```sh
python -m venv /tmp/formation-env
/tmp/formation-env/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/formation-env/bin/pip install numpy scipy numba shapely matplotlib pandas Pillow arabic-reshaper python-bidi ipython
# Or, after installing the official CPU wheel, install the exact used versions:
/tmp/formation-env/bin/pip install -r requirements.txt
```

The exact versions used here are recorded in the run's `runtime.json`.
The frozen checkpoint is included, loaded with `weights_only=True`, and checked
against SHA256 `af2f21b195e2d25b877482cd527110054f42bf92200f64e39572375219e5e581`.
The Numba planning evaluator is checked against the original float32 PyTorch
forward. Final acceptance always uses PyTorch/RK45 and independent analytical
Poiseuille/DOP853 integrations, with all-pair shared-clock minimum refinement.

From this directory:

```sh
python run_formation_campaign.py --stage prepare
python run_formation_campaign.py --stage check
python run_formation_campaign.py --stage history --resume
python run_formation_campaign.py --stage smoke --workers 3 --resume
python run_formation_campaign.py --stage core --workers 4 --resume
python run_formation_campaign.py --stage extended --workers 4 --resume
python run_formation_campaign.py --stage reload
python plot_formation_campaign.py
python run_robustness.py
python refresh_outputs.py
python save_diagnostics.py
python export_perturbation_inputs.py
python finalize_results.py
```

`finalize_results.py` is intended after the selected workers finish. It refreshes
historical rerun replays, conditional states and all indexes, saves critical-pair
events, renders figures, checks saved data and figure totals, and writes
`SHA256SUMS`. It runs no optimizer. A comparison-table row keeps duration,
emission gap, collision objective, layout and cap separate, so distinct settings
cannot be pooled into one three-seed comparison.

The first N=100, seed-11 D1 screening batch used scalar candidate pair costs.
Later runs use the same cost in a vectorized evaluator; the numerical selfcheck
compares both implementations. The scalar backend is preserved for reproducing
that batch and recorded in `execution_provenance.csv`. Both use the same
24/50/35 evaluation caps. Final acceptance is unchanged.

`run_robustness.py` freezes representative low/high-margin selections, then runs
64 draws in each predeclared completed scenario at fixed nominal T. The
postprocessor `export_perturbation_inputs.py` reconstructs the exact seeded
inputs, labels their reconstruction provenance, checks all 64 saved bounds
flags, and repeats the first valid reference replay in each scenario. It never
replaces the original trial metrics. Per-scenario input arrays, RNG seeds,
bounds-conditional fractions, invalid draws and replay checks are portable.

`--group`, `--N`, `--seed` and `--limit` select pending trials without changing
the 24/50/35 optimizer caps. The complete matrix remains in `cases.json` and
`cases_manifest.csv`. Three seeds (11, 29, 47) are recorded independently;
individual starts are not measurement uncertainty. A matched comparison is
complete only after all three planned starts have results at the same budget.
Configured capacities are enforced in allocation and swaps, and independently
checked in saved-schedule verification. Free locations have no configured port
count. Infeasible-before-solve, running, unexecuted and attempted failures remain
distinct. No failed finite search establishes physical impossibility.

The pickle files in `runs/*/cache` are disposable local seed/path caches; they
are not scientific inputs and are excluded from Git and from the portable
mirror. Delete them only when recomputation is desired. Every published case
retains its settings, targets, ports, final schedule, endpoints, metrics and
trajectory arrays. Plotting reads these files and never invokes the optimizer.

Historical triangle/S/C/T payloads were absent from the source archive and a
fresh request to the public Kaggle page was denied HTTP 403 by the proxy.
The original manifest and logs are preserved under `inputs/historical/`.
To run the selected original definitions without executing training cells:

```sh
python rerun_historical.py --case triangle
python rerun_historical.py --case letters
```

Those commands preserve original historical search settings, including the
six-station letter duration bound of 8 rather than the new matched bound of 5.
They refuse to overwrite existing reruns. Their outputs are separately labelled
and must not be described as recovered original Kaggle artifacts.
