# Reproduce the Stokes PINN thesis evidence

Run from the repository root with Python 3.12 and the packages in
[New_Main_Results/requirements.txt](../../New_Main_Results/requirements.txt).
Install the pinned CPU PyTorch wheel before the other dependencies if starting
with a fresh virtual environment.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
python -m pip install -r New_Main_Results/requirements.txt
```

The runner checks the frozen checkpoint SHA256 and verifies that model,
inference, validation and historical formation inputs match research commit
`c5b1223934156ce3a59dc771910d2ab4beb8e2d8`. The output is a versioned folder.
Use a new run ID for new conditions or budgets.

```sh
run=parametric_particle_pinn/results/thesis_figures/stokes_cpu_v2
python -m parametric_particle_pinn.thesis_results --output "$run" --stage validation
python -m parametric_particle_pinn.thesis_results --output "$run" --stage smoke
python -m parametric_particle_pinn.thesis_results --output "$run" --stage core --jobs 3
python -m parametric_particle_pinn.thesis_results --output "$run" --stage extension --jobs 3
python -m parametric_particle_pinn.thesis_results --output "$run" --stage slices
python -m parametric_particle_pinn.thesis_results --output "$run" --stage audit
python -m parametric_particle_pinn.thesis_results --output "$run" --stage plots
```

The validation stage invokes the existing validator on a fresh run directory.
The core stage resumes from recorded case metrics and executes smoke, A, B and
C in order. Use `--stage A`, `B` or `C` to resume a specific phase. The
`--retry-errors` option retries execution errors while retaining normal
preflight and numerical outcomes. Run one controller per output directory.
Rendering reads the saved evidence, including the archived training log. It
does not launch a search, fit weights, or query Kaggle.

## Formation method and saved data

The primary planning engine evaluates the frozen direct ParticlePINN. At each
of 181 candidate ages the learned y map is linear in the signed launch
velocity. The learned x map is affine in the streamwise velocity and shift
invariant before channel boundaries are considered. The search solves those
relations in the candidate bank, uses five streamwise velocity nodes for
free launches, and retains up to eight diverse age/velocity alternatives per
target and port or wall. It scores sampled spacing on the shared formation
clock and improves a seed-dependent assignment for up to eight sweeps. For
small formations it then reuses the established normalized sparse refinement
objective with a 35-evaluation cap. Larger runs use the bounded discrete
search only. These budgets are recorded before search in the manifest. They
are shared by matched trials, with algorithmic differences between fixed and
free release locations documented in the candidate bank. Elapsed time in
`summary.csv` covers the case work after group-shared candidate generation,
when a bank is shared across seeds. Manifest timings cover entire stages.

The replay checks the original target IDs, launch side and port positions,
caps, configured gaps and ages, both methods' completion at the same T,
endpoint and transfer errors, and all-pair spacing. The common-time tolerance
is 1e-10. The 0.0263 search spacing is a design reserve and acceptance uses
0.026. Positive disk overlap means separation below 0.025. Spacing on an
incomplete case is computed only over valid shared presence. Reported
margins in accepted cases cover all pairs through T, with overlap boundaries
and local refinements at both mesh resolutions.

| Evidence | Shape or contents |
| --- | --- |
| `validation/time_series.npz` | 70 × 121 × 6 states per method at common times, plus 70 × 401 display samples per method on its own interval |
| `validation/events.csv` | One row per method per case, including events at release |
| `formation/matrix.json` | Frozen target coordinates, layout and cap, seed, search budget, historical source, target hash and outline where available |
| `formation/<case>/schedule.csv` | Frozen release inputs and assigned target IDs before independent replay |
| `formation/<case>/endpoints.csv` | Physical endpoints, stop events and completion flags |
| `formation/<case>/trajectories.npz` | (N×401×4) arrays in x, y, vx, vy order, shared global time, physical masks, release times, terminal states and target endpoints |
| `formation/<case>/pair_minima_*.csv` | One row per overlapping pair, with global time, both resolutions, sensitivity and termination flags |
| `formation/summary.csv` | Every trial, with search, preflight, replay and acceptance outcomes |
| `figure_catalog.csv` | Figure IDs, files, methods, data sources, case IDs and Persian captions |

The historical rerun coordinates are reused for the triangle and S/C/T; both
the six-port layout and target assignments are frozen in the matrix. All new
release searches are cold starts. The numerical cutoff for ages is 5 for
every case, including the letters originally planned with a cap of 8.
