# Parametric particle-trajectory PINN

One saved PyTorch model maps nondimensional `[elapsed_time, x0, y0, vx0, vy0]` to
particle-center position `[xp, yp]`. Automatic differentiation of the same
position map gives velocity and acceleration. The one-way carrier is analytical
plane Poiseuille flow with linear Stokes drag. The fixed parameters are
`L=2.2`, `H=0.30`, `Uc=1`, `Re=10`, `dp=0.025`, and
`rho_p/rho_f=1000`, giving `tau_p=0.3472222222`. Length, velocity, and time
are scaled by the Master3 reference units `L0`, `U0`, and `L0/U0`.

## Supported releases

| Input | Inclusive range |
| --- | --- |
| Elapsed time | 0 to 5, the formation planner's age limit within Master3's trajectory horizon of 20 |
| `x0` | 0 to 2.2 |
| `y0` | 0.0125 to 0.2875 |
| `vx0` | −0.40 to 1.50 |
| Cartesian `vy0` | −1.20 to 1.20 |

The wall adapter sets the lower launcher to `y0=0.01251` with
`vy0 ∈ [0.15, 1.20]`, or the upper launcher to `y0=0.28749` with
`vy0 ∈ [−1.20, −0.15]`. The Cartesian interface also covers near-wall
interior releases moving toward a wall. A center exactly on a wall-contact
boundary can move inward from release. A tangential or outward initial wall
contact stops at time zero. Channel ends are open and use the particle center
for exit detection.

## One-checkpoint inference

Run from the repository root with Python 3.12, PyTorch, NumPy, and SciPy.

```python
import numpy as np
from parametric_particle_pinn.inference import predict_trajectory, wall_release
from parametric_particle_pinn.model import load_checkpoint

model = load_checkpoint("parametric_particle_pinn/results/cpu/best.pt")
releases = np.vstack([
    wall_release(model.config, "lower", 0.439, -0.18, 0.41),
    wall_release(model.config, "upper", 0.751, 0.83, 0.27),
    [0.684, 0.12, 0.29, 0.0],
    [1.531, 0.228, -0.17, 0.65],
])
result = predict_trajectory(model, *releases.T, np.linspace(0, 5, 101))
print(result["event"], result["valid"].sum(axis=1))
```

You can pass a checkpoint path in place of a loaded model. Release arguments
broadcast to the batch shape. Queries can share a 1-D time array or use one
time grid per release. The result contains `elapsed_time` and Boolean
`valid` arrays of shape `(*batch, T)`, two-component `position`, `velocity`
and `acceleration` arrays of shape `(*batch, T, 2)`, and `event` with
`type` and `time` for each release. State is NaN after the first event.
The event types are `lower_wall`, `upper_wall`, `inlet_exit`, `outlet`, and
`time_limit`. The API brackets crossings and local minima on a temporal mesh
and refines candidates by scalar root finding on the neural path. Event
detection is numerical, with the scan step recorded in the configuration.

## Training and validation

[Training.ipynb](Training.ipynb) runs training, demonstrates inference on four
new conditions, and performs independent validation. The repository environment
already has the required packages. For a fresh run from the root:

```sh
python -m parametric_particle_pinn.train --output /tmp/particle-pinn-run
python -m parametric_particle_pinn.validate /tmp/particle-pinn-run/best.pt \
  --output /tmp/particle-pinn-final
```

The hard initial-condition ansatz is `q0 + t*v0 + t²/(t+s)*N(t,c)`. The
conditional network uses smooth Chebyshev time features with trainable linear
readouts. Stokes dynamics make the transverse correction linear in signed
`vy0`. Streamwise translation invariance, centerline reflection symmetry and
the quadratic carrier give four streamwise release features
`[1−z², z*w, w², vx0/Uc]`, where `z=2*y0/H−1` and `w=vy0/1.2`.
All five time coefficients are learned. The fixed QR transform conditions
the time features using the differential operator on a time grid. It uses
neither release trajectories nor reference labels. All derivatives act on
the original nondimensional time, through the log time normalization.

Training uses the component residuals scaled as `tau_p * [rx, ry] / [1, 1.2]`.
Adam first trains the transverse readout, then trains the horizontal readout
with the transverse one fixed, following the one-way dependence of the
equations. Optional horizontal L-BFGS refinement uses one fixed bank across
all closure calls. Collocation samples cover continuous release and time
ranges, faces of the domain, launcher speed limits, and the initial-time
neighborhood. The carrier polynomial is smoothly continued past walls and
open ends for residual training only. Reported physical trajectories stop at
the first event.

Distinct seeds produce training, development, and final-test conditions.
Development residuals choose `best.pt`. The independent validation module
uses high-accuracy DOP853 and the exact transverse solution. It compares
positions and velocities on the intersection of the predicted and reference
physical intervals and scores event types and times separately. Numerical
acceptance limits are in [config.json](config.json).

The [CPU run](results/cpu) includes the best and final checkpoints, training
log and summary, and final-test results. Validation saves per-case errors,
an acceptance summary, trajectory overlays and absolute error plots.

### Verified run

The local CPU run finished at step 4,808 after 4,600 Adam updates and 208
L-BFGS iterations. Development tests passed 32/32 and the held-out final
tests passed 70/70. All five event categories were exercised. The largest
absolute errors were `8.23e-8` in x, `5.58e-10` in y, `1.05e-6` in vx,
`2.85e-10` in vy, and `2.50e-7` in event time. Initial-condition error was
zero. For 70 releases with 121 queries each, this CPU run took 0.313 s in
batched inference and 0.288 s in DOP853, including event detection in both.

The private Kaggle CPU run independently completed and reported 70/70 final
tests passed. Its Output tab contains both checkpoints, per-case metrics,
and both plots. [Kaggle notebook](https://www.kaggle.com/code/mohrezsayeh/parametric-particle-pinn-stokes-poiseuille)

The [analytical solution](ANALYTICAL_SOLUTION.md) gives the full position and
velocity formulas and their DOP853 cross-check. The [exploratory runs](results/exploratory)
preserve earlier configurations, checkpoints and diagnostics. The [Kaggle run
archive](results/kaggle) contains the remote notebook source, log and output
inventory.
