# Analytical Stokes–Poiseuille trajectory

The straight-channel benchmark uses the fixed scales and parameters in
[config.json](config.json). Write `tau = rho_ratio * Re * dp² / 18` and
`E = exp(-t/tau)`. With analytical carrier velocity
`(4 Uc (y/H)(1-y/H), 0)` and linear Stokes drag, the vertical equation gives

```text
vy(t) = vy0 E
y(t)  = y0 + tau vy0 (1 - E)
```

For the horizontal component, define `m = (y0 + tau vy0)/H` and
`n = tau vy0/H`. The particle sees `y/H = m - n E`, so the carrier
velocity along its path is `A + B E + C E²`, with

```text
A =  4 Uc m (1 - m)
B =  4 Uc (2 m - 1) n
C = -4 Uc n²
```

Solving `tau dvx/dt + vx = A + B E + C E²` with the integrating factor
`1/E` gives the velocity. Integrating that velocity from the release gives
the position. For `D = vx0 - A + C`, the two results are

```text
vx(t) = A + D E + B (t/tau) E - C E²
x(t)  = x0 + A t
           + D tau (1 - E)
           + B tau [1 - (1 + t/tau) E]
           - (C tau/2) (1 - E²)
```

Both initial velocity components may be nonzero. Vertical motion changes the
horizontal velocity the particle sees, while the vertical equation remains
independent under this carrier flow and drag law.

The formulas give the trajectory on the common physical interval. Stop at the
first wall contact, open-end exit, or supported horizon. Vertical contact can
be solved from the exponential when the corresponding wall is reached within
the interval. Open-end events can use scalar root finding on `x(t)`. Check
the earliest event and the direction of travel at a release on a boundary.

The trained PINN used physics residuals. The published numerical benchmark
compares its output against independent DOP853 integration and checks the
exact transverse solution. The full formula above also provides a reference
without time stepping. The existing runtime comparison is against DOP853.

The full formula was checked against independent DOP853 on 70 releases with
121 times sampled over each reference physical interval. The maximum errors
by component are recorded in the [cross-check results](results/cpu/analytical_crosscheck.json).
