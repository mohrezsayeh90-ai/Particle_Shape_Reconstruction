# Reproduce the article figure set

From the repository root, use the same Python 3.12 CPU environment as the
[Stokes campaign](../thesis_results/README.md). The default destination is
`parametric_particle_pinn/results/article_figures/stokes_article_v1`.

```sh
OPENBLAS_NUM_THREADS=1 python -m parametric_particle_pinn.article_results
```

Use `--output /tmp/stokes-article` to render in another folder.
`--stage mandatory` renders F06 and F07 and exports their data. The full
command additionally runs the archived audit in an isolated directory,
renders the revised and supplementary figures, and writes the manifest
and bilingual catalog. The Persian review is delivered with the canonical
run. The source archive is addressed relative to the repository root.
In the catalog, archived paths such as `formation/<case>` and `validation/`
are relative to the source archive specified in the manifest. New `data/`
paths are relative to the article output directory. Paths beginning with
`parametric_particle_pinn/` are relative to the repository root.

The DOP853 pair replay reads each frozen schedule, evaluates both trajectories
on their saved overlapping global-time interval, and checks the refined
minimum at the recorded time. The NPZ arrays `time_0` and `distance_0` use
the endpoint-only schedule. Index 1 uses the spacing-aware schedule. The
CSV also records pair IDs, refined point flags, and target times. The
selected-snapshots NPZ uses keys of the form `<case id>__<field>`; state and
time conventions match the archived campaign. Final snapshots and aggregate
counts retain their recorded acceptance status.

PDFs are drawn at 177.8 mm width. P03 also has an 85 mm single-column
layout. PNG previews use 320 DPI.
