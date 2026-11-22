"""Freeze historical coordinates and the staged formation matrix before search."""
import hashlib
import itertools
import json

import numpy as np

from New_Main_Results.formation.targets import make_ports, mask_shape, target_points
from .evidence import ROOT, read_csv

SEEDS = (11, 29, 47)
LEGACY = ROOT / "New_Main_Results"
STAGES = ("smoke", "A", "B", "C", "extension")


def cases():
    matrix = []
    historical = LEGACY / "runs/formation_20261112_cpu/historical_reruns"
    triangle, = sorted(historical.glob("triangle_*/original_result.json"))
    letters = sorted(historical.glob("letters_*/six_fixed_stations_*/formations/*/result.json"))

    def add(stage, study, shape, n, mode="fixed", m=3, layout="equispaced", cap=None,
            source=None, points=None, edges=None, old_max_age=None, old_ages=None,
            ports=None, center=(1.52, .15), collision=True, q=1., theta=0.):
        base = dict(stage=stage, study=study, shape=shape, N=n, mode=mode, m=m, layout=layout,
                    cap=cap, min_age=.03, max_age=5., min_gap=0., center=center, q=q, theta=theta,
                    collision_aware=collision, old_max_age=old_max_age,
                    legacy_ages_above_horizon=sum(a > 5 for a in old_ages or []),
                    budgets=dict(age_nodes=181, free_velocity_nodes=5, candidates_per_slot=8,
                                 allocation_sweeps=8, nfev=35, dense_refinement=False))
        base["ports"] = make_ports(base) if ports is None else ports
        coordinates = np.asarray(target_points(base) if points is None else points)
        base["targets"] = coordinates.tolist()
        base["target_sha256"] = hashlib.sha256(coordinates.astype("<f8").tobytes()).hexdigest()
        base["target_edges"] = edges
        base["source"] = source or "New_Main_Results.formation.targets"
        shape_name = "stadium" if shape == "small_stadium" else shape
        if shape_name in ("stadium", "fish", "arrow"):
            dims = (.32, .11) if shape == "small_stadium" else (.88, .23)
            base["outline"] = np.asarray(mask_shape(shape_name, center, *dims).exterior.coords).tolist()
        else:
            base["outline"] = None
        for seed in (11,) if stage == "smoke" else SEEDS:
            identity = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()[:10]
            matrix.append(dict(base, seed=seed, id=f"{stage}_{study}_{identity}_s{seed}"))

    for mode in ("fixed", "free"):
        add("smoke", mode, "smoke", 6, mode=mode)
    tri = json.loads(triangle.read_text())
    add("A", "triangle", "triangle", 3, mode="free", points=tri["targets"],
        edges=[[0,1],[1,2],[2,0]], source=str(triangle.relative_to(ROOT)),
        old_ages=[r["age_at_target"] for r in tri["schedules"]])
    for path in letters:
        doc = json.loads(path.read_text())
        add("A", path.parent.name, path.parent.name, 6, cap=1, points=doc["targets"],
            edges=doc["target_edges"], source=str(path.relative_to(ROOT)), old_max_age=8.,
            old_ages=[r["age_at_target"] for r in doc["schedules"]],
            ports=[dict(id=p["station_id"], side=p["side"], x=p["x_launch"]) for p in doc["stations"]])
    for n, mode in itertools.product((24,48,72,100), ("fixed","free")):
        add("B", "density", "stadium", n, mode=mode)
    for folder in ("duck_six_fixed","duck_independent_wall_ports","dense_duck_six_fixed",
                   "dense_duck_independent_wall_ports"):
        path = LEGACY / "inputs/historical" / folder / "release_schedule.csv"
        rows = read_csv(path)
        points = [[float(r["target_x"]), float(r["target_y"])] for r in rows]
        add("B", folder, "duck", len(rows), mode="fixed" if "six_fixed" in folder else "free",
            source=str(path.relative_to(ROOT)), points=points,
            old_ages=[float(r["age_at_target"]) for r in rows])
    for m in range(1,7):
        add("C", "equispaced", "stadium", 24, m=m)
        add("C", "nested", "stadium", 24, m=m, layout="nested")
    add("C", "free", "stadium", 24, mode="free")
    for cap in (3,4,6,None):
        add("C", "capacity", "stadium", 24, cap=cap)
    for x, y in itertools.product((1.15,1.40,1.65), (.115,.15,.185)):
        add("extension", "center", "small_stadium", 18, center=(x,y))
    for q, theta in itertools.product((1.,1.5,2.0), (0.,45.,90.)):
        if q != 1 or theta == 0:
            add("extension", "ellipse", "ellipse", 12, center=(1.40,.15), q=q, theta=theta)
    for reflected in (False,True):
        ports = make_ports(dict(mode="fixed", layout="wall", top_count=2, reflected=reflected))
        add("extension", "reflection", "small_stadium", 18, center=(1.40,.15),
            layout="wall_reflected" if reflected else "wall", ports=ports)
    for collision in (False,True):
        add("extension", "clearance_objective", "small_stadium", 18,
            center=(1.40,.15), collision=collision)
    for shape in ("fish","arrow"):
        add("extension", "shape", shape, 24)
    return matrix
