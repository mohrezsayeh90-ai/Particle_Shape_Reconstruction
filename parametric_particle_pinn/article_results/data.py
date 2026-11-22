"""Read pinned campaign evidence and reconstruct selected DOP853 pair traces."""
import json
from pathlib import Path

import numpy as np

from ..model import read_config
from ..thesis_results.evidence import ROOT, read_csv, sha, write_csv
from ..validate import reference

SOURCE = ROOT / "parametric_particle_pinn/results/thesis_figures/stokes_cpu_v1"
OUTPUT = SOURCE.parent.parent / "article_figures/stokes_article_v1"
COMMIT = "0ad80a0e1ca89c337665e0a58c0028a92189d73f"
DUCK = ("B_duck_six_fixed_0c2964bf02_s11", "B_duck_independent_wall_ports_85350f38d0_s11")
ABLATION = ("extension_clearance_objective_e125479fde_s11",
            "extension_clearance_objective_475d448df1_s11")
DENSITY = ("B_density_c57b315571_s11", "B_density_6e628ef14f_s11")
LETTERS = ("A_S_6_61d0d57437_s11", "A_C_6_019ad014b8_s11", "A_T_6_11333ea077_s11")
TRIANGLE = "A_triangle_bcb6848497_s11"
GALLERY = ("extension_shape_74ea224c8d_s11", "extension_shape_5b580fd890_s11",
           "B_density_2dd140b6ba_s11")
RADIUS, CLEARANCE, CONTACT_TOLERANCE = .0125, .026, 1e-10


def document(path):
    return json.loads(path.read_text())


class Evidence:
    def __init__(self):
        self.manifest = document(SOURCE/"manifest.json")
        self.config = self.manifest["config"]
        assert self.config == read_config()
        self.matrix = {r["id"]:r for r in document(SOURCE/"formation/matrix.json")}
        self.rows = read_csv(SOURCE/"formation/summary.csv")
        self.by_id = {r["id"]:r for r in self.rows}
        self.used = set()

    def case(self, key):
        self.used.add(key)
        folder = SOURCE/"formation"/key
        metrics = document(folder/"metrics.json")
        endpoints = read_csv(folder/"endpoints.csv")
        schedule = read_csv(folder/"schedule.csv")
        with np.load(folder/"trajectories.npz") as source:
            arrays = {k:source[k] for k in source.files}
        cfg = self.matrix[key]
        count = cfg["N"]
        assert [int(r["target_id"]) for r in endpoints] == list(range(1,count+1))
        assert [int(r["target_id"]) for r in schedule] == list(range(1,count+1))
        np.testing.assert_array_equal(arrays["target"], cfg["targets"])
        np.testing.assert_array_equal(arrays["dop853_endpoint"],
                                      [[float(r[k]) for k in ("dop853_x","dop853_y")] for r in endpoints])
        for method in ("pinn","dop853"):
            summary = metrics["methods"][method]
            pairs = read_csv(folder/f"pair_minima_{method}.csv")
            assert len(pairs)==count*(count-1)//2
            lowest = min(pairs,key=lambda row:float(row["distance"]))
            assert float(lowest["distance"])==summary["minimum"]["distance"]
        return cfg,metrics,arrays,schedule

    def trace(self,key):
        _,metrics,_,schedule = self.case(key)
        minimum = metrics["methods"]["dop853"]["minimum"]
        times = np.unique(np.r_[np.linspace(minimum["overlap_start"],minimum["overlap_end"],601),
                                minimum["time"]])
        subpath = []
        for target in (minimum["i"],minimum["j"]):
            row = schedule[target-1]
            release = np.array([float(row[k]) for k in ("x0","y0","vx0","vy0")])
            _,end,solver = reference(self.config,release)
            ages = times-float(row["release_time"])
            assert ages.min()>=-1e-10 and ages.max()<=end+1e-10
            subpath.append(solver(ages.clip(0,end))[:,:2])
        distance = np.linalg.norm(subpath[0]-subpath[1],axis=1)
        k = np.searchsorted(times,minimum["time"])
        np.testing.assert_allclose(distance[k],minimum["distance"],rtol=0,atol=1e-10)
        return times,distance,minimum,metrics["target_time"]

    def summary(self,rows,path):
        write_csv(path, (dict(**r,dop853_minimum_distance=(float(r["ref_clearance_margin"])+CLEARANCE
                   if r["ref_clearance_margin"] else ""),attempted=r["status"]!="preflight_rejected",
                   disk_overlap=(float(r["ref_clearance_margin"])+CLEARANCE<2*RADIUS-CONTACT_TOLERANCE
                   if r["ref_clearance_margin"] else "")) for r in rows))

    def source_hashes(self):
        keys = ("metrics.json","schedule.csv","endpoints.csv","trajectories.npz",
                "pair_minima_pinn.csv","pair_minima_dop853.csv")
        paths = [SOURCE/rel for rel in (
            "formation/summary.csv","formation/matrix.json","validation/time_series.npz",
            "validation/summary.json","validation/case_diagnostics.csv","validation/error_percentiles.csv",
            "slices/grid.csv","figure_catalog.csv","RESULTS_FA.md")]
        paths.extend(SOURCE/"formation"/case/name for case in sorted(self.used) for name in keys)
        return {str(path.relative_to(ROOT)):sha(path) for path in paths}
