"""Generate, resume, and plot the thesis evidence without modifying thesis sources."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLBACKEND","Agg")
os.environ.setdefault("MPLCONFIGDIR","/tmp/particle-thesis-mpl")

import matplotlib
import numba
import numpy as np
import scipy
import shapely
import torch

from ..model import load_checkpoint
from .backend import DirectPINN
from .evidence import CHECKPOINT, ROOT, SOURCE, WEIGHT_HASH, extract_validation, sha, write_csv, write_json
from .formation import candidates, run_case
from .matrix import STAGES, cases

THESIS_SOURCE = "9690560dd817fc3bea96007fd4c200a1141db712"
DEFAULT = ROOT / "parametric_particle_pinn/results/thesis_figures/stokes_cpu_v1"
PACKAGE = "parametric_particle_pinn"
PINN_INPUTS = [f"{PACKAGE}/{name}" for name in
               ("model.py","domain.py","inference.py","train.py","validate.py","config.json","Training.ipynb")]
_ENGINE = None
_MODEL = None


def init():
    global _MODEL, _ENGINE
    torch.set_num_threads(1)
    _MODEL = load_checkpoint(CHECKPOINT)
    _ENGINE = DirectPINN(_MODEL)


def index(output, matrix):
    rows = []
    for case in matrix:
        path = output / case["id"] / "metrics.json"
        data = json.loads(path.read_text()) if path.is_file() else {}
        ref = data.get("methods",{}).get("dop853",{})
        pinn = data.get("methods",{}).get("pinn",{})
        rows.append(dict(id=case["id"], stage=case["stage"], study=case["study"], shape=case["shape"],
                         mode=case["mode"], layout=case["layout"], N=case["N"], seed=case["seed"],
                         port_count=len(case["ports"]) if case["mode"]=="fixed" else None,
                         cap=case["cap"], min_age=case["min_age"], max_age=case["max_age"],
                         min_gap=case["min_gap"], legacy_ages_above_horizon=case["legacy_ages_above_horizon"],
                         source=case["source"], target_sha256=case["target_sha256"],
                         status=data.get("status","unattempted"), reasons=";".join(data.get("reasons",[])),
                         elapsed_seconds=data.get("elapsed_seconds"), completed=ref.get("completed"),
                         ref_endpoint_max=ref.get("endpoint_max"), ref_endpoint_rms=ref.get("endpoint_rms"),
                         ref_clearance_margin=ref.get("clearance_margin"), ref_interval=ref.get("interval"),
                         pinn_endpoint_max=pinn.get("endpoint_max"), pinn_clearance_margin=pinn.get("clearance_margin"),
                         transfer_max=data.get("transfer_max"), target_time=data.get("target_time")))
    write_csv(output/"summary.csv",rows)
    return {stage: dict(Counter(r["status"] for r in rows if r["stage"]==stage)) for stage in STAGES}


def group_run(output, group):
    options = None
    if len(group) > 1:
        c = group[0]
        from New_Main_Results.formation.targets import preflight
        xy = np.asarray(c["targets"])
        if not preflight(xy,c["N"],c["ports"],c["cap"])[0]:
            options = candidates(_ENGINE,c,xy)
    for case in group:
        folder = output / case["id"]
        try:
            result = run_case(_ENGINE,_MODEL,case,folder,options)
        except Exception as error:
            folder.mkdir(parents=True,exist_ok=True)
            result = dict(status="error",reasons=[f"{type(error).__name__}: {error}"],case_id=case["id"])
            write_json(folder/"metrics.json",result)
        print(case["id"], result["status"], result.get("reasons"), flush=True)


def run_stage(output, matrix, stage, jobs, retry):
    groups = defaultdict(list)
    for c in matrix:
        exists = output/c["id"]/"metrics.json"
        if c["stage"]!=stage or (exists.is_file() and not (
                retry and json.loads(exists.read_text())["status"]=="error")):
            continue
        signature = json.dumps([c[k] for k in ("target_sha256","ports","cap","mode","budgets",
                                               "min_age","max_age")], sort_keys=True)
        groups[signature].append(c)
    if jobs==1:
        init()
        for group in groups.values():
            group_run(output,group)
            index(output,matrix)
    else:
        with ProcessPoolExecutor(max_workers=jobs,initializer=init) as executor:
            from concurrent.futures import as_completed
            futures = [executor.submit(group_run, output, group) for group in groups.values()]
            for future in as_completed(futures):
                future.result()
                index(output,matrix)


def cpu_quota():
    path = Path("/sys/fs/cgroup/cpu.max")
    return path.read_text().strip() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--stage", choices=("validation","slices","core","plots","audit",*STAGES), required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if sha(CHECKPOINT) != WEIGHT_HASH:
        raise ValueError("Unexpected frozen checkpoint hash")
    subprocess.run(["git","diff","--quiet",SOURCE,"--",*PINN_INPUTS,
                    f"{PACKAGE}/results/cpu","New_Main_Results/formation","New_Main_Results/inputs",
                    "New_Main_Results/runs/formation_20261112_cpu/historical_reruns"],
                   cwd=ROOT,check=True)
    os.chdir(ROOT)
    head = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    metadata = output/"manifest.json"
    init()
    cfg = _MODEL.config
    config_hash = hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    manifest = (json.loads(metadata.read_text()) if metadata.is_file() else
                dict(source_commit=SOURCE,execution_head=head,thesis_main_reviewed=THESIS_SOURCE,
                     checkpoint=str(CHECKPOINT.relative_to(ROOT)),weight_sha256=WEIGHT_HASH,
                     config=cfg,config_canonical_json_sha256=config_hash,
                     trainable_parameters=sum(p.numel() for p in _MODEL.parameters() if p.is_floating_point()),
                     seeds=dict(training=42,development=1201,final=9327,formation=[11,29,47]),
                     platform=platform.platform(),python=sys.version,logical_cpus=os.cpu_count(),
                     environment=dict(numpy=np.__version__,torch=str(torch.__version__),
                                      scipy=scipy.__version__,matplotlib=matplotlib.__version__),
                     dimensionless=True,commands=[],timings={},stages={}))
    assert manifest["weight_sha256"] == WEIGHT_HASH and manifest["config_canonical_json_sha256"] == config_hash
    manifest["environment"].update(numba=numba.__version__, shapely=shapely.__version__)
    manifest["resources"] = dict(formation_threads_per_worker=1, validation_torch_threads=cfg["training"]["threads"],
                                 cpu_quota=cpu_quota())
    manifest["generator_sha256"] = {path.name:sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    manifest["thesis_figure_convention"] = "main, chapters/results.tex: vector PDF sized in LaTeX; captions external"
    formation = output/"formation"
    formation.mkdir(exist_ok=True)
    matrix = cases()
    plan = formation/"matrix.json"
    if plan.is_file():
        assert json.loads(plan.read_text()) == json.loads(json.dumps(matrix)), "Use a new run ID after changing the matrix"
    else:
        write_json(plan,matrix)
        write_csv(formation/"cases_manifest.csv", (
            {k:v for k,v in c.items() if k not in ("targets","outline","target_edges","ports","budgets")}
            for c in matrix))
    started = perf_counter()
    manifest["commands"].append("python -m parametric_particle_pinn.thesis_results "+" ".join(sys.argv[1:]))
    write_json(metadata,manifest)
    if args.stage == "validation":
        from ..validate import validate
        torch.set_num_threads(cfg["training"]["threads"])
        dest = output/"validation"
        if not (dest/"summary.json").is_file():
            if dest.exists():
                raise ValueError(f"Incomplete validation directory at {dest}; move it aside for a fresh validation.")
            with TemporaryDirectory(prefix=".validation-",dir=output) as staging:
                replay = Path(staging)/"validation"
                validate(CHECKPOINT.relative_to(ROOT),replay)
                replay.rename(dest)
        manifest["stages"]["validation"] = extract_validation(_MODEL,dest)
    elif args.stage == "slices":
        from .slices import sweep
        summary = output/"slices"/"summary.json"
        manifest["stages"]["slices"] = (json.loads(summary.read_text()) if summary.is_file() and "elapsed_seconds" in json.loads(summary.read_text())
                                         else sweep(_MODEL,output))
    elif args.stage == "audit":
        from .audit import check
        manifest["stages"]["audit"] = check(_MODEL,_ENGINE,output,matrix)
    elif args.stage == "plots":
        from .plots import generate
        manifest["stages"]["plots"] = generate(output)
    else:
        manifest["stages"]["fast_pinn_check"] = _ENGINE.verify(_MODEL)
        for stage in STAGES[:-1] if args.stage=="core" else (args.stage,):
            run_stage(formation,matrix,stage,args.jobs,args.retry_errors)
    manifest["stages"]["formation"] = index(formation,matrix)
    manifest["timings"][args.stage] = perf_counter()-started
    write_json(metadata,manifest)
    print("Run:",output,flush=True)


if __name__ == "__main__":
    main()
