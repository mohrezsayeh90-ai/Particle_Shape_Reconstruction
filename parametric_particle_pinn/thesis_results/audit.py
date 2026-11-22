"""Cross-check the recorded release clock, event masks and acceptance evidence."""
from collections import Counter
import json

import numpy as np

from ..inference import predict_trajectory
from .evidence import exact_state, read_csv, write_json
from .formation import CLEARANCE


def check(model, engine, output, matrix):
    validation = output/"validation"
    with np.load(validation/"time_series.npz") as data:
        assert data["common_time"].shape == (70,121)
        assert data["common_valid"].all()
    events = read_csv(validation/"events.csv")
    assert len(events)==140 and any(float(r["elapsed"])==0 for r in events)
    assert len({(r["case"],r["method"]) for r in events}) == 140
    assert json.loads((validation/"summary.json").read_text())["passed_count"] == 70

    attempts = read_csv(output/"formation/summary.csv")
    definitions = {r["id"]:r for r in matrix}
    assert {r["id"] for r in attempts} == set(definitions)
    count = Counter(r["status"] for r in attempts)
    if count["error"] or count["unattempted"]:
        raise ValueError(f"Unresolved formation records: {dict(count)}")
    exact_max = 0.
    position_diff = 0.
    direct_check = 0.
    refinement_max = 0.
    for row in attempts:
        definition = definitions[row["id"]]
        folder = output/"formation"/row["id"]
        metric = json.loads((folder/"metrics.json").read_text())
        assert row["status"] == metric["status"]
        if metric["status"] in ("preflight_rejected","search_failed"):
            assert not (folder/"schedule.csv").exists()
            continue
        schedule = read_csv(folder/"schedule.csv")
        endpoints = read_csv(folder/"endpoints.csv")
        n = definition["N"]
        assert [int(r["target_id"]) for r in schedule] == list(range(1,n+1))
        assert [int(r["target_id"]) for r in endpoints] == list(range(1,n+1))
        with np.load(folder/"trajectories.npz") as data:
            raw = np.array([[float(r[k]) for k in ("x0","y0","vx0","vy0")] for r in schedule])
            np.testing.assert_array_equal(raw,data["release"])
            np.testing.assert_array_equal(data["target"],definition["targets"])
            np.testing.assert_allclose(data["release_time"]+data["age_at_target"],metric["target_time"],
                                       rtol=0,atol=1e-12)
            for label in ("pinn","dop853"):
                valid = data[f"{label}_valid"]
                values = data[label]
                assert np.isfinite(values[valid]).all() and np.isnan(values[~valid]).all()
                pairs = read_csv(folder/f"pair_minima_{label}.csv")
                actual = metric["methods"][label]
                if actual["interval"]=="full_to_T":
                    assert len(pairs)==n*(n-1)//2
                if pairs:
                    closest = min(pairs,key=lambda r:float(r["distance"]))
                    assert float(closest["distance"]) == actual["minimum"]["distance"]
                    assert float(closest["time"]) == actual["minimum"]["time"]
                    refinement_max = max(refinement_max,actual["maximum_refined_resolution_delta"])
                    pair = np.array([int(closest[k])-1 for k in ("i","j")])
                    time = float(closest["time"])-data["release_time"][pair]
                    if label=="pinn":
                        xy = engine.position(raw[pair],time)
                    else:
                        xy = exact_state(model.config,raw[pair],time[:,None])[:,0,:2]
                    direct_check = max(direct_check,abs(float(closest["distance"])-np.linalg.norm(xy[0]-xy[1])))
                if metric["status"]=="accepted":
                    assert actual["interval"]=="full_to_T" and actual["completed"]==n
                    assert actual["minimum"]["distance"]>=CLEARANCE
            if row["stage"]=="smoke":
                pred = predict_trajectory(model,*raw.T,data["age_at_target"][:,None])
                assert pred["valid"].all()
                position_diff = max(position_diff,abs(pred["position"][:,0]-data["pinn_endpoint"]).max())
            exact_max = max(exact_max,metric["exact_vs_dop853_max"] or 0.)
    # Forward velocity of the batched fast evaluation, away from events.
    releases = np.array([engine.wall(0,.55,.34,.35),engine.wall(1,1.10,.62,.30)])
    times = np.array([.13,.26])
    step = 1e-6
    finite = (engine.position(releases,times+step)-engine.position(releases,times-step))/(2*step)
    derivatives = engine.position(releases,times,velocity=True)[:,2:]
    finite_diff = abs(derivatives-finite).max()
    assert max(direct_check,position_diff) < 1e-9 and finite_diff < 1e-8
    result = dict(formation_status_counts=dict(count), saved_schedule_and_mask_checks=True,
                  matched_core_and_extension_records=len(attempts), expected_final_cases=70,
                  max_critical_pair_direct_check=direct_check,
                  max_refined_resolution_sensitivity=refinement_max,
                  max_smoke_endpoint_fast_vs_public=position_diff,
                  max_fast_velocity_finite_difference_check=finite_diff,
                  max_closed_form_dop853_formation_endpoint_error=exact_max)
    write_json(output/"audit.json",result)
    return result
