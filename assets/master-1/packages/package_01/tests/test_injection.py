"""One-way independence, initial data, conservation, and event regression tests."""
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from particle_pinn.injection import (
    HALF_WIDTH, LENGTH, SCOPE, InjectionCase, ParameterRanges, FixedGeometryFlow,
    sample_injections, sample_flow_batch, flow_objective, channel_coordinates,
    trace_injection, load_injection_model,
)
from particle_pinn.parametric import map_points

torch.set_num_threads(1)


class InjectionTests(unittest.TestCase):
    def test_six_parameter_sampling_preserves_sign_and_sphere_clearance(self):
        cases = sample_injections(np.random.default_rng(27), 4000, ParameterRanges())
        values = np.stack([case.array() for case in cases])
        self.assertEqual(values.shape, (4000, 6))
        self.assertTrue(np.all(abs(values[:, 3]) + values[:, 1] / 2 < HALF_WIDTH))
        for i, key in enumerate(("re", "diameter", "density_ratio", "x_p0", "u_p0", "v_p0")):
            low, high = getattr(ParameterRanges(), key)
            self.assertGreaterEqual(values[:, i].min(), low)
            self.assertLessEqual(values[:, i].max(), high)
            self.assertGreater(np.ptp(values[:, i]), 0.8 * (high - low))
        self.assertLess(values[:, 4].min(), 0)
        self.assertGreater(values[:, 4].max(), 0)

    def test_fixed_ranges_and_impossible_combinations(self):
        ranges = ParameterRanges(re=(1, 1), diameter=(.025, .025), density_ratio=(1000, 1000),
                                 x_p0=(.03, .03), u_p0=(0, 0), v_p0=(0, 0))
        for case in sample_injections(np.random.default_rng(8), 10, ranges):
            np.testing.assert_allclose(case.array(), [1, .025, 1000, .03, 0, 0])
        with self.assertRaises(ValueError):
            ParameterRanges(x_p0=(.1, .14)).validate()
        with self.assertRaises(ValueError):
            InjectionCase(x_p0=.14, diameter=.025).validate()

    def test_actual_mlp_has_three_inputs_and_exact_one_way_independence(self):
        model = FixedGeometryFlow(16, 2)
        self.assertEqual(model.net[0].in_features, 3)
        with torch.no_grad():
            model.net[-1].weight.normal_(0, .1)
        points = np.array([[.1, .05], [.3, .25]])
        a = InjectionCase()
        b = replace(a, x_p0=.06, u_p0=-.3, v_p0=.4, diameter=.01, density_ratio=200)
        np.testing.assert_array_equal(model.evaluate(points, a), model.evaluate(points, b))
        self.assertFalse(np.allclose(model.evaluate(points, a), model.evaluate(points, replace(a, re=5))))

    def test_boundary_conditions_gauge_and_pde_backward(self):
        model = FixedGeometryFlow(16, 2)
        with torch.no_grad():
            model.net[-1].weight.normal_(0, .1)
        q = torch.linspace(0, 1, 50)
        re = torch.logspace(-1, 1, len(q))[:, None]
        p = model.geometry_parameters(re)
        for side in (-1, 1):
            uv = model(map_points(q, torch.full_like(q, side), p), re)[:, :2]
            torch.testing.assert_close(uv, torch.zeros_like(uv), atol=2e-5, rtol=0)
        eta = torch.linspace(-.95, .95, len(q))
        for end in (0, 1):
            uv = model(map_points(torch.full_like(q, end), eta, p), re)[:, :2]
            torch.testing.assert_close(uv[:, 0], torch.zeros_like(q), atol=2e-5, rtol=0)
            torch.testing.assert_close(uv[:, 1], 1 - eta.square(), atol=2e-5, rtol=1e-5)
        gauge = model(torch.tensor([[1.4, 0.]]), torch.ones((1, 1)))[0, 2]
        self.assertLess(abs(float(gauge.detach())), 1e-3)
        xy, re = sample_flow_batch(np.random.default_rng(6), 32, (.1, 10))
        loss = flow_objective(model, xy, re)[0]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_analytic_coordinates_roundtrip_and_signed_cap_extension(self):
        q = torch.linspace(.001, .999, 200, dtype=torch.float64)
        eta = torch.linspace(-.9, .9, len(q), dtype=torch.float64)
        p = FixedGeometryFlow.geometry_parameters(torch.ones((len(q), 1), dtype=torch.float64))
        xy = map_points(q, eta, p).numpy()
        s, actual_eta = channel_coordinates(xy)
        np.testing.assert_allclose(s / LENGTH, q.numpy(), atol=1e-14)
        np.testing.assert_allclose(actual_eta, eta.numpy(), atol=1e-14)
        s, _ = channel_coordinates([[0, -.01], [1.4, .01]])
        np.testing.assert_allclose(s, [-.01, LENGTH + .01], atol=1e-14)

    def test_stokes_ivp_matches_exact_relaxation_with_both_initial_velocities(self):
        fluid = np.array([.2, .8])
        field = lambda points: np.tile([*fluid, 0.0], (len(points), 1))
        case = InjectionCase(x_p0=.01, u_p0=-.1, v_p0=.3)
        track = trace_injection(field, case, tmax=.02, drag="stokes")
        self.assertEqual(track["outcome"], "time_limit")
        t = track["t"][:, None]
        initial_velocity = np.array([case.u_p0, case.v_p0])
        tau = case.density_ratio * case.re * case.diameter ** 2 / 18
        velocity = fluid + (initial_velocity - fluid) * np.exp(-t / tau)
        position = [case.x_p0, 0] + t * fluid + tau * (initial_velocity - fluid) * (1 - np.exp(-t / tau))
        np.testing.assert_allclose(track["state"][:, :2], position, atol=2e-8)
        np.testing.assert_allclose(track["state"][:, 2:], velocity, atol=2e-7)
        np.testing.assert_array_equal(track["state"][0], [case.x_p0, 0, case.u_p0, case.v_p0])

    def test_initial_rest_and_backflow_events(self):
        field = lambda points: np.tile([0., 1., 0.], (len(points), 1))
        resting = trace_injection(field, InjectionCase(v_p0=0), tmax=.02)
        self.assertEqual(resting["outcome"], "time_limit")
        self.assertGreater(resting["state"][-1, 1], 0)
        reverse = trace_injection(field, InjectionCase(v_p0=-.2), tmax=.02)
        self.assertEqual(reverse["outcome"], "backflow_exit")
        self.assertLess(reverse["t"][-1], 1e-8)

    def test_wall_contact_preserves_real_initial_data(self):
        field = lambda points: np.tile([1., .1, 0.], (len(points), 1))
        case = InjectionCase(x_p0=.12, u_p0=1., v_p0=.1)
        track = trace_injection(field, case, tmax=.1)
        self.assertEqual(track["outcome"], "stuck")
        _, eta = channel_coordinates(track["state"][-1:, :2])
        self.assertAlmostEqual(HALF_WIDTH * (1 - abs(eta[0])), case.diameter / 2, places=7)
        np.testing.assert_array_equal(track["state"][0], [.12, 0., 1., .1])

    def test_weights_only_checkpoint_roundtrip_and_old_scope_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "model.pt"
            model = FixedGeometryFlow(16, 2)
            torch.save(dict(scope=SCOPE, config=dict(width=16, depth=2), model=model.state_dict()), file)
            loaded, _ = load_injection_model(file)
            np.testing.assert_array_equal(model.evaluate([[.1, .05]], 1), loaded.evaluate([[.1, .05]], 1))
            torch.save(dict(scope="parametric-s-channel-v1"), file)
            with self.assertRaises(ValueError):
                load_injection_model(file)


if __name__ == "__main__":
    unittest.main()
