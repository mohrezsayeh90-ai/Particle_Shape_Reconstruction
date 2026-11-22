"""CPU evaluation of the frozen ParticlePINN with event-aware paths."""
import numpy as np
from scipy.optimize import brentq

from ..domain import wall_release
from ..inference import predict_trajectory
from ..validate import conditions


class DirectPINN:
    def __init__(self, model):
        self.config = model.config
        self.physics = self.config["physics"]
        self.domain = self.config["domain"]
        self.scale = self.config["model"]["time_feature_scale"]
        self.freq = model.frequencies.numpy()
        self.weights = model.basis_transform.numpy() @ np.concatenate(
            (model.horizontal.weight.numpy().T, model.transverse.weight.numpy().T), axis=1)
        self.horizon = self.domain["horizon"]
        self.mesh = np.linspace(0, self.horizon, 1 + int(np.ceil(
            self.horizon/self.config["validation"]["event_scan_step"])))
        self.mesh_basis = self.basis(self.mesh, velocity=True)

    def basis(self, t, *, velocity=False):
        t = np.asarray(t)
        s = self.scale
        scale = 1.998 / np.log1p(self.horizon/s)
        z = scale*np.log1p(t/s)-.999
        angle = np.arccos(z)[..., None]*self.freq
        envelope = t*t/(t+s)
        cosine = np.cos(angle)
        value = (envelope[..., None]*cosine) @ self.weights
        if not velocity:
            return value
        dz = scale/(t+s)
        derivative = (t*(t+2*s)/(t+s)**2)[..., None]*cosine
        derivative += (envelope*dz/np.sqrt(1-z*z))[..., None]*np.sin(angle)*self.freq
        return value, derivative @ self.weights

    def position(self, release, t, basis=None, *, velocity=False):
        t = np.asarray(t)
        release = np.asarray(release)
        basis = self.basis(t, velocity=velocity) if basis is None else basis
        z = 2*release[..., 1]/self.physics["height"]-1
        w = release[..., 3]/self.domain["interior_vy0"][1]
        first = np.stack((1-z*z, z*w, w*w, release[..., 2]/self.physics["centerline_speed"]), axis=-1)

        def correction(coefficients):
            u = (coefficients[..., :4]*first).sum(axis=-1)
            return np.stack((u, coefficients[..., 4]*w), axis=-1)

        q = release[..., :2] + t[..., None]*release[..., 2:]
        if velocity:
            return np.concatenate((q+correction(basis[0]), release[..., 2:]+correction(basis[1])), axis=-1)
        return q+correction(basis)

    def event(self, release):
        r = self.physics["diameter"]/2
        end = self.horizon
        status = "time_limit"
        data = self.position(release, self.mesh, self.mesh_basis, velocity=True)
        for label, axis, bound, sign in (
            ("lower_wall", 1, r, 1), ("upper_wall", 1, self.physics["height"]-r, -1),
            ("inlet_exit", 0, 0, 1), ("outlet", 0, self.physics["length"], -1),
        ):
            face = sign*(data[:, axis]-bound)
            rate = sign*data[:, axis+2]
            if face[0] == 0 and (rate[0] < 0 or (axis == 1 and rate[0] == 0)):
                return label, 0.
            crossing = (face[:-1] >= 0) & (face[1:] < 0)
            minima = (rate[:-1] < 0) & (rate[1:] >= 0)

            def distance(t):
                return sign*(self.position(release, t)[axis]-bound)

            for i in np.flatnonzero(crossing | minima):
                lo, hi = self.mesh[i:i+2]
                if lo >= end:
                    break
                if crossing[i]:
                    root = brentq(distance, lo, hi, xtol=1e-12)
                else:
                    t = brentq(lambda t: sign*self.position(release, t, velocity=True)[axis+2],
                               lo, hi, xtol=1e-12)
                    clearance = distance(t)
                    if axis == 1 and abs(clearance) <= 1e-12:
                        root = t
                    elif clearance < 0 and face[i] > 0:
                        root = brentq(distance, lo, t, xtol=1e-12)
                    else:
                        continue
                if root < end:
                    end, status = root, label
                break
            if face[-1] == 0 and rate[-1] < 0 and status == "time_limit":
                status = label
        return status, end

    def wall(self, side, x, vx, speed):
        return wall_release(self.config, ("lower", "upper")[side], x, vx, speed)

    def verify(self, model):
        _, release = zip(*conditions(self.config, "final"))
        releases = np.asarray(release)
        times = np.linspace(0, self.horizon, 81)
        pred = predict_trajectory(model, *releases.T, times)
        state = self.position(releases[:, None, :], times, velocity=True)
        valid = pred["valid"]
        position = abs(state[..., :2][valid]-pred["position"][valid]).max()
        velocity = abs(state[..., 2:][valid]-pred["velocity"][valid]).max()
        kind, time = zip(*(self.event(row) for row in releases))
        event_error = abs(np.asarray(time)-pred["event"]["time"]).max()
        assert np.array_equal(kind, pred["event"]["type"])
        assert max(position, velocity, event_error) < 2e-11
        return dict(case_count=len(releases), position_max=position, velocity_max=velocity,
                    event_time_max=event_error, event_types_agree=True)
