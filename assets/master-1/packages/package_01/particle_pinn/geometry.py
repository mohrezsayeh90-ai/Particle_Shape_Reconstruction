"""Planar constant-width channels built from an editable centerline."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree


@dataclass
class Channel:
    kind: str = "serpentine"
    rin: float = 0.2
    rout: float = 0.5
    path: str | None = None

    def __post_init__(self):
        if not 0 < self.rin < self.rout:
            raise ValueError("Require 0 < rin < rout.")
        self.half_width = (self.rout - self.rin) / 2
        radius = (self.rin + self.rout) / 2
        if self.path:
            xy = np.loadtxt(self.path, delimiter=",", skiprows=1, ndmin=2)
            if xy.shape[1] != 2 or len(xy) < 4:
                raise ValueError("Centerline CSV needs header x,y and at least four points.")
            self.kind = "custom"
        elif self.kind == "straight":
            xy = np.column_stack((np.linspace(0, 2, 200), np.zeros(200)))
        elif self.kind == "bend":
            a = np.linspace(np.pi, 0, 401)
            xy = np.column_stack((radius + radius * np.cos(a), radius * np.sin(a)))
        elif self.kind == "serpentine":
            a = np.linspace(np.pi, 0, 301)
            first = np.column_stack((radius + radius * np.cos(a), radius * np.sin(a)))
            second = np.column_stack((3 * radius + radius * np.cos(a), -radius * np.sin(a)))
            xy = np.vstack((first, second[1:]))
        else:
            raise ValueError(f"Unknown geometry: {self.kind}")
        if not np.all(np.isfinite(xy)):
            raise ValueError("Centerline contains non-finite coordinates.")
        ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if np.any(ds <= 1e-10):
            raise ValueError("Centerline contains duplicate consecutive points.")
        s = np.r_[0., np.cumsum(ds)]
        self.length = float(s[-1])
        self.curve = CubicSpline(s, xy, axis=0)
        self.lookup_s = np.linspace(0, self.length, 2401)
        centers = self.curve(self.lookup_s)
        self.tree = cKDTree(centers)
        d, dd = self.curve(self.lookup_s, 1), self.curve(self.lookup_s, 2)
        k = np.abs(d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]) / np.linalg.norm(d, axis=1)**3
        if np.max(k) * self.half_width > .92:
            raise ValueError("Channel is too wide for this centerline curvature; reduce width.")
        # Detect remote portions of the tube intersecting. Adjacent samples are excluded.
        pairs = self.tree.query_pairs(1.999*self.half_width, output_type="ndarray")
        if len(pairs) and np.any(np.abs(self.lookup_s[pairs[:, 0]]-self.lookup_s[pairs[:, 1]]) > 4*self.half_width):
            raise ValueError("Channel overlaps itself; use a nonintersecting centerline.")

    def frame(self, s):
        s = np.asarray(s)
        c = self.curve(s)
        t = self.curve(s, 1)
        t = t / np.linalg.norm(t, axis=-1, keepdims=True)
        n = np.stack((-t[..., 1], t[..., 0]), axis=-1)
        return c, t, n

    def map(self, s, eta):
        c, _, n = self.frame(s)
        return c + self.half_width * np.asarray(eta)[..., None] * n

    def coordinates(self, xy):
        """Closest centerline coordinate, with tangent extrapolation beyond end caps."""
        xy = np.atleast_2d(xy)
        _, idx = self.tree.query(xy)
        s = self.lookup_s[idx].copy()
        for _ in range(4):
            c, d, dd = self.curve(s), self.curve(s, 1), self.curve(s, 2)
            residual = c - xy
            denom = np.sum(d*d + residual*dd, axis=1)
            step = np.sum(residual*d, axis=1) / np.maximum(denom, .1)
            s = np.clip(s - step, 0, self.length)
        c, t, n = self.frame(s)
        eta = np.sum((xy-c)*n, axis=1) / self.half_width
        ext = np.sum((xy-c)*t, axis=1)
        s = s + np.where((s < 1e-9) | (s > self.length-1e-9), ext, 0)
        return s, eta

    def sample(self, n, rng, margin=.002):
        s = rng.uniform(margin * self.length, (1-margin) * self.length, n)
        eta = rng.uniform(-1+margin, 1-margin, n)
        return self.map(s, eta)

    def boundary(self, n):
        s = np.linspace(0, self.length, n)
        wall = np.vstack((self.map(s, -np.ones(n)), self.map(s, np.ones(n))))
        eta = np.linspace(-1, 1, n)
        caps, vel = [], []
        for end in (0., self.length):
            caps.append(self.map(np.full(n, end), eta))
            _, t, _ = self.frame(end)
            vel.append((1-eta**2)[:, None] * t)
        return np.vstack((wall, *caps)), np.vstack((np.zeros_like(wall), *vel))

    def grid(self, ns=100, nn=19):
        s, eta = np.meshgrid(np.linspace(0, self.length, ns), np.linspace(-1, 1, nn), indexing="ij")
        return self.map(s.ravel(), eta.ravel()), s, eta
