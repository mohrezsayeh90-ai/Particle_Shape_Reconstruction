"""Release validation and independent banks of training/evaluation conditions."""

import numpy as np

from .model import input_bounds


def wall_release(config, side, x0, vx0, inward_speed):
    """Build Cartesian release inputs from a lower or upper wall launcher."""
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'.")
    x0, vx0, speed = np.broadcast_arrays(x0, vx0, inward_speed)
    low, high = config["domain"]["inward_speed"]
    if not np.isfinite(speed).all() or np.any((speed < low) | (speed > high)):
        raise ValueError(f"inward_speed must be in [{low}, {high}].")
    p = config["physics"]
    y0 = p["diameter"] / 2 + p["launch_gap"]
    if side == "upper":
        y0, speed = p["height"] - y0, -speed
    releases = np.stack((x0, np.full(x0.shape, y0), vx0, speed), axis=-1)
    check_releases(config, releases)
    return releases


def check_releases(config, releases):
    releases = np.asarray(releases, dtype=float)
    if releases.shape[-1:] != (4,) or not releases.size or not np.isfinite(releases).all():
        raise ValueError("Provide finite release conditions with last axis [x0, y0, vx0, vy0].")
    lower, upper = np.array(input_bounds(config)[1:]).T
    if np.any((releases < lower) | (releases > upper)):
        raise ValueError("Initial condition is outside the supported Cartesian bounds.")


def sample_releases(config, rng, count, *, boundaries=False):
    lower, upper = np.array(input_bounds(config)[1:]).T
    releases = rng.uniform(lower, upper, size=(count, 4))
    # Include both exact Master3 launcher heights as well as the continuous interior.
    groups = np.arange(count) % 4
    for group, side in ((0, "lower"), (1, "upper")):
        subset = groups == group
        speed = rng.uniform(*config["domain"]["inward_speed"], size=int(subset.sum()))
        releases[subset] = wall_release(config, side, releases[subset, 0],
                                       releases[subset, 2], speed)
    releases[groups == 2, 3] = 0
    if boundaries:
        faces = rng.permutation(count)[:count // 4]
        for row in faces:
            edge = rng.integers(0, 2)
            if groups[row] < 2:
                axis = rng.choice((0, 2, 3))
                if axis == 3:
                    speed = config["domain"]["inward_speed"][edge]
                    releases[row, 3] = speed if groups[row] == 0 else -speed
                    continue
            else:
                axis = rng.integers(0, 4)
            releases[row, axis] = (lower, upper)[edge][axis]
    return releases
