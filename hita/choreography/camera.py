"""Camera choreography helpers (elev/azim orbits, shortest-path lerp)."""
from __future__ import annotations

import numpy as np

from hita.primitives.math import lerp, smoothstep


def shortest_azim_lerp(az0: float, az1: float, u: float) -> float:
    """Interpolate azimuth along the shortest arc (degrees)."""
    d = (float(az1) - float(az0) + 180.0) % 360.0 - 180.0
    return float(az0) + d * float(np.clip(u, 0.0, 1.0))


def elev_azim_lerp(
    elev0: float,
    az0: float,
    elev1: float,
    az1: float,
    u: float,
    *,
    ease: bool = True,
) -> tuple[float, float]:
    t = smoothstep(u) if ease else float(np.clip(u, 0.0, 1.0))
    return lerp(elev0, elev1, t), shortest_azim_lerp(az0, az1, t)


def orbit_azims(az0: float, degrees: float = 360.0, n: int = 96, *, endpoint: bool = False) -> np.ndarray:
    """Azimuth schedule for a camera orbit."""
    return float(az0) + np.linspace(0.0, float(degrees), int(n), endpoint=endpoint)


def elev_ramp(elev0: float, elev1: float, n: int) -> np.ndarray:
    return np.linspace(float(elev0), float(elev1), int(n))
