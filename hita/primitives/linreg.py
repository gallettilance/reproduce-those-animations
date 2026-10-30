"""Linear-regression plant datasets and SSE helpers for Chapter 8."""
from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np

# Perfect line: Age = Height / 3
CORE_POINTS = (
    (6.0, 2.0),
    (12.0, 4.0),
    (18.0, 6.0),
    (24.0, 8.0),
    (30.0, 10.0),
)
PLUS_POINT = (18.0, 8.0)

# Quadratic extras: a U around the linear fit so
#   short  → older plants  → younger predictions
#   middle → younger plants → older predictions
#   tall   → older plants  → younger predictions again
QUAD_EXTRA = (
    (3.0, 5.0),
    (5.0, 5.4),
    (8.0, 4.2),
    (14.0, 2.2),
    (16.5, 2.6),
    (21.0, 3.8),
    (23.5, 4.2),
    (31.5, 14.2),
    (34.0, 16.2),
    (36.0, 17.8),
)

# A few counter-points: young shorts, old middles, young talls.
NQUAD_EXTRA = (
    (3.8, 1.2),
    (7.0, 1.4),
    (14.0, 8.2),
    (17.5, 9.4),
    (21.0, 10.4),
    (24.5, 11.2),
    (33.0, 8.2),
    (36.0, 9.0),
)

# Fan: spread grows with height.
HETERO_EXTRA = (
    (7.0, 2.2),
    (7.0, 2.5),
    (14.0, 4.4),
    (14.0, 5.0),
    (22.0, 5.5),
    (22.0, 9.5),
    (25.0, 5.8),
    (25.0, 12.2),
    (28.0, 5.0),
    (28.0, 13.0),
    (30.0, 4.5),
    (30.0, 15.0),
    (32.0, 4.0),
    (32.0, 16.5),
    (35.0, 3.5),
    (35.0, 17.8),
)

HEIGHTS = (6.0, 12.0, 18.0, 24.0, 30.0)
COND_MEANS = (2.0, 4.0, 7.0, 8.0, 10.0)  # mean at 18 cm is the average of 6 and 8
COND_SIGMA = 1.0
X0_SLICE = 18.0  # SSE slice: prediction at height = 18 cm


@dataclass(frozen=True)
class PlantDataset:
    name: str
    height: np.ndarray
    age: np.ndarray
    n_base: int  # how many points belong to the previous stage

    @property
    def n(self) -> int:
        return int(self.height.size)

    @property
    def n_new(self) -> int:
        return self.n - int(self.n_base)


def _xy(points) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(points, dtype=np.float64)
    return arr[:, 0].copy(), arr[:, 1].copy()


def plant_datasets() -> dict[str, PlantDataset]:
    core_h, core_a = _xy(CORE_POINTS)
    plus_h, plus_a = _xy(CORE_POINTS + (PLUS_POINT,))
    quad_h, quad_a = _xy(CORE_POINTS + (PLUS_POINT,) + QUAD_EXTRA)
    nquad_h, nquad_a = _xy(CORE_POINTS + (PLUS_POINT,) + QUAD_EXTRA + NQUAD_EXTRA)
    het_h, het_a = _xy(CORE_POINTS + (PLUS_POINT,) + QUAD_EXTRA + NQUAD_EXTRA + HETERO_EXTRA)
    return {
        "core": PlantDataset("core", core_h, core_a, n_base=0),
        "plus": PlantDataset("plus", plus_h, plus_a, n_base=5),
        "quad": PlantDataset("quad", quad_h, quad_a, n_base=6),
        "nquad": PlantDataset("nquad", nquad_h, nquad_a, n_base=int(quad_h.size)),
        "hetero": PlantDataset("hetero", het_h, het_a, n_base=int(nquad_h.size)),
    }


def ols_fit(height, age) -> tuple[float, float]:
    """Return (intercept, slope) for age ~ intercept + slope * height."""
    x = np.asarray(height, dtype=np.float64).reshape(-1)
    y = np.asarray(age, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def residuals(height, age, intercept: float, slope: float) -> np.ndarray:
    x = np.asarray(height, dtype=np.float64)
    y = np.asarray(age, dtype=np.float64)
    return y - (intercept + slope * x)


def sse_two_point(p) -> np.ndarray:
    """SSE of ages {6, 8} for a constant prediction p at height 18."""
    p = np.asarray(p, dtype=np.float64)
    return (p - 6.0) ** 2 + (p - 8.0) ** 2


def sse_line(height, age, p, slope, x0: float = X0_SLICE) -> np.ndarray:
    """SSE of age = p + slope * (height − x0).

    ``p`` is the predicted age at height ``x0`` (the 18 cm slice).
    """
    x = np.asarray(height, dtype=np.float64).reshape(-1)
    y = np.asarray(age, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64)
    m = np.asarray(slope, dtype=np.float64)
    dx = x - float(x0)
    # Broadcast to (...grid, n)
    pred = p[..., None] + m[..., None] * dx.reshape((1,) * p.ndim + (x.size,))
    resid = y.reshape((1,) * p.ndim + (y.size,)) - pred
    return np.sum(resid ** 2, axis=-1)


def sse_mesh(
    height,
    age,
    p_grid,
    m_grid,
    x0: float = X0_SLICE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    P, M = np.meshgrid(np.asarray(p_grid, dtype=np.float64), np.asarray(m_grid, dtype=np.float64))
    Z = sse_line(height, age, P, M, x0=x0)
    return P, M, Z


def sse_ib(height, age, intercept, slope) -> np.ndarray:
    """SSE of age = intercept + slope * height."""
    x = np.asarray(height, dtype=np.float64).reshape(-1)
    y = np.asarray(age, dtype=np.float64).reshape(-1)
    b0 = np.asarray(intercept, dtype=np.float64)
    b1 = np.asarray(slope, dtype=np.float64)
    pred = b0[..., None] + b1[..., None] * x.reshape((1,) * b0.ndim + (x.size,))
    resid = y.reshape((1,) * b0.ndim + (y.size,)) - pred
    return np.sum(resid ** 2, axis=-1)


def sse_mesh_ib(
    height,
    age,
    b0_grid,
    m_grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B0, M = np.meshgrid(np.asarray(b0_grid, dtype=np.float64), np.asarray(m_grid, dtype=np.float64))
    Z = sse_ib(height, age, B0, M)
    return B0, M, Z


DIST_LO, DIST_HI = 0.35, 13.65  # stay inside the clip-03 age window
SHAPE_KINDS = ("normal", "heavy", "skew", "bimodal")


def gaussian_pdf(y, mu: float, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    s = max(float(sigma), 1e-9)
    return np.exp(-0.5 * ((y - float(mu)) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


def laplace_pdf(y, mu: float, b: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    bb = max(float(b), 1e-9)
    return 0.5 / bb * np.exp(-np.abs(y - float(mu)) / bb)


def shape_pdf(y, mu: float, kind: str, *, sigma: float = COND_SIGMA) -> np.ndarray:
    """Analytic density for a named Age | Height shape; mean is ``mu``."""
    y = np.asarray(y, dtype=np.float64)
    m = float(mu)
    kind = str(kind)
    if kind == "heavy":
        # Laplace: peaky center, long tails on both sides.
        return laplace_pdf(y, m, 0.95)
    if kind == "skew":
        # Gamma shifted so the mean is ``m``; long tail toward older ages.
        k, theta = 1.75, 0.92
        loc = m - k * theta
        z = np.clip(y - loc, 1e-12, None)
        logp = (k - 1.0) * np.log(z) - z / theta - k * np.log(theta) - lgamma(k)
        p = np.exp(logp)
        p = np.where(y >= loc, p, 0.0)
        return p
    if kind == "bimodal":
        d, s = 1.42, 0.38
        return 0.5 * gaussian_pdf(y, m - d, s) + 0.5 * gaussian_pdf(y, m + d, s)
    return gaussian_pdf(y, m, sigma)


def _clip_recenter(ages, mu: float, lo: float = DIST_LO, hi: float = DIST_HI) -> np.ndarray:
    out = np.asarray(ages, dtype=np.float64).copy()
    target = float(mu)
    for _ in range(4):
        out = np.clip(out, lo, hi)
        out = out - float(np.mean(out)) + target
    return np.clip(out, lo, hi)


def sample_conditional_ages(
    *,
    n: int = 320,
    seed: int = 8,
    sigma: float = COND_SIGMA,
) -> dict[float, np.ndarray]:
    """Fixed samples of plant ages at each canonical height."""
    rng = np.random.default_rng(int(seed))
    out: dict[float, np.ndarray] = {}
    for h, mu in zip(HEIGHTS, COND_MEANS):
        ages = rng.normal(float(mu), float(sigma), size=int(n))
        out[float(h)] = np.clip(ages, 0.35, 16.5)
    return out


def sample_shape_ages(
    *,
    n: int = 320,
    seed: int = 8,
    kind: str = "normal",
    sigma: float = COND_SIGMA,
) -> dict[float, np.ndarray]:
    """Age samples at each height for a named shape, all with the canonical means."""
    kind = str(kind)
    if kind == "normal":
        return sample_conditional_ages(n=n, seed=seed, sigma=sigma)
    rng = np.random.default_rng(int(seed) + 17 * (1 + SHAPE_KINDS.index(kind) if kind in SHAPE_KINDS else 1))
    out: dict[float, np.ndarray] = {}
    nn = int(n)
    for h, mu in zip(HEIGHTS, COND_MEANS):
        m = float(mu)
        if kind == "heavy":
            ages = rng.laplace(m, 0.95, size=nn)
        elif kind == "skew":
            k, theta = 1.75, 0.92
            ages = rng.gamma(k, theta, size=nn) - k * theta + m
        elif kind == "bimodal":
            d, s = 1.42, 0.38
            which = rng.random(nn) < 0.5
            ages = np.empty(nn, dtype=np.float64)
            ages[which] = rng.normal(m - d, s, size=int(np.count_nonzero(which)))
            ages[~which] = rng.normal(m + d, s, size=int(np.count_nonzero(~which)))
        else:
            ages = rng.normal(m, float(sigma), size=nn)
        out[float(h)] = _clip_recenter(ages, m)
    return out
