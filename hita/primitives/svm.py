"""Support-vector geometry on the logistic-regression students.

The separable roster is the Chapter 1 survey before the noisy students arrive.
Labels here are the SVM signs, +1 pass and −1 fail.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import linprog, minimize

# Chapter 1 separable students, then the six who break a straight cut.
SEPARABLE = (
    (2, 3, 0), (4, 5, 0), (5, 6, 0), (1, 3, 0), (2, 4, 0), (4, 6, 0),
    (1, 4, 0), (3, 6, 0), (1, 6, 0),
    (3, 2, 1), (5, 4, 1), (6, 5, 1), (4, 2, 1), (6, 4, 1), (3, 1, 1),
    (4, 1, 1), (5, 2, 1), (6, 3, 1), (6, 2, 1), (6, 1, 1),
)
NOISY = (
    (2, 1, 0), (1, 2, 1), (3, 4, 1), (4, 3, 0), (3, 5, 1), (5, 3, 0),
)

# Orientations whose centered separators have visibly different margins.
# Steep and narrow, flatter and middling, then the max-margin diagonal.
NARROW_ANG = -1.075
MEDIUM_ANG = -0.629
WIDE_ANG = -0.25 * np.pi

# Hard-margin solution on the separable roster: w = (1, −1), b = 0.
# The six students with |study − exam| = 1 share α = 1/3.
OPT_W = (1.0, -1.0)
OPT_B = 0.0
SV_INDEX = (0, 1, 2, 9, 10, 11)
SV_ALPHA = 1.0 / 3.0

# A feasible point on the same ray, weights too long (strip too narrow).
OUTER_W = (2.2, -2.2)
OUTER_B = 0.0

# The survey indexes we keep once the picture is small enough to solve by hand.
# Fail (2, 4) and pass (4, 2). Their max-margin line is the perpendicular
# bisector: w = (1/2, −1/2), b = 0, each α = 1/4, objective 1/4.
CORE_FAIL = 4
CORE_PASS = 12
LEFT_PT = 8     # fail (1, 6), farther out on the fail side
RIGHT_PT = 19   # pass (6, 1), farther out on the pass side
MID_PT = 9      # pass (3, 2), inside the two-point strip
PAIR_W = (0.5, -0.5)
PAIR_B = 0.0
PAIR_ALPHA = 0.25
# After the inside pass joins: the old pass lets go. Objective rises to 2/5.
MOVED_W = (0.4, -0.8)
MOVED_B = 1.4
MOVED_ALPHA = 0.4
# Overlapping logistic students, appended after the separable 20.
NOISE_FAIL = 20   # (4, 3)
NOISE_PASS = 21   # (3, 4)
# (w1, w2, b) window: the origin, both solutions, and a long feasible weight.
SIMPLE_BOUNDS = (-0.15, 1.55, -1.70, 0.30, -1.55, 2.05)

# Perceptron that stops at a thin but valid separator.
PERC_ANG0 = -0.60
PERC_B0 = -2.0
PERC_ETA = 0.30

# Chapter 8 workshop: three students, then two who sit far outside the strip.
TOY = (
    (-1.0, 0.0, -1.0),  # A fail
    (1.0, -1.0, 1.0),   # B pass
    (1.0, 1.0, 1.0),    # C pass
    (-3.0, 0.0, -1.0),  # distant fail
    (3.0, 0.0, 1.0),    # distant pass
)
TOY_NAMES = ("A", "B", "C", "", "")


def _pack_points(rows) -> dict[str, np.ndarray]:
    arr = np.asarray(rows, dtype=np.float64)
    raw = arr[:, 2].astype(np.float64)
    binary = np.all((raw == 0.0) | (raw == 1.0))
    signed = np.all((raw == -1.0) | (raw == 1.0))
    if not (binary or signed):
        raise ValueError("labels must be 0/1 or ±1")
    ypm = np.where(raw > 0.5, 1.0, -1.0)
    y01_icon = np.where(ypm > 0, 1.0, 0.0)
    return {
        "x": arr[:, 0].copy(),
        "e": arr[:, 1].copy(),
        "y01": y01_icon,
        "ypm": ypm,
    }


def separable_set() -> dict[str, np.ndarray]:
    return _pack_points(SEPARABLE)


def realistic_set() -> dict[str, np.ndarray]:
    return _pack_points(SEPARABLE + NOISY)


def toy_set() -> dict[str, np.ndarray]:
    return _pack_points(TOY)


def moon_set() -> dict[str, np.ndarray]:
    """Two nested crescents. No straight line separates them."""
    n = 7
    t = np.linspace(0.08, np.pi - 0.08, n)
    upper = np.column_stack([np.cos(t), np.sin(t)])
    lower = np.column_stack([1.0 - np.cos(t), 1.0 - np.sin(t) - 0.5])

    def _fit(xy: np.ndarray) -> np.ndarray:
        x = (xy[:, 0] + 1.1) / 3.2 * 5.0 + 0.8
        y = (xy[:, 1] + 0.6) / 1.7 * 4.2 + 1.2
        return np.column_stack([x, y])

    passed = _fit(upper)
    failed = _fit(lower)
    rows = [(float(x), float(y), 0) for x, y in failed] + [
        (float(x), float(y), 1) for x, y in passed
    ]
    return _pack_points(rows)


def linearly_separable(x: np.ndarray, e: np.ndarray, ypm: np.ndarray) -> bool:
    """True when some line puts every point strictly on its own side."""
    y = np.asarray(ypm, dtype=np.float64)
    X = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(e, dtype=np.float64)])
    A = np.column_stack([-y * X[:, 0], -y * X[:, 1], -y])
    res = linprog(
        np.zeros(3),
        A_ub=A,
        b_ub=-np.ones(len(y)),
        bounds=[(None, None)] * 3,
        method="highs",
    )
    return bool(res.success)


def separator_at_angle(ang: float, x=None, e=None, ypm=None) -> tuple[float, float, float, float]:
    """Unit-weight separator centered in the gap at this normal angle.

    Returns ``(w1, w2, b, geometric_margin)``. Margin is negative when the
    angle cannot separate the separable roster.
    """
    if x is None:
        ds = separable_set()
        x, e, ypm = ds["x"], ds["e"], ds["ypm"]
    X = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(e, dtype=np.float64)])
    y = np.asarray(ypm, dtype=np.float64)
    w = np.array([np.cos(ang), np.sin(ang)], dtype=np.float64)
    proj = X @ w
    gap = 0.5 * ((-proj[y < 0]).min() - (-proj[y > 0]).max())
    if gap <= 1e-8:
        return float(w[0]), float(w[1]), 0.0, float(gap)
    lo = float((-proj[y > 0]).max() + gap)
    hi = float((-proj[y < 0]).min() - gap)
    b = 0.5 * (lo + hi)
    return float(w[0]), float(w[1]), float(b), float(gap)


def functional_scores(w1: float, w2: float, b: float, x, e) -> np.ndarray:
    return float(w1) * np.asarray(x, dtype=np.float64) + float(w2) * np.asarray(e, dtype=np.float64) + float(b)


def signed_scores(w1, w2, b, x, e, ypm) -> np.ndarray:
    return np.asarray(ypm, dtype=np.float64) * functional_scores(w1, w2, b, x, e)


def nearest_index(w1, w2, b, x, e) -> int:
    f = functional_scores(w1, w2, b, x, e)
    nrm = max(float(np.hypot(w1, w2)), 1e-12)
    return int(np.argmin(np.abs(f) / nrm))


def perceptron_keyframes() -> list[dict[str, float]]:
    """Real perceptron updates. The last line separates, with a thin margin."""
    ds = separable_set()
    X = np.column_stack([ds["x"], ds["e"]])
    y = ds["ypm"]
    w = np.array([np.cos(PERC_ANG0), np.sin(PERC_ANG0)], dtype=np.float64)
    b = float(PERC_B0)
    order = list(range(len(y)))
    frames = [{"w1": float(w[0]), "w2": float(w[1]), "b": float(b), "focus": -1.0}]
    for _ in range(12):
        wrong = [i for i in order if y[i] * (X[i] @ w + b) <= 1e-9]
        if not wrong:
            break
        i = wrong[0]
        order = order[order.index(i) + 1 :] + order[: order.index(i) + 1]
        w = w + PERC_ETA * y[i] * X[i]
        b = b + PERC_ETA * y[i]
        frames.append({"w1": float(w[0]), "w2": float(w[1]), "b": float(b), "focus": float(i)})
    return frames


def normalize_to_unit_margin(w1: float, w2: float, b: float, x, e, ypm) -> tuple[float, float, float]:
    """Rescale so the smallest label-times-score is exactly 1."""
    m = float(np.min(signed_scores(w1, w2, b, x, e, ypm)))
    if m <= 1e-8:
        raise ValueError("cannot normalize a line that does not strictly separate")
    return float(w1) / m, float(w2) / m, float(b) / m


def smo_state(seg: int, u: float) -> dict[str, float]:
    """Exact pair steps from the narration, then the short last move to the optimum.

    ``seg`` 0..3 and ``u`` in [0, 1] walk A–B, then A–C, then B–C, then the
    remaining step onto ``α = (1/2, 1/4, 1/4)``.
    """
    u = float(np.clip(u, 0.0, 1.0))
    seg = int(seg)
    if seg <= 0:
        a = 0.4 * u
        alphas = (a, a, 0.0)
        # Pair-consistent offset from student A: b = −1 + w1, w = (2a, −a).
        w1, w2 = 2.0 * a, -a
        b = -1.0 + w1
    elif seg == 1:
        t = 0.16 * u
        alphas = (0.4 + t, 0.4, t)
        w1, w2 = 0.8 + 2.0 * t, -0.4 + t
        b = -0.2 + 2.0 * t
    elif seg == 2:
        t = 0.12 * u
        alphas = (0.56, 0.4 - t, 0.16 + t)
        w1, w2 = 1.12, -0.24 + 2.0 * t
        b = 0.12 + (-0.12 - 0.12) * u
    else:
        alphas = tuple(float((1.0 - u) * a + u * b_) for a, b_ in zip((0.56, 0.28, 0.28), (0.5, 0.25, 0.25)))
        w1 = alphas[0] + alphas[1] + alphas[2]
        w2 = -alphas[1] + alphas[2]
        b = -0.12 * (1.0 - u)
    dual = _toy_dual(alphas)
    return {
        "a0": float(alphas[0]),
        "a1": float(alphas[1]),
        "a2": float(alphas[2]),
        "w1": float(w1),
        "w2": float(w2),
        "b": float(b),
        "dual": float(dual),
    }


def _toy_dual(alphas) -> float:
    X = np.array([[-1.0, 0.0], [1.0, -1.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([-1.0, 1.0, 1.0])
    a = np.asarray(alphas, dtype=np.float64)
    gram = (y[:, None] * y[None, :]) * (X @ X.T)
    return float(a.sum() - 0.5 * a @ gram @ a)


def smo_curve(seg: int, n: int = 80) -> tuple[np.ndarray, np.ndarray, float]:
    """1D dual slice for one pair step. Returns ``(axis, dual, peak_axis)``."""
    seg = int(seg)
    if seg <= 0:
        axis = np.linspace(0.0, 0.7, n)
        # D(a) = 2a − 2.5 a²
        vals = 2.0 * axis - 2.5 * axis ** 2
        peak = 0.4
    elif seg == 1:
        axis = np.linspace(0.0, 0.36, n)
        vals = np.array([smo_state(1, float(t / 0.16) if t <= 0.16 else 1.0)["dual"] for t in axis])
        # Outside the step, evaluate the quadratic directly.
        vals = 0.4 + 0.8 * axis - 2.5 * axis ** 2
        peak = 0.16
    elif seg == 2:
        axis = np.linspace(0.0, 0.28, n)
        vals = np.array([_toy_dual((0.56, 0.4 - t, 0.16 + t)) for t in axis])
        peak = 0.12
    else:
        axis = np.linspace(0.0, 1.0, n)
        vals = np.array([smo_state(3, float(t))["dual"] for t in axis])
        peak = 1.0
    return axis, vals, peak


def ray_dual(t: float) -> float:
    """Dual value when the optimal alphas are scaled by ``t`` (balance holds)."""
    t = float(t)
    return 2.0 * t - t * t


def pair_dual(t: float) -> float:
    """Dual along α_fail = α_pass = t for the two kept students. Peak 1/4 at t = 1/4."""
    t = float(t)
    return 2.0 * t - 4.0 * t * t


def moved_dual(t: float) -> float:
    """Dual along the active pair after the inside pass joins. Peak 2/5 at t = 2/5."""
    t = float(t)
    return 2.0 * t - 2.5 * t * t


@lru_cache(maxsize=1)
def soft_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft-margin solutions on the realistic roster, low C to high C.

    Returns ``C`` (n,), ``wb`` (n, 3), ``xi`` (n, n_students).
    """
    ds = realistic_set()
    X = np.column_stack([ds["x"], ds["e"]])
    y = ds["ypm"]
    n = len(y)
    Cs = np.geomspace(0.02, 0.2, 7)

    def _solve(C: float) -> tuple[np.ndarray, np.ndarray]:
        def obj(v):
            return 0.5 * (v[0] ** 2 + v[1] ** 2) + float(C) * np.sum(v[3:])

        cons = [
            {
                "type": "ineq",
                "fun": lambda v, i=i: y[i] * (v[0] * X[i, 0] + v[1] * X[i, 1] + v[2]) - 1.0 + v[3 + i],
            }
            for i in range(n)
        ]
        bounds = [(None, None), (None, None), (None, None)] + [(0.0, None)] * n
        v0 = np.zeros(3 + n)
        v0[0], v0[1] = 0.4, -0.4
        res = minimize(obj, v0, bounds=bounds, constraints=cons, method="SLSQP", tol=1e-12, options={"maxiter": 800})
        if not res.success:
            raise RuntimeError(f"soft-margin solve failed at C={C}: {res.message}")
        return res.x[:3].astype(np.float64), np.clip(res.x[3:], 0.0, None).astype(np.float64)

    wbs = []
    xis = []
    for C in Cs:
        wb, xi = _solve(float(C))
        wbs.append(wb)
        xis.append(xi)
    return Cs.astype(np.float64), np.vstack(wbs), np.vstack(xis)


@lru_cache(maxsize=1)
def logistic_fit() -> tuple[float, float, float]:
    """Unregularized logistic MLE on the realistic roster, labels ±1."""
    ds = realistic_set()
    X = np.column_stack([ds["x"], ds["e"]])
    y = ds["ypm"]

    def nll(v):
        m = y * (X @ v[:2] + v[2])
        return float(np.sum(np.logaddexp(0.0, -m)))

    res = minimize(nll, np.array([0.7, -0.7, 0.0]), method="BFGS")
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def cast_rows() -> list[tuple[float, float, float]]:
    """Separable survey, plus the two logistic students who cross each other."""
    return list(SEPARABLE) + [(4.0, 3.0, 0.0), (3.0, 4.0, 1.0)]


@lru_cache(maxsize=1)
def small_soft_table() -> tuple[np.ndarray, np.ndarray]:
    """Soft-margin path once those two crossing students are back.

    Returns ``C`` (n,) and ``wb`` (n, 3) for the seven-student cast.
    """
    rows = [
        SEPARABLE[CORE_FAIL],
        SEPARABLE[CORE_PASS],
        SEPARABLE[LEFT_PT],
        SEPARABLE[RIGHT_PT],
        SEPARABLE[MID_PT],
        (4.0, 3.0, 0.0),
        (3.0, 4.0, 1.0),
    ]
    X = np.array([[r[0], r[1]] for r in rows], dtype=np.float64)
    y = np.array([1.0 if r[2] else -1.0 for r in rows])
    n = len(y)
    # Stop before C pushes the line flat. The last value still leaves both crossers in trouble.
    Cs = np.array([0.05, 0.08, 0.12, 0.18], dtype=np.float64)

    def _solve(C: float) -> np.ndarray:
        def obj(v):
            return 0.5 * (v[0] ** 2 + v[1] ** 2) + float(C) * np.sum(v[3:])

        cons = [
            {
                "type": "ineq",
                "fun": lambda v, i=i: y[i] * (v[0] * X[i, 0] + v[1] * X[i, 1] + v[2]) - 1.0 + v[3 + i],
            }
            for i in range(n)
        ]
        bounds = [(None, None)] * 3 + [(0.0, None)] * n
        v0 = np.zeros(3 + n)
        v0[0], v0[1], v0[2] = MOVED_W[0], MOVED_W[1], MOVED_B
        res = minimize(obj, v0, bounds=bounds, constraints=cons, method="SLSQP", tol=1e-12, options={"maxiter": 800})
        if not res.success:
            raise RuntimeError(f"small soft-margin failed at C={C}: {res.message}")
        return res.x[:3].astype(np.float64)

    return Cs, np.vstack([_solve(float(C)) for C in Cs])


@lru_cache(maxsize=1)
def small_logistic_fit() -> tuple[float, float, float]:
    """Logistic MLE on the same seven students the soft-margin clip uses."""
    rows = [
        SEPARABLE[CORE_FAIL],
        SEPARABLE[CORE_PASS],
        SEPARABLE[LEFT_PT],
        SEPARABLE[RIGHT_PT],
        SEPARABLE[MID_PT],
        (4.0, 3.0, 0.0),
        (3.0, 4.0, 1.0),
    ]
    X = np.array([[r[0], r[1]] for r in rows], dtype=np.float64)
    y = np.array([1.0 if r[2] else -1.0 for r in rows])

    def nll(v):
        m = y * (X @ v[:2] + v[2])
        return float(np.sum(np.logaddexp(0.0, -m)))

    res = minimize(nll, np.array([0.2, -0.5, 1.0]), method="BFGS")
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def hinge(m: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - np.asarray(m, dtype=np.float64))


def logistic_loss(m: np.ndarray) -> np.ndarray:
    z = -np.asarray(m, dtype=np.float64)
    return np.logaddexp(0.0, z)


def voxel_lattice(n, X: np.ndarray, y: np.ndarray, bounds=None) -> dict[str, np.ndarray]:
    """Checkerboard cubes in (w1, w2, b). Color is the objective, the 4th coordinate.

    ``n`` is one count per axis, or ``(n1, n2, n3)`` so the cubes stay nearly cubic
    when the window is not a cube. Objective ``½(w1² + w2²)`` does not depend on b,
    so color is constant along that axis. ``min_yf`` is the tightest student
    constraint at the cell center.
    """
    if bounds is None:
        # Window holds the origin (unconstrained bottom), the outer feasible point,
        # and the hard-margin solution (1, −1, 0).
        bounds = (-0.55, 2.65, -2.65, 0.55, -1.45, 1.45)
    if isinstance(n, int):
        n1 = n2 = n3 = int(n)
    else:
        n1, n2, n3 = (int(v) for v in n)
    lo1, hi1, lo2, hi2, lob, hib = bounds
    e1 = np.linspace(lo1, hi1, n1 + 1)
    e2 = np.linspace(lo2, hi2, n2 + 1)
    e3 = np.linspace(lob, hib, n3 + 1)
    c1 = 0.5 * (e1[:-1] + e1[1:])
    c2 = 0.5 * (e2[:-1] + e2[1:])
    c3 = 0.5 * (e3[:-1] + e3[1:])
    W1, W2, B = np.meshgrid(c1, c2, c3, indexing="ij")
    ii, jj, kk = np.indices(W1.shape)
    checker = ((ii + jj + kk) % 2) == 0
    obj = 0.5 * (W1 ** 2 + W2 ** 2)
    # Constraints of the separable students.
    min_yf = np.full(W1.shape, np.inf, dtype=np.float64)
    for i in range(len(y)):
        f = y[i] * (W1 * X[i, 0] + W2 * X[i, 1] + B)
        min_yf = np.minimum(min_yf, f)
    # Diagonal reveal, low corner → high corner, matching the Ch4/Ch6 sweep.
    start = np.array([lo1, lo2, lob], dtype=np.float64)
    end = np.array([hi1, hi2, hib], dtype=np.float64)
    diag = end - start
    proj = (
        (W1 - start[0]) * diag[0]
        + (W2 - start[1]) * diag[1]
        + (B - start[2]) * diag[2]
    )
    sel = checker
    vmax = float(np.quantile(obj[sel], 0.92))
    vmax = max(vmax, 1e-6)
    # Same navy → red ramp as the Chapter 4 negative-log-likelihood voxels.
    ramp = _nll_rgb(np.clip(obj[sel] / vmax, 0.0, 1.0))
    rel = np.clip(obj[sel] / vmax, 0.0, 1.0)
    alpha = 0.90 * (1.0 - rel) + 0.22 * rel
    return {
        "x": e1[:-1][ii[sel]].astype(np.float64),
        "y": e2[:-1][jj[sel]].astype(np.float64),
        "z": e3[:-1][kk[sel]].astype(np.float64),
        "dx": np.float64(e1[1] - e1[0]),
        "dy": np.float64(e2[1] - e2[0]),
        "dz": np.float64(e3[1] - e3[0]),
        "rgb": ramp.astype(np.float64),
        "alpha0": alpha.astype(np.float64),
        "min_yf": min_yf[sel].astype(np.float64),
        "proj": proj[sel].astype(np.float64),
        "max_proj": np.float64(np.dot(diag, diag)),
        "lim": np.asarray(bounds, dtype=np.float64),
    }


def _nll_rgb(t: np.ndarray) -> np.ndarray:
    """CH4_NLL_HEATMAP_COLORS sampled along ``t`` in [0, 1]. Low objective is navy."""
    anchors = np.array(
        [
            [8, 17, 31],
            [18, 42, 136],
            [45, 91, 255],
            [122, 77, 255],
            [193, 60, 255],
            [255, 59, 92],
        ],
        dtype=np.float64,
    ) / 255.0
    stops = np.linspace(0.0, 1.0, len(anchors))
    t = np.asarray(t, dtype=np.float64).ravel()
    out = np.empty((t.size, 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.interp(t, stops, anchors[:, k])
    return out


if __name__ == "__main__":
    sep = separable_set()
    w1, w2, b, m = separator_at_angle(WIDE_ANG)
    assert abs(m - 1.0 / np.sqrt(2.0)) < 1e-6, m
    frames = perceptron_keyframes()
    f = signed_scores(frames[-1]["w1"], frames[-1]["w2"], frames[-1]["b"], sep["x"], sep["e"], sep["ypm"])
    assert f.min() > 0, f.min()
    st = smo_state(0, 1)
    assert abs(st["dual"] - 0.4) < 1e-9, st
    st = smo_state(1, 1)
    assert abs(st["dual"] - 0.464) < 1e-9, st
    st = smo_state(3, 1)
    assert abs(st["w1"] - 1.0) < 1e-9 and abs(st["dual"] - 0.5) < 1e-9, st
    moons = moon_set()
    assert not linearly_separable(moons["x"], moons["e"], moons["ypm"])
    Cs, wb, xi = soft_table()
    assert wb[0, 0] < wb[-1, 0]
    assert abs(pair_dual(PAIR_ALPHA) - 0.25) < 1e-9
    assert abs(moved_dual(MOVED_ALPHA) - 0.4) < 1e-9
    print("svm checks ok", "perc steps", len(frames) - 1, "pair", pair_dual(0.25), "moved", moved_dual(0.4))
