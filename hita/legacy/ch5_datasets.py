"""Chapter 6 — four canonical trust datasets."""
from __future__ import annotations

import numpy as np

# Shared plot window for D1, D2, D4 (matches chapters 1–5).
CH5_STANDARD_XLIM = (0.0, 7.0)
CH5_STANDARD_YLIM = (0.0, 7.0)

# D1: clean separable roster from chapters 1–5.
CH5_D1_POINTS = [
    (2, 3, 0),
    (4, 5, 0), (5, 6, 0),
    (1, 3, 0), (2, 4, 0), (4, 6, 0),
    (1, 4, 0), (3, 6, 0), (1, 6, 0),
    (3, 2, 1),
    (5, 4, 1), (6, 5, 1),
    (4, 2, 1), (6, 4, 1), (3, 1, 1),
    (4, 1, 1), (5, 2, 1), (6, 3, 1),
    (6, 2, 1),
    (6, 1, 1),
]

# D2: D1 + two mislabeled points on each side of w·x = 0 at (0.5, -0.5, 0).
CH5_D2_NOISE = [
    (2, 1, 0), (4, 3, 0),   # above boundary, labeled fail
    (1, 2, 1), (3, 4, 1),   # below boundary, labeled pass
]
CH5_D2_POINTS = CH5_D1_POINTS + CH5_D2_NOISE

# Generating logistic for D3 (all points in [1, 6] × [1, 6]).
CH5_D3_W = (0.5, -0.5, 0.0)
CH5_D3_LINE_STEP = 0.5
CH5_D3_POINTS_PER_LINE = 4  # 15 bands × 4 = 60 pts
CH5_D3_RNG_SEED = 5
CH5_D3_ST_MIN = 1
CH5_D3_ST_MAX = 6
CH5_D3_EL_MIN = 1
CH5_D3_EL_MAX = 6
CH5_D3_ORTH_HALF = 3.5  # d = st − el spans [−3.5, 3.5] in ½ steps → 15 bands
CH5_D3_ORTH_PAD = 1.0  # extra viewport margin on each side ⊥ threshold (d = st − el)

# D4: minimally informative — 10 alternating points on a circle at (3.5, 3.5), r=1.5.
# Neighbors and diametric opposites carry the opposite class → MAP ≈ 0 and the
# belief surface stays nearly as wide as the prior (see ch5_posterior_display_density).
CH5_D4_CENTER = (3.5, 3.5)
CH5_D4_RADIUS = 1.5
CH5_D4_N = 10


def _ch5_gen_d4_minimal(*, center=CH5_D4_CENTER, radius=CH5_D4_RADIUS, n=CH5_D4_N):
    """Regular n-gon on a circle; labels alternate so adjacent and opposite points disagree."""
    cx = float(center[0])
    cy = float(center[1])
    r = float(radius)
    angles = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    pts = []
    for i, ang in enumerate(angles):
        x = round(cx + r * np.cos(ang), 3)
        y = round(cy + r * np.sin(ang), 3)
        pts.append((x, y, i % 2))
    return pts


CH5_D4_POINTS = _ch5_gen_d4_minimal()


def _ch5_d4_xor_points():
    """Legacy name — returns the current D4 roster."""
    return list(_ch5_gen_d4_minimal())


def _ch5_gen_d4_uniform(*, rng_seed=14):
    """Deprecated — kept for notebooks that still call the old generator name."""
    return _ch5_gen_d4_minimal()


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def _ch5_gen_d3_logistic(
    *,
    w_st=CH5_D3_W[0],
    w_el=CH5_D3_W[1],
    b=CH5_D3_W[2],
    line_step=CH5_D3_LINE_STEP,
    n_per_line=CH5_D3_POINTS_PER_LINE,
    rng_seed=CH5_D3_RNG_SEED,
    st_min=CH5_D3_ST_MIN,
    st_max=CH5_D3_ST_MAX,
    el_min=CH5_D3_EL_MIN,
    el_max=CH5_D3_EL_MAX,
    orth_half=CH5_D3_ORTH_HALF,
):
    """Logistic cloud with study time and exam score in [st_min, st_max] × [el_min, el_max].

    Threshold: w = (0.5, −0.5, 0)  →  st − el = 0  (half pass on the boundary).

    • Perpendicular offset d = st − el, stepped by ``line_step`` (default ½) from
      −``orth_half`` to +``orth_half``.
    • Parallel coordinate t = st + el is sampled on each band so both coordinates
      stay inside the box.
    • Label at each point: Bernoulli(σ(w·x)).
    """
    rng = np.random.default_rng(rng_seed)
    points = []
    st_min, st_max = float(st_min), float(st_max)
    el_min, el_max = float(el_min), float(el_max)
    for d in np.arange(-float(orth_half), float(orth_half) + 0.5 * line_step, line_step):
        t_lo = max(2.0 * st_min - d, 2.0 * el_min + d)
        t_hi = min(2.0 * st_max - d, 2.0 * el_max + d)
        if t_hi < t_lo - 1e-12:
            continue
        for t in np.linspace(t_lo, t_hi, n_per_line):
            st = 0.5 * (t + d)
            el = 0.5 * (t - d)
            if not (st_min <= st <= st_max and el_min <= el <= el_max):
                continue
            logit = w_st * st + w_el * el + b
            p = float(_sigmoid(logit))
            y = int(rng.binomial(1, p))
            points.append((float(st), float(el), y))
    return points


def _ch5_axis_limits_from_points(points, *, pad=0.35, floor_x=CH5_STANDARD_XLIM, floor_y=CH5_STANDARD_YLIM):
    arr = np.asarray(points, dtype=float)
    xlim = (
        min(float(floor_x[0]), float(np.min(arr[:, 0])) - pad),
        max(float(floor_x[1]), float(np.max(arr[:, 0])) + pad),
    )
    ylim = (
        min(float(floor_y[0]), float(np.min(arr[:, 1])) - pad),
        max(float(floor_y[1]), float(np.max(arr[:, 1])) + pad),
    )
    return xlim, ylim


def _ch5_d3_plot_limits(points, *, orth_pad=1.0):
    """Square D3 viewport: equal st/el span, centered on the cloud.

    Extends the visible range by ``orth_pad`` on each side along d = st − el
    (orthogonal to the threshold w·x = 0 at w = (0.5, −0.5, 0)).
    """
    arr = np.asarray(points, dtype=float)
    st, el = arr[:, 0], arr[:, 1]
    d = st - el
    t = st + el
    d_lo = float(np.min(d)) - float(orth_pad)
    d_hi = float(np.max(d)) + float(orth_pad)
    t_lo, t_hi = float(np.min(t)), float(np.max(t))
    st_vals = [0.5 * (tv + dv) for tv in (t_lo, t_hi) for dv in (d_lo, d_hi)]
    el_vals = [0.5 * (tv - dv) for tv in (t_lo, t_hi) for dv in (d_lo, d_hi)]
    st_lo, st_hi = min(st_vals), max(st_vals)
    el_lo, el_hi = min(el_vals), max(el_vals)
    span = max(st_hi - st_lo, el_hi - el_lo)
    cx = 0.5 * (st_lo + st_hi)
    cy = 0.5 * (el_lo + el_hi)
    half = 0.5 * span
    return (cx - half, cx + half), (cy - half, cy + half)


CH5_D3_POINTS = _ch5_gen_d3_logistic(rng_seed=CH5_D3_RNG_SEED)

CH5_DATASET_KEYS = ("D1", "D2", "D3", "D4")

# D3 uses the same (0, 7) window as D1/D2/D4 (custom orth-pad lims kept available
# via `_ch5_d3_plot_limits` for niche call sites only).
CH5_DATASET_META = {
    "D1": {
        "title": "Convincing",
        "credible_target": 0.90,
        "weights": (0.50, -0.50, 0.0),
        "xlim": CH5_STANDARD_XLIM,
        "ylim": CH5_STANDARD_YLIM,
    },
    "D2": {
        "title": "Compromise",
        "credible_target": 0.95,
        "weights": (0.50, -0.50, 0.0),
        "xlim": CH5_STANDARD_XLIM,
        "ylim": CH5_STANDARD_YLIM,
    },
    "D3": {
        "title": "One solution",
        "credible_target": 0.99,
        "weights": CH5_D3_W,
        "xlim": CH5_STANDARD_XLIM,
        "ylim": CH5_STANDARD_YLIM,
    },
    "D4": {
        "title": "No pattern",
        "credible_target": 0.95,
        "weights": (0.0, 0.0, 0.0),
        "xlim": CH5_STANDARD_XLIM,
        "ylim": CH5_STANDARD_YLIM,
    },
}

CH5_DATASET_POINTS = {
    "D1": CH5_D1_POINTS,
    "D2": CH5_D2_POINTS,
    "D3": CH5_D3_POINTS,
    "D4": CH5_D4_POINTS,
}


def ch5_plot_limits(key: str):
    meta = CH5_DATASET_META[str(key)]
    return meta["xlim"], meta["ylim"]


def ch5_unpack_dataset(key: str):
    arr = np.array(CH5_DATASET_POINTS[str(key)], dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2].astype(int)
