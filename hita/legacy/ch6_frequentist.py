"""Chapter 6 — frequentist sampling variability for logistic regression.

The generative story (fixed design, random labels)
-------------------------------------------------
D1's feature locations ``x_i = (study, exam)`` are treated as a fixed roster of
students.  Randomness lives only in the Bernoulli labels:

    Y_i | x_i  ~  Bern( σ(w_true · x_i) )

with ``w_true`` set from the Ch6 mixture via LDA (shared class covariance) — see
``ch6_lda_logistic_weights``.  Each draw is one "alternate classroom."

Why not just resample D1's observed labels (nonparametric bootstrap)?
--------------------------------------------------------------------
Observed D1 is *completely separable*.  Resampling those hard labels keeps the
fitted boundary almost glued in place (angle std ≈ 2°).  That understates the
uncertainty implied by the logistic model, which assigns soft probabilities near
the diagonal.  Parametric resampling at ``w_true`` is the process the likelihood
itself assumes.

Why the MLE needs a ridge
-------------------------
With n = 20 and occasional near-separation, the unpenalized MLE diverges along
the separating direction (likelihood ridge, not a bowl).  Wald / CLT intervals
then fail.  We fit a ridge-penalized MLE with λ = 1/σ² = 0.25 to match Ch5's
Gaussian prior σ = 2 — same math, frequentist reading (penalized likelihood).

Likelihood shape → confidence → wiggle
--------------------------------------
* Curvature of ℓ(w) at the MLE (Fisher / observed information) sets the Wald
  ellipse: flat directions → large SE → large line wiggle.
* Likelihood-ratio regions ``{w : 2(ℓ̂ − ℓ(w)) ≤ χ²}`` follow the true bowl
  shape without needing normality; they are the frequentist twin of Ch5's
  credible regions (cut the likelihood, not the posterior mass).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from ch5_datasets import (
    CH5_DATASET_KEYS,
    CH5_DATASET_META,
    ch5_plot_limits,
    ch5_unpack_dataset,
)

_CH3_DRAFT = os.environ.get("CH3_DRAFT_EXPORT", "").strip().lower() in {"1", "true", "yes", "y"}

# Set after mixture constants below — LDA logistic weights for resample clips.
CH6_W_TRUE: np.ndarray

# Penalized MLE: λ = 1/σ² with σ = CH5_PRIOR_SIGMA = 2.0.
CH6_RIDGE = 0.25
# Population reels: moderate ridge (lighter than CH6_RIDGE so spread shrinks with n).
CH6_POPULATION_FIT_RIDGE = 0.07

CH6_VIEW_BOUNDS_W12 = (-3.0, 3.0, -3.0, 3.0)
# Triple the sampling density vs the first cut (draft / hq).
CH6_N_REPS_CLOUD = 144 if _CH3_DRAFT else 480
CH6_N_REEL = 48 if _CH3_DRAFT else 120
CH6_MS = 110 if _CH3_DRAFT else 95
CH6_N_HOLD = 3 if _CH3_DRAFT else 8
CH6_N_SEQ_HOLD = 1 if _CH3_DRAFT else 2
CH6_N_FLASH = 1 if _CH3_DRAFT else 2
CH6_LR_GRID = 24 if _CH3_DRAFT else 48
# Denser (w_ST, w_EL) samples for smooth likelihood bowls; wireframe stride
# still targets CH5_SURFACE_GRID_SPACING so the visible mesh density is unchanged.
CH6_LR_GRID_SMOOTH = 52 if _CH3_DRAFT else 132
CH6_HIST_BINS = 12 if _CH3_DRAFT else 24
CH6_HIST_BINS_MAX = 28 if _CH3_DRAFT else 72
CH6_HIST_TARGET_CLUSTER_BINS = 7  # aim for ~this many bins across the landing cloud
CH6_CONF_MASS = 0.95
CH6_SURFACE_Z_HI = 1.0
CH6_N_GHOST_LINES_SHOW = 108 if _CH3_DRAFT else 120  # display cap for readability

CH6_LINE_COLOR = "#2c3e50"
CH6_TRUE_LINE_COLOR = "#c0392b"
CH6_GHOST_COLOR = "#5dade2"
CH6_CLOUD_COLOR = "#1a5276"
CH6_LR_COLOR = "#e67e22"
CH6_WALD_COLOR = "#8e44ad"


def ch6_sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def ch6_design(study, exam):
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    return np.column_stack([study, exam, np.ones(len(study))])


def ch6_nll(w, Xd, y, *, ridge=CH6_RIDGE):
    w = np.asarray(w, dtype=np.float64)
    p = ch6_sigmoid(Xd @ w)
    ll = y * np.log(p + 1e-15) + (1.0 - y) * np.log(1.0 - p + 1e-15)
    return float(-ll.sum() + 0.5 * float(ridge) * np.sum(w * w))


def ch6_nll_grad(w, Xd, y, *, ridge=CH6_RIDGE):
    """∇_w of ridge NLL: Xᵀ(p − y) + ridge · w."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    Xd = np.asarray(Xd, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = ch6_sigmoid(Xd @ w)
    return Xd.T @ (p - y) + float(ridge) * w


def ch6_loglik(w, Xd, y, *, ridge=0.0):
    """Log-likelihood; ``ridge`` defaults to 0 so LR regions use the model likelihood."""
    return -ch6_nll(w, Xd, y, ridge=ridge)


def ch6_fit(Xd, y, *, ridge=CH6_RIDGE, w0=None):
    Xd = np.asarray(Xd, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w0 = np.zeros(Xd.shape[1], dtype=np.float64) if w0 is None else np.asarray(w0, dtype=np.float64)
    res = minimize(lambda w: ch6_nll(w, Xd, y, ridge=ridge), w0, method="BFGS")
    return np.asarray(res.x, dtype=np.float64), bool(res.success)


def ch6_fit_dataset(study, exam, y, *, ridge=CH6_RIDGE):
    Xd = ch6_design(study, exam)
    w, ok = ch6_fit(Xd, y, ridge=ridge)
    return w, ok


def ch6_fit_population_classroom(study, exam, y, *, w0=None):
    """Ridge-penalized MLE for population-reel / sampling-variability clips."""
    Xd = ch6_design(study, exam)
    if w0 is None:
        w0 = np.asarray(CH6_W_TRUE, dtype=np.float64)
    w, ok = ch6_fit(Xd, y, ridge=CH6_POPULATION_FIT_RIDGE, w0=w0)
    return w, ok


def ch6_p_true(study, exam, *, w_true=None):
    w_true = CH6_W_TRUE if w_true is None else np.asarray(w_true, dtype=np.float64)
    return ch6_sigmoid(ch6_design(study, exam) @ w_true)


def ch6_resample_labels(study, exam, *, w_true=None, rng=None):
    """Parametric bootstrap: fix X, redraw Y ~ Bern(σ(w_true·x))."""
    rng = np.random.default_rng(rng)
    p = ch6_p_true(study, exam, w_true=w_true)
    return rng.binomial(1, p).astype(np.float64)


def ch6_boundary_angle_deg(w):
    w = np.asarray(w, dtype=np.float64)
    return float(np.degrees(np.arctan2(w[1], w[0])))


def ch6_observed_information(w, Xd, y, *, ridge=CH6_RIDGE):
    w = np.asarray(w, dtype=np.float64)
    p = ch6_sigmoid(Xd @ w)
    W = p * (1.0 - p)
    return Xd.T @ (W[:, None] * Xd) + float(ridge) * np.eye(Xd.shape[1])


def ch6_fisher_information(Xd, *, w_true=None, ridge=0.0):
    """Expected information under the true Bernoulli model (labels not needed)."""
    w_true = CH6_W_TRUE if w_true is None else np.asarray(w_true, dtype=np.float64)
    p = ch6_sigmoid(Xd @ w_true)
    W = p * (1.0 - p)
    return Xd.T @ (W[:, None] * Xd) + float(ridge) * np.eye(Xd.shape[1])


def ch6_wald_cov(w, Xd, y, *, ridge=CH6_RIDGE):
    H = ch6_observed_information(w, Xd, y, ridge=ridge)
    return np.linalg.inv(H)


def ch6_wald_ellipse_w12(mean_w, cov, *, mass=CH6_CONF_MASS, n=80, b_fixed=None):
    """2D Wald ellipse in (w_ST, w_EL), optionally conditioning on b = b_fixed.

    If ``b_fixed`` is None, uses the marginal covariance of (w1, w2).
    """
    mean_w = np.asarray(mean_w, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    if b_fixed is None:
        mu = mean_w[:2]
        C = cov[:2, :2]
        df = 2
    else:
        # Conditional on b: Schur complement
        Cbb = cov[2, 2]
        C12 = cov[:2, 2]
        C22 = cov[:2, :2]
        C = C22 - np.outer(C12, C12) / max(Cbb, 1e-12)
        mu = mean_w[:2] + C12 / max(Cbb, 1e-12) * (float(b_fixed) - mean_w[2])
        df = 2
    # χ² quantile for the Mahalanobis ellipse
    r2 = float(chi2.ppf(mass, df))
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-12, None)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=True)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    axes = vecs @ (np.sqrt(r2 * vals)[:, None] * circle)
    return mu[0] + axes[0], mu[1] + axes[1]


def ch6_lr_stat(w, w_hat, Xd, y, *, ridge=0.0):
    return 2.0 * (ch6_loglik(w_hat, Xd, y, ridge=ridge) - ch6_loglik(w, Xd, y, ridge=ridge))


def ch6_lr_grid_w12(
    Xd,
    y,
    w_hat,
    *,
    ridge=0.0,
    b_fixed=None,
    bounds=CH6_VIEW_BOUNDS_W12,
    grid=CH6_LR_GRID,
):
    """2(ℓ̂−ℓ) on a (w1, w2) grid with b fixed (default: MLE's b)."""
    Xd = np.asarray(Xd, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    b_fixed = float(w_hat[2] if b_fixed is None else b_fixed)
    gn = int(grid)
    w1 = np.linspace(bounds[0], bounds[1], gn)
    w2 = np.linspace(bounds[2], bounds[3], gn)
    W1, W2 = np.meshgrid(w1, w2)
    # Vectorized log-likelihood over the (w1, w2) mesh (b fixed).
    W = np.stack([W1, W2, np.full_like(W1, b_fixed)], axis=-1)  # (g, g, 3)
    logits = np.einsum("nd,ijd->ijn", Xd, W)
    p = ch6_sigmoid(logits)
    ll = (
        y[None, None, :] * np.log(p + 1e-15)
        + (1.0 - y[None, None, :]) * np.log(1.0 - p + 1e-15)
    ).sum(axis=-1)
    if float(ridge) > 0.0:
        ll = ll - 0.5 * float(ridge) * (W1 * W1 + W2 * W2 + b_fixed * b_fixed)
    ll_hat = ch6_loglik(w_hat, Xd, y, ridge=ridge)
    Z = 2.0 * (ll_hat - ll)
    return w1, w2, Z


def ch6_lr_threshold(mass=CH6_CONF_MASS, df=2):
    return float(chi2.ppf(mass, df))


@lru_cache(maxsize=32)
def ch6_sampling_cloud(
    key: str,
    n_reps: int = CH6_N_REPS_CLOUD,
    *,
    ridge: float = CH6_RIDGE,
    seed: int = 0,
    w_true: tuple[float, float, float] | None = None,
):
    """Fit penalized MLE on ``n_reps`` parametric resamples of dataset ``key``.

    Returns dict with arrays ``weights`` (n,3), ``angles``, ``study``, ``exam``,
    and the fixed design / true probs used to resample.
    """
    study, exam, y_obs = ch5_unpack_dataset(key)
    Xd = ch6_design(study, exam)
    wt = CH6_W_TRUE if w_true is None else np.asarray(w_true, dtype=np.float64)
    # For D4 meta weights are (0,0,0) — use that as the null generative process.
    if w_true is None and key == "D4":
        wt = np.asarray(CH5_DATASET_META["D4"]["weights"], dtype=np.float64)
    rng = np.random.default_rng(int(seed) + 17 * (CH5_DATASET_KEYS.index(key) + 1))
    weights = np.zeros((int(n_reps), 3), dtype=np.float64)
    for i in range(int(n_reps)):
        yb = rng.binomial(1, ch6_sigmoid(Xd @ wt)).astype(np.float64)
        w, _ = ch6_fit(Xd, yb, ridge=float(ridge))
        weights[i] = w
    angles = np.array([ch6_boundary_angle_deg(w) for w in weights])
    w_obs, _ = ch6_fit(Xd, y_obs.astype(np.float64), ridge=float(ridge))
    return {
        "key": key,
        "study": np.asarray(study, dtype=np.float64),
        "exam": np.asarray(exam, dtype=np.float64),
        "y_obs": np.asarray(y_obs, dtype=np.float64),
        "w_true": wt,
        "w_obs": w_obs,
        "weights": weights,
        "angles": angles,
        "ridge": float(ridge),
        "xlim": ch5_plot_limits(key)[0],
        "ylim": ch5_plot_limits(key)[1],
    }


def ch6_threshold_segments(w, xlim, ylim, *, n=2):
    """Return (xs, ys) polyline for w1·x + w2·y + b = 0 clipped to the plot box."""
    del n
    w1, w2, b = float(w[0]), float(w[1]), float(w[2])
    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    pts = []
    if abs(w2) > 1e-12:
        for x in (x0, x1):
            y = -(w1 * x + b) / w2
            if y0 - 1e-9 <= y <= y1 + 1e-9:
                pts.append((x, float(np.clip(y, y0, y1))))
    if abs(w1) > 1e-12:
        for y in (y0, y1):
            x = -(w2 * y + b) / w1
            if x0 - 1e-9 <= x <= x1 + 1e-9:
                pts.append((float(np.clip(x, x0, x1)), y))
    if len(pts) < 2:
        return np.array([x0, x1]), np.array([y0, y1])
    arr = np.unique(np.round(np.asarray(pts, dtype=np.float64), 6), axis=0)
    if len(arr) < 2:
        return arr[:, 0], arr[:, 1]
    d = arr[-1] - arr[0]
    order = np.argsort(arr @ d)
    arr = arr[order]
    return arr[[0, -1], 0], arr[[0, -1], 1]


def ch6_rel_likelihood_w12(
    Xd,
    y,
    w_hat,
    *,
    ridge=0.0,
    b_fixed=None,
    bounds=CH6_VIEW_BOUNDS_W12,
    grid=CH6_LR_GRID,
):
    """Relative likelihood L(w)/L(ŵ) on (w_ST, w_EL) with b fixed at the MLE."""
    w1, w2, Z_lr = ch6_lr_grid_w12(
        Xd, y, w_hat, ridge=ridge, b_fixed=b_fixed, bounds=bounds, grid=grid,
    )
    rel = np.exp(-0.5 * np.clip(Z_lr, 0.0, 60.0))
    W1, W2 = np.meshgrid(w1, w2)
    return {
        "w1": w1,
        "w2": w2,
        "W1": W1,
        "W2": W2,
        "Z": rel,
        "Z_lr": Z_lr,
        "z_lim": (0.0, CH6_SURFACE_Z_HI),
    }


def ch6_adaptive_hist_bins(
    weights,
    *,
    bounds=CH6_VIEW_BOUNDS_W12,
    base=CH6_HIST_BINS,
    bins_max=CH6_HIST_BINS_MAX,
    target_cluster_bins=CH6_HIST_TARGET_CLUSTER_BINS,
):
    """More bins when MLE landings cluster tightly (avoids a single fat pillar)."""
    Ws = np.asarray(weights, dtype=np.float64)
    base_n = int(base)
    max_n = int(bins_max)
    if Ws.ndim != 2 or Ws.shape[0] < 3 or Ws.shape[1] < 2:
        return base_n
    span = float(max(
        float(np.ptp(Ws[:, 0])),
        float(np.ptp(Ws[:, 1])),
        2.5 * float(np.std(Ws[:, 0])),
        2.5 * float(np.std(Ws[:, 1])),
        1e-3,
    ))
    full = float(max(bounds[1] - bounds[0], bounds[3] - bounds[2], 1e-6))
    # full/bins ≈ span/target  →  bins ≈ full * target / span
    bins = int(round(full * float(target_cluster_bins) / span))
    return int(np.clip(bins, base_n, max_n))


def ch6_landing_histogram(
    weights,
    *,
    bounds=CH6_VIEW_BOUNDS_W12,
    bins=None,
    adaptive=True,
):
    """2D histogram of MLE landings on (w_ST, w_EL), normalized to peak 1.

    When ``adaptive`` and landings are tight, bin count increases so the cloud
    resolves into several pillars instead of one block.
    """
    Ws = np.asarray(weights, dtype=np.float64)
    if bins is None:
        bins = (
            ch6_adaptive_hist_bins(Ws, bounds=bounds)
            if adaptive else int(CH6_HIST_BINS)
        )
    bins = int(bins)
    if Ws.size == 0:
        w1 = np.linspace(bounds[0], bounds[1], bins)
        w2 = np.linspace(bounds[2], bounds[3], bins)
        W1, W2 = np.meshgrid(w1, w2)
        Z = np.zeros_like(W1)
        return {"w1": w1, "w2": w2, "W1": W1, "W2": W2, "Z": Z, "counts": Z.copy(), "bins": bins}
    counts, xedges, yedges = np.histogram2d(
        Ws[:, 0], Ws[:, 1],
        bins=bins,
        range=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
    )
    # histogram2d returns shape (nx, ny) with x along axis 0; meshgrid needs (ny, nx)
    counts = counts.T
    w1 = 0.5 * (xedges[:-1] + xedges[1:])
    w2 = 0.5 * (yedges[:-1] + yedges[1:])
    W1, W2 = np.meshgrid(w1, w2)
    peak = float(counts.max()) if counts.size else 1.0
    Z = counts.astype(np.float64) / max(peak, 1.0)
    return {
        "w1": w1,
        "w2": w2,
        "W1": W1,
        "W2": W2,
        "Z": Z,
        "counts": counts.astype(np.float64),
        "z_lim": (0.0, CH6_SURFACE_Z_HI),
        "bins": bins,
    }


def ch6_precomputed_resamples(key="D1", n_reps=None, *, seed=1, ridge=CH6_RIDGE):
    """List of (y_resampled, w_fit) for the reel — labels match the displayed line."""
    n_reps = int(CH6_N_REEL if n_reps is None else n_reps)
    study, exam, _ = ch5_unpack_dataset(key)
    Xd = ch6_design(study, exam)
    cloud = ch6_sampling_cloud(key, n_reps=n_reps, seed=seed, ridge=ridge)
    rng = np.random.default_rng(int(seed) + 91)
    p = ch6_sigmoid(Xd @ cloud["w_true"])
    out = []
    for _ in range(n_reps):
        yb = rng.binomial(1, p).astype(np.float64)
        w, _ = ch6_fit(Xd, yb, ridge=float(ridge))
        out.append((yb, w))
    return cloud, out


# ---------------------------------------------------------------------------
# Class-conditional Gaussians inferred from D1 (elongated // to the diagonal)
# ---------------------------------------------------------------------------
# Fail / pass means match D1 sample means. Covariance is shared and elongated
# along u∥ = (1,1)/√2 (parallel to st−el = 0) with a thin perpendicular width
# so the clouds rarely cross the boundary — but still can.

_SQRT2 = float(np.sqrt(2.0))
CH6_U_PARALLEL = np.array([1.0, 1.0], dtype=np.float64) / _SQRT2
CH6_U_PERP = np.array([1.0, -1.0], dtype=np.float64) / _SQRT2  # toward pass from fail

# Empirically: fail μ≈(2.56, 4.78), pass μ≈(4.91, 2.45) on D1 — blended toward
# their midpoint so population MLE landings stay inside the ±3 parameter view.
CH6_MU_FAIL_D1 = np.array([2.56, 4.78], dtype=np.float64)
CH6_MU_PASS_D1 = np.array([4.91, 2.45], dtype=np.float64)
CH6_CLASS_MEAN_BLEND = 0.55  # 0 = D1 sample means, 1 = both at midpoint
_CH6_CLASS_MID = 0.5 * (CH6_MU_FAIL_D1 + CH6_MU_PASS_D1)
CH6_MU_FAIL = CH6_MU_FAIL_D1 + CH6_CLASS_MEAN_BLEND * (_CH6_CLASS_MID - CH6_MU_FAIL_D1)
CH6_MU_PASS = CH6_MU_PASS_D1 + CH6_CLASS_MEAN_BLEND * (_CH6_CLASS_MID - CH6_MU_PASS_D1)
CH6_PI_FAIL = 9.0 / 20.0  # D1 class prior
CH6_PI_PASS = 11.0 / 20.0

# Elongation along the diagonal; wider σ⊥ so the two clouds often overlap.
CH6_SIGMA_PARALLEL = 1.75
CH6_SIGMA_PERP = 1.05


def ch6_class_cov(*, sigma_par=None, sigma_perp=None):
    sp = float(CH6_SIGMA_PARALLEL if sigma_par is None else sigma_par)
    sq = float(CH6_SIGMA_PERP if sigma_perp is None else sigma_perp)
    u = CH6_U_PARALLEL.reshape(2, 1)
    v = CH6_U_PERP.reshape(2, 1)
    return (sp * sp) * (u @ u.T) + (sq * sq) * (v @ v.T)


CH6_CLASS_COV = ch6_class_cov()


def ch6_lda_logistic_weights(
    *,
    mu_fail=None,
    mu_pass=None,
    cov=None,
    pi_fail=None,
    pi_pass=None,
):
    """Logistic weights implied by the shared-covariance Gaussian mixture (LDA).

    When class covariances match, ``P(y|x)`` has linear log-odds in ``x`` and these
    are the population logistic coefficients.  Vary ``cov`` (e.g. separate Σ per class)
    to break the logistic story later.
    """
    mu_f = CH6_MU_FAIL if mu_fail is None else np.asarray(mu_fail, dtype=np.float64)
    mu_p = CH6_MU_PASS if mu_pass is None else np.asarray(mu_pass, dtype=np.float64)
    cov = CH6_CLASS_COV if cov is None else np.asarray(cov, dtype=np.float64)
    pi_f = float(CH6_PI_FAIL if pi_fail is None else pi_fail)
    pi_p = float(CH6_PI_PASS if pi_pass is None else pi_pass)
    sigma_inv = np.linalg.inv(cov)
    w2 = sigma_inv @ (mu_p - mu_f)
    b = np.log(pi_p / pi_f) - 0.5 * (
        float(mu_p @ sigma_inv @ mu_p) - float(mu_f @ sigma_inv @ mu_f)
    )
    return np.array([w2[0], w2[1], b], dtype=np.float64)


CH6_W_TRUE = ch6_lda_logistic_weights()


def ch6_class_gaussian_params():
    """Population parameters for each class (means, shared elongated cov, priors)."""
    cov = np.asarray(CH6_CLASS_COV, dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    return {
        "fail": {
            "label": 0,
            "name": "fail",
            "mu": CH6_MU_FAIL.copy(),
            "cov": cov.copy(),
            "pi": float(CH6_PI_FAIL),
            "color": "#d62728",
        },
        "pass": {
            "label": 1,
            "name": "pass",
            "mu": CH6_MU_PASS.copy(),
            "cov": cov.copy(),
            "pi": float(CH6_PI_PASS),
            "color": "#2ca02c",
        },
        "sigma_parallel": float(CH6_SIGMA_PARALLEL),
        "sigma_perp": float(CH6_SIGMA_PERP),
        "u_parallel": CH6_U_PARALLEL.copy(),
        "u_perp": CH6_U_PERP.copy(),
        "eigs": evals,
        "evecs": evecs,
        "mean_sep_perp": float(
            abs((CH6_MU_PASS - CH6_MU_FAIL) @ CH6_U_PERP) / (CH6_SIGMA_PERP * _SQRT2)
        ),
    }


def ch6_gaussian_ellipse_points(mu, cov, *, n_std=2.0, n=80):
    """2D ellipse at ``n_std`` Mahalanobis radii for plotting."""
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=True)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    axes = vecs @ (float(n_std) * np.sqrt(vals)[:, None] * circle)
    return mu[0] + axes[0], mu[1] + axes[1]


def ch6_sample_class_points(n, *, label, rng, cov=None, mu=None):
    """Sample ``n`` feature vectors from one class Gaussian."""
    rng = np.random.default_rng(rng)
    if label == 0:
        mu = CH6_MU_FAIL if mu is None else np.asarray(mu, dtype=np.float64)
    else:
        mu = CH6_MU_PASS if mu is None else np.asarray(mu, dtype=np.float64)
    cov = CH6_CLASS_COV if cov is None else np.asarray(cov, dtype=np.float64)
    X = rng.multivariate_normal(mu, cov, size=int(n))
    y = np.full(int(n), int(label), dtype=np.float64)
    return X[:, 0], X[:, 1], y


def ch6_sample_population(n_total, *, seed=0, pi_fail=None):
    """Draw a large ghost population from the two-class Gaussian mixture."""
    rng = np.random.default_rng(int(seed))
    pi_f = float(CH6_PI_FAIL if pi_fail is None else pi_fail)
    n_fail = int(rng.binomial(int(n_total), pi_f))
    n_pass = int(n_total) - n_fail
    s0, e0, y0 = ch6_sample_class_points(n_fail, label=0, rng=rng)
    s1, e1, y1 = ch6_sample_class_points(n_pass, label=1, rng=rng)
    study = np.concatenate([s0, s1])
    exam = np.concatenate([e0, e1])
    y = np.concatenate([y0, y1])
    # shuffle for nicer draw order
    perm = rng.permutation(len(y))
    return study[perm], exam[perm], y[perm]


def ch6_draw_classroom_from_population(study, exam, y, n_class, *, rng):
    """Select ``n_class`` students without replacement from a frozen population."""
    rng = np.random.default_rng(rng)
    n_pop = len(y)
    n_class = int(min(max(1, n_class), n_pop))
    idx = rng.choice(n_pop, size=n_class, replace=False)
    return (
        np.asarray(study, dtype=np.float64)[idx],
        np.asarray(exam, dtype=np.float64)[idx],
        np.asarray(y, dtype=np.float64)[idx],
        idx,
    )


def ch6_match_roster_from_population(pop_s, pop_e, pop_y, tgt_s, tgt_e, tgt_y):
    """Pick the population students that best match a target roster (e.g. D1).

    Same-label bipartite matching via Hungarian assignment on Euclidean (st, el)
    distance — one unique population twin per target student.
    """
    from scipy.optimize import linear_sum_assignment

    pop_s = np.asarray(pop_s, dtype=np.float64)
    pop_e = np.asarray(pop_e, dtype=np.float64)
    pop_y = np.asarray(pop_y, dtype=np.float64)
    tgt_s = np.asarray(tgt_s, dtype=np.float64)
    tgt_e = np.asarray(tgt_e, dtype=np.float64)
    tgt_y = np.asarray(tgt_y, dtype=np.float64)
    pop_xy = np.column_stack([pop_s, pop_e])
    tgt_xy = np.column_stack([tgt_s, tgt_e])
    matched = np.empty(len(tgt_y), dtype=np.int64)
    for lab in (0, 1):
        t_idx = np.flatnonzero(tgt_y == lab)
        p_idx = np.flatnonzero(pop_y == lab)
        if t_idx.size == 0:
            continue
        if p_idx.size < t_idx.size:
            raise ValueError(
                f"population has {p_idx.size} label={lab} students; "
                f"need ≥ {t_idx.size} to match the roster"
            )
        # cost[i, j] = distance(target_i, pop_j)
        diff = tgt_xy[t_idx, None, :] - pop_xy[p_idx][None, :, :]
        cost = np.linalg.norm(diff, axis=-1)
        rows, cols = linear_sum_assignment(cost)
        matched[t_idx[rows]] = p_idx[cols]
    return pop_s[matched], pop_e[matched], pop_y[matched], matched


def ch6_opening_classroom_from_population(pop_s, pop_e, pop_y, n_class, *, seed):
    """One opening classroom drawn from the frozen population pool."""
    return ch6_draw_classroom_from_population(
        pop_s, pop_e, pop_y, int(n_class), rng=int(seed) + 7,
    )[:3]


def ch6_population_param_cloud_pack(
    n_class,
    *,
    seed=20,
    seed_from_key=None,
    n_reel=None,
):
    """Population-reel classroom landings in (w_ST, w_EL, b) — shared by story builders."""
    from ch5_datasets import CH5_STANDARD_XLIM, CH5_STANDARD_YLIM, ch5_unpack_dataset

    xlim, ylim = CH5_STANDARD_XLIM, CH5_STANDARD_YLIM
    n_reel = int(CH6_N_POP_REEL if n_reel is None else n_reel)
    pop_s, pop_e, pop_y = ch6_sample_population(CH6_POP_SIZE, seed=int(seed) + 11)

    if seed_from_key is not None:
        tgt_s, tgt_e, tgt_y = ch5_unpack_dataset(str(seed_from_key))
        open_s, open_e, open_y = (
            np.asarray(tgt_s, dtype=np.float64),
            np.asarray(tgt_e, dtype=np.float64),
            np.asarray(tgt_y, dtype=np.float64),
        )
        match_s, match_e, match_y, _ = ch6_match_roster_from_population(
            pop_s, pop_e, pop_y, open_s, open_e, open_y,
        )
        seed_s, seed_e, seed_y = match_s, match_e, match_y
    else:
        open_s, open_e, open_y = ch6_opening_classroom_from_population(
            pop_s, pop_e, pop_y, n_class, seed=seed,
        )
        seed_s, seed_e, seed_y = open_s, open_e, open_y
        match_s = match_e = match_y = None

    w_open, _ = ch6_fit_population_classroom(open_s, open_e, open_y)
    w_seed, _ = ch6_fit_population_classroom(seed_s, seed_e, seed_y)

    rng = np.random.default_rng(int(seed) + 99)
    ghosts: list[np.ndarray] = []
    landed: list[np.ndarray] = []
    reel_steps: list[dict] = []
    for i in range(n_reel):
        if i == 0 and match_s is not None:
            cs, ce, cy = seed_s, seed_e, seed_y
        else:
            cs, ce, cy, _ = ch6_draw_classroom_from_population(
                pop_s, pop_e, pop_y, int(n_class), rng=rng,
            )
        w, _ = ch6_fit_population_classroom(cs, ce, cy)
        landed.append(w)
        reel_steps.append({
            "study": cs, "exam": ce, "y": cy, "w": w,
            "markers": np.asarray(landed), "ghosts": list(ghosts),
        })
        ghosts.append(w)

    return {
        "xlim": xlim,
        "ylim": ylim,
        "pop_s": pop_s,
        "pop_e": pop_e,
        "pop_y": pop_y,
        "open_s": open_s,
        "open_e": open_e,
        "open_y": open_y,
        "match_s": match_s,
        "match_e": match_e,
        "match_y": match_y,
        "w_open": w_open,
        "w_seed": w_seed,
        "landed": np.asarray(landed, dtype=np.float64),
        "reel_steps": reel_steps,
        "n_class": int(n_class),
        "seed": int(seed),
        "seed_from_key": seed_from_key,
    }


def ch6_asymptotic_mle_cov(inverse_hessians, n_class):
    """Asymptotic sampling covariance: ``E[H^{-1}] / n`` over MC classrooms."""
    invs = np.asarray(inverse_hessians, dtype=np.float64)
    if invs.ndim != 3 or len(invs) == 0:
        return np.eye(3, dtype=np.float64)
    return np.mean(invs, axis=0) / float(max(int(n_class), 1))


def ch6_population_reel_asymptotic_cov(reel_steps, n_class, *, ridge=None):
    """Asymptotic Cov(ŵ) from a population-reel trace of classroom fits."""
    ridge = CH6_POPULATION_FIT_RIDGE if ridge is None else float(ridge)
    invs: list[np.ndarray] = []
    for step in reel_steps:
        w = np.asarray(step["w"], dtype=np.float64)
        cs = np.asarray(step["study"], dtype=np.float64)
        ce = np.asarray(step["exam"], dtype=np.float64)
        cy = np.asarray(step["y"], dtype=np.float64)
        Xd = ch6_design(cs, ce)
        H = ch6_observed_information(w, Xd, cy, ridge=ridge)
        invs.append(np.linalg.inv(H))
    return ch6_asymptotic_mle_cov(invs, n_class)


def ch6_make_seed_classroom(n_class, *, seed=0):
    """One 'observed' classroom of size ``n_class`` from the population model."""
    # Draw a modest pool then take a stratified-ish sample via mixture.
    pop_s, pop_e, pop_y = ch6_sample_population(max(int(n_class) * 8, 40), seed=seed)
    return ch6_draw_classroom_from_population(
        pop_s, pop_e, pop_y, n_class, rng=seed + 3,
    )[:3]


CH6_N_CLASS_BASE = 20
CH6_N_CLASS_SMALL = 8
CH6_N_CLASS_LARGE = 60
CH6_N_CLASS_XLARGE = 300
CH6_POP_MULTIPLIER = 100
# Fixed ghost population for all classroom sizes (do not scale with ``n_class``).
CH6_POP_SIZE = int(CH6_N_CLASS_LARGE) * int(CH6_POP_MULTIPLIER)
CH6_N_POP_REEL = 72 if _CH3_DRAFT else 144

# Dist-overlay clips (n=8 / 20 / 60 / 300) share one calibrated parameter cube.
CH6_POPULATION_DIST_N_REEL = 1000
CH6_POPULATION_DIST_CAL_SPECS = (
    (CH6_N_CLASS_SMALL, 8, None),
    (CH6_N_CLASS_BASE, 17, "D1"),
    (CH6_N_CLASS_LARGE, 60, None),
    (CH6_N_CLASS_XLARGE, 300, None),
)


@lru_cache(maxsize=1)
def ch6_population_param_axis_lim(
    *,
    n_reel=CH6_POPULATION_DIST_N_REEL,
    percentile=99.5,
    pad_frac=0.12,
):
    """Symmetric (w_ST, w_EL, b) limits containing all dist-reel landings."""
    weights: list[np.ndarray] = []
    for n_class, seed, key in CH6_POPULATION_DIST_CAL_SPECS:
        kw = {"n_reel": int(n_reel)}
        if key is not None:
            kw["seed_from_key"] = key
        pack = ch6_population_param_cloud_pack(n_class, seed=seed, **kw)
        weights.append(np.asarray(pack["landed"], dtype=np.float64))
        weights.append(np.stack([pack["w_open"], pack["w_seed"]], axis=0))
    W = np.vstack(weights)
    mag = float(np.percentile(np.abs(W), float(percentile)))
    lim = float(np.ceil(mag * (1.0 + float(pad_frac)) * 2.0) / 2.0)
    lim = max(lim, 1.0)
    return (-lim, lim)


CH6_POPULATION_PARAM_AXIS_LIM = ch6_population_param_axis_lim()


def ch6_population_n_sweep_ns():
    """Sample sizes for the n=6…500 population cloud sweep reel."""
    ns = list(range(6, 51))
    ns.extend(range(51, 101, 5))
    if 100 not in ns:
        ns.append(100)
    ns.extend(range(101, 501, 10))
    if 500 not in ns:
        ns.append(500)
    return ns


CH6_POPULATION_SWEEP_N_REEL_MIN = 500
CH6_POPULATION_SWEEP_N_REEL_MAX = 3000
CH6_POPULATION_SWEEP_N_LO = 6
CH6_POPULATION_SWEEP_N_HI = 500


def ch6_population_n_sweep_n_reel(n_class):
    """More classroom draws at larger ``n`` (denser cloud when spread is tighter)."""
    n = int(max(CH6_POPULATION_SWEEP_N_LO, min(CH6_POPULATION_SWEEP_N_HI, int(n_class))))
    t = (n - CH6_POPULATION_SWEEP_N_LO) / float(CH6_POPULATION_SWEEP_N_HI - CH6_POPULATION_SWEEP_N_LO)
    return int(round(
        CH6_POPULATION_SWEEP_N_REEL_MIN
        + t * (CH6_POPULATION_SWEEP_N_REEL_MAX - CH6_POPULATION_SWEEP_N_REEL_MIN)
    ))


CH6_POPULATION_SWEEP_MARKER_S_LO = 28.0  # default cloud marker at n=6
CH6_POPULATION_SWEEP_MARKER_S_HI = 5.0
CH6_POPULATION_SWEEP_MARKER_SHRINK_POWER = 1.75


def ch6_population_n_sweep_marker_size(n_class):
    """Sublinear marker shrink — full size at small ``n``, much smaller at n=500."""
    n = int(max(CH6_POPULATION_SWEEP_N_LO, min(CH6_POPULATION_SWEEP_N_HI, int(n_class))))
    t = (n - CH6_POPULATION_SWEEP_N_LO) / float(CH6_POPULATION_SWEEP_N_HI - CH6_POPULATION_SWEEP_N_LO)
    u = float(t) ** float(CH6_POPULATION_SWEEP_MARKER_SHRINK_POWER)
    s = CH6_POPULATION_SWEEP_MARKER_S_LO + (
        (CH6_POPULATION_SWEEP_MARKER_S_HI - CH6_POPULATION_SWEEP_MARKER_S_LO) * u
    )
    return float(max(CH6_POPULATION_SWEEP_MARKER_S_HI, s))


CH6_POPULATION_SWEEP_DENSITY_PAD_LO = 0.48
CH6_POPULATION_SWEEP_DENSITY_PAD_HI = 0.26
CH6_POPULATION_SWEEP_DENSITY_PAD_POWER = 1.75


def ch6_population_n_sweep_density_bar_pad(n_class):
    """Thinner histogram bars at larger ``n`` (matches marker shrink curve)."""
    n = int(max(CH6_POPULATION_SWEEP_N_LO, min(CH6_POPULATION_SWEEP_N_HI, int(n_class))))
    t = (n - CH6_POPULATION_SWEEP_N_LO) / float(CH6_POPULATION_SWEEP_N_HI - CH6_POPULATION_SWEEP_N_LO)
    u = float(t) ** float(CH6_POPULATION_SWEEP_DENSITY_PAD_POWER)
    pad = CH6_POPULATION_SWEEP_DENSITY_PAD_LO + (
        (CH6_POPULATION_SWEEP_DENSITY_PAD_HI - CH6_POPULATION_SWEEP_DENSITY_PAD_LO) * u
    )
    return float(max(CH6_POPULATION_SWEEP_DENSITY_PAD_HI, pad))


CH6_POPULATION_SWEEP_DENSITY_BINS_LO = 16
CH6_POPULATION_SWEEP_DENSITY_BINS_HI = 96
CH6_POPULATION_SWEEP_DENSITY_BINS_DRAFT_LO = 14
CH6_POPULATION_SWEEP_DENSITY_BINS_DRAFT_HI = 72
CH6_POPULATION_SWEEP_DENSITY_BINS_POWER = 1.75


def ch6_population_n_sweep_density_bins(n_class, *, draft=False):
    """More histogram bins at larger ``n`` (finer grid as bars get thinner)."""
    lo = CH6_POPULATION_SWEEP_DENSITY_BINS_DRAFT_LO if draft else CH6_POPULATION_SWEEP_DENSITY_BINS_LO
    hi = CH6_POPULATION_SWEEP_DENSITY_BINS_DRAFT_HI if draft else CH6_POPULATION_SWEEP_DENSITY_BINS_HI
    n = int(max(CH6_POPULATION_SWEEP_N_LO, min(CH6_POPULATION_SWEEP_N_HI, int(n_class))))
    t = (n - CH6_POPULATION_SWEEP_N_LO) / float(CH6_POPULATION_SWEEP_N_HI - CH6_POPULATION_SWEEP_N_LO)
    u = float(t) ** float(CH6_POPULATION_SWEEP_DENSITY_BINS_POWER)
    bins = lo + u * (hi - lo)
    return int(max(lo, round(bins)))


_SWEEP_STATE_CACHE: dict[tuple, tuple] = {}


def ch6_population_n_sweep_states(
    *,
    seed=17,
    pad_frac=0.12,
):
    """Precompute final-reel state for every ``n`` in ``ch6_population_n_sweep_ns``.

    Returns ``(axis_lim, states)`` where each state dict holds the left/right
    panel inputs for one frame (final cloud after ``n_reel`` classrooms).
    """
    key = (int(seed), float(pad_frac))
    cached = _SWEEP_STATE_CACHE.get(key)
    if cached is not None:
        return cached

    states: list[dict] = []
    weights: list[np.ndarray] = []
    for n_class in ch6_population_n_sweep_ns():
        n_reel = ch6_population_n_sweep_n_reel(n_class)
        pack = ch6_population_param_cloud_pack(
            int(n_class), seed=int(seed), n_reel=int(n_reel),
        )
        last = pack["reel_steps"][-1]
        landed = np.asarray(pack["landed"], dtype=np.float64)
        stats = ch6_param_stats(landed)
        weights.append(landed)
        weights.append(np.stack([pack["w_open"], pack["w_seed"]], axis=0))
        weights.append(stats["mean"].reshape(1, 3))
        states.append({
            "n": int(n_class),
            "n_reel": int(n_reel),
            "xlim": pack["xlim"],
            "ylim": pack["ylim"],
            "pop_s": pack["pop_s"],
            "pop_e": pack["pop_e"],
            "pop_y": pack["pop_y"],
            "cs": last["study"],
            "ce": last["exam"],
            "cy": last["y"],
            "w": last["w"],
            "markers": last["markers"],
            "ghosts": last["ghosts"],
            "mean": stats["mean"],
        })
    W = np.vstack(weights)
    mag = float(np.abs(W).max())
    lim = float(np.ceil(mag * (1.0 + float(pad_frac)) * 2.0) / 2.0)
    lim = max(lim, 1.0)
    axis_lim = (-lim, lim)
    out = (axis_lim, states)
    _SWEEP_STATE_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# Script pedagogy helpers — mean / variance / covariance / expected likelihood
# ---------------------------------------------------------------------------

CH6_MEAN_LINE_COLOR = "#16a085"
CH6_VAR_COLOR = "#e74c3c"
CH6_COV_COLOR = "#8e44ad"
CH6_STEEP_COLOR = "#27ae60"
CH6_FLAT_COLOR = "#e67e22"
CH6_MATRIX_BG = "#f7f9fb"


def ch6_gaussian_ellipsoid_loops(
    mean,
    cov,
    *,
    masses=(0.50, 0.68, 0.95),
    df=3,
    n_u=48,
    n_v=24,
):
    """Wireframe latitude rings for 3D Gaussian confidence ellipsoids."""
    from scipy.stats import chi2

    mu = np.asarray(mean, dtype=np.float64).reshape(3)
    cov = np.asarray(cov, dtype=np.float64).reshape(3, 3)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    u = np.linspace(0.0, 2.0 * np.pi, int(n_u), endpoint=False)
    v = np.linspace(0.0, np.pi, int(n_v), endpoint=True)
    out: list[list[np.ndarray]] = []
    for mass in masses:
        r = float(np.sqrt(chi2.ppf(float(mass), int(df))))
        scales = r * np.sqrt(vals)
        rings: list[np.ndarray] = []
        for vv in v:
            ring = []
            for uu in u:
                sph = np.array([
                    np.sin(vv) * np.cos(uu),
                    np.sin(vv) * np.sin(uu),
                    np.cos(vv),
                ], dtype=np.float64)
                ring.append(mu + vecs @ (scales * sph))
            rings.append(np.asarray(ring, dtype=np.float64).T)
        out.append(rings)
    return out


def ch6_population_sampling_distribution(
    n_class,
    *,
    seed=20,
    seed_from_key=None,
    n_mc=5000,
    ridge=CH6_POPULATION_FIT_RIDGE,
):
    """Mean and covariance of classroom MLE weights under the Ch6 mixture population.

    Monte Carlo over ``n_mc`` classrooms of size ``n_class`` drawn from the same
    frozen population / RNG protocol as the population-reel clips.

    Uses ``CH6_POPULATION_FIT_RIDGE`` (not ``CH6_RIDGE``) and empirical ``np.cov``
    of the fitted weights.
    """
    from ch5_datasets import ch5_unpack_dataset

    pop_s, pop_e, pop_y = ch6_sample_population(CH6_POP_SIZE, seed=int(seed) + 11)

    if seed_from_key is not None:
        tgt_s, tgt_e, tgt_y = ch5_unpack_dataset(str(seed_from_key))
        match_s, match_e, match_y, _ = ch6_match_roster_from_population(
            pop_s, pop_e, pop_y,
            np.asarray(tgt_s, dtype=np.float64),
            np.asarray(tgt_e, dtype=np.float64),
            np.asarray(tgt_y, dtype=np.float64),
        )
    else:
        match_s = match_e = match_y = None

    rng = np.random.default_rng(int(seed) + 99)
    hats: list[np.ndarray] = []
    for i in range(int(n_mc)):
        if i == 0 and match_s is not None:
            cs, ce, cy = match_s, match_e, match_y
        else:
            cs, ce, cy, _ = ch6_draw_classroom_from_population(
                pop_s, pop_e, pop_y, int(n_class), rng=rng,
            )
        w, _ = ch6_fit_dataset(cs, ce, cy, ridge=float(ridge))
        hats.append(w)
    Ws = np.asarray(hats, dtype=np.float64)
    mu = Ws.mean(axis=0)
    cov = np.cov(Ws.T, ddof=1) if len(Ws) >= 2 else np.eye(3, dtype=np.float64)
    return {
        "mean": mu,
        "cov": cov,
        "weights": Ws,
        "pop_s": pop_s,
        "pop_e": pop_e,
        "pop_y": pop_y,
        "n_class": int(n_class),
        "n_mc": int(n_mc),
    }


def ch6_param_stats(weights):
    """Empirical mean + covariance of parameter landings (n, 3)."""
    Ws = np.asarray(weights, dtype=np.float64)
    if Ws.ndim != 2 or Ws.shape[0] < 2:
        mu = Ws.reshape(-1)[:3] if Ws.size else np.zeros(3)
        return {
            "mean": np.asarray(mu, dtype=np.float64).reshape(3),
            "cov": np.eye(3),
            "var": np.ones(3),
            "std": np.ones(3),
        }
    mu = Ws.mean(axis=0)
    C = np.cov(Ws.T, ddof=1)
    var = np.diag(C)
    return {
        "mean": mu,
        "cov": C,
        "var": var,
        "std": np.sqrt(np.clip(var, 0.0, None)),
    }


def ch6_iso_vs_corr_clouds(n=180, *, seed=42, sx=0.55, sy=0.55, rho=0.82):
    """Two (w1,w2) clouds with identical marginal variances: circular vs elliptical.

    Returns dict with ``iso`` (rho=0) and ``corr`` (correlated) arrays of shape (n, 3)
    with b fixed at 0 for visualization in the w12 plane.
    """
    rng = np.random.default_rng(int(seed))
    n = int(n)
    # Independent isotropic
    iso_xy = rng.normal(0.0, 1.0, size=(n, 2)) * np.array([sx, sy])
    # Correlated with same marginal std
    z = rng.normal(0.0, 1.0, size=(n, 2))
    L = np.array([[sx, 0.0], [rho * sy, sy * np.sqrt(max(1.0 - rho * rho, 1e-9))]])
    corr_xy = z @ L.T
    # Center both near a plausible MLE
    center = np.array([0.85, -0.85, 0.0])
    iso = np.column_stack([iso_xy[:, 0] + center[0], iso_xy[:, 1] + center[1],
                           np.full(n, center[2])])
    corr = np.column_stack([corr_xy[:, 0] + center[0], corr_xy[:, 1] + center[1],
                            np.full(n, center[2])])
    return {
        "iso": iso,
        "corr": corr,
        "center": center,
        "sx": float(sx),
        "sy": float(sy),
        "rho": float(rho),
        "cov_iso": np.diag([sx * sx, sy * sy, 1e-6]),
        "cov_corr": np.array([
            [sx * sx, rho * sx * sy, 0.0],
            [rho * sx * sy, sy * sy, 0.0],
            [0.0, 0.0, 1e-6],
        ]),
    }


def ch6_rescale_study_hours_to_seconds(weights):
    """If study time is measured in seconds, w_ST shrinks by 1/3600."""
    Ws = np.asarray(weights, dtype=np.float64).copy()
    Ws[:, 0] = Ws[:, 0] / 3600.0
    return Ws


def ch6_expected_rel_likelihood(
    key="D1",
    n_avg=24,
    *,
    seed=5,
    ridge=CH6_RIDGE,
    grid=None,
):
    """Average relative-likelihood surfaces across parametric resamples.

    Each classroom's surface is evaluated on a common (w1,w2) grid with b fixed
    at the mean of the per-classroom MLEs (so shapes are comparable).
    """
    cloud = ch6_sampling_cloud(key, n_reps=int(n_avg), seed=seed, ridge=ridge)
    study, exam = cloud["study"], cloud["exam"]
    Xd = ch6_design(study, exam)
    Wt = cloud["weights"]
    b_fixed = float(np.mean(Wt[:, 2]))
    grid = int(CH6_LR_GRID if grid is None else grid)
    rng = np.random.default_rng(int(seed) + 201)
    p = ch6_sigmoid(Xd @ cloud["w_true"])
    acc = None
    hats = []
    for i in range(int(n_avg)):
        yb = rng.binomial(1, p).astype(np.float64)
        w_hat, _ = ch6_fit(Xd, yb, ridge=float(ridge))
        hats.append(w_hat)
        surf = ch6_rel_likelihood_w12(
            Xd, yb, w_hat, ridge=float(ridge), b_fixed=b_fixed, grid=grid,
        )
        Z = np.asarray(surf["Z"], dtype=np.float64)
        acc = Z.copy() if acc is None else acc + Z
    Z_avg = acc / float(n_avg)
    peak = float(np.nanmax(Z_avg))
    if peak > 1e-12:
        Z_avg = Z_avg / peak
    W1, W2 = surf["W1"], surf["W2"]
    return {
        "W1": W1,
        "W2": W2,
        "Z": Z_avg,
        "z_lim": (0.0, CH6_SURFACE_Z_HI),
        "hats": np.asarray(hats, dtype=np.float64),
        "mean_hat": np.mean(hats, axis=0),
        "b_fixed": b_fixed,
        "cloud": cloud,
        "surfaces_meta": {"n_avg": int(n_avg), "grid": grid},
    }


def ch6_classroom_likelihood_stack(
    key="D1",
    n_show=6,
    *,
    seed=5,
    ridge=CH6_RIDGE,
    grid=None,
):
    """Return a short list of relative-likelihood surfaces for different classrooms."""
    cloud = ch6_sampling_cloud(key, n_reps=max(int(n_show), 8), seed=seed, ridge=ridge)
    study, exam = cloud["study"], cloud["exam"]
    Xd = ch6_design(study, exam)
    b_fixed = float(np.mean(cloud["weights"][:, 2]))
    grid = int(CH6_LR_GRID if grid is None else grid)
    rng = np.random.default_rng(int(seed) + 77)
    p = ch6_sigmoid(Xd @ cloud["w_true"])
    out = []
    for _ in range(int(n_show)):
        yb = rng.binomial(1, p).astype(np.float64)
        w_hat, _ = ch6_fit(Xd, yb, ridge=float(ridge))
        surf = ch6_rel_likelihood_w12(
            Xd, yb, w_hat, ridge=float(ridge), b_fixed=b_fixed, grid=grid,
        )
        out.append({"y": yb, "w": w_hat, "surface": surf})
    return cloud, out


def ch6_average_rel_likelihood_population(
    *,
    n_class: int,
    n_avg: int = 500,
    seed: int = 45,
    ridge=CH6_RIDGE,
    grid=None,
    seed_from_key: str | None = None,
):
    """Average relative-likelihood over ``n_avg`` classrooms of size ``n_class``.

    Returns the mean surface, mean MLE, one held-out classroom (the 'observed'
    dataset), and that classroom's surface on the shared grid.
    """
    from ch5_datasets import CH5_STANDARD_XLIM, CH5_STANDARD_YLIM

    grid = int(CH6_LR_GRID if grid is None else grid)
    n_pop = int(CH6_POP_SIZE)
    n_avg = int(n_avg)
    pop_s, pop_e, pop_y = ch6_sample_population(n_pop, seed=int(seed) + 11)
    rng = np.random.default_rng(int(seed) + 99)

    if seed_from_key is not None:
        from ch5_datasets import ch5_unpack_dataset
        tgt_s, tgt_e, tgt_y = ch5_unpack_dataset(str(seed_from_key))
        obs_s, obs_e, obs_y, _ = ch6_match_roster_from_population(
            pop_s, pop_e, pop_y,
            np.asarray(tgt_s, dtype=np.float64),
            np.asarray(tgt_e, dtype=np.float64),
            np.asarray(tgt_y, dtype=np.float64),
        )
    else:
        obs_s, obs_e, obs_y, _ = ch6_draw_classroom_from_population(
            pop_s, pop_e, pop_y, int(n_class), rng=rng,
        )

    # Fit many classrooms for the average landscape (+ mean MLE).
    hats = []
    classrooms = []
    for i in range(int(n_avg)):
        if i == 0:
            cs, ce, cy = obs_s, obs_e, obs_y
        else:
            cs, ce, cy, _ = ch6_draw_classroom_from_population(
                pop_s, pop_e, pop_y, int(n_class), rng=rng,
            )
        w_hat, _ = ch6_fit_dataset(cs, ce, cy, ridge=float(ridge))
        hats.append(w_hat)
        classrooms.append((cs, ce, cy, w_hat))
    hats = np.asarray(hats, dtype=np.float64)
    b_fixed = float(np.mean(hats[:, 2]))
    mean_hat = np.mean(hats, axis=0)

    acc = None
    surf_last = None
    for cs, ce, cy, w_hat in classrooms:
        Xd = ch6_design(cs, ce)
        surf = ch6_rel_likelihood_w12(
            Xd, cy, w_hat, ridge=float(ridge), b_fixed=b_fixed, grid=grid,
        )
        Z = np.asarray(surf["Z"], dtype=np.float64)
        acc = Z.copy() if acc is None else acc + Z
        surf_last = surf
    Z_avg = acc / float(n_avg)
    peak = float(np.nanmax(Z_avg))
    if peak > 1e-12:
        Z_avg = Z_avg / peak

    # Observed classroom surface on the same grid / b_fixed.
    Xd_obs = ch6_design(obs_s, obs_e)
    w_obs, _ = ch6_fit_dataset(obs_s, obs_e, obs_y, ridge=float(ridge))
    surf_obs = ch6_rel_likelihood_w12(
        Xd_obs, obs_y, w_obs, ridge=float(ridge), b_fixed=b_fixed, grid=grid,
    )
    avg_surf = {
        "W1": surf_last["W1"],
        "W2": surf_last["W2"],
        "Z": Z_avg,
        "z_lim": (0.0, CH6_SURFACE_Z_HI),
    }
    return {
        "avg_surf": avg_surf,
        "obs_surf": surf_obs,
        "mean_hat": mean_hat,
        "w_obs": w_obs,
        "hats": hats,
        "b_fixed": b_fixed,
        "study": np.asarray(obs_s, dtype=np.float64),
        "exam": np.asarray(obs_e, dtype=np.float64),
        "y": np.asarray(obs_y, dtype=np.float64),
        "xlim": CH5_STANDARD_XLIM,
        "ylim": CH5_STANDARD_YLIM,
        "n_avg": int(n_avg),
        "n_class": int(n_class),
        "pop_s": pop_s,
        "pop_e": pop_e,
        "pop_y": pop_y,
    }


def ch6_hessian_eigen_w12(w, Xd, y, *, ridge=CH6_RIDGE):
    """Eigenframe of the observed information in the (w_ST, w_EL) plane (b fixed via Schur)."""
    H = ch6_observed_information(w, Xd, y, ridge=ridge)
    # Conditional information on (w1,w2) given b — Schur complement of H
    Hbb = H[2, 2]
    H12 = H[:2, 2]
    H22 = H[:2, :2]
    Hcond = H22 - np.outer(H12, H12) / max(float(Hbb), 1e-12)
    vals, vecs = np.linalg.eigh(Hcond)
    order = np.argsort(vals)[::-1]  # steepest first
    vals = vals[order]
    vecs = vecs[:, order]
    cov = ch6_wald_cov(w, Xd, y, ridge=ridge)
    return {
        "H": H,
        "Hcond": Hcond,
        "eigvals": vals,
        "eigvecs": vecs,
        "cov": cov,
        "steep_dir": vecs[:, 0],
        "flat_dir": vecs[:, -1],
    }


def ch6_probe_path(w_hat, direction, *, lengths, b_fixed=None):
    """Parameter points along ±direction from w_hat (2D direction in w12)."""
    w_hat = np.asarray(w_hat, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64).reshape(2)
    d = d / max(np.linalg.norm(d), 1e-12)
    b = float(w_hat[2] if b_fixed is None else b_fixed)
    pts = []
    for t in lengths:
        pts.append(np.array([w_hat[0] + t * d[0], w_hat[1] + t * d[1], b]))
    return np.asarray(pts, dtype=np.float64)
