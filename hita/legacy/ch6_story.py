"""Chapter 6 — frequentist sampling variability on the Ch4 duo (2D | 3D) stage."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch

from ch5_core import CH5_BELIEF_SURFACE_ALPHA, CH5_VIEW_BOUNDS, ch5_hq_land_elev, ch5_plot_belief_surface_with_grid
from ch5_datasets import ch5_plot_limits, ch5_unpack_dataset
from ch5_layout import ch5_uniform_belief_facecolors, ch5_uniform_belief_rgba_at_pdf
from ch6_frequentist import (
    CH6_CLOUD_COLOR,
    CH6_CONF_MASS,
    CH6_COV_COLOR,
    CH6_FLAT_COLOR,
    CH6_GHOST_COLOR,
    CH6_LINE_COLOR,
    CH6_LR_COLOR,
    CH6_MEAN_LINE_COLOR,
    CH6_MS,
    CH6_N_CLASS_BASE,
    CH6_N_CLASS_LARGE,
    CH6_N_CLASS_SMALL,
    CH6_N_CLASS_XLARGE,
    CH6_POPULATION_DIST_N_REEL,
    CH6_N_FLASH,
    CH6_N_GHOST_LINES_SHOW,
    CH6_N_HOLD,
    CH6_N_POP_REEL,
    CH6_N_REEL,
    CH6_N_REPS_CLOUD,
    CH6_N_SEQ_HOLD,
    CH6_POP_MULTIPLIER,
    CH6_POP_SIZE,
    CH6_POPULATION_PARAM_AXIS_LIM,
    CH6_RIDGE,
    CH6_LR_GRID_SMOOTH,
    CH6_STEEP_COLOR,
    CH6_SURFACE_Z_HI,
    CH6_TRUE_LINE_COLOR,
    CH6_VAR_COLOR,
    CH6_VIEW_BOUNDS_W12,
    CH6_W_TRUE,
    CH6_WALD_COLOR,
    ch6_class_gaussian_params,
    ch6_classroom_likelihood_stack,
    ch6_design,
    ch6_draw_classroom_from_population,
    ch6_average_rel_likelihood_population,
    ch6_expected_rel_likelihood,
    ch6_fit,
    ch6_fit_dataset,
    ch6_fit_population_classroom,
    ch6_gaussian_ellipsoid_loops,
    ch6_nll_grad,
    ch6_gaussian_ellipse_points,
    ch6_hessian_eigen_w12,
    ch6_iso_vs_corr_clouds,
    ch6_landing_histogram,
    ch6_lr_threshold,
    ch6_match_roster_from_population,
    ch6_opening_classroom_from_population,
    ch6_p_true,
    ch6_param_stats,
    ch6_population_param_cloud_pack,
    ch6_population_n_sweep_states,
    ch6_population_n_sweep_marker_size,
    ch6_population_n_sweep_n_reel,
    ch6_population_n_sweep_density_bar_pad,
    ch6_population_n_sweep_density_bins,
    ch6_precomputed_resamples,
    ch6_probe_path,
    ch6_rel_likelihood_w12,
    ch6_rescale_study_hours_to_seconds,
    ch6_resample_labels,
    ch6_sample_population,
    ch6_sampling_cloud,
    ch6_sigmoid,
    ch6_threshold_segments,
    ch6_wald_cov,
    ch6_wald_ellipse_w12,
)

_G: dict[str, Any] = {}

# Ghost check/cross: rotate hues toward the ghost-line blue.
_CH6_GHOST_HUE_SHIFT = 0.42
_CH6_GHOST_ICON_ALPHA = 0.72
_CH6_GHOST_ICON_ZOOM = 0.18
_GHOST_ICON_CACHE: dict[str, np.ndarray] = {}


def install(globals_dict: dict[str, Any]) -> None:
    global _G
    _G = globals_dict
    ch6_sampling_cloud.cache_clear()
    _GHOST_ICON_CACHE.clear()
    _G["CH6_EXPORT_SPECS"] = CH6_EXPORT_SPECS
    _G["ch6_export_clip"] = ch6_export_clip


def _g(name: str):
    return _G[name]


def _draft_short(n_full: int, n_draft: int) -> int:
    from ch6_frequentist import _CH3_DRAFT

    return n_draft if _CH3_DRAFT else n_full


def _hold(frame, n):
    return [frame] * int(max(0, n))


def _finish(img, clip_id: str):
    """Full-bleed duo on the chapter-4 export canvas (same as Ch5 landscape clips)."""
    del clip_id
    return _g("compose_tutorial")(
        img,
        right_blocks=[],
        bottom_blocks=[],
        corner_blocks=[],
        right_title="",
        bottom_title="",
        corner_title="",
        layout_u=0.0,
        panel_u=0.0,
        title_write_progress=0.0,
        write_progress=0.0,
        plot_start_rect=_g("CH4_LIK_PLOT_START_RECT"),
        theme="classic_light",
    )


def _fig_to_plot(fig):
    return _g("fig_to_image")(fig, dpi=_g("CH3_ANIM_DPI"))


def _plot_threshold(ax, w, xlim, ylim, *, color, lw=2.0, alpha=1.0, ls="-", zorder=5):
    xs, ys = ch6_threshold_segments(w, xlim, ylim)
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, ls=ls, zorder=zorder, solid_capstyle="round")


def _hue_shift_rgba(rgba: np.ndarray, *, hue_shift: float, alpha_scale: float) -> np.ndarray:
    """Tint opaque icon pixels toward ghost-line blue (keeps check vs cross distinct)."""
    del hue_shift  # kept for call-site compatibility; we blend to a fixed ghost blue
    arr = np.asarray(rgba, dtype=np.float64)
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    out = arr.copy()
    rgb = out[..., :3]
    a = out[..., 3]
    mask = a > 0.05
    if not np.any(mask):
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)
    # Ghost blue matching CH6_GHOST_COLOR (#5dade2)
    target = np.array([0.365, 0.682, 0.886], dtype=np.float64)
    mix = 0.72
    rgb_m = rgb[mask]
    # Preserve a bit of original hue so check stays cooler-greenish and cross warmer.
    rgb[mask] = (1.0 - mix) * rgb_m + mix * target
    out[..., :3] = rgb
    out[..., 3] = a * float(alpha_scale)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _ghost_icon(kind: str) -> np.ndarray:
    key = str(kind)
    cached = _GHOST_ICON_CACHE.get(key)
    if cached is not None:
        return cached
    src = _g("CHECK_ICON") if kind == "check" else _g("CROSS_ICON")
    out = _hue_shift_rgba(
        src, hue_shift=_CH6_GHOST_HUE_SHIFT, alpha_scale=_CH6_GHOST_ICON_ALPHA,
    )
    _GHOST_ICON_CACHE[key] = out
    return out


def _draw_base_dataset(ax, study, exam, y, *, xlim, ylim):
    _g("draw_dataset")(ax, study, exam, y)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _draw_ghost_dataset(ax, study, exam, y, *, zoom=None, offset=0.14):
    """Hue-tinted check/cross icons — slight offset so they don't bury the base roster."""
    zoom = float(_CH6_GHOST_ICON_ZOOM if zoom is None else zoom)
    check = _ghost_icon("check")
    cross = _ghost_icon("cross")
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    off = float(offset)
    for xi, yi, lab in zip(study, exam, y):
        img = check if int(lab) else cross
        ab = AnnotationBbox(
            OffsetImage(img, zoom=zoom),
            (float(xi) + off, float(yi) + off),
            frameon=False,
            zorder=9,
        )
        ax.add_artist(ab)


def _ghost_line_alpha(n_ghosts: int) -> float:
    # More lines → slightly more transparent so the bundle stays readable.
    return float(np.clip(0.55 * (12.0 / max(n_ghosts, 12)), 0.12, 0.45))


def _pick_ghost_lines(weights, n_show=None):
    Ws = np.asarray(weights, dtype=np.float64)
    if Ws.size == 0:
        return []
    n_show = int(CH6_N_GHOST_LINES_SHOW if n_show is None else n_show)
    if len(Ws) <= n_show:
        return [w for w in Ws]
    idx = np.linspace(0, len(Ws) - 1, n_show).astype(int)
    return [Ws[j] for j in idx]


def _style_ax3d(
    ax3d,
    *,
    z_lim=(0.0, CH6_SURFACE_Z_HI),
    zlabel="Likelihood",
    xy_lim=None,
    azim=None,
    elev=None,
    minimal_ui=False,
):
    """Match Ch5 HQ landscape camera (CT view_init + hq elev offset)."""
    dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS
    if xy_lim is not None:
        lo, hi = float(xy_lim[0]), float(xy_lim[1])
        dlo1, dhi1, dlo2, dhi2 = lo, hi, lo, hi
    ax3d.set_xlim(dlo1, dhi1)
    ax3d.set_ylim(dlo2, dhi2)
    ax3d.set_zlim(float(z_lim[0]), float(z_lim[1]))
    if minimal_ui:
        ax3d.set_xlabel("")
        ax3d.set_ylabel("")
        ax3d.set_zlabel("")
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
    else:
        fs = float(_g("AXIS_LABEL_SIZE")) * float(_g("CH3_LIK_3D_AXIS_LABEL_SCALE"))
        ax3d.set_xlabel(r"$w_{\mathrm{ST}}$", fontsize=fs, labelpad=8)
        ax3d.set_ylabel(r"$w_{\mathrm{EL}}$", fontsize=fs, labelpad=8)
        ax3d.set_zlabel(zlabel, fontsize=fs, labelpad=8)
        ax3d.tick_params(labelsize=6)
    _g("ch4_lik_ct_view_init")(ax3d)
    base = float(_g("CH3_LIK_W12_CT_ELEV"))
    ax3d.view_init(
        elev=float(elev if elev is not None else ch5_hq_land_elev(base)),
        azim=float(_g("CH3_LIK_W12_CT_AZIM") if azim is None else azim),
    )


def _draw_sampling_ellipsoids_ch5_style(
    ax3d,
    mean,
    cov,
    *,
    reveal_u=1.0,
    mass=0.95,
    n_layers=None,
):
    """Nested translucent shells like ch5_66 (Gaussian level sets from ``cov``)."""
    from scipy.stats import chi2

    from ch5_core import (
        CH5_ELLIPSOID_EDGE_ALPHAS,
        CH5_ELLIPSOID_FACE_ALPHAS,
        CH5_ELLIPSOID_LAYER_SCALES,
        CH5_ELLIPSOID_MESH_U,
        CH5_ELLIPSOID_MESH_V,
    )
    from ch5_layout import ch5_draw_laplace_ellipsoid

    mu = np.asarray(mean, dtype=np.float64).reshape(3)
    cov = np.asarray(cov, dtype=np.float64).reshape(3, 3)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    base_radii = np.sqrt(float(chi2.ppf(float(mass), 3)) * evals)
    dirs = tuple(evecs[:, i] for i in range(3))
    scales = list(CH5_ELLIPSOID_LAYER_SCALES)
    face_as = list(CH5_ELLIPSOID_FACE_ALPHAS)
    edge_as = list(CH5_ELLIPSOID_EDGE_ALPHAS)
    face_cols = ("#3b82f6", "#60a5fa", "#93c5fd")
    edge_cols = ("#1e3a8a", "#1d4ed8", "#3b82f6")
    n_show = len(scales) if n_layers is None else int(n_layers)
    u = float(np.clip(reveal_u, 0.0, 1.0))
    for i in range(n_show):
        local_u = u if i == n_show - 1 else 1.0
        if local_u <= 1e-4:
            continue
        sc = float(scales[i]) * (0.35 + 0.65 * local_u)
        ch5_draw_laplace_ellipsoid(
            ax3d,
            mu,
            dirs,
            base_radii * sc,
            n_u=int(CH5_ELLIPSOID_MESH_U),
            n_v=int(CH5_ELLIPSOID_MESH_V),
            face_color=face_cols[min(i, len(face_cols) - 1)],
            face_alpha=float(face_as[i]) * local_u,
            edge_color=edge_cols[min(i, len(edge_cols) - 1)],
            edge_alpha=float(edge_as[i]) * local_u,
            zorder=12 + i,
        )


def _draw_gaussian_ellipsoids(ax3d, loops, *, color=CH6_COV_COLOR, alpha=0.85, lw=1.6):
    """Nested 3D ellipsoid wireframes from ``ch6_gaussian_ellipsoid_loops``."""
    if not loops:
        return
    for li, loop in enumerate(loops):
        al = float(alpha) * (0.55 + 0.45 * li / max(len(loops) - 1, 1))
        for ring in loop:
            pts = np.asarray(ring, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
            ax3d.plot(
                pts[0], pts[1], pts[2],
                color=color, lw=lw, alpha=al, zorder=12,
            )


def _project_to_axes_fraction(ax3d, point):
    """Map a 3D data point to normalized (0–1) coordinates on the 3D axes."""
    from mpl_toolkits.mplot3d import proj3d

    mu = np.asarray(point, dtype=np.float64).reshape(3)
    fig = ax3d.figure
    fig.canvas.draw()
    sx, sy, _ = proj3d.proj_transform(
        float(mu[0]), float(mu[1]), float(mu[2]), ax3d.get_proj(),
    )
    disp = ax3d.transData.transform((sx, sy))
    bbox = ax3d.bbox
    if bbox.width <= 1e-6 or bbox.height <= 1e-6:
        return 0.5, 0.5
    return (
        (float(disp[0]) - float(bbox.x0)) / float(bbox.width),
        (float(disp[1]) - float(bbox.y0)) / float(bbox.height),
    )


def _draw_mean_x(ax3d, fig, mean, *, color="#ef4444", halo="#ffffff", fontsize=30):
    """Red × at the sampling-distribution center (screen overlay for mplot3d visibility)."""
    import matplotlib.patheffects as pe

    mu = np.asarray(mean, dtype=np.float64).reshape(3)
    fx, fy = _project_to_axes_fraction(ax3d, mu)
    ax3d.text2D(
        fx, fy, "\u2715",
        transform=ax3d.transAxes,
        fontsize=float(fontsize),
        color=color,
        ha="center",
        va="center",
        fontweight="bold",
        zorder=1000,
        path_effects=[
            pe.withStroke(linewidth=5.0, foreground=halo),
            pe.withStroke(linewidth=2.0, foreground="#991b1b"),
        ],
    )


def _draw_stems(ax3d, hist, *, alpha=1.0):
    """3D histogram pillars — Ch4/Ch5 belief heatmap colors."""
    W1 = np.asarray(hist["W1"], dtype=np.float64)
    W2 = np.asarray(hist["W2"], dtype=np.float64)
    Z = np.asarray(hist["Z"], dtype=np.float64) * float(alpha)
    if W1.size < 2:
        return
    dw1 = float(W1[0, 1] - W1[0, 0]) if W1.shape[1] > 1 else 0.2
    dw2 = float(W2[1, 0] - W2[0, 0]) if W2.shape[0] > 1 else 0.2
    dx = 0.85 * abs(dw1)
    dy = 0.85 * abs(dw2)
    z_lim = (0.0, CH6_SURFACE_Z_HI)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            h = float(Z[i, j])
            if h <= 1e-4:
                continue
            rgba = ch5_uniform_belief_rgba_at_pdf(h, z_lim=z_lim, alpha=0.92)
            ax3d.bar3d(
                float(W1[i, j]) - 0.5 * dx,
                float(W2[i, j]) - 0.5 * dy,
                0.0, dx, dy, h,
                color=rgba, shade=True, linewidth=0.0, zorder=2,
            )


def _draw_surface(ax3d, surf, *, alpha=None, palette="belief"):
    """Likelihood surface with belief or cloud-density (ch6_69+) facecolors."""
    W1 = np.asarray(surf["W1"], dtype=np.float64)
    W2 = np.asarray(surf["W2"], dtype=np.float64)
    Z = np.asarray(surf["Z"], dtype=np.float64)
    z_hi = max(float(np.nanmax(Z)), float(surf.get("z_lim", (0.0, 1.0))[1]), 1e-6)
    al = float(CH5_BELIEF_SURFACE_ALPHA if alpha is None else alpha)
    if palette == "cloud":
        fc = _cloud_density_facecolors(Z, z_lim=(0.0, z_hi), surface_alpha=al)
    else:
        fc = ch5_uniform_belief_facecolors(Z, z_lim=(0.0, z_hi), surface_alpha=al)
    ch5_plot_belief_surface_with_grid(
        ax3d, W1, W2, Z, facecolors=fc, zorder=1.0, antialiased=True,
    )


def _draw_surface_ghost(ax3d, surf, *, alpha=0.16, grey_mix=0.78):
    """Past likelihood as a toned-down near-grey ghost (not pure grey)."""
    W1 = np.asarray(surf["W1"], dtype=np.float64)
    W2 = np.asarray(surf["W2"], dtype=np.float64)
    Z = np.asarray(surf["Z"], dtype=np.float64)
    z_hi = max(float(np.nanmax(Z)), float(surf.get("z_lim", (0.0, 1.0))[1]), 1e-6)
    fc = ch5_uniform_belief_facecolors(Z, z_lim=(0.0, z_hi), surface_alpha=1.0)
    fc = np.asarray(fc, dtype=np.float64).copy()
    rgb = fc[..., :3]
    luma = (0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3])
    # Pull toward grey while keeping a whisper of the belief hue.
    cooled = (1.0 - float(grey_mix)) * rgb + float(grey_mix) * luma
    # Slight cool-slate cast so ghosts don't read as flat silver.
    slate = np.array([0.62, 0.64, 0.68], dtype=np.float64)
    fc[..., :3] = np.clip(0.82 * cooled + 0.18 * slate, 0.0, 1.0)
    fc[..., 3] = float(alpha)
    ch5_plot_belief_surface_with_grid(
        ax3d, W1, W2, Z, facecolors=fc, zorder=0.6, antialiased=True,
    )


def _pick_ghost_surfaces(surfaces, n_show=None):
    """Subsample past landscapes so the ghost stack stays renderable."""
    if not surfaces:
        return []
    n_show = int(16 if n_show is None else n_show)
    if len(surfaces) <= n_show:
        return list(surfaces)
    idx = np.linspace(0, len(surfaces) - 1, n_show).astype(int)
    # Always keep the most recent ghost
    idx[-1] = len(surfaces) - 1
    idx = np.unique(idx)
    return [surfaces[int(i)] for i in idx]


def _draw_markers(ax3d, weights, *, color=CH6_CLOUD_COLOR, s=28, alpha=0.85, z=None):
    Ws = np.asarray(weights, dtype=np.float64)
    if Ws.size == 0:
        return
    if z is None:
        zz = np.full(len(Ws), 0.02)
    else:
        zz = np.asarray(z, dtype=np.float64)
    ax3d.scatter(
        Ws[:, 0], Ws[:, 1], zz,
        s=s, c=color, alpha=alpha, depthshade=False, edgecolors="white", linewidths=0.4, zorder=8,
    )


def _cloud_density_rgba_at_t(t, *, alpha_scale=0.92):
    """Single blue density color for one histogram bin height."""
    t = float(np.clip(t, 0.0, 1.0))
    light = np.array([0.84, 0.90, 0.95], dtype=np.float64)
    dark = np.array([0.10, 0.32, 0.46], dtype=np.float64)
    rgb = light * (1.0 - t) + dark * t
    return (*rgb.tolist(), (0.22 + 0.72 * t) * float(alpha_scale))


def _cloud_density_facecolors(Z, *, z_lim=None, surface_alpha=None):
    """RGBA face colors for likelihood surfaces — matches ch6_69+ histogram blues."""
    Z = np.asarray(Z, dtype=np.float64)
    if z_lim is None:
        z_lim = (0.0, CH6_SURFACE_Z_HI)
    z_hi = max(float(np.nanmax(Z)), float(z_lim[1]), 1e-6)
    t = np.clip(Z / z_hi, 0.0, 1.0)
    light = np.array([0.84, 0.90, 0.95], dtype=np.float64)
    dark = np.array([0.10, 0.32, 0.46], dtype=np.float64)
    rgb = light.reshape(1, 1, 3) * (1.0 - t[..., None]) + dark.reshape(1, 1, 3) * t[..., None]
    fc = np.zeros((*Z.shape, 4), dtype=np.float64)
    fc[..., :3] = rgb
    al = float(CH5_BELIEF_SURFACE_ALPHA if surface_alpha is None else surface_alpha)
    fc[..., 3] = (0.22 + 0.72 * t) * al
    return fc


def _projected_cloud_histogram(W, plane_i, plane_j, *, axis_lim, bins=24):
    """Raw 2D count histogram on coordinate plane ``(plane_i, plane_j)``."""
    W = np.asarray(W, dtype=np.float64)
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    H, xe, ye = np.histogram2d(
        W[:, plane_i], W[:, plane_j], bins=int(bins), range=[[lo, hi], [lo, hi]],
    )
    if float(H.max()) > 0.0:
        H = H / float(H.max())
    return H, xe, ye


def _inward_hist_side(W, height_k, axis_lim):
    """Pick the bounding face so bars grow inward toward the cloud center."""
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    mu_k = float(np.mean(np.asarray(W, dtype=np.float64)[:, height_k]))
    return -1 if mu_k >= 0.5 * (lo + hi) else +1


def _density_hist_side(name, W, height_k, axis_lim):
    """Face per plane: ``wst_el`` always on low-``b`` floor; ``wel_b`` opposite auto."""
    if name == "wst_el":
        return -1
    side = _inward_hist_side(W, height_k, axis_lim)
    if name == "wel_b":
        side = -side
    return side


CH6_DENSITY_BAR_PAD = 0.48


def _draw_inward_density_histogram(
    ax3d,
    W,
    *,
    plane_i,
    plane_j,
    height_k,
    axis_lim,
    bins=24,
    side=None,
    reveal_u=1.0,
    bar_pad=None,
):
    """Histogram pillars on a coordinate plane, extruded inward along ``height_k``."""
    W = np.asarray(W, dtype=np.float64)
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    side = int(side) if side is not None else _inward_hist_side(W, height_k, axis_lim)
    H, xe, ye = _projected_cloud_histogram(
        W, plane_i, plane_j, axis_lim=axis_lim, bins=bins,
    )
    if float(H.max()) <= 1e-8:
        return

    mu_k = float(np.mean(W[:, height_k]))
    if side < 0:
        k_plane = lo
        peak_h = 0.5 * max(mu_k - k_plane, 1e-6)
    else:
        k_plane = hi
        peak_h = 0.5 * max(k_plane - mu_k, 1e-6)

    peak_h *= float(np.clip(reveal_u, 0.0, 1.0))
    if peak_h <= 1e-8:
        return

    du = float(xe[1] - xe[0]) if len(xe) > 1 else 0.2
    dv = float(ye[1] - ye[0]) if len(ye) > 1 else 0.2
    pad = float(CH6_DENSITY_BAR_PAD if bar_pad is None else bar_pad)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            t = float(H[i, j])
            if t <= 1e-4:
                continue
            bh = t * peak_h
            origin = np.zeros(3, dtype=np.float64)
            extent = np.zeros(3, dtype=np.float64)
            origin[plane_i] = float(xe[i])
            origin[plane_j] = float(ye[j])
            extent[plane_i] = du * pad
            extent[plane_j] = dv * pad
            extent[height_k] = bh
            if side < 0:
                origin[height_k] = k_plane
            else:
                origin[height_k] = k_plane - bh
            rgba = _cloud_density_rgba_at_t(t)
            ax3d.bar3d(
                float(origin[0]), float(origin[1]), float(origin[2]),
                float(extent[0]), float(extent[1]), float(extent[2]),
                color=rgba, shade=True, linewidth=0.0, zorder=3,
            )


def _draw_all_density_histograms(
    ax3d,
    W,
    axis_lim,
    *,
    bins=24,
    reveal_u=1.0,
    bar_pad=None,
):
    """All three coordinate-plane histograms at once."""
    for name, pi, pj, hk in CH6_DENSITY_PROJ_SPECS:
        _draw_inward_density_histogram(
            ax3d, W,
            plane_i=pi, plane_j=pj, height_k=hk,
            axis_lim=axis_lim, bins=bins,
            side=_density_hist_side(name, W, hk, axis_lim),
            reveal_u=reveal_u, bar_pad=bar_pad,
        )


def _draw_plane_projection_points(
    ax3d,
    W,
    *,
    plane_i,
    plane_j,
    height_k,
    axis_lim,
    side,
    reveal_u=1.0,
    marker_s=14,
):
    """Scatter of cloud points orthogonally projected onto one coordinate plane."""
    W = np.asarray(W, dtype=np.float64)
    if len(W) == 0:
        return
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    k_plane = lo if int(side) < 0 else hi
    pts = np.zeros((len(W), 3), dtype=np.float64)
    pts[:, plane_i] = W[:, plane_i]
    pts[:, plane_j] = W[:, plane_j]
    pts[:, height_k] = k_plane
    u = float(np.clip(reveal_u, 0.0, 1.0))
    if u <= 1e-6:
        return
    ax3d.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=float(marker_s) * (0.35 + 0.65 * u),
        c=CH6_CLOUD_COLOR,
        alpha=0.18 + 0.50 * u,
        depthshade=False,
        edgecolors="white",
        linewidths=0.25,
        zorder=5,
    )


def _draw_all_plane_projections(
    ax3d,
    W,
    axis_lim,
    *,
    reveal_u=1.0,
    marker_s=14,
):
    """Project the cloud onto all three coordinate planes."""
    for name, pi, pj, hk in CH6_DENSITY_PROJ_SPECS:
        _draw_plane_projection_points(
            ax3d, W,
            plane_i=pi, plane_j=pj, height_k=hk,
            axis_lim=axis_lim,
            side=_density_hist_side(name, W, hk, axis_lim),
            reveal_u=reveal_u, marker_s=marker_s,
        )

# One inward histogram per coordinate plane (3 faces total).
CH6_DENSITY_PROJ_SPECS: tuple[tuple[str, int, int, int], ...] = (
    ("wst_el", 0, 1, 2),
    ("wst_b", 0, 2, 1),
    ("wel_b", 1, 2, 0),
)

CH6_VARIANCE_RED = "#ef4444"
CH6_VARIANCE_GREY = "#b8bcc4"
CH6_VARIANCE_MIN_ANCHOR_DIST = 2.5
CH6_VARIANCE_AXIS_NAMES = ("w_st", "w_el", "b")
CH6_VARIANCE_VIEWS: dict[str, tuple[float, float]] = {
    # elev=0; cumulative +90° azim per pass (bias → w_ST → w_EL)
    "b": (0.0, 0.0),
    "w_st_face": (0.0, 90.0),
    "w_el_face": (0.0, 180.0),
}
CH6_VARIANCE_INTERVAL_OFFSET = 0.10


def _variance_init_view():
    base_elev = float(ch5_hq_land_elev(float(_g("CH3_LIK_W12_CT_ELEV"))))
    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))
    return base_elev, base_azim


def _variance_resolve_view(name):
    if name == "init":
        return _variance_init_view()
    elev, azim = CH6_VARIANCE_VIEWS[name]
    return float(elev), float(azim)


def _variance_interp_view(name_a, name_b, u):
    ea, aa = _variance_resolve_view(name_a)
    eb, ab = _variance_resolve_view(name_b)
    t = float(np.clip(u, 0.0, 1.0))
    elev = (1.0 - t) * ea + t * eb
    az_delta = (float(ab) - float(aa) + 180.0) % 360.0 - 180.0
    azim = float(aa) + t * az_delta
    return elev, azim


def _variance_marker_colors(W, *, revealed, grey_u):
    W = np.asarray(W, dtype=np.float64)
    n = len(W)
    base = np.array(plt.matplotlib.colors.to_rgb(CH6_CLOUD_COLOR), dtype=np.float64)
    grey = np.array(plt.matplotlib.colors.to_rgb(CH6_VARIANCE_GREY), dtype=np.float64)
    cols = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        if bool(revealed[i]):
            rgb = base
            al = 0.88
        else:
            u = float(np.clip(grey_u, 0.0, 1.0))
            rgb = base * (1.0 - u) + grey * u
            al = 0.28 + 0.22 * (1.0 - u)
        cols[i] = (*rgb.tolist(), al)
    return cols


def _variance_plot_seg(ax3d, p0, p1, *, color, lw, ls, alpha):
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    if float(alpha) <= 1e-4:
        return
    ax3d.plot(
        [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
        color=color, lw=float(lw), ls=ls, alpha=float(alpha), zorder=7,
    )


def _variance_axis_corner(mu, pt, axis):
    corner = np.asarray(mu, dtype=np.float64).reshape(3).copy()
    corner[int(axis)] = float(np.asarray(pt, dtype=np.float64).reshape(3)[int(axis)])
    return corner


def _variance_dotted_end(mu, pt, axis):
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    pt = np.asarray(pt, dtype=np.float64).reshape(3)
    end = pt.copy()
    if int(axis) == 0:
        end[0] = mu[0]
    elif int(axis) == 1:
        end[1] = mu[1]
    else:
        end[2] = mu[2]
    return end


def _variance_draw_axis_component(ax3d, mu, pt, axis, *, alpha=1.0):
    corner = _variance_axis_corner(mu, pt, axis)
    dotted_end = _variance_dotted_end(mu, pt, axis)
    _variance_plot_seg(
        ax3d, mu, corner, color=CH6_VARIANCE_RED, lw=2.2, ls="-", alpha=alpha,
    )
    _variance_plot_seg(
        ax3d, pt, dotted_end, color=CH6_VARIANCE_RED, lw=1.8, ls=":", alpha=alpha,
    )


def _variance_draw_full_decomp(ax3d, mu, pt, *, alpha=1.0):
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    pt = np.asarray(pt, dtype=np.float64).reshape(3)
    c_st = mu.copy(); c_st[0] = pt[0]
    c_el = c_st.copy(); c_el[1] = pt[1]
    _variance_plot_seg(ax3d, mu, c_st, color=CH6_VARIANCE_RED, lw=2.2, ls="-", alpha=alpha)
    _variance_plot_seg(ax3d, c_st, c_el, color=CH6_VARIANCE_RED, lw=1.8, ls=":", alpha=alpha)
    _variance_plot_seg(ax3d, c_el, pt, color=CH6_VARIANCE_RED, lw=1.8, ls=":", alpha=alpha)


def _variance_draw_centered_range(
    ax3d,
    mu,
    axis,
    lo,
    hi,
    *,
    alpha=1.0,
    labels=False,
    lw=2.8,
):
    """Axis-aligned segment through the cloud mean."""
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    a = mu.copy()
    b = mu.copy()
    a[int(axis)] = float(lo)
    b[int(axis)] = float(hi)
    _variance_plot_seg(ax3d, a, b, color=CH6_VARIANCE_RED, lw=lw, ls="-", alpha=alpha)
    if labels and float(alpha) > 0.35:
        lbl = r"$\sigma^2$"
        ax3d.text(
            float(a[0]), float(a[1]), float(a[2]), lbl,
            color=CH6_VARIANCE_RED, fontsize=9, ha="right", va="center",
        )
        ax3d.text(
            float(b[0]), float(b[1]), float(b[2]), lbl,
            color=CH6_VARIANCE_RED, fontsize=9, ha="left", va="center",
        )


def _variance_draw_axis_through(ax3d, mu, axis, lo, hi, *, alpha=1.0, lw=2.4):
    _variance_draw_centered_range(
        ax3d, mu, axis, lo, hi, alpha=alpha, labels=False, lw=lw,
    )


def _variance_box_edges(bounds):
    x0, x1 = bounds[0]
    y0, y1 = bounds[1]
    z0, z1 = bounds[2]
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return corners, edges


def _variance_draw_box(ax3d, bounds, *, edge_alpha=0.85, fill_alpha=0.06, inner_bounds=None, gap_u=0.0):
    corners, edges = _variance_box_edges(bounds)
    for i, j in edges:
        _variance_plot_seg(
            ax3d, corners[i], corners[j],
            color=CH6_VARIANCE_RED, lw=1.8, ls="-", alpha=edge_alpha,
        )
    if inner_bounds is not None and float(gap_u) > 1e-4:
        ic, ie = _variance_box_edges(inner_bounds)
        for i, j in ie:
            _variance_plot_seg(
                ax3d, ic[i], ic[j],
                color=CH6_VARIANCE_RED, lw=1.4, ls=":", alpha=0.55 * gap_u,
            )
        faces = (
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5),
        )
        for f in faces:
            pts = corners[list(f)]
            ax3d.plot_surface(
                np.array([[pts[0, 0], pts[1, 0]], [pts[3, 0], pts[2, 0]]]),
                np.array([[pts[0, 1], pts[1, 1]], [pts[3, 1], pts[2, 1]]]),
                np.array([[pts[0, 2], pts[1, 2]], [pts[3, 2], pts[2, 2]]]),
                color=CH6_VARIANCE_RED, alpha=float(fill_alpha) * float(gap_u),
                linewidth=0, shade=False, zorder=2,
            )


def _frame_population_variance_duo(
    *,
    xlim,
    ylim,
    base_study,
    base_exam,
    base_y,
    W,
    mu,
    axis_lim,
    revealed_mask,
    grey_u=1.0,
    show_center=True,
    axis_components=None,
    decomp_alpha=1.0,
    range_state=None,
    finale_ranges=None,
    show_box=False,
    box_bounds=None,
    inner_bounds=None,
    gap_u=0.0,
    view_elev=None,
    view_azim=None,
    marker_s=28,
    knob_w=None,
):
    """Duo frame for sampling-variance story (continues from density_end cloud)."""
    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()
    if len(base_study) > 0:
        _draw_base_dataset(ax_data, base_study, base_exam, base_y, xlim=xlim, ylim=ylim)
    ax_data.set_xlim(*xlim)
    ax_data.set_ylim(*ylim)
    ax_data.set_xlabel("Study time (hours)", fontsize=_g("AXIS_LABEL_SIZE"), labelpad=10)
    ax_data.set_ylabel("Exam length (hours)", fontsize=_g("AXIS_LABEL_SIZE"), labelpad=10)
    ax_data.grid(alpha=0.2)

    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    z_lim = (lo, hi)
    xy_lim = (lo, hi)
    _style_ax3d(
        ax3d, z_lim=z_lim, zlabel="b", xy_lim=xy_lim,
        elev=view_elev, azim=view_azim,
    )

    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    revealed_mask = np.asarray(revealed_mask, dtype=bool)
    cols = _variance_marker_colors(W, revealed=revealed_mask, grey_u=float(grey_u))
    ax3d.scatter(
        W[:, 0], W[:, 1], W[:, 2],
        s=float(marker_s), c=cols, depthshade=False,
        edgecolors="white", linewidths=0.35, zorder=8,
    )

    if show_center:
        ax3d.scatter(
            [mu[0]], [mu[1]], [mu[2]],
            s=95, c=CH6_VARIANCE_RED, alpha=1.0, depthshade=False,
            edgecolors="white", linewidths=0.6, zorder=12,
        )

    if axis_components:
        for axis, pairs in axis_components.items():
            al = float(pairs.get("alpha", decomp_alpha))
            for idx in pairs.get("indices", []):
                _variance_draw_axis_component(ax3d, mu, W[int(idx)], int(axis), alpha=al)

    if range_state:
        for axis, st in range_state.items():
            _variance_draw_centered_range(
                ax3d, mu, int(axis), float(st["lo"]), float(st["hi"]),
                alpha=float(st.get("alpha", 1.0)),
                labels=bool(st.get("labels", False)),
                lw=float(st.get("lw", 2.8)),
            )
    if finale_ranges:
        for axis, st in finale_ranges.items():
            _variance_draw_axis_through(
                ax3d, mu, int(axis), float(st["lo"]), float(st["hi"]),
                alpha=float(st.get("alpha", 1.0)),
            )
    if show_box and box_bounds is not None:
        _variance_draw_box(
            ax3d, box_bounds, edge_alpha=0.9, fill_alpha=0.07,
            inner_bounds=inner_bounds, gap_u=float(gap_u),
        )

    if knob_w is not None:
        kw = np.asarray(knob_w, dtype=np.float64).reshape(3)
        ws, we, bb = float(kw[0]), float(kw[1]), float(kw[2])
        knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
        _g("ch3_draw_knob_row")(
            fig, axes_k, ws, we, bb, "st", knob_rgbs, canvas_sides,
            rot_strip_deg=0.0, strip_scale=1.0,
            knob_rots=_g("ch3_k1_knob_rots_at")(ws, we, bb),
            knob_scales=[1.0, 1.0, 1.0], ax_data=ax_data,
        )
    _g("_ch3_align_knob_axes_under_data")(fig, ax_data, axes_k)
    _g("ch3_layout_knob_axes_like_bridge_end")(fig, ax_data, axes_k)

    return _fig_to_plot(fig)


def _frame_cloud_density_project(
    W,
    *,
    axis_lim,
    marker_s=18,
    bins=24,
    minimal_ui=False,
    reveal_u=1.0,
):
    """Single 3D panel: point cloud + all three inward histograms."""
    figsize = _g("CH4_DUO_FIGSIZE") if "CH4_DUO_FIGSIZE" in _G else (12.8, 7.2)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax3d = fig.add_subplot(111, projection="3d")
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    _draw_all_density_histograms(
        ax3d, W, axis_lim, bins=bins, reveal_u=reveal_u,
    )
    W = np.asarray(W, dtype=np.float64)
    _draw_markers(ax3d, W, s=marker_s, z=W[:, 2])
    _style_ax3d(
        ax3d, z_lim=(lo, hi), zlabel="b", xy_lim=axis_lim, minimal_ui=minimal_ui,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return _fig_to_plot(fig)


def _build_cloud_density_project_previews(clip_id, *, n_class=60, seed=17):
    """Low-res previews: three inward histograms (one per coordinate plane)."""
    from ch6_frequentist import _CH3_DRAFT

    n_reel = ch6_population_n_sweep_n_reel(n_class)
    pack = ch6_population_param_cloud_pack(n_class, seed=seed, n_reel=n_reel)
    W = pack["landed"]
    axis_lim = ch6_population_n_sweep_states(seed=seed)[0]
    marker_s = ch6_population_n_sweep_marker_size(n_class) * 0.75
    bins = 20 if _CH3_DRAFT else 28
    n_hold = max(8, CH6_N_HOLD * 4)
    frames = []
    img = _frame_cloud_density_project(
        W, axis_lim=axis_lim, marker_s=marker_s, bins=bins,
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold))
    return frames


def build_ch6_77_cloud_density_project_previews(clip_id):
    return _build_cloud_density_project_previews(clip_id, n_class=60, seed=17)


def _zlim_from_bias_points(markers=None, highlight_w=None, *, pad_frac=0.14, min_span=0.8):
    """Dynamic z-range for parameter cloud views where z is bias b."""
    zs: list[float] = []
    if markers is not None:
        Ws = np.asarray(markers, dtype=np.float64)
        if Ws.ndim == 2 and Ws.shape[1] >= 3 and len(Ws) > 0:
            zs.extend(np.asarray(Ws[:, 2], dtype=np.float64).tolist())
    if highlight_w is not None:
        hw = np.asarray(highlight_w, dtype=np.float64).reshape(-1)
        if hw.size >= 3:
            zs.append(float(hw[2]))
    if not zs:
        return (0.0, CH6_SURFACE_Z_HI)
    lo = float(np.min(zs))
    hi = float(np.max(zs))
    span = max(float(hi - lo), float(min_span))
    pad = float(pad_frac) * span
    return (lo - pad, hi + pad)


def _surface_w12_axes(surf):
    """1D w_ST / w_EL coordinate axes for a likelihood mesh."""
    W1 = np.asarray(surf["W1"], dtype=np.float64)
    W2 = np.asarray(surf["W2"], dtype=np.float64)
    w1 = surf.get("w1")
    w2 = surf.get("w2")
    if w1 is None:
        w1 = W1[0, :]
    if w2 is None:
        w2 = W2[:, 0]
    return np.asarray(w1, dtype=np.float64), np.asarray(w2, dtype=np.float64)


def _surface_z_at(surf, w1, w2):
    """Bilinear lookup of relative-likelihood height on (w_ST, w_EL)."""
    Z = np.asarray(surf["Z"], dtype=np.float64)
    w1a, w2a = _surface_w12_axes(surf)
    w1f = float(np.clip(float(w1), float(w1a[0]), float(w1a[-1])))
    w2f = float(np.clip(float(w2), float(w2a[0]), float(w2a[-1])))
    i1 = int(np.searchsorted(w1a, w1f) - 1)
    i2 = int(np.searchsorted(w2a, w2f) - 1)
    i1 = int(np.clip(i1, 0, len(w1a) - 2))
    i2 = int(np.clip(i2, 0, len(w2a) - 2))
    t1 = (w1f - float(w1a[i1])) / max(float(w1a[i1 + 1] - w1a[i1]), 1e-12)
    t2 = (w2f - float(w2a[i2])) / max(float(w2a[i2 + 1] - w2a[i2]), 1e-12)
    t1 = float(np.clip(t1, 0.0, 1.0))
    t2 = float(np.clip(t2, 0.0, 1.0))
    z00 = float(Z[i2, i1])
    z10 = float(Z[i2, i1 + 1])
    z01 = float(Z[i2 + 1, i1])
    z11 = float(Z[i2 + 1, i1 + 1])
    z0 = (1.0 - t1) * z00 + t1 * z10
    z1 = (1.0 - t1) * z01 + t1 * z11
    return float((1.0 - t2) * z0 + t2 * z1)


def _avg_surf_peak_w12(surf):
    """Argmax of the averaged relative-likelihood surface (b fixed slice)."""
    Z = np.asarray(surf["Z"], dtype=np.float64)
    W1 = np.asarray(surf["W1"], dtype=np.float64)
    W2 = np.asarray(surf["W2"], dtype=np.float64)
    k = np.unravel_index(int(np.nanargmax(Z)), Z.shape)
    return np.array([float(W1[k]), float(W2[k])], dtype=np.float64)


def _marker_z_on_surface(surface, weights, *, lift=0.012):
    """Lift (w_ST, w_EL) landings onto a likelihood surface."""
    W = np.asarray(weights, dtype=np.float64)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    return np.array(
        [_surface_z_at(surface, w[0], w[1]) + float(lift) for w in W],
        dtype=np.float64,
    )


def _draw_weight_grad_arrow(
    ax3d, w, grad, *, surface=None, color="#d500f9", scale=0.70, ascent=False,
    pin_to_surface=True,
):
    """Floor-plane push arrow in (w_ST, w_EL) — same spot as ch6_49/50 markers.

    Axes are (w_ST, w_EL, likelihood). The arrow shaft lives at z≈0 in the
    (w_ST, w_EL) floor. Pass the display direction with ``ascent=False``;
    set ``ascent=True`` only when ``grad`` is raw ∇NLL.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    g = np.asarray(grad, dtype=np.float64).reshape(-1)
    g2 = g[:2].astype(np.float64, copy=True)
    d = -g2 if ascent else g2
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return
    d = d / nrm

    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    span = min(hi1 - lo1, hi2 - lo2)
    if float(scale) < 1e-3:
        return
    L = float(np.clip(float(scale), 0.0, 0.20 * max(span, 1e-6)))

    x0 = float(np.clip(float(w[0]), lo1, hi1))
    y0 = float(np.clip(float(w[1]), lo2, hi2))
    z = 0.04
    x1 = float(np.clip(x0 + L * d[0], lo1, hi1))
    y1 = float(np.clip(y0 + L * d[1], lo2, hi2))
    dx, dy = x1 - x0, y1 - y0
    ln = float(np.hypot(dx, dy))
    if ln < 1e-9:
        return
    ux, uy = dx / ln, dy / ln

    if pin_to_surface and surface is not None:
        z_top = _surface_z_at(surface, x0, y0) + 0.03
        ax3d.plot(
            [x0, x0], [y0, y0], [z, z_top],
            color="#1a001f", linewidth=1.2, linestyle=":",
            alpha=0.75, zorder=28,
        )
        ax3d.scatter(
            [x0], [y0], [z_top],
            s=28, c=color, edgecolors="#1a001f", linewidths=0.4,
            depthshade=False, zorder=29,
        )

    for col, width in (("#1a001f", 4.8), (color, 3.0)):
        ax3d.plot(
            [x0, x1], [y0, y1], [z, z],
            color=col, linewidth=width, solid_capstyle="round",
            alpha=0.98, zorder=30,
        )
    head = max(0.30 * ln, 0.12)
    wing = 0.50 * head
    hx, hy = x1 - head * ux, y1 - head * uy
    for sx, sy in ((-uy, ux), (uy, -ux)):
        ax3d.plot(
            [x1, hx + wing * sx], [y1, hy + wing * sy], [z, z],
            color=color, linewidth=2.8, solid_capstyle="round",
            alpha=0.98, zorder=31,
        )
    ax3d.scatter(
        [x0], [y0], [z],
        s=60, c=color, edgecolors="#1a001f", linewidths=0.5,
        depthshade=False, zorder=32,
    )


def _draw_spring_compress_arrow(
    ax3d,
    surf,
    head_xy,
    away_dir,
    *,
    compress_u=1.0,
    scale=1.0,
    color=CH6_VARIANCE_RED,
):
    """Compressing spring arrow on the far side of the point, tip at the ball."""
    head = np.asarray(head_xy, dtype=np.float64).reshape(2)
    away = np.asarray(away_dir, dtype=np.float64).reshape(2)
    away = away / (float(np.linalg.norm(away)) + 1e-12)
    u = float(np.clip(compress_u, 0.0, 1.0))
    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    span = min(hi1 - lo1, hi2 - lo2)
    L = float(scale) * u * 0.24 * span
    if L < 1e-4:
        return
    tail = head + away * L
    xh, yh = float(head[0]), float(head[1])
    xt, yt = float(tail[0]), float(tail[1])
    zh = _surface_z_at(surf, xh, yh) + 0.018
    zt = _surface_z_at(surf, xt, yt) + 0.018
    dx, dy = xh - xt, yh - yt
    ln = float(np.hypot(dx, dy))
    if ln < 1e-9:
        return
    ux, uy = dx / ln, dy / ln
    head_len = min(0.042 * span, 0.17 * ln)
    bx, by = xh - head_len * ux, yh - head_len * uy
    zb = _surface_z_at(surf, bx, by) + 0.018
    ax3d.plot(
        [xt, bx], [yt, by], [zt, zb],
        color=color, linewidth=3.6, solid_capstyle="round",
        alpha=0.98, zorder=35,
    )
    hw = head_len * 0.40
    tri_x = [xh, bx + hw * (-uy), bx + hw * uy, xh]
    tri_y = [yh, by + hw * ux, by - hw * ux, yh]
    tri_z = [zh, zb, zb, zh]
    ax3d.plot(
        tri_x, tri_y, tri_z,
        color=color, linewidth=2.4, solid_capstyle="round",
        alpha=0.98, zorder=36,
    )


def _resample_path_arc_length(path, n_out, *, ease_out=2.4):
    """Uniformly resample a polyline with optional ease-out deceleration."""
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    n_out = max(int(n_out), 2)
    if len(pts) < 2:
        return np.tile(pts[-1], (n_out, 1)) if len(pts) else pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-9:
        return np.tile(pts[-1], (n_out, 1))
    u = np.linspace(0.0, 1.0, n_out)
    t_query = total * (1.0 - (1.0 - u) ** float(ease_out))
    out = np.zeros((n_out, 2), dtype=np.float64)
    for d in range(2):
        out[:, d] = np.interp(t_query, s, pts[:, d])
    out[-1] = pts[-1].copy()
    return out


def _spring_crush_profile(t):
    """Slow load then snap shut — high ``dL/dt`` at ``t=1`` (Newton's-cradle strike)."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.68:
        return 0.10 * (t / 0.68) ** 1.15
    u = (t - 0.68) / 0.32
    return 0.10 + 0.90 * (u ** 1.22)


def _min_uphill_speed(surf, p0, p_target):
    """Kinetic speed (mass=1) needed to climb the bowl from ``p0`` to the peak."""
    p0 = np.asarray(p0, dtype=np.float64).reshape(2)
    target = np.asarray(p_target, dtype=np.float64).reshape(2)
    toward = target - p0
    path_d = float(np.linalg.norm(toward)) + 1e-12
    toward_u = toward / path_d
    z0 = _surface_z_at(surf, float(p0[0]), float(p0[1]))
    zp = _surface_z_at(surf, float(target[0]), float(target[1]))
    climb = max(float(zp - z0), 1e-5)
    grad0 = _avg_surf_lik_grad_w12(surf, float(p0[0]), float(p0[1]))
    hill = max(float(grad0 @ toward_u), 1e-5)
    # Sample mid-path steepness — steeper bowl demands more launch energy.
    mid = 0.5 * (p0 + target)
    grad_m = _avg_surf_lik_grad_w12(surf, float(mid[0]), float(mid[1]))
    gn_mid = float(np.linalg.norm(grad_m))
    potential = climb * 3.4 + hill * path_d * 2.4 + gn_mid * path_d * 0.55
    return float(np.sqrt(max(2.0 * potential, 1e-8)))


def _surface_spring_launch_sequence(
    surf,
    p0,
    p_target,
    away_dir,
    *,
    n_compress=11,
    n_coast=52,
    dt=0.042,
    spring_k=14.0,
    spring_len_frac=0.34,
    roll_drag=2.4,
    slope_drag=11.5,
    g_eff=5.8,
):
    """Full arrow collapse (ball still) → cradle impulse ∝ dL/dt → uphill coast.

    Phase 1: spring compresses completely while the ball rests at the classroom.
    Phase 2: on the final collapse step, speed ``∝ |dL/dt|`` is transferred along
    the uphill direction, boosted to exceed the landscape's steepness barrier.
    Phase 3: motion decelerates with gradient-weighted drag (steeper → slower).
    """
    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    span = min(hi1 - lo1, hi2 - lo2)
    L_max = float(spring_len_frac) * span
    away = np.asarray(away_dir, dtype=np.float64).reshape(2)
    away = away / (float(np.linalg.norm(away)) + 1e-12)
    toward = -away
    target = np.asarray(p_target, dtype=np.float64).reshape(2)
    p0 = np.asarray(p0, dtype=np.float64).reshape(2)
    out: list = []

    n_c = max(int(n_compress), 3)
    lengths = []
    for i in range(1, n_c + 1):
        t = float(i) / float(n_c)
        crush = _spring_crush_profile(t)
        L = L_max * (1.0 - crush)
        lengths.append(float(L))
        compress_u = L / max(L_max, 1e-12)
        out.append({
            "xy": p0.copy(),
            "compress_u": float(compress_u),
            "spring_scale": 1.42,
        })
    out[-1]["compress_u"] = 0.0

    t_prev = float(n_c - 1) / float(n_c)
    L_prev = L_max * (1.0 - _spring_crush_profile(t_prev))
    dL_release = max(L_prev - 0.0, 1e-8)
    collapse_rate = dL_release / float(dt)
    v_cradle = float(spring_k) * collapse_rate
    v_energy = float(np.sqrt(float(spring_k))) * L_max * 0.92
    v_hill = _min_uphill_speed(surf, p0, target)
    v0_mag = max(v_cradle + 0.40 * v_energy, v_hill * 1.04)
    v0_mag = min(v0_mag, v_hill * 1.18)
    v = v0_mag * toward

    p = p0.copy()
    dense = [p.copy()]
    frame_stride = 3
    step_i = 0
    for _ in range(max(int(n_coast) * 12, 320)):
        grad = _avg_surf_lik_grad_w12(surf, float(p[0]), float(p[1]))
        gn = float(np.linalg.norm(grad))
        to_peak = target - p
        dist = float(np.linalg.norm(to_peak))
        spd = float(np.linalg.norm(v))
        v_hat = v / spd if spd > 1e-9 else toward
        climb_resist = float(g_eff) * max(float(grad @ v_hat), 0.0)
        drag_c = float(roll_drag) + float(slope_drag) * gn + 2.8 * gn * gn
        a = -climb_resist * v_hat - drag_c * v
        if dist < 0.14:
            settle = (1.0 - dist / 0.14) ** 1.6
            wn = 4.5 + 16.0 * settle
            a = a - (2.55 * wn) * v - (wn * wn) * to_peak
        if dist < 0.022:
            lock = (0.022 - dist) / 0.022
            p = (1.0 - lock) * p + lock * target
            v *= max(0.0, 1.0 - 4.0 * lock)
        v = v + float(dt) * a
        p = p + float(dt) * v
        p[0] = float(np.clip(p[0], lo1, hi1))
        p[1] = float(np.clip(p[1], lo2, hi2))
        step_i += 1
        if step_i % frame_stride == 0:
            dense.append(p.copy())
        if dist < 0.006:
            p = target.copy()
            if not np.allclose(dense[-1], p, atol=1e-6):
                dense.append(p.copy())
            break
    if not np.allclose(dense[-1], target, atol=1e-6):
        dense.append(target.copy())
    coast_xy = _resample_path_arc_length(dense, max(int(n_coast), 2), ease_out=1.0)
    for xy in coast_xy[1:]:
        out.append({"xy": np.asarray(xy, dtype=np.float64), "compress_u": None})
    if out:
        out[-1]["xy"] = target.copy()
    return out


def _surface_ball_coast_path(
    surf,
    p0,
    p_target,
    *,
    impulse,
    n_out=58,
    substeps=360,
    mass=1.0,
    damp=9.2,
    slope_gain=4.6,
    dt=0.022,
):
    """Overdamped roll uphill on the averaged bowl; dense integrate then smooth."""
    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    p = np.asarray(p0, dtype=np.float64).reshape(2).copy()
    target = np.asarray(p_target, dtype=np.float64).reshape(2)
    v = np.asarray(impulse, dtype=np.float64).reshape(2).copy()
    dense = [p.copy()]
    for _ in range(int(substeps)):
        grad = _avg_surf_lik_grad_w12(surf, float(p[0]), float(p[1]))
        gn = float(np.linalg.norm(grad))
        to_peak = target - p
        dist = float(np.linalg.norm(to_peak))
        a_uh = float(slope_gain) * grad if gn > 1e-9 else np.zeros(2, dtype=np.float64)
        pull = min(1.0, dist / 0.18)
        a = a_uh + 1.8 * pull * to_peak - float(damp) * v
        v = v + (float(dt) / float(mass)) * a
        vn = float(np.linalg.norm(v))
        if vn > 1.35:
            v *= 1.35 / vn
        p = p + float(dt) * v
        p[0] = float(np.clip(p[0], lo1, hi1))
        p[1] = float(np.clip(p[1], lo2, hi2))
        dense.append(p.copy())
        if dist < 0.018 and vn < 0.03:
            break
    dense.append(target.copy())
    return _resample_path_arc_length(dense, int(n_out), ease_out=2.6)


def _spring_away_d1_landscape_pack(*, seed=51, n_avg=500, grid=132):
    """Shared D1 average-landscape pack for spring-away clips."""
    study, exam, y = ch5_unpack_dataset("D1")
    pack = ch6_average_rel_likelihood_population(
        n_class=20, n_avg=int(n_avg), seed=int(seed), ridge=CH6_RIDGE, grid=int(grid),
        seed_from_key="D1",
    )
    b_fixed = float(pack["b_fixed"])
    w_avg = np.asarray(pack["mean_hat"], dtype=np.float64).copy()
    w_avg[2] = b_fixed
    w_obs = np.asarray(pack["w_obs"], dtype=np.float64).copy()
    w_obs[2] = b_fixed
    return pack, study, exam, y, w_avg, w_obs, b_fixed, pack["avg_surf"]


def _draw_population_scatter(ax, study, exam, y, *, alpha=0.18, s=8):
    """Ghost population as tiny colored dots (icons would be too heavy at 100×)."""
    study = np.asarray(study, dtype=np.float64)
    exam = np.asarray(exam, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fail = y < 0.5
    pass_ = ~fail
    if np.any(fail):
        ax.scatter(
            study[fail], exam[fail],
            s=s, c="#d62728", alpha=float(alpha), linewidths=0, zorder=2, rasterized=True,
        )
    if np.any(pass_):
        ax.scatter(
            study[pass_], exam[pass_],
            s=s, c="#2ca02c", alpha=float(alpha), linewidths=0, zorder=2, rasterized=True,
        )


def _draw_class_ellipses(ax, *, n_std=(1.0, 2.0), show_means=True, show_labels=True):
    params = ch6_class_gaussian_params()
    for key in ("fail", "pass"):
        cls = params[key]
        mu = cls["mu"]
        cov = cls["cov"]
        color = cls["color"]
        for ns in n_std:
            xs, ys = ch6_gaussian_ellipse_points(mu, cov, n_std=float(ns))
            ax.plot(xs, ys, color=color, lw=1.6 if ns >= 1.9 else 1.1,
                    alpha=0.85 if ns >= 1.9 else 0.55, zorder=4)
        if show_means:
            ax.scatter([mu[0]], [mu[1]], c=color, s=55, marker="x", linewidths=2.0, zorder=5)
        if show_labels:
            dy = 0.42 if key == "fail" else -0.55
            va = "bottom" if key == "fail" else "top"
            ax.text(
                float(mu[0]), float(mu[1]) + dy,
                f"{cls['name']}  ($\\pi={cls['pi']:.2f}$)\n"
                f"$\\mu=({mu[0]:.2f},\\,{mu[1]:.2f})$",
                ha="center", va=va, fontsize=7.5, color=color, zorder=6,
            )
    if show_labels:
        ax.text(
            0.02, 0.98,
            rf"shared $\sigma_{{\parallel}}={params['sigma_parallel']:.2f}$, "
            rf"$\sigma_{{\perp}}={params['sigma_perp']:.2f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.0, color="#444444", zorder=7,
        )


def _frame_duo(
    *,
    xlim,
    ylim,
    base_study,
    base_exam,
    base_y,
    ghost_study=None,
    ghost_exam=None,
    ghost_y=None,
    pop_study=None,
    pop_exam=None,
    pop_y=None,
    pop_alpha=0.16,
    show_ellipses=False,
    ellipse_labels=False,
    show_base=True,
    w_live=None,
    ghost_ws=None,
    w_true=None,
    w_mean=None,
    knob_w=None,
    stems=None,
    surface=None,
    surface_alpha=None,
    surface_palette="belief",
    ghost_surfaces=None,
    ghost_alpha=None,
    morph_u=0.0,
    markers=None,
    highlight_w=None,
    lr_level=None,
    show_wald=False,
    floor_ellipse=None,
    probe_pts=None,
    probe_color=CH6_STEEP_COLOR,
    probe_flat=False,
    weight_grad=None,
    weight_grad_w=None,
    weight_grad_scale=0.90,
    weight_grad_ascent=False,
    weight_grad_floor_only=False,
    weight_grad_color="#d500f9",
    highlight_color=None,
    spring_arrow=None,
    title_left=None,
    title_right=None,
    zlabel="Likelihood",
    p_annotate=False,
    show_legend=True,
    xlabel=None,
    show_point_grads=False,
    marker_z_mode="floor",
    marker_axis_lim=None,
    view_azim=None,
    ellipsoid_loops=None,
    ellipsoid_alpha=0.85,
    sampling_ellipsoid=None,
    sampling_ellipsoid_reveal_u=1.0,
    show_mean_x=False,
    mean_x=None,
    mean_x_fontsize=30,
    minimal_ui=False,
    marker_s=28,
    highlight_marker_s=70,
    density_W=None,
    density_axis_lim=None,
    density_reveal_u=1.0,
    density_projection_u=0.0,
    density_bins=24,
    density_bar_pad=None,
):
    """Ch4 duo: roster + optional population/ellipses | 3D landings/likelihood."""
    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()

    if pop_study is not None:
        _draw_population_scatter(
            ax_data, pop_study, pop_exam, pop_y, alpha=pop_alpha,
        )
        ax_data.set_xlim(*xlim)
        ax_data.set_ylim(*ylim)

    if show_base and base_study is not None and len(base_study) > 0:
        _draw_base_dataset(ax_data, base_study, base_exam, base_y, xlim=xlim, ylim=ylim)
        if minimal_ui:
            ax_data.set_xlabel("")
            ax_data.set_ylabel("")
            ax_data.tick_params(labelbottom=False, labelleft=False)
    else:
        ax_data.set_xlim(*xlim)
        ax_data.set_ylim(*ylim)
        if minimal_ui:
            ax_data.set_xlabel("")
            ax_data.set_ylabel("")
            ax_data.tick_params(labelbottom=False, labelleft=False)
            ax_data.grid(alpha=0.2)
        else:
            ax_data.set_xlabel(
                xlabel or "Study time (hours)",
                fontsize=_g("AXIS_LABEL_SIZE"), labelpad=10,
            )
            ax_data.set_ylabel("Exam length (hours)", fontsize=_g("AXIS_LABEL_SIZE"), labelpad=10)
            ax_data.grid(alpha=0.2)

    if show_ellipses:
        _draw_class_ellipses(ax_data, show_labels=bool(ellipse_labels))

    if ghost_study is not None:
        _draw_ghost_dataset(ax_data, ghost_study, ghost_exam, ghost_y)

    n_g = len(ghost_ws) if ghost_ws else 0
    g_alpha = _ghost_line_alpha(n_g)
    if ghost_ws:
        for wg in ghost_ws:
            _plot_threshold(
                ax_data, wg, xlim, ylim,
                color=CH6_GHOST_COLOR, lw=1.2, alpha=g_alpha, zorder=3,
            )
    if w_true is not None:
        _plot_threshold(
            ax_data, w_true, xlim, ylim,
            color=CH6_TRUE_LINE_COLOR, lw=1.6, alpha=0.75, ls=":", zorder=4,
        )
    if w_mean is not None:
        _plot_threshold(
            ax_data, w_mean, xlim, ylim,
            color=CH6_MEAN_LINE_COLOR, lw=2.8, alpha=0.95, zorder=5,
        )
    if w_live is not None:
        _plot_threshold(
            ax_data, w_live, xlim, ylim,
            color=CH6_LINE_COLOR, lw=2.5, alpha=1.0, zorder=6,
        )
    if (
        show_point_grads
        and w_live is not None
        and base_study is not None
        and len(base_study) > 0
    ):
        G = ch6_point_nll_grad_contrib(w_live, base_study, base_exam, base_y)
        _draw_point_grad_quivers_2d(ax_data, base_study, base_exam, G, w_live)
    if p_annotate and base_study is not None:
        p = ch6_p_true(base_study, base_exam, w_true=w_true if w_true is not None else CH6_W_TRUE)
        for xi, yi, pi in zip(base_study, base_exam, p):
            ax_data.text(
                float(xi), float(yi) + 0.18, f"{pi:.2f}",
                ha="center", va="bottom", fontsize=6.5, color="#555", zorder=7,
            )
    if title_left:
        ax_data.text(
            0.02, 0.98, title_left, transform=ax_data.transAxes,
            va="top", ha="left", fontsize=9.5, color="#222", fontweight="bold",
        )
    if not show_legend:
        leg = ax_data.get_legend()
        if leg is not None:
            leg.remove()
    else:
        _g("finalize_style_legend_tex")(ax_data)

    kw = knob_w if knob_w is not None else (w_live if w_live is not None else CH6_W_TRUE)
    ws, we, bb = float(kw[0]), float(kw[1]), float(kw[2])
    knob_rgbs, canvas_sides = _g("ch4_knob_asset_pack")()
    _g("ch3_draw_knob_row")(
        fig, axes_k, ws, we, bb, "st", knob_rgbs, canvas_sides,
        rot_strip_deg=0.0, strip_scale=1.0,
        knob_rots=_g("ch3_k1_knob_rots_at")(ws, we, bb),
        knob_scales=[1.0, 1.0, 1.0], ax_data=ax_data,
    )
    _g("_ch3_align_knob_axes_under_data")(fig, ax_data, axes_k)
    _g("ch3_layout_knob_axes_like_bridge_end")(fig, ax_data, axes_k)

    marker_z_mode = str(marker_z_mode)
    z_lim = (0.0, CH6_SURFACE_Z_HI)
    xy_lim = None
    if marker_z_mode == "bias" and surface is None and stems is None:
        if marker_axis_lim is not None:
            lo, hi = float(marker_axis_lim[0]), float(marker_axis_lim[1])
            z_lim = (lo, hi)
            xy_lim = (lo, hi)
        else:
            z_lim = _zlim_from_bias_points(markers=markers, highlight_w=highlight_w)
    _style_ax3d(
        ax3d, z_lim=z_lim, zlabel=zlabel, xy_lim=xy_lim, azim=view_azim,
        minimal_ui=minimal_ui,
    )

    if density_W is not None and density_axis_lim is not None:
        proj_u = float(density_projection_u)
        rev_u = float(density_reveal_u)
        proj_fade = max(0.0, 1.0 - rev_u)
        eff_proj_u = proj_u * proj_fade
        if eff_proj_u > 1e-6:
            _draw_all_plane_projections(
                ax3d, density_W, density_axis_lim,
                reveal_u=eff_proj_u,
                marker_s=max(6.0, float(marker_s) * 0.42),
            )
        if rev_u > 1e-6:
            _draw_all_density_histograms(
                ax3d, density_W, density_axis_lim,
                bins=int(density_bins), reveal_u=rev_u,
                bar_pad=density_bar_pad,
            )

    u = float(np.clip(morph_u, 0.0, 1.0))
    if stems is not None and u < 1.0 - 1e-6:
        _draw_stems(ax3d, stems, alpha=1.0 - u)
    if ghost_surfaces:
        n_g = len(ghost_surfaces)
        for gi, gs in enumerate(ghost_surfaces):
            if ghost_alpha is not None:
                al = float(ghost_alpha)
            else:
                # Older ghosts slightly more transparent
                t = gi / max(n_g - 1, 1)
                al = 0.10 + 0.10 * t
            _draw_surface_ghost(ax3d, gs, alpha=al)
    if surface is not None and u > 1e-6:
        _draw_surface(
            ax3d, surface,
            alpha=(0.20 + 0.80 * u) * CH5_BELIEF_SURFACE_ALPHA,
            palette=surface_palette,
        )
    elif surface is not None and stems is None:
        _draw_surface(ax3d, surface, alpha=surface_alpha, palette=surface_palette)

    if markers is not None:
        if marker_z_mode == "bias":
            _draw_markers(
                ax3d, markers, s=marker_s,
                z=np.asarray(markers, dtype=np.float64)[:, 2],
            )
        elif marker_z_mode == "surface" and surface is not None:
            _draw_markers(
                ax3d, markers, s=marker_s,
                z=_marker_z_on_surface(surface, markers),
            )
        else:
            _draw_markers(ax3d, markers, s=marker_s)
    if highlight_w is not None:
        hw = np.asarray(highlight_w, dtype=np.float64).reshape(1, 3)
        hcol = CH6_LINE_COLOR if highlight_color is None else str(highlight_color)
        if marker_z_mode == "bias":
            _draw_markers(
                ax3d, hw, color=hcol, s=highlight_marker_s, alpha=1.0,
                z=[float(hw[0, 2])],
            )
        elif marker_z_mode == "surface" and surface is not None:
            _draw_markers(
                ax3d, hw, color=hcol, s=highlight_marker_s, alpha=1.0,
                z=_marker_z_on_surface(surface, hw),
            )
        else:
            _draw_markers(
                ax3d, hw, color=hcol, s=highlight_marker_s, alpha=1.0, z=[0.05],
            )

    if ellipsoid_loops:
        _draw_gaussian_ellipsoids(
            ax3d, ellipsoid_loops, color=CH6_COV_COLOR, alpha=float(ellipsoid_alpha),
        )
    if sampling_ellipsoid is not None:
        se = sampling_ellipsoid
        _draw_sampling_ellipsoids_ch5_style(
            ax3d,
            se["mean"],
            se["cov"],
            reveal_u=float(sampling_ellipsoid_reveal_u),
            mass=float(se.get("mass", 0.95)),
        )
    if lr_level is not None and surface is not None and "Z_lr" in surface:
        thr = float(lr_level)
        ax3d.contour(
            surface["W1"], surface["W2"], surface["Z_lr"],
            levels=[thr], offset=0.0, colors=[CH6_LR_COLOR], linewidths=2.0, zorder=6,
        )
    if show_wald and highlight_w is not None and base_study is not None:
        Xd = ch6_design(base_study, base_exam)
        cov = ch6_wald_cov(highlight_w, Xd, base_y, ridge=CH6_RIDGE)
        ex, ey = ch6_wald_ellipse_w12(
            highlight_w, cov, mass=CH6_CONF_MASS, b_fixed=float(highlight_w[2]),
        )
        ax3d.plot(ex, ey, np.zeros_like(ex), color=CH6_WALD_COLOR, lw=2.0, zorder=7)

    if floor_ellipse is not None:
        # floor_ellipse: dict with mean (len>=2), cov 2x2, color, mass optional
        mu = np.asarray(floor_ellipse["mean"], dtype=np.float64)
        C = np.asarray(floor_ellipse["cov"], dtype=np.float64)
        mass = float(floor_ellipse.get("mass", CH6_CONF_MASS))
        color = floor_ellipse.get("color", CH6_COV_COLOR)
        # Build a fake 3-vector / 3x3 so wald helper's marginal path works
        mean3 = np.array([mu[0], mu[1], 0.0])
        cov3 = np.eye(3)
        cov3[:2, :2] = C
        ex, ey = ch6_wald_ellipse_w12(mean3, cov3, mass=mass, b_fixed=None)
        ax3d.plot(ex, ey, np.zeros_like(ex), color=color, lw=2.2, zorder=7)

    if probe_pts is not None:
        P = np.asarray(probe_pts, dtype=np.float64)
        if P.ndim == 2 and len(P) >= 2:
            # Default: lift onto the surface. Flat: stay in the (w_ST, w_EL) floor.
            if probe_flat or surface is None:
                zz = np.full(len(P), 0.02)
            else:
                zz = np.full(len(P), 0.08)
                for i, pt in enumerate(P):
                    zz[i] = _surface_z_at(surface, pt[0], pt[1]) + 0.02
            ax3d.plot(
                P[:, 0], P[:, 1], zz,
                color=probe_color, lw=3.4 if len(P) >= 20 else 2.8, zorder=9,
            )
            ax3d.scatter(
                P[[0, -1], 0], P[[0, -1], 1], zz[[0, -1]],
                s=55, c=probe_color, depthshade=False, zorder=10,
            )

    if weight_grad is not None:
        origin = weight_grad_w if weight_grad_w is not None else (
            highlight_w if highlight_w is not None else w_live
        )
        if origin is not None:
            _draw_weight_grad_arrow(
                ax3d, origin, weight_grad,
                surface=surface, scale=float(weight_grad_scale),
                ascent=bool(weight_grad_ascent),
                pin_to_surface=not bool(weight_grad_floor_only),
                color=str(weight_grad_color),
            )

    if spring_arrow is not None and surface is not None:
        sa = dict(spring_arrow)
        head = sa.pop("head_xy", None)
        if head is None and highlight_w is not None:
            head = np.asarray(highlight_w, dtype=np.float64)[:2]
        if head is not None and "away_dir" in sa:
            _draw_spring_compress_arrow(
                ax3d, surface, head, sa.pop("away_dir"),
                compress_u=float(sa.get("compress_u", 1.0)),
                scale=float(sa.get("scale", 1.0)),
                color=str(sa.get("color", CH6_VARIANCE_RED)),
            )

    if show_mean_x and mean_x is not None:
        _draw_mean_x(ax3d, fig, mean_x, fontsize=mean_x_fontsize)

    if title_right:
        ax3d.text2D(
            0.02, 0.98, title_right, transform=ax3d.transAxes,
            va="top", ha="left", fontsize=9.5, color="#222", fontweight="bold",
        )
    return _fig_to_plot(fig)


def _frame_card(lines, *, title=None, subtitle=None):
    """Full-bleed text card on the classic light canvas (Ch6_12 style)."""
    fig, ax = plt.subplots(figsize=_g("CH4_DUO_FIGSIZE") if "CH4_DUO_FIGSIZE" in _G else (12.8, 7.2))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box = FancyBboxPatch(
        (0.08, 0.14), 0.84, 0.72,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#f7f9fb", edgecolor="#bbb", lw=1.2,
    )
    ax.add_patch(box)
    y = 0.78
    if title:
        ax.text(0.5, y, title, ha="center", va="center",
                fontsize=15, fontweight="bold", color="#222")
        y -= 0.10
    if subtitle:
        ax.text(0.5, y, subtitle, ha="center", va="center",
                fontsize=11, color="#444")
        y -= 0.08
    for line in lines:
        ax.text(0.5, y, line, ha="center", va="center",
                fontsize=10.5, color="#333", family="monospace")
        y -= 0.07
    return _fig_to_plot(fig)


def _frame_marginal_hist(
    values, *, mean, std, title_left, title_right, xlabel=r"$w_{\mathrm{ST}}$",
):
    """1D histogram of a single parameter with mean ± std brackets."""
    fig, ax_data, ax3d, axes_k = _g("ch4_figure_duo_weight3d")()
    # Hide the unused 3D / knobs by painting a focused 1D panel on ax_data,
    # and a twin annotation on ax3d.
    for a in (ax3d, *axes_k):
        a.set_visible(False)
    vals = np.asarray(values, dtype=np.float64)
    ax_data.cla()
    ax_data.hist(vals, bins=18, color=CH6_GHOST_COLOR, edgecolor="white",
                 alpha=0.85, density=True)
    ax_data.axvline(mean, color=CH6_MEAN_LINE_COLOR, lw=2.4, label="mean")
    ax_data.axvspan(mean - std, mean + std, color=CH6_VAR_COLOR, alpha=0.18)
    ax_data.axvline(mean - std, color=CH6_VAR_COLOR, lw=1.4, ls="--")
    ax_data.axvline(mean + std, color=CH6_VAR_COLOR, lw=1.4, ls="--")
    ax_data.set_xlabel(xlabel, fontsize=_g("AXIS_LABEL_SIZE"))
    ax_data.set_ylabel("density", fontsize=_g("AXIS_LABEL_SIZE"))
    ax_data.set_title("")
    ax_data.text(
        0.02, 0.98, title_left, transform=ax_data.transAxes,
        va="top", ha="left", fontsize=9.5, color="#222", fontweight="bold",
    )
    ax_data.text(
        0.98, 0.98, title_right, transform=ax_data.transAxes,
        va="top", ha="right", fontsize=9.5, color="#222", fontweight="bold",
    )
    ax_data.legend(loc="upper right", fontsize=8)
    # Place a variance callout where the 3D panel was.
    ax3d.set_visible(True)
    ax3d.cla()
    ax3d.set_axis_off()
    ax3d.text2D(
        0.5, 0.55,
        f"variance  ≈  {std * std:.3f}\n"
        f"std      ≈  {std:.3f}",
        transform=ax3d.transAxes, ha="center", va="center",
        fontsize=13, family="monospace", color="#222",
    )
    return _fig_to_plot(fig)


# ---------------------------------------------------------------------------
# Clip builders — progression: sample → landings → likelihood
# ---------------------------------------------------------------------------

def build_ch6_01_observed_line(clip_id):
    """Observed D1 + empty 3D floor: pose the wiggle question."""
    study, exam, y = ch5_unpack_dataset("D1")
    xlim, ylim = ch5_plot_limits("D1")
    w, _ = ch6_fit_dataset(study, exam, y)
    frames = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=w, knob_w=w,
        title_left="D1 — one classroom",
        title_right="Where do lines land?",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=w, knob_w=w,
        title_left="How much would this line move",
        title_right="on another random classroom?",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_02_generative_coins(clip_id):
    """Fixed X, soft Bernoulli coins — the generative model."""
    study, exam, y = ch5_unpack_dataset("D1")
    xlim, ylim = ch5_plot_limits("D1")
    frames = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_true=CH6_W_TRUE, knob_w=CH6_W_TRUE, p_annotate=True,
        title_left=r"$Y_i\sim\mathrm{Bern}(\sigma(w_{\mathrm{true}}\cdot x_i))$",
        title_right="labels are coin flips",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    rng = np.random.default_rng(7)
    y2 = ch6_resample_labels(study, exam, rng=rng)
    w2, _ = ch6_fit_dataset(study, exam, y2)
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        ghost_study=study, ghost_exam=exam, ghost_y=y2,
        w_live=w2, w_true=CH6_W_TRUE, knob_w=w2,
        title_left="one redraw of the coins",
        title_right="ghost labels → new line",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_03_resample_reel(clip_id):
    """D1 stays; each resample flashes as hue-shifted icons + line, then icons leave."""
    cloud, draws = ch6_precomputed_resamples("D1", n_reps=CH6_N_REEL, seed=1)
    study, exam = cloud["study"], cloud["exam"]
    y_obs = cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    frames = []
    ghosts: list[np.ndarray] = []
    landed: list[np.ndarray] = []

    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y_obs,
        w_live=cloud["w_obs"], knob_w=cloud["w_obs"],
        title_left="Observed D1 stays",
        title_right="start sampling…",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))

    for i, (yb, w) in enumerate(draws):
        landed.append(w)
        markers = np.asarray(landed)
        stems = ch6_landing_histogram(markers) if len(landed) >= 3 else None
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y_obs,
            ghost_study=study, ghost_exam=exam, ghost_y=yb,
            w_live=w, ghost_ws=ghosts, w_true=cloud["w_true"], knob_w=w,
            stems=stems, markers=markers, highlight_w=w,
            title_left=f"classroom {i + 1} / {len(draws)}",
            title_right="ghost dataset + its line",
            zlabel="landings",
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_FLASH))
        ghosts.append(w)
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y_obs,
            ghost_ws=ghosts, w_true=cloud["w_true"], knob_w=w,
            stems=stems, markers=markers, highlight_w=w,
            title_left=f"{len(ghosts)} ghost lines stay",
            title_right=f"{len(landed)} landings",
            zlabel="landings",
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_SEQ_HOLD))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))
    return frames


def build_ch6_04_landing_histogram(clip_id):
    """Keep track of how often MLEs land in the same place — stems rise."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Ws = cloud["weights"]
    frames = []
    steps = _draft_short(28, 10)
    for t in range(steps + 1):
        u = t / steps
        k = max(1, int(round(u * len(Ws))))
        hist = ch6_landing_histogram(Ws[:k])
        ghosts = _pick_ghost_lines(Ws[:k])
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            ghost_ws=ghosts, w_true=cloud["w_true"], knob_w=cloud["w_obs"],
            stems=hist, markers=Ws[:k],
            title_left=f"{k} ghost lines",
            title_right="how often do they land here?",
            zlabel="landing frequency",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_05_looks_like_likelihood(clip_id):
    """Morph landing histogram → relative-likelihood landscape of observed D1."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Xd = ch6_design(study, exam)
    hist = ch6_landing_histogram(cloud["weights"])
    surf = ch6_rel_likelihood_w12(Xd, y, cloud["w_obs"], ridge=CH6_RIDGE)
    ghosts = _pick_ghost_lines(cloud["weights"])
    frames = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        ghost_ws=ghosts, knob_w=cloud["w_obs"],
        stems=hist, morph_u=0.0,
        title_left="landing histogram",
        title_right="where MLEs pile up",
        zlabel="landing frequency",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))
    for u in np.linspace(0.0, 1.0, _draft_short(24, 8)):
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            ghost_ws=ghosts, knob_w=cloud["w_obs"],
            stems=hist, surface=surf, morph_u=float(u),
            highlight_w=cloud["w_obs"],
            title_left="same shape…",
            title_right="as the likelihood!",
            zlabel="Likelihood" if u > 0.5 else "landing frequency",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=cloud["w_obs"], ghost_ws=ghosts, knob_w=cloud["w_obs"],
        surface=surf, morph_u=1.0, highlight_w=cloud["w_obs"],
        title_left="likelihood of the observed D1",
        title_right=r"penalized $L_\lambda(w)/L_\lambda(\hat w)$",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_06_likelihood_bowl(clip_id):
    """Likelihood landscape with sampling cloud sitting on it."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Xd = ch6_design(study, exam)
    surf = ch6_rel_likelihood_w12(Xd, y, cloud["w_obs"], ridge=CH6_RIDGE)
    ghosts = _pick_ghost_lines(cloud["weights"])
    frames = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=cloud["w_obs"], ghost_ws=ghosts, knob_w=cloud["w_obs"],
        surface=surf, markers=cloud["weights"], highlight_w=cloud["w_obs"],
        title_left="observed fit + ghost lines",
        title_right="MLE cloud on the likelihood",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_07_lr_confidence(clip_id):
    """Grow LR confidence contour on the likelihood floor."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Xd = ch6_design(study, exam)
    surf = ch6_rel_likelihood_w12(Xd, y, cloud["w_obs"], ridge=CH6_RIDGE)
    ghosts = _pick_ghost_lines(cloud["weights"])
    frames = []
    masses = np.linspace(0.3, CH6_CONF_MASS, _draft_short(16, 6))
    for m in masses:
        thr = ch6_lr_threshold(float(m))
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            w_live=cloud["w_obs"], ghost_ws=ghosts, knob_w=cloud["w_obs"],
            surface=surf, markers=cloud["weights"], highlight_w=cloud["w_obs"],
            lr_level=thr,
            title_left="likelihood-ratio region",
            title_right=f"{100 * m:.0f}% confidence",
            zlabel="Likelihood",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_08_wald_vs_lr(clip_id):
    """Wald ellipse as quadratic approximation to the LR cut."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Xd = ch6_design(study, exam)
    surf = ch6_rel_likelihood_w12(Xd, y, cloud["w_obs"], ridge=CH6_RIDGE)
    thr = ch6_lr_threshold(CH6_CONF_MASS)
    frames = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=cloud["w_obs"], knob_w=cloud["w_obs"],
        surface=surf, markers=cloud["weights"], highlight_w=cloud["w_obs"],
        lr_level=thr,
        title_left="LR 95% contour",
        title_right="cut the likelihood bowl",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=cloud["w_obs"], knob_w=cloud["w_obs"],
        surface=surf, markers=cloud["weights"], highlight_w=cloud["w_obs"],
        lr_level=thr, show_wald=True,
        title_left="Wald ellipse ≈ quadratic bowl",
        title_right="same 95%, local curvature",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_09_separation(clip_id):
    """Near-separation: unpenalized MLE runs up the ridge."""
    study, exam, y_obs = ch5_unpack_dataset("D1")
    xlim, ylim = ch5_plot_limits("D1")
    Xd = ch6_design(study, exam)
    rng = np.random.default_rng(11)
    found = None
    for _ in range(400):
        yb = rng.binomial(1, ch6_sigmoid(Xd @ CH6_W_TRUE)).astype(np.float64)
        w_tiny, _ = ch6_fit(Xd, yb, ridge=1e-4)
        if np.linalg.norm(w_tiny) > 8.0:
            found = (yb, w_tiny)
            break
    if found is None:
        yb = ch6_resample_labels(study, exam, rng=rng)
        w_tiny, _ = ch6_fit(Xd, yb, ridge=1e-4)
        found = (yb, w_tiny)
    yb, w_blow = found
    w_pen, _ = ch6_fit(Xd, yb, ridge=CH6_RIDGE)
    surf = ch6_rel_likelihood_w12(Xd, yb, w_pen, ridge=CH6_RIDGE)
    frames = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y_obs,
        ghost_study=study, ghost_exam=exam, ghost_y=yb,
        w_live=w_blow, w_true=CH6_W_TRUE, knob_w=w_blow,
        surface=surf, highlight_w=w_blow,
        title_left="near-separation — MLE blows up",
        title_right=f"‖w‖ ≈ {np.linalg.norm(w_blow):.0f}",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y_obs,
        ghost_study=study, ghost_exam=exam, ghost_y=yb,
        w_live=w_pen, w_true=CH6_W_TRUE, knob_w=w_pen,
        surface=surf, highlight_w=w_pen,
        title_left="penalized MLE stays finite",
        title_right=r"ridge $\lambda=0.25$",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_10_n_effect(clip_id):
    """D1 vs D3: more data → tighter landings / likelihood."""
    frames = []
    for key, label in (("D1", "D1 · n=20"), ("D3", "D3 · n=60")):
        cloud = ch6_sampling_cloud(key, n_reps=CH6_N_REPS_CLOUD, seed=3)
        study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
        Xd = ch6_design(study, exam)
        surf = ch6_rel_likelihood_w12(Xd, y, cloud["w_obs"], ridge=CH6_RIDGE)
        hist = ch6_landing_histogram(cloud["weights"])
        ghosts = _pick_ghost_lines(cloud["weights"])
        img = _frame_duo(
            xlim=cloud["xlim"], ylim=cloud["ylim"],
            base_study=study, base_exam=exam, base_y=y,
            w_live=cloud["w_obs"], ghost_ws=ghosts, knob_w=cloud["w_obs"],
            stems=hist, surface=surf, morph_u=0.55, markers=cloud["weights"],
            highlight_w=cloud["w_obs"],
            title_left=label,
            title_right=f"angle std ≈ {np.std(cloud['angles']):.1f}°",
            zlabel="Likelihood",
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_11_d4_chaos(clip_id):
    """D4 null process — landings everywhere."""
    cloud = ch6_sampling_cloud("D4", n_reps=CH6_N_REPS_CLOUD, seed=4)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    Xd = ch6_design(study, exam)
    surf = ch6_rel_likelihood_w12(Xd, y, cloud["w_obs"], ridge=CH6_RIDGE)
    hist = ch6_landing_histogram(cloud["weights"])
    ghosts = _pick_ghost_lines(cloud["weights"])
    frames = []
    img = _frame_duo(
        xlim=cloud["xlim"], ylim=cloud["ylim"],
        base_study=study, base_exam=exam, base_y=y,
        w_live=cloud["w_obs"], ghost_ws=ghosts, knob_w=cloud["w_obs"],
        stems=hist, markers=cloud["weights"],
        title_left="D4 — no pattern",
        title_right="landings everywhere",
        zlabel="landing frequency",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))
    img = _frame_duo(
        xlim=cloud["xlim"], ylim=cloud["ylim"],
        base_study=study, base_exam=exam, base_y=y,
        w_live=cloud["w_obs"], ghost_ws=ghosts, knob_w=cloud["w_obs"],
        stems=hist, surface=surf, morph_u=1.0, highlight_w=cloud["w_obs"],
        title_left="flat likelihood",
        title_right=f"angle std ≈ {np.std(cloud['angles']):.1f}°",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_12_bayes_vs_freq(clip_id):
    """Same bowl, different cut — frequentist vs Bayesian reading."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Xd = ch6_design(study, exam)
    surf = ch6_rel_likelihood_w12(Xd, y, cloud["w_obs"], ridge=CH6_RIDGE)
    thr = ch6_lr_threshold(CH6_CONF_MASS)
    frames = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=cloud["w_obs"], knob_w=cloud["w_obs"],
        surface=surf, markers=cloud["weights"], highlight_w=cloud["w_obs"],
        lr_level=thr,
        title_left="Frequentist — cut the likelihood",
        title_right="confidence region",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    fig, ax = plt.subplots(figsize=_g("CH4_DUO_FIGSIZE") if "CH4_DUO_FIGSIZE" in _G else (12.8, 7.2))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box = FancyBboxPatch(
        (0.08, 0.18), 0.84, 0.64,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#f7f9fb", edgecolor="#bbb", lw=1.2,
    )
    ax.add_patch(box)
    ax.text(0.5, 0.72, "Same bowl. Different cut.", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#222")
    ax.text(
        0.5, 0.50,
        "Ch5 Bayesian: prior × likelihood → credible region\n"
        "          (mass under the posterior belief surface)\n\n"
        "Ch6 Frequentist: resample classrooms → landings\n"
        "          pile up in the shape of the likelihood",
        ha="center", va="center", fontsize=10, color="#333", family="monospace",
    )
    ax.text(
        0.5, 0.28,
        "The ridge penalty is the same math as Ch5's Gaussian prior —\n"
        "read as regularization here, not as belief.",
        ha="center", va="center", fontsize=9, color="#555",
    )
    frames.append(_finish(_fig_to_plot(fig), clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_13_class_gaussians(clip_id):
    """Reveal per-class elongated Gaussians inferred from D1 (params on the 2D plot)."""
    from ch5_datasets import CH5_STANDARD_XLIM, CH5_STANDARD_YLIM

    study, exam, y = ch5_unpack_dataset("D1")
    xlim, ylim = CH5_STANDARD_XLIM, CH5_STANDARD_YLIM
    w, _ = ch6_fit_dataset(study, exam, y)
    frames = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=w, knob_w=w,
        title_left="D1 roster",
        title_right="infer the population…",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        show_ellipses=True, ellipse_labels=True, knob_w=w,
        title_left="class Gaussians (elongated ∥ diagonal)",
        title_right="means · priors · shared σ on the plot",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 4))
    return frames


def _build_population_classroom_reel(
    clip_id, *, n_class, seed=20, seed_from_key=None, keep_classroom_on_stage=False,
):
    """Classroom of size n → ghost population (100×) → resample classrooms → likelihood.

    If ``seed_from_key`` is set (e.g. ``\"D1\"``), open on that chapter roster, then
    snap to the nearest same-label population twins before resampling.
    """
    from ch5_datasets import CH5_STANDARD_XLIM, CH5_STANDARD_YLIM

    xlim, ylim = CH5_STANDARD_XLIM, CH5_STANDARD_YLIM
    n_pop = int(CH6_POP_SIZE)
    n_reel = int(CH6_N_POP_REEL)
    pop_s, pop_e, pop_y = ch6_sample_population(n_pop, seed=seed + 11)

    if seed_from_key is not None:
        tgt_s, tgt_e, tgt_y = ch5_unpack_dataset(str(seed_from_key))
        # Opening roster is the canonical dataset; matched twins live in the pop.
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
    frames = []
    # Clean stage: no 2D legend / overlay titles (HQ population reels).
    duo_kw = dict(show_legend=False, title_left=None, title_right=None)

    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=open_s, base_exam=open_e, base_y=open_y,
        w_live=w_open, knob_w=w_open, show_ellipses=True,
        zlabel="landings", **duo_kw,
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    for u in np.linspace(0.0, 1.0, _draft_short(12, 5)):
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=open_s, base_exam=open_e, base_y=open_y,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            pop_alpha=0.04 + 0.14 * float(u),
            w_live=w_open, knob_w=w_open, show_ellipses=(u < 0.35),
            zlabel="landings", **duo_kw,
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))

    # Snap canonical roster → nearest population twins (no-op when already seeded).
    if match_s is not None:
        for u in np.linspace(0.0, 1.0, _draft_short(10, 4)):
            uu = float(u)
            # Crossfade positions: interpolate features toward matched twins.
            bs = (1.0 - uu) * open_s + uu * match_s
            be = (1.0 - uu) * open_e + uu * match_e
            ww = (1.0 - uu) * w_open + uu * w_seed
            img = _frame_duo(
                xlim=xlim, ylim=ylim,
                base_study=bs, base_exam=be, base_y=open_y,
                pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                w_live=ww, knob_w=ww,
                zlabel="landings", **duo_kw,
            )
            frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD))

    rng = np.random.default_rng(seed + 99)
    ghosts: list[np.ndarray] = []
    landed: list[np.ndarray] = []
    last_s, last_e, last_y, last_w = seed_s, seed_e, seed_y, w_seed
    for i in range(n_reel):
        if i == 0 and match_s is not None:
            # First landing is the D1-matched classroom itself.
            cs, ce, cy = seed_s, seed_e, seed_y
        else:
            cs, ce, cy, _ = ch6_draw_classroom_from_population(
                pop_s, pop_e, pop_y, n_class, rng=rng,
            )
        w, _ = ch6_fit_population_classroom(cs, ce, cy)
        landed.append(w)
        markers = np.asarray(landed)
        stems = ch6_landing_histogram(markers) if len(landed) >= 3 else None
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=cs, base_exam=ce, base_y=cy,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            w_live=w, ghost_ws=ghosts, knob_w=w,
            stems=stems, markers=markers, highlight_w=w,
            zlabel="landings", **duo_kw,
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_FLASH))
        ghosts.append(w)
        if not keep_classroom_on_stage:
            img = _frame_duo(
                xlim=xlim, ylim=ylim,
                base_study=cs[:0], base_exam=ce[:0], base_y=cy[:0],
                show_base=False,
                pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                ghost_ws=ghosts, knob_w=w,
                stems=stems, markers=markers, highlight_w=w,
                zlabel="landings", **duo_kw,
            )
            frames.append(_finish(img, clip_id))
            frames.extend(_hold(frames[-1], CH6_N_SEQ_HOLD))
        else:
            frames.extend(_hold(frames[-1], CH6_N_SEQ_HOLD))
        last_s, last_e, last_y, last_w = cs, ce, cy, w

    Xd = ch6_design(last_s, last_e)
    surf = ch6_rel_likelihood_w12(
        Xd, last_y, last_w, ridge=CH6_RIDGE, grid=CH6_LR_GRID_SMOOTH,
    )
    hist = ch6_landing_histogram(np.asarray(landed))
    for u in np.linspace(0.0, 1.0, _draft_short(18, 7)):
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=last_s, base_exam=last_e, base_y=last_y,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            pop_alpha=0.10,
            w_live=last_w, ghost_ws=ghosts, knob_w=last_w,
            stems=hist, surface=surf, morph_u=float(u),
            markers=np.asarray(landed), highlight_w=last_w,
            zlabel="Likelihood" if u > 0.5 else "landing frequency",
            **duo_kw,
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_14_population_reel_n20(clip_id):
    return _build_population_classroom_reel(clip_id, n_class=CH6_N_CLASS_BASE, seed=20)


def build_ch6_15_population_reel_n8(clip_id):
    return _build_population_classroom_reel(clip_id, n_class=CH6_N_CLASS_SMALL, seed=8)


def build_ch6_16_population_reel_n60(clip_id):
    return _build_population_classroom_reel(clip_id, n_class=CH6_N_CLASS_LARGE, seed=60)


def build_ch6_17_population_reel_d1(clip_id):
    """Like ch6_14, but opens on Ch5 D1 and snaps to nearest population twins."""
    return _build_population_classroom_reel(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
    )


def _population_param_cloud_landings(
    *,
    n_class,
    seed=20,
    seed_from_key=None,
    n_reel=None,
):
    """Shared population reel: return classroom landings and opening state."""
    return ch6_population_param_cloud_pack(
        n_class, seed=seed, seed_from_key=seed_from_key, n_reel=n_reel,
    )


def _build_population_classroom_reel_param_cloud(
    clip_id, *, n_class, seed=20, seed_from_key=None, n_reel=None, fast=False,
    marker_axis_lim=None, keep_classroom_on_stage=False,
):
    """Like ch6_14-17 but right panel is a 3D parameter cloud: (w_ST, w_EL, b)."""
    pack = _population_param_cloud_landings(
        n_class=n_class, seed=seed, seed_from_key=seed_from_key, n_reel=n_reel,
    )
    xlim, ylim = pack["xlim"], pack["ylim"]
    pop_s, pop_e, pop_y = pack["pop_s"], pack["pop_e"], pack["pop_y"]
    open_s, open_e, open_y = pack["open_s"], pack["open_e"], pack["open_y"]
    match_s = pack["match_s"]
    w_open, w_seed = pack["w_open"], pack["w_seed"]
    frames = []
    duo_kw = dict(show_legend=False, title_left=None, title_right=None)
    n_hold_open = CH6_N_HOLD if fast else CH6_N_HOLD * 2
    n_hold_pop = max(1, CH6_N_HOLD // 2) if fast else CH6_N_HOLD
    n_flash = 0 if fast else CH6_N_FLASH
    n_seq_hold = 0 if fast else CH6_N_SEQ_HOLD
    n_end_hold = CH6_N_HOLD if fast else CH6_N_HOLD * 2
    cloud_kw = dict(marker_z_mode="bias", marker_axis_lim=marker_axis_lim, zlabel="b")

    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=open_s, base_exam=open_e, base_y=open_y,
        w_live=w_open, knob_w=w_open, show_ellipses=True,
        markers=np.asarray([w_open]), highlight_w=w_open,
        **cloud_kw, **duo_kw,
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold_open))

    for u in np.linspace(0.0, 1.0, _draft_short(12, 5)):
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=open_s, base_exam=open_e, base_y=open_y,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            pop_alpha=0.04 + 0.14 * float(u),
            w_live=w_open, knob_w=w_open, show_ellipses=(u < 0.35),
            markers=np.asarray([w_open]), highlight_w=w_open,
            **cloud_kw, **duo_kw,
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold_pop))

    if match_s is not None:
        for u in np.linspace(0.0, 1.0, _draft_short(10, 4)):
            uu = float(u)
            bs = (1.0 - uu) * open_s + uu * match_s
            be = (1.0 - uu) * open_e + uu * pack["match_e"]
            ww = (1.0 - uu) * w_open + uu * w_seed
            img = _frame_duo(
                xlim=xlim, ylim=ylim,
                base_study=bs, base_exam=be, base_y=open_y,
                pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                w_live=ww, knob_w=ww,
                markers=np.asarray([ww]), highlight_w=ww,
                **cloud_kw, **duo_kw,
            )
            frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], n_hold_pop))

    for step in pack["reel_steps"]:
        cs, ce, cy = step["study"], step["exam"], step["y"]
        w = step["w"]
        markers = step["markers"]
        ghosts = step["ghosts"]
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=cs, base_exam=ce, base_y=cy,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            w_live=w, ghost_ws=ghosts, knob_w=w,
            markers=markers, highlight_w=w,
            **cloud_kw, **duo_kw,
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], n_flash))
        if not keep_classroom_on_stage:
            img = _frame_duo(
                xlim=xlim, ylim=ylim,
                base_study=cs[:0], base_exam=ce[:0], base_y=cy[:0],
                show_base=False,
                pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                ghost_ws=ghosts + [w], knob_w=w,
                markers=markers, highlight_w=w,
                **cloud_kw, **duo_kw,
            )
            frames.append(_finish(img, clip_id))
            frames.extend(_hold(frames[-1], n_seq_hold))
        else:
            frames.extend(_hold(frames[-1], n_seq_hold))

    frames.extend(_hold(frames[-1], n_end_hold))
    return frames


def _build_population_reel_param_cloud_dist(
    clip_id,
    *,
    n_class,
    seed=20,
    seed_from_key=None,
    n_reel=None,
    fast=False,
    marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    keep_classroom_on_stage=False,
    density_histograms=False,
    density_at_end=False,
    density_show_mean_x=False,
    density_bins=None,
    density_bar_pad=None,
):
    """Point-cloud reel; ellipsoids + spin, live density, or end-of-reel density reveal."""
    from ch6_frequentist import _CH3_DRAFT

    pack = _population_param_cloud_landings(
        n_class=n_class, seed=seed, seed_from_key=seed_from_key, n_reel=n_reel,
    )
    reel_stats = ch6_param_stats(pack["landed"])
    sampling_ellipsoid = {
        "mean": reel_stats["mean"],
        "cov": reel_stats["cov"],
        "mass": 0.95,
    }
    dist_mean = reel_stats["mean"]
    if density_bins is None:
        density_bins = 20 if _CH3_DRAFT else 28

    xlim, ylim = pack["xlim"], pack["ylim"]
    pop_s, pop_e, pop_y = pack["pop_s"], pack["pop_e"], pack["pop_y"]
    open_s, open_e, open_y = pack["open_s"], pack["open_e"], pack["open_y"]
    match_s = pack["match_s"]
    w_open, w_seed = pack["w_open"], pack["w_seed"]
    frames = []
    duo_kw = dict(show_legend=False, title_left=None, title_right=None)
    n_hold_open = CH6_N_HOLD if fast else CH6_N_HOLD * 2
    n_hold_pop = max(1, CH6_N_HOLD // 2) if fast else CH6_N_HOLD
    n_flash = 0 if fast else CH6_N_FLASH
    n_seq_hold = 0 if fast else CH6_N_SEQ_HOLD
    cloud_kw = dict(marker_z_mode="bias", marker_axis_lim=marker_axis_lim, zlabel="b")
    base_azim = float(_g("CH3_LIK_W12_CT_AZIM"))

    def _cloud_frame(**extra):
        return _frame_duo(xlim=xlim, ylim=ylim, pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                          **cloud_kw, **duo_kw, **extra)

    def _density_kw(markers_arr, *, reveal_u=1.0, projection_u=0.0):
        if not density_histograms:
            return {}
        return dict(
            density_W=np.asarray(markers_arr, dtype=np.float64),
            density_axis_lim=marker_axis_lim,
            density_reveal_u=float(reveal_u),
            density_projection_u=float(projection_u),
            density_bins=density_bins,
            density_bar_pad=density_bar_pad,
        )

    def _end_density_kw(*, reveal_u=0.0, projection_u=0.0):
        if not density_at_end:
            return {}
        return dict(
            density_W=markers,
            density_axis_lim=marker_axis_lim,
            density_reveal_u=float(reveal_u),
            density_projection_u=float(projection_u),
            density_bins=density_bins,
            density_bar_pad=density_bar_pad,
        )

    # --- opening (same as cloud reel) ---
    img = _cloud_frame(
        base_study=open_s, base_exam=open_e, base_y=open_y,
        w_live=w_open, knob_w=w_open, show_ellipses=True,
        markers=np.asarray([w_open]), highlight_w=w_open,
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold_open))

    for u in np.linspace(0.0, 1.0, _draft_short(12, 5)):
        img = _cloud_frame(
            base_study=open_s, base_exam=open_e, base_y=open_y,
            pop_alpha=0.04 + 0.14 * float(u),
            w_live=w_open, knob_w=w_open, show_ellipses=(u < 0.35),
            markers=np.asarray([w_open]), highlight_w=w_open,
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_hold_pop))

    if match_s is not None:
        for u in np.linspace(0.0, 1.0, _draft_short(10, 4)):
            uu = float(u)
            bs = (1.0 - uu) * open_s + uu * match_s
            be = (1.0 - uu) * open_e + uu * pack["match_e"]
            ww = (1.0 - uu) * w_open + uu * w_seed
            img = _cloud_frame(
                base_study=bs, base_exam=be, base_y=open_y,
                w_live=ww, knob_w=ww,
                markers=np.asarray([ww]), highlight_w=ww,
            )
            frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], n_hold_pop))

    last_step = pack["reel_steps"][-1]
    for step in pack["reel_steps"]:
        cs, ce, cy = step["study"], step["exam"], step["y"]
        w = step["w"]
        markers = step["markers"]
        ghosts = step["ghosts"]
        img = _cloud_frame(
            base_study=cs, base_exam=ce, base_y=cy,
            w_live=w, ghost_ws=ghosts, knob_w=w,
            markers=markers, highlight_w=w,
            **_density_kw(markers),
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], n_flash))
        if not keep_classroom_on_stage:
            img = _cloud_frame(
                base_study=cs[:0], base_exam=ce[:0], base_y=cy[:0],
                show_base=False,
                ghost_ws=ghosts + [w], knob_w=w,
                markers=markers, highlight_w=w,
                **_density_kw(markers),
            )
            frames.append(_finish(img, clip_id))
            frames.extend(_hold(frames[-1], n_seq_hold))
        else:
            frames.extend(_hold(frames[-1], n_seq_hold))

    # --- overlay sampling distribution on the final cloud ---
    cs, ce, cy = last_step["study"], last_step["exam"], last_step["y"]
    markers = last_step["markers"]
    ghosts = last_step["ghosts"]
    w = last_step["w"]

    def _final_frame(
        *,
        ell_alpha=0.0,
        show_x=False,
        azim=base_azim,
        density_reveal_u=0.0,
        density_projection_u=0.0,
    ):
        roster_kw = (
            dict(base_study=cs, base_exam=ce, base_y=cy, show_base=True)
            if keep_classroom_on_stage
            else dict(base_study=cs[:0], base_exam=ce[:0], base_y=cy[:0], show_base=False)
        )
        dens_kw = {}
        if density_histograms:
            dens_kw = dict(
                density_W=markers,
                density_axis_lim=marker_axis_lim,
                density_reveal_u=float(density_reveal_u),
                density_projection_u=float(density_projection_u),
                density_bins=density_bins,
                density_bar_pad=density_bar_pad,
            )
        elif density_at_end:
            dens_kw = _end_density_kw(
                reveal_u=density_reveal_u,
                projection_u=density_projection_u,
            )
        return _cloud_frame(
            **roster_kw,
            ghost_ws=ghosts, knob_w=w,
            markers=markers, highlight_w=w,
            sampling_ellipsoid=(
                None if (density_histograms or density_at_end) else (
                    sampling_ellipsoid if ell_alpha > 0 else None
                )
            ),
            sampling_ellipsoid_reveal_u=float(ell_alpha),
            show_mean_x=bool(show_x) and (
                density_show_mean_x if density_histograms else True
            ),
            mean_x=dist_mean,
            view_azim=azim,
            **dens_kw,
        )

    frames.extend(_hold(_finish(_final_frame(), clip_id), CH6_N_HOLD))

    if density_at_end:
        n_proj = _draft_short(32, 16)
        for u in np.linspace(0.0, 1.0, n_proj):
            img = _final_frame(density_projection_u=float(u), density_reveal_u=0.0)
            frames.append(_finish(img, clip_id))
        frames.extend(_hold(
            _finish(_final_frame(density_projection_u=1.0, density_reveal_u=0.0), clip_id),
            CH6_N_HOLD * 2,
        ))
        n_rise = _draft_short(72, 36)
        for u in np.linspace(0.0, 1.0, n_rise):
            uu = float(u) ** 1.35
            img = _final_frame(density_projection_u=1.0, density_reveal_u=uu)
            frames.append(_finish(img, clip_id))
        frames.extend(_hold(
            _finish(_final_frame(density_projection_u=1.0, density_reveal_u=1.0), clip_id),
            CH6_N_HOLD * 2,
        ))
        n_spin = _draft_short(72, 28)
        for t in range(n_spin + 1):
            az = base_azim + 360.0 * t / n_spin
            img = _final_frame(
                density_projection_u=1.0,
                density_reveal_u=1.0,
                azim=az,
            )
            frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
        return frames

    if density_histograms:
        frames.extend(_hold(
            _finish(_final_frame(density_reveal_u=1.0), clip_id),
            CH6_N_HOLD * 2,
        ))
        if density_show_mean_x:
            for u in np.linspace(0.0, 1.0, _draft_short(8, 4)):
                img = _final_frame(density_reveal_u=1.0, show_x=(u > 0.2))
                frames.append(_finish(img, clip_id))
            frames.extend(_hold(
                _finish(_final_frame(density_reveal_u=1.0, show_x=True), clip_id),
                CH6_N_HOLD * 2,
            ))
        n_spin = _draft_short(72, 28)
        for t in range(n_spin + 1):
            az = base_azim + 360.0 * t / n_spin
            img = _final_frame(
                density_reveal_u=1.0,
                show_x=density_show_mean_x,
                azim=az,
            )
            frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
        return frames

    for u in np.linspace(0.0, 1.0, _draft_short(14, 6)):
        img = _final_frame(ell_alpha=float(u))
        frames.append(_finish(img, clip_id))
    for u in np.linspace(0.0, 1.0, _draft_short(8, 4)):
        img = _final_frame(ell_alpha=1.0, show_x=(u > 0.2))
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(_finish(_final_frame(ell_alpha=1.0, show_x=True), clip_id), CH6_N_HOLD * 2))

    n_spin = _draft_short(72, 28)
    for t in range(n_spin + 1):
        az = base_azim + 360.0 * t / n_spin
        img = _final_frame(ell_alpha=1.0, show_x=True, azim=az)
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_53_population_reel_cloud_n20(clip_id):
    return _build_population_classroom_reel_param_cloud(
        clip_id,
        n_class=CH6_N_CLASS_BASE,
        seed=17,
        seed_from_key="D1",
        n_reel=CH6_POPULATION_DIST_N_REEL,
        fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    )


def build_ch6_54_population_reel_cloud_n8(clip_id):
    return _build_population_classroom_reel_param_cloud(
        clip_id, n_class=CH6_N_CLASS_SMALL, seed=8,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    )


def build_ch6_55_population_reel_cloud_n60(clip_id):
    return _build_population_classroom_reel_param_cloud(
        clip_id, n_class=CH6_N_CLASS_LARGE, seed=60,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    )


def build_ch6_56_population_reel_cloud_d1(clip_id):
    return _build_population_classroom_reel_param_cloud(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    )


def build_ch6_57_population_reel_cloud_dist_n20(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id,
        n_class=CH6_N_CLASS_BASE,
        seed=17,
        seed_from_key="D1",
        n_reel=CH6_POPULATION_DIST_N_REEL,
        fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    )


def build_ch6_58_population_reel_cloud_dist_n8(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id,
        n_class=CH6_N_CLASS_SMALL,
        seed=8,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    )


def build_ch6_59_population_reel_cloud_dist_n60(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id,
        n_class=CH6_N_CLASS_LARGE,
        seed=60,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    )


def build_ch6_60_population_reel_cloud_dist_d1(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id,
        n_class=CH6_N_CLASS_BASE,
        seed=17,
        seed_from_key="D1",
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
    )


def build_ch6_61_population_reel_n20_keep(clip_id):
    return _build_population_classroom_reel(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=20, keep_classroom_on_stage=True,
    )


def build_ch6_62_population_reel_n8_keep(clip_id):
    return _build_population_classroom_reel(
        clip_id, n_class=CH6_N_CLASS_SMALL, seed=8, keep_classroom_on_stage=True,
    )


def build_ch6_63_population_reel_n60_keep(clip_id):
    return _build_population_classroom_reel(
        clip_id, n_class=CH6_N_CLASS_LARGE, seed=60, keep_classroom_on_stage=True,
    )


def build_ch6_64_population_reel_d1_keep(clip_id):
    return _build_population_classroom_reel(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
        keep_classroom_on_stage=True,
    )


def build_ch6_65_population_reel_cloud_n20_keep(clip_id):
    return _build_population_classroom_reel_param_cloud(
        clip_id,
        n_class=CH6_N_CLASS_BASE,
        seed=17,
        seed_from_key="D1",
        n_reel=CH6_POPULATION_DIST_N_REEL,
        fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True,
    )


def build_ch6_66_population_reel_cloud_n8_keep(clip_id):
    return _build_population_classroom_reel_param_cloud(
        clip_id, n_class=CH6_N_CLASS_SMALL, seed=8,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True,
    )


def build_ch6_67_population_reel_cloud_n60_keep(clip_id):
    return _build_population_classroom_reel_param_cloud(
        clip_id, n_class=CH6_N_CLASS_LARGE, seed=60,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True,
    )


def build_ch6_68_population_reel_cloud_d1_keep(clip_id):
    return _build_population_classroom_reel_param_cloud(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True,
    )


def build_ch6_69_population_reel_cloud_dist_n20_keep(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id,
        n_class=CH6_N_CLASS_BASE,
        seed=17,
        seed_from_key="D1",
        n_reel=CH6_POPULATION_DIST_N_REEL,
        fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True,
    )


def build_ch6_70_population_reel_cloud_dist_n8_keep(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id,
        n_class=CH6_N_CLASS_SMALL,
        seed=8,
        n_reel=CH6_POPULATION_DIST_N_REEL,
        fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True,
    )


def build_ch6_71_population_reel_cloud_dist_n60_keep(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id,
        n_class=CH6_N_CLASS_LARGE,
        seed=60,
        n_reel=CH6_POPULATION_DIST_N_REEL,
        fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True,
    )


def build_ch6_72_population_reel_cloud_dist_d1_keep(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id,
        n_class=CH6_N_CLASS_BASE,
        seed=17,
        seed_from_key="D1",
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True,
    )


def build_ch6_73_population_reel_cloud_dist_n300_keep(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id,
        n_class=CH6_N_CLASS_XLARGE,
        seed=300,
        n_reel=CH6_POPULATION_DIST_N_REEL,
        fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True,
    )


def _build_population_n_sweep_cloud(
    clip_id,
    *,
    rotate=False,
    show_mean_x=False,
    mean_x_fontsize=20,
    density_histograms=False,
):
    """n=6…500 sweep: left ghost lines + roster, right 3D cloud."""
    from ch6_frequentist import _CH3_DRAFT

    axis_lim, states = ch6_population_n_sweep_states(seed=17)
    base_azim = float(_g("CH3_LIK_W12_CT_AZIM")) if rotate else None
    n_steps = len(states)
    frames = []
    duo_kw = dict(show_legend=False, title_left=None, title_right=None)
    cloud_kw = dict(marker_z_mode="bias", marker_axis_lim=axis_lim, zlabel="b")

    for i, st in enumerate(states):
        azim = None
        if rotate and base_azim is not None:
            azim = base_azim + 360.0 * float(i) / max(n_steps - 1, 1)
        n_class = int(st["n"])
        marker_s = ch6_population_n_sweep_marker_size(n_class)
        highlight_marker_s = marker_s * (70.0 / 28.0)
        dens_kw = {}
        if density_histograms:
            dens_kw = dict(
                density_W=st["markers"],
                density_axis_lim=axis_lim,
                density_reveal_u=1.0,
                density_bins=ch6_population_n_sweep_density_bins(n_class, draft=_CH3_DRAFT),
                density_bar_pad=ch6_population_n_sweep_density_bar_pad(n_class),
            )
        img = _frame_duo(
            xlim=st["xlim"],
            ylim=st["ylim"],
            base_study=st["cs"],
            base_exam=st["ce"],
            base_y=st["cy"],
            show_base=True,
            ghost_ws=st["ghosts"],
            w_live=st["w"],
            knob_w=st["w"],
            markers=st["markers"],
            highlight_w=st["w"],
            view_azim=azim,
            show_mean_x=show_mean_x,
            mean_x=st["mean"] if show_mean_x else None,
            mean_x_fontsize=mean_x_fontsize,
            marker_s=marker_s,
            highlight_marker_s=highlight_marker_s,
            **dens_kw,
            **cloud_kw,
            **duo_kw,
        )
        frames.append(_finish(img, clip_id))
    return frames


def build_ch6_74_population_n_sweep_cloud(clip_id):
    return _build_population_n_sweep_cloud(clip_id)


def build_ch6_75_population_n_sweep_cloud_spin(clip_id):
    return _build_population_n_sweep_cloud(clip_id, rotate=True)


def build_ch6_76_population_n_sweep_cloud_mean_x(clip_id):
    return _build_population_n_sweep_cloud(clip_id, show_mean_x=True, mean_x_fontsize=20)


def build_ch6_78_population_reel_cloud_dist_n20_density_keep(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_histograms=True,
    )


def build_ch6_79_population_reel_cloud_dist_n8_density_keep(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id, n_class=CH6_N_CLASS_SMALL, seed=8,
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_histograms=True,
    )


def build_ch6_80_population_reel_cloud_dist_n60_density_keep(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id, n_class=CH6_N_CLASS_LARGE, seed=60,
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_histograms=True,
    )


def build_ch6_81_population_reel_cloud_dist_d1_density_keep(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_histograms=True,
    )


def build_ch6_82_population_reel_cloud_dist_n300_density_keep(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id, n_class=CH6_N_CLASS_XLARGE, seed=300,
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_histograms=True,
    )


def build_ch6_86_population_reel_cloud_dist_n20_density_end(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_at_end=True,
    )


def build_ch6_87_population_reel_cloud_dist_n8_density_end(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id, n_class=CH6_N_CLASS_SMALL, seed=8,
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_at_end=True,
    )


def build_ch6_88_population_reel_cloud_dist_n60_density_end(clip_id):
    from ch6_frequentist import _CH3_DRAFT

    return _build_population_reel_param_cloud_dist(
        clip_id, n_class=CH6_N_CLASS_LARGE, seed=60,
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_at_end=True,
        density_bins=36 if _CH3_DRAFT else 48,
        density_bar_pad=0.32,
    )


def build_ch6_89_population_reel_cloud_dist_d1_density_end(clip_id):
    return _build_population_reel_param_cloud_dist(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_at_end=True,
    )


def build_ch6_90_population_reel_cloud_dist_n300_density_end(clip_id):
    from ch6_frequentist import _CH3_DRAFT

    return _build_population_reel_param_cloud_dist(
        clip_id, n_class=CH6_N_CLASS_XLARGE, seed=300,
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
        marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
        keep_classroom_on_stage=True, density_at_end=True,
        density_bins=60 if _CH3_DRAFT else 80,
        density_bar_pad=0.24,
    )


def _build_population_variance_story(
    clip_id,
    *,
    n_class,
    seed=20,
    seed_from_key=None,
    n_reel=None,
    fast=True,
    marker_axis_lim=CH6_POPULATION_PARAM_AXIS_LIM,
):
    """Variance story continuing from the density_end cloud (center, axis ranges, box)."""
    pack = _population_param_cloud_landings(
        n_class=n_class, seed=seed, seed_from_key=seed_from_key, n_reel=n_reel,
    )
    last = pack["reel_steps"][-1]
    W = np.asarray(last["markers"], dtype=np.float64)
    landed = np.asarray(pack["landed"], dtype=np.float64)
    stats = ch6_param_stats(landed)
    mu = np.asarray(stats["mean"], dtype=np.float64).reshape(3)
    std = np.asarray(stats["std"], dtype=np.float64).reshape(3)
    n_pts = len(W)
    revealed = np.zeros(n_pts, dtype=bool)

    xlim, ylim = pack["xlim"], pack["ylim"]
    cs, ce, cy = last["study"], last["exam"], last["y"]
    w_knob = np.asarray(last["w"], dtype=np.float64).reshape(3)
    hold = CH6_N_HOLD if fast else CH6_N_HOLD * 2
    init_elev, init_azim = _variance_init_view()
    data_bounds = tuple((float(W[:, i].min()), float(W[:, i].max())) for i in range(3))
    inner_bounds = tuple(
        (float(np.percentile(W[:, i], 8)), float(np.percentile(W[:, i], 92))) for i in range(3)
    )
    accumulated_ranges: dict[int, dict] = {}

    def _vf(**kw):
        base = dict(
            xlim=xlim, ylim=ylim,
            base_study=cs, base_exam=ce, base_y=cy,
            W=W, mu=mu, axis_lim=marker_axis_lim,
            revealed_mask=revealed, knob_w=w_knob,
        )
        base.update(kw)
        return _finish(_frame_population_variance_duo(**base), clip_id)

    frames: list = []
    revealed[:] = True
    frames.extend(_hold(_vf(grey_u=0.0, view_elev=init_elev, view_azim=init_azim), max(1, hold // 2)))

    revealed[:] = False
    frames.extend(_hold(_vf(
        grey_u=1.0, show_center=True, view_elev=init_elev, view_azim=init_azim,
    ), hold))

    # b (z) → +90° → w_ST (x) → +90° → w_EL (y)
    axis_passes = ((2, "b"), (0, "w_st_face"), (1, "w_el_face"))
    prev_view = "init"

    for axis_idx, view_name in axis_passes:
        ea_a, aa_a = _variance_resolve_view(prev_view)
        ea_b, aa_b = _variance_resolve_view(view_name)
        az_delta = abs((float(aa_b) - float(aa_a) + 180.0) % 360.0 - 180.0)
        elev_delta = abs(float(ea_b) - float(ea_a))
        if az_delta > 0.5 or elev_delta > 0.5:
            for u in np.linspace(0.0, 1.0, _draft_short(24, 12)):
                elev, azim = _variance_interp_view(prev_view, view_name, float(u))
                frames.append(_vf(
                    grey_u=1.0, show_center=True,
                    range_state=dict(accumulated_ranges),
                    view_elev=elev, view_azim=azim,
                ))
        prev_view = view_name
        elev, azim = _variance_resolve_view(view_name)

        revealed[:] = False
        order = np.argsort(np.abs(W[:, axis_idx] - mu[axis_idx]))
        n_steps = _draft_short(56, 28)
        for step_i in range(1, n_steps + 1):
            n_show = int(np.round(step_i / n_steps * n_pts))
            rev = np.zeros(n_pts, dtype=bool)
            rev[order[:max(1, n_show)]] = True
            show_idx = np.flatnonzero(rev).tolist()
            frames.append(_vf(
                grey_u=1.0, revealed_mask=rev, show_center=True,
                view_elev=elev, view_azim=azim,
                range_state=dict(accumulated_ranges),
                axis_components={
                    axis_idx: {"indices": show_idx, "alpha": 1.0},
                },
            ))
        revealed[:] = True
        all_idx = list(range(n_pts))
        frames.extend(_hold(_vf(
            grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
            range_state=dict(accumulated_ranges),
            axis_components={axis_idx: {"indices": all_idx, "alpha": 1.0}},
        ), max(1, hold // 2)))

        for u in np.linspace(1.0, 0.0, _draft_short(14, 7)):
            frames.append(_vf(
                grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=dict(accumulated_ranges),
                axis_components={axis_idx: {"indices": all_idx, "alpha": float(u)}},
            ))

        half = 2.0 * float(std[axis_idx])
        sigma_lo = float(mu[axis_idx] - half)
        sigma_hi = float(mu[axis_idx] + half)
        data_lo, data_hi = data_bounds[axis_idx]
        cur_rng = {
            axis_idx: {
                "lo": sigma_lo, "hi": sigma_hi, "alpha": 0.0, "labels": False, "lw": 3.0,
            },
        }
        for u in np.linspace(0.0, 1.0, _draft_short(18, 9)):
            cur_rng[axis_idx]["alpha"] = float(u)
            cur_rng[axis_idx]["labels"] = float(u) > 0.55
            merged = {**accumulated_ranges, **cur_rng}
            frames.append(_vf(
                grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=merged,
            ))
        frames.extend(_hold(_vf(
            grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
            range_state={**accumulated_ranges, **cur_rng},
        ), max(1, hold // 2)))

        extend_rng = {axis_idx: {"lo": sigma_lo, "hi": sigma_hi, "alpha": 1.0, "labels": True, "lw": 2.8}}
        for u in np.linspace(0.0, 1.0, _draft_short(18, 9)):
            lo = sigma_lo + float(u) * (data_lo - sigma_lo)
            hi = sigma_hi + float(u) * (data_hi - sigma_hi)
            extend_rng[axis_idx] = {
                "lo": lo, "hi": hi, "alpha": 1.0, "labels": True, "lw": 2.8,
            }
            merged = {**accumulated_ranges, **extend_rng}
            frames.append(_vf(
                grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=merged,
            ))
        accumulated_ranges[axis_idx] = {
            "lo": data_lo, "hi": data_hi, "alpha": 1.0, "labels": False, "lw": 2.8,
        }
        frames.extend(_hold(_vf(
            grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
            range_state=dict(accumulated_ranges),
        ), max(1, hold // 2)))

        # Grey before the next axis pass; after the final pass keep full color.
        if view_name != axis_passes[-1][1]:
            revealed[:] = False
            frames.extend(_hold(_vf(
                grey_u=1.0, show_center=True, view_elev=elev, view_azim=azim,
                range_state=dict(accumulated_ranges),
            ), max(1, hold // 3)))

    revealed[:] = True
    full_ranges = dict(accumulated_ranges)
    for u in np.linspace(0.0, 1.0, _draft_short(28, 14)):
        elev, azim = _variance_interp_view(prev_view, "init", float(u))
        frames.append(_vf(
            grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
            range_state=full_ranges,
        ))

    elev, azim = _variance_init_view()
    frames.extend(_hold(_vf(
        grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
        range_state=full_ranges,
    ), hold))

    for u in np.linspace(0.0, 1.0, _draft_short(20, 10)):
        frames.append(_vf(
            grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
            range_state=full_ranges,
            show_box=True, box_bounds=data_bounds, inner_bounds=inner_bounds, gap_u=float(u),
        ))
    frames.extend(_hold(_vf(
        grey_u=0.0, show_center=True, view_elev=elev, view_azim=azim,
        range_state=full_ranges,
        show_box=True, box_bounds=data_bounds, inner_bounds=inner_bounds, gap_u=1.0,
    ), hold * 2))

    n_spin = _draft_short(72, 28)
    for t in range(n_spin + 1):
        spin_az = float(azim) + 360.0 * float(t) / float(n_spin)
        frames.append(_vf(
            grey_u=0.0, show_center=True, view_elev=elev, view_azim=spin_az,
            range_state=full_ranges,
            show_box=True, box_bounds=data_bounds, inner_bounds=inner_bounds, gap_u=1.0,
        ))
    frames.extend(_hold(frames[-1], hold * 2))
    return frames


def build_ch6_91_population_reel_cloud_dist_n20_variance(clip_id):
    return _build_population_variance_story(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
    )


def build_ch6_92_population_reel_cloud_dist_n8_variance(clip_id):
    return _build_population_variance_story(
        clip_id, n_class=CH6_N_CLASS_SMALL, seed=8,
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
    )


def build_ch6_93_population_reel_cloud_dist_n60_variance(clip_id):
    return _build_population_variance_story(
        clip_id, n_class=CH6_N_CLASS_LARGE, seed=60,
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
    )


def build_ch6_94_population_reel_cloud_dist_d1_variance(clip_id):
    return _build_population_variance_story(
        clip_id, n_class=CH6_N_CLASS_BASE, seed=17, seed_from_key="D1",
    )


def build_ch6_95_population_reel_cloud_dist_n300_variance(clip_id):
    return _build_population_variance_story(
        clip_id, n_class=CH6_N_CLASS_XLARGE, seed=300,
        n_reel=CH6_POPULATION_DIST_N_REEL, fast=True,
    )


def build_ch6_83_population_n_sweep_cloud_density(clip_id):
    return _build_population_n_sweep_cloud(clip_id, density_histograms=True)


def build_ch6_84_population_n_sweep_cloud_density_spin(clip_id):
    return _build_population_n_sweep_cloud(
        clip_id, density_histograms=True, rotate=True,
    )


def build_ch6_85_population_n_sweep_cloud_density_mean_x(clip_id):
    return _build_population_n_sweep_cloud(
        clip_id, density_histograms=True, show_mean_x=True, mean_x_fontsize=20,
    )


def _build_landscape_wobble_d1(
    clip_id,
    *,
    ghost_landscapes: bool = False,
    n_reel: int | None = None,
    seed: int = 35,
    n_class: int | None = None,
    seed_from_key: str | None = "D1",
):
    """Population opening, then classrooms back-to-back with morphing likelihoods.

    Each dataset + best line stays on stage; only the likelihood morphs between
    classrooms (no blank clear).  If ``ghost_landscapes``, each finished bowl is
    frozen into a FIFO of near-grey ghosts that never reshape.

    ``n_class`` sets classroom size (default 20).  With ``seed_from_key="D1"`` the
    opening matches Ch5 D1 into the population (n should be 20); otherwise a fresh
    seed classroom of size ``n_class`` is drawn.
    """
    from ch5_datasets import CH5_STANDARD_XLIM, CH5_STANDARD_YLIM
    from ch6_frequentist import _CH3_DRAFT

    xlim, ylim = CH5_STANDARD_XLIM, CH5_STANDARD_YLIM
    n_class = int(CH6_N_CLASS_BASE if n_class is None else n_class)
    n_pop = int(CH6_POP_SIZE)
    # HQ: 200 classrooms; draft: short preview.
    if n_reel is None:
        n_reel = 12 if _CH3_DRAFT else 200
    n_reel = int(n_reel)
    n_morph = 2 if _CH3_DRAFT else 4
    # Denser than normal HQ LR grid (48) — smoother bowls for the wobble reel.
    grid = 28 if _CH3_DRAFT else 80
    n_ghost_show = 8 if _CH3_DRAFT else 18

    pop_s, pop_e, pop_y = ch6_sample_population(n_pop, seed=seed + 11)

    if seed_from_key is not None:
        tgt_s, tgt_e, tgt_y = ch5_unpack_dataset(str(seed_from_key))
        open_s = np.asarray(tgt_s, dtype=np.float64)
        open_e = np.asarray(tgt_e, dtype=np.float64)
        open_y = np.asarray(tgt_y, dtype=np.float64)
        match_s, match_e, match_y, _ = ch6_match_roster_from_population(
            pop_s, pop_e, pop_y, open_s, open_e, open_y,
        )
    else:
        open_s, open_e, open_y = ch6_opening_classroom_from_population(
            pop_s, pop_e, pop_y, n_class, seed=seed,
        )
        match_s, match_e, match_y = open_s, open_e, open_y

    w_open, _ = ch6_fit_population_classroom(open_s, open_e, open_y)
    w_seed, _ = ch6_fit_population_classroom(match_s, match_e, match_y)

    frames = []
    duo_kw = dict(show_legend=False, title_left=None, title_right=None)

    # --- Opening: seed classroom → population → optional snap to twins ---
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=open_s, base_exam=open_e, base_y=open_y,
        w_live=w_open, knob_w=w_open, show_ellipses=True,
        zlabel="Likelihood", **duo_kw,
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    for u in np.linspace(0.0, 1.0, _draft_short(12, 5)):
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=open_s, base_exam=open_e, base_y=open_y,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            pop_alpha=0.04 + 0.14 * float(u),
            w_live=w_open, knob_w=w_open, show_ellipses=(u < 0.35),
            zlabel="Likelihood", **duo_kw,
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))

    if seed_from_key is not None:
        for u in np.linspace(0.0, 1.0, _draft_short(10, 4)):
            uu = float(u)
            bs = (1.0 - uu) * open_s + uu * match_s
            be = (1.0 - uu) * open_e + uu * match_e
            ww = (1.0 - uu) * w_open + uu * w_seed
            img = _frame_duo(
                xlim=xlim, ylim=ylim,
                base_study=bs, base_exam=be, base_y=open_y,
                pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                w_live=ww, knob_w=ww,
                zlabel="Likelihood", **duo_kw,
            )
            frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD))

    # --- Precompute classrooms + shared-grid likelihoods ---
    print(
        f"  wobble: n={n_class}  fitting {n_reel} classrooms…",
        flush=True,
    )
    rng = np.random.default_rng(seed + 99)
    classrooms = []
    for i in range(n_reel):
        if i == 0:
            cs, ce, cy = match_s, match_e, match_y
        else:
            cs, ce, cy, _ = ch6_draw_classroom_from_population(
                pop_s, pop_e, pop_y, n_class, rng=rng,
            )
        w, _ = ch6_fit_population_classroom(cs, ce, cy)
        classrooms.append((cs, ce, cy, w))

    Ws = np.asarray([c[3] for c in classrooms], dtype=np.float64)
    b_fixed = float(np.mean(Ws[:, 2]))

    print(f"  wobble: building {n_reel} likelihood surfaces (grid={grid})…", flush=True)
    surfaces = []
    for cs, ce, cy, w in classrooms:
        Xd = ch6_design(cs, ce)
        surf = ch6_rel_likelihood_w12(
            Xd, cy, w, ridge=CH6_RIDGE, b_fixed=b_fixed, grid=grid,
        )
        surfaces.append(surf)

    W1 = surfaces[0]["W1"]
    W2 = surfaces[0]["W2"]
    ghost_ws: list[np.ndarray] = []
    ghost_surfs: list[dict] = []  # FIFO — once appended, a ghost never reshapes
    landed: list[np.ndarray] = []

    def _ghost_kw():
        if not ghost_landscapes:
            return None
        return list(ghost_surfs)

    def _push_ghost(surf):
        """Freeze this landscape as a ghost; drop oldest if over the display cap."""
        if not ghost_landscapes:
            return
        # Store a copy of Z so later morph blends can't alias into ghosts.
        ghost_surfs.append({
            "W1": W1,
            "W2": W2,
            "Z": np.asarray(surf["Z"], dtype=np.float64).copy(),
            "z_lim": (0.0, CH6_SURFACE_Z_HI),
        })
        while len(ghost_surfs) > n_ghost_show:
            ghost_surfs.pop(0)

    def _blend_surf(Za, Zb, u):
        Z = (1.0 - u) * Za + u * Zb
        peak = float(np.nanmax(Z))
        if peak > 1e-12:
            Z = Z / peak
        return {"W1": W1, "W2": W2, "Z": Z, "z_lim": (0.0, CH6_SURFACE_Z_HI)}

    # --- Reel: dataset + best line one after another; only likelihood morphs between ---
    for i, (cs, ce, cy, w) in enumerate(classrooms):
        landed.append(w)
        markers = np.asarray(landed)
        gsurfs = _ghost_kw()

        # Classroom + its best line + its landscape (no blank clear phase)
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=cs, base_exam=ce, base_y=cy,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            pop_alpha=0.10,
            w_live=w, ghost_ws=ghost_ws, knob_w=w,
            surface=surfaces[i], morph_u=1.0,
            ghost_surfaces=gsurfs,
            markers=markers, highlight_w=w,
            zlabel="Likelihood", **duo_kw,
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_FLASH + CH6_N_SEQ_HOLD))

        ghost_ws.append(w)

        if i >= n_reel - 1:
            _push_ghost(surfaces[i])
            continue

        # Freeze current landscape as a ghost, then morph the *live* surface to the next.
        _push_ghost(surfaces[i])
        gsurfs = _ghost_kw()  # fixed set for the whole morph
        w_next = classrooms[i + 1][3]
        Za = np.asarray(surfaces[i]["Z"], dtype=np.float64)
        Zb = np.asarray(surfaces[i + 1]["Z"], dtype=np.float64)

        # Keep this classroom on stage while the likelihood wobbles to the next bowl.
        for u in np.linspace(0.0, 1.0, n_morph + 1)[1:]:
            uu = float(u)
            img = _frame_duo(
                xlim=xlim, ylim=ylim,
                base_study=cs, base_exam=ce, base_y=cy,
                pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                pop_alpha=0.10,
                w_live=w, ghost_ws=ghost_ws, knob_w=w,
                surface=_blend_surf(Za, Zb, uu), morph_u=1.0,
                ghost_surfaces=gsurfs,
                markers=markers, highlight_w=w,
                zlabel="Likelihood", **duo_kw,
            )
            frames.append(_finish(img, clip_id))

        if (i + 1) % 25 == 0:
            print(f"  wobble: classroom {i + 1}/{n_reel}  ({len(frames)} frames)", flush=True)

    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    print(f"  wobble: done — {len(frames)} frames", flush=True)
    return frames


def build_ch6_35_landscape_wobble_d1(clip_id):
    """200 classrooms (n=20 / D1): datasets + best lines back-to-back; likelihood morphs."""
    return _build_landscape_wobble_d1(
        clip_id, ghost_landscapes=False, seed=35, n_class=20, seed_from_key="D1",
    )


def build_ch6_36_landscape_wobble_ghosts_d1(clip_id):
    """Same as ch6_35 with FIFO near-grey ghost landscapes."""
    return _build_landscape_wobble_d1(
        clip_id, ghost_landscapes=True, seed=36, n_class=20, seed_from_key="D1",
    )


def build_ch6_37_landscape_wobble_n6(clip_id):
    """Like ch6_35 but each classroom has n=6 students."""
    return _build_landscape_wobble_d1(
        clip_id, ghost_landscapes=False, seed=37, n_class=6, seed_from_key=None,
    )


def build_ch6_38_landscape_wobble_ghosts_n6(clip_id):
    """Like ch6_36 but each classroom has n=6 students."""
    return _build_landscape_wobble_d1(
        clip_id, ghost_landscapes=True, seed=38, n_class=6, seed_from_key=None,
    )


def build_ch6_39_landscape_wobble_n60(clip_id):
    """Like ch6_35 but each classroom has n=60 students."""
    return _build_landscape_wobble_d1(
        clip_id, ghost_landscapes=False, seed=39, n_class=60, seed_from_key=None,
    )


def build_ch6_40_landscape_wobble_ghosts_n60(clip_id):
    """Like ch6_36 but each classroom has n=60 students."""
    return _build_landscape_wobble_d1(
        clip_id, ghost_landscapes=True, seed=40, n_class=60, seed_from_key=None,
    )


# ---------------------------------------------------------------------------
# 2D dataset + per-point gradient projections | likelihood wobble
# ---------------------------------------------------------------------------

_CH6_GRAD_VECTOR_COLOR = "#d500f9"
_CH6_GRAD_VECTOR_UNDER = "#1a001f"


def ch6_point_nll_grad_contrib(w, study, exam, y):
    """Per-point NLL gradient contributions ``(p_i - y_i) x_i`` — shape (n, 3)."""
    Xd = ch6_design(study, exam)
    w = np.asarray(w, dtype=np.float64).reshape(3)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = ch6_sigmoid(Xd @ w)
    return (p - y)[:, None] * Xd


def _draw_point_grad_quivers_2d(
    ax,
    study,
    exam,
    contrib,
    w,
    *,
    color=None,
    span_frac=0.22,
    lw=2.4,
    alpha=0.98,
):
    """Project each point's feature ∇NLL onto (w_ST, w_EL); draw along the line normal.

    Bias is ignored. Quivers stay perpendicular to the fitted line
    (signed length ∝ (g_ST, g_EL) · n̂).
    """
    color = _CH6_GRAD_VECTOR_COLOR if color is None else color
    study = np.asarray(study, dtype=np.float64).reshape(-1)
    exam = np.asarray(exam, dtype=np.float64).reshape(-1)
    G = np.asarray(contrib, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(3)
    if G.ndim != 2 or G.shape[0] == 0:
        return
    n_xy = w[:2].copy()
    n_len = float(np.linalg.norm(n_xy))
    if n_len < 1e-12:
        return
    n_hat = n_xy / n_len

    # Feature components only — drop bias from both g and the projection axis.
    G2 = G[:, :2]
    signed = G2 @ n_hat
    peak = float(np.max(np.abs(signed))) + 1e-12
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    target = float(span_frac) * min(float(xlim[1] - xlim[0]), float(ylim[1] - ylim[0]))
    n = len(study)
    if n <= 12:
        target *= 1.30
        shaft_w = 0.014
        head_w, head_l, head_a = 5.5, 6.5, 5.0
    elif n >= 40:
        target *= 0.80
        shaft_w = 0.008
        head_w, head_l, head_a = 4.2, 5.0, 4.0
    else:
        shaft_w = 0.011
        head_w, head_l, head_a = 5.0, 6.0, 4.6

    U = np.zeros(n, dtype=np.float64)
    V = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = float(signed[i])
        if abs(s) < 1e-9:
            continue
        length = (s / peak) * target
        if abs(length) < 0.05 * target:
            continue
        U[i] = length * float(n_hat[0])
        V[i] = length * float(n_hat[1])

    mask = (np.abs(U) + np.abs(V)) > 0
    if not np.any(mask):
        return
    # Dark under-quiver then bright overlay — large heads read clearly on scatter.
    for col, wid, zo, a in (
        (_CH6_GRAD_VECTOR_UNDER, shaft_w * 1.55, 8, 0.90),
        (color, shaft_w, 9, float(alpha)),
    ):
        ax.quiver(
            study[mask], exam[mask], U[mask], V[mask],
            angles="xy", scale_units="xy", scale=1.0,
            width=wid,
            headwidth=head_w,
            headlength=head_l,
            headaxislength=head_a,
            color=col,
            alpha=a,
            pivot="tail",
            zorder=zo,
            minshaft=1.5,
            minlength=0.4,
        )


def _build_sigmoid_grad_wobble(
    clip_id,
    *,
    n_class: int | None = None,
    seed_from_key: str | None = "D1",
    seed: int = 41,
    n_reel: int | None = None,
):
    """Like ch6_36: 2D dataset + 3D likelihood, with ∇NLL projections on each point."""
    from ch5_datasets import CH5_STANDARD_XLIM, CH5_STANDARD_YLIM
    from ch6_frequentist import _CH3_DRAFT

    xlim, ylim = CH5_STANDARD_XLIM, CH5_STANDARD_YLIM
    n_class = int(CH6_N_CLASS_BASE if n_class is None else n_class)
    n_pop = int(CH6_POP_SIZE)
    if n_reel is None:
        n_reel = 12 if _CH3_DRAFT else 200
    n_reel = int(n_reel)
    n_morph = 2 if _CH3_DRAFT else 4
    grid = 28 if _CH3_DRAFT else 80

    pop_s, pop_e, pop_y = ch6_sample_population(n_pop, seed=seed + 11)
    if seed_from_key is not None:
        tgt_s, tgt_e, tgt_y = ch5_unpack_dataset(str(seed_from_key))
        open_s = np.asarray(tgt_s, dtype=np.float64)
        open_e = np.asarray(tgt_e, dtype=np.float64)
        open_y = np.asarray(tgt_y, dtype=np.float64)
        match_s, match_e, match_y, _ = ch6_match_roster_from_population(
            pop_s, pop_e, pop_y, open_s, open_e, open_y,
        )
    else:
        open_s, open_e, open_y = ch6_opening_classroom_from_population(
            pop_s, pop_e, pop_y, n_class, seed=seed,
        )
        match_s, match_e, match_y = open_s, open_e, open_y

    w_open, _ = ch6_fit_population_classroom(open_s, open_e, open_y)
    w_seed, _ = ch6_fit_population_classroom(match_s, match_e, match_y)
    frames = []
    duo_kw = dict(
        show_legend=False,
        title_left=None,
        title_right=None,
        show_point_grads=True,
    )

    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=open_s, base_exam=open_e, base_y=open_y,
        w_live=w_open, knob_w=w_open, show_ellipses=True,
        zlabel="Likelihood", **duo_kw,
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    for u in np.linspace(0.0, 1.0, _draft_short(12, 5)):
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=open_s, base_exam=open_e, base_y=open_y,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            pop_alpha=0.04 + 0.14 * float(u),
            w_live=w_open, knob_w=w_open, show_ellipses=(u < 0.35),
            zlabel="Likelihood", **duo_kw,
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))

    if seed_from_key is not None:
        for u in np.linspace(0.0, 1.0, _draft_short(10, 4)):
            uu = float(u)
            bs = (1.0 - uu) * open_s + uu * match_s
            be = (1.0 - uu) * open_e + uu * match_e
            ww = (1.0 - uu) * w_open + uu * w_seed
            img = _frame_duo(
                xlim=xlim, ylim=ylim,
                base_study=bs, base_exam=be, base_y=open_y,
                pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                w_live=ww, knob_w=ww,
                zlabel="Likelihood", **duo_kw,
            )
            frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD))

    print(f"  grad wobble: n={n_class}  fitting {n_reel} classrooms…", flush=True)
    rng = np.random.default_rng(seed + 99)
    classrooms = []
    for i in range(n_reel):
        if i == 0:
            cs, ce, cy = match_s, match_e, match_y
        else:
            cs, ce, cy, _ = ch6_draw_classroom_from_population(
                pop_s, pop_e, pop_y, n_class, rng=rng,
            )
        ww, _ = ch6_fit_population_classroom(cs, ce, cy)
        classrooms.append((cs, ce, cy, ww))

    Ws = np.asarray([c[3] for c in classrooms], dtype=np.float64)
    b_fixed = float(np.mean(Ws[:, 2]))
    print(f"  grad wobble: building {n_reel} likelihoods (grid={grid})…", flush=True)
    surfaces = []
    for cs, ce, cy, ww in classrooms:
        surf = ch6_rel_likelihood_w12(
            ch6_design(cs, ce), cy, ww, ridge=CH6_RIDGE, b_fixed=b_fixed, grid=grid,
        )
        surfaces.append(surf)

    W1, W2 = surfaces[0]["W1"], surfaces[0]["W2"]
    ghost_ws: list[np.ndarray] = []
    landed: list[np.ndarray] = []

    def _blend(Za, Zb, u):
        Z = (1.0 - u) * Za + u * Zb
        peak = float(np.nanmax(Z))
        if peak > 1e-12:
            Z = Z / peak
        return {"W1": W1, "W2": W2, "Z": Z, "z_lim": (0.0, CH6_SURFACE_Z_HI)}

    for i, (cs, ce, cy, ww) in enumerate(classrooms):
        landed.append(ww)
        markers = np.asarray(landed)
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=cs, base_exam=ce, base_y=cy,
            pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
            pop_alpha=0.10,
            w_live=ww, ghost_ws=ghost_ws, knob_w=ww,
            surface=surfaces[i], morph_u=1.0,
            markers=markers, highlight_w=ww,
            zlabel="Likelihood", **duo_kw,
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_FLASH + CH6_N_SEQ_HOLD))
        ghost_ws.append(ww)

        if i >= n_reel - 1:
            continue
        Za = np.asarray(surfaces[i]["Z"], dtype=np.float64)
        Zb = np.asarray(surfaces[i + 1]["Z"], dtype=np.float64)
        for u in np.linspace(0.0, 1.0, n_morph + 1)[1:]:
            uu = float(u)
            img = _frame_duo(
                xlim=xlim, ylim=ylim,
                base_study=cs, base_exam=ce, base_y=cy,
                pop_study=pop_s, pop_exam=pop_e, pop_y=pop_y,
                pop_alpha=0.10,
                w_live=ww, ghost_ws=ghost_ws, knob_w=ww,
                surface=_blend(Za, Zb, uu), morph_u=1.0,
                markers=markers, highlight_w=ww,
                zlabel="Likelihood", **duo_kw,
            )
            frames.append(_finish(img, clip_id))

        if (i + 1) % 25 == 0:
            print(
                f"  grad wobble: classroom {i + 1}/{n_reel}  ({len(frames)} frames)",
                flush=True,
            )

    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    print(f"  grad wobble: done — {len(frames)} frames", flush=True)
    return frames


def build_ch6_41_sigmoid_grad_wobble_d1(clip_id):
    """2D data + ∇NLL projections | likelihood wobble (n=20 / D1)."""
    return _build_sigmoid_grad_wobble(
        clip_id, n_class=20, seed_from_key="D1", seed=41,
    )


def build_ch6_42_sigmoid_grad_wobble_n6(clip_id):
    """2D data + ∇NLL projections | likelihood wobble (n=6)."""
    return _build_sigmoid_grad_wobble(
        clip_id, n_class=6, seed_from_key=None, seed=42,
    )


def build_ch6_43_sigmoid_grad_wobble_n60(clip_id):
    """2D data + ∇NLL projections | likelihood wobble (n=60)."""
    return _build_sigmoid_grad_wobble(
        clip_id, n_class=60, seed_from_key=None, seed=43,
    )


# ---------------------------------------------------------------------------
# Bowl nudge physics — Hooke's law: stiff rides along, soft valley rolls
# ---------------------------------------------------------------------------

def _bowl_surface_z(x, y, *, cx, cy, kx, ky):
    """Paraboloid bowl: z = ½ k_x (x−c_x)² + ½ k_y (y−c_y)² (Hooke potential shape)."""
    return 0.5 * float(kx) * (x - cx) ** 2 + 0.5 * float(ky) * (y - cy) ** 2


def _frame_bowl_ball(
    *,
    bx, by, vx, vy,
    cx, cy, kx, ky,
    elev=22.0, azim=-48.0,
    show_nudge_arrow=False,
    nudge_dir=(0.0, 1.0),
    title=None,
    trail=None,
    bounds=2.6,
    grid_n=56,
):
    """Full-bleed 3D bowl with a ball resting on the surface."""
    from matplotlib.colors import LinearSegmentedColormap

    fig = plt.figure(figsize=_g("CH4_DUO_FIGSIZE") if "CH4_DUO_FIGSIZE" in _G else (12.8, 7.2))
    fig.patch.set_facecolor("#f7f8fb")
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.92], projection="3d")
    ax.set_facecolor("#f7f8fb")

    lo, hi = -float(bounds), float(bounds)
    gn = int(grid_n)
    xs = np.linspace(lo, hi, gn)
    ys = np.linspace(lo, hi, gn)
    X, Y = np.meshgrid(xs, ys)
    Z = _bowl_surface_z(X, Y, cx=cx, cy=cy, kx=kx, ky=ky)
    z_rim = 2.35
    Z_draw = np.where(Z <= z_rim, Z, np.nan)

    cmap = LinearSegmentedColormap.from_list(
        "bowl", ["#dfe7f2", "#8fa6c4", "#3d5a80", "#1b2838"],
    )
    surf = ax.plot_surface(
        X, Y, Z_draw, cmap=cmap, vmin=0.0, vmax=z_rim,
        linewidth=0.0, antialiased=True, alpha=0.38, shade=True,
        rstride=1, cstride=1,
    )
    try:
        surf.set_edgecolor((0.35, 0.40, 0.48, 0.08))
    except Exception:
        pass

    ax.contour(
        X, Y, Z_draw, levels=7, offset=0.0, colors="#6b7c93",
        linewidths=0.45, alpha=0.30, zdir="z",
    )

    bz = float(_bowl_surface_z(bx, by, cx=cx, cy=cy, kx=kx, ky=ky))
    ball_r = 0.15
    speed = float(np.hypot(vx, vy))
    bob = 0.012 * np.tanh(speed / 2.0)
    lift = ball_r + 0.12 + bob
    center = np.array([float(bx), float(by), float(bz) + lift], dtype=np.float64)

    u = np.linspace(0.0, np.pi, 16)
    v = np.linspace(0.0, 2.0 * np.pi, 24)
    uu, vv = np.meshgrid(u, v)
    xsph = center[0] + ball_r * np.sin(uu) * np.cos(vv)
    ysph = center[1] + ball_r * np.sin(uu) * np.sin(vv)
    zsph = center[2] + ball_r * np.cos(uu)
    ax.plot_surface(
        xsph, ysph, zsph,
        color="#e74c3c", linewidth=0.0, antialiased=True, shade=True, alpha=1.0,
        zorder=10,
    )
    ax.scatter(
        [bx], [by], [bz + 0.03],
        s=160, c="#1a1a1a", alpha=0.18, depthshade=False, linewidths=0, zorder=9,
    )

    if trail is not None and len(trail) >= 2:
        tr = np.asarray(trail, dtype=np.float64)
        tz = _bowl_surface_z(tr[:, 0], tr[:, 1], cx=cx, cy=cy, kx=kx, ky=ky) + 0.08
        ax.plot(
            tr[:, 0], tr[:, 1], tz,
            color="#c0392b", lw=1.8, alpha=0.40, zorder=8,
        )

    if show_nudge_arrow:
        nd = np.asarray(nudge_dir, dtype=np.float64)
        nd = nd / max(float(np.linalg.norm(nd)), 1e-9)
        base = np.array([cx - 1.55 * nd[0], cy - 1.55 * nd[1], 0.35])
        tip = base + 0.85 * np.array([nd[0], nd[1], 0.0])
        ax.plot(
            [base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]],
            color="#c0392b", lw=3.2, solid_capstyle="round", zorder=12,
        )
        ax.scatter([tip[0]], [tip[1]], [tip[2]], s=70, c="#c0392b", depthshade=False, zorder=13)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_zlim(0.0, z_rim * 1.05)
    ax.set_xlabel(r"$x$", labelpad=6)
    ax.set_ylabel(r"$y$", labelpad=6)
    ax.set_zlabel("height", labelpad=6)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=float(elev), azim=float(azim))
    try:
        ax.set_box_aspect((1.0, 1.0, 0.55))
    except Exception:
        pass
    if title:
        ax.text2D(
            0.02, 0.97, title, transform=ax.transAxes,
            va="top", ha="left", fontsize=12, color="#222", fontweight="bold",
        )
    return _fig_to_plot(fig)


def _bowl_rk4_step(p, v, *, c, kx, ky, m, damp, dt, f_ext):
    """RK4 for Hooke's law: m a = −K⊙(p−c) − damp v + f_ext."""
    K = np.array([float(kx), float(ky)], dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).reshape(2)
    f_ext = np.asarray(f_ext, dtype=np.float64).reshape(2)
    p = np.asarray(p, dtype=np.float64).reshape(2)
    v = np.asarray(v, dtype=np.float64).reshape(2)

    def deriv(pp, vv):
        aa = (-K * (pp - c) - float(damp) * vv + f_ext) / float(m)
        return vv, aa

    k1_p, k1_v = deriv(p, v)
    k2_p, k2_v = deriv(p + 0.5 * dt * k1_p, v + 0.5 * dt * k1_v)
    k3_p, k3_v = deriv(p + 0.5 * dt * k2_p, v + 0.5 * dt * k2_v)
    k4_p, k4_v = deriv(p + dt * k3_p, v + dt * k3_v)
    p_n = p + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
    v_n = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return p_n, v_n


def _simulate_bowl_nudge(
    *,
    kx, ky,
    c_start=(0.0, 0.0),
    c_end=(0.0, 0.55),
    p0=None,
    v0=None,
    m=1.0,
    zeta=0.55,
    dt=1.0 / 120.0,
    t_rise=0.85,
    t_settle=2.0,
):
    """Translate bowl minimum c(t); ball follows Hooke restoring −K(p−c).

    Equilibrium for a held nudge is p→c (Hooke: F=0 iff p=c). Stiff K → ball
    tracks c with tiny lag; soft K along a valley → larger lag / travel for the
    same Δc.
    """
    c0 = np.asarray(c_start, dtype=np.float64).reshape(2)
    c1 = np.asarray(c_end, dtype=np.float64).reshape(2)
    # Per-axis critical damping scale from geometric mean (anisotropic bowl).
    k_geom = float(np.sqrt(max(float(kx), 1e-6) * max(float(ky), 1e-6)))
    damp = float(zeta) * 2.0 * np.sqrt(float(m) * max(k_geom, 0.25))

    def center_at(t):
        if t <= 0.0:
            u = 0.0
        elif t >= float(t_rise):
            u = 1.0
        else:
            s = t / float(t_rise)
            u = s * s * (3.0 - 2.0 * s)
        return c0 + u * (c1 - c0)

    p = np.asarray(c_start if p0 is None else p0, dtype=np.float64).reshape(2).copy()
    v = np.zeros(2, dtype=np.float64) if v0 is None else np.asarray(v0, dtype=np.float64).reshape(2).copy()
    n = int(round((float(t_rise) + float(t_settle)) / float(dt)))
    path_p = [p.copy()]
    path_c = [center_at(0.0).copy()]
    path_v = [v.copy()]
    for i in range(n):
        t = (i + 1) * float(dt)
        c = center_at(t)
        p, v = _bowl_rk4_step(
            p, v, c=c, kx=kx, ky=ky, m=m, damp=damp, dt=dt,
            f_ext=np.zeros(2),
        )
        path_p.append(p.copy())
        path_c.append(c.copy())
        path_v.append(v.copy())
    return (
        np.asarray(path_p, dtype=np.float64),
        np.asarray(path_c, dtype=np.float64),
        np.asarray(path_v, dtype=np.float64),
    )


def build_ch6_44_bowl_nudge_physics(clip_id):
    """Hooke bowl: stiff → tiny ride-along; very stretched valley → long roll."""
    from ch6_frequentist import _CH3_DRAFT

    frames = []
    # Hooke rates: F = −k (p − c). Steep ⇒ large k ⇒ tiny response to a nudge.
    k_stiff = 48.0
    # Extremely elongated valley: hard walls, nearly flat trough.
    k_wall = 55.0
    k_valley = 0.10
    # Tiny landscape nudge on the steep bowl (ball barely moves).
    nudge_amp = 0.20
    # Soft trough: large travel along y (Hooke: soft k ⇒ big Δ for a shove).
    soft_roll = 2.85
    axis = np.array([0.0, 1.0], dtype=np.float64)
    dt = 1.0 / 120.0
    emit_every = 3 if _CH3_DRAFT else 2
    grid_n = 28 if _CH3_DRAFT else 64
    trail_len = 50 if _CH3_DRAFT else 90
    bounds_round = 2.4
    bounds_valley = 5.6

    def _emit_state(
        bx, by, vx, vy, *, cx, cy, kx, ky, title, trail,
        nudge=False, nudge_dir=None, n_hold=1, bounds=bounds_round,
    ):
        nd = axis if nudge_dir is None else nudge_dir
        img = _frame_bowl_ball(
            bx=bx, by=by, vx=vx, vy=vy,
            cx=cx, cy=cy, kx=kx, ky=ky,
            show_nudge_arrow=nudge, nudge_dir=nd,
            title=title, trail=trail, grid_n=grid_n, bounds=bounds,
        )
        frames.append(_finish(img, clip_id))
        if n_hold > 1:
            frames.extend(_hold(frames[-1], n_hold - 1))

    def _emit_path(
        path_p, path_c, path_v, *, kx, ky, title, bounds,
        nudge_dir=None, trail=None,
    ):
        if trail is None:
            trail = []
        travel = float(np.linalg.norm(path_c[-1] - path_c[0]))
        for i in range(len(path_p)):
            p, c, vv = path_p[i], path_c[i], path_v[i]
            trail.append(p.copy())
            if len(trail) > trail_len:
                trail.pop(0)
            if i % emit_every != 0:
                continue
            progress = float(np.linalg.norm(c - path_c[0]))
            moving = progress < travel * 0.98 if travel > 1e-9 else False
            _emit_state(
                float(p[0]), float(p[1]), float(vv[0]), float(vv[1]),
                cx=float(c[0]), cy=float(c[1]), kx=kx, ky=ky,
                title=title, trail=trail, nudge=moving,
                nudge_dir=nudge_dir, bounds=bounds,
            )
        return trail, path_p[-1].copy(), path_c[-1].copy(), path_v[-1].copy()

    # --- 1. Stiff bowl at rest ---
    _emit_state(
        0.0, 0.0, 0.0, 0.0,
        cx=0.0, cy=0.0, kx=k_stiff, ky=k_stiff,
        title="A steep bowl — ball at rest",
        trail=None, n_hold=_draft_short(14, 5),
    )

    # --- 2. Tiny landscape nudge; stiff Hooke → ball barely moves ---
    c_plus = nudge_amp * axis
    _emit_state(
        0.0, 0.0, 0.0, 0.0,
        cx=0.0, cy=0.0, kx=k_stiff, ky=k_stiff,
        title="Nudge the landscape…",
        trail=None, nudge=True, nudge_dir=axis,
        n_hold=_draft_short(7, 3),
    )
    path_p, path_c, path_v = _simulate_bowl_nudge(
        kx=k_stiff, ky=k_stiff,
        c_start=(0.0, 0.0), c_end=c_plus,
        m=1.0, zeta=0.95, dt=dt,
        t_rise=0.80 if not _CH3_DRAFT else 0.45,
        t_settle=1.0 if not _CH3_DRAFT else 0.55,
    )
    trail, p_now, c_now, _v_now = _emit_path(
        path_p, path_c, path_v,
        kx=k_stiff, ky=k_stiff,
        title="…ball barely moves",
        bounds=bounds_round, nudge_dir=axis,
    )
    frames.extend(_hold(frames[-1], _draft_short(10, 4)))

    # --- 3. Stretch into a very long, narrow valley in place ---
    n_elong = _draft_short(48, 16)
    for t in range(n_elong + 1):
        u = t / max(n_elong, 1)
        uu = u * u * (3.0 - 2.0 * u)
        kx = k_stiff + (k_wall - k_stiff) * uu
        ky = k_stiff + (k_valley - k_stiff) * uu
        bd = bounds_round + (bounds_valley - bounds_round) * uu
        _emit_state(
            float(p_now[0]), float(p_now[1]), 0.0, 0.0,
            cx=float(c_now[0]), cy=float(c_now[1]), kx=kx, ky=ky,
            title="Stretch into a long soft valley",
            trail=None, bounds=bd,
        )
    frames.extend(_hold(frames[-1], _draft_short(10, 4)))

    # --- 4. Soft trough — Hooke Δ ∝ 1/k ⇒ much larger travel along the valley ---
    c_far = -soft_roll * axis
    _emit_state(
        float(p_now[0]), float(p_now[1]), 0.0, 0.0,
        cx=float(c_now[0]), cy=float(c_now[1]),
        kx=k_wall, ky=k_valley,
        title="Same shove on the soft trough…",
        trail=None, nudge=True, nudge_dir=-axis,
        n_hold=_draft_short(7, 3),
        bounds=bounds_valley,
    )
    path_p, path_c, path_v = _simulate_bowl_nudge(
        kx=k_wall, ky=k_valley,
        c_start=c_now, c_end=c_far,
        p0=p_now, v0=np.zeros(2),
        m=1.0, zeta=0.28, dt=dt,
        t_rise=1.35 if not _CH3_DRAFT else 0.70,
        t_settle=3.6 if not _CH3_DRAFT else 1.8,
    )
    _emit_path(
        path_p, path_c, path_v,
        kx=k_wall, ky=k_valley,
        title="…rolls far along the valley",
        bounds=bounds_valley, nudge_dir=-axis, trail=trail,
    )
    frames.extend(_hold(frames[-1], _draft_short(16, 6)))
    return frames


# ---------------------------------------------------------------------------
# Average landscape + dataset ∇ ascent (2D quivers track the moving line)
# ---------------------------------------------------------------------------

def _build_avg_grad_ascent(
    clip_id,
    *,
    n_class: int = 20,
    seed_from_key: str | None = "D1",
    seed: int = 45,
    n_avg: int | None = None,
):
    """Overlay E[landscape] + one classroom; ascend from mean MLE with live 2D grads."""
    from ch6_frequentist import _CH3_DRAFT

    n_avg = int((80 if _CH3_DRAFT else 500) if n_avg is None else n_avg)
    grid = 28 if _CH3_DRAFT else 64
    n_steps = 10 if _CH3_DRAFT else 18
    n_interp = 2 if _CH3_DRAFT else 3

    print(
        f"  avg-grad: n={n_class}  averaging {n_avg} classrooms (grid={grid})…",
        flush=True,
    )
    pack = ch6_average_rel_likelihood_population(
        n_class=int(n_class),
        n_avg=int(n_avg),
        seed=int(seed),
        ridge=CH6_RIDGE,
        grid=grid,
        seed_from_key=seed_from_key,
    )
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    xlim, ylim = pack["xlim"], pack["ylim"]
    Xd = ch6_design(study, exam)
    avg_surf = pack["avg_surf"]
    obs_surf = pack["obs_surf"]
    w_avg = np.asarray(pack["mean_hat"], dtype=np.float64)
    w_obs = np.asarray(pack["w_obs"], dtype=np.float64)
    # Hold b near the shared slice used for the surfaces.
    b_fixed = float(pack["b_fixed"])
    w_avg = w_avg.copy()
    w_avg[2] = b_fixed
    w_obs = w_obs.copy()
    w_obs[2] = b_fixed

    frames = []
    duo_base = dict(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        pop_study=pack["pop_s"], pop_exam=pack["pop_e"], pop_y=pack["pop_y"],
        pop_alpha=0.08,
        show_legend=False,
        ghost_alpha=0.30,
        zlabel="Likelihood",
    )

    # --- 1. Average landscape alone + its best line ---
    img = _frame_duo(
        **duo_base,
        w_mean=w_avg, knob_w=w_avg, w_live=w_avg,
        surface=avg_surf, morph_u=1.0, highlight_w=w_avg,
        title_left=f"average of {n_avg} classrooms",
        title_right="expected likelihood landscape",
        show_point_grads=False,
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    # --- 2. Fade in this classroom's landscape on top of the average ---
    for u in np.linspace(0.0, 1.0, _draft_short(12, 5)):
        uu = float(u)
        img = _frame_duo(
            **duo_base,
            w_mean=w_avg, w_live=w_avg, knob_w=w_avg,
            ghost_surfaces=[avg_surf],
            surface=obs_surf, morph_u=uu, highlight_w=w_avg,
            show_point_grads=True,
            title_left="this classroom vs the average",
            title_right="same 3D plot · two bowls",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))

    # --- 3. Away-push at the average (same convention as ch6_49) ---
    g_away = -ch6_nll_grad(w_avg, Xd, y, ridge=CH6_RIDGE)[:2]
    img = _frame_duo(
        **duo_base,
        w_mean=w_avg, w_live=w_avg, knob_w=w_avg,
        ghost_surfaces=[avg_surf],
        surface=obs_surf, morph_u=1.0, highlight_w=w_avg,
        show_point_grads=True,
        weight_grad=g_away, weight_grad_w=w_avg,
        weight_grad_scale=0.55, weight_grad_ascent=False,
        title_left="average best line on this dataset",
        title_right=r"away-push $\propto -\nabla_{w_{\mathrm{ST}},w_{\mathrm{EL}}}$",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    # --- 4. Stable ascent in (ST, EL): unit −∇NLL steps, clipped to the view ---
    # Raw GD overshoots the [-3,3] floor (arrows/path then "fly off"). Match
    # ch6_49/50: keep every weight on the plotted parameter plane.
    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    step = 0.12
    w = w_avg.copy()
    path = [w.copy()]
    grads = [g_away.copy()]
    target = w_obs[:2]
    for _ in range(int(n_steps)):
        g_nll = ch6_nll_grad(w, Xd, y, ridge=CH6_RIDGE)[:2]
        nrm = float(np.linalg.norm(g_nll)) + 1e-12
        direction = -g_nll / nrm  # likelihood ascent on the floor
        # Blend toward the classroom MLE so we land cleanly (Hooke bridge like 49).
        to_mle = target - w[:2]
        dist = float(np.linalg.norm(to_mle))
        if dist < 1.5 * step:
            w = w_obs.copy()
            path.append(w.copy())
            grads.append((-g_nll).copy())
            break
        pull = to_mle / dist
        move = 0.65 * direction + 0.35 * pull
        move = move / (float(np.linalg.norm(move)) + 1e-12)
        w = w.copy()
        w[0] = float(np.clip(w[0] + step * move[0], lo1, hi1))
        w[1] = float(np.clip(w[1] + step * move[1], lo2, hi2))
        w[2] = b_fixed
        path.append(w.copy())
        grads.append(direction.copy())
    if not np.allclose(path[-1][:2], w_obs[:2], atol=1e-3):
        path.append(w_obs.copy())
        grads.append((-ch6_nll_grad(w_obs, Xd, y, ridge=CH6_RIDGE)[:2]).copy())

    print(f"  avg-grad: walking {len(path) - 1} ascent steps…", flush=True)
    for i in range(len(path) - 1):
        wa, wb = path[i], path[i + 1]
        ga, gb = grads[i], grads[i + 1]
        us = np.linspace(0.0, 1.0, n_interp + 1)
        if i < len(path) - 2:
            us = us[:-1]
        for u in us:
            uu = float(u)
            ww = (1.0 - uu) * wa + uu * wb
            gg = (1.0 - uu) * ga + uu * gb
            trail = np.asarray(path[: i + 1] + [ww], dtype=np.float64)
            img = _frame_duo(
                **duo_base,
                w_mean=w_avg, w_live=ww, knob_w=ww,
                ghost_surfaces=[avg_surf],
                surface=obs_surf, morph_u=1.0, highlight_w=ww,
                show_point_grads=True,
                markers=trail,
                probe_pts=trail[:, :2], probe_color="#d500f9", probe_flat=True,
                weight_grad=gg, weight_grad_w=ww,
                weight_grad_scale=0.55, weight_grad_ascent=False,
                title_left="ascent on this classroom",
                title_right=f"step {i + 1}/{len(path) - 1}",
            )
            frames.append(_finish(img, clip_id))

    img = _frame_duo(
        **duo_base,
        w_mean=w_avg, w_live=w_obs, knob_w=w_obs,
        ghost_surfaces=[avg_surf],
        surface=obs_surf, morph_u=1.0, highlight_w=w_obs,
        show_point_grads=True,
        markers=np.asarray(path, dtype=np.float64),
        probe_pts=np.asarray(path, dtype=np.float64)[:, :2],
        probe_color="#d500f9",
        probe_flat=True,
        title_left="landed at this classroom's best line",
        title_right="average line was the start",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    print(f"  avg-grad: done — {len(frames)} frames", flush=True)
    return frames


def build_ch6_45_avg_grad_ascent_d1(clip_id):
    """Avg landscape (500× n=20) + dataset ∇ ascent with live 2D quivers."""
    return _build_avg_grad_ascent(
        clip_id, n_class=20, seed_from_key="D1", seed=45,
    )


def build_ch6_46_avg_grad_ascent_n6(clip_id):
    """Avg landscape (500× n=6) + dataset ∇ ascent with live 2D quivers."""
    return _build_avg_grad_ascent(
        clip_id, n_class=6, seed_from_key=None, seed=46,
    )


def build_ch6_47_avg_grad_ascent_n60(clip_id):
    """Avg landscape (500× n=60) + dataset ∇ ascent with live 2D quivers."""
    return _build_avg_grad_ascent(
        clip_id, n_class=60, seed_from_key=None, seed=47,
    )


# ---------------------------------------------------------------------------
# Script-aligned — ambiguity ridge / push↔restore / three equivalent views
# ---------------------------------------------------------------------------

def build_ch6_48_ambiguous_ridge(clip_id):
    """Correlated features → landscape ridge → ST/EL trade off along the flat direction."""
    rng = np.random.default_rng(48)
    n = 24
    # Ambiguous roster: long exams come with more study time.
    t = rng.uniform(0.4, 5.6, size=n)
    study = np.clip(t + 0.15 * rng.normal(size=n), 0.2, 6.0)
    exam = np.clip(0.35 + 0.85 * t + 0.20 * rng.normal(size=n), 0.3, 6.0)
    logits = 0.55 * study - 0.35 * exam - 0.4
    y = (rng.random(n) < ch6_sigmoid(logits)).astype(np.float64)
    xlim = (-0.3, 6.5)
    ylim = (-0.3, 6.5)
    w, _ = ch6_fit_dataset(study, exam, y)
    Xd = ch6_design(study, exam)
    surf = ch6_rel_likelihood_w12(Xd, y, w, ridge=CH6_RIDGE, grid=_draft_short(40, 22))
    eigen = ch6_hessian_eigen_w12(w, Xd, y, ridge=CH6_RIDGE)
    flat = np.asarray(eigen["flat_dir"], dtype=np.float64)
    # Orient so ST and EL move opposite along the ridge slide.
    if flat[0] * flat[1] > 0:
        flat = flat * np.array([1.0, -1.0])
    flat = flat / (np.linalg.norm(flat) + 1e-12)

    frames = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=w, knob_w=w,
        title_left="study ↑ with exam length",
        title_right="which factor caused the pass?",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=w, knob_w=w,
        surface=surf, morph_u=1.0, highlight_w=w,
        title_left="data is ambiguous along a ridge",
        title_right="fit almost unchanged either way",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    amps = np.linspace(-1.15, 1.15, _draft_short(22, 9))
    for a in amps:
        ww = w.copy()
        ww[0] = float(w[0] + a * flat[0])
        ww[1] = float(w[1] + a * flat[1])
        pts = ch6_probe_path(w, flat, lengths=[-abs(float(a)), abs(float(a))])
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            w_live=ww, knob_w=ww,
            surface=surf, morph_u=1.0, highlight_w=ww,
            probe_pts=pts, probe_color=CH6_FLAT_COLOR, probe_flat=True,
            title_left="slide along the ridge",
            title_right="ST ↑  ·  EL ↓   (or reverse)",
            zlabel="Likelihood",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    card = _frame_card(
        [
            "Ambiguous data → a ridge.",
            "Best lines from different classrooms",
            "tend to slide along that ridge together.",
            "",
            "That shared slide is covariance.",
        ],
        title="Why parameters co-vary",
    )
    frames.append(_finish(card, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_49_push_vs_restore(clip_id):
    """Away-push at the average vs restoring push at the classroom best — same force, opposing."""
    from ch6_frequentist import _CH3_DRAFT

    n_avg = 60 if _CH3_DRAFT else 240
    grid = 24 if _CH3_DRAFT else 48
    pack = ch6_average_rel_likelihood_population(
        n_class=20, n_avg=n_avg, seed=49, ridge=CH6_RIDGE, grid=grid, seed_from_key="D1",
    )
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    xlim, ylim = pack["xlim"], pack["ylim"]
    Xd = ch6_design(study, exam)
    avg_surf = pack["avg_surf"]
    obs_surf = pack["obs_surf"]
    w_avg = np.asarray(pack["mean_hat"], dtype=np.float64).copy()
    w_obs = np.asarray(pack["w_obs"], dtype=np.float64).copy()
    b_fixed = float(pack["b_fixed"])
    w_avg[2] = b_fixed
    w_obs[2] = b_fixed

    # Classroom push at the average line (ascent on this dataset).
    g_away = -ch6_nll_grad(w_avg, Xd, y, ridge=CH6_RIDGE)[:2]
    # Restoring push on the average landscape at this classroom's best line.
    # Approximate avg landscape gradient via finite differences on avg_surf grid if needed;
    # use mean classroom as proxy: gradient of expected NLL ≈ average of grads.
    # Practical stand-in: opposite of away direction scaled by displacement (Hooke).
    delta = (w_obs - w_avg)[:2]
    # Restoring direction toward average on the average bowl.
    g_restore = -delta  # points from w_obs toward w_avg in (ST, EL)

    duo_base = dict(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        pop_study=pack["pop_s"], pop_exam=pack["pop_e"], pop_y=pack["pop_y"],
        pop_alpha=0.08,
        show_legend=False,
        zlabel="Likelihood",
    )
    frames = []

    img = _frame_duo(
        **duo_base,
        w_mean=w_avg, w_live=w_avg, knob_w=w_avg,
        ghost_surfaces=[avg_surf],
        surface=obs_surf, morph_u=1.0, highlight_w=w_avg,
        weight_grad=g_away, weight_grad_w=w_avg,
        weight_grad_scale=0.55, weight_grad_ascent=False,
        title_left="classroom landscape at the average line",
        title_right="away-push",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    # Walk a few steps away to land near w_obs (visual bridge).
    path = np.linspace(0.0, 1.0, _draft_short(10, 5))
    for u in path:
        ww = (1.0 - u) * w_avg + u * w_obs
        img = _frame_duo(
            **duo_base,
            w_mean=w_avg, w_live=ww, knob_w=ww,
            ghost_surfaces=[avg_surf],
            surface=obs_surf, morph_u=1.0, highlight_w=ww,
            title_left="classroom settles at its best line",
            title_right="departure from the average",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))

    img = _frame_duo(
        **duo_base,
        w_mean=w_avg, w_live=w_obs, knob_w=w_obs,
        surface=avg_surf, morph_u=1.0, highlight_w=w_obs,
        weight_grad=g_restore, weight_grad_w=w_obs,
        weight_grad_scale=0.55, weight_grad_ascent=False,
        title_left="average landscape at this best line",
        title_right="restoring push",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))

    # Opposing pair: show average bowl with both markers and restore arrow.
    img = _frame_duo(
        **duo_base,
        w_mean=w_avg, w_live=w_obs, knob_w=w_obs,
        surface=avg_surf, morph_u=1.0, highlight_w=w_obs,
        markers=np.vstack([w_avg, w_obs]),
        weight_grad=g_restore, weight_grad_w=w_obs,
        weight_grad_scale=0.55, weight_grad_ascent=False,
        title_left="same force — opposing",
        title_right="restore = ∇(avg) at best ÷ curvature",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))

    card = _frame_card(
        [
            "Away-push: classroom gradient at the average.",
            "Restore: average-landscape gradient at the best line.",
            "",
            "Same departure. Opposite reading.",
            "Curvature turns either push into how far you move.",
        ],
        title="Push ↔ restore",
    )
    frames.append(_finish(card, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_50_three_equivalent_views(clip_id):
    """Landings ↔ away-pushes at average ↔ restoring pushes — one variation, three views."""
    from ch6_frequentist import _CH3_DRAFT

    n_show = 8 if _CH3_DRAFT else 14
    n_avg = 50 if _CH3_DRAFT else 200
    grid = 22 if _CH3_DRAFT else 40
    pack = ch6_average_rel_likelihood_population(
        n_class=20, n_avg=n_avg, seed=50, ridge=CH6_RIDGE, grid=grid, seed_from_key="D1",
    )
    cloud = ch6_sampling_cloud("D1", n_reps=max(48, n_show * 6), seed=50)
    study, exam, y = pack["study"], pack["exam"], pack["y"]
    xlim, ylim = pack["xlim"], pack["ylim"]
    avg_surf = pack["avg_surf"]
    w_avg = np.asarray(pack["mean_hat"], dtype=np.float64).copy()
    b_fixed = float(pack["b_fixed"])
    w_avg[2] = b_fixed

    Ws = np.asarray(cloud["weights"], dtype=np.float64)[:n_show].copy()
    Ws[:, 2] = b_fixed
    # Precompute away-pushes (classroom grad at average) via label resamples already in cloud.
    # Use displacement as Hooke proxy for both views (consistent opposing arrows).
    deltas = Ws[:, :2] - w_avg[:2]
    away = deltas  # from average toward each landing
    restore = -deltas

    duo_base = dict(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        show_legend=False,
        zlabel="Likelihood",
    )
    frames = []

    # View 1 — landings
    for k in range(1, n_show + 1):
        img = _frame_duo(
            **duo_base,
            w_mean=w_avg, knob_w=w_avg,
            surface=avg_surf, morph_u=1.0, highlight_w=w_avg,
            markers=Ws[:k],
            ghost_ws=list(Ws[:k]),
            title_left="view 1 — classroom best lines",
            title_right="spread of landings",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    # View 2 — away pushes at the average (cycle)
    for i in range(n_show):
        img = _frame_duo(
            **duo_base,
            w_mean=w_avg, w_live=w_avg, knob_w=w_avg,
            surface=avg_surf, morph_u=1.0, highlight_w=w_avg,
            markers=Ws[: i + 1],
            weight_grad=away[i], weight_grad_w=w_avg,
            weight_grad_scale=0.55, weight_grad_ascent=False,
            title_left="view 2 — away-pushes at the average",
            title_right=f"classroom {i + 1}/{n_show}",
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], max(1, CH6_N_FLASH)))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    # View 3 — restoring pushes at each landing
    for i in range(n_show):
        img = _frame_duo(
            **duo_base,
            w_mean=w_avg, w_live=Ws[i], knob_w=Ws[i],
            surface=avg_surf, morph_u=1.0, highlight_w=Ws[i],
            markers=Ws[: i + 1],
            weight_grad=restore[i], weight_grad_w=Ws[i],
            weight_grad_scale=0.55, weight_grad_ascent=False,
            title_left="view 3 — restoring pushes",
            title_right=f"at landing {i + 1}/{n_show}",
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], max(1, CH6_N_FLASH)))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    card = _frame_card(
        [
            "Landings, away-pushes, restoring pushes:",
            "three views of the same classroom-to-classroom variation.",
            "",
            "Curvature of the average landscape",
            "is the shared translator: push → departure.",
        ],
        title="One variation · three readings",
    )
    frames.append(_finish(card, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 4))
    return frames


# ---------------------------------------------------------------------------
# Spring push / restore — D1, no on-screen text (script: force → departure)
# ---------------------------------------------------------------------------

def _floor_spring_pts(a_xy, b_xy, *, n_coils=9, amp=0.10, n_pts=48):
    """Zigzag spring polyline in the (w_ST, w_EL) floor between two landings."""
    a = np.asarray(a_xy, dtype=np.float64).reshape(2)
    b = np.asarray(b_xy, dtype=np.float64).reshape(2)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-8:
        return a.reshape(1, 2)
    u = d / length
    v = np.array([-u[1], u[0]], dtype=np.float64)
    # Longer stretch → slightly larger coil amplitude (tension reads).
    amp = float(amp) * (0.55 + 0.45 * np.tanh(length / 0.6))
    ts = np.linspace(0.0, 1.0, int(n_pts))
    pts = np.empty((len(ts), 2), dtype=np.float64)
    for i, t in enumerate(ts):
        env = float(np.sin(np.pi * t))  # taper at ends
        pts[i] = a + t * d + (amp * env * np.sin(2.0 * np.pi * n_coils * t)) * v
    return pts


def _avg_surf_fd_step(surf):
    W1 = np.asarray(surf["W1"], dtype=np.float64)
    W2 = np.asarray(surf["W2"], dtype=np.float64)
    dw1 = abs(float(W1[0, 1] - W1[0, 0])) if W1.shape[1] > 1 else 0.05
    dw2 = abs(float(W2[1, 0] - W2[0, 0])) if W2.shape[0] > 1 else 0.05
    return 0.5 * (dw1 + dw2)


def _avg_surf_lik_grad_w12(surf, w1, w2, *, h=None):
    """∇(relative likelihood) on the averaged landscape (w_ST, w_EL)."""
    h = float(_avg_surf_fd_step(surf) if h is None else h)
    zx1 = _surface_z_at(surf, w1 + h, w2)
    zx0 = _surface_z_at(surf, w1 - h, w2)
    zy1 = _surface_z_at(surf, w1, w2 + h)
    zy0 = _surface_z_at(surf, w1, w2 - h)
    return np.array([(zx1 - zx0) / (2.0 * h), (zy1 - zy0) / (2.0 * h)], dtype=np.float64)


def _avg_surf_nll_hess_w12(surf, w1, w2, *, h=None):
    """Hessian of −log(rel. likelihood) on the averaged landscape."""
    h = float(_avg_surf_fd_step(surf) if h is None else h)

    def nll(x, y):
        z = max(_surface_z_at(surf, x, y), 1e-12)
        return -float(np.log(z))

    f0 = nll(w1, w2)
    fxp = nll(w1 + h, w2)
    fxm = nll(w1 - h, w2)
    fyp = nll(w1, w2 + h)
    fym = nll(w1, w2 - h)
    fxyp = nll(w1 + h, w2 + h)
    fxym = nll(w1 + h, w2 - h)
    fmxp = nll(w1 - h, w2 + h)
    fxmy = nll(w1 - h, w2 - h)
    dxx = (fxp - 2.0 * f0 + fxm) / (h * h)
    dyy = (fyp - 2.0 * f0 + fym) / (h * h)
    dxy = (fxyp - fxym - fmxp + fxmy) / (4.0 * h * h)
    H = np.array([[dxx, dxy], [dxy, dyy]], dtype=np.float64)
    # Positive-definite bowl for stable Hooke integration.
    evals = np.linalg.eigvalsh(H)
    floor = max(0.35, -float(np.min(evals)) + 0.08)
    H = H + floor * np.eye(2, dtype=np.float64)
    return H


def _hessian_restore_path(
    p0,
    p_rest,
    H,
    *,
    n_frames=20,
    damp=1.2,
    m=1.0,
    dt=0.014,
    stiffness=4.8,
    max_substeps=520,
):
    """Underdamped Hooke return: m a = −k H (p − p_rest) − c v.

    Stiffness ``stiffness`` scales the landscape Hessian (tighter spring →
    higher ω_n).  Damping ``damp`` is chosen below critical so the point
    snaps back quickly with at most a slight overshoot (ζ ≈ 0.35–0.55).
    Dense substeps are time-resampled to ``n_frames`` for a smooth clip.
    """
    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    H_eff = float(stiffness) * np.asarray(H, dtype=np.float64).reshape(2, 2)
    p = np.asarray(p0, dtype=np.float64).reshape(2).copy()
    rest = np.asarray(p_rest, dtype=np.float64).reshape(2)
    v = np.zeros(2, dtype=np.float64)
    traj = [p.copy()]
    times = [0.0]
    t = 0.0
    for _ in range(int(max_substeps)):
        disp = p - rest
        a = (-H_eff @ disp - float(damp) * v) / float(m)
        v = v + float(dt) * a
        p = p + float(dt) * v
        p[0] = float(np.clip(p[0], lo1, hi1))
        p[1] = float(np.clip(p[1], lo2, hi2))
        t += float(dt)
        traj.append(p.copy())
        times.append(t)
        dist = float(np.linalg.norm(disp))
        spd = float(np.linalg.norm(v))
        if dist < 0.005 and spd < 0.02:
            break
    traj.append(rest.copy())
    times.append(t + float(dt))
    pts = np.asarray(traj, dtype=np.float64)
    ts = np.asarray(times, dtype=np.float64)
    n_out = max(int(n_frames), 2)
    t_query = np.linspace(0.0, float(ts[-1]), n_out)
    out = np.zeros((n_out, 2), dtype=np.float64)
    for d in range(2):
        out[:, d] = np.interp(t_query, ts, pts[:, d])
    out[-1] = rest.copy()
    return out


def _spring_sim_path(p0, p_rest, *, n_steps, k=22.0, damp=4.2, m=1.0, dt=0.055):
    """Underdamped Hooke path from ``p0`` toward rest ``p_rest`` (2D)."""
    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    p = np.asarray(p0, dtype=np.float64).reshape(2).copy()
    rest = np.asarray(p_rest, dtype=np.float64).reshape(2)
    v = np.zeros(2, dtype=np.float64)
    path = [p.copy()]
    for _ in range(int(n_steps)):
        a = (-float(k) * (p - rest) - float(damp) * v) / float(m)
        v = v + float(dt) * a
        p = p + float(dt) * v
        p[0] = float(np.clip(p[0], lo1, hi1))
        p[1] = float(np.clip(p[1], lo2, hi2))
        path.append(p.copy())
    path[-1] = rest.copy()
    return np.asarray(path, dtype=np.float64)


def _tension_wobble_path(p0, toward, *, n_frames, amp=0.055):
    """Load the spring: oscillating pulls along the force direction, growing amp."""
    p0 = np.asarray(p0, dtype=np.float64).reshape(2)
    toward = np.asarray(toward, dtype=np.float64).reshape(2)
    d = toward - p0
    nrm = float(np.linalg.norm(d)) + 1e-12
    u = d / nrm
    out = []
    n = max(int(n_frames), 2)
    for i in range(n):
        t = i / (n - 1)
        # Growing shake; a few cycles, then linger at peak stretch.
        env = 0.25 + 0.75 * t
        phase = 2.0 * np.pi * (1.6 + 1.2 * t) * t * 3.0
        stretch = float(amp) * env * (0.55 + 0.45 * np.sin(phase))
        # Bias toward the target so tension "leans" into the snap.
        out.append(p0 + stretch * u)
    return np.asarray(out, dtype=np.float64)


def _pack_w3(xy, b_fixed):
    w = np.zeros(3, dtype=np.float64)
    w[0] = float(xy[0])
    w[1] = float(xy[1])
    w[2] = float(b_fixed)
    return w


def _build_spring_away_d1_hq(
    clip_id,
    *,
    pack,
    study,
    exam,
    y,
    n_push,
    n_return,
    n_settle_hold,
):
    """HQ spring-away: floor point → classroom → avg-likelihood ∇ → H⁻¹ return."""
    b_fixed = float(pack["b_fixed"])
    w_avg = np.asarray(pack["mean_hat"], dtype=np.float64).copy()
    w_avg[2] = b_fixed
    w_obs = np.asarray(pack["w_obs"], dtype=np.float64).copy()
    w_obs[2] = b_fixed
    avg_surf = pack["avg_surf"]
    xlim, ylim = pack["xlim"], pack["ylim"]

    disp0 = w_obs[:2] - w_avg[:2]
    H_avg = _avg_surf_nll_hess_w12(avg_surf, float(w_avg[0]), float(w_avg[1]))
    H_inv = np.linalg.inv(H_avg)
    hinv_disp0 = float(np.sqrt(max(disp0 @ H_inv @ disp0, 1e-18)))
    g_avg_at_obs = _avg_surf_lik_grad_w12(avg_surf, float(w_obs[0]), float(w_obs[1]))
    if float(np.linalg.norm(g_avg_at_obs)) < 1e-12:
        g_avg_at_obs = -disp0.copy()
    g_scale = 0.55

    duo_base = dict(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        pop_study=pack["pop_s"], pop_exam=pack["pop_e"], pop_y=pack["pop_y"],
        pop_alpha=0.07,
        show_legend=False,
        zlabel="Likelihood",
        surface_palette="cloud",
        marker_z_mode="floor",
        weight_grad_floor_only=True,
        highlight_color=CH6_VARIANCE_RED,
        weight_grad_color=CH6_VARIANCE_RED,
        highlight_marker_s=100,
        show_point_grads=False,
        title_left=None,
        title_right=None,
    )

    def _vf(**kw):
        return _finish(_frame_duo(**duo_base, **kw), clip_id)

    frames: list = []

    # 1. Average best line as a floor-plane point (no arrow yet).
    w0 = w_avg.copy()
    frames.extend(_hold(_vf(
        w_mean=w_avg, w_live=w0, knob_w=w0,
        surface=avg_surf, morph_u=1.0, highlight_w=w0,
    ), CH6_N_HOLD * 2))

    # 2. Push to classroom; average-likelihood arrow grows with departure.
    n_push_i = max(int(n_push), 2)
    for i in range(1, n_push_i + 1):
        u = float(i) / float(n_push_i)
        xy = (1.0 - u) * w_avg[:2] + u * w_obs[:2]
        ww = _pack_w3(xy, b_fixed)
        g_now = _avg_surf_lik_grad_w12(avg_surf, float(xy[0]), float(xy[1]))
        if float(np.linalg.norm(g_now)) < 1e-12:
            g_now = (xy - w_avg[:2]).astype(np.float64)
        frames.append(_vf(
            w_mean=w_avg, w_live=ww, knob_w=ww,
            surface=avg_surf, morph_u=1.0, highlight_w=ww,
            weight_grad=g_now, weight_grad_scale=g_scale * u,
        ))

    # 3. At the classroom line: full gradient arrow.
    frames.extend(_hold(_vf(
        w_mean=w_avg, w_live=w_obs, knob_w=w_obs,
        surface=avg_surf, morph_u=1.0, highlight_w=w_obs,
        weight_grad=g_avg_at_obs, weight_grad_scale=g_scale,
    ), CH6_N_HOLD * 2))

    # 4. Spring return; arrow decays to 0 at the optimum (H⁻¹ metric).
    ret_path = _hessian_restore_path(
        w_obs[:2], w_avg[:2], H_avg,
        n_frames=20,
        damp=1.15,
        stiffness=5.2,
        dt=0.013,
    )
    for xy in ret_path[1:]:
        ww = _pack_w3(xy, b_fixed)
        disp = xy - w_avg[:2]
        arrow_scale = float(np.sqrt(max(disp @ H_inv @ disp, 0.0))) / hinv_disp0
        grad_kw = {}
        if arrow_scale > 1e-3:
            grad_kw = dict(
                weight_grad=g_avg_at_obs,
                weight_grad_scale=g_scale * arrow_scale,
            )
        frames.append(_vf(
            w_mean=w_avg, w_live=ww, knob_w=ww,
            surface=avg_surf, morph_u=1.0, highlight_w=ww,
            **grad_kw,
        ))

    # 5. Settled back at the average.
    frames.extend(_hold(_vf(
        w_mean=w_avg, w_live=w_avg, knob_w=w_avg,
        surface=avg_surf, morph_u=1.0, highlight_w=w_avg,
    ), int(n_settle_hold)))

    print(f"  spring-away: done — {len(frames)} frames", flush=True)
    return frames


def _build_spring_away_d1_physics(clip_id, *, seed=51, n_avg=500, grid=200):
    """Surface-ball spring: depart from peak → compress → smooth coast to optimum."""
    print(f"  spring-physics: averaging {n_avg} classrooms (grid={grid})…", flush=True)
    pack, study, exam, y, w_avg, w_obs, b_fixed, avg_surf = _spring_away_d1_landscape_pack(
        seed=seed, n_avg=n_avg, grid=grid,
    )
    xlim, ylim = pack["xlim"], pack["ylim"]
    peak_xy = _avg_surf_peak_w12(avg_surf)
    w_peak = _pack_w3(peak_xy, b_fixed)
    away = w_obs[:2] - peak_xy
    away_u = away / (float(np.linalg.norm(away)) + 1e-12)

    duo_base = dict(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        pop_study=pack["pop_s"], pop_exam=pack["pop_e"], pop_y=pack["pop_y"],
        pop_alpha=0.07,
        show_legend=False,
        zlabel="Likelihood",
        surface_palette="cloud",
        marker_z_mode="surface",
        show_point_grads=False,
        highlight_color=CH6_VARIANCE_RED,
        highlight_marker_s=105,
        title_left=None,
        title_right=None,
    )

    def _vf(**kw):
        return _finish(_frame_duo(**duo_base, **kw), clip_id)

    frames: list = []
    n_hold = CH6_N_HOLD * 2
    n_push = 26
    n_compress = 11
    n_coast = 52

    # 1. Resting at the averaged-landscape peak (population mean line in 2D).
    frames.extend(_hold(_vf(
        w_mean=w_avg, w_live=w_peak, knob_w=w_peak,
        surface=avg_surf, morph_u=1.0, highlight_w=w_peak,
    ), n_hold))

    # 2. Classroom line departs from the average — live 2D line + surface point move.
    n_push_i = max(int(n_push), 2)
    for i in range(1, n_push_i + 1):
        u = float(i) / float(n_push_i)
        ease = 0.5 - 0.5 * np.cos(np.pi * u)
        xy = (1.0 - ease) * peak_xy + ease * w_obs[:2]
        ww = _pack_w3(xy, b_fixed)
        frames.append(_vf(
            w_mean=w_avg, w_live=ww, knob_w=ww,
            surface=avg_surf, morph_u=1.0, highlight_w=ww,
        ))

    frames.extend(_hold(_vf(
        w_mean=w_avg, w_live=w_obs, knob_w=w_obs,
        surface=avg_surf, morph_u=1.0, highlight_w=w_obs,
    ), max(2, CH6_N_HOLD // 2)))

    # 3. Full arrow collapse (ball still) → cradle launch → uphill coast.
    launch = _surface_spring_launch_sequence(
        avg_surf, w_obs[:2], peak_xy, away_u,
        n_compress=n_compress, n_coast=n_coast,
    )
    for step in launch:
        xy = np.asarray(step["xy"], dtype=np.float64)
        ww = _pack_w3(xy, b_fixed)
        kw = dict(
            w_mean=w_avg, w_live=ww, knob_w=ww,
            surface=avg_surf, morph_u=1.0, highlight_w=ww,
        )
        cu = step.get("compress_u")
        if cu is not None and float(cu) > 1e-3:
            kw["spring_arrow"] = dict(
                away_dir=away_u,
                compress_u=float(cu),
                scale=float(step.get("spring_scale", 1.38)),
                color=CH6_VARIANCE_RED,
            )
        frames.append(_vf(**kw))

    # 4. Settled at the likelihood peak (b fixed slice).
    frames.extend(_hold(_vf(
        w_mean=w_avg, w_live=w_peak, knob_w=w_peak,
        surface=avg_surf, morph_u=1.0, highlight_w=w_peak,
    ), CH6_N_HOLD * 4))

    print(f"  spring-physics: done — {len(frames)} frames", flush=True)
    return frames


def _build_spring_force_d1(
    clip_id,
    *,
    mode: str,
    seed: int,
    hq: bool = False,
):
    """Text-free spring: ``away`` (classroom pulls off average) or ``restore``."""
    from ch6_frequentist import _CH3_DRAFT

    assert mode in ("away", "restore")
    if hq:
        n_avg = 500
        grid = 132
        n_tension = 22
        n_flight = 28
        n_settle_hold = CH6_N_HOLD * 4
        surface_palette = "cloud"
        marker_z_mode = "floor"
        probe_flat = False
    else:
        n_avg = 60 if _CH3_DRAFT else 280
        grid = 24 if _CH3_DRAFT else 52
        n_tension = _draft_short(18, 8)
        n_flight = _draft_short(36, 16)
        n_settle_hold = CH6_N_HOLD * (2 if _CH3_DRAFT else 3)
        surface_palette = "belief"
        marker_z_mode = "floor"
        probe_flat = True

    # True D1 roster as the classroom; average landscape from many twins.
    study, exam, y = ch5_unpack_dataset("D1")
    xlim, ylim = ch5_plot_limits("D1")
    print(f"  spring-{mode}: averaging {n_avg} classrooms (grid={grid})…", flush=True)
    pack = ch6_average_rel_likelihood_population(
        n_class=20, n_avg=n_avg, seed=int(seed), ridge=CH6_RIDGE, grid=grid,
        seed_from_key="D1",
    )

    if mode == "away" and hq:
        return _build_spring_away_d1_hq(
            clip_id, pack=pack, study=study, exam=exam, y=y,
            n_push=n_flight, n_return=n_flight + 4, n_settle_hold=n_settle_hold,
        )

    b_fixed = float(pack["b_fixed"])
    w_avg = np.asarray(pack["mean_hat"], dtype=np.float64).copy()
    w_avg[2] = b_fixed
    w_obs, _ = ch6_fit_dataset(study, exam, y, ridge=CH6_RIDGE)
    w_obs = np.asarray(w_obs, dtype=np.float64).copy()
    w_obs[2] = b_fixed
    Xd = ch6_design(study, exam)
    obs_surf = ch6_rel_likelihood_w12(
        Xd, y, w_obs, ridge=CH6_RIDGE, b_fixed=b_fixed, grid=grid,
    )
    avg_surf = pack["avg_surf"]

    # Classroom away-push at the average; restore = Hooke toward average.
    g_away = -ch6_nll_grad(w_avg, Xd, y, ridge=CH6_RIDGE)[:2]
    delta = (w_obs - w_avg)[:2]
    dist = float(np.linalg.norm(delta)) + 1e-12
    # Pedagogue exaggeration: tiny D1 departures still need a readable spring throw.
    travel = float(min(1.55, max(dist, 1.15)))
    delta_vis = (travel / dist) * delta
    g_restore = -delta_vis

    lo1, hi1, lo2, hi2 = CH6_VIEW_BOUNDS_W12
    if mode == "away":
        anchor = w_avg[:2].copy()
        start = w_avg[:2].copy()
        rest = w_avg[:2] + delta_vis
        rest[0] = float(np.clip(rest[0], lo1, hi1))
        rest[1] = float(np.clip(rest[1], lo2, hi2))
        rest_true = w_obs[:2].copy()
        force0 = g_away.copy()
        surf = avg_surf if hq else obs_surf
        ghost = None if hq else [avg_surf]
        show_grads = False if hq else True
    else:
        anchor = w_avg[:2].copy()
        start = w_avg[:2] + delta_vis
        start[0] = float(np.clip(start[0], lo1, hi1))
        start[1] = float(np.clip(start[1], lo2, hi2))
        rest = w_avg[:2].copy()
        rest_true = w_avg[:2].copy()
        force0 = g_restore.copy()
        surf = avg_surf
        ghost = None
        show_grads = False

    duo_base = dict(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        pop_study=pack["pop_s"], pop_exam=pack["pop_e"], pop_y=pack["pop_y"],
        pop_alpha=0.07,
        show_legend=False,
        ghost_alpha=0.28,
        zlabel="Likelihood",
        surface_palette=surface_palette,
        marker_z_mode=marker_z_mode,
        # No titles / cards — motion only.
        title_left=None,
        title_right=None,
    )

    frames = []

    # --- Establish: average line + landscapes, quiet ---
    w0 = _pack_w3(start, b_fixed)
    img = _frame_duo(
        **duo_base,
        w_mean=w_avg, w_live=w0, knob_w=w0,
        ghost_surfaces=ghost,
        surface=surf, morph_u=1.0, highlight_w=w0,
        markers=np.vstack([w_avg, w0]) if mode == "restore" else None,
        show_point_grads=show_grads,
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))

    # --- Tension: wobble + growing force arrow + coiling spring ---
    wobble = _tension_wobble_path(start, rest, n_frames=n_tension, amp=0.10)
    for i, xy in enumerate(wobble):
        t = i / max(len(wobble) - 1, 1)
        ww = _pack_w3(xy, b_fixed)
        spring = _floor_spring_pts(anchor, xy, n_coils=10, amp=0.14 + 0.08 * t)
        force_scale = 0.35 + 0.55 * t
        img = _frame_duo(
            **duo_base,
            w_mean=w_avg, w_live=ww, knob_w=ww,
            ghost_surfaces=ghost,
            surface=surf, morph_u=1.0, highlight_w=ww,
            markers=np.vstack([_pack_w3(anchor, b_fixed), ww]),
            probe_pts=spring, probe_color="#d500f9", probe_flat=probe_flat,
            weight_grad=force0, weight_grad_w=ww,
            weight_grad_scale=force_scale, weight_grad_ascent=False,
            show_point_grads=show_grads,
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], max(2, CH6_N_HOLD)))

    # --- Snap / flight: underdamped spring to rest ---
    flight = _spring_sim_path(
        wobble[-1], rest,
        n_steps=n_flight,
        k=28.0 if mode == "away" else 22.0,
        damp=3.2 if mode == "away" else 3.8,
    )
    trail_xy = [start.copy()]
    trail_stride = 28 if hq else 12
    for i, xy in enumerate(flight):
        t = i / max(len(flight) - 1, 1)
        ww = _pack_w3(xy, b_fixed)
        trail_xy.append(xy.copy())
        spring = _floor_spring_pts(anchor, xy, n_coils=10, amp=0.16)
        force_scale = 0.85 * (1.0 - 0.85 * t)
        g_now = force0 if mode == "away" else (anchor - xy)
        if float(np.linalg.norm(g_now)) < 1e-9:
            g_now = force0
        path_probe = np.asarray(trail_xy, dtype=np.float64) if hq else spring
        img = _frame_duo(
            **duo_base,
            w_mean=w_avg, w_live=ww, knob_w=ww,
            ghost_surfaces=ghost,
            surface=surf, morph_u=1.0, highlight_w=ww,
            markers=np.asarray(
                [_pack_w3(p, b_fixed) for p in trail_xy[:: max(1, len(trail_xy) // trail_stride)]],
                dtype=np.float64,
            ),
            probe_pts=path_probe, probe_color="#d500f9", probe_flat=probe_flat,
            weight_grad=g_now if force_scale > 0.08 else None,
            weight_grad_w=ww,
            weight_grad_scale=max(force_scale, 0.20),
            weight_grad_ascent=False,
            show_point_grads=show_grads,
        )
        frames.append(_finish(img, clip_id))

    # --- Settled at the true endpoint (classroom MLE or average) ---
    w_end = _pack_w3(rest_true, b_fixed)
    spring = _floor_spring_pts(anchor, rest_true, n_coils=8, amp=0.06)
    settle_probe = np.asarray(trail_xy + [rest_true], dtype=np.float64) if hq else spring
    img = _frame_duo(
        **duo_base,
        w_mean=w_avg, w_live=w_end, knob_w=w_end,
        ghost_surfaces=ghost,
        surface=surf, morph_u=1.0, highlight_w=w_end,
        markers=np.vstack([_pack_w3(anchor, b_fixed), w_end]),
        probe_pts=settle_probe, probe_color="#d500f9", probe_flat=probe_flat,
        show_point_grads=show_grads,
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], n_settle_hold))
    print(f"  spring-{mode}: done — {len(frames)} frames", flush=True)
    return frames


def build_ch6_51_spring_away_d1(clip_id):
    """D1 classroom landscape: tension at the average line, then spring away. No text."""
    return _build_spring_force_d1(clip_id, mode="away", seed=51, hq=True)


def build_ch6_52_spring_restore_d1(clip_id):
    """Average landscape: tension at D1's best line, then spring back. No text."""
    return _build_spring_force_d1(clip_id, mode="restore", seed=52)


def build_ch6_96_spring_away_d1_physics(clip_id):
    """Surface spring push: compressing arrow → ball coasts up the averaged bowl."""
    return _build_spring_away_d1_physics(clip_id, seed=51, n_avg=500, grid=200)


# ---------------------------------------------------------------------------
# Script-aligned clips — variance / covariance / expected likelihood / Hessian
# ---------------------------------------------------------------------------

def build_ch6_18_questions_intro(clip_id):
    """Pose the chapter questions, then land on 'how I think about logistic regression'."""
    study, exam, y = ch5_unpack_dataset("D1")
    xlim, ylim = ch5_plot_limits("D1")
    w, _ = ch6_fit_dataset(study, exam, y)
    frames = []
    beats = [
        ("One classroom among many", "What if we repeated everything?"),
        ("Would the line change a lot?", "Or only a little?"),
        ("What forces drive the line?", "Which directions move most?"),
        ("How do we quantify stability?", "That's this video."),
    ]
    for left, right in beats:
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            w_live=w, knob_w=w,
            title_left=left, title_right=right, zlabel="landings",
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    # How-I-think title card
    try:
        from ch5_layout import ch5_overlay_howithink_center_right
        base = frames[-1]
        for u in np.linspace(0.0, 1.0, _draft_short(10, 4)):
            frames.append(ch5_overlay_howithink_center_right(
                base, dim_u=0.55 * float(u), logo_u=float(u),
            ))
        frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    except Exception:
        card = _frame_card(
            ["This… is how I think about", "logistic regression."],
            title="Chapter 6",
        )
        frames.append(_finish(card, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_19_classroom_walk(clip_id):
    """Walk classroom → classroom; each produces a slightly different best line."""
    cloud, draws = ch6_precomputed_resamples("D1", n_reps=_draft_short(16, 6), seed=19)
    study, exam = cloud["study"], cloud["exam"]
    y_obs = cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    frames = []
    ghosts: list[np.ndarray] = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y_obs,
        w_live=cloud["w_obs"], knob_w=cloud["w_obs"],
        title_left="Walk into another classroom…",
        title_right="fit another best line",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))
    for i, (yb, w) in enumerate(draws):
        ghosts.append(w)
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y_obs,
            ghost_study=study, ghost_exam=exam, ghost_y=yb,
            w_live=w, ghost_ws=ghosts[:-1], knob_w=w,
            title_left=f"classroom {i + 1}",
            title_right="different students · different line",
            zlabel="landings",
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_FLASH + 1))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))
    return frames


def build_ch6_20_line_wiggle(clip_id):
    """Fan all ghost lines together — watch the decision boundary wiggle."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Ws = cloud["weights"]
    frames = []
    steps = _draft_short(24, 8)
    for t in range(steps + 1):
        u = t / steps
        k = max(1, int(round(u * min(len(Ws), CH6_N_GHOST_LINES_SHOW))))
        ghosts = list(Ws[:k])
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            w_live=cloud["w_obs"], ghost_ws=ghosts, knob_w=cloud["w_obs"],
            markers=Ws[:k],
            title_left=f"{k} fitted lines",
            title_right="watch the boundary wiggle",
            zlabel="landings",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_21_record_parameters(clip_id):
    """Every line ↔ three numbers; landings accumulate in parameter space."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Ws = cloud["weights"]
    frames = []
    # Emphasize the three knobs first
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=cloud["w_obs"], knob_w=cloud["w_obs"],
        title_left="Every line = three numbers",
        title_right=r"$w_{\mathrm{ST}},\; w_{\mathrm{EL}},\; b$",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    steps = _draft_short(28, 10)
    for t in range(steps + 1):
        u = t / steps
        k = max(1, int(round(u * len(Ws))))
        ghosts = _pick_ghost_lines(Ws[:k])
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            ghost_ws=ghosts, knob_w=Ws[k - 1],
            markers=Ws[:k], highlight_w=Ws[k - 1],
            title_left=f"classroom → ({Ws[k-1,0]:+.2f}, {Ws[k-1,1]:+.2f}, {Ws[k-1,2]:+.2f})",
            title_right="record the parameters",
            zlabel="landings",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_22_parameter_mean(clip_id):
    """Average of each parameter → the typical line we'd expect."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Ws = cloud["weights"]
    stats = ch6_param_stats(Ws)
    mu = stats["mean"]
    frames = []
    ghosts = _pick_ghost_lines(Ws)
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        ghost_ws=ghosts, knob_w=cloud["w_obs"],
        markers=Ws,
        title_left="a cloud of parameters",
        title_right="where do they tend to be?",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))
    # Fade in the mean
    for u in np.linspace(0.0, 1.0, _draft_short(16, 6)):
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            ghost_ws=ghosts, w_mean=mu, knob_w=mu,
            markers=Ws, highlight_w=mu,
            title_left="average of each parameter",
            title_right="→ the typical line",
            zlabel="landings",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_23_variance_study(clip_id):
    """1D spread of the study-time coefficient — introduce variance."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    Ws = cloud["weights"]
    stats = ch6_param_stats(Ws)
    mu_st, std_st = float(stats["mean"][0]), float(stats["std"][0])
    frames = []
    # Grow the histogram
    steps = _draft_short(20, 8)
    for t in range(steps + 1):
        u = t / steps
        k = max(8, int(round(u * len(Ws))))
        vals = Ws[:k, 0]
        st = ch6_param_stats(Ws[:k])
        img = _frame_marginal_hist(
            vals,
            mean=float(st["mean"][0]),
            std=float(st["std"][0]),
            title_left=f"study-time coefficient  (n={k})",
            title_right="how spread out?",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))
    img = _frame_marginal_hist(
        Ws[:, 0], mean=mu_st, std=std_st,
        title_left="that spread is the variance",
        title_right="large → classrooms disagree",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_24_variance_units(clip_id):
    """Hours → seconds: coefficient shrinks, variance changes, uncertainty doesn't."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    Ws = cloud["weights"]
    Ws_s = ch6_rescale_study_hours_to_seconds(Ws)
    st_h = ch6_param_stats(Ws)
    st_s = ch6_param_stats(Ws_s)
    frames = []
    img = _frame_marginal_hist(
        Ws[:, 0],
        mean=float(st_h["mean"][0]),
        std=float(st_h["std"][0]),
        title_left="study time in hours",
        title_right=f"Var(w_ST) ≈ {st_h['var'][0]:.3f}",
        xlabel=r"$w_{\mathrm{ST}}$  (per hour)",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    img = _frame_marginal_hist(
        Ws_s[:, 0],
        mean=float(st_s["mean"][0]),
        std=float(st_s["std"][0]),
        title_left="same data — study time in seconds",
        title_right=f"Var(w_ST) ≈ {st_s['var'][0]:.2e}",
        xlabel=r"$w_{\mathrm{ST}}$  (per second)",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    card = _frame_card(
        [
            "The numerical variance changed.",
            "The underlying uncertainty did not.",
            "Variance alone depends on our units.",
        ],
        title="Units matter",
        subtitle="hours → seconds shrinks w_ST by 1/3600",
    )
    frames.append(_finish(card, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_25_three_variances(clip_id):
    """Three marginal variances — still not the full story."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Ws = cloud["weights"]
    stats = ch6_param_stats(Ws)
    frames = []
    ghosts = _pick_ghost_lines(Ws)
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        ghost_ws=ghosts, w_mean=stats["mean"], knob_w=stats["mean"],
        markers=Ws,
        title_left="three variances",
        title_right=(
            f"ST {stats['var'][0]:.2f} · EL {stats['var'][1]:.2f} · b {stats['var'][2]:.2f}"
        ),
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    card = _frame_card(
        [
            "Have we completely described",
            "how the line moves?",
            "",
            "Not quite.",
        ],
        title="Marginals miss the joint shape",
    )
    frames.append(_finish(card, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_26_circle_vs_ellipse(clip_id):
    """Same horizontal & vertical spread — circle vs long thin ellipse."""
    demo = ch6_iso_vs_corr_clouds(n=_draft_short(160, 60), seed=26)
    study, exam, y = ch5_unpack_dataset("D1")
    xlim, ylim = ch5_plot_limits("D1")
    frames = []
    for label, key, color in (
        ("circular cloud — same spreads", "iso", CH6_GHOST_COLOR),
        ("elliptical cloud — same spreads", "corr", CH6_COV_COLOR),
    ):
        Ws = demo[key]
        mu = Ws.mean(axis=0)
        C = np.cov(Ws[:, :2].T, ddof=1)
        ghosts = _pick_ghost_lines(Ws, n_show=40)
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            ghost_ws=ghosts, w_mean=mu, knob_w=mu,
            markers=Ws, highlight_w=mu,
            floor_ellipse={"mean": mu[:2], "cov": C, "color": color, "mass": 0.90},
            title_left=label,
            title_right="variance alone can't tell them apart",
            zlabel="landings",
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_27_covariance_matrix(clip_id):
    """Introduce covariance, then assemble the covariance matrix."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Ws = cloud["weights"]
    stats = ch6_param_stats(Ws)
    C = stats["cov"]
    frames = []
    ghosts = _pick_ghost_lines(Ws)
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        ghost_ws=ghosts, w_mean=stats["mean"], knob_w=stats["mean"],
        markers=Ws,
        floor_ellipse={
            "mean": stats["mean"][:2],
            "cov": C[:2, :2],
            "color": CH6_COV_COLOR,
            "mass": 0.95,
        },
        title_left="parameters move together",
        title_right="covariance captures that dance",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    # Reveal matrix entries progressively
    labels = [
        (0, 0, r"Var(w_ST)"),
        (1, 1, r"Var(w_EL)"),
        (2, 2, r"Var(b)"),
        (0, 1, r"Cov(ST,EL)"),
        (0, 2, r"Cov(ST,b)"),
        (1, 2, r"Cov(EL,b)"),
    ]
    shown = set()
    for (i, j, name) in labels:
        shown.add((i, j))
        shown.add((j, i))
        rows = []
        for r in range(3):
            cells = []
            for c in range(3):
                if (r, c) in shown:
                    cells.append(f"{C[r, c]:+.2f}")
                else:
                    cells.append("  ·  ")
            rows.append("  ".join(cells))
        card = _frame_card(
            rows,
            title="Covariance matrix Σ",
            subtitle=f"fill in {name}",
        )
        frames.append(_finish(card, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD))
    card = _frame_card(
        [
            "Σ completely describes how the",
            "parameters vary around their average —",
            "in every direction through parameter space.",
        ],
        title="Stability of the line itself",
    )
    frames.append(_finish(card, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_28_many_likelihoods(clip_id):
    """Each classroom has its own likelihood landscape — similar overall shape."""
    cloud, stack = ch6_classroom_likelihood_stack(
        "D1", n_show=_draft_short(6, 3), seed=28, grid=_draft_short(36, 20),
    )
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    frames = []
    for i, item in enumerate(stack):
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            ghost_study=study, ghost_exam=exam, ghost_y=item["y"],
            w_live=item["w"], knob_w=item["w"],
            surface=item["surface"], morph_u=1.0, highlight_w=item["w"],
            title_left=f"classroom {i + 1} likelihood",
            title_right="peak shifts · shape rhymes",
            zlabel="Likelihood",
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD + 1))
    return frames


def build_ch6_29_expected_likelihood(clip_id):
    """Average many landscapes → expected likelihood landscape."""
    n_avg = _draft_short(16, 6)
    grid = _draft_short(36, 20)
    pack = ch6_expected_rel_likelihood(
        "D1", n_avg=n_avg, seed=29, grid=grid,
    )
    cloud = pack["cloud"]
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    frames = []
    # Same grid as expected surface so the morph blends cleanly
    Xd = ch6_design(study, exam)
    surf_one = ch6_rel_likelihood_w12(
        Xd, y, cloud["w_obs"], ridge=CH6_RIDGE,
        b_fixed=pack["b_fixed"], grid=grid,
    )
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=cloud["w_obs"], knob_w=cloud["w_obs"],
        surface=surf_one, morph_u=1.0, highlight_w=cloud["w_obs"],
        title_left="one observed landscape",
        title_right="now average many…",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD))
    # Morph toward the expected landscape
    exp_surf = {"W1": pack["W1"], "W2": pack["W2"], "Z": pack["Z"], "z_lim": pack["z_lim"]}
    for u in np.linspace(0.0, 1.0, _draft_short(20, 8)):
        # Blend Z fields
        Z = (1.0 - u) * np.asarray(surf_one["Z"]) + u * np.asarray(pack["Z"])
        blend = {"W1": pack["W1"], "W2": pack["W2"], "Z": Z, "z_lim": pack["z_lim"]}
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            w_mean=pack["mean_hat"], knob_w=pack["mean_hat"],
            surface=blend, morph_u=1.0, highlight_w=pack["mean_hat"],
            markers=pack["hats"],
            title_left="expected likelihood landscape",
            title_right=f"average of {n_avg} classrooms",
            zlabel="Likelihood",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    # Overlay sampling cloud — cov shape matches landscape
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_mean=pack["mean_hat"], knob_w=pack["mean_hat"],
        surface=exp_surf, morph_u=1.0, highlight_w=pack["mean_hat"],
        markers=cloud["weights"],
        title_left="Σ is written in the curvature",
        title_right="of this average landscape",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_30_steep_vs_flat(clip_id):
    """Steep directions = stable; flat directions = high variance."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Xd = ch6_design(study, exam)
    w = cloud["w_obs"]
    surf = ch6_rel_likelihood_w12(Xd, y, w, ridge=CH6_RIDGE)
    eigen = ch6_hessian_eigen_w12(w, Xd, y, ridge=CH6_RIDGE)
    frames = []
    # Steep probe
    for label, direction, color in (
        ("steep direction — likelihood drops fast", eigen["steep_dir"], CH6_STEEP_COLOR),
        ("flat direction — many near-equally good lines", eigen["flat_dir"], CH6_FLAT_COLOR),
    ):
        lengths = np.linspace(-1.4, 1.4, _draft_short(18, 7))
        for t in lengths:
            pts = ch6_probe_path(w, direction, lengths=[-abs(t), abs(t)])
            img = _frame_duo(
                xlim=xlim, ylim=ylim,
                base_study=study, base_exam=exam, base_y=y,
                w_live=w, knob_w=w,
                surface=surf, morph_u=1.0, highlight_w=w,
                markers=cloud["weights"],
                probe_pts=pts, probe_color=color,
                title_left=label,
                title_right="stable" if color == CH6_STEEP_COLOR else "unstable",
                zlabel="Likelihood",
            )
            frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    card = _frame_card(
        [
            "Steep  →  small uncertainty",
            "Flat   →  large uncertainty",
            "",
            "The landscape tells us how tightly",
            "the data constrains the parameters.",
        ],
        title="Curvature ↔ variance",
    )
    frames.append(_finish(card, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_31_one_classroom_problem(clip_id):
    """The thought experiment is impossible — we only have one classroom."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Ws = cloud["weights"]
    frames = []
    ghosts = _pick_ghost_lines(Ws)
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        ghost_ws=ghosts, knob_w=cloud["w_obs"],
        markers=Ws,
        title_left="hundreds of classrooms…",
        title_right="an impossible experiment",
        zlabel="landings",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    # Collapse ghosts away
    for u in np.linspace(1.0, 0.0, _draft_short(16, 6)):
        k = max(0, int(round(u * len(ghosts))))
        mk = max(1, int(round(u * len(Ws))))
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            w_live=cloud["w_obs"], ghost_ws=ghosts[:k], knob_w=cloud["w_obs"],
            markers=Ws[:mk] if u > 0.05 else None,
            highlight_w=cloud["w_obs"],
            title_left="in reality we only have one",
            title_right="so how do we estimate Σ?",
            zlabel="landings",
        )
        frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    return frames


def build_ch6_32_large_n_approx(clip_id):
    """Large datasets: one observed likelihood ≈ expected likelihood."""
    frames = []
    for key, label, n_note in (
        ("D1", "n = 20 — noisier shape", "small n"),
        ("D3", "n = 60 — closer to the average", "large n"),
    ):
        cloud = ch6_sampling_cloud(key, n_reps=_draft_short(80, 24), seed=32)
        study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
        Xd = ch6_design(study, exam)
        surf = ch6_rel_likelihood_w12(Xd, y, cloud["w_obs"], ridge=CH6_RIDGE)
        ghosts = _pick_ghost_lines(cloud["weights"])
        img = _frame_duo(
            xlim=cloud["xlim"], ylim=cloud["ylim"],
            base_study=study, base_exam=exam, base_y=y,
            w_live=cloud["w_obs"], ghost_ws=ghosts, knob_w=cloud["w_obs"],
            surface=surf, morph_u=1.0,
            markers=cloud["weights"], highlight_w=cloud["w_obs"],
            title_left=label,
            title_right=f"{n_note} · observed ≈ expected",
            zlabel="Likelihood",
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    card = _frame_card(
        [
            "One sufficiently large dataset",
            "is often enough.",
            "",
            "We don't need hundreds of classrooms.",
        ],
        title="Large-n approximation",
    )
    frames.append(_finish(card, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_33_hessian_to_cov(clip_id):
    """Curvature of the observed likelihood (Hessian) → estimate of Σ."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Xd = ch6_design(study, exam)
    w = cloud["w_obs"]
    surf = ch6_rel_likelihood_w12(Xd, y, w, ridge=CH6_RIDGE)
    cov = ch6_wald_cov(w, Xd, y, ridge=CH6_RIDGE)
    eigen = ch6_hessian_eigen_w12(w, Xd, y, ridge=CH6_RIDGE)
    frames = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=w, knob_w=w,
        surface=surf, morph_u=1.0, highlight_w=w,
        title_left="observed likelihood",
        title_right="curvature = Hessian H",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    # Eigen arrows
    for color, direction, tag in (
        (CH6_STEEP_COLOR, eigen["steep_dir"], "steep eigen-direction"),
        (CH6_FLAT_COLOR, eigen["flat_dir"], "flat eigen-direction"),
    ):
        pts = ch6_probe_path(w, direction, lengths=[-1.2, 1.2])
        img = _frame_duo(
            xlim=xlim, ylim=ylim,
            base_study=study, base_exam=exam, base_y=y,
            w_live=w, knob_w=w,
            surface=surf, morph_u=1.0, highlight_w=w,
            probe_pts=pts, probe_color=color,
            title_left=tag,
            title_right=r"$\Sigma \approx H^{-1}$",
            zlabel="Likelihood",
        )
        frames.append(_finish(img, clip_id))
        frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    # Wald ellipse + empirical cloud
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=w, knob_w=w,
        surface=surf, morph_u=1.0, highlight_w=w,
        markers=cloud["weights"], show_wald=True,
        title_left="Hessian inverse ≈ sampling cloud",
        title_right="Wald ellipse from one dataset",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    card = _frame_card(
        [
            f"Σ̂_ST ≈ {cov[0,0]:.3f}   Σ̂_EL ≈ {cov[1,1]:.3f}",
            f"Cov̂(ST,EL) ≈ {cov[0,1]:.3f}",
            "",
            "One landscape → estimate of how the",
            "best line would move if we could repeat.",
        ],
        title="From curvature to covariance",
    )
    frames.append(_finish(card, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    return frames


def build_ch6_34_stability_outro(clip_id):
    """From finding an optimum → measuring how stable it is; teaser for distributions."""
    cloud = ch6_sampling_cloud("D1", n_reps=CH6_N_REPS_CLOUD, seed=2)
    study, exam, y = cloud["study"], cloud["exam"], cloud["y_obs"]
    xlim, ylim = cloud["xlim"], cloud["ylim"]
    Xd = ch6_design(study, exam)
    w = cloud["w_obs"]
    surf = ch6_rel_likelihood_w12(Xd, y, w, ridge=CH6_RIDGE)
    stats = ch6_param_stats(cloud["weights"])
    frames = []
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=w, knob_w=w,
        surface=surf, morph_u=1.0, highlight_w=w, show_wald=True,
        title_left="Before: find the best line",
        title_right="Now: how stable is it?",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 2))
    # SE callout on knobs
    se = stats["std"]
    img = _frame_duo(
        xlim=xlim, ylim=ylim,
        base_study=study, base_exam=exam, base_y=y,
        w_live=w, ghost_ws=_pick_ghost_lines(cloud["weights"]), knob_w=w,
        markers=cloud["weights"], highlight_w=w, show_wald=True,
        title_left=f"SE(ST)≈{se[0]:.2f}  SE(EL)≈{se[1]:.2f}  SE(b)≈{se[2]:.2f}",
        title_right="large SE — accident or pattern?",
        zlabel="Likelihood",
    )
    frames.append(_finish(img, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 3))
    card = _frame_card(
        [
            "How much the parameters vary",
            "is only half the story.",
            "",
            "Next: how those variations",
            "are distributed.",
        ],
        title="One step further",
    )
    frames.append(_finish(card, clip_id))
    frames.extend(_hold(frames[-1], CH6_N_HOLD * 4))
    return frames


def _ch6_build_export_specs() -> list[tuple[str, str, Callable[[str], list]]]:
    specs: list[tuple[str, str, Callable[[str], list]]] = []

    def add(slug: str, builder):
        n = len(specs) + 1
        clip_id = f"ch6_{n:02d}"
        specs.append((clip_id, f"{clip_id}_{slug}.mp4", builder))

    add("observed_line", build_ch6_01_observed_line)
    add("generative_coins", build_ch6_02_generative_coins)
    add("resample_reel", build_ch6_03_resample_reel)
    add("landing_histogram", build_ch6_04_landing_histogram)
    add("looks_like_likelihood", build_ch6_05_looks_like_likelihood)
    add("likelihood_bowl", build_ch6_06_likelihood_bowl)
    add("lr_confidence", build_ch6_07_lr_confidence)
    add("wald_vs_lr", build_ch6_08_wald_vs_lr)
    add("separation", build_ch6_09_separation)
    add("n_effect", build_ch6_10_n_effect)
    add("d4_chaos", build_ch6_11_d4_chaos)
    add("bayes_vs_freq", build_ch6_12_bayes_vs_freq)
    add("class_gaussians", build_ch6_13_class_gaussians)
    add("population_reel_n20", build_ch6_14_population_reel_n20)
    add("population_reel_n8", build_ch6_15_population_reel_n8)
    add("population_reel_n60", build_ch6_16_population_reel_n60)
    add("population_reel_d1", build_ch6_17_population_reel_d1)
    # Script-aligned pedagogy (variance → covariance → Hessian)
    add("questions_intro", build_ch6_18_questions_intro)
    add("classroom_walk", build_ch6_19_classroom_walk)
    add("line_wiggle", build_ch6_20_line_wiggle)
    add("record_parameters", build_ch6_21_record_parameters)
    add("parameter_mean", build_ch6_22_parameter_mean)
    add("variance_study", build_ch6_23_variance_study)
    add("variance_units", build_ch6_24_variance_units)
    add("three_variances", build_ch6_25_three_variances)
    add("circle_vs_ellipse", build_ch6_26_circle_vs_ellipse)
    add("covariance_matrix", build_ch6_27_covariance_matrix)
    add("many_likelihoods", build_ch6_28_many_likelihoods)
    add("expected_likelihood", build_ch6_29_expected_likelihood)
    add("steep_vs_flat", build_ch6_30_steep_vs_flat)
    add("one_classroom_problem", build_ch6_31_one_classroom_problem)
    add("large_n_approx", build_ch6_32_large_n_approx)
    add("hessian_to_cov", build_ch6_33_hessian_to_cov)
    add("stability_outro", build_ch6_34_stability_outro)
    add("landscape_wobble_d1", build_ch6_35_landscape_wobble_d1)
    add("landscape_wobble_ghosts_d1", build_ch6_36_landscape_wobble_ghosts_d1)
    add("landscape_wobble_n6", build_ch6_37_landscape_wobble_n6)
    add("landscape_wobble_ghosts_n6", build_ch6_38_landscape_wobble_ghosts_n6)
    add("landscape_wobble_n60", build_ch6_39_landscape_wobble_n60)
    add("landscape_wobble_ghosts_n60", build_ch6_40_landscape_wobble_ghosts_n60)
    add("sigmoid_grad_wobble_d1", build_ch6_41_sigmoid_grad_wobble_d1)
    add("sigmoid_grad_wobble_n6", build_ch6_42_sigmoid_grad_wobble_n6)
    add("sigmoid_grad_wobble_n60", build_ch6_43_sigmoid_grad_wobble_n60)
    add("bowl_nudge_physics", build_ch6_44_bowl_nudge_physics)
    add("avg_grad_ascent_d1", build_ch6_45_avg_grad_ascent_d1)
    add("avg_grad_ascent_n6", build_ch6_46_avg_grad_ascent_n6)
    add("avg_grad_ascent_n60", build_ch6_47_avg_grad_ascent_n60)
    add("ambiguous_ridge", build_ch6_48_ambiguous_ridge)
    add("push_vs_restore", build_ch6_49_push_vs_restore)
    add("three_equivalent_views", build_ch6_50_three_equivalent_views)
    add("spring_away_d1", build_ch6_51_spring_away_d1)
    add("spring_restore_d1", build_ch6_52_spring_restore_d1)
    add("population_reel_cloud_n20", build_ch6_53_population_reel_cloud_n20)
    add("population_reel_cloud_n8", build_ch6_54_population_reel_cloud_n8)
    add("population_reel_cloud_n60", build_ch6_55_population_reel_cloud_n60)
    add("population_reel_cloud_d1", build_ch6_56_population_reel_cloud_d1)
    add("population_reel_cloud_dist_n20", build_ch6_57_population_reel_cloud_dist_n20)
    add("population_reel_cloud_dist_n8", build_ch6_58_population_reel_cloud_dist_n8)
    add("population_reel_cloud_dist_n60", build_ch6_59_population_reel_cloud_dist_n60)
    add("population_reel_cloud_dist_d1", build_ch6_60_population_reel_cloud_dist_d1)
    add("population_reel_n20_keep", build_ch6_61_population_reel_n20_keep)
    add("population_reel_n8_keep", build_ch6_62_population_reel_n8_keep)
    add("population_reel_n60_keep", build_ch6_63_population_reel_n60_keep)
    add("population_reel_d1_keep", build_ch6_64_population_reel_d1_keep)
    add("population_reel_cloud_n20_keep", build_ch6_65_population_reel_cloud_n20_keep)
    add("population_reel_cloud_n8_keep", build_ch6_66_population_reel_cloud_n8_keep)
    add("population_reel_cloud_n60_keep", build_ch6_67_population_reel_cloud_n60_keep)
    add("population_reel_cloud_d1_keep", build_ch6_68_population_reel_cloud_d1_keep)
    add("population_reel_cloud_dist_n20_keep", build_ch6_69_population_reel_cloud_dist_n20_keep)
    add("population_reel_cloud_dist_n8_keep", build_ch6_70_population_reel_cloud_dist_n8_keep)
    add("population_reel_cloud_dist_n60_keep", build_ch6_71_population_reel_cloud_dist_n60_keep)
    add("population_reel_cloud_dist_d1_keep", build_ch6_72_population_reel_cloud_dist_d1_keep)
    add("population_reel_cloud_dist_n300_keep", build_ch6_73_population_reel_cloud_dist_n300_keep)
    add("population_n_sweep_cloud", build_ch6_74_population_n_sweep_cloud)
    add("population_n_sweep_cloud_spin", build_ch6_75_population_n_sweep_cloud_spin)
    add("population_n_sweep_cloud_mean_x", build_ch6_76_population_n_sweep_cloud_mean_x)
    add("cloud_density_project_previews", build_ch6_77_cloud_density_project_previews)
    add("population_reel_cloud_dist_n20_density_keep", build_ch6_78_population_reel_cloud_dist_n20_density_keep)
    add("population_reel_cloud_dist_n8_density_keep", build_ch6_79_population_reel_cloud_dist_n8_density_keep)
    add("population_reel_cloud_dist_n60_density_keep", build_ch6_80_population_reel_cloud_dist_n60_density_keep)
    add("population_reel_cloud_dist_d1_density_keep", build_ch6_81_population_reel_cloud_dist_d1_density_keep)
    add("population_reel_cloud_dist_n300_density_keep", build_ch6_82_population_reel_cloud_dist_n300_density_keep)
    add("population_n_sweep_cloud_density", build_ch6_83_population_n_sweep_cloud_density)
    add("population_n_sweep_cloud_density_spin", build_ch6_84_population_n_sweep_cloud_density_spin)
    add("population_n_sweep_cloud_density_mean_x", build_ch6_85_population_n_sweep_cloud_density_mean_x)
    add("population_reel_cloud_dist_n20_density_end", build_ch6_86_population_reel_cloud_dist_n20_density_end)
    add("population_reel_cloud_dist_n8_density_end", build_ch6_87_population_reel_cloud_dist_n8_density_end)
    add("population_reel_cloud_dist_n60_density_end", build_ch6_88_population_reel_cloud_dist_n60_density_end)
    add("population_reel_cloud_dist_d1_density_end", build_ch6_89_population_reel_cloud_dist_d1_density_end)
    add("population_reel_cloud_dist_n300_density_end", build_ch6_90_population_reel_cloud_dist_n300_density_end)
    add("population_reel_cloud_dist_n20_variance", build_ch6_91_population_reel_cloud_dist_n20_variance)
    add("population_reel_cloud_dist_n8_variance", build_ch6_92_population_reel_cloud_dist_n8_variance)
    add("population_reel_cloud_dist_n60_variance", build_ch6_93_population_reel_cloud_dist_n60_variance)
    add("population_reel_cloud_dist_d1_variance", build_ch6_94_population_reel_cloud_dist_d1_variance)
    add("population_reel_cloud_dist_n300_variance", build_ch6_95_population_reel_cloud_dist_n300_variance)
    add("spring_away_d1_physics", build_ch6_96_spring_away_d1_physics)
    return specs


CH6_EXPORT_SPECS: list[tuple[str, str, Callable[[str], list]]] = _ch6_build_export_specs()

# Smoother playback for the physics bowl (≈25 fps).
CH6_BOWL_MS = 40


def ch6_export_clip(filename: str) -> Path:
    spec = next((s for s in CH6_EXPORT_SPECS if s[1] == filename or s[0] == filename), None)
    if spec is None:
        key = filename.replace(".mp4", "")
        spec = next((s for s in CH6_EXPORT_SPECS if s[0] == key or s[1].startswith(key)), None)
    if spec is None:
        raise KeyError(f"unknown export: {filename!r}")
    clip_id, fn, builder = spec
    frames = builder(clip_id)
    ms = CH6_BOWL_MS if "bowl_nudge" in fn else CH6_MS
    if fn in {
        "ch6_53_population_reel_cloud_n20.mp4",
        "ch6_57_population_reel_cloud_dist_n20.mp4",
        "ch6_65_population_reel_cloud_n20_keep.mp4",
        "ch6_69_population_reel_cloud_dist_n20_keep.mp4",
    } or "population_n_sweep_cloud" in fn:
        ms = max(55, int(CH6_MS * 0.7))
    _g("save_mp4")(frames, fn, duration=ms)
    path = _g("OUTPUT_DIR") / fn
    print("wrote", path, f"({len(frames)} frames, ms={ms})")
    return path
